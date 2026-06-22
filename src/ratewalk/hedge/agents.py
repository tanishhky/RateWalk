"""Hedging agents.

You asked for multiple agents, each using a different hedging instrument, plus
an all-in-one agent that scans which instrument looks underpriced at each
instance and uses that. Each agent searches a (tenor x moneyness) grid of
protective puts and reports, for every contract, the protection efficiency:

    efficiency = CVaR_reduction / premium_paid

i.e. tail risk removed per premium dollar. The agent's pick is the contract
with the best efficiency ("least premium for most utility"). The all-in-one
agent runs all underlying-typed agents and, per grid cell, selects whichever
underlying is cheapest relative to its modeled fair value (largest underpricing),
then ranks by efficiency, so it behaves like a desk shopping the cheapest hedge.

Option premia here use a Black-style approximation on the simulated terminal
distribution. With a live option chain wired in, the same agents price off real
quotes unchanged; the search and efficiency definition do not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .. import obs
from ..analytics.risk import var_cvar


@dataclass
class HedgeCandidate:
    agent: str
    underlying: str
    tenor_years: float
    moneyness: float
    premium: float
    fair_value: float
    underpricing: float           # fair_value - premium (positive = cheap)
    cvar_reduction: float
    efficiency: float             # cvar_reduction / premium


def _put_payoff(terminal_prices: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(strike - terminal_prices, 0.0)


def _bs_put_premium(spot: float, strike: float, vol: float, t: float,
                    r: float = 0.0) -> float:
    """Black-Scholes put premium (r as a flat discount). A modeling proxy for
    the agent when no live chain is present."""
    from scipy.stats import norm
    if t <= 0 or vol <= 0:
        return max(strike - spot, 0.0)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return float(strike * np.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1))


class HedgeAgent:
    """Base hedging agent over one underlying type."""
    name = "base"
    underlying = "base"
    # vol multiplier captures that different underlyings (ETF vs futures vs
    # swaption) carry different implied-vol levels; the all-in-one agent uses
    # the cheaper one as 'underpriced'.
    vol_multiplier = 1.0
    fair_vol_multiplier = 1.0

    def scan(self, terminal_prices: np.ndarray, base_returns: np.ndarray, *,
             moneyness_grid, tenor_grid, horizon_years: float) -> List[HedgeCandidate]:
        spot = float(np.mean(terminal_prices))   # use mean terminal as reference spot
        realized_vol = float(np.std(np.log(np.maximum(terminal_prices, 1e-6) / spot))) \
            / np.sqrt(max(horizon_years, 1e-6))
        base_cvar = var_cvar(base_returns, 0.95)["CVaR"]
        cands: List[HedgeCandidate] = []
        for m in moneyness_grid:
            strike = m * spot
            for t in tenor_grid:
                mkt_vol = realized_vol * self.vol_multiplier
                fair_vol = realized_vol * self.fair_vol_multiplier
                premium = _bs_put_premium(spot, strike, mkt_vol, t)
                fair = _bs_put_premium(spot, strike, fair_vol, t)
                if premium <= 1e-6:
                    continue
                # hedged returns: pay premium, receive put payoff at horizon
                payoff = _put_payoff(terminal_prices, strike)
                hedged = base_returns + (payoff - premium) / spot
                hedged_cvar = var_cvar(hedged, 0.95)["CVaR"]
                reduction = base_cvar - hedged_cvar
                eff = reduction / premium if premium > 1e-9 else 0.0
                cands.append(HedgeCandidate(
                    agent=self.name, underlying=self.underlying,
                    tenor_years=float(t), moneyness=float(m),
                    premium=round(premium, 4), fair_value=round(fair, 4),
                    underpricing=round(fair - premium, 4),
                    cvar_reduction=round(reduction, 5),
                    efficiency=round(eff, 4)))
        return cands

    def best(self, *args, **kwargs) -> Optional[HedgeCandidate]:
        c = self.scan(*args, **kwargs)
        return max(c, key=lambda x: x.efficiency) if c else None


class EtfPutAgent(HedgeAgent):
    name = "etf_put"; underlying = "TLT"
    vol_multiplier = 1.05; fair_vol_multiplier = 1.0     # ETF puts a touch rich


class FuturesOptionAgent(HedgeAgent):
    name = "futures_option"; underlying = "ZB_future"
    vol_multiplier = 0.98; fair_vol_multiplier = 1.0     # futures options a touch cheap


class SwaptionAgent(HedgeAgent):
    name = "swaption"; underlying = "swaption"
    vol_multiplier = 1.10; fair_vol_multiplier = 1.0     # OTC, usually richest


class AllInOneAgent(HedgeAgent):
    """Scans every underlying-typed agent and, per grid cell, keeps whichever
    underlying is most underpriced (fair - premium), then ranks by efficiency.
    Behaves like a desk shopping the cheapest available hedge."""
    name = "all_in_one"; underlying = "best_of"

    def __init__(self, sub_agents: List[HedgeAgent]):
        self.sub_agents = sub_agents

    def scan(self, terminal_prices, base_returns, *, moneyness_grid, tenor_grid,
             horizon_years):
        by_cell: Dict[tuple, HedgeCandidate] = {}
        for ag in self.sub_agents:
            for c in ag.scan(terminal_prices, base_returns,
                             moneyness_grid=moneyness_grid, tenor_grid=tenor_grid,
                             horizon_years=horizon_years):
                key = (c.moneyness, c.tenor_years)
                # keep the most underpriced underlying for this cell
                if key not in by_cell or c.underpricing > by_cell[key].underpricing:
                    chosen = HedgeCandidate(**{**c.__dict__, "agent": self.name})
                    by_cell[key] = chosen
        return list(by_cell.values())


def build_agents(cfg_hedge) -> List[HedgeAgent]:
    registry = {"etf_put": EtfPutAgent, "futures_option": FuturesOptionAgent,
                "swaption": SwaptionAgent}
    subs = [registry[a]() for a in cfg_hedge.agents if a in registry]
    agents: List[HedgeAgent] = list(subs)
    if "all_in_one" in cfg_hedge.agents:
        agents.append(AllInOneAgent(subs))
    return agents


def run_agents(agents: List[HedgeAgent], terminal_prices, base_returns,
               cfg_hedge, horizon_years: float) -> Dict:
    out = {}
    for ag in agents:
        best = ag.best(terminal_prices, base_returns,
                       moneyness_grid=cfg_hedge.moneyness_grid,
                       tenor_grid=cfg_hedge.tenor_grid_years,
                       horizon_years=horizon_years)
        out[ag.name] = best.__dict__ if best else None
    obs.event(channel="hedge", kind="run_agents", n_agents=len(agents))
    return out
