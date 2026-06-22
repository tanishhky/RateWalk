"""Monte Carlo engine.

Per path: sample a rate-state trajectory from the Markov chain (conditioned on
a simulated CPI regime when the conditional model is used), cumulate the
increments into a short-rate path, apply jumps, map each step to a full curve,
reprice the held book, accrue and reinvest coupons across the tenor ladder,
and record total wealth. Pricing is vectorized across paths; only the
(few-hundred) time steps loop in Python, so 5000 paths run quickly.

Reinvestment model (v1): coupons and any maturing principal accrue into a
reinvestment balance that compounds at the prevailing blended ladder yield.
Full bond-by-bond reinvestment is a documented extension; the interface does
not change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .. import obs
from ..curve import CurveModel
from ..instruments.book import InstrumentBook
from .jumps import JumpModel


@dataclass
class SimResult:
    horizon_years: float
    steps_per_year: int
    init_investment: float
    terminal_wealth: np.ndarray      # (n_paths,)
    horizon_return: np.ndarray       # (n_paths,) total return over horizon
    annualized_return: np.ndarray    # (n_paths,)
    wealth_paths: np.ndarray         # (n_paths, n_steps+1)
    rate_paths: np.ndarray           # (n_paths, n_steps+1) short rate (percent)
    config_hash: str = ""
    seed: int = 0


# ── increment label parsing ────────────────────────────────────────────────

def _increment_bps(label: str) -> float:
    m = re.match(r"([+-]?\d+)bps", label)
    return float(m.group(1)) / 100.0 if m else 0.0


class _InterpPlan:
    """Linear-interpolation weights for a set of times on a fixed tenor grid,
    so curve -> yields-at-times is a vectorized matmul. The times are
    measured from NOW (time-to-cashflow), so a fresh plan is built each step as
    the bond ages."""

    def __init__(self, tenors: List[float], times):
        self.t = np.asarray(tenors, dtype=float)
        self.times = np.asarray(times, dtype=float)
        if self.times.size == 0:
            self.i_lo = self.i_hi = np.array([], dtype=int)
            self.w_lo = self.w_hi = np.array([])
            return
        self.i_lo = np.clip(np.searchsorted(self.t, self.times, side="right") - 1,
                            0, len(self.t) - 2)
        self.i_hi = self.i_lo + 1
        span = self.t[self.i_hi] - self.t[self.i_lo]
        span[span == 0] = 1.0
        self.w_hi = (self.times - self.t[self.i_lo]) / span
        self.w_lo = 1.0 - self.w_hi

    def yields_at(self, curve: np.ndarray) -> np.ndarray:
        """curve (n_paths, n_tenors) -> (n_paths, n_times) yields in percent."""
        if self.times.size == 0:
            return np.zeros((curve.shape[0], 0))
        return curve[:, self.i_lo] * self.w_lo[None, :] + curve[:, self.i_hi] * self.w_hi[None, :]


def _simulate_rate_states(rate_P: np.ndarray, n_paths: int, n_steps: int,
                          start_idx: int, rng: np.random.Generator,
                          cpi_P: Optional[np.ndarray] = None,
                          conditional_P: Optional[List[np.ndarray]] = None,
                          cpi_start_idx: int = 0):
    """Return (rate_state_idx paths). If conditional_P is given, a CPI regime
    is simulated in lockstep and selects the rate matrix each step."""
    states = np.zeros((n_paths, n_steps + 1), dtype=int)
    states[:, 0] = start_idx
    n_rate = rate_P.shape[0]

    if conditional_P is None:
        cdf = np.cumsum(rate_P, axis=1)
        for s in range(1, n_steps + 1):
            u = rng.random(n_paths)
            cur = states[:, s - 1]
            states[:, s] = (u[:, None] > cdf[cur]).sum(axis=1).clip(0, n_rate - 1)
        return states, None

    # conditional: simulate CPI regime, pick matrix per regime
    cpi = np.zeros((n_paths, n_steps + 1), dtype=int)
    cpi[:, 0] = cpi_start_idx
    cpi_cdf = np.cumsum(cpi_P, axis=1)
    cond_cdf = [np.cumsum(P, axis=1) for P in conditional_P]
    n_cpi = cpi_P.shape[0]
    for s in range(1, n_steps + 1):
        uc = rng.random(n_paths)
        cur_c = cpi[:, s - 1]
        cpi[:, s] = (uc[:, None] > cpi_cdf[cur_c]).sum(axis=1).clip(0, n_cpi - 1)
        ur = rng.random(n_paths)
        cur_r = states[:, s - 1]
        nxt = np.empty(n_paths, dtype=int)
        for regime in range(n_cpi):
            mask = cpi[:, s] == regime
            if not mask.any():
                continue
            cdf_r = cond_cdf[regime]
            nxt[mask] = (ur[mask, None] > cdf_r[cur_r[mask]]).sum(axis=1)
        states[:, s] = nxt.clip(0, n_rate - 1)
    return states, cpi


def run_simulation(cfg, *, curve_model: CurveModel, book: InstrumentBook,
                   rate_space, rate_P: np.ndarray, start_rate: float,
                   horizon_years: float, jump_models: Optional[List[JumpModel]] = None,
                   cpi_P: Optional[np.ndarray] = None,
                   conditional_P: Optional[List[np.ndarray]] = None,
                   neutral_rate: Optional[float] = None,
                   rng: Optional[np.random.Generator] = None) -> SimResult:
    from ..markov.diagnostics import stationary_distribution
    spy = cfg.sim.steps_per_year
    n_paths = cfg.sim.n_paths
    n_steps = int(round(horizon_years * spy))
    dt = 1.0 / spy
    rng = rng or np.random.default_rng(cfg.sim.seed)
    neutral = start_rate if neutral_rate is None else neutral_rate
    kappa = cfg.sim.mean_reversion

    # 1) rate-state paths -> short-rate level paths.
    # The Markov chain supplies each step's increment; we de-drift it (subtract
    # the chain's long-run mean increment so a free random walk has no
    # systematic trend) and add an OU pull toward the neutral level so the
    # level stays realistic over long horizons.
    incr = np.array([_increment_bps(lab) for lab in rate_space.labels])
    e_incr = float(stationary_distribution(rate_P) @ incr)   # long-run drift/step
    start_idx = int(np.argmin(np.abs(incr)))
    state_idx, _ = _simulate_rate_states(
        rate_P, n_paths, n_steps, start_idx, rng,
        cpi_P=cpi_P, conditional_P=conditional_P)
    shocks = incr[state_idx] - e_incr           # de-drifted per-step shocks
    shocks[:, 0] = 0.0
    rate_paths = np.empty((n_paths, n_steps + 1))
    rate_paths[:, 0] = start_rate
    for s in range(1, n_steps + 1):
        prev = rate_paths[:, s - 1]
        rate_paths[:, s] = prev + shocks[:, s] - kappa * dt * (prev - neutral)
    np.clip(rate_paths, 0.0, None, out=rate_paths)

    # 2) jumps, then bound the short rate to a realistic ceiling. Without an
    # upper bound, a fat-tailed jump can push the rate arbitrarily high and the
    # reinvestment account then compounds at that rate, producing a handful of
    # explosive paths that dominate the MEAN. Central-bank rates do not run to
    # 50%, so we clip. (The robust central tendency to read is the median.)
    for jm in (jump_models or []):
        jm.apply(rate_paths, steps_per_year=spy, rng=rng)
    np.clip(rate_paths, 0.0, cfg.sim.max_short_rate, out=rate_paths)

    # 3) ladder reinvestment plan (fixed tenors -> can be precomputed)
    ladder_tenors = list(book.reinvest_ladder.keys())
    ladder_w = np.array([book.reinvest_ladder[t] for t in ladder_tenors])
    ladder_plan = _InterpPlan(curve_model.tenors, ladder_tenors)
    issue_times = [(w, bond, np.asarray(bond.cashflow_times(), dtype=float))
                   for w, bond in book.holdings]

    # 4) march forward, vectorized over paths
    wealth = np.zeros((n_paths, n_steps + 1))
    reinvest = np.zeros(n_paths)
    matured = [False] * len(issue_times)
    init_inv = None

    for s in range(n_steps + 1):
        elapsed = s * dt
        curve = curve_model.curve_from_rate(rate_paths[:, s], rng=rng)  # (n_paths, n_tenors)
        held = np.zeros(n_paths)
        coupon = np.zeros(n_paths)
        for bi, (w, bond, times0) in enumerate(issue_times):
            rem = bond.maturity_years - elapsed
            if rem > 1e-9:
                # remaining cashflows measured FROM NOW (the bond ages each step)
                rem_times = times0[times0 > elapsed + 1e-9] - elapsed
                plan = _InterpPlan(curve_model.tenors, rem_times)
                ya = plan.yields_at(curve) / 100.0           # (n_paths, n_rem_cf)
                cpn = bond.coupon_rate * bond.face / bond.freq
                disc = (1.0 + ya) ** (-rem_times[None, :])
                pv = (cpn * disc).sum(axis=1)
                # principal at remaining maturity
                pplan = _InterpPlan(curve_model.tenors, [rem])
                yT = pplan.yields_at(curve)[:, 0] / 100.0
                pv = pv + bond.face / (1.0 + yT) ** rem
                held += w * pv
                if s > 0:
                    coupon += w * bond.coupon_rate * bond.face * dt   # accrual approx
            elif not matured[bi]:
                reinvest += w * bond.face                # principal rolls in once
                matured[bi] = True

        # reinvestment compounds at the blended ladder yield
        if s > 0:
            blended = (ladder_plan.yields_at(curve) * ladder_w[None, :]).sum(axis=1) / 100.0
            reinvest = reinvest * (1.0 + blended * dt) + coupon

        wealth[:, s] = held + reinvest
        if s == 0:
            init_inv = float(np.mean(held))              # par-ish initial outlay

    terminal = wealth[:, -1]
    ret = terminal / init_inv - 1.0
    ann = (terminal / init_inv) ** (1.0 / horizon_years) - 1.0
    obs.event(channel="sim", kind="engine.run", n_paths=n_paths, n_steps=n_steps,
              horizon_years=horizon_years, init_investment=round(init_inv, 4),
              mean_ann_return=round(float(np.mean(ann)), 5))
    return SimResult(horizon_years=horizon_years, steps_per_year=spy,
                     init_investment=init_inv, terminal_wealth=terminal,
                     horizon_return=ret, annualized_return=ann,
                     wealth_paths=wealth, rate_paths=rate_paths,
                     config_hash=cfg.content_hash(), seed=cfg.sim.seed)


def _nearest(tenors, t):
    return int(np.argmin(np.abs(np.asarray(tenors) - t)))
