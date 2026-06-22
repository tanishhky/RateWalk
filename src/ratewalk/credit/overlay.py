"""Credit / payment-risk overlay.

Generic and sovereign-agnostic by design: the same code models a US Treasury
technical-default (debt-ceiling delay), an EM sovereign default with recovery,
or a corporate/muni coupon miss. It operates on a SimResult's wealth at the
terminal step by applying, per path, payment-risk haircuts:

  * default      : with annual probability p_d, the issuer defaults at a random
                   time; the position recovers ``recovery_rate`` of remaining
                   value and stops accruing.
  * coupon delay : each coupon is delayed with probability p_delay (time value
                   of the delayed cash is lost).
  * coupon miss  : each coupon is missed outright with probability p_miss.

For true US Treasuries set p_d = 0 and use the delay channel for debt-ceiling
stress. For other issuers, raise p_d and lower recovery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import obs


@dataclass
class CreditResult:
    terminal_wealth: np.ndarray
    annualized_return: np.ndarray
    n_defaults: int
    loss_vs_riskfree: float          # mean wealth reduction from the overlay


def apply_credit_overlay(sim_result, cfg_credit, horizon_years: float,
                         coupon_rate: float, rng: np.random.Generator) -> CreditResult:
    """Apply payment risk to a (risk-free) SimResult, returning haircut wealth."""
    tw = sim_result.terminal_wealth.copy()
    n = len(tw)
    if not cfg_credit.enabled:
        return CreditResult(tw, sim_result.annualized_return, 0, 0.0)

    base_mean = float(np.mean(tw))

    # Default channel: Bernoulli over the horizon (annual prob compounded).
    p_default_horizon = 1.0 - (1.0 - cfg_credit.annual_default_prob) ** horizon_years
    defaulted = rng.random(n) < p_default_horizon
    # default time uniform over horizon -> fraction of value already accrued
    frac = rng.uniform(0.0, 1.0, n)
    tw[defaulted] = (sim_result.init_investment
                     + (tw[defaulted] - sim_result.init_investment) * frac[defaulted]) \
        * cfg_credit.recovery_rate

    # Coupon delay / miss channels: approximate as a fractional income haircut.
    # Expected number of coupons over the horizon ~ horizon * freq(=2).
    exp_coupons = horizon_years * 2.0
    coupon_value = coupon_rate * sim_result.init_investment / 2.0
    miss_loss = cfg_credit.coupon_miss_prob * exp_coupons * coupon_value
    delay_loss = cfg_credit.coupon_delay_prob * exp_coupons * coupon_value * 0.1  # ~10% TV loss
    tw = tw - miss_loss - delay_loss

    ann = (np.maximum(tw, 1e-9) / sim_result.init_investment) ** (1.0 / horizon_years) - 1.0
    res = CreditResult(terminal_wealth=tw, annualized_return=ann,
                       n_defaults=int(defaulted.sum()),
                       loss_vs_riskfree=base_mean - float(np.mean(tw)))
    obs.event(channel="credit", kind="overlay", n_defaults=res.n_defaults,
              mean_loss=round(res.loss_vs_riskfree, 4))
    return res


def single_coupon_sensitivity(sim_result, coupon_rate: float, horizon_years: float,
                              which_coupon: int = 1) -> float:
    """Impact (in mean terminal wealth) of missing exactly ONE specific coupon.

    Answers 'sensitivity to any single/individual coupon miss': the present
    value of one coupon as a fraction of terminal wealth, by coupon index."""
    coupon_value = coupon_rate * sim_result.init_investment / 2.0
    # the k-th coupon, reinvested for the remaining horizon at a nominal blended
    # rate (approximate with the realized annualized return per path)
    remaining_years = max(horizon_years - which_coupon / 2.0, 0.0)
    growth = (1.0 + np.maximum(sim_result.annualized_return, -0.99)) ** remaining_years
    lost = coupon_value * growth
    return float(np.mean(lost))
