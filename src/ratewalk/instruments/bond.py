"""Generic coupon bond: pricing, par coupon, duration.

Pricing discounts each cashflow at the curve yield for its own maturity (a
zero-style discount read off the simulated curve), so a held bond is repriced
each step as the curve moves. The instrument is sovereign-agnostic: a
``country`` and an optional credit hazard make the same object usable for US
Treasuries, EM sovereigns, corporates, or munis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Bond:
    maturity_years: float
    coupon_rate: float            # annual, as a fraction (0.04 = 4%)
    face: float = 100.0
    freq: int = 2                 # coupons per year
    country: str = "US"

    def cashflow_times(self, t0: float = 0.0):
        """Coupon/principal times (in years from t0) still outstanding."""
        n = int(round(self.maturity_years * self.freq))
        times = [(k + 1) / self.freq for k in range(n)]
        return [t for t in times if t > t0 + 1e-9]


def _yield_for(curve_yield_fn: Callable[[float], float], t: float) -> float:
    return max(curve_yield_fn(t), 0.0) / 100.0


def price_bond(bond: Bond, curve_yield_fn: Callable[[float], float], *,
               elapsed_years: float = 0.0) -> float:
    """Present value of remaining cashflows. ``curve_yield_fn(t)`` returns the
    annualized yield (percent) for maturity ``t`` years."""
    rem = bond.maturity_years - elapsed_years
    if rem <= 0:
        return 0.0
    coupon = bond.coupon_rate * bond.face / bond.freq
    pv = 0.0
    for t in bond.cashflow_times(t0=0.0):
        if t > rem + 1e-9:
            continue
        y = _yield_for(curve_yield_fn, t)
        pv += coupon / (1.0 + y) ** t
    # principal at remaining maturity
    yT = _yield_for(curve_yield_fn, rem)
    pv += bond.face / (1.0 + yT) ** rem
    return pv


def par_coupon(curve_yield_fn: Callable[[float], float], maturity_years: float,
               face: float = 100.0, freq: int = 2) -> float:
    """Coupon rate that prices the bond at par given the curve."""
    times = [(k + 1) / freq for k in range(int(round(maturity_years * freq)))]
    disc = [1.0 / (1.0 + _yield_for(curve_yield_fn, t)) ** t for t in times]
    yT = _yield_for(curve_yield_fn, maturity_years)
    disc_principal = 1.0 / (1.0 + yT) ** maturity_years
    annuity = sum(disc) / freq
    if annuity <= 0:
        return 0.0
    return (1.0 - disc_principal) / annuity


def macaulay_duration(bond: Bond, curve_yield_fn: Callable[[float], float], *,
                      elapsed_years: float = 0.0) -> float:
    """Macaulay duration (years), PV-weighted average cashflow time."""
    rem = bond.maturity_years - elapsed_years
    if rem <= 0:
        return 0.0
    coupon = bond.coupon_rate * bond.face / bond.freq
    num = 0.0
    price = price_bond(bond, curve_yield_fn, elapsed_years=elapsed_years)
    if price <= 0:
        return 0.0
    for t in bond.cashflow_times():
        if t > rem + 1e-9:
            continue
        y = _yield_for(curve_yield_fn, t)
        cf = coupon
        num += t * cf / (1.0 + y) ** t
    yT = _yield_for(curve_yield_fn, rem)
    num += rem * bond.face / (1.0 + yT) ** rem
    return num / price
