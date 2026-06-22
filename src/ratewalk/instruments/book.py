"""Instrument book: the held position(s).

v1 supports three ``kind`` values from config, all behind one interface so the
simulator does not care which it holds:
  * single_bond : one held bond (the default).
  * ladder      : equal/weighted rungs across tenors (held + reinvestment).
  * portfolio   : arbitrary tenor -> weight map.

This is the extension point you asked for: the same engine prices a single
Treasury today and a multi-rung ladder or a sovereign portfolio tomorrow by
swapping config, not code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .bond import Bond, par_coupon


@dataclass
class InstrumentBook:
    """A weighted set of bonds plus the reinvestment tenor ladder."""
    holdings: List[tuple]            # list of (weight, Bond)
    reinvest_ladder: Dict[float, float]  # tenor_years -> weight (sums to 1)
    country: str = "US"

    def total_weight(self) -> float:
        return sum(w for w, _ in self.holdings)


def build_book(cfg_instrument, curve_yield_fn: Callable[[float], float],
               country: str = "US") -> InstrumentBook:
    """Construct the held book at issue, setting par coupons from the curve."""
    ladder = {float(k): float(v) for k, v in dict(cfg_instrument.reinvestment_ladder).items()}
    s = sum(ladder.values()) or 1.0
    ladder = {k: v / s for k, v in ladder.items()}

    holdings: List[tuple] = []
    if cfg_instrument.kind == "single_bond":
        tenor = float(cfg_instrument.held_tenor_years)
        cpn = (cfg_instrument.coupon_rate
               if cfg_instrument.coupon_rate is not None
               else par_coupon(curve_yield_fn, tenor, face=cfg_instrument.face))
        holdings.append((1.0, Bond(maturity_years=tenor, coupon_rate=cpn,
                                   face=cfg_instrument.face, country=country)))
    elif cfg_instrument.kind in ("ladder", "portfolio"):
        weights = (ladder if cfg_instrument.kind == "ladder"
                   else {float(k): float(v) for k, v in dict(cfg_instrument.portfolio_tenors).items()})
        sw = sum(weights.values()) or 1.0
        for tenor, w in weights.items():
            cpn = par_coupon(curve_yield_fn, tenor, face=cfg_instrument.face)
            holdings.append((w / sw, Bond(maturity_years=tenor, coupon_rate=cpn,
                                          face=cfg_instrument.face, country=country)))
    else:
        raise ValueError(f"unknown instrument kind {cfg_instrument.kind!r}")

    return InstrumentBook(holdings=holdings, reinvest_ladder=ladder, country=country)
