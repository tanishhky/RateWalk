"""Volatility allocation study.

Two sizing policies compared on the same simulated ensemble, answering: do we
hold risk constant, or lean into high-confidence signals?

  * constant_risk    : scale exposure so realized portfolio vol hits a target
                       (vol targeting).
  * confidence_scaled: scale exposure UP when the signal is high-confidence,
                       where confidence is the inverse width of the transition
                       confidence interval (tight band -> strong conviction ->
                       bigger position). A Kelly-flavored 'double down'.

Both are applied to the base return distribution as leverage multipliers, then
compared on risk-adjusted return and tail metrics.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .. import obs
from ..analytics.risk import var_cvar


def _apply_leverage(returns: np.ndarray, lev: float) -> np.ndarray:
    return returns * lev


def compare_policies(base_returns: np.ndarray, *, target_annual_vol: float,
                     horizon_years: float, signal_confidence: float,
                     confidence_scaling: bool = True) -> Dict:
    """signal_confidence in [0,1]: 1 = very tight transition CI (strong)."""
    r = np.asarray(base_returns)
    realized_vol = float(np.std(r, ddof=1)) / np.sqrt(max(horizon_years, 1e-6))

    # constant-risk leverage
    lev_const = target_annual_vol / realized_vol if realized_vol > 1e-9 else 1.0
    lev_const = float(np.clip(lev_const, 0.0, 5.0))

    # confidence-scaled leverage: base 1.0, scaled by conviction up to 2x
    lev_conf = 1.0 + (signal_confidence if confidence_scaling else 0.0)
    lev_conf = float(np.clip(lev_conf, 0.0, 5.0))

    def summary(lev):
        x = _apply_leverage(r, lev)
        sd = np.std(x, ddof=1)
        return {
            "leverage": round(lev, 3),
            "mean_return": float(np.mean(x)),
            "sharpe": float(np.mean(x) / sd) if sd > 0 else 0.0,
            "VaR95": var_cvar(x, 0.95)["VaR"],
            "CVaR95": var_cvar(x, 0.95)["CVaR"],
        }

    out = {
        "realized_annual_vol": realized_vol,
        "signal_confidence": signal_confidence,
        "constant_risk": summary(lev_const),
        "confidence_scaled": summary(lev_conf),
        "unlevered": summary(1.0),
    }
    # recommendation: prefer the policy with the higher CVaR-adjusted return
    def cadj(s):
        return s["mean_return"] / s["CVaR95"] if s["CVaR95"] > 1e-9 else -np.inf
    out["recommended"] = ("confidence_scaled"
                          if cadj(out["confidence_scaled"]) > cadj(out["constant_risk"])
                          else "constant_risk")
    obs.event(channel="vol", kind="compare_policies",
              recommended=out["recommended"])
    return out
