"""Dynamic horizon policy.

You did not want to fix the investment horizon by hand, so RateWalk picks it.
``mode='dynamic'`` evaluates each candidate horizon by running the simulation
and scoring the objective (the same family used for duration), then returns the
horizon that maximizes risk-adjusted outcome. ``mode='fixed'`` just returns the
configured horizon. The selection surface is returned so the UI can show why.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .. import obs
from ..analytics.risk import objective_value, var_cvar


def select_horizon(cfg_horizon, run_fn: Callable[[float], np.ndarray]) -> Dict:
    """``run_fn(h)`` returns the annualized-return array for horizon ``h``."""
    if cfg_horizon.mode == "fixed":
        return {"mode": "fixed", "selected": float(cfg_horizon.fixed_years),
                "surface": []}

    surface: List[Dict] = []
    for h in cfg_horizon.candidate_years:
        r = run_fn(float(h))
        surface.append({
            "horizon": float(h),
            "objective": objective_value(r, cfg_horizon.objective),
            "mean_return": float(np.mean(r)),
            "CVaR95": var_cvar(r, 0.95)["CVaR"],
        })
    best = max(surface, key=lambda x: x["objective"])
    obs.event(channel="horizon", kind="select", selected=best["horizon"],
              objective=cfg_horizon.objective)
    return {"mode": "dynamic", "selected": best["horizon"],
            "objective": cfg_horizon.objective, "surface": surface}
