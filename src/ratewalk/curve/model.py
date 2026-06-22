"""State -> yield-curve mapping.

A bond is priced off the WHOLE curve, not the policy rate, so the simulator
must turn each simulated short rate into a full curve. We fit, per tenor, a
linear map

    yield_tenor = a_tenor + b_tenor * policy_rate   (+ residual noise)

by OLS on history. This captures the empirical fact that long yields move
less than one-for-one with the policy rate (b < 1 at the long end), and the
residual std lets the curve carry its own risk rather than being a
deterministic function of the short rate.

``model='nss_regression'`` uses the per-tenor regression above. A richer
Nelson-Siegel-Svensson factor model (regress level/slope/curvature on the
state) is the documented extension; the interface below does not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import obs


@dataclass
class CurveModel:
    tenors: List[float]              # in years
    a: np.ndarray                    # intercepts per tenor
    b: np.ndarray                    # slopes vs policy rate per tenor
    resid_std: np.ndarray            # residual std per tenor (percent)
    add_noise: bool = True
    noise_scale_bps: float = 15.0

    def curve_from_rate(self, policy_rate, rng=None) -> np.ndarray:
        """Vectorized: policy_rate may be a scalar or an array of shape (k,).
        Returns yields of shape (k, n_tenors) in percent."""
        r = np.atleast_1d(np.asarray(policy_rate, dtype=float))
        base = self.a[None, :] + self.b[None, :] * r[:, None]
        if self.add_noise and rng is not None:
            extra = (self.noise_scale_bps / 100.0)
            sigma = np.sqrt(self.resid_std[None, :] ** 2 + extra ** 2)
            base = base + rng.standard_normal(base.shape) * sigma
        return np.clip(base, 0.0, None)

    def yield_at(self, policy_rate: float, tenor_years: float, rng=None) -> float:
        """Single tenor (interpolated across the fitted grid)."""
        full = self.curve_from_rate(policy_rate, rng=rng)[0]
        return float(np.interp(tenor_years, self.tenors, full))


def fit_curve_model(curve_df: pd.DataFrame, policy_df: pd.DataFrame, *,
                    add_noise: bool = True, noise_scale_bps: float = 15.0
                    ) -> CurveModel:
    """Fit the per-tenor linear map from history (aligned on date)."""
    c = curve_df.copy()
    c["date"] = pd.to_datetime(c["date"])
    p = policy_df.copy()
    p["date"] = pd.to_datetime(p["date"])
    merged = pd.merge(c, p, on="date", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    tenor_cols = [col for col in c.columns if col != "date"]
    tenors = [float(t) for t in tenor_cols]
    x = np.ascontiguousarray(merged["rate"].values, dtype=float)
    A = np.ascontiguousarray(np.column_stack([np.ones_like(x), x]))
    a_list, b_list, sd_list = [], [], []
    for col in tenor_cols:
        y = np.ascontiguousarray(merged[col].values, dtype=float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A.dot(coef)
        a_list.append(coef[0]); b_list.append(coef[1])
        sd_list.append(float(np.std(resid)))
    obs.event(channel="curve", kind="fit", n_tenors=len(tenors),
              n_obs=len(merged))
    return CurveModel(tenors=tenors, a=np.array(a_list), b=np.array(b_list),
                      resid_std=np.array(sd_list), add_noise=add_noise,
                      noise_scale_bps=noise_scale_bps)
