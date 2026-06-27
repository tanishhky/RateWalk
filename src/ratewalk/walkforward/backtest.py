"""Walk-forward duration backtest.

Turns the forecaster's view into a position and measures realized P&L out of
sample, against constant-duration benchmarks. Each month, using only past
data, the conditional model gives an expected next move (probability-weighted
bps). The strategy tilts duration with that view:

    expect cuts (E[move] < 0)  -> go LONG duration  (rates fall -> bonds rally)
    expect hikes (E[move] > 0) -> go SHORT duration (limit the drawdown)

Realized return over the month is computed from the ACTUAL Treasury curve: we
hold a par bond of the chosen tenor and revalue it one month later on the
realized curve (price change + one month of carry). Benchmarks hold a fixed
tenor throughout. Everything is out of sample: the duration at month t uses
only data through t-1.

This is a deliberately simple, transparent return model (par bond, monthly
revalue, linear carry). It is enough to test whether the model's timing adds
value; it is not a full portfolio accountant.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import obs
from .forecast import WFSeries, _row_counts


def _curve_yield(curve_df: pd.DataFrame, tenor_cols: List[str], tenor_vals: np.ndarray,
                 as_of: pd.Timestamp, tenor: float) -> Optional[float]:
    """Realized par yield (percent) for `tenor` years at the last curve date
    on or before `as_of`. ``tenor_cols`` are the actual column names; their
    float values are ``tenor_vals`` (kept aligned so interpolation works
    regardless of how the columns are spelled)."""
    sub = curve_df[curve_df["date"] <= as_of]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    ys = np.array([row[c] for c in tenor_cols], dtype=float)
    return float(np.interp(tenor, tenor_vals, ys))


def _bond_total_return(y0: float, y1: float, tenor: float, dt: float) -> float:
    """Approximate one-period total return of a par bond: carry plus a
    duration/convexity price move as the yield goes y0 -> y1 (percent)."""
    y0d, y1d = y0 / 100.0, y1 / 100.0
    # modified duration and convexity of a par bond, closed-form-ish proxies
    dur = (1 - (1 + y0d) ** (-tenor)) / y0d if y0d > 1e-6 else tenor
    conv = tenor * (tenor + 1) / (1 + y0d) ** 2
    dy = y1d - y0d
    price_ret = -dur * dy + 0.5 * conv * dy ** 2
    carry = y0d * dt
    return carry + price_ret


def duration_backtest(s: WFSeries, curve_df: pd.DataFrame, tenors=None, *,
                      min_train: int = 120, short_tenor: float = 2.0,
                      long_tenor: float = 10.0, benchmarks=(2.0, 10.0),
                      prior: float = 1.0) -> Dict:
    """Run the duration-timing strategy and the benchmarks month by month."""
    tenor_cols = [c for c in curve_df.columns if c != "date"]
    tenor_vals = np.array([float(c) for c in tenor_cols])
    dates = pd.to_datetime(s.dates)
    from_arr, to_arr, reg_arr = s.state[:-1], s.state[1:], s.regime[:-1]
    n_states = s.rate_space.n

    strat_ret, bench_ret = [], {b: [] for b in benchmarks}
    chosen, eval_dates = [], []

    for i in range(min_train, len(s.state) - 1):
        t0, t1 = dates[i], dates[i + 1]
        dt = max((t1 - t0).days / 365.25, 1e-6)
        # model view as of t0, past-only
        q_from, q_reg = s.state[i - 1], s.regime[i - 1]
        counts = _row_counts(from_arr[:i - 1], to_arr[:i - 1], reg_arr[:i - 1],
                             q_from=q_from, q_reg=q_reg, n_states=n_states)
        probs = (counts + prior) / (counts + prior).sum()
        exp_bps = float(probs @ s.incr_bps)
        tenor = long_tenor if exp_bps < 0 else short_tenor  # tilt with the view

        y0 = _curve_yield(curve_df, tenor_cols, tenor_vals, t0, tenor)
        y1 = _curve_yield(curve_df, tenor_cols, tenor_vals, t1, tenor)
        if y0 is None or y1 is None:
            continue
        strat_ret.append(_bond_total_return(y0, y1, tenor, dt))
        chosen.append(tenor)
        eval_dates.append(str(t0.date()))
        for b in benchmarks:
            by0 = _curve_yield(curve_df, tenor_cols, tenor_vals, t0, b)
            by1 = _curve_yield(curve_df, tenor_cols, tenor_vals, t1, b)
            bench_ret[b].append(_bond_total_return(by0, by1, b, dt)
                                if by0 is not None and by1 is not None else 0.0)

    def stats(rets):
        r = np.asarray(rets)
        if len(r) == 0:
            return {"n": 0}
        ann = 12  # monthly steps
        cum = float(np.prod(1 + r) - 1)
        mean_ann = float(np.mean(r) * ann)
        vol_ann = float(np.std(r, ddof=1) * np.sqrt(ann)) if len(r) > 1 else 0.0
        sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
        eq = np.cumprod(1 + r)
        dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
        return {"n": len(r), "cum_return": round(cum, 4),
                "ann_return": round(mean_ann, 4), "ann_vol": round(vol_ann, 4),
                "sharpe": round(sharpe, 3), "max_drawdown": round(dd, 4)}

    result = {
        "eval_start": eval_dates[0] if eval_dates else None,
        "eval_end": eval_dates[-1] if eval_dates else None,
        "n_months": len(strat_ret),
        "strategy": stats(strat_ret),
        "benchmarks": {f"{b}y": stats(bench_ret[b]) for b in benchmarks},
        "pct_long": round(float(np.mean(np.array(chosen) == long_tenor)), 3) if chosen else None,
        "equity_curves": {
            "dates": eval_dates,
            "strategy": np.cumprod(1 + np.array(strat_ret)).round(4).tolist(),
            **{f"{b}y": np.cumprod(1 + np.array(bench_ret[b])).round(4).tolist() for b in benchmarks},
        },
    }
    obs.event(channel="walkforward", kind="duration_backtest",
              n_months=result["n_months"],
              strat_sharpe=result["strategy"].get("sharpe"))
    return result
