"""Market-implied benchmark: correctness + the no-look-ahead guarantee.

Same discipline as test_walkforward.py: predictions stamped at period t must
be bit-identical whether or not later periods exist; the grid-integration
mapping is checked against a closed-form case; and on synthetic data whose
moves are DRIVEN by the signal, the market model must beat climatology
(power check - the calibration can actually find a real signal).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ratewalk import config as cfg
from ratewalk.data import load_macro
from ratewalk.walkforward import (prepare_series, walk_forward_forecast,
                                  score_forecasts, market_signal,
                                  walk_forward_market, blend_adaptive)
from ratewalk.walkforward.market_benchmark import _grid_bin_probs


def _series(end):
    c = cfg.load()
    md = load_macro(country="US", source="synthetic", start="1990-01-01", end=end)
    return md, prepare_series(md, c)


def test_grid_bin_probs_closed_form():
    """Tight normal centered exactly on one grid value puts ~all mass there,
    and the bins always sum to 1."""
    grid = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
    p = _grid_bin_probs(mu=25.0, sigma=0.5, incr_bps=grid)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[3] > 0.999
    # symmetric wide normal at 0 -> symmetric mass on +/- moves
    p2 = _grid_bin_probs(mu=0.0, sigma=30.0, incr_bps=grid)
    assert abs(p2[1] - p2[3]) < 1e-9 and abs(p2[0] - p2[4]) < 1e-9


def test_market_forecast_rows_are_distributions():
    md, s = _series("2015-01-01")
    sig = market_signal(md, s)
    df = walk_forward_market(s, sig, min_train=60)
    assert len(df) > 0
    for p in df["probs"]:
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p >= 0).all()


def test_market_forecast_is_no_lookahead():
    """Truncate the series + signal arrays: earlier predictions must not
    change. Direct future-data-injection test on the calibration loop."""
    md, s_full = _series("2024-01-01")
    sig_full = market_signal(md, s_full)
    T = len(s_full.state) - 50
    s_trunc = dataclasses.replace(
        s_full, dates=s_full.dates[:T], state=s_full.state[:T],
        regime=s_full.regime[:T])

    df_full = walk_forward_market(s_full, sig_full, min_train=60)
    df_trunc = walk_forward_market(s_trunc, sig_full[:T], min_train=60)
    merged = df_trunc.merge(df_full, on="date", suffixes=("_t", "_f"))
    assert len(merged) > 100
    for _, row in merged.iterrows():
        assert np.allclose(row["probs_t"], row["probs_f"], atol=1e-12), (
            f"future data leaked into the market forecast at {row['date']}")


def test_market_model_finds_a_real_signal():
    """Power check on a rigged series: when next-period moves are literally
    a linear function of the signal plus small noise, the walk-forward
    calibration must recover it and beat move-frequency climatology."""
    rng = np.random.default_rng(42)
    md, s = _series("2020-01-01")
    n = len(s.state)
    grid = s.incr_bps
    # Rig: outcome at index i is a noisy linear function of signal[i-1],
    # snapped to the move grid. walk_forward_market fits pairs
    # (signal[k], move[k+1]) and predicts i from signal[i-1], so it should
    # recover this relationship almost perfectly.
    sig = rng.normal(0.0, 20.0, n)
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        target = sig[i - 1] + rng.normal(0.0, 6.0)
        state[i] = int(np.argmin(np.abs(grid - target)))
    s_rig = dataclasses.replace(s, state=state)

    df = walk_forward_market(s_rig, sig, min_train=60)
    sc_mkt = score_forecasts(df, s.rate_space.n)
    df_clim = walk_forward_forecast(s_rig, model="climatology", min_train=60,
                                    n_dirichlet=1, rng=np.random.default_rng(0))
    sc_clim = score_forecasts(df_clim, s.rate_space.n)
    assert sc_mkt["mean_log_loss"] < sc_clim["mean_log_loss"], (
        f"market calibration failed to exploit a planted signal: "
        f"{sc_mkt['mean_log_loss']} vs climatology {sc_clim['mean_log_loss']}")


def test_blend_is_no_lookahead_and_aligned():
    md, s = _series("2018-01-01")
    sig = market_signal(md, s)
    df_m = walk_forward_market(s, sig, min_train=60)
    df_c = walk_forward_forecast(s, model="unconditional", min_train=60,
                                 n_dirichlet=1, rng=np.random.default_rng(0))
    df_b = blend_adaptive(df_m, df_c, s.rate_space.n)
    assert len(df_b) == len(df_m)
    for p in df_b["probs"]:
        assert abs(p.sum() - 1.0) < 1e-9
    # warmup rows are exactly 50/50
    assert (df_b["w_a"].iloc[:24] == 0.5).all()
    # blend weights at row i depend only on rows < i: recompute on a prefix
    k = len(df_b) - 30
    df_b_prefix = blend_adaptive(df_m.iloc[:k].reset_index(drop=True),
                                 df_c.iloc[:k].reset_index(drop=True),
                                 s.rate_space.n)
    assert np.allclose(df_b_prefix["w_a"].to_numpy(),
                       df_b["w_a"].to_numpy()[:k], atol=1e-12)
