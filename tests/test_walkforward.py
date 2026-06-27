"""Walk-forward forecaster: correctness + the no-look-ahead guarantee.

The headline invariant: a prediction stamped at month t must not change when
future months are appended to the data. If it does, the walk-forward loop is
leaking the future into the past.
"""
from __future__ import annotations

import numpy as np

from ratewalk import config as cfg
from ratewalk.data import load_macro
from ratewalk.walkforward import (prepare_series, walk_forward_forecast,
                                  score_forecasts, compare_models, nowcast)


def _series(end):
    c = cfg.load()
    md = load_macro(country="US", source="synthetic", start="1990-01-01", end=end)
    return prepare_series(md, c), c


def test_forecast_rows_are_probability_distributions():
    s, _ = _series("2015-01-01")
    df = walk_forward_forecast(s, model="conditional", min_train=60,
                               n_dirichlet=50, rng=np.random.default_rng(0))
    assert len(df) > 0
    for p in df["probs"]:
        assert abs(p.sum() - 1.0) < 1e-9
        assert (p >= 0).all()


def test_walk_forward_is_no_lookahead():
    """A forecast made at a past month must be identical whether or not future
    months exist in the data. Tested by truncating ONE fixed series' arrays
    (so the comparison is not confounded by the data generator), which directly
    exercises the loop's past-only counting."""
    import dataclasses
    s_full, _ = _series("2024-01-01")
    T = len(s_full.state) - 50                      # cut off the last 50 months
    s_trunc = dataclasses.replace(
        s_full, dates=s_full.dates[:T], state=s_full.state[:T], regime=s_full.regime[:T])

    df_full = walk_forward_forecast(s_full, model="conditional", min_train=60,
                                    n_dirichlet=1, rng=np.random.default_rng(0))
    df_trunc = walk_forward_forecast(s_trunc, model="conditional", min_train=60,
                                     n_dirichlet=1, rng=np.random.default_rng(0))
    # every eval point in the truncated run must match the full run exactly:
    # the future 50 months cannot have changed any earlier prediction.
    merged = df_trunc.merge(df_full, on="date", suffixes=("_t", "_f"))
    assert len(merged) > 100
    for _, row in merged.iterrows():
        assert np.allclose(row["probs_t"], row["probs_f"], atol=1e-12), (
            f"future data leaked into the forecast at {row['date']}")


def test_compare_models_and_scoring():
    s, _ = _series("2015-01-01")
    out = compare_models(s, min_train=60, n_dirichlet=50, rng=np.random.default_rng(0))
    for m in ("climatology", "unconditional", "conditional"):
        assert out[m]["n"] > 0
        assert 0.0 <= out[m]["accuracy"] <= 1.0
        assert out[m]["mean_log_loss"] > 0
    assert "chain_beats_climatology" in out["summary"]


def test_nowcast_is_a_distribution_with_ci():
    s, _ = _series("2024-01-01")
    nc = nowcast(s, model="conditional", n_dirichlet=500)
    probs = [d["prob"] for d in nc["distribution"]]
    assert abs(sum(probs) - 1.0) < 1e-3        # probs are rounded to 4dp
    for d in nc["distribution"]:
        lo, hi = d["ci"]
        assert 0.0 <= lo <= hi <= 1.0          # a valid, ordered confidence band
        assert 0.0 <= d["prob"] <= 1.0
