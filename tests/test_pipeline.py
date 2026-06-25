"""End-to-end pipeline integration: a complete report on synthetic data."""
from __future__ import annotations

import dataclasses

from ratewalk import config as cfg
from ratewalk.cli import run_pipeline


def _fast_cfg():
    c = cfg.load()
    c = dataclasses.replace(c, data=dataclasses.replace(c.data, source="synthetic"))
    c = dataclasses.replace(c, sim=dataclasses.replace(c.sim, n_paths=600))
    c = dataclasses.replace(c, analytics=dataclasses.replace(c.analytics, sensitivity_draws=20))
    return c


def test_pipeline_produces_complete_report():
    r = run_pipeline(_fast_cfg())
    for key in ("markov", "horizon_selection", "headline", "risk",
                "duration_grid", "transition_sensitivity", "credit", "viz"):
        assert key in r, f"missing report section: {key}"
    # headline sanity
    h = r["headline"]["annualized_return_pct"]
    assert h["p5"] <= h["p50"] <= h["p95"]
    # viz payloads present and shaped
    v = r["viz"]
    assert len(v["fan_chart"]["t_years"]) > 1
    P = v["transition_matrix"]["P"]
    assert len(P) == len(v["transition_matrix"]["labels"])
    assert len(v["return_histogram"]["centers"]) == len(v["return_histogram"]["counts"])


def test_pipeline_respects_country_override():
    c = dataclasses.replace(_fast_cfg(), country="GB")
    # synthetic source -> no network; just confirms country flows through
    r = run_pipeline(c)
    assert r["country"] == "GB"
