"""No-look-ahead invariance test (the platform's signature test).

A point-in-time accessor stamped at as_of_ts must NOT change when we append
future data. We build macro data, snapshot a state-estimation result computed
as of a date, then inject later observations and recompute. The result must be
identical: if it changed, the future leaked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ratewalk.data import load_macro
from ratewalk.states import build_rate_states, build_cpi_states, build_joint_series
from ratewalk.markov import estimate_chain


def _joint_asof(md, as_of):
    """Estimate the conditional joint series using ONLY data public at as_of."""
    pol = md.policy_asof(as_of)
    cpi = md.cpi_asof(as_of)
    rs, rspace = build_rate_states(pol, mode="increments")
    cs, cspace = build_cpi_states(cpi)
    return build_joint_series(rs, cs), rspace, cspace


def test_pit_filter_excludes_future():
    md = load_macro(country="US", source="synthetic", start="1995-01-01")
    asof = "2012-01-01"
    pol = md.policy_asof(asof)
    cpi = md.cpi_asof(asof)
    assert pol["date"].max() <= pd.Timestamp(asof)
    # CPI is filtered on the RELEASE date (publication lag honored).
    assert cpi["release_date"].max() <= pd.Timestamp(asof)


def test_estimation_invariant_to_future_data():
    md_now = load_macro(country="US", source="synthetic", start="1995-01-01", end="2012-01-01")
    md_future = load_macro(country="US", source="synthetic", start="1995-01-01", end="2024-01-01")

    asof = "2012-01-01"
    j1, rsp1, _ = _joint_asof(md_now, asof)
    j2, rsp2, _ = _joint_asof(md_future, asof)

    m1 = estimate_chain(j1.rename(columns={"rate_state": "state"}), rsp1, estimation="full")
    m2 = estimate_chain(j2.rename(columns={"rate_state": "state"}), rsp2, estimation="full")

    # Same labels, and the transition matrix estimated as-of 2012 is identical
    # whether or not 2012-2024 data exists on disk.
    assert rsp1.labels == rsp2.labels
    assert np.allclose(m1.P, m2.P, atol=1e-12), "future data leaked into a past estimate"
