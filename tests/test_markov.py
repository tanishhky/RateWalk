"""Markov estimation invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ratewalk.data import load_macro
from ratewalk.states import build_rate_states
from ratewalk.markov import estimate_chain, resample_dirichlet
from ratewalk.markov.diagnostics import (stationary_distribution, log_likelihood,
                                         persistence_baseline_ll)


def _model():
    md = load_macro(country="US", start="1995-01-01")
    rs, rspace = build_rate_states(md.policy_rate, mode="increments")
    return rs, rspace, estimate_chain(rs, rspace, estimation="exp_weighted")


def test_transition_matrix_is_row_stochastic():
    _, _, m = _model()
    assert np.allclose(m.P.sum(axis=1), 1.0, atol=1e-9)
    assert (m.P >= 0).all()


def test_stationary_distribution_is_a_distribution():
    _, _, m = _model()
    pi = stationary_distribution(m.P)
    assert abs(pi.sum() - 1.0) < 1e-6
    assert (pi >= -1e-9).all()


def test_chain_beats_persistence_baseline():
    rs, rspace, m = _model()
    ll = log_likelihood(m.P, rs["state"], rspace)
    base = persistence_baseline_ll(rs["state"], rspace)
    assert ll > base, "estimated chain should beat the naive baseline in-sample"


def test_dirichlet_draws_are_row_stochastic():
    _, _, m = _model()
    draws = resample_dirichlet(m, 10, np.random.default_rng(0))
    assert len(draws) == 10
    for P in draws:
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)
