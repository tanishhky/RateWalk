"""Simulation engine invariants: reproducibility, horizon-stability, jumps."""
from __future__ import annotations

import dataclasses

import numpy as np

from ratewalk import config as cfg
from ratewalk.data import load_macro
from ratewalk.states import build_rate_states
from ratewalk.markov import estimate_chain
from ratewalk.curve import fit_curve_model
from ratewalk.instruments.book import build_book
from ratewalk.sim import build_jump_models, run_simulation


def _ctx(n_paths=1000):
    c = cfg.load()
    c = dataclasses.replace(c, sim=dataclasses.replace(c.sim, n_paths=n_paths))
    md = load_macro(country="US", source="synthetic", start="1995-01-01")
    rs, rspace = build_rate_states(md.policy_rate, mode=c.state.rate_mode,
                                   increment_grid_bps=c.state.increment_grid_bps)
    cm = fit_curve_model(md.curve, md.policy_rate)
    start = float(md.policy_rate["rate"].iloc[-1])
    neutral = float(md.policy_rate["rate"].mean())
    book = build_book(c.instrument, lambda t: cm.yield_at(start, t))
    m = estimate_chain(rs, rspace, half_life_years=c.markov.half_life_years)
    return c, cm, rspace, m, book, start, neutral


def _run(c, cm, rspace, m, book, start, neutral, horizon, seed=42, jumps=True):
    jm = build_jump_models(c.sim.jumps) if jumps else []
    return run_simulation(c, curve_model=cm, book=book, rate_space=rspace,
                          rate_P=m.P, start_rate=start, horizon_years=horizon,
                          jump_models=jm, neutral_rate=neutral,
                          rng=np.random.default_rng(seed))


def test_reproducible_under_same_seed():
    ctx = _ctx()
    a = _run(*ctx, horizon=10.0, seed=7)
    b = _run(*ctx, horizon=10.0, seed=7)
    assert np.allclose(a.terminal_wealth, b.terminal_wealth)


def test_returns_are_horizon_stable():
    """Annualized return should not systematically blow up with horizon (the
    mean-reversion anchor prevents the random-walk drift artifact)."""
    ctx = _ctx(n_paths=1500)
    med5 = np.median(_run(*ctx, horizon=5.0).annualized_return)
    med30 = np.median(_run(*ctx, horizon=30.0).annualized_return)
    assert abs(med30 - med5) < 0.03, f"horizon drift too large: {med5:.3f} vs {med30:.3f}"
    # and both are in a plausible band for a high-grade bond
    assert 0.0 < med30 < 0.12


def test_jumps_inject_rate_dispersion():
    """Turning jumps on must add dispersion to the simulated rate paths (the
    whole point of keeping the outliers). The effect on horizon returns is
    deliberately damped because jumps are transient and decay, but the rate
    paths themselves must be measurably more dispersed."""
    ctx = _ctx(n_paths=2500)
    no_j = _run(*ctx, horizon=20.0, jumps=False).rate_paths[:, -1]
    with_j = _run(*ctx, horizon=20.0, jumps=True).rate_paths[:, -1]
    assert with_j.std() > no_j.std(), "jumps should add rate-path dispersion"
