"""Analytics + credit overlay invariants."""
from __future__ import annotations

import numpy as np

from ratewalk.analytics import var_cvar, distribution_moments, objective_value
from ratewalk.analytics.risk import fit_gmm


def test_var_cvar_ordering():
    rng = np.random.default_rng(0)
    r = rng.normal(0.03, 0.05, 50000)
    tc = var_cvar(r, 0.95)
    # CVaR (mean tail loss) is at least as severe as VaR
    assert tc["CVaR"] >= tc["VaR"] - 1e-9


def test_moments_detect_non_gaussian():
    rng = np.random.default_rng(0)
    # a heavy-tailed mixture should be flagged non-gaussian
    r = np.concatenate([rng.normal(0.03, 0.01, 9000), rng.normal(-0.2, 0.05, 1000)])
    m = distribution_moments(r)
    assert not m["is_gaussian_5pct"]
    gmm = fit_gmm(r, max_components=3)
    assert gmm["n_components"] >= 2


def test_objective_is_finite_for_riskless():
    # near-riskless positive returns must not blow the objective up
    r = np.full(1000, 0.03) + np.random.default_rng(0).normal(0, 1e-4, 1000)
    v = objective_value(r, "cvar_adjusted_return")
    assert np.isfinite(v) and v > 0


def test_credit_overlay_reduces_wealth():
    import dataclasses
    from ratewalk import config as cfg
    from ratewalk.credit import apply_credit_overlay

    class _Res:
        init_investment = 100.0
        terminal_wealth = np.full(5000, 140.0)
        annualized_return = np.full(5000, 0.034)

    c = cfg.load()
    credit_on = dataclasses.replace(c.credit, enabled=True,
                                    annual_default_prob=0.05, recovery_rate=0.4)
    out = apply_credit_overlay(_Res(), credit_on, 10.0, 0.033,
                               np.random.default_rng(1))
    assert out.n_defaults > 0
    assert out.terminal_wealth.mean() < 140.0
