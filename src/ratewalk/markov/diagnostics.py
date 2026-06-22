"""Markov diagnostics: stationary distribution, out-of-sample likelihood."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..states import StateSpace


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalized to a distribution."""
    vals, vecs = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    pi = np.abs(pi)
    s = pi.sum()
    return pi / s if s > 0 else np.full(P.shape[0], 1.0 / P.shape[0])


def log_likelihood(P: np.ndarray, states: pd.Series, space: StateSpace) -> float:
    """Mean log-likelihood per transition of an observed state sequence under
    P. Used to compare models out of sample against a baseline."""
    idx = states.map({lab: i for i, lab in enumerate(space.labels)}).dropna().astype(int).values
    if len(idx) < 2:
        return float("nan")
    eps = 1e-12
    ll = sum(np.log(P[idx[k], idx[k + 1]] + eps) for k in range(len(idx) - 1))
    return float(ll / (len(idx) - 1))


def persistence_baseline_ll(states: pd.Series, space: StateSpace) -> float:
    """Baseline: 'state stays put' (identity-ish) chain, for comparison."""
    n = space.n
    P = np.eye(n) * 0.8 + (np.ones((n, n)) - np.eye(n)) * (0.2 / max(n - 1, 1))
    return log_likelihood(P, states, space)
