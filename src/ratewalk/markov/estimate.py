"""Transition-matrix estimation.

Models (config-selectable):
  * rate        : univariate chain over rate states.
  * cpi         : univariate chain over CPI regimes.
  * conditional : P(rate_{t+1} | rate_t, cpi_regime_t). The CPI regime
                  modulates the rate transition matrix (a Markov-switching
                  structure). This is where the macro signal lives.

Estimation weighting:
  * full        : equal weight on all transitions.
  * exp_weighted: recent transitions weighted up (half-life in years) so the
                  matrix reflects the current regime, not a 60-year average
                  that resembles no actual era.

Regularization: a Dirichlet/Laplace prior keeps sparse rows well-defined.

Uncertainty: ``resample_dirichlet`` draws transition matrices from the
posterior Dirichlet so every downstream metric can carry confidence bands
(this is the basis of the transition-sensitivity analysis).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import obs
from ..states import StateSpace


@dataclass
class TransitionModel:
    """An estimated chain: row-stochastic matrix P over a labeled state space,
    plus the raw transition counts (for resampling) and the conditioning
    context if any."""
    space: StateSpace
    P: np.ndarray                 # (n, n) row-stochastic
    counts: np.ndarray            # (n, n) integer-ish transition counts
    context: Optional[str] = None # e.g. cpi regime label for conditional chains

    def sample_next(self, state_idx: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.space.n, p=self.P[state_idx]))


def _weighted_counts(seq_idx: np.ndarray, n: int, *, weights: Optional[np.ndarray]) -> np.ndarray:
    counts = np.zeros((n, n), dtype=float)
    for k in range(len(seq_idx) - 1):
        i, j = seq_idx[k], seq_idx[k + 1]
        w = 1.0 if weights is None else weights[k]
        counts[i, j] += w
    return counts


def _exp_weights(dates: pd.Series, half_life_years: float) -> np.ndarray:
    d = pd.to_datetime(dates).values.astype("datetime64[D]").astype(float)
    age_years = (d.max() - d) / 365.25
    lam = np.log(2.0) / max(half_life_years, 1e-6)
    return np.exp(-lam * age_years)


def _normalize(counts: np.ndarray, *, prior: str, prior_strength: float) -> np.ndarray:
    c = counts.copy()
    if prior in ("dirichlet", "laplace"):
        c = c + prior_strength
    row = c.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return c / row


def estimate_chain(states: pd.DataFrame, space: StateSpace, *,
                   state_col: str = "state", date_col: str = "date",
                   estimation: str = "exp_weighted", half_life_years: float = 8.0,
                   prior: str = "dirichlet", prior_strength: float = 1.0) -> TransitionModel:
    """Estimate a univariate transition model over ``space``."""
    df = states.dropna(subset=[state_col]).reset_index(drop=True)
    idx = df[state_col].map({lab: i for i, lab in enumerate(space.labels)}).values
    if estimation == "exp_weighted":
        w = _exp_weights(df[date_col], half_life_years)
    else:
        w = None
    counts = _weighted_counts(idx.astype(int), space.n, weights=w)
    P = _normalize(counts, prior=prior, prior_strength=prior_strength)
    obs.event(channel="markov", kind="estimate.chain", n_states=space.n,
              n_transitions=int(len(df) - 1), estimation=estimation)
    return TransitionModel(space=space, P=P, counts=counts)


def estimate_conditional(joint: pd.DataFrame, rate_space: StateSpace,
                         cpi_space: StateSpace, *,
                         estimation: str = "exp_weighted", half_life_years: float = 8.0,
                         prior: str = "dirichlet", prior_strength: float = 1.0
                         ) -> Dict[str, TransitionModel]:
    """One rate-transition model per CPI regime: P(rate' | rate, cpi).

    Returns {cpi_label: TransitionModel}. Regimes never observed fall back to
    the pooled (unconditional) rate chain so simulation never hits a hole.
    """
    pooled = estimate_chain(
        joint.rename(columns={"rate_state": "state"}), rate_space,
        estimation=estimation, half_life_years=half_life_years,
        prior=prior, prior_strength=prior_strength)

    models: Dict[str, TransitionModel] = {}
    for cpi_label in cpi_space.labels:
        sub = joint[joint["cpi_state"] == cpi_label]
        if len(sub) < 3:
            models[cpi_label] = TransitionModel(
                space=rate_space, P=pooled.P.copy(), counts=pooled.counts.copy(),
                context=cpi_label)
            continue
        sub2 = sub.rename(columns={"rate_state": "state"})
        m = estimate_chain(sub2, rate_space, estimation=estimation,
                           half_life_years=half_life_years, prior=prior,
                           prior_strength=prior_strength)
        m.context = cpi_label
        models[cpi_label] = m
    obs.event(channel="markov", kind="estimate.conditional",
              n_regimes=len(models), n_states=rate_space.n)
    return models


def resample_dirichlet(model: TransitionModel, n_draws: int,
                       rng: np.random.Generator) -> List[np.ndarray]:
    """Draw ``n_draws`` transition matrices from the row-wise Dirichlet
    posterior implied by the counts. Each draw is a plausible 'true' P given
    finite data; propagating these gives confidence bands on every metric."""
    draws: List[np.ndarray] = []
    alpha = model.counts + 1.0    # Dirichlet posterior with a flat prior
    for _ in range(n_draws):
        P = np.vstack([rng.dirichlet(alpha[i]) for i in range(model.space.n)])
        draws.append(P)
    return draws
