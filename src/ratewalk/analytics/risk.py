"""Risk analytics on a simulated return distribution.

All functions are pure on numpy arrays, except the two orchestration helpers
(``duration_grid_search``, ``transition_sensitivity``) which take a ``run_fn``
closure so the caller controls path counts and wiring.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from scipy import stats

from .. import obs


# ── Tail risk ───────────────────────────────────────────────────────────────

def var_cvar(returns: np.ndarray, level: float = 0.95) -> Dict[str, float]:
    """Historical VaR and CVaR (Expected Shortfall) as POSITIVE loss numbers.

    VaR_level is the loss not exceeded with probability ``level``; CVaR is the
    mean loss in the worst ``1-level`` tail."""
    losses = -np.asarray(returns)
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return {"level": level, "VaR": var, "CVaR": cvar}


def distribution_moments(returns: np.ndarray) -> Dict[str, float]:
    r = np.asarray(returns)
    jb_stat, jb_p = stats.jarque_bera(r)
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),    # Fisher (excess)
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "is_gaussian_5pct": bool(jb_p > 0.05),
    }


def fit_gmm(returns: np.ndarray, max_components: int = 4) -> Dict:
    """Fit a Gaussian Mixture, choosing the number of components by BIC.

    When the residual distribution is non-Gaussian (it will be, once jumps are
    on), this expresses it as a mixture of Gaussians, as requested."""
    import warnings
    from sklearn.mixture import GaussianMixture
    X = np.asarray(returns).reshape(-1, 1)
    best = None
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=0, n_init=2)
        with warnings.catch_warnings():
            # numpy 2.0 emits spurious matmul RuntimeWarnings from sklearn's
            # k-means init on some BLAS builds; they are harmless here.
            warnings.simplefilter("ignore", RuntimeWarning)
            gm.fit(X)
            bic = gm.bic(X)
        if best is None or bic < best["bic"]:
            best = {
                "n_components": k, "bic": float(bic),
                "weights": gm.weights_.tolist(),
                "means": gm.means_.flatten().tolist(),
                "stds": np.sqrt(gm.covariances_.flatten()).tolist(),
            }
    obs.event(channel="analytics", kind="gmm.fit",
              n_components=best["n_components"], bic=round(best["bic"], 2))
    return best


# ── Objectives ──────────────────────────────────────────────────────────────

def objective_value(returns: np.ndarray, objective: str, *,
                    var_level: float = 0.95, crra_gamma: float = 3.0) -> float:
    r = np.asarray(returns)
    if objective == "sharpe":
        sd = np.std(r, ddof=1)
        return float(np.mean(r) / sd) if sd > 0 else float("-inf")
    if objective == "cvar_adjusted_return":
        cvar = var_cvar(r, var_level)["CVaR"]
        # Floor the tail-risk denominator so near-riskless durations do not make
        # the ratio explode (a 0.5% CVaR floor keeps the objective in a sane,
        # comparable range across the grid).
        denom = max(cvar, 0.005)
        return float(np.mean(r) / denom)
    if objective == "crra_utility":
        g = crra_gamma
        w = 1.0 + r            # wealth relative
        w = np.maximum(w, 1e-6)
        if abs(g - 1.0) < 1e-9:
            u = np.log(w)
        else:
            u = (w ** (1.0 - g) - 1.0) / (1.0 - g)
        ce = (np.mean(u) * (1.0 - g) + 1.0) ** (1.0 / (1.0 - g)) - 1.0 if abs(g - 1.0) > 1e-9 \
            else np.exp(np.mean(u)) - 1.0
        return float(ce)
    raise ValueError(f"unknown objective {objective!r}")


# ── Orchestration: duration grid + transition sensitivity ───────────────────

def duration_grid_search(run_fn: Callable[[float], np.ndarray],
                         durations: List[float], objective: str, *,
                         var_level: float = 0.95, crra_gamma: float = 3.0) -> Dict:
    """Search the held duration that maximizes the objective. ``run_fn(d)``
    returns a return array for duration ``d`` years. Returns the full surface,
    not just the argmax."""
    surface = []
    for d in durations:
        r = run_fn(float(d))
        surface.append({
            "duration": float(d),
            "objective": objective_value(r, objective, var_level=var_level, crra_gamma=crra_gamma),
            "mean_return": float(np.mean(r)),
            "VaR95": var_cvar(r, 0.95)["VaR"],
            "CVaR95": var_cvar(r, 0.95)["CVaR"],
        })
    best = max(surface, key=lambda x: x["objective"])
    obs.event(channel="analytics", kind="duration.grid",
              best_duration=best["duration"], objective=objective)
    return {"objective": objective, "best": best, "surface": surface}


def transition_sensitivity(run_fn_P: Callable[[np.ndarray], np.ndarray],
                           P_draws: List[np.ndarray], *,
                           var_level: float = 0.95) -> Dict:
    """Propagate transition-matrix uncertainty into metric confidence bands.

    ``run_fn_P(P)`` runs the sim under transition matrix P and returns the
    return array. The spread of metrics across the Dirichlet draws is the
    'sensitivity to Fed rate probability shifts'."""
    means, vars95, cvars95 = [], [], []
    for P in P_draws:
        r = run_fn_P(P)
        means.append(float(np.mean(r)))
        tc = var_cvar(r, var_level)
        vars95.append(tc["VaR"]); cvars95.append(tc["CVaR"])

    def band(x):
        a = np.asarray(x)
        return {"mean": float(a.mean()), "p5": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)), "std": float(a.std())}

    obs.event(channel="analytics", kind="transition.sensitivity",
              n_draws=len(P_draws))
    return {"n_draws": len(P_draws), "mean_return": band(means),
            "VaR": band(vars95), "CVaR": band(cvars95)}
