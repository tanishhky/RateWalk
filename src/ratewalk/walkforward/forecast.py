"""Walk-forward FOMC-decision forecaster.

At each historical month t, using ONLY data public before t, we predict the
probability distribution over the next rate increment (-50/-25/0/+25/+50/+75
bps) and attach a confidence interval, then compare to the move that actually
happened. This both answers "what is the likelihood of the Fed's next decision,
with a confidence interval" and validates, out of sample, whether the model was
historically right.

No look-ahead is structural: for the prediction at month t we only count
transitions that completed strictly before t, and we condition on the CPI
regime that was public before t (CPI carries its real release date).

Unit note: the engine works on MONTHLY increments (consistent with the rest of
RateWalk). Most months are "+0bps" because not every month has an FOMC meeting
or a move, so accuracy is dominated by the hold class; the honest tests are
log-loss / Brier and whether the model beats the climatology and unconditional
baselines. Meeting-level forecasting (using the FOMC calendar) is a documented
refinement.

Three models are compared:
  * climatology   - the unconditional marginal frequency of moves (ignores
                    the current state entirely). The "dumb" baseline.
  * unconditional - a first-order chain P(next | current move). No macro.
  * conditional   - P(next | current move, CPI regime). The macro-aware model.

Confidence interval: the predicted row is a Dirichlet posterior on the
transition counts; we draw from it, so sparse history yields honestly wide
bands.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import obs
from ..states import StateSpace, build_rate_states, build_cpi_states, build_joint_series


def _increment_bps(label: str) -> float:
    m = re.match(r"([+-]?\d+)bps", label)
    return float(m.group(1)) if m else 0.0


@dataclass
class WFSeries:
    """Aligned, no-look-ahead monthly series ready for the walk-forward loop."""
    dates: np.ndarray              # month-end dates
    state: np.ndarray              # increment-state index per month
    regime: np.ndarray             # CPI-regime index per month (public that month)
    rate_space: StateSpace
    cpi_space: StateSpace
    incr_bps: np.ndarray           # bps value per state index


# Decision-cadence resample frequencies. 'monthly' is 12/yr (the default, but
# inflates the hold class with no-meeting months). 'meeting' is a ~46-day grid,
# ~8/yr, matching the FOMC's eight scheduled meetings (an approximation of the
# cadence, not the exact published calendar, which would be a data-ingestion
# refinement). The net rate change in each bucket is snapped to the move grid.
_DECISION_FREQ = {"monthly": "ME", "meeting": "46D"}


def prepare_series(md, cfg, *, decision_freq: str = "monthly") -> WFSeries:
    resample = _DECISION_FREQ.get(decision_freq, decision_freq)
    rs, rate_space = build_rate_states(md.policy_rate, mode="increments",
                                       increment_grid_bps=cfg.state.increment_grid_bps,
                                       resample=resample)
    cs, cpi_space = build_cpi_states(md.cpi_yoy, bins_yoy=cfg.state.cpi_bins_yoy)
    joint = build_joint_series(rs, cs, alignment=cfg.state.alignment)
    rmap = {lab: i for i, lab in enumerate(rate_space.labels)}
    cmap = {lab: i for i, lab in enumerate(cpi_space.labels)}
    state = joint["rate_state"].map(rmap).to_numpy()
    regime = joint["cpi_state"].map(cmap).to_numpy()
    incr_bps = np.array([_increment_bps(l) for l in rate_space.labels])
    return WFSeries(dates=pd.to_datetime(joint["date"]).to_numpy(), state=state,
                    regime=regime, rate_space=rate_space, cpi_space=cpi_space,
                    incr_bps=incr_bps)


def _row_counts(from_arr, to_arr, reg_arr, *, q_from, q_reg, n_states) -> np.ndarray:
    """Transition counts out of state q_from (optionally within regime q_reg),
    using only the transitions passed in (the caller passes past-only slices)."""
    mask = from_arr == q_from
    if q_reg is not None:
        mask = mask & (reg_arr == q_reg)
    return np.bincount(to_arr[mask], minlength=n_states).astype(float)


def estimate_eb_tau(from_arr, to_arr, reg_arr, n_states, n_regimes, *,
                    bounds=(1.0, 3000.0)) -> float:
    """Data-driven shrinkage strength by empirical Bayes (type-II MLE).

    The hierarchical model: each (from-state, CPI-regime) transition row is
    Multinomial with a Dirichlet(tau * pooled_prob) prior, where pooled_prob is
    the unconditional row for that from-state. Marginalizing the Multinomial
    gives a Dirichlet-multinomial whose only free parameter is the
    concentration tau. We pick tau to maximize the marginal likelihood of the
    observed regime rows (a Polya / Dirichlet-multinomial concentration
    estimate). Large tau means the regimes look like the pooled chain (little
    extra signal); small tau means the regimes genuinely differ.

    Pass ONLY past transitions so the estimate stays no-look-ahead.
    """
    from scipy.special import gammaln
    from scipy.optimize import minimize_scalar

    rows = []  # (counts_vec, pooled_prob_vec)
    for i in np.unique(from_arr):
        mi = from_arr == i
        pooled = np.bincount(to_arr[mi], minlength=n_states).astype(float)
        if pooled.sum() == 0:
            continue
        pooled_prob = (pooled + 1e-6) / (pooled + 1e-6).sum()
        for r in range(n_regimes):
            c = np.bincount(to_arr[mi & (reg_arr == r)], minlength=n_states).astype(float)
            if c.sum() > 0:
                rows.append((c, pooled_prob))
    if not rows:
        return bounds[0]

    def neg_ll(log_tau):
        tau = float(np.exp(log_tau))
        ll = 0.0
        for c, p in rows:
            n = c.sum()
            a = tau * p
            ll += gammaln(tau) - gammaln(tau + n) + np.sum(gammaln(c + a) - gammaln(a))
        return -ll

    res = minimize_scalar(neg_ll, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded")
    return float(np.clip(np.exp(res.x), bounds[0], bounds[1]))


def walk_forward_forecast(s: WFSeries, *, model: str = "conditional",
                          min_train: int = 120, prior: float = 1.0,
                          n_dirichlet: int = 300, ci: float = 0.90,
                          shrink_tau: float = 20.0, eb_refit_every: int = 12,
                          rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Return one row per evaluated month with the predicted distribution, its
    confidence band, and the realized outcome.

    ``shrink_tau`` is the fixed shrinkage strength for 'conditional_shrunk'.
    'conditional_eb' instead estimates tau by empirical Bayes from past-only
    data, re-fitting every ``eb_refit_every`` steps (kept no-look-ahead)."""
    rng = rng or np.random.default_rng(0)
    n = len(s.state)
    n_states = s.rate_space.n
    n_regimes = s.cpi_space.n
    # transition arrays: transition k is state[k] -> state[k+1] under regime[k]
    from_arr = s.state[:-1]
    to_arr = s.state[1:]
    reg_arr = s.regime[:-1]
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    cur_tau = shrink_tau
    eb_taus = []

    rows = []
    for i in range(min_train, n):
        q_from = s.state[i - 1]
        actual = s.state[i]
        # past-only transitions: those completed before month i (k+1 <= i-1)
        end = i - 1
        if model == "climatology":
            counts = np.bincount(s.state[:i], minlength=n_states).astype(float)
            alpha = counts + prior
        elif model == "unconditional":
            counts = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                 q_from=q_from, q_reg=None, n_states=n_states)
            alpha = counts + prior
        elif model == "conditional":
            counts = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                 q_from=q_from, q_reg=s.regime[i - 1], n_states=n_states)
            alpha = counts + prior
        elif model == "conditional_shrunk":
            # Empirical-Bayes shrinkage: pull the (sparse) regime-specific row
            # toward the pooled unconditional row. The pooled distribution acts
            # as the Dirichlet prior with strength shrink_tau pseudo-counts, so
            # a data-rich regime keeps its own estimate while a data-poor regime
            # falls back to the unconditional chain. shrink_tau -> 0 recovers
            # raw conditional; shrink_tau -> inf recovers unconditional.
            reg_counts = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                     q_from=q_from, q_reg=s.regime[i - 1], n_states=n_states)
            pooled = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                 q_from=q_from, q_reg=None, n_states=n_states)
            pooled_prob = (pooled + prior) / (pooled + prior).sum()
            alpha = reg_counts + shrink_tau * pooled_prob
            counts = reg_counts
        elif model == "conditional_eb":
            # Same shrinkage, but tau is estimated from past-only data by
            # empirical Bayes, refit periodically (no look-ahead).
            if (i - min_train) % eb_refit_every == 0:
                cur_tau = estimate_eb_tau(from_arr[:end], to_arr[:end], reg_arr[:end],
                                          n_states, n_regimes)
                eb_taus.append((str(s.dates[i]), round(cur_tau, 1)))
            reg_counts = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                     q_from=q_from, q_reg=s.regime[i - 1], n_states=n_states)
            pooled = _row_counts(from_arr[:end], to_arr[:end], reg_arr[:end],
                                 q_from=q_from, q_reg=None, n_states=n_states)
            pooled_prob = (pooled + prior) / (pooled + prior).sum()
            alpha = reg_counts + cur_tau * pooled_prob
            counts = reg_counts
        else:
            raise ValueError(f"unknown model {model!r}")
        if alpha.sum() <= 0:        # degenerate (e.g. tau=0 with an empty row)
            alpha = np.ones(n_states)
        probs = alpha / alpha.sum()
        draws = rng.dirichlet(alpha, n_dirichlet)
        lo = np.percentile(draws, lo_q, axis=0)
        hi = np.percentile(draws, hi_q, axis=0)
        rows.append({
            "date": s.dates[i], "from_state": int(q_from),
            "regime": int(s.regime[i - 1]), "actual": int(actual),
            "n_support": int(counts.sum()), "probs": probs, "lo": lo, "hi": hi,
        })
    df = pd.DataFrame(rows)
    if eb_taus:
        df.attrs["eb_taus"] = eb_taus
    obs.event(channel="walkforward", kind="forecast", model=model,
              n_eval=len(df), min_train=min_train)
    return df


def score_forecasts(df: pd.DataFrame, n_states: int) -> Dict:
    """Accuracy, mean log-loss, Brier score, and a confidence-vs-accuracy
    calibration table."""
    if df.empty:
        return {"n": 0}
    eps = 1e-12
    probs = np.vstack(df["probs"].to_numpy())
    actual = df["actual"].to_numpy()
    pred = probs.argmax(axis=1)
    onehot = np.eye(n_states)[actual]
    logloss = -np.log(probs[np.arange(len(df)), actual] + eps)
    brier = ((probs - onehot) ** 2).sum(axis=1)
    # calibration: bin by the model's confidence (max prob), compare to accuracy
    conf = probs.max(axis=1)
    correct = (pred == actual).astype(float)
    bins = np.linspace(0, 1, 6)
    cal = []
    for b in range(len(bins) - 1):
        m = (conf >= bins[b]) & (conf < bins[b + 1] if b < len(bins) - 2 else conf <= bins[b + 1])
        if m.sum() > 0:
            cal.append({"conf_bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                        "n": int(m.sum()), "mean_conf": round(float(conf[m].mean()), 3),
                        "accuracy": round(float(correct[m].mean()), 3)})
    return {
        "n": int(len(df)),
        "accuracy": round(float(correct.mean()), 4),
        "mean_log_loss": round(float(logloss.mean()), 4),
        "brier": round(float(brier.mean()), 4),
        "calibration": cal,
    }


def tau_sweep(s: WFSeries, taus, *, min_train: int = 120, n_dirichlet: int = 1,
              rng: Optional[np.random.Generator] = None) -> List[Dict]:
    """Out-of-sample log-loss of the shrunk conditional model across shrinkage
    strengths. tau=0 is raw conditional; large tau approaches unconditional.
    If an interior tau beats both ends, shrinkage recovered a real CPI signal."""
    rng = rng or np.random.default_rng(0)
    out = []
    for tau in taus:
        df = walk_forward_forecast(s, model="conditional_shrunk", min_train=min_train,
                                   n_dirichlet=n_dirichlet, shrink_tau=float(tau), rng=rng)
        sc = score_forecasts(df, s.rate_space.n)
        out.append({"tau": float(tau), "log_loss": sc["mean_log_loss"],
                    "brier": sc["brier"], "accuracy": sc["accuracy"]})
    return out


def compare_models(s: WFSeries, *, min_train: int = 120, n_dirichlet: int = 200,
                   shrink_tau: float = 20.0,
                   rng: Optional[np.random.Generator] = None) -> Dict:
    """Run the models over the same evaluation window and score them. Headline
    comparisons: does the chain beat climatology; does raw CPI conditioning beat
    the unconditional chain; and does empirical-Bayes shrinkage of the CPI
    conditioning recover any edge?"""
    rng = rng or np.random.default_rng(0)
    out = {}
    for model in ("climatology", "unconditional", "conditional"):
        df = walk_forward_forecast(s, model=model, min_train=min_train,
                                   n_dirichlet=n_dirichlet, rng=rng)
        out[model] = score_forecasts(df, s.rate_space.n)
    df_sh = walk_forward_forecast(s, model="conditional_shrunk", min_train=min_train,
                                  n_dirichlet=n_dirichlet, shrink_tau=shrink_tau, rng=rng)
    out["conditional_shrunk"] = score_forecasts(df_sh, s.rate_space.n)
    out["conditional_shrunk"]["shrink_tau"] = shrink_tau

    df_eb = walk_forward_forecast(s, model="conditional_eb", min_train=min_train,
                                  n_dirichlet=n_dirichlet, rng=rng)
    out["conditional_eb"] = score_forecasts(df_eb, s.rate_space.n)
    out["conditional_eb"]["eb_taus"] = df_eb.attrs.get("eb_taus", [])

    cu = out["unconditional"]["mean_log_loss"]
    cc = out["conditional"]["mean_log_loss"]
    cs = out["conditional_shrunk"]["mean_log_loss"]
    ce = out["conditional_eb"]["mean_log_loss"]
    cl = out["climatology"]["mean_log_loss"]
    out["summary"] = {
        "eval_points": out["conditional"]["n"],
        "chain_beats_climatology": bool(cu < cl),
        "raw_cpi_conditioning_helps": bool(cc < cu),
        "shrunk_cpi_conditioning_helps": bool(cs < cu),
        "eb_cpi_conditioning_helps": bool(ce < cu),
        "shrinkage_beats_raw_conditional": bool(cs < cc),
        "logloss_climatology": cl, "logloss_unconditional": cu,
        "logloss_conditional_raw": cc, "logloss_conditional_shrunk": cs,
        "logloss_conditional_eb": ce,
        "shrink_tau_fixed": shrink_tau,
    }
    return out


def nowcast(s: WFSeries, *, model: str = "conditional_shrunk", prior: float = 1.0,
            shrink_tau: float = 50.0, n_dirichlet: int = 2000, ci: float = 0.90,
            rng: Optional[np.random.Generator] = None) -> Dict:
    """The live, forward-looking call: predict the NEXT month's increment from
    all available data, with a confidence interval. Defaults to the best
    validated model (CPI-conditional with empirical-Bayes shrinkage). No score
    (no actual yet)."""
    rng = rng or np.random.default_rng(0)
    n_states = s.rate_space.n
    q_from = s.state[-1]
    from_arr, to_arr, reg_arr = s.state[:-1], s.state[1:], s.regime[:-1]
    if model == "climatology":
        counts = np.bincount(s.state, minlength=n_states).astype(float)
        alpha = counts + prior
    elif model == "unconditional":
        counts = _row_counts(from_arr, to_arr, reg_arr, q_from=q_from, q_reg=None, n_states=n_states)
        alpha = counts + prior
    elif model == "conditional":
        counts = _row_counts(from_arr, to_arr, reg_arr, q_from=q_from, q_reg=s.regime[-1], n_states=n_states)
        alpha = counts + prior
    elif model == "conditional_shrunk":
        reg_counts = _row_counts(from_arr, to_arr, reg_arr, q_from=q_from, q_reg=s.regime[-1], n_states=n_states)
        pooled = _row_counts(from_arr, to_arr, reg_arr, q_from=q_from, q_reg=None, n_states=n_states)
        pooled_prob = (pooled + prior) / (pooled + prior).sum()
        alpha = reg_counts + shrink_tau * pooled_prob
        counts = reg_counts
    else:
        raise ValueError(f"unknown model {model!r}")
    if alpha.sum() <= 0:
        alpha = np.ones(n_states)
    probs = alpha / alpha.sum()
    draws = rng.dirichlet(alpha, n_dirichlet)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lo = np.percentile(draws, lo_q, axis=0)
    hi = np.percentile(draws, hi_q, axis=0)
    exp_bps = float(probs @ s.incr_bps)
    return {
        "as_of": str(pd.Timestamp(s.dates[-1]).date()),
        "model": model,
        "current_state": s.rate_space.labels[int(q_from)],
        "cpi_regime": s.cpi_space.labels[int(s.regime[-1])],
        "n_support": int(counts.sum()),
        "expected_move_bps": round(exp_bps, 1),
        "distribution": [
            {"move": s.rate_space.labels[k],
             "prob": round(float(probs[k]), 4),
             "ci": [round(float(lo[k]), 4), round(float(hi[k]), 4)]}
            for k in range(n_states)
        ],
    }
