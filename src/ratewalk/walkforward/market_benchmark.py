"""Market-implied benchmark for the FOMC walk-forward forecaster.

The question every rates desk asks of a policy-rate model: does it beat the
market's own pricing? The canonical market forecast lives in fed funds
futures, which are paywalled; the free, defensible proxy used here is the
short end of the Treasury curve. The 3-month bill yield embeds the expected
average policy rate over the next quarter, so the spread

    x_t = y3m(t) - policy_rate(t)

carries the market's directional view. Rather than hardcoding how many bps
of next-month move a given spread implies, the mapping is CALIBRATED
walk-forward: at each evaluation date we regress realized next-period moves
on past spreads (past pairs only), predict the mean move from today's
spread, estimate the residual scale from past errors only, and integrate a
normal over the move grid to get a probability distribution scored with
exactly the same log-loss as the Markov models. No fixed scaling constants;
the market's information content is whatever past data says it is.

Two artifacts come out:

  * ``walk_forward_market`` - the market-proxy forecaster, same row schema
    as ``walk_forward_forecast`` so ``score_forecasts`` applies unchanged.
  * ``blend_adaptive`` - a parameter-free linear opinion pool of any two
    forecast tables, weighted per date by inverse mean past log-loss of
    each component (past evaluation points only, so no look-ahead).

No-look-ahead discipline (ADR 0004): the signal at index i uses market data
dated <= s.dates[i-1]; regression pairs end at transitions completed before
i; blend weights use scores of forecasts already realized before i. The
invariance test in tests/test_market_benchmark.py injects future data and
asserts unchanged predictions.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import obs
from .forecast import WFSeries, score_forecasts


def market_signal(md, s: WFSeries, *, tenor_col: str = "0.25") -> np.ndarray:
    """Curve-minus-policy spread aligned to the WFSeries decision grid.

    signal[k] = (short-tenor yield - policy rate), both as of the last
    observation dated <= s.dates[k]. NaN where either series has no
    observation yet.
    """
    curve = md.curve.sort_values("date")
    policy = md.policy_rate.sort_values("date")
    if tenor_col not in curve.columns:
        # fall back to the shortest available tenor
        tenors = sorted((c for c in curve.columns if c != "date"), key=float)
        if not tenors:
            raise ValueError("curve has no tenor columns")
        tenor_col = tenors[0]

    grid = pd.DataFrame({"date": pd.to_datetime(s.dates)})
    y = pd.merge_asof(grid, curve[["date", tenor_col]], on="date")
    r = pd.merge_asof(grid, policy[["date", "rate"]], on="date")
    sig = y[tenor_col].to_numpy(dtype=float) - r["rate"].to_numpy(dtype=float)
    obs.event(channel="walkforward", kind="market_signal",
              tenor=tenor_col, n=len(sig), n_nan=int(np.isnan(sig).sum()))
    return sig


def _grid_bin_probs(mu: float, sigma: float, incr_bps: np.ndarray) -> np.ndarray:
    """Integrate N(mu, sigma) over bins whose edges are midpoints between
    the (sorted) grid values; open-ended tails. Returned in the original
    state order."""
    from math import erf, sqrt

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))

    order = np.argsort(incr_bps)
    vals = incr_bps[order]
    edges = (vals[1:] + vals[:-1]) / 2.0
    upper = np.concatenate([edges, [np.inf]])
    lower = np.concatenate([[-np.inf], edges])
    p_sorted = np.array([cdf(u) - cdf(l) for l, u in zip(lower, upper)])
    out = np.empty_like(p_sorted)
    out[order] = p_sorted
    return out


def walk_forward_market(s: WFSeries, signal: np.ndarray, *,
                        min_train: int = 120, prior: float = 1.0,
                        min_pairs: int = 24) -> pd.DataFrame:
    """Walk-forward market-proxy forecast, one row per evaluated period.

    At index i (predicting period i from information at i-1):
      pairs  = {(signal[k], move_bps[k+1]) : k+1 <= i-1, signal[k] finite}
      fit    = OLS move ~ a + b*signal on pairs (mean-only if x degenerate)
      mu     = a + b*signal[i-1]
      sigma  = std of past residuals (population), floored at 1e-6
      probs  = normal integrated over the move grid, then Dirichlet-style
               smoothed with the same ``prior`` pseudo-count budget the
               Markov models receive: (n*probs + prior) / (n + prior*K)

    Rows where the signal is unavailable or pairs < min_pairs fall back to
    the past empirical move distribution (climatology on realized moves),
    flagged with used_signal=False, so the comparison covers the identical
    evaluation window as the Markov models.
    """
    n = len(s.state)
    n_states = s.rate_space.n
    y_bps = s.incr_bps[s.state]      # realized move per period
    rows = []
    used = 0
    for i in range(min_train, n):
        x_now = signal[i - 1]
        # past pairs: predictor at k, outcome at k+1, completed before i
        xs = signal[:i - 1]
        ys = y_bps[1:i]
        m = np.isfinite(xs)
        xs, ys = xs[m], ys[m]
        n_pairs = len(xs)

        if np.isfinite(x_now) and n_pairs >= min_pairs:
            xvar = float(np.var(xs))
            if xvar > 1e-12:
                b = float(np.cov(xs, ys, bias=True)[0, 1] / xvar)
            else:
                b = 0.0
            a = float(np.mean(ys) - b * np.mean(xs))
            resid = ys - (a + b * xs)
            sigma = max(float(np.std(resid)), 1e-6)
            mu = a + b * x_now
            probs = _grid_bin_probs(mu, sigma, s.incr_bps)
            used_signal = True
            used += 1
        else:
            counts = np.bincount(s.state[:i], minlength=n_states).astype(float)
            probs = counts / counts.sum() if counts.sum() > 0 else np.full(n_states, 1.0 / n_states)
            mu, sigma, n_pairs = float("nan"), float("nan"), n_pairs
            used_signal = False

        probs = (n_pairs * probs + prior) / (n_pairs + prior * n_states)
        rows.append({
            "date": s.dates[i], "from_state": int(s.state[i - 1]),
            "regime": int(s.regime[i - 1]), "actual": int(s.state[i]),
            "n_support": int(n_pairs), "probs": probs,
            "lo": probs, "hi": probs,
            "mu_bps": mu, "sigma_bps": sigma, "used_signal": used_signal,
        })
    df = pd.DataFrame(rows)
    obs.event(channel="walkforward", kind="market_forecast",
              n_eval=len(df), n_signal_used=used, min_pairs=min_pairs)
    return df


def blend_adaptive(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   n_states: int, *, warmup: int = 24) -> pd.DataFrame:
    """Parameter-free linear opinion pool of two aligned forecast tables.

    Weight at row i uses ONLY rows < i: w_a = exp(-La) / (exp(-La)+exp(-Lb))
    with La, Lb the components' mean log-losses so far. During the warmup
    (too few realized scores) the pool is 50/50. This is the natural
    no-look-ahead combination: whichever source has been more informative
    lately earns weight, with no tuned constant.
    """
    assert len(df_a) == len(df_b), "forecast tables must cover the same window"
    eps = 1e-12
    pa = np.vstack(df_a["probs"].to_numpy())
    pb = np.vstack(df_b["probs"].to_numpy())
    actual = df_a["actual"].to_numpy()
    assert (actual == df_b["actual"].to_numpy()).all(), "misaligned actuals"

    la = -np.log(pa[np.arange(len(df_a)), actual] + eps)
    lb = -np.log(pb[np.arange(len(df_b)), actual] + eps)

    rows = []
    for i in range(len(df_a)):
        if i < warmup:
            w = 0.5
        else:
            ma, mb = la[:i].mean(), lb[:i].mean()
            ea, eb_ = np.exp(-ma), np.exp(-mb)
            w = float(ea / (ea + eb_))
        probs = w * pa[i] + (1 - w) * pb[i]
        r = dict(df_a.iloc[i][["date", "from_state", "regime", "actual"]])
        r.update({"n_support": int(df_a.iloc[i]["n_support"]),
                  "probs": probs, "lo": probs, "hi": probs, "w_a": w})
        rows.append(r)
    return pd.DataFrame(rows)


def compare_with_market(s: WFSeries, md, *, min_train: int = 120,
                        shrink_tau: float = 50.0, n_dirichlet: int = 200,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """Model-vs-market head-to-head on the identical evaluation window.

    Returns scores for the market proxy, the EB-shrunk conditional chain,
    the adaptive blend, and a divergence diagnosis: on which periods do the
    two disagree most, and who was right there.
    """
    from .forecast import walk_forward_forecast

    rng = rng or np.random.default_rng(0)
    sig = market_signal(md, s)
    df_mkt = walk_forward_market(s, sig, min_train=min_train)
    df_eb = walk_forward_forecast(s, model="conditional_eb", min_train=min_train,
                                  n_dirichlet=n_dirichlet, rng=rng)
    df_blend = blend_adaptive(df_mkt, df_eb, s.rate_space.n)

    out = {
        "market": score_forecasts(df_mkt, s.rate_space.n),
        "conditional_eb": score_forecasts(df_eb, s.rate_space.n),
        "blend_adaptive": score_forecasts(df_blend, s.rate_space.n),
    }
    out["market"]["signal_coverage"] = round(
        float(df_mkt["used_signal"].mean()), 4)

    # ── Divergence diagnosis ──
    eps = 1e-12
    pm = np.vstack(df_mkt["probs"].to_numpy())
    pe = np.vstack(df_eb["probs"].to_numpy())
    actual = df_mkt["actual"].to_numpy()
    ll_m = -np.log(pm[np.arange(len(actual)), actual] + eps)
    ll_e = -np.log(pe[np.arange(len(actual)), actual] + eps)
    # total-variation distance between the two predictive distributions
    tv = 0.5 * np.abs(pm - pe).sum(axis=1)
    moved = s.incr_bps[actual] != 0.0
    hi_div = tv > np.quantile(tv, 0.8)
    out["divergence"] = {
        "mean_tv": round(float(tv.mean()), 4),
        "n_high_divergence": int(hi_div.sum()),
        "market_better_overall_pct": round(float((ll_m < ll_e).mean()), 4),
        "market_better_on_moves_pct": round(
            float((ll_m[moved] < ll_e[moved]).mean()), 4) if moved.any() else None,
        "market_better_on_holds_pct": round(
            float((ll_m[~moved] < ll_e[~moved]).mean()), 4) if (~moved).any() else None,
        "market_better_when_diverging_pct": round(
            float((ll_m[hi_div] < ll_e[hi_div]).mean()), 4) if hi_div.any() else None,
    }
    out["summary"] = {
        "eval_points": int(len(df_mkt)),
        "logloss_market": out["market"]["mean_log_loss"],
        "logloss_conditional_eb": out["conditional_eb"]["mean_log_loss"],
        "logloss_blend_adaptive": out["blend_adaptive"]["mean_log_loss"],
        "model_beats_market": bool(out["conditional_eb"]["mean_log_loss"]
                                   < out["market"]["mean_log_loss"]),
        "blend_beats_both": bool(
            out["blend_adaptive"]["mean_log_loss"]
            < min(out["market"]["mean_log_loss"],
                  out["conditional_eb"]["mean_log_loss"])),
    }
    obs.event(channel="walkforward", kind="market_compare", **out["summary"])
    return out
