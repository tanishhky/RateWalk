"""Generate the model-vs-market figures for the paper (paper/figs/).

Re-runs the two walk-forward forecasters on real FRED data (same settings
as `ratewalk walkforward`) and renders:

  figs/market_cumll.png   - cumulative log-loss gap, model vs market proxy,
                            with the adaptive blend
  figs/market_decomp.png  - who wins where: holds vs moves vs divergence
  figs/market_reliability.png - reliability of P(hold): predicted vs realized

Run from the repo root:  .venv/bin/python scripts/make_market_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratewalk import config as cfg
from ratewalk.data import load_macro
from ratewalk.walkforward import (prepare_series, walk_forward_forecast,
                                  market_signal, walk_forward_market,
                                  blend_adaptive)

OUT = Path(__file__).resolve().parents[1] / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
MIN_TRAIN = 120
EPS = 1e-12

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def main() -> None:
    c = cfg.load()
    md = load_macro(country="US", source=c.data.source, start=c.data.start,
                    cpi_vintage=c.data.cpi_vintage)
    if md.source != "fred":
        raise SystemExit("refusing to make paper figures from synthetic data "
                         "(set FRED_API_KEY); see honesty note in README")
    s = prepare_series(md, c, decision_freq=c.state.decision_freq)

    rng = np.random.default_rng(c.sim.seed)
    sig = market_signal(md, s)
    df_m = walk_forward_market(s, sig, min_train=MIN_TRAIN)
    df_e = walk_forward_forecast(s, model="conditional_eb", min_train=MIN_TRAIN,
                                 n_dirichlet=1, rng=rng)
    df_b = blend_adaptive(df_m, df_e, s.rate_space.n)

    pm = np.vstack(df_m["probs"].to_numpy())
    pe = np.vstack(df_e["probs"].to_numpy())
    pb = np.vstack(df_b["probs"].to_numpy())
    actual = df_m["actual"].to_numpy()
    idx = np.arange(len(actual))
    ll_m = -np.log(pm[idx, actual] + EPS)
    ll_e = -np.log(pe[idx, actual] + EPS)
    ll_b = -np.log(pb[idx, actual] + EPS)
    dates = pd.to_datetime(df_m["date"])
    moved = s.incr_bps[actual] != 0.0

    # ── Figure 1: cumulative log-loss ──
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(dates, np.cumsum(ll_m), label=f"market proxy ({ll_m.mean():.3f}/obs)",
            color="#b22222", lw=1.4)
    ax.plot(dates, np.cumsum(ll_e), label=f"Markov EB ({ll_e.mean():.3f}/obs)",
            color="#1f4e8c", lw=1.4)
    ax.plot(dates, np.cumsum(ll_b), label=f"adaptive blend ({ll_b.mean():.3f}/obs)",
            color="#2e7d32", lw=1.6)
    ax.set_ylabel("cumulative log-loss")
    ax.set_title("Out-of-sample cumulative log-loss: model vs curve-implied market proxy")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "market_cumll.png")

    # ── Figure 2: who wins where ──
    cats = {
        "all periods": np.ones(len(actual), dtype=bool),
        "hold periods": ~moved,
        "move periods": moved,
    }
    frac_mkt = [float((ll_m[m] < ll_e[m]).mean()) for m in cats.values()]
    ns = [int(m.sum()) for m in cats.values()]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(cats))
    ax.bar(x - 0.18, frac_mkt, width=0.36, label="market proxy better",
           color="#b22222")
    ax.bar(x + 0.18, [1 - f for f in frac_mkt], width=0.36,
           label="Markov EB better", color="#1f4e8c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(cats, ns)])
    ax.set_ylabel("share of periods with lower log-loss")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_title("Who wins where: market on moves, model on holds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "market_decomp.png")

    # ── Figure 3: reliability of P(hold) ──
    hold_idx = int(np.argmin(np.abs(s.incr_bps)))
    is_hold = (actual == hold_idx).astype(float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    bins = np.linspace(0, 1, 9)
    for probs, name, color in ((pm, "market proxy", "#b22222"),
                               (pe, "Markov EB", "#1f4e8c")):
        p_hold = probs[:, hold_idx]
        xs, ys, sz = [], [], []
        for b in range(len(bins) - 1):
            m = (p_hold >= bins[b]) & (p_hold < bins[b + 1])
            if m.sum() >= 5:
                xs.append(float(p_hold[m].mean()))
                ys.append(float(is_hold[m].mean()))
                sz.append(int(m.sum()))
        ax.plot(xs, ys, "o-", label=name, color=color, ms=4, lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted P(hold)")
    ax.set_ylabel("realized hold frequency")
    ax.set_title("Reliability of the hold probability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "market_reliability.png")

    print(f"figures written to {OUT}")
    print(f"log-loss: market={ll_m.mean():.4f} eb={ll_e.mean():.4f} "
          f"blend={ll_b.mean():.4f} over {len(actual)} decisions")


if __name__ == "__main__":
    main()
