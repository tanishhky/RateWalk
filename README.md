# RateWalk

![tests](https://github.com/tanishhky/RateWalk/actions/workflows/tests.yml/badge.svg)

**A Markov-driven Monte Carlo engine for fixed-income path simulation, risk, and hedging.**

RateWalk learns how the monetary regime moves (policy rate and CPI) as a Markov
process estimated from history, maps each rate state onto a full yield curve,
then runs large Monte Carlo ensembles of a bond-investment strategy (coupons
reinvested across a configurable tenor ladder) with programmable,
regime-realistic jumps (GFC, Covid, SVB). From the simulated distribution it
computes VaR/CVaR, distributional moments and a Gaussian-mixture fit, the best
investment duration by grid search, and the sensitivity of everything to shifts
in the transition probabilities and to coupon delays/defaults. Optional modules
add multi-agent put hedging and a volatility-allocation decision.

It is **multi-sovereign** (US, GB, DE, ... via config), the held instrument
scales from a **single bond to a ladder or portfolio**, and the investment
**horizon is selected dynamically**.

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[api,dev]"
./.venv/bin/python -m ratewalk.cli run            # full pipeline -> runs/report-<hash>.json
./.venv/bin/python -m ratewalk.cli walkforward    # out-of-sample forecast + backtest
./.venv/bin/python -m ratewalk.api.server         # web UI at http://127.0.0.1:8780
./.venv/bin/python -m pytest -q                   # tests (incl. no-lookahead)
```

## Walk-forward validation (the honest part)

`ratewalk walkforward` answers two questions on out-of-sample data, with no
look-ahead (every prediction at month t uses only data public before t):

1. **What is the likelihood of the Fed's next move, with a confidence interval?**
   A live nowcast gives `P(-50/-25/0/+25/+50/+75 bps)` with a Dirichlet band,
   e.g. `P(hold) = 60% [49%, 70%]`.
2. **Were we right historically?** Each month's prediction is scored against the
   realized move and compared to a climatology baseline and an unconditional
   chain (accuracy, log-loss, Brier, calibration), plus a duration-timing
   backtest vs constant-duration benchmarks.

### The headline finding

Out of sample (1990-2026), conditioning the rate-transition chain on the CPI
regime behaves in a way that is itself the result:

| model | US log-loss | what it is |
|---|---|---|
| climatology | 1.162 | unconditional marginal of moves |
| unconditional chain | 1.071 | P(next \| current move) |
| **raw** CPI-conditional | 1.136 | P(next \| current, CPI) - *worse*, it overfits |
| **shrunk** CPI-conditional | **1.059** | same, shrunk toward the pooled chain - *best* |

So the naive macro model **hurts** (it fragments sparse regime data), but pulling
each regime's row toward the pooled chain with an empirical-Bayes-style prior
**recovers a real edge** and beats the unconditional model. The flip replicates
out of sample across the **US, UK, and Germany**, and it is a broad plateau in
the shrinkage strength (tau ~ 20-500), not a knife-edge tuned to the test set.

The duration-timing strategy built on the signal still does **not** beat a
constant-2y bond risk-adjusted (Sharpe 1.06 vs 1.39), which is an honest
negative. See DESIGN.md for what would move that.

### Real data (FRED / ALFRED)

Put a free [FRED API key](https://fredaccount.stlouisfed.org/apikeys) in `.env`:

```bash
cp .env.example .env        # then edit: FRED_API_KEY=your_key
```

With a key, `source: auto` pulls **real data**: the policy rate, the Treasury
curve, and **true point-in-time CPI via ALFRED** (initial-release values dated
by their real publication date, so revisions never leak). It is
**multi-sovereign**: `country: US | GB | DE | JP | CA` (US has a full daily
curve; others use the policy rate plus the 10Y yield as curve anchors, which is
all FRED carries for them). With no key, it falls back to a deterministic
**synthetic** generator so the pipeline and tests still run offline.

## What one run produces

```
horizon=5.0y  mean ann return=2.62%  VaR95=-0.0063  best duration=7.0y
```

plus a JSON report with: the estimated transition matrix and its stationary
distribution (and whether it beats a persistence baseline), the dynamic horizon
selection surface, the headline return distribution (mean/median/p5/p95),
VaR/CVaR, distributional moments, a Gaussian-mixture fit when non-Gaussian, the
duration grid surface, transition-probability sensitivity bands (via Dirichlet
resampling), the credit overlay (defaults + single-coupon-miss impact), and
optionally the hedging agents' picks and the vol-policy comparison.

## Configuration

Everything in the model path is in `config/default.yaml` (schema and per-field
docs in `src/ratewalk/config.py`). Pass your own with `--config`. Highlights:

- `state.rate_mode`: `increments` (recommended, stationary) or `levels`
- `markov.model`: `rate` | `cpi` | `conditional` (CPI regime modulates the rate chain)
- `instrument.kind`: `single_bond` | `ladder` | `portfolio`
- `sim.jumps`: a stackable list of `scenario_replay` and `jump_diffusion` models
- `sim.mean_reversion`: OU anchor that keeps long-horizon rates realistic
- `credit.enabled`: turn on for corporates / munis / EM sovereigns / debt-ceiling stress
- `hedge.enabled`, `vol.enabled`, `horizon.mode`: optional / dynamic modules

## Layout

```
src/ratewalk/
  config.py  obs.py            config + observability spine
  data/      states/  markov/  point-in-time data, discretization, chain estimation
  curve/                       state -> yield-curve mapping
  instruments/                 generic bond / ladder / portfolio
  sim/                         Monte Carlo engine + pluggable jump models
  credit/                      multi-sovereign default / delay / coupon-miss overlay
  analytics/                   VaR/CVaR, moments, GMM, duration grid, sensitivity
  hedge/  vol/  horizon/       optional: hedge agents, vol policy, dynamic horizon
  api/                         FastAPI + minimal web UI
```

## Writeups

- [`notebooks/RateWalk-research.ipynb`](./notebooks/RateWalk-research.ipynb) - an
  executable research notebook that walks the whole story end to end (idea ->
  surprise -> diagnosis -> shrinkage fix -> robustness -> honest negatives), with
  outputs and figures generated from live FRED data.
- [`paper/ratewalk.pdf`](./paper/ratewalk.pdf) - a short working paper of the
  same study.

See [DESIGN.md](./DESIGN.md) for the methodology, the modeling decisions, and
the planned extensions. This is research software, not investment advice.
