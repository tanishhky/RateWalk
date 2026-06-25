# RateWalk

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
./.venv/bin/python -m ratewalk.api.server         # web UI at http://127.0.0.1:8780
./.venv/bin/python -m pytest -q                   # tests (incl. no-lookahead)
```

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

See [DESIGN.md](./DESIGN.md) for the methodology, the modeling decisions, and
the planned extensions. This is research software, not investment advice.
