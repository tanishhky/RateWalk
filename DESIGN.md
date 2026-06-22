# RateWalk: design and methodology

## Pipeline

```
data -> states -> markov -> curve -> horizon select -> sim (+jumps) -> analytics
                                                         |-> credit overlay
                                                         |-> hedge agents (optional)
                                                         |-> vol allocation (optional)
```

## Modeling decisions (and how each is handled)

1. **Policy rate is not the curve.** A bond is priced off the whole curve, so
   `curve/` fits a per-tenor linear map `yield_tenor = a + b * policy_rate (+ noise)`
   from history. Long yields move less than one-for-one with the policy rate, and
   the residual noise lets the curve carry its own risk. A Nelson-Siegel-Svensson
   factor model is the documented richer option behind the same interface.

2. **Markov memorylessness vs cycle persistence.** A pure increment chain
   simulated forward is a memoryless random walk with no level anchor, so rates
   drift unrealistically over long horizons. The engine de-drifts the increments
   (subtract the chain's stationary mean increment) and adds an OU pull toward a
   neutral level (`sim.mean_reversion`). This keeps the Markov step distribution
   while anchoring the level, so annualized returns are horizon-stable. A
   semi-Markov (dwell-time) variant is the next refinement.

3. **Regime non-stationarity.** Estimation is exponentially time-weighted
   (`markov.half_life_years`) so the matrix reflects the current regime rather
   than a multi-decade average. Sub-period and regime-conditional matrices are
   the documented alternatives.

4. **CPI revisions are a look-ahead trap.** CPI is read on its **release date**
   (publication lag honored), not its observation month, via `cpi_asof`. The
   real FRED path uses ALFRED vintages. Enforced by the no-look-ahead test.

5. **Treasuries do not default in the normal course.** The `credit/` overlay is
   generic and sovereign-agnostic on purpose: set `annual_default_prob = 0` and
   use the delay channel for US debt-ceiling stress, or raise it with a recovery
   rate for EM sovereigns / corporates / munis. The pipeline is built as a
   multi-sovereign tool, not a US-only one.

6. **Tail vs mean.** Reinvestment compounds high-rate paths, so terminal wealth
   is right-skewed; the **median** is the robust central tendency and the report
   leads with it. The short rate is also clipped to a realistic ceiling
   (`sim.max_short_rate`) so a fat-tailed jump cannot produce a few explosive
   paths that dominate the mean.

## Components

- **states/** `increments` (FOMC-native, stationary) or `levels`; CPI binned
  into inflation regimes; joint alignment via `merge_asof` on the CPI release.
- **markov/** MLE transition matrices with a Dirichlet/Laplace prior; univariate
  rate and CPI chains plus a **conditional** chain `P(rate'|rate, cpi)`;
  `resample_dirichlet` draws matrices from the posterior for sensitivity bands.
- **sim/** vectorized Monte Carlo (pricing vectorized across paths, only the
  time steps loop). Jumps are a **plugin family**: `ScenarioReplay` (replay GFC /
  Covid / SVB / taper trajectories at sampled, per-path times) and
  `JumpDiffusion` (Poisson arrivals, fat-tailed shocks, decay over a recovery
  window). Add your own by subclassing `JumpModel`.
- **instruments/** generic coupon bond with par-coupon, pricing, and Macaulay
  duration; the `InstrumentBook` scales single bond -> ladder -> portfolio.
- **analytics/** historical VaR and CVaR; mean/std/skew/excess-kurtosis with a
  Jarque-Bera test; a BIC-selected Gaussian-mixture fit when non-Gaussian;
  duration grid search over a configurable objective (Sharpe, CVaR-adjusted
  return, CRRA utility); transition sensitivity via Dirichlet draws.
- **hedge/** multiple agents, one per underlying (ETF put, futures option,
  swaption), each searching a (tenor x moneyness) grid for the best protection
  efficiency (CVaR reduction per premium dollar), plus an **all-in-one** agent
  that, per grid cell, keeps whichever underlying is most underpriced and ranks
  by efficiency, like a desk shopping the cheapest hedge.
- **vol/** compares constant-risk (vol targeting) vs confidence-scaled
  ("double down" when the transition CI is tight) sizing.
- **horizon/** dynamic selection: score each candidate horizon and pick the best
  risk-adjusted one; the surface is returned so the UI shows why.

## Engineering discipline (from the PinSight / DriftEdge / ChronoFund platform)

- **No look-ahead.** Every data reader takes `as_of_ts` and returns only what was
  public then; `tests/test_no_lookahead.py` proves a past estimate does not
  change when future data is appended.
- **Config-first.** Nothing in the model path is hard-coded; a content hash of
  the config is stamped on every report.
- **Observability.** Every estimate / simulation / analytic emits a JSONL event
  via `obs`.
- **Reproducible.** A seed plus the config hash reproduces every number; the
  synthetic data generator is deterministic per country.

## Extensions (documented, interfaces already in place)

- FRED/ALFRED HTTP client (the hook is in `data/sources.py`).
- Nelson-Siegel-Svensson factor curve; semi-Markov dwell times; regime-conditional
  matrices.
- Bond-by-bond reinvestment (v1 uses a blended reinvestment account).
- Live option-chain pricing for the hedge agents (the search and efficiency
  definitions do not change).
- A richer React/Plotly front end consuming the existing API endpoints.
