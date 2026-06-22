"""RateWalk: Markov-driven Monte Carlo engine for fixed-income path
simulation, risk, and hedging.

Pipeline
--------
data -> states -> markov -> curve -> sim (+jumps) -> analytics
                                       |-> credit overlay
                                       |-> hedge agents (optional)
                                       |-> vol allocation (optional)

Design rules (carried from the PinSight / DriftEdge / ChronoFund platform):
  * No look-ahead. Every reader of historical data takes an explicit
    ``as_of_ts`` and returns only what was public at that time. Model code
    never calls ``datetime.now()``.
  * Config-first. Everything tweakable lives in a YAML config; nothing is
    hard-coded in the model path.
  * Observability. Every estimate / simulation / analytic emits a structured
    JSONL event via ``obs``.
  * Reproducible. A seed plus a config hash reproduces every number.
"""

__version__ = "0.1.0"
