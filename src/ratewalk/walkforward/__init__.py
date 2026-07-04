"""Walk-forward engine: out-of-sample forecasting and backtesting.

One no-look-ahead loop powers two readouts:
  * forecast.py  - predict the next rate decision (with a confidence interval)
                   at each historical date using ONLY past data, then score
                   the predictions against what actually happened.
  * backtest.py  - turn the same forecasts into a duration choice and compare
                   the realized return to constant-duration benchmarks.
"""
from .forecast import (  # noqa: F401
    prepare_series, walk_forward_forecast, score_forecasts, compare_models, nowcast,
)
from .backtest import duration_backtest  # noqa: F401
from .market_benchmark import (  # noqa: F401
    market_signal, walk_forward_market, blend_adaptive, compare_with_market,
)
