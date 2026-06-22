"""Typed, YAML-driven configuration.

Everything tweakable in the model path lives here. Load a config from a YAML
file (or the packaged default), get a frozen, validated object, and a stable
content hash so every output is traceable to its exact inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class StateConfig:
    rate_mode: str = "increments"            # "levels" | "increments"
    increment_grid_bps: tuple = (-50, -25, 0, 25, 50, 75)
    level_bin_width_bps: float = 25.0
    cpi_bins_yoy: tuple = (0.0, 2.0, 3.0, 5.0)   # edges -> len+1 regimes
    alignment: str = "decoupled"             # event_time | calendar_time | decoupled


@dataclass(frozen=True)
class MarkovConfig:
    model: str = "conditional"               # rate | cpi | conditional | semi_markov
    estimation: str = "exp_weighted"         # full | exp_weighted | window
    half_life_years: float = 8.0
    window_years: float = 20.0
    prior: str = "dirichlet"                 # none | dirichlet | laplace
    prior_strength: float = 1.0


@dataclass(frozen=True)
class CurveConfig:
    model: str = "nss_regression"            # nss_regression | parallel_shift
    add_residual_noise: bool = True
    residual_scale_bps: float = 15.0
    base_tenors_years: tuple = (0.25, 1, 2, 3, 5, 7, 10, 20, 30)


@dataclass(frozen=True)
class InstrumentConfig:
    kind: str = "single_bond"                # single_bond | ladder | portfolio
    held_tenor_years: float = 10.0
    coupon_rate: Optional[float] = None      # None -> set from issue curve (par)
    face: float = 100.0
    # For ladder / portfolio: tenor -> weight
    reinvestment_ladder: dict = field(default_factory=lambda: {"2": 0.3, "5": 0.4, "10": 0.3})
    portfolio_tenors: dict = field(default_factory=lambda: {"10": 1.0})


@dataclass(frozen=True)
class JumpSpec:
    type: str = "jump_diffusion"             # jump_diffusion | scenario_replay
    enabled: bool = True
    # jump_diffusion
    intensity_per_year: float = 0.10
    size_dist: str = "student_t"
    size_df: float = 3.0
    size_scale_bps: float = 75.0
    recovery_periods: int = 6
    # scenario_replay
    scenario: str = "covid_2020"
    inject_year_low: float = 1.0
    inject_year_high: float = 8.0
    fraction_of_paths: float = 0.15
    systemic: bool = False


@dataclass(frozen=True)
class SimConfig:
    n_paths: int = 5000
    steps_per_year: int = 12
    seed: int = 42
    # Mean reversion of the simulated short rate toward a neutral level (per
    # year). A pure Markov increment chain is a memoryless random walk with no
    # level anchor, so simulated rates drift unrealistically over long
    # horizons. This pulls the level back like an OU process while the Markov
    # chain still supplies the (regime-dependent) step distribution. Set to 0
    # for a pure random walk.
    mean_reversion: float = 0.25
    max_short_rate: float = 20.0
    jumps_enabled: bool = True
    jumps: tuple = ()                        # tuple[JumpSpec]; filled by loader


@dataclass(frozen=True)
class CreditConfig:
    enabled: bool = False                    # generic, multi-sovereign
    annual_default_prob: float = 0.0
    recovery_rate: float = 0.4
    coupon_delay_prob: float = 0.0
    coupon_miss_prob: float = 0.0


@dataclass(frozen=True)
class AnalyticsConfig:
    var_levels: tuple = (0.95, 0.99)
    duration_grid_years: tuple = (1, 2, 3, 5, 7, 10, 20, 30)
    objective: str = "cvar_adjusted_return"  # sharpe | cvar_adjusted_return | crra_utility
    crra_gamma: float = 3.0
    gmm_max_components: int = 4
    sensitivity_draws: int = 300


@dataclass(frozen=True)
class HedgeConfig:
    enabled: bool = False
    agents: tuple = ("etf_put", "futures_option", "swaption", "all_in_one")
    moneyness_grid: tuple = (0.90, 0.95, 1.00)
    tenor_grid_years: tuple = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class VolConfig:
    enabled: bool = False
    target_annual_vol: float = 0.06
    confidence_scaling: bool = True


@dataclass(frozen=True)
class HorizonConfig:
    mode: str = "dynamic"                    # fixed | dynamic
    fixed_years: float = 10.0
    candidate_years: tuple = (2, 5, 10, 20, 30)
    objective: str = "cvar_adjusted_return"


@dataclass(frozen=True)
class Config:
    country: str = "US"                      # multi-sovereign by design
    state: StateConfig = field(default_factory=StateConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    horizon: HorizonConfig = field(default_factory=HorizonConfig)

    def content_hash(self) -> str:
        """Stable hash of the full config, stamped on every output."""
        blob = json.dumps(_to_plain(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _build(dc_type, raw: dict):
    """Construct a frozen dataclass from a (possibly partial) dict, recursing
    into nested dataclass fields. Tuples are preserved as tuples."""
    if not raw:
        return dc_type()
    kwargs = {}
    type_hints = {f.name: f.type for f in fields(dc_type)}
    for f in fields(dc_type):
        if f.name not in raw:
            continue
        val = raw[f.name]
        # Nested dataclass?
        default = f.default_factory() if f.default_factory is not MISSING else None  # type: ignore
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[f.name] = _build(type(default), val)
        elif isinstance(val, list):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


# dataclasses.MISSING import without polluting namespace above
from dataclasses import MISSING  # noqa: E402


def load(path: Optional[Path] = None) -> Config:
    """Load a Config from YAML, falling back to packaged defaults. Unknown
    keys are ignored; missing keys take the dataclass default."""
    raw: dict = {}
    p = Path(path) if path else _DEFAULT_PATH
    if p.exists():
        with open(p) as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = Config(
        country=raw.get("country", "US"),
        state=_build(StateConfig, raw.get("state", {})),
        markov=_build(MarkovConfig, raw.get("markov", {})),
        curve=_build(CurveConfig, raw.get("curve", {})),
        instrument=_build(InstrumentConfig, raw.get("instrument", {})),
        sim=_build(SimConfig, raw.get("sim", {})),
        credit=_build(CreditConfig, raw.get("credit", {})),
        analytics=_build(AnalyticsConfig, raw.get("analytics", {})),
        hedge=_build(HedgeConfig, raw.get("hedge", {})),
        vol=_build(VolConfig, raw.get("vol", {})),
        horizon=_build(HorizonConfig, raw.get("horizon", {})),
    )
    # Jumps are a list of specs nested under sim.jumps.
    jump_specs = tuple(_build(JumpSpec, j) for j in raw.get("sim", {}).get("jumps", []))
    cfg = _replace_sim_jumps(cfg, jump_specs)
    return cfg


def _replace_sim_jumps(cfg: Config, jumps: tuple) -> Config:
    from dataclasses import replace
    return replace(cfg, sim=replace(cfg.sim, jumps=jumps))
