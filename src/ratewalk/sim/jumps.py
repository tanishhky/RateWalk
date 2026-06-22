"""Programmable jump injection.

Jumps are a pluggable family applied to the simulated short-rate path. Two
ship by default; add your own by subclassing ``JumpModel`` and registering it.

  * JumpDiffusion : Poisson arrivals per path (so different paths jump at
                    different times, as requested), each a fat-tailed shock
                    that decays back over ``recovery_periods``.
  * ScenarioReplay: replay an actual stress trajectory (GFC, Covid, SVB) at a
                    sampled injection step in a configurable fraction of paths.
                    "Keep the outliers, because we see jumps like that in
                    reality too."

Every model edits the rate path IN PLACE (shape (n_paths, n_steps+1)) and is
deterministic given the rng, so a seed reproduces the whole ensemble.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .. import obs

# Stress templates as bps shock-then-recover trajectories on the short rate.
# (Illustrative shapes; calibrate to realized episodes when wiring real data.)
_SCENARIOS: Dict[str, List[float]] = {
    # Covid 2020: emergency cuts, rates slammed to the floor, slow normalize.
    "covid_2020": [-100, -50, -25, 0, 10, 10, 15, 15, 20, 20],
    # GFC 2008: sustained easing over many months.
    "gfc_2008": [-50, -75, -75, -50, -50, -25, -25, -10, 0, 0],
    # SVB 2023: a sharp risk-off dip then a snap-back as crisis contained.
    "svb_2023": [-60, -20, 30, 40, 20, 10, 0, 0, 0, 0],
    # Taper tantrum 2013: a sharp sell-off (yields up) then partial fade.
    "taper_2013": [40, 30, 20, 10, 0, -5, -10, 0, 0, 0],
}


class JumpModel:
    """Base class. Override ``apply``."""

    def apply(self, rate_paths: np.ndarray, *, steps_per_year: int,
              rng: np.random.Generator) -> None:
        raise NotImplementedError


class JumpDiffusion(JumpModel):
    def __init__(self, intensity_per_year: float, size_dist: str, size_df: float,
                 size_scale_bps: float, recovery_periods: int):
        self.intensity = intensity_per_year
        self.size_dist = size_dist
        self.size_df = size_df
        self.size_scale = size_scale_bps / 100.0   # percent
        self.recovery = max(int(recovery_periods), 1)

    def _draw_size(self, rng: np.random.Generator) -> float:
        if self.size_dist == "student_t":
            return float(rng.standard_t(self.size_df) * self.size_scale)
        if self.size_dist == "normal":
            return float(rng.standard_normal() * self.size_scale)
        raise ValueError(f"unknown size_dist {self.size_dist!r}")

    def apply(self, rate_paths: np.ndarray, *, steps_per_year: int,
              rng: np.random.Generator) -> None:
        n_paths, n_steps = rate_paths.shape[0], rate_paths.shape[1] - 1
        p_step = self.intensity / steps_per_year     # arrival prob per step
        n_jumps = 0
        for p in range(n_paths):
            for s in range(1, n_steps + 1):
                if rng.random() < p_step:
                    n_jumps += 1
                    size = self._draw_size(rng)
                    # shock decays linearly over the recovery window
                    for k in range(self.recovery):
                        idx = s + k
                        if idx > n_steps:
                            break
                        decay = (self.recovery - k) / self.recovery
                        rate_paths[p, idx] += size * decay
        np.clip(rate_paths, 0.0, None, out=rate_paths)
        obs.event(channel="sim", kind="jumps.diffusion", n_jumps=int(n_jumps),
                  intensity=self.intensity)


class ScenarioReplay(JumpModel):
    def __init__(self, scenario: str, inject_year_low: float, inject_year_high: float,
                 fraction_of_paths: float, systemic: bool = False):
        if scenario not in _SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; have {list(_SCENARIOS)}")
        self.scenario = scenario
        self.traj = np.array(_SCENARIOS[scenario], dtype=float) / 100.0  # percent
        self.low = inject_year_low
        self.high = inject_year_high
        self.fraction = fraction_of_paths
        self.systemic = systemic

    def apply(self, rate_paths: np.ndarray, *, steps_per_year: int,
              rng: np.random.Generator) -> None:
        n_paths, n_steps = rate_paths.shape[0], rate_paths.shape[1] - 1
        chosen = rng.random(n_paths) < self.fraction
        # systemic: every chosen path is hit at the SAME step; else idiosyncratic.
        common_step = None
        if self.systemic:
            yr = rng.uniform(self.low, self.high)
            common_step = int(np.clip(yr * steps_per_year, 1, n_steps))
        n_hit = 0
        for p in range(n_paths):
            if not chosen[p]:
                continue
            n_hit += 1
            if common_step is not None:
                start = common_step
            else:
                yr = rng.uniform(self.low, self.high)
                start = int(np.clip(yr * steps_per_year, 1, n_steps))
            for k, shock in enumerate(self.traj):
                idx = start + k
                if idx > n_steps:
                    break
                rate_paths[p, idx] += shock
        np.clip(rate_paths, 0.0, None, out=rate_paths)
        obs.event(channel="sim", kind="jumps.scenario", scenario=self.scenario,
                  n_paths_hit=int(n_hit))


def build_jump_models(jump_specs) -> List[JumpModel]:
    """Instantiate jump models from config specs (a tuple of JumpSpec)."""
    models: List[JumpModel] = []
    for spec in jump_specs:
        if not getattr(spec, "enabled", True):
            continue
        if spec.type == "jump_diffusion":
            models.append(JumpDiffusion(
                spec.intensity_per_year, spec.size_dist, spec.size_df,
                spec.size_scale_bps, spec.recovery_periods))
        elif spec.type == "scenario_replay":
            models.append(ScenarioReplay(
                spec.scenario, spec.inject_year_low, spec.inject_year_high,
                spec.fraction_of_paths, spec.systemic))
        else:
            raise ValueError(f"unknown jump type {spec.type!r}")
    return models
