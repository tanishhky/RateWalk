"""Turn continuous macro series into discrete Markov states.

Two rate-state conventions (config-selectable):
  * ``levels``     : bucket the policy rate into fixed-width bins. Intuitive,
                     but rate levels are non-stationary.
  * ``increments`` : the FOMC decision itself, in {-50,-25,0,+25,+50,+75} bps.
                     Far more stationary and event-native. Recommended.

CPI is bucketed into inflation regimes by configurable YoY edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateSpace:
    """Labels for a discrete state dimension, in canonical order."""
    name: str
    labels: List[str]

    @property
    def n(self) -> int:
        return len(self.labels)

    def index(self, label: str) -> int:
        return self.labels.index(label)


# ── Rate states ────────────────────────────────────────────────────────────

def build_rate_states(policy: pd.DataFrame, *, mode: str = "increments",
                      increment_grid_bps=(-50, -25, 0, 25, 50, 75),
                      level_bin_width_bps: float = 25.0,
                      resample: str = "ME"):
    """Return (series, StateSpace). ``series`` is a DataFrame [date, state].

    For ``increments`` we resample the daily rate to month-end, diff it, and
    snap each change to the nearest grid increment (in bps).
    """
    p = policy.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date").resample(resample)["rate"].last().dropna()

    if mode == "increments":
        grid = np.array(sorted(increment_grid_bps), dtype=float)
        labels = [f"{int(g):+d}bps" for g in grid]
        changes_bps = (p.diff() * 100).fillna(0.0)
        snapped = changes_bps.apply(lambda x: grid[int(np.argmin(np.abs(grid - x)))])
        states = snapped.apply(lambda g: f"{int(g):+d}bps")
        ss = StateSpace(name="rate_increment", labels=labels)
    elif mode == "levels":
        w = level_bin_width_bps / 100.0
        lo = float(np.floor(p.min() / w) * w)
        hi = float(np.ceil(p.max() / w) * w)
        edges = np.arange(lo, hi + w, w)
        labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
        idx = np.clip(np.digitize(p.values, edges) - 1, 0, len(labels) - 1)
        states = pd.Series([labels[i] for i in idx], index=p.index)
        ss = StateSpace(name="rate_level", labels=labels)
    else:
        raise ValueError(f"unknown rate mode {mode!r}")

    out = states.rename("state").reset_index()
    return out, ss


# ── CPI states ─────────────────────────────────────────────────────────────

def build_cpi_states(cpi: pd.DataFrame, *, bins_yoy=(0.0, 2.0, 3.0, 5.0)):
    """Bucket CPI YoY into inflation regimes by the given edges."""
    edges = list(bins_yoy)
    labels = (["<{:.0f}".format(edges[0])]
              + ["{:.0f}-{:.0f}".format(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
              + [">{:.0f}".format(edges[-1])])
    full_edges = [-np.inf] + edges + [np.inf]
    idx = np.digitize(cpi["cpi_yoy"].values, full_edges) - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    out = pd.DataFrame({
        "date": pd.to_datetime(cpi["date"]),
        "release_date": pd.to_datetime(cpi["release_date"]),
        "state": [labels[i] for i in idx],
    })
    ss = StateSpace(name="cpi_regime", labels=labels)
    return out, ss


# ── Joint alignment ─────────────────────────────────────────────────────────

def build_joint_series(rate_states: pd.DataFrame, cpi_states: pd.DataFrame, *,
                       alignment: str = "decoupled") -> pd.DataFrame:
    """Align rate states with the CPI regime that was PUBLIC at each rate date.

    Returns [date, rate_state, cpi_state]. The key no-look-ahead step is the
    merge_asof on the CPI *release_date*, so each rate observation only sees a
    CPI print that had already been published.
    """
    r = rate_states.rename(columns={"state": "rate_state"}).copy()
    r["date"] = pd.to_datetime(r["date"])
    c = cpi_states.rename(columns={"state": "cpi_state"}).copy()
    # Use the release date as the as-of key (publication lag honored).
    c = c.sort_values("release_date")
    r = r.sort_values("date")
    joint = pd.merge_asof(
        r, c[["release_date", "cpi_state"]],
        left_on="date", right_on="release_date", direction="backward",
    )
    joint["cpi_state"] = joint["cpi_state"].fillna(cpi_states["state"].iloc[0])
    return joint[["date", "rate_state", "cpi_state"]].reset_index(drop=True)
