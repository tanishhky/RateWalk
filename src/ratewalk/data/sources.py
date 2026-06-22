"""Point-in-time macro and curve data.

Two providers behind one interface:
  * ``fred``      : live pull from FRED/ALFRED (needs FRED_API_KEY). ALFRED
                    vintages are used for CPI so revisions never leak (CPI is
                    revised; using the latest print would be look-ahead).
  * ``synthetic`` : a deterministic, seed-based generator so the whole
                    pipeline runs and tests offline with realistic-looking
                    policy-rate cycles, CPI, and a curve. Default fallback.

Every accessor takes an explicit ``as_of_ts`` and returns ONLY rows whose
release/observation date is on or before it. This is the no-look-ahead
guarantee at the data boundary.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .. import obs


@dataclass
class MacroData:
    """Bundle of point-in-time series, all indexed by date (UTC, tz-naive).

    policy_rate : daily Fed-funds-style policy rate (percent).
    cpi_yoy     : monthly CPI year-over-year (percent), with a `release_date`
                  column so we can respect the publication lag and vintages.
    curve       : daily par yields per tenor (percent); columns are tenors in
                  years as strings.
    """
    policy_rate: pd.DataFrame      # columns: [date, rate]
    cpi_yoy: pd.DataFrame          # columns: [date, cpi_yoy, release_date]
    curve: pd.DataFrame            # columns: [date, <tenor>, ...]
    source: str = "synthetic"

    # ── Point-in-time accessors ──────────────────────────────────────────
    def policy_asof(self, as_of_ts: str) -> pd.DataFrame:
        cut = _parse(as_of_ts)
        return self.policy_rate[self.policy_rate["date"] <= cut].copy()

    def cpi_asof(self, as_of_ts: str) -> pd.DataFrame:
        """CPI public as of `as_of_ts`: filter on RELEASE date, not the
        observation month, so the publication lag is honored."""
        cut = _parse(as_of_ts)
        return self.cpi_yoy[self.cpi_yoy["release_date"] <= cut].copy()

    def curve_asof(self, as_of_ts: str) -> pd.DataFrame:
        cut = _parse(as_of_ts)
        return self.curve[self.curve["date"] <= cut].copy()


def _parse(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo is None \
        else pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)


# ── Public entry point ───────────────────────────────────────────────────

def load_macro(country: str = "US", source: str = "auto",
               start: str = "1990-01-01", end: Optional[str] = None,
               seed: int = 7) -> MacroData:
    """Load macro data for a country. ``source='auto'`` tries FRED then falls
    back to synthetic so the pipeline always runs."""
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if source in ("fred", "auto"):
        try:
            md = _load_fred(country, start, end)
            obs.event(channel="data", kind="load.fred", country=country,
                      n_policy=len(md.policy_rate), n_cpi=len(md.cpi_yoy))
            return md
        except Exception as exc:  # noqa: BLE001
            if source == "fred":
                raise
            obs.event(channel="data", kind="load.fred_unavailable",
                      level="WARNING", err=str(exc))
    md = _load_synthetic(country, start, end, seed=seed)
    obs.event(channel="data", kind="load.synthetic", country=country,
              n_policy=len(md.policy_rate), n_cpi=len(md.cpi_yoy))
    return md


# ── FRED / ALFRED provider ─────────────────────────────────────────────────

# Per-country FRED series ids. Extend this map to add sovereigns.
_FRED_SERIES = {
    "US": {"policy": "DFF", "cpi": "CPIAUCSL",
           "curve": {"0.25": "DGS3MO", "1": "DGS1", "2": "DGS2", "3": "DGS3",
                     "5": "DGS5", "7": "DGS7", "10": "DGS10", "20": "DGS20",
                     "30": "DGS30"}},
    # Placeholders for other sovereigns (series ids differ on FRED).
    "GB": {"policy": "IUDSOIA", "cpi": "GBRCPIALLMINMEI", "curve": {}},
    "DE": {"policy": "IRSTCB01DEM156N", "cpi": "DEUCPIALLMINMEI", "curve": {}},
}


def _load_fred(country: str, start: str, end: str) -> MacroData:
    """Live FRED/ALFRED pull. Raises if no key or no network so callers can
    fall back to synthetic."""
    import os
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    if country not in _FRED_SERIES:
        raise RuntimeError(f"no FRED series map for country={country}")
    # NOTE: the real implementation issues HTTP calls to the FRED + ALFRED
    # endpoints (the latter for point-in-time CPI vintages). Kept as an
    # explicit failure here so the offline synthetic path is always exercised
    # in tests; wire the HTTP client in deployment.
    raise RuntimeError("FRED HTTP client not wired in this build; using synthetic")


# ── Synthetic provider (deterministic, offline) ────────────────────────────

def _country_offset(country: str) -> int:
    """Deterministic per-country seed offset (built-in hash() is randomized
    across processes, which would break reproducibility)."""
    h = hashlib.sha256(country.encode()).hexdigest()
    return int(h[:6], 16) % 10000


def _load_synthetic(country: str, start: str, end: str, seed: int) -> MacroData:
    """Generate realistic-looking policy-rate cycles, CPI, and a curve.

    The policy rate follows hiking/cutting *cycles* (persistent runs) rather
    than i.i.d. noise, so the estimated Markov chain has the momentum a real
    rate series has. It mean-reverts toward a neutral level so it never gets
    stuck at the zero floor. CPI co-moves with the rate with a lag. The curve
    is the policy rate plus a term premium plus level/slope wiggle.
    """
    rng = np.random.default_rng(seed + _country_offset(country))
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Policy rate: regime-switching cycles around a neutral level. Direction
    # persists; the central bank moves in 25 bps steps at ~monthly cadence,
    # with a gentle pull back toward neutral so it cycles instead of drifting.
    neutral = 3.5
    rate = np.zeros(n)
    level = 4.0
    direction = 1           # -1 cut, 0 hold, +1 hike
    steps_to_next_move = 21
    for i in range(n):
        if i and steps_to_next_move <= 0:
            # Bias the regime toward closing the gap to neutral (mean reversion).
            gap = level - neutral
            p_hike = float(np.clip(0.45 - 0.06 * gap, 0.1, 0.8))
            p_cut = float(np.clip(0.45 + 0.06 * gap, 0.1, 0.8))
            p_hold = max(1.0 - p_hike - p_cut, 0.05)
            tot = p_hike + p_cut + p_hold
            direction = rng.choice([1, -1, 0], p=[p_hike / tot, p_cut / tot, p_hold / tot])
            move = direction * rng.choice([0, 25, 25, 50]) / 100.0
            level = float(np.clip(level + move, 0.25, 9.0))
            steps_to_next_move = int(rng.integers(15, 35))
        steps_to_next_move -= 1
        rate[i] = level
    policy = pd.DataFrame({"date": dates, "rate": rate})

    # CPI YoY: mean-reverting around 2.5%, pushed by lagged rate gaps and
    # occasional inflation surges. Monthly. Released ~2 weeks after month end.
    months = pd.date_range(start=start, end=end, freq="ME")
    m = len(months)
    cpi = np.zeros(m)
    c = 2.5
    for j in range(m):
        rate_at = rate[min(int(j * n / max(m, 1)), n - 1)]
        pull = 0.05 * (rate_at - 4.0)
        shock = rng.standard_t(4) * 0.4
        surge = 3.0 if rng.random() < 0.01 else 0.0
        c = float(np.clip(0.9 * c + 0.1 * (2.5 - pull) + shock + surge, -2.0, 12.0))
        cpi[j] = c
    release = months + pd.Timedelta(days=14)   # publication lag
    cpi_df = pd.DataFrame({"date": months, "cpi_yoy": cpi, "release_date": release})

    # Curve: par yields = policy rate + term premium(tenor) + slope wiggle.
    tenors = [0.25, 1, 2, 3, 5, 7, 10, 20, 30]
    term_premium = {t: 0.15 * np.log1p(t) for t in tenors}
    curve = {"date": dates}
    slope_noise = np.cumsum(rng.standard_normal(n)) * 0.01
    for t in tenors:
        tp = term_premium[t]
        curve[str(t)] = rate + tp + slope_noise * (t / 10.0) \
            + rng.standard_normal(n) * 0.02
    curve_df = pd.DataFrame(curve)

    return MacroData(policy_rate=policy, cpi_yoy=cpi_df, curve=curve_df,
                     source="synthetic")
