"""Observability spine: structured JSONL events.

The same pattern used across PinSight / DriftEdge / Sentinel. Every estimate,
simulation batch, and analytic emits one line of JSON with a timestamp,
channel, kind, duration, and status, so a run is fully traceable and a
Sentinel-style monitor can read it.

Usage
-----
    from . import obs
    obs.event(channel="markov", kind="estimate.done", level="INFO",
              n_obs=512, n_states=6, duration_ms=12.4)

    with obs.timed(channel="sim", kind="engine.run"):
        ...   # duration is measured and emitted automatically
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOG_PATH: Optional[Path] = None
_RUN_ID: Optional[str] = None


def configure(log_dir: Optional[Path] = None, run_id: Optional[str] = None) -> None:
    """Point the spine at a log directory. If never called, events go to
    stderr only. Idempotent."""
    global _LOG_PATH, _RUN_ID
    _RUN_ID = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / f"run-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"


def event(*, channel: str, kind: str, level: str = "INFO", **fields: Any) -> None:
    """Emit one structured event."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "run_id": _RUN_ID,
        "pid": os.getpid(),
        "channel": channel,
        "kind": kind,
        "level": level,
        **fields,
    }
    line = json.dumps(rec, default=str)
    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    if level in ("WARNING", "ERROR") or os.getenv("RATEWALK_VERBOSE"):
        print(line, file=sys.stderr)


@contextmanager
def timed(*, channel: str, kind: str, level: str = "INFO", **fields: Any):
    """Context manager that times a block and emits duration_ms on exit."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - we re-raise after logging
        status = "error"
        fields["err"] = str(exc)
        fields["exc_type"] = type(exc).__name__
        raise
    finally:
        event(channel=channel, kind=kind, level=("ERROR" if status == "error" else level),
              duration_ms=round((time.perf_counter() - t0) * 1000, 3), status=status, **fields)
