"""Small helpers shared by several stages."""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path


def find_column(columns, candidates) -> str | None:
    """Return the first candidate present in columns (case-insensitive), as spelled in columns."""
    lowered = {str(c).lower(): c for c in columns}
    return next((lowered[c] for c in candidates if c in lowered), None)


def safe_name(value) -> str:
    """Filesystem-safe fire ID; image filenames are traced back to their fire with the same rule."""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).strip())


def parse_date(value) -> datetime:
    import pandas as pd

    parsed = pd.to_datetime(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.to_pydatetime()


class _Tee:
    """Write to the console and a log file at the same time."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def start_log(log_dir: Path) -> Path:
    """Copy everything printed or logged from now on into log_dir/run_<date>_<time>.txt."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path
