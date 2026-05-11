"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_dataset_files(data_root: str | Path) -> list[Path]:
    """Return a sorted list of files under the raw dataset root."""
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def load_time_bin_table(path: str | Path) -> pd.DataFrame:
    """Load the aggregated ping time-bin table with parsed timestamps."""
    frame = pd.read_csv(path)
    if frame.empty:
        return frame

    frame["bin_start_utc"] = pd.to_datetime(frame["bin_start_utc"], utc=False)
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce")
    frame = frame.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    frame["session_bin_index"] = frame.groupby("relative_path").cumcount()
    return frame


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
