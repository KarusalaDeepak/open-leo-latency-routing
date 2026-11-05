"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TIME_BIN_REQUIRED_COLUMNS = [
    "relative_path",
    "bin_epoch",
    "bin_start_utc",
    "session_date",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_max_ms",
    "observed_replies",
    "path_state",
    "location",
    "target_hint",
    "window_duration",
    "probe_interval",
]


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
    validate_time_bin_table(frame)
    return frame


def validate_time_bin_table(frame: pd.DataFrame, required_columns: Iterable[str] | None = None) -> None:
    """Validate the schema required by the open LEO forecasting pipeline.

    The pipeline is dataset-agnostic as long as the processed table preserves
    these columns and the same row semantics. This lets the repository be
    pointed at other LEO measurement tables with the same schema.
    """

    required = list(required_columns or TIME_BIN_REQUIRED_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "time-bin table is missing required columns: "
            + ", ".join(missing)
            + ". The pipeline expects a processed table with the same schema as "
            "the LENS-derived aggregates."
        )


def audit_trace_concurrency(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    """Summarize timestamp concurrency without inferring controller authority."""

    if frame.empty:
        return {
            "epoch_count": 0,
            "concurrent_epoch_count": 0,
            "max_concurrent_paths": 0,
            "median_concurrent_paths": 0.0,
            "concurrent_row_fraction": 0.0,
            "has_temporally_concurrent_candidates": False,
        }
    counts = frame.groupby("bin_epoch")["relative_path"].nunique()
    concurrent_epochs = counts[counts >= 2].index
    return {
        "epoch_count": int(len(counts)),
        "concurrent_epoch_count": int((counts >= 2).sum()),
        "max_concurrent_paths": int(counts.max()),
        "median_concurrent_paths": float(counts.median()),
        "concurrent_row_fraction": float(
            frame["bin_epoch"].isin(concurrent_epochs).mean()
        ),
        "has_temporally_concurrent_candidates": bool((counts >= 2).any()),
    }


def assign_decision_groups(
    frame: pd.DataFrame,
    *,
    allow_normalized_counterfactual: bool = False,
    literal_single_controller_steering: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign actual-time groups without conflating timing and topology proof."""

    output = frame.copy()
    audit = audit_trace_concurrency(output)
    if audit["has_temporally_concurrent_candidates"]:
        output["session_bin_index"] = pd.factorize(
            output["bin_epoch"],
            sort=True,
        )[0]
        alignment = "actual_timestamp"
    elif allow_normalized_counterfactual:
        output["session_bin_index"] = output.groupby("relative_path").cumcount()
        alignment = "normalized_stage_counterfactual"
    else:
        raise ValueError(
            "the trace contains no simultaneously observed alternative paths; "
            "online path-selection evaluation is invalid. Use "
            "--allow-normalized-counterfactual only for an explicitly labeled "
            "matched-stage diagnostic."
        )
    return output, {
        **audit,
        "decision_alignment": alignment,
        "supports_candidate_outcome_shadow_replay": alignment == "actual_timestamp",
        "supports_shadow_policy_replay": alignment == "actual_timestamp",
        "supports_literal_single_controller_steering": bool(
            alignment == "actual_timestamp" and literal_single_controller_steering
        ),
        "supports_closed_loop_deployment_evidence": False,
        "controller_topology_scope": (
            "source-specific adapter declares one steering controller; offline "
            "shadow replay only"
            if alignment == "actual_timestamp" and literal_single_controller_steering
            else "not established by timestamp concurrency alone"
        ),
    }


def load_compatible_latency_trace(
    path: str | Path,
    column_map: dict[str, str],
    dataset_name: str,
    bin_seconds: int,
) -> pd.DataFrame:
    """Convert a compatible external latency trace to the canonical schema.

    `column_map` maps canonical names to source-column names. At minimum an
    external trace must provide path identity, timestamp, and latency. Optional
    metadata is filled with explicit dataset-specific defaults so the same
    forecasting and decision code can evaluate LENS, Hypatia-derived, or other
    public Starlink traces after a transparent one-time mapping.
    """

    source = pd.read_csv(path)
    essential = {"relative_path", "bin_epoch", "latency_mean_ms"}
    missing_mapping = sorted(essential - set(column_map))
    if missing_mapping:
        raise ValueError(
            "external trace mapping is missing canonical fields: "
            + ", ".join(missing_mapping)
        )
    missing_source = sorted(
        source_name
        for canonical_name, source_name in column_map.items()
        if canonical_name in essential and source_name not in source
    )
    if missing_source:
        raise ValueError(
            "external trace is missing mapped source columns: "
            + ", ".join(missing_source)
        )

    output = pd.DataFrame(index=source.index)
    for canonical_name, source_name in column_map.items():
        if source_name in source:
            output[canonical_name] = source[source_name]

    output["bin_epoch"] = pd.to_numeric(output["bin_epoch"], errors="raise")
    output["latency_mean_ms"] = pd.to_numeric(
        output["latency_mean_ms"], errors="raise"
    )
    output["bin_start_utc"] = pd.to_datetime(
        output["bin_epoch"], unit="s", utc=True
    ).dt.tz_localize(None)
    output["session_date"] = output["bin_start_utc"].dt.normalize()
    output["latency_std_ms"] = output.get("latency_std_ms", 0.0)
    output["latency_max_ms"] = output.get(
        "latency_max_ms", output["latency_mean_ms"]
    )
    output["observed_replies"] = output.get("observed_replies", 1)
    output["path_state"] = output.get("path_state", "available")
    output["location"] = output.get("location", dataset_name)
    output["target_hint"] = output.get("target_hint", output["relative_path"])
    output["window_duration"] = output.get("window_duration", "trace")
    output["probe_interval"] = output.get("probe_interval", f"{bin_seconds * 1000}ms")
    output["measurement_family"] = output.get("measurement_family", dataset_name)
    output["bin_seconds"] = int(bin_seconds)
    output = output.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    output["session_bin_index"] = output.groupby("relative_path").cumcount()
    validate_time_bin_table(output)
    return output


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
