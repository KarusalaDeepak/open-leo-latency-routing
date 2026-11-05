#!/usr/bin/env python3
"""Convert the public WetLinks release to the canonical latency-bin schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(path: Path) -> pd.DataFrame:
    columns = [
        "site_name",
        "timestamp_start",
        "timestamp_end",
        "server",
        "ping_packet_loss",
        "ping_packets_send",
        "ping_avg",
        "ping_worst",
        "ping_stddev",
        "download",
        "upload",
    ]
    frame = pd.read_csv(path, usecols=columns)
    # Parse on an explicit UTC timeline, then retain timezone-naive UTC values
    # for compatibility with the canonical CSV schema. This avoids local-time
    # dependence for source files that omit an offset.
    frame["timestamp_start"] = pd.to_datetime(
        frame["timestamp_start"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    frame["timestamp_end"] = pd.to_datetime(
        frame["timestamp_end"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    numeric = [column for column in columns if column not in {
        "site_name", "timestamp_start", "timestamp_end", "server"
    }]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["site_name", "timestamp_start", "ping_avg"])


def build_table(input_dir: Path, bin_minutes: int) -> tuple[pd.DataFrame, dict]:
    source_paths = sorted(input_dir.glob("analysis_data_*.csv"))
    if len(source_paths) < 2:
        raise ValueError(
            f"expected at least two WetLinks site files under {input_dir}"
        )

    sources = []
    source_metadata = []
    for path in source_paths:
        source = _load_source(path)
        sources.append(source)
        source_metadata.append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "source_rows": int(len(source)),
                "start": source["timestamp_start"].min().isoformat(),
                "end": source["timestamp_start"].max().isoformat(),
            }
        )

    raw = pd.concat(sources, ignore_index=True)
    raw["bin_start_utc"] = raw["timestamp_start"].dt.floor(f"{bin_minutes}min")
    raw["expected_replies"] = raw["ping_packets_send"] * (
        1.0 - raw["ping_packet_loss"].clip(0.0, 100.0) / 100.0
    )

    grouped = raw.groupby(["site_name", "bin_start_utc"], as_index=False)
    table = grouped.agg(
        latency_mean_ms=("ping_avg", "mean"),
        latency_std_ms=("ping_stddev", "mean"),
        latency_max_ms=("ping_worst", "max"),
        observed_replies=("expected_replies", "sum"),
        packet_loss_pct=("ping_packet_loss", "mean"),
        measurement_count=("ping_avg", "count"),
        download_bps=("download", "mean"),
        upload_bps=("upload", "mean"),
        target_hint=("server", "first"),
    )
    table["relative_path"] = "wetlinks/" + table["site_name"].astype(str)
    table["location"] = table["site_name"].astype(str)
    # Do not infer the datetime storage unit from ``astype('int64')``: pandas
    # 2.x commonly used nanoseconds while pandas 3.x may expose microseconds.
    # Computing a timedifference gives stable Unix seconds in either version.
    unix_epoch = pd.Timestamp("1970-01-01")
    table["bin_epoch"] = (
        (table["bin_start_utc"] - unix_epoch)
        .dt.total_seconds()
        .round()
        .astype("int64")
    )
    table["session_date"] = table["bin_start_utc"].dt.normalize()
    table["path_state"] = np.where(table["observed_replies"] > 0, "active", "unavailable")
    table["window_duration"] = f"{bin_minutes / 60.0:.6f}h"
    table["probe_interval"] = "200ms"
    table["measurement_family"] = "wetlinks_starlink"
    table["bin_seconds"] = int(bin_minutes * 60)

    canonical_order = [
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
        "measurement_family",
        "bin_seconds",
        "packet_loss_pct",
        "measurement_count",
        "download_bps",
        "upload_bps",
    ]
    table = table[canonical_order].sort_values(
        ["relative_path", "bin_epoch"]
    ).reset_index(drop=True)

    epoch_path_counts = table.groupby("bin_epoch")["relative_path"].nunique()
    metadata = {
        "dataset": "WetLinks",
        "dataset_repository": "https://github.com/sys-uos/WetLinks",
        "license": "CC BY-SA 4.0",
        "source_independent_of_lens": True,
        "bin_minutes": bin_minutes,
        "rows": int(len(table)),
        "sites": int(table["relative_path"].nunique()),
        "start": table["bin_start_utc"].min().isoformat(),
        "end": table["bin_start_utc"].max().isoformat(),
        "shared_epoch_count": int((epoch_path_counts >= 2).sum()),
        "candidate_set_semantics": "distributed_observations_not_interchangeable_paths",
        "has_temporally_concurrent_candidates": False,
        "supports_candidate_outcome_shadow_replay": False,
        "supports_literal_single_controller_steering": False,
        "valid_use": "independent longitudinal prediction and risk-calibration validation",
        "invalid_use": "concurrent service-path policy evaluation",
        "source_files": source_metadata,
    }
    return table, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/external/wetlinks_dataset/Preprocessed_Data",
    )
    parser.add_argument(
        "--output",
        default="data/processed/wetlinks_latency_5min.csv",
    )
    parser.add_argument("--bin-minutes", type=int, default=5)
    args = parser.parse_args()

    table, metadata = build_table(_resolve(args.input_dir), args.bin_minutes)
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wetlinks_table_written={output_path}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
