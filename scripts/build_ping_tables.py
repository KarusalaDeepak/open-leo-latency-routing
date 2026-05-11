#!/usr/bin/env python3
"""Parse LENS ping logs into modeling-ready session and observation tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.aggregations import aggregate_ping_file
from open_leo_latency_routing.data.ping_logs import summarize_ping_file


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_target_files(data_root: Path, manifest_csv: Path | None, max_files: int | None) -> list[Path]:
    """Load candidate ping files either from manifest or by scanning the dataset root."""
    if manifest_csv and manifest_csv.exists():
        frame = pd.read_csv(manifest_csv)
        files = [data_root / relative_path for relative_path in frame["relative_path"].tolist()]
    else:
        files = sorted(path for path in data_root.rglob("ping-*.txt") if path.is_file())

    if max_files is not None:
        return files[:max_files]
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-csv", default="results/candidate_manifest.csv")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--sample-rows-per-file", type=int, default=250)
    parser.add_argument("--time-bin-seconds", type=int, default=60)
    parser.add_argument("--session-out", default="data/processed/ping_session_summary.csv")
    parser.add_argument("--observations-out", default="data/processed/ping_observations_sample.csv")
    parser.add_argument("--aggregates-out", default="data/processed/ping_time_bins.csv")
    args = parser.parse_args()

    data_root = _resolve_repo_path(args.data_root)
    manifest_csv = _resolve_repo_path(args.manifest_csv)
    session_out = _resolve_repo_path(args.session_out)
    observations_out = _resolve_repo_path(args.observations_out)
    aggregates_out = _resolve_repo_path(args.aggregates_out)

    target_files = _load_target_files(data_root, manifest_csv, args.max_files)
    session_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    for path in target_files:
        summary, sampled_rows = summarize_ping_file(
            path=path,
            data_root=data_root,
            sample_rows_per_file=args.sample_rows_per_file,
        )
        session_rows.append(summary)
        observation_rows.extend(sampled_rows)
        aggregate_rows.extend(
            aggregate_ping_file(
                path=path,
                data_root=data_root,
                bin_seconds=args.time_bin_seconds,
            )
        )

    session_out.parent.mkdir(parents=True, exist_ok=True)
    observations_out.parent.mkdir(parents=True, exist_ok=True)
    aggregates_out.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(session_rows).to_csv(session_out, index=False)
    pd.DataFrame(observation_rows).to_csv(observations_out, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregates_out, index=False)

    print(f"session_summary_written={session_out}")
    print(f"observation_sample_written={observations_out}")
    print(f"time_bin_aggregates_written={aggregates_out}")
    print(f"parsed_files={len(session_rows)}")
    print(f"sampled_observations={len(observation_rows)}")
    print(f"time_bins={len(aggregate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
