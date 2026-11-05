#!/usr/bin/env python3
"""Build and evaluate a larger LENS ping subset when raw logs are available."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _count_ping_logs(data_root: Path) -> int:
    if not data_root.exists():
        return 0
    return sum(1 for path in data_root.rglob("ping-*.txt") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/lens_2025_03/LENS-2025-03")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--time-bin-seconds", type=int, default=60)
    parser.add_argument("--aggregates-out", default="data/processed/ping_time_bins_large.csv")
    parser.add_argument("--session-out", default="data/processed/ping_session_summary_large.csv")
    parser.add_argument("--observations-out", default="data/processed/ping_observations_sample_large.csv")
    parser.add_argument("--output-dir", default="results/service_path_experiments_large")
    args = parser.parse_args()

    data_root = _resolve_repo_path(args.data_root)
    available_logs = _count_ping_logs(data_root)
    if available_logs == 0:
        print(f"raw_ping_logs_found=0")
        print(f"expected_data_root={data_root}")
        print("larger_subset_status=skipped_missing_raw_logs")
        return 2

    selected_files = min(args.max_files, available_logs)
    print(f"raw_ping_logs_found={available_logs}")
    print(f"selected_ping_logs={selected_files}")

    build_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_ping_tables.py"),
        "--data-root",
        str(data_root),
        "--manifest-csv",
        "",
        "--max-files",
        str(selected_files),
        "--time-bin-seconds",
        str(args.time_bin_seconds),
        "--session-out",
        str(_resolve_repo_path(args.session_out)),
        "--observations-out",
        str(_resolve_repo_path(args.observations_out)),
        "--aggregates-out",
        str(_resolve_repo_path(args.aggregates_out)),
    ]
    subprocess.run(build_command, check=True)

    experiment_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_service_path_experiments.py"),
        "--time-bins",
        str(_resolve_repo_path(args.aggregates_out)),
        "--output-dir",
        str(_resolve_repo_path(args.output_dir)),
    ]
    subprocess.run(experiment_command, check=True)
    print(f"larger_subset_status=completed")
    print(f"larger_subset_results={_resolve_repo_path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
