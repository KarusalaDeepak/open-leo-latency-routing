#!/usr/bin/env python3
"""Run next-bin policy experiments at 5/10/30/60-second control cadences."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--tables-dir",
        default="data/processed/temporal_resolutions",
    )
    parser.add_argument(
        "--output-dir",
        default="results/temporal_resolution_evaluation",
    )
    parser.add_argument("--resolutions", nargs="+", type=int, default=[5, 10, 30, 60])
    args = parser.parse_args()

    tables_dir = _resolve(args.tables_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = []
    for resolution in sorted(set(args.resolutions)):
        input_path = tables_dir / f"ping_time_bins_{resolution}s.csv"
        if not input_path.exists():
            raise FileNotFoundError(
                f"missing {input_path}; run build_temporal_resolution_tables.py first"
            )
        resolution_output = output_dir / f"{resolution}s"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_service_path_experiments.py"),
                "--config",
                str(_resolve(args.config)),
                "--time-bins",
                str(input_path),
                "--output-dir",
                str(resolution_output),
                "--horizon-seconds",
                str(resolution),
                "--allow-normalized-counterfactual",
            ],
            check=True,
        )
        summary = pd.read_csv(resolution_output / "policy_summary.csv")
        summary["resolution_seconds"] = resolution
        summary["forecast_horizon_seconds"] = resolution
        combined_rows.append(summary)

    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(output_dir / "temporal_resolution_policy_summary.csv", index=False)
    print(f"temporal_resolution_summary_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
