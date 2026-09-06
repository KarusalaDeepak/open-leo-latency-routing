#!/usr/bin/env python3
"""Evaluate policy robustness across simulator severity profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "benign": {
        "load_multiplier": 0.7,
        "handover_penalty_ms": 6.0,
        "gateway_attenuation_ms": 15.0,
        "satellite_incident_ms": 25.0,
        "invisible_path_penalty_ms": 70.0,
    },
    "nominal": {
        "load_multiplier": 1.0,
        "handover_penalty_ms": 10.0,
        "gateway_attenuation_ms": 24.0,
        "satellite_incident_ms": 38.0,
        "invisible_path_penalty_ms": 85.0,
    },
    "adverse": {
        "load_multiplier": 1.5,
        "handover_penalty_ms": 18.0,
        "gateway_attenuation_ms": 40.0,
        "satellite_incident_ms": 60.0,
        "invisible_path_penalty_ms": 110.0,
    },
}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=list(PROFILES),
    )
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        default="results/simulator_parameter_sensitivity",
    )
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for profile_name in args.profiles:
        profile = PROFILES[profile_name]
        trace_path = output_dir / f"trace_{profile_name}.csv"
        run_dir = output_dir / profile_name
        generator_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_physics_informed_multipath_trace.py"),
            "--output",
            str(trace_path),
            "--duration-hours",
            str(args.duration_hours),
            "--seed",
            str(args.seed),
        ]
        for name, value in profile.items():
            generator_command.extend(
                ["--" + name.replace("_", "-"), str(value)]
            )
        subprocess.run(generator_command, check=True)
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_independent_multipath_validation.py"),
                "--trace",
                str(trace_path),
                "--output-dir",
                str(run_dir),
            ],
            check=True,
        )
        summary = pd.read_csv(run_dir / "independent_policy_summary.csv")
        summary["severity_profile"] = profile_name
        rows.append(summary)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "parameter_sensitivity_summary.csv", index=False)
    (output_dir / "parameter_profiles.json").write_text(
        json.dumps(
            {name: PROFILES[name] for name in args.profiles},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"parameter_sensitivity_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
