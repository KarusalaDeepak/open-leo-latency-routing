#!/usr/bin/env python3
"""Rebuild and aggregate rolling COMMECT timestamp-skew diagnostics.

Each case preserves the canonical raw wall-clock fold boundaries and removes
forecast epochs only after those boundaries are declared.  The cases are a
robustness diagnostic and are never used to select an admission configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--output-dir",
        default="results/commect_rolling_timestamp_sensitivity",
    )
    parser.add_argument(
        "--skew-ms",
        nargs="*",
        type=float,
        default=[500.0, 1000.0, 2000.0, 5000.0],
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if any(value < 0 for value in args.skew_ms):
        raise ValueError("skew limits must be nonnegative")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cases: list[tuple[str, float | None]] = [
        *((f"le_{value:g}ms", value) for value in args.skew_ms),
        ("full", None),
    ]

    for label, threshold in cases:
        case_dir = output / label
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_commect_rolling_origin_validation.py"),
            "--config",
            str(_resolve(args.config)),
            "--output-dir",
            str(case_dir),
            "--folds",
            str(args.folds),
        ]
        if threshold is not None:
            command.extend(["--max-skew-ms", str(threshold)])
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        summary = pd.read_csv(case_dir / "rolling_policy_summary.csv")
        opportunity = pd.read_csv(case_dir / "rolling_opportunity_audit.csv")
        opportunity_row = opportunity[
            opportunity["threshold_ms"].eq(60.0)
        ].iloc[0]
        fold_manifest = pd.read_csv(case_dir / "rolling_fold_manifest.csv")
        metadata = json.loads(
            (case_dir / "rolling_validation_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        gate_selected = set(fold_manifest["gate_selected_policy"].astype(str))
        gate_reasons = sorted(
            set(fold_manifest["gate_selection_reason"].astype(str))
        )
        all_folds_abstained = gate_selected == {"reactive"}

        for policy in (
            "reactive_greedy",
            "qos_shielded_operational_selector",
            "validation_gated_qos_selector",
        ):
            policy_row = summary[summary["policy_name"].eq(policy)].iloc[0]
            rows.append(
                {
                    "skew_case": label,
                    "maximum_skew_ms": threshold,
                    "policy_name": policy,
                    "decision_count": int(policy_row["decision_count"]),
                    "decision_opportunity_count": int(
                        opportunity_row["decision_opportunity_count"]
                    ),
                    "decision_opportunity_rate": float(
                        opportunity_row["decision_opportunity_rate"]
                    ),
                    "success_rate_under_60ms": float(
                        policy_row["success_rate_under_60ms"]
                    ),
                    "mean_realized_latency_ms": float(
                        policy_row["mean_realized_latency_ms"]
                    ),
                    "p95_realized_latency_ms": float(
                        policy_row["p95_realized_latency_ms"]
                    ),
                    "cvar95_realized_latency_ms": float(
                        policy_row["cvar95_realized_latency_ms"]
                    ),
                    "gate_all_folds_abstained": all_folds_abstained,
                    "gate_selected_policy_set": "|".join(sorted(gate_selected)),
                    "gate_selection_reason_set": "|".join(gate_reasons),
                }
            )

        if metadata.get("maximum_inter_path_skew_ms") != threshold:
            raise RuntimeError(f"skew metadata mismatch for {label}")

    result = pd.DataFrame(rows)
    result.to_csv(
        output / "rolling_timestamp_skew_policy_sensitivity.csv",
        index=False,
    )
    diagnostic_metadata = {
        "dataset": "COMMECT",
        "protocol": "rolling-origin maximum-skew robustness rebuild",
        "skew_limits_ms": [*args.skew_ms, None],
        "fold_count": int(args.folds),
        "boundary_semantics": (
            "all cases reuse boundaries declared on the same unfiltered raw "
            "wall-clock schedule; the skew condition then restricts eligible "
            "forecast epochs within each partition"
        ),
        "opportunity_definition_ms": 60.0,
        "used_for_policy_or_gate_selection": False,
        "claim_boundary": (
            "post-specified robustness diagnostic; not evidence for choosing "
            "a skew limit or admission configuration"
        ),
    }
    (output / "rolling_timestamp_skew_metadata.json").write_text(
        json.dumps(diagnostic_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"rolling_timestamp_sensitivity_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
