#!/usr/bin/env python3
"""Run and aggregate COMMECT timestamp-skew sensitivity experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from open_leo_latency_routing.evaluation.decision_opportunity import (
    build_candidate_opportunity_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/commect_timestamp_sensitivity")
    parser.add_argument(
        "--skew-ms",
        nargs="*",
        type=float,
        default=[500.0, 1000.0, 2000.0, 5000.0],
    )
    args = parser.parse_args()
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    cases: list[tuple[str, float | None]] = [
        *( (f"le_{value:g}ms", value) for value in args.skew_ms ),
        ("full", None),
    ]
    for gate_index, (label, threshold) in enumerate(cases, start=1):
        case_dir = output / label
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_commect_multiaccess_validation.py"),
            "--config",
            str(_resolve(args.config)),
            "--output-dir",
            str(case_dir),
            "--gate-planned-uses",
            str(len(cases)),
            "--gate-use-index",
            str(gate_index),
        ]
        if threshold is not None:
            command.extend(["--max-skew-ms", str(threshold)])
        subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        summary = pd.read_csv(case_dir / "policy_summary.csv")
        candidates = pd.read_csv(
            case_dir / "candidate_predictions.csv",
            low_memory=False,
        )
        opportunity_audit, _ = build_candidate_opportunity_audit(
            candidates,
            thresholds_ms=(60.0,),
        )
        opportunity_row = opportunity_audit.iloc[0]
        metadata = json.loads((case_dir / "validation_metadata.json").read_text())
        gate_selected_values = set(
            candidates["validation_gated_fallback_policy"].astype(str)
        )
        if len(gate_selected_values) != 1:
            raise RuntimeError(f"non-constant fixed gate choice for {label}")
        gate_selected_policy = next(iter(gate_selected_values))
        for policy in (
            "reactive_greedy",
            "age_aware_reactive_selector",
            "qos_shielded_operational_selector",
            "validation_gated_qos_selector",
        ):
            row = summary[summary["policy_name"].eq(policy)].iloc[0]
            rows.append(
                {
                    "skew_case": label,
                    "maximum_skew_ms": threshold,
                    "policy_name": policy,
                    "decision_count": int(row["decision_count"]),
                    "decision_opportunity_count": int(
                        opportunity_row["decision_opportunity_count"]
                    ),
                    "decision_opportunity_rate": float(
                        opportunity_row["decision_opportunity_rate"]
                    ),
                    "success_rate_under_60ms": float(row["success_rate_under_60ms"]),
                    "mean_realized_latency_ms": float(row["mean_realized_latency_ms"]),
                    "p95_realized_latency_ms": float(row["p95_realized_latency_ms"]),
                    "cvar95_realized_latency_ms": float(row["cvar95_realized_latency_ms"]),
                    "gate_abstained": gate_selected_policy == "reactive",
                    "gate_selected_policy": gate_selected_policy,
                    "gate_selection_reason": metadata["gate_selection_reason"],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "timestamp_skew_policy_sensitivity.csv", index=False)
    (output / "timestamp_skew_sensitivity_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "COMMECT",
                "protocol": "fixed four-block maximum-skew robustness rebuild",
                "skew_limits_ms": [*args.skew_ms, None],
                "opportunity_definition_ms": 60.0,
                "used_for_policy_or_gate_selection": False,
                "claim_boundary": (
                    "overlapping fixed-holdout robustness diagnostic; not a "
                    "basis for selecting a skew threshold"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
