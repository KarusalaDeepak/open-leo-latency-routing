#!/usr/bin/env python3
"""Re-select the complete COMMECT policy gate for each latency objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.evaluation.decision_opportunity import (
    build_candidate_opportunity_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--trace", default="data/processed/commect_multiaccess_10s.csv")
    parser.add_argument(
        "--output-dir",
        default="results/commect_threshold_gate_sensitivity",
    )
    parser.add_argument(
        "--thresholds-ms",
        nargs="+",
        type=float,
        default=(40.0, 60.0, 100.0, 200.0),
    )
    args = parser.parse_args()

    output = REPO_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[pd.DataFrame] = []
    evidence_rows: list[pd.DataFrame] = []

    for gate_index, threshold in enumerate(args.thresholds_ms, start=1):
        label = f"tau_{threshold:g}ms"
        case_dir = output / label
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_commect_multiaccess_validation.py"),
            "--config",
            args.config,
            "--trace",
            args.trace,
            "--output-dir",
            str(case_dir.relative_to(REPO_ROOT)),
            "--latency-budget-ms",
            str(threshold),
            "--gate-planned-uses",
            str(len(args.thresholds_ms)),
            "--gate-use-index",
            str(gate_index),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

        summary = pd.read_csv(case_dir / "policy_summary.csv")
        # The core evaluator retains its historical 60-ms column name for
        # compatibility; expose the correct objective-specific name here.
        summary["success_rate_at_threshold"] = summary[
            "success_rate_under_60ms"
        ]
        summary["success_count_at_threshold"] = (
            summary["success_rate_at_threshold"] * summary["decision_count"]
        ).round().astype(int)
        candidates = pd.read_csv(
            case_dir / "candidate_predictions.csv",
            low_memory=False,
        )
        opportunity, _ = build_candidate_opportunity_audit(
            candidates,
            thresholds_ms=(threshold,),
        )
        opportunity_row = opportunity.iloc[0]
        summary["test_decision_opportunity_count"] = int(
            opportunity_row["decision_opportunity_count"]
        )
        summary["test_decision_opportunity_rate"] = float(
            opportunity_row["decision_opportunity_rate"]
        )
        gate_selected_values = set(
            candidates["validation_gated_fallback_policy"].astype(str)
        )
        if len(gate_selected_values) != 1:
            raise RuntimeError(f"non-constant gate choice for threshold {threshold:g}")
        gate_selected_policy = next(iter(gate_selected_values))
        summary["gate_selected_policy"] = gate_selected_policy
        summary["gate_abstained"] = gate_selected_policy == "reactive"
        summary.insert(0, "threshold_ms", threshold)
        summary_rows.append(summary)

        evidence = pd.read_csv(case_dir / "gate_selection_evidence.csv")
        evidence.insert(0, "threshold_ms", threshold)
        evidence_rows.append(evidence)

    combined_summary = pd.concat(summary_rows, ignore_index=True)
    combined_evidence = pd.concat(evidence_rows, ignore_index=True)
    combined_summary.to_csv(output / "threshold_policy_results.csv", index=False)
    combined_summary[
        combined_summary["policy_name"].isin(
            [
                "reactive_greedy",
                "qos_shielded_operational_selector",
                "validation_gated_qos_selector",
            ]
        )
    ].to_csv(output / "threshold_primary_policy_results.csv", index=False)
    combined_evidence.to_csv(output / "threshold_gate_evidence.csv", index=False)
    manifest = {
        "thresholds_ms": list(args.thresholds_ms),
        "protocol": (
            "Each threshold independently reruns training, residual calibration, "
            "opportunity auditing, confidence-gated policy selection, and final "
            "test; gate alpha is allocated across all thresholds."
        ),
        "gate_family_size": len(args.thresholds_ms),
        "objective_specific_retraining": True,
        "used_for_threshold_selection": False,
        "distinct_from_frozen_score_diagnostic": (
            "This full rerun is the source for objective-specific claims. The "
            "canonical frozen-score threshold-matched diagnostic changes only "
            "the deterministic shield cutoff and does not refit predictors, "
            "calibration quantities, or fallback selection."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(combined_evidence[combined_evidence["selected"]].to_string(index=False))
    print(f"threshold_gate_sensitivity_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
