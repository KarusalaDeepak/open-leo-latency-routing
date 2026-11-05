#!/usr/bin/env python3
"""Rebuild the exact 1-s COMMECT result with strict alignment sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from open_leo_latency_routing.evaluation.significance import (
    build_paired_policy_significance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "reactive_greedy",
    "qos_filter_then_ensemble_selector",
    "qos_shielded_operational_selector",
    "validation_gated_qos_selector",
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--trace",
        default="data/processed/commect_multiaccess_1s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/commect_1s_alignment_audit",
    )
    parser.add_argument("--strict-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Aggregate existing case outputs instead of rerunning validation.",
    )
    args = parser.parse_args()
    if args.strict_skew_ms <= 0:
        raise ValueError("--strict-skew-ms must be positive")

    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = (("full", None), ("strict", float(args.strict_skew_ms)))
    summary_rows: list[pd.DataFrame] = []
    significance_rows: list[pd.DataFrame] = []
    scheduled_rows: list[pd.DataFrame] = []

    for label, skew_ms in cases:
        case_dir = output / label
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_commect_multiaccess_validation.py"),
            "--config",
            str(_resolve(args.config)),
            "--trace",
            str(_resolve(args.trace)),
            "--output-dir",
            str(case_dir),
            "--gate-planned-uses",
            str(len(cases)),
            "--gate-use-index",
            str(1 if label == "full" else 2),
        ]
        if skew_ms is not None:
            command.extend(["--max-skew-ms", str(skew_ms)])
        if not args.reuse_existing or not (case_dir / "policy_summary.csv").exists():
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

        summary = pd.read_csv(case_dir / "policy_summary.csv")
        selected = summary[summary["policy_name"].isin(POLICIES)].copy()
        selected.insert(0, "alignment_case", label)
        selected.insert(1, "maximum_inter_path_skew_ms", skew_ms)
        summary_rows.append(selected)

        decisions = pd.read_csv(case_dir / "policy_decisions.csv")
        comparison = build_paired_policy_significance(
            decisions,
            comparisons=[
                (
                    "ungated_ensemble_vs_reactive",
                    "qos_filter_then_ensemble_selector",
                    "reactive_greedy",
                ),
                (
                    "shield_vs_reactive",
                    "qos_shielded_operational_selector",
                    "reactive_greedy",
                ),
                (
                    "gate_vs_reactive",
                    "validation_gated_qos_selector",
                    "reactive_greedy",
                ),
            ],
            metric_columns=["success_under_budget", "realized_next_latency_ms"],
            block_length=max(2, round(decisions["session_bin_index"].nunique() ** (1 / 3))),
            segment_columns="continuity_segment_id",
        )
        comparison.insert(0, "alignment_case", label)
        comparison.insert(1, "maximum_inter_path_skew_ms", skew_ms)
        significance_rows.append(comparison)

        scheduled = pd.read_csv(case_dir / "scheduled_epoch_service_success.csv")
        scheduled = scheduled[scheduled["policy_name"].isin(POLICIES)].copy()
        scheduled.insert(0, "alignment_case", label)
        scheduled.insert(1, "maximum_inter_path_skew_ms", skew_ms)
        scheduled_rows.append(scheduled)

    pd.concat(summary_rows, ignore_index=True).to_csv(
        output / "alignment_policy_summary.csv", index=False
    )
    pd.concat(significance_rows, ignore_index=True).to_csv(
        output / "alignment_paired_diagnostics.csv", index=False
    )
    pd.concat(scheduled_rows, ignore_index=True).to_csv(
        output / "alignment_scheduled_epoch_success.csv", index=False
    )
    metadata = {
        "dataset": "COMMECT",
        "cadence_seconds": 1,
        "strict_maximum_inter_path_skew_ms": float(args.strict_skew_ms),
        "alignment_cases_predeclared_for_this_audit": ["full", "strict"],
        "used_for_candidate_or_gate_selection": False,
        "inference_boundary": (
            "paired moving-block intervals are descriptive within one drive; "
            "the evidence gate still sees one independent collection"
        ),
        "unconditional_endpoint": (
            "scheduled-epoch lower bound counts every unevaluable or no-action "
            "test slot as failure"
        ),
    }
    (output / "alignment_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"alignment_audit_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
