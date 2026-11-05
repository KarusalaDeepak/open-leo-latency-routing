#!/usr/bin/env python3
"""Audit whether generated evidence supports the manuscript's strongest claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPATIA_TARGET_TRACE = "data/processed/hypatia_service_paths_10s.csv"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _concurrency_evidence(audit: dict) -> dict[str, object]:
    """Keep the audit factual without inheriting legacy deployment labels."""

    keys = (
        "epoch_count",
        "concurrent_epoch_count",
        "max_concurrent_paths",
        "median_concurrent_paths",
        "concurrent_row_fraction",
        "has_temporally_concurrent_candidates",
        "decision_alignment",
        "supports_candidate_outcome_shadow_replay",
        "supports_shadow_policy_replay",
        "supports_literal_single_controller_steering",
        "supports_closed_loop_deployment_evidence",
        "controller_topology_scope",
    )
    return {key: audit[key] for key in keys if key in audit}


def _supports_hypatia_zero_shot_replication(
    trace_metadata: dict,
    zero_shot_metadata: dict,
) -> bool:
    """Bind the established-simulator pass to the canonical Hypatia target."""

    concurrency = zero_shot_metadata.get("target_concurrency_audit", {})
    return bool(
        trace_metadata.get("is_hypatia_output")
        and trace_metadata.get("uses_tle_orbital_propagation")
        and trace_metadata.get("uses_dynamic_shortest_path_state")
        and zero_shot_metadata.get("zero_shot_transfer", False)
        and zero_shot_metadata.get("target_trace") == HYPATIA_TARGET_TRACE
        and zero_shot_metadata.get("target_family")
        == trace_metadata.get("dataset_name")
        and zero_shot_metadata.get("target_rows_used_for_training") == 0
        and zero_shot_metadata.get("target_rows_used_for_calibration") == 0
        and concurrency.get("has_temporally_concurrent_candidates")
        and concurrency.get("decision_alignment") == "actual_timestamp"
        and concurrency.get("supports_shadow_policy_replay")
    )


def _add(
    rows: list[dict[str, object]],
    category: str,
    check: str,
    passed: bool,
    evidence: str,
    likely_comment: str,
    required_action: str,
) -> None:
    rows.append(
        {
            "category": category,
            "check": check,
            "status": "PASS" if passed else "PENDING",
            "evidence": evidence,
            "likely_future_reviewer_comment": likely_comment,
            "required_action": "" if passed else required_action,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/reviewer_readiness",
    )
    parser.add_argument(
        "--independent-dir",
        default="results/transactions_orbital_validation",
    )
    parser.add_argument(
        "--seed-matrix-dir",
        default="results/transactions_seed_matrix",
    )
    parser.add_argument(
        "--sensitivity-dir",
        default="results/simulator_parameter_sensitivity",
    )
    parser.add_argument(
        "--evidence-dir",
        default="results/transactions_evidence",
        help=(
            "Evidence tree to audit. The rebuild runner points this at its "
            "validated staging tree before publishing it."
        ),
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help=(
            "Write the audit and return success even when external or deployment "
            "evidence remains pending. Pending items are never relabeled as passes."
        ),
    )
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    independent_dir = _resolve(args.independent_dir)
    seed_dir = _resolve(args.seed_matrix_dir)
    sensitivity_dir = _resolve(args.sensitivity_dir)
    evidence_dir = _resolve(args.evidence_dir)
    rows: list[dict[str, object]] = []

    independent_metadata = _read_json(independent_dir / "run_metadata.json")
    concurrency = independent_metadata.get("concurrency_audit", {})
    concurrent_pass = bool(
        concurrency.get("supports_shadow_policy_replay")
        and concurrency.get("max_concurrent_paths", 0) >= 2
    )
    _add(
        rows,
        "artifact_integrity",
        "concurrent alternative-path evaluation",
        concurrent_pass,
        json.dumps(_concurrency_evidence(concurrency), sort_keys=True),
        "The policy is evaluated on unrelated sessions rather than simultaneous alternatives.",
        "Run the complete policy matrix on a trace with at least two feasible paths at each timestamp.",
    )

    partition = independent_metadata.get("data_partition_protocol", {})
    _add(
        rows,
        "artifact_integrity",
        "leakage-safe partitions and graph context",
        {
            "session_holdout",
            "temporal_holdout",
            "graph_context_isolation",
        }.issubset(partition),
        json.dumps(partition, sort_keys=True),
        "Autocorrelated bins or graph aggregates leak across train and test partitions.",
        "Regenerate results with disjoint session/temporal splits and partition-local graph features.",
    )

    exact_horizon_path = evidence_dir / "exact_horizon_audit.csv"
    exact_horizon = (
        pd.read_csv(exact_horizon_path)
        if exact_horizon_path.exists()
        else pd.DataFrame()
    )
    expected_semantics = (
        "exact_target_bin_start_with_complete_intermediate_sequence"
    )
    exact_horizon_pass = bool(
        len(exact_horizon) >= 4
        and exact_horizon["endpoint_semantics"].eq(expected_semantics).all()
        and exact_horizon["scheduled_history_semantics"].eq(
            "exact_wall_clock_slots"
        ).all()
        and exact_horizon["decision_cadence_seconds"].gt(0).all()
        and exact_horizon["require_complete_decision_epochs"].eq(True).all()
        and exact_horizon["retained_exact_target_row_count"].gt(0).all()
        and exact_horizon["excluded_nonexact_gap_row_count"].ge(0).all()
        and exact_horizon["excluded_incomplete_decision_epoch_count"].ge(0).all()
        and exact_horizon["scheduled_grid_missing_row_count"].ge(0).all()
        and exact_horizon["history_gap_row_count"].ge(0).all()
    )
    _add(
        rows,
        "artifact_integrity",
        "exact adjacent-bin target starts and complete candidate epochs",
        exact_horizon_pass,
        exact_horizon.to_json(orient="records"),
        "A missing aggregate bin is treated as the next forecast step or silently removes only one candidate path.",
        "Rebuild all policy evidence with explicit cadence, exact adjacent-bin start matching, and complete decision-epoch targets.",
    )

    gate_operating_path = (
        REPO_ROOT
        / "results"
        / "gate_operating_characteristics"
        / "gate_operating_characteristics_summary.csv"
    )
    gate_operating_metadata = _read_json(
        REPO_ROOT
        / "results"
        / "gate_operating_characteristics"
        / "gate_operating_characteristics_metadata.json"
    )
    gate_operating = (
        pd.read_csv(gate_operating_path)
        if gate_operating_path.exists()
        else pd.DataFrame()
    )
    null_or_unsafe = gate_operating[
        gate_operating.get("scenario", pd.Series(dtype=str)).isin(
            ["paired_null", "unsafe_success"]
        )
    ]
    strong_large = gate_operating[
        gate_operating.get("scenario", pd.Series(dtype=str)).eq(
            "strong_positive_control"
        )
        & gate_operating.get("group_count", pd.Series(dtype=float)).eq(5000)
    ]
    gate_operating_pass = bool(
        not gate_operating.empty
        and len(gate_operating) == 16
        and not null_or_unsafe.empty
        and float(null_or_unsafe["admission_rate"].max()) == 0.0
        and len(strong_large) == 1
        and float(strong_large["admission_rate"].iloc[0]) == 1.0
        and gate_operating_metadata.get("measured_efficacy_evidence") is False
        and gate_operating_metadata.get("risk_control_config", {}).get(
            "cvar_grid_points"
        )
        == 1_200_001
    )
    _add(
        rows,
        "artifact_integrity",
        "evidence-gate operating-characteristic audit",
        gate_operating_pass,
        json.dumps(
            {
                "summary_cells": len(gate_operating),
                "maximum_null_or_unsafe_admission_rate": (
                    float(null_or_unsafe["admission_rate"].max())
                    if not null_or_unsafe.empty
                    else None
                ),
                "strong_positive_control_admission_rate_at_5000": (
                    float(strong_large["admission_rate"].iloc[0])
                    if len(strong_large) == 1
                    else None
                ),
                "cvar_grid_points": gate_operating_metadata.get(
                    "risk_control_config", {}
                ).get("cvar_grid_points"),
                "measured_efficacy_evidence": gate_operating_metadata.get(
                    "measured_efficacy_evidence"
                ),
            },
            sort_keys=True,
        ),
        "The gate has no positive control or an unreported false-admission calibration.",
        "Run the predeclared synthetic calibration audit and retain its null, unsafe, moderate, and strong-control cells.",
    )

    policy_summary_path = independent_dir / "policy_summary.csv"
    policy_names = (
        set(pd.read_csv(policy_summary_path)["policy_name"])
        if policy_summary_path.exists()
        else set()
    )
    _add(
        rows,
        "artifact_integrity",
        "online switching-cost decision",
        "switch_aware_operational_selector" in policy_names,
        ", ".join(sorted(policy_names)),
        "Switching cost is evaluated only after selection and cannot influence the operational decision.",
        "Run the switch-aware selector that adds handover cost before path ranking.",
    )

    seed_metadata = _read_json(seed_dir / "seed_matrix_metadata.json")
    seeds = seed_metadata.get("seeds", [])
    _add(
        rows,
        "artifact_integrity",
        "ten regenerated simulator seeds",
        len(seeds) >= 10
        and (seed_dir / "multi_seed_pairwise_deltas.csv").exists(),
        f"completed_seed_count={len(seeds)}",
        "A single simulator seed may make the reported advantage accidental.",
        "Run run_independent_multipath_seed_matrix.py with at least ten seeds.",
    )

    sensitivity_path = sensitivity_dir / "parameter_sensitivity_summary.csv"
    sensitivity_profiles: set[str] = set()
    if sensitivity_path.exists():
        sensitivity_profiles = set(
            pd.read_csv(sensitivity_path)["severity_profile"].dropna()
        )
    _add(
        rows,
        "artifact_integrity",
        "simulator parameter sensitivity",
        {"benign", "nominal", "adverse"}.issubset(sensitivity_profiles),
        ", ".join(sorted(sensitivity_profiles)),
        "The result may depend on one hand-tuned synthetic parameter profile.",
        "Run benign, nominal, and adverse profiles and report all outcomes.",
    )

    resolution_root = REPO_ROOT / "results" / "temporal_resolution_evaluation"
    resolutions = {
        int(path.name.removesuffix("s"))
        for path in resolution_root.glob("*s")
        if path.is_dir()
        and path.name.removesuffix("s").isdigit()
        and (path / "policy_summary.csv").exists()
    }
    _add(
        rows,
        "artifact_integrity",
        "5/10/30/60-second resolution sensitivity",
        {5, 10, 30, 60}.issubset(resolutions),
        f"completed_resolutions={sorted(resolutions)}",
        "The claimed short-horizon result is an artifact of coarse 60-second bins.",
        "Complete the full policy matrix at 5, 10, 30, and 60 seconds.",
    )

    control_path = (
        REPO_ROOT
        / "results"
        / "reviewer_validation"
        / "control_loop_latency_sensitivity.csv"
    )
    control_delays: set[float] = set()
    if control_path.exists():
        control_delays = set(
            pd.read_csv(control_path)["control_loop_latency_ms"].dropna()
        )
    _add(
        rows,
        "artifact_integrity",
        "control-loop delay sensitivity",
        {0.0, 10.0, 50.0, 100.0, 500.0, 1000.0}.issubset(
            control_delays
        ),
        f"completed_delays_ms={sorted(control_delays)}",
        "State collection and dissemination make next-bin decisions stale.",
        "Evaluate collection, inference, and dissemination delay over the configured range.",
    )

    external_metadata = _read_json(
        REPO_ROOT
        / "results"
        / "commect_validation_gated_audit"
        / "validation_metadata.json"
    )
    external_concurrency = external_metadata.get("concurrency_audit", {})
    external_trace = external_metadata.get("trace_metadata", {})
    _add(
        rows,
        "evaluation_evidence",
        "external-source measured heterogeneous-access shadow replay",
        bool(
            external_metadata.get("policy_level_evaluation")
            and external_metadata.get("measured_concurrent_paths")
            and external_metadata.get("independent_of_lens")
            and external_trace.get("is_measured_dataset")
            and external_trace.get("is_independent_of_lens")
            and external_trace.get("concurrent_alternative_paths")
            and external_concurrency.get("supports_shadow_policy_replay")
            and external_concurrency.get("median_concurrent_paths", 0) >= 2
        ),
        json.dumps(
            {
                "dataset_doi": external_metadata.get("dataset_doi"),
                "independent_of_lens": external_metadata.get(
                    "independent_of_lens"
                ),
                "collection_scope": "one continuous COMMECT drive",
                "concurrency": _concurrency_evidence(external_concurrency),
                "valid_claim": external_metadata.get("valid_claim"),
                "invalid_claim": external_metadata.get("invalid_claim"),
            },
            sort_keys=True,
        ),
        "The independent evidence does not contain simultaneous selectable alternatives.",
        "Evaluate a frozen shadow policy on an independent measured trace with simultaneous alternatives and actual timestamp alignment.",
    )

    measured_multihomed = _read_json(
        REPO_ROOT
        / "results"
        / "measured_multihomed_holdout_validation"
        / "measured_validation_metadata.json"
    )
    measured_concurrency = measured_multihomed.get("concurrency_audit", {})
    _add(
        rows,
        "evaluation_evidence",
        "measured concurrent same-campaign Starlink shadow replay",
        bool(
            measured_multihomed.get("policy_level_evaluation")
            and measured_multihomed.get("measured_concurrent_paths")
            and measured_concurrency.get("supports_shadow_policy_replay")
        ),
        json.dumps(
            {
                "independent_of_lens": measured_multihomed.get(
                    "independent_of_lens"
                ),
                "collection_scope": (
                    "two co-located Victoria terminals in the LENS release"
                ),
                "concurrency": _concurrency_evidence(measured_concurrency),
                "valid_claim": measured_multihomed.get("valid_claim"),
                "invalid_claim": measured_multihomed.get("invalid_claim"),
            },
            sort_keys=True,
        ),
        "The LEO concurrency check is supported only by synthetic alternatives.",
        "Evaluate the frozen shadow policy on simultaneous measured LEO alternatives.",
    )

    zero_shot_validation = _read_json(
        REPO_ROOT
        / "results"
        / "zero_shot_transfer_validation"
        / "zero_shot_metadata.json"
    )
    _add(
        rows,
        "evaluation_evidence",
        "zero-shot cross-simulator replay",
        bool(
            zero_shot_validation.get("zero_shot_transfer", False)
            and zero_shot_validation.get("target_rows_used_for_training") == 0
            and zero_shot_validation.get("target_rows_used_for_calibration")
            == 0
        ),
        json.dumps(zero_shot_validation, sort_keys=True),
        "Retraining on every target trace does not test frozen cross-simulator replay.",
        "Freeze all experts, calibrators, gates, and weights on one simulator trace, then replay unchanged on a compatible target simulator trace.",
    )

    hypatia_trace_metadata = _read_json(
        REPO_ROOT
        / "data"
        / "processed"
        / "hypatia_service_paths_10s.metadata.json"
    )
    hypatia_concurrency = zero_shot_validation.get(
        "target_concurrency_audit", {}
    )
    established_simulator = _supports_hypatia_zero_shot_replication(
        hypatia_trace_metadata,
        zero_shot_validation,
    )
    _add(
        rows,
        "evaluation_evidence",
        "established orbital-simulator replication",
        established_simulator,
        json.dumps(
            {
                "trace": hypatia_trace_metadata,
                "concurrency": _concurrency_evidence(
                    hypatia_concurrency
                ),
                "evaluation_scope": (
                    "frozen zero-shot shadow replay; no target fitting"
                ),
            },
            sort_keys=True,
        ),
        "The custom sinusoidal simulator may encode assumptions favorable to the proposed rule.",
        "Replicate on Hypatia/TLE-SGP4 or another established LEO simulator.",
    )

    commercial_status = _read_json(
        REPO_ROOT / "docs" / "commercial_multileo_acquisition_status.json"
    )
    commercial_validation = _read_json(
        REPO_ROOT
        / "results"
        / "commercial_multileo_validation"
        / "validation_metadata.json"
    )
    commercial_trace_gate = commercial_validation.get("trace_metadata", {})
    if not isinstance(commercial_trace_gate, dict):
        commercial_trace_gate = {}
    commercial_split_audit = commercial_validation.get("split_audit", {})
    if not isinstance(commercial_split_audit, dict):
        commercial_split_audit = {}
    commercial_campaign_audit = commercial_validation.get(
        "campaign_independence_audit",
        {},
    )
    if not isinstance(commercial_campaign_audit, dict):
        commercial_campaign_audit = {}
    commercial_group_column = commercial_campaign_audit.get(
        "risk_control_group_column"
    )
    commercial_gate_inference_safe = bool(
        commercial_validation.get(
            "gate_inference_fail_closed_without_audited_campaign_ids"
        )
        and commercial_validation.get("unaudited_campaign_reactive_guard_pass")
        and (
            (
                commercial_group_column is None
                and commercial_validation.get("gate_independence_group_count")
                in (0, 1)
            )
            or (
                commercial_group_column == "campaign_id"
                and commercial_campaign_audit.get(
                    "forecast_campaign_ids_audited"
                )
                and commercial_trace_gate.get(
                    "independent_campaign_grouping_pass"
                )
            )
        )
    )
    commercial_complete = bool(
        commercial_status.get("raw_data_received")
        and commercial_status.get("claim_gate_status") == "open"
        and commercial_validation.get("same_controller_selectable_path_evidence")
        and commercial_validation.get("policy_level_evaluation")
        and commercial_trace_gate.get("topology_claim_gate_version", 0) >= 1
        and commercial_trace_gate.get("spatial_colocation_pass")
        and commercial_trace_gate.get("shared_controller_provenance_pass")
        and commercial_trace_gate.get("same_controller_selectable_path_evidence")
        and commercial_split_audit.get("global_wall_clock_partitioning")
        and commercial_split_audit.get("pairwise_epoch_disjoint")
        and commercial_split_audit.get("strict_chronological_order")
        and commercial_split_audit.get("one_step_target_boundary_closed")
        and commercial_split_audit.get("multi_bin_target_boundaries_closed")
        and commercial_gate_inference_safe
        and commercial_validation.get(
            "closes_independent_longitudinal_multileo_limitation"
        )
    )
    _add(
        rows,
        "deployment_evidence",
        "authorized independent commercial multi-LEO trace",
        commercial_complete,
        json.dumps(
            {
                "acquisition_status": commercial_status,
                "validation_metadata": commercial_validation,
            },
            sort_keys=True,
        ),
        "The artifact has not evaluated an authorized synchronized Starlink--OneWeb trace.",
        "Receive and checksum the authorized raw trace, then pass the temporal, complete-GPS co-location, shared-controller, duration, and policy-replay gates before enabling the claim.",
    )

    deployment_metadata = _read_json(
        REPO_ROOT
        / "results"
        / "closed_loop_deployment"
        / "deployment_metadata.json"
    )
    closed_loop_complete = bool(
        deployment_metadata.get("closed_loop_field_trial")
        and deployment_metadata.get("policy_installed_before_outcomes")
        and deployment_metadata.get("network_actions_executed")
    )
    closed_loop_protocol = (
        REPO_ROOT / "docs" / "closed_loop_field_validation_protocol.md"
    )
    _add(
        rows,
        "deployment_evidence",
        "closed-loop field deployment",
        closed_loop_complete,
        json.dumps(
            {
                "deployment_metadata": deployment_metadata,
                "prospective_protocol": str(
                    closed_loop_protocol.relative_to(REPO_ROOT)
                ),
                "protocol_present": closed_loop_protocol.exists(),
                "protocol_is_deployment_evidence": False,
            },
            sort_keys=True,
        ),
        "All current policy results are replay or simulation rather than installed network actions.",
        "Run and document a prospective closed-loop field trial; do not describe shadow replay as deployment evidence.",
    )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "reviewer_readiness.csv", index=False)
    category_summary = {
        category: {
            "pass_count": int((frame["status"] == "PASS").sum()),
            "pending_count": int((frame["status"] == "PENDING").sum()),
            "complete": bool((frame["status"] == "PASS").all()),
        }
        for category, frame in table.groupby("category", sort=True)
    }
    summary = {
        "pass_count": int((table["status"] == "PASS").sum()),
        "pending_count": int((table["status"] == "PENDING").sum()),
        "all_review_items_complete": bool((table["status"] == "PASS").all()),
        "artifact_checks_complete": bool(
            (table.loc[table["category"].eq("artifact_integrity"), "status"] == "PASS").all()
        ),
        "external_or_deployment_evidence_complete": bool(
            (
                table.loc[
                    table["category"].isin(
                        ["evaluation_evidence", "deployment_evidence"]
                    ),
                    "status",
                ]
                == "PASS"
            ).all()
        ),
        "allow_pending_acknowledged": bool(args.allow_pending),
        "by_category": category_summary,
    }
    (output_dir / "reviewer_readiness.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    markdown_table = table.astype(str)
    headers = markdown_table.columns.tolist()
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(row) + " |"
        for row in markdown_table.itertuples(index=False, name=None)
    )
    markdown = [
        "# Reviewer Readiness Audit",
        "",
        f"Passed: {summary['pass_count']}; pending: {summary['pending_count']}.",
        "",
        (
            "Artifact-integrity checks verify generated files and protocols. "
            "Evaluation-evidence and deployment-evidence rows are separate claim "
            "boundaries; a replay pass is not a closed-loop deployment result."
        ),
        "",
        *markdown_rows,
        "",
    ]
    (output_dir / "reviewer_readiness.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(json.dumps(summary))
    if summary["pending_count"] and not args.allow_pending:
        print(
            "Pending evidence remains; outputs were written. "
            "Use --allow-pending only to acknowledge those explicit limitations."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
