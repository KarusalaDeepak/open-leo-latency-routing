#!/usr/bin/env python3
"""Build canonical Transactions tables and figures from concurrent-path runs."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
from pathlib import Path
from pathlib import PurePosixPath
import sys
import tempfile

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from open_leo_latency_routing.evaluation.confidence_intervals import (
    build_bootstrap_policy_intervals,
)
from open_leo_latency_routing.evaluation.significance import (
    build_paired_policy_significance,
)
from open_leo_latency_routing.evaluation.decision_opportunity import (
    build_candidate_opportunity_audit,
    build_opportunity_conditioned_results,
    build_pairwise_success_gap_bounds,
    build_policy_choice_agreement,
)
from open_leo_latency_routing.evaluation.delayed_execution import (
    replay_delayed_execution,
)
from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar
from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.optimization.policies import (
    add_qos_shielded_scores,
    evaluate_decision_policies,
    summarize_switch_transitions,
)
from open_leo_latency_routing.visualization import (
    IEEE_TEXT_WIDTH_IN,
    OPAQUE_GRID_COLOR,
    configure_ieee_figure_style,
    save_png_pdf_pair,
)


configure_ieee_figure_style(base_font_size=9.0)


RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "Pillow",
    "xgboost",
)
TEST_DISTRIBUTIONS = ("pytest",)


POLICIES = [
    "reactive_greedy",
    "age_aware_reactive_selector",
    "robust_persistence_selector",
    "predictive_greedy",
    "predictive_graph_greedy",
    "predictive_simple_fusion_greedy",
    "calibrated_fusion_selector",
    "disagreement_only_selector",
    "ensemble_uncertainty_selector",
    "cvar_proxy_selector",
    "conformal_uncertainty_selector",
    "calibrated_operational_selector",
    "switch_aware_operational_selector",
    "qos_context_fallback_selector",
    "qos_ensemble_fallback_selector",
    "reactive_hysteresis_selector",
    "qos_hysteresis_selector",
    "qos_shielded_operational_selector",
    "validation_gated_qos_selector",
    "qos_filter_then_context_selector",
    "qos_filter_then_ensemble_selector",
]

DISPLAY = {
    "reactive_greedy": "Reactive",
    "age_aware_reactive_selector": "Age-aware reactive",
    "robust_persistence_selector": "Robust persistence",
    "predictive_greedy": "Temporal",
    "predictive_graph_greedy": "Context",
    "ensemble_uncertainty_selector": "Ensemble",
    "cvar_proxy_selector": "CVaR proxy",
    "conformal_uncertainty_selector": "Conformal",
    "calibrated_operational_selector": "Calibrated",
    "switch_aware_operational_selector": "Predictive + hysteresis",
    "qos_context_fallback_selector": "QoS + context",
    "qos_ensemble_fallback_selector": "QoS + ensemble",
    "reactive_hysteresis_selector": "Reactive + hysteresis",
    "qos_hysteresis_selector": "QoS shield + hysteresis",
    "qos_shielded_operational_selector": "Predictive shield",
    "validation_gated_qos_selector": "Evidence-gated policy",
    "qos_filter_then_context_selector": "QoS filter + context",
    "qos_filter_then_ensemble_selector": "QoS filter + ensemble",
}


_CANDIDATE_CACHE: dict[Path, pd.DataFrame] = {}


SUPPORTING_EVIDENCE = {
    "component_ablation.csv": "results/reviewer_validation/component_ablation.csv",
    "disagreement_diagnostics.csv": "results/reviewer_validation/disagreement_diagnostics.csv",
    "matched_model_pair_audit.csv": "results/reviewer_validation/matched_model_pair_audit.csv",
    "predictor_combination_audit.csv": "results/reviewer_validation/predictor_combination_audit.csv",
    "stale_state_sensitivity.csv": "results/reviewer_validation/stale_state_sensitivity.csv",
    "temporal_resolution_policy_summary.csv": "results/temporal_resolution_evaluation/temporal_resolution_policy_summary.csv",
    "propagation_bounds.csv": "results/physical_feasibility/propagation_bounds.csv",
    "control_horizon_bounds.csv": "results/physical_feasibility/control_horizon_bounds.csv",
    "commect_rolling_origin_summary.csv": "results/commect_validation_gated_rolling/rolling_policy_summary.csv",
    "commect_rolling_fold_manifest.csv": "results/commect_validation_gated_rolling/rolling_fold_manifest.csv",
    "commect_rolling_protocol_metadata.json": "results/commect_validation_gated_rolling/rolling_validation_metadata.json",
    "commect_rolling_opportunity_audit.csv": "results/commect_validation_gated_rolling/rolling_opportunity_audit.csv",
    "commect_rolling_opportunity_conditioned_results.csv": "results/commect_validation_gated_rolling/rolling_opportunity_conditioned_results.csv",
    "commect_rolling_policy_significance.csv": "results/commect_validation_gated_rolling/rolling_policy_significance.csv",
    "commect_rolling_success_gap_bounds.csv": "results/commect_validation_gated_rolling/rolling_success_gap_bounds.csv",
    "commect_rolling_delayed_state_replay.csv": "results/commect_validation_gated_rolling/rolling_delayed_state_replay.csv",
    "commect_rolling_gate_selection_evidence.csv": "results/commect_validation_gated_rolling/rolling_gate_selection_evidence.csv",
    "commect_fixed_gate_selection_evidence.csv": "results/commect_validation_gated_audit/gate_selection_evidence.csv",
    "commect_predictor_information_audit.csv": "results/commect_validation_gated_audit/predictor_information_audit.csv",
    "commect_predictor_information_audit_metadata.json": "results/commect_validation_gated_audit/predictor_information_audit_metadata.json",
    "victoria_gate_selection_evidence.csv": "results/measured_multihomed_holdout_validation/gate_selection_evidence.csv",
    "commect_timestamp_skew_sensitivity.csv": "results/commect_timestamp_sensitivity/timestamp_skew_policy_sensitivity.csv",
    "commect_timestamp_skew_sensitivity_metadata.json": "results/commect_timestamp_sensitivity/timestamp_skew_sensitivity_metadata.json",
    "commect_rolling_timestamp_skew_sensitivity.csv": "results/commect_rolling_timestamp_sensitivity/rolling_timestamp_skew_policy_sensitivity.csv",
    "commect_rolling_timestamp_skew_metadata.json": "results/commect_rolling_timestamp_sensitivity/rolling_timestamp_skew_metadata.json",
    "commect_objective_specific_threshold_policy_results.csv": "results/commect_threshold_gate_sensitivity/threshold_policy_results.csv",
    "commect_objective_specific_threshold_primary_results.csv": "results/commect_threshold_gate_sensitivity/threshold_primary_policy_results.csv",
    "commect_objective_specific_threshold_gate_evidence.csv": "results/commect_threshold_gate_sensitivity/threshold_gate_evidence.csv",
    "commect_objective_specific_threshold_metadata.json": "results/commect_threshold_gate_sensitivity/manifest.json",
    "prospective_gate_design_sensitivity.csv": "results/gate_design_sensitivity/gate_design_sensitivity.csv",
    "prospective_gate_design_canonical_reference.csv": "results/gate_design_sensitivity/gate_design_sensitivity_canonical_reference.csv",
    "prospective_gate_design_sensitivity_metadata.json": "results/gate_design_sensitivity/gate_design_sensitivity_metadata.json",
    "short_30_seed_policy_summary.csv": "results/transactions_seed_matrix_30_short/multi_seed_policy_summary.csv",
    "short_30_seed_pairwise_deltas.csv": "results/transactions_seed_matrix_30_short/multi_seed_pairwise_deltas.csv",
    "short_30_seed_metadata.json": "results/transactions_seed_matrix_30_short/seed_matrix_metadata.json",
    "gate_operating_characteristics_detailed.csv": "results/gate_operating_characteristics/gate_operating_characteristics_detailed.csv",
    "gate_operating_characteristics_summary.csv": "results/gate_operating_characteristics/gate_operating_characteristics_summary.csv",
    "gate_operating_characteristics_metadata.json": "results/gate_operating_characteristics/gate_operating_characteristics_metadata.json",
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_sources() -> list[dict[str, object]]:
    return [
        {
            "dataset": "COMMECT",
            "scenario": "measured_multi_access",
            "source_type": "external-source measured",
            "summary": REPO_ROOT / "results/commect_validation_gated_audit/policy_summary.csv",
            "decisions": REPO_ROOT / "results/commect_validation_gated_audit/policy_decisions.csv",
            "candidates": REPO_ROOT / "results/commect_validation_gated_audit/candidate_predictions.csv",
            "metadata": REPO_ROOT / "results/commect_validation_gated_audit/validation_metadata.json",
        },
        {
            "dataset": "LENS Victoria holdout",
            "scenario": "measured_multihomed",
            "source_type": "measured LEO terminals",
            "summary": REPO_ROOT / "results/measured_multihomed_holdout_validation/measured_policy_summary.csv",
            "decisions": REPO_ROOT / "results/measured_multihomed_holdout_validation/measured_policy_decisions.csv",
            "candidates": REPO_ROOT / "results/measured_multihomed_holdout_validation/measured_candidate_predictions.csv",
            "metadata": REPO_ROOT / "results/measured_multihomed_holdout_validation/measured_validation_metadata.json",
        },
        {
            "dataset": "Physics-informed orbital trace",
            "scenario": "session_holdout",
            "source_type": "concurrent simulator",
            "summary": REPO_ROOT / "results/transactions_orbital_validation/policy_summary.csv",
            "decisions": REPO_ROOT / "results/transactions_orbital_validation/policy_decisions.csv",
            "candidates": REPO_ROOT / "results/transactions_orbital_validation/candidate_predictions.csv",
            "metadata": REPO_ROOT / "results/transactions_orbital_validation/independent_validation_metadata.json",
        },
        {
            "dataset": "Physics-informed orbital trace",
            "scenario": "operational_moderate",
            "source_type": "concurrent simulator stress",
            "summary": REPO_ROOT / "results/transactions_orbital_validation/policy_summary.csv",
            "decisions": REPO_ROOT / "results/transactions_orbital_validation/policy_decisions.csv",
            "candidates": REPO_ROOT / "results/transactions_orbital_validation/candidate_predictions.csv",
            "metadata": REPO_ROOT / "results/transactions_orbital_validation/independent_validation_metadata.json",
        },
        {
            "dataset": "Physics-informed orbital trace",
            "scenario": "operational_severe",
            "source_type": "concurrent simulator stress",
            "summary": REPO_ROOT / "results/transactions_orbital_validation/policy_summary.csv",
            "decisions": REPO_ROOT / "results/transactions_orbital_validation/policy_decisions.csv",
            "candidates": REPO_ROOT / "results/transactions_orbital_validation/candidate_predictions.csv",
            "metadata": REPO_ROOT / "results/transactions_orbital_validation/independent_validation_metadata.json",
        },
    ]


def _direct_qos_fallback_comparators(
    source: dict[str, object],
    base_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed context and ensemble fallbacks behind the same shield.

    These comparators isolate the value of clean-validation fallback selection
    from the lexicographic shield. They are generated from frozen candidate
    predictions and never choose a fallback using evaluation outcomes.
    """

    candidate_path = Path(source["candidates"])
    if candidate_path not in _CANDIDATE_CACHE:
        _CANDIDATE_CACHE[candidate_path] = pd.read_csv(
            candidate_path,
            low_memory=False,
        )
    candidates = _CANDIDATE_CACHE[candidate_path]
    scenario = str(source["scenario"])
    if "scenario_name" in candidates.columns:
        candidates = candidates[candidates["scenario_name"].eq(scenario)].copy()
    else:
        candidates = candidates.copy()

    candidates = add_qos_shielded_scores(
        candidates,
        fallback_column="pred_graph",
        latency_budget_ms=60.0,
        output_column="pred_qos_context_fallback",
    )
    candidates = add_qos_shielded_scores(
        candidates,
        fallback_column="pred_ensemble_uncertainty",
        latency_budget_ms=60.0,
        output_column="pred_qos_ensemble_fallback",
    )
    shield = base_summary[
        base_summary["policy_name"].eq("qos_shielded_operational_selector")
    ]
    if not shield.empty and "mean_model_scoring_time_us" in shield:
        candidates.attrs["model_scoring_time_us_per_decision"] = float(
            shield.iloc[0]["mean_model_scoring_time_us"]
        )
    summary, decisions = evaluate_decision_policies(
        candidates,
        latency_budget_ms=60.0,
        policy_columns={
            "qos_context_fallback_selector": "pred_qos_context_fallback",
            "qos_ensemble_fallback_selector": "pred_qos_ensemble_fallback",
            "reactive_hysteresis_selector": "latency_mean_ms",
            "qos_hysteresis_selector": "pred_qos_shielded_operational",
        },
        online_switch_penalties_ms={
            "reactive_hysteresis_selector": 10.0,
            "qos_hysteresis_selector": 10.0,
        },
    )
    if "scenario_name" in candidates.columns:
        summary["scenario_name"] = scenario
        decisions["scenario_name"] = scenario
    return summary, decisions


def _source_frame(source: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(Path(source["summary"]))
    decisions = pd.read_csv(Path(source["decisions"]), low_memory=False)
    scenario = str(source["scenario"])
    if "scenario_name" in summary.columns:
        summary = summary[summary["scenario_name"].eq(scenario)]
    if "scenario_name" in decisions.columns:
        decisions = decisions[decisions["scenario_name"].eq(scenario)]
    summary = summary[summary["policy_name"].isin(POLICIES)].copy()
    decisions = decisions[decisions["policy_name"].isin(POLICIES)].copy()
    fallback_summary, fallback_decisions = _direct_qos_fallback_comparators(
        source,
        summary,
    )
    summary = pd.concat([summary, fallback_summary], ignore_index=True)
    decisions = pd.concat([decisions, fallback_decisions], ignore_index=True)
    for frame in (summary, decisions):
        frame["dataset"] = source["dataset"]
        frame["evaluation_case"] = scenario
        frame["source_type"] = source["source_type"]
    return summary, decisions


def _dataset_audit(sources: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    seen: set[Path] = set()
    for source in sources:
        metadata_path = Path(source["metadata"])
        if metadata_path in seen:
            continue
        seen.add(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trace_metadata = metadata.get("trace_metadata", {})
        audit = metadata.get("concurrency_audit") or trace_metadata.get(
            "concurrency_audit", {}
        )
        declared_concurrent = bool(
            metadata.get("concurrent_alternative_paths")
            or trace_metadata.get("concurrent_alternative_paths")
        )
        concurrent_epoch_count = int(audit.get("concurrent_epoch_count", 0))
        max_concurrent_paths = int(audit.get("max_concurrent_paths", 0))
        if declared_concurrent and not concurrent_epoch_count:
            # Simulator metadata declares simultaneous candidate paths even
            # when it predates the row-level concurrency-audit schema.
            decision_count = 0
            path_count = int(trace_metadata.get("candidate_paths_per_decision", 0))
            matching_source = next(
                item for item in sources if Path(item["metadata"]) == metadata_path
            )
            source_decisions = pd.read_csv(Path(matching_source["decisions"]))
            decision_count = int(source_decisions["session_bin_index"].nunique())
            concurrent_epoch_count = decision_count
            max_concurrent_paths = path_count
        rows.append(
            {
                "dataset": source["dataset"],
                "source_type": source["source_type"],
                "measured": bool(
                    metadata.get("measured_concurrent_paths")
                    or trace_metadata.get("is_measured_dataset")
                ),
                "concurrent_epoch_count": concurrent_epoch_count,
                "max_concurrent_paths": max_concurrent_paths,
                "concurrent_row_fraction": float(
                    audit.get("concurrent_row_fraction", 1.0 if declared_concurrent else 0.0)
                ),
                "has_temporally_concurrent_candidates": bool(
                    audit.get(
                        "has_temporally_concurrent_candidates",
                        declared_concurrent,
                    )
                ),
                "supports_candidate_outcome_shadow_replay": bool(
                    audit.get(
                        "supports_candidate_outcome_shadow_replay",
                        audit.get("supports_shadow_policy_replay", declared_concurrent),
                    )
                ),
                "supports_literal_single_controller_steering": bool(
                    audit.get("supports_literal_single_controller_steering", False)
                ),
                "supports_closed_loop_deployment_evidence": bool(
                    audit.get("supports_closed_loop_deployment_evidence", False)
                ),
                "controller_topology_scope": str(
                    audit.get(
                        "controller_topology_scope",
                        "not established by timestamp concurrency alone",
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _missing_outcome_envelopes(
    fixed_summary: pd.DataFrame,
    fixed_split_audit: dict[str, object],
    rolling_fold_manifest: pd.DataFrame,
    rolling_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Instantiate conservative aggregate-success bounds over scheduled tests."""

    def policy_counts(frame: pd.DataFrame, *, protocol: str) -> tuple[int, int]:
        selected = frame[frame["policy_name"].eq("reactive_greedy")]
        if len(selected) != 1:
            raise ValueError(
                f"{protocol} reactive summary must contain exactly one row"
            )
        row = selected.iloc[0]
        retained = int(row["decision_count"])
        successes_float = retained * float(row["success_rate_under_60ms"])
        successes = int(round(successes_float))
        if not np.isclose(successes_float, successes, atol=1e-8):
            raise ValueError(
                f"{protocol} success rate does not map to an integer count"
            )
        return retained, successes

    fixed_partitions = fixed_split_audit.get("partitions")
    if not isinstance(fixed_partitions, dict) or not isinstance(
        fixed_partitions.get("test"), dict
    ):
        raise ValueError("fixed split audit lacks the scheduled test partition")
    fixed_scheduled = int(
        fixed_partitions["test"]["scheduled_decision_epoch_count"]
    )
    fixed_retained, fixed_successes = policy_counts(
        fixed_summary,
        protocol="fixed",
    )

    required_rolling = {"scheduled_test_decision_epochs"}
    if not required_rolling.issubset(rolling_fold_manifest.columns):
        raise ValueError("rolling manifest lacks scheduled test counts")
    rolling_scheduled = int(
        rolling_fold_manifest["scheduled_test_decision_epochs"].sum()
    )
    rolling_retained, rolling_successes = policy_counts(
        rolling_summary,
        protocol="rolling",
    )

    rows: list[dict[str, object]] = []
    for protocol, scheduled, retained, successes in (
        ("fixed", fixed_scheduled, fixed_retained, fixed_successes),
        ("rolling", rolling_scheduled, rolling_retained, rolling_successes),
    ):
        unknown = scheduled - retained
        if scheduled <= 0 or retained < 0 or successes < 0 or unknown < 0:
            raise ValueError(f"invalid {protocol} missing-outcome counts")
        rows.append(
            {
                "protocol": protocol,
                "policy_name": "reactive_greedy",
                "scheduled_test_decision_epochs": scheduled,
                "retained_evaluable_test_decision_epochs": retained,
                "observed_success_count": successes,
                "unevaluable_scheduled_test_decision_epochs": unknown,
                "worst_case_success_lower_bound": successes / scheduled,
                "best_case_success_upper_bound": (successes + unknown) / scheduled,
                "unknown_slot_definition": (
                    "scheduled test slot lacking an evaluable complete adjacent-bin "
                    "candidate vector after raw-grid missingness, target completeness, "
                    "and protocol boundary closure"
                ),
                "interpretation": (
                    "aggregate-success complete-case envelope; not an outage estimate "
                    "and not a bound for opportunity capture or CVaR"
                ),
            }
        )
    return pd.DataFrame(rows)


def _timestamp_alignment_audit(sources: list[dict[str, object]]) -> pd.DataFrame:
    """Export source-level timing evidence behind concurrency claims."""

    rows = []
    seen: set[Path] = set()
    for source in sources:
        metadata_path = Path(source["metadata"])
        if metadata_path in seen:
            continue
        seen.add(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trace_metadata = metadata.get("trace_metadata", metadata)
        alignment = trace_metadata.get("timestamp_alignment_audit", {})
        rows.append(
            {
                "dataset": source["dataset"],
                "source_type": source["source_type"],
                "alignment_rule": trace_metadata.get("alignment", "exact shared bin key"),
                "timezone_note": trace_metadata.get("source_timezone_note", "not applicable or inherited from source"),
                "complete_concurrent_bins": alignment.get("complete_concurrent_bins", np.nan),
                "median_inter_path_median_skew_ms": alignment.get("median_inter_path_median_skew_ms", np.nan),
                "p95_inter_path_median_skew_ms": alignment.get("p95_inter_path_median_skew_ms", np.nan),
                "maximum_inter_path_median_skew_ms": alignment.get("maximum_inter_path_median_skew_ms", np.nan),
                "raw_probe_skew_available": bool(alignment),
            }
        )
    return pd.DataFrame(rows)


def _exact_horizon_audit(sources: list[dict[str, object]]) -> pd.DataFrame:
    """Export exact wall-clock target retention and gap exclusions."""

    metadata_entries: list[tuple[str, str, Path]] = []
    seen: set[Path] = set()
    for source in sources:
        metadata_path = Path(source["metadata"])
        if metadata_path in seen:
            continue
        seen.add(metadata_path)
        metadata_entries.append(
            (str(source["dataset"]), "fixed", metadata_path)
        )
    metadata_entries.append(
        (
            "COMMECT",
            "rolling_origin",
            REPO_ROOT
            / "results/commect_validation_gated_rolling"
            / "rolling_validation_metadata.json",
        )
    )

    rows: list[dict[str, object]] = []
    for dataset, evaluation_protocol, metadata_path in metadata_entries:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        audit = metadata.get("exact_horizon_audit", {})
        rows.append(
            {
                "dataset": dataset,
                "evaluation_protocol": evaluation_protocol,
                "endpoint_semantics": audit.get("endpoint_semantics"),
                "scheduled_history_semantics": audit.get(
                    "scheduled_history_semantics"
                ),
                "decision_cadence_seconds": audit.get(
                    "decision_cadence_seconds"
                ),
                "one_step_horizon_bins": audit.get("one_step_horizon_bins"),
                "one_step_expected_horizon_seconds": audit.get(
                    "one_step_expected_horizon_seconds"
                ),
                "input_row_count": audit.get("input_row_count"),
                "scheduled_grid_expected_row_count": audit.get(
                    "scheduled_grid_expected_row_count"
                ),
                "scheduled_grid_observed_row_count": audit.get(
                    "scheduled_grid_observed_row_count"
                ),
                "scheduled_grid_missing_row_count": audit.get(
                    "scheduled_grid_missing_row_count"
                ),
                "scheduled_grid_off_phase_row_count": audit.get(
                    "scheduled_grid_off_phase_row_count"
                ),
                "history_gap_row_count": audit.get("history_gap_row_count"),
                "full_lag_history_row_count": audit.get(
                    "full_lag_history_row_count"
                ),
                "positional_target_row_count": audit.get(
                    "positional_target_row_count"
                ),
                "excluded_nonexact_gap_row_count": audit.get(
                    "excluded_nonexact_gap_row_count"
                ),
                "require_complete_decision_epochs": audit.get(
                    "require_complete_decision_epochs"
                ),
                "excluded_incomplete_decision_epoch_count": audit.get(
                    "excluded_incomplete_decision_epoch_count"
                ),
                "asymmetric_missing_candidate_epoch_count": audit.get(
                    "asymmetric_missing_candidate_epoch_count"
                ),
                "retained_exact_target_row_count": audit.get(
                    "retained_exact_target_row_count"
                ),
                "retained_full_lag_history_row_count": audit.get(
                    "retained_full_lag_history_row_count"
                ),
                "retained_decision_epoch_count": audit.get(
                    "retained_decision_epoch_count"
                ),
            }
        )
    return pd.DataFrame(rows)


def _validation_gate_audit(sources: list[dict[str, object]]) -> pd.DataFrame:
    """Export the fallback frozen before each fixed or rolling test interval."""

    rows: list[dict[str, object]] = []
    allowed = {"reactive", "graph", "ensemble"}
    for source in sources:
        candidate_path = Path(source["candidates"])
        if candidate_path not in _CANDIDATE_CACHE:
            _CANDIDATE_CACHE[candidate_path] = pd.read_csv(
                candidate_path,
                low_memory=False,
            )
        candidates = _CANDIDATE_CACHE[candidate_path]
        scenario = str(source["scenario"])
        if "scenario_name" in candidates.columns:
            candidates = candidates[candidates["scenario_name"].eq(scenario)].copy()
        gated = sorted(
            set(candidates["validation_gated_fallback_policy"].dropna().astype(str))
        )
        predictive = sorted(
            set(candidates["risk_fallback_policy"].dropna().astype(str))
        )
        if len(gated) != 1 or gated[0] not in allowed:
            raise AssertionError(
                f"{source['dataset']} {scenario} has invalid frozen gate choices: {gated}"
            )
        if len(predictive) != 1 or predictive[0] not in {"graph", "ensemble"}:
            raise AssertionError(
                f"{source['dataset']} {scenario} has invalid predictive fallback: {predictive}"
            )
        rows.append(
            {
                "dataset": source["dataset"],
                "evaluation_case": scenario,
                "protocol": "fixed chronological holdout",
                "fold": "fixed",
                "predictive_only_fallback": predictive[0],
                "validation_gated_fallback": gated[0],
                "admissible_fallbacks": "reactive|graph|ensemble",
                "selection_objective": (
                    "minimum opportunity-bearing groups, simultaneous aggregate "
                    "and opportunity-conditioned QoS non-inferiority, and "
                    "practical bounded-CVaR gain"
                ),
                "selection_reason": str(
                    candidates.get("gate_selection_reason", pd.Series(["legacy"])).iloc[0]
                ),
                "selection_opportunity_count": int(
                    candidates.get("gate_opportunity_count", pd.Series([0])).iloc[0]
                ),
                "effective_opportunity_count": float(
                    candidates.get(
                        "gate_effective_opportunity_count", pd.Series([0.0])
                    ).iloc[0]
                ),
                "selected_success_lcb": float(
                    candidates.get("gate_selected_success_lcb", pd.Series([0.0])).iloc[0]
                ),
                "selected_opportunity_success_lcb": float(
                    candidates.get(
                        "gate_selected_opportunity_success_lcb",
                        pd.Series([0.0]),
                    ).iloc[0]
                ),
                "noninferiority_margin": float(
                    candidates.get("gate_noninferiority_margin", pd.Series([0.0])).iloc[0]
                ),
                "opportunity_noninferiority_margin": float(
                    candidates.get(
                        "gate_opportunity_noninferiority_margin",
                        pd.Series([0.0]),
                    ).iloc[0]
                ),
                "aggregate_success_noninferior": bool(
                    candidates.get(
                        "gate_selected_aggregate_success_noninferior",
                        pd.Series([0]),
                    ).iloc[0]
                ),
                "opportunity_success_noninferior": bool(
                    candidates.get(
                        "gate_selected_opportunity_success_noninferior",
                        pd.Series([0]),
                    ).iloc[0]
                ),
                "frozen_before_test": True,
                "test_outcomes_used": False,
                "test_decision_epochs": int(
                    candidates["session_bin_index"].nunique()
                ),
            }
        )

    rolling_path = (
        REPO_ROOT
        / "results/commect_validation_gated_rolling/rolling_candidate_predictions.csv"
    )
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path, low_memory=False)
        for fold, frame in rolling.groupby("rolling_fold", sort=True):
            gated = sorted(
                set(frame["validation_gated_fallback_policy"].dropna().astype(str))
            )
            predictive = sorted(
                set(frame["risk_fallback_policy"].dropna().astype(str))
            )
            if len(gated) != 1 or gated[0] not in allowed:
                raise AssertionError(f"COMMECT rolling fold {fold} gate choices: {gated}")
            if len(predictive) != 1 or predictive[0] not in {"graph", "ensemble"}:
                raise AssertionError(
                    f"COMMECT rolling fold {fold} predictive fallback: {predictive}"
                )
            rows.append(
                {
                    "dataset": "COMMECT",
                    "evaluation_case": "rolling_origin",
                    "protocol": (
                        "primary measured prequential expanding-window; "
                        "within-fold train/calibration/selection/test"
                    ),
                    "cross_fold_history_reuse": (
                        "past scored test blocks may enter later-fold history"
                    ),
                    "fold": int(fold),
                    "predictive_only_fallback": predictive[0],
                    "validation_gated_fallback": gated[0],
                    "admissible_fallbacks": "reactive|graph|ensemble",
                    "selection_objective": (
                        "minimum opportunity-bearing groups, simultaneous aggregate "
                        "and opportunity-conditioned QoS non-inferiority, and "
                        "practical bounded-CVaR gain"
                    ),
                    "selection_reason": str(
                        frame.get("gate_selection_reason", pd.Series(["legacy"])).iloc[0]
                    ),
                    "selection_opportunity_count": int(
                        frame.get("gate_opportunity_count", pd.Series([0])).iloc[0]
                    ),
                    "effective_opportunity_count": float(
                        frame.get(
                            "gate_effective_opportunity_count", pd.Series([0.0])
                        ).iloc[0]
                    ),
                    "selected_success_lcb": float(
                        frame.get("gate_selected_success_lcb", pd.Series([0.0])).iloc[0]
                    ),
                    "selected_opportunity_success_lcb": float(
                        frame.get(
                            "gate_selected_opportunity_success_lcb",
                            pd.Series([0.0]),
                        ).iloc[0]
                    ),
                    "noninferiority_margin": float(
                        frame.get("gate_noninferiority_margin", pd.Series([0.0])).iloc[0]
                    ),
                    "opportunity_noninferiority_margin": float(
                        frame.get(
                            "gate_opportunity_noninferiority_margin",
                            pd.Series([0.0]),
                        ).iloc[0]
                    ),
                    "aggregate_success_noninferior": bool(
                        frame.get(
                            "gate_selected_aggregate_success_noninferior",
                            pd.Series([0]),
                        ).iloc[0]
                    ),
                    "opportunity_success_noninferior": bool(
                        frame.get(
                            "gate_selected_opportunity_success_noninferior",
                            pd.Series([0]),
                        ).iloc[0]
                    ),
                    "frozen_before_test": True,
                    "test_outcomes_used": False,
                    "test_decision_epochs": int(frame["session_bin_index"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _decision_opportunity_evidence(
    sources: list[dict[str, object]],
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build source-specific actionability and policy-separation evidence."""

    audit_frames = []
    label_frames = []
    conditioned_frames = []
    agreement_frames = []
    bound_frames = []
    for source in sources:
        candidates = pd.read_csv(Path(source["candidates"]), low_memory=False)
        scenario = str(source["scenario"])
        if "scenario_name" in candidates.columns:
            candidates = candidates[candidates["scenario_name"].eq(scenario)].copy()
        source_decisions = decisions[
            decisions["dataset"].eq(source["dataset"])
            & decisions["evaluation_case"].eq(scenario)
        ].copy()
        audit, labels = build_candidate_opportunity_audit(candidates)
        conditioned = build_opportunity_conditioned_results(source_decisions, labels)
        agreement = build_policy_choice_agreement(source_decisions)
        bounds = build_pairwise_success_gap_bounds(source_decisions, labels)
        for frame in (audit, labels, conditioned, agreement, bounds):
            frame["dataset"] = source["dataset"]
            frame["evaluation_case"] = scenario
            frame["source_type"] = source["source_type"]
        audit_frames.append(audit)
        label_frames.append(labels)
        conditioned_frames.append(conditioned)
        agreement_frames.append(agreement)
        bound_frames.append(bounds)
    return (
        pd.concat(audit_frames, ignore_index=True),
        pd.concat(label_frames, ignore_index=True),
        pd.concat(conditioned_frames, ignore_index=True),
        pd.concat(agreement_frames, ignore_index=True),
        pd.concat(bound_frames, ignore_index=True),
    )


def _delayed_replay_evidence(
    sources: list[dict[str, object]],
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Replay frozen choices against later trace state and availability."""

    policies = {
        "reactive_greedy",
        "predictive_graph_greedy",
        "ensemble_uncertainty_selector",
        "qos_shielded_operational_selector",
        "validation_gated_qos_selector",
        "validation_gated_qos_selector",
    }
    frames = []
    for source in sources:
        candidates = pd.read_csv(Path(source["candidates"]), low_memory=False)
        scenario = str(source["scenario"])
        if "scenario_name" in candidates.columns:
            candidates = candidates[candidates["scenario_name"].eq(scenario)].copy()
        source_decisions = decisions[
            decisions["dataset"].eq(source["dataset"])
            & decisions["evaluation_case"].eq(scenario)
            & decisions["policy_name"].isin(policies)
        ].copy()
        summary, _ = replay_delayed_execution(
            candidates,
            source_decisions,
            latency_budget_ms=60.0,
            delay_bins=(0, 1, 2, 3),
            decision_cadence_seconds=float(
                candidates["target_expected_cadence_seconds"].iloc[0]
            ),
        )
        summary["dataset"] = source["dataset"]
        summary["evaluation_case"] = scenario
        summary["source_type"] = source["source_type"]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _threshold_sensitivity(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measured = decisions[decisions["source_type"].str.contains("measured")]
    for keys, frame in measured.groupby(
        ["dataset", "evaluation_case", "policy_name"], sort=False
    ):
        for threshold in (40.0, 60.0, 100.0, 200.0):
            rows.append(
                {
                    "dataset": keys[0],
                    "evaluation_case": keys[1],
                    "policy_name": keys[2],
                    "threshold_ms": threshold,
                    "success_rate": float(
                        frame["realized_next_latency_ms"].le(threshold).mean()
                    ),
                    "decision_count": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def _threshold_matched_policy_sensitivity(
    sources: list[dict[str, object]],
) -> pd.DataFrame:
    """Re-score frozen fitted artifacts with each deterministic shield cutoff.

    This differs from merely scoring the frozen 60-ms decisions at another
    cutoff. Predictor weights and the validation-selected fallback remain
    frozen; only the explicit threshold in the deterministic shield changes.
    It is not the objective-specific refit/recalibration/reselection protocol
    generated by ``run_commect_threshold_gate_sensitivity.py``.
    """

    rows = []
    for source in sources:
        # The paper's threshold table is a measured-data robustness check.
        # Keeping simulator sweeps separate avoids conflating an application
        # threshold audit with the injected-stress experiment.
        if "measured" not in str(source["source_type"]):
            continue
        candidates = pd.read_csv(Path(source["candidates"]), low_memory=False)
        scenario = str(source["scenario"])
        if "scenario_name" in candidates.columns:
            candidates = candidates[candidates["scenario_name"].eq(scenario)].copy()
        fallback_name = (
            str(candidates["risk_fallback_policy"].mode().iloc[0])
            if "risk_fallback_policy" in candidates
            else "graph"
        )
        fallback_column = (
            "pred_ensemble_uncertainty"
            if fallback_name == "ensemble"
            else "pred_graph"
        )
        gated_fallback_name = (
            str(candidates["validation_gated_fallback_policy"].mode().iloc[0])
            if "validation_gated_fallback_policy" in candidates
            else fallback_name
        )
        gated_fallback_column = {
            "reactive": "latency_mean_ms",
            "graph": "pred_graph",
            "ensemble": "pred_ensemble_uncertainty",
        }[gated_fallback_name]
        for threshold in (40.0, 60.0, 100.0, 200.0):
            scored = add_qos_shielded_scores(
                candidates,
                fallback_column=fallback_column,
                latency_budget_ms=threshold,
                output_column="pred_qos_threshold_matched",
            )
            scored = add_qos_shielded_scores(
                scored,
                fallback_column=gated_fallback_column,
                latency_budget_ms=threshold,
                output_column="pred_gated_qos_threshold_matched",
            )
            summary, decisions = evaluate_decision_policies(
                scored,
                latency_budget_ms=threshold,
                policy_columns={
                    "reactive_greedy": "latency_mean_ms",
                    "predictive_graph_greedy": "pred_graph",
                    "ensemble_uncertainty_selector": "pred_ensemble_uncertainty",
                    "qos_shielded_threshold_matched": "pred_qos_threshold_matched",
                    "validation_gated_threshold_matched": "pred_gated_qos_threshold_matched",
                },
            )
            audit, labels = build_candidate_opportunity_audit(
                candidates,
                thresholds_ms=(threshold,),
            )
            conditioned = build_opportunity_conditioned_results(decisions, labels)
            audit_row = audit.iloc[0]
            merged = summary.merge(
                conditioned[
                    [
                        "policy_name",
                        "opportunity_count",
                        "opportunity_capture_rate",
                        "missed_opportunity_count",
                    ]
                ],
                on="policy_name",
                how="left",
            )
            for _, result in merged.iterrows():
                rows.append(
                    {
                        "dataset": source["dataset"],
                        "evaluation_case": scenario,
                        "source_type": source["source_type"],
                        "diagnostic_scope": "frozen_fitted_artifacts_shield_cutoff_only",
                        "objective_specific_retraining": False,
                        "used_for_threshold_selection": False,
                        "threshold_ms": threshold,
                        "fallback_policy_frozen_on_validation": fallback_name,
                        "gated_fallback_policy_frozen_on_validation": gated_fallback_name,
                        "policy_name": result["policy_name"],
                        "decision_count": int(result["decision_count"]),
                        "success_rate": float(result["success_rate_under_60ms"]),
                        "mean_realized_latency_ms": float(result["mean_realized_latency_ms"]),
                        "p95_realized_latency_ms": float(result["p95_realized_latency_ms"]),
                        "decision_opportunity_rate": float(
                            audit_row["decision_opportunity_rate"]
                        ),
                        "opportunity_count": int(result["opportunity_count"]),
                        "opportunity_capture_rate": float(
                            result["opportunity_capture_rate"]
                        ) if pd.notna(result["opportunity_capture_rate"]) else np.nan,
                        "missed_opportunity_count": int(
                            result["missed_opportunity_count"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _qos_branch_frequency(decisions: pd.DataFrame) -> pd.DataFrame:
    """Count the exact online branch executed by the final selector."""

    selected = decisions[
        decisions["policy_name"].eq("qos_shielded_operational_selector")
    ].copy()
    rows = []
    for keys, frame in selected.groupby(["dataset", "evaluation_case"], sort=False):
        counts = frame["qos_shield_mode"].value_counts()
        total = int(len(frame))
        for branch in (
            "mixed_qos_safeguard",
            "all_qos_fallback",
            "no_qos_fallback",
        ):
            count = int(counts.get(branch, 0))
            rows.append(
                {
                    "dataset": keys[0],
                    "evaluation_case": keys[1],
                    "branch": branch,
                    "decision_count": count,
                    "branch_fraction": count / total if total else 0.0,
                    "total_decisions": total,
                }
            )
    return pd.DataFrame(rows)


def _fallback_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    policies = [
        "reactive_greedy",
        "qos_context_fallback_selector",
        "qos_ensemble_fallback_selector",
        "reactive_hysteresis_selector",
        "qos_hysteresis_selector",
        "qos_shielded_operational_selector",
    ]
    columns = [
        "dataset",
        "evaluation_case",
        "policy_name",
        "decision_count",
        "mean_realized_latency_ms",
        "success_rate_under_60ms",
        "p95_realized_latency_ms",
    ]
    return summary[summary["policy_name"].isin(policies)][columns].copy()


def _branch_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
    """Report next-epoch outcomes for each executed shield branch."""

    selected = decisions[
        decisions["policy_name"].eq("qos_shielded_operational_selector")
    ].copy()
    selected["selected_current_qos"] = selected["reactive_latency_ms"].le(60.0)
    selected["current_to_next_violation"] = (
        selected["selected_current_qos"] & ~selected["success_under_budget"].astype(bool)
    )
    selected["current_to_next_recovery"] = (
        ~selected["selected_current_qos"] & selected["success_under_budget"].astype(bool)
    )
    rows = []
    for keys, frame in selected.groupby(
        ["dataset", "evaluation_case", "qos_shield_mode"], sort=False
    ):
        current_compliant = frame["selected_current_qos"].sum()
        current_noncompliant = len(frame) - current_compliant
        rows.append(
            {
                "dataset": keys[0],
                "evaluation_case": keys[1],
                "branch": keys[2],
                "decision_count": int(len(frame)),
                "next_epoch_success_rate": float(frame["success_under_budget"].mean()),
                "mean_realized_latency_ms": float(frame["realized_next_latency_ms"].mean()),
                "p95_realized_latency_ms": float(frame["realized_next_latency_ms"].quantile(0.95)),
                "mean_decision_gap_ms": float(frame["decision_gap_ms"].mean()),
                "current_compliant_count": int(current_compliant),
                "current_to_next_violation_rate": (
                    float(frame["current_to_next_violation"].sum() / current_compliant)
                    if current_compliant
                    else np.nan
                ),
                "current_noncompliant_count": int(current_noncompliant),
                "current_to_next_recovery_rate": (
                    float(frame["current_to_next_recovery"].sum() / current_noncompliant)
                    if current_noncompliant
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _operational_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Collect the secondary metrics promised by the experimental protocol."""

    policies = [
        "reactive_greedy",
        "reactive_hysteresis_selector",
        "switch_aware_operational_selector",
        "qos_hysteresis_selector",
        "qos_shielded_operational_selector",
    ]
    columns = [
        "dataset",
        "evaluation_case",
        "policy_name",
        "decision_count",
        "success_rate_under_60ms",
        "cvar95_realized_latency_ms",
        "switch_rate",
        "mean_model_and_ranking_time_us",
        "stale_decision_rate",
        "mean_feasible_candidate_count",
        "no_feasible_candidate_rate",
    ]
    return summary[summary["policy_name"].isin(policies)][columns].copy()


def _numerical_consistency_audit(
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    opportunity_audit: pd.DataFrame,
    validation_gate_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Fail the evidence build when reported denominators become inconsistent."""

    rows: list[dict[str, object]] = []
    keys = ["dataset", "evaluation_case", "policy_name"]
    required_switch_summary_fields = {
        "switch_rate",
        "switch_count",
        "eligible_switch_transition_count",
        # Required compatibility alias; see PolicyDecision.
        "switch_transition_count",
        "continuity_reset_count",
        "continuity_segment_count",
    }
    missing_switch_summary_fields = sorted(
        required_switch_summary_fields.difference(summary.columns)
    )
    rows.append(
        {
            "check": "policy_summary_switch_audit_schema",
            "dataset": "all",
            "evaluation_case": "all",
            "policy_name": "not_applicable",
            "observed": (
                "missing="
                + ("|".join(missing_switch_summary_fields) or "none")
            ),
            "expected": "explicit switch numerator, denominator, and continuity counts",
            "passed": not missing_switch_summary_fields,
        }
    )
    decision_key_counts = (
        decisions.groupby(keys, sort=False, dropna=False)
        .size()
        .rename("decision_row_count")
        .reset_index()
    )
    summary_key_counts = (
        summary.groupby(keys, sort=False, dropna=False)
        .size()
        .rename("summary_row_count")
        .reset_index()
    )
    key_cardinality = decision_key_counts.merge(
        summary_key_counts,
        on=keys,
        how="outer",
    )
    for _, key_row in key_cardinality.iterrows():
        decision_count_value = key_row.get("decision_row_count", np.nan)
        summary_count_value = key_row.get("summary_row_count", np.nan)
        decision_row_count = (
            int(decision_count_value) if pd.notna(decision_count_value) else 0
        )
        summary_row_count = (
            int(summary_count_value) if pd.notna(summary_count_value) else 0
        )
        rows.append(
            {
                "check": "policy_summary_decision_key_cardinality",
                "dataset": key_row["dataset"],
                "evaluation_case": key_row["evaluation_case"],
                "policy_name": key_row["policy_name"],
                "observed": (
                    f"decision_rows={decision_row_count};"
                    f"summary_rows={summary_row_count}"
                ),
                "expected": (
                    "at least one decision row and exactly one summary row"
                ),
                "passed": bool(
                    decision_row_count >= 1 and summary_row_count == 1
                ),
            }
        )

    for group_keys, frame in decisions.groupby(
        keys,
        sort=False,
        dropna=False,
    ):
        matching_mask = pd.Series(True, index=summary.index)
        for column, value in zip(keys, group_keys):
            matching_mask &= (
                summary[column].isna()
                if pd.isna(value)
                else summary[column].eq(value)
            )
        matching = summary[matching_mask]
        if len(matching) != 1:
            # The bidirectional cardinality row above records this mismatch;
            # skip ambiguous metric comparisons and fail at the audit gate.
            continue
        observed_count = int(len(frame))
        observed_success_count = int(frame["success_under_budget"].sum())
        recomputed_rate = observed_success_count / observed_count
        reported_rate = float(matching.iloc[0]["success_rate_under_60ms"])
        passed = bool(np.isclose(recomputed_rate, reported_rate, atol=1e-12))
        rows.append(
            {
                "check": "policy_success_denominator",
                "dataset": group_keys[0],
                "evaluation_case": group_keys[1],
                "policy_name": group_keys[2],
                "observed": f"{observed_success_count}/{observed_count}={recomputed_rate:.12f}",
                "expected": f"reported={reported_rate:.12f}",
                "passed": passed,
            }
        )
        p95 = float(frame["realized_next_latency_ms"].quantile(0.95))
        switch_metrics = summarize_switch_transitions(frame)
        if "continuity_segment_start" in frame.columns:
            continuity_segment_count = int(
                frame["continuity_segment_start"].sum()
            )
            expected_eligible_transition_count = (
                observed_count - continuity_segment_count
            )
            rows.append(
                {
                    "check": "switch_transition_t_minus_s_closure",
                    "dataset": group_keys[0],
                    "evaluation_case": group_keys[1],
                    "policy_name": group_keys[2],
                    "observed": (
                        "eligible_switch_transition_count="
                        f"{switch_metrics.eligible_transition_count}"
                    ),
                    "expected": (
                        f"T-S={observed_count}-{continuity_segment_count}="
                        f"{expected_eligible_transition_count}"
                    ),
                    "passed": bool(
                        switch_metrics.eligible_transition_count
                        == expected_eligible_transition_count
                    ),
                }
            )
        recomputed_metrics = {
            "decision_count": float(observed_count),
            "mean_realized_latency_ms": float(
                frame["realized_next_latency_ms"].mean()
            ),
            "mean_decision_gap_ms": float(frame["decision_gap_ms"].mean()),
            "p95_realized_latency_ms": p95,
            "cvar95_realized_latency_ms": empirical_upper_cvar(
                frame["realized_next_latency_ms"].to_numpy(dtype=float),
                0.95,
            ),
            "switch_rate": switch_metrics.switch_rate,
        }
        summary_row = matching.iloc[0]
        switch_count_metrics = {
            "switch_count": switch_metrics.switch_count,
            "eligible_switch_transition_count": (
                switch_metrics.eligible_transition_count
            ),
            # Backward-compatible denominator alias.
            "switch_transition_count": switch_metrics.eligible_transition_count,
            "continuity_reset_count": int(frame["continuity_reset"].sum())
            if "continuity_reset" in frame.columns
            else None,
            "continuity_segment_count": int(
                frame["continuity_segment_start"].sum()
            )
            if "continuity_segment_start" in frame.columns
            else None,
        }
        for metric, recomputed in switch_count_metrics.items():
            if (
                recomputed is not None
                and metric in matching.columns
                and pd.notna(summary_row[metric])
            ):
                recomputed_metrics[metric] = float(recomputed)
        for metric, recomputed in recomputed_metrics.items():
            reported = float(summary_row[metric])
            rows.append(
                {
                    "check": "policy_summary_metric_recomputed",
                    "dataset": group_keys[0],
                    "evaluation_case": group_keys[1],
                    "policy_name": group_keys[2],
                    "observed": f"{metric}={recomputed:.12f}",
                    "expected": f"reported={reported:.12f}",
                    "passed": bool(np.isclose(recomputed, reported, atol=1e-12)),
                }
            )

    regime_count_columns = [
        "all_candidates_succeed_count",
        "mixed_outcome_opportunity_count",
        "all_candidates_fail_count",
        "single_candidate_count",
        "emergency_no_current_feasible_count",
    ]
    for _, row in opportunity_audit.iterrows():
        partition_total = int(sum(int(row[column]) for column in regime_count_columns))
        expected_total = int(row["decision_epoch_count"])
        rows.append(
            {
                "check": "opportunity_regimes_are_disjoint_and_exhaustive",
                "dataset": row["dataset"],
                "evaluation_case": row["evaluation_case"],
                "policy_name": "not_applicable",
                "threshold_ms": float(row["threshold_ms"]),
                "observed": partition_total,
                "expected": expected_total,
                "passed": partition_total == expected_total,
            }
        )

    for _, row in validation_gate_audit.iterrows():
        selected = str(row["validation_gated_fallback"])
        allowed = set(str(row["admissible_fallbacks"]).split("|"))
        rows.append(
            {
                "check": "validation_fallback_frozen_and_allowed",
                "dataset": row["dataset"],
                "evaluation_case": row["evaluation_case"],
                "policy_name": f"fold_{row['fold']}",
                "threshold_ms": "not_applicable",
                "observed": selected,
                "expected": "one of reactive|graph|ensemble; no test outcomes",
                "passed": bool(
                    selected in allowed
                    and bool(row["frozen_before_test"])
                    and not bool(row["test_outcomes_used"])
                ),
            }
        )

    rolling_manifest = (
        REPO_ROOT
        / "results/commect_validation_gated_rolling/rolling_fold_manifest.csv"
    )
    if rolling_manifest.exists():
        manifest = pd.read_csv(rolling_manifest)
        ordered = manifest.sort_values("rolling_fold")
        nonoverlap = bool(
            all(
                int(left["test_last_epoch"]) < int(right["test_first_epoch"])
                for (_, left), (_, right) in zip(
                    ordered.iloc[:-1].iterrows(),
                    ordered.iloc[1:].iterrows(),
                )
            )
        )
        if "expected_boundary_closed_test_decision_epochs" in ordered:
            expected_test_epoch_total = int(
                ordered[
                    "expected_boundary_closed_test_decision_epochs"
                ].sum()
            )
            expected_test_epoch_source = (
                "sum of per-fold wall-clock-boundary-closed planned epochs"
            )
        else:
            # Backward-compatible derivation for archived manifests written
            # before the explicit expected-count column was introduced.  For
            # a contiguous one-step fold, the final decision's target crosses
            # the boundary, so last - first is the closed epoch count.
            expected_test_epoch_total = int(
                (
                    ordered["test_last_epoch"]
                    - ordered["test_first_epoch"]
                ).sum()
            )
            expected_test_epoch_source = (
                "derived from archived per-fold one-step interval boundaries"
            )
        observed_test_epoch_total = int(
            ordered["test_decision_epochs"].sum()
        )
        rows.append(
            {
                "check": "rolling_test_epochs_do_not_overlap",
                "dataset": "COMMECT",
                "evaluation_case": "five_fold_rolling_origin",
                "policy_name": "not_applicable",
                "threshold_ms": "not_applicable",
                "observed": observed_test_epoch_total,
                "expected": expected_test_epoch_total,
                "passed": nonoverlap
                and observed_test_epoch_total == expected_test_epoch_total,
                "expected_source": expected_test_epoch_source,
            }
        )
        chronological = bool(
            (
                ordered["train_last_epoch"]
                < ordered["calibration_first_epoch"]
            ).all()
            and (
                ordered["calibration_last_epoch"]
                < ordered["selection_first_epoch"]
            ).all()
            and (
                ordered["selection_last_epoch"]
                < ordered["test_first_epoch"]
            ).all()
        )
        rows.append(
            {
                "check": "rolling_train_calibration_selection_test_are_chronological",
                "dataset": "COMMECT",
                "evaluation_case": "five_fold_rolling_origin",
                "policy_name": "not_applicable",
                "threshold_ms": "not_applicable",
                "observed": "train < calibration < selection < test",
                "expected": "strict order in every fold",
                "passed": chronological,
            }
        )

    rolling_skew_path = (
        REPO_ROOT
        / "results/commect_rolling_timestamp_sensitivity"
        / "rolling_timestamp_skew_policy_sensitivity.csv"
    )
    if rolling_skew_path.exists():
        rolling_skew = pd.read_csv(rolling_skew_path)
        requested_cases = {
            "le_500ms",
            "le_1000ms",
            "le_2000ms",
            "le_5000ms",
            "full",
        }
        observed_cases = set(rolling_skew["skew_case"].astype(str))
        rows.append(
            {
                "check": "rolling_skew_requested_cases_complete",
                "dataset": "COMMECT",
                "evaluation_case": "rolling_skew_sensitivity",
                "policy_name": "not_applicable",
                "threshold_ms": 60.0,
                "observed": "|".join(sorted(observed_cases)),
                "expected": "|".join(sorted(requested_cases)),
                "passed": observed_cases == requested_cases,
            }
        )
        for skew_case, case in rolling_skew.groupby("skew_case", sort=False):
            indexed = case.set_index("policy_name")
            reactive = indexed.loc["reactive_greedy"]
            gated = indexed.loc["validation_gated_qos_selector"]
            same_metrics = all(
                np.isclose(
                    float(reactive[column]),
                    float(gated[column]),
                    atol=1e-12,
                )
                for column in (
                    "decision_count",
                    "success_rate_under_60ms",
                    "mean_realized_latency_ms",
                    "p95_realized_latency_ms",
                    "cvar95_realized_latency_ms",
                )
            )
            valid_opportunities = bool(
                int(reactive["decision_opportunity_count"])
                <= int(reactive["decision_count"])
            )
            rows.append(
                {
                    "check": "rolling_skew_gate_abstention_closure",
                    "dataset": "COMMECT",
                    "evaluation_case": str(skew_case),
                    "policy_name": "validation_gated_qos_selector",
                    "threshold_ms": 60.0,
                    "observed": (
                        f"N={int(gated['decision_count'])};"
                        f"opp={int(gated['decision_opportunity_count'])};"
                        f"success={float(gated['success_rate_under_60ms']):.12f}"
                    ),
                    "expected": "all folds abstain and gated metrics equal reactive",
                    "passed": bool(
                        case["gate_all_folds_abstained"].astype(bool).all()
                        and same_metrics
                        and valid_opportunities
                    ),
                }
            )

    design_path = (
        REPO_ROOT
        / "results/gate_design_sensitivity/gate_design_sensitivity.csv"
    )
    if design_path.exists():
        design = pd.read_csv(design_path)
        unique_grid = design[
            [
                "n_min_opportunity_bearing_groups",
                "epsilon_aggregate",
                "practical_cvar_gain_ms",
                "latency_cap_ms",
            ]
        ].drop_duplicates()
        rows.append(
            {
                "check": "prospective_design_grid_cardinality",
                "dataset": "prospective_design",
                "evaluation_case": "not_measured_efficacy",
                "policy_name": "not_applicable",
                "threshold_ms": "not_applicable",
                "observed": f"rows={len(design)};unique={len(unique_grid)}",
                "expected": "108 requested Cartesian configurations",
                "passed": len(design) == 108 and len(unique_grid) == 108,
            }
        )
        for row_index, design_row in design.iterrows():
            floor_closure = all(
                int(design_row[f"u{uses}_joint_best_case_group_floor"])
                == max(
                    int(design_row["n_min_opportunity_bearing_groups"]),
                    int(
                        design_row[
                            f"u{uses}_success_zero_harm_group_floor"
                        ]
                    ),
                    int(design_row[f"u{uses}_tail_best_case_group_floor"]),
                )
                for uses in (1, 5)
            )
            rows.append(
                {
                    "check": "prospective_design_floor_closure",
                    "dataset": "prospective_design",
                    "evaluation_case": f"configuration_{row_index + 1}",
                    "policy_name": "reactive_abstention",
                    "threshold_ms": "not_applicable",
                    "observed": (
                        f"u1={int(design_row['u1_joint_best_case_group_floor'])};"
                        f"u5={int(design_row['u5_joint_best_case_group_floor'])}"
                    ),
                    "expected": "joint=max(n_min, success floor, tail floor); measured admission false",
                    "passed": bool(
                        floor_closure
                        and not bool(design_row["measured_admitted"])
                        and int(design_row["measured_opportunity_group_count"])
                        == 1
                    ),
                }
            )

    audit = pd.DataFrame(rows)
    failures = audit[~audit["passed"].astype(bool)]
    if not failures.empty:
        raise AssertionError(
            "numerical consistency audit failed:\n" + failures.to_string(index=False)
        )
    return audit


def _xai_case_studies(decisions: pd.DataFrame) -> pd.DataFrame:
    """Select one auditable final-policy decision for each branch type."""

    selected = decisions[
        decisions["policy_name"].eq("qos_shielded_operational_selector")
    ].copy()
    selected["explanation_fidelity_error_ms"] = (
        selected["selected_online_score"]
        - selected["xai_explained_signed_total_ms"]
    ).abs()
    rows = []
    for branch in (
        "mixed_qos_safeguard",
        "all_qos_fallback",
        "no_qos_fallback",
    ):
        candidates = selected[selected["qos_shield_mode"].eq(branch)].copy()
        if candidates.empty:
            continue
        candidates["display_margin"] = candidates["xai_score_margin_ms"].where(
            candidates["xai_score_margin_ms"].lt(1.0e6), np.inf
        )
        example = candidates.sort_values(
            ["display_margin", "dataset", "session_bin_index"]
        ).iloc[0]
        rows.append(
            {
                "branch": branch,
                "dataset": example["dataset"],
                "evaluation_case": example["evaluation_case"],
                "decision_epoch": int(example["session_bin_index"]),
                "selected_path": example["chosen_relative_path"],
                "runner_up_path": example["xai_runner_up_relative_path"],
                "selected_current_latency_ms": float(example["reactive_latency_ms"]),
                "selected_online_score_ms": float(example["selected_online_score"]),
                "runner_up_score_ms": float(example["xai_runner_up_score_ms"]),
                "score_margin_ms": float(example["xai_score_margin_ms"]),
                "realized_next_latency_ms": float(example["realized_next_latency_ms"]),
                "next_epoch_success": int(example["success_under_budget"]),
                "fallback_policy": example["xai_fallback_policy"],
                "counterfactual_reason": example["xai_counterfactual_reason"],
                "explanation_fidelity_error_ms": float(
                    example["explanation_fidelity_error_ms"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _training_hyperparameters() -> pd.DataFrame:
    config = load_config(REPO_ROOT / "configs/experiment.yaml")
    forecasting = config["forecasting"]
    optimization = config["optimization"]
    disagreement = optimization["disagreement_aware"]
    ensemble = optimization["ensemble_uncertainty"]
    policy_split = forecasting["policy_evaluation_ratios"]
    risk_control = optimization["risk_control"]
    rows = [
        ("Temporal/context model", f"{forecasting['temporal_model']} / {forecasting['graph_context_model']}"),
        ("Lag steps", ", ".join(str(value) for value in forecasting["lag_steps"])),
        ("Train/validation/test", f"{forecasting['train_ratio']:.2f}/{forecasting['val_ratio']:.2f}/{forecasting['test_ratio']:.2f}"),
        (
            "Policy train/calibration/selection/test",
            "/".join(
                f"{policy_split[name]:.2f}"
                for name in ("train", "calibration", "selection", "test")
            ),
        ),
        ("QoS threshold", f"{optimization['latency_budget_ms']:g} ms"),
        ("Residual-risk fit", disagreement["residual_risk_fit"]),
        ("Risk-gate quantile", f"{disagreement['gate_validation_quantile']:.2f}"),
        ("Conformal coverage", f"{disagreement['calibration_coverage']:.2f}"),
        ("Ensemble members", str(ensemble["ensemble_members"])),
        ("Ensemble row/feature fraction", f"{ensemble['row_fraction']:.2f}/{ensemble['feature_fraction']:.2f}"),
        ("Ensemble uncertainty multiplier", f"{ensemble['lambda_ens']:.2f}"),
        ("Hysteresis penalty", f"{optimization['online_switch_penalty_ms']:g} ms"),
        ("Gate familywise alpha", f"{risk_control['familywise_alpha']:.2f}"),
        (
            "Gate QoS non-inferiority margin",
            f"{100 * risk_control['noninferiority_margin']:.1f} percentage points",
        ),
        (
            "Gate opportunity-conditioned QoS non-inferiority margin",
            f"{100 * risk_control.get('opportunity_noninferiority_margin', 0.02):.1f} "
            "percentage points",
        ),
        (
            "Minimum effective opportunities",
            f"{risk_control['minimum_effective_opportunities']:g}",
        ),
        (
            "Minimum practical CVaR gain",
            f"{risk_control['practical_cvar_gain_ms']:g} ms",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


def _runtime_environment() -> dict[str, object]:
    def installed_versions(distributions: tuple[str, ...]) -> dict[str, str]:
        packages: dict[str, str] = {}
        for distribution in distributions:
            try:
                packages[distribution] = version(distribution)
            except PackageNotFoundError:
                packages[distribution] = "not installed"
        return packages

    return {
        "schema_version": 2,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "package_groups": {
            "runtime": installed_versions(RUNTIME_DISTRIBUTIONS),
            "test": installed_versions(TEST_DISTRIBUTIONS),
        },
        "dependency_declarations": {
            "canonical_direct_and_test_pins": "requirements-lock.txt",
            "canonical_install_requirements": "requirements.txt",
            "optional_hypatia_requirements": "requirements-hypatia.txt",
            "optional_hypatia_in_canonical_build": False,
            "optional_hypatia_dependency_status": "non-canonical and unverified",
        },
        "gpu_used": False,
        "descriptive_bootstrap_protocol": {
            "method": "segment_stratified_circular_moving_block",
            "fixed_protocol_segment_columns": ["continuity_segment_id"],
            "rolling_protocol_segment_columns": [
                "rolling_fold",
                "continuity_segment_id",
            ],
            "fixed_segment_sample_sizes": True,
            "within_segment_circular_wrap_only": True,
            "metric_missingness_starts_new_segment": True,
            "inferential_scope": (
                "descriptive dependence-aware intervals and centered-null "
                "diagnostics; not an independent-group guarantee"
            ),
        },
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "model_scoring": (
                "all frozen expert predictions and online score construction, "
                "divided by the number of decision epochs"
            ),
            "ranking": "per-epoch candidate filtering, sorting, and selection",
            "excluded": "data loading, model fitting, and model serialization",
        },
    }


def _simulator_parameters() -> pd.DataFrame:
    metadata_path = (
        REPO_ROOT
        / "data/processed/physics_informed_orbital_multipath_5s.metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    altitude = float(metadata["altitude_km"])
    satellites = int(metadata["satellites"])
    return pd.DataFrame(
        [
            ("Decision cadence", f"{metadata['bin_seconds']} s"),
            ("Trace duration per seed", f"{metadata['duration_hours']} h"),
            ("Satellites / gateways", f"{satellites} / {metadata['gateways']}"),
            ("Concurrent paths", str(metadata["candidate_paths_per_decision"])),
            ("Satellite altitude", f"{altitude:.0f}-{altitude + 15 * 2:.0f} km"),
            ("Visible elevation", "> 10 degrees"),
            ("Handover trigger", "elevation < 18 degrees or visibility transition"),
            ("Orbital-period range", f"5550-{5550 + 90 * (satellites - 1)} s"),
            ("Gateway base delay", "12-22 ms"),
            ("Handover penalty", f"{metadata.get('handover_penalty_ms', 10.0):g} ms"),
            ("Gateway attenuation", f"{metadata.get('gateway_attenuation_ms', 24.0):g} ms"),
            ("Satellite incident", f"{metadata.get('satellite_incident_ms', 38.0):g} ms"),
            ("Invisible-path penalty", f"{metadata.get('invisible_path_penalty_ms', 85.0):g} ms"),
            ("Independent seeds", "2026-2035"),
        ],
        columns=["parameter", "value"],
    )


def _simulator_model_specification() -> dict[str, object]:
    metadata_path = (
        REPO_ROOT
        / "data/processed/physics_informed_orbital_multipath_5s.metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    keys = [
        "latency_equation",
        "common_load_equation",
        "queue_equation",
        "reply_equation",
        "latency_std_equation",
        "gateway_attenuation_schedule",
        "satellite_incident_schedule",
        "physical_components",
    ]
    return {key: metadata[key] for key in keys}


def _collect_supporting_evidence(output: Path) -> None:
    """Copy reviewer-facing audits into the canonical evidence directory."""

    for output_name, relative_source in SUPPORTING_EVIDENCE.items():
        source = REPO_ROOT / relative_source
        if not source.exists():
            raise FileNotFoundError(f"Required supporting evidence is missing: {source}")
        destination = output / output_name
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            destination.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            pd.read_csv(source).to_csv(destination, index=False)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_file_records(output: Path) -> list[dict[str, object]]:
    """Return exact records for every evidence file except the manifest."""

    files: list[dict[str, object]] = []
    manifest_path = output / "evidence_manifest.json"
    for path in sorted(output.rglob("*")):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise ValueError(f"evidence tree must not contain symlinks: {path}")
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _stream_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return files


def _write_evidence_manifest(output: Path) -> None:
    """Atomically record content hashes for every generated evidence file."""

    files = _evidence_file_records(output)
    payload = json.dumps(
        {"schema_version": 1, "files": files},
        indent=2,
    ) + "\n"
    manifest_path = output / "evidence_manifest.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output,
            prefix=".evidence_manifest.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _verify_evidence_manifest(output: Path) -> dict[str, object]:
    """Reject any missing, extra, size-drifted, or hash-drifted evidence file."""

    manifest_path = output / "evidence_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing evidence manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing evidence manifest schema_version")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("evidence manifest 'files' must be a list")

    expected: dict[str, dict[str, object]] = {}
    for index, record in enumerate(raw_files):
        if not isinstance(record, dict):
            raise ValueError(f"manifest file record {index} is not an object")
        relative = record.get("path")
        sha256 = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"manifest file record {index} has an invalid path")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative != pure.as_posix():
            raise ValueError(f"unsafe or non-canonical manifest path: {relative!r}")
        if relative == "evidence_manifest.json":
            raise ValueError("evidence manifest must not hash itself")
        if relative in expected:
            raise ValueError(f"duplicate evidence manifest path: {relative}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"invalid SHA-256 for manifest path: {relative}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid byte size for manifest path: {relative}")
        expected[relative] = record

    observed_records = _evidence_file_records(output)
    observed = {
        str(record["path"]): record
        for record in observed_records
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    size_drift = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path]["bytes"] != observed[path]["bytes"]
    )
    hash_drift = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path]["sha256"] != observed[path]["sha256"]
    )
    if missing or extra or size_drift or hash_drift:
        raise ValueError(
            "evidence manifest verification failed: "
            f"missing={missing}; extra={extra}; size_drift={size_drift}; "
            f"hash_drift={hash_drift}"
        )
    return {
        "schema_version": 1,
        "verified_file_count": len(observed),
        "manifest_sha256": _stream_sha256(manifest_path),
    }


def _control_loop_sensitivity(decisions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a hypothetical network-RTT-plus-added-delay total."""

    rows = []
    selected = decisions[
        decisions["policy_name"].eq("qos_shielded_operational_selector")
    ]
    for keys, frame in selected.groupby(["dataset", "evaluation_case"], sort=False):
        for delay_ms in (0.0, 5.0, 10.0, 25.0, 50.0, 100.0):
            network_rtt_plus_added_delay = (
                frame["realized_next_latency_ms"] + delay_ms
            )
            rows.append(
                {
                    "dataset": keys[0],
                    "evaluation_case": keys[1],
                    "control_loop_latency_ms": delay_ms,
                    "success_rate_under_60ms": float(
                        network_rtt_plus_added_delay.le(60.0).mean()
                    ),
                    "mean_network_rtt_plus_added_delay_ms": float(
                        network_rtt_plus_added_delay.mean()
                    ),
                    "p95_network_rtt_plus_added_delay_ms": float(
                        network_rtt_plus_added_delay.quantile(0.95)
                    ),
                    "decision_count": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def _paired_statistics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("shield_vs_reactive", "qos_shielded_operational_selector", "reactive_greedy"),
        ("shield_vs_context", "qos_shielded_operational_selector", "predictive_graph_greedy"),
        ("shield_vs_ensemble", "qos_shielded_operational_selector", "ensemble_uncertainty_selector"),
        ("shield_vs_calibrated", "qos_shielded_operational_selector", "calibrated_operational_selector"),
        ("shield_vs_qos_context", "qos_shielded_operational_selector", "qos_context_fallback_selector"),
        ("shield_vs_qos_ensemble", "qos_shielded_operational_selector", "qos_ensemble_fallback_selector"),
        ("shield_vs_reactive_hysteresis", "qos_shielded_operational_selector", "reactive_hysteresis_selector"),
        ("shield_vs_predictive_hysteresis", "qos_shielded_operational_selector", "switch_aware_operational_selector"),
        ("shield_vs_qos_hysteresis", "qos_shielded_operational_selector", "qos_hysteresis_selector"),
        ("gated_vs_reactive", "validation_gated_qos_selector", "reactive_greedy"),
        ("gated_vs_shield", "validation_gated_qos_selector", "qos_shielded_operational_selector"),
    ]
    metrics = ["realized_next_latency_ms", "decision_gap_ms", "success_under_budget"]
    for keys, frame in decisions.groupby(["dataset", "evaluation_case"], sort=False):
        result = build_paired_policy_significance(
            frame,
            comparisons=comparisons,
            metric_columns=metrics,
            block_length=max(2, round(frame["session_bin_index"].nunique() ** (1 / 3))),
            segment_columns=("continuity_segment_id",),
        )
        if result.empty:
            continue
        result["dataset"] = keys[0]
        result["evaluation_case"] = keys[1]
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _confidence_intervals(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["realized_next_latency_ms", "decision_gap_ms", "success_under_budget"]
    for keys, frame in decisions.groupby(["dataset", "evaluation_case"], sort=False):
        intervals = build_bootstrap_policy_intervals(
            frame,
            metric_columns=metrics,
            block_length=max(2, round(frame["session_bin_index"].nunique() ** (1 / 3))),
            segment_columns=("continuity_segment_id",),
        )
        intervals["dataset"] = keys[0]
        intervals["evaluation_case"] = keys[1]
        rows.append(intervals)
    return pd.concat(rows, ignore_index=True)


def _plot_primary(summary: pd.DataFrame, output: Path) -> None:
    configure_ieee_figure_style(base_font_size=9.0)
    plot = summary[summary["policy_name"].isin(
        [
            "reactive_greedy",
            "predictive_graph_greedy",
            "ensemble_uncertainty_selector",
            "qos_shielded_operational_selector",
            "validation_gated_qos_selector",
        ]
    )].copy()
    case_labels = {
        ("COMMECT", "measured_multi_access"): "COMMECT",
        ("LENS Victoria holdout", "measured_multihomed"): "Victoria",
        ("Physics-informed orbital trace", "session_holdout"): "Orbital\nsession",
        ("Physics-informed orbital trace", "operational_moderate"): "Orbital\nmoderate",
        ("Physics-informed orbital trace", "operational_severe"): "Orbital\nsevere",
    }
    plot["case"] = [
        case_labels.get((row.dataset, row.evaluation_case), row.dataset)
        for row in plot.itertuples(index=False)
    ]
    cases = plot["case"].drop_duplicates().tolist()
    policies = plot["policy_name"].drop_duplicates().tolist()
    styles = {
        "reactive_greedy": ("#4d4d4d", ""),
        "predictive_graph_greedy": ("#1f77b4", "//"),
        "ensemble_uncertainty_selector": ("#e69f00", "xx"),
        "qos_shielded_operational_selector": ("#009e73", "\\\\"),
        "validation_gated_qos_selector": ("#cc79a7", ".."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_TEXT_WIDTH_IN, 3.55))
    width = 0.16
    x = np.arange(len(cases))
    for index, policy in enumerate(policies):
        frame = plot[plot["policy_name"].eq(policy)].set_index("case").reindex(cases)
        offset = (index - (len(policies) - 1) / 2) * width
        color, hatch = styles[policy]
        bar_options = {
            "label": DISPLAY[policy],
            "color": color,
            "edgecolor": "#202020",
            "linewidth": 0.45,
            "hatch": hatch,
        }
        axes[0].bar(
            x + offset,
            frame["success_rate_under_60ms"],
            width,
            **bar_options,
        )
        axes[1].bar(
            x + offset,
            frame["p95_realized_latency_ms"],
            width,
            **bar_options,
        )
    axes[0].set_title("(a) Latency-QoS success")
    axes[0].set_ylabel("Success rate at 60 ms")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_title("(b) Selected-path tail latency")
    axes[1].set_ylabel("P95 realized latency (ms)")
    for axis in axes:
        axis.set_xticks(x, cases, rotation=20, ha="right")
        axis.grid(axis="y", color=OPAQUE_GRID_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.91,
        bottom=0.29,
        wspace=0.27,
    )
    save_png_pdf_pair(fig, output)
    plt.close(fig)


def _plot_delayed_state_replay(replay: pd.DataFrame, output: Path) -> None:
    """Show how selected paths behave when execution uses a later trace state."""

    configure_ieee_figure_style(base_font_size=9.0)

    cases = [
        ("COMMECT", "measured_multi_access", "COMMECT measured alternatives"),
        (
            "Physics-informed orbital trace",
            "operational_severe",
            "Orbital severe injected stress",
        ),
    ]
    policies = [
        "reactive_greedy",
        "qos_shielded_operational_selector",
        "validation_gated_qos_selector",
    ]
    styles = {
        "reactive_greedy": ("#355070", "o", "-"),
        "qos_shielded_operational_selector": ("#c1121f", "s", "--"),
        "validation_gated_qos_selector": ("#087e8b", "^", "-."),
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_TEXT_WIDTH_IN, 3.15),
        sharey=True,
    )
    for axis, (dataset, scenario, title) in zip(axes, cases):
        frame = replay[
            replay["dataset"].eq(dataset)
            & replay["evaluation_case"].eq(scenario)
            & replay["policy_name"].isin(policies)
        ].copy()
        for policy in policies:
            values = frame[frame["policy_name"].eq(policy)].sort_values("delay_bins")
            color, marker, linestyle = styles[policy]
            axis.plot(
                values["delay_bins"],
                values["network_qos_success_rate"],
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                color=color,
                label=DISPLAY[policy],
            )
        availability = (
            frame.groupby("delay_bins", as_index=False)["execution_availability_rate"]
            .mean()
            .sort_values("delay_bins")
        )
        axis.plot(
            availability["delay_bins"],
            availability["execution_availability_rate"],
            color="#6c757d",
            linestyle=":",
            marker="D",
            linewidth=1.7,
            label="Execution availability",
        )
        axis.set_title(title)
        axis.set_xlabel("Execution delay (trace bins)")
        axis.set_xticks(sorted(frame["delay_bins"].unique()))
        axis.set_ylim(0.0, 1.04)
        axis.grid(color=OPAQUE_GRID_COLOR)
    axes[0].set_ylabel("Rate at delayed trace state")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Delayed-state replay: ranking degradation and path availability",
        y=0.975,
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.82,
        bottom=0.24,
        wspace=0.18,
    )
    save_png_pdf_pair(fig, output)
    plt.close(fig)


def _plot_decision_opportunity(
    audit: pd.DataFrame,
    conditioned: pd.DataFrame,
    output: Path,
) -> None:
    """Visualize candidate-set actionability and policy capture at 60 ms."""

    configure_ieee_figure_style(base_font_size=9.0)

    cases = [
        ("COMMECT", "measured_multi_access", "COMMECT"),
        ("LENS Victoria holdout", "measured_multihomed", "Victoria"),
        ("Physics-informed orbital trace", "session_holdout", "Orbital\nsession"),
        ("Physics-informed orbital trace", "operational_moderate", "Orbital\nmoderate"),
        ("Physics-informed orbital trace", "operational_severe", "Orbital\nsevere"),
    ]
    regimes = [
        ("all_candidates_succeed_rate", "All succeed", "#7bc8a4", ""),
        ("mixed_outcome_opportunity_rate", "Choice changes QoS", "#f4b942", "//"),
        ("all_candidates_fail_rate", "All fail", "#e76f51", "xx"),
        ("single_candidate_rate", "Single candidate", "#8da0cb", ".."),
        (
            "emergency_no_current_feasible_rate",
            "No feasible path",
            "#6c757d",
            "\\\\",
        ),
    ]
    at_sixty = audit[audit["threshold_ms"].eq(60.0)].copy()
    rows = []
    for dataset, scenario, label in cases:
        row = at_sixty[
            at_sixty["dataset"].eq(dataset)
            & at_sixty["evaluation_case"].eq(scenario)
        ].iloc[0].to_dict()
        row["case"] = label
        rows.append(row)
    plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_TEXT_WIDTH_IN, 3.5))
    x = np.arange(len(plot))
    bottom = np.zeros(len(plot))
    for column, label, color, hatch in regimes:
        values = plot[column].to_numpy(dtype=float)
        axes[0].bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="#303030",
            linewidth=0.45,
            hatch=hatch,
        )
        bottom += values
    axes[0].set_title("(a) Candidate-set outcome regimes")
    axes[0].set_ylabel("Fraction of decision epochs")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(x, plot["case"])
    axes[0].grid(axis="y", color=OPAQUE_GRID_COLOR)

    policies = [
        "reactive_greedy",
        "predictive_greedy",
        "predictive_graph_greedy",
        "ensemble_uncertainty_selector",
        "qos_shielded_operational_selector",
    ]
    commect = conditioned[
        conditioned["dataset"].eq("COMMECT")
        & conditioned["evaluation_case"].eq("measured_multi_access")
        & conditioned["threshold_ms"].eq(60.0)
        & conditioned["policy_name"].isin(policies)
    ].set_index("policy_name").reindex(policies)
    labels = ["Reactive", "Temporal", "Context", "Ensemble", "Predictive shield"]
    capture_bars = axes[1].bar(
        np.arange(len(policies)),
        commect["opportunity_capture_rate"],
        color=["#355070", "#6d597a", "#087e8b", "#f4a261", "#c1121f"],
        edgecolor="#303030",
        linewidth=0.45,
    )
    for patch, hatch in zip(
        capture_bars.patches,
        ["", "//", "xx", "..", "\\\\"],
    ):
        patch.set_hatch(hatch)
    axes[1].set_title("(b) COMMECT opportunity capture")
    axes[1].set_ylabel("Opportunity capture rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(np.arange(len(policies)), labels, rotation=20, ha="right")
    axes[1].grid(axis="y", color=OPAQUE_GRID_COLOR)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.91,
        bottom=0.25,
        wspace=0.28,
    )
    save_png_pdf_pair(fig, output)
    plt.close(fig)


def _plot_thresholds(thresholds: pd.DataFrame, output: Path) -> None:
    plot = thresholds[thresholds["policy_name"].isin(
        ["reactive_greedy", "predictive_graph_greedy", "qos_shielded_operational_selector"]
    )]
    datasets = plot["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.5, 4.2), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        frame = plot[plot["dataset"].eq(dataset)]
        for policy, policy_frame in frame.groupby("policy_name", sort=False):
            axis.plot(
                policy_frame["threshold_ms"],
                policy_frame["success_rate"],
                marker="o",
                linewidth=1.8,
                label=DISPLAY[policy],
            )
        axis.set_title(dataset)
        axis.set_xlabel("QoS threshold (ms)")
        axis.set_ylim(0.0, 1.05)
        axis.grid(color=OPAQUE_GRID_COLOR)
    axes[0, 0].set_ylabel("Success rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.24)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation(summary: pd.DataFrame, output: Path) -> None:
    policies = [
        "predictive_greedy",
        "predictive_graph_greedy",
        "predictive_simple_fusion_greedy",
        "disagreement_only_selector",
        "ensemble_uncertainty_selector",
        "calibrated_operational_selector",
        "qos_shielded_operational_selector",
    ]
    labels = [
        "Temporal",
        "Context",
        "Fusion",
        "Disagreement",
        "Ensemble",
        "Calibrated",
        "Predictive shield",
    ]
    frame = summary[
        summary["dataset"].eq("COMMECT") & summary["policy_name"].isin(policies)
    ].set_index("policy_name").reindex(policies)
    x = np.arange(len(policies))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    axes[0].bar(x, frame["success_rate_under_60ms"], color="#087e8b")
    axes[1].bar(x, frame["p95_realized_latency_ms"], color="#ff5a5f")
    axes[0].set_title("Component attribution: QoS success")
    axes[0].set_ylabel("Success rate under 60 ms")
    axes[0].set_ylim(0.0, 0.72)
    axes[1].set_title("Component attribution: tail latency")
    axes[1].set_ylabel("P95 realized latency (ms)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", color=OPAQUE_GRID_COLOR)
    fig.subplots_adjust(bottom=0.27, wspace=0.28)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_control_loop(sensitivity: pd.DataFrame, output: Path) -> None:
    cases = [
        ("COMMECT", "measured_multi_access", "COMMECT measured"),
        ("Physics-informed orbital trace", "operational_severe", "Orbital severe"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for dataset, scenario, label in cases:
        frame = sensitivity[
            sensitivity["dataset"].eq(dataset)
            & sensitivity["evaluation_case"].eq(scenario)
        ]
        axes[0].plot(
            frame["control_loop_latency_ms"],
            frame["success_rate_under_60ms"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            frame["control_loop_latency_ms"],
            frame["p95_network_rtt_plus_added_delay_ms"],
            marker="o",
            label=label,
        )
    axes[0].set_title("QoS sensitivity to hypothetical added delay")
    axes[0].set_ylabel("Success rate under 60 ms")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_title("P95 network RTT plus added delay")
    axes[1].set_ylabel("P95 total latency (ms)")
    for axis in axes:
        axis.set_xlabel("Hypothetical added delay (ms)")
        axis.grid(color=OPAQUE_GRID_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_dashboard(
    thresholds: pd.DataFrame,
    control_loop: pd.DataFrame,
    output: Path,
) -> None:
    """Combine threshold and control-delay sensitivity in one journal figure."""

    configure_ieee_figure_style(base_font_size=9.0)

    threshold_policies = [
        "reactive_greedy",
        "predictive_graph_greedy",
        "qos_shielded_operational_selector",
    ]
    measured_cases = [
        ("COMMECT", "(a) COMMECT measured"),
        ("LENS Victoria holdout", "(b) Victoria holdout"),
    ]
    control_cases = [
        ("COMMECT", "measured_multi_access", "COMMECT measured"),
        ("Physics-informed orbital trace", "operational_severe", "Orbital severe"),
    ]
    threshold_styles = {
        "reactive_greedy": ("#4d4d4d", "o", "-"),
        "predictive_graph_greedy": ("#1f77b4", "s", "--"),
        "qos_shielded_operational_selector": ("#009e73", "^", "-."),
    }
    control_styles = {
        "COMMECT measured": ("#355070", "o", "-"),
        "Orbital severe": ("#c1121f", "s", "--"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_TEXT_WIDTH_IN, 4.55))
    for axis, (dataset, title) in zip(axes[0], measured_cases):
        frame = thresholds[thresholds["dataset"].eq(dataset)]
        for policy in threshold_policies:
            policy_frame = frame[frame["policy_name"].eq(policy)]
            color, marker, linestyle = threshold_styles[policy]
            axis.plot(
                policy_frame["threshold_ms"],
                policy_frame["success_rate"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                label=DISPLAY[policy],
            )
        axis.set_title(title)
        axis.set_xlabel("QoS threshold (ms)")
        axis.set_ylim(0.0, 1.05)
        axis.grid(color=OPAQUE_GRID_COLOR)
    axes[0, 0].set_ylabel("Success rate")

    for dataset, scenario, label in control_cases:
        frame = control_loop[
            control_loop["dataset"].eq(dataset)
            & control_loop["evaluation_case"].eq(scenario)
        ]
        color, marker, linestyle = control_styles[label]
        axes[1, 0].plot(
            frame["control_loop_latency_ms"],
            frame["success_rate_under_60ms"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
        axes[1, 1].plot(
            frame["control_loop_latency_ms"],
            frame["p95_network_rtt_plus_added_delay_ms"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
    axes[1, 0].set_title("(c) Added delay: success")
    axes[1, 0].set_ylabel("Success rate under 60 ms")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("(d) Added delay: P95 latency")
    axes[1, 1].set_ylabel("P95 latency (ms)")
    for axis in axes[1]:
        axis.set_xlabel("Hypothetical added delay (ms)")
        axis.grid(color=OPAQUE_GRID_COLOR)

    threshold_handles, threshold_labels = axes[0, 0].get_legend_handles_labels()
    control_handles, control_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        threshold_handles + control_handles,
        threshold_labels + control_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.19,
        hspace=0.52,
        wspace=0.27,
    )
    save_png_pdf_pair(fig, output)
    plt.close(fig)


def _plot_method_pipeline(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.0, 3.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.25, "Concurrent\ntelemetry", "#d8ecff"),
        (2.15, "Matched temporal\nand context experts", "#dff3e3"),
        (4.45, "Calibration-only\nresidual fitting", "#fff1c9"),
        (6.55, "Independent selection\nLCB + opportunity + CVaR", "#ffe0d7"),
        (8.55, "Latency-QoS\nfilter", "#e7dcff"),
        (10.45, "Selected path +\nexact audit", "#d9f1ef"),
    ]
    for index, (x, label, color) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, 1.0),
            1.35,
            1.0,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="#25364a",
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + 0.675, 1.5, label, ha="center", va="center", fontsize=9)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.38, 1.5),
                    (next_x - 0.05, 1.5),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.1,
                    color="#25364a",
                )
            )
    axis.text(
        6.0,
        0.35,
        "Train, calibration, policy selection, and test are chronological and disjoint; test outcomes never affect admission.",
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_multiseed(output: Path) -> pd.DataFrame:
    source = REPO_ROOT / "results/transactions_seed_matrix/multi_seed_policy_summary.csv"
    summary = pd.read_csv(source)
    scenarios = ["session_holdout", "operational_moderate", "operational_severe"]
    policies = [
        "reactive_greedy",
        "predictive_graph_greedy",
        "ensemble_uncertainty_selector",
        "qos_shielded_operational_selector",
    ]
    plot = summary[
        summary["scenario_name"].isin(scenarios)
        & summary["policy_name"].isin(policies)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    width = 0.19
    x = np.arange(len(scenarios))
    for index, policy in enumerate(policies):
        offset = (index - 1.5) * width
        for axis, metric in zip(
            axes,
            ["mean_realized_latency_ms", "success_rate_under_60ms"],
        ):
            frame = plot[
                plot["policy_name"].eq(policy) & plot["metric_name"].eq(metric)
            ].set_index("scenario_name").reindex(scenarios)
            errors = np.vstack(
                [
                    frame["mean_value"] - frame["ci95_lower"],
                    frame["ci95_upper"] - frame["mean_value"],
                ]
            )
            axis.bar(
                x + offset,
                frame["mean_value"],
                width,
                yerr=errors,
                capsize=2.5,
                label=DISPLAY[policy],
            )
    labels = ["Session", "Moderate", "Severe"]
    axes[0].set_title("Mean latency across 10 regenerated traces")
    axes[0].set_ylabel("Realized latency (ms)")
    axes[1].set_title("QoS success across 10 regenerated traces")
    axes[1].set_ylabel("Success rate under 60 ms")
    axes[1].set_ylim(0.60, 1.01)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color=OPAQUE_GRID_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.20, wspace=0.28)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/transactions_evidence")
    manifest_mode = parser.add_mutually_exclusive_group()
    manifest_mode.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Refresh evidence_manifest.json from an already-built output tree "
            "without regenerating tables or figures."
        ),
    )
    manifest_mode.add_argument(
        "--verify-manifest",
        action="store_true",
        help=(
            "Verify that the output tree matches evidence_manifest.json "
            "exactly, rejecting missing, extra, size-drifted, or hash-drifted "
            "files."
        ),
    )
    args = parser.parse_args()
    output = _resolve(args.output_dir)
    if args.manifest_only:
        if not output.is_dir():
            raise FileNotFoundError(
                f"cannot refresh a manifest for missing output directory: {output}"
            )
        _write_evidence_manifest(output)
        print(f"transactions_evidence_manifest_refreshed={output}")
        return 0
    if args.verify_manifest:
        if not output.is_dir():
            raise FileNotFoundError(
                f"cannot verify a manifest for missing output directory: {output}"
            )
        verification = _verify_evidence_manifest(output)
        print(
            "transactions_evidence_manifest_verified="
            f"{output}; files={verification['verified_file_count']}; "
            f"manifest_sha256={verification['manifest_sha256']}"
        )
        return 0
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sources = _load_sources()
    summaries = []
    decisions = []
    for source in sources:
        summary, decision = _source_frame(source)
        summaries.append(summary)
        decisions.append(decision)
    summary = pd.concat(summaries, ignore_index=True)
    decision = pd.concat(decisions, ignore_index=True)
    audit = _dataset_audit(sources)
    alignment_audit = _timestamp_alignment_audit(sources)
    exact_horizon_audit = _exact_horizon_audit(sources)
    validation_gate_audit = _validation_gate_audit(sources)
    (
        opportunity_audit,
        opportunity_labels,
        opportunity_results,
        policy_agreement,
        success_gap_bounds,
    ) = _decision_opportunity_evidence(sources, decision)
    delayed_replay = _delayed_replay_evidence(sources, decision)
    thresholds = _threshold_sensitivity(decision)
    threshold_matched = _threshold_matched_policy_sensitivity(sources)
    branch_frequency = _qos_branch_frequency(decision)
    branch_outcomes = _branch_outcomes(decision)
    fallback_comparison = _fallback_comparison(summary)
    operational_metrics = _operational_metrics(summary)
    xai_case_studies = _xai_case_studies(decision)
    control_loop = _control_loop_sensitivity(decision)
    significance = _paired_statistics(decision)
    intervals = _confidence_intervals(decision)
    fixed_metadata = json.loads(
        (
            REPO_ROOT
            / "results/commect_validation_gated_audit/validation_metadata.json"
        ).read_text(encoding="utf-8")
    )
    missing_outcome_envelopes = _missing_outcome_envelopes(
        summary[
            summary["dataset"].eq("COMMECT")
            & summary["evaluation_case"].eq("measured_multi_access")
        ],
        fixed_metadata["split_boundary_audit"],
        pd.read_csv(
            REPO_ROOT
            / "results/commect_validation_gated_rolling/rolling_fold_manifest.csv"
        ),
        pd.read_csv(
            REPO_ROOT
            / "results/commect_validation_gated_rolling/rolling_policy_summary.csv"
        ),
    )
    numerical_audit = _numerical_consistency_audit(
        summary,
        decision,
        opportunity_audit,
        validation_gate_audit,
    )

    summary.to_csv(output / "concurrent_policy_results.csv", index=False)
    decision.to_csv(output / "concurrent_policy_decisions.csv", index=False)
    audit.to_csv(output / "concurrency_audit.csv", index=False)
    alignment_audit.to_csv(output / "timestamp_alignment_audit.csv", index=False)
    exact_horizon_audit.to_csv(output / "exact_horizon_audit.csv", index=False)
    validation_gate_audit.to_csv(output / "validation_gate_selection_audit.csv", index=False)
    opportunity_audit.to_csv(output / "decision_opportunity_audit.csv", index=False)
    opportunity_labels.to_csv(output / "decision_opportunity_labels.csv", index=False)
    opportunity_results.to_csv(
        output / "opportunity_conditioned_policy_results.csv", index=False
    )
    policy_agreement.to_csv(output / "policy_choice_agreement.csv", index=False)
    success_gap_bounds.to_csv(output / "pairwise_success_gap_bounds.csv", index=False)
    delayed_replay.to_csv(output / "delayed_state_replay.csv", index=False)
    thresholds.to_csv(output / "qos_threshold_sensitivity.csv", index=False)
    threshold_matched.to_csv(
        output / "threshold_matched_policy_sensitivity.csv", index=False
    )
    threshold_matched.to_csv(
        output / "frozen_score_threshold_shield_diagnostic.csv", index=False
    )
    branch_frequency.to_csv(output / "qos_branch_frequency.csv", index=False)
    branch_outcomes.to_csv(output / "qos_branch_outcomes.csv", index=False)
    fallback_comparison.to_csv(output / "qos_fallback_comparison.csv", index=False)
    operational_metrics.to_csv(output / "operational_secondary_metrics.csv", index=False)
    missing_outcome_envelopes.to_csv(
        output / "missing_outcome_success_envelopes.csv",
        index=False,
    )
    xai_case_studies.to_csv(output / "xai_case_studies.csv", index=False)
    _training_hyperparameters().to_csv(
        output / "training_hyperparameters.csv", index=False
    )
    _simulator_parameters().to_csv(output / "simulator_parameters.csv", index=False)
    (output / "simulator_model_specification.json").write_text(
        json.dumps(_simulator_model_specification(), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "runtime_environment.json").write_text(
        json.dumps(_runtime_environment(), indent=2) + "\n",
        encoding="utf-8",
    )
    control_loop.to_csv(output / "control_loop_sensitivity.csv", index=False)
    significance.to_csv(output / "paired_block_significance.csv", index=False)
    intervals.to_csv(output / "block_bootstrap_confidence_intervals.csv", index=False)
    numerical_audit.to_csv(output / "numerical_consistency_audit.csv", index=False)
    _plot_primary(summary, figures / "concurrent_primary_results.png")
    _plot_delayed_state_replay(
        delayed_replay,
        figures / "delayed_state_replay.png",
    )
    _plot_decision_opportunity(
        opportunity_audit,
        opportunity_results,
        figures / "decision_opportunity_analysis.png",
    )
    _plot_thresholds(thresholds, figures / "measured_qos_threshold_sensitivity.png")
    _plot_ablation(summary, figures / "component_ablation.png")
    _plot_control_loop(control_loop, figures / "control_loop_sensitivity.png")
    _plot_sensitivity_dashboard(
        thresholds,
        control_loop,
        figures / "qos_and_control_sensitivity.png",
    )
    _plot_method_pipeline(figures / "operational_pipeline.png")
    seed_summary = _plot_multiseed(figures / "multiseed_orbital_robustness.png")
    seed_summary.to_csv(output / "multiseed_orbital_summary.csv", index=False)
    _collect_supporting_evidence(output)
    _write_evidence_manifest(output)
    print(f"transactions_evidence_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
