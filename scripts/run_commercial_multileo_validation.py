#!/usr/bin/env python3
"""Evaluate policies on an audited commercial Starlink--OneWeb trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from open_leo_latency_routing.config import load_config  # noqa: E402
from open_leo_latency_routing.data.loaders import (  # noqa: E402
    assign_decision_groups,
    load_time_bin_table,
)
from open_leo_latency_routing.features.temporal import (  # noqa: E402
    build_forecast_table,
    split_train_calibration_selection_test,
)
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features  # noqa: E402
from open_leo_latency_routing.optimization.policies import (  # noqa: E402
    ConsensusPolicyConfig,
    evaluate_decision_policies,
)
from open_leo_latency_routing.optimization.risk_control import (  # noqa: E402
    RiskControlConfig,
)
from scripts.run_service_path_experiments import (  # noqa: E402
    POLICY_COLUMNS,
    _fit_models,
    _make_candidate_frame,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_validation_scope(
    trace_metadata: dict[str, object],
    *,
    allow_scoped_paired_replay: bool,
) -> bool:
    """Return literal-selectability status or reject an unscoped replay."""

    if not trace_metadata.get("all_candidate_outcomes_observed", False):
        raise ValueError("trace failed the all-candidate concurrent-outcome audit")
    operators = trace_metadata.get("operators", [])
    if not isinstance(operators, (list, tuple, set)) or set(operators) != {
        "starlink",
        "oneweb",
    }:
        raise ValueError("trace metadata does not identify Starlink and OneWeb")
    same_controller_evidence = bool(
        trace_metadata.get("same_controller_selectable_path_evidence", False)
    )
    if not same_controller_evidence and not allow_scoped_paired_replay:
        raise ValueError(
            "trace does not establish co-located Starlink and OneWeb paths "
            "selectable by one controller; pass --allow-scoped-paired-replay "
            "only for a restricted offline paired comparison"
        )
    return same_controller_evidence


def _resolve_campaign_gate_grouping(
    trace_metadata: dict[str, object],
    forecast: pd.DataFrame,
) -> tuple[str | None, dict[str, object]]:
    """Use campaign IDs only after a complete explicit independence audit."""

    audited_ids_raw = trace_metadata.get("audited_campaign_ids", [])
    audited_ids = (
        {str(value) for value in audited_ids_raw}
        if isinstance(audited_ids_raw, list)
        else set()
    )
    declared_pass = bool(
        trace_metadata.get("independent_campaign_grouping_pass", False)
        and trace_metadata.get("mapped_campaign_id_pass", False)
        and trace_metadata.get("independent_campaign_ids_asserted", False)
        and str(trace_metadata.get("campaign_independence_note", "")).strip()
        and len(audited_ids) >= 2
    )
    observed_ids: set[str] = set()
    frame_complete = False
    if "campaign_id" in forecast:
        campaign_values = forecast["campaign_id"].astype("string").str.strip()
        frame_complete = bool(campaign_values.notna().all() and campaign_values.ne("").all())
        observed_ids = set(campaign_values.dropna().astype(str).unique())
    observed_ids_audited = bool(
        frame_complete
        and len(observed_ids) >= 2
        and observed_ids.issubset(audited_ids)
    )
    grouping_pass = bool(declared_pass and observed_ids_audited)
    if grouping_pass:
        reason = (
            "mapped campaign_id values passed complete paired-ID and documented "
            "independence audits"
        )
        group_column = "campaign_id"
    else:
        reason = (
            "fail closed: the complete imported trace is one gate inference "
            "group because independent campaign IDs were not fully audited"
        )
        group_column = None
    return group_column, {
        "declared_campaign_independence_pass": declared_pass,
        "forecast_campaign_id_complete": frame_complete,
        "audited_campaign_ids": sorted(audited_ids),
        "forecast_campaign_ids": sorted(observed_ids),
        "forecast_campaign_ids_audited": observed_ids_audited,
        "risk_control_group_column": group_column,
        "grouping_reason": reason,
    }


def _four_way_split_audit(
    partitions: dict[str, pd.DataFrame],
    source_forecast: pd.DataFrame,
    split_ratios: dict[str, float],
) -> dict[str, object]:
    """Describe global chronology, disjointness, and one-step target closure."""

    manifest: dict[str, dict[str, object]] = {}
    epoch_sets: dict[str, set[float]] = {}
    chronological_bounds: list[tuple[float, float]] = []
    target_closed = True
    multi_bin_target_closed = True
    source_epochs = sorted(
        pd.to_numeric(source_forecast["bin_epoch"], errors="raise")
        .astype(float)
        .unique()
        .tolist()
    )
    source_size = len(source_epochs)
    if source_size < 4:
        raise ValueError("split audit requires at least four source epochs")
    train_ratio = float(split_ratios["train"])
    calibration_ratio = float(split_ratios["calibration"])
    selection_ratio = float(split_ratios["selection"])
    test_ratio = float(split_ratios["test"])
    if round(
        train_ratio + calibration_ratio + selection_ratio + test_ratio,
        6,
    ) != 1.0:
        raise ValueError("split audit ratios must sum to one")
    train_end = min(max(1, int(source_size * train_ratio)), source_size - 3)
    calibration_end = min(
        max(
            train_end + 1,
            int(source_size * (train_ratio + calibration_ratio)),
        ),
        source_size - 2,
    )
    selection_end = min(
        max(
            calibration_end + 1,
            int(
                source_size
                * (train_ratio + calibration_ratio + selection_ratio)
            ),
        ),
        source_size - 1,
    )
    split_upper_bounds = {
        "train": source_epochs[train_end - 1],
        "calibration": source_epochs[calibration_end - 1],
        "selection": source_epochs[selection_end - 1],
        "test": None,
    }

    def _source_partition(values: pd.Series) -> pd.Series:
        return pd.Series(
            pd.cut(
                pd.to_numeric(values, errors="coerce"),
                bins=[
                    float("-inf"),
                    split_upper_bounds["train"],
                    split_upper_bounds["calibration"],
                    split_upper_bounds["selection"],
                    float("inf"),
                ],
                labels=["train", "calibration", "selection", "test"],
                include_lowest=True,
            ).astype("string"),
            index=values.index,
        )

    horizon_numbers = sorted(
        int(column.removeprefix("target_available_"))
        for column in source_forecast.columns
        if column.startswith("target_available_")
        and column.removeprefix("target_available_").isdigit()
    )

    for name in ("train", "calibration", "selection", "test"):
        frame = partitions[name]
        epochs = set(pd.to_numeric(frame["bin_epoch"], errors="raise").astype(float))
        epoch_sets[name] = epochs
        minimum_epoch = float(min(epochs)) if epochs else None
        maximum_epoch = float(max(epochs)) if epochs else None
        if minimum_epoch is not None and maximum_epoch is not None:
            chronological_bounds.append((minimum_epoch, maximum_epoch))
        target_boundary_closed = True
        if "target_next_bin_epoch" in frame and not frame.empty:
            targets = pd.to_numeric(frame["target_next_bin_epoch"], errors="coerce")
            target_boundary_closed = bool(
                targets.notna().all()
                and _source_partition(targets).eq(name).all()
            )
        target_closed = target_closed and target_boundary_closed
        horizon_audit: dict[str, dict[str, object]] = {}
        for horizon in horizon_numbers:
            available_column = f"target_available_{horizon}"
            endpoint_column = f"target_end_bin_epoch_{horizon}"
            target_columns = [
                f"target_cumulative_{horizon}",
                f"target_mean_{horizon}",
            ]
            if endpoint_column not in frame or available_column not in frame:
                horizon_closed = False
                crossing_targets_cleared = False
                available_count = 0
            else:
                endpoints = pd.to_numeric(
                    frame[endpoint_column],
                    errors="coerce",
                )
                endpoint_partition = _source_partition(endpoints)
                available = frame[available_column].fillna(0).astype(bool)
                horizon_closed = bool(
                    endpoints.loc[available].notna().all()
                    and endpoint_partition.loc[available].eq(name).all()
                )
                crossing = endpoints.notna() & ~endpoint_partition.eq(name)
                crossing_targets_cleared = bool((~available.loc[crossing]).all())
                for target_column in target_columns:
                    if target_column not in frame:
                        crossing_targets_cleared = False
                    elif crossing.any():
                        crossing_targets_cleared = bool(
                            crossing_targets_cleared
                            and frame.loc[crossing, target_column].isna().all()
                        )
                available_count = int(available.sum())
            multi_bin_target_closed = bool(
                multi_bin_target_closed
                and horizon_closed
                and crossing_targets_cleared
            )
            horizon_audit[str(horizon)] = {
                "available_row_count": available_count,
                "available_targets_within_partition": horizon_closed,
                "crossing_targets_unavailable_and_cleared": (
                    crossing_targets_cleared
                ),
            }
        manifest[name] = {
            "row_count": int(len(frame)),
            "decision_epoch_count": int(len(epochs)),
            "minimum_bin_epoch": minimum_epoch,
            "maximum_bin_epoch": maximum_epoch,
            "target_boundary_closed": target_boundary_closed,
            "multi_bin_target_boundary_audit": horizon_audit,
            "campaign_ids": (
                sorted(frame["campaign_id"].dropna().astype(str).unique().tolist())
                if "campaign_id" in frame
                else []
            ),
        }
    names = list(epoch_sets)
    pairwise_disjoint = all(
        epoch_sets[names[left]].isdisjoint(epoch_sets[names[right]])
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )
    all_partitions_nonempty = len(chronological_bounds) == 4
    chronology_pass = all_partitions_nonempty and all(
        chronological_bounds[index][1] < chronological_bounds[index + 1][0]
        for index in range(len(chronological_bounds) - 1)
    )
    return {
        "partition_manifest": manifest,
        "source_forecast_epoch_count": source_size,
        "split_ratios": {
            "train": train_ratio,
            "calibration": calibration_ratio,
            "selection": selection_ratio,
            "test": test_ratio,
        },
        "split_upper_bin_epochs": split_upper_bounds,
        "all_partitions_nonempty": all_partitions_nonempty,
        "global_wall_clock_partitioning": True,
        "pairwise_epoch_disjoint": pairwise_disjoint,
        "strict_chronological_order": chronology_pass,
        "one_step_target_boundary_closed": target_closed,
        "multi_bin_target_boundaries_closed": multi_bin_target_closed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--trace", default="data/processed/commercial_multileo_10s.csv")
    parser.add_argument("--output-dir", default="results/commercial_multileo_validation")
    parser.add_argument(
        "--allow-scoped-paired-replay",
        action="store_true",
        help=(
            "Permit an explicitly restricted offline comparison when the trace "
            "does not establish co-located paths under one controller."
        ),
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    trace_path = _resolve(args.trace)
    trace_metadata = json.loads(
        trace_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    same_controller_evidence = _require_validation_scope(
        trace_metadata,
        allow_scoped_paired_replay=args.allow_scoped_paired_replay,
    )

    frame, concurrency = assign_decision_groups(
        load_time_bin_table(trace_path),
        literal_single_controller_steering=same_controller_evidence,
    )
    forecast = build_forecast_table(
        frame,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=float(frame["bin_seconds"].iloc[0]),
        require_complete_decision_epochs=True,
    )
    split_cfg = config["forecasting"]["policy_evaluation_ratios"]
    train, calibration, selection, test = split_train_calibration_selection_test(
        forecast,
        train_ratio=float(split_cfg["train"]),
        calibration_ratio=float(split_cfg["calibration"]),
        selection_ratio=float(split_cfg["selection"]),
        test_ratio=float(split_cfg["test"]),
    )
    split_audit = _four_way_split_audit(
        {
            "train": train,
            "calibration": calibration,
            "selection": selection,
            "test": test,
        },
        forecast,
        {
            "train": float(split_cfg["train"]),
            "calibration": float(split_cfg["calibration"]),
            "selection": float(split_cfg["selection"]),
            "test": float(split_cfg["test"]),
        },
    )
    required_split_guards = (
        "all_partitions_nonempty",
        "pairwise_epoch_disjoint",
        "strict_chronological_order",
        "one_step_target_boundary_closed",
        "multi_bin_target_boundaries_closed",
    )
    if not all(bool(split_audit[guard]) for guard in required_split_guards):
        raise ValueError(
            "commercial four-way split failed chronology or future-target "
            f"closure audit: {split_audit}"
        )
    graph_train = add_graph_snapshot_features(train)
    graph_calibration = add_graph_snapshot_features(calibration)
    graph_selection = add_graph_snapshot_features(selection)
    graph_test = add_graph_snapshot_features(test)
    ensemble_cfg = config["optimization"]["ensemble_uncertainty"]
    risk_cfg = config["optimization"]["risk_control"]
    gate_group_column, campaign_independence_audit = (
        _resolve_campaign_gate_grouping(trace_metadata, forecast)
    )
    (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_features,
        graph_features,
        historical_latency,
        historical_fallback,
        temporal_calibration,
        graph_calibration,
    ) = _fit_models(
        train,
        calibration,
        graph_train,
        graph_calibration,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        ensemble_members=int(ensemble_cfg["ensemble_members"]),
        ensemble_row_fraction=float(ensemble_cfg["row_fraction"]),
        ensemble_feature_fraction=float(ensemble_cfg["feature_fraction"]),
        selection_frame=selection,
        graph_selection=graph_selection,
        risk_control_group_column=gate_group_column,
        risk_control_config=RiskControlConfig(
            alpha=float(risk_cfg["familywise_alpha"]),
            noninferiority_margin=float(risk_cfg["noninferiority_margin"]),
            opportunity_noninferiority_margin=float(
                risk_cfg.get("opportunity_noninferiority_margin", 0.02)
            ),
            minimum_effective_opportunities=float(
                risk_cfg["minimum_effective_opportunities"]
            ),
            practical_cvar_gain_ms=float(risk_cfg["practical_cvar_gain_ms"]),
            cvar_quantile=float(risk_cfg["cvar_quantile"]),
            # Without an explicitly audited campaign_id column, None makes the
            # risk-control implementation treat the full selection interval as
            # one collection. Arbitrary time blocks are never promoted to
            # independent replications in this runner.
            block_length=None,
            latency_cap_ms=float(risk_cfg.get("latency_cap_ms", 60_000.0)),
            cvar_grid_points=int(risk_cfg.get("cvar_grid_points", 101)),
            planned_gate_uses=int(risk_cfg.get("planned_gate_uses", 1)),
            gate_use_index=int(risk_cfg.get("gate_use_index", 1)),
            bootstrap_samples=int(risk_cfg["bootstrap_samples"]),
            random_seed=int(risk_cfg["random_seed"]),
        ),
    )
    gate_evidence_records = json.loads(
        temporal_calibration.gate_selection_evidence_json
    )
    if gate_group_column is None:
        evidence_block_counts = {
            int(record.get("inference_block_count", -1))
            for record in gate_evidence_records
        }
        if temporal_calibration.validation_gated_fallback_policy != "reactive":
            raise RuntimeError(
                "unaudited commercial campaign unexpectedly admitted a learned "
                "policy instead of failing closed to reactive"
            )
        if not evidence_block_counts.issubset({0, 1}):
            raise RuntimeError(
                "unaudited commercial campaign was split into pseudo-independent "
                f"gate blocks: {sorted(evidence_block_counts)}"
            )
    consensus_cfg = config["optimization"]["consensus"]
    disagreement_cfg = config["optimization"]["disagreement_aware"]
    service_cfg = config["optimization"]["service_risk"]
    candidates = _make_candidate_frame(
        test,
        graph_test,
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_features,
        graph_features,
        historical_latency,
        historical_fallback,
        temporal_calibration,
        graph_calibration,
        consensus_config=ConsensusPolicyConfig(
            temporal_weight=float(consensus_cfg["temporal_weight"]),
            graph_weight=float(consensus_cfg["graph_weight"]),
            disagreement_penalty=float(consensus_cfg["disagreement_penalty"]),
        ),
        disagreement_temporal_weight=0.5,
        disagreement_graph_weight=0.5,
        disagreement_penalty=float(disagreement_cfg["uncertainty_multiplier"]),
        ensemble_penalty=float(ensemble_cfg["lambda_ens"]),
        service_risk_reply_weight=float(service_cfg["reply_pressure_penalty"]),
        service_risk_volatility_weight=float(service_cfg["volatility_penalty"]),
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
    )
    if "campaign_id" in test:
        candidates["campaign_id"] = (
            test.reset_index(drop=True)["campaign_id"].astype("string")
        )
    summary, decisions = evaluate_decision_policies(
        candidates,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        policy_columns=POLICY_COLUMNS,
        decision_window_seconds=float(frame["bin_seconds"].iloc[0]),
        online_switch_penalties_ms={
            "switch_aware_operational_selector": float(
                config["optimization"]["online_switch_penalty_ms"]
            )
        },
    )
    if "campaign_id" in test:
        campaign_by_epoch = (
            test.groupby("bin_epoch", sort=False)["campaign_id"].first()
        )
        decisions["campaign_id"] = decisions["decision_bin_epoch"].map(
            campaign_by_epoch
        )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    decisions.to_csv(output_dir / "policy_decisions.csv", index=False)
    candidates.to_csv(output_dir / "candidate_predictions.csv", index=False)
    gate_evidence = pd.DataFrame(gate_evidence_records)
    gate_evidence.to_csv(output_dir / "gate_selection_evidence.csv", index=False)
    if gate_group_column is None:
        gate_independence_group_count = 1 if not selection.empty else 0
        gate_inference_unit = (
            "one complete imported campaign; time bins, dates, files, and "
            "segments are dependent observations within that campaign"
        )
    else:
        gate_independence_group_count = int(
            selection[gate_group_column].dropna().nunique()
        )
        gate_inference_unit = (
            "explicit campaign_id values with complete paired mapping and "
            "documented cross-campaign independence"
        )
    gate_estimand_metadata = {
        key: gate_evidence_records[0].get(key)
        for key in (
            "gate_estimand_population",
            "success_estimand_population",
            "aggregate_success_estimand_population",
            "opportunity_conditioned_success_estimand_population",
            "opportunity_conditioning",
            "cvar_estimand_population",
            "group_population_weighting",
            "within_group_population_weighting",
            "emergency_epoch_treatment",
            "inference_unit_source",
            "inference_block_count",
            "effective_block_count",
            "effective_opportunity_count",
            "opportunity_conditioned_inference_group_count",
            "success_gate_requires_both_endpoints",
        )
        if gate_evidence_records and key in gate_evidence_records[0]
    }
    claim_safe_concurrency = {
        **concurrency,
        "generic_timestamp_concurrency_detected": bool(
            concurrency["has_temporally_concurrent_candidates"]
        ),
        "supports_candidate_outcome_shadow_replay": True,
        "supports_literal_single_controller_steering": same_controller_evidence,
        "supports_single_controller_shadow_replay": same_controller_evidence,
        "supports_closed_loop_deployment_evidence": False,
        "topology_gate_applied": True,
    }
    validation_metadata = {
        "policy_level_evaluation": same_controller_evidence,
        "offline_paired_replay": True,
        "measured_concurrent_paths": same_controller_evidence,
        "time_aligned_dual_operator_measurements": True,
        "same_controller_selectable_path_evidence": same_controller_evidence,
        "evaluation_scope": trace_metadata.get(
            "evidence_scope",
            "scoped_location_unverified_time_aligned_comparison",
        ),
        "commercial_multileo": True,
        "independent_of_lens": trace_metadata["is_independent_of_lens"],
        "long_duration_pass": trace_metadata["long_duration_pass"],
        "closes_independent_longitudinal_multileo_limitation": bool(
            trace_metadata.get(
                "closes_independent_longitudinal_multileo_limitation",
                False,
            )
            and same_controller_evidence
        ),
        "temporal_concurrency_audit": concurrency,
        "concurrency_audit": claim_safe_concurrency,
        "trace_metadata": trace_metadata,
        "split_protocol": (
            "global shared-wall-clock train/calibration/policy-selection/test "
            "partition with exact future-target boundary closure"
        ),
        "split_audit": split_audit,
        "training_row_count": int(len(train)),
        "calibration_row_count": int(len(calibration)),
        "policy_selection_row_count": int(len(selection)),
        "evaluation_row_count": int(len(test)),
        "exact_horizon_audit": forecast.attrs.get("exact_horizon_audit", {}),
        "campaign_independence_audit": campaign_independence_audit,
        "gate_inference_unit": gate_inference_unit,
        "gate_independence_group_count": gate_independence_group_count,
        "gate_inference_fail_closed_without_audited_campaign_ids": True,
        "gate_selected_policy": (
            temporal_calibration.validation_gated_fallback_policy
        ),
        "gate_evidence_inference_block_counts": sorted(
            {
                int(record.get("inference_block_count", -1))
                for record in gate_evidence_records
            }
        ),
        "unaudited_campaign_reactive_guard_pass": bool(
            gate_group_column is not None
            or temporal_calibration.validation_gated_fallback_policy == "reactive"
        ),
        "gate_estimand": gate_estimand_metadata,
        "gate_selection_reason": temporal_calibration.gate_selection_reason,
        "claim_restriction": trace_metadata.get(
            "claim_restriction",
            "do not claim same-controller commercial multi-LEO validation",
        ),
        "valid_claim": trace_metadata.get("valid_claim", "scoped paired replay"),
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(validation_metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"validation_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
