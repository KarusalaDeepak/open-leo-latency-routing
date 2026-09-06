#!/usr/bin/env python3
"""Run leakage-safe expanding-window validation on COMMECT alternatives."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import assign_decision_groups, load_time_bin_table
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    build_rolling_origin_split_plan,
    build_wall_clock_decision_schedule,
)
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.evaluation.decision_opportunity import (
    build_candidate_opportunity_audit,
    build_opportunity_conditioned_results,
    build_pairwise_success_gap_bounds,
)
from open_leo_latency_routing.evaluation.delayed_execution import replay_delayed_execution
from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar
from open_leo_latency_routing.evaluation.significance import (
    build_paired_policy_significance,
)
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    evaluate_decision_policies,
    summarize_switch_transitions,
)
from open_leo_latency_routing.optimization.risk_control import RiskControlConfig
from scripts.run_service_path_experiments import POLICY_COLUMNS, _fit_models, _make_candidate_frame


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _closed_partition(
    frame: pd.DataFrame,
    epochs: list[int] | None = None,
    multi_bin_horizons: Iterable[int] | None = None,
    *,
    partition_start_epoch: float | None = None,
    partition_end_epoch: float | None = None,
) -> pd.DataFrame:
    """Close all configured future outcomes within one time partition.

    The primary label row is removed when its target epoch crosses the block
    boundary. Multi-bin outcomes remain usable for one-step evaluation, but
    each crossing outcome is marked unavailable and its values are cleared.
    """

    explicit_wall_clock_interval = (
        partition_start_epoch is not None or partition_end_epoch is not None
    )
    if explicit_wall_clock_interval and (
        partition_start_epoch is None or partition_end_epoch is None
    ):
        raise ValueError("both wall-clock partition endpoints are required")
    if not explicit_wall_clock_interval and not epochs:
        return frame.iloc[0:0].copy()
    if multi_bin_horizons is None:
        horizons = tuple(
            sorted(
                int(column.removeprefix("target_available_"))
                for column in frame.columns
                if column.startswith("target_available_")
                and column.removeprefix("target_available_").isdigit()
            )
        )
    else:
        horizons = tuple(
            dict.fromkeys(int(horizon) for horizon in multi_bin_horizons)
        )
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("multi-bin horizons must be positive")

    prepared = frame.copy()
    for horizon in horizons:
        available_column = f"target_available_{horizon}"
        target_columns = (
            f"target_cumulative_{horizon}",
            f"target_mean_{horizon}",
        )
        missing = [
            column
            for column in (available_column, *target_columns)
            if column not in prepared
        ]
        if missing:
            raise ValueError(
                f"configured {horizon}-bin target is missing columns: {missing}"
            )
        endpoint_column = f"target_end_bin_epoch_{horizon}"
        if endpoint_column not in prepared:
            required_endpoint_inputs = {
                "bin_epoch",
                "target_next_bin_epoch",
            }
            if not required_endpoint_inputs.issubset(prepared.columns):
                raise ValueError(
                    f"cannot derive {endpoint_column} without exact one-step "
                    "endpoint metadata"
                )
            if "target_expected_cadence_seconds" in prepared:
                exact_cadence = prepared["target_expected_cadence_seconds"]
            elif "bin_seconds" in prepared:
                exact_cadence = prepared["bin_seconds"]
            else:
                # This runner's primary target is predeclared as one bin.
                exact_cadence = (
                    prepared["target_next_bin_epoch"] - prepared["bin_epoch"]
                )
            prepared[endpoint_column] = (
                prepared["bin_epoch"] + horizon * exact_cadence
            )

    if explicit_wall_clock_interval:
        interval_start = float(partition_start_epoch)
        interval_end = float(partition_end_epoch)
        if interval_end < interval_start:
            raise ValueError("wall-clock partition end precedes its start")
        partition = prepared[
            prepared["bin_epoch"].between(
                interval_start,
                interval_end,
                inclusive="both",
            )
        ].copy()
    else:
        epoch_set = set(epochs or ())
        partition = prepared[
            prepared["session_bin_index"].isin(epoch_set)
        ].copy()
    if partition.empty:
        return partition
    # Close outcomes against the predeclared wall-clock interval, not against
    # the subset of timestamps that also happen to be eligible forecast rows.
    # Requiring endpoint membership in that eligible subset would condition a
    # decision at t on whether the endpoint can itself forecast still later
    # outcomes (for example t + 2H), and would drop otherwise observed labels.
    if explicit_wall_clock_interval:
        frozen_start_epoch = float(partition_start_epoch)
        frozen_end_epoch = float(partition_end_epoch)
    else:
        frozen_start_epoch = float(partition["bin_epoch"].min())
        frozen_end_epoch = float(partition["bin_epoch"].max())
    one_step_closed = partition["target_next_bin_epoch"].between(
        frozen_start_epoch,
        frozen_end_epoch,
        inclusive="both",
    )
    partition = partition[one_step_closed].copy()
    for horizon in horizons:
        available_column = f"target_available_{horizon}"
        target_columns = [
            f"target_cumulative_{horizon}",
            f"target_mean_{horizon}",
        ]
        endpoint_column = f"target_end_bin_epoch_{horizon}"
        horizon_closed = partition[endpoint_column].between(
            frozen_start_epoch,
            frozen_end_epoch,
            inclusive="both",
        )
        horizon_available = (
            partition[available_column].astype(bool) & horizon_closed
        )
        partition.loc[~horizon_available, target_columns] = float("nan")
        partition[available_column] = horizon_available.astype(int)
    return partition


def _aggregate_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, frame in decisions.groupby("policy_name", sort=False):
        switch_metrics = summarize_switch_transitions(frame)
        continuity_reset_count = int(frame["continuity_reset"].sum())
        continuity_segment_count = int(
            frame["continuity_segment_start"].sum()
        )
        rows.append(
            {
                "policy_name": policy,
                "decision_count": int(len(frame)),
                "mean_realized_latency_ms": float(frame["realized_next_latency_ms"].mean()),
                "mean_decision_gap_ms": float(frame["decision_gap_ms"].mean()),
                "success_rate_under_60ms": float(frame["success_under_budget"].mean()),
                "p95_realized_latency_ms": float(frame["realized_next_latency_ms"].quantile(0.95)),
                "cvar95_realized_latency_ms": empirical_upper_cvar(
                    frame["realized_next_latency_ms"].to_numpy(dtype=float),
                    0.95,
                ),
                "retrospective_best_path_match_rate": float(
                    frame["retrospective_best_path_match"].mean()
                ),
                "switch_rate": switch_metrics.switch_rate,
                "switch_count": switch_metrics.switch_count,
                "eligible_switch_transition_count": (
                    switch_metrics.eligible_transition_count
                ),
                # Compatibility alias retained for existing artifact readers.
                # It is the eligible-transition denominator, not switch count.
                "switch_transition_count": (
                    switch_metrics.eligible_transition_count
                ),
                "continuity_reset_count": continuity_reset_count,
                "continuity_segment_count": continuity_segment_count,
                "mean_model_and_ranking_time_us": float(
                    frame["model_and_ranking_time_us"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--trace", default="data/processed/commect_multiaccess_10s.csv")
    parser.add_argument(
        "--output-dir",
        default="results/commect_validation_gated_rolling",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--max-skew-ms",
        type=float,
        default=None,
        help=(
            "Restrict eligible forecast epochs by the recorded cross-path "
            "median-time skew while retaining split boundaries from the "
            "unfiltered wall-clock schedule."
        ),
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    multi_bin_horizons = list(
        dict.fromkeys(
            int(horizon)
            for horizon in config["optimization"].get(
                "multi_bin_horizons", [3, 5]
            )
        )
    )
    all_target_horizons = list(dict.fromkeys((1, *multi_bin_horizons)))
    frame, concurrency = assign_decision_groups(
        load_time_bin_table(_resolve(args.trace)),
        literal_single_controller_steering=True,
    )
    decision_cadence_seconds = float(frame["bin_seconds"].iloc[0])
    decision_schedule = build_wall_clock_decision_schedule(
        frame,
        decision_cadence_seconds=decision_cadence_seconds,
    )
    forecast = build_forecast_table(
        frame,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=decision_cadence_seconds,
        multi_bin_horizons=multi_bin_horizons,
        require_complete_decision_epochs=True,
    )
    exact_horizon_audit = forecast.attrs.get("exact_horizon_audit", {})
    unfiltered_forecast = forecast.copy()
    if args.max_skew_ms is not None:
        if args.max_skew_ms < 0:
            raise ValueError("maximum skew must be nonnegative")
        if "inter_path_skew_ms" not in forecast:
            raise ValueError("trace does not expose inter_path_skew_ms")
        epoch_skew = forecast.groupby("session_bin_index")[
            "inter_path_skew_ms"
        ].max()
        eligible_epochs = set(
            epoch_skew[epoch_skew.le(args.max_skew_ms)].index.astype(int)
        )
        forecast = forecast[
            forecast["session_bin_index"].isin(eligible_epochs)
        ].copy()
        if forecast["session_bin_index"].nunique() < 40:
            raise ValueError("skew restriction leaves too few decision epochs")
    folds = max(2, int(args.folds))
    rolling_split_plan = build_rolling_origin_split_plan(
        decision_schedule,
        fold_count=folds,
        minimum_block_size=10,
    )

    ensemble_cfg = config["optimization"]["ensemble_uncertainty"]
    consensus_cfg = config["optimization"]["consensus"]
    disagreement_cfg = config["optimization"]["disagreement_aware"]
    service_cfg = config["optimization"]["service_risk"]
    risk_cfg = config["optimization"]["risk_control"]
    latency_budget_ms = float(config["optimization"]["latency_budget_ms"])
    switch_penalty = float(config["optimization"]["online_switch_penalty_ms"])
    all_candidates = []
    all_decisions = []
    manifests = []
    all_gate_evidence = []

    for fold_plan in rolling_split_plan["folds"]:
        fold_index = int(fold_plan["rolling_fold"]) - 1
        partition_plan = fold_plan["partitions"]
        test_plan = partition_plan["test"]
        test_interval_start_epoch = float(test_plan["first_bin_epoch"])
        test_interval_end_epoch = float(test_plan["last_bin_epoch"])
        raw_test = frame[
            frame["bin_epoch"].between(
                test_interval_start_epoch,
                test_interval_end_epoch,
                inclusive="both",
            )
        ]
        unfiltered_planned_test = unfiltered_forecast[
            unfiltered_forecast["bin_epoch"].between(
                test_interval_start_epoch,
                test_interval_end_epoch,
                inclusive="both",
            )
        ]
        planned_test = forecast[
            forecast["bin_epoch"].between(
                test_interval_start_epoch,
                test_interval_end_epoch,
                inclusive="both",
            )
        ]
        expected_test_decision_epochs = int(
            planned_test.loc[
                planned_test["target_next_bin_epoch"].between(
                    test_interval_start_epoch,
                    test_interval_end_epoch,
                    inclusive="both",
                ),
                "session_bin_index",
            ].nunique()
        )
        partition_frames = {
            name: _closed_partition(
                forecast,
                multi_bin_horizons=multi_bin_horizons,
                partition_start_epoch=float(interval["first_bin_epoch"]),
                partition_end_epoch=float(interval["last_bin_epoch"]),
            )
            for name, interval in partition_plan.items()
        }
        train = partition_frames["train"]
        calibration = partition_frames["calibration"]
        selection = partition_frames["selection"]
        test = partition_frames["test"]
        if train.empty or calibration.empty or selection.empty or test.empty:
            raise ValueError(f"fold {fold_index + 1} has an empty partition")

        graph_train = add_graph_snapshot_features(train)
        graph_calibration = add_graph_snapshot_features(calibration)
        graph_selection = add_graph_snapshot_features(selection)
        graph_test = add_graph_snapshot_features(test)
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
            latency_budget_ms=latency_budget_ms,
            ensemble_members=int(ensemble_cfg["ensemble_members"]),
            ensemble_row_fraction=float(ensemble_cfg["row_fraction"]),
            ensemble_feature_fraction=float(ensemble_cfg["feature_fraction"]),
            selection_frame=selection,
            graph_selection=graph_selection,
            risk_control_group_column="session_date",
            risk_control_config=RiskControlConfig(
                alpha=float(risk_cfg["familywise_alpha"]),
                noninferiority_margin=float(
                    risk_cfg["noninferiority_margin"]
                ),
                opportunity_noninferiority_margin=float(
                    risk_cfg.get("opportunity_noninferiority_margin", 0.02)
                ),
                minimum_effective_opportunities=float(
                    risk_cfg["minimum_effective_opportunities"]
                ),
                practical_cvar_gain_ms=float(
                    risk_cfg["practical_cvar_gain_ms"]
                ),
                cvar_quantile=float(risk_cfg["cvar_quantile"]),
                block_length=(
                    int(risk_cfg["block_length"])
                    if risk_cfg.get("block_length") is not None
                    else None
                ),
                latency_cap_ms=float(risk_cfg.get("latency_cap_ms", 60_000.0)),
                cvar_grid_points=int(risk_cfg.get("cvar_grid_points", 101)),
                planned_gate_uses=folds,
                gate_use_index=fold_index + 1,
                bootstrap_samples=int(risk_cfg["bootstrap_samples"]),
                random_seed=int(risk_cfg["random_seed"]) + fold_index,
            ),
        )
        gate_evidence = pd.DataFrame(
            json.loads(temporal_calibration.gate_selection_evidence_json)
        )
        gate_evidence.insert(0, "rolling_fold", fold_index + 1)
        all_gate_evidence.append(gate_evidence)
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
            latency_budget_ms=latency_budget_ms,
        )
        _, decisions = evaluate_decision_policies(
            candidates,
            latency_budget_ms=latency_budget_ms,
            policy_columns=POLICY_COLUMNS,
            decision_window_seconds=float(frame["bin_seconds"].iloc[0]),
            online_switch_penalties_ms={"switch_aware_operational_selector": switch_penalty},
        )
        candidates["rolling_fold"] = fold_index + 1
        decisions["rolling_fold"] = fold_index + 1
        all_candidates.append(candidates)
        all_decisions.append(decisions)
        manifest = {
            "rolling_fold": fold_index + 1,
            "boundary_basis": "pre_target_wall_clock_schedule",
            "boundaries_declared_before_target_filtering": True,
            "target_availability_used_for_boundary_derivation": False,
            "train_first_epoch": partition_plan["train"]["first_bin_epoch"],
            "train_last_epoch": partition_plan["train"]["last_bin_epoch"],
            "calibration_first_epoch": partition_plan["calibration"]["first_bin_epoch"],
            "calibration_last_epoch": partition_plan["calibration"]["last_bin_epoch"],
            "selection_first_epoch": partition_plan["selection"]["first_bin_epoch"],
            "selection_last_epoch": partition_plan["selection"]["last_bin_epoch"],
            "test_first_epoch": test_interval_start_epoch,
            "test_last_epoch": test_interval_end_epoch,
            "training_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "selection_rows": int(len(selection)),
            "test_rows": int(len(test)),
            "test_decision_epochs": int(test["bin_epoch"].nunique()),
            "planned_test_decision_epochs": int(
                test_plan["scheduled_decision_epoch_count"]
            ),
            "scheduled_test_decision_epochs": int(
                test_plan["scheduled_decision_epoch_count"]
            ),
            "observed_raw_test_decision_epochs": int(
                raw_test["bin_epoch"].nunique()
            ),
            "scheduled_missing_raw_test_decision_epochs": int(
                test_plan["scheduled_decision_epoch_count"]
                - raw_test["bin_epoch"].nunique()
            ),
            "target_complete_test_decision_epochs_before_boundary": int(
                unfiltered_planned_test["bin_epoch"].nunique()
            ),
            "target_availability_excluded_test_decision_epochs": int(
                raw_test["bin_epoch"].nunique()
                - unfiltered_planned_test["bin_epoch"].nunique()
            ),
            "skew_excluded_test_decision_epochs": int(
                unfiltered_planned_test["bin_epoch"].nunique()
                - planned_test["bin_epoch"].nunique()
            ),
            "expected_boundary_closed_test_decision_epochs": (
                expected_test_decision_epochs
            ),
            "one_step_boundary_excluded_test_decision_epochs": (
                planned_test["bin_epoch"].nunique()
                - expected_test_decision_epochs
            ),
            "schedule_grid_decision_epoch_count": int(
                rolling_split_plan["schedule"]["scheduled_decision_epoch_count"]
            ),
            "schedule_grid_observed_raw_decision_epoch_count": int(
                rolling_split_plan["schedule"]["observed_raw_decision_epoch_count"]
            ),
            "schedule_grid_missing_raw_decision_epoch_count": int(
                rolling_split_plan["schedule"]["missing_raw_decision_epoch_count"]
            ),
            "rolling_block_size_scheduled_epochs": int(
                rolling_split_plan["block_size_scheduled_epochs"]
            ),
            "gate_selection_reason": temporal_calibration.gate_selection_reason,
            "gate_selected_policy": (
                temporal_calibration.validation_gated_fallback_policy
            ),
            "gate_opportunity_count": temporal_calibration.gate_opportunity_count,
            "gate_effective_opportunity_count": (
                temporal_calibration.gate_effective_opportunity_count
            ),
            "closed_target_horizons_bins": "|".join(
                str(horizon) for horizon in all_target_horizons
            ),
            "maximum_inter_path_skew_ms": args.max_skew_ms,
        }
        for partition_name, interval in partition_plan.items():
            manifest[f"{partition_name}_first_schedule_index"] = int(
                interval["first_schedule_index"]
            )
            manifest[f"{partition_name}_last_schedule_index"] = int(
                interval["last_schedule_index"]
            )
            manifest[f"{partition_name}_scheduled_decision_epochs"] = int(
                interval["scheduled_decision_epoch_count"]
            )
        for partition_name, partition_frame in (
            ("train", train),
            ("calibration", calibration),
            ("selection", selection),
            ("test", test),
        ):
            for horizon in multi_bin_horizons:
                manifest[
                    f"{partition_name}_target_available_{horizon}_rows"
                ] = int(partition_frame[f"target_available_{horizon}"].sum())
        manifests.append(manifest)

    candidates = pd.concat(all_candidates, ignore_index=True)
    decisions = pd.concat(all_decisions, ignore_index=True)
    summary = _aggregate_decisions(decisions)
    opportunity_audit, opportunity_labels = build_candidate_opportunity_audit(candidates)
    opportunity_results = build_opportunity_conditioned_results(
        decisions,
        opportunity_labels,
    )
    significance = build_paired_policy_significance(
        decisions,
        comparisons=[
            ("shield_vs_reactive", "qos_shielded_operational_selector", "reactive_greedy"),
            ("shield_vs_context", "qos_shielded_operational_selector", "predictive_graph_greedy"),
            ("shield_vs_ensemble", "qos_shielded_operational_selector", "ensemble_uncertainty_selector"),
            ("gated_vs_reactive", "validation_gated_qos_selector", "reactive_greedy"),
            ("gated_vs_shield", "validation_gated_qos_selector", "qos_shielded_operational_selector"),
        ],
        metric_columns=["realized_next_latency_ms", "success_under_budget"],
        block_length=max(2, round(len(decisions["session_bin_index"].unique()) ** (1 / 3))),
        segment_columns=("rolling_fold", "continuity_segment_id"),
    )
    success_gap_bounds = build_pairwise_success_gap_bounds(
        decisions,
        opportunity_labels,
    )
    delayed_replay, _ = replay_delayed_execution(
        candidates,
        decisions[
            decisions["policy_name"].isin(
                [
                    "reactive_greedy",
                    "qos_shielded_operational_selector",
                    "validation_gated_qos_selector",
                ]
            )
        ],
        latency_budget_ms=latency_budget_ms,
        delay_bins=(0, 1, 2, 3),
        decision_cadence_seconds=float(frame["bin_seconds"].iloc[0]),
    )
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / "rolling_candidate_predictions.csv", index=False)
    decisions.to_csv(output / "rolling_policy_decisions.csv", index=False)
    summary.to_csv(output / "rolling_policy_summary.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "rolling_fold_manifest.csv", index=False)
    pd.concat(all_gate_evidence, ignore_index=True).to_csv(
        output / "rolling_gate_selection_evidence.csv", index=False
    )
    opportunity_audit.to_csv(output / "rolling_opportunity_audit.csv", index=False)
    opportunity_results.to_csv(
        output / "rolling_opportunity_conditioned_results.csv", index=False
    )
    significance.to_csv(output / "rolling_policy_significance.csv", index=False)
    success_gap_bounds.to_csv(output / "rolling_success_gap_bounds.csv", index=False)
    delayed_replay.to_csv(output / "rolling_delayed_state_replay.csv", index=False)
    metadata = {
        "dataset": "COMMECT",
        "protocol": (
            "primary measured prequential expanding-window evaluation with "
            "within-fold disjoint train, calibration, policy-selection, and "
            "test intervals frozen on the unfiltered raw exact-cadence grid "
            "before target availability filtering"
        ),
        "fold_count": folds,
        "primary_measured_protocol": True,
        "within_fold_intervals_disjoint": True,
        "test_epoch_overlap": False,
        "same_fold_test_outcomes_used_for_admission": False,
        "boundaries_declared_before_target_filtering": True,
        "target_availability_used_for_boundary_derivation": False,
        "maximum_inter_path_skew_ms": args.max_skew_ms,
        "skew_filter_semantics": (
            "the maximum cross-path median-time skew is evaluated per "
            "forecast epoch after split boundaries are frozen on the "
            "unfiltered wall-clock schedule; retained epochs keep every "
            "concurrently observed candidate row"
        ),
        "rolling_split_boundary_plan": rolling_split_plan,
        "cross_fold_history_reuse": (
            "after a test block is scored, its outcomes may enter the history "
            "of later folds; no current or future test outcome is used before "
            "that fold is scored"
        ),
        "future_target_boundary_guard": True,
        "future_target_boundary_guard_horizons_bins": all_target_horizons,
        "descriptive_bootstrap_protocol": {
            "method": "segment_stratified_circular_moving_block",
            "segment_columns": [
                "rolling_fold",
                "continuity_segment_id",
            ],
            "fixed_segment_sample_sizes": True,
            "within_segment_circular_wrap_only": True,
            "metric_missingness_starts_new_segment": True,
            "boundary_semantics": (
                "blocks cannot cross rolling-fold, telemetry-gap, or "
                "session/campaign continuity boundaries"
            ),
        },
        "exact_horizon_audit": exact_horizon_audit,
        "total_out_of_sample_decision_epochs": int(
            decisions["session_bin_index"].nunique()
        ),
        "concurrency_audit": concurrency,
        "claim_boundary": (
            "primary measured protocol; pooled estimates aggregate disjoint "
            "out-of-sample test blocks without claiming independent folds"
        ),
        "validation_gated_fallback_counts": {
            str(key): int(value)
            for key, value in candidates.drop_duplicates("rolling_fold")[
                "validation_gated_fallback_policy"
            ].value_counts().items()
        },
    }
    (output / "rolling_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"rolling_origin_validation_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
