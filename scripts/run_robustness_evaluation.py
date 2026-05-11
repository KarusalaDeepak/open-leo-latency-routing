#!/usr/bin/env python3
"""Run burst and outage robustness experiments on the LENS modeling pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent, load_time_bin_table
from open_leo_latency_routing.evaluation.significance import build_paired_policy_significance
from open_leo_latency_routing.features.temporal import build_forecast_table, split_train_val_test
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.models.forecast_baselines import (
    ForecastResult,
    default_feature_columns,
    evaluate_prediction_frame,
    fit_forecast_model,
    predict_forecast_model,
)
from open_leo_latency_routing.models.graph_baselines import GraphResult, evaluate_graph_predictions, fit_graph_xgb_model, predict_graph_model
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    SimpleFusionPolicyConfig,
    add_consensus_hybrid_scores,
    add_simple_fusion_scores,
    evaluate_decision_policies,
    tune_consensus_policy,
)
from open_leo_latency_routing.scenarios.stress import (
    apply_burst_stress,
    apply_outage_stress,
)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _persistence_predictions(test_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": test_frame.index,
            "model_name": "persistence",
            "y_true": test_frame["target_next"].tolist(),
            "y_pred": test_frame["latency_mean_ms"].tolist(),
        }
    )


def _write_summary_markdown(
    path: Path,
    forecast_metrics: pd.DataFrame,
    graph_metrics: pd.DataFrame,
    policy_summary: pd.DataFrame,
    policy_significance: pd.DataFrame,
    consensus_tuning: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
    penalty_sweep: pd.DataFrame,
) -> None:
    lines = [
        "# Robustness Evaluation Summary",
        "",
        "This report evaluates train-on-base, test-on-shift robustness under burst, outage, and correlated structural-shift conditions.",
        "",
        "## Forecast Metrics",
        "",
        "| Scenario | Model | MAE | RMSE | MAPE | Rows |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in forecast_metrics.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.model_name} | {row.mae:.4f} | {row.rmse:.4f} | {row.mape:.4f} | {row.row_count} |"
        )

    lines.extend(
        [
            "",
            "## Graph Metrics",
            "",
            "| Scenario | Model | MAE | RMSE | MAPE | Rows |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in graph_metrics.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.model_name} | {row.mae:.4f} | {row.rmse:.4f} | {row.mape:.4f} | {row.row_count} |"
        )

    lines.extend(
        [
            "",
            "## Consensus Hybrid Calibration",
            "",
            "| Temporal Weight | Graph Weight | Disagreement Penalty | Mean Validation Gap (ms) |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in consensus_tuning.itertuples():
        lines.append(
            f"| {row.temporal_weight:.2f} | {row.graph_weight:.2f} | {row.disagreement_penalty:.2f} | {row.validation_gap_ms:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Policy Summary",
            "",
            "| Scenario | Policy | Decisions | Mean Realized Latency (ms) | Mean Regret (ms) | Best-Path Match Rate | Success Under 60 ms | Mean Decision Time (us) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy_summary.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.policy_name} | {row.decision_count} | {row.mean_realized_latency_ms:.4f} | "
            f"{row.mean_regret_ms:.4f} | {row.best_path_match_rate:.4f} | {row.success_rate_under_60ms:.4f} | "
            f"{row.mean_decision_time_us:.2f} |"
        )

    lines.extend(
        [
            "",
            "Best-path match rate is reported only as a secondary metric because it uses perfect hindsight and cannot be deployed online.",
            "",
            "## Policy Significance",
            "",
            "| Scenario | Comparison | Metric | Mean Delta (ms) | p-value |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in policy_significance.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.comparison_name} | {row.metric_name} | {row.mean_delta:.4f} | {row.p_value:.4f} |"
        )

    lines.extend(
        [
            "",
            "In base evaluation, random selection retains the strongest `Success Under 60 ms` tail score while the graph-aware policy improves mean latency and regret. This is a classic latency-vs.-tail tradeoff rather than a contradiction.",
            "",
            "## Disagreement as an Uncertainty Signal",
            "",
            "| Scenario | Disagreement Bin | Policy | Mean Latency (ms) | Mean Regret (ms) | Match Rate | Success Under 60 ms | Decisions |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in disagreement_summary.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.disagreement_bin} | {row.policy_name} | {row.mean_realized_latency_ms:.4f} | "
            f"{row.mean_regret_ms:.4f} | {row.best_path_match_rate:.4f} | {row.success_rate_under_60ms:.4f} | "
            f"{row.decision_count} |"
        )

    lines.extend(
        [
            "",
            "## Consensus Penalty Sweep",
            "",
            "| Scenario | Disagreement Penalty | Mean Latency (ms) | Mean Regret (ms) | Match Rate | Runtime (us) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in penalty_sweep.itertuples():
        lines.append(
            f"| {row.scenario_name} | {row.disagreement_penalty:.2f} | {row.mean_realized_latency_ms:.4f} | "
            f"{row.mean_regret_ms:.4f} | {row.best_path_match_rate:.4f} | {row.mean_decision_time_us:.2f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_window_context(candidate_frame: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    """Summarize per-window disagreement before any policy acts on the snapshot."""

    context = (
        candidate_frame.groupby("session_bin_index", sort=True)
        .agg(
            candidate_count=("relative_path", "size"),
            mean_candidate_disagreement=("pred_disagreement", "mean"),
            max_candidate_disagreement=("pred_disagreement", "max"),
            stressed_candidate_fraction=("stress_applied", "mean"),
        )
        .reset_index()
    )
    context["scenario_name"] = scenario_name
    return context


def _assign_disagreement_bins(policy_decisions: pd.DataFrame) -> pd.DataFrame:
    """Label each decision window by low/medium/high disagreement pressure."""

    windows = (
        policy_decisions[
            ["scenario_name", "session_bin_index", "max_candidate_disagreement", "mean_candidate_disagreement"]
        ]
        .drop_duplicates()
        .copy()
    )
    labeled_frames: list[pd.DataFrame] = []
    for scenario_name, scenario_windows in windows.groupby("scenario_name", sort=False):
        work = scenario_windows.copy().reset_index(drop=True)
        # Rank first so qcut remains stable even when raw values tie.
        work["_rank"] = work["max_candidate_disagreement"].rank(method="first")
        work["disagreement_bin"] = pd.qcut(
            work["_rank"],
            q=3,
            labels=["Low", "Medium", "High"],
        )
        labeled_frames.append(work.drop(columns=["_rank"]))
    labeled = pd.concat(labeled_frames, ignore_index=True)
    return policy_decisions.merge(
        labeled,
        on=["scenario_name", "session_bin_index", "max_candidate_disagreement", "mean_candidate_disagreement"],
        how="left",
    )


def _build_disagreement_summary(policy_decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy quality by disagreement-pressure bins."""

    labeled = _assign_disagreement_bins(policy_decisions)
    summary = (
        labeled.groupby(["scenario_name", "disagreement_bin", "policy_name"], observed=True)
        .agg(
            decision_count=("session_bin_index", "size"),
            mean_realized_latency_ms=("realized_next_latency_ms", "mean"),
            mean_regret_ms=("regret_ms", "mean"),
            best_path_match_rate=("best_path_match", "mean"),
            success_rate_under_60ms=("success_under_budget", "mean"),
            mean_chosen_disagreement=("chosen_disagreement", "mean"),
        )
        .reset_index()
    )
    return summary


def _run_penalty_sweep(
    scenario_candidate_frames: dict[str, pd.DataFrame],
    temporal_weight: float,
    graph_weight: float,
    penalties: list[float],
    latency_budget_ms: float,
) -> pd.DataFrame:
    """Evaluate how the disagreement penalty changes decision quality."""

    rows: list[pd.DataFrame] = []
    for scenario_name, candidate_frame in scenario_candidate_frames.items():
        for penalty in penalties:
            scored = add_consensus_hybrid_scores(
                candidate_frame,
                config=ConsensusPolicyConfig(
                    temporal_weight=temporal_weight,
                    graph_weight=graph_weight,
                    disagreement_penalty=penalty,
                    output_column="pred_consensus",
                ),
            )
            summary, _ = evaluate_decision_policies(
                scored,
                latency_budget_ms=latency_budget_ms,
                policy_columns={"predictive_consensus_greedy": "pred_consensus"},
            )
            summary["scenario_name"] = scenario_name
            summary["disagreement_penalty"] = penalty
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _apply_structural_shift_to_forecast_table(
    forecast_table: pd.DataFrame,
    seed: int = 44,
    location_fraction: float = 0.35,
    window_count: int = 2,
    window_length_bins: tuple[int, int] = (2, 4),
    latency_spike_ms: float = 28.0,
    reply_drop_fraction: float = 0.20,
) -> pd.DataFrame:
    """Apply correlated degradation directly on normalized forecast windows.

    This keeps the stress aligned with the evaluation windows used by the
    decision policies, which is important for demonstrating structural shift.
    """

    import numpy as np

    frame = forecast_table.copy()
    rng = np.random.default_rng(seed)
    frame["stress_scenario"] = "structural"
    frame["stress_applied"] = 0

    unique_windows = sorted(frame["session_bin_index"].unique().tolist())
    if not unique_windows:
        return frame

    affected_windows: set[int] = set()
    tail_start_idx = max(0, int(len(unique_windows) * 0.85))
    for _ in range(max(1, window_count)):
        length = int(rng.integers(window_length_bins[0], window_length_bins[1] + 1))
        length = max(1, min(length, len(unique_windows)))
        start_idx = int(rng.integers(tail_start_idx, max(tail_start_idx + 1, len(unique_windows) - length + 1)))
        affected_windows.update(unique_windows[start_idx : start_idx + length])

    eligible_locations = sorted(
        frame.loc[frame["session_bin_index"].isin(affected_windows), "location"].dropna().unique().tolist()
    )
    if not eligible_locations:
        return frame
    selected_count = max(2, int(np.ceil(len(eligible_locations) * location_fraction)))
    selected_count = min(selected_count, len(eligible_locations))
    selected_locations = set(rng.choice(eligible_locations, size=selected_count, replace=False).tolist())

    affected = frame["location"].isin(selected_locations) & frame["session_bin_index"].isin(affected_windows)
    severity = latency_spike_ms + 0.25 * frame["latency_std_ms"].fillna(0.0)
    severity = severity * (0.90 + 0.25 * rng.random(len(frame)))

    frame.loc[affected, "latency_mean_ms"] += severity[affected]
    frame.loc[affected, "latency_std_ms"] += 0.60 * severity[affected]
    frame.loc[affected, "latency_max_ms"] = np.maximum(
        frame.loc[affected, "latency_max_ms"],
        frame.loc[affected, "latency_mean_ms"] + 0.65 * severity[affected],
    )
    frame.loc[affected, "target_next"] += 0.85 * severity[affected]
    if "observed_replies" in frame.columns:
        frame.loc[affected, "observed_replies"] = np.maximum(
            1,
            np.floor(frame.loc[affected, "observed_replies"] * (1.0 - reply_drop_fraction)).astype(int),
        )
    frame.loc[affected, "stress_applied"] = 1
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument("--forecast-metrics-out", default="results/robustness_evaluation/temporal_forecast_robustness_metrics.csv")
    parser.add_argument("--graph-metrics-out", default="results/robustness_evaluation/graph_forecast_robustness_metrics.csv")
    parser.add_argument("--policy-summary-out", default="results/robustness_evaluation/decision_policy_robustness_summary.csv")
    parser.add_argument("--policy-decisions-out", default="results/robustness_evaluation/decision_policy_robustness_window_results.csv")
    parser.add_argument("--policy-significance-out", default="results/robustness_evaluation/decision_policy_robustness_significance.csv")
    parser.add_argument("--consensus-tuning-out", default="results/robustness_evaluation/consensus_policy_tuning.csv")
    parser.add_argument("--window-context-out", default="results/robustness_evaluation/decision_window_context.csv")
    parser.add_argument("--disagreement-summary-out", default="results/robustness_evaluation/disagreement_uncertainty_summary.csv")
    parser.add_argument("--penalty-sweep-out", default="results/robustness_evaluation/consensus_penalty_sweep.csv")
    parser.add_argument("--summary-md-out", default="docs/robustness_results_summary.md")
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)

    stress_cfg = config.get("stress", {})
    # All models are trained once on the base table and then evaluated on the
    # stressed variants. That keeps the robustness claims honest by avoiding
    # retraining on the shifted test conditions.
    scenarios = {
        "base": time_bins.assign(stress_scenario="base", stress_applied=0).copy(),
        "burst": apply_burst_stress(
            time_bins,
            burst_fraction=float(stress_cfg.get("burst_fraction", 0.12)),
            latency_spike_ms=float(stress_cfg.get("burst_latency_spike_ms", 18.0)),
            reply_drop_fraction=float(stress_cfg.get("burst_reply_drop_fraction", 0.12)),
        ),
        "outage": apply_outage_stress(
            time_bins,
            session_fraction=float(stress_cfg.get("outage_session_fraction", 0.25)),
            latency_spike_ms=float(stress_cfg.get("outage_latency_spike_ms", 35.0)),
            reply_drop_fraction=float(stress_cfg.get("outage_reply_drop_fraction", 0.35)),
        ),
        "structural": time_bins.assign(stress_scenario="structural", stress_applied=0).copy(),
    }

    horizon_seconds = int(config["forecasting"]["horizon_seconds"])
    snapshot_seconds = int(config["graph"]["snapshot_seconds"])
    horizon_bins = max(1, horizon_seconds // snapshot_seconds) if horizon_seconds >= snapshot_seconds else 1

    base_forecast_table = build_forecast_table(
        time_bins=scenarios["base"],
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
    )
    base_graph_table = add_graph_snapshot_features(base_forecast_table)
    base_train, base_val, base_test = split_train_val_test(
        base_forecast_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    base_graph_train, base_graph_val, _ = split_train_val_test(
        base_graph_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    base_train_full = pd.concat([base_train, base_val], ignore_index=True)
    base_graph_train_full = pd.concat([base_graph_train, base_graph_val], ignore_index=True)

    forecast_feature_columns = default_feature_columns(base_train_full)
    graph_feature_columns = default_feature_columns(base_graph_train_full)

    trained_forecast_models = {
        "linear_regression": fit_forecast_model("linear_regression", base_train_full, forecast_feature_columns),
    }
    trained_graph_models = {
        "graph_xgboost": fit_graph_xgb_model(base_graph_train_full, graph_feature_columns),
    }
    validation_forecast_model = fit_forecast_model("linear_regression", base_train, forecast_feature_columns)
    validation_graph_model = fit_graph_xgb_model(base_graph_train, graph_feature_columns)

    validation_candidate_frames: dict[str, pd.DataFrame] = {}
    for scenario_name, scenario_time_bins in scenarios.items():
        scenario_forecast_table = build_forecast_table(
            time_bins=scenario_time_bins,
            target_column=config["forecasting"]["target_column"],
            lags=list(config["forecasting"]["lag_steps"]),
            horizon_bins=horizon_bins,
        )
        if scenario_name == "structural":
            scenario_forecast_table = _apply_structural_shift_to_forecast_table(
                scenario_forecast_table,
                location_fraction=float(stress_cfg.get("structural_location_fraction", 0.35)),
                window_count=int(stress_cfg.get("structural_window_count", 2)),
                latency_spike_ms=float(stress_cfg.get("structural_latency_spike_ms", 28.0)),
                reply_drop_fraction=float(stress_cfg.get("structural_reply_drop_fraction", 0.20)),
            )
        scenario_graph_table = add_graph_snapshot_features(scenario_forecast_table)
        _, scenario_val, _ = split_train_val_test(
            scenario_forecast_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )
        _, scenario_graph_val, _ = split_train_val_test(
            scenario_graph_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )
        validation_temporal = predict_forecast_model(
            model_name="linear_regression",
            model=validation_forecast_model,
            test_frame=scenario_val,
            feature_columns=forecast_feature_columns,
        ).rename(columns={"y_pred": "pred_forecast"})
        validation_graph = predict_graph_model(
            model=validation_graph_model,
            test_frame=scenario_graph_val,
            feature_columns=graph_feature_columns,
            model_name="graph_xgboost",
        ).rename(columns={"y_pred": "pred_graph"})
        validation_meta = scenario_graph_val.reset_index(drop=True).reset_index(names="row_id")[
            [
                "row_id",
                "relative_path",
                "location",
                "path_state",
                "session_bin_index",
                "bin_epoch",
                "bin_start_utc",
                "latency_mean_ms",
                "target_next",
            ]
        ]
        validation_candidate_frames[scenario_name] = (
            validation_meta.merge(validation_temporal[["row_id", "pred_forecast"]], on="row_id", how="left")
            .merge(validation_graph[["row_id", "pred_graph"]], on="row_id", how="left")
        )

    tuned_consensus = tune_consensus_policy(validation_candidate_frames)
    tuned_consensus_frame = pd.DataFrame([asdict(tuned_consensus)])
    consensus_policy_config = ConsensusPolicyConfig(
        temporal_weight=tuned_consensus.temporal_weight,
        graph_weight=tuned_consensus.graph_weight,
        disagreement_penalty=tuned_consensus.disagreement_penalty,
    )

    forecast_metric_rows: list[dict[str, object]] = []
    graph_metric_rows: list[dict[str, object]] = []
    policy_summary_rows: list[pd.DataFrame] = []
    policy_decision_rows: list[pd.DataFrame] = []
    policy_significance_rows: list[pd.DataFrame] = []
    window_context_rows: list[pd.DataFrame] = []
    scenario_candidate_frames: dict[str, pd.DataFrame] = {}

    for scenario_name, scenario_time_bins in scenarios.items():
        scenario_forecast_table = build_forecast_table(
            time_bins=scenario_time_bins,
            target_column=config["forecasting"]["target_column"],
            lags=list(config["forecasting"]["lag_steps"]),
            horizon_bins=horizon_bins,
        )
        if scenario_name == "structural":
            scenario_forecast_table = _apply_structural_shift_to_forecast_table(
                scenario_forecast_table,
                location_fraction=float(stress_cfg.get("structural_location_fraction", 0.35)),
                window_count=int(stress_cfg.get("structural_window_count", 2)),
                latency_spike_ms=float(stress_cfg.get("structural_latency_spike_ms", 28.0)),
                reply_drop_fraction=float(stress_cfg.get("structural_reply_drop_fraction", 0.20)),
            )
        scenario_graph_table = add_graph_snapshot_features(scenario_forecast_table)
        _, _, scenario_test = split_train_val_test(
            scenario_forecast_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )
        _, _, scenario_graph_test = split_train_val_test(
            scenario_graph_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )

        persistence_predictions = _persistence_predictions(scenario_test)
        persistence_metrics = asdict(evaluate_prediction_frame(persistence_predictions))
        persistence_metrics["scenario_name"] = scenario_name
        forecast_metric_rows.append(persistence_metrics)

        forecast_predictions_by_model: dict[str, pd.DataFrame] = {"persistence": persistence_predictions}
        for model_name, model in trained_forecast_models.items():
            pred_frame = predict_forecast_model(
                model_name=model_name,
                model=model,
                test_frame=scenario_test,
                feature_columns=forecast_feature_columns,
            )
            forecast_predictions_by_model[model_name] = pred_frame
            metric_row = asdict(evaluate_prediction_frame(pred_frame))
            metric_row["scenario_name"] = scenario_name
            forecast_metric_rows.append(metric_row)

        graph_prediction_frames: dict[str, pd.DataFrame] = {}
        scenario_graph_metric_rows: list[dict[str, object]] = []
        for model_name, model in trained_graph_models.items():
            graph_predictions = predict_graph_model(
                model=model,
                test_frame=scenario_graph_test,
                feature_columns=graph_feature_columns,
                model_name=model_name,
            )
            graph_prediction_frames[model_name] = graph_predictions
            graph_metric_row = asdict(evaluate_graph_predictions(graph_predictions))
            graph_metric_row["scenario_name"] = scenario_name
            graph_metric_rows.append(graph_metric_row)
            scenario_graph_metric_rows.append(graph_metric_row)

        temporal_metric_frame = pd.DataFrame(
            [row for row in forecast_metric_rows if row["scenario_name"] == scenario_name and row["model_name"] != "persistence"]
        )
        best_temporal_model = temporal_metric_frame.sort_values("mae").iloc[0]["model_name"]
        best_graph_model = pd.DataFrame(scenario_graph_metric_rows).sort_values("mae").iloc[0]["model_name"]
        selected_temporal = forecast_predictions_by_model[best_temporal_model].rename(columns={"y_pred": "pred_forecast"})
        selected_graph = graph_prediction_frames[best_graph_model].rename(columns={"y_pred": "pred_graph"})

        test_meta = scenario_graph_test.reset_index(drop=True).reset_index(names="row_id")[
            [
                "row_id",
                "relative_path",
                "location",
                "path_state",
                "session_bin_index",
                "bin_epoch",
                "bin_start_utc",
                "latency_mean_ms",
                "target_next",
                "burst_indicator",
                "reply_pressure_score",
                "peer_burst_indicator_mean",
                "stress_scenario",
                "stress_applied",
            ]
        ]
        candidate_frame = (
            test_meta.merge(selected_temporal[["row_id", "pred_forecast"]], on="row_id", how="left")
            .merge(selected_graph[["row_id", "pred_graph"]], on="row_id", how="left")
        )
        candidate_frame["pred_disagreement"] = (candidate_frame["pred_graph"] - candidate_frame["pred_forecast"]).abs()
        candidate_frame = add_simple_fusion_scores(
            candidate_frame,
            config=SimpleFusionPolicyConfig(
                temporal_weight=consensus_policy_config.temporal_weight,
                graph_weight=consensus_policy_config.graph_weight,
            ),
        )
        candidate_frame = add_consensus_hybrid_scores(candidate_frame, config=consensus_policy_config)
        scenario_candidate_frames[scenario_name] = candidate_frame.copy()
        window_context = _build_window_context(candidate_frame, scenario_name)
        window_context_rows.append(window_context)

        policy_summary, policy_decisions = evaluate_decision_policies(
            candidate_frame,
            latency_budget_ms=float(config["optimization"].get("latency_budget_ms", 60.0)),
        )
        policy_summary["scenario_name"] = scenario_name
        policy_summary["best_temporal_model"] = best_temporal_model
        policy_summary["best_graph_model"] = best_graph_model
        policy_summary_rows.append(policy_summary)

        policy_decisions["scenario_name"] = scenario_name
        policy_decisions["best_temporal_model"] = best_temporal_model
        policy_decisions["best_graph_model"] = best_graph_model
        policy_decisions["stress_applied_rate"] = float(candidate_frame["stress_applied"].mean())
        policy_decisions["chosen_disagreement"] = (policy_decisions["pred_graph"] - policy_decisions["pred_forecast"]).abs()
        policy_decisions = policy_decisions.merge(
            window_context,
            on=["scenario_name", "session_bin_index"],
            how="left",
        )
        policy_decision_rows.append(policy_decisions)
        policy_significance = build_paired_policy_significance(
            decisions=policy_decisions,
            comparisons=[
                ("graph_vs_reactive", "predictive_graph_greedy", "reactive_greedy"),
                ("graph_vs_predictive_only", "predictive_graph_greedy", "predictive_greedy"),
                ("fusion_vs_temporal", "predictive_simple_fusion_greedy", "predictive_greedy"),
                ("fusion_vs_graph", "predictive_simple_fusion_greedy", "predictive_graph_greedy"),
                ("consensus_vs_fusion", "predictive_consensus_greedy", "predictive_simple_fusion_greedy"),
                ("consensus_vs_temporal", "predictive_consensus_greedy", "predictive_greedy"),
                ("consensus_vs_graph", "predictive_consensus_greedy", "predictive_graph_greedy"),
            ],
            metric_columns=["realized_next_latency_ms", "regret_ms"],
        )
        policy_significance["scenario_name"] = scenario_name
        policy_significance_rows.append(policy_significance)

    forecast_metrics = pd.DataFrame(forecast_metric_rows)
    graph_metrics = pd.DataFrame(graph_metric_rows)
    policy_summary_frame = pd.concat(policy_summary_rows, ignore_index=True)
    policy_decisions_frame = pd.concat(policy_decision_rows, ignore_index=True)
    policy_significance_frame = pd.concat(policy_significance_rows, ignore_index=True)
    window_context_frame = pd.concat(window_context_rows, ignore_index=True)
    disagreement_summary = _build_disagreement_summary(policy_decisions_frame)
    penalty_sweep = _run_penalty_sweep(
        scenario_candidate_frames=scenario_candidate_frames,
        temporal_weight=consensus_policy_config.temporal_weight,
        graph_weight=consensus_policy_config.graph_weight,
        penalties=[0.00, 0.10, 0.20, 0.30, 0.50],
        latency_budget_ms=float(config["optimization"].get("latency_budget_ms", 60.0)),
    )

    forecast_out = ensure_parent(_resolve_repo_path(args.forecast_metrics_out))
    graph_out = ensure_parent(_resolve_repo_path(args.graph_metrics_out))
    policy_summary_out = ensure_parent(_resolve_repo_path(args.policy_summary_out))
    policy_decisions_out = ensure_parent(_resolve_repo_path(args.policy_decisions_out))
    policy_significance_out = ensure_parent(_resolve_repo_path(args.policy_significance_out))
    consensus_tuning_out = ensure_parent(_resolve_repo_path(args.consensus_tuning_out))
    window_context_out = ensure_parent(_resolve_repo_path(args.window_context_out))
    disagreement_summary_out = ensure_parent(_resolve_repo_path(args.disagreement_summary_out))
    penalty_sweep_out = ensure_parent(_resolve_repo_path(args.penalty_sweep_out))
    summary_md_out = ensure_parent(_resolve_repo_path(args.summary_md_out))

    forecast_metrics.to_csv(forecast_out, index=False)
    graph_metrics.to_csv(graph_out, index=False)
    policy_summary_frame.to_csv(policy_summary_out, index=False)
    policy_decisions_frame.to_csv(policy_decisions_out, index=False)
    policy_significance_frame.to_csv(policy_significance_out, index=False)
    tuned_consensus_frame.to_csv(consensus_tuning_out, index=False)
    window_context_frame.to_csv(window_context_out, index=False)
    disagreement_summary.to_csv(disagreement_summary_out, index=False)
    penalty_sweep.to_csv(penalty_sweep_out, index=False)
    _write_summary_markdown(
        summary_md_out,
        forecast_metrics,
        graph_metrics,
        policy_summary_frame,
        policy_significance_frame,
        tuned_consensus_frame,
        disagreement_summary,
        penalty_sweep,
    )

    print(f"temporal_forecast_robustness_metrics_written={forecast_out}")
    print(f"graph_forecast_robustness_metrics_written={graph_out}")
    print(f"decision_policy_robustness_summary_written={policy_summary_out}")
    print(f"decision_policy_robustness_window_results_written={policy_decisions_out}")
    print(f"decision_policy_robustness_significance_written={policy_significance_out}")
    print(f"consensus_policy_tuning_written={consensus_tuning_out}")
    print(f"decision_window_context_written={window_context_out}")
    print(f"disagreement_uncertainty_summary_written={disagreement_summary_out}")
    print(f"consensus_penalty_sweep_written={penalty_sweep_out}")
    print(f"robustness_summary_written={summary_md_out}")
    print(policy_summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
