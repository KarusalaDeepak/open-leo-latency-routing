#!/usr/bin/env python3
"""Run feature-family ablations with bootstrap confidence intervals."""

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
from open_leo_latency_routing.evaluation.confidence_intervals import build_bootstrap_policy_intervals
from open_leo_latency_routing.evaluation.significance import build_paired_policy_significance
from open_leo_latency_routing.features.temporal import build_forecast_table, split_train_val_test
from open_leo_latency_routing.graphs.snapshots import GRAPH_SNAPSHOT_FEATURE_COLUMNS, add_graph_snapshot_features
from open_leo_latency_routing.models.forecast_baselines import default_feature_columns, fit_forecast_model, predict_forecast_model
from open_leo_latency_routing.models.graph_baselines import fit_graph_xgb_model, predict_graph_model
from open_leo_latency_routing.optimization.policies import evaluate_decision_policies
from open_leo_latency_routing.scenarios.stress import apply_burst_stress, apply_outage_stress


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument("--summary-out", default="results/ablation_evaluation/ablation_policy_summary.csv")
    parser.add_argument("--decisions-out", default="results/ablation_evaluation/ablation_policy_window_results.csv")
    parser.add_argument("--significance-out", default="results/ablation_evaluation/ablation_policy_significance.csv")
    parser.add_argument("--ci-out", default="results/ablation_evaluation/ablation_policy_confidence_intervals.csv")
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)

    stress_cfg = config.get("stress", {})
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
    base_train, base_val, _ = split_train_val_test(
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

    temporal_feature_columns = default_feature_columns(base_train_full)
    graph_full_feature_columns = default_feature_columns(base_graph_train_full)
    graph_only_feature_columns = [column for column in GRAPH_SNAPSHOT_FEATURE_COLUMNS if column in base_graph_train_full.columns]

    temporal_model = fit_forecast_model("linear_regression", base_train_full, temporal_feature_columns)
    graph_full_model = fit_graph_xgb_model(base_graph_train_full, graph_full_feature_columns)
    graph_only_model = fit_graph_xgb_model(base_graph_train_full, graph_only_feature_columns)

    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    significance_rows: list[pd.DataFrame] = []
    ci_rows: list[pd.DataFrame] = []

    for scenario_name, scenario_time_bins in scenarios.items():
        forecast_table = build_forecast_table(
            time_bins=scenario_time_bins,
            target_column=config["forecasting"]["target_column"],
            lags=list(config["forecasting"]["lag_steps"]),
            horizon_bins=horizon_bins,
        )
        graph_table = add_graph_snapshot_features(forecast_table)
        _, _, test_frame = split_train_val_test(
            forecast_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )
        _, _, graph_test = split_train_val_test(
            graph_table,
            train_ratio=float(config["forecasting"]["train_ratio"]),
            val_ratio=float(config["forecasting"]["val_ratio"]),
            test_ratio=float(config["forecasting"]["test_ratio"]),
        )

        temporal_predictions = predict_forecast_model(
            "linear_regression",
            temporal_model,
            test_frame,
            temporal_feature_columns,
        ).rename(columns={"y_pred": "pred_temporal_only"})
        graph_full_predictions = predict_graph_model(
            graph_full_model,
            graph_test,
            graph_full_feature_columns,
            model_name="graph_xgboost_full",
        ).rename(columns={"y_pred": "pred_graph_full"})
        graph_only_predictions = predict_graph_model(
            graph_only_model,
            graph_test,
            graph_only_feature_columns,
            model_name="graph_xgboost_graph_only",
        ).rename(columns={"y_pred": "pred_graph_only"})

        test_meta = graph_test.reset_index(drop=True).reset_index(names="row_id")[
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
                "stress_scenario",
                "stress_applied",
            ]
        ]
        candidate_frame = (
            test_meta.merge(temporal_predictions[["row_id", "pred_temporal_only"]], on="row_id", how="left")
            .merge(graph_only_predictions[["row_id", "pred_graph_only"]], on="row_id", how="left")
            .merge(graph_full_predictions[["row_id", "pred_graph_full"]], on="row_id", how="left")
        )

        summary, decisions = evaluate_decision_policies(
            candidate_frame,
            latency_budget_ms=float(config["optimization"].get("latency_budget_ms", 60.0)),
            policy_columns={
                "predictive_temporal_only": "pred_temporal_only",
                "predictive_graph_only": "pred_graph_only",
                "predictive_graph_greedy": "pred_graph_full",
            },
        )
        summary["scenario_name"] = scenario_name
        decisions["scenario_name"] = scenario_name
        summary_rows.append(summary)
        decision_rows.append(decisions)

        significance = build_paired_policy_significance(
            decisions=decisions,
            comparisons=[
                ("full_vs_temporal", "predictive_graph_greedy", "predictive_temporal_only"),
                ("full_vs_graph_only", "predictive_graph_greedy", "predictive_graph_only"),
            ],
            metric_columns=["realized_next_latency_ms", "regret_ms"],
        )
        significance["scenario_name"] = scenario_name
        significance_rows.append(significance)

        ci_frame = build_bootstrap_policy_intervals(
            decisions=decisions,
            metric_columns=[
                "realized_next_latency_ms",
                "regret_ms",
                "best_path_match",
                "success_under_budget",
            ],
            n_bootstrap=3000,
            ci=0.95,
            random_state=42,
        )
        ci_frame["scenario_name"] = scenario_name
        ci_rows.append(ci_frame)

    summary_frame = pd.concat(summary_rows, ignore_index=True)
    decision_frame = pd.concat(decision_rows, ignore_index=True)
    significance_frame = pd.concat(significance_rows, ignore_index=True)
    ci_frame = pd.concat(ci_rows, ignore_index=True)

    ci_wide = (
        ci_frame.pivot(index=["scenario_name", "policy_name"], columns="metric_name", values=["ci_lower", "ci_upper"])
        .sort_index(axis=1)
        .reset_index()
    )
    ci_wide.columns = [
        column if isinstance(column, str) else (
            column[0] if column[1] == "" else f"{column[1]}_{column[0]}"
        )
        for column in ci_wide.columns.to_flat_index()
    ]
    summary_frame = summary_frame.merge(ci_wide, on=["scenario_name", "policy_name"], how="left")

    summary_out = ensure_parent(_resolve_repo_path(args.summary_out))
    decisions_out = ensure_parent(_resolve_repo_path(args.decisions_out))
    significance_out = ensure_parent(_resolve_repo_path(args.significance_out))
    ci_out = ensure_parent(_resolve_repo_path(args.ci_out))
    summary_frame.to_csv(summary_out, index=False)
    decision_frame.to_csv(decisions_out, index=False)
    significance_frame.to_csv(significance_out, index=False)
    ci_frame.to_csv(ci_out, index=False)

    print(f"ablation_policy_summary_written={summary_out}")
    print(f"ablation_policy_window_results_written={decisions_out}")
    print(f"ablation_policy_significance_written={significance_out}")
    print(f"ablation_policy_confidence_intervals_written={ci_out}")
    print(summary_frame.to_string(index=False))
    print(significance_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
