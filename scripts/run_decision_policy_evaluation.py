#!/usr/bin/env python3
"""Evaluate the final reactive and predictive decision policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent
from open_leo_latency_routing.evaluation.significance import build_paired_policy_significance
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    SimpleFusionPolicyConfig,
    add_consensus_hybrid_scores,
    add_simple_fusion_scores,
    evaluate_decision_policies,
)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--forecast-metrics", default="results/temporal_forecasting/temporal_forecast_metrics.csv")
    parser.add_argument("--forecast-predictions", default="results/temporal_forecasting/temporal_forecast_predictions.csv")
    parser.add_argument("--graph-metrics", default="results/graph_forecasting/graph_forecast_metrics.csv")
    parser.add_argument("--graph-predictions", default="results/graph_forecasting/graph_forecast_predictions.csv")
    parser.add_argument("--summary-out", default="results/decision_policy_evaluation/decision_policy_summary.csv")
    parser.add_argument("--decisions-out", default="results/decision_policy_evaluation/decision_policy_window_results.csv")
    parser.add_argument("--significance-out", default="results/decision_policy_evaluation/decision_policy_significance.csv")
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    forecast_metrics = pd.read_csv(_resolve_repo_path(args.forecast_metrics))
    forecast_predictions = pd.read_csv(_resolve_repo_path(args.forecast_predictions))
    graph_metrics = pd.read_csv(_resolve_repo_path(args.graph_metrics))
    graph_predictions = pd.read_csv(_resolve_repo_path(args.graph_predictions))

    best_forecast_model = forecast_metrics.sort_values("mae").iloc[0]["model_name"]
    best_graph_model = graph_metrics.sort_values("mae").iloc[0]["model_name"]
    selected_forecast = (
        forecast_predictions[forecast_predictions["model_name"] == best_forecast_model]
        .rename(columns={"y_pred": "pred_forecast"})
        .copy()
    )
    selected_graph = (
        graph_predictions[graph_predictions["model_name"] == best_graph_model]
        .rename(columns={"y_pred": "pred_graph"})
        .copy()
    )

    # The policy stage evaluates one decision window at a time. It merges the
    # best temporal forecaster with the best graph-aware forecaster so the
    # policy comparison is based on each family's strongest available model.
    candidate_frame = selected_forecast[
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
            "pred_forecast",
            "burst_indicator",
            "reply_pressure_score",
        ]
    ].merge(
        selected_graph[["row_id", "pred_graph", "peer_burst_indicator_mean"]],
        on="row_id",
        how="inner",
    )
    consensus_cfg = config["optimization"].get("consensus", {})
    candidate_frame = add_simple_fusion_scores(
        candidate_frame,
        config=SimpleFusionPolicyConfig(
            temporal_weight=float(consensus_cfg.get("temporal_weight", 0.65)),
            graph_weight=float(consensus_cfg.get("graph_weight", 0.35)),
        ),
    )
    candidate_frame = add_consensus_hybrid_scores(
        candidate_frame,
        config=ConsensusPolicyConfig(
            temporal_weight=float(consensus_cfg.get("temporal_weight", 0.65)),
            graph_weight=float(consensus_cfg.get("graph_weight", 0.35)),
            disagreement_penalty=float(consensus_cfg.get("disagreement_penalty", 0.30)),
        ),
    )

    summary, decisions = evaluate_decision_policies(
        candidate_frame,
        latency_budget_ms=float(config["optimization"].get("latency_budget_ms", 60.0)),
    )

    summary_out = ensure_parent(_resolve_repo_path(args.summary_out))
    decisions_out = ensure_parent(_resolve_repo_path(args.decisions_out))
    significance_out = ensure_parent(_resolve_repo_path(args.significance_out))
    significance = build_paired_policy_significance(
        decisions=decisions,
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
    summary.to_csv(summary_out, index=False)
    decisions.to_csv(decisions_out, index=False)
    significance.to_csv(significance_out, index=False)

    print(f"best_forecast_model={best_forecast_model}")
    print(f"best_graph_model={best_graph_model}")
    print(f"decision_policy_summary_written={summary_out}")
    print(f"decision_policy_window_results_written={decisions_out}")
    print(f"decision_policy_significance_written={significance_out}")
    print(summary.to_string(index=False))
    print(significance.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
