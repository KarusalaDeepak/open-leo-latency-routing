#!/usr/bin/env python3
"""Run a leave-one-session-out forecasting sanity check.

This sanity check is intentionally lightweight. It does not replace the main
chronological split used in the paper; instead, it asks whether the high-level
forecasting ranking changes when each measurement session is held out entirely.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent, load_time_bin_table
from open_leo_latency_routing.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from open_leo_latency_routing.features.temporal import build_forecast_table
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.models.forecast_baselines import default_feature_columns, fit_forecast_model, predict_forecast_model
from open_leo_latency_routing.models.graph_baselines import fit_graph_xgb_model, predict_graph_model


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_std(values: list[float]) -> float:
    return float(pd.Series(values).std(ddof=0)) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument(
        "--summary-out",
        default="results/sanity_checks/session_holdout_forecast_summary.csv",
    )
    parser.add_argument(
        "--folds-out",
        default="results/sanity_checks/session_holdout_forecast_folds.csv",
    )
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)

    horizon_seconds = int(config["forecasting"]["horizon_seconds"])
    snapshot_seconds = int(config["graph"]["snapshot_seconds"])
    horizon_bins = max(1, horizon_seconds // snapshot_seconds) if horizon_seconds >= snapshot_seconds else 1

    forecast_table = build_forecast_table(
        time_bins=time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
    )
    graph_table = add_graph_snapshot_features(forecast_table)

    temporal_feature_columns = default_feature_columns(forecast_table)
    graph_feature_columns = default_feature_columns(graph_table)

    fold_rows: list[dict[str, object]] = []

    for held_out_path in sorted(forecast_table["relative_path"].unique()):
        temporal_train = forecast_table[forecast_table["relative_path"] != held_out_path].reset_index(drop=True)
        temporal_test = forecast_table[forecast_table["relative_path"] == held_out_path].reset_index(drop=True)
        graph_train = graph_table[graph_table["relative_path"] != held_out_path].reset_index(drop=True)
        graph_test = graph_table[graph_table["relative_path"] == held_out_path].reset_index(drop=True)

        if temporal_test.empty or graph_test.empty:
            continue

        temporal_model = fit_forecast_model("linear_regression", temporal_train, temporal_feature_columns)
        graph_model = fit_graph_xgb_model(graph_train, graph_feature_columns)

        temporal_predictions = predict_forecast_model(
            "linear_regression",
            temporal_model,
            temporal_test,
            temporal_feature_columns,
        )
        graph_predictions = predict_graph_model(
            graph_model,
            graph_test,
            graph_feature_columns,
            model_name="graph_xgboost",
        )

        temporal_mae = mean_absolute_error(temporal_predictions["y_true"], temporal_predictions["y_pred"])
        graph_mae = mean_absolute_error(graph_predictions["y_true"], graph_predictions["y_pred"])
        temporal_rmse = root_mean_squared_error(temporal_predictions["y_true"], temporal_predictions["y_pred"])
        graph_rmse = root_mean_squared_error(graph_predictions["y_true"], graph_predictions["y_pred"])

        fold_rows.append(
            {
                "held_out_relative_path": held_out_path,
                "row_count": int(len(temporal_test)),
                "temporal_mae": temporal_mae,
                "graph_context_mae": graph_mae,
                "temporal_rmse": temporal_rmse,
                "graph_context_rmse": graph_rmse,
                "mae_delta_graph_minus_temporal": graph_mae - temporal_mae,
                "rmse_delta_graph_minus_temporal": graph_rmse - temporal_rmse,
                "temporal_better_mae": int(temporal_mae < graph_mae),
                "graph_context_better_mae": int(graph_mae < temporal_mae),
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    summary_frame = pd.DataFrame(
        [
            {
                "fold_count": int(len(fold_frame)),
                "temporal_mae_mean": float(fold_frame["temporal_mae"].mean()),
                "graph_context_mae_mean": float(fold_frame["graph_context_mae"].mean()),
                "temporal_mae_std": _safe_std(fold_frame["temporal_mae"].tolist()),
                "graph_context_mae_std": _safe_std(fold_frame["graph_context_mae"].tolist()),
                "temporal_rmse_mean": float(fold_frame["temporal_rmse"].mean()),
                "graph_context_rmse_mean": float(fold_frame["graph_context_rmse"].mean()),
                "mean_mae_delta_graph_minus_temporal": float(
                    fold_frame["mae_delta_graph_minus_temporal"].mean()
                ),
                "temporal_better_fold_count": int(fold_frame["temporal_better_mae"].sum()),
                "graph_context_better_fold_count": int(fold_frame["graph_context_better_mae"].sum()),
            }
        ]
    )

    summary_out = ensure_parent(_resolve_repo_path(args.summary_out))
    folds_out = ensure_parent(_resolve_repo_path(args.folds_out))
    summary_frame.to_csv(summary_out, index=False)
    fold_frame.to_csv(folds_out, index=False)

    print(f"session_holdout_summary_written={summary_out}")
    print(f"session_holdout_folds_written={folds_out}")
    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
