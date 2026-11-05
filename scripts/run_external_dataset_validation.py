#!/usr/bin/env python3
"""Validate predictor and risk-signal transfer on independent Starlink IRTT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import load_time_bin_table
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_train_val_test,
)
from open_leo_latency_routing.graphs.snapshots import (
    add_graph_snapshot_features,
    graph_context_feature_columns,
)
from open_leo_latency_routing.models.forecast_baselines import (
    default_feature_columns,
    fit_forecast_model,
)
from open_leo_latency_routing.models.graph_baselines import fit_graph_context_model
from open_leo_latency_routing.optimization.calibrated_risk import (
    add_calibrated_mixture_risk_scores,
    fit_expert_calibration,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _fit_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    graph_train: pd.DataFrame,
    graph_validation: pd.DataFrame,
    graph_test: pd.DataFrame,
) -> pd.DataFrame:
    temporal_features = default_feature_columns(train)
    graph_features = graph_context_feature_columns(graph_train)
    temporal_model = fit_forecast_model(
        "ridge_regression", train, temporal_features
    )
    graph_model = fit_graph_context_model(
        "ridge_regression", graph_train, graph_features
    )
    temporal_validation_prediction = temporal_model.predict(
        validation[temporal_features].fillna(0.0)
    )
    graph_validation_prediction = graph_model.predict(
        graph_validation[graph_features].fillna(0.0)
    )
    temporal_calibration = fit_expert_calibration(
        validation["target_next"], temporal_validation_prediction
    )
    graph_calibration = fit_expert_calibration(
        graph_validation["target_next"], graph_validation_prediction
    )
    output = test[
        ["relative_path", "session_bin_index", "target_next"]
    ].reset_index(drop=True)
    output["pred_forecast"] = temporal_model.predict(
        test[temporal_features].fillna(0.0)
    )
    output["pred_graph"] = graph_model.predict(
        graph_test[graph_features].fillna(0.0)
    )
    return add_calibrated_mixture_risk_scores(
        output,
        temporal_calibration,
        graph_calibration,
    )


def _summarize_predictions(predictions: pd.DataFrame, evaluation_name: str) -> dict[str, object]:
    temporal_error = (
        predictions["target_next"] - predictions["pred_temporal_calibrated"]
    ).abs()
    graph_error = (
        predictions["target_next"] - predictions["pred_graph_calibrated"]
    ).abs()
    fusion_error = (
        predictions["target_next"] - predictions["pred_calibrated_fusion"]
    ).abs()
    disagreement = predictions["pred_disagreement_normalized"]
    corr = spearmanr(disagreement, fusion_error, nan_policy="omit")
    high_error = (fusion_error >= fusion_error.quantile(0.75)).astype(int)
    auroc = (
        float(roc_auc_score(high_error, disagreement))
        if high_error.nunique() == 2
        else float("nan")
    )
    return {
        "evaluation_name": evaluation_name,
        "rows": len(predictions),
        "sessions": predictions["relative_path"].nunique(),
        "temporal_mae_ms": float(temporal_error.mean()),
        "graph_mae_ms": float(graph_error.mean()),
        "calibrated_fusion_mae_ms": float(fusion_error.mean()),
        "spearman_disagreement_vs_fusion_error": float(corr.statistic),
        "spearman_p_value": float(corr.pvalue),
        "high_error_detection_auroc": auroc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--external-table",
        default="data/processed/external_starlink_irtt_10s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/external_dataset_validation",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    external_bins = load_time_bin_table(_resolve(args.external_table))
    resolution_seconds = int(external_bins["bin_seconds"].iloc[0])
    horizon_bins = max(
        1,
        int(np.ceil(config["forecasting"]["horizon_seconds"] / resolution_seconds)),
    )
    external_forecast = build_forecast_table(
        external_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
        decision_cadence_seconds=resolution_seconds,
    )
    external_graph = add_graph_snapshot_features(external_forecast)
    train, validation, test = split_train_val_test(
        external_forecast, 0.70, 0.15, 0.15
    )
    graph_train, graph_validation, graph_test = split_train_val_test(
        external_graph, 0.70, 0.15, 0.15
    )
    predictions = _fit_and_predict(
        train,
        validation,
        test,
        graph_train,
        graph_validation,
        graph_test,
    )
    summary = pd.DataFrame(
        [_summarize_predictions(predictions, "independent_dataset_retrained")]
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "external_prediction_rows.csv", index=False)
    summary.to_csv(output_dir / "external_prediction_summary.csv", index=False)
    metadata = {
        "dataset_doi": "10.17632/479v4mym7j.2",
        "license": "CC BY 4.0",
        "dataset_independent_of_lens": True,
        "bin_seconds": resolution_seconds,
        "forecast_horizon_bins": horizon_bins,
        "forecast_horizon_seconds": horizon_bins * resolution_seconds,
        "exact_horizon_audit": external_forecast.attrs.get(
            "exact_horizon_audit", {}
        ),
        "valid_claim": (
            "predictor and disagreement-risk behavior on an independent "
            "Starlink measurement dataset"
        ),
        "invalid_claim": (
            "service-path policy generalization, because the archive contains "
            "repeated single-endpoint experiments rather than concurrent "
            "alternative service paths"
        ),
    }
    (output_dir / "external_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"external_validation_written={output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
