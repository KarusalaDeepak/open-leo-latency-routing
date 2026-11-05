#!/usr/bin/env python3
"""Run leakage-safe longitudinal and cross-site validation on WetLinks."""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from open_leo_latency_routing.data.loaders import load_time_bin_table
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_train_calibration_selection_test,
)
from open_leo_latency_routing.graphs.snapshots import (
    add_graph_snapshot_features,
    graph_context_feature_columns,
)
from open_leo_latency_routing.models.forecast_baselines import (
    default_feature_columns,
    fit_forecast_model,
)
from open_leo_latency_routing.optimization.calibrated_risk import (
    ExpertCalibration,
    add_calibrated_mixture_risk_scores,
    fit_expert_calibration,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _time_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create globally timed, future-target-closed four-way partitions."""

    epochs = np.sort(frame["bin_epoch"].unique())
    train_end = int(len(epochs) * 0.60)
    calibration_end = int(len(epochs) * 0.75)
    selection_end = int(len(epochs) * 0.85)
    train, calibration, selection, test = (
        split_train_calibration_selection_test(
            frame,
            train_ratio=0.60,
            calibration_ratio=0.15,
            selection_ratio=0.10,
            test_ratio=0.15,
        )
    )

    train_upper = epochs[train_end - 1]
    calibration_upper = epochs[calibration_end - 1]
    selection_upper = epochs[selection_end - 1]

    def _partition(values: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [
                    values.le(train_upper),
                    values.le(calibration_upper),
                    values.le(selection_upper),
                ],
                ["train", "calibration", "selection"],
                default="test",
            ),
            index=values.index,
            dtype="object",
        )

    partitions = {
        "train": train,
        "calibration": calibration,
        "selection": selection,
        "test": test,
    }
    for name, partition in partitions.items():
        if "target_next_bin_epoch" in partition:
            closed = partition["target_next_bin_epoch"].notna() & _partition(
                partition["target_next_bin_epoch"]
            ).eq(name)
            if not bool(closed.all()):
                raise AssertionError(
                    f"WetLinks {name} contains a cross-boundary one-step target"
                )
        for available_column in (
            column
            for column in partition.columns
            if column.startswith("target_available_")
            and column.removeprefix("target_available_").isdigit()
        ):
            horizon = int(available_column.removeprefix("target_available_"))
            endpoint_column = f"target_end_bin_epoch_{horizon}"
            if endpoint_column not in partition:
                continue
            available = partition[available_column].astype(bool)
            endpoint_closed = _partition(partition[endpoint_column]).eq(name)
            if not bool((~available | endpoint_closed).all()):
                raise AssertionError(
                    f"WetLinks {name} exposes a cross-boundary {horizon}-bin target"
                )

    planned_split = _partition(frame["bin_epoch"])
    retained_row_count = sum(len(partition) for partition in partitions.values())
    metadata = {
        "train_end_epoch": int(epochs[train_end - 1]),
        "calibration_start_epoch": int(epochs[train_end]),
        "calibration_end_epoch": int(epochs[calibration_end - 1]),
        "selection_start_epoch": int(epochs[calibration_end]),
        "selection_end_epoch": int(epochs[selection_end - 1]),
        "test_start_epoch": int(epochs[selection_end]),
        "test_end_epoch": int(epochs[-1]),
        "global_wall_clock_partitions": True,
        "future_target_boundary_guard": True,
        "planned_rows_before_boundary_closure": int(len(planned_split)),
        "one_step_boundary_crossing_rows_removed": int(
            len(frame) - retained_row_count
        ),
    }
    return (
        train,
        calibration,
        selection,
        test,
        metadata,
    )


def _attach_paired_residual_covariance(
    temporal_calibration: ExpertCalibration,
    graph_calibration: ExpertCalibration,
    truth: pd.Series | np.ndarray,
    temporal_prediction: pd.Series | np.ndarray,
    graph_prediction: pd.Series | np.ndarray,
) -> tuple[ExpertCalibration, ExpertCalibration]:
    """Store calibration-only covariance shared by the two expert residuals."""

    target = np.asarray(truth, dtype=float)
    temporal_centered_residual = (
        target
        - np.asarray(temporal_prediction, dtype=float)
        - temporal_calibration.residual_bias_ms
    )
    graph_centered_residual = (
        target
        - np.asarray(graph_prediction, dtype=float)
        - graph_calibration.residual_bias_ms
    )
    covariance = 0.0
    if len(target) > 1:
        covariance = float(
            np.cov(
                temporal_centered_residual,
                graph_centered_residual,
                ddof=1,
            )[0, 1]
        )
        if not np.isfinite(covariance):
            covariance = 0.0
    temporal_variance = float(
        temporal_calibration.residual_variance_ms2
        if temporal_calibration.residual_variance_ms2 is not None
        else temporal_calibration.residual_scale_ms**2
    )
    graph_variance = float(
        graph_calibration.residual_variance_ms2
        if graph_calibration.residual_variance_ms2 is not None
        else graph_calibration.residual_scale_ms**2
    )
    covariance_limit = float(
        np.sqrt(max(temporal_variance * graph_variance, 0.0))
    )
    covariance = float(np.clip(covariance, -covariance_limit, covariance_limit))
    return (
        replace(
            temporal_calibration,
            paired_residual_covariance_ms2=covariance,
        ),
        replace(
            graph_calibration,
            paired_residual_covariance_ms2=covariance,
        ),
    )


def _fit_pair(
    train: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    selection: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    temporal_features = default_feature_columns(train)
    graph_features = graph_context_feature_columns(train)
    temporal_model = fit_forecast_model("ridge_regression", train, temporal_features)
    graph_model = fit_forecast_model("ridge_regression", train, graph_features)

    temporal_calibration_prediction = temporal_model.predict(
        calibration_frame[temporal_features].fillna(0.0)
    )
    graph_calibration_prediction = graph_model.predict(
        calibration_frame[graph_features].fillna(0.0)
    )
    temporal_calibration = fit_expert_calibration(
        calibration_frame["target_next"], temporal_calibration_prediction
    )
    graph_calibration = fit_expert_calibration(
        calibration_frame["target_next"], graph_calibration_prediction
    )
    temporal_calibration, graph_calibration = _attach_paired_residual_covariance(
        temporal_calibration,
        graph_calibration,
        calibration_frame["target_next"],
        temporal_calibration_prediction,
        graph_calibration_prediction,
    )

    def score(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[
            ["relative_path", "location", "bin_epoch", "bin_start_utc", "target_next"]
        ].copy()
        output["pred_persistence"] = frame["latency_mean_ms"].to_numpy()
        output["pred_forecast"] = temporal_model.predict(
            frame[temporal_features].fillna(0.0)
        )
        output["pred_graph"] = graph_model.predict(
            frame[graph_features].fillna(0.0)
        )
        return add_calibrated_mixture_risk_scores(
            output, temporal_calibration, graph_calibration
        )

    selection_predictions = score(selection)
    candidate_columns = [
        "pred_temporal_calibrated",
        "pred_graph_calibrated",
        "pred_calibrated_fusion",
    ]
    selection_mae = {
        column: float(
            (selection_predictions["target_next"] - selection_predictions[column])
            .abs()
            .mean()
        )
        for column in candidate_columns
    }
    selected_column = min(selection_mae, key=selection_mae.get)
    output = score(test)
    output["pred_validation_selected"] = output[selected_column]
    calibration = {
        "temporal_calibration_mae_ms": temporal_calibration.mae_ms,
        "context_calibration_mae_ms": graph_calibration.mae_ms,
        "temporal_weight": float(output["temporal_expert_weight"].iloc[0]),
        "context_weight": float(output["graph_expert_weight"].iloc[0]),
        "selection_mae_ms": selection_mae,
        "selected_prediction_column": selected_column,
    }
    return output, calibration


def _metric_rows(predictions: pd.DataFrame, evaluation: str) -> list[dict]:
    columns = {
        "persistence": "pred_persistence",
        "temporal_ridge": "pred_temporal_calibrated",
        "context_ridge": "pred_graph_calibrated",
        "calibrated_fusion": "pred_calibrated_fusion",
        "validation_selected": "pred_validation_selected",
    }
    rows = []
    truth = predictions["target_next"].to_numpy(float)
    for model, column in columns.items():
        residual = truth - predictions[column].to_numpy(float)
        absolute = np.abs(residual)
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "rows": len(predictions),
                "mae_ms": float(absolute.mean()),
                "rmse_ms": float(np.sqrt(np.mean(residual**2))),
                "p95_absolute_error_ms": float(np.quantile(absolute, 0.95)),
                "bias_ms": float(residual.mean()),
            }
        )
    return rows


def _paired_day_bootstrap(
    predictions: pd.DataFrame,
    left_column: str,
    right_column: str,
    seed: int = 2026,
    repetitions: int = 4000,
) -> dict:
    frame = predictions.copy()
    frame["day"] = pd.to_datetime(frame["bin_start_utc"]).dt.date
    frame["delta"] = (
        (frame["target_next"] - frame[left_column]).abs()
        - (frame["target_next"] - frame[right_column]).abs()
    )
    day_means = frame.groupby("day")["delta"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(day_means, size=(repetitions, len(day_means)), replace=True)
    boot = draws.mean(axis=1)
    return {
        "left": left_column,
        "right": right_column,
        "day_blocks": int(len(day_means)),
        "mean_mae_delta_ms": float(day_means.mean()),
        "ci_low_ms": float(np.quantile(boot, 0.025)),
        "ci_high_ms": float(np.quantile(boot, 0.975)),
    }


def _risk_diagnostics(predictions: pd.DataFrame) -> dict:
    error = (
        predictions["target_next"] - predictions["pred_calibrated_fusion"]
    ).abs()
    disagreement = predictions["pred_disagreement_normalized"]
    corr = spearmanr(disagreement, error, nan_policy="omit")
    high_error = (error >= error.quantile(0.75)).astype(int)
    upper_90 = predictions["pred_calibrated_fusion"] + 1.645 * predictions["pred_mixture_std"]
    return {
        "rows": int(len(predictions)),
        "spearman_disagreement_vs_absolute_error": float(corr.statistic),
        "spearman_p_value": float(corr.pvalue),
        "high_error_detection_auroc": float(roc_auc_score(high_error, disagreement)),
        "nominal_upper_coverage": 0.90,
        "empirical_upper_coverage": float((predictions["target_next"] <= upper_90).mean()),
    }


def _cross_site_temporal_transfer(
    forecast: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_rows = []
    sites = sorted(forecast["relative_path"].unique())
    features = default_feature_columns(train)
    for source in sites:
        target = next(site for site in sites if site != source)
        source_train = train[train["relative_path"] == source]
        source_val = validation[validation["relative_path"] == source]
        target_test = test[test["relative_path"] == target]
        model = fit_forecast_model("ridge_regression", source_train, features)
        validation_prediction = model.predict(source_val[features].fillna(0.0))
        calibration = fit_expert_calibration(
            source_val["target_next"], validation_prediction
        )
        prediction = model.predict(target_test[features].fillna(0.0)) + calibration.residual_bias_ms
        persistence = target_test["latency_mean_ms"].to_numpy(float)
        truth = target_test["target_next"].to_numpy(float)
        rows.append(
            {
                "source_site": source,
                "held_out_target_site": target,
                "test_rows": len(target_test),
                "persistence_mae_ms": float(np.mean(np.abs(truth - persistence))),
                "transferred_temporal_mae_ms": float(np.mean(np.abs(truth - prediction))),
                "mae_delta_vs_persistence_ms": float(
                    np.mean(np.abs(truth - prediction) - np.abs(truth - persistence))
                ),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "source_site": source,
                    "held_out_target_site": target,
                    "bin_epoch": target_test["bin_epoch"].to_numpy(),
                    "target_next": truth,
                    "pred_persistence": persistence,
                    "pred_transferred_temporal": prediction,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-bins", default="data/processed/wetlinks_latency_5min.csv"
    )
    parser.add_argument(
        "--output-dir", default="results/wetlinks_longitudinal_validation"
    )
    args = parser.parse_args()

    bins = load_time_bin_table(_resolve(args.time_bins))
    bin_seconds = int(bins["bin_seconds"].iloc[0])
    shared_epochs = bins.groupby("bin_epoch")["relative_path"].nunique()
    bins = bins[bins["bin_epoch"].isin(shared_epochs[shared_epochs >= 2].index)].copy()
    bins["session_bin_index"] = pd.factorize(bins["bin_epoch"], sort=True)[0]
    forecast = build_forecast_table(
        bins,
        target_column="latency_mean_ms",
        lags=[1, 2, 3, 6, 12],
        horizon_bins=1,
        decision_cadence_seconds=bin_seconds,
    )
    graph_forecast = add_graph_snapshot_features(forecast)
    train, calibration_frame, selection, test, split_metadata = _time_split(
        graph_forecast
    )
    predictions, calibration = _fit_pair(
        train, calibration_frame, selection, test
    )

    metric_rows = _metric_rows(predictions, "late_period_two_site_holdout")
    metrics = pd.DataFrame(metric_rows)
    risk = _risk_diagnostics(predictions)
    paired = pd.DataFrame(
        [
            _paired_day_bootstrap(
                predictions, "pred_calibrated_fusion", "pred_persistence"
            ),
            _paired_day_bootstrap(
                predictions, "pred_calibrated_fusion", "pred_temporal_calibrated"
            ),
        ]
    )
    transfer, transfer_predictions = _cross_site_temporal_transfer(
        graph_forecast, train, calibration_frame, test
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "late_holdout_predictions.csv", index=False)
    metrics.to_csv(output_dir / "late_holdout_model_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_day_block_intervals.csv", index=False)
    transfer.to_csv(output_dir / "cross_site_transfer_summary.csv", index=False)
    transfer_predictions.to_csv(
        output_dir / "cross_site_transfer_predictions.csv", index=False
    )
    pd.DataFrame([risk]).to_csv(output_dir / "risk_diagnostics.csv", index=False)
    metadata = {
        "dataset": "WetLinks",
        "dataset_repository": "https://github.com/sys-uos/WetLinks",
        "license": "CC BY-SA 4.0",
        "evaluation_semantics": "independent_longitudinal_prediction_validation",
        "has_temporally_concurrent_candidates": False,
        "supports_candidate_outcome_shadow_replay": False,
        "supports_literal_single_controller_steering": False,
        "reason_not_policy_evidence": "sites are geographically distinct and not interchangeable controller choices",
        "bin_seconds": bin_seconds,
        "fixed_forecast_horizon_seconds": bin_seconds,
        "shared_epochs_before_target_filter": int(len(shared_epochs[shared_epochs >= 2])),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration_frame)),
        "selection_rows": int(len(selection)),
        "test_rows": int(len(test)),
        "split": split_metadata,
        "calibration": calibration,
        "risk_diagnostics": risk,
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print("\nCross-site transfer")
    print(transfer.to_string(index=False))
    print("\nRisk diagnostics")
    print(json.dumps(risk, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
