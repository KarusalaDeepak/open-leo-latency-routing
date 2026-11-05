#!/usr/bin/env python3
"""Generate code-backed evidence for the major reviewer concerns."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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
    ExpertCalibration,
    add_calibrated_mixture_risk_scores,
    fit_expert_calibration,
)
from open_leo_latency_routing.optimization.policies import evaluate_decision_policies


ABLATION_POLICIES = {
    "temporal_only": "pred_forecast",
    "graph_only": "pred_graph",
    "simple_fusion": "pred_simple_fusion",
    "calibrated_fusion": "pred_calibrated_fusion",
    "disagreement_only": "pred_disagreement_only",
    "service_risk_without_disagreement": "pred_service_risk_only",
    "learned_risk_without_disagreement": "pred_learned_risk_no_disagreement",
    "ungated_calibrated_risk": "pred_calibrated_risk_ungated",
    "ensemble_uncertainty": "pred_ensemble_uncertainty",
    "full_gated_operational_rule": "pred_calibrated_operational",
}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "scenario_name",
        "relative_path",
        "session_bin_index",
        "target_next",
        "pred_forecast",
        "pred_graph",
        "pred_calibrated_operational",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(
            f"candidate prediction table is missing columns: {', '.join(missing)}"
        )
    frame["pred_service_risk_only"] = (
        frame["pred_calibrated_fusion"] + frame["service_risk_ms"]
    )
    return frame


def _run_ablation(candidates: pd.DataFrame, latency_budget_ms: float) -> pd.DataFrame:
    rows = []
    for scenario_name, frame in candidates.groupby("scenario_name", sort=False):
        summary, _ = evaluate_decision_policies(
            frame,
            latency_budget_ms=latency_budget_ms,
            policy_columns=ABLATION_POLICIES,
        )
        summary["scenario_name"] = scenario_name
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _run_control_loop_sensitivity(
    candidates: pd.DataFrame,
    latency_budget_ms: float,
    decision_window_seconds: float,
    delays_ms: list[float],
) -> pd.DataFrame:
    rows = []
    policies = {
        "temporal_only": "pred_forecast",
        "graph_only": "pred_graph",
        "ensemble_uncertainty": "pred_ensemble_uncertainty",
        "full_gated_operational_rule": "pred_calibrated_operational",
    }
    for scenario_name, frame in candidates.groupby("scenario_name", sort=False):
        for delay_ms in delays_ms:
            summary, _ = evaluate_decision_policies(
                frame,
                latency_budget_ms=latency_budget_ms,
                policy_columns=policies,
                control_loop_latency_ms=delay_ms,
                decision_window_seconds=decision_window_seconds,
            )
            summary["scenario_name"] = scenario_name
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _run_stale_state_sensitivity(
    candidates: pd.DataFrame,
    latency_budget_ms: float,
) -> pd.DataFrame:
    """Evaluate decisions made from scores delayed by complete measurement bins."""

    rows = []
    policies = {
        "temporal_only": "pred_forecast",
        "graph_only": "pred_graph",
        "ensemble_uncertainty": "pred_ensemble_uncertainty",
        "full_gated_operational_rule": "pred_calibrated_operational",
    }
    for scenario_name, scenario_frame in candidates.groupby("scenario_name", sort=False):
        ordered = scenario_frame.sort_values(["relative_path", "session_bin_index"]).copy()
        for state_age_bins in (0, 1, 2):
            work = ordered.copy()
            delayed_columns = {}
            for policy_name, score_column in policies.items():
                delayed_column = f"{score_column}_age_{state_age_bins}"
                work[delayed_column] = work.groupby("relative_path")[score_column].shift(
                    state_age_bins
                )
                delayed_columns[policy_name] = delayed_column
            work = work.dropna(subset=list(delayed_columns.values()))
            if work.empty:
                continue
            summary, _ = evaluate_decision_policies(
                work,
                latency_budget_ms=latency_budget_ms,
                policy_columns=delayed_columns,
            )
            summary["scenario_name"] = scenario_name
            summary["state_age_bins"] = state_age_bins
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _run_disagreement_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name, frame in candidates.groupby("scenario_name", sort=False):
        work = frame.copy()
        work["fusion_error_ms"] = (
            work["pred_calibrated_fusion"] - work["target_next"]
        ).abs()
        work["temporal_error_ms"] = (
            work["pred_temporal_calibrated"] - work["target_next"]
        ).abs()
        work["graph_error_ms"] = (
            work["pred_graph_calibrated"] - work["target_next"]
        ).abs()
        high_error_threshold = float(work["fusion_error_ms"].quantile(0.75))
        work["high_error"] = (work["fusion_error_ms"] >= high_error_threshold).astype(int)
        if work["high_error"].nunique() == 2:
            auroc = float(
                roc_auc_score(
                    work["high_error"],
                    work["pred_disagreement_normalized"],
                )
            )
        else:
            auroc = float("nan")
        corr = spearmanr(
            work["pred_disagreement_normalized"],
            work["fusion_error_ms"],
            nan_policy="omit",
        )
        ensemble_corr = spearmanr(
            work["pred_ensemble_std"],
            work["fusion_error_ms"],
            nan_policy="omit",
        )
        if work["high_error"].nunique() == 2:
            ensemble_auroc = float(
                roc_auc_score(
                    work["high_error"],
                    work["pred_ensemble_std"],
                )
            )
        else:
            ensemble_auroc = float("nan")
        burst_corr = spearmanr(
            work["pred_disagreement_normalized"],
            work.get("burst_indicator", pd.Series(0.0, index=work.index)),
            nan_policy="omit",
        )
        reply_pressure_corr = spearmanr(
            work["pred_disagreement_normalized"],
            work.get("reply_pressure_score", pd.Series(0.0, index=work.index)),
            nan_policy="omit",
        )
        temporal_bad = work["temporal_error_ms"] >= work["temporal_error_ms"].quantile(0.75)
        graph_bad = work["graph_error_ms"] >= work["graph_error_ms"].quantile(0.75)
        low_disagreement = (
            work["pred_disagreement_normalized"]
            <= work["pred_disagreement_normalized"].median()
        )
        shared_failure = temporal_bad & graph_bad
        shared_failure_count = int(shared_failure.sum())
        rows.append(
            {
                "scenario_name": scenario_name,
                "candidate_count": len(work),
                "spearman_disagreement_vs_fusion_error": float(corr.statistic),
                "spearman_p_value": float(corr.pvalue),
                "spearman_disagreement_vs_burst_proxy": float(
                    burst_corr.statistic
                ),
                "spearman_disagreement_vs_reply_pressure": float(
                    reply_pressure_corr.statistic
                ),
                "high_error_detection_auroc": auroc,
                "spearman_ensemble_spread_vs_fusion_error": float(
                    ensemble_corr.statistic
                ),
                "ensemble_spread_p_value": float(ensemble_corr.pvalue),
                "ensemble_spread_high_error_auroc": ensemble_auroc,
                "mean_normalized_disagreement": float(
                    work["pred_disagreement_normalized"].mean()
                ),
                "mean_trust_gate": float(work["disagreement_trust_gate"].mean()),
                "shared_failure_count": shared_failure_count,
                "shared_failure_low_disagreement_rate": (
                    float((shared_failure & low_disagreement).sum() / shared_failure_count)
                    if shared_failure_count
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_empirical_risk_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    """Measure how often calibrated fusion radii cover realized latency."""

    rows = []
    for scenario_name, frame in candidates.groupby("scenario_name", sort=False):
        absolute_error = (
            frame["target_next"] - frame["pred_calibrated_fusion"]
        ).abs()
        for multiplier in (1.0, 1.5, 2.0, 3.0):
            radius = multiplier * frame["pred_mixture_std"]
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "radius_multiplier": multiplier,
                    "empirical_coverage": float((absolute_error <= radius).mean()),
                    "mean_radius_ms": float(radius.mean()),
                    "mean_absolute_error_ms": float(absolute_error.mean()),
                }
            )
    return pd.DataFrame(rows)


def _fit_paired_expert_calibrations(
    y_true: pd.Series | np.ndarray,
    temporal_prediction: pd.Series | np.ndarray,
    graph_prediction: pd.Series | np.ndarray,
) -> tuple[ExpertCalibration, ExpertCalibration]:
    """Fit both expert calibrations and their covariance on one block.

    Pairing is essential: covariance is meaningful only when both residuals
    refer to the same calibration target.  The Cauchy--Schwarz clipping is the
    same finite-precision safeguard used by the runtime fusion path.
    """

    truth = np.asarray(y_true, dtype=float)
    temporal = np.asarray(temporal_prediction, dtype=float)
    graph = np.asarray(graph_prediction, dtype=float)
    if not (len(truth) == len(temporal) == len(graph)):
        raise ValueError("paired expert calibration arrays must have equal length")

    temporal_calibration = fit_expert_calibration(truth, temporal)
    graph_calibration = fit_expert_calibration(truth, graph)
    temporal_centered_residual = (
        truth - temporal - temporal_calibration.residual_bias_ms
    )
    graph_centered_residual = truth - graph - graph_calibration.residual_bias_ms
    covariance = 0.0
    if len(truth) > 1:
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


def _run_matched_model_pair_audit(
    time_bins: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    forecast = build_forecast_table(
        time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=float(time_bins["bin_seconds"].iloc[0]),
    )
    train, val, test = split_train_val_test(forecast, 0.70, 0.15, 0.15)
    graph_train = add_graph_snapshot_features(train)
    graph_val = add_graph_snapshot_features(val)
    graph_test = add_graph_snapshot_features(test)
    temporal_features = default_feature_columns(train)
    graph_features = graph_context_feature_columns(graph_train)
    rows = []
    for model_name in (
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "small_mlp_regressor",
    ):
        temporal_model = fit_forecast_model(
            model_name, train, temporal_features
        )
        graph_model = fit_graph_context_model(
            model_name, graph_train, graph_features
        )
        temporal_calibration_prediction = temporal_model.predict(
            val[temporal_features].fillna(0.0)
        )
        graph_calibration_prediction = graph_model.predict(
            graph_val[graph_features].fillna(0.0)
        )
        temporal_calibration, graph_calibration = (
            _fit_paired_expert_calibrations(
                val["target_next"].to_numpy(dtype=float),
                temporal_calibration_prediction,
                graph_calibration_prediction,
            )
        )
        temporal_prediction = temporal_model.predict(
            test[temporal_features].fillna(0.0)
        )
        graph_prediction = graph_model.predict(
            graph_test[graph_features].fillna(0.0)
        )
        scored = add_calibrated_mixture_risk_scores(
            pd.DataFrame(
                {
                    "pred_forecast": temporal_prediction,
                    "pred_graph": graph_prediction,
                }
            ),
            temporal_calibration,
            graph_calibration,
        )
        truth = test["target_next"].to_numpy(dtype=float)
        temporal_error = np.abs(
            truth - scored["pred_temporal_calibrated"].to_numpy(dtype=float)
        )
        graph_error = np.abs(
            truth - scored["pred_graph_calibrated"].to_numpy(dtype=float)
        )
        fusion_error = np.abs(
            truth - scored["pred_calibrated_fusion"].to_numpy(dtype=float)
        )
        disagreement = scored["pred_disagreement"].to_numpy(dtype=float)
        corr = spearmanr(disagreement, fusion_error, nan_policy="omit")
        rows.append(
            {
                "model_family": model_name,
                "temporal_mae_ms": float(temporal_error.mean()),
                "graph_mae_ms": float(graph_error.mean()),
                "graph_to_temporal_mae_ratio": float(
                    graph_error.mean() / max(temporal_error.mean(), 1e-9)
                ),
                "fusion_mae_ms": float(fusion_error.mean()),
                "mean_disagreement_ms": float(disagreement.mean()),
                "spearman_disagreement_vs_fusion_error": float(corr.statistic),
                "spearman_p_value": float(corr.pvalue),
                "calibration_count": int(len(val)),
                "temporal_weight": float(
                    scored["temporal_expert_weight"].iloc[0]
                ),
                "graph_weight": float(scored["graph_expert_weight"].iloc[0]),
                "paired_residual_covariance_ms2": float(
                    scored["paired_residual_covariance_ms2"].iloc[0]
                ),
                "fusion_error_std_ms": float(
                    scored["pred_fusion_error_std"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_predictor_combination_audit(
    time_bins: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Evaluate all temporal/graph estimator-family combinations on one test split."""

    forecast = build_forecast_table(
        time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=float(time_bins["bin_seconds"].iloc[0]),
    )
    train, val, test = split_train_val_test(forecast, 0.70, 0.15, 0.15)
    graph_train = add_graph_snapshot_features(train)
    graph_val = add_graph_snapshot_features(val)
    graph_test = add_graph_snapshot_features(test)
    train_full = pd.concat([train, val], ignore_index=True)
    graph_train_full = pd.concat([graph_train, graph_val], ignore_index=True)
    temporal_features = default_feature_columns(train_full)
    graph_features = graph_context_feature_columns(graph_train_full)
    families = (
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "small_mlp_regressor",
    )
    temporal_predictions = {}
    graph_predictions = {}
    for family in families:
        temporal_model = fit_forecast_model(
            family,
            train_full,
            temporal_features,
        )
        graph_model = fit_graph_context_model(
            family,
            graph_train_full,
            graph_features,
        )
        temporal_predictions[family] = temporal_model.predict(
            test[temporal_features].fillna(0.0)
        )
        graph_predictions[family] = graph_model.predict(
            graph_test[graph_features].fillna(0.0)
        )

    truth = test["target_next"].to_numpy()
    rows = []
    for temporal_family, temporal_prediction in temporal_predictions.items():
        for graph_family, graph_prediction in graph_predictions.items():
            temporal_error = np.abs(truth - temporal_prediction)
            graph_error = np.abs(truth - graph_prediction)
            fusion_prediction = 0.5 * (temporal_prediction + graph_prediction)
            fusion_error = np.abs(truth - fusion_prediction)
            disagreement = np.abs(temporal_prediction - graph_prediction)
            corr = spearmanr(disagreement, fusion_error, nan_policy="omit")
            rows.append(
                {
                    "temporal_model_family": temporal_family,
                    "graph_model_family": graph_family,
                    "capacity_matched": temporal_family == graph_family,
                    "temporal_mae_ms": float(temporal_error.mean()),
                    "graph_mae_ms": float(graph_error.mean()),
                    "fusion_mae_ms": float(fusion_error.mean()),
                    "mean_disagreement_ms": float(disagreement.mean()),
                    "spearman_disagreement_vs_fusion_error": float(corr.statistic),
                    "spearman_p_value": float(corr.pvalue),
                }
            )
    return pd.DataFrame(rows)


def _write_validation_figures(
    output_dir: Path,
    ablation: pd.DataFrame,
    control_loop: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_pairs: pd.DataFrame,
    predictor_combinations: pd.DataFrame,
) -> None:
    """Render the four reviewer-facing diagnostics from their source tables."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    operational = ablation[
        ablation["scenario_name"].str.startswith("operational_")
    ].copy()
    pivot = operational.pivot(
        index="scenario_name",
        columns="policy_name",
        values="success_rate_under_60ms",
    )
    selected = [
        name
        for name in (
            "temporal_only",
            "graph_only",
            "simple_fusion",
            "disagreement_only",
            "learned_risk_without_disagreement",
            "ensemble_uncertainty",
            "full_gated_operational_rule",
        )
        if name in pivot
    ]
    ax = pivot[selected].plot(kind="bar", figsize=(11.5, 5.2), width=0.84)
    ax.set_title("Component attribution under injected degradation")
    ax.set_xlabel("Evaluation stress")
    ax.set_ylabel("Success rate under 60 ms")
    ax.legend(title="Policy variant", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "component_ablation.png", dpi=300, bbox_inches="tight")
    ax.figure.savefig(figures_dir / "component_ablation.pdf", bbox_inches="tight")
    plt.close(ax.figure)

    diagnostic_plot = diagnostics.copy()
    labels = diagnostic_plot["scenario_name"].str.replace("operational_", "", regex=False)
    x_positions = np.arange(len(diagnostic_plot))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        x_positions - 0.24,
        diagnostic_plot["high_error_detection_auroc"],
        width=0.24,
        label="Disagreement high-error AUROC",
        color="#2b6cb0",
    )
    ax.bar(
        x_positions,
        diagnostic_plot["ensemble_spread_high_error_auroc"],
        width=0.24,
        label="Ensemble-spread high-error AUROC",
        color="#2f855a",
    )
    ax.bar(
        x_positions + 0.24,
        diagnostic_plot["shared_failure_low_disagreement_rate"],
        width=0.24,
        label="Shared failures missed by low disagreement",
        color="#c53030",
    )
    ax.axhline(0.5, color="#4a5568", linestyle="--", linewidth=1, label="Random AUROC")
    ax.set_xticks(x_positions, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("When predictor disagreement is informative and when it fails")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "disagreement_diagnostics.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "disagreement_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    severe = control_loop[
        control_loop["scenario_name"].eq("operational_severe")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for policy_name, group in severe.groupby("policy_name", sort=False):
        ax.plot(
            group["control_loop_latency_ms"],
            group["success_rate_under_60ms"],
            marker="o",
            label=policy_name,
        )
    ax.set_xlabel("Collection + inference + dissemination delay (ms)")
    ax.set_ylabel("End-to-end success rate under 60 ms")
    ax.set_title("Control-loop latency sensitivity under severe degradation")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "control_loop_latency_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "control_loop_latency_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    family_labels = model_pairs["model_family"].str.replace("_", " ").str.title()
    x_positions = np.arange(len(model_pairs))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x_positions - 0.18, model_pairs["temporal_mae_ms"], width=0.36, label="Temporal view")
    ax.bar(x_positions + 0.18, model_pairs["graph_mae_ms"], width=0.36, label="Graph view")
    ax.set_xticks(x_positions, family_labels, rotation=15, ha="right")
    ax.set_ylabel("Test MAE (ms)")
    ax.set_title("Matched-capacity predictor-pair audit")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "matched_model_pair_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "matched_model_pair_audit.pdf", bbox_inches="tight")
    plt.close(fig)

    heatmap = predictor_combinations.pivot(
        index="temporal_model_family",
        columns="graph_model_family",
        values="spearman_disagreement_vs_fusion_error",
    )
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    image = ax.imshow(heatmap.to_numpy(), cmap="RdYlBu", vmin=-1.0, vmax=1.0)
    labels_x = [value.replace("_", " ").title() for value in heatmap.columns]
    labels_y = [value.replace("_", " ").title() for value in heatmap.index]
    ax.set_xticks(np.arange(len(labels_x)), labels_x, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels_y)), labels_y)
    ax.set_xlabel("Graph-context estimator")
    ax.set_ylabel("Temporal estimator")
    ax.set_title("Disagreement-error correlation across predictor combinations")
    for row_index in range(heatmap.shape[0]):
        for column_index in range(heatmap.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{heatmap.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(figures_dir / "predictor_combination_audit.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "predictor_combination_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--experiment-dir",
        default="results/service_path_reviewer_revision",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reviewer_validation",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    experiment_dir = _resolve(args.experiment_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(experiment_dir / "candidate_predictions.csv")
    latency_budget_ms = float(config["optimization"]["latency_budget_ms"])
    decision_window_seconds = float(config["graph"]["snapshot_seconds"])

    ablation = _run_ablation(candidates, latency_budget_ms)
    control_loop = _run_control_loop_sensitivity(
        candidates,
        latency_budget_ms,
        decision_window_seconds,
        [
            float(value)
            for value in config["optimization"].get(
                "control_loop_latency_ms", [0, 10, 50, 100, 500, 1000]
            )
        ],
    )
    stale_state = _run_stale_state_sensitivity(candidates, latency_budget_ms)
    diagnostics = _run_disagreement_diagnostics(candidates)
    risk_coverage = _run_empirical_risk_coverage(candidates)
    time_bins = load_time_bin_table(
        _resolve(config["dataset"]["time_bins_path"])
    )
    model_pairs = _run_matched_model_pair_audit(time_bins, config)
    predictor_combinations = _run_predictor_combination_audit(time_bins, config)

    ablation.to_csv(output_dir / "component_ablation.csv", index=False)
    control_loop.to_csv(output_dir / "control_loop_latency_sensitivity.csv", index=False)
    stale_state.to_csv(output_dir / "stale_state_sensitivity.csv", index=False)
    diagnostics.to_csv(output_dir / "disagreement_diagnostics.csv", index=False)
    risk_coverage.to_csv(output_dir / "empirical_risk_coverage.csv", index=False)
    model_pairs.to_csv(output_dir / "matched_model_pair_audit.csv", index=False)
    predictor_combinations.to_csv(
        output_dir / "predictor_combination_audit.csv",
        index=False,
    )
    _write_validation_figures(
        output_dir,
        ablation,
        control_loop,
        diagnostics,
        model_pairs,
        predictor_combinations,
    )
    metadata = {
        "evaluation_candidates": str(
            (experiment_dir / "candidate_predictions.csv").relative_to(REPO_ROOT)
        ),
        "control_loop_definition": (
            "state collection + controller inference + decision dissemination"
        ),
        "decision_window_seconds": decision_window_seconds,
        "structural_shift_source": (
            "controlled perturbations applied to evaluation rows only"
        ),
        "mathematical_score": (
            "deterministic covariance-aware linear pool with calibration-residual "
            "bias correction, constrained generalized-least-squares weights, and "
            "fixed-weight fusion-error variance; disagreement is a separate "
            "diagnostic and ranking feature"
        ),
        "external_dataset_status": (
            "independent Starlink IRTT measurements were parsed and evaluated; "
            "the single-endpoint archive supports predictor transfer analysis "
            "but not concurrent alternative-path policy generalization"
        ),
    }
    (output_dir / "reviewer_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"reviewer_validation_written={output_dir}")
    print(diagnostics.to_string(index=False))
    print(model_pairs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
