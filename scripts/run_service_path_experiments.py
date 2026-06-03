#!/usr/bin/env python3
"""Run paper-facing service path-selection experiments on open LEO measurements.

The script produces compact tables and figures for structural-shift evaluation:
normal split, temporal shift, site holdout, controlled operational degradation,
disagreement calibration, ablation, and runtime analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent, load_time_bin_table
from open_leo_latency_routing.evaluation.confidence_intervals import build_bootstrap_policy_intervals
from open_leo_latency_routing.evaluation.significance import build_paired_policy_significance
from open_leo_latency_routing.features.temporal import build_forecast_table, split_train_val_test
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.models.forecast_baselines import (
    default_feature_columns,
    fit_forecast_model,
    predict_forecast_model,
)
from open_leo_latency_routing.models.graph_baselines import fit_graph_xgb_model, predict_graph_model
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    SimpleFusionPolicyConfig,
    add_consensus_hybrid_scores,
    add_simple_fusion_scores,
    evaluate_decision_policies,
    summarize_multibin_decisions,
    summarize_stochastic_switching_costs,
    summarize_switching_costs,
)


plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


POLICY_COLUMNS = {
    "random": "random_score",
    "best_historical_path": "historical_mean_latency_ms",
    "highest_reply_availability": "reply_availability_score",
    "reactive_greedy": "latency_mean_ms",
    "predictive_greedy": "pred_forecast",
    "predictive_graph_greedy": "pred_graph",
    "predictive_simple_fusion_greedy": "pred_simple_fusion",
    "predictive_consensus_greedy": "pred_consensus",
    "ensemble_uncertainty_selector": "pred_ensemble_uncertainty",
    "conformal_uncertainty_selector": "pred_conformal_uncertainty",
    "disagreement_aware_selector": "pred_disagreement_aware",
}

DISPLAY_POLICIES = {
    "random": "Random",
    "best_historical_path": "Historical",
    "highest_reply_availability": "Reply availability",
    "reactive_greedy": "Reactive",
    "predictive_greedy": "Temporal",
    "predictive_graph_greedy": "Graph",
    "predictive_simple_fusion_greedy": "Fusion",
    "predictive_consensus_greedy": "Disagreement-aware",
    "ensemble_uncertainty_selector": "Ensemble uncertainty",
    "conformal_uncertainty_selector": "Conformal uncertainty",
    "disagreement_aware_selector": "Risk-adjusted",
}


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_sigmoid(values: pd.Series, scale: float) -> pd.Series:
    scaled = values.clip(lower=-60 * scale, upper=60 * scale) / max(scale, 1e-6)
    return 1.0 / (1.0 + np.exp(scaled))


def _assign_random_split_by_path(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a reproducible in-distribution split within each measured path."""

    rng = np.random.default_rng(random_state)
    split_frames: list[pd.DataFrame] = []
    for _, group in frame.groupby("relative_path", sort=False):
        work = group.copy()
        order = rng.permutation(len(work))
        train_end = max(1, int(len(work) * train_ratio))
        val_end = max(train_end + 1, int(len(work) * (train_ratio + val_ratio)))
        split = np.array(["test"] * len(work), dtype=object)
        split[order[:train_end]] = "train"
        split[order[train_end:val_end]] = "val"
        work["split"] = split
        split_frames.append(work)
    return pd.concat(split_frames, ignore_index=True)


def _split_frame_by_label(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        frame[frame["split"] == "train"].reset_index(drop=True),
        frame[frame["split"] == "val"].reset_index(drop=True),
        frame[frame["split"] == "test"].reset_index(drop=True),
    )


def _choose_holdout_locations(forecast_table: pd.DataFrame, holdout_count: int = 4) -> list[str]:
    """Select a compact, diverse held-out site set from available locations."""

    counts = forecast_table.groupby("location").size().sort_values(ascending=False)
    ordered = counts.index.tolist()
    if len(ordered) <= 1:
        return []

    effective_holdout_count = min(holdout_count, len(ordered) - 1)
    candidate_positions = np.linspace(0, len(ordered) - 1, effective_holdout_count, dtype=int)
    return [ordered[position] for position in candidate_positions]


def _apply_operational_shift(
    frame: pd.DataFrame,
    scenario_name: str,
    latency_spike_ms: float,
    reply_drop_fraction: float,
    affected_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Inject controlled network degradation into the evaluation split only."""

    rng = np.random.default_rng(seed)
    output = frame.copy()
    output["stress_scenario"] = scenario_name
    output["stress_applied"] = 0
    if output.empty:
        return output

    affected = rng.random(len(output)) < affected_fraction
    volatility = output["latency_std_ms"].fillna(0.0).to_numpy()
    spike = latency_spike_ms * (0.85 + 0.30 * rng.random(len(output))) + 0.20 * volatility

    for column, factor in [
        ("latency_mean_ms", 1.00),
        ("target_next", 1.00),
        ("latency_max_ms", 1.15),
        ("latency_std_ms", 0.35),
    ]:
        if column in output.columns:
            output.loc[affected, column] = output.loc[affected, column] + factor * spike[affected]

    if "observed_replies" in output.columns:
        output.loc[affected, "observed_replies"] = np.maximum(
            1,
            np.floor(output.loc[affected, "observed_replies"] * (1.0 - reply_drop_fraction)).astype(int),
        )
    output.loc[affected, "stress_applied"] = 1
    return output


def _historical_latency_map(train_frame: pd.DataFrame) -> tuple[dict[str, float], float]:
    values = train_frame.groupby("relative_path")["target_next"].mean().to_dict()
    global_mean = float(train_frame["target_next"].mean())
    return values, global_mean


def _fit_temporal_uncertainty_ensemble(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    n_members: int = 9,
    row_fraction: float = 0.82,
    feature_fraction: float = 0.78,
    random_state: int = 314,
) -> list[dict[str, object]]:
    """Fit small bootstrapped temporal models for an uncertainty baseline."""

    rng = np.random.default_rng(random_state)
    members: list[dict[str, object]] = []
    if train_frame.empty or not feature_columns:
        return members

    sample_size = max(2, int(len(train_frame) * row_fraction))
    feature_count = max(1, int(len(feature_columns) * feature_fraction))
    for _ in range(n_members):
        row_positions = rng.choice(len(train_frame), size=sample_size, replace=True)
        selected_features = sorted(rng.choice(feature_columns, size=feature_count, replace=False).tolist())
        model = fit_forecast_model(
            "linear_regression",
            train_frame.iloc[row_positions].reset_index(drop=True),
            selected_features,
        )
        members.append({"model": model, "feature_columns": selected_features})
    return members


def _predict_temporal_uncertainty_ensemble(
    ensemble_members: list[dict[str, object]],
    test_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return ensemble mean and spread for each candidate path."""

    if not ensemble_members:
        fallback = pd.Series(np.zeros(len(test_frame)), index=test_frame.index)
        return fallback, fallback

    predictions = []
    for member in ensemble_members:
        pred = predict_forecast_model(
            "linear_regression",
            member["model"],
            test_frame,
            member["feature_columns"],
        )["y_pred"].to_numpy()
        predictions.append(pred)
    matrix = np.vstack(predictions)
    return (
        pd.Series(matrix.mean(axis=0), index=test_frame.index),
        pd.Series(matrix.std(axis=0), index=test_frame.index),
    )


def _service_risk_ms(
    candidate: pd.DataFrame,
    reply_weight: float = 2.5,
    volatility_weight: float = 1.5,
) -> pd.Series:
    """Compute the lightweight service-risk penalty shared by uncertainty-aware policies."""

    reply_pressure = candidate.get("reply_pressure_score", pd.Series(0.0, index=candidate.index)).clip(lower=0.0)
    volatility = candidate["latency_std_ms"].fillna(0.0)
    volatility_scale = max(1.0, float(volatility.median()))
    return reply_weight * reply_pressure + volatility_weight * (volatility / volatility_scale)


def _fit_conformal_uncertainty_calibrator(
    temporal_model,
    validation_frame: pd.DataFrame,
    temporal_feature_columns: list[str],
    quantile: float = 0.90,
    n_bins: int = 4,
) -> dict[str, Any]:
    """Fit a simple split-conformal uncertainty calibrator on validation residuals.

    The uncertainty radius is conditioned on a lightweight online proxy built
    from latency volatility and reply pressure so the resulting selector varies
    meaningfully across candidate paths.
    """

    predictions = predict_forecast_model(
        "linear_regression",
        temporal_model,
        validation_frame,
        temporal_feature_columns,
    )
    work = validation_frame.reset_index(drop=True).copy()
    work["pred_forecast"] = predictions["y_pred"].to_numpy()
    work["abs_residual"] = (work["target_next"] - work["pred_forecast"]).abs()
    work["uncertainty_proxy"] = _service_risk_ms(work)
    proxy_values = work["uncertainty_proxy"].to_numpy(dtype=float)
    if len(proxy_values) == 0:
        return {
            "thresholds": np.array([], dtype=float),
            "radii": np.array([0.0], dtype=float),
            "global_radius": 0.0,
        }

    ranks = pd.Series(proxy_values).rank(method="first")
    bin_count = max(1, min(n_bins, len(work)))
    if bin_count == 1:
        work["proxy_bin"] = 0
        thresholds = np.array([], dtype=float)
    else:
        work["proxy_bin"] = pd.qcut(ranks, q=bin_count, labels=False, duplicates="drop")
        thresholds = np.quantile(proxy_values, q=np.linspace(0, 1, bin_count + 1)[1:-1])
    radii = (
        work.groupby("proxy_bin")["abs_residual"]
        .quantile(quantile)
        .sort_index()
        .to_numpy(dtype=float)
    )
    global_radius = float(work["abs_residual"].quantile(quantile))
    if len(radii) == 0:
        radii = np.array([global_radius], dtype=float)
    return {
        "thresholds": np.array(thresholds, dtype=float),
        "radii": np.array(radii, dtype=float),
        "global_radius": global_radius,
        "quantile": quantile,
    }


def _apply_conformal_uncertainty_selector(
    candidate: pd.DataFrame,
    calibrator: dict[str, Any],
    service_risk_ms: pd.Series,
) -> pd.DataFrame:
    """Add a conformalized uncertainty selector to the candidate table."""

    output = candidate.copy()
    proxy = _service_risk_ms(output)
    thresholds = np.asarray(calibrator.get("thresholds", []), dtype=float)
    radii = np.asarray(calibrator.get("radii", []), dtype=float)
    if len(radii) == 0:
        radii = np.array([float(calibrator.get("global_radius", 0.0))], dtype=float)
    bin_index = np.searchsorted(thresholds, proxy.to_numpy(dtype=float), side="right")
    bin_index = np.clip(bin_index, 0, len(radii) - 1)
    conformal_radius = pd.Series(radii[bin_index], index=output.index)
    output["pred_conformal_radius"] = conformal_radius
    output["pred_conformal_uncertainty"] = output["pred_forecast"] + conformal_radius + service_risk_ms
    return output


def _make_candidate_frame(
    test_frame: pd.DataFrame,
    graph_test: pd.DataFrame,
    temporal_model,
    graph_model,
    temporal_ensemble: list[dict[str, object]],
    conformal_calibrator: dict[str, Any],
    temporal_feature_columns: list[str],
    graph_feature_columns: list[str],
    historical_latency: dict[str, float],
    historical_fallback: float,
    consensus_config: ConsensusPolicyConfig,
    disagreement_temporal_weight: float,
    disagreement_graph_weight: float,
    disagreement_penalty: float,
    ensemble_penalty: float,
    service_risk_reply_weight: float,
    service_risk_volatility_weight: float,
) -> pd.DataFrame:
    """Merge predictions, current measurements, and policy scores."""

    test_frame = test_frame.reset_index(drop=True)
    graph_test = graph_test.reset_index(drop=True)
    temporal_predictions = predict_forecast_model(
        "linear_regression",
        temporal_model,
        test_frame,
        temporal_feature_columns,
    ).rename(columns={"y_pred": "pred_forecast"})
    graph_predictions = predict_graph_model(
        graph_model,
        graph_test,
        graph_feature_columns,
        model_name="graph_xgboost",
    ).rename(columns={"y_pred": "pred_graph"})
    ensemble_mean, ensemble_std = _predict_temporal_uncertainty_ensemble(temporal_ensemble, test_frame)

    meta_columns = [
        "relative_path",
        "measurement_family",
        "path_state",
        "location",
        "session_date",
        "session_bin_index",
        "bin_epoch",
        "bin_start_utc",
        "observed_replies",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_max_ms",
        "target_next",
        "target_cumulative_3",
        "target_mean_3",
        "target_cumulative_5",
        "target_mean_5",
    ]
    optional_columns = ["burst_indicator", "reply_pressure_score", "stress_scenario", "stress_applied"]
    selected_columns = [column for column in meta_columns + optional_columns if column in graph_test.columns]
    candidate = graph_test.reset_index(names="row_id")[selected_columns + ["row_id"]]
    candidate = (
        candidate.merge(temporal_predictions[["row_id", "pred_forecast"]], on="row_id", how="left")
        .merge(graph_predictions[["row_id", "pred_graph"]], on="row_id", how="left")
        .copy()
    )
    candidate["pred_ensemble_mean"] = ensemble_mean.to_numpy()
    candidate["pred_ensemble_std"] = ensemble_std.to_numpy()
    candidate["pred_disagreement"] = (candidate["pred_graph"] - candidate["pred_forecast"]).abs()
    candidate["historical_mean_latency_ms"] = (
        candidate["relative_path"].map(historical_latency).fillna(historical_fallback)
    )
    candidate["reply_availability_score"] = -candidate["observed_replies"].astype(float)
    candidate = add_simple_fusion_scores(
        candidate,
        config=SimpleFusionPolicyConfig(
            temporal_weight=consensus_config.temporal_weight,
            graph_weight=consensus_config.graph_weight,
        ),
    )
    candidate = add_consensus_hybrid_scores(candidate, config=consensus_config)
    service_risk_ms = _service_risk_ms(
        candidate,
        reply_weight=service_risk_reply_weight,
        volatility_weight=service_risk_volatility_weight,
    )
    candidate["pred_ensemble_uncertainty"] = (
        candidate["pred_ensemble_mean"] + ensemble_penalty * candidate["pred_ensemble_std"] + service_risk_ms
    )
    candidate = _apply_conformal_uncertainty_selector(candidate, conformal_calibrator, service_risk_ms)
    # The proposed selector keeps the stable temporal forecast as the anchor,
    # uses graph context lightly, and avoids paths with high model disagreement.
    candidate["pred_disagreement_aware"] = (
        disagreement_temporal_weight * candidate["pred_forecast"]
        + disagreement_graph_weight * candidate["pred_graph"]
        + disagreement_penalty * candidate["pred_disagreement"]
        + service_risk_ms
    )
    return candidate


def _fit_models(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    graph_train: pd.DataFrame,
    graph_val: pd.DataFrame,
    ensemble_members: int = 9,
    ensemble_row_fraction: float = 0.82,
    ensemble_feature_fraction: float = 0.78,
) -> tuple[object, object, list[dict[str, object]], list[str], list[str], dict[str, float], float]:
    train_full = pd.concat([train_frame, val_frame], ignore_index=True)
    graph_train_full = pd.concat([graph_train, graph_val], ignore_index=True)
    temporal_feature_columns = default_feature_columns(train_full)
    graph_feature_columns = default_feature_columns(graph_train_full)
    temporal_model = fit_forecast_model("linear_regression", train_full, temporal_feature_columns)
    graph_model = fit_graph_xgb_model(graph_train_full, graph_feature_columns)
    temporal_ensemble = _fit_temporal_uncertainty_ensemble(
        train_full,
        temporal_feature_columns,
        n_members=ensemble_members,
        row_fraction=ensemble_row_fraction,
        feature_fraction=ensemble_feature_fraction,
    )
    conformal_calibrator = _fit_conformal_uncertainty_calibrator(
        temporal_model,
        val_frame if not val_frame.empty else train_frame,
        temporal_feature_columns,
    )
    historical_latency, historical_fallback = _historical_latency_map(train_full)
    return (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
    )


def _evaluate_candidate_frame(
    scenario_name: str,
    candidate_frame: pd.DataFrame,
    latency_budget_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, decisions = evaluate_decision_policies(
        candidate_frame,
        latency_budget_ms=latency_budget_ms,
        policy_columns=POLICY_COLUMNS,
    )
    summary["scenario_name"] = scenario_name
    decisions["scenario_name"] = scenario_name

    significance = build_paired_policy_significance(
        decisions=decisions,
        comparisons=[
            ("risk_adjusted_vs_temporal", "disagreement_aware_selector", "predictive_greedy"),
            ("risk_adjusted_vs_graph", "disagreement_aware_selector", "predictive_graph_greedy"),
            ("risk_adjusted_vs_fusion", "disagreement_aware_selector", "predictive_simple_fusion_greedy"),
            ("risk_adjusted_vs_ensemble_uncertainty", "disagreement_aware_selector", "ensemble_uncertainty_selector"),
            ("risk_adjusted_vs_conformal_uncertainty", "disagreement_aware_selector", "conformal_uncertainty_selector"),
            ("risk_adjusted_vs_historical", "disagreement_aware_selector", "best_historical_path"),
            ("graph_vs_temporal", "predictive_graph_greedy", "predictive_greedy"),
        ],
        metric_columns=[
            "realized_next_latency_ms",
            "decision_gap_ms",
            "success_under_budget",
            "realized_cumulative_latency_3_ms",
            "cumulative_decision_gap_3_ms",
            "realized_cumulative_latency_5_ms",
            "cumulative_decision_gap_5_ms",
        ],
    )
    significance["scenario_name"] = scenario_name
    return summary, decisions, significance


def _build_calibration_tables(
    scenario_candidates: dict[str, pd.DataFrame],
    latency_budget_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for scenario_name, frame in scenario_candidates.items():
        work = frame.copy()
        if work["pred_disagreement"].nunique() < 3:
            work["disagreement_bin"] = "Single"
        else:
            work["_rank"] = work["pred_disagreement"].rank(method="first")
            work["disagreement_bin"] = pd.qcut(work["_rank"], q=3, labels=["Low", "Medium", "High"])
        work["risk_adjusted_abs_error_ms"] = (work["pred_disagreement_aware"] - work["target_next"]).abs()
        work["temporal_abs_error_ms"] = (work["pred_forecast"] - work["target_next"]).abs()
        work["graph_abs_error_ms"] = (work["pred_graph"] - work["target_next"]).abs()
        work["service_failure"] = (work["target_next"] > latency_budget_ms).astype(int)

        scale = max(5.0, float(work["target_next"].std()))
        work["predicted_success_probability"] = _safe_sigmoid(work["pred_disagreement_aware"] - latency_budget_ms, scale)
        work["probability_bin"] = pd.cut(
            work["predicted_success_probability"],
            bins=np.linspace(0.0, 1.0, 6),
            include_lowest=True,
        )

        for bin_name, group in work.groupby("disagreement_bin", observed=True):
            bin_rows.append(
                {
                    "scenario_name": scenario_name,
                    "disagreement_bin": str(bin_name),
                    "candidate_count": len(group),
                    "mean_disagreement_ms": float(group["pred_disagreement"].mean()),
                    "mean_risk_adjusted_abs_error_ms": float(group["risk_adjusted_abs_error_ms"].mean()),
                    "mean_temporal_abs_error_ms": float(group["temporal_abs_error_ms"].mean()),
                    "mean_graph_abs_error_ms": float(group["graph_abs_error_ms"].mean()),
                    "service_failure_rate": float(group["service_failure"].mean()),
                }
            )

        prob_bins = (
            work.groupby("probability_bin", observed=True)
            .agg(
                bin_count=("predicted_success_probability", "size"),
                mean_probability=("predicted_success_probability", "mean"),
                observed_success=("service_failure", lambda values: 1.0 - float(np.mean(values))),
            )
            .reset_index()
        )
        ece = float(
            (
                (prob_bins["bin_count"] / max(1, len(work)))
                * (prob_bins["mean_probability"] - prob_bins["observed_success"]).abs()
            ).sum()
        )
        brier = float(((work["predicted_success_probability"] - (1 - work["service_failure"])) ** 2).mean())
        corr = spearmanr(work["pred_disagreement"], work["risk_adjusted_abs_error_ms"], nan_policy="omit")
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "spearman_disagreement_error": float(corr.statistic) if not math.isnan(corr.statistic) else 0.0,
                "spearman_p_value": float(corr.pvalue) if not math.isnan(corr.pvalue) else 1.0,
                "ece": ece,
                "brier_score": brier,
                "candidate_count": len(work),
            }
        )

    return pd.DataFrame(bin_rows), pd.DataFrame(summary_rows)


def _build_stratified_disagreement_analysis(
    scenario_candidates: dict[str, pd.DataFrame],
    latency_budget_ms: float,
) -> pd.DataFrame:
    """Stratify disagreement-error behavior by plausible online covariates."""

    rows: list[dict[str, object]] = []
    for scenario_name, frame in scenario_candidates.items():
        work = frame.copy()
        work["abs_temporal_error_ms"] = (work["pred_forecast"] - work["target_next"]).abs()
        work["service_failure"] = (work["target_next"] > latency_budget_ms).astype(int)
        if "bin_start_utc" in work.columns:
            hours = pd.to_datetime(work["bin_start_utc"], errors="coerce").dt.hour.fillna(0).astype(int)
            work["time_of_day_stratum"] = np.where(hours.between(6, 17), "day", "night")
        else:
            work["time_of_day_stratum"] = "unknown"
        strata_defs = {
            "reply_pressure": np.where(
                work["reply_pressure_score"].fillna(0.0) >= work["reply_pressure_score"].fillna(0.0).median(),
                "high",
                "low",
            ),
            "volatility": np.where(
                work["latency_std_ms"].fillna(0.0) >= work["latency_std_ms"].fillna(0.0).median(),
                "high",
                "low",
            ),
            "time_of_day": work["time_of_day_stratum"],
        }
        if "location" in work.columns:
            location_counts = work["location"].value_counts()
            frequent_locations = location_counts[location_counts >= max(8, int(location_counts.median()))].index.tolist()
            if frequent_locations:
                strata_defs["site"] = np.where(work["location"].isin(frequent_locations), work["location"], "other")
        for stratum_name, labels in strata_defs.items():
            work[f"{stratum_name}_stratum"] = labels
            for level_name, group in work.groupby(f"{stratum_name}_stratum", sort=False):
                if len(group) < 3:
                    continue
                corr = spearmanr(group["pred_disagreement"], group["abs_temporal_error_ms"], nan_policy="omit")
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "stratifier": stratum_name,
                        "stratum_level": str(level_name),
                        "candidate_count": len(group),
                        "mean_disagreement_ms": float(group["pred_disagreement"].mean()),
                        "mean_abs_temporal_error_ms": float(group["abs_temporal_error_ms"].mean()),
                        "service_failure_rate": float(group["service_failure"].mean()),
                        "spearman_disagreement_error": float(corr.statistic) if not math.isnan(corr.statistic) else 0.0,
                        "spearman_p_value": float(corr.pvalue) if not math.isnan(corr.pvalue) else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _build_dataset_summary(time_bins: pd.DataFrame, forecast_table: pd.DataFrame, holdout_locations: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "measurement_rows": len(time_bins),
                "forecast_rows": len(forecast_table),
                "locations": time_bins["location"].nunique(),
                "paths": time_bins["relative_path"].nunique(),
                "date_start": str(time_bins["session_date"].min().date()),
                "date_end": str(time_bins["session_date"].max().date()),
                "held_out_locations": ", ".join(holdout_locations),
            }
        ]
    )


def _save_plot(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _plot_shift_performance(summary: pd.DataFrame, fig_path: Path) -> None:
    selected = [
        "best_historical_path",
        "predictive_greedy",
        "predictive_graph_greedy",
        "conformal_uncertainty_selector",
        "ensemble_uncertainty_selector",
        "disagreement_aware_selector",
    ]
    frame = summary[summary["policy_name"].isin(selected)].copy()
    scenario_order = [
        "in_distribution",
        "temporal_shift",
        "site_holdout",
        "operational_mild",
        "operational_moderate",
        "operational_severe",
    ]
    scenario_order = [name for name in scenario_order if name in frame["scenario_name"].unique()]
    policy_labels = {name: DISPLAY_POLICIES[name] for name in selected}
    success = frame.pivot(index="scenario_name", columns="policy_name", values="success_rate_under_60ms")
    gap = frame.pivot(index="scenario_name", columns="policy_name", values="mean_decision_gap_ms")
    success = success.loc[scenario_order, selected].rename(columns=policy_labels)
    gap = gap.loc[scenario_order, selected].rename(columns=policy_labels)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharex=True)
    success.plot(kind="line", marker="o", ax=axes[0])
    axes[0].set_title("QoS Success Under Shift")
    axes[0].set_ylabel("Success rate")
    axes[0].set_xlabel("")
    axes[0].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[0].legend().remove()

    gap.plot(kind="line", marker="o", ax=axes[1])
    axes[1].set_title("Decision Gap Under Shift")
    axes[1].set_ylabel("Decision gap (ms)")
    axes[1].set_xlabel("")
    axes[1].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    _save_plot(fig_path)


def _plot_operational_severity(summary: pd.DataFrame, fig_path: Path) -> None:
    scenarios = ["operational_mild", "operational_moderate", "operational_severe"]
    selected = [
        "predictive_greedy",
        "predictive_graph_greedy",
        "predictive_simple_fusion_greedy",
        "ensemble_uncertainty_selector",
        "disagreement_aware_selector",
    ]
    frame = summary[summary["scenario_name"].isin(scenarios) & summary["policy_name"].isin(selected)].copy()
    if frame.empty:
        return
    latency = frame.pivot(index="scenario_name", columns="policy_name", values="p95_realized_latency_ms")
    latency = latency.loc[scenarios, selected].rename(columns={name: DISPLAY_POLICIES[name] for name in selected})
    latency.index = ["Mild", "Moderate", "Severe"]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    latency.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Tail Latency Under Operational Degradation")
    ax.set_ylabel("95th percentile latency (ms)")
    ax.set_xlabel("Degradation severity")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    _save_plot(fig_path)


def _plot_calibration(calibration_bins: pd.DataFrame, fig_path: Path) -> None:
    frame = calibration_bins[calibration_bins["scenario_name"].isin(["in_distribution", "temporal_shift", "site_holdout", "operational_severe"])].copy()
    if frame.empty:
        return
    order = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for scenario_name, group in frame.groupby("scenario_name", sort=False):
        group = group.set_index("disagreement_bin").reindex(order).dropna().reset_index()
        axes[0].plot(group["disagreement_bin"], group["mean_risk_adjusted_abs_error_ms"], marker="o", label=scenario_name)
        axes[1].plot(group["disagreement_bin"], group["service_failure_rate"], marker="o", label=scenario_name)
    axes[0].set_title("Prediction Error by Disagreement")
    axes[0].set_ylabel("Absolute error (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].set_title("Service Failure by Disagreement")
    axes[1].set_ylabel("Failure rate")
    axes[1].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    _save_plot(fig_path)


def _plot_ablation_runtime(summary: pd.DataFrame, fig_path: Path) -> None:
    selected = [
        "predictive_greedy",
        "predictive_graph_greedy",
        "predictive_simple_fusion_greedy",
        "ensemble_uncertainty_selector",
        "disagreement_aware_selector",
    ]
    base = summary[(summary["scenario_name"] == "temporal_shift") & summary["policy_name"].isin(selected)].copy()
    base = base.set_index("policy_name").loc[selected].reset_index()
    labels = [DISPLAY_POLICIES[name] for name in selected]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axes[0].bar(labels, base["mean_decision_gap_ms"], color=["#718096", "#2b6cb0", "#dd6b20", "#805ad5", "#2f855a"])
    axes[0].set_title("Ablation: Decision Gap")
    axes[0].set_ylabel("Decision gap (ms)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].bar(labels, base["mean_decision_time_us"], color=["#718096", "#2b6cb0", "#dd6b20", "#805ad5", "#2f855a"])
    axes[1].set_title("Runtime Per Decision")
    axes[1].set_ylabel("Runtime (microseconds)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    _save_plot(fig_path)


def _write_markdown_summary(
    path: Path,
    dataset_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    switching_summary: pd.DataFrame,
    stochastic_switching_summary: pd.DataFrame,
    multibin_summary: pd.DataFrame,
    stratified_analysis: pd.DataFrame,
) -> None:
    def to_markdown(frame: pd.DataFrame) -> str:
        """Render a small DataFrame as Markdown without optional dependencies."""

        text_frame = frame.copy().astype(str)
        headers = list(text_frame.columns)
        rows = text_frame.values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = [
        "# Service Path Experiment Summary",
        "",
        "This report summarizes the structural-shift experiments for disagreement-aware LEO service path selection.",
        "",
        "## Dataset",
        "",
        to_markdown(dataset_summary),
        "",
        "## Main Policy Results",
        "",
        to_markdown(
            policy_summary[
                [
                    "scenario_name",
                    "policy_name",
                    "decision_count",
                    "mean_realized_latency_ms",
                    "mean_decision_gap_ms",
                    "success_rate_under_60ms",
                    "p95_realized_latency_ms",
                    "mean_decision_time_us",
                ]
            ]
        ),
        "",
        "## Disagreement Calibration",
        "",
        to_markdown(calibration_summary),
        "",
        "## Switching-Cost Summary",
        "",
        to_markdown(
            switching_summary[
                [
                    "scenario_name",
                    "policy_name",
                    "switch_penalty_ms",
                    "switch_rate",
                    "mean_penalized_latency_ms",
                    "mean_penalized_decision_gap_ms",
                    "success_rate_under_60ms",
                ]
            ].head(20)
        ) if not switching_summary.empty else "No switching-cost rows generated.",
        "",
        "## Multi-Bin Summary",
        "",
        to_markdown(multibin_summary.head(20)) if not multibin_summary.empty else "No multi-bin rows generated.",
        "",
        "## Stochastic Switching Summary",
        "",
        to_markdown(
            stochastic_switching_summary[
                [
                    "scenario_name",
                    "policy_name",
                    "switch_rate",
                    "base_penalty_ms",
                    "spike_penalty_ms",
                    "spike_probability",
                    "mean_penalized_latency_ms",
                    "mean_penalized_latency_ci_low_ms",
                    "mean_penalized_latency_ci_high_ms",
                ]
            ].head(20)
        ) if not stochastic_switching_summary.empty else "No stochastic switching rows generated.",
        "",
        "## Stratified Disagreement Analysis",
        "",
        to_markdown(stratified_analysis.head(20)) if not stratified_analysis.empty else "No stratified rows generated.",
        "",
        "The post-measurement reference path is used only for computing decision gap. It is not a deployable online method.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument("--output-dir", default="results/service_path_experiments")
    parser.add_argument("--holdout-count", type=int, default=4)
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    output_dir = _resolve_repo_path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)
    horizon_seconds = int(config["forecasting"]["horizon_seconds"])
    snapshot_seconds = int(config["graph"]["snapshot_seconds"])
    horizon_bins = max(1, horizon_seconds // snapshot_seconds) if horizon_seconds >= snapshot_seconds else 1
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))
    consensus_cfg = config["optimization"].get("consensus", {})
    disagreement_cfg = config["optimization"].get("disagreement_aware", {})
    ensemble_cfg = config["optimization"].get("ensemble_uncertainty", {})
    service_risk_cfg = config["optimization"].get("service_risk", {})
    switching_penalties_ms = list(config["optimization"].get("switching_penalties_ms", [5.0, 10.0, 20.0]))
    stochastic_switching_cfg = config["optimization"].get("stochastic_switching", {})
    multi_bin_horizons = list(config["optimization"].get("multi_bin_horizons", [3, 5]))
    consensus_config = ConsensusPolicyConfig(
        temporal_weight=float(consensus_cfg.get("temporal_weight", 0.65)),
        graph_weight=float(consensus_cfg.get("graph_weight", 0.35)),
        disagreement_penalty=float(consensus_cfg.get("disagreement_penalty", 0.30)),
    )
    disagreement_temporal_weight = float(disagreement_cfg.get("temporal_weight", 0.85))
    disagreement_graph_weight = float(disagreement_cfg.get("graph_weight", 0.15))
    disagreement_penalty = float(disagreement_cfg.get("disagreement_penalty", 0.60))
    ensemble_penalty = float(ensemble_cfg.get("lambda_ens", 0.75))
    ensemble_members = int(ensemble_cfg.get("ensemble_members", 9))
    ensemble_row_fraction = float(ensemble_cfg.get("row_fraction", 0.82))
    ensemble_feature_fraction = float(ensemble_cfg.get("feature_fraction", 0.78))
    service_risk_reply_weight = float(service_risk_cfg.get("reply_pressure_penalty", 2.5))
    service_risk_volatility_weight = float(service_risk_cfg.get("volatility_penalty", 1.5))

    forecast_table = build_forecast_table(
        time_bins=time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
    )
    holdout_locations = _choose_holdout_locations(forecast_table, holdout_count=args.holdout_count)
    dataset_summary = _build_dataset_summary(time_bins, forecast_table, holdout_locations)

    scenario_candidates: dict[str, pd.DataFrame] = {}
    policy_summaries: list[pd.DataFrame] = []
    policy_decisions: list[pd.DataFrame] = []
    significance_rows: list[pd.DataFrame] = []

    # Scenario 1: in-distribution random split.
    random_split = _assign_random_split_by_path(
        forecast_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
    )
    random_graph = add_graph_snapshot_features(random_split)
    train, val, test = _split_frame_by_label(random_split)
    graph_train, graph_val, graph_test = _split_frame_by_label(random_graph)
    models = _fit_models(
        train,
        val,
        graph_train,
        graph_val,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
    )
    candidate = _make_candidate_frame(
        test,
        graph_test,
        *models,
        consensus_config=consensus_config,
        disagreement_temporal_weight=disagreement_temporal_weight,
        disagreement_graph_weight=disagreement_graph_weight,
        disagreement_penalty=disagreement_penalty,
        ensemble_penalty=ensemble_penalty,
        service_risk_reply_weight=service_risk_reply_weight,
        service_risk_volatility_weight=service_risk_volatility_weight,
    )
    scenario_candidates["in_distribution"] = candidate

    # Scenario 2: temporal structural shift, trained on earlier bins and tested on later bins.
    temporal_train, temporal_val, temporal_test = split_train_val_test(
        forecast_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    temporal_graph = add_graph_snapshot_features(forecast_table)
    temporal_graph_train, temporal_graph_val, temporal_graph_test = split_train_val_test(
        temporal_graph,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    temporal_models = _fit_models(
        temporal_train,
        temporal_val,
        temporal_graph_train,
        temporal_graph_val,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
    )
    temporal_candidate = _make_candidate_frame(
        temporal_test,
        temporal_graph_test,
        *temporal_models,
        consensus_config=consensus_config,
        disagreement_temporal_weight=disagreement_temporal_weight,
        disagreement_graph_weight=disagreement_graph_weight,
        disagreement_penalty=disagreement_penalty,
        ensemble_penalty=ensemble_penalty,
        service_risk_reply_weight=service_risk_reply_weight,
        service_risk_volatility_weight=service_risk_volatility_weight,
    )
    scenario_candidates["temporal_shift"] = temporal_candidate

    # Scenario 3: site holdout, trained without selected locations.
    site_train_table = forecast_table[~forecast_table["location"].isin(holdout_locations)].reset_index(drop=True)
    site_test = forecast_table[forecast_table["location"].isin(holdout_locations)].reset_index(drop=True)
    site_train, site_val, _ = split_train_val_test(
        site_train_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    site_graph_train_table = add_graph_snapshot_features(site_train_table)
    site_graph_test = add_graph_snapshot_features(site_test)
    site_graph_train, site_graph_val, _ = split_train_val_test(
        site_graph_train_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    site_models = _fit_models(
        site_train,
        site_val,
        site_graph_train,
        site_graph_val,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
    )
    site_candidate = _make_candidate_frame(
        site_test,
        site_graph_test,
        *site_models,
        consensus_config=consensus_config,
        disagreement_temporal_weight=disagreement_temporal_weight,
        disagreement_graph_weight=disagreement_graph_weight,
        disagreement_penalty=disagreement_penalty,
        ensemble_penalty=ensemble_penalty,
        service_risk_reply_weight=service_risk_reply_weight,
        service_risk_volatility_weight=service_risk_volatility_weight,
    )
    scenario_candidates["site_holdout"] = site_candidate

    # Scenarios 4-6: controlled operational shifts on the temporal test split.
    severity_settings = {
        "operational_mild": (15.0, 0.10, 0.30),
        "operational_moderate": (30.0, 0.25, 0.45),
        "operational_severe": (50.0, 0.40, 0.60),
    }
    for seed_offset, (scenario_name, settings) in enumerate(severity_settings.items(), start=1):
        shifted_test = _apply_operational_shift(
            temporal_test,
            scenario_name=scenario_name,
            latency_spike_ms=settings[0],
            reply_drop_fraction=settings[1],
            affected_fraction=settings[2],
            seed=100 + seed_offset,
        )
        shifted_graph_test = add_graph_snapshot_features(shifted_test)
        shifted_candidate = _make_candidate_frame(
            shifted_test,
            shifted_graph_test,
            *temporal_models,
            consensus_config=consensus_config,
            disagreement_temporal_weight=disagreement_temporal_weight,
            disagreement_graph_weight=disagreement_graph_weight,
            disagreement_penalty=disagreement_penalty,
            ensemble_penalty=ensemble_penalty,
            service_risk_reply_weight=service_risk_reply_weight,
            service_risk_volatility_weight=service_risk_volatility_weight,
        )
        scenario_candidates[scenario_name] = shifted_candidate

    for scenario_name, candidate_frame in scenario_candidates.items():
        summary, decisions, significance = _evaluate_candidate_frame(
            scenario_name,
            candidate_frame,
            latency_budget_ms=latency_budget_ms,
        )
        policy_summaries.append(summary)
        policy_decisions.append(decisions)
        significance_rows.append(significance)

    policy_summary = pd.concat(policy_summaries, ignore_index=True)
    decision_results = pd.concat(policy_decisions, ignore_index=True)
    significance = pd.concat(significance_rows, ignore_index=True)
    calibration_bins, calibration_summary = _build_calibration_tables(scenario_candidates, latency_budget_ms)
    stratified_analysis = _build_stratified_disagreement_analysis(scenario_candidates, latency_budget_ms)
    switching_summary = summarize_switching_costs(
        decision_results,
        penalty_levels_ms=switching_penalties_ms,
        latency_budget_ms=latency_budget_ms,
    )
    stochastic_switching_summary = summarize_stochastic_switching_costs(
        decision_results,
        base_penalty_ms=float(stochastic_switching_cfg.get("base_penalty_ms", 10.0)),
        spike_penalty_ms=float(stochastic_switching_cfg.get("spike_penalty_ms", 75.0)),
        spike_probability=float(stochastic_switching_cfg.get("spike_probability", 0.10)),
        n_trials=int(stochastic_switching_cfg.get("n_trials", 256)),
        latency_budget_ms=latency_budget_ms,
        random_state=42,
    )
    multibin_summary = summarize_multibin_decisions(
        decision_results,
        horizons=multi_bin_horizons,
    )
    ci_frames: list[pd.DataFrame] = []
    for scenario_name, scenario_decisions in decision_results.groupby("scenario_name", sort=False):
        scenario_ci = build_bootstrap_policy_intervals(
            scenario_decisions,
            metric_columns=[
                "realized_next_latency_ms",
                "decision_gap_ms",
                "success_under_budget",
                "retrospective_best_path_match",
                "realized_cumulative_latency_3_ms",
                "cumulative_decision_gap_3_ms",
                "realized_cumulative_latency_5_ms",
                "cumulative_decision_gap_5_ms",
            ],
            n_bootstrap=2000,
            random_state=42,
        )
        scenario_ci["scenario_name"] = scenario_name
        ci_frames.append(scenario_ci)
    confidence_intervals = pd.concat(ci_frames, ignore_index=True)

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)
    decision_results.to_csv(output_dir / "policy_decisions.csv", index=False)
    significance.to_csv(output_dir / "policy_significance.csv", index=False)
    calibration_bins.to_csv(output_dir / "disagreement_calibration_bins.csv", index=False)
    calibration_summary.to_csv(output_dir / "disagreement_calibration_summary.csv", index=False)
    stratified_analysis.to_csv(output_dir / "stratified_disagreement_analysis.csv", index=False)
    switching_summary.to_csv(output_dir / "switching_cost_summary.csv", index=False)
    stochastic_switching_summary.to_csv(output_dir / "stochastic_switching_summary.csv", index=False)
    multibin_summary.to_csv(output_dir / "multi_bin_summary.csv", index=False)
    confidence_intervals.to_csv(output_dir / "policy_confidence_intervals.csv", index=False)

    metadata = {
        "latency_budget_ms": latency_budget_ms,
        "holdout_locations": holdout_locations,
        "policy_columns": POLICY_COLUMNS,
        "consensus_config": asdict(consensus_config),
        "disagreement_aware_config": {
            "temporal_weight": disagreement_temporal_weight,
            "graph_weight": disagreement_graph_weight,
            "disagreement_penalty": disagreement_penalty,
        },
        "ensemble_uncertainty_config": {
            "lambda_ens": ensemble_penalty,
            "ensemble_members": ensemble_members,
            "row_fraction": ensemble_row_fraction,
            "feature_fraction": ensemble_feature_fraction,
        },
        "service_risk_config": {
            "reply_pressure_penalty": service_risk_reply_weight,
            "volatility_penalty": service_risk_volatility_weight,
        },
        "switching_penalties_ms": switching_penalties_ms,
        "stochastic_switching": {
            "base_penalty_ms": float(stochastic_switching_cfg.get("base_penalty_ms", 10.0)),
            "spike_penalty_ms": float(stochastic_switching_cfg.get("spike_penalty_ms", 75.0)),
            "spike_probability": float(stochastic_switching_cfg.get("spike_probability", 0.10)),
            "n_trials": int(stochastic_switching_cfg.get("n_trials", 256)),
            "random_state": 42,
        },
        "multi_bin_horizons": multi_bin_horizons,
        "config_path": _resolve_repo_path(args.config).relative_to(REPO_ROOT).as_posix(),
        "time_bins_path": time_bins_path.relative_to(REPO_ROOT).as_posix(),
        "split_random_seeds": {
            "in_distribution_split": 42,
            "operational_mild": 101,
            "operational_moderate": 102,
            "operational_severe": 103,
            "bootstrap_confidence_intervals": 42,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _plot_shift_performance(policy_summary, figures_dir / "shift_policy_success_and_gap.png")
    _plot_operational_severity(policy_summary, figures_dir / "operational_severity_tail_latency.png")
    _plot_calibration(calibration_bins, figures_dir / "disagreement_calibration.png")
    _plot_ablation_runtime(policy_summary, figures_dir / "ablation_and_runtime.png")
    _write_markdown_summary(
        output_dir / "service_path_results_summary.md",
        dataset_summary,
        policy_summary,
        calibration_summary,
        switching_summary,
        stochastic_switching_summary,
        multibin_summary,
        stratified_analysis,
    )

    print(f"outputs_written={output_dir}")
    print(f"figures_written={figures_dir}")
    print(dataset_summary.to_string(index=False))
    print(
        policy_summary[
            policy_summary["policy_name"].isin(
                ["best_historical_path", "predictive_greedy", "predictive_graph_greedy", "disagreement_aware_selector"]
            )
        ][
            [
                "scenario_name",
                "policy_name",
                "mean_realized_latency_ms",
                "mean_decision_gap_ms",
                "success_rate_under_60ms",
                "p95_realized_latency_ms",
                "switch_rate",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
