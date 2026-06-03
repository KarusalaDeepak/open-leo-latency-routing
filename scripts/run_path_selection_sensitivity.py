#!/usr/bin/env python3
"""Run extended robustness analyses for the open LEO service-path paper.

This script keeps heavier appendix-style sweeps separate from the core
pipeline while reusing the same data processing and path-selection logic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import load_time_bin_table
from open_leo_latency_routing.features.temporal import build_forecast_table, split_train_val_test
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.models.forecast_baselines import (
    default_feature_columns,
    evaluate_prediction_frame,
    fit_forecast_model,
    predict_forecast_model,
)
from open_leo_latency_routing.optimization.policies import evaluate_decision_policies

import run_service_path_experiments as service_exp


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path_value


def _prepare_base_scenarios(
    forecast_table: pd.DataFrame,
    config: dict,
    holdout_count: int,
    ensemble_members: int,
    ensemble_row_fraction: float,
    ensemble_feature_fraction: float,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Build the key structural-shift candidate tables used in the paper."""

    holdout_locations = service_exp._choose_holdout_locations(forecast_table, holdout_count=holdout_count)
    consensus_cfg = config["optimization"].get("consensus", {})
    consensus_config = service_exp.ConsensusPolicyConfig(
        temporal_weight=float(consensus_cfg.get("temporal_weight", 0.65)),
        graph_weight=float(consensus_cfg.get("graph_weight", 0.35)),
        disagreement_penalty=float(consensus_cfg.get("disagreement_penalty", 0.30)),
    )
    disagreement_cfg = config["optimization"].get("disagreement_aware", {})
    ensemble_cfg = config["optimization"].get("ensemble_uncertainty", {})
    service_risk_cfg = config["optimization"].get("service_risk", {})

    disagreement_temporal_weight = float(disagreement_cfg.get("temporal_weight", 0.85))
    disagreement_graph_weight = float(disagreement_cfg.get("graph_weight", 0.15))
    disagreement_penalty = float(disagreement_cfg.get("disagreement_penalty", 0.60))
    ensemble_penalty = float(ensemble_cfg.get("lambda_ens", 0.75))
    service_risk_reply_weight = float(service_risk_cfg.get("reply_pressure_penalty", 2.5))
    service_risk_volatility_weight = float(service_risk_cfg.get("volatility_penalty", 1.5))

    scenario_candidates: dict[str, pd.DataFrame] = {}

    # Temporal shift scenario.
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
    temporal_models = service_exp._fit_models(
        temporal_train,
        temporal_val,
        temporal_graph_train,
        temporal_graph_val,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
    )
    scenario_candidates["temporal_shift"] = service_exp._make_candidate_frame(
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

    # Site holdout scenario.
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
    site_models = service_exp._fit_models(
        site_train,
        site_val,
        site_graph_train,
        site_graph_val,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
    )
    scenario_candidates["site_holdout"] = service_exp._make_candidate_frame(
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

    # Operational moderate and severe shifts reusing temporal models.
    severity_settings = {
        "operational_moderate": (30.0, 0.25, 0.45),
        "operational_severe": (50.0, 0.40, 0.60),
    }
    for seed_offset, (scenario_name, settings) in enumerate(severity_settings.items(), start=1):
        shifted_test = service_exp._apply_operational_shift(
            temporal_test,
            scenario_name=scenario_name,
            latency_spike_ms=settings[0],
            reply_drop_fraction=settings[1],
            affected_fraction=settings[2],
            seed=200 + seed_offset,
        )
        shifted_graph_test = add_graph_snapshot_features(shifted_test)
        scenario_candidates[scenario_name] = service_exp._make_candidate_frame(
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

    return scenario_candidates, holdout_locations


def _evaluate_single_score(candidate: pd.DataFrame, score_column: str, latency_budget_ms: float) -> pd.DataFrame:
    summary, _ = evaluate_decision_policies(
        candidate,
        latency_budget_ms=latency_budget_ms,
        policy_columns={score_column: score_column},
    )
    return summary


def _run_weight_sensitivity(scenario_candidates: dict[str, pd.DataFrame], latency_budget_ms: float) -> pd.DataFrame:
    rows = []
    for scenario_name, candidate in scenario_candidates.items():
        service_risk = service_exp._service_risk_ms(candidate)
        for beta in [0.60, 0.70, 0.80, 0.85, 0.90]:
            score = (
                beta * candidate["pred_forecast"]
                + (1.0 - beta) * candidate["pred_graph"]
                + 0.60 * candidate["pred_disagreement"]
                + service_risk
            )
            work = candidate.copy()
            work["sweep_score"] = score
            summary = _evaluate_single_score(work, "sweep_score", latency_budget_ms)
            summary["scenario_name"] = scenario_name
            summary["parameter_name"] = "beta"
            summary["parameter_value"] = beta
            rows.append(summary)
        for lam in [0.20, 0.40, 0.60, 0.80, 1.00]:
            score = (
                0.85 * candidate["pred_forecast"]
                + 0.15 * candidate["pred_graph"]
                + lam * candidate["pred_disagreement"]
                + service_risk
            )
            work = candidate.copy()
            work["sweep_score"] = score
            summary = _evaluate_single_score(work, "sweep_score", latency_budget_ms)
            summary["scenario_name"] = scenario_name
            summary["parameter_name"] = "lambda"
            summary["parameter_value"] = lam
            rows.append(summary)
        for lam_ens in [0.25, 0.50, 0.75, 1.00, 1.25]:
            work = candidate.copy()
            work["sweep_score"] = candidate["pred_ensemble_mean"] + lam_ens * candidate["pred_ensemble_std"] + service_risk
            summary = _evaluate_single_score(work, "sweep_score", latency_budget_ms)
            summary["scenario_name"] = scenario_name
            summary["parameter_name"] = "lambda_ens"
            summary["parameter_value"] = lam_ens
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _run_temporal_model_comparison(forecast_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    train, val, test = split_train_val_test(
        forecast_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    train_full = pd.concat([train, val], ignore_index=True)
    feature_columns = default_feature_columns(train_full)
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))
    rows = []
    for model_name in [
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "small_mlp_regressor",
    ]:
        model = fit_forecast_model(model_name, train_full, feature_columns)
        prediction_frame = predict_forecast_model(model_name, model, test, feature_columns)
        metrics = evaluate_prediction_frame(prediction_frame)
        candidate = test.reset_index(drop=True).copy()
        candidate["model_score"] = prediction_frame["y_pred"].to_numpy()
        decision_summary, _ = evaluate_decision_policies(
            candidate,
            latency_budget_ms=latency_budget_ms,
            policy_columns={model_name: "model_score"},
        )
        decision_row = decision_summary.iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "row_count": metrics.row_count,
                "decision_count": int(decision_row["decision_count"]),
                "mean_realized_latency_ms": float(decision_row["mean_realized_latency_ms"]),
                "mean_decision_gap_ms": float(decision_row["mean_decision_gap_ms"]),
                "success_rate_under_60ms": float(decision_row["success_rate_under_60ms"]),
            }
        )
    return pd.DataFrame(rows)


def _run_horizon_sensitivity(time_bins: pd.DataFrame, config: dict, holdout_count: int) -> pd.DataFrame:
    rows = []
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))
    for horizon_bins in [1, 2, 3]:
        forecast_table = build_forecast_table(
            time_bins=time_bins,
            target_column=config["forecasting"]["target_column"],
            lags=list(config["forecasting"]["lag_steps"]),
            horizon_bins=horizon_bins,
        )
        scenario_candidates, _ = _prepare_base_scenarios(
            forecast_table,
            config,
            holdout_count=holdout_count,
            ensemble_members=int(config["optimization"].get("ensemble_uncertainty", {}).get("ensemble_members", 9)),
            ensemble_row_fraction=float(config["optimization"].get("ensemble_uncertainty", {}).get("row_fraction", 0.82)),
            ensemble_feature_fraction=float(config["optimization"].get("ensemble_uncertainty", {}).get("feature_fraction", 0.78)),
        )
        for scenario_name, candidate in scenario_candidates.items():
            summary, _ = evaluate_decision_policies(
                candidate,
                latency_budget_ms=latency_budget_ms,
                policy_columns={
                    "temporal": "pred_forecast",
                    "ensemble_uncertainty": "pred_ensemble_uncertainty",
                    "risk_adjusted": "pred_disagreement_aware",
                },
            )
            summary["scenario_name"] = scenario_name
            summary["horizon_bins"] = horizon_bins
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def _run_ensemble_member_sensitivity(forecast_table: pd.DataFrame, config: dict, holdout_count: int) -> pd.DataFrame:
    rows = []
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))
    for ensemble_members in [3, 5, 9, 15]:
        scenario_candidates, _ = _prepare_base_scenarios(
            forecast_table,
            config,
            holdout_count=holdout_count,
            ensemble_members=ensemble_members,
            ensemble_row_fraction=float(config["optimization"].get("ensemble_uncertainty", {}).get("row_fraction", 0.82)),
            ensemble_feature_fraction=float(config["optimization"].get("ensemble_uncertainty", {}).get("feature_fraction", 0.78)),
        )
        for scenario_name, candidate in scenario_candidates.items():
            summary, _ = evaluate_decision_policies(
                candidate,
                latency_budget_ms=latency_budget_ms,
                policy_columns={
                    "ensemble_uncertainty": "pred_ensemble_uncertainty",
                    "risk_adjusted": "pred_disagreement_aware",
                },
            )
            summary["scenario_name"] = scenario_name
            summary["ensemble_members"] = ensemble_members
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/path_selection_sensitivity_analysis",
    )
    parser.add_argument("--holdout-count", type=int, default=4)
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)
    snapshot_seconds = int(config["graph"]["snapshot_seconds"])
    horizon_seconds = int(config["forecasting"]["horizon_seconds"])
    horizon_bins = max(1, horizon_seconds // snapshot_seconds) if horizon_seconds >= snapshot_seconds else 1
    forecast_table = build_forecast_table(
        time_bins=time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
    )

    ensemble_cfg = config["optimization"].get("ensemble_uncertainty", {})
    scenario_candidates, holdout_locations = _prepare_base_scenarios(
        forecast_table,
        config,
        holdout_count=args.holdout_count,
        ensemble_members=int(ensemble_cfg.get("ensemble_members", 9)),
        ensemble_row_fraction=float(ensemble_cfg.get("row_fraction", 0.82)),
        ensemble_feature_fraction=float(ensemble_cfg.get("feature_fraction", 0.78)),
    )
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))

    hyperparameter_sensitivity = _run_weight_sensitivity(scenario_candidates, latency_budget_ms)
    temporal_model_comparison = _run_temporal_model_comparison(forecast_table, config)
    horizon_sensitivity = _run_horizon_sensitivity(time_bins, config, args.holdout_count)
    ensemble_member_sensitivity = _run_ensemble_member_sensitivity(forecast_table, config, args.holdout_count)

    hyperparameter_sensitivity.to_csv(output_dir / "hyperparameter_sensitivity.csv", index=False)
    temporal_model_comparison.to_csv(output_dir / "temporal_model_comparison.csv", index=False)
    horizon_sensitivity.to_csv(output_dir / "horizon_sensitivity.csv", index=False)
    ensemble_member_sensitivity.to_csv(output_dir / "ensemble_member_sensitivity.csv", index=False)

    metadata = {
        "time_bins_path": str(time_bins_path),
        "config_path": str(_resolve_repo_path(args.config)),
        "holdout_locations": holdout_locations,
        "scenarios": sorted(scenario_candidates.keys()),
        "random_seeds": {
            "operational_moderate": 201,
            "operational_severe": 202,
        },
    }
    (output_dir / "path_selection_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"outputs_written={output_dir}")
    print(temporal_model_comparison.to_string(index=False))
    print(hyperparameter_sensitivity[["scenario_name", "parameter_name", "parameter_value", "mean_decision_gap_ms", "success_rate_under_60ms"]].head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
