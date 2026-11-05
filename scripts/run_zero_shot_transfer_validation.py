#!/usr/bin/env python3
"""Evaluate a completely frozen source policy on an unseen target simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import (
    assign_decision_groups,
    load_time_bin_table,
)
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_train_val_test,
)
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    evaluate_decision_policies,
)
from scripts.run_service_path_experiments import (
    POLICY_COLUMNS,
    _fit_models,
    _make_candidate_frame,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _ensure_source_trace(path: Path) -> None:
    if path.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_physics_informed_multipath_trace.py"),
            "--output",
            str(path),
            "--bin-seconds",
            "10",
            "--duration-hours",
            "0.5",
            "--seed",
            "2026",
        ],
        check=True,
    )


def _ensure_target_trace(path: Path) -> None:
    if path.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_hypatia_service_trace.py"),
            "--output",
            str(path),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--source-trace",
        default="data/processed/physics_informed_orbital_multipath_10s.csv",
    )
    parser.add_argument(
        "--target-trace",
        default="data/processed/hypatia_service_paths_10s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/zero_shot_transfer_validation",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    source_path = _resolve(args.source_trace)
    target_path = _resolve(args.target_trace)
    _ensure_source_trace(source_path)
    _ensure_target_trace(target_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source, source_audit = assign_decision_groups(
        load_time_bin_table(source_path)
    )
    target, target_audit = assign_decision_groups(
        load_time_bin_table(target_path)
    )
    source_resolution = int(source["bin_seconds"].iloc[0])
    target_resolution = int(target["bin_seconds"].iloc[0])
    if source_resolution != target_resolution:
        raise ValueError(
            "zero-shot source and target traces must use the same decision cadence"
        )

    lags = list(config["forecasting"]["lag_steps"])
    source_forecast = build_forecast_table(
        source,
        target_column=config["forecasting"]["target_column"],
        lags=lags,
        horizon_bins=1,
        decision_cadence_seconds=source_resolution,
        require_complete_decision_epochs=True,
    )
    source_train, source_val, _ = split_train_val_test(
        source_forecast,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    source_graph_train = add_graph_snapshot_features(source_train)
    source_graph_val = add_graph_snapshot_features(source_val)
    ensemble_cfg = config["optimization"]["ensemble_uncertainty"]
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
        source_train,
        source_val,
        source_graph_train,
        source_graph_val,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        ensemble_members=int(ensemble_cfg["ensemble_members"]),
        ensemble_row_fraction=float(ensemble_cfg["row_fraction"]),
        ensemble_feature_fraction=float(ensemble_cfg["feature_fraction"]),
    )

    # No target row participates in fitting, calibration, threshold selection
    # or fallback selection. Missing source-schema features receive the source
    # preprocessing default of zero.
    target_forecast = build_forecast_table(
        target,
        target_column=config["forecasting"]["target_column"],
        lags=lags,
        horizon_bins=1,
        decision_cadence_seconds=target_resolution,
        require_complete_decision_epochs=True,
    )
    target_graph = add_graph_snapshot_features(target_forecast)
    for column in temporal_features:
        if column not in target_forecast:
            target_forecast[column] = 0.0
    for column in graph_features:
        if column not in target_graph:
            target_graph[column] = 0.0

    consensus_cfg = config["optimization"]["consensus"]
    disagreement_cfg = config["optimization"]["disagreement_aware"]
    service_cfg = config["optimization"]["service_risk"]
    candidates = _make_candidate_frame(
        target_forecast,
        target_graph,
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
        disagreement_penalty=float(
            disagreement_cfg["uncertainty_multiplier"]
        ),
        ensemble_penalty=float(ensemble_cfg["lambda_ens"]),
        service_risk_reply_weight=float(
            service_cfg["reply_pressure_penalty"]
        ),
        service_risk_volatility_weight=float(
            service_cfg["volatility_penalty"]
        ),
    )
    online_switch_penalty = float(
        config["optimization"]["online_switch_penalty_ms"]
    )
    summary, decisions = evaluate_decision_policies(
        candidates,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        policy_columns=POLICY_COLUMNS,
        decision_window_seconds=target_resolution,
        online_switch_penalties_ms={
            "switch_aware_operational_selector": online_switch_penalty
        },
    )
    summary.to_csv(output_dir / "zero_shot_policy_summary.csv", index=False)
    decisions.to_csv(output_dir / "zero_shot_policy_decisions.csv", index=False)
    candidates.to_csv(output_dir / "zero_shot_candidate_predictions.csv", index=False)

    source_metadata = json.loads(
        source_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    target_metadata = json.loads(
        target_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "zero_shot_transfer": True,
        "source_trace": source_path.relative_to(REPO_ROOT).as_posix(),
        "target_trace": target_path.relative_to(REPO_ROOT).as_posix(),
        "source_family": source_metadata.get("dataset_name"),
        "target_family": target_metadata.get("dataset_name"),
        "source_concurrency_audit": source_audit,
        "target_concurrency_audit": target_audit,
        "decision_cadence_seconds": source_resolution,
        "source_exact_horizon_audit": source_forecast.attrs.get(
            "exact_horizon_audit", {}
        ),
        "target_exact_horizon_audit": target_forecast.attrs.get(
            "exact_horizon_audit", {}
        ),
        "target_rows_used_for_training": 0,
        "target_rows_used_for_calibration": 0,
        "frozen_components": [
            "temporal expert",
            "graph-context expert",
            "temporal ensemble",
            "conformal calibrator",
            "expert residual calibration",
            "residual-risk coefficients",
            "trust-gate threshold",
            "fallback policy",
        ],
        "valid_claim": (
            "zero-shot cross-simulator transfer from the custom "
            "physics-informed source to official Hypatia dynamic state"
        ),
        "invalid_claim": "zero-shot transfer to an independent measurement deployment",
    }
    (output_dir / "zero_shot_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"zero_shot_results_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
