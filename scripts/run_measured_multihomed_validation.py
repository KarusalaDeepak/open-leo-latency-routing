#!/usr/bin/env python3
"""Evaluate service-path policies on measured concurrent Victoria terminals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import (
    assign_decision_groups,
    load_time_bin_table,
)
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_train_calibration_selection_test,
)
from open_leo_latency_routing.graphs.snapshots import add_graph_snapshot_features
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    evaluate_decision_policies,
)
from open_leo_latency_routing.optimization.risk_control import RiskControlConfig
from scripts.run_service_path_experiments import (
    POLICY_COLUMNS,
    _fit_models,
    _make_candidate_frame,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--trace",
        default="data/processed/lens_victoria_multihomed_holdout_10s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/measured_multihomed_holdout_validation",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    trace_path = _resolve(args.trace)
    frame, concurrency = assign_decision_groups(
        load_time_bin_table(trace_path)
    )
    forecast = build_forecast_table(
        frame,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=float(frame["bin_seconds"].iloc[0]),
        require_complete_decision_epochs=True,
    )
    split_cfg = config["forecasting"]["policy_evaluation_ratios"]
    train, calibration, selection, test = split_train_calibration_selection_test(
        forecast,
        train_ratio=float(split_cfg["train"]),
        calibration_ratio=float(split_cfg["calibration"]),
        selection_ratio=float(split_cfg["selection"]),
        test_ratio=float(split_cfg["test"]),
    )
    graph_train = add_graph_snapshot_features(train)
    graph_calibration = add_graph_snapshot_features(calibration)
    graph_selection = add_graph_snapshot_features(selection)
    graph_test = add_graph_snapshot_features(test)
    ensemble_cfg = config["optimization"]["ensemble_uncertainty"]
    risk_cfg = config["optimization"]["risk_control"]
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
        train,
        calibration,
        graph_train,
        graph_calibration,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        ensemble_members=int(ensemble_cfg["ensemble_members"]),
        ensemble_row_fraction=float(ensemble_cfg["row_fraction"]),
        ensemble_feature_fraction=float(ensemble_cfg["feature_fraction"]),
        selection_frame=selection,
        graph_selection=graph_selection,
        # The 12 adjacent hourly files form one continuous measurement block.
        # File boundaries are administrative and cannot be promoted to
        # independent campaigns, so the gate must fail closed on one group.
        risk_control_group_column=None,
        risk_control_config=RiskControlConfig(
            alpha=float(risk_cfg["familywise_alpha"]),
            noninferiority_margin=float(
                risk_cfg["noninferiority_margin"]
            ),
            opportunity_noninferiority_margin=float(
                risk_cfg.get("opportunity_noninferiority_margin", 0.02)
            ),
            minimum_effective_opportunities=float(
                risk_cfg["minimum_effective_opportunities"]
            ),
            practical_cvar_gain_ms=float(
                risk_cfg["practical_cvar_gain_ms"]
            ),
            cvar_quantile=float(risk_cfg["cvar_quantile"]),
            block_length=(
                int(risk_cfg["block_length"])
                if risk_cfg.get("block_length") is not None
                else None
            ),
            latency_cap_ms=float(risk_cfg.get("latency_cap_ms", 60_000.0)),
            cvar_grid_points=int(risk_cfg.get("cvar_grid_points", 101)),
            planned_gate_uses=int(risk_cfg.get("planned_gate_uses", 1)),
            gate_use_index=int(risk_cfg.get("gate_use_index", 1)),
            bootstrap_samples=int(risk_cfg["bootstrap_samples"]),
            random_seed=int(risk_cfg["random_seed"]),
        ),
    )
    consensus_cfg = config["optimization"]["consensus"]
    disagreement_cfg = config["optimization"]["disagreement_aware"]
    service_cfg = config["optimization"]["service_risk"]
    candidates = _make_candidate_frame(
        test,
        graph_test,
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
    switch_penalty = float(
        config["optimization"]["online_switch_penalty_ms"]
    )
    summary, decisions = evaluate_decision_policies(
        candidates,
        latency_budget_ms=float(config["optimization"]["latency_budget_ms"]),
        policy_columns=POLICY_COLUMNS,
        decision_window_seconds=float(frame["bin_seconds"].iloc[0]),
        online_switch_penalties_ms={
            "switch_aware_operational_selector": switch_penalty
        },
    )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "measured_policy_summary.csv", index=False)
    decisions.to_csv(output_dir / "measured_policy_decisions.csv", index=False)
    candidates.to_csv(
        output_dir / "measured_candidate_predictions.csv",
        index=False,
    )
    pd.DataFrame(json.loads(temporal_calibration.gate_selection_evidence_json)).to_csv(
        output_dir / "gate_selection_evidence.csv", index=False
    )
    trace_metadata = json.loads(
        trace_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "policy_level_evaluation": True,
        "measured_concurrent_paths": True,
        "independent_of_lens": False,
        "concurrency_audit": concurrency,
        "trace_metadata": trace_metadata,
        "split_protocol": (
            "strict train/calibration/policy-selection/test split per terminal"
        ),
        "training_row_count": int(len(train)),
        "calibration_row_count": int(len(calibration)),
        "policy_selection_row_count": int(len(selection)),
        "evaluation_row_count": int(len(test)),
        "exact_horizon_audit": forecast.attrs.get("exact_horizon_audit", {}),
        "gate_inference_unit": (
            "one continuous 12-hour collection; hourly source files are "
            "descriptive windows, not independent groups"
        ),
        "gate_independence_group_count": 1,
        "gate_selection_reason": temporal_calibration.gate_selection_reason,
        "valid_claim": (
            "path-selection validation on measured concurrent multi-homed "
            "Starlink access links"
        ),
        "invalid_claim": (
            "validation on a measurement dataset independent of LENS"
        ),
    }
    (output_dir / "measured_validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"measured_validation_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
