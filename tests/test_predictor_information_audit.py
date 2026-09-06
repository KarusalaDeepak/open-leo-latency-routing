"""Focused regression tests for the COMMECT predictor-information audit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from open_leo_latency_routing.optimization.calibrated_risk import ExpertCalibration
from scripts import run_commect_multiaccess_validation as validation


def _temporal_partition(offset: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": ["2023-10-11", "2023-10-11"],
            "session_bin_index": [offset, offset],
            "relative_path": [
                "commect/operator_a_5g",
                "commect/starlink",
            ],
            "bin_epoch": [1_000 + offset, 1_000 + offset],
            "target_hint": ["operator_a_5g", "starlink"],
            "temporal_complete": [1.0, 2.0],
            "temporal_sparse": [None, 3.0],
        }
    )


def _context_partition(temporal: pd.DataFrame) -> pd.DataFrame:
    context = temporal.copy()
    context["peer_latency_mean"] = [10.0, 20.0]
    context["peer_latency_available"] = [1, 1]
    context["target_peer_latency_mean"] = [None, None]
    context["target_peer_latency_std"] = [None, None]
    context["target_peer_latency_observed_count"] = [0, 0]
    context["target_peer_latency_available"] = [0, 0]
    return context


def _calibration(
    *,
    bias: float,
    scale: float,
    mae: float,
) -> ExpertCalibration:
    return ExpertCalibration(
        residual_bias_ms=bias,
        residual_scale_ms=scale,
        upper_residual_quantile_ms=9.0,
        mae_ms=mae,
        sample_count=2,
        paired_residual_covariance_ms2=6.0,
        residual_variance_ms2=25.0,
        residual_risk_intercept_ms=1.5,
        residual_risk_disagreement_weight=0.0,
        residual_risk_ensemble_weight=2.0,
        residual_risk_service_weight=0.5,
        residual_risk_gate_ms=12.0,
    )


def _audit_inputs():
    temporal = {
        partition: _temporal_partition(offset)
        for offset, partition in enumerate(validation.PREDICTOR_AUDIT_PARTITIONS)
    }
    context = {
        partition: _context_partition(frame) for partition, frame in temporal.items()
    }
    candidates = pd.DataFrame(
        {
            "temporal_expert_weight": [0.75, 0.75],
            "graph_expert_weight": [0.25, 0.25],
        }
    )
    return temporal, context, candidates


def test_predictor_information_audit_exposes_balance_semantics_and_calibration():
    temporal, context, candidates = _audit_inputs()

    audit, metadata = validation._build_predictor_information_audit(
        temporal_partitions=temporal,
        context_partitions=context,
        temporal_features=["temporal_complete", "temporal_sparse"],
        context_features=[
            "peer_latency_mean",
            "peer_latency_available",
            "target_peer_latency_mean",
            "target_peer_latency_std",
            "target_peer_latency_observed_count",
            "target_peer_latency_available",
        ],
        temporal_calibration=_calibration(bias=1.0, scale=5.0, mae=4.0),
        context_calibration=_calibration(bias=2.0, scale=9.0, mae=8.0),
        executed_candidates=candidates,
    )

    assert tuple(audit.columns) == validation.PREDICTOR_INFORMATION_AUDIT_COLUMNS
    assert len(audit) == 8
    assert metadata["evaluation_performance_metrics_included"] is False
    assert metadata["evaluation_outcomes_used"] is False
    assert metadata["all_partitions_same_rows_across_views"] is True
    assert metadata["all_partitions_same_row_order_across_views"] is True
    assert metadata["calibration"]["row_count"] == 2
    assert metadata["calibration"]["decision_epoch_count"] == 1

    train_context = audit[
        audit["partition"].eq("train") & audit["view"].eq("context")
    ].iloc[0]
    assert train_context["feature_count"] == 6
    assert train_context["raw_feature_value_count"] == 12
    assert train_context["raw_feature_missing_count"] == 4
    assert train_context["raw_feature_missing_fraction"] == pytest.approx(1 / 3)
    assert json.loads(train_context["wholly_missing_feature_columns_json"]) == [
        "target_peer_latency_mean",
        "target_peer_latency_std",
    ]
    assert train_context["peer_summary_availability_field_count"] == 2
    assert train_context["peer_summary_availability_value_count"] == 4
    assert train_context["peer_summary_available_value_count"] == 2
    assert train_context["peer_summary_available_fraction"] == 0.5
    assert train_context["target_hint_path_identity_match_count"] == 2
    assert train_context["target_hint_is_path_identity"]
    assert train_context["target_peer_signal_available_row_count"] == 0
    assert train_context["target_peer_moment_missing_fraction"] == 1.0
    assert str(train_context["target_peer_semantic_status"]).startswith(
        "structurally_unavailable"
    )
    assert train_context["calibration_residual_bias_ms"] == 2.0
    assert train_context["calibration_residual_scale_ms"] == 9.0
    assert train_context["calibration_mae_ms"] == 8.0
    assert train_context["paired_residual_covariance_ms2"] == 6.0
    assert train_context["executed_temporal_fusion_weight_min"] == 0.75
    assert train_context["executed_context_fusion_weight_max"] == 0.25
    assert json.loads(train_context["residual_risk_active_terms_json"]) == [
        "ensemble_spread",
        "service_risk",
    ]


def test_predictor_information_audit_fails_closed_on_cross_view_row_mismatch():
    temporal, context, candidates = _audit_inputs()
    context["test"] = context["test"].copy()
    context["test"].loc[0, "relative_path"] = "commect/unmatched"

    with pytest.raises(ValueError, match="identical ordered rows"):
        validation._build_predictor_information_audit(
            temporal_partitions=temporal,
            context_partitions=context,
            temporal_features=["temporal_complete", "temporal_sparse"],
            context_features=["peer_latency_mean", "peer_latency_available"],
            temporal_calibration=_calibration(bias=1.0, scale=5.0, mae=4.0),
            context_calibration=_calibration(bias=2.0, scale=9.0, mae=8.0),
            executed_candidates=candidates,
        )


def test_nondefault_objective_reaches_candidate_shield_construction(monkeypatch):
    captured: dict[str, object] = {}

    def fake_candidate_builder(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(validation, "_make_candidate_frame", fake_candidate_builder)
    output = validation._make_candidate_frame_with_budget(
        "test-frame",
        latency_budget_ms=137.5,
        marker="preserved",
    )

    assert output["ok"].tolist() == [1]
    assert captured["args"] == ("test-frame",)
    assert captured["kwargs"] == {
        "latency_budget_ms": 137.5,
        "marker": "preserved",
    }
