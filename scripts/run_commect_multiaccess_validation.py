#!/usr/bin/env python3
"""Evaluate path-selection policies on independent COMMECT measurements."""

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
    WALL_CLOCK_SPLIT_AUDIT_ATTR,
    build_forecast_table,
    build_wall_clock_decision_schedule,
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


PREDICTOR_AUDIT_PARTITIONS = ("train", "calibration", "selection", "test")
PREDICTOR_AUDIT_ROW_IDENTITY_CANDIDATES = (
    "session_date",
    "session_bin_index",
    "relative_path",
    "bin_epoch",
)
PEER_SUMMARY_AVAILABILITY_COLUMNS = (
    "peer_latency_available",
    "state_peer_latency_available",
    "target_peer_latency_available",
    "peer_reply_available",
    "peer_burst_indicator_available",
)
TARGET_PEER_MOMENT_COLUMNS = (
    "target_peer_latency_mean",
    "target_peer_latency_std",
)
PREDICTOR_INFORMATION_AUDIT_SCOPE = (
    "information-view/data-quality balance plus fixed calibration diagnostics; "
    "not evaluation-performance evidence"
)
PREDICTOR_INFORMATION_AUDIT_COLUMNS = (
    "audit_scope",
    "diagnostic_interpretation",
    "partition",
    "view",
    "feature_count",
    "feature_columns_json",
    "row_count",
    "decision_epoch_count",
    "calibration_row_count",
    "calibration_decision_epoch_count",
    "row_identity_columns_json",
    "row_identity_unique",
    "same_rows_across_views",
    "same_row_order_across_views",
    "raw_feature_value_count",
    "raw_feature_missing_count",
    "raw_feature_missing_fraction",
    "raw_feature_missing_by_column_json",
    "rows_with_any_feature_missing_count",
    "rows_with_any_feature_missing_fraction",
    "wholly_missing_feature_count",
    "wholly_missing_feature_columns_json",
    "peer_summary_availability_field_count",
    "peer_summary_availability_by_field_json",
    "peer_summary_availability_value_count",
    "peer_summary_available_value_count",
    "peer_summary_available_fraction",
    "rows_with_any_peer_summary_available_count",
    "rows_with_any_peer_summary_available_fraction",
    "rows_with_all_peer_summaries_available_count",
    "rows_with_all_peer_summaries_available_fraction",
    "target_hint_path_identity_match_count",
    "target_hint_path_identity_comparable_count",
    "target_hint_path_identity_match_fraction",
    "target_hint_is_path_identity",
    "target_peer_signal_available_row_count",
    "target_peer_signal_available_fraction",
    "target_peer_moment_value_count",
    "target_peer_moment_missing_count",
    "target_peer_moment_missing_fraction",
    "target_peer_semantic_status",
    "calibration_sample_count",
    "calibration_residual_bias_ms",
    "calibration_residual_scale_ms",
    "calibration_mae_ms",
    "calibration_residual_variance_ms2",
    "paired_residual_covariance_ms2",
    "executed_temporal_fusion_weight_min",
    "executed_temporal_fusion_weight_max",
    "executed_context_fusion_weight_min",
    "executed_context_fusion_weight_max",
    "residual_risk_intercept_ms",
    "residual_risk_disagreement_weight",
    "residual_risk_ensemble_weight",
    "residual_risk_service_weight",
    "residual_risk_active_terms_json",
    "residual_risk_gate_ms",
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _make_candidate_frame_with_budget(
    *args: object,
    latency_budget_ms: float,
    **kwargs: object,
) -> pd.DataFrame:
    """Forward the resolved objective to shield construction explicitly."""

    return _make_candidate_frame(
        *args,
        latency_budget_ms=latency_budget_ms,
        **kwargs,
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _normalized_row_identity(
    frame: pd.DataFrame,
    identity_columns: list[str],
) -> pd.DataFrame:
    identity = frame.loc[:, identity_columns].copy()
    for column in identity_columns:
        identity[column] = identity[column].astype("string").fillna("<NA>")
    return identity.reset_index(drop=True)


def _raw_feature_missingness(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    missing_columns = [
        column for column in feature_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "predictor audit cannot inspect absent feature columns: "
            f"{missing_columns}"
        )
    matrix = frame.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    missing = matrix.isna()
    row_count = int(len(frame))
    feature_count = int(len(feature_columns))
    value_count = row_count * feature_count
    missing_by_column = {
        column: int(missing[column].sum()) for column in feature_columns
    }
    missing_count = int(sum(missing_by_column.values()))
    rows_with_missing = int(missing.any(axis=1).sum()) if feature_count else 0
    wholly_missing_columns = [
        column
        for column, count in missing_by_column.items()
        if row_count > 0 and count == row_count
    ]
    return {
        "raw_feature_value_count": value_count,
        "raw_feature_missing_count": missing_count,
        "raw_feature_missing_fraction": _fraction(missing_count, value_count),
        "raw_feature_missing_by_column": missing_by_column,
        "rows_with_any_feature_missing_count": rows_with_missing,
        "rows_with_any_feature_missing_fraction": _fraction(
            rows_with_missing,
            row_count,
        ),
        "wholly_missing_feature_columns": wholly_missing_columns,
    }


def _peer_summary_availability(frame: pd.DataFrame) -> dict[str, object]:
    fields = [
        column
        for column in PEER_SUMMARY_AVAILABILITY_COLUMNS
        if column in frame.columns
    ]
    row_count = int(len(frame))
    if not fields:
        return {
            "fields": [],
            "by_field": {},
            "availability_value_count": 0,
            "available_value_count": 0,
            "available_fraction": 0.0,
            "rows_with_any_available_count": 0,
            "rows_with_any_available_fraction": 0.0,
            "rows_with_all_available_count": 0,
            "rows_with_all_available_fraction": 0.0,
        }
    availability = (
        frame.loc[:, fields]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
    )
    available_value_count = int(availability.to_numpy().sum())
    availability_value_count = row_count * len(fields)
    rows_with_any = int(availability.any(axis=1).sum())
    rows_with_all = int(availability.all(axis=1).sum())
    by_field = {
        column: {
            "available_count": int(availability[column].sum()),
            "row_count": row_count,
            "available_fraction": _fraction(
                int(availability[column].sum()),
                row_count,
            ),
        }
        for column in fields
    }
    return {
        "fields": fields,
        "by_field": by_field,
        "availability_value_count": availability_value_count,
        "available_value_count": available_value_count,
        "available_fraction": _fraction(
            available_value_count,
            availability_value_count,
        ),
        "rows_with_any_available_count": rows_with_any,
        "rows_with_any_available_fraction": _fraction(rows_with_any, row_count),
        "rows_with_all_available_count": rows_with_all,
        "rows_with_all_available_fraction": _fraction(rows_with_all, row_count),
    }


def _target_peer_semantics(
    temporal_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> dict[str, object]:
    if "relative_path" not in temporal_frame or "target_hint" not in temporal_frame:
        comparable = pd.Series(False, index=temporal_frame.index)
        matches = pd.Series(False, index=temporal_frame.index)
    else:
        relative_path = temporal_frame["relative_path"].astype("string")
        target_hint = temporal_frame["target_hint"].astype("string")
        path_identity = relative_path.str.rsplit("/", n=1).str[-1]
        comparable = relative_path.notna() & target_hint.notna()
        matches = comparable & path_identity.eq(target_hint)
    comparable_count = int(comparable.sum())
    match_count = int(matches.sum())
    target_hint_is_path_identity = bool(
        comparable_count == len(temporal_frame)
        and comparable_count > 0
        and match_count == comparable_count
    )

    if "target_peer_latency_available" in context_frame:
        target_peer_available = (
            pd.to_numeric(
                context_frame["target_peer_latency_available"],
                errors="coerce",
            )
            .fillna(0.0)
            .gt(0.0)
        )
        signal_available_count = int(target_peer_available.sum())
    else:
        signal_available_count = 0
    moment_columns = [
        column
        for column in TARGET_PEER_MOMENT_COLUMNS
        if column in context_frame.columns
    ]
    if moment_columns:
        target_peer_moments = context_frame.loc[:, moment_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        moment_value_count = int(target_peer_moments.size)
        moment_missing_count = int(target_peer_moments.isna().to_numpy().sum())
    else:
        moment_value_count = 0
        moment_missing_count = 0

    if target_hint_is_path_identity and signal_available_count == 0:
        semantic_status = (
            "structurally_unavailable: target_hint equals the relative-path "
            "identity, so leave-one-out target peers do not exist"
        )
    elif signal_available_count == 0:
        semantic_status = "unavailable: source semantics do not establish why"
    elif target_hint_is_path_identity:
        semantic_status = (
            "partly available despite target_hint/path-identity equivalence; "
            "inspect duplicate path rows"
        )
    else:
        semantic_status = "available for at least one row"
    return {
        "target_hint_path_identity_match_count": match_count,
        "target_hint_path_identity_comparable_count": comparable_count,
        "target_hint_path_identity_match_fraction": _fraction(
            match_count,
            comparable_count,
        ),
        "target_hint_is_path_identity": target_hint_is_path_identity,
        "target_peer_signal_available_row_count": signal_available_count,
        "target_peer_signal_available_fraction": _fraction(
            signal_available_count,
            int(len(context_frame)),
        ),
        "target_peer_moment_columns": moment_columns,
        "target_peer_moment_value_count": moment_value_count,
        "target_peer_moment_missing_count": moment_missing_count,
        "target_peer_moment_missing_fraction": _fraction(
            moment_missing_count,
            moment_value_count,
        ),
        "target_peer_semantic_status": semantic_status,
    }


def _calibration_diagnostics(calibration: object) -> dict[str, object]:
    return {
        "sample_count": int(getattr(calibration, "sample_count")),
        "residual_bias_ms": float(getattr(calibration, "residual_bias_ms")),
        "residual_scale_ms": float(getattr(calibration, "residual_scale_ms")),
        "mae_ms": float(getattr(calibration, "mae_ms")),
        "residual_variance_ms2": float(
            getattr(calibration, "residual_variance_ms2")
        ),
        "paired_residual_covariance_ms2": float(
            getattr(calibration, "paired_residual_covariance_ms2")
        ),
    }


def _build_predictor_information_audit(
    temporal_partitions: dict[str, pd.DataFrame],
    context_partitions: dict[str, pd.DataFrame],
    temporal_features: list[str],
    context_features: list[str],
    temporal_calibration: object,
    context_calibration: object,
    executed_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit matched predictor inputs and fixed calibration diagnostics.

    No evaluation outcome or policy metric is read here.  The artifact is
    intentionally limited to information-view balance, raw input availability,
    row identity, and parameters frozen by the calibration split.
    """

    for partition in PREDICTOR_AUDIT_PARTITIONS:
        if partition not in temporal_partitions or partition not in context_partitions:
            raise ValueError(f"predictor audit is missing partition {partition!r}")
    all_frames = [
        temporal_partitions[partition]
        for partition in PREDICTOR_AUDIT_PARTITIONS
    ] + [
        context_partitions[partition]
        for partition in PREDICTOR_AUDIT_PARTITIONS
    ]
    identity_columns = [
        column
        for column in PREDICTOR_AUDIT_ROW_IDENTITY_CANDIDATES
        if all(column in frame.columns for frame in all_frames)
    ]
    if not identity_columns:
        raise ValueError("predictor audit could not establish a row identity")

    required_weight_columns = (
        "temporal_expert_weight",
        "graph_expert_weight",
    )
    missing_weight_columns = [
        column
        for column in required_weight_columns
        if column not in executed_candidates.columns
    ]
    if missing_weight_columns or executed_candidates.empty:
        raise ValueError(
            "predictor audit requires executed fusion weights; missing "
            f"{missing_weight_columns}"
        )
    temporal_weights = pd.to_numeric(
        executed_candidates["temporal_expert_weight"],
        errors="coerce",
    )
    context_weights = pd.to_numeric(
        executed_candidates["graph_expert_weight"],
        errors="coerce",
    )
    if temporal_weights.isna().any() or context_weights.isna().any():
        raise ValueError("executed fusion weights contain missing/non-numeric values")
    fusion_diagnostics = {
        "source": "executed candidate rows; weights are fixed by calibration",
        "row_count": int(len(executed_candidates)),
        "temporal_weight_min": float(temporal_weights.min()),
        "temporal_weight_max": float(temporal_weights.max()),
        "context_weight_min": float(context_weights.min()),
        "context_weight_max": float(context_weights.max()),
        "weight_sum_min": float((temporal_weights + context_weights).min()),
        "weight_sum_max": float((temporal_weights + context_weights).max()),
        "constant_across_rows": bool(
            temporal_weights.nunique(dropna=False) == 1
            and context_weights.nunique(dropna=False) == 1
        ),
    }

    calibration_diagnostics = {
        "temporal": _calibration_diagnostics(temporal_calibration),
        "context": _calibration_diagnostics(context_calibration),
    }
    risk_term_attributes = {
        "disagreement": "residual_risk_disagreement_weight",
        "ensemble_spread": "residual_risk_ensemble_weight",
        "service_risk": "residual_risk_service_weight",
    }
    risk_terms = {
        term: {
            "coefficient": float(getattr(temporal_calibration, attribute)),
            "active": bool(
                abs(float(getattr(temporal_calibration, attribute))) > 1e-12
            ),
        }
        for term, attribute in risk_term_attributes.items()
    }
    risk_diagnostics = {
        "fit_partition": "calibration",
        "nonnegative_linear_fit": True,
        "activity_threshold_absolute_coefficient": 1e-12,
        "intercept_ms": float(
            getattr(temporal_calibration, "residual_risk_intercept_ms")
        ),
        "terms": risk_terms,
        "active_terms": [
            term for term, values in risk_terms.items() if values["active"]
        ],
        "gate_ms": float(getattr(temporal_calibration, "residual_risk_gate_ms")),
        "shared_between_views": all(
            float(getattr(temporal_calibration, attribute))
            == float(getattr(context_calibration, attribute))
            for attribute in (
                "residual_risk_intercept_ms",
                *risk_term_attributes.values(),
                "residual_risk_gate_ms",
            )
        ),
    }

    partition_metadata: dict[str, object] = {}
    audit_rows: list[dict[str, object]] = []
    calibration_row_count = int(len(temporal_partitions["calibration"]))
    calibration_decision_count = int(
        temporal_partitions["calibration"]["session_bin_index"].nunique()
    )
    for partition in PREDICTOR_AUDIT_PARTITIONS:
        temporal_frame = temporal_partitions[partition]
        context_frame = context_partitions[partition]
        temporal_identity = _normalized_row_identity(
            temporal_frame,
            identity_columns,
        )
        context_identity = _normalized_row_identity(
            context_frame,
            identity_columns,
        )
        same_order = bool(temporal_identity.equals(context_identity))
        temporal_records = sorted(map(tuple, temporal_identity.to_numpy().tolist()))
        context_records = sorted(map(tuple, context_identity.to_numpy().tolist()))
        same_rows = bool(temporal_records == context_records)
        if not same_rows or not same_order:
            raise ValueError(
                "temporal/context predictor views do not preserve identical "
                f"ordered rows in partition {partition!r}"
            )
        row_identity_unique = bool(
            not temporal_identity.duplicated(keep=False).any()
            and not context_identity.duplicated(keep=False).any()
        )
        temporal_missingness = _raw_feature_missingness(
            temporal_frame,
            temporal_features,
        )
        context_missingness = _raw_feature_missingness(
            context_frame,
            context_features,
        )
        peer_availability = _peer_summary_availability(context_frame)
        target_peer_semantics = _target_peer_semantics(
            temporal_frame,
            context_frame,
        )
        row_count = int(len(temporal_frame))
        decision_count = int(temporal_frame["session_bin_index"].nunique())
        partition_metadata[partition] = {
            "row_count": row_count,
            "decision_epoch_count": decision_count,
            "row_identity_unique": row_identity_unique,
            "same_rows_across_views": same_rows,
            "same_row_order_across_views": same_order,
            "views": {
                "temporal": temporal_missingness,
                "context": {
                    **context_missingness,
                    "peer_summary_availability": peer_availability,
                    "target_peer_semantics": target_peer_semantics,
                },
            },
        }
        for view, features, missingness, calibration in (
            (
                "temporal",
                temporal_features,
                temporal_missingness,
                calibration_diagnostics["temporal"],
            ),
            (
                "context",
                context_features,
                context_missingness,
                calibration_diagnostics["context"],
            ),
        ):
            is_context = view == "context"
            audit_rows.append(
                {
                    "audit_scope": PREDICTOR_INFORMATION_AUDIT_SCOPE,
                    "diagnostic_interpretation": (
                        "calibration diagnostics are fixed-split diagnostics, "
                        "not held-out evaluation-performance claims"
                    ),
                    "partition": partition,
                    "view": view,
                    "feature_count": int(len(features)),
                    "feature_columns_json": _compact_json(features),
                    "row_count": row_count,
                    "decision_epoch_count": decision_count,
                    "calibration_row_count": calibration_row_count,
                    "calibration_decision_epoch_count": calibration_decision_count,
                    "row_identity_columns_json": _compact_json(identity_columns),
                    "row_identity_unique": row_identity_unique,
                    "same_rows_across_views": same_rows,
                    "same_row_order_across_views": same_order,
                    "raw_feature_value_count": missingness[
                        "raw_feature_value_count"
                    ],
                    "raw_feature_missing_count": missingness[
                        "raw_feature_missing_count"
                    ],
                    "raw_feature_missing_fraction": missingness[
                        "raw_feature_missing_fraction"
                    ],
                    "raw_feature_missing_by_column_json": _compact_json(
                        missingness["raw_feature_missing_by_column"]
                    ),
                    "rows_with_any_feature_missing_count": missingness[
                        "rows_with_any_feature_missing_count"
                    ],
                    "rows_with_any_feature_missing_fraction": missingness[
                        "rows_with_any_feature_missing_fraction"
                    ],
                    "wholly_missing_feature_count": len(
                        missingness["wholly_missing_feature_columns"]
                    ),
                    "wholly_missing_feature_columns_json": _compact_json(
                        missingness["wholly_missing_feature_columns"]
                    ),
                    "peer_summary_availability_field_count": (
                        len(peer_availability["fields"]) if is_context else 0
                    ),
                    "peer_summary_availability_by_field_json": _compact_json(
                        peer_availability["by_field"] if is_context else {}
                    ),
                    "peer_summary_availability_value_count": (
                        peer_availability["availability_value_count"]
                        if is_context
                        else 0
                    ),
                    "peer_summary_available_value_count": (
                        peer_availability["available_value_count"]
                        if is_context
                        else 0
                    ),
                    "peer_summary_available_fraction": (
                        peer_availability["available_fraction"]
                        if is_context
                        else None
                    ),
                    "rows_with_any_peer_summary_available_count": (
                        peer_availability["rows_with_any_available_count"]
                        if is_context
                        else 0
                    ),
                    "rows_with_any_peer_summary_available_fraction": (
                        peer_availability["rows_with_any_available_fraction"]
                        if is_context
                        else None
                    ),
                    "rows_with_all_peer_summaries_available_count": (
                        peer_availability["rows_with_all_available_count"]
                        if is_context
                        else 0
                    ),
                    "rows_with_all_peer_summaries_available_fraction": (
                        peer_availability["rows_with_all_available_fraction"]
                        if is_context
                        else None
                    ),
                    "target_hint_path_identity_match_count": (
                        target_peer_semantics[
                            "target_hint_path_identity_match_count"
                        ]
                        if is_context
                        else 0
                    ),
                    "target_hint_path_identity_comparable_count": (
                        target_peer_semantics[
                            "target_hint_path_identity_comparable_count"
                        ]
                        if is_context
                        else 0
                    ),
                    "target_hint_path_identity_match_fraction": (
                        target_peer_semantics[
                            "target_hint_path_identity_match_fraction"
                        ]
                        if is_context
                        else None
                    ),
                    "target_hint_is_path_identity": (
                        target_peer_semantics["target_hint_is_path_identity"]
                        if is_context
                        else None
                    ),
                    "target_peer_signal_available_row_count": (
                        target_peer_semantics[
                            "target_peer_signal_available_row_count"
                        ]
                        if is_context
                        else 0
                    ),
                    "target_peer_signal_available_fraction": (
                        target_peer_semantics[
                            "target_peer_signal_available_fraction"
                        ]
                        if is_context
                        else None
                    ),
                    "target_peer_moment_value_count": (
                        target_peer_semantics["target_peer_moment_value_count"]
                        if is_context
                        else 0
                    ),
                    "target_peer_moment_missing_count": (
                        target_peer_semantics[
                            "target_peer_moment_missing_count"
                        ]
                        if is_context
                        else 0
                    ),
                    "target_peer_moment_missing_fraction": (
                        target_peer_semantics[
                            "target_peer_moment_missing_fraction"
                        ]
                        if is_context
                        else None
                    ),
                    "target_peer_semantic_status": (
                        target_peer_semantics["target_peer_semantic_status"]
                        if is_context
                        else "not applicable to temporal view"
                    ),
                    "calibration_sample_count": calibration["sample_count"],
                    "calibration_residual_bias_ms": calibration[
                        "residual_bias_ms"
                    ],
                    "calibration_residual_scale_ms": calibration[
                        "residual_scale_ms"
                    ],
                    "calibration_mae_ms": calibration["mae_ms"],
                    "calibration_residual_variance_ms2": calibration[
                        "residual_variance_ms2"
                    ],
                    "paired_residual_covariance_ms2": calibration[
                        "paired_residual_covariance_ms2"
                    ],
                    "executed_temporal_fusion_weight_min": fusion_diagnostics[
                        "temporal_weight_min"
                    ],
                    "executed_temporal_fusion_weight_max": fusion_diagnostics[
                        "temporal_weight_max"
                    ],
                    "executed_context_fusion_weight_min": fusion_diagnostics[
                        "context_weight_min"
                    ],
                    "executed_context_fusion_weight_max": fusion_diagnostics[
                        "context_weight_max"
                    ],
                    "residual_risk_intercept_ms": risk_diagnostics[
                        "intercept_ms"
                    ],
                    "residual_risk_disagreement_weight": risk_terms[
                        "disagreement"
                    ]["coefficient"],
                    "residual_risk_ensemble_weight": risk_terms[
                        "ensemble_spread"
                    ]["coefficient"],
                    "residual_risk_service_weight": risk_terms[
                        "service_risk"
                    ]["coefficient"],
                    "residual_risk_active_terms_json": _compact_json(
                        risk_diagnostics["active_terms"]
                    ),
                    "residual_risk_gate_ms": risk_diagnostics["gate_ms"],
                }
            )

    metadata: dict[str, object] = {
        "schema_version": 1,
        "csv_file": "predictor_information_audit.csv",
        "audit_scope": PREDICTOR_INFORMATION_AUDIT_SCOPE,
        "interpretation": (
            "This artifact audits matched information views, raw feature "
            "availability, and diagnostics fixed on calibration data. It does "
            "not establish held-out predictive or policy superiority."
        ),
        "evaluation_performance_metrics_included": False,
        "evaluation_outcomes_used": False,
        "row_identity_columns": identity_columns,
        "all_partitions_same_rows_across_views": all(
            bool(partition_metadata[partition]["same_rows_across_views"])
            for partition in PREDICTOR_AUDIT_PARTITIONS
        ),
        "all_partitions_same_row_order_across_views": all(
            bool(partition_metadata[partition]["same_row_order_across_views"])
            for partition in PREDICTOR_AUDIT_PARTITIONS
        ),
        "views": {
            "temporal": {
                "feature_count": int(len(temporal_features)),
                "feature_columns": temporal_features,
            },
            "context": {
                "feature_count": int(len(context_features)),
                "feature_columns": context_features,
            },
        },
        "partitions": partition_metadata,
        "calibration": {
            "partition": "calibration",
            "row_count": calibration_row_count,
            "decision_epoch_count": calibration_decision_count,
            "same_rows_across_views": bool(
                partition_metadata["calibration"]["same_rows_across_views"]
            ),
            "residual_definition": "target_next minus raw prediction",
            "diagnostics": calibration_diagnostics,
        },
        "executed_fusion": fusion_diagnostics,
        "residual_risk_fit": risk_diagnostics,
        "semantic_availability_note": (
            "target_peer_latency availability is reported separately from "
            "generic missingness because COMMECT target_hint may encode the "
            "relative-path identity, leaving no distinct target peer after "
            "leave-one-out construction"
        ),
    }
    audit = pd.DataFrame(audit_rows).loc[
        :, list(PREDICTOR_INFORMATION_AUDIT_COLUMNS)
    ]
    return audit, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--trace",
        default="data/processed/commect_multiaccess_10s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/commect_validation_gated_audit",
    )
    parser.add_argument(
        "--max-skew-ms",
        type=float,
        default=None,
        help="Restrict decision epochs by recorded cross-path median-time skew.",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=None,
        help=(
            "Override the configured latency objective. Models, calibration, "
            "opportunity counts, and policy admission are recomputed for this value."
        ),
    )
    parser.add_argument(
        "--gate-planned-uses",
        type=int,
        default=None,
        help="Total pre-declared invocations sharing the gate alpha family.",
    )
    parser.add_argument(
        "--gate-use-index",
        type=int,
        default=None,
        help="One-based index of this invocation in the shared alpha family.",
    )
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    latency_budget_ms = (
        float(args.latency_budget_ms)
        if args.latency_budget_ms is not None
        else float(config["optimization"]["latency_budget_ms"])
    )
    trace_path = _resolve(args.trace)
    frame, concurrency = assign_decision_groups(
        load_time_bin_table(trace_path),
        literal_single_controller_steering=True,
    )
    decision_cadence_seconds = float(frame["bin_seconds"].iloc[0])
    decision_schedule = build_wall_clock_decision_schedule(
        frame,
        decision_cadence_seconds=decision_cadence_seconds,
    )
    forecast = build_forecast_table(
        frame,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=1,
        decision_cadence_seconds=decision_cadence_seconds,
        require_complete_decision_epochs=True,
    )
    exact_horizon_audit = forecast.attrs.get("exact_horizon_audit", {})
    if args.max_skew_ms is not None:
        if "inter_path_skew_ms" not in forecast:
            raise ValueError("trace does not expose inter_path_skew_ms")
        eligible_epochs = set(
            forecast.loc[
                forecast["inter_path_skew_ms"].le(args.max_skew_ms),
                "session_bin_index",
            ].astype(int)
        )
        forecast = forecast[
            forecast["session_bin_index"].isin(eligible_epochs)
        ].copy()
        if forecast["session_bin_index"].nunique() < 20:
            raise ValueError("skew restriction leaves too few decision epochs")
    split_cfg = config["forecasting"]["policy_evaluation_ratios"]
    train, calibration, selection, test = split_train_calibration_selection_test(
        forecast,
        train_ratio=float(split_cfg["train"]),
        calibration_ratio=float(split_cfg["calibration"]),
        selection_ratio=float(split_cfg["selection"]),
        test_ratio=float(split_cfg["test"]),
        decision_schedule=decision_schedule,
    )
    split_boundary_audit = train.attrs.get(WALL_CLOCK_SPLIT_AUDIT_ATTR, {})
    graph_train = add_graph_snapshot_features(train)
    graph_calibration_frame = add_graph_snapshot_features(calibration)
    graph_selection = add_graph_snapshot_features(selection)
    graph_test = add_graph_snapshot_features(test)
    ensemble_cfg = config["optimization"]["ensemble_uncertainty"]
    risk_cfg = config["optimization"]["risk_control"]
    planned_gate_uses = (
        int(args.gate_planned_uses)
        if args.gate_planned_uses is not None
        else int(risk_cfg.get("planned_gate_uses", 1))
    )
    gate_use_index = (
        int(args.gate_use_index)
        if args.gate_use_index is not None
        else int(risk_cfg.get("gate_use_index", 1))
    )
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
        graph_calibration_frame,
        latency_budget_ms=latency_budget_ms,
        ensemble_members=int(ensemble_cfg["ensemble_members"]),
        ensemble_row_fraction=float(ensemble_cfg["row_fraction"]),
        ensemble_feature_fraction=float(ensemble_cfg["feature_fraction"]),
        selection_frame=selection,
        graph_selection=graph_selection,
        # COMMECT is one continuous drive.  Treating its bins as independent
        # would manufacture replication, so the collection date is the unit.
        risk_control_group_column="session_date",
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
            planned_gate_uses=planned_gate_uses,
            gate_use_index=gate_use_index,
            bootstrap_samples=int(risk_cfg["bootstrap_samples"]),
            random_seed=int(risk_cfg["random_seed"]),
        ),
    )
    consensus_cfg = config["optimization"]["consensus"]
    disagreement_cfg = config["optimization"]["disagreement_aware"]
    service_cfg = config["optimization"]["service_risk"]
    candidates = _make_candidate_frame_with_budget(
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
        latency_budget_ms=latency_budget_ms,
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
    predictor_audit, predictor_audit_metadata = _build_predictor_information_audit(
        temporal_partitions={
            "train": train,
            "calibration": calibration,
            "selection": selection,
            "test": test,
        },
        context_partitions={
            "train": graph_train,
            "calibration": graph_calibration_frame,
            "selection": graph_selection,
            "test": graph_test,
        },
        temporal_features=temporal_features,
        context_features=graph_features,
        temporal_calibration=temporal_calibration,
        context_calibration=graph_calibration,
        executed_candidates=candidates,
    )
    switch_penalty = float(
        config["optimization"]["online_switch_penalty_ms"]
    )
    summary, decisions = evaluate_decision_policies(
        candidates,
        latency_budget_ms=latency_budget_ms,
        policy_columns=POLICY_COLUMNS,
        decision_window_seconds=float(frame["bin_seconds"].iloc[0]),
        online_switch_penalties_ms={
            "switch_aware_operational_selector": switch_penalty
        },
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "policy_summary.csv", index=False)
    decisions.to_csv(output_dir / "policy_decisions.csv", index=False)
    candidates.to_csv(output_dir / "candidate_predictions.csv", index=False)
    predictor_audit.to_csv(
        output_dir / "predictor_information_audit.csv",
        index=False,
    )
    (output_dir / "predictor_information_audit_metadata.json").write_text(
        json.dumps(predictor_audit_metadata, indent=2),
        encoding="utf-8",
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
        "independent_of_lens": True,
        "dataset_doi": trace_metadata["dataset_doi"],
        "concurrency_audit": concurrency,
        "trace_metadata": trace_metadata,
        "split_protocol": (
            "train/calibration/policy-selection/test boundaries are frozen on "
            "the unfiltered raw min/max exact-cadence wall-clock grid before "
            "target availability or completeness filtering; exact-horizon "
            "targets crossing a frozen boundary are excluded"
        ),
        "boundaries_declared_before_target_filtering": True,
        "split_boundary_audit": split_boundary_audit,
        "training_row_count": int(len(train)),
        "calibration_row_count": int(len(calibration)),
        "policy_selection_row_count": int(len(selection)),
        "evaluation_row_count": int(len(test)),
        "exact_horizon_audit": exact_horizon_audit,
        "gate_selection_reason": temporal_calibration.gate_selection_reason,
        "predictor_information_audit": predictor_audit_metadata,
        "maximum_inter_path_skew_ms": args.max_skew_ms,
        "latency_budget_ms": latency_budget_ms,
        "zero_shot_transfer": False,
        "valid_claim": (
            "external-source measured shadow evaluation on concurrent 5G and "
            "Starlink access alternatives"
        ),
        "invalid_claim": (
            "zero-shot deployment transfer; models are trained on the early "
            "portion of this independent campaign"
        ),
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"validation_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
