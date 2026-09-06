#!/usr/bin/env python3
"""Run paper-facing service path-selection experiments on open LEO measurements.

The script produces compact tables and figures for structural-shift evaluation:
normal split, temporal shift, site holdout, controlled operational degradation,
disagreement calibration, ablation, and runtime analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import (
    assign_decision_groups,
    ensure_parent,
    load_time_bin_table,
)
from open_leo_latency_routing.evaluation.confidence_intervals import build_bootstrap_policy_intervals
from open_leo_latency_routing.evaluation.significance import build_paired_policy_significance
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_group_train_calibration_selection_test,
    split_group_holdout,
    split_train_calibration_selection_test,
    split_train_val_test,
)
from open_leo_latency_routing.graphs.snapshots import (
    add_graph_snapshot_features,
    graph_context_feature_columns,
)
from open_leo_latency_routing.models.forecast_baselines import (
    default_feature_columns,
    fit_forecast_model,
    predict_forecast_model,
)
from open_leo_latency_routing.models.graph_baselines import fit_graph_context_model
from open_leo_latency_routing.optimization.calibrated_risk import (
    CalibratedRiskConfig,
    ExpertCalibration,
    add_calibrated_mixture_risk_scores,
    calibration_to_dict,
    fit_expert_calibration,
)
from open_leo_latency_routing.optimization.explainability import summarize_xai_attribution
from open_leo_latency_routing.optimization.policies import (
    ConsensusPolicyConfig,
    SimpleFusionPolicyConfig,
    add_consensus_hybrid_scores,
    add_qos_filter_then_rank_scores,
    add_qos_shielded_scores,
    add_simple_fusion_scores,
    evaluate_decision_policies,
    select_validation_gated_fallback,
    summarize_multibin_decisions,
    summarize_stochastic_switching_costs,
    summarize_switching_costs,
)
from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    select_opportunity_aware_risk_controlled_policy,
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
    "age_aware_reactive_selector": "pred_age_aware_reactive",
    "robust_persistence_selector": "pred_robust_persistence",
    "predictive_greedy": "pred_forecast",
    "predictive_graph_greedy": "pred_graph",
    "predictive_simple_fusion_greedy": "pred_simple_fusion",
    "predictive_consensus_greedy": "pred_consensus",
    "calibrated_fusion_selector": "pred_calibrated_fusion",
    "disagreement_only_selector": "pred_disagreement_only",
    "ensemble_uncertainty_selector": "pred_ensemble_uncertainty",
    "cvar_proxy_selector": "pred_cvar_proxy",
    "conformal_uncertainty_selector": "pred_conformal_uncertainty",
    "calibrated_operational_selector": "pred_calibrated_operational",
    "qos_shielded_operational_selector": "pred_qos_shielded_operational",
    "validation_gated_qos_selector": "pred_qos_validation_gated",
    "qos_filter_then_context_selector": "pred_qos_filter_then_context",
    "qos_filter_then_ensemble_selector": "pred_qos_filter_then_ensemble",
    "switch_aware_operational_selector": "pred_calibrated_operational",
}

DISPLAY_POLICIES = {
    "random": "Random",
    "best_historical_path": "Historical",
    "highest_reply_availability": "Reply availability",
    "reactive_greedy": "Reactive",
    "age_aware_reactive_selector": "Age-aware reactive",
    "robust_persistence_selector": "Robust persistence",
    "predictive_greedy": "Temporal",
    "predictive_graph_greedy": "Context",
    "predictive_simple_fusion_greedy": "Fusion",
    "predictive_consensus_greedy": "Legacy disagreement",
    "calibrated_fusion_selector": "Calibrated fusion",
    "disagreement_only_selector": "Disagreement only",
    "ensemble_uncertainty_selector": "Ensemble uncertainty",
    "cvar_proxy_selector": "CVaR proxy",
    "conformal_uncertainty_selector": "Conformal uncertainty",
    "calibrated_operational_selector": "Calibrated operational",
    "qos_shielded_operational_selector": "QoS-shielded operational",
    "validation_gated_qos_selector": "Evidence-gated QoS",
    "qos_filter_then_context_selector": "QoS filter + context",
    "qos_filter_then_ensemble_selector": "QoS filter + ensemble",
    "switch_aware_operational_selector": "Switch-aware operational",
}


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _target_retention_protocol(
    concurrency_audit: dict[str, object],
) -> dict[str, object]:
    """Choose target completeness semantics from the audited decision topology."""

    supports_shadow_replay = bool(
        concurrency_audit.get("supports_shadow_policy_replay", False)
    )
    normalized_counterfactual = (
        concurrency_audit.get("decision_alignment")
        == "normalized_stage_counterfactual"
    )
    require_complete_candidates = supports_shadow_replay and not normalized_counterfactual
    return {
        "mode": (
            "complete_candidate_decision_epoch"
            if require_complete_candidates
            else "per_row_exact_target"
        ),
        "require_complete_decision_epochs": require_complete_candidates,
        "normalized_session_index_used_for_candidate_completeness": False,
        "reason": (
            "literal candidate-outcome replay requires every currently feasible "
            "candidate target"
            if require_complete_candidates
            else "non-concurrent normalized-stage analyses retain each row only "
            "when its own exact target exists; unrelated sessions are not treated "
            "as one candidate set"
        ),
    }


def _safe_sigmoid(values: pd.Series, scale: float) -> pd.Series:
    scaled = values.clip(lower=-60 * scale, upper=60 * scale) / max(scale, 1e-6)
    return 1.0 / (1.0 + np.exp(scaled))


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
            "ridge_regression",
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
    volatility_scale_ms: float | None = None,
) -> pd.Series:
    """Compute the lightweight service-risk penalty shared by uncertainty-aware policies."""

    reply_pressure = candidate.get("reply_pressure_score", pd.Series(0.0, index=candidate.index)).clip(lower=0.0)
    volatility = candidate["latency_std_ms"].fillna(0.0)
    volatility_scale = (
        max(1.0, float(volatility.median()))
        if volatility_scale_ms is None
        else max(1.0, float(volatility_scale_ms))
    )
    return reply_weight * reply_pressure + volatility_weight * (volatility / volatility_scale)


def _finite_sample_split_conformal_radius(
    absolute_residuals: np.ndarray | pd.Series,
    coverage: float,
) -> float:
    """Return the corrected split-conformal residual order statistic.

    The rank is ``ceil((n + 1) * coverage)``. If the requested coverage cannot
    be certified with ``n`` calibration scores, the conformal set is
    unbounded rather than silently replacing the rank by an interpolated
    sample quantile.
    """

    coverage = float(coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("split-conformal coverage must lie in (0, 1)")
    scores = np.asarray(absolute_residuals, dtype=float).reshape(-1)
    if not scores.size:
        raise ValueError("split-conformal calibration requires residuals")
    if not np.isfinite(scores).all() or (scores < 0.0).any():
        raise ValueError(
            "split-conformal residual scores must be finite and non-negative"
        )
    rank = int(math.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _fit_conformal_uncertainty_calibrator(
    temporal_model,
    validation_frame: pd.DataFrame,
    temporal_feature_columns: list[str],
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Fit a global finite-sample split-conformal residual radius."""

    predictions = predict_forecast_model(
        "linear_regression",
        temporal_model,
        validation_frame,
        temporal_feature_columns,
    )
    work = validation_frame.reset_index(drop=True).copy()
    work["pred_forecast"] = predictions["y_pred"].to_numpy()
    work["abs_residual"] = (work["target_next"] - work["pred_forecast"]).abs()
    volatility_scale_ms = max(
        1.0,
        float(work["latency_std_ms"].fillna(0.0).median()),
    )
    global_radius = _finite_sample_split_conformal_radius(
        work["abs_residual"].to_numpy(dtype=float),
        quantile,
    )
    return {
        "thresholds": np.array([], dtype=float),
        "radii": np.array([global_radius], dtype=float),
        "global_radius": global_radius,
        "coverage_target": quantile,
        "calibration_score_count": int(len(work)),
        "method": "global_split_conformal_corrected_order_statistic",
        "volatility_scale_ms": volatility_scale_ms,
    }


def _apply_conformal_uncertainty_selector(
    candidate: pd.DataFrame,
    calibrator: dict[str, Any],
    service_risk_ms: pd.Series,
) -> pd.DataFrame:
    """Add a global split-conformal uncertainty selector."""

    output = candidate.copy()
    proxy = _service_risk_ms(
        output,
        volatility_scale_ms=float(
            calibrator.get("volatility_scale_ms", 1.0)
        ),
    )
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


def _apply_feasible_snapshot_residual_gate(
    candidate: pd.DataFrame,
    residual_risk_gate_ms: float,
) -> pd.DataFrame:
    """Apply the trust gate using only currently feasible candidates.

    ``is_feasible_path`` is the online availability predicate (allowed path
    state and at least one current reply).  Unavailable rows are masked before
    the snapshot median is formed, so neither an attractive nor an extreme
    unavailable score can change the branch used by an actionable candidate.

    A snapshot with no feasible candidate is an outage/no-action epoch.  Such
    a snapshot is forced onto the conservative fallback branch for a stable
    audit score, while ``snapshot_trust_gate_state`` records that the score is
    only an emergency placeholder and not an executable routing action.  A
    non-finite residual risk on any feasible row also activates the fallback
    rather than silently omitting that candidate from the median.
    """

    required = {
        "session_bin_index",
        "is_feasible_path",
        "pred_learned_residual_risk",
    }
    missing = sorted(required.difference(candidate.columns))
    if missing:
        raise KeyError(
            "snapshot residual-risk gate is missing columns: "
            f"{missing}"
        )
    threshold = float(residual_risk_gate_ms)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "snapshot residual-risk gate threshold must be finite and "
            "non-negative"
        )

    output = candidate.copy()
    feasible_raw = pd.to_numeric(
        output["is_feasible_path"], errors="coerce"
    )
    if feasible_raw.isna().any() or not feasible_raw.isin((0.0, 1.0)).all():
        raise ValueError("is_feasible_path must be complete and binary")
    feasible = feasible_raw.eq(1.0)
    residual_risk = pd.to_numeric(
        output["pred_learned_residual_risk"], errors="coerce"
    )
    finite_risk = pd.Series(
        np.isfinite(residual_risk.to_numpy(dtype=float)),
        index=output.index,
    )
    snapshot_key = output["session_bin_index"]
    feasible_count = feasible.astype(int).groupby(
        snapshot_key,
        dropna=False,
    ).transform("sum")
    invalid_feasible_risk = (feasible & ~finite_risk).astype(int).groupby(
        snapshot_key,
        dropna=False,
    ).transform("sum").gt(0)
    snapshot_risk = residual_risk.where(feasible).groupby(
        snapshot_key,
        dropna=False,
    ).transform("median")
    emergency_no_action = feasible_count.eq(0)
    fail_closed = emergency_no_action | invalid_feasible_risk
    gate = (fail_closed | snapshot_risk.ge(threshold)).astype(float)

    output["snapshot_feasible_candidate_count"] = feasible_count.astype(int)
    output["snapshot_learned_residual_risk_ms"] = snapshot_risk
    output["snapshot_residual_risk_valid"] = (
        ~emergency_no_action & ~invalid_feasible_risk
    ).astype(int)
    output["snapshot_trust_gate_state"] = np.select(
        [
            emergency_no_action,
            invalid_feasible_risk,
            gate.eq(1.0),
        ],
        [
            "emergency_no_action",
            "invalid_feasible_risk_fail_closed",
            "risk_fallback",
        ],
        default="calibrated_risk",
    )
    output["disagreement_trust_gate"] = gate
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
    temporal_validation_mae: ExpertCalibration,
    graph_validation_mae: ExpertCalibration,
    consensus_config: ConsensusPolicyConfig,
    disagreement_temporal_weight: float,
    disagreement_graph_weight: float,
    disagreement_penalty: float,
    ensemble_penalty: float,
    service_risk_reply_weight: float,
    service_risk_volatility_weight: float,
    latency_budget_ms: float = 60.0,
) -> pd.DataFrame:
    """Merge predictions, current measurements, and policy scores."""

    scoring_started_ns = time.perf_counter_ns()
    test_frame = test_frame.reset_index(drop=True)
    graph_test = graph_test.reset_index(drop=True)
    temporal_predictions = predict_forecast_model(
        "ridge_regression",
        temporal_model,
        test_frame,
        temporal_feature_columns,
    ).rename(columns={"y_pred": "pred_forecast"})
    graph_predictions = predict_forecast_model(
        "ridge_regression",
        graph_model,
        graph_test,
        graph_feature_columns,
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
        "bin_seconds",
        "bin_start_utc",
        "observed_replies",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_max_ms",
        "target_next",
        "target_next_bin_epoch",
        "target_expected_cadence_seconds",
        "target_expected_horizon_seconds",
        "target_exact_wall_clock",
        "target_complete_decision_epoch",
        "target_cumulative_3",
        "target_mean_3",
        "target_cumulative_5",
        "target_mean_5",
    ]
    optional_columns = [
        "burst_indicator",
        "reply_pressure_score",
        "stress_scenario",
        "stress_applied",
        # Physics-informed simulator labels are retained strictly for post-hoc
        # event analysis; the feature selectors do not admit these columns.
        "handover_event",
        "attenuation_event",
        "elevation_degrees",
        "propagation_lower_bound_ms",
        "observation_age_ms",
        "observation_span_ms",
        "inter_path_skew_ms",
    ]
    selected_columns = [column for column in meta_columns + optional_columns if column in graph_test.columns]
    candidate = graph_test.reset_index(names="row_id")[selected_columns + ["row_id"]]
    candidate = (
        candidate.merge(temporal_predictions[["row_id", "pred_forecast"]], on="row_id", how="left")
        .merge(graph_predictions[["row_id", "pred_graph"]], on="row_id", how="left")
        .copy()
    )
    candidate["pred_ensemble_mean"] = ensemble_mean.to_numpy()
    candidate["pred_ensemble_std"] = ensemble_std.to_numpy()
    candidate["historical_mean_latency_ms"] = (
        candidate["relative_path"].map(historical_latency).fillna(historical_fallback)
    )
    candidate["reply_availability_score"] = -candidate["observed_replies"].astype(float)
    feasible_states = {"active", "available", "up", "healthy"}
    candidate["is_feasible_path"] = (
        candidate["path_state"].astype(str).str.lower().isin(feasible_states)
        & candidate["observed_replies"].astype(float).gt(0.0)
    ).astype(int)
    observation_age_ms = (
        candidate["observation_age_ms"].astype(float).clip(lower=0.0)
        if "observation_age_ms" in candidate
        else pd.Series(0.0, index=candidate.index)
    )
    candidate["pred_staleness_margin_ms"] = (
        temporal_validation_mae.age_margin_intercept_ms
        + temporal_validation_mae.age_margin_slope_ms_per_second
        * observation_age_ms
        / 1000.0
    )
    candidate["pred_age_aware_reactive"] = (
        candidate["latency_mean_ms"] + candidate["pred_staleness_margin_ms"]
    )
    candidate["pred_robust_persistence"] = (
        candidate["pred_age_aware_reactive"]
        + candidate["latency_std_ms"].astype(float).clip(lower=0.0)
    )
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
        volatility_scale_ms=temporal_validation_mae.service_risk_volatility_scale_ms,
    )
    candidate["pred_ensemble_uncertainty"] = (
        candidate["pred_ensemble_mean"] + ensemble_penalty * candidate["pred_ensemble_std"] + service_risk_ms
    )
    # A fixed Gaussian-tail proxy is included as a decision-level risk baseline;
    # unlike the proposed gate, it has no learned admission mechanism.
    candidate["pred_cvar_proxy"] = (
        candidate["pred_ensemble_mean"]
        + 2.0627 * candidate["pred_ensemble_std"]
        + service_risk_ms
    )
    candidate = _apply_conformal_uncertainty_selector(candidate, conformal_calibrator, service_risk_ms)
    candidate["service_risk_ms"] = service_risk_ms
    candidate = add_calibrated_mixture_risk_scores(
        candidate,
        temporal_calibration=temporal_validation_mae,
        graph_calibration=graph_validation_mae,
        config=CalibratedRiskConfig(
            uncertainty_multiplier=disagreement_penalty,
            service_risk_multiplier=1.0,
            output_column="pred_disagreement_aware",
        ),
        service_risk_column="service_risk_ms",
    )
    learned_residual_risk = (
        temporal_validation_mae.residual_risk_intercept_ms
        + temporal_validation_mae.residual_risk_disagreement_weight
        * candidate["pred_disagreement_normalized"]
        + temporal_validation_mae.residual_risk_ensemble_weight
        * candidate["pred_ensemble_std"]
        + temporal_validation_mae.residual_risk_service_weight
        * candidate["service_risk_ms"]
    ).clip(lower=0.0)
    raw_residual_risk = (
        temporal_validation_mae.residual_risk_intercept_ms
        + temporal_validation_mae.residual_risk_disagreement_weight
        * candidate["pred_disagreement_normalized"]
        + temporal_validation_mae.residual_risk_ensemble_weight
        * candidate["pred_ensemble_std"]
        + temporal_validation_mae.residual_risk_service_weight
        * candidate["service_risk_ms"]
    )
    candidate["pred_learned_residual_risk"] = learned_residual_risk
    candidate["pred_learned_risk_no_disagreement"] = (
        candidate["pred_calibrated_fusion"]
        + temporal_validation_mae.residual_risk_intercept_ms
        + temporal_validation_mae.residual_risk_ensemble_weight
        * candidate["pred_ensemble_std"]
        + temporal_validation_mae.residual_risk_service_weight
        * candidate["service_risk_ms"]
    )
    candidate["pred_calibrated_risk_ungated"] = (
        candidate["pred_calibrated_fusion"] + learned_residual_risk
    )
    # The gate is driven by validation-fitted expected residual risk over the
    # currently executable candidate set. Unavailable paths cannot influence
    # the online branch, and a zero-feasible snapshot is an explicit
    # emergency/no-action epoch with a fail-closed audit score.
    candidate = _apply_feasible_snapshot_residual_gate(
        candidate,
        temporal_validation_mae.residual_risk_gate_ms,
    )
    gate = candidate["disagreement_trust_gate"]
    fallback_score = (
        candidate["pred_graph"]
        if temporal_validation_mae.fallback_policy == "graph"
        else candidate["pred_ensemble_uncertainty"]
    )
    candidate["pred_disagreement_aware"] = (
        (1.0 - gate) * candidate["pred_calibrated_risk_ungated"]
        + gate * fallback_score
    )
    # Keep the historical column as a compatibility alias while exposing the
    # paper-facing method as a calibrated multi-signal operational selector.
    candidate["pred_calibrated_operational"] = candidate[
        "pred_disagreement_aware"
    ]
    candidate["risk_fallback_policy"] = temporal_validation_mae.fallback_policy
    candidate["validation_gated_fallback_policy"] = (
        temporal_validation_mae.validation_gated_fallback_policy
    )
    candidate["gate_selection_reason"] = temporal_validation_mae.gate_selection_reason
    candidate["gate_opportunity_count"] = temporal_validation_mae.gate_opportunity_count
    candidate["gate_effective_opportunity_count"] = (
        temporal_validation_mae.gate_effective_opportunity_count
    )
    candidate["gate_selected_success_lcb"] = (
        temporal_validation_mae.gate_selected_success_lcb
    )
    candidate["gate_selected_opportunity_success_lcb"] = (
        temporal_validation_mae.gate_selected_opportunity_success_lcb
    )
    candidate["gate_noninferiority_margin"] = (
        temporal_validation_mae.gate_noninferiority_margin
    )
    candidate["gate_opportunity_noninferiority_margin"] = (
        temporal_validation_mae.gate_opportunity_noninferiority_margin
    )
    candidate["gate_selected_aggregate_success_noninferior"] = int(
        temporal_validation_mae.gate_selected_aggregate_success_noninferior
    )
    candidate["gate_selected_opportunity_success_noninferior"] = int(
        temporal_validation_mae.gate_selected_opportunity_success_noninferior
    )
    candidate["gate_practical_cvar_gain_ms"] = (
        temporal_validation_mae.gate_practical_cvar_gain_ms
    )

    candidate = add_qos_shielded_scores(
        candidate,
        fallback_column=(
            "pred_graph"
            if temporal_validation_mae.fallback_policy == "graph"
            else "pred_ensemble_uncertainty"
        ),
        latency_budget_ms=latency_budget_ms,
    )
    validation_gated_fallback_column = {
        "reactive": "latency_mean_ms",
        "graph": "pred_graph",
        "ensemble": "pred_ensemble_uncertainty",
    }[temporal_validation_mae.validation_gated_fallback_policy]
    candidate = add_qos_shielded_scores(
        candidate,
        fallback_column=validation_gated_fallback_column,
        latency_budget_ms=latency_budget_ms,
        output_column="pred_qos_validation_gated",
    )
    candidate = add_qos_filter_then_rank_scores(
        candidate,
        ranking_column="pred_graph",
        latency_budget_ms=latency_budget_ms,
        output_column="pred_qos_filter_then_context",
    )
    candidate = add_qos_filter_then_rank_scores(
        candidate,
        ranking_column="pred_ensemble_uncertainty",
        latency_budget_ms=latency_budget_ms,
        output_column="pred_qos_filter_then_ensemble",
    )

    # Store the exact score terms used by the active branch. These columns make
    # decision explanations algebraically faithful instead of reconstructing
    # approximate contributions after selection.
    ungated = gate.eq(0.0)
    graph_fallback = (
        gate.eq(1.0)
        if temporal_validation_mae.fallback_policy == "graph"
        else pd.Series(False, index=candidate.index)
    )
    ensemble_fallback = (
        gate.eq(1.0)
        if temporal_validation_mae.fallback_policy == "ensemble"
        else pd.Series(False, index=candidate.index)
    )
    candidate["score_branch"] = np.select(
        [
            candidate["snapshot_trust_gate_state"].eq(
                "emergency_no_action"
            ),
            ungated,
        ],
        [
            "emergency_no_action",
            "calibrated_risk",
        ],
        default=f"{temporal_validation_mae.fallback_policy}_fallback",
    )
    candidate["score_component_latency_ms"] = np.select(
        [ungated, graph_fallback, ensemble_fallback],
        [
            candidate["pred_calibrated_fusion"],
            candidate["pred_graph"],
            candidate["pred_ensemble_mean"],
        ],
        default=candidate["pred_calibrated_fusion"],
    )
    candidate["score_component_disagreement_ms"] = np.where(
        ungated,
        temporal_validation_mae.residual_risk_disagreement_weight
        * candidate["pred_disagreement_normalized"],
        0.0,
    )
    candidate["score_component_uncertainty_ms"] = np.where(
        ungated,
        temporal_validation_mae.residual_risk_ensemble_weight
        * candidate["pred_ensemble_std"],
        np.where(
            ensemble_fallback,
            ensemble_penalty * candidate["pred_ensemble_std"],
            0.0,
        ),
    )
    candidate["score_component_service_risk_ms"] = np.where(
        ungated,
        temporal_validation_mae.residual_risk_service_weight
        * candidate["service_risk_ms"],
        np.where(ensemble_fallback, candidate["service_risk_ms"], 0.0),
    )
    candidate["score_component_calibration_ms"] = np.where(
        ungated,
        temporal_validation_mae.residual_risk_intercept_ms
        + (-raw_residual_risk).clip(lower=0.0),
        0.0,
    )
    component_sum = candidate[
        [
            "score_component_latency_ms",
            "score_component_disagreement_ms",
            "score_component_uncertainty_ms",
            "score_component_service_risk_ms",
            "score_component_calibration_ms",
        ]
    ].sum(axis=1)
    if not np.allclose(
        component_sum.to_numpy(dtype=float),
        candidate["pred_calibrated_operational"].to_numpy(dtype=float),
        atol=1e-8,
        rtol=1e-8,
    ):
        raise AssertionError("operational score components do not sum to the final score")
    decision_count = max(1, int(candidate["session_bin_index"].nunique()))
    candidate.attrs["model_scoring_time_us_per_decision"] = (
        (time.perf_counter_ns() - scoring_started_ns) / 1000.0 / decision_count
    )
    return candidate


@dataclass(frozen=True)
class GateSelectionSamples:
    """Actionable policy-selection epochs used by the admission gate.

    An epoch with no currently feasible path is an outage/no-action event, not
    a routing decision.  Keeping the excluded count next to the aligned arrays
    makes that conditioning explicit and prevents an emergency snapshot from
    silently re-entering opportunity, success, tail-risk, or grouping logic.
    """

    realized_latency: dict[str, list[float]]
    opportunity_mask: list[bool]
    independence_group_ids: list[object] | None
    total_epoch_count: int
    excluded_emergency_epoch_count: int

    @property
    def actionable_epoch_count(self) -> int:
        return len(self.opportunity_mask)


def _build_actionable_gate_selection_samples(
    selection_candidates: pd.DataFrame,
    base_selection: pd.DataFrame,
    latency_budget_ms: float,
    risk_control_group_column: str | None = None,
) -> GateSelectionSamples:
    """Build gate arrays conditional on at least one currently feasible path."""

    policy_scores = {
        "reactive": "latency_mean_ms",
        "graph": "selection_graph_shield",
        "ensemble": "selection_ensemble_shield",
    }
    required = {
        "session_bin_index",
        "target_next",
        "is_feasible_path",
        *policy_scores.values(),
    }
    missing = required.difference(selection_candidates.columns)
    if missing:
        raise KeyError(
            "gate selection candidates are missing required columns: "
            f"{sorted(missing)}"
        )
    if risk_control_group_column is not None:
        if risk_control_group_column not in base_selection:
            raise KeyError(
                "risk-control independence group is missing from the "
                f"selection frame: {risk_control_group_column}"
            )
        if "session_bin_index" not in base_selection:
            raise KeyError("selection frame is missing session_bin_index")

    realized_latency = {name: [] for name in policy_scores}
    opportunity_mask: list[bool] = []
    independence_group_ids: list[object] | None = (
        [] if risk_control_group_column is not None else None
    )
    excluded_emergency_epoch_count = 0
    grouped = selection_candidates.groupby("session_bin_index", sort=True)

    for epoch_id, snapshot in grouped:
        feasible = snapshot[
            snapshot["is_feasible_path"].fillna(0).astype(bool)
        ]
        if feasible.empty:
            # Deployment records this as outage/no-action.  It contributes no
            # candidate outcome and therefore no gate inference unit.
            excluded_emergency_epoch_count += 1
            continue

        outcomes = feasible["target_next"].astype(float).le(latency_budget_ms)
        opportunity_mask.append(bool(outcomes.any() and not outcomes.all()))

        if independence_group_ids is not None:
            epoch_groups = base_selection.loc[
                base_selection["session_bin_index"].eq(epoch_id),
                risk_control_group_column,
            ].dropna().unique()
            if len(epoch_groups) != 1:
                raise ValueError(
                    "each actionable decision epoch must map to exactly one "
                    "risk-control independence group in "
                    f"{risk_control_group_column}"
                )
            independence_group_ids.append(epoch_groups[0])

        for policy, score in policy_scores.items():
            chosen = feasible.loc[feasible[score].astype(float).idxmin()]
            realized_latency[policy].append(float(chosen["target_next"]))

    return GateSelectionSamples(
        realized_latency=realized_latency,
        opportunity_mask=opportunity_mask,
        independence_group_ids=independence_group_ids,
        total_epoch_count=int(selection_candidates["session_bin_index"].nunique()),
        excluded_emergency_epoch_count=excluded_emergency_epoch_count,
    )


def _fit_models(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    graph_train: pd.DataFrame,
    graph_val: pd.DataFrame,
    latency_budget_ms: float = 60.0,
    ensemble_members: int = 9,
    ensemble_row_fraction: float = 0.82,
    ensemble_feature_fraction: float = 0.78,
    selection_frame: pd.DataFrame | None = None,
    graph_selection: pd.DataFrame | None = None,
    risk_control_config: RiskControlConfig | None = None,
    risk_control_group_column: str | None = None,
) -> tuple[
    object,
    object,
    list[dict[str, object]],
    dict[str, Any],
    list[str],
    list[str],
    dict[str, float],
    float,
    ExpertCalibration,
    ExpertCalibration,
]:
    # Calibration rows never fit experts. When a separate selection frame is
    # supplied, calibration outcomes also never select the deployed fallback.
    train_full = train_frame.reset_index(drop=True)
    graph_train_full = graph_train.reset_index(drop=True)
    temporal_feature_columns = default_feature_columns(train_full)
    graph_feature_columns = graph_context_feature_columns(graph_train_full)
    # Matched model capacity isolates graph-context information from estimator
    # complexity. The legacy OLS/XGBoost pair is evaluated separately.
    temporal_model = fit_forecast_model("ridge_regression", train_full, temporal_feature_columns)
    graph_model = fit_graph_context_model(
        "ridge_regression",
        graph_train_full,
        graph_feature_columns,
    )
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
    temporal_validation_predictions = predict_forecast_model(
        "ridge_regression",
        temporal_model,
        val_frame if not val_frame.empty else train_frame,
        temporal_feature_columns,
    )
    graph_validation_predictions = predict_forecast_model(
        "ridge_regression",
        graph_model,
        graph_val if not graph_val.empty else graph_train,
        graph_feature_columns,
    )
    temporal_validation_mae = fit_expert_calibration(
        temporal_validation_predictions["y_true"],
        temporal_validation_predictions["y_pred"],
    )
    graph_validation_mae = fit_expert_calibration(
        graph_validation_predictions["y_true"],
        graph_validation_predictions["y_pred"],
    )
    temporal_centered_residual = (
        temporal_validation_predictions["y_true"].to_numpy(dtype=float)
        - temporal_validation_predictions["y_pred"].to_numpy(dtype=float)
        - temporal_validation_mae.residual_bias_ms
    )
    graph_centered_residual = (
        graph_validation_predictions["y_true"].to_numpy(dtype=float)
        - graph_validation_predictions["y_pred"].to_numpy(dtype=float)
        - graph_validation_mae.residual_bias_ms
    )
    paired_residual_covariance = 0.0
    if len(temporal_centered_residual) > 1:
        paired_residual_covariance = float(
            np.cov(
                temporal_centered_residual,
                graph_centered_residual,
                ddof=1,
            )[0, 1]
        )
        if not np.isfinite(paired_residual_covariance):
            paired_residual_covariance = 0.0
    temporal_residual_variance = float(
        temporal_validation_mae.residual_variance_ms2
        if temporal_validation_mae.residual_variance_ms2 is not None
        else temporal_validation_mae.residual_scale_ms**2
    )
    graph_residual_variance = float(
        graph_validation_mae.residual_variance_ms2
        if graph_validation_mae.residual_variance_ms2 is not None
        else graph_validation_mae.residual_scale_ms**2
    )
    covariance_limit = math.sqrt(
        max(temporal_residual_variance * graph_residual_variance, 0.0)
    )
    paired_residual_covariance = float(
        np.clip(
            paired_residual_covariance,
            -covariance_limit,
            covariance_limit,
        )
    )
    temporal_validation_mae = replace(
        temporal_validation_mae,
        paired_residual_covariance_ms2=paired_residual_covariance,
    )
    graph_validation_mae = replace(
        graph_validation_mae,
        paired_residual_covariance_ms2=paired_residual_covariance,
    )
    temporal_corrected = (
        temporal_validation_predictions["y_pred"]
        + temporal_validation_mae.residual_bias_ms
    )
    graph_corrected = (
        graph_validation_predictions["y_pred"]
        + graph_validation_mae.residual_bias_ms
    )
    pair_scale = math.sqrt(
        max(
            temporal_residual_variance
            + graph_residual_variance
            - 2.0 * paired_residual_covariance,
            1e-12,
        )
    )
    validation_disagreement = (
        (temporal_corrected - graph_corrected).abs() / max(pair_scale, 1e-6)
    )
    validation_index = (
        val_frame if not val_frame.empty else train_frame
    )["session_bin_index"].reset_index(drop=True)
    validation_snapshot_disagreement = validation_disagreement.groupby(
        validation_index
    ).median()
    gate_threshold = float(
        validation_snapshot_disagreement.quantile(0.90)
    )
    temporal_validation_mae = replace(
        temporal_validation_mae,
        normalized_disagreement_gate=gate_threshold,
    )
    graph_validation_mae = replace(
        graph_validation_mae,
        normalized_disagreement_gate=gate_threshold,
    )
    base_validation = (
        val_frame if not val_frame.empty else train_frame
    ).reset_index(drop=True)
    base_graph_validation = (
        graph_val if not graph_val.empty else graph_train
    ).reset_index(drop=True)
    service_risk_volatility_scale_ms = max(
        1.0,
        float(base_validation["latency_std_ms"].fillna(0.0).median()),
    )
    risk_feature_frames: list[pd.DataFrame] = []
    risk_target_frames: list[pd.Series] = []
    base_risk_features: pd.DataFrame | None = None

    # Calibration is deliberately restricted to the clean validation split.
    # Injected shift families belong only to evaluation; exposing the same
    # perturbation generator during calibration would overstate robustness.
    validation_variants = [("clean_validation", base_validation, base_graph_validation)]

    for variant_name, variant_frame, variant_graph_frame in validation_variants:
        variant_temporal_prediction = predict_forecast_model(
            "ridge_regression",
            temporal_model,
            variant_frame,
            temporal_feature_columns,
        )
        variant_graph_prediction = predict_forecast_model(
            "ridge_regression",
            graph_model,
            variant_graph_frame,
            graph_feature_columns,
        )
        variant_risk_frame = add_calibrated_mixture_risk_scores(
            pd.DataFrame(
                {
                    "pred_forecast": variant_temporal_prediction["y_pred"],
                    "pred_graph": variant_graph_prediction["y_pred"],
                }
            ),
            temporal_validation_mae,
            graph_validation_mae,
        )
        variant_ensemble_mean, variant_ensemble_std = _predict_temporal_uncertainty_ensemble(
            temporal_ensemble,
            variant_frame,
        )
        variant_service_risk = _service_risk_ms(
            variant_frame,
            volatility_scale_ms=service_risk_volatility_scale_ms,
        ).reset_index(drop=True)
        variant_features = pd.DataFrame(
            {
                "disagreement": variant_risk_frame[
                    "pred_disagreement_normalized"
                ],
                "ensemble_spread": variant_ensemble_std.reset_index(drop=True),
                "service_risk": variant_service_risk,
            }
        ).fillna(0.0)
        variant_error = (
            variant_temporal_prediction["y_true"].reset_index(drop=True)
            - variant_risk_frame["pred_calibrated_fusion"]
        ).abs()
        risk_feature_frames.append(variant_features)
        risk_target_frames.append(variant_error)
        if variant_name == "clean_validation":
            base_risk_features = variant_features

    validation_risk_features = pd.concat(risk_feature_frames, ignore_index=True)
    validation_absolute_error = pd.concat(risk_target_frames, ignore_index=True)
    residual_risk_model = LinearRegression(positive=True)
    residual_risk_model.fit(validation_risk_features, validation_absolute_error)
    if base_risk_features is None:
        raise ValueError("base validation risk features were not generated")
    validation_predicted_risk = np.clip(
        residual_risk_model.predict(base_risk_features),
        0.0,
        None,
    )
    risk_gate_ms = float(np.quantile(validation_predicted_risk, 0.90))

    # Fit a conservative positive-drift margin against per-path observation
    # age. This is zero for datasets that do not expose source-sample age.
    age_intercept_ms = 0.0
    age_slope_ms_per_second = 0.0
    if "observation_age_ms" in base_validation:
        age_seconds = (
            base_validation["observation_age_ms"].astype(float).clip(lower=0.0)
            / 1000.0
        ).to_numpy().reshape(-1, 1)
        positive_drift = (
            base_validation["target_next"].astype(float)
            - base_validation["latency_mean_ms"].astype(float)
        ).clip(lower=0.0).to_numpy()
        age_model = LinearRegression(positive=True)
        age_model.fit(age_seconds, positive_drift)
        fitted = age_model.predict(age_seconds)
        residual_radius = float(np.quantile(positive_drift - fitted, 0.90))
        age_intercept_ms = max(
            0.0,
            float(age_model.intercept_) + max(0.0, residual_radius),
        )
        age_slope_ms_per_second = max(0.0, float(age_model.coef_[0]))

    # Policy admission uses a fourth, independent chronological interval when
    # provided. Falling back to calibration is retained only for legacy scripts
    # and is explicitly marked in the exported reason.
    base_selection = (
        selection_frame
        if selection_frame is not None and not selection_frame.empty
        else base_validation
    ).reset_index(drop=True)
    base_graph_selection = (
        graph_selection
        if graph_selection is not None and not graph_selection.empty
        else base_graph_validation
    ).reset_index(drop=True)
    selection_temporal_prediction = predict_forecast_model(
        "ridge_regression",
        temporal_model,
        base_selection,
        temporal_feature_columns,
    )
    selection_graph_prediction = predict_forecast_model(
        "ridge_regression",
        graph_model,
        base_graph_selection,
        graph_feature_columns,
    )
    selection_ensemble_mean, selection_ensemble_std = (
        _predict_temporal_uncertainty_ensemble(
            temporal_ensemble,
            base_selection,
        )
    )
    selection_service_risk = _service_risk_ms(
        base_selection,
        volatility_scale_ms=service_risk_volatility_scale_ms,
    ).reset_index(drop=True)
    selection_candidates = pd.DataFrame(
        {
            "session_bin_index": base_selection["session_bin_index"].reset_index(
                drop=True
            ),
            "relative_path": base_selection["relative_path"].reset_index(drop=True),
            "latency_mean_ms": base_selection["latency_mean_ms"].reset_index(
                drop=True
            ),
            "target_next": base_selection["target_next"].reset_index(drop=True),
            "pred_graph": selection_graph_prediction["y_pred"].reset_index(drop=True),
            "pred_ensemble_uncertainty": (
                selection_ensemble_mean.reset_index(drop=True)
                + 0.75 * selection_ensemble_std.reset_index(drop=True)
                + selection_service_risk
            ),
        }
    )
    feasible_states = {"active", "available", "up", "healthy"}
    selection_candidates["is_feasible_path"] = (
        base_selection["path_state"].astype(str).str.lower().isin(feasible_states)
        & base_selection["observed_replies"].astype(float).gt(0.0)
    ).astype(int).to_numpy()
    selection_age_ms = (
        base_selection["observation_age_ms"].astype(float).clip(lower=0.0)
        if "observation_age_ms" in base_selection
        else pd.Series(0.0, index=base_selection.index)
    )
    selection_candidates["pred_staleness_margin_ms"] = (
        age_intercept_ms
        + age_slope_ms_per_second * selection_age_ms.to_numpy() / 1000.0
    )
    selection_candidates = add_qos_shielded_scores(
        selection_candidates,
        fallback_column="pred_graph",
        latency_budget_ms=latency_budget_ms,
        output_column="selection_graph_shield",
    )
    selection_candidates = add_qos_shielded_scores(
        selection_candidates,
        fallback_column="pred_ensemble_uncertainty",
        latency_budget_ms=latency_budget_ms,
        output_column="selection_ensemble_shield",
    )
    gate_samples = _build_actionable_gate_selection_samples(
        selection_candidates,
        base_selection,
        latency_budget_ms,
        risk_control_group_column=risk_control_group_column,
    )

    # Preserve the predictive-only variant, but select it on the independent
    # policy-selection interval rather than on calibration residuals.
    fallback_policy = select_validation_gated_fallback(
        gate_samples.realized_latency,
        latency_budget_ms,
        allowed_policies=("graph", "ensemble"),
    )
    gate_result = select_opportunity_aware_risk_controlled_policy(
        gate_samples.realized_latency,
        gate_samples.opportunity_mask,
        latency_budget_ms,
        config=risk_control_config,
        independence_group_ids=gate_samples.independence_group_ids,
    )
    validation_gated_fallback_policy = gate_result.selected_policy
    gate_evidence = gate_result.evidence_frame()
    gate_evidence["selection_epoch_count_total"] = gate_samples.total_epoch_count
    gate_evidence["actionable_decision_count"] = gate_samples.actionable_epoch_count
    gate_evidence["emergency_epoch_count_excluded"] = (
        gate_samples.excluded_emergency_epoch_count
    )
    gate_evidence["gate_conditioning_population"] = (
        "epochs_with_at_least_one_currently_feasible_path"
    )
    gate_evidence["actionable_conditioning"] = (
        "condition_on_at_least_one_currently_feasible_path_before_drawing_"
        "an_independent_group_and_then_an_epoch_within_group"
    )
    gate_evidence["emergency_epoch_treatment"] = "excluded_outage_no_action"
    gate_reason = gate_result.reason
    if gate_samples.excluded_emergency_epoch_count:
        gate_reason = (
            f"{gate_reason}; excluded "
            f"{gate_samples.excluded_emergency_epoch_count} emergency "
            "outage/no-action epochs"
        )
    if selection_frame is None:
        gate_reason = f"legacy calibration reuse: {gate_reason}"
    gate_evidence["selection_reason"] = gate_reason
    selected_gate_row = gate_evidence[
        gate_evidence["policy"].eq(validation_gated_fallback_policy)
    ].iloc[0]
    temporal_validation_mae = replace(
        temporal_validation_mae,
        residual_risk_intercept_ms=float(residual_risk_model.intercept_),
        residual_risk_disagreement_weight=float(residual_risk_model.coef_[0]),
        residual_risk_ensemble_weight=float(residual_risk_model.coef_[1]),
        residual_risk_service_weight=float(residual_risk_model.coef_[2]),
        residual_risk_gate_ms=risk_gate_ms,
        service_risk_volatility_scale_ms=service_risk_volatility_scale_ms,
        fallback_policy=fallback_policy,
        validation_gated_fallback_policy=validation_gated_fallback_policy,
        gate_selection_reason=gate_reason,
        gate_opportunity_count=int(selected_gate_row["opportunity_count"]),
        gate_effective_opportunity_count=float(
            selected_gate_row["effective_opportunity_count"]
        ),
        gate_selected_success_lcb=float(selected_gate_row["simultaneous_lcb"]),
        gate_selected_opportunity_success_lcb=float(
            selected_gate_row["opportunity_conditioned_success_delta_lcb"]
        ),
        gate_noninferiority_margin=float(
            selected_gate_row["noninferiority_margin"]
        ),
        gate_opportunity_noninferiority_margin=float(
            selected_gate_row["opportunity_noninferiority_margin"]
        ),
        gate_selected_aggregate_success_noninferior=bool(
            selected_gate_row["aggregate_actionable_success_noninferior"]
        ),
        gate_selected_opportunity_success_noninferior=bool(
            selected_gate_row["opportunity_conditioned_success_noninferior"]
        ),
        gate_practical_cvar_gain_ms=float(
            selected_gate_row["practical_cvar_gain_ms"]
        ),
        gate_selection_evidence_json=gate_evidence.to_json(orient="records"),
        age_margin_intercept_ms=age_intercept_ms,
        age_margin_slope_ms_per_second=age_slope_ms_per_second,
    )
    graph_validation_mae = replace(
        graph_validation_mae,
        residual_risk_intercept_ms=float(residual_risk_model.intercept_),
        residual_risk_disagreement_weight=float(residual_risk_model.coef_[0]),
        residual_risk_ensemble_weight=float(residual_risk_model.coef_[1]),
        residual_risk_service_weight=float(residual_risk_model.coef_[2]),
        residual_risk_gate_ms=risk_gate_ms,
        service_risk_volatility_scale_ms=service_risk_volatility_scale_ms,
        fallback_policy=fallback_policy,
        validation_gated_fallback_policy=validation_gated_fallback_policy,
        gate_selection_reason=gate_reason,
        gate_opportunity_count=int(selected_gate_row["opportunity_count"]),
        gate_effective_opportunity_count=float(
            selected_gate_row["effective_opportunity_count"]
        ),
        gate_selected_success_lcb=float(selected_gate_row["simultaneous_lcb"]),
        gate_selected_opportunity_success_lcb=float(
            selected_gate_row["opportunity_conditioned_success_delta_lcb"]
        ),
        gate_noninferiority_margin=float(
            selected_gate_row["noninferiority_margin"]
        ),
        gate_opportunity_noninferiority_margin=float(
            selected_gate_row["opportunity_noninferiority_margin"]
        ),
        gate_selected_aggregate_success_noninferior=bool(
            selected_gate_row["aggregate_actionable_success_noninferior"]
        ),
        gate_selected_opportunity_success_noninferior=bool(
            selected_gate_row["opportunity_conditioned_success_noninferior"]
        ),
        gate_practical_cvar_gain_ms=float(
            selected_gate_row["practical_cvar_gain_ms"]
        ),
        gate_selection_evidence_json=gate_evidence.to_json(orient="records"),
        age_margin_intercept_ms=age_intercept_ms,
        age_margin_slope_ms_per_second=age_slope_ms_per_second,
    )
    return (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
    )


def _evaluate_candidate_frame(
    scenario_name: str,
    candidate_frame: pd.DataFrame,
    latency_budget_ms: float,
    online_switch_penalty_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, decisions = evaluate_decision_policies(
        candidate_frame,
        latency_budget_ms=latency_budget_ms,
        policy_columns=POLICY_COLUMNS,
        online_switch_penalties_ms={
            "switch_aware_operational_selector": online_switch_penalty_ms,
        },
    )
    summary["scenario_name"] = scenario_name
    decisions["scenario_name"] = scenario_name

    significance = build_paired_policy_significance(
        decisions=decisions,
        comparisons=[
            ("qos_shielded_vs_reactive", "qos_shielded_operational_selector", "reactive_greedy"),
            ("qos_shielded_vs_graph", "qos_shielded_operational_selector", "predictive_graph_greedy"),
            ("qos_shielded_vs_ensemble", "qos_shielded_operational_selector", "ensemble_uncertainty_selector"),
            ("operational_vs_temporal", "calibrated_operational_selector", "predictive_greedy"),
            ("operational_vs_graph", "calibrated_operational_selector", "predictive_graph_greedy"),
            ("operational_vs_fusion", "calibrated_operational_selector", "predictive_simple_fusion_greedy"),
            ("operational_vs_ensemble_uncertainty", "calibrated_operational_selector", "ensemble_uncertainty_selector"),
            ("operational_vs_conformal_uncertainty", "calibrated_operational_selector", "conformal_uncertainty_selector"),
            ("operational_vs_historical", "calibrated_operational_selector", "best_historical_path"),
            ("switch_aware_vs_operational", "switch_aware_operational_selector", "calibrated_operational_selector"),
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
        segment_columns=("continuity_segment_id",),
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
        work["risk_adjusted_abs_error_ms"] = (work["pred_calibrated_operational"] - work["target_next"]).abs()
        work["temporal_abs_error_ms"] = (work["pred_forecast"] - work["target_next"]).abs()
        work["graph_abs_error_ms"] = (work["pred_graph"] - work["target_next"]).abs()
        work["service_failure"] = (work["target_next"] > latency_budget_ms).astype(int)

        scale = max(5.0, float(work["target_next"].std()))
        work["predicted_success_probability"] = _safe_sigmoid(work["pred_calibrated_operational"] - latency_budget_ms, scale)
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
        "calibrated_operational_selector",
    ]
    frame = summary[summary["policy_name"].isin(selected)].copy()
    scenario_order = [
        "session_holdout",
        "temporal_holdout",
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
        "calibrated_operational_selector",
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
    frame = calibration_bins[calibration_bins["scenario_name"].isin(["session_holdout", "temporal_holdout", "site_holdout", "operational_severe"])].copy()
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
        "calibrated_operational_selector",
    ]
    base = summary[(summary["scenario_name"] == "temporal_holdout") & summary["policy_name"].isin(selected)].copy()
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
        "This report summarizes structural-shift experiments for calibrated multi-signal LEO service-path selection.",
        "",
        "The disagreement score is validation-scaled before it enters the operational selector. That keeps the risk term tied to observed model fit rather than raw capacity mismatch.",
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
        "The structural-shift scenarios are injected only into the evaluation split. They are controlled stress tests, not claims that the raw release contains separate outage traces.",
        "",
        "The calibrated multi-signal policy is a lightweight operational decision rule rather than a new prediction-model family.",
        "",
        "The ensemble uncertainty selector is retained because it can be more conservative under moderate and severe degradation, where forecast spread becomes informative.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument("--output-dir", default="results/service_path_experiments")
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=None,
        help="Override the configured forecast horizon for cadence-sensitivity runs.",
    )
    parser.add_argument(
        "--allow-normalized-counterfactual",
        action="store_true",
        help=(
            "Align non-concurrent sessions by within-session stage. This is "
            "counterfactual diagnostic evidence, not deployable path selection."
        ),
    )
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
    time_bins, concurrency_audit = assign_decision_groups(
        time_bins,
        allow_normalized_counterfactual=args.allow_normalized_counterfactual,
    )
    target_retention_protocol = _target_retention_protocol(concurrency_audit)
    horizon_seconds = int(
        args.horizon_seconds
        if args.horizon_seconds is not None
        else config["forecasting"]["horizon_seconds"]
    )
    if horizon_seconds <= 0:
        raise ValueError("--horizon-seconds must be positive")
    observed_resolutions = sorted(
        int(value)
        for value in time_bins.get(
            "bin_seconds",
            pd.Series([config["graph"]["snapshot_seconds"]]),
        ).dropna().unique()
    )
    if len(observed_resolutions) != 1:
        raise ValueError(
            "one experiment run requires a single temporal resolution; found "
            + ", ".join(str(value) for value in observed_resolutions)
        )
    snapshot_seconds = observed_resolutions[0]
    horizon_bins = max(1, math.ceil(horizon_seconds / snapshot_seconds))
    latency_budget_ms = float(config["optimization"].get("latency_budget_ms", 60.0))
    consensus_cfg = config["optimization"].get("consensus", {})
    disagreement_cfg = config["optimization"].get("disagreement_aware", {})
    ensemble_cfg = config["optimization"].get("ensemble_uncertainty", {})
    service_risk_cfg = config["optimization"].get("service_risk", {})
    switching_penalties_ms = list(config["optimization"].get("switching_penalties_ms", [5.0, 10.0, 20.0]))
    online_switch_penalty_ms = float(
        config["optimization"].get("online_switch_penalty_ms", 10.0)
    )
    stochastic_switching_cfg = config["optimization"].get("stochastic_switching", {})
    multi_bin_horizons = list(config["optimization"].get("multi_bin_horizons", [3, 5]))
    consensus_config = ConsensusPolicyConfig(
        temporal_weight=float(consensus_cfg.get("temporal_weight", 0.65)),
        graph_weight=float(consensus_cfg.get("graph_weight", 0.35)),
        disagreement_penalty=float(consensus_cfg.get("disagreement_penalty", 0.30)),
    )
    disagreement_temporal_weight = float(disagreement_cfg.get("temporal_weight", 0.50))
    disagreement_graph_weight = float(disagreement_cfg.get("graph_weight", 0.50))
    disagreement_penalty = float(
        disagreement_cfg.get(
            "uncertainty_multiplier",
            disagreement_cfg.get("disagreement_penalty", 0.75),
        )
    )
    ensemble_penalty = float(ensemble_cfg.get("lambda_ens", 0.75))
    ensemble_members = int(ensemble_cfg.get("ensemble_members", 9))
    ensemble_row_fraction = float(ensemble_cfg.get("row_fraction", 0.82))
    ensemble_feature_fraction = float(ensemble_cfg.get("feature_fraction", 0.78))
    service_risk_reply_weight = float(service_risk_cfg.get("reply_pressure_penalty", 2.5))
    service_risk_volatility_weight = float(service_risk_cfg.get("volatility_penalty", 1.5))
    split_cfg = config["forecasting"].get(
        "policy_evaluation_ratios",
        {"train": 0.55, "calibration": 0.15, "selection": 0.15, "test": 0.15},
    )
    risk_cfg = config["optimization"].get("risk_control", {})
    risk_control_config = RiskControlConfig(
        alpha=float(risk_cfg.get("familywise_alpha", 0.05)),
        noninferiority_margin=float(risk_cfg.get("noninferiority_margin", 0.02)),
        opportunity_noninferiority_margin=float(
            risk_cfg.get("opportunity_noninferiority_margin", 0.02)
        ),
        minimum_effective_opportunities=float(
            risk_cfg.get("minimum_effective_opportunities", 5.0)
        ),
        practical_cvar_gain_ms=float(risk_cfg.get("practical_cvar_gain_ms", 1.0)),
        cvar_quantile=float(risk_cfg.get("cvar_quantile", 0.95)),
        block_length=(
            int(risk_cfg["block_length"])
            if risk_cfg.get("block_length") is not None
            else None
        ),
        latency_cap_ms=float(risk_cfg.get("latency_cap_ms", 60_000.0)),
        cvar_grid_points=int(risk_cfg.get("cvar_grid_points", 101)),
        planned_gate_uses=int(risk_cfg.get("planned_gate_uses", 1)),
        gate_use_index=int(risk_cfg.get("gate_use_index", 1)),
        bootstrap_samples=int(risk_cfg.get("bootstrap_samples", 4000)),
        random_seed=int(risk_cfg.get("random_seed", 2026)),
    )

    forecast_table = build_forecast_table(
        time_bins=time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
        decision_cadence_seconds=snapshot_seconds,
        multi_bin_horizons=multi_bin_horizons,
        require_complete_decision_epochs=bool(
            target_retention_protocol["require_complete_decision_epochs"]
        ),
    )
    holdout_locations = _choose_holdout_locations(forecast_table, holdout_count=args.holdout_count)
    dataset_summary = _build_dataset_summary(time_bins, forecast_table, holdout_locations)

    scenario_candidates: dict[str, pd.DataFrame] = {}
    policy_summaries: list[pd.DataFrame] = []
    policy_decisions: list[pd.DataFrame] = []
    significance_rows: list[pd.DataFrame] = []

    # Scenario 1: in-domain generalization across disjoint measurement sessions.
    # Complete sessions are held out so autocorrelated bins cannot cross splits.
    train, calibration, selection, test = split_group_train_calibration_selection_test(
        forecast_table,
        train_ratio=float(split_cfg["train"]),
        calibration_ratio=float(split_cfg["calibration"]),
        selection_ratio=float(split_cfg["selection"]),
        test_ratio=float(split_cfg["test"]),
        random_state=42,
    )
    graph_train = add_graph_snapshot_features(train)
    graph_calibration = add_graph_snapshot_features(calibration)
    graph_selection = add_graph_snapshot_features(selection)
    graph_test = add_graph_snapshot_features(test)
    (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
    ) = _fit_models(
        train,
        calibration,
        graph_train,
        graph_calibration,
        latency_budget_ms=latency_budget_ms,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
        selection_frame=selection,
        graph_selection=graph_selection,
        risk_control_config=risk_control_config,
    )
    candidate = _make_candidate_frame(
        test,
        graph_test,
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
        consensus_config=consensus_config,
        disagreement_temporal_weight=disagreement_temporal_weight,
        disagreement_graph_weight=disagreement_graph_weight,
        disagreement_penalty=disagreement_penalty,
        ensemble_penalty=ensemble_penalty,
        service_risk_reply_weight=service_risk_reply_weight,
        service_risk_volatility_weight=service_risk_volatility_weight,
    )
    scenario_candidates["session_holdout"] = candidate

    # Scenario 2: strict early-to-late temporal holdout within every session.
    (
        temporal_train,
        temporal_calibration,
        temporal_selection,
        temporal_test,
    ) = split_train_calibration_selection_test(
        forecast_table,
        train_ratio=float(split_cfg["train"]),
        calibration_ratio=float(split_cfg["calibration"]),
        selection_ratio=float(split_cfg["selection"]),
        test_ratio=float(split_cfg["test"]),
    )
    temporal_graph_train = add_graph_snapshot_features(temporal_train)
    temporal_graph_calibration = add_graph_snapshot_features(temporal_calibration)
    temporal_graph_selection = add_graph_snapshot_features(temporal_selection)
    temporal_graph_test = add_graph_snapshot_features(temporal_test)
    (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
    ) = _fit_models(
        temporal_train,
        temporal_calibration,
        temporal_graph_train,
        temporal_graph_calibration,
        latency_budget_ms=latency_budget_ms,
        ensemble_members=ensemble_members,
        ensemble_row_fraction=ensemble_row_fraction,
        ensemble_feature_fraction=ensemble_feature_fraction,
        selection_frame=temporal_selection,
        graph_selection=temporal_graph_selection,
        risk_control_config=risk_control_config,
    )
    temporal_candidate = _make_candidate_frame(
        temporal_test,
        temporal_graph_test,
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
        consensus_config=consensus_config,
        disagreement_temporal_weight=disagreement_temporal_weight,
        disagreement_graph_weight=disagreement_graph_weight,
        disagreement_penalty=disagreement_penalty,
        ensemble_penalty=ensemble_penalty,
        service_risk_reply_weight=service_risk_reply_weight,
        service_risk_volatility_weight=service_risk_volatility_weight,
    )
    scenario_candidates["temporal_holdout"] = temporal_candidate
    operational_model_bundle = (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
    )

    # Scenario 3: site holdout, trained without selected locations.
    if holdout_locations:
        site_train_table = forecast_table[
            ~forecast_table["location"].isin(holdout_locations)
        ].reset_index(drop=True)
        site_test = forecast_table[
            forecast_table["location"].isin(holdout_locations)
        ].reset_index(drop=True)
        site_train, site_calibration, site_selection, _ = (
            split_train_calibration_selection_test(
            site_train_table,
            train_ratio=float(split_cfg["train"]),
            calibration_ratio=float(split_cfg["calibration"]),
            selection_ratio=float(split_cfg["selection"]),
            test_ratio=float(split_cfg["test"]),
            )
        )
        site_graph_test = add_graph_snapshot_features(site_test)
        site_graph_train = add_graph_snapshot_features(site_train)
        site_graph_calibration = add_graph_snapshot_features(site_calibration)
        site_graph_selection = add_graph_snapshot_features(site_selection)
        (
            site_temporal_model,
            site_graph_model,
            site_temporal_ensemble,
            site_conformal_calibrator,
            site_temporal_feature_columns,
            site_graph_feature_columns,
            site_historical_latency,
            site_historical_fallback,
            site_temporal_validation,
            site_graph_validation,
        ) = _fit_models(
            site_train,
            site_calibration,
            site_graph_train,
            site_graph_calibration,
            latency_budget_ms=latency_budget_ms,
            ensemble_members=ensemble_members,
            ensemble_row_fraction=ensemble_row_fraction,
            ensemble_feature_fraction=ensemble_feature_fraction,
            selection_frame=site_selection,
            graph_selection=site_graph_selection,
            risk_control_config=risk_control_config,
        )
        site_candidate = _make_candidate_frame(
            site_test,
            site_graph_test,
            site_temporal_model,
            site_graph_model,
            site_temporal_ensemble,
            site_conformal_calibrator,
            site_temporal_feature_columns,
            site_graph_feature_columns,
            site_historical_latency,
            site_historical_fallback,
            site_temporal_validation,
            site_graph_validation,
            consensus_config=consensus_config,
            disagreement_temporal_weight=disagreement_temporal_weight,
            disagreement_graph_weight=disagreement_graph_weight,
            disagreement_penalty=disagreement_penalty,
            ensemble_penalty=ensemble_penalty,
            service_risk_reply_weight=service_risk_reply_weight,
            service_risk_volatility_weight=service_risk_volatility_weight,
        )
        scenario_candidates["site_holdout"] = site_candidate

    # Operational stress must use the earlier-to-later temporal model, not a
    # site-holdout model fitted for a different generalization question.
    (
        temporal_model,
        graph_model,
        temporal_ensemble,
        conformal_calibrator,
        temporal_feature_columns,
        graph_feature_columns,
        historical_latency,
        historical_fallback,
        temporal_validation_mae,
        graph_validation_mae,
    ) = operational_model_bundle

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
            temporal_model,
            graph_model,
            temporal_ensemble,
            conformal_calibrator,
            temporal_feature_columns,
            graph_feature_columns,
            historical_latency,
            historical_fallback,
            temporal_validation_mae,
            graph_validation_mae,
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
            online_switch_penalty_ms=online_switch_penalty_ms,
        )
        policy_summaries.append(summary)
        policy_decisions.append(decisions)
        significance_rows.append(significance)

    policy_summary = pd.concat(policy_summaries, ignore_index=True)
    decision_results = pd.concat(policy_decisions, ignore_index=True)
    xai_attribution_summary = summarize_xai_attribution(decision_results)
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
            segment_columns=("continuity_segment_id",),
        )
        scenario_ci["scenario_name"] = scenario_name
        ci_frames.append(scenario_ci)
    confidence_intervals = pd.concat(ci_frames, ignore_index=True)

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)
    decision_results.to_csv(output_dir / "policy_decisions.csv", index=False)
    xai_attribution_summary.to_csv(output_dir / "xai_attribution_summary.csv", index=False)
    pd.concat(
        [
            frame.assign(scenario_name=scenario_name)
            for scenario_name, frame in scenario_candidates.items()
        ],
        ignore_index=True,
    ).to_csv(output_dir / "candidate_predictions.csv", index=False)
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
        "concurrency_audit": concurrency_audit,
        "decision_window_seconds": snapshot_seconds,
        "forecast_horizon_seconds": horizon_seconds,
        "forecast_horizon_bins": horizon_bins,
        "target_retention_protocol": target_retention_protocol,
        "exact_horizon_audit": forecast_table.attrs.get(
            "exact_horizon_audit", {}
        ),
        "holdout_locations": holdout_locations,
        "policy_columns": POLICY_COLUMNS,
        "proposed_method_name": "qos_shielded_operational_selector",
        "compatibility_aliases": {
            "pred_disagreement_aware": "pred_calibrated_operational"
        },
        "consensus_config": asdict(consensus_config),
        "disagreement_aware_config": {
            "expert_pair": "matched_ridge",
            "uncertainty_multiplier": disagreement_penalty,
            "calibration_shift_exposure": "clean_calibration_only",
            "residual_risk_threshold": (
                "90th percentile of calibration-predicted absolute residual risk; "
                "the online snapshot median uses currently feasible paths only, "
                "zero-feasible snapshots are emergency/no-action epochs that "
                "fail closed to the audit fallback, and the normalized-"
                "disagreement threshold is retained only as an audit statistic"
            ),
            "fallback_selection": (
                "policy-selection-only lexicographic objective: maximize "
                "success under latency budget, then minimize realized latency"
            ),
            "temporal_calibration": calibration_to_dict(temporal_validation_mae),
            "graph_calibration": calibration_to_dict(graph_validation_mae),
        },
        "calibration_threshold_separation": {
            "conformal_radius": (
                "split-conformal quantile fitted from calibration residuals and "
                "used only by the conformal ranking baseline"
            ),
            "residual_risk_gate": (
                "90th percentile of calibration-predicted absolute residual "
                "risk and used only by the calibrated-risk ablation"
            ),
            "shared_test_information": False,
            "note": "the two thresholds are fitted independently and are not reused",
        },
        "ensemble_uncertainty_config": {
            "member_model": "ridge_regression",
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
        "online_switch_penalty_ms": online_switch_penalty_ms,
        "stochastic_switching": {
            "base_penalty_ms": float(stochastic_switching_cfg.get("base_penalty_ms", 10.0)),
            "spike_penalty_ms": float(stochastic_switching_cfg.get("spike_penalty_ms", 75.0)),
            "spike_probability": float(stochastic_switching_cfg.get("spike_probability", 0.10)),
            "n_trials": int(stochastic_switching_cfg.get("n_trials", 256)),
            "random_state": 42,
        },
        "multi_bin_horizons": multi_bin_horizons,
        "xai_outputs": {
            "decision_level_columns": [
                "xai_latency_component_ms",
                "xai_disagreement_component_ms",
                "xai_uncertainty_component_ms",
                "xai_service_risk_component_ms",
                "xai_switch_component_ms",
                "xai_calibration_component_ms",
                "xai_score_branch",
                "xai_gate_active",
                "xai_fallback_policy",
                "xai_dominant_component",
                "xai_runner_up_relative_path",
                "xai_score_margin_ms",
                "xai_counterfactual_reason",
            ],
            "aggregate_file": "xai_attribution_summary.csv",
            "explanation_type": "faithful additive score attribution plus nearest-rejected-path counterfactual",
        },
        "config_path": _resolve_repo_path(args.config).relative_to(REPO_ROOT).as_posix(),
        "time_bins_path": time_bins_path.relative_to(REPO_ROOT).as_posix(),
        "split_random_seeds": {
            "session_holdout_split": 42,
            "operational_mild": 101,
            "operational_moderate": 102,
            "operational_severe": 103,
            "bootstrap_confidence_intervals": 42,
        },
        "statistical_dependence_control": {
            "confidence_intervals": (
                "segment-stratified circular moving-block bootstrap"
            ),
            "paired_delta_intervals": (
                "segment-stratified circular moving-block bootstrap"
            ),
            "segment_columns": ["continuity_segment_id"],
            "segment_boundary_semantics": (
                "continuity segments reset at telemetry gaps and explicit "
                "session/campaign changes; circular wraps remain within a "
                "segment and metric-missing rows create additional breaks"
            ),
            "block_length_rule": "cube root of decision-window count",
            "additional_test": "centered block-bootstrap p-value",
        },
        "data_partition_protocol": {
            "target_retention": target_retention_protocol["reason"],
            "session_holdout": (
                "whole relative_path groups are assigned to disjoint training, "
                "calibration, policy-selection, and test partitions"
            ),
            "temporal_holdout": (
                "the shared wall-clock axis is split chronologically into "
                "training, calibration, policy-selection, and test blocks; "
                "boundary-crossing targets are removed"
            ),
            "graph_context_isolation": (
                "graph snapshot features are constructed separately inside "
                "each partition after splitting"
            ),
            "multi_seed_claim": (
                "each regenerated seed is trained and calibrated within that "
                "trace; this tests environment-level reproducibility with "
                "per-seed calibration, not one-model transfer to unseen seeds"
            ),
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
                ["best_historical_path", "predictive_greedy", "predictive_graph_greedy", "calibrated_operational_selector"]
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
