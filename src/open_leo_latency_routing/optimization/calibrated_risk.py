"""Validation-calibrated risk scores for two complementary path experts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpertCalibration:
    """Robust residual statistics estimated without evaluation data."""

    residual_bias_ms: float
    residual_scale_ms: float
    upper_residual_quantile_ms: float
    mae_ms: float
    sample_count: int
    paired_residual_covariance_ms2: float = 0.0
    residual_variance_ms2: float | None = None
    normalized_disagreement_gate: float = 0.5
    residual_risk_intercept_ms: float = 0.0
    residual_risk_disagreement_weight: float = 0.0
    residual_risk_ensemble_weight: float = 0.0
    residual_risk_service_weight: float = 0.0
    residual_risk_gate_ms: float = 0.0
    service_risk_volatility_scale_ms: float = 1.0
    fallback_policy: str = "ensemble"
    validation_gated_fallback_policy: str = "reactive"
    gate_selection_reason: str = "legacy point-estimate selection"
    gate_opportunity_count: int = 0
    gate_effective_opportunity_count: float = 0.0
    gate_selected_success_lcb: float = 0.0
    gate_selected_opportunity_success_lcb: float = 0.0
    gate_noninferiority_margin: float = 0.0
    gate_opportunity_noninferiority_margin: float = 0.0
    gate_selected_aggregate_success_noninferior: bool = False
    gate_selected_opportunity_success_noninferior: bool = False
    gate_practical_cvar_gain_ms: float = 0.0
    gate_selection_evidence_json: str = "[]"
    age_margin_intercept_ms: float = 0.0
    age_margin_slope_ms_per_second: float = 0.0


@dataclass(frozen=True)
class CalibratedRiskConfig:
    """Configuration for the covariance-aware fusion-risk score."""

    uncertainty_multiplier: float = 0.75
    service_risk_multiplier: float = 1.0
    output_column: str = "pred_calibrated_risk"


def fit_expert_calibration(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    coverage: float = 0.90,
) -> ExpertCalibration:
    """Estimate robust bias, scale, and one-sided residual radius.

    The residual scale is a robust Gaussian-consistent MAD estimate, with MAE
    as a fallback for nearly constant validation residuals.
    """

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    residual = truth - prediction
    bias = float(np.median(residual))
    centered = residual - bias
    mae = float(np.mean(np.abs(residual)))
    robust_scale = float(1.4826 * np.median(np.abs(centered)))
    scale = max(robust_scale, mae, 1e-6)
    variance = (
        float(np.var(centered, ddof=1))
        if len(centered) > 1
        else float(scale**2)
    )
    upper_quantile = float(np.quantile(residual, coverage))
    return ExpertCalibration(
        residual_bias_ms=bias,
        residual_scale_ms=scale,
        upper_residual_quantile_ms=upper_quantile,
        mae_ms=mae,
        sample_count=int(len(residual)),
        residual_variance_ms2=max(variance, 1e-12),
    )


def add_calibrated_mixture_risk_scores(
    candidate_frame: pd.DataFrame,
    temporal_calibration: ExpertCalibration,
    graph_calibration: ExpertCalibration,
    config: CalibratedRiskConfig | None = None,
    temporal_column: str = "pred_forecast",
    graph_column: str = "pred_graph",
    service_risk_column: str | None = None,
) -> pd.DataFrame:
    """Add calibrated fusion, disagreement, and operational-risk scores.

    The executed predictor is a linear pool, not a random mixture.  Its error
    variance therefore uses squared weights and the paired residual covariance
    estimated on calibration data.  Expert disagreement is exported as a
    separate diagnostic/ranking feature; it is not folded into the variance as
    though the controller randomly selected one of the two experts.

    The historical function name and ``pred_mixture_std`` column are retained
    for artifact compatibility.  ``pred_fusion_error_std`` is the preferred
    name for new outputs.
    """

    risk_config = config or CalibratedRiskConfig()
    output = candidate_frame.copy()

    temporal_mean = (
        output[temporal_column].astype(float) + temporal_calibration.residual_bias_ms
    )
    graph_mean = output[graph_column].astype(float) + graph_calibration.residual_bias_ms
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
    covariance = float(
        0.5
        * (
            temporal_calibration.paired_residual_covariance_ms2
            + graph_calibration.paired_residual_covariance_ms2
        )
    )
    covariance_limit = math.sqrt(max(temporal_variance * graph_variance, 0.0))
    covariance = float(np.clip(covariance, -covariance_limit, covariance_limit))
    pooling_denominator = temporal_variance + graph_variance - 2.0 * covariance
    if pooling_denominator > 1e-12:
        temporal_weight = float(
            np.clip(
                (graph_variance - covariance) / pooling_denominator,
                0.0,
                1.0,
            )
        )
    else:
        temporal_precision = 1.0 / max(temporal_variance, 1e-12)
        graph_precision = 1.0 / max(graph_variance, 1e-12)
        temporal_weight = temporal_precision / (
            temporal_precision + graph_precision
        )
    graph_weight = 1.0 - temporal_weight

    mixture_mean = temporal_weight * temporal_mean + graph_weight * graph_mean
    raw_disagreement = (temporal_mean - graph_mean).abs()
    disagreement_variance = max(pooling_denominator, 1e-12)
    disagreement_scale = np.sqrt(disagreement_variance)
    normalized_disagreement = raw_disagreement / max(disagreement_scale, 1e-6)
    fusion_variance = (
        temporal_weight**2 * temporal_variance
        + graph_weight**2 * graph_variance
        + 2.0 * temporal_weight * graph_weight * covariance
    )
    fusion_std = float(np.sqrt(max(fusion_variance, 0.0)))

    if service_risk_column and service_risk_column in output:
        service_risk = output[service_risk_column].astype(float)
    else:
        service_risk = pd.Series(0.0, index=output.index)

    output["pred_temporal_calibrated"] = temporal_mean
    output["pred_graph_calibrated"] = graph_mean
    output["pred_calibrated_fusion"] = mixture_mean
    output["pred_disagreement"] = raw_disagreement
    output["pred_disagreement_normalized"] = normalized_disagreement
    output["pred_fusion_error_std"] = fusion_std
    output["pred_mixture_std"] = fusion_std
    output["pred_disagreement_only"] = (
        mixture_mean
        + risk_config.uncertainty_multiplier
        * np.sqrt(temporal_weight * graph_weight)
        * raw_disagreement
    )
    output[risk_config.output_column] = (
        mixture_mean
        + risk_config.uncertainty_multiplier * fusion_std
        + risk_config.service_risk_multiplier * service_risk
    )
    output["temporal_expert_weight"] = temporal_weight
    output["graph_expert_weight"] = graph_weight
    output["paired_residual_covariance_ms2"] = covariance
    return output


def calibration_to_dict(calibration: ExpertCalibration) -> dict[str, object]:
    """Return JSON-safe expert calibration metadata."""

    return asdict(calibration)
