"""Frozen external gate audit for the CoNEXT'23 mobility measurements.

The public release contains timestamped UDP-Ping RTT records for a Starlink
Mobility terminal and cellular handsets measured side by side in a vehicle.
This module aggregates successful replies into exact one-second bins.  It does
not invent timestamps for source rows recorded only as ``timeout``.

The experiment reserves 2023-04-23 for model development.  The five following
calendar dates are treated as separately restarted acquisition groups.  That
day-level independence and exchangeability assumption is declared rather than
inferred from the observations, and all five dates remain one acquisition
family from one published field trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


PRIMARY_PROVIDERS = ("mob", "tm", "vz")
ROAM_SENSITIVITY_PROVIDERS = ("mob", "rm", "tm", "vz")
ALL_SOURCE_PROVIDERS = ("mob", "rm", "att", "tm", "vz")
# Backward-compatible name for the frozen primary audit.
PROVIDERS = PRIMARY_PROVIDERS
DEVELOPMENT_DATE = "2023-04-23"
EVIDENCE_DATES = tuple(f"2023-04-{day:02d}" for day in range(24, 29))


@dataclass(frozen=True)
class FrozenProviderModels:
    """Two frozen ridge experts and calibration-only fusion parameters."""

    temporal: Pipeline
    context: Pipeline
    temporal_features: tuple[str, ...]
    context_features: tuple[str, ...]
    temporal_bias_ms: float
    context_bias_ms: float
    temporal_weight: float
    context_weight: float


def minimum_variance_fusion_weights(
    temporal_residual: np.ndarray,
    context_residual: np.ndarray,
) -> tuple[float, float, dict[str, float | str]]:
    """Return bounded Bates--Granger weights from paired residuals.

    Residuals must come from the same held-out calibration targets.  The
    covariance is clipped to its Cauchy--Schwarz limit for numerical safety.
    If the forecast-difference variance is degenerate, inverse-variance
    weights provide a deterministic fallback.  No evidence-campaign outcome
    is involved in this calculation.
    """

    temporal = np.asarray(temporal_residual, dtype=float)
    context = np.asarray(context_residual, dtype=float)
    if temporal.ndim != 1 or context.ndim != 1 or len(temporal) != len(context):
        raise ValueError(
            "paired residual arrays must be one-dimensional and equal length"
        )
    if len(temporal) < 2:
        raise ValueError("at least two paired calibration residuals are required")
    if not (np.isfinite(temporal).all() and np.isfinite(context).all()):
        raise ValueError("paired calibration residuals must be finite")

    temporal_variance = float(np.var(temporal, ddof=1))
    context_variance = float(np.var(context, ddof=1))
    covariance = float(np.cov(temporal, context, ddof=1)[0, 1])
    covariance_limit = float(
        np.sqrt(max(temporal_variance * context_variance, 0.0))
    )
    covariance = float(np.clip(covariance, -covariance_limit, covariance_limit))
    denominator = temporal_variance + context_variance - 2.0 * covariance
    if denominator > 1e-12:
        temporal_weight = float(
            np.clip((context_variance - covariance) / denominator, 0.0, 1.0)
        )
        method = "bounded_minimum_variance"
    else:
        temporal_precision = 1.0 / max(temporal_variance, 1e-12)
        context_precision = 1.0 / max(context_variance, 1e-12)
        temporal_weight = float(
            temporal_precision / (temporal_precision + context_precision)
        )
        method = "inverse_variance_degenerate_fallback"
    context_weight = 1.0 - temporal_weight
    diagnostics: dict[str, float | str] = {
        "method": method,
        "temporal_residual_variance_ms2": temporal_variance,
        "context_residual_variance_ms2": context_variance,
        "paired_residual_covariance_ms2": covariance,
        "forecast_difference_variance_ms2": denominator,
    }
    return temporal_weight, context_weight, diagnostics


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one source artifact."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provider_source_path(dataset_root: Path, provider: str) -> Path:
    """Resolve either the original ZIP or an extracted provider CSV."""

    candidates = (
        dataset_root / f"{provider}_udpping_summary.zip",
        dataset_root / f"{provider}_udpping_summary.csv",
        dataset_root
        / f"{provider}_udpping_summary"
        / f"{provider}_udpping_summary.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"missing {provider} UDP-Ping archive under {dataset_root}"
    )


def load_provider_seconds(dataset_root: Path, provider: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load valid replies and aggregate them into absolute one-second bins."""

    source = provider_source_path(dataset_root, provider)
    raw = pd.read_csv(source, dtype=str)
    if set(raw.columns) != {"time", "RTT"}:
        raise ValueError(f"unexpected UDP-Ping schema in {source}")

    timestamps = pd.to_datetime(
        raw["time"].where(raw["time"].str.startswith("2023-")),
        format="%Y-%m-%d %H:%M:%S.%f CDT",
        errors="coerce",
    )
    rtt = pd.to_numeric(raw["RTT"], errors="coerce")
    valid = pd.DataFrame({"timestamp": timestamps, "rtt_ms": rtt}).dropna()
    valid = valid[valid["rtt_ms"].gt(0.0)].copy()
    valid["second"] = valid["timestamp"].dt.floor("s")

    aggregate = valid.groupby("second", sort=True)["rtt_ms"].agg(
        ["mean", "std", "count", "max"]
    )
    aggregate = aggregate.rename(
        columns={
            "mean": f"{provider}_rtt",
            "std": f"{provider}_std",
            "count": f"{provider}_replies",
            "max": f"{provider}_max",
        }
    )
    aggregate[f"{provider}_std"] = aggregate[f"{provider}_std"].fillna(0.0)
    metadata = {
        "provider": provider,
        "source_path": source.relative_to(dataset_root).as_posix(),
        "source_sha256": sha256_file(source),
        "raw_row_count": int(len(raw)),
        "valid_timestamped_reply_count": int(len(valid)),
        "unassignable_timeout_or_malformed_row_count": int(
            len(raw) - len(valid)
        ),
        "aggregated_second_count": int(len(aggregate)),
    }
    return aggregate, metadata


def audit_provider_date_coverage(
    dataset_root: Path,
    providers: tuple[str, ...] = ALL_SOURCE_PROVIDERS,
) -> pd.DataFrame:
    """Report timestamped one-second coverage for every released service."""

    rows: list[dict[str, Any]] = []
    required_dates = (DEVELOPMENT_DATE, *EVIDENCE_DATES)
    for provider in providers:
        frame, metadata = load_provider_seconds(dataset_root, provider)
        available_dates = set(frame.index.strftime("%Y-%m-%d"))
        missing_dates = [date for date in required_dates if date not in available_dates]
        rows.append(
            {
                "provider": provider,
                "aggregated_second_count": int(len(frame)),
                "first_timestamp": frame.index.min().isoformat(),
                "last_timestamp": frame.index.max().isoformat(),
                "development_date_present": DEVELOPMENT_DATE in available_dates,
                "evidence_date_count": sum(
                    date in available_dates for date in EVIDENCE_DATES
                ),
                "all_required_dates_present": not missing_dates,
                "missing_required_dates": ";".join(missing_dates),
                "source_sha256": metadata["source_sha256"],
            }
        )
    return pd.DataFrame.from_records(rows)


def build_concurrent_seconds(
    dataset_root: Path,
    providers: tuple[str, ...] = PROVIDERS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Inner-align one-second summaries from every declared candidate path."""

    frames: list[pd.DataFrame] = []
    source_metadata: list[dict[str, Any]] = []
    for provider in providers:
        frame, metadata = load_provider_seconds(dataset_root, provider)
        frames.append(frame)
        source_metadata.append(metadata)
    concurrent = pd.concat(frames, axis=1, join="inner").sort_index()
    concurrent.index.name = "decision_second"
    concurrent["campaign_id"] = concurrent.index.strftime(
        "conext23_%Y%m%d"
    )
    concurrent["collection_date"] = concurrent.index.strftime("%Y-%m-%d")
    return concurrent, source_metadata


def add_exact_forecast_features(
    concurrent: pd.DataFrame,
    providers: tuple[str, ...] = PROVIDERS,
) -> pd.DataFrame:
    """Add exact one-second histories and adjacent-bin outcomes.

    Reindexing by timestamps prevents a later observation after a gap from
    being misinterpreted as the next scheduled second.
    """

    frame = concurrent.copy()
    for provider in providers:
        current = frame[f"{provider}_rtt"]
        for lag in (1, 2):
            frame[f"{provider}_lag{lag}"] = current.reindex(
                frame.index - pd.Timedelta(seconds=lag)
            ).to_numpy()
        frame[f"{provider}_roll3"] = frame[
            [f"{provider}_rtt", f"{provider}_lag1", f"{provider}_lag2"]
        ].mean(axis=1)
        frame[f"{provider}_target"] = current.reindex(
            frame.index + pd.Timedelta(seconds=1)
        ).to_numpy()
        peers = [candidate for candidate in providers if candidate != provider]
        frame[f"{provider}_peer_min"] = frame[
            [f"{peer}_rtt" for peer in peers]
        ].min(axis=1)
        frame[f"{provider}_peer_mean"] = frame[
            [f"{peer}_rtt" for peer in peers]
        ].mean(axis=1)

    required: list[str] = []
    for provider in providers:
        required.extend(
            [
                f"{provider}_rtt",
                f"{provider}_std",
                f"{provider}_replies",
                f"{provider}_lag1",
                f"{provider}_lag2",
                f"{provider}_roll3",
                f"{provider}_target",
                f"{provider}_peer_min",
                f"{provider}_peer_mean",
            ]
        )
    return frame.dropna(subset=required).copy()


def _feature_columns(provider: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    temporal = (
        f"{provider}_rtt",
        f"{provider}_lag1",
        f"{provider}_lag2",
        f"{provider}_roll3",
        f"{provider}_std",
        f"{provider}_replies",
    )
    context = temporal + (
        f"{provider}_peer_min",
        f"{provider}_peer_mean",
    )
    return temporal, context


def fit_frozen_models(
    development: pd.DataFrame,
    providers: tuple[str, ...] = PROVIDERS,
    *,
    train_fraction: float = 0.70,
    ridge_alpha: float = 1.0,
) -> tuple[dict[str, FrozenProviderModels], dict[str, Any]]:
    """Fit and calibrate the fixed candidate on development-only rows."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    ordered = development.sort_index()
    if len(ordered) < 20:
        raise ValueError("development campaign is too short")
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * train_fraction)))
    split_second = ordered.index[split_index]
    # Keep an unused one-second boundary between fit and calibration outcomes.
    train = ordered[ordered.index < split_second - pd.Timedelta(seconds=1)]
    calibration = ordered[ordered.index >= split_second]
    if train.empty or calibration.empty:
        raise ValueError("development split produced an empty partition")

    models: dict[str, FrozenProviderModels] = {}
    manifest: dict[str, Any] = {
        "development_date": DEVELOPMENT_DATE,
        "train_fraction": train_fraction,
        "ridge_alpha": ridge_alpha,
        "fit_row_count": int(len(train)),
        "calibration_row_count": int(len(calibration)),
        "fit_last_second": train.index.max().isoformat(),
        "calibration_first_second": calibration.index.min().isoformat(),
        "providers": {},
    }
    for provider in providers:
        temporal_features, context_features = _feature_columns(provider)
        temporal = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        context = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        target = f"{provider}_target"
        temporal.fit(train[list(temporal_features)], train[target])
        context.fit(train[list(context_features)], train[target])
        temporal_calibration_prediction = temporal.predict(
            calibration[list(temporal_features)]
        )
        context_calibration_prediction = context.predict(
            calibration[list(context_features)]
        )
        calibration_truth = calibration[target].to_numpy(dtype=float)
        temporal_bias = float(
            np.median(calibration_truth - temporal_calibration_prediction)
        )
        context_bias = float(
            np.median(calibration_truth - context_calibration_prediction)
        )
        temporal_residual = (
            calibration_truth - temporal_calibration_prediction - temporal_bias
        )
        context_residual = (
            calibration_truth - context_calibration_prediction - context_bias
        )
        temporal_weight, context_weight, fusion_diagnostics = (
            minimum_variance_fusion_weights(temporal_residual, context_residual)
        )
        models[provider] = FrozenProviderModels(
            temporal=temporal,
            context=context,
            temporal_features=temporal_features,
            context_features=context_features,
            temporal_bias_ms=temporal_bias,
            context_bias_ms=context_bias,
            temporal_weight=temporal_weight,
            context_weight=context_weight,
        )
        manifest["providers"][provider] = {
            "temporal_features": temporal_features,
            "context_features": context_features,
            "temporal_bias_ms": temporal_bias,
            "context_bias_ms": context_bias,
            "fusion_weights": {
                "temporal": temporal_weight,
                "context": context_weight,
            },
            "fusion_weight_diagnostics": fusion_diagnostics,
        }
    return models, manifest


def score_frozen_candidate(
    evidence: pd.DataFrame,
    models: dict[str, FrozenProviderModels],
    providers: tuple[str, ...] = PROVIDERS,
    *,
    latency_target_ms: float = 60.0,
) -> pd.DataFrame:
    """Apply the frozen current-state filter and predictive ranking rule."""

    scored = evidence.copy()
    for provider in providers:
        model = models[provider]
        temporal_prediction = (
            model.temporal.predict(scored[list(model.temporal_features)])
            + model.temporal_bias_ms
        )
        context_prediction = (
            model.context.predict(scored[list(model.context_features)])
            + model.context_bias_ms
        )
        scored[f"{provider}_prediction"] = (
            model.temporal_weight * temporal_prediction
            + model.context_weight * context_prediction
        )

    records: list[dict[str, Any]] = []
    for timestamp, row in scored.iterrows():
        current = {provider: float(row[f"{provider}_rtt"]) for provider in providers}
        outcome = {
            provider: float(row[f"{provider}_target"]) for provider in providers
        }
        reactive_path = min(providers, key=current.__getitem__)
        compliant = [
            provider
            for provider in providers
            if current[provider] <= latency_target_ms
        ]
        ranking_set = compliant if compliant else list(providers)
        candidate_path = min(
            ranking_set,
            key=lambda provider: float(row[f"{provider}_prediction"]),
        )
        future_success = {
            provider: outcome[provider] <= latency_target_ms
            for provider in providers
        }
        generic_opportunity = any(future_success.values()) and not all(
            future_success.values()
        )
        records.append(
            {
                "decision_second": timestamp,
                "campaign_id": str(row["campaign_id"]),
                "collection_date": str(row["collection_date"]),
                "reactive_path": reactive_path,
                "candidate_path": candidate_path,
                "reactive_latency_ms": outcome[reactive_path],
                "candidate_latency_ms": outcome[candidate_path],
                "reactive_success": int(future_success[reactive_path]),
                "candidate_success": int(future_success[candidate_path]),
                "generic_opportunity": int(generic_opportunity),
                "policy_active_opportunity": int(
                    generic_opportunity and candidate_path != reactive_path
                ),
                **{
                    f"{provider}_current_rtt_ms": current[provider]
                    for provider in providers
                },
                **{
                    f"{provider}_next_rtt_ms": outcome[provider]
                    for provider in providers
                },
                **{
                    f"{provider}_predicted_next_rtt_ms": float(
                        row[f"{provider}_prediction"]
                    )
                    for provider in providers
                },
            }
        )
    return pd.DataFrame.from_records(records)
