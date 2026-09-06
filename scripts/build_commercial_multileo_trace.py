#!/usr/bin/env python3
"""Build and audit a synchronized commercial multi-LEO latency trace.

The importer is deliberately schema-driven because access to the candidate
Starlink--OneWeb campaign must be granted by its owners. It refuses to label a
trace as same-controller path-selection evidence unless temporal concurrency,
complete GPS co-location, and shared-controller provenance all pass. Duration
and independence are additional requirements for limitation-closing claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.loaders import (  # noqa: E402
    audit_trace_concurrency,
    validate_time_bin_table,
)


REQUIRED_MAP_FIELDS = {"timestamp", "operator", "latency_ms"}
OPERATOR_ALIASES = {
    "starlink": "starlink",
    "spacex": "starlink",
    "spacex starlink": "starlink",
    "oneweb": "oneweb",
    "eutelsat oneweb": "oneweb",
    "eutelsat_oneweb": "oneweb",
}
CLAIM_MINIMUM_DURATION_DAYS = 30.0
CLAIM_MINIMUM_CONCURRENT_HOURS = 24.0
CLAIM_MAXIMUM_P95_SKEW_MS = 100.0
# A synchronized trace is not a selectable-path trace when the terminals are
# geographically separated.  The hard floor below is intentionally a maximum:
# callers may require tighter co-location, but cannot relax the claim gate.
CLAIM_MAXIMUM_INTER_OPERATOR_DISTANCE_METERS = 100.0


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_column_map(path: Path) -> dict[str, str]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_MAP_FIELDS - set(mapping))
    if missing:
        raise ValueError("column map is missing: " + ", ".join(missing))
    return {str(key): str(value) for key, value in mapping.items()}


def _normalize_operator(value: object) -> str | None:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    return OPERATOR_ALIASES.get(normalized)


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "yes", "received", "success", "up"}
    false_values = {"0", "false", "no", "lost", "failure", "down", "nan", "none"}
    unknown = ~normalized.isin(true_values | false_values)
    if unknown.any():
        examples = sorted(normalized[unknown].drop_duplicates().head(5).tolist())
        raise ValueError(f"unrecognized packet_received values: {examples}")
    return normalized.isin(true_values)


def _coerce_coordinate(series: pd.Series, *, name: str) -> pd.Series:
    """Parse a coordinate while distinguishing missing from malformed values."""

    numeric = pd.to_numeric(series, errors="coerce")
    supplied = series.notna() & series.astype(str).str.strip().ne("")
    malformed = supplied & numeric.isna()
    if malformed.any():
        examples = sorted(series[malformed].astype(str).drop_duplicates().head(5).tolist())
        raise ValueError(f"{name} contains non-numeric values: {examples}")
    lower, upper = (-90.0, 90.0) if name == "latitude" else (-180.0, 180.0)
    out_of_range = numeric.notna() & ~numeric.between(lower, upper)
    if out_of_range.any():
        examples = sorted(numeric[out_of_range].drop_duplicates().head(5).tolist())
        raise ValueError(f"{name} is outside [{lower}, {upper}]: {examples}")
    return numeric.astype(float)


def _haversine_meters(
    latitude_a: pd.Series,
    longitude_a: pd.Series,
    latitude_b: pd.Series,
    longitude_b: pd.Series,
) -> pd.Series:
    """Return great-circle distance between paired WGS84 coordinates."""

    radius_m = 6_371_008.8
    lat_a = np.radians(latitude_a.astype(float))
    lon_a = np.radians(longitude_a.astype(float))
    lat_b = np.radians(latitude_b.astype(float))
    lon_b = np.radians(longitude_b.astype(float))
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arctan2(
        np.sqrt(haversine),
        np.sqrt(np.maximum(0.0, 1.0 - haversine)),
    )
    return pd.Series(radius_m * central_angle, index=latitude_a.index)


def build_commercial_multileo_trace(
    source_path: Path,
    column_map: dict[str, str],
    *,
    bin_seconds: int,
    timeout_ms: float,
    minimum_duration_days: float,
    minimum_concurrent_hours: float,
    maximum_p95_skew_ms: float,
    dataset_name: str,
    dataset_url: str,
    dataset_doi: str,
    license_name: str,
    independent_provenance: bool,
    maximum_inter_operator_distance_meters: float = (
        CLAIM_MAXIMUM_INTER_OPERATOR_DISTANCE_METERS
    ),
    same_controller_provenance: bool = False,
    controller_provenance_note: str = "",
    independent_campaign_ids_audited: bool = False,
    campaign_independence_note: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert packet observations and return a claim-gated metadata record."""

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive")
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be positive")
    if minimum_duration_days <= 0:
        raise ValueError("minimum_duration_days must be positive")
    if minimum_concurrent_hours <= 0:
        raise ValueError("minimum_concurrent_hours must be positive")
    if maximum_p95_skew_ms <= 0:
        raise ValueError("maximum_p95_skew_ms must be positive")
    if maximum_inter_operator_distance_meters <= 0:
        raise ValueError("maximum_inter_operator_distance_meters must be positive")
    controller_provenance_note = controller_provenance_note.strip()
    if same_controller_provenance and not controller_provenance_note:
        raise ValueError(
            "same_controller_provenance requires a non-empty documented "
            "controller_provenance_note"
        )
    campaign_independence_note = campaign_independence_note.strip()
    if independent_campaign_ids_audited and not campaign_independence_note:
        raise ValueError(
            "independent_campaign_ids_audited requires a non-empty documented "
            "campaign_independence_note"
        )

    source = pd.read_csv(source_path)
    missing_columns = sorted(
        source_name for source_name in column_map.values() if source_name not in source
    )
    if missing_columns:
        raise ValueError("source CSV is missing columns: " + ", ".join(missing_columns))

    work = pd.DataFrame(index=source.index)
    for canonical_name, source_name in column_map.items():
        work[canonical_name] = source[source_name]
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if work["timestamp"].isna().any():
        raise ValueError("timestamp parsing failed; provide timezone-aware ISO timestamps")
    work["operator"] = work["operator"].map(_normalize_operator)
    unknown_operator_rows = int(work["operator"].isna().sum())
    work = work.dropna(subset=["operator"]).copy()
    observed_operators = set(work["operator"].unique())
    if observed_operators != {"starlink", "oneweb"}:
        raise ValueError(
            "trace must contain commercial Starlink and OneWeb observations; "
            f"found {sorted(observed_operators)}"
        )

    work["latency_ms"] = pd.to_numeric(work["latency_ms"], errors="coerce")
    if "packet_received" in work:
        work["packet_received"] = _coerce_boolean(work["packet_received"])
        inconsistent = work["packet_received"] & work["latency_ms"].isna()
        if inconsistent.any():
            raise ValueError("received packets must have a finite latency outcome")
    else:
        work["packet_received"] = work["latency_ms"].notna()
    work.loc[~work["packet_received"], "latency_ms"] = np.nan
    coordinate_fields_present = {"latitude", "longitude"}.issubset(work.columns)
    if "latitude" in work:
        work["latitude"] = _coerce_coordinate(work["latitude"], name="latitude")
    if "longitude" in work:
        work["longitude"] = _coerce_coordinate(work["longitude"], name="longitude")
    if "controller_id" in work:
        work["controller_id"] = work["controller_id"].astype("string").str.strip()
        work.loc[work["controller_id"].eq(""), "controller_id"] = pd.NA
    if "campaign_id" in work:
        work["campaign_id"] = work["campaign_id"].astype("string").str.strip()
        work.loc[work["campaign_id"].eq(""), "campaign_id"] = pd.NA
    work["bin_start_utc"] = work["timestamp"].dt.floor(f"{bin_seconds}s")

    group_columns = ["operator", "bin_start_utc"]
    grouped = (
        work.groupby(group_columns, as_index=False)
        .agg(
            attempted_packets=("packet_received", "size"),
            observed_replies=("packet_received", "sum"),
            latency_mean_ms=("latency_ms", "mean"),
            latency_std_ms=("latency_ms", "std"),
            latency_max_ms=("latency_ms", "max"),
            source_timestamp_median=("timestamp", "median"),
        )
        .sort_values(group_columns)
    )
    grouped["packet_loss_rate"] = 1.0 - (
        grouped["observed_replies"] / grouped["attempted_packets"].clip(lower=1)
    )
    no_reply = grouped["observed_replies"].eq(0)
    grouped.loc[no_reply, "latency_mean_ms"] = float(timeout_ms)
    grouped.loc[no_reply, "latency_max_ms"] = float(timeout_ms)
    grouped["latency_std_ms"] = grouped["latency_std_ms"].fillna(0.0)
    grouped["path_state"] = np.where(no_reply, "unavailable", "active")

    for optional in ("scenario", "direction"):
        if optional not in work:
            continue
        values = work.groupby(group_columns, as_index=False)[optional].first()
        grouped = grouped.merge(values, on=group_columns, how="left")
    if coordinate_fields_present:
        coordinate_summary = work.groupby(group_columns, as_index=False).agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
        )
        grouped = grouped.merge(coordinate_summary, on=group_columns, how="left")
    else:
        for coordinate in ("latitude", "longitude"):
            if coordinate not in work:
                continue
            values = work.groupby(group_columns, as_index=False)[coordinate].median()
            grouped = grouped.merge(values, on=group_columns, how="left")
    if "controller_id" in work:
        controller_counts = work.groupby(group_columns)["controller_id"].nunique(dropna=True)
        if controller_counts.gt(1).any():
            raise ValueError(
                "controller_id changes within an operator decision bin; "
                "the shared-controller audit is ambiguous"
            )
        values = work.groupby(group_columns, as_index=False)["controller_id"].first()
        grouped = grouped.merge(values, on=group_columns, how="left")
    if "campaign_id" in work:
        campaign_counts = work.groupby(group_columns)["campaign_id"].nunique(dropna=True)
        if campaign_counts.gt(1).any():
            raise ValueError(
                "campaign_id changes within an operator decision bin; "
                "independence grouping is ambiguous"
            )
        values = work.groupby(group_columns, as_index=False)["campaign_id"].first()
        grouped = grouped.merge(values, on=group_columns, how="left")

    grouped["bin_epoch"] = (
        grouped["bin_start_utc"] - pd.Timestamp("1970-01-01", tz="UTC")
    ).dt.total_seconds().astype("int64")
    grouped["bin_start_utc"] = grouped["bin_start_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    grouped["measurement_family"] = "independent_commercial_multileo"
    grouped["location"] = grouped.get("scenario", pd.Series(dataset_name, index=grouped.index))
    grouped["location"] = grouped["location"].fillna(dataset_name).astype(str)
    grouped["session_date"] = grouped["bin_start_utc"].dt.normalize()
    grouped["target_hint"] = grouped["operator"]
    grouped["window_duration"] = "continuous_campaign"
    grouped["probe_interval"] = "source_packet_stream"
    grouped["bin_seconds"] = int(bin_seconds)
    grouped["access_technology"] = "commercial_leo"

    pivot = grouped.pivot(
        index="bin_start_utc",
        columns="operator",
        values="source_timestamp_median",
    )
    complete_pivot = pivot.dropna(subset=["starlink", "oneweb"])
    complete_epochs = set(complete_pivot.index)
    grouped["complete_counterfactual_epoch"] = grouped["bin_start_utc"].isin(
        complete_epochs
    )
    complete = grouped[grouped["complete_counterfactual_epoch"]].copy()
    if complete.empty:
        raise ValueError("no time bin contains both Starlink and OneWeb outcomes")

    skew_ms = (
        complete_pivot[["starlink", "oneweb"]].max(axis=1)
        - complete_pivot[["starlink", "oneweb"]].min(axis=1)
    ).dt.total_seconds() * 1000.0
    first_timestamp = work["timestamp"].min()
    last_timestamp = work["timestamp"].max()
    duration_days = float((last_timestamp - first_timestamp).total_seconds() / 86400.0)
    all_epoch_count = int(grouped["bin_epoch"].nunique())
    complete_epoch_count = int(len(complete_epochs))
    complete_epoch_fraction = complete_epoch_count / max(all_epoch_count, 1)

    effective_maximum_distance_meters = min(
        float(maximum_inter_operator_distance_meters),
        CLAIM_MAXIMUM_INTER_OPERATOR_DISTANCE_METERS,
    )
    coordinate_pair_count = 0
    coordinate_pair_fraction = 0.0
    complete_source_coordinate_row_count = 0
    complete_source_coordinate_row_fraction = 0.0
    median_inter_operator_distance_meters: float | None = None
    p95_inter_operator_distance_meters: float | None = None
    maximum_inter_operator_distance_meters_observed: float | None = None
    spatial_colocation_pass = False
    if coordinate_fields_present:
        work_bin_start_naive = (
            work["bin_start_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        )
        complete_source_rows = work[work_bin_start_naive.isin(complete_epochs)]
        complete_source_coordinate_mask = complete_source_rows[
            ["latitude", "longitude"]
        ].notna().all(axis=1)
        complete_source_coordinate_row_count = int(
            complete_source_coordinate_mask.sum()
        )
        complete_source_coordinate_row_fraction = (
            complete_source_coordinate_row_count / max(len(complete_source_rows), 1)
        )
        coordinate_pivot = grouped.pivot(
            index="bin_start_utc",
            columns="operator",
            values=["latitude", "longitude"],
        ).reindex(complete_pivot.index)
        required_coordinate_columns = [
            ("latitude", "starlink"),
            ("longitude", "starlink"),
            ("latitude", "oneweb"),
            ("longitude", "oneweb"),
        ]
        valid_coordinate_pairs = coordinate_pivot[required_coordinate_columns].notna().all(axis=1)
        coordinate_pair_count = int(valid_coordinate_pairs.sum())
        coordinate_pair_fraction = coordinate_pair_count / max(complete_epoch_count, 1)
        if coordinate_pair_count:
            valid_coordinates = coordinate_pivot.loc[valid_coordinate_pairs]
            distances_m = _haversine_meters(
                valid_coordinates[("latitude", "starlink")],
                valid_coordinates[("longitude", "starlink")],
                valid_coordinates[("latitude", "oneweb")],
                valid_coordinates[("longitude", "oneweb")],
            )
            median_inter_operator_distance_meters = float(distances_m.median())
            p95_inter_operator_distance_meters = float(distances_m.quantile(0.95))
            maximum_inter_operator_distance_meters_observed = float(distances_m.max())
            # Complete raw-coordinate coverage and a strict maximum over the
            # per-bin representative terminal positions prevent missing or
            # spatially separated epochs from being hidden by a campaign mean.
            spatial_colocation_pass = bool(
                coordinate_pair_count == complete_epoch_count
                and complete_source_coordinate_row_count == len(complete_source_rows)
                and maximum_inter_operator_distance_meters_observed
                <= effective_maximum_distance_meters
            )

    mapped_controller_ids_present = "controller_id" in grouped
    controller_pair_count = 0
    matching_controller_pair_count = 0
    conflicting_controller_pair_count = 0
    mapped_controller_id_pass = False
    if mapped_controller_ids_present:
        controller_pivot = grouped.pivot(
            index="bin_start_utc",
            columns="operator",
            values="controller_id",
        ).reindex(complete_pivot.index)
        valid_controller_pairs = controller_pivot[["starlink", "oneweb"]].notna().all(axis=1)
        matching_controller_pairs = valid_controller_pairs & controller_pivot["starlink"].eq(
            controller_pivot["oneweb"]
        )
        conflicting_controller_pairs = valid_controller_pairs & ~matching_controller_pairs
        controller_pair_count = int(valid_controller_pairs.sum())
        matching_controller_pair_count = int(matching_controller_pairs.sum())
        conflicting_controller_pair_count = int(conflicting_controller_pairs.sum())
        mapped_controller_id_pass = bool(
            controller_pair_count == complete_epoch_count
            and matching_controller_pair_count == complete_epoch_count
        )
    controller_conflict_free = conflicting_controller_pair_count == 0
    shared_controller_provenance_pass = bool(
        controller_conflict_free
        and (mapped_controller_id_pass or same_controller_provenance)
    )
    if mapped_controller_id_pass:
        controller_provenance_mode = "mapped_controller_id"
    elif same_controller_provenance and controller_conflict_free:
        controller_provenance_mode = "documented_external_attestation"
    elif conflicting_controller_pair_count:
        controller_provenance_mode = "mapped_controller_id_conflict"
    else:
        controller_provenance_mode = "none"

    mapped_campaign_ids_present = "campaign_id" in grouped
    campaign_pair_count = 0
    matching_campaign_pair_count = 0
    conflicting_campaign_pair_count = 0
    mapped_campaign_id_pass = False
    mapped_campaign_ids: list[str] = []
    if mapped_campaign_ids_present:
        campaign_pivot = grouped.pivot(
            index="bin_start_utc",
            columns="operator",
            values="campaign_id",
        ).reindex(complete_pivot.index)
        valid_campaign_pairs = campaign_pivot[["starlink", "oneweb"]].notna().all(axis=1)
        matching_campaign_pairs = valid_campaign_pairs & campaign_pivot["starlink"].eq(
            campaign_pivot["oneweb"]
        )
        conflicting_campaign_pairs = valid_campaign_pairs & ~matching_campaign_pairs
        campaign_pair_count = int(valid_campaign_pairs.sum())
        matching_campaign_pair_count = int(matching_campaign_pairs.sum())
        conflicting_campaign_pair_count = int(conflicting_campaign_pairs.sum())
        mapped_campaign_id_pass = bool(
            campaign_pair_count == complete_epoch_count
            and matching_campaign_pair_count == complete_epoch_count
        )
        if mapped_campaign_id_pass:
            mapped_campaign_ids = sorted(
                campaign_pivot.loc[matching_campaign_pairs, "starlink"]
                .astype(str)
                .unique()
                .tolist()
            )
    independent_campaign_grouping_pass = bool(
        independent_campaign_ids_audited
        and mapped_campaign_id_pass
        and conflicting_campaign_pair_count == 0
        and len(mapped_campaign_ids) >= 2
    )
    if independent_campaign_grouping_pass:
        campaign_independence_mode = "mapped_ids_with_documented_independence_audit"
    elif independent_campaign_ids_audited and not mapped_campaign_id_pass:
        campaign_independence_mode = "mapped_campaign_id_audit_failed"
    elif independent_campaign_ids_audited and len(mapped_campaign_ids) < 2:
        campaign_independence_mode = "insufficient_distinct_campaigns"
    elif mapped_campaign_ids_present:
        campaign_independence_mode = "mapped_ids_not_asserted_independent"
    else:
        campaign_independence_mode = "single_campaign_default"

    # A forecast target must be the immediately following decision bin, not
    # merely the next surviving row after an outage in data collection.
    complete_epoch_table = (
        complete[["bin_epoch"]].drop_duplicates().sort_values("bin_epoch").copy()
    )
    new_segment = complete_epoch_table["bin_epoch"].diff().fillna(
        bin_seconds
    ).ne(bin_seconds)
    if mapped_campaign_ids_present:
        epoch_campaign = (
            complete.groupby("bin_epoch", sort=True)["campaign_id"]
            .first()
            .reindex(complete_epoch_table["bin_epoch"])
            .astype("string")
            .fillna("__missing_campaign__")
            .reset_index(drop=True)
        )
        new_segment = new_segment.reset_index(drop=True) | epoch_campaign.ne(
            epoch_campaign.shift(1)
        )
        new_segment.iloc[0] = False
    complete_epoch_table["segment_id"] = new_segment.cumsum()
    complete = complete.merge(complete_epoch_table, on="bin_epoch", how="left")
    complete["relative_path"] = (
        "commercial_multileo/"
        + complete["operator"]
        + "/segment_"
        + complete["segment_id"].astype(int).astype(str)
    )
    segment_sizes = complete_epoch_table.groupby("segment_id").size()
    observed_concurrent_hours = complete_epoch_count * bin_seconds / 3600.0

    complete = complete.drop(columns=["source_timestamp_median"])
    complete = complete.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    validate_time_bin_table(complete)
    concurrency = audit_trace_concurrency(complete)
    effective_minimum_duration_days = max(
        minimum_duration_days,
        CLAIM_MINIMUM_DURATION_DAYS,
    )
    effective_minimum_concurrent_hours = max(
        minimum_concurrent_hours,
        CLAIM_MINIMUM_CONCURRENT_HOURS,
    )
    effective_maximum_p95_skew_ms = min(
        maximum_p95_skew_ms,
        CLAIM_MAXIMUM_P95_SKEW_MS,
    )
    duration_span_pass = duration_days >= effective_minimum_duration_days
    observation_volume_pass = (
        observed_concurrent_hours >= effective_minimum_concurrent_hours
    )
    synchronization_pass = (
        float(skew_ms.quantile(0.95)) <= effective_maximum_p95_skew_ms
    )
    long_duration_pass = duration_span_pass and observation_volume_pass
    paired_outcome_pass = bool(
        concurrency["has_temporally_concurrent_candidates"]
        and concurrency["median_concurrent_paths"] >= 2
        and complete_epoch_fraction >= 0.95
        and synchronization_pass
    )
    same_controller_selectable_path_evidence = bool(
        paired_outcome_pass
        and spatial_colocation_pass
        and shared_controller_provenance_pass
    )
    closes_limitation = bool(
        independent_provenance
        and long_duration_pass
        and same_controller_selectable_path_evidence
    )
    if same_controller_selectable_path_evidence:
        evidence_scope = "literal_same_controller_selectable_path_replay"
        valid_claim = (
            "long-duration independent measured policy-level validation on "
            "co-located, shared-controller commercial Starlink and OneWeb paths"
            if closes_limitation
            else "scoped same-controller commercial multi-LEO path replay"
        )
    elif not paired_outcome_pass:
        evidence_scope = "non_counterfactual_dual_operator_observations"
        valid_claim = (
            "dual-operator measurements without a complete synchronized "
            "counterfactual replay"
        )
    elif spatial_colocation_pass:
        evidence_scope = "scoped_near_concurrent_colocated_or_convoy_replay"
        valid_claim = (
            "scoped near-concurrent co-located or convoy replay; shared-controller "
            "selectability is unverified"
        )
    elif coordinate_pair_count and (
        coordinate_pair_fraction < 1.0
        or complete_source_coordinate_row_fraction < 1.0
    ):
        evidence_scope = "scoped_incomplete_location_audit_paired_comparison"
        valid_claim = (
            "scoped time-aligned dual-operator comparison; terminal-location "
            "coverage is incomplete"
        )
    elif coordinate_pair_count:
        evidence_scope = "scoped_spatially_separated_time_aligned_comparison"
        valid_claim = (
            "scoped time-aligned dual-operator comparison; the terminals exceed "
            "the co-location threshold"
        )
    else:
        evidence_scope = "scoped_location_unverified_time_aligned_comparison"
        valid_claim = (
            "scoped time-aligned dual-operator comparison; co-location and "
            "shared-controller selectability are unverified"
        )
    if closes_limitation:
        claim_restriction = "none for the audited dataset and topology criteria"
    elif not paired_outcome_pass:
        claim_restriction = (
            "do not claim counterfactual commercial multi-LEO replay or "
            "same-controller path-selection validation"
        )
    elif not spatial_colocation_pass:
        claim_restriction = (
            "report only a time-aligned dual-operator comparison; do not claim "
            "co-located or same-controller selectable paths"
        )
    elif not shared_controller_provenance_pass:
        claim_restriction = (
            "report only scoped near-concurrent co-located or convoy replay; "
            "do not claim literal same-controller path-selection validation"
        )
    else:
        claim_restriction = (
            "do not claim long-duration independent multi-LEO validation"
        )
    claim_safe_concurrency_audit = {
        **concurrency,
        "generic_timestamp_concurrency_detected": bool(
            concurrency["has_temporally_concurrent_candidates"]
        ),
        "supports_scoped_paired_replay": paired_outcome_pass,
        "supports_candidate_outcome_shadow_replay": paired_outcome_pass,
        "supports_literal_single_controller_steering": (
            same_controller_selectable_path_evidence
        ),
        "supports_single_controller_shadow_replay": (
            same_controller_selectable_path_evidence
        ),
        "supports_closed_loop_deployment_evidence": False,
        "topology_gate_applied": True,
    }
    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_url": dataset_url,
        "dataset_doi": dataset_doi,
        "license": license_name,
        "source_sha256": _sha256(source_path),
        "source_row_count": int(len(source)),
        "retained_commercial_leo_row_count": int(len(work)),
        "ignored_unknown_operator_row_count": unknown_operator_rows,
        "processed_row_count": int(len(complete)),
        "operators": ["starlink", "oneweb"],
        "is_measured_dataset": True,
        "is_independent_of_lens": bool(independent_provenance),
        "commercial_leo_paths": True,
        "time_aligned_dual_operator_measurements": True,
        "concurrent_interchangeable_paths": same_controller_selectable_path_evidence,
        "same_controller_selectable_path_evidence": same_controller_selectable_path_evidence,
        "supports_scoped_paired_replay": paired_outcome_pass,
        "all_candidate_outcomes_observed": paired_outcome_pass,
        "post_selection_outcome_interpretation": (
            "next-bin realized outcomes exist for both operators; they support "
            "literal selectable-path replay only when the separate spatial and "
            "shared-controller topology gates also pass"
        ),
        "bin_seconds": int(bin_seconds),
        "timeout_ms": float(timeout_ms),
        "duration_days": duration_days,
        "requested_minimum_duration_days": float(minimum_duration_days),
        "effective_minimum_duration_days": effective_minimum_duration_days,
        "duration_span_pass": duration_span_pass,
        "observed_concurrent_hours": observed_concurrent_hours,
        "requested_minimum_concurrent_hours": float(minimum_concurrent_hours),
        "effective_minimum_concurrent_hours": effective_minimum_concurrent_hours,
        "observation_volume_pass": observation_volume_pass,
        "long_duration_pass": long_duration_pass,
        "all_epoch_count": all_epoch_count,
        "complete_concurrent_epoch_count": complete_epoch_count,
        "complete_concurrent_epoch_fraction": complete_epoch_fraction,
        "median_inter_operator_timestamp_skew_ms": float(skew_ms.median()),
        "p95_inter_operator_timestamp_skew_ms": float(skew_ms.quantile(0.95)),
        "maximum_inter_operator_timestamp_skew_ms": float(skew_ms.max()),
        "requested_maximum_p95_skew_ms": float(maximum_p95_skew_ms),
        "effective_maximum_p95_skew_ms": effective_maximum_p95_skew_ms,
        "synchronization_pass": synchronization_pass,
        "coordinate_fields_present": coordinate_fields_present,
        "complete_coordinate_pair_count": coordinate_pair_count,
        "complete_coordinate_pair_fraction": coordinate_pair_fraction,
        "complete_source_coordinate_row_count": complete_source_coordinate_row_count,
        "complete_source_coordinate_row_fraction": (
            complete_source_coordinate_row_fraction
        ),
        "median_inter_operator_distance_meters": median_inter_operator_distance_meters,
        "p95_inter_operator_distance_meters": p95_inter_operator_distance_meters,
        "maximum_inter_operator_distance_meters": (
            maximum_inter_operator_distance_meters_observed
        ),
        "requested_maximum_inter_operator_distance_meters": float(
            maximum_inter_operator_distance_meters
        ),
        "effective_maximum_inter_operator_distance_meters": (
            effective_maximum_distance_meters
        ),
        "spatial_colocation_pass": spatial_colocation_pass,
        "mapped_controller_ids_present": mapped_controller_ids_present,
        "complete_controller_pair_count": controller_pair_count,
        "matching_controller_pair_count": matching_controller_pair_count,
        "conflicting_controller_pair_count": conflicting_controller_pair_count,
        "mapped_controller_id_pass": mapped_controller_id_pass,
        "same_controller_provenance_asserted": bool(same_controller_provenance),
        "controller_provenance_note": controller_provenance_note,
        "controller_provenance_mode": controller_provenance_mode,
        "shared_controller_provenance_pass": shared_controller_provenance_pass,
        "mapped_campaign_ids_present": mapped_campaign_ids_present,
        "complete_campaign_pair_count": campaign_pair_count,
        "matching_campaign_pair_count": matching_campaign_pair_count,
        "conflicting_campaign_pair_count": conflicting_campaign_pair_count,
        "mapped_campaign_id_pass": mapped_campaign_id_pass,
        "independent_campaign_ids_asserted": bool(
            independent_campaign_ids_audited
        ),
        "campaign_independence_note": campaign_independence_note,
        "mapped_campaign_ids": mapped_campaign_ids,
        "mapped_campaign_count": len(mapped_campaign_ids),
        "audited_campaign_ids": (
            mapped_campaign_ids if independent_campaign_grouping_pass else []
        ),
        "audited_campaign_count": (
            len(mapped_campaign_ids) if independent_campaign_grouping_pass else 0
        ),
        "campaign_independence_mode": campaign_independence_mode,
        "independent_campaign_grouping_pass": (
            independent_campaign_grouping_pass
        ),
        "gate_inference_default_without_audited_campaign_ids": (
            "one_complete_imported_campaign"
        ),
        "spatial_coordinate_semantics": (
            "latitude and longitude must identify the candidate access terminals, "
            "not remote servers or unrelated probes"
        ),
        "topology_claim_gate_version": 1,
        "continuous_segment_count": int(len(segment_sizes)),
        "median_continuous_segment_bins": float(segment_sizes.median()),
        "maximum_continuous_segment_bins": int(segment_sizes.max()),
        "temporal_concurrency_audit": concurrency,
        "concurrency_audit": claim_safe_concurrency_audit,
        "closes_independent_longitudinal_multileo_limitation": closes_limitation,
        "evidence_scope": evidence_scope,
        "valid_claim": valid_claim,
        "claim_restriction": claim_restriction,
    }
    return complete, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--column-map", required=True)
    parser.add_argument("--output", default="data/processed/commercial_multileo_10s.csv")
    parser.add_argument("--bin-seconds", type=int, default=10)
    parser.add_argument("--timeout-ms", type=float, required=True)
    parser.add_argument("--minimum-duration-days", type=float, default=30.0)
    parser.add_argument("--minimum-concurrent-hours", type=float, default=24.0)
    parser.add_argument("--maximum-p95-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--maximum-inter-operator-distance-meters",
        type=float,
        default=CLAIM_MAXIMUM_INTER_OPERATOR_DISTANCE_METERS,
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--dataset-doi", default="")
    parser.add_argument("--license", dest="license_name", required=True)
    parser.add_argument(
        "--independent-of-lens",
        action="store_true",
        help="Assert only when provenance is documented and independent of LENS.",
    )
    parser.add_argument(
        "--same-controller-provenance",
        action="store_true",
        help=(
            "Assert only with campaign documentation that both terminals were "
            "simultaneously selectable by one controller. Matching GPS alone is "
            "insufficient."
        ),
    )
    parser.add_argument(
        "--controller-provenance-note",
        default="",
        help="Citation or data-owner confirmation supporting the controller assertion.",
    )
    parser.add_argument(
        "--independent-campaign-ids-audited",
        action="store_true",
        help=(
            "Assert only after verifying that mapped campaign_id values denote "
            "genuinely independent collection campaigns, not sessions, days, "
            "files, vehicles within one campaign, or arbitrary time blocks."
        ),
    )
    parser.add_argument(
        "--campaign-independence-note",
        default="",
        help="Citation or data-owner confirmation supporting campaign independence.",
    )
    args = parser.parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    frame, metadata = build_commercial_multileo_trace(
        input_path,
        _read_column_map(_resolve(args.column_map)),
        bin_seconds=args.bin_seconds,
        timeout_ms=args.timeout_ms,
        minimum_duration_days=args.minimum_duration_days,
        minimum_concurrent_hours=args.minimum_concurrent_hours,
        maximum_p95_skew_ms=args.maximum_p95_skew_ms,
        dataset_name=args.dataset_name,
        dataset_url=args.dataset_url,
        dataset_doi=args.dataset_doi,
        license_name=args.license_name,
        independent_provenance=args.independent_of_lens,
        maximum_inter_operator_distance_meters=(
            args.maximum_inter_operator_distance_meters
        ),
        same_controller_provenance=args.same_controller_provenance,
        controller_provenance_note=args.controller_provenance_note,
        independent_campaign_ids_audited=(
            args.independent_campaign_ids_audited
        ),
        campaign_independence_note=args.campaign_independence_note,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"trace_written={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
