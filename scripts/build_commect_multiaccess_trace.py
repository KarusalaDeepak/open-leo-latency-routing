#!/usr/bin/env python3
"""Build a concurrent 5G/Starlink path table from the COMMECT measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_leo_latency_routing.data.loaders import audit_trace_concurrency


DATASET_DOI = "10.5281/zenodo.14620779"
EXPECTED_ARCHIVE_MD5 = "fa4a17432128c5ee23662e0f98dbc0c1"
SOURCE_FILES = {
    "operator_a_5g": "Operator_A_RTT.csv",
    "operator_b_5g": "Operator_B_RTT.csv",
    "starlink": "Satellite_RTT.csv",
}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_basename(member_name: str) -> str:
    """Return a ZIP member basename without assuming one archive root."""

    # ZIP member names normally use POSIX separators, but archives written by
    # some Windows tools contain backslashes.  Normalizing both forms avoids
    # coupling the provenance check to the archive's top-level directory.
    return PurePosixPath(member_name.replace("\\", "/")).name


def _verify_extracted_sources_against_archive(
    archive_path: Path,
    source_dir: Path,
    source_files: dict[str, str] = SOURCE_FILES,
) -> dict[str, dict[str, object]]:
    """Prove that each extracted CSV is byte-identical to a ZIP member.

    Matching is by exact basename so a harmless archive-root rename does not
    invalidate the build.  If an archive contains duplicate basenames, the
    extracted file must still match at least one member exactly; every matching
    member is recorded so the provenance remains unambiguous.
    """

    verification: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members_by_basename: dict[str, list[zipfile.ZipInfo]] = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            members_by_basename.setdefault(
                _zip_member_basename(member.filename), []
            ).append(member)

        for path_name, filename in source_files.items():
            source_path = source_dir / filename
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"missing COMMECT source file: {source_path}"
                )
            source_sha256 = _sha256(source_path)
            source_bytes = source_path.stat().st_size
            candidates = members_by_basename.get(filename, [])
            if not candidates:
                raise ValueError(
                    f"verified COMMECT archive has no member named {filename!r}"
                )

            matching_members: list[str] = []
            candidate_records: list[dict[str, object]] = []
            for member in candidates:
                digest = hashlib.sha256()
                with archive.open(member, "r") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                member_sha256 = digest.hexdigest()
                candidate_records.append(
                    {
                        "member": member.filename,
                        "sha256": member_sha256,
                        "bytes": member.file_size,
                    }
                )
                if (
                    member.file_size == source_bytes
                    and member_sha256 == source_sha256
                ):
                    matching_members.append(member.filename)

            if not matching_members:
                raise ValueError(
                    "extracted COMMECT source does not match the verified ZIP "
                    f"member bytes for {filename}: extracted sha256="
                    f"{source_sha256}, bytes={source_bytes}, archive_candidates="
                    f"{candidate_records}"
                )
            verification[path_name] = {
                "filename": filename,
                "extracted_sha256": source_sha256,
                "extracted_bytes": source_bytes,
                "matching_archive_members": sorted(matching_members),
                "archive_member_match_count": len(matching_members),
                "byte_identity_verified": True,
            }
    return verification


def _load_and_bin(path: Path, path_name: str, bin_seconds: int) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="utf-8-sig")
    source["timestamp"] = pd.to_datetime(
        source["Time"].astype(str).str.strip("'"),
        format="%d-%b-%Y %H:%M:%S.%f",
        errors="raise",
    )
    source["latency_ms"] = pd.to_numeric(
        source["Latency (ms)"],
        errors="coerce",
    )
    source = source.dropna(subset=["timestamp", "latency_ms"]).copy()
    source["bin_start_utc"] = source["timestamp"].dt.floor(f"{bin_seconds}s")

    grouped = (
        source.groupby("bin_start_utc", as_index=False)
        .agg(
            observed_replies=("latency_ms", "size"),
            latency_mean_ms=("latency_ms", "mean"),
            latency_std_ms=("latency_ms", "std"),
            latency_min_ms=("latency_ms", "min"),
            latency_max_ms=("latency_ms", "max"),
            source_timestamp_first=("timestamp", "min"),
            source_timestamp_median=("timestamp", "median"),
            source_timestamp_last=("timestamp", "max"),
        )
        .sort_values("bin_start_utc")
    )
    grouped["latency_std_ms"] = grouped["latency_std_ms"].fillna(0.0)

    for source_column, output_column in (
        ("RSRP (dBm)", "rsrp_mean_dbm"),
        ("RSRQ (dB)", "rsrq_mean_db"),
        ("SNR (dB)", "snr_mean_db"),
        ("Lon", "longitude"),
        ("Lat", "latitude"),
    ):
        if source_column in source:
            values = (
                source.assign(
                    **{
                        output_column: pd.to_numeric(
                            source[source_column],
                            errors="coerce",
                        )
                    }
                )
                .groupby("bin_start_utc", as_index=False)[output_column]
                .mean()
            )
            grouped = grouped.merge(values, on="bin_start_utc", how="left")

    # Source timestamps have no timezone. Integer conversion is used only as a
    # stable, shared decision key; no geographic UTC interpretation is claimed.
    grouped["bin_epoch"] = (
        grouped["bin_start_utc"]
        .astype("datetime64[ns]")
        .astype("int64")
        // 1_000_000_000
    )
    if grouped["bin_epoch"].nunique() != grouped["bin_start_utc"].nunique():
        raise ValueError("decision-key conversion collapsed distinct time bins")
    grouped["relative_path"] = f"commect_denmark/{path_name}"
    grouped["measurement_family"] = "commect_measured_multiaccess"
    grouped["path_state"] = "available"
    grouped["location"] = "northern_denmark_rural_route"
    grouped["session_date"] = grouped["bin_start_utc"].dt.normalize()
    grouped["target_hint"] = path_name
    grouped["probe_interval"] = "approximately_100ms"
    grouped["window_duration"] = "continuous_drive"
    grouped["bin_seconds"] = int(bin_seconds)
    bin_end = grouped["bin_start_utc"] + pd.to_timedelta(bin_seconds, unit="s")
    grouped["observation_age_ms"] = (
        bin_end - grouped["source_timestamp_median"]
    ).dt.total_seconds() * 1000.0
    grouped["observation_span_ms"] = (
        grouped["source_timestamp_last"] - grouped["source_timestamp_first"]
    ).dt.total_seconds() * 1000.0
    grouped["access_technology"] = (
        "leo_satellite" if path_name == "starlink" else "5g_nsa"
    )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        default=(
            "data/external/commect_latency/"
            "Mutli-connectivity_KPIs_Latency"
        ),
    )
    parser.add_argument(
        "--archive",
        default=(
            "data/external/commect_latency/"
            "Mutli-connectivity_KPIs_Latency.zip"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/processed/commect_multiaccess_10s.csv",
    )
    parser.add_argument("--bin-seconds", type=int, default=10)
    args = parser.parse_args()

    source_dir = _resolve(args.source_dir)
    archive_path = _resolve(args.archive)
    output_path = _resolve(args.output)
    if args.bin_seconds <= 0:
        raise ValueError("--bin-seconds must be positive")
    if not archive_path.exists():
        raise FileNotFoundError(
            f"missing source archive: {archive_path}; see README for download"
        )
    archive_md5 = _md5(archive_path)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"archive checksum mismatch: {archive_md5} != "
            f"{EXPECTED_ARCHIVE_MD5}"
        )

    archive_source_verification = _verify_extracted_sources_against_archive(
        archive_path,
        source_dir,
    )

    frames = []
    source_rows: dict[str, int] = {}
    for path_name, filename in SOURCE_FILES.items():
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"missing COMMECT source file: {source_path}")
        source_rows[path_name] = int(
            sum(1 for _ in source_path.open(encoding="utf-8-sig")) - 1
        )
        frames.append(_load_and_bin(source_path, path_name, args.bin_seconds))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["relative_path", "bin_epoch"]
    ).reset_index(drop=True)
    concurrency = audit_trace_concurrency(combined)
    concurrency.update(
        {
            "supports_candidate_outcome_shadow_replay": True,
            "supports_shadow_policy_replay": True,
            "supports_literal_single_controller_steering": True,
            "supports_closed_loop_deployment_evidence": False,
            "controller_topology_scope": (
                "one acquisition controller with heterogeneous access interfaces; "
                "offline shadow replay only"
            ),
        }
    )
    timestamp_pivot = combined.pivot(
        index="bin_start_utc",
        columns="relative_path",
        values="source_timestamp_median",
    ).dropna()
    inter_path_skew_ms = (
        timestamp_pivot.max(axis=1) - timestamp_pivot.min(axis=1)
    ).dt.total_seconds() * 1000.0
    skew_by_bin = inter_path_skew_ms.rename("inter_path_skew_ms")
    combined = combined.merge(
        skew_by_bin,
        left_on="bin_start_utc",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    within_path_span_ms = (
        combined["source_timestamp_last"] - combined["source_timestamp_first"]
    ).dt.total_seconds() * 1000.0
    alignment_audit = {
        "complete_concurrent_bins": int(len(timestamp_pivot)),
        "median_inter_path_median_skew_ms": float(inter_path_skew_ms.median()),
        "p95_inter_path_median_skew_ms": float(inter_path_skew_ms.quantile(0.95)),
        "maximum_inter_path_median_skew_ms": float(inter_path_skew_ms.max()),
        "median_within_path_observation_span_ms": float(within_path_span_ms.median()),
        "p95_within_path_observation_span_ms": float(within_path_span_ms.quantile(0.95)),
        "maximum_within_path_observation_span_ms": float(within_path_span_ms.max()),
    }
    expected_minimum_epochs = 60 * 60 // args.bin_seconds
    if concurrency["epoch_count"] < expected_minimum_epochs:
        raise ValueError(
            "too few decision epochs for the measured campaign: "
            f"{concurrency['epoch_count']} < {expected_minimum_epochs}"
        )
    if concurrency["median_concurrent_paths"] < 3:
        raise ValueError(
            "COMMECT trace did not preserve all three concurrent interfaces "
            "in the median decision epoch"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Numeric age and skew fields remain in the canonical trace so every
    # sensitivity result can be replayed without reopening the raw archive.
    combined.drop(
        columns=["source_timestamp_first", "source_timestamp_last"]
    ).to_csv(output_path, index=False)
    metadata = {
        "dataset_name": "commect_denmark_multiaccess_latency",
        "dataset_doi": DATASET_DOI,
        "dataset_url": f"https://doi.org/{DATASET_DOI}",
        "archive_md5": archive_md5,
        "archive_sha256": _sha256(archive_path),
        "archive_source_verification": archive_source_verification,
        "license": "CC BY-SA 4.0",
        "provenance": (
            "Aalborg University COMMECT measured rural-drive latency dataset"
        ),
        "is_measured_dataset": True,
        "is_independent_of_lens": True,
        "concurrent_alternative_paths": True,
        "path_interpretation": (
            "one controller may select Operator A 5G, Operator B 5G, or "
            "Starlink at each aligned decision epoch"
        ),
        "paths": list(SOURCE_FILES),
        "source_rows": source_rows,
        "source_files": [
            {
                "path": str((source_dir / filename).relative_to(REPO_ROOT)),
                "sha256": _sha256(source_dir / filename),
                "bytes": (source_dir / filename).stat().st_size,
            }
            for filename in SOURCE_FILES.values()
        ],
        "bin_seconds": int(args.bin_seconds),
        "alignment": (
            "source-local timestamps floored to equal-width bins; no "
            "interpolation, backfill, or future observations"
        ),
        "source_timezone_note": (
            "the public CSV timestamps are timezone-naive; they are used only "
            "for within-campaign ordering and concurrent alignment"
        ),
        "concurrency_audit": concurrency,
        "timestamp_alignment_audit": alignment_audit,
        "valid_claim": (
            "separate-source measured shadow replay on bin-aligned "
            "concurrent terrestrial and LEO access alternatives"
        ),
        "invalid_claim": (
            "an independent statistical replication, commercial multi-LEO "
            "service-replica routing, or "
            "satellite-to-satellite path selection"
        ),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"trace_written={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
