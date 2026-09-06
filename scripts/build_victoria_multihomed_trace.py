#!/usr/bin/env python3
"""Build simultaneous measured paths from two co-located LENS terminals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from open_leo_latency_routing.data.aggregations import aggregate_ping_file
from open_leo_latency_routing.data.loaders import (
    audit_trace_concurrency,
    validate_time_bin_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$"
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _session_key(path: Path) -> str:
    match = SESSION_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"cannot parse session time from {path}")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_sessions(directory: Path) -> dict[str, Path]:
    return {
        _session_key(path): path
        for path in directory.rglob("ping-*.txt")
        if path.is_file()
    }


def _pool_boundary_bins(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pool overlapping hourly-file boundary bins without dropping replies.

    Some LENS hourly files include one or two replies from the first bucket of
    the next hour, while the following file contains the remaining replies in
    that same absolute bucket.  Aggregate rows are therefore combined with
    sample-count weights.  The pooled population variance is recovered from
    each row's count, mean, and population standard deviation; extrema and TTL
    summaries are combined exactly.  Metadata comes from the later hourly
    window, whose nominal start contains the bucket.  This half-open-window
    convention affects metadata only: every observed reply remains in the
    outcome statistics.
    """

    keys = ["relative_path", "bin_epoch"]
    duplicate_mask = frame.duplicated(keys, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_pairs = int(frame.loc[duplicate_mask, keys].drop_duplicates().shape[0])
    input_reply_count = int(
        pd.to_numeric(frame["observed_replies"], errors="raise").sum()
    )
    overshoot_replies_pooled = 0
    if duplicate_rows:
        pooled_rows: list[dict[str, object]] = []
        for _, group in frame.groupby(keys, sort=False, dropna=False):
            if len(group) == 1:
                pooled_rows.append(group.iloc[0].to_dict())
                continue

            ordered = group.sort_values("window_start", kind="mergesort")
            pooled = ordered.iloc[-1].copy()
            counts = pd.to_numeric(
                ordered["observed_replies"], errors="raise"
            ).astype(float)
            total = float(counts.sum())
            if total <= 0.0:
                raise ValueError("cannot pool a boundary bin without replies")

            latency_mean = pd.to_numeric(
                ordered["latency_mean_ms"], errors="raise"
            ).astype(float)
            latency_std = pd.to_numeric(
                ordered["latency_std_ms"], errors="raise"
            ).astype(float)
            combined_mean = float((counts * latency_mean).sum() / total)
            combined_second_moment = float(
                (counts * (latency_std.pow(2) + latency_mean.pow(2))).sum()
                / total
            )
            pooled["observed_replies"] = int(total)
            pooled["latency_mean_ms"] = combined_mean
            pooled["latency_std_ms"] = max(
                0.0,
                combined_second_moment - combined_mean**2,
            ) ** 0.5
            pooled["latency_min_ms"] = float(
                pd.to_numeric(ordered["latency_min_ms"], errors="raise").min()
            )
            pooled["latency_max_ms"] = float(
                pd.to_numeric(ordered["latency_max_ms"], errors="raise").max()
            )
            ttl_mean = pd.to_numeric(
                ordered["ttl_mean"], errors="raise"
            ).astype(float)
            pooled["ttl_mean"] = float((counts * ttl_mean).sum() / total)
            pooled["ttl_min"] = int(
                pd.to_numeric(ordered["ttl_min"], errors="raise").min()
            )
            pooled["ttl_max"] = int(
                pd.to_numeric(ordered["ttl_max"], errors="raise").max()
            )
            pooled["icmp_seq_min"] = int(
                pd.to_numeric(ordered["icmp_seq_min"], errors="raise").min()
            )
            pooled["icmp_seq_max"] = int(
                pd.to_numeric(ordered["icmp_seq_max"], errors="raise").max()
            )
            overshoot_replies_pooled += int(counts.iloc[:-1].sum())
            pooled_rows.append(pooled.to_dict())

        frame = pd.DataFrame(pooled_rows, columns=frame.columns)
        frame = frame.sort_values(keys).reset_index(drop=True)
    if frame.duplicated(keys).any():
        raise AssertionError("Victoria trace contains duplicate path-epoch rows")
    output_reply_count = int(
        pd.to_numeric(frame["observed_replies"], errors="raise").sum()
    )
    if output_reply_count != input_reply_count:
        raise AssertionError("boundary pooling changed the observed reply count")
    return frame, {
        "overlapping_boundary_rows": duplicate_rows,
        "overlapping_boundary_pairs": duplicate_pairs,
        "aggregate_rows_combined": duplicate_rows - duplicate_pairs,
        "overshoot_replies_pooled": overshoot_replies_pooled,
        "input_reply_count": input_reply_count,
        "output_reply_count": output_reply_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/raw/lens_2025_03/LENS-2025-03",
    )
    parser.add_argument("--session-count", type=int, default=6)
    parser.add_argument(
        "--session-offset",
        type=int,
        default=0,
        help="Start index within the chronologically sorted common sessions.",
    )
    parser.add_argument("--bin-seconds", type=int, default=10)
    parser.add_argument(
        "--output",
        default="data/processed/lens_victoria_multihomed_10s.csv",
    )
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    active_root = data_root / "inside-out" / "active"
    terminal_dirs = {
        "victoria_standard": active_root / "victoria",
        "victoria_priority": active_root / "victoria_hp1_proto2",
    }
    indices = {
        terminal: _index_sessions(directory)
        for terminal, directory in terminal_dirs.items()
    }
    common_sessions = sorted(
        set.intersection(*(set(index) for index in indices.values()))
    )
    if args.session_offset < 0:
        raise ValueError("--session-offset must be non-negative")
    selection_end = args.session_offset + args.session_count
    if len(common_sessions) < selection_end:
        raise ValueError(
            f"requested sessions [{args.session_offset}:{selection_end}] but "
            f"only {len(common_sessions)} overlap"
        )
    # Prefer one continuous block to preserve time-series interpretation.
    selected = common_sessions[args.session_offset:selection_end]
    rows: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    for terminal_name, index in indices.items():
        for session_key in selected:
            source_path = index[session_key]
            source_files.append(
                {
                    "terminal": terminal_name,
                    "session": session_key,
                    "path": str(source_path.relative_to(data_root)),
                    "sha256": _sha256(source_path),
                    "bytes": source_path.stat().st_size,
                }
            )
            aggregate_rows = aggregate_ping_file(
                source_path,
                data_root,
                bin_seconds=args.bin_seconds,
            )
            for row in aggregate_rows:
                row["relative_path"] = (
                    f"lens_victoria_multihomed/{terminal_name}"
                )
                row["location"] = "victoria_multihomed_site"
                row["target_hint"] = terminal_name
                row["measurement_family"] = (
                    "lens_measured_concurrent_multihomed"
                )
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["bin_start_utc"] = pd.to_datetime(frame["bin_start_utc"])
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame, boundary_audit = _pool_boundary_bins(frame)
    validate_time_bin_table(frame)
    concurrency = audit_trace_concurrency(frame)
    if not concurrency["has_temporally_concurrent_candidates"]:
        raise ValueError("selected terminal sessions are not concurrent")
    concurrency.update(
        {
            "supports_candidate_outcome_shadow_replay": True,
            "supports_shadow_policy_replay": True,
            "supports_literal_single_controller_steering": False,
            "supports_closed_loop_deployment_evidence": False,
            "controller_topology_scope": (
                "co-located terminals; shared steering authority unverified"
            ),
        }
    )

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata = {
        "dataset_name": "lens_victoria_multihomed",
        "provenance": "LENS 2025-03 raw measured ping logs",
        "license": "CC BY-SA 4.0",
        "is_measured_dataset": True,
        "is_independent_of_lens": False,
        "concurrent_alternative_paths": True,
        "selection_interpretation": (
            "candidate-outcome counterfactual between two co-located Victoria "
            "Starlink terminals measured at the same timestamps; public metadata "
            "do not verify one steering controller"
        ),
        "terminal_paths": list(terminal_dirs),
        "shared_target": "Seattle LENS endpoint",
        "available_overlapping_sessions": len(common_sessions),
        "selected_sessions": selected,
        "source_files": source_files,
        "session_offset": args.session_offset,
        "bin_seconds": args.bin_seconds,
        "boundary_overlap_resolution": {
            **boundary_audit,
            "rule": (
                "pool all aggregate statistics by observed-reply count; use "
                "the later nominal hourly window only for metadata"
            ),
        },
        "concurrency_audit": concurrency,
        "valid_claim": (
            "co-located two-terminal candidate-outcome counterfactual diagnostic"
        ),
        "invalid_claim": (
            "literal single-controller steering, closed-loop deployment, or "
            "generalization to a measurement dataset independent of LENS"
        ),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"trace_written={output}")
    print(
        f"rows={len(frame)} paths={frame['relative_path'].nunique()} "
        f"decision_bins={frame['bin_epoch'].nunique()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
