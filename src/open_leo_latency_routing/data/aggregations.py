"""Time-binned aggregation helpers for parsed LENS ping sessions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from open_leo_latency_routing.data.ping_logs import (
    iter_ping_observations,
    parse_ping_session_metadata,
)


def aggregate_ping_file(path: Path, data_root: Path, bin_seconds: int = 60) -> list[dict[str, Any]]:
    """Aggregate one ping log into fixed-width time bins."""
    metadata = parse_ping_session_metadata(path, data_root)
    buckets: dict[int, dict[str, list[float] | list[int]]] = defaultdict(
        lambda: {
            "latency": [],
            "ttl": [],
            "icmp_seq": [],
        }
    )

    for obs in iter_ping_observations(path):
        # Align every reply to the start of its fixed-width UTC bucket. The
        # downstream forecasting target is one bucket ahead of this value.
        bucket_epoch = int(obs.epoch // bin_seconds) * bin_seconds
        bucket = buckets[bucket_epoch]
        bucket["latency"].append(obs.latency_ms)
        bucket["ttl"].append(obs.ttl)
        bucket["icmp_seq"].append(obs.icmp_seq)

    rows: list[dict[str, Any]] = []
    for bucket_epoch in sorted(buckets):
        bucket = buckets[bucket_epoch]
        latency_values = [float(value) for value in bucket["latency"]]
        ttl_values = [int(value) for value in bucket["ttl"]]
        icmp_values = [int(value) for value in bucket["icmp_seq"]]
        rows.append(
            {
                **metadata.__dict__,
                "bin_seconds": bin_seconds,
                "bin_epoch": bucket_epoch,
                "bin_start_utc": datetime.utcfromtimestamp(bucket_epoch).isoformat(),
                "observed_replies": len(latency_values),
                "latency_mean_ms": mean(latency_values),
                "latency_std_ms": pstdev(latency_values) if len(latency_values) > 1 else 0.0,
                "latency_min_ms": min(latency_values),
                "latency_max_ms": max(latency_values),
                "ttl_mean": mean(ttl_values),
                "ttl_min": min(ttl_values),
                "ttl_max": max(ttl_values),
                "icmp_seq_min": min(icmp_values),
                "icmp_seq_max": max(icmp_values),
            }
        )
    return rows
