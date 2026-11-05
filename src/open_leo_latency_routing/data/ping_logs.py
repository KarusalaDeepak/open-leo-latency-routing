"""Parsing helpers for LENS ping-log files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from statistics import mean, pstdev
from typing import Iterator

from open_leo_latency_routing.data.inventory import PING_HEADER_PATTERN, PING_SAMPLE_PATTERN


PING_FILENAME_PATTERN = re.compile(
    r"^ping-(?P<target_hint>.+)-(?P<probe_interval>\d+ms)-(?P<window_duration>\d+h)-"
    r"(?P<window_start>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$"
)


@dataclass
class PingSessionMetadata:
    """Metadata extracted from the ping-log path, filename, and header."""

    relative_path: str
    measurement_family: str
    path_state: str
    location: str
    session_date: str
    target_hint: str
    probe_interval: str
    window_duration: str
    window_start: str
    header_target: str
    header_target_ip: str
    header_source_ip: str
    interface: str


@dataclass
class PingObservation:
    """One parsed ICMP reply line."""

    epoch: float
    icmp_seq: int
    ttl: int
    latency_ms: float
    reply_ip: str


def _parse_filename(name: str) -> dict[str, str]:
    """Parse the LENS ping filename into structured components."""
    match = PING_FILENAME_PATTERN.match(name)
    if not match:
        return {
            "target_hint": "",
            "probe_interval": "",
            "window_duration": "",
            "window_start": "",
        }
    return match.groupdict()


def parse_ping_session_metadata(path: Path, data_root: Path) -> PingSessionMetadata:
    """Extract session metadata from the file path and header."""
    relative_path = str(path.relative_to(data_root))
    parts = path.relative_to(data_root).parts
    measurement_family = parts[0] if len(parts) > 0 else ""
    path_state = parts[1] if len(parts) > 1 else ""
    location = parts[2] if len(parts) > 2 else ""
    session_date = parts[3] if len(parts) > 3 else ""
    filename_info = _parse_filename(path.name)

    header_target = ""
    header_target_ip = ""
    header_source_ip = ""
    interface = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PING_HEADER_PATTERN.match(line.strip())
            if match:
                header_target = match.group("target")
                header_target_ip = match.group("target_ip")
                header_source_ip = match.group("source_ip")
                interface = match.group("interface")
                break

    return PingSessionMetadata(
        relative_path=relative_path,
        measurement_family=measurement_family,
        path_state=path_state,
        location=location,
        session_date=session_date,
        target_hint=filename_info["target_hint"],
        probe_interval=filename_info["probe_interval"],
        window_duration=filename_info["window_duration"],
        window_start=filename_info["window_start"],
        header_target=header_target,
        header_target_ip=header_target_ip,
        header_source_ip=header_source_ip,
        interface=interface,
    )


def iter_ping_observations(path: Path) -> Iterator[PingObservation]:
    """Yield parsed ICMP reply rows from a ping-log file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PING_SAMPLE_PATTERN.match(line.strip())
            if not match:
                continue
            yield PingObservation(
                epoch=float(match.group("epoch")),
                icmp_seq=int(match.group("icmp_seq")),
                ttl=int(match.group("ttl")),
                latency_ms=float(match.group("latency_ms")),
                reply_ip=match.group("reply_ip"),
            )


def summarize_ping_file(path: Path, data_root: Path, sample_rows_per_file: int = 1000) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build one session summary row and sampled observation rows for a ping log."""
    metadata = parse_ping_session_metadata(path, data_root)

    latency_values: list[float] = []
    ttl_values: list[int] = []
    sampled_rows: list[dict[str, object]] = []
    sample_stride = 1
    observations = 0
    epoch_start: float | None = None
    epoch_end: float | None = None

    for obs in iter_ping_observations(path):
        observations += 1
        if epoch_start is None:
            epoch_start = obs.epoch
        epoch_end = obs.epoch
        latency_values.append(obs.latency_ms)
        ttl_values.append(obs.ttl)

    if observations > sample_rows_per_file and sample_rows_per_file > 0:
        sample_stride = max(1, observations // sample_rows_per_file)

    for index, obs in enumerate(iter_ping_observations(path)):
        if sample_rows_per_file > 0 and index % sample_stride == 0 and len(sampled_rows) < sample_rows_per_file:
            sampled_rows.append(
                {
                    **metadata.__dict__,
                    "epoch": obs.epoch,
                    "timestamp_utc": datetime.utcfromtimestamp(obs.epoch).isoformat(),
                    "icmp_seq": obs.icmp_seq,
                    "ttl": obs.ttl,
                    "latency_ms": obs.latency_ms,
                    "reply_ip": obs.reply_ip,
                }
            )

    summary = {
        **metadata.__dict__,
        "file_size_bytes": path.stat().st_size,
        "observation_count": observations,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "timestamp_start_utc": datetime.utcfromtimestamp(epoch_start).isoformat() if epoch_start is not None else "",
        "timestamp_end_utc": datetime.utcfromtimestamp(epoch_end).isoformat() if epoch_end is not None else "",
        "latency_mean_ms": mean(latency_values) if latency_values else None,
        "latency_std_ms": pstdev(latency_values) if len(latency_values) > 1 else 0.0,
        "latency_min_ms": min(latency_values) if latency_values else None,
        "latency_max_ms": max(latency_values) if latency_values else None,
        "ttl_mean": mean(ttl_values) if ttl_values else None,
        "ttl_min": min(ttl_values) if ttl_values else None,
        "ttl_max": max(ttl_values) if ttl_values else None,
        "sample_stride": sample_stride,
        "sampled_observation_count": len(sampled_rows),
    }
    return summary, sampled_rows
