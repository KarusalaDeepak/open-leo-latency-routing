#!/usr/bin/env python3
"""Generate concurrent service-gateway paths from official Hypatia state.

Hypatia supplies the TLE-derived orbital positions, plus-grid ISLs and dynamic
shortest-path forwarding state. This adapter converts three destination
gateways reachable from one source terminal into the repository's canonical
time-bin schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPATIA_ROOT = REPO_ROOT / "external" / "hypatia"
SATGENPY_ROOT = HYPATIA_ROOT / "satgenpy"
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
if str(SATGENPY_ROOT) not in sys.path:
    sys.path.insert(0, str(SATGENPY_ROOT))

import satgen

from open_leo_latency_routing.data.loaders import validate_time_bin_table

EARTH_RADIUS_M = 6_378_135.0
ALTITUDE_M = 550_000.0
MAX_GSL_LENGTH_M = math.sqrt(940_700.0**2 + ALTITUDE_M**2)
MAX_ISL_LENGTH_M = 2.0 * math.sqrt(
    (EARTH_RADIUS_M + ALTITUDE_M) ** 2
    - (EARTH_RADIUS_M + 80_000.0) ** 2
)
SPEED_OF_LIGHT_M_S = 299_792_458.0


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_ground_stations(path: Path) -> None:
    stations = [
        (0, "Source-London", 51.5074, -0.1278, 0.0),
        (1, "Replica-New-York", 40.7128, -74.0060, 0.0),
        (2, "Replica-Tokyo", 35.6762, 139.6503, 0.0),
        (3, "Replica-Singapore", 1.3521, 103.8198, 0.0),
    ]
    basic = path.with_name("ground_stations.basic.txt")
    basic.write_text(
        "\n".join(
            f"{identifier},{name},{latitude},{longitude},{elevation}"
            for identifier, name, latitude, longitude, elevation in stations
        )
        + "\n",
        encoding="utf-8",
    )
    satgen.extend_ground_stations(str(basic), str(path))


def _generate_hypatia_state(
    state_root: Path,
    duration_seconds: int,
    step_seconds: int,
    orbits: int,
    satellites_per_orbit: int,
) -> Path:
    scenario_name = "oats_hypatia_starlink"
    scenario_dir = state_root / scenario_name
    if state_root.exists():
        shutil.rmtree(state_root)
    scenario_dir.mkdir(parents=True)

    _write_ground_stations(scenario_dir / "ground_stations.txt")
    satgen.generate_tles_from_scratch_manual(
        str(scenario_dir / "tles.txt"),
        "OATS-Hypatia-Starlink",
        orbits,
        satellites_per_orbit,
        True,
        53.0,
        0.0000001,
        0.0,
        15.19,
    )
    satgen.generate_plus_grid_isls(
        str(scenario_dir / "isls.txt"),
        orbits,
        satellites_per_orbit,
        isl_shift=0,
        idx_offset=0,
    )
    satgen.generate_description(
        str(scenario_dir / "description.txt"),
        MAX_GSL_LENGTH_M,
        MAX_ISL_LENGTH_M,
    )
    satgen.generate_simple_gsl_interfaces_info(
        str(scenario_dir / "gsl_interfaces_info.txt"),
        orbits * satellites_per_orbit,
        4,
        1,
        1,
        1,
        1,
    )
    satgen.help_dynamic_state(
        str(state_root),
        1,
        scenario_name,
        step_seconds * 1000,
        duration_seconds,
        MAX_GSL_LENGTH_M,
        MAX_ISL_LENGTH_M,
        "algorithm_free_one_only_over_isls",
        True,
    )
    return scenario_dir


def _read_fstate_updates(path: Path, state: dict[tuple[int, int], int]) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        current, destination, next_hop, _, _ = line.split(",")
        state[(int(current), int(destination))] = int(next_hop)


def _route_rtt_ms(
    source: int,
    destination: int,
    fstate: dict[tuple[int, int], int],
    epoch,
    time_ns: int,
    satellites,
    ground_stations,
    isls,
) -> tuple[float, list[int] | None]:
    forward = satgen.get_path(source, destination, fstate)
    reverse = satgen.get_path(destination, source, fstate)
    if forward is None or reverse is None:
        return float("nan"), None
    forward_length = satgen.compute_path_length_without_graph(
        forward,
        epoch,
        time_ns,
        satellites,
        ground_stations,
        isls,
        MAX_GSL_LENGTH_M,
        MAX_ISL_LENGTH_M,
    )
    reverse_length = satgen.compute_path_length_without_graph(
        reverse,
        epoch,
        time_ns,
        satellites,
        ground_stations,
        isls,
        MAX_GSL_LENGTH_M,
        MAX_ISL_LENGTH_M,
    )
    return (
        1000.0 * (forward_length + reverse_length) / SPEED_OF_LIGHT_M_S,
        forward,
    )


def build_trace(
    state_root: Path,
    duration_seconds: int,
    step_seconds: int,
    orbits: int,
    satellites_per_orbit: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    scenario_dir = _generate_hypatia_state(
        state_root,
        duration_seconds,
        step_seconds,
        orbits,
        satellites_per_orbit,
    )
    dynamic_dir = scenario_dir / (
        f"dynamic_state_{step_seconds * 1000}ms_for_{duration_seconds}s"
    )
    tles = satgen.read_tles(str(scenario_dir / "tles.txt"))
    satellites = tles["satellites"]
    epoch = tles["epoch"]
    ground_stations = satgen.read_ground_stations_extended(
        str(scenario_dir / "ground_stations.txt")
    )
    isls = satgen.read_isls(
        str(scenario_dir / "isls.txt"),
        len(satellites),
    )
    satellite_count = len(satellites)
    source = satellite_count
    destinations = {
        "new_york": satellite_count + 1,
        "tokyo": satellite_count + 2,
        "singapore": satellite_count + 3,
    }
    rng = np.random.default_rng(seed)
    start_epoch = int(pd.Timestamp("2025-05-01T00:00:00Z").timestamp())
    fstate: dict[tuple[int, int], int] = {}
    previous_route: dict[str, tuple[int, ...] | None] = {
        name: None for name in destinations
    }
    rows: list[dict[str, object]] = []
    for elapsed_seconds in range(0, duration_seconds, step_seconds):
        time_ns = elapsed_seconds * 1_000_000_000
        _read_fstate_updates(dynamic_dir / f"fstate_{time_ns}.txt", fstate)
        for replica_name, destination in destinations.items():
            propagation_ms, route = _route_rtt_ms(
                source,
                destination,
                fstate,
                epoch,
                time_ns,
                satellites,
                ground_stations,
                isls,
            )
            route_tuple = tuple(route) if route is not None else None
            handover = int(
                previous_route[replica_name] is not None
                and route_tuple != previous_route[replica_name]
            )
            previous_route[replica_name] = route_tuple
            active = route is not None and np.isfinite(propagation_ms)
            # Hypatia determines the route and propagation component. A small
            # seeded processing term avoids unrealistically identical replicas.
            processing_ms = (
                2.0
                + 0.7 * list(destinations).index(replica_name)
                + rng.normal(0.0, 0.15)
            )
            latency_ms = (
                propagation_ms + processing_ms if active else 500.0
            )
            epoch_seconds = start_epoch + elapsed_seconds
            rows.append(
                {
                    "relative_path": f"hypatia/london/{replica_name}",
                    "bin_epoch": epoch_seconds,
                    "bin_start_utc": pd.to_datetime(epoch_seconds, unit="s"),
                    "session_date": pd.to_datetime(
                        epoch_seconds, unit="s"
                    ).normalize(),
                    "latency_mean_ms": float(latency_ms),
                    "latency_std_ms": 0.5 + 0.5 * handover,
                    "latency_max_ms": float(latency_ms + 2.5 + 2.0 * handover),
                    "observed_replies": 100 if active else 0,
                    "path_state": "active" if active else "unavailable",
                    "location": "london",
                    "target_hint": replica_name,
                    "window_duration": f"{duration_seconds}s",
                    "probe_interval": f"{step_seconds * 1000}ms",
                    "bin_seconds": step_seconds,
                    "measurement_family": "hypatia_tle_dynamic_state",
                    "hypatia_route": (
                        "-".join(str(node) for node in route)
                        if route is not None
                        else ""
                    ),
                    "route_hop_count": len(route) - 1 if route is not None else 0,
                    "handover_event": handover,
                    "propagation_lower_bound_ms": (
                        float(propagation_ms) if active else np.nan
                    ),
                }
            )

    frame = pd.DataFrame(rows).sort_values(
        ["relative_path", "bin_epoch"]
    ).reset_index(drop=True)
    validate_time_bin_table(frame)
    commit = subprocess.run(
        ["git", "-C", str(HYPATIA_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "dataset_name": "hypatia_service_replica_paths",
        "provenance": "official Hypatia satgenpy dynamic state",
        "hypatia_commit": commit,
        "is_measured_dataset": False,
        "is_hypatia_output": True,
        "uses_tle_orbital_propagation": True,
        "uses_dynamic_shortest_path_state": True,
        "uses_ns3_packet_simulation": False,
        "concurrent_alternative_paths": True,
        "source_gateway": "London",
        "service_replicas": list(destinations),
        "candidate_paths_per_decision": len(destinations),
        "duration_seconds": duration_seconds,
        "step_seconds": step_seconds,
        "orbits": orbits,
        "satellites_per_orbit": satellites_per_orbit,
        "satellite_count": satellite_count,
        "random_seed": seed,
        "valid_claim": (
            "policy replication on official Hypatia TLE-derived dynamic "
            "routing state with concurrent service replicas"
        ),
        "invalid_claim": "packet-level ns-3 or independent measured validation",
    }
    return frame, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/hypatia_service_paths_10s.csv",
    )
    parser.add_argument(
        "--state-root",
        default="results/hypatia_dynamic_state",
    )
    parser.add_argument("--duration-seconds", type=int, default=1800)
    parser.add_argument("--step-seconds", type=int, default=10)
    parser.add_argument("--orbits", type=int, default=12)
    parser.add_argument("--satellites-per-orbit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, metadata = build_trace(
        _resolve(args.state_root),
        args.duration_seconds,
        args.step_seconds,
        args.orbits,
        args.satellites_per_orbit,
        args.seed,
    )
    frame.to_csv(output, index=False)
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
