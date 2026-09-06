#!/usr/bin/env python3
"""Generate an independent physics-informed LEO multi-path latency trace.

This dataset is a reproducible simulator trace, not a measurement release and
not a Hypatia output. It provides concurrent satellite/gateway alternatives so
the complete path-selection policy can be evaluated outside the LENS data
generation process.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.loaders import validate_time_bin_table
from open_leo_latency_routing.models.orbital_physics import (
    propagation_rtt_lower_bound_ms,
)


# The canonical four-way protocol reserves the last 15% for testing. Injected
# service events are confined to that untouched interval so they cannot affect
# model fitting, residual calibration, or policy admission.
TEST_START_FRACTION = 0.85
GATEWAY_ATTENUATION_WINDOW = (0.88, 0.92)
SATELLITE_INCIDENT_WINDOW = (0.94, 0.98)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _elevation_degrees(
    elapsed_seconds: np.ndarray,
    period_seconds: float,
    phase_radians: float,
    peak_elevation_degrees: float,
) -> np.ndarray:
    """Approximate repeated passes with a clipped sinusoidal elevation arc."""

    orbital_phase = 2.0 * np.pi * elapsed_seconds / period_seconds + phase_radians
    visible_arc = np.maximum(np.sin(orbital_phase), 0.0)
    return np.maximum(5.0, peak_elevation_degrees * visible_arc)


def build_trace(
    *,
    bin_seconds: int,
    duration_hours: float,
    satellites: int,
    gateways: int,
    altitude_km: float,
    seed: int,
    load_multiplier: float = 1.0,
    handover_penalty_ms: float = 10.0,
    gateway_attenuation_ms: float = 24.0,
    satellite_incident_ms: float = 38.0,
    invisible_path_penalty_ms: float = 85.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if bin_seconds <= 0 or duration_hours <= 0:
        raise ValueError("bin duration and experiment duration must be positive")
    if satellites < 3 or gateways < 2:
        raise ValueError("at least three satellites and two gateways are required")

    rng = np.random.default_rng(seed)
    bin_count = int(duration_hours * 3600 / bin_seconds)
    elapsed = np.arange(bin_count, dtype=float) * bin_seconds
    start_epoch = int(pd.Timestamp("2025-05-01T00:00:00Z").timestamp())

    # A shared load process creates correlated queueing without directly
    # revealing future path quality to either predictor.
    common_load = np.zeros(bin_count, dtype=float)
    for index in range(1, bin_count):
        common_load[index] = (
            0.93 * common_load[index - 1] + rng.normal(0.0, 0.75)
        )
    common_load = np.maximum(common_load, 0.0)

    gateway_names = [f"gateway_{index + 1}" for index in range(gateways)]
    gateway_base_ms = np.linspace(12.0, 22.0, gateways)
    rows: list[dict[str, object]] = []
    handover_event_count = 0
    attenuation_event_count = 0

    for satellite_index in range(satellites):
        period_seconds = 5550.0 + 90.0 * satellite_index
        phase = 2.0 * np.pi * satellite_index / satellites
        peak_elevation = 55.0 + 30.0 * ((satellite_index % 3) / 2.0)
        elevation = _elevation_degrees(
            elapsed,
            period_seconds,
            phase,
            peak_elevation,
        )
        visible = elevation > 10.0
        propagation = np.array(
            [
                propagation_rtt_lower_bound_ms(
                    altitude_km + 15.0 * (satellite_index % 3),
                    float(angle),
                    space_legs=4,
                )
                for angle in elevation
            ]
        )

        for gateway_index, gateway_name in enumerate(gateway_names):
            relative_path = (
                f"physics_informed/{gateway_name}/satellite_{satellite_index + 1}"
            )
            queue_state = np.zeros(bin_count, dtype=float)
            independent_noise = rng.normal(0.0, 1.2, bin_count)
            for index in range(1, bin_count):
                queue_state[index] = max(
                    0.0,
                    0.86 * queue_state[index - 1]
                    + 0.35 * common_load[index]
                    + rng.normal(0.0, 0.45),
                )

            # Deterministic intervals represent gateway attenuation and a
            # satellite-specific service incident. Their labels are retained
            # only for post-hoc diagnostics and are not model features.
            attenuation = np.zeros(bin_count, dtype=float)
            attenuation_mask = (
                (elapsed >= duration_hours * 3600 * GATEWAY_ATTENUATION_WINDOW[0])
                & (elapsed < duration_hours * 3600 * GATEWAY_ATTENUATION_WINDOW[1])
                & (gateway_index == gateways - 1)
            )
            attenuation[attenuation_mask] = gateway_attenuation_ms
            incident_mask = (
                (elapsed >= duration_hours * 3600 * SATELLITE_INCIDENT_WINDOW[0])
                & (elapsed < duration_hours * 3600 * SATELLITE_INCIDENT_WINDOW[1])
                & (satellite_index == 1)
            )
            attenuation[incident_mask] += satellite_incident_ms

            handover_mask = visible & (
                (elevation < 18.0)
                | (np.roll(visible, 1) != visible)
            )
            handover_mask[0] = False
            handover_penalty = (
                handover_mask.astype(float) * handover_penalty_ms
            )
            outage_penalty = (
                (~visible).astype(float) * invisible_path_penalty_ms
            )
            latency = (
                gateway_base_ms[gateway_index]
                + propagation
                + load_multiplier * 1.4 * queue_state
                + load_multiplier * 0.8 * common_load
                + handover_penalty
                + outage_penalty
                + attenuation
                + independent_noise
            )
            latency = np.maximum(latency, propagation + 2.0)
            replies = np.clip(
                100
                - 70 * (~visible).astype(int)
                - 25 * handover_mask.astype(int)
                - 35 * (attenuation > 0).astype(int)
                + rng.integers(-3, 4, bin_count),
                0,
                100,
            )
            latency_std = (
                1.5
                + 0.12 * queue_state
                + 3.0 * handover_mask.astype(float)
                + 5.0 * (attenuation > 0).astype(float)
            )

            handover_event_count += int(handover_mask.sum())
            attenuation_event_count += int((attenuation > 0).sum())
            for bin_index in range(bin_count):
                epoch = start_epoch + bin_index * bin_seconds
                rows.append(
                    {
                        "relative_path": relative_path,
                        "bin_epoch": epoch,
                        "bin_start_utc": pd.to_datetime(epoch, unit="s"),
                        "session_date": pd.to_datetime(epoch, unit="s").normalize(),
                        "latency_mean_ms": float(latency[bin_index]),
                        "latency_std_ms": float(latency_std[bin_index]),
                        "latency_max_ms": float(
                            latency[bin_index] + 2.5 * latency_std[bin_index]
                        ),
                        "observed_replies": int(replies[bin_index]),
                        "path_state": (
                            "active" if visible[bin_index] else "degraded"
                        ),
                        "location": gateway_name,
                        "target_hint": f"satellite_{satellite_index + 1}",
                        "window_duration": f"{duration_hours}h",
                        "probe_interval": f"{bin_seconds * 1000}ms",
                        "bin_seconds": bin_seconds,
                        "measurement_family": "physics_informed_orbital_simulation",
                        "satellite_altitude_km": (
                            altitude_km + 15.0 * (satellite_index % 3)
                        ),
                        "elevation_degrees": float(elevation[bin_index]),
                        "propagation_lower_bound_ms": float(propagation[bin_index]),
                        "handover_event": int(handover_mask[bin_index]),
                        "attenuation_event": int(attenuation[bin_index] > 0),
                        "common_load_state": float(common_load[bin_index]),
                        "queue_state": float(queue_state[bin_index]),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["relative_path", "bin_epoch"]
    ).reset_index(drop=True)
    test_start_epoch = start_epoch + int(
        math.floor(bin_count * TEST_START_FRACTION)
    ) * bin_seconds
    injected_event_rows = frame["attenuation_event"].astype(bool)
    if injected_event_rows.any() and bool(
        frame.loc[injected_event_rows, "bin_epoch"].lt(test_start_epoch).any()
    ):
        raise AssertionError(
            "injected attenuation/incident events must be confined to test"
        )
    validate_time_bin_table(frame)
    metadata = {
        "dataset_name": "physics_informed_orbital_multipath",
        "provenance": "independently generated simulator trace",
        "is_measured_dataset": False,
        "is_hypatia_output": False,
        "independent_of_lens": True,
        "concurrent_alternative_paths": True,
        "bin_seconds": bin_seconds,
        "duration_hours": duration_hours,
        "satellites": satellites,
        "gateways": gateways,
        "candidate_paths_per_decision": satellites * gateways,
        "altitude_km": altitude_km,
        "altitude_range_km": [altitude_km, altitude_km + 30.0],
        "orbital_period_range_seconds": [
            5550.0,
            5550.0 + 90.0 * (satellites - 1),
        ],
        "minimum_visible_elevation_degrees": 10.0,
        "handover_elevation_threshold_degrees": 18.0,
        "gateway_base_delay_range_ms": [12.0, 22.0],
        "space_legs": 4,
        "common_load_ar_coefficient": 0.93,
        "common_load_innovation_std": 0.75,
        "queue_ar_coefficient": 0.86,
        "queue_common_load_coefficient": 0.35,
        "queue_innovation_std": 0.45,
        "independent_latency_noise_std_ms": 1.2,
        "latency_equation": (
            "gateway_base + propagation_rtt + 1.4*load_multiplier*queue "
            "+ 0.8*load_multiplier*common_load + handover + invisible "
            "+ attenuation + independent_noise"
        ),
        "common_load_equation": "L_t=max(0,0.93*L_{t-1}+Normal(0,0.75))",
        "queue_equation": (
            "Q_t=max(0,0.86*Q_{t-1}+0.35*L_t+Normal(0,0.45))"
        ),
        "reply_equation": (
            "clip(100-70*invisible-25*handover-35*attenuation+Integer[-3,3],0,100)"
        ),
        "latency_std_equation": (
            "1.5+0.12*queue+3*handover+5*attenuation"
        ),
        "gateway_attenuation_schedule": {
            "start_fraction": GATEWAY_ATTENUATION_WINDOW[0],
            "end_fraction": GATEWAY_ATTENUATION_WINDOW[1],
            "affected_gateway": "last gateway",
        },
        "satellite_incident_schedule": {
            "start_fraction": SATELLITE_INCIDENT_WINDOW[0],
            "end_fraction": SATELLITE_INCIDENT_WINDOW[1],
            "affected_satellite_index": 1,
        },
        "test_start_fraction": TEST_START_FRACTION,
        "injected_events_confined_to_test": True,
        "random_seed": seed,
        "load_multiplier": load_multiplier,
        "handover_penalty_ms": handover_penalty_ms,
        "gateway_attenuation_ms": gateway_attenuation_ms,
        "satellite_incident_ms": satellite_incident_ms,
        "invisible_path_penalty_ms": invisible_path_penalty_ms,
        "handover_event_rows": handover_event_count,
        "attenuation_event_rows": attenuation_event_count,
        "physical_components": [
            "elevation-dependent slant range",
            "vacuum propagation lower bound",
            "correlated offered load",
            "path queue state",
            "handover transition penalty",
            "gateway attenuation",
            "satellite service incident",
        ],
        "valid_claim": (
            "policy behavior on an independent physics-informed concurrent-path "
            "simulator trace"
        ),
        "invalid_claim": (
            "generalization to an independent measured multi-path deployment"
        ),
    }
    return frame, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/processed/physics_informed_orbital_multipath_5s.csv",
    )
    parser.add_argument("--bin-seconds", type=int, default=5)
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--satellites", type=int, default=6)
    parser.add_argument("--gateways", type=int, default=3)
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-multiplier", type=float, default=1.0)
    parser.add_argument("--handover-penalty-ms", type=float, default=10.0)
    parser.add_argument("--gateway-attenuation-ms", type=float, default=24.0)
    parser.add_argument("--satellite-incident-ms", type=float, default=38.0)
    parser.add_argument("--invisible-path-penalty-ms", type=float, default=85.0)
    args = parser.parse_args()

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame, metadata = build_trace(
        bin_seconds=args.bin_seconds,
        duration_hours=args.duration_hours,
        satellites=args.satellites,
        gateways=args.gateways,
        altitude_km=args.altitude_km,
        seed=args.seed,
        load_multiplier=args.load_multiplier,
        handover_penalty_ms=args.handover_penalty_ms,
        gateway_attenuation_ms=args.gateway_attenuation_ms,
        satellite_incident_ms=args.satellite_incident_ms,
        invisible_path_penalty_ms=args.invisible_path_penalty_ms,
    )
    frame.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"trace_written={output_path}")
    print(f"metadata_written={metadata_path}")
    print(
        f"rows={len(frame)} paths={frame['relative_path'].nunique()} "
        f"decision_bins={frame['bin_epoch'].nunique()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
