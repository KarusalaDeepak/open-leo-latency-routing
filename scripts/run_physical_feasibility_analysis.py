#!/usr/bin/env python3
"""Generate propagation and control-horizon sanity bounds."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.models.orbital_physics import (
    circular_orbit_speed_km_s,
    control_horizon_margin_ms,
    propagation_rtt_lower_bound_ms,
    slant_range_km,
)


def main() -> int:
    output_dir = REPO_ROOT / "results" / "physical_feasibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    propagation_rows = []
    for altitude_km in (550.0, 1200.0):
        for elevation_degrees in (10.0, 30.0, 60.0, 90.0):
            propagation_rows.append(
                {
                    "altitude_km": altitude_km,
                    "elevation_degrees": elevation_degrees,
                    "slant_range_km": slant_range_km(
                        altitude_km, elevation_degrees
                    ),
                    "two_space_leg_propagation_ms": propagation_rtt_lower_bound_ms(
                        altitude_km, elevation_degrees, space_legs=2
                    ),
                    "four_space_leg_propagation_ms": propagation_rtt_lower_bound_ms(
                        altitude_km, elevation_degrees, space_legs=4
                    ),
                    "circular_orbit_speed_km_s": circular_orbit_speed_km_s(
                        altitude_km
                    ),
                }
            )
    control_rows = []
    for horizon_seconds in (5.0, 10.0, 30.0, 60.0):
        for control_latency_ms in (10.0, 50.0, 100.0, 500.0, 1000.0):
            margin = control_horizon_margin_ms(
                horizon_seconds, control_latency_ms
            )
            control_rows.append(
                {
                    "forecast_horizon_seconds": horizon_seconds,
                    "control_loop_latency_ms": control_latency_ms,
                    "remaining_horizon_ms": margin,
                    "decision_not_stale": int(margin > 0.0),
                }
            )
    pd.DataFrame(propagation_rows).to_csv(
        output_dir / "propagation_bounds.csv", index=False
    )
    pd.DataFrame(control_rows).to_csv(
        output_dir / "control_horizon_bounds.csv", index=False
    )
    print(f"physical_feasibility_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
