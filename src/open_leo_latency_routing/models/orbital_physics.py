"""Minimal physical bounds used to sanity-check short-horizon decisions."""

from __future__ import annotations

import math


EARTH_RADIUS_KM = 6371.0
LIGHT_SPEED_KM_S = 299_792.458
EARTH_MU_KM3_S2 = 398_600.4418


def slant_range_km(
    altitude_km: float,
    elevation_degrees: float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """Return ground-to-satellite slant range for a spherical Earth."""

    if altitude_km <= 0:
        raise ValueError("satellite altitude must be positive")
    if not 0.0 <= elevation_degrees <= 90.0:
        raise ValueError("elevation must lie in [0, 90] degrees")
    elevation = math.radians(elevation_degrees)
    orbital_radius = earth_radius_km + altitude_km
    return (
        math.sqrt(
            orbital_radius**2
            - (earth_radius_km * math.cos(elevation)) ** 2
        )
        - earth_radius_km * math.sin(elevation)
    )


def propagation_rtt_lower_bound_ms(
    altitude_km: float,
    elevation_degrees: float,
    space_legs: int = 2,
) -> float:
    """Return a vacuum-propagation RTT lower bound for repeated slant legs."""

    if space_legs <= 0:
        raise ValueError("space_legs must be positive")
    distance_km = space_legs * slant_range_km(altitude_km, elevation_degrees)
    return 1000.0 * distance_km / LIGHT_SPEED_KM_S


def circular_orbit_speed_km_s(altitude_km: float) -> float:
    """Return circular-orbit speed from the two-body approximation."""

    return math.sqrt(EARTH_MU_KM3_S2 / (EARTH_RADIUS_KM + altitude_km))


def control_horizon_margin_ms(
    forecast_horizon_seconds: float,
    control_loop_latency_ms: float,
) -> float:
    """Return remaining forecast horizon after control-state collection."""

    return forecast_horizon_seconds * 1000.0 - control_loop_latency_ms
