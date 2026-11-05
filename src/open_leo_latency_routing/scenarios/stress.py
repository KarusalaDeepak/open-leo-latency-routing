"""Synthetic stress generators applied to the processed LENS time-bin table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_burst_stress(
    time_bins: pd.DataFrame,
    seed: int = 42,
    burst_fraction: float = 0.12,
    latency_spike_ms: float = 18.0,
    reply_drop_fraction: float = 0.12,
) -> pd.DataFrame:
    """Inject short-lived burst spikes into a subset of time bins."""
    frame = time_bins.copy()
    rng = np.random.default_rng(seed)

    std_scale = frame["latency_std_ms"].fillna(0.0)
    std_scale = (std_scale - std_scale.min()) / max(1e-9, std_scale.max() - std_scale.min())
    active_scale = np.where(frame["path_state"].eq("active"), 1.0, 0.75)
    event_prob = np.clip(burst_fraction * (1.0 + 0.9 * std_scale) * active_scale, 0.0, 0.85)
    affected = rng.random(len(frame)) < event_prob

    severity = latency_spike_ms + 0.45 * frame["latency_std_ms"].fillna(0.0).to_numpy()
    severity = severity * (0.8 + 0.4 * rng.random(len(frame)))

    frame.loc[affected, "latency_mean_ms"] += severity[affected]
    frame.loc[affected, "latency_std_ms"] += 0.55 * severity[affected]
    frame.loc[affected, "latency_max_ms"] = np.maximum(
        frame.loc[affected, "latency_max_ms"],
        frame.loc[affected, "latency_mean_ms"] + 0.5 * severity[affected],
    )
    frame.loc[affected, "observed_replies"] = np.maximum(
        1,
        np.floor(frame.loc[affected, "observed_replies"] * (1.0 - reply_drop_fraction)).astype(int),
    )
    frame["stress_scenario"] = "burst"
    frame["stress_applied"] = affected.astype(int)
    return frame


def apply_outage_stress(
    time_bins: pd.DataFrame,
    seed: int = 43,
    session_fraction: float = 0.25,
    outage_length_bins: tuple[int, int] = (2, 5),
    latency_spike_ms: float = 35.0,
    reply_drop_fraction: float = 0.35,
) -> pd.DataFrame:
    """Inject contiguous outage windows into a subset of sessions."""
    frame = time_bins.copy()
    rng = np.random.default_rng(seed)
    frame["stress_applied"] = 0

    for relative_path, group in frame.groupby("relative_path", sort=False):
        inactive_bonus = 0.18 if group["path_state"].iloc[0] == "inactive" else 0.0
        select_prob = min(0.95, session_fraction + inactive_bonus)
        if rng.random() >= select_prob or len(group) < outage_length_bins[0]:
            continue

        length = int(rng.integers(outage_length_bins[0], outage_length_bins[1] + 1))
        length = min(length, len(group))
        tail_start_min = max(0, int(len(group) * 0.65) - length)
        start = int(rng.integers(tail_start_min, len(group) - length + 1))
        affected_index = group.index[start : start + length]
        severity = latency_spike_ms + 0.35 * frame.loc[affected_index, "latency_std_ms"].fillna(0.0)
        if group["path_state"].iloc[0] == "inactive":
            severity += 10.0

        frame.loc[affected_index, "latency_mean_ms"] += severity
        frame.loc[affected_index, "latency_std_ms"] += 0.7 * severity
        frame.loc[affected_index, "latency_max_ms"] = np.maximum(
            frame.loc[affected_index, "latency_max_ms"],
            frame.loc[affected_index, "latency_mean_ms"] + 0.75 * severity,
        )
        frame.loc[affected_index, "observed_replies"] = np.maximum(
            1,
            np.floor(
                frame.loc[affected_index, "observed_replies"] * (1.0 - reply_drop_fraction)
            ).astype(int),
        )
        frame.loc[affected_index, "stress_applied"] = 1

    frame["stress_scenario"] = "outage"
    return frame


def apply_structural_shift_stress(
    time_bins: pd.DataFrame,
    seed: int = 44,
    location_fraction: float = 0.35,
    window_count: int = 2,
    window_length_bins: tuple[int, int] = (2, 4),
    latency_spike_ms: float = 28.0,
    reply_drop_fraction: float = 0.20,
) -> pd.DataFrame:
    """Inject correlated degradation across location groups and normalized time windows.

    The goal is to create a shift that is not just local to one path. We first
    select a subset of locations, then apply contiguous degradation windows at
    the same relative progression positions across all selected locations.
    """

    frame = time_bins.copy()
    rng = np.random.default_rng(seed)
    frame = frame.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    frame["stress_applied"] = 0

    # Relative path progression approximates the normalized decision window used
    # later in the path-selection stage.
    frame["_relative_position"] = frame.groupby("relative_path").cumcount()

    unique_locations = sorted(frame["location"].dropna().unique().tolist())
    if not unique_locations:
        frame["stress_scenario"] = "structural"
        return frame.drop(columns=["_relative_position"])

    selected_count = max(2, int(np.ceil(len(unique_locations) * location_fraction)))
    selected_count = min(selected_count, len(unique_locations))
    selected_locations = set(rng.choice(unique_locations, size=selected_count, replace=False).tolist())

    max_position = int(frame["_relative_position"].max())
    affected_positions: set[int] = set()
    for _ in range(max(1, window_count)):
        length = int(rng.integers(window_length_bins[0], window_length_bins[1] + 1))
        length = max(1, min(length, max_position + 1))
        tail_start_min = max(0, int((max_position + 1) * 0.65) - length)
        start = int(rng.integers(tail_start_min, max(1, max_position - length + 2)))
        affected_positions.update(range(start, start + length))

    affected = frame["location"].isin(selected_locations) & frame["_relative_position"].isin(affected_positions)
    severity = latency_spike_ms + 0.30 * frame["latency_std_ms"].fillna(0.0)
    severity = severity * (0.90 + 0.25 * rng.random(len(frame)))

    frame.loc[affected, "latency_mean_ms"] += severity[affected]
    frame.loc[affected, "latency_std_ms"] += 0.60 * severity[affected]
    frame.loc[affected, "latency_max_ms"] = np.maximum(
        frame.loc[affected, "latency_max_ms"],
        frame.loc[affected, "latency_mean_ms"] + 0.65 * severity[affected],
    )
    frame.loc[affected, "observed_replies"] = np.maximum(
        1,
        np.floor(frame.loc[affected, "observed_replies"] * (1.0 - reply_drop_fraction)).astype(int),
    )
    frame.loc[affected, "stress_applied"] = 1
    frame["stress_scenario"] = "structural"
    return frame.drop(columns=["_relative_position"])


def build_stress_scenarios(time_bins: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the base table plus predefined stress variants."""
    base = time_bins.copy()
    base["stress_scenario"] = "base"
    base["stress_applied"] = 0
    return {
        "base": base,
        "burst": apply_burst_stress(time_bins),
        "outage": apply_outage_stress(time_bins),
        "structural": apply_structural_shift_stress(time_bins),
    }
