"""Temporal feature construction for forecasting baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np
import pandas as pd


def _exact_history_values(
    frame: pd.DataFrame,
    column: str,
    steps_back: int,
    decision_cadence_seconds: float,
) -> pd.Series:
    """Look up one scheduled history slot without using row adjacency."""

    lookup_index = pd.MultiIndex.from_frame(
        frame[["relative_path", "bin_epoch"]]
    )
    lookup = pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        index=lookup_index,
    )
    requested = pd.MultiIndex.from_arrays(
        [
            frame["relative_path"].to_numpy(),
            frame["bin_epoch"].to_numpy(dtype=float)
            - steps_back * decision_cadence_seconds,
        ]
    )
    return pd.Series(lookup.reindex(requested).to_numpy(), index=frame.index)


def add_lag_features(
    frame: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    decision_cadence_seconds: float,
) -> pd.DataFrame:
    """Add exact scheduled lags and an observability flag for each lag."""

    output = frame.copy()
    for lag in lags:
        lag = int(lag)
        if lag < 1:
            raise ValueError("lag steps must be positive")
        lag_column = f"{column}_lag_{lag}"
        output[lag_column] = _exact_history_values(
            output,
            column,
            lag,
            decision_cadence_seconds,
        )
        output[f"{lag_column}_available"] = output[lag_column].notna().astype(int)
    return output


def add_rolling_features(
    frame: pd.DataFrame,
    column: str,
    windows: Iterable[int],
    decision_cadence_seconds: float,
) -> pd.DataFrame:
    """Add rolling features over scheduled slots, retaining gap indicators."""

    output = frame.copy()
    for window in windows:
        window = int(window)
        if window < 1:
            raise ValueError("rolling windows must be positive")
        scheduled = pd.concat(
            [
                _exact_history_values(
                    output,
                    column,
                    steps_back,
                    decision_cadence_seconds,
                ).rename(str(steps_back))
                for steps_back in range(window)
            ],
            axis=1,
        )
        observed_count = scheduled.notna().sum(axis=1)
        output[f"{column}_roll_mean_{window}"] = scheduled.mean(axis=1)
        output[f"{column}_roll_std_{window}"] = scheduled.std(axis=1).fillna(0.0)
        output[f"{column}_roll_observed_count_{window}"] = observed_count
        output[f"{column}_roll_coverage_{window}"] = observed_count / window
        output[f"{column}_roll_complete_{window}"] = observed_count.eq(window).astype(int)
    return output


def add_static_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert compact metadata fields into numeric columns."""
    output = frame.copy()
    output["path_state_flag"] = (output["path_state"] == "active").astype(int)
    output["window_duration_hours"] = pd.to_numeric(
        output["window_duration"].astype(str).str.extract(r"([\d.]+)")[0],
        errors="coerce",
    ).fillna(0.0)
    output["probe_interval_ms"] = pd.to_numeric(
        output["probe_interval"].astype(str).str.extract(r"([\d.]+)")[0],
        errors="coerce",
    ).fillna(0.0)
    output["session_day_of_month"] = output["session_date"].dt.day.fillna(0).astype(int)
    return output


def build_forecast_table(
    time_bins: pd.DataFrame,
    target_column: str,
    lags: list[int],
    horizon_bins: int,
    *,
    decision_cadence_seconds: float,
    multi_bin_horizons: Iterable[int] = (3, 5),
    require_complete_decision_epochs: bool = False,
) -> pd.DataFrame:
    """Create exact adjacent-bin aggregate targets from session time bins.

    ``decision_cadence_seconds`` is a protocol input, not an estimated data
    statistic.  ``bin_epoch`` identifies the start of an aggregate interval.
    After the current interval closes, a one-step target is the aggregate over
    the immediately following interval, whose start key is exactly one cadence
    later.  A multi-bin target likewise requires every intervening start key.
    Thus a missing aggregate bin is never reinterpreted as a longer, unknown
    forecasting horizon merely because another observation appears later.
    """
    if horizon_bins < 1:
        raise ValueError("horizon_bins must be positive")
    decision_cadence_seconds = float(decision_cadence_seconds)
    if (
        not np.isfinite(decision_cadence_seconds)
        or decision_cadence_seconds <= 0.0
    ):
        raise ValueError("decision_cadence_seconds must be finite and positive")
    normalized_multi_bin_horizons = tuple(
        dict.fromkeys(int(horizon) for horizon in multi_bin_horizons)
    )
    if any(horizon < 1 for horizon in normalized_multi_bin_horizons):
        raise ValueError("multi-bin horizons must be positive")

    required_columns = {"relative_path", "bin_epoch", target_column}
    missing_columns = required_columns.difference(time_bins.columns)
    if missing_columns:
        raise KeyError(
            "forecast table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = time_bins.copy()
    if "bin_seconds" in frame:
        declared_cadence = pd.to_numeric(frame["bin_seconds"], errors="coerce")
        if declared_cadence.isna().any() or not np.isfinite(
            declared_cadence.to_numpy(dtype=float)
        ).all():
            raise ValueError("bin_seconds must contain only finite values")
        if not np.isclose(
            declared_cadence.to_numpy(dtype=float),
            decision_cadence_seconds,
            rtol=0.0,
            atol=max(1.0e-9, abs(decision_cadence_seconds) * 1.0e-12),
        ).all():
            raise ValueError(
                "decision_cadence_seconds does not match the source bin_seconds"
            )
    frame["bin_epoch"] = pd.to_numeric(frame["bin_epoch"], errors="coerce")
    if frame["bin_epoch"].isna().any() or not np.isfinite(
        frame["bin_epoch"].to_numpy(dtype=float)
    ).all():
        raise ValueError("bin_epoch must contain only finite numeric epochs")
    if frame.duplicated(["relative_path", "bin_epoch"]).any():
        raise ValueError(
            "forecast targets require unique bin_epoch values within each path"
        )
    frame = frame.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    scheduled_grid_expected_row_count = 0
    scheduled_grid_off_phase_row_count = 0
    for _, path_frame in frame.groupby("relative_path", sort=False):
        offsets = (
            path_frame["bin_epoch"].to_numpy(dtype=float)
            - float(path_frame["bin_epoch"].min())
        ) / decision_cadence_seconds
        rounded_offsets = np.rint(offsets)
        on_phase = np.isclose(
            offsets,
            rounded_offsets,
            rtol=0.0,
            atol=1.0e-9,
        )
        scheduled_grid_off_phase_row_count += int((~on_phase).sum())
        scheduled_grid_expected_row_count += int(rounded_offsets.max()) + 1
    scheduled_grid_missing_row_count = max(
        0,
        scheduled_grid_expected_row_count
        - (len(frame) - scheduled_grid_off_phase_row_count),
    )
    frame = add_static_numeric_features(frame)
    normalized_lags = tuple(dict.fromkeys(int(lag) for lag in lags))
    frame = add_lag_features(
        frame,
        target_column,
        normalized_lags,
        decision_cadence_seconds,
    )
    frame = add_rolling_features(
        frame,
        target_column,
        windows=[3, 5],
        decision_cadence_seconds=decision_cadence_seconds,
    )

    # Reply-count pressure and latency jump features are intentionally simple:
    # they expose burst-like behavior without requiring private telemetry.
    frame = add_lag_features(
        frame,
        "observed_replies",
        lags=[1],
        decision_cadence_seconds=decision_cadence_seconds,
    )
    frame = add_rolling_features(
        frame,
        "observed_replies",
        windows=[3],
        decision_cadence_seconds=decision_cadence_seconds,
    )
    lag_availability_columns = [
        f"{target_column}_lag_{lag}_available" for lag in normalized_lags
    ]
    frame["history_lag_coverage_ratio"] = (
        frame[lag_availability_columns].mean(axis=1)
        if lag_availability_columns
        else 1.0
    )
    full_lag_history = frame["history_lag_coverage_ratio"].eq(1.0)
    history_gap_row_count = int((~full_lag_history).sum())
    frame["latency_delta_1"] = frame[target_column] - frame[f"{target_column}_lag_1"]
    frame["latency_delta_roll3"] = frame[target_column] - frame[f"{target_column}_roll_mean_3"]
    frame["latency_jump_ratio"] = frame[target_column] / frame[f"{target_column}_roll_mean_3"].clip(lower=1e-6)
    frame["latency_volatility_ratio"] = frame[f"{target_column}_roll_std_3"] / frame[
        f"{target_column}_roll_mean_3"
    ].clip(lower=1e-6)
    frame["reply_delta_1"] = frame["observed_replies"] - frame["observed_replies_lag_1"]
    frame["reply_gap_roll3"] = frame["observed_replies"] - frame["observed_replies_roll_mean_3"]
    frame["reply_pressure_score"] = 1.0 - (
        frame["observed_replies"] / frame["observed_replies_roll_mean_3"].clip(lower=1e-6)
    )
    frame["burst_indicator"] = (
        frame["latency_jump_ratio"].clip(lower=0.0)
        + frame["latency_volatility_ratio"].clip(lower=0.0)
        + frame["reply_pressure_score"].clip(lower=0.0)
    ) / 3.0

    grouped_target = frame.groupby("relative_path", sort=False)[target_column]
    grouped_epoch = frame.groupby("relative_path", sort=False)["bin_epoch"]
    current_epoch = frame["bin_epoch"].astype(float)
    epoch_tolerance_seconds = max(
        1.0e-9,
        abs(decision_cadence_seconds) * 1.0e-12,
    )

    def _exact_future_sequence(
        horizon: int,
    ) -> tuple[list[pd.Series], pd.Series]:
        shifted_epochs = [
            grouped_epoch.shift(-step) for step in range(1, horizon + 1)
        ]
        exact = pd.Series(True, index=frame.index, dtype=bool)
        for step, shifted_epoch in enumerate(shifted_epochs, start=1):
            expected_delta = step * decision_cadence_seconds
            delta = shifted_epoch.astype(float) - current_epoch
            exact &= shifted_epoch.notna() & pd.Series(
                np.isclose(
                    delta.to_numpy(dtype=float),
                    expected_delta,
                    rtol=0.0,
                    atol=epoch_tolerance_seconds,
                    equal_nan=False,
                ),
                index=frame.index,
            )
        return shifted_epochs, exact

    # A horizon of more than one bin still requires every intermediate epoch;
    # observing only the endpoint is not enough to establish exact cadence.
    one_step_epochs, one_step_exact = _exact_future_sequence(horizon_bins)
    raw_target_next = grouped_target.shift(-horizon_bins)
    target_available = one_step_exact & raw_target_next.notna()
    frame["target_next"] = raw_target_next.where(target_available)
    frame["target_next_bin_epoch"] = one_step_epochs[-1].where(target_available)
    frame["target_available"] = target_available.astype(int)
    frame["target_expected_cadence_seconds"] = decision_cadence_seconds
    frame["target_expected_horizon_seconds"] = (
        horizon_bins * decision_cadence_seconds
    )
    # Historical field name retained for artifact compatibility: the exact
    # timestamp is the target aggregate bin's start key, not a point outcome.
    frame["target_exact_wall_clock"] = target_available.astype(int)

    positional_target_count = int(raw_target_next.notna().sum())
    excluded_nonexact_gap_rows = int(
        (raw_target_next.notna() & ~one_step_exact).sum()
    )

    # Journal-facing analyses also inspect short multi-bin service outcomes.
    # These future-window targets remain NaN when insufficient future bins are
    # available so downstream evaluation can track the effective sample size.
    future_target_columns: list[str] = []
    future_target_epoch_columns: list[str] = []
    multi_bin_audit: dict[str, dict[str, int | float]] = {}
    for horizon in normalized_multi_bin_horizons:
        shifted_terms = [grouped_target.shift(-step) for step in range(1, horizon + 1)]
        shifted_epochs, exact_sequence = _exact_future_sequence(horizon)
        future_sum = shifted_terms[0].copy()
        for term in shifted_terms[1:]:
            future_sum = future_sum + term
        available = exact_sequence.copy()
        for term in shifted_terms:
            available &= term.notna()
        frame[f"target_cumulative_{horizon}"] = future_sum.where(available)
        frame[f"target_mean_{horizon}"] = (future_sum / horizon).where(available)
        frame[f"target_available_{horizon}"] = available.astype(int)
        # Store the exact final epoch used by this outcome. Splitters can then
        # prove that the complete future window stays inside one partition,
        # including for horizons other than the historical 3/5-bin defaults.
        endpoint_column = f"target_end_bin_epoch_{horizon}"
        frame[endpoint_column] = shifted_epochs[-1].where(available)
        positional_complete = pd.Series(True, index=frame.index, dtype=bool)
        for term in shifted_terms:
            positional_complete &= term.notna()
        multi_bin_audit[str(horizon)] = {
            "horizon_bins": horizon,
            "expected_horizon_seconds": horizon * decision_cadence_seconds,
            "positional_complete_row_count": int(positional_complete.sum()),
            "retained_exact_row_count": int(available.sum()),
            "excluded_nonexact_gap_row_count": int(
                (positional_complete & ~exact_sequence).sum()
            ),
        }
        future_target_columns.extend(
            [
                f"target_cumulative_{horizon}",
                f"target_mean_{horizon}",
            ]
        )
        future_target_epoch_columns.append(endpoint_column)

    numeric_columns = [
        column
        for column in frame.select_dtypes(include=["number"]).columns
        if column
        not in {
            "target_next",
            "target_next_bin_epoch",
            *future_target_columns,
            *future_target_epoch_columns,
        }
    ]
    frame[numeric_columns] = frame[numeric_columns].fillna(0.0)

    incomplete_decision_epoch_count = 0
    asymmetric_missing_candidate_epoch_count = 0
    incomplete_required_candidate_row_count = 0
    complete_decision_epoch = pd.Series(True, index=frame.index, dtype=bool)
    if require_complete_decision_epochs:
        if "session_bin_index" not in frame:
            raise KeyError(
                "complete decision-epoch targets require session_bin_index"
            )
        feasible_states = {"active", "available", "up", "healthy"}
        required_candidate = pd.Series(True, index=frame.index, dtype=bool)
        if "path_state" in frame:
            required_candidate &= frame["path_state"].astype(str).str.lower().isin(
                feasible_states
            )
        if "observed_replies" in frame:
            required_candidate &= frame["observed_replies"].astype(float).gt(0.0)
        missing_required = required_candidate & ~target_available
        missing_by_epoch = missing_required.groupby(
            frame["session_bin_index"]
        ).transform("any")
        required_exact_by_epoch = (required_candidate & target_available).groupby(
            frame["session_bin_index"]
        ).transform("any")
        complete_decision_epoch = ~missing_by_epoch
        incomplete_decision_epoch_count = int(
            frame.loc[missing_by_epoch, "session_bin_index"].nunique()
        )
        asymmetric_missing_candidate_epoch_count = int(
            frame.loc[
                missing_by_epoch & required_exact_by_epoch,
                "session_bin_index",
            ].nunique()
        )
        incomplete_required_candidate_row_count = int(missing_required.sum())
    frame["target_complete_decision_epoch"] = complete_decision_epoch.astype(int)

    retention_mask = target_available & complete_decision_epoch
    retained = frame.loc[retention_mask].copy().reset_index(drop=True)
    retained.attrs["exact_horizon_audit"] = {
        "endpoint_semantics": (
            "exact_target_bin_start_with_complete_intermediate_sequence"
        ),
        "decision_cadence_seconds": decision_cadence_seconds,
        "one_step_horizon_bins": horizon_bins,
        "one_step_expected_horizon_seconds": (
            horizon_bins * decision_cadence_seconds
        ),
        "input_row_count": int(len(frame)),
        "scheduled_history_semantics": "exact_wall_clock_slots",
        "scheduled_history_lags": list(normalized_lags),
        "scheduled_grid_expected_row_count": (
            scheduled_grid_expected_row_count
        ),
        "scheduled_grid_observed_row_count": int(len(frame)),
        "scheduled_grid_missing_row_count": scheduled_grid_missing_row_count,
        "scheduled_grid_off_phase_row_count": (
            scheduled_grid_off_phase_row_count
        ),
        "history_gap_row_count": history_gap_row_count,
        "full_lag_history_row_count": int(full_lag_history.sum()),
        "positional_target_row_count": positional_target_count,
        "exact_target_row_count_before_decision_completeness": int(
            target_available.sum()
        ),
        "excluded_nonexact_gap_row_count": excluded_nonexact_gap_rows,
        "require_complete_decision_epochs": bool(
            require_complete_decision_epochs
        ),
        "excluded_incomplete_decision_epoch_count": (
            incomplete_decision_epoch_count
        ),
        "asymmetric_missing_candidate_epoch_count": (
            asymmetric_missing_candidate_epoch_count
        ),
        "incomplete_required_candidate_row_count": (
            incomplete_required_candidate_row_count
        ),
        "retained_exact_target_row_count": int(len(retained)),
        "retained_full_lag_history_row_count": int(
            retained["history_lag_coverage_ratio"].eq(1.0).sum()
        ),
        "retained_decision_epoch_count": (
            int(retained["session_bin_index"].nunique())
            if "session_bin_index" in retained
            else None
        ),
        "multi_bin_horizons": multi_bin_audit,
    }
    return retained


WALL_CLOCK_SCHEDULE_AUDIT_ATTR = "wall_clock_decision_schedule_audit"
WALL_CLOCK_SPLIT_AUDIT_ATTR = "wall_clock_split_audit"


def build_wall_clock_decision_schedule(
    frame: pd.DataFrame,
    decision_cadence_seconds: float | None = None,
) -> pd.DataFrame:
    """Build the full exact-cadence decision grid before outcome filtering.

    The grid spans the inclusive raw minimum and maximum ``bin_epoch`` values.
    Missing observed bins remain scheduled decision slots, so deleting or adding
    rows with usable future outcomes cannot move a chronological boundary.
    """

    if frame.empty:
        raise ValueError("wall-clock decision schedule requires non-empty data")
    if "bin_epoch" not in frame:
        raise ValueError("wall-clock decision schedule requires bin_epoch")

    if decision_cadence_seconds is None:
        if "bin_seconds" not in frame:
            raise ValueError(
                "decision_cadence_seconds or a constant bin_seconds column is required"
            )
        declared = pd.to_numeric(frame["bin_seconds"], errors="coerce")
        distinct = np.sort(declared.dropna().unique())
        if len(distinct) != 1 or declared.isna().any():
            raise ValueError("bin_seconds must contain one finite cadence value")
        decision_cadence_seconds = float(distinct[0])

    cadence = float(decision_cadence_seconds)
    if not np.isfinite(cadence) or cadence <= 0.0:
        raise ValueError("decision_cadence_seconds must be finite and positive")

    observed = pd.to_numeric(frame["bin_epoch"], errors="coerce")
    if observed.isna().any() or not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError("bin_epoch must contain only finite numeric epochs")
    observed_epochs = np.sort(observed.unique().astype(float))
    first_epoch = float(observed_epochs[0])
    last_epoch = float(observed_epochs[-1])
    tolerance = max(1.0e-9, abs(cadence) * 1.0e-9)
    offsets = (observed_epochs - first_epoch) / cadence
    rounded_offsets = np.rint(offsets)
    on_phase = np.isclose(offsets, rounded_offsets, rtol=0.0, atol=1.0e-9)
    if not bool(on_phase.all()):
        off_phase = observed_epochs[~on_phase]
        raise ValueError(
            "observed bin_epoch values are off the declared cadence grid: "
            f"{off_phase[:5].tolist()}"
        )

    span_steps_float = (last_epoch - first_epoch) / cadence
    span_steps = int(np.rint(span_steps_float))
    if not np.isclose(
        span_steps_float,
        span_steps,
        rtol=0.0,
        atol=tolerance / cadence,
    ):
        raise ValueError("raw minimum and maximum do not close an exact cadence grid")
    scheduled_epochs = first_epoch + cadence * np.arange(span_steps + 1, dtype=float)
    schedule = pd.DataFrame(
        {
            "schedule_index": np.arange(len(scheduled_epochs), dtype=int),
            "bin_epoch": scheduled_epochs,
        }
    )
    observed_slot_indices = np.rint(offsets).astype(int)
    schedule["observed_in_raw_trace"] = 0
    schedule.loc[observed_slot_indices, "observed_in_raw_trace"] = 1
    integer_epoch_grid = np.isclose(
        scheduled_epochs,
        np.rint(scheduled_epochs),
        rtol=0.0,
        atol=tolerance,
    ).all()
    if integer_epoch_grid:
        schedule["bin_epoch"] = np.rint(schedule["bin_epoch"]).astype("int64")
    schedule.attrs[WALL_CLOCK_SCHEDULE_AUDIT_ATTR] = {
        "boundary_schedule_source": "raw_min_max_exact_cadence_grid",
        "target_availability_used_for_schedule": False,
        "decision_cadence_seconds": cadence,
        "first_scheduled_bin_epoch": first_epoch,
        "last_scheduled_bin_epoch": last_epoch,
        "scheduled_decision_epoch_count": int(len(schedule)),
        "observed_raw_decision_epoch_count": int(len(observed_epochs)),
        "missing_raw_decision_epoch_count": int(
            len(schedule) - len(observed_epochs)
        ),
        "off_phase_raw_decision_epoch_count": 0,
    }
    return schedule


def _coerce_wall_clock_schedule(
    decision_schedule: pd.DataFrame | Sequence[float],
) -> tuple[np.ndarray, dict[str, object]]:
    if isinstance(decision_schedule, pd.DataFrame):
        if "bin_epoch" not in decision_schedule:
            raise ValueError("decision schedule requires bin_epoch")
        values = decision_schedule["bin_epoch"]
        inherited_audit = decision_schedule.attrs.get(
            WALL_CLOCK_SCHEDULE_AUDIT_ATTR,
            {},
        )
    else:
        values = pd.Series(list(decision_schedule), dtype=float)
        inherited_audit = {}
    epochs = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(epochs) == 0 or not np.isfinite(epochs).all():
        raise ValueError("decision schedule must contain finite epochs")
    epochs = np.unique(epochs)
    epochs.sort()
    if len(epochs) < 4:
        raise ValueError("decision schedule requires at least four epochs")
    if not bool((np.diff(epochs) > 0.0).all()):
        raise ValueError("decision schedule epochs must be strictly increasing")
    audit = {
        "boundary_schedule_source": inherited_audit.get(
            "boundary_schedule_source",
            "provided_wall_clock_schedule",
        ),
        "target_availability_used_for_schedule": bool(
            inherited_audit.get("target_availability_used_for_schedule", False)
        ),
        "first_scheduled_bin_epoch": float(epochs[0]),
        "last_scheduled_bin_epoch": float(epochs[-1]),
        "scheduled_decision_epoch_count": int(len(epochs)),
        **inherited_audit,
    }
    return epochs, audit


def _interval_audit(
    epochs: np.ndarray,
    start_index: int,
    end_index: int,
) -> dict[str, int | float]:
    if start_index < 0 or end_index <= start_index or end_index > len(epochs):
        raise ValueError("invalid wall-clock partition interval")
    return {
        "first_schedule_index": int(start_index),
        "last_schedule_index": int(end_index - 1),
        "first_bin_epoch": float(epochs[start_index]),
        "last_bin_epoch": float(epochs[end_index - 1]),
        "scheduled_decision_epoch_count": int(end_index - start_index),
    }


def build_four_way_split_plan(
    decision_schedule: pd.DataFrame | Sequence[float],
    *,
    train_ratio: float,
    calibration_ratio: float,
    selection_ratio: float,
    test_ratio: float,
) -> dict[str, object]:
    """Freeze four chronological intervals on a pre-outcome schedule."""

    ratios = train_ratio + calibration_ratio + selection_ratio + test_ratio
    if round(ratios, 6) != 1.0:
        raise ValueError("train/calibration/selection/test ratios must sum to 1.0")
    if min(train_ratio, calibration_ratio, selection_ratio, test_ratio) <= 0:
        raise ValueError("all four split ratios must be positive")

    epochs, schedule_audit = _coerce_wall_clock_schedule(decision_schedule)
    size = len(epochs)
    train_end = min(max(1, int(size * train_ratio)), size - 3)
    calibration_end = min(
        max(train_end + 1, int(size * (train_ratio + calibration_ratio))),
        size - 2,
    )
    selection_end = min(
        max(
            calibration_end + 1,
            int(size * (train_ratio + calibration_ratio + selection_ratio)),
        ),
        size - 1,
    )
    partitions = {
        "train": _interval_audit(epochs, 0, train_end),
        "calibration": _interval_audit(epochs, train_end, calibration_end),
        "selection": _interval_audit(
            epochs,
            calibration_end,
            selection_end,
        ),
        "test": _interval_audit(epochs, selection_end, size),
    }
    return {
        "boundary_basis": "pre_target_wall_clock_schedule",
        "boundaries_declared_before_target_filtering": True,
        "target_availability_used_for_boundary_derivation": False,
        "schedule": schedule_audit,
        "ratios": {
            "train": float(train_ratio),
            "calibration": float(calibration_ratio),
            "selection": float(selection_ratio),
            "test": float(test_ratio),
        },
        "partitions": partitions,
    }


def build_rolling_origin_split_plan(
    decision_schedule: pd.DataFrame | Sequence[float],
    *,
    fold_count: int,
    minimum_block_size: int = 10,
) -> dict[str, object]:
    """Freeze expanding-window folds on scheduled, not outcome-eligible, bins."""

    folds = int(fold_count)
    if folds < 2:
        raise ValueError("rolling-origin validation requires at least two folds")
    minimum = int(minimum_block_size)
    if minimum < 1:
        raise ValueError("minimum_block_size must be positive")
    epochs, schedule_audit = _coerce_wall_clock_schedule(decision_schedule)
    block_size = max(minimum, len(epochs) // (2 * folds))
    first_test = len(epochs) - folds * block_size
    if first_test <= 2 * block_size:
        raise ValueError("insufficient scheduled epochs for four disjoint intervals")

    fold_plans: list[dict[str, object]] = []
    for fold_index in range(folds):
        test_start = first_test + fold_index * block_size
        test_end = (
            len(epochs)
            if fold_index == folds - 1
            else test_start + block_size
        )
        selection_start = test_start - block_size
        calibration_start = selection_start - block_size
        fold_plans.append(
            {
                "rolling_fold": fold_index + 1,
                "partitions": {
                    "train": _interval_audit(epochs, 0, calibration_start),
                    "calibration": _interval_audit(
                        epochs,
                        calibration_start,
                        selection_start,
                    ),
                    "selection": _interval_audit(
                        epochs,
                        selection_start,
                        test_start,
                    ),
                    "test": _interval_audit(epochs, test_start, test_end),
                },
            }
        )
    return {
        "boundary_basis": "pre_target_wall_clock_schedule",
        "boundaries_declared_before_target_filtering": True,
        "target_availability_used_for_boundary_derivation": False,
        "schedule": schedule_audit,
        "fold_count": folds,
        "minimum_block_size_scheduled_epochs": minimum,
        "block_size_scheduled_epochs": int(block_size),
        "first_test_schedule_index": int(first_test),
        "folds": fold_plans,
    }


def split_train_val_test(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each session chronologically and close every future target.

    One-step rows whose targets cross an internal boundary are removed.  A
    multi-bin target that crosses a boundary is instead marked unavailable so
    the same row can still be used for its closed one-step outcome.
    """
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    split_frames: list[pd.DataFrame] = []
    split_boundaries: dict[object, tuple[float, float]] = {}
    for relative_path, group in frame.groupby("relative_path", sort=False):
        ordered = group.sort_values("bin_epoch").copy()
        size = len(ordered)
        ordered["split_index"] = range(size)
        train_end = max(1, int(size * train_ratio))
        val_end = max(train_end + 1, int(size * (train_ratio + val_ratio)))
        val_end = min(val_end, size)
        ordered["split"] = "test"
        ordered.loc[ordered["split_index"] < train_end, "split"] = "train"
        ordered.loc[
            (ordered["split_index"] >= train_end) & (ordered["split_index"] < val_end),
            "split",
        ] = "val"
        split_boundaries[relative_path] = (
            float(ordered.iloc[train_end - 1]["bin_epoch"]),
            float(ordered.iloc[val_end - 1]["bin_epoch"]),
        )
        split_frames.append(ordered.drop(columns=["split_index"]))

    combined = pd.concat(split_frames, ignore_index=True)
    train_upper = combined["relative_path"].map(
        {key: value[0] for key, value in split_boundaries.items()}
    )
    val_upper = combined["relative_path"].map(
        {key: value[1] for key, value in split_boundaries.items()}
    )

    def _endpoint_partition(values: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [values.le(train_upper), values.le(val_upper)],
                ["train", "val"],
                default="test",
            ),
            index=combined.index,
            dtype="object",
        )

    horizon_columns = {
        int(column.removeprefix("target_available_")): column
        for column in combined.columns
        if column.startswith("target_available_")
        and column.removeprefix("target_available_").isdigit()
    }
    horizon_endpoints: dict[int, pd.Series] = {}
    for horizon in horizon_columns:
        endpoint_column = f"target_end_bin_epoch_{horizon}"
        if endpoint_column in combined:
            horizon_endpoints[horizon] = combined[endpoint_column].copy()
        else:
            if "target_next_bin_epoch" not in combined:
                raise ValueError(
                    f"cannot derive {endpoint_column} without exact one-step "
                    "endpoint metadata"
                )
            if "target_expected_cadence_seconds" in combined:
                exact_cadence = combined["target_expected_cadence_seconds"]
            elif "bin_seconds" in combined:
                exact_cadence = combined["bin_seconds"]
            else:
                # Legacy three-way tables used a one-bin primary target.
                exact_cadence = (
                    combined["target_next_bin_epoch"] - combined["bin_epoch"]
                )
            horizon_endpoints[horizon] = (
                combined["bin_epoch"] + horizon * exact_cadence
            )

    if "target_next_bin_epoch" in combined:
        target_split = _endpoint_partition(combined["target_next_bin_epoch"])
        combined = combined.loc[
            combined["target_next_bin_epoch"].notna()
            & combined["split"].eq(target_split)
        ].copy()

    for horizon, available_column in horizon_columns.items():
        target_columns = [
            f"target_cumulative_{horizon}",
            f"target_mean_{horizon}",
        ]
        if not all(column in combined for column in target_columns):
            continue
        endpoint = horizon_endpoints[horizon].reindex(combined.index)
        # Rebuild path-specific bounds on the retained index before classifying
        # endpoints; bounds themselves remain frozen from the original split.
        train_upper = combined["relative_path"].map(
            {key: value[0] for key, value in split_boundaries.items()}
        )
        val_upper = combined["relative_path"].map(
            {key: value[1] for key, value in split_boundaries.items()}
        )
        endpoint_split = _endpoint_partition(endpoint)
        horizon_closed = endpoint.notna() & combined["split"].eq(endpoint_split)
        horizon_available = combined[available_column].astype(bool) & horizon_closed
        combined.loc[~horizon_available, target_columns] = np.nan
        combined[available_column] = horizon_available.astype(int)

    return (
        combined[combined["split"] == "train"].reset_index(drop=True),
        combined[combined["split"] == "val"].reset_index(drop=True),
        combined[combined["split"] == "test"].reset_index(drop=True),
    )


def split_train_calibration_selection_test(
    frame: pd.DataFrame,
    train_ratio: float,
    calibration_ratio: float,
    selection_ratio: float,
    test_ratio: float,
    *,
    decision_schedule: pd.DataFrame | Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create four globally disjoint chronological partitions.

    Model parameters are fitted on ``train``; residual and uncertainty
    quantities are fitted on ``calibration``; policy admission is decided on
    ``selection``; and ``test`` is evaluated once. Keeping calibration and
    policy selection separate prevents a calibrated candidate from being
    judged on the same outcomes used to calibrate it. Boundaries are defined
    from the shared wall-clock axis, rather than independently within each
    path, so paths with missing observations cannot place the same time epoch
    in adjacent partitions. Rows whose forecasting target crosses a boundary
    are removed from that partition.
    """

    ratios = train_ratio + calibration_ratio + selection_ratio + test_ratio
    if round(ratios, 6) != 1.0:
        raise ValueError("train/calibration/selection/test ratios must sum to 1.0")
    if min(train_ratio, calibration_ratio, selection_ratio, test_ratio) <= 0:
        raise ValueError("all four split ratios must be positive")

    if frame.empty:
        raise ValueError("four-way chronological split requires non-empty data")
    if "bin_epoch" not in frame:
        raise ValueError("four-way chronological split requires bin_epoch")

    if decision_schedule is None:
        # Backward-compatible fallback for callers that have not yet supplied a
        # pre-outcome schedule. COMMECT runners pass the raw exact-cadence grid.
        fallback_schedule = pd.DataFrame(
            {"bin_epoch": np.sort(frame["bin_epoch"].dropna().unique())}
        )
        fallback_schedule.attrs[WALL_CLOCK_SCHEDULE_AUDIT_ATTR] = {
            "boundary_schedule_source": "input_frame_observed_epochs",
            "target_availability_used_for_schedule": True,
            "scheduled_decision_epoch_count": int(len(fallback_schedule)),
            "observed_raw_decision_epoch_count": int(len(fallback_schedule)),
            "missing_raw_decision_epoch_count": 0,
        }
        boundary_schedule = fallback_schedule
        declared_before_target_filtering = False
    else:
        boundary_schedule = decision_schedule
        declared_before_target_filtering = True

    split_plan = build_four_way_split_plan(
        boundary_schedule,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
        selection_ratio=selection_ratio,
        test_ratio=test_ratio,
    )
    split_plan["boundaries_declared_before_target_filtering"] = (
        declared_before_target_filtering
    )
    split_plan["target_availability_used_for_boundary_derivation"] = (
        not declared_before_target_filtering
    )
    partition_plan = split_plan["partitions"]
    schedule_lower = float(split_plan["schedule"]["first_scheduled_bin_epoch"])
    schedule_upper = float(split_plan["schedule"]["last_scheduled_bin_epoch"])
    train_upper = float(partition_plan["train"]["last_bin_epoch"])
    calibration_upper = float(
        partition_plan["calibration"]["last_bin_epoch"]
    )
    selection_upper = float(partition_plan["selection"]["last_bin_epoch"])

    def _partition(values: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [
                    values.le(train_upper),
                    values.le(calibration_upper),
                    values.le(selection_upper),
                ],
                ["train", "calibration", "selection"],
                default="test",
            ),
            index=values.index,
            dtype="object",
        )

    combined = frame.sort_values(["relative_path", "bin_epoch"]).copy()
    combined["split"] = _partition(combined["bin_epoch"])
    before_boundary_filter = combined.copy()

    # Resolve every multi-bin endpoint before removing one-step boundary rows.
    # Deriving an endpoint after filtering would skip removed epochs and could
    # either overrun or over-trim a future outcome window.
    horizon_columns = {
        int(column.removeprefix("target_available_")): column
        for column in combined.columns
        if column.startswith("target_available_")
        and column.removeprefix("target_available_").isdigit()
    }
    horizon_endpoints: dict[int, pd.Series] = {}
    for horizon in horizon_columns:
        endpoint_column = f"target_end_bin_epoch_{horizon}"
        if endpoint_column in combined:
            horizon_endpoints[horizon] = combined[endpoint_column].copy()
        else:
            # Compatibility for tables generated before explicit endpoint
            # columns were introduced.  Derive from the exact one-step cadence
            # rather than from later forecast-row membership.
            if "target_next_bin_epoch" not in combined:
                raise ValueError(
                    f"cannot derive {endpoint_column} without exact one-step "
                    "endpoint metadata"
                )
            if "target_expected_cadence_seconds" in combined:
                exact_cadence = combined["target_expected_cadence_seconds"]
            elif "bin_seconds" in combined:
                exact_cadence = combined["bin_seconds"]
            else:
                # Legacy four-way tables used a one-bin primary target.
                exact_cadence = (
                    combined["target_next_bin_epoch"] - combined["bin_epoch"]
                )
            horizon_endpoints[horizon] = (
                combined["bin_epoch"] + horizon * exact_cadence
            )

    # A target observed after a boundary belongs to the later partition and
    # must not supervise or evaluate a decision made in the earlier one.
    one_step_boundary_excluded = pd.Series(False, index=combined.index, dtype=bool)
    one_step_missing_endpoint = pd.Series(False, index=combined.index, dtype=bool)
    if "target_next_bin_epoch" in combined:
        target_split = _partition(combined["target_next_bin_epoch"])
        one_step_missing_endpoint = combined["target_next_bin_epoch"].isna()
        target_within_schedule = (
            combined["target_next_bin_epoch"].between(
                schedule_lower,
                schedule_upper,
                inclusive="both",
            )
            if declared_before_target_filtering
            else pd.Series(True, index=combined.index, dtype=bool)
        )
        one_step_boundary_excluded = (
            combined["target_next_bin_epoch"].notna()
            & (
                ~target_within_schedule
                | ~combined["split"].eq(target_split)
            )
        )
        combined = combined.loc[
            combined["target_next_bin_epoch"].notna()
            & target_within_schedule
            & combined["split"].eq(target_split)
        ].copy()

    # Apply the same closed-partition rule to optional multi-bin targets while
    # retaining the row for one-step evaluation.
    for horizon, available_column in horizon_columns.items():
        target_columns = [
            f"target_cumulative_{horizon}",
            f"target_mean_{horizon}",
        ]
        if available_column not in combined or not all(
            column in combined for column in target_columns
        ):
            continue
        ordered_endpoint = horizon_endpoints[horizon].reindex(combined.index)
        endpoint_within_schedule = (
            ordered_endpoint.between(
                schedule_lower,
                schedule_upper,
                inclusive="both",
            )
            if declared_before_target_filtering
            else pd.Series(True, index=combined.index, dtype=bool)
        )
        horizon_closed = (
            ordered_endpoint.notna()
            & endpoint_within_schedule
            & combined["split"].eq(_partition(ordered_endpoint))
        )
        horizon_available = combined[available_column].astype(bool) & horizon_closed
        combined.loc[~horizon_available, target_columns] = np.nan
        combined[available_column] = horizon_available.astype(int)

    combined = combined.reset_index(drop=True)
    names = ("train", "calibration", "selection", "test")
    split_audit = {
        **split_plan,
        "input_frame_row_count": int(len(before_boundary_filter)),
        "input_frame_decision_epoch_count": int(
            before_boundary_filter["bin_epoch"].nunique()
        ),
        "one_step_missing_endpoint_row_count": int(
            one_step_missing_endpoint.sum()
        ),
        "one_step_boundary_excluded_row_count": int(
            one_step_boundary_excluded.sum()
        ),
        "partitions": {
            name: dict(partition_plan[name]) for name in names
        },
    }
    outputs: list[pd.DataFrame] = []
    for name in names:
        part = combined[combined["split"].eq(name)].reset_index(drop=True)
        source_part = before_boundary_filter[
            before_boundary_filter["split"].eq(name)
        ]
        split_audit["partitions"][name].update(
            {
                "input_outcome_eligible_row_count": int(len(source_part)),
                "input_outcome_eligible_decision_epoch_count": int(
                    source_part["bin_epoch"].nunique()
                ),
                "one_step_missing_endpoint_row_count": int(
                    one_step_missing_endpoint.reindex(source_part.index).sum()
                ),
                "one_step_boundary_excluded_row_count": int(
                    one_step_boundary_excluded.reindex(source_part.index).sum()
                ),
                "retained_row_count": int(len(part)),
                "retained_decision_epoch_count": int(
                    part["bin_epoch"].nunique()
                ),
            }
        )
        part.attrs.update(frame.attrs)
        part.attrs[WALL_CLOCK_SPLIT_AUDIT_ATTR] = split_audit
        outputs.append(part)
    return tuple(outputs)


def split_group_holdout(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_column: str = "relative_path",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign complete measurement sessions to disjoint data partitions.

    This split supports an in-domain generalization test without placing
    autocorrelated bins from the same session in both training and evaluation.
    """

    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")
    if group_column not in frame:
        raise ValueError(f"missing split group column: {group_column}")

    groups = frame[group_column].drop_duplicates().to_numpy()
    if len(groups) < 3:
        raise ValueError("group holdout requires at least three independent groups")

    rng = np.random.default_rng(random_state)
    groups = rng.permutation(groups)
    train_end = min(max(1, int(len(groups) * train_ratio)), len(groups) - 2)
    val_count = max(1, int(len(groups) * val_ratio))
    val_end = min(train_end + val_count, len(groups) - 1)

    train_groups = set(groups[:train_end])
    val_groups = set(groups[train_end:val_end])
    test_groups = set(groups[val_end:])

    train = frame[frame[group_column].isin(train_groups)].copy()
    val = frame[frame[group_column].isin(val_groups)].copy()
    test = frame[frame[group_column].isin(test_groups)].copy()
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_group_train_calibration_selection_test(
    frame: pd.DataFrame,
    train_ratio: float,
    calibration_ratio: float,
    selection_ratio: float,
    test_ratio: float,
    group_column: str = "relative_path",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign complete autocorrelated groups to four disjoint partitions."""

    if round(train_ratio + calibration_ratio + selection_ratio + test_ratio, 6) != 1.0:
        raise ValueError("group split ratios must sum to 1.0")
    groups = frame[group_column].drop_duplicates().to_numpy()
    if len(groups) < 4:
        raise ValueError("four-way group holdout requires at least four groups")
    groups = np.random.default_rng(random_state).permutation(groups)
    n = len(groups)
    train_end = min(max(1, int(n * train_ratio)), n - 3)
    calibration_end = min(
        max(train_end + 1, int(n * (train_ratio + calibration_ratio))), n - 2
    )
    selection_end = min(
        max(
            calibration_end + 1,
            int(n * (train_ratio + calibration_ratio + selection_ratio)),
        ),
        n - 1,
    )
    group_sets = (
        set(groups[:train_end]),
        set(groups[train_end:calibration_end]),
        set(groups[calibration_end:selection_end]),
        set(groups[selection_end:]),
    )
    return tuple(
        frame[frame[group_column].isin(group_set)].reset_index(drop=True)
        for group_set in group_sets
    )
