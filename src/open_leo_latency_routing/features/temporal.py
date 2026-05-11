"""Temporal feature construction for forecasting baselines."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_lag_features(frame: pd.DataFrame, column: str, lags: Iterable[int]) -> pd.DataFrame:
    """Add groupwise lag features for a target column."""
    output = frame.copy()
    for lag in lags:
        output[f"{column}_lag_{lag}"] = output.groupby("relative_path")[column].shift(lag)
    return output


def add_rolling_features(frame: pd.DataFrame, column: str, windows: Iterable[int]) -> pd.DataFrame:
    """Add simple rolling mean and std features over each session."""
    output = frame.copy()
    grouped = output.groupby("relative_path")[column]
    for window in windows:
        rolling = grouped.rolling(window=window, min_periods=1)
        output[f"{column}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        output[f"{column}_roll_std_{window}"] = (
            rolling.std().fillna(0.0).reset_index(level=0, drop=True)
        )
    return output


def add_static_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert compact metadata fields into numeric columns."""
    output = frame.copy()
    output["path_state_flag"] = (output["path_state"] == "active").astype(int)
    output["window_duration_hours"] = (
        output["window_duration"].astype(str).str.replace("h", "", regex=False).astype(float)
    )
    output["probe_interval_ms"] = (
        output["probe_interval"].astype(str).str.replace("ms", "", regex=False).astype(float)
    )
    output["session_day_of_month"] = output["session_date"].dt.day.fillna(0).astype(int)
    return output


def build_forecast_table(
    time_bins: pd.DataFrame,
    target_column: str,
    lags: list[int],
    horizon_bins: int,
) -> pd.DataFrame:
    """Create a supervised forecasting table from session time bins."""
    frame = time_bins.copy()
    frame = frame.sort_values(["relative_path", "bin_epoch"]).reset_index(drop=True)
    frame = add_static_numeric_features(frame)
    frame = add_lag_features(frame, target_column, lags)
    frame = add_rolling_features(frame, target_column, windows=[3, 5])

    # Reply-count pressure and latency jump features are intentionally simple:
    # they expose burst-like behavior without requiring private telemetry.
    frame["observed_replies_roll_mean_3"] = (
        frame.groupby("relative_path")["observed_replies"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame["observed_replies_roll_std_3"] = (
        frame.groupby("relative_path")["observed_replies"]
        .rolling(window=3, min_periods=1)
        .std()
        .fillna(0.0)
        .reset_index(level=0, drop=True)
    )
    frame["observed_replies_lag_1"] = frame.groupby("relative_path")["observed_replies"].shift(1)
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

    # The target is the realized latency in the next available aggregate bin.
    frame["target_next"] = frame.groupby("relative_path")[target_column].shift(-horizon_bins)
    frame["target_next_bin_epoch"] = frame.groupby("relative_path")["bin_epoch"].shift(-horizon_bins)
    frame["target_available"] = frame["target_next"].notna().astype(int)
    numeric_columns = frame.select_dtypes(include=["number"]).columns
    frame[numeric_columns] = frame[numeric_columns].fillna(0.0)
    frame = frame[frame["target_available"] == 1].copy()
    return frame.reset_index(drop=True)


def split_train_val_test(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each session chronologically to avoid leakage across future bins."""
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train/val/test ratios must sum to 1.0")

    split_frames: list[pd.DataFrame] = []
    for _, group in frame.groupby("relative_path", sort=False):
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
        split_frames.append(ordered.drop(columns=["split_index"]))

    combined = pd.concat(split_frames, ignore_index=True)
    return (
        combined[combined["split"] == "train"].reset_index(drop=True),
        combined[combined["split"] == "val"].reset_index(drop=True),
        combined[combined["split"] == "test"].reset_index(drop=True),
    )
