#!/usr/bin/env python3
"""Run temporal forecasting baselines on the processed LENS time-bin table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent, load_time_bin_table
from open_leo_latency_routing.features.temporal import build_forecast_table, split_train_val_test
from open_leo_latency_routing.models.forecast_baselines import default_feature_columns, run_forecast_baselines


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_display_path(path: Path) -> str:
    """Return repo-relative paths in public run metadata when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--time-bins", default=None)
    parser.add_argument("--metrics-out", default="results/temporal_forecasting/temporal_forecast_metrics.csv")
    parser.add_argument("--predictions-out", default="results/temporal_forecasting/temporal_forecast_predictions.csv")
    parser.add_argument("--metadata-out", default="results/temporal_forecasting/temporal_forecast_run_metadata.json")
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    time_bins_path = _resolve_repo_path(
        args.time_bins or config["dataset"].get("time_bins_path", "data/processed/ping_time_bins.csv")
    )
    time_bins = load_time_bin_table(time_bins_path)

    snapshot_seconds = int(config["graph"]["snapshot_seconds"])
    horizon_seconds = int(config["forecasting"]["horizon_seconds"])
    horizon_bins = max(1, math.ceil(horizon_seconds / snapshot_seconds))

    forecast_table = build_forecast_table(
        time_bins=time_bins,
        target_column=config["forecasting"]["target_column"],
        lags=list(config["forecasting"]["lag_steps"]),
        horizon_bins=horizon_bins,
    )
    train_frame, val_frame, test_frame = split_train_val_test(
        forecast_table,
        train_ratio=float(config["forecasting"]["train_ratio"]),
        val_ratio=float(config["forecasting"]["val_ratio"]),
        test_ratio=float(config["forecasting"]["test_ratio"]),
    )
    train_full = pd.concat([train_frame, val_frame], ignore_index=True)
    feature_columns = default_feature_columns(train_full)

    metrics, predictions = run_forecast_baselines(
        train_frame=train_full,
        test_frame=test_frame,
        feature_columns=feature_columns,
    )

    # These columns are carried into the saved prediction table so later stages
    # can evaluate not only forecast error, but also burst sensitivity and the
    # resulting decision quality.
    test_meta = test_frame.reset_index(drop=True).reset_index(names="row_id")[
        [
            "row_id",
            "relative_path",
            "location",
            "path_state",
            "session_bin_index",
            "bin_epoch",
            "bin_start_utc",
            "latency_mean_ms",
            "target_next",
            "burst_indicator",
            "reply_pressure_score",
        ]
    ]
    predictions = predictions.merge(test_meta, on="row_id", how="left")

    metrics_out = ensure_parent(_resolve_repo_path(args.metrics_out))
    predictions_out = ensure_parent(_resolve_repo_path(args.predictions_out))
    metadata_out = ensure_parent(_resolve_repo_path(args.metadata_out))

    metrics.to_csv(metrics_out, index=False)
    predictions.to_csv(predictions_out, index=False)
    metadata_out.write_text(
        json.dumps(
            {
                "time_bins_path": _repo_display_path(time_bins_path),
                "snapshot_seconds": snapshot_seconds,
                "horizon_seconds": horizon_seconds,
                "horizon_bins": horizon_bins,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "train_rows": len(train_frame),
                "val_rows": len(val_frame),
                "test_rows": len(test_frame),
                "session_count": int(forecast_table["relative_path"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"temporal_forecast_metrics_written={metrics_out}")
    print(f"temporal_forecast_predictions_written={predictions_out}")
    print(f"temporal_forecast_metadata_written={metadata_out}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
