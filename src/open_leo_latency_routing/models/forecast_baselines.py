"""Temporal forecasting models for the conference pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression

from open_leo_latency_routing.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


@dataclass
class ForecastResult:
    """Compact forecasting metrics for one model."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    row_count: int


def build_forecast_model(model_name: str):
    """Create one configured temporal forecasting model by name."""
    if model_name == "linear_regression":
        return LinearRegression()
    raise ValueError(f"unsupported forecast model: {model_name}")


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the numeric feature columns used by the forecasting baselines."""
    excluded = {
        "target_next",
        "target_next_bin_epoch",
        "target_available",
        "split",
        "relative_path",
        "measurement_family",
        "path_state",
        "location",
        "session_date",
        "target_hint",
        "probe_interval",
        "window_duration",
        "window_start",
        "header_target",
        "header_target_ip",
        "header_source_ip",
        "interface",
        "bin_start_utc",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _fit_predict_regressor(
    model,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> list[float]:
    train_x = train_frame[feature_columns].fillna(0.0)
    test_x = test_frame[feature_columns].fillna(0.0)
    train_y = train_frame["target_next"]
    model.fit(train_x, train_y)
    return model.predict(test_x).tolist()


def fit_forecast_model(
    model_name: str,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
):
    """Fit and return one reusable forecasting model."""
    model = build_forecast_model(model_name)
    train_x = train_frame[feature_columns].fillna(0.0)
    train_y = train_frame["target_next"]
    model.fit(train_x, train_y)
    return model


def predict_forecast_model(
    model_name: str,
    model,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate a standardized predictions table for one forecast model."""
    test_x = test_frame[feature_columns].fillna(0.0)
    pred = model.predict(test_x).tolist()
    return pd.DataFrame(
        {
            "row_id": test_frame.index,
            "model_name": model_name,
            "y_true": test_frame["target_next"].tolist(),
            "y_pred": pred,
        }
    )


def evaluate_prediction_frame(predictions: pd.DataFrame) -> ForecastResult:
    """Convert a standardized prediction frame into scalar metrics."""
    return ForecastResult(
        model_name=str(predictions["model_name"].iloc[0]),
        mae=mean_absolute_error(predictions["y_true"], predictions["y_pred"]),
        rmse=root_mean_squared_error(predictions["y_true"], predictions["y_pred"]),
        mape=mean_absolute_percentage_error(predictions["y_true"], predictions["y_pred"]),
        row_count=len(predictions),
    )


def run_forecast_baselines(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the temporal baselines used in the final conference comparison."""
    predictions: list[pd.DataFrame] = []
    metrics: list[ForecastResult] = []

    persistence_pred = test_frame["latency_mean_ms"].tolist()
    predictions.append(
        pd.DataFrame(
            {
                "row_id": test_frame.index,
                "model_name": "persistence",
                "y_true": test_frame["target_next"].tolist(),
                "y_pred": persistence_pred,
            }
        )
    )
    metrics.append(
        ForecastResult(
            model_name="persistence",
            mae=mean_absolute_error(test_frame["target_next"], persistence_pred),
            rmse=root_mean_squared_error(test_frame["target_next"], persistence_pred),
            mape=mean_absolute_percentage_error(test_frame["target_next"], persistence_pred),
            row_count=len(test_frame),
        )
    )

    # The conference version keeps one simple predictive baseline and one
    # non-learning persistence reference so the comparison stays concise.
    for model_name in ("linear_regression",):
        model = fit_forecast_model(
            model_name=model_name,
            train_frame=train_frame,
            feature_columns=feature_columns,
        )
        prediction_frame = predict_forecast_model(
            model_name=model_name,
            model=model,
            test_frame=test_frame,
            feature_columns=feature_columns,
        )
        predictions.append(prediction_frame)
        metrics.append(evaluate_prediction_frame(prediction_frame))

    return pd.DataFrame([asdict(item) for item in metrics]), pd.concat(predictions, ignore_index=True)
