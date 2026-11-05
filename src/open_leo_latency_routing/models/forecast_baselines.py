"""Temporal forecasting models for the manuscript pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

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


@dataclass(frozen=True)
class RidgeSelectionResult:
    """Training-internal development result for the ridge penalty."""

    selected_alpha: float
    development_rmse: float
    fit_rows: int
    development_rows: int
    candidate_scores: tuple[tuple[float, float], ...]


def build_forecast_model(
    model_name: str,
    *,
    ridge_alpha: float = 1.0,
):
    """Create one configured temporal forecasting model by name."""
    if model_name == "linear_regression":
        return LinearRegression()
    if model_name == "ridge_regression":
        ridge_alpha = float(ridge_alpha)
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha, random_state=42)),
            ]
        )
    if model_name == "decision_tree_regressor":
        return DecisionTreeRegressor(max_depth=5, min_samples_leaf=8, random_state=42)
    if model_name == "small_mlp_regressor":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(24,),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=5e-3,
                        early_stopping=True,
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"unsupported forecast model: {model_name}")


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the portable temporal feature view used by all datasets.

    Raw dataset-specific fields such as TTL, absolute epoch, and ICMP sequence
    are deliberately excluded. This keeps transfer tests from silently relying
    on columns that exist only in LENS.
    """

    exact_features = {
        "latency_mean_ms",
        "latency_std_ms",
        "latency_max_ms",
        "observed_replies",
        "path_state_flag",
        "window_duration_hours",
        "probe_interval_ms",
        "session_day_of_month",
        "observed_replies_roll_mean_3",
        "observed_replies_roll_std_3",
        "observed_replies_roll_observed_count_3",
        "observed_replies_roll_coverage_3",
        "observed_replies_roll_complete_3",
        "observed_replies_lag_1",
        "observed_replies_lag_1_available",
        "history_lag_coverage_ratio",
        "latency_delta_1",
        "latency_delta_roll3",
        "latency_jump_ratio",
        "latency_volatility_ratio",
        "reply_delta_1",
        "reply_gap_roll3",
        "reply_pressure_score",
        "burst_indicator",
    }
    allowed_prefixes = (
        "latency_mean_ms_lag_",
        "latency_mean_ms_roll_mean_",
        "latency_mean_ms_roll_std_",
        "latency_mean_ms_roll_observed_count_",
        "latency_mean_ms_roll_coverage_",
        "latency_mean_ms_roll_complete_",
    )
    return [
        column
        for column in frame.columns
        if (
            (column in exact_features or column.startswith(allowed_prefixes))
            and not column.startswith("target_")
            and pd.api.types.is_numeric_dtype(frame[column])
        )
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
    *,
    ridge_alpha: float = 1.0,
):
    """Fit and return one reusable forecasting model."""
    model = build_forecast_model(model_name, ridge_alpha=ridge_alpha)
    train_x = train_frame[feature_columns].fillna(0.0)
    train_y = train_frame["target_next"]
    model.fit(train_x, train_y)
    return model


def select_ridge_alpha(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    development_fraction: float = 0.20,
) -> RidgeSelectionResult:
    """Select ridge regularization inside the training partition only.

    The latest chronological epochs form an internal development block.  The
    returned value may then be frozen and used to refit on the complete outer
    training block.  No calibration, policy-selection, or test outcome enters
    this choice.
    """

    if not feature_columns:
        raise ValueError("ridge selection requires at least one feature")
    if not 0.0 < development_fraction < 0.5:
        raise ValueError("development_fraction must lie in (0, 0.5)")
    candidates = tuple(float(value) for value in alpha_grid)
    if not candidates or any(
        not np.isfinite(value) or value < 0.0 for value in candidates
    ):
        raise ValueError("ridge alpha grid must be finite, non-negative, and non-empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("ridge alpha grid must not contain duplicates")

    order_column = next(
        (
            name
            for name in ("bin_epoch", "target_next_bin_epoch", "session_bin_index")
            if name in training_frame
        ),
        None,
    )
    if order_column is None:
        raise KeyError(
            "training-internal ridge selection requires bin_epoch, "
            "target_next_bin_epoch, or session_bin_index"
        )
    ordered = training_frame.sort_values(order_column, kind="stable").reset_index(drop=True)
    epochs = ordered[order_column].drop_duplicates().to_numpy()
    if len(epochs) < 2:
        raise ValueError("ridge selection requires at least two chronological epochs")
    development_epoch_count = max(1, int(np.ceil(len(epochs) * development_fraction)))
    development_epochs = set(epochs[-development_epoch_count:].tolist())
    development = ordered[ordered[order_column].isin(development_epochs)]
    fit = ordered[~ordered[order_column].isin(development_epochs)]
    if fit.empty or development.empty:
        raise ValueError("training-internal ridge split produced an empty block")

    scores: list[tuple[float, float]] = []
    for alpha in sorted(candidates):
        model = fit_forecast_model(
            "ridge_regression",
            fit,
            feature_columns,
            ridge_alpha=alpha,
        )
        prediction = np.asarray(
            model.predict(development[feature_columns].fillna(0.0)),
            dtype=float,
        )
        truth = development["target_next"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(np.square(truth - prediction))))
        scores.append((alpha, rmse))
    selected_alpha, selected_rmse = min(scores, key=lambda item: (item[1], item[0]))
    return RidgeSelectionResult(
        selected_alpha=selected_alpha,
        development_rmse=selected_rmse,
        fit_rows=len(fit),
        development_rows=len(development),
        candidate_scores=tuple(scores),
    )


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
    """Fit the temporal baselines used in the final manuscript comparison."""
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

    # The manuscript version keeps one simple predictive baseline and one
    # non-learning persistence reference so the comparison stays concise.
    for model_name in (
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "small_mlp_regressor",
    ):
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
