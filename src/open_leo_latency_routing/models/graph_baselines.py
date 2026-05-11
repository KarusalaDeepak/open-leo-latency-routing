"""Graph-aware forecasting models built on snapshot features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from xgboost import XGBRegressor

from open_leo_latency_routing.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


@dataclass
class GraphResult:
    """Summary metrics for a graph-based model run."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    row_count: int


def build_graph_xgb_model() -> XGBRegressor:
    """Create the graph-aware regressor used in the final manuscript pipeline."""
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def fit_graph_xgb_model(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
):
    """Fit and return the boosted graph-aware regressor."""
    model = build_graph_xgb_model()
    train_x = train_frame[feature_columns].fillna(0.0)
    train_y = train_frame["target_next"]
    model.fit(train_x, train_y)
    return model


def predict_graph_model(
    model,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str = "graph_xgboost",
) -> pd.DataFrame:
    """Generate standardized graph-model predictions."""
    test_x = test_frame[feature_columns].fillna(0.0)
    pred = model.predict(test_x)
    return pd.DataFrame(
        {
            "row_id": test_frame.index,
            "model_name": model_name,
            "y_true": test_frame["target_next"].tolist(),
            "y_pred": pred.tolist(),
        }
    )


def evaluate_graph_predictions(predictions: pd.DataFrame) -> GraphResult:
    """Convert graph predictions into scalar metrics."""
    return GraphResult(
        model_name=str(predictions["model_name"].iloc[0]),
        mae=mean_absolute_error(predictions["y_true"], predictions["y_pred"]),
        rmse=root_mean_squared_error(predictions["y_true"], predictions["y_pred"]),
        mape=mean_absolute_percentage_error(predictions["y_true"], predictions["y_pred"]),
        row_count=len(predictions),
    )


def run_graph_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the final graph-aware model and return metrics plus predictions."""
    metric_rows: list[GraphResult] = []
    prediction_frames: list[pd.DataFrame] = []

    xgb_model = fit_graph_xgb_model(train_frame=train_frame, feature_columns=feature_columns)
    xgb_predictions = predict_graph_model(
        model=xgb_model,
        test_frame=test_frame,
        feature_columns=feature_columns,
        model_name="graph_xgboost",
    )
    metric_rows.append(evaluate_graph_predictions(xgb_predictions))
    prediction_frames.append(xgb_predictions)

    return pd.DataFrame([asdict(item) for item in metric_rows]), pd.concat(prediction_frames, ignore_index=True)
