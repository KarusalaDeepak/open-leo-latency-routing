"""Evaluation metrics shared across forecasting, graph learning, and optimization."""

from __future__ import annotations

import math
from typing import Iterable


def _to_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = _to_list(y_true)
    pred_values = _to_list(y_pred)
    values = [abs(a - b) for a, b in zip(true_values, pred_values)]
    return sum(values) / len(values) if values else math.nan


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = _to_list(y_true)
    pred_values = _to_list(y_pred)
    squared = [(a - b) ** 2 for a, b in zip(true_values, pred_values)]
    return math.sqrt(sum(squared) / len(squared)) if squared else math.nan


def mean_absolute_percentage_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = _to_list(y_true)
    pred_values = _to_list(y_pred)
    ratios = [
        abs((a - b) / a)
        for a, b in zip(true_values, pred_values)
        if abs(a) > 1e-9
    ]
    return sum(ratios) / len(ratios) if ratios else math.nan
