"""Statistical comparison helpers for paired policy evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass
class PairedSignificanceResult:
    """Stores one paired statistical comparison."""

    comparison_name: str
    metric_name: str
    sample_count: int
    mean_delta: float
    median_delta: float
    ci_lower: float
    ci_upper: float
    statistic: float
    p_value: float
    holm_adjusted_p_value: float
    effect_size_dz: float
    win_rate: float
    loss_rate: float


def _bootstrap_delta_interval(
    deltas: pd.Series,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the paired mean delta."""

    values = deltas.to_numpy(dtype=float)
    if len(values) <= 1:
        value = float(values[0]) if len(values) == 1 else 0.0
        return value, value
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - ci
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(float(sample.mean()))
    return (
        float(np.percentile(bootstrap_means, lower_q)),
        float(np.percentile(bootstrap_means, upper_q)),
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm-Bonferroni correction to a vector of p-values."""

    if p_values.empty:
        return p_values
    order = p_values.sort_values().index.tolist()
    m = len(order)
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        candidate = min(1.0, float(p_values.loc[index]) * (m - rank + 1))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted


def build_paired_policy_significance(
    decisions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    metric_columns: list[str],
    group_column: str = "session_bin_index",
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run paired Wilcoxon tests on per-window policy outcomes.

    Each comparison aligns two policies on the same decision window so the test
    measures whether one policy consistently improves over the other.
    """

    rows: list[PairedSignificanceResult] = []
    for comparison_name, left_policy, right_policy in comparisons:
        left = decisions[decisions["policy_name"] == left_policy].copy()
        right = decisions[decisions["policy_name"] == right_policy].copy()
        paired = left.merge(
            right,
            on=group_column,
            suffixes=("_left", "_right"),
            how="inner",
        )
        if paired.empty:
            continue

        for metric_name in metric_columns:
            left_column = f"{metric_name}_left"
            right_column = f"{metric_name}_right"
            if left_column not in paired.columns or right_column not in paired.columns:
                continue
            deltas = (paired[left_column] - paired[right_column]).dropna()
            if deltas.empty:
                continue
            nonzero = deltas[deltas != 0]
            if nonzero.empty:
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(deltas)
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            ci_lower, ci_upper = _bootstrap_delta_interval(
                deltas,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            std_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
            effect_size_dz = float(deltas.mean() / std_delta) if std_delta > 1e-9 else 0.0
            win_rate = float((deltas < 0).mean())
            loss_rate = float((deltas > 0).mean())

            rows.append(
                PairedSignificanceResult(
                    comparison_name=comparison_name,
                    metric_name=metric_name,
                    sample_count=len(deltas),
                    mean_delta=float(deltas.mean()),
                    median_delta=float(deltas.median()),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    statistic=statistic,
                    p_value=p_value,
                    holm_adjusted_p_value=p_value,
                    effect_size_dz=effect_size_dz,
                    win_rate=win_rate,
                    loss_rate=loss_rate,
                )
            )

    result = pd.DataFrame([asdict(item) for item in rows])
    if result.empty:
        return result
    for metric_name, metric_frame in result.groupby("metric_name"):
        adjusted = _holm_adjust(metric_frame["p_value"])
        result.loc[metric_frame.index, "holm_adjusted_p_value"] = adjusted
    return result
