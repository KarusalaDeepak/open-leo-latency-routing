"""Statistical comparison helpers for paired policy evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    statistic: float
    p_value: float


def build_paired_policy_significance(
    decisions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    metric_columns: list[str],
    group_column: str = "session_bin_index",
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
            deltas = paired[f"{metric_name}_left"] - paired[f"{metric_name}_right"]
            nonzero = deltas[deltas != 0]
            if nonzero.empty:
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(deltas)
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            rows.append(
                PairedSignificanceResult(
                    comparison_name=comparison_name,
                    metric_name=metric_name,
                    sample_count=len(deltas),
                    mean_delta=float(deltas.mean()),
                    median_delta=float(deltas.median()),
                    statistic=statistic,
                    p_value=p_value,
                )
            )

    return pd.DataFrame([asdict(item) for item in rows])
