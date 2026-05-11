"""Bootstrap confidence intervals for per-window policy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapMetricInterval:
    """Stores one bootstrap confidence interval."""

    policy_name: str
    metric_name: str
    mean_value: float
    ci_lower: float
    ci_upper: float
    sample_count: int


def build_bootstrap_policy_intervals(
    decisions: pd.DataFrame,
    metric_columns: list[str],
    policy_column: str = "policy_name",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate percentile bootstrap confidence intervals per policy.

    The bootstrap is run across aligned decision windows for each policy. This
    keeps the uncertainty estimate tied to the unit of operational interest:
    one path-selection opportunity.
    """

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - ci
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    rows: list[BootstrapMetricInterval] = []
    for policy_name, policy_frame in decisions.groupby(policy_column, sort=True):
        sample_count = len(policy_frame)
        if sample_count == 0:
            continue

        for metric_name in metric_columns:
            values = policy_frame[metric_name].to_numpy(dtype=float)
            mean_value = float(values.mean())
            if sample_count == 1:
                ci_lower = mean_value
                ci_upper = mean_value
            else:
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = rng.choice(values, size=sample_count, replace=True)
                    bootstrap_means.append(sample.mean())
                ci_lower = float(np.percentile(bootstrap_means, lower_q))
                ci_upper = float(np.percentile(bootstrap_means, upper_q))

            rows.append(
                BootstrapMetricInterval(
                    policy_name=str(policy_name),
                    metric_name=metric_name,
                    mean_value=mean_value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    sample_count=sample_count,
                )
            )

    return pd.DataFrame(rows)
