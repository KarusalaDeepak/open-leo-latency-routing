"""Segment-safe bootstrap confidence intervals for policy metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from open_leo_latency_routing.evaluation.significance import (
    SEGMENTED_CIRCULAR_BLOCK_METHOD,
    _moving_block_means,
    _normalize_segment_columns,
    _validate_bootstrap_parameters,
    build_bootstrap_segment_ids,
)


@dataclass
class BootstrapMetricInterval:
    """Stores one bootstrap confidence interval and its segment audit."""

    policy_name: str
    metric_name: str
    mean_value: float
    ci_lower: float
    ci_upper: float
    sample_count: int
    bootstrap_method: str
    bootstrap_segment_columns: str
    bootstrap_segment_count: int


def build_bootstrap_policy_intervals(
    decisions: pd.DataFrame,
    metric_columns: list[str],
    policy_column: str = "policy_name",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
    block_length: int | None = None,
    segment_columns: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Estimate segment-safe percentile bootstrap intervals per policy.

    Callers must explicitly pass columns that identify every uninterrupted
    sequence. Fixed evaluations use ``continuity_segment_id``, which already
    resets at telemetry gaps and session/campaign changes; pooled rolling
    evaluations must additionally include ``rolling_fold``. Circular moving
    blocks are drawn separately within each segment while preserving its row
    count, so neither a block nor its wrap can cross a declared boundary.

    Missing/nonfinite metric rows split a segment rather than reconnecting the
    observations on either side. Passing no segment columns is rejected.
    Legacy single-series behavior is available only by passing an explicit
    constant segment column after independently validating that the rows form
    one uninterrupted sequence.
    """

    columns = _normalize_segment_columns(segment_columns)
    if policy_column not in decisions.columns:
        raise KeyError(f"policy column is missing: {policy_column}")
    missing = [column for column in columns if column not in decisions.columns]
    if missing:
        raise KeyError(f"bootstrap segment columns are missing: {missing}")
    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be strictly between zero and one")
    _validate_bootstrap_parameters(
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )

    alpha = 1.0 - ci
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    segment_label = "|".join(columns)

    rows: list[BootstrapMetricInterval] = []
    for policy_name, policy_frame in decisions.groupby(
        policy_column,
        sort=True,
    ):
        if policy_frame.empty:
            continue
        # Validate the unfiltered segmentation even if a requested metric is
        # absent or entirely missing.
        build_bootstrap_segment_ids(policy_frame, columns)

        for metric_name in metric_columns:
            if metric_name not in policy_frame.columns:
                continue
            numeric = pd.to_numeric(policy_frame[metric_name], errors="coerce")
            valid_mask = np.isfinite(numeric.to_numpy(dtype=float))
            if not valid_mask.any():
                continue
            values = numeric.to_numpy(dtype=float)[valid_mask]
            segment_ids = build_bootstrap_segment_ids(
                policy_frame,
                columns,
                valid_mask=valid_mask,
            )
            mean_value = float(values.mean())
            if len(values) == 1:
                ci_lower = mean_value
                ci_upper = mean_value
            else:
                bootstrap_means = _moving_block_means(
                    values,
                    segment_ids=segment_ids,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                    block_length=block_length,
                )
                ci_lower = float(np.percentile(bootstrap_means, lower_q))
                ci_upper = float(np.percentile(bootstrap_means, upper_q))

            rows.append(
                BootstrapMetricInterval(
                    policy_name=str(policy_name),
                    metric_name=metric_name,
                    mean_value=mean_value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    sample_count=len(values),
                    bootstrap_method=SEGMENTED_CIRCULAR_BLOCK_METHOD,
                    bootstrap_segment_columns=segment_label,
                    bootstrap_segment_count=int(len(np.unique(segment_ids))),
                )
            )

    return pd.DataFrame([asdict(item) for item in rows])
