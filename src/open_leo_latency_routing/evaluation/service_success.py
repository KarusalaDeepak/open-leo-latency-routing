"""Conditional and scheduled-epoch service-success accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_scheduled_epoch_success(
    decisions: pd.DataFrame,
    *,
    scheduled_epoch_count: int,
    latency_budget_ms: float,
    policy_column: str = "policy_name",
    latency_column: str = "realized_next_latency_ms",
) -> pd.DataFrame:
    """Report complete-case success and conservative scheduled-epoch bounds.

    The lower endpoint treats every scheduled epoch without an evaluable
    policy outcome as service failure. The upper endpoint treats every such
    epoch as success. This does not assume that missingness is random and keeps
    conditional path-selection performance separate from operational service
    availability.
    """

    if isinstance(scheduled_epoch_count, bool) or scheduled_epoch_count < 0:
        raise ValueError("scheduled_epoch_count must be a non-negative integer")
    if int(scheduled_epoch_count) != scheduled_epoch_count:
        raise ValueError("scheduled_epoch_count must be an integer")
    if not np.isfinite(latency_budget_ms) or latency_budget_ms < 0:
        raise ValueError("latency_budget_ms must be finite and non-negative")
    missing = {policy_column, latency_column}.difference(decisions.columns)
    if missing:
        raise KeyError(f"scheduled success is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for policy, frame in decisions.groupby(policy_column, sort=False):
        latency = pd.to_numeric(frame[latency_column], errors="coerce")
        evaluable = latency[np.isfinite(latency.to_numpy(dtype=float))]
        decision_count = int(len(evaluable))
        if decision_count > int(scheduled_epoch_count):
            raise ValueError(
                f"policy {policy} has more evaluable decisions than scheduled epochs"
            )
        success_count = int(evaluable.le(float(latency_budget_ms)).sum())
        unknown_count = int(scheduled_epoch_count) - decision_count
        conditional_rate = (
            success_count / decision_count if decision_count else np.nan
        )
        lower = (
            success_count / int(scheduled_epoch_count)
            if scheduled_epoch_count
            else np.nan
        )
        upper = (
            (success_count + unknown_count) / int(scheduled_epoch_count)
            if scheduled_epoch_count
            else np.nan
        )
        rows.append(
            {
                "policy_name": policy,
                "latency_budget_ms": float(latency_budget_ms),
                "scheduled_epoch_count": int(scheduled_epoch_count),
                "evaluable_decision_count": decision_count,
                "success_count": success_count,
                "unknown_or_no_action_count": unknown_count,
                "conditional_success_rate": conditional_rate,
                "scheduled_epoch_success_lower": lower,
                "scheduled_epoch_success_upper": upper,
                "lower_endpoint_semantics": (
                    "all unknown_or_no_action scheduled epochs counted as failures"
                ),
                "upper_endpoint_semantics": (
                    "all unknown_or_no_action scheduled epochs counted as successes"
                ),
            }
        )
    return pd.DataFrame(rows)
