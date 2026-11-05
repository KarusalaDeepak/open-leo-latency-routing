"""Trace-replayed execution after controller state has aged."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_decision_cadence_seconds(
    candidates: pd.DataFrame,
    decision_cadence_seconds: float | None,
) -> float:
    """Resolve a declared cadence without estimating it from observed gaps."""

    declared_columns = (
        "target_expected_cadence_seconds",
        "bin_seconds",
    )
    declared_values: list[float] = []
    for column in declared_columns:
        if column not in candidates:
            continue
        values = pd.to_numeric(candidates[column], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise ValueError(f"delayed replay requires one stable {column} value")
        declared_values.append(float(values[0]))
    if decision_cadence_seconds is None:
        if not declared_values:
            raise ValueError(
                "delayed replay requires an explicit decision cadence or a "
                "stable source cadence column"
            )
        decision_cadence_seconds = declared_values[0]
    decision_cadence_seconds = float(decision_cadence_seconds)
    if not np.isfinite(decision_cadence_seconds) or decision_cadence_seconds <= 0:
        raise ValueError("decision cadence must be finite and positive")
    if any(
        not np.isclose(value, decision_cadence_seconds, rtol=0.0, atol=1.0e-9)
        for value in declared_values
    ):
        raise ValueError("explicit delayed-replay cadence conflicts with source data")
    return decision_cadence_seconds


def replay_delayed_execution(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    latency_budget_ms: float = 60.0,
    delay_bins: tuple[int, ...] = (0, 1, 2, 3),
    *,
    decision_cadence_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay a frozen decision against later observed candidate state.

    Delay zero reproduces the original next-bin target. For delay ``k``, the
    path selected at wall-clock epoch ``t`` is evaluated only at an exact
    ``t + k * decision_cadence_seconds`` row. A later row after a trace gap is
    not reinterpreted as a one-bin delay. Availability, path state, handover,
    queue, and incident effects already present in that exact later row are
    therefore respected. The
    network-only QoS metric excludes waiting time so stale-ranking degradation
    is visible; the end-to-end metric additionally includes controller waiting
    and will correctly fail when that waiting alone exceeds the QoS budget.
    """

    candidate_required = {
        "session_bin_index",
        "bin_epoch",
        "relative_path",
        "target_next",
    }
    missing = candidate_required.difference(candidates.columns)
    if missing:
        raise KeyError(f"delayed replay is missing candidate columns: {sorted(missing)}")
    decision_required = {
        "session_bin_index",
        "decision_bin_epoch",
        "policy_name",
        "chosen_relative_path",
    }
    missing = decision_required.difference(decisions.columns)
    if missing:
        raise KeyError(f"delayed replay is missing decision columns: {sorted(missing)}")
    candidate_epochs = pd.to_numeric(candidates["bin_epoch"], errors="coerce")
    decision_epochs = pd.to_numeric(
        decisions["decision_bin_epoch"], errors="coerce"
    )
    if (
        candidate_epochs.isna().any()
        or decision_epochs.isna().any()
        or not np.isfinite(candidate_epochs.to_numpy(dtype=float)).all()
        or not np.isfinite(decision_epochs.to_numpy(dtype=float)).all()
    ):
        raise ValueError("delayed replay requires finite wall-clock epochs")
    candidates = candidates.copy()
    candidates["bin_epoch"] = candidate_epochs.astype(float)
    decisions = decisions.copy()
    decisions["decision_bin_epoch"] = decision_epochs.astype(float)
    if any(int(value) < 0 for value in delay_bins):
        raise ValueError("delay bins must be non-negative")

    bin_seconds = _resolve_decision_cadence_seconds(
        candidates,
        decision_cadence_seconds,
    )
    if candidates.duplicated(["bin_epoch", "relative_path"]).any():
        raise ValueError(
            "delayed replay requires unique wall-clock epochs within each path"
        )
    lookup = candidates.drop_duplicates(
        ["bin_epoch", "relative_path"], keep="last"
    ).set_index(["bin_epoch", "relative_path"])
    detail_rows: list[dict[str, object]] = []
    for delay in delay_bins:
        delay = int(delay)
        wait_ms = delay * bin_seconds * 1000.0
        for row in decisions.itertuples(index=False):
            decision_epoch = int(row.session_bin_index)
            decision_bin_epoch = float(row.decision_bin_epoch)
            path = str(row.chosen_relative_path)
            expected_replay_bin_epoch = decision_bin_epoch + delay * bin_seconds
            key = (expected_replay_bin_epoch, path)
            matched = key in lookup.index
            replay = lookup.loc[key] if matched else None
            if isinstance(replay, pd.DataFrame):
                replay = replay.iloc[0]
            endpoint_observed = bool(matched)
            available = endpoint_observed
            if available and "is_feasible_path" in lookup.columns:
                available = bool(replay["is_feasible_path"])
            replay_epoch = (
                int(replay["session_bin_index"])
                if matched and pd.notna(replay["session_bin_index"])
                else np.nan
            )
            network_latency = (
                float(replay["target_next"])
                if available and pd.notna(replay["target_next"])
                else np.nan
            )
            network_outcome_evaluable = bool(
                endpoint_observed
                and (not available or np.isfinite(network_latency))
            )
            network_success = (
                int(
                    available
                    and np.isfinite(network_latency)
                    and network_latency <= latency_budget_ms
                )
                if network_outcome_evaluable
                else np.nan
            )
            end_to_end_latency = (
                network_latency + wait_ms if np.isfinite(network_latency) else np.nan
            )
            end_to_end_success = (
                int(
                    bool(network_success)
                    and np.isfinite(end_to_end_latency)
                    and end_to_end_latency <= latency_budget_ms
                )
                if network_outcome_evaluable
                else np.nan
            )
            detail_rows.append(
                {
                    "policy_name": row.policy_name,
                    "decision_epoch": decision_epoch,
                    "replay_epoch": replay_epoch,
                    "decision_bin_epoch": decision_bin_epoch,
                    "expected_replay_bin_epoch": expected_replay_bin_epoch,
                    "replay_bin_epoch": (
                        float(replay.name[0]) if matched else np.nan
                    ),
                    "chosen_relative_path": path,
                    "delay_bins": delay,
                    "delay_ms": wait_ms,
                    "trace_endpoint_observed": int(endpoint_observed),
                    "trace_row_matched": int(matched),
                    "execution_available": int(available),
                    "endpoint_observation_status": (
                        "unobserved_acquisition_endpoint"
                        if not endpoint_observed
                        else (
                            "observed_feasible"
                            if available
                            else "observed_infeasible"
                        )
                    ),
                    "network_outcome_evaluable": int(network_outcome_evaluable),
                    "replayed_network_latency_ms": network_latency,
                    "replayed_end_to_end_latency_ms": end_to_end_latency,
                    "network_qos_success": network_success,
                    "end_to_end_qos_success": end_to_end_success,
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, object]] = []
    for keys, frame in detail.groupby(["policy_name", "delay_bins"], sort=False):
        available_latency = frame["replayed_network_latency_ms"].dropna()
        observed_endpoints = frame[frame["trace_endpoint_observed"].astype(bool)]
        summary_rows.append(
            {
                "policy_name": keys[0],
                "delay_bins": int(keys[1]),
                "delay_ms": float(frame["delay_ms"].iloc[0]),
                "decision_cadence_seconds": bin_seconds,
                "replay_lookup": "exact_wall_clock_epoch",
                "decision_count": int(len(frame)),
                "trace_endpoint_observability_rate": float(
                    frame["trace_endpoint_observed"].mean()
                ),
                "trace_row_match_rate": float(frame["trace_row_matched"].mean()),
                "execution_availability_rate": float(frame["execution_available"].mean()),
                "execution_availability_definition": (
                    "exact_endpoint_observed_and_path_feasible"
                ),
                "execution_feasibility_rate_when_observed": (
                    float(observed_endpoints["execution_available"].mean())
                    if len(observed_endpoints)
                    else np.nan
                ),
                "network_qos_evaluable_count": int(
                    frame["network_outcome_evaluable"].sum()
                ),
                "network_qos_evaluable_rate": float(
                    frame["network_outcome_evaluable"].mean()
                ),
                "network_qos_success_rate": float(frame["network_qos_success"].mean()),
                "network_qos_success_rate_definition": (
                    "conditional_on_observed_evaluable_endpoint"
                ),
                "end_to_end_qos_success_rate": float(
                    frame["end_to_end_qos_success"].mean()
                ),
                "end_to_end_qos_success_rate_definition": (
                    "conditional_on_observed_evaluable_endpoint"
                ),
                "mean_network_latency_when_available_ms": (
                    float(available_latency.mean()) if len(available_latency) else np.nan
                ),
                "p95_network_latency_when_available_ms": (
                    float(available_latency.quantile(0.95))
                    if len(available_latency)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summary_rows), detail
