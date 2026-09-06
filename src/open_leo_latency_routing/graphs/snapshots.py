"""Temporal graph snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRAPH_SNAPSHOT_FEATURE_COLUMNS = [
    "peer_latency_mean",
    "peer_latency_std",
    "peer_latency_observed_count",
    "peer_latency_available",
    "state_peer_latency_mean",
    "state_peer_latency_std",
    "state_peer_latency_observed_count",
    "state_peer_latency_available",
    "target_peer_latency_mean",
    "target_peer_latency_std",
    "target_peer_latency_observed_count",
    "target_peer_latency_available",
    "peer_reply_mean",
    "peer_reply_std",
    "peer_reply_observed_count",
    "peer_reply_available",
    "peer_burst_indicator_mean",
    "peer_burst_indicator_std",
    "peer_burst_indicator_observed_count",
    "peer_burst_indicator_available",
    "peer_latency_gap",
    "location_degree",
    "target_degree",
    "snapshot_candidate_count",
]


def graph_context_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the graph expert's deliberately separate information view.

    The graph expert does not reuse temporal lag or rolling-history features.
    This makes expert divergence attributable to temporal-history versus
    neighborhood-context evidence instead of nested feature sets.
    """

    return [column for column in GRAPH_SNAPSHOT_FEATURE_COLUMNS if column in frame]


@dataclass
class GraphSnapshotSpec:
    """Configuration for graph snapshot creation."""

    snapshot_seconds: int = 60
    edge_rule: str = "co_observation"
    min_shared_events: int = 2


def add_graph_snapshot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach graph-derived peer features to each forecast row.

    Snapshot time is normalized by `session_bin_index` so sessions from different
    calendar dates can still contribute to the same decision stage.
    """

    output = frame.copy()
    snapshot_key = ["session_bin_index"]

    def excluding_self_stats(
        column: str,
        group_keys: list[str],
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return leave-one-row-out moments and their observability fields.

        Counts refer to peers with an observed value for ``column``. A row with
        no observed peer therefore receives missing moments instead of a copy
        of its own value. The graph model's normal zero imputation remains safe
        because the paired availability and count fields distinguish a genuine
        zero-valued peer statistic from an unavailable one.
        """

        values = pd.to_numeric(output[column], errors="coerce")
        groupers = [output[key] for key in group_keys]
        grouped = values.groupby(groupers, dropna=False)
        observed_count = grouped.transform("count").astype(float)
        total = grouped.transform("sum").astype(float)
        total_square = values.pow(2).groupby(
            groupers,
            dropna=False,
        ).transform("sum")
        own_observed = values.notna().astype(float)
        peer_count = (observed_count - own_observed).clip(lower=0.0)
        denominator = peer_count.where(peer_count > 0.0)
        own_value = values.fillna(0.0)
        peer_mean = (total - own_value) / denominator
        peer_second_moment = (
            total_square - own_value.pow(2)
        ) / denominator
        peer_variance = (
            peer_second_moment - peer_mean.pow(2)
        ).clip(lower=0.0)
        peer_std = np.sqrt(peer_variance).where(peer_count > 0.0)
        return (
            peer_mean,
            peer_std,
            peer_count.astype(int),
            peer_count.gt(0.0).astype(int),
        )

    (
        output["peer_latency_mean"],
        output["peer_latency_std"],
        output["peer_latency_observed_count"],
        output["peer_latency_available"],
    ) = excluding_self_stats("latency_mean_ms", snapshot_key)
    (
        output["state_peer_latency_mean"],
        output["state_peer_latency_std"],
        output["state_peer_latency_observed_count"],
        output["state_peer_latency_available"],
    ) = excluding_self_stats(
        "latency_mean_ms",
        ["session_bin_index", "path_state"],
    )
    (
        output["target_peer_latency_mean"],
        output["target_peer_latency_std"],
        output["target_peer_latency_observed_count"],
        output["target_peer_latency_available"],
    ) = excluding_self_stats(
        "latency_mean_ms",
        ["session_bin_index", "target_hint"],
    )
    (
        output["peer_reply_mean"],
        output["peer_reply_std"],
        output["peer_reply_observed_count"],
        output["peer_reply_available"],
    ) = excluding_self_stats("observed_replies", snapshot_key)
    (
        output["peer_burst_indicator_mean"],
        output["peer_burst_indicator_std"],
        output["peer_burst_indicator_observed_count"],
        output["peer_burst_indicator_available"],
    ) = excluding_self_stats("burst_indicator", snapshot_key)
    output["peer_latency_gap"] = (
        output["latency_mean_ms"] - output["peer_latency_mean"]
    )
    output["location_degree"] = output.groupby(
        ["session_bin_index", "location"], dropna=False
    )["target_hint"].transform("nunique")
    output["target_degree"] = output.groupby(
        ["session_bin_index", "target_hint"], dropna=False
    )["location"].transform("nunique")
    output["snapshot_candidate_count"] = output.groupby(
        "session_bin_index"
    )["relative_path"].transform("count")
    return output
