"""Temporal graph snapshot helpers."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

GRAPH_SNAPSHOT_FEATURE_COLUMNS = [
    "peer_latency_mean",
    "peer_latency_std",
    "state_peer_latency_mean",
    "target_peer_latency_mean",
    "peer_reply_mean",
    "peer_reply_std",
    "peer_burst_indicator_mean",
    "peer_burst_indicator_std",
    "peer_latency_gap",
    "state_peer_latency_std",
    "target_peer_latency_std",
    "location_degree",
    "target_degree",
    "snapshot_candidate_count",
]


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
    for column in GRAPH_SNAPSHOT_FEATURE_COLUMNS:
        output[column] = 0.0

    for snapshot_index, snapshot_frame in output.groupby("session_bin_index"):
        # The graph connects locations to targets observed in the same
        # normalized decision window, giving each candidate peer context without
        # looking into future latency values.
        graph = nx.Graph()
        for row in snapshot_frame.itertuples():
            location_node = f"loc::{row.location}"
            target_node = f"tgt::{row.target_hint}"
            graph.add_edge(location_node, target_node)

        snapshot_size = len(snapshot_frame)
        for row in snapshot_frame.itertuples():
            row_mask = output.index == row.Index
            peer_frame = snapshot_frame.drop(index=row.Index, errors="ignore")
            state_peer_frame = peer_frame[peer_frame["path_state"] == row.path_state]
            target_peer_frame = peer_frame[peer_frame["target_hint"] == row.target_hint]

            output.loc[row_mask, "peer_latency_mean"] = (
                peer_frame["latency_mean_ms"].mean() if not peer_frame.empty else row.latency_mean_ms
            )
            output.loc[row_mask, "peer_latency_std"] = (
                peer_frame["latency_mean_ms"].std(ddof=0) if len(peer_frame) > 1 else 0.0
            )
            output.loc[row_mask, "state_peer_latency_mean"] = (
                state_peer_frame["latency_mean_ms"].mean()
                if not state_peer_frame.empty
                else row.latency_mean_ms
            )
            output.loc[row_mask, "target_peer_latency_mean"] = (
                target_peer_frame["latency_mean_ms"].mean()
                if not target_peer_frame.empty
                else row.latency_mean_ms
            )
            output.loc[row_mask, "peer_reply_mean"] = (
                peer_frame["observed_replies"].mean() if not peer_frame.empty else row.observed_replies
            )
            output.loc[row_mask, "peer_reply_std"] = (
                peer_frame["observed_replies"].std(ddof=0) if len(peer_frame) > 1 else 0.0
            )
            output.loc[row_mask, "peer_burst_indicator_mean"] = (
                peer_frame["burst_indicator"].mean() if not peer_frame.empty else row.burst_indicator
            )
            output.loc[row_mask, "peer_burst_indicator_std"] = (
                peer_frame["burst_indicator"].std(ddof=0) if len(peer_frame) > 1 else 0.0
            )
            output.loc[row_mask, "peer_latency_gap"] = row.latency_mean_ms - (
                peer_frame["latency_mean_ms"].mean() if not peer_frame.empty else row.latency_mean_ms
            )
            output.loc[row_mask, "state_peer_latency_std"] = (
                state_peer_frame["latency_mean_ms"].std(ddof=0) if len(state_peer_frame) > 1 else 0.0
            )
            output.loc[row_mask, "target_peer_latency_std"] = (
                target_peer_frame["latency_mean_ms"].std(ddof=0) if len(target_peer_frame) > 1 else 0.0
            )
            output.loc[row_mask, "location_degree"] = graph.degree(f"loc::{row.location}")
            output.loc[row_mask, "target_degree"] = graph.degree(f"tgt::{row.target_hint}")
            output.loc[row_mask, "snapshot_candidate_count"] = snapshot_size

    return output
