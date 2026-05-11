"""Decision policies for the final manuscript evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import pandas as pd


@dataclass
class ConsensusPolicyConfig:
    """Configuration for the consensus-regularized hybrid policy."""

    temporal_weight: float = 0.65
    graph_weight: float = 0.35
    disagreement_penalty: float = 0.30
    output_column: str = "pred_consensus"


@dataclass
class SimpleFusionPolicyConfig:
    """Configuration for the unregularized weighted fusion baseline."""

    temporal_weight: float = 0.65
    graph_weight: float = 0.35
    output_column: str = "pred_simple_fusion"


@dataclass
class ConsensusPolicyTuningResult:
    """Validation-selected hyperparameters for the hybrid policy."""

    temporal_weight: float
    graph_weight: float
    disagreement_penalty: float
    validation_gap_ms: float


@dataclass
class PolicyDecision:
    """Container for one policy summary."""

    policy_name: str
    decision_count: int
    mean_realized_latency_ms: float
    mean_regret_ms: float
    best_path_match_rate: float
    success_rate_under_60ms: float
    mean_decision_time_us: float


def add_consensus_hybrid_scores(
    candidate_frame: pd.DataFrame,
    temporal_column: str = "pred_forecast",
    graph_column: str = "pred_graph",
    config: ConsensusPolicyConfig | None = None,
) -> pd.DataFrame:
    """Add the consensus-regularized hybrid score to a candidate table.

    The score blends the temporal and graph forecasts, then penalizes
    disagreement between the two experts as a lightweight uncertainty proxy.
    """

    policy_config = config or ConsensusPolicyConfig()
    output = candidate_frame.copy()
    disagreement = (output[graph_column] - output[temporal_column]).abs()
    output[policy_config.output_column] = (
        policy_config.temporal_weight * output[temporal_column]
        + policy_config.graph_weight * output[graph_column]
        + policy_config.disagreement_penalty * disagreement
    )
    return output


def add_simple_fusion_scores(
    candidate_frame: pd.DataFrame,
    temporal_column: str = "pred_forecast",
    graph_column: str = "pred_graph",
    config: SimpleFusionPolicyConfig | None = None,
) -> pd.DataFrame:
    """Add the weighted fusion baseline without disagreement regularization."""

    policy_config = config or SimpleFusionPolicyConfig()
    output = candidate_frame.copy()
    output[policy_config.output_column] = (
        policy_config.temporal_weight * output[temporal_column]
        + policy_config.graph_weight * output[graph_column]
    )
    return output


def tune_consensus_policy(
    validation_candidate_frames: dict[str, pd.DataFrame],
    temporal_column: str = "pred_forecast",
    graph_column: str = "pred_graph",
    temporal_weight_grid: list[float] | None = None,
    disagreement_penalty_grid: list[float] | None = None,
) -> ConsensusPolicyTuningResult:
    """Select hybrid weights on validation scenarios without touching test data.

    The objective minimizes the mean latency gap to the best single-expert
    baseline (temporal or graph) averaged across the supplied validation
    scenarios.
    """

    temporal_weights = temporal_weight_grid or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    disagreement_penalties = disagreement_penalty_grid or [0.00, 0.10, 0.20, 0.25, 0.30]
    best_result: ConsensusPolicyTuningResult | None = None

    for temporal_weight in temporal_weights:
        graph_weight = 1.0 - temporal_weight
        if graph_weight < 0.0:
            continue
        for disagreement_penalty in disagreement_penalties:
            validation_gap_ms = 0.0
            for frame in validation_candidate_frames.values():
                baseline_summary, _ = evaluate_decision_policies(
                    frame,
                    policy_columns={
                        "temporal_only": temporal_column,
                        "graph_only": graph_column,
                    },
                )
                best_single_latency = float(baseline_summary["mean_realized_latency_ms"].min())
                hybrid_frame = add_consensus_hybrid_scores(
                    frame,
                    temporal_column=temporal_column,
                    graph_column=graph_column,
                    config=ConsensusPolicyConfig(
                        temporal_weight=temporal_weight,
                        graph_weight=graph_weight,
                        disagreement_penalty=disagreement_penalty,
                        output_column="pred_consensus",
                    ),
                )
                hybrid_summary, _ = evaluate_decision_policies(
                    hybrid_frame,
                    policy_columns={"predictive_consensus_greedy": "pred_consensus"},
                )
                hybrid_latency = float(hybrid_summary["mean_realized_latency_ms"].iloc[0])
                validation_gap_ms += hybrid_latency - best_single_latency

            validation_gap_ms /= max(1, len(validation_candidate_frames))
            candidate_result = ConsensusPolicyTuningResult(
                temporal_weight=temporal_weight,
                graph_weight=graph_weight,
                disagreement_penalty=disagreement_penalty,
                validation_gap_ms=validation_gap_ms,
            )
            if best_result is None or candidate_result.validation_gap_ms < best_result.validation_gap_ms:
                best_result = candidate_result

    if best_result is None:
        raise ValueError("unable to tune consensus policy without validation frames")
    return best_result


def evaluate_decision_policies(
    candidate_frame: pd.DataFrame,
    latency_budget_ms: float = 60.0,
    policy_columns: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare reactive and predictive path-selection policies.

    Each `session_bin_index` is treated as one normalized decision window across
    concurrently available candidate sessions.
    """

    policies = policy_columns or {
        "random": "random_score",
        "reactive_greedy": "latency_mean_ms",
        "predictive_greedy": "pred_forecast",
        "predictive_graph_greedy": "pred_graph",
        "predictive_simple_fusion_greedy": "pred_simple_fusion",
        "predictive_consensus_greedy": "pred_consensus",
    }
    work = candidate_frame.copy()
    work["random_score"] = (
        work["session_bin_index"] * 100000 + work.groupby("session_bin_index").cumcount() * 7919
    ) % 104729
    prediction_columns = [column for column in work.columns if column.startswith("pred_")]

    decision_rows: list[dict[str, object]] = []
    summary_rows: list[PolicyDecision] = []

    for policy_name, sort_column in policies.items():
        chosen_rows: list[dict[str, object]] = []
        for _, snapshot in work.groupby("session_bin_index", sort=True):
            snapshot = snapshot.copy()
            ascending = policy_name != "random"
            start_ns = time.perf_counter_ns()
            # Policies are evaluated at the decision-window level: the chosen
            # path uses only current or predicted scores, while regret is
            # computed afterward against the realized next-bin latency.
            chosen = snapshot.sort_values(sort_column, ascending=ascending).iloc[0]
            hindsight_best = snapshot.sort_values("target_next", ascending=True).iloc[0]
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            chosen_rows.append(
                {
                    "policy_name": policy_name,
                    "session_bin_index": int(chosen["session_bin_index"]),
                    "chosen_relative_path": chosen["relative_path"],
                    "chosen_location": chosen["location"],
                    "chosen_path_state": chosen["path_state"],
                    "reactive_latency_ms": float(chosen["latency_mean_ms"]),
                    "realized_next_latency_ms": float(chosen["target_next"]),
                    "hindsight_best_latency_ms": float(hindsight_best["target_next"]),
                    "regret_ms": float(chosen["target_next"] - hindsight_best["target_next"]),
                    "best_path_match": int(chosen["relative_path"] == hindsight_best["relative_path"]),
                    "success_under_budget": int(chosen["target_next"] <= latency_budget_ms),
                    "decision_time_us": elapsed_us,
                }
            )
            for prediction_column in prediction_columns:
                chosen_rows[-1][prediction_column] = float(chosen[prediction_column])

        decisions = pd.DataFrame(chosen_rows)
        decision_rows.append(decisions)
        summary_rows.append(
            PolicyDecision(
                policy_name=policy_name,
                decision_count=len(decisions),
                mean_realized_latency_ms=float(decisions["realized_next_latency_ms"].mean()),
                mean_regret_ms=float(decisions["regret_ms"].mean()),
                best_path_match_rate=float(decisions["best_path_match"].mean()),
                success_rate_under_60ms=float(decisions["success_under_budget"].mean()),
                mean_decision_time_us=float(decisions["decision_time_us"].mean()),
            )
        )

    return pd.DataFrame([asdict(item) for item in summary_rows]), pd.concat(decision_rows, ignore_index=True)
