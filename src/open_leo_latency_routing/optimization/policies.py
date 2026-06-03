"""Decision policies for the final manuscript evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
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
    mean_decision_gap_ms: float
    retrospective_best_path_match_rate: float
    success_rate_under_60ms: float
    p95_realized_latency_ms: float
    cvar95_realized_latency_ms: float
    mean_decision_time_us: float
    switch_rate: float


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
        previous_path: str | None = None
        for _, snapshot in work.groupby("session_bin_index", sort=True):
            snapshot = snapshot.copy()
            ascending = policy_name != "random"
            start_ns = time.perf_counter_ns()
            # Policies are evaluated at the decision-window level. The chosen
            # path uses only online scores; the retrospective benchmark is
            # computed afterward from observed next-bin latency.
            chosen = snapshot.sort_values(sort_column, ascending=ascending).iloc[0]
            retrospective_best = snapshot.sort_values("target_next", ascending=True).iloc[0]
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            switched_path = int(previous_path is not None and chosen["relative_path"] != previous_path)
            previous_path = str(chosen["relative_path"])
            chosen_rows.append(
                {
                    "policy_name": policy_name,
                    "session_bin_index": int(chosen["session_bin_index"]),
                    "chosen_relative_path": chosen["relative_path"],
                    "chosen_location": chosen["location"],
                    "chosen_path_state": chosen["path_state"],
                    "reactive_latency_ms": float(chosen["latency_mean_ms"]),
                    "realized_next_latency_ms": float(chosen["target_next"]),
                    "retrospective_best_path_latency_ms": float(retrospective_best["target_next"]),
                    "decision_gap_ms": float(chosen["target_next"] - retrospective_best["target_next"]),
                    "retrospective_best_path_match": int(
                        chosen["relative_path"] == retrospective_best["relative_path"]
                    ),
                    "success_under_budget": int(chosen["target_next"] <= latency_budget_ms),
                    "decision_time_us": elapsed_us,
                    "switched_path": switched_path,
                }
            )
            for prediction_column in prediction_columns:
                chosen_rows[-1][prediction_column] = float(chosen[prediction_column])
            for horizon in (3, 5):
                cumulative_column = f"target_cumulative_{horizon}"
                mean_column = f"target_mean_{horizon}"
                if cumulative_column in snapshot.columns:
                    available_snapshot = snapshot[snapshot[cumulative_column].notna()].copy()
                    if not available_snapshot.empty and pd.notna(chosen.get(cumulative_column)):
                        retrospective_best_cumulative = available_snapshot.sort_values(
                            cumulative_column, ascending=True
                        ).iloc[0]
                        chosen_rows[-1][f"realized_cumulative_latency_{horizon}_ms"] = float(
                            chosen[cumulative_column]
                        )
                        chosen_rows[-1][f"retrospective_best_cumulative_latency_{horizon}_ms"] = float(
                            retrospective_best_cumulative[cumulative_column]
                        )
                        chosen_rows[-1][f"cumulative_decision_gap_{horizon}_ms"] = float(
                            chosen[cumulative_column] - retrospective_best_cumulative[cumulative_column]
                        )
                        chosen_rows[-1][f"cumulative_success_under_budget_{horizon}"] = int(
                            float(chosen[mean_column]) <= latency_budget_ms
                        )
                    else:
                        chosen_rows[-1][f"realized_cumulative_latency_{horizon}_ms"] = np.nan
                        chosen_rows[-1][f"retrospective_best_cumulative_latency_{horizon}_ms"] = np.nan
                        chosen_rows[-1][f"cumulative_decision_gap_{horizon}_ms"] = np.nan
                        chosen_rows[-1][f"cumulative_success_under_budget_{horizon}"] = np.nan

        decisions = pd.DataFrame(chosen_rows)
        p95_latency = float(decisions["realized_next_latency_ms"].quantile(0.95))
        cvar95_latency = float(
            decisions.loc[decisions["realized_next_latency_ms"] >= p95_latency, "realized_next_latency_ms"].mean()
        )
        decision_rows.append(decisions)
        summary_rows.append(
            PolicyDecision(
                policy_name=policy_name,
                decision_count=len(decisions),
                mean_realized_latency_ms=float(decisions["realized_next_latency_ms"].mean()),
                mean_decision_gap_ms=float(decisions["decision_gap_ms"].mean()),
                retrospective_best_path_match_rate=float(decisions["retrospective_best_path_match"].mean()),
                success_rate_under_60ms=float(decisions["success_under_budget"].mean()),
                p95_realized_latency_ms=p95_latency,
                cvar95_realized_latency_ms=cvar95_latency,
                mean_decision_time_us=float(decisions["decision_time_us"].mean()),
                switch_rate=float(decisions["switched_path"].mean()),
            )
        )

    return pd.DataFrame([asdict(item) for item in summary_rows]), pd.concat(decision_rows, ignore_index=True)


def summarize_switching_costs(
    decisions: pd.DataFrame,
    penalty_levels_ms: list[float],
    latency_budget_ms: float = 60.0,
    scenario_column: str = "scenario_name",
) -> pd.DataFrame:
    """Evaluate how path-switch penalties change practical policy utility."""

    rows: list[dict[str, object]] = []
    grouping = [scenario_column, "policy_name"] if scenario_column in decisions.columns else ["policy_name"]
    for keys, policy_frame in decisions.groupby(grouping, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))
        for penalty_ms in penalty_levels_ms:
            penalized_latency = (
                policy_frame["realized_next_latency_ms"] + penalty_ms * policy_frame["switched_path"]
            )
            penalized_gap = policy_frame["decision_gap_ms"] + penalty_ms * policy_frame["switched_path"]
            p95_latency = float(penalized_latency.quantile(0.95))
            rows.append(
                {
                    **base,
                    "switch_penalty_ms": penalty_ms,
                    "decision_count": len(policy_frame),
                    "switch_rate": float(policy_frame["switched_path"].mean()),
                    "mean_penalized_latency_ms": float(penalized_latency.mean()),
                    "mean_penalized_decision_gap_ms": float(penalized_gap.mean()),
                    "success_rate_under_60ms": float((penalized_latency <= latency_budget_ms).mean()),
                    "p95_penalized_latency_ms": p95_latency,
                    "cvar95_penalized_latency_ms": float(
                        penalized_latency[penalized_latency >= p95_latency].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_stochastic_switching_costs(
    decisions: pd.DataFrame,
    base_penalty_ms: float,
    spike_penalty_ms: float,
    spike_probability: float,
    n_trials: int = 256,
    latency_budget_ms: float = 60.0,
    scenario_column: str = "scenario_name",
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate switching utility under mostly mild but occasionally severe handovers.

    Each switch always pays the base penalty. With probability `spike_probability`,
    the switch also incurs an additional spike penalty, modeling transient
    outages or unusually slow handovers.
    """

    rows: list[dict[str, object]] = []
    grouping = [scenario_column, "policy_name"] if scenario_column in decisions.columns else ["policy_name"]
    for keys, policy_frame in decisions.groupby(grouping, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))
        rng_seed = abs(hash(keys)) % (2**32)
        rng = np.random.default_rng(random_state + rng_seed)
        switch_mask = policy_frame["switched_path"].to_numpy(dtype=float)
        base_latency = policy_frame["realized_next_latency_ms"].to_numpy(dtype=float)
        base_gap = policy_frame["decision_gap_ms"].to_numpy(dtype=float)

        trial_latency_means = []
        trial_gap_means = []
        trial_success = []
        trial_p95 = []
        trial_cvar95 = []
        spike_rates = []
        for _ in range(n_trials):
            spike_events = rng.random(len(policy_frame)) < spike_probability
            trial_penalty = switch_mask * (base_penalty_ms + spike_penalty_ms * spike_events.astype(float))
            penalized_latency = base_latency + trial_penalty
            penalized_gap = base_gap + trial_penalty
            p95_latency = float(np.quantile(penalized_latency, 0.95))
            trial_latency_means.append(float(np.mean(penalized_latency)))
            trial_gap_means.append(float(np.mean(penalized_gap)))
            trial_success.append(float(np.mean(penalized_latency <= latency_budget_ms)))
            trial_p95.append(p95_latency)
            trial_cvar95.append(float(np.mean(penalized_latency[penalized_latency >= p95_latency])))
            spike_rates.append(float(np.mean(spike_events & (switch_mask > 0))))

        rows.append(
            {
                **base,
                "decision_count": len(policy_frame),
                "switch_rate": float(policy_frame["switched_path"].mean()),
                "base_penalty_ms": float(base_penalty_ms),
                "spike_penalty_ms": float(spike_penalty_ms),
                "spike_probability": float(spike_probability),
                "n_trials": int(n_trials),
                "mean_penalized_latency_ms": float(np.mean(trial_latency_means)),
                "mean_penalized_latency_ci_low_ms": float(np.quantile(trial_latency_means, 0.025)),
                "mean_penalized_latency_ci_high_ms": float(np.quantile(trial_latency_means, 0.975)),
                "mean_penalized_decision_gap_ms": float(np.mean(trial_gap_means)),
                "mean_penalized_decision_gap_ci_low_ms": float(np.quantile(trial_gap_means, 0.025)),
                "mean_penalized_decision_gap_ci_high_ms": float(np.quantile(trial_gap_means, 0.975)),
                "success_rate_under_60ms": float(np.mean(trial_success)),
                "success_rate_ci_low": float(np.quantile(trial_success, 0.025)),
                "success_rate_ci_high": float(np.quantile(trial_success, 0.975)),
                "p95_penalized_latency_ms": float(np.mean(trial_p95)),
                "cvar95_penalized_latency_ms": float(np.mean(trial_cvar95)),
                "mean_spike_event_rate": float(np.mean(spike_rates)),
            }
        )
    return pd.DataFrame(rows)


def summarize_multibin_decisions(
    decisions: pd.DataFrame,
    horizons: list[int] | None = None,
    scenario_column: str = "scenario_name",
) -> pd.DataFrame:
    """Aggregate short-horizon cumulative service outcomes for each policy."""

    horizons = horizons or [3, 5]
    rows: list[dict[str, object]] = []
    grouping = [scenario_column, "policy_name"] if scenario_column in decisions.columns else ["policy_name"]
    for keys, policy_frame in decisions.groupby(grouping, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(grouping, keys))
        for horizon in horizons:
            cumulative_column = f"realized_cumulative_latency_{horizon}_ms"
            gap_column = f"cumulative_decision_gap_{horizon}_ms"
            success_column = f"cumulative_success_under_budget_{horizon}"
            if cumulative_column not in policy_frame.columns:
                continue
            valid = policy_frame[policy_frame[cumulative_column].notna()].copy()
            if valid.empty:
                continue
            rows.append(
                {
                    **base,
                    "horizon_bins": horizon,
                    "decision_count": len(valid),
                    "mean_cumulative_latency_ms": float(valid[cumulative_column].mean()),
                    "mean_cumulative_decision_gap_ms": float(valid[gap_column].mean()),
                    "success_rate_under_budget": float(valid[success_column].mean()),
                }
            )
    return pd.DataFrame(rows)
