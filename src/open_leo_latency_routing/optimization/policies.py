"""Decision policies for the final manuscript evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import pandas as pd

from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar
from open_leo_latency_routing.optimization.explainability import (
    counterfactual_decision_explanation,
)


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
    mean_end_to_end_latency_ms: float
    p95_end_to_end_latency_ms: float
    control_loop_latency_ms: float
    stale_decision_rate: float
    mean_feasible_candidate_count: float
    no_feasible_candidate_rate: float
    switch_count: int
    eligible_switch_transition_count: int
    # Backward-compatible alias for eligible_switch_transition_count.  This
    # historical field is a denominator, not the number of observed switches.
    switch_transition_count: int
    continuity_reset_count: int
    continuity_segment_count: int
    continuity_key_column: str
    continuity_cadence_seconds: float
    continuity_cadence_source: str


@dataclass(frozen=True)
class SwitchTransitionMetrics:
    """Validated switch statistics over contiguous decision transitions."""

    switch_count: int
    eligible_transition_count: int
    switch_rate: float


def select_validation_gated_fallback(
    realized_validation_latency: dict[str, list[float]],
    latency_budget_ms: float,
    allowed_policies: tuple[str, ...] | None = None,
) -> str:
    """Select a fallback without observing test outcomes.

    Validation QoS success is optimized first. Mean realized validation
    latency breaks exact success ties. Including ``reactive`` in the allowed
    set lets the deployed rule abstain from predictive refinement when neither
    predictive view earns that refinement on the disjoint validation window.
    """

    policies = allowed_policies or tuple(realized_validation_latency)
    if not policies:
        raise ValueError("at least one validation fallback policy is required")
    missing = set(policies).difference(realized_validation_latency)
    if missing:
        raise KeyError(f"missing validation latency for policies: {sorted(missing)}")

    def objective(name: str) -> tuple[float, float, str]:
        values = np.asarray(realized_validation_latency[name], dtype=float)
        if values.size == 0:
            return (0.0, float("inf"), name)
        return (
            -float(np.mean(values <= latency_budget_ms)),
            float(np.mean(values)),
            name,
        )

    return min(policies, key=objective)


def add_qos_shielded_scores(
    candidate_frame: pd.DataFrame,
    fallback_column: str,
    latency_budget_ms: float = 60.0,
    output_column: str = "pred_qos_shielded_operational",
    staleness_margin_column: str | None = None,
) -> pd.DataFrame:
    """Add a lexicographic QoS-preserving operational score.

    The shield uses only information observable at decision time. If a
    snapshot mixes currently QoS-compliant and non-compliant paths, it ranks
    compliant paths by current latency and excludes the others. When every
    available path is compliant, or none is, it defers to the validation-
    selected predictive fallback. The output remains a numeric score so it can
    be evaluated by the same policy runner as every comparator.
    """

    required = {
        "session_bin_index",
        "latency_mean_ms",
        fallback_column,
    }
    missing = required.difference(candidate_frame.columns)
    if missing:
        raise KeyError(f"QoS shield is missing required columns: {sorted(missing)}")

    output = candidate_frame.copy()
    available = (
        output["is_feasible_path"].astype(bool)
        if "is_feasible_path" in output
        else pd.Series(True, index=output.index)
    )
    staleness_margin = (
        output[staleness_margin_column].astype(float).clip(lower=0.0)
        if staleness_margin_column and staleness_margin_column in output
        else pd.Series(0.0, index=output.index)
    )
    output["age_adjusted_current_latency_ms"] = (
        output["latency_mean_ms"].astype(float) + staleness_margin
    )
    current_qos_feasible = available & output[
        "age_adjusted_current_latency_ms"
    ].le(latency_budget_ms)
    feasible_count = available.astype(int).groupby(
        output["session_bin_index"]
    ).transform("sum")
    qos_count = current_qos_feasible.astype(int).groupby(
        output["session_bin_index"]
    ).transform("sum")
    all_qos_feasible = feasible_count.gt(0) & qos_count.eq(feasible_count)
    some_qos_feasible = qos_count.gt(0) & qos_count.lt(feasible_count)

    # The finite offset preserves deterministic ordering while ensuring that
    # every currently compliant path outranks every non-compliant alternative.
    exclusion_offset = max(
        1.0e9,
        float(output["latency_mean_ms"].abs().max()) * 1.0e6,
    )
    mixed_snapshot_score = np.where(
        current_qos_feasible,
        output["age_adjusted_current_latency_ms"],
        exclusion_offset + output["age_adjusted_current_latency_ms"],
    )
    output[output_column] = np.where(
        all_qos_feasible | qos_count.eq(0),
        output[fallback_column],
        mixed_snapshot_score,
    )
    output["qos_shield_mode"] = np.select(
        [all_qos_feasible, some_qos_feasible],
        ["all_qos_fallback", "mixed_qos_safeguard"],
        default="no_qos_fallback",
    )
    return output


def add_qos_filter_then_rank_scores(
    candidate_frame: pd.DataFrame,
    ranking_column: str,
    latency_budget_ms: float = 60.0,
    output_column: str = "pred_qos_filter_then_rank",
    staleness_margin_column: str | None = None,
) -> pd.DataFrame:
    """Filter current QoS violations, then rank remaining paths predictively.

    This decision-level baseline differs from the conservative shield: when at
    least one adjusted-current path is compliant, it permits the learned score
    to choose among all compliant paths. If none is compliant, it ranks every
    available path with the same frozen score.
    """

    required = {"session_bin_index", "latency_mean_ms", ranking_column}
    missing = required.difference(candidate_frame.columns)
    if missing:
        raise KeyError(f"QoS filter baseline is missing columns: {sorted(missing)}")
    output = candidate_frame.copy()
    available = (
        output["is_feasible_path"].astype(bool)
        if "is_feasible_path" in output
        else pd.Series(True, index=output.index)
    )
    margin = (
        output[staleness_margin_column].astype(float).clip(lower=0.0)
        if staleness_margin_column and staleness_margin_column in output
        else pd.Series(0.0, index=output.index)
    )
    adjusted = output["latency_mean_ms"].astype(float) + margin
    compliant = available & adjusted.le(latency_budget_ms)
    compliant_count = compliant.astype(int).groupby(
        output["session_bin_index"]
    ).transform("sum")
    exclusion_offset = max(
        1.0e9,
        float(output[ranking_column].abs().max()) * 1.0e6,
    )
    output[output_column] = np.where(
        compliant_count.gt(0),
        np.where(compliant, output[ranking_column], exclusion_offset + output[ranking_column]),
        output[ranking_column],
    )
    return output


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


_CONTINUITY_KEY_CANDIDATES = (
    "continuity_id",
    "session_id",
    "campaign_id",
    "session_date",
)


def _resolve_continuity_key_column(
    frame: pd.DataFrame,
    requested_column: str | None,
) -> str | None:
    """Resolve an explicit or canonical sequence-continuity key."""

    if requested_column is not None:
        if requested_column not in frame.columns:
            raise KeyError(
                "continuity key column is missing from candidate frame: "
                f"{requested_column}"
            )
        return requested_column
    return next(
        (column for column in _CONTINUITY_KEY_CANDIDATES if column in frame),
        None,
    )


def _positive_finite_scalar(value: object, label: str) -> float:
    """Return a validated positive finite cadence value."""

    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return parsed


def _resolve_continuity_cadence(
    frame: pd.DataFrame,
    decision_cadence_seconds: float | None,
    decision_window_seconds: float | None,
) -> tuple[float | None, str]:
    """Resolve declared cadence without learning it from observed gaps."""

    if decision_cadence_seconds is not None:
        return (
            _positive_finite_scalar(
                decision_cadence_seconds,
                "decision_cadence_seconds",
            ),
            "decision_cadence_seconds_argument",
        )
    if decision_window_seconds is not None:
        return (
            _positive_finite_scalar(
                decision_window_seconds,
                "decision_window_seconds",
            ),
            "decision_window_seconds_argument",
        )
    for attr_name in ("decision_cadence_seconds", "bin_seconds"):
        if attr_name in frame.attrs:
            return (
                _positive_finite_scalar(
                    frame.attrs[attr_name],
                    f"attrs[{attr_name!r}]",
                ),
                f"frame_attr:{attr_name}",
            )
    if "bin_seconds" in frame:
        declared = pd.to_numeric(frame["bin_seconds"], errors="coerce")
        values = declared.dropna().to_numpy(dtype=float)
        if len(values) != len(frame):
            raise ValueError("bin_seconds must be complete when used as cadence")
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("bin_seconds must contain finite positive values")
        reference = float(values[0])
        if not np.isclose(
            values,
            reference,
            rtol=0.0,
            atol=max(1.0e-9, abs(reference) * 1.0e-12),
        ).all():
            raise ValueError("bin_seconds must declare one common decision cadence")
        return reference, "column:bin_seconds"
    return None, "unavailable"


def summarize_switch_transitions(
    decisions: pd.DataFrame,
) -> SwitchTransitionMetrics:
    """Compute switches per eligible within-sequence transition.

    Continuity resets (including session, campaign, and cadence gaps) are not
    transitions.  Require the explicit eligibility audit column so callers
    cannot silently revert to a decision-count denominator. When segment-start
    metadata is present, enforce both the row-wise eligibility complement and
    the aggregate ``T - S`` denominator closure.
    """

    required = {"switched_path", "switch_transition_eligible"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(
            "switch-transition metrics require audit columns: "
            f"missing={missing}"
        )
    switched = pd.to_numeric(
        decisions["switched_path"], errors="coerce"
    ).to_numpy(dtype=float)
    eligible = pd.to_numeric(
        decisions["switch_transition_eligible"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(switched).all() or not np.isfinite(eligible).all():
        raise ValueError("switch-transition audit columns must be finite")
    if not np.isin(switched, (0.0, 1.0)).all():
        raise ValueError("switched_path must be binary")
    if not np.isin(eligible, (0.0, 1.0)).all():
        raise ValueError("switch_transition_eligible must be binary")
    if ((eligible == 0.0) & (switched == 1.0)).any():
        raise ValueError(
            "switched_path cannot be true on an ineligible continuity-reset "
            "transition"
        )

    eligible_mask = eligible == 1.0
    eligible_count = int(eligible_mask.sum())
    switch_count = int(switched[eligible_mask].sum())
    if "continuity_segment_start" in decisions.columns:
        segment_start = pd.to_numeric(
            decisions["continuity_segment_start"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(segment_start).all():
            raise ValueError("continuity_segment_start must be finite")
        if not np.isin(segment_start, (0.0, 1.0)).all():
            raise ValueError("continuity_segment_start must be binary")
        segment_count = int(segment_start.sum())
        if len(decisions) and segment_count == 0:
            raise ValueError(
                "a nonempty decision frame must contain a continuity-segment "
                "start"
            )
        expected_eligible_count = len(decisions) - segment_count
        if eligible_count != expected_eligible_count:
            raise ValueError(
                "eligible switch-transition count violates T-S continuity "
                "closure: "
                f"observed={eligible_count}, decisions={len(decisions)}, "
                f"segments={segment_count}, expected={expected_eligible_count}"
            )
        if not np.array_equal(eligible, 1.0 - segment_start):
            raise ValueError(
                "switch_transition_eligible must be the row-wise complement "
                "of continuity_segment_start"
            )
    if "continuity_reset" in decisions.columns:
        continuity_reset = pd.to_numeric(
            decisions["continuity_reset"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(continuity_reset).all():
            raise ValueError("continuity_reset must be finite")
        if not np.isin(continuity_reset, (0.0, 1.0)).all():
            raise ValueError("continuity_reset must be binary")
        if "continuity_segment_start" in decisions.columns:
            segment_start = pd.to_numeric(
                decisions["continuity_segment_start"], errors="raise"
            ).to_numpy(dtype=float)
            if ((continuity_reset == 1.0) & (segment_start == 0.0)).any():
                raise ValueError(
                    "continuity_reset must begin a continuity segment"
                )
    switch_rate = (
        float(switch_count / eligible_count) if eligible_count else 0.0
    )
    return SwitchTransitionMetrics(
        switch_count=switch_count,
        eligible_transition_count=eligible_count,
        switch_rate=switch_rate,
    )


def evaluate_decision_policies(
    candidate_frame: pd.DataFrame,
    latency_budget_ms: float = 60.0,
    policy_columns: dict[str, str] | None = None,
    control_loop_latency_ms: float = 0.0,
    decision_window_seconds: float | None = None,
    online_switch_penalties_ms: dict[str, float] | None = None,
    decision_cadence_seconds: float | None = None,
    continuity_key_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare reactive and predictive path-selection policies.

    Each `session_bin_index` is treated as one normalized decision window across
    concurrently available candidate sessions. Control-loop latency includes
    state collection, controller processing, and decision dissemination. It is
    added to the observed network latency and is marked stale when it reaches
    or exceeds the configured decision window. Stateful switch penalties and
    switch counts apply only across contiguous decision epochs in the same
    session or campaign. A declared cadence is preferred; the observed epoch
    spacing is never used to infer away a telemetry gap.
    """

    policies = policy_columns or {
        "random": "random_score",
        "reactive_greedy": "latency_mean_ms",
        "predictive_greedy": "pred_forecast",
        "predictive_graph_greedy": "pred_graph",
        "predictive_simple_fusion_greedy": "pred_simple_fusion",
        "predictive_consensus_greedy": "pred_consensus",
    }
    switch_penalties = online_switch_penalties_ms or {}
    work = candidate_frame.copy()
    resolved_continuity_key = _resolve_continuity_key_column(
        work,
        continuity_key_column,
    )
    continuity_cadence_seconds, continuity_cadence_source = (
        _resolve_continuity_cadence(
            work,
            decision_cadence_seconds,
            decision_window_seconds,
        )
    )
    model_scoring_time_us = float(
        candidate_frame.attrs.get("model_scoring_time_us_per_decision", np.nan)
    )
    work["_continuity_input_order"] = np.arange(len(work), dtype=int)
    decision_group_columns = ["session_bin_index"]
    if resolved_continuity_key is not None:
        decision_group_columns.insert(0, resolved_continuity_key)
    work["_decision_candidate_ordinal"] = work.groupby(
        decision_group_columns,
        sort=False,
        dropna=False,
    ).cumcount()
    work["random_score"] = (
        work["session_bin_index"] * 100000
        + work["_decision_candidate_ordinal"] * 7919
    ) % 104729
    prediction_columns = [column for column in work.columns if column.startswith("pred_")]

    snapshots = [
        snapshot.copy()
        for _, snapshot in work.groupby(
            decision_group_columns,
            sort=False,
            dropna=False,
        )
    ]

    def _snapshot_sort_key(snapshot: pd.DataFrame) -> tuple[float, float, int]:
        if "bin_epoch" in snapshot:
            epoch_values = pd.to_numeric(
                snapshot["bin_epoch"],
                errors="coerce",
            ).dropna()
            if not epoch_values.empty:
                return (
                    0.0,
                    float(epoch_values.min()),
                    int(snapshot["_continuity_input_order"].min()),
                )
        index_values = pd.to_numeric(
            snapshot["session_bin_index"],
            errors="coerce",
        ).dropna()
        return (
            1.0,
            float(index_values.min()) if not index_values.empty else float("inf"),
            int(snapshot["_continuity_input_order"].min()),
        )

    snapshots.sort(key=_snapshot_sort_key)

    decision_rows: list[dict[str, object]] = []
    summary_rows: list[PolicyDecision] = []
    window_ms = (
        _positive_finite_scalar(
            decision_window_seconds,
            "decision_window_seconds",
        )
        * 1000.0
        if decision_window_seconds is not None
        else float("inf")
    )
    stale_decision = int(float(control_loop_latency_ms) >= window_ms)

    for policy_name, sort_column in policies.items():
        chosen_rows: list[dict[str, object]] = []
        previous_path: str | None = None
        previous_decision_epoch: float | None = None
        previous_decision_index: float | None = None
        previous_continuity_key: object | None = None
        has_previous_decision = False
        continuity_segment_id = 0
        for snapshot in snapshots:
            snapshot = snapshot.copy()
            current_epoch: float | None = None
            if "bin_epoch" in snapshot:
                epoch_values = pd.to_numeric(
                    snapshot["bin_epoch"],
                    errors="coerce",
                ).dropna().unique()
                if len(epoch_values) == 1 and np.isfinite(float(epoch_values[0])):
                    current_epoch = float(epoch_values[0])
            decision_index_values = pd.to_numeric(
                snapshot["session_bin_index"],
                errors="coerce",
            ).dropna().unique()
            current_decision_index = (
                float(decision_index_values[0])
                if len(decision_index_values) == 1
                else None
            )
            current_continuity_key: object | None = None
            if resolved_continuity_key is not None:
                key_values = snapshot[resolved_continuity_key].drop_duplicates()
                if len(key_values) != 1:
                    raise ValueError(
                        "a decision snapshot must have exactly one continuity "
                        f"key value in {resolved_continuity_key!r}"
                    )
                current_continuity_key = key_values.iloc[0]

            reset_reasons: list[str] = []
            epoch_gap_seconds = np.nan
            decision_index_gap = np.nan
            if has_previous_decision:
                if resolved_continuity_key is not None:
                    current_key_missing = bool(pd.isna(current_continuity_key))
                    previous_key_missing = bool(pd.isna(previous_continuity_key))
                    if current_key_missing or previous_key_missing:
                        reset_reasons.append("missing_continuity_key")
                    elif current_continuity_key != previous_continuity_key:
                        reset_reasons.append("continuity_key_change")
                if current_epoch is not None and previous_decision_epoch is not None:
                    epoch_gap_seconds = current_epoch - previous_decision_epoch
                    if continuity_cadence_seconds is not None:
                        cadence_tolerance = max(
                            1.0e-9,
                            abs(continuity_cadence_seconds) * 1.0e-12,
                        )
                        if not np.isclose(
                            epoch_gap_seconds,
                            continuity_cadence_seconds,
                            rtol=0.0,
                            atol=cadence_tolerance,
                        ):
                            if epoch_gap_seconds <= 0.0:
                                reset_reasons.append("nonincreasing_epoch")
                            elif epoch_gap_seconds > (
                                continuity_cadence_seconds + cadence_tolerance
                            ):
                                reset_reasons.append("epoch_gap")
                            else:
                                reset_reasons.append("off_cadence_epoch")
                    elif epoch_gap_seconds <= 0.0:
                        reset_reasons.append("nonincreasing_epoch")
                if (
                    current_decision_index is not None
                    and previous_decision_index is not None
                ):
                    decision_index_gap = (
                        current_decision_index - previous_decision_index
                    )
                    if not np.isclose(
                        decision_index_gap,
                        1.0,
                        rtol=0.0,
                        atol=1.0e-9,
                    ):
                        reset_reasons.append("decision_index_gap")

            # De-duplicate reasons while retaining their diagnostic order.
            reset_reasons = list(dict.fromkeys(reset_reasons))
            continuity_reset = int(has_previous_decision and bool(reset_reasons))
            continuity_segment_start = int(
                not has_previous_decision or continuity_reset
            )
            if continuity_segment_start:
                continuity_segment_id += 1
            prior_path_before_reset = previous_path
            if continuity_reset:
                previous_path = None
            switch_transition_eligible = int(previous_path is not None)
            feasible_count = len(snapshot)
            no_feasible_candidate = 0
            if "is_feasible_path" in snapshot:
                feasible = snapshot[
                    snapshot["is_feasible_path"].astype(bool)
                ].copy()
                feasible_count = len(feasible)
                if not feasible.empty:
                    snapshot = feasible
                else:
                    # Retain a deterministic emergency fallback so the run
                    # remains auditable instead of silently dropping a window.
                    no_feasible_candidate = 1
            ascending = policy_name != "random"
            start_ns = time.perf_counter_ns()
            # Policies are evaluated at the decision-window level. The chosen
            # path uses only online scores; the retrospective benchmark is
            # computed afterward from observed next-bin latency.
            selection_score = snapshot[sort_column].astype(float).copy()
            switch_penalty_ms = float(switch_penalties.get(policy_name, 0.0))
            if previous_path is not None and switch_penalty_ms > 0.0:
                selection_score = selection_score + switch_penalty_ms * (
                    snapshot["relative_path"].astype(str) != previous_path
                ).astype(float)
            snapshot["_selection_score"] = selection_score
            sorted_snapshot = snapshot.sort_values("_selection_score", ascending=ascending)
            chosen = sorted_snapshot.iloc[0]
            runner_up = sorted_snapshot.iloc[1] if len(sorted_snapshot) > 1 else None
            retrospective_best = snapshot.sort_values("target_next", ascending=True).iloc[0]
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            switched_path = int(previous_path is not None and chosen["relative_path"] != previous_path)
            runner_up_switched = bool(
                previous_path is not None
                and runner_up is not None
                and runner_up["relative_path"] != previous_path
            )
            xai_fields = counterfactual_decision_explanation(
                chosen=chosen,
                runner_up=runner_up,
                selected_score=float(chosen["_selection_score"]),
                runner_up_score=(
                    float(runner_up["_selection_score"]) if runner_up is not None else None
                ),
                sort_column=sort_column,
                switch_penalty_ms=switch_penalty_ms,
                chosen_switched=bool(switched_path),
                runner_up_switched=runner_up_switched,
            )
            previous_path = str(chosen["relative_path"])
            previous_decision_epoch = current_epoch
            previous_decision_index = current_decision_index
            previous_continuity_key = current_continuity_key
            has_previous_decision = True
            chosen_rows.append(
                {
                    "policy_name": policy_name,
                    "session_bin_index": int(chosen["session_bin_index"]),
                    "decision_bin_epoch": (
                        float(chosen["bin_epoch"])
                        if "bin_epoch" in chosen.index
                        else np.nan
                    ),
                    "chosen_relative_path": chosen["relative_path"],
                    "chosen_location": chosen["location"],
                    "chosen_path_state": chosen["path_state"],
                    "reactive_latency_ms": float(chosen["latency_mean_ms"]),
                    "realized_next_latency_ms": float(chosen["target_next"]),
                    "end_to_end_realized_latency_ms": float(
                        chosen["target_next"] + control_loop_latency_ms
                    ),
                    "retrospective_best_path_latency_ms": float(retrospective_best["target_next"]),
                    "decision_gap_ms": float(chosen["target_next"] - retrospective_best["target_next"]),
                    "retrospective_best_path_match": int(
                        chosen["relative_path"] == retrospective_best["relative_path"]
                    ),
                    "success_under_budget": int(
                        chosen["target_next"] + control_loop_latency_ms <= latency_budget_ms
                    ),
                    "control_loop_latency_ms": float(control_loop_latency_ms),
                    "decision_window_seconds": (
                        float(decision_window_seconds)
                        if decision_window_seconds is not None
                        else np.nan
                    ),
                    "stale_decision": stale_decision,
                    "feasible_candidate_count": feasible_count,
                    "no_feasible_candidate": no_feasible_candidate,
                    "decision_time_us": elapsed_us,
                    "model_scoring_time_us": model_scoring_time_us,
                    "model_and_ranking_time_us": (
                        model_scoring_time_us + elapsed_us
                        if np.isfinite(model_scoring_time_us)
                        else np.nan
                    ),
                    "switched_path": switched_path,
                    "switch_transition_eligible": switch_transition_eligible,
                    "online_switch_penalty_ms": switch_penalty_ms,
                    "selected_online_score": float(chosen["_selection_score"]),
                    "continuity_reset": continuity_reset,
                    "continuity_segment_start": continuity_segment_start,
                    "continuity_segment_id": continuity_segment_id,
                    "continuity_reset_reason": (
                        "+".join(reset_reasons)
                        if reset_reasons
                        else (
                            "initial_decision"
                            if continuity_segment_start
                            else "continuous"
                        )
                    ),
                    "continuity_key_column": resolved_continuity_key or "",
                    "continuity_key_value": (
                        ""
                        if resolved_continuity_key is None
                        or pd.isna(current_continuity_key)
                        else str(current_continuity_key)
                    ),
                    "continuity_cadence_seconds": (
                        float(continuity_cadence_seconds)
                        if continuity_cadence_seconds is not None
                        else np.nan
                    ),
                    "continuity_cadence_source": continuity_cadence_source,
                    "continuity_previous_path_before_reset": (
                        prior_path_before_reset or ""
                    ),
                    "continuity_previous_decision_bin_epoch": (
                        float(current_epoch - epoch_gap_seconds)
                        if current_epoch is not None
                        and np.isfinite(epoch_gap_seconds)
                        else np.nan
                    ),
                    "continuity_epoch_gap_seconds": epoch_gap_seconds,
                    "continuity_decision_index_gap": decision_index_gap,
                    **xai_fields,
                }
            )
            for prediction_column in prediction_columns:
                chosen_rows[-1][prediction_column] = float(chosen[prediction_column])
            for source_column, output_column in (
                ("handover_event", "chosen_handover_event"),
                ("attenuation_event", "chosen_attenuation_event"),
                ("elevation_degrees", "chosen_elevation_degrees"),
                (
                    "propagation_lower_bound_ms",
                    "chosen_propagation_lower_bound_ms",
                ),
            ):
                if source_column in chosen.index and pd.notna(chosen[source_column]):
                    chosen_rows[-1][output_column] = float(chosen[source_column])
            if "qos_shield_mode" in chosen.index:
                chosen_rows[-1]["qos_shield_mode"] = str(
                    chosen["qos_shield_mode"]
                )
            for metadata_column in (
                "risk_fallback_policy",
                "validation_gated_fallback_policy",
            ):
                if metadata_column in chosen.index:
                    chosen_rows[-1][metadata_column] = str(
                        chosen[metadata_column]
                    )
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
        p95_end_to_end_latency = float(
            decisions["end_to_end_realized_latency_ms"].quantile(0.95)
        )
        cvar95_latency = empirical_upper_cvar(
            decisions["realized_next_latency_ms"].to_numpy(dtype=float),
            0.95,
        )
        switch_metrics = summarize_switch_transitions(decisions)
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
                switch_rate=switch_metrics.switch_rate,
                mean_end_to_end_latency_ms=float(
                    decisions["end_to_end_realized_latency_ms"].mean()
                ),
                p95_end_to_end_latency_ms=p95_end_to_end_latency,
                control_loop_latency_ms=float(control_loop_latency_ms),
                stale_decision_rate=float(decisions["stale_decision"].mean()),
                mean_feasible_candidate_count=float(
                    decisions["feasible_candidate_count"].mean()
                ),
                no_feasible_candidate_rate=float(
                    decisions["no_feasible_candidate"].mean()
                ),
                switch_count=switch_metrics.switch_count,
                eligible_switch_transition_count=(
                    switch_metrics.eligible_transition_count
                ),
                switch_transition_count=(
                    switch_metrics.eligible_transition_count
                ),
                continuity_reset_count=int(decisions["continuity_reset"].sum()),
                continuity_segment_count=int(
                    decisions["continuity_segment_start"].sum()
                ),
                continuity_key_column=resolved_continuity_key or "",
                continuity_cadence_seconds=(
                    float(continuity_cadence_seconds)
                    if continuity_cadence_seconds is not None
                    else np.nan
                ),
                continuity_cadence_source=continuity_cadence_source,
            )
        )

    summary = pd.DataFrame([asdict(item) for item in summary_rows])
    summary["mean_model_scoring_time_us"] = model_scoring_time_us
    summary["mean_model_and_ranking_time_us"] = (
        summary["mean_decision_time_us"] + model_scoring_time_us
        if np.isfinite(model_scoring_time_us)
        else np.nan
    )
    decisions = pd.concat(decision_rows, ignore_index=True)
    continuity_audit = {
        "continuity_key_column": resolved_continuity_key,
        "continuity_cadence_seconds": continuity_cadence_seconds,
        "continuity_cadence_source": continuity_cadence_source,
        "state_reset_rule": (
            "reset previous path, online switch penalty, and switch transition "
            "counter at continuity-key changes, missing continuity keys, "
            "off-cadence epochs, or decision-index gaps"
        ),
    }
    summary.attrs["continuity_audit"] = continuity_audit
    decisions.attrs["continuity_audit"] = continuity_audit
    return summary, decisions


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
                    "switch_rate": summarize_switch_transitions(
                        policy_frame
                    ).switch_rate,
                    "mean_penalized_latency_ms": float(penalized_latency.mean()),
                    "mean_penalized_decision_gap_ms": float(penalized_gap.mean()),
                    "success_rate_under_60ms": float((penalized_latency <= latency_budget_ms).mean()),
                    "p95_penalized_latency_ms": p95_latency,
                    "cvar95_penalized_latency_ms": empirical_upper_cvar(
                        penalized_latency.to_numpy(dtype=float), 0.95
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
            trial_cvar95.append(empirical_upper_cvar(penalized_latency, 0.95))
            spike_rates.append(float(np.mean(spike_events & (switch_mask > 0))))

        rows.append(
            {
                **base,
                "decision_count": len(policy_frame),
                "switch_rate": summarize_switch_transitions(
                    policy_frame
                ).switch_rate,
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
