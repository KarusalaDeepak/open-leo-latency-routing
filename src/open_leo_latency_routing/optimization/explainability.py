"""Explainable decision-audit helpers for path-selection policies."""

from __future__ import annotations

import numpy as np
import pandas as pd


XAI_COMPONENT_COLUMNS = [
    "xai_latency_component_ms",
    "xai_disagreement_component_ms",
    "xai_uncertainty_component_ms",
    "xai_service_risk_component_ms",
    "xai_switch_component_ms",
    "xai_calibration_component_ms",
]


def _value(row: pd.Series, column: str, default: float = 0.0) -> float:
    """Return a finite numeric value from a row if the column exists."""

    if column not in row.index or pd.isna(row[column]):
        return default
    value = float(row[column])
    return value if np.isfinite(value) else default


def explain_candidate_score(
    row: pd.Series,
    *,
    selected_score: float,
    sort_column: str,
    switch_penalty_ms: float = 0.0,
    switched_path: bool = False,
) -> dict[str, float]:
    """Decompose one candidate's online decision score into auditable terms.

    The calibrated operational policy exports its exact branch-aware weighted
    terms. Other policies are represented by their complete online score as the
    latency/ranking component. In both cases, the reported terms sum exactly to
    the score used for selection.
    """
    is_operational_score = (
        sort_column in {"pred_calibrated_operational", "pred_disagreement_aware"}
        and "score_component_latency_ms" in row.index
    )
    if is_operational_score:
        latency_component = _value(row, "score_component_latency_ms")
        disagreement_component = _value(row, "score_component_disagreement_ms")
        uncertainty_component = _value(row, "score_component_uncertainty_ms")
        service_risk_component = _value(row, "score_component_service_risk_ms")
        calibration_component = _value(row, "score_component_calibration_ms")
    else:
        base_score = _value(row, sort_column, float(selected_score))
        latency_component = base_score
        disagreement_component = 0.0
        uncertainty_component = 0.0
        service_risk_component = 0.0
        calibration_component = 0.0
    switch_component = float(switch_penalty_ms) if switched_path else 0.0

    explained_sum = (
        latency_component
        + disagreement_component
        + uncertainty_component
        + service_risk_component
        + calibration_component
        + switch_component
    )
    if not np.isclose(explained_sum, float(selected_score), atol=1e-8, rtol=1e-8):
        raise AssertionError(
            "explanation terms do not sum to the online selection score: "
            f"{explained_sum} != {selected_score}"
        )

    absolute_total = (
        abs(latency_component)
        + abs(disagreement_component)
        + abs(uncertainty_component)
        + abs(service_risk_component)
        + abs(calibration_component)
        + abs(switch_component)
    )
    if absolute_total <= 1e-12:
        dominant_component = "none"
    else:
        components = {
            "latency": abs(latency_component),
            "disagreement": abs(disagreement_component),
            "uncertainty": abs(uncertainty_component),
            "service_risk": abs(service_risk_component),
            "calibration": abs(calibration_component),
            "switching": abs(switch_component),
        }
        dominant_component = max(components, key=components.get)

    if sort_column == "pred_qos_shielded_operational":
        score_branch = str(row.get("qos_shield_mode", "qos_shield"))
        gate_active = int(
            score_branch != "mixed_qos_safeguard"
            and _value(row, "disagreement_trust_gate", 0.0) > 0.0
        )
        fallback_policy = (
            "current_qos_latency"
            if score_branch == "mixed_qos_safeguard"
            else str(row.get("risk_fallback_policy", ""))
        )
    else:
        score_branch = str(row.get("score_branch", "direct_score"))
        gate_active = int(_value(row, "disagreement_trust_gate", 0.0) > 0.0)
        fallback_policy = str(row.get("risk_fallback_policy", ""))

    return {
        "xai_latency_component_ms": float(latency_component),
        "xai_disagreement_component_ms": float(disagreement_component),
        "xai_uncertainty_component_ms": float(uncertainty_component),
        "xai_service_risk_component_ms": float(service_risk_component),
        "xai_switch_component_ms": float(switch_component),
        "xai_calibration_component_ms": float(calibration_component),
        "xai_explained_signed_total_ms": float(explained_sum),
        "xai_explained_abs_total_ms": float(absolute_total),
        "xai_dominant_component": dominant_component,
        "xai_score_branch": score_branch,
        "xai_gate_active": gate_active,
        "xai_fallback_policy": fallback_policy,
    }


def counterfactual_decision_explanation(
    *,
    chosen: pd.Series,
    runner_up: pd.Series | None,
    selected_score: float,
    runner_up_score: float | None,
    sort_column: str,
    switch_penalty_ms: float,
    chosen_switched: bool,
    runner_up_switched: bool,
) -> dict[str, object]:
    """Explain why the selected path beat the nearest rejected alternative."""

    chosen_terms = explain_candidate_score(
        chosen,
        selected_score=selected_score,
        sort_column=sort_column,
        switch_penalty_ms=switch_penalty_ms,
        switched_path=chosen_switched,
    )
    output: dict[str, object] = dict(chosen_terms)
    if runner_up is None or runner_up_score is None:
        output.update(
            {
                "xai_runner_up_relative_path": "",
                "xai_runner_up_score_ms": np.nan,
                "xai_score_margin_ms": np.nan,
                "xai_counterfactual_reason": "single_candidate",
            }
        )
        return output

    runner_terms = explain_candidate_score(
        runner_up,
        selected_score=runner_up_score,
        sort_column=sort_column,
        switch_penalty_ms=switch_penalty_ms,
        switched_path=runner_up_switched,
    )
    component_deltas = {
        name.removeprefix("xai_").removesuffix("_component_ms"): float(
            runner_terms[name] - chosen_terms[name]
        )
        for name in XAI_COMPONENT_COLUMNS
    }
    positive_deltas = {k: v for k, v in component_deltas.items() if v > 0.0}
    if positive_deltas:
        reason_component = max(positive_deltas, key=positive_deltas.get)
        reason = f"selected_lower_{reason_component}"
    else:
        reason = "selected_lower_combined_score"

    output.update(
        {
            "xai_runner_up_relative_path": str(runner_up.get("relative_path", "")),
            "xai_runner_up_score_ms": float(runner_up_score),
            "xai_score_margin_ms": float(runner_up_score - selected_score),
            "xai_counterfactual_reason": reason,
        }
    )
    for component, delta in component_deltas.items():
        output[f"xai_delta_runner_minus_selected_{component}_ms"] = delta
    return output


def summarize_xai_attribution(decisions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized score-term attribution from decision rows."""

    if decisions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["policy_name"]
    if "scenario_name" in decisions.columns:
        group_columns.insert(0, "scenario_name")

    for keys, frame in decisions.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        denominator = frame[XAI_COMPONENT_COLUMNS].abs().sum(axis=1).replace(0.0, np.nan)
        for column in XAI_COMPONENT_COLUMNS:
            row[column.replace("xai_", "mean_attr_").replace("_component_ms", "")] = float(
                (frame[column].abs() / denominator).mean(skipna=True)
            )
        row["decision_count"] = int(len(frame))
        row["mean_score_margin_ms"] = float(frame["xai_score_margin_ms"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
