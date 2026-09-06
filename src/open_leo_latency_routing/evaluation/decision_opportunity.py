"""Audit whether a path-selection decision can change the QoS outcome.

Aggregate success can hide two uninformative regimes: every candidate succeeds,
or every candidate fails. This module separates those regimes from mixed
candidate sets, where selecting a different path can change deadline success.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _runtime_candidate_set(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Mirror the evaluator's feasibility rule for one decision epoch."""

    if "is_feasible_path" not in snapshot:
        return snapshot.copy(), False
    feasible = snapshot[snapshot["is_feasible_path"].astype(bool)].copy()
    if not feasible.empty:
        return feasible, False
    # The evaluator keeps all rows as an explicitly marked emergency fallback.
    return snapshot.copy(), True


def build_candidate_opportunity_audit(
    candidates: pd.DataFrame,
    thresholds_ms: tuple[float, ...] = (40.0, 60.0, 100.0, 200.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify epochs and summarize how often policy choice can matter.

    An epoch is an opportunity only when at least two runtime candidates are
    present and their next-epoch QoS outcomes differ. ``target_next`` is used
    only after execution as an evaluation label; it never enters online scores.
    """

    required = {"session_bin_index", "relative_path", "target_next"}
    missing = required.difference(candidates.columns)
    if missing:
        raise KeyError(f"opportunity audit is missing columns: {sorted(missing)}")

    labels: list[dict[str, object]] = []
    for epoch, snapshot in candidates.groupby("session_bin_index", sort=True):
        runtime, emergency = _runtime_candidate_set(snapshot)
        candidate_count = int(len(runtime))
        spread = (
            float(runtime["target_next"].max() - runtime["target_next"].min())
            if candidate_count
            else np.nan
        )
        for threshold in thresholds_ms:
            outcomes = runtime["target_next"].le(float(threshold))
            success_count = int(outcomes.sum())
            mixed = candidate_count >= 2 and 0 < success_count < candidate_count
            if emergency:
                regime = "emergency_no_current_feasible"
            elif candidate_count == 1:
                regime = "single_candidate"
            elif success_count == candidate_count:
                regime = "all_candidates_succeed"
            elif success_count == 0:
                regime = "all_candidates_fail"
            else:
                regime = "mixed_outcome_opportunity"
            labels.append(
                {
                    "session_bin_index": epoch,
                    "threshold_ms": float(threshold),
                    "runtime_candidate_count": candidate_count,
                    "emergency_fallback": int(emergency),
                    "candidate_success_count": success_count,
                    "candidate_latency_spread_ms": spread,
                    "best_available_success_with_runtime_fallback": int(success_count > 0),
                    # Unlike decision_opportunity, this label includes an
                    # emergency epoch when the retained candidates have mixed
                    # future outcomes. It supports a general pairwise bound.
                    "binary_outcome_discriminative": int(mixed),
                    "decision_opportunity": int(mixed and not emergency),
                    "outcome_regime": regime,
                }
            )

    label_frame = pd.DataFrame(labels)
    summary_rows: list[dict[str, object]] = []
    regimes = (
        "all_candidates_succeed",
        "mixed_outcome_opportunity",
        "all_candidates_fail",
        "single_candidate",
        "emergency_no_current_feasible",
    )
    for threshold, frame in label_frame.groupby("threshold_ms", sort=True):
        counts = frame["outcome_regime"].value_counts()
        total = int(len(frame))
        row: dict[str, object] = {
            "threshold_ms": float(threshold),
            "decision_epoch_count": total,
            "decision_opportunity_count": int(frame["decision_opportunity"].sum()),
            "decision_opportunity_rate": float(frame["decision_opportunity"].mean()),
            "binary_outcome_discriminative_count": int(
                frame["binary_outcome_discriminative"].sum()
            ),
            "binary_outcome_discriminative_rate": float(
                frame["binary_outcome_discriminative"].mean()
            ),
            "best_available_success_with_runtime_fallback": float(
                frame["best_available_success_with_runtime_fallback"].mean()
            ),
            "median_runtime_candidate_count": float(
                frame["runtime_candidate_count"].median()
            ),
            "median_candidate_latency_spread_ms": float(
                frame["candidate_latency_spread_ms"].median()
            ),
        }
        for regime in regimes:
            count = int(counts.get(regime, 0))
            row[f"{regime}_count"] = count
            row[f"{regime}_rate"] = count / total if total else np.nan
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), label_frame


def build_opportunity_conditioned_results(
    decisions: pd.DataFrame,
    opportunity_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Report policy success specifically on mixed-outcome decision epochs."""

    required = {
        "session_bin_index",
        "policy_name",
        "realized_next_latency_ms",
        "decision_gap_ms",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise KeyError(f"conditioned results are missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for threshold, labels in opportunity_labels.groupby("threshold_ms", sort=True):
        opportunities = labels[labels["decision_opportunity"].astype(bool)]
        opportunity_ids = set(opportunities["session_bin_index"])
        best_available_rate = float(
            labels["best_available_success_with_runtime_fallback"].mean()
        )
        for policy, frame in decisions.groupby("policy_name", sort=False):
            selected = frame[frame["session_bin_index"].isin(opportunity_ids)]
            success = selected["realized_next_latency_ms"].le(float(threshold))
            rows.append(
                {
                    "threshold_ms": float(threshold),
                    "policy_name": policy,
                    "decision_count": int(len(frame)),
                    "overall_success_rate": float(
                        frame["realized_next_latency_ms"].le(float(threshold)).mean()
                    ),
                    "best_available_success_with_runtime_fallback": best_available_rate,
                    "opportunity_count": int(len(selected)),
                    "opportunity_capture_rate": float(success.mean()) if len(selected) else np.nan,
                    "missed_opportunity_count": int((~success).sum()) if len(selected) else 0,
                    "mean_gap_on_opportunities_ms": (
                        float(selected["decision_gap_ms"].mean())
                        if len(selected)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_success_gap_bounds(
    decisions: pd.DataFrame,
    opportunity_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the finite-sample success-gap identifiability bound.

    For policies choosing from the same runtime candidate set, their binary
    QoS outcomes can differ only in an epoch containing both a successful and
    an unsuccessful runtime candidate. Therefore the absolute empirical
    success-rate difference is upper-bounded by the fraction of such epochs.
    """

    required_decisions = {
        "session_bin_index",
        "policy_name",
        "realized_next_latency_ms",
    }
    missing = required_decisions.difference(decisions.columns)
    if missing:
        raise KeyError(f"success-gap bound is missing decision columns: {sorted(missing)}")
    required_labels = {
        "session_bin_index",
        "threshold_ms",
        "binary_outcome_discriminative",
    }
    missing = required_labels.difference(opportunity_labels.columns)
    if missing:
        raise KeyError(f"success-gap bound is missing label columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    tolerance = 1e-12
    for threshold, labels in opportunity_labels.groupby("threshold_ms", sort=True):
        bound = float(labels["binary_outcome_discriminative"].mean())
        wide = decisions.pivot_table(
            index="session_bin_index",
            columns="policy_name",
            values="realized_next_latency_ms",
            aggfunc="first",
        )
        for policy_a, policy_b in combinations(wide.columns, 2):
            pair = wide[[policy_a, policy_b]].dropna()
            success_a = pair[policy_a].le(float(threshold)).astype(float)
            success_b = pair[policy_b].le(float(threshold)).astype(float)
            gap = float(abs(success_a.mean() - success_b.mean()))
            rows.append(
                {
                    "threshold_ms": float(threshold),
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "common_decision_count": int(len(pair)),
                    "absolute_success_rate_gap": gap,
                    "discriminative_epoch_rate_bound": bound,
                    "bound_slack": bound - gap,
                    "bound_holds": bool(gap <= bound + tolerance),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty and not result["bound_holds"].all():
        failures = result[~result["bound_holds"]]
        raise AssertionError("pairwise success-gap bound failed:\n" + failures.to_string(index=False))
    return result


def build_policy_choice_agreement(decisions: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise selected-path agreement on common decision epochs."""

    required = {"session_bin_index", "policy_name", "chosen_relative_path"}
    missing = required.difference(decisions.columns)
    if missing:
        raise KeyError(f"agreement audit is missing columns: {sorted(missing)}")
    wide = decisions.pivot_table(
        index="session_bin_index",
        columns="policy_name",
        values="chosen_relative_path",
        aggfunc="first",
    )
    rows: list[dict[str, object]] = []
    for policy_a, policy_b in combinations(wide.columns, 2):
        pair = wide[[policy_a, policy_b]].dropna()
        rows.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "common_decision_count": int(len(pair)),
                "selected_path_agreement_rate": (
                    float(pair[policy_a].eq(pair[policy_b]).mean())
                    if len(pair)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
