#!/usr/bin/env python3
"""Audit cadence, mixed-snapshot QoS filtering, and missing outcomes.

The audit reads frozen policy outputs. It never refits a model, changes a
candidate, or re-runs the evidence gate, so its tables are explanatory rather
than another policy-selection stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REACTIVE = "reactive_greedy"
SHIELD = "qos_shielded_operational_selector"
FILTER_RANK = "qos_filter_then_ensemble_selector"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _policy_row(summary: pd.DataFrame, policy: str) -> pd.Series:
    rows = summary.loc[summary["policy_name"].eq(policy)]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {policy!r}, found {len(rows)}")
    return rows.iloc[0]


def build_cadence_comparison(
    cases: tuple[tuple[str, float, Path], ...],
) -> pd.DataFrame:
    """Compare cadences under the same five-fold rolling protocol."""

    rows: list[dict[str, object]] = []
    for label, cadence_seconds, directory in cases:
        summary = pd.read_csv(directory / "rolling_policy_summary.csv")
        manifest = pd.read_csv(directory / "rolling_fold_manifest.csv")
        reactive = _policy_row(summary, REACTIVE)
        shield = _policy_row(summary, SHIELD)
        filter_rank = _policy_row(summary, FILTER_RANK)
        rows.append(
            {
                "cadence_label": label,
                "decision_interval_seconds": cadence_seconds,
                "next_bin_prediction_horizon_seconds": cadence_seconds,
                "aggregation_window_seconds": cadence_seconds,
                "fold_count": int(manifest["rolling_fold"].nunique()),
                "training_rows_min": int(manifest["training_rows"].min()),
                "training_rows_max": int(manifest["training_rows"].max()),
                "scheduled_test_epochs": int(
                    manifest["scheduled_test_decision_epochs"].sum()
                ),
                "retained_test_decisions": int(reactive["decision_count"]),
                "reactive_success": float(reactive["success_rate_under_60ms"]),
                "shield_success": float(shield["success_rate_under_60ms"]),
                "shield_minus_reactive_success": float(
                    shield["success_rate_under_60ms"]
                    - reactive["success_rate_under_60ms"]
                ),
                "filter_then_rank_success": float(
                    filter_rank["success_rate_under_60ms"]
                ),
                "filter_then_rank_minus_reactive_success": float(
                    filter_rank["success_rate_under_60ms"]
                    - reactive["success_rate_under_60ms"]
                ),
                "reactive_mean_rtt_ms": float(reactive["mean_realized_latency_ms"]),
                "shield_mean_rtt_ms": float(shield["mean_realized_latency_ms"]),
                "filter_then_rank_mean_rtt_ms": float(
                    filter_rank["mean_realized_latency_ms"]
                ),
                "interpretation_boundary": (
                    "cadence changes aggregation, physical forecast horizon, and "
                    "retained sample count together; this is a matched-protocol "
                    "comparison, not a causal decomposition"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_qos_rule_ablation(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    cadence_label: str,
    key_columns: tuple[str, ...],
    latency_budget_ms: float = 60.0,
) -> pd.DataFrame:
    """Compare reactive-in-mixed shielding with rank-among-compliant filtering."""

    keys = [*key_columns, "session_bin_index"]
    required_candidates = {
        *keys,
        "relative_path",
        "latency_mean_ms",
        "is_feasible_path",
        "qos_shield_mode",
    }
    required_decisions = {
        *keys,
        "policy_name",
        "chosen_relative_path",
        "success_under_budget",
        "realized_next_latency_ms",
    }
    missing_candidates = required_candidates.difference(candidates.columns)
    missing_decisions = required_decisions.difference(decisions.columns)
    if missing_candidates or missing_decisions:
        raise KeyError(
            "QoS-rule audit missing columns: "
            f"candidates={sorted(missing_candidates)}, "
            f"decisions={sorted(missing_decisions)}"
        )

    candidate = candidates.copy()
    candidate["currently_compliant"] = (
        candidate["is_feasible_path"].astype(bool)
        & candidate["latency_mean_ms"].astype(float).le(latency_budget_ms)
    )
    epoch = (
        candidate.groupby(keys, sort=False)
        .agg(
            feasible_path_count=("is_feasible_path", "sum"),
            compliant_path_count=("currently_compliant", "sum"),
            qos_shield_mode=("qos_shield_mode", "first"),
        )
        .reset_index()
    )
    path_state = candidate[
        [*keys, "relative_path", "latency_mean_ms", "currently_compliant"]
    ].copy()

    def selected(policy: str, prefix: str) -> pd.DataFrame:
        frame = decisions.loc[decisions["policy_name"].eq(policy)].copy()
        columns = [
            *keys,
            "chosen_relative_path",
            "success_under_budget",
            "realized_next_latency_ms",
        ]
        frame = frame[columns].rename(
            columns={
                "chosen_relative_path": f"{prefix}_path",
                "success_under_budget": f"{prefix}_success",
                "realized_next_latency_ms": f"{prefix}_latency_ms",
            }
        )
        selected_state = path_state.rename(
            columns={
                "relative_path": f"{prefix}_path",
                "latency_mean_ms": f"{prefix}_current_latency_ms",
                "currently_compliant": f"{prefix}_currently_compliant",
            }
        )
        return frame.merge(selected_state, on=[*keys, f"{prefix}_path"], how="left")

    paired = epoch.merge(selected(SHIELD, "shield"), on=keys, validate="one_to_one")
    paired = paired.merge(
        selected(FILTER_RANK, "filter_rank"),
        on=keys,
        validate="one_to_one",
    )
    paired["different_choice"] = paired["shield_path"].ne(
        paired["filter_rank_path"]
    )
    paired["shield_only_success"] = paired["shield_success"].gt(
        paired["filter_rank_success"]
    )
    paired["filter_rank_only_success"] = paired["filter_rank_success"].gt(
        paired["shield_success"]
    )

    scopes = {
        "all_retained": pd.Series(True, index=paired.index),
        "mixed_snapshot": paired["qos_shield_mode"].eq("mixed_qos_safeguard"),
        "mixed_with_multiple_compliant": (
            paired["qos_shield_mode"].eq("mixed_qos_safeguard")
            & paired["compliant_path_count"].ge(2)
        ),
        "different_action": paired["different_choice"],
    }
    rows: list[dict[str, object]] = []
    for scope, mask in scopes.items():
        frame = paired.loc[mask]
        count = int(len(frame))
        rows.append(
            {
                "cadence_label": cadence_label,
                "scope": scope,
                "decision_count": count,
                "different_action_count": int(frame["different_choice"].sum()),
                "shield_success_rate": (
                    float(frame["shield_success"].mean()) if count else math.nan
                ),
                "filter_then_rank_success_rate": (
                    float(frame["filter_rank_success"].mean()) if count else math.nan
                ),
                "filter_then_rank_minus_shield_success": (
                    float(
                        frame["filter_rank_success"].mean()
                        - frame["shield_success"].mean()
                    )
                    if count
                    else math.nan
                ),
                "shield_only_success_count": int(frame["shield_only_success"].sum()),
                "filter_then_rank_only_success_count": int(
                    frame["filter_rank_only_success"].sum()
                ),
                "shield_mean_rtt_ms": (
                    float(frame["shield_latency_ms"].mean()) if count else math.nan
                ),
                "filter_then_rank_mean_rtt_ms": (
                    float(frame["filter_rank_latency_ms"].mean())
                    if count
                    else math.nan
                ),
                "shield_current_violation_count": int(
                    (~frame["shield_currently_compliant"]).sum()
                ),
                "filter_then_rank_current_violation_count": int(
                    (~frame["filter_rank_currently_compliant"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_unavailability_audit(
    *,
    trace: pd.DataFrame,
    full_metadata: dict[str, object],
    full_decisions: pd.DataFrame,
    strict_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Partition scheduled test slots into observable exclusion reasons."""

    partition = full_metadata["split_boundary_audit"]["partitions"]["test"]
    first_epoch = float(partition["first_bin_epoch"])
    last_epoch = float(partition["last_bin_epoch"])
    scheduled = int(partition["scheduled_decision_epoch_count"])
    observed = int(
        trace.loc[trace["bin_epoch"].between(first_epoch, last_epoch), "bin_epoch"]
        .nunique()
    )

    def policy_count(decisions: pd.DataFrame) -> tuple[int, int]:
        rows = decisions.loc[decisions["policy_name"].eq(REACTIVE)]
        return int(len(rows)), int(rows["no_feasible_candidate"].sum())

    full_count, full_no_action = policy_count(full_decisions)
    strict_count, strict_no_action = policy_count(strict_decisions)
    raw_gap = scheduled - observed
    target_or_completeness = observed - full_count
    skew_excluded = full_count - strict_count
    rows = []
    for label, retained, skew, no_action in (
        ("full", full_count, 0, full_no_action),
        ("strict_100ms_skew", strict_count, skew_excluded, strict_no_action),
    ):
        unknown = scheduled - retained
        unknown_or_no_action = unknown + no_action
        rows.append(
            {
                "alignment_case": label,
                "scheduled_test_epochs": scheduled,
                "raw_observed_epochs": observed,
                "scheduled_raw_gap_epochs": raw_gap,
                "observed_but_no_complete_exact_target_or_candidate_set": (
                    target_or_completeness
                ),
                "additional_skew_excluded_epochs": skew,
                "retained_decision_slots": retained,
                "retained_evaluable_action_decisions": retained - no_action,
                "retained_no_feasible_action_count": no_action,
                "unknown_or_no_action_count": unknown_or_no_action,
                "unknown_or_no_action_fraction": unknown_or_no_action / scheduled,
                "interpretation": (
                    "raw gaps and incomplete exact next-bin concurrent outcomes "
                    "are evaluation unobservability; the strict case additionally "
                    "excludes excessive recorded cross-path timestamp skew"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rolling-1s",
        default="results/commect_rolling_1s_active_opportunity",
    )
    parser.add_argument(
        "--rolling-10s",
        default="results/commect_rolling_active_opportunity",
    )
    parser.add_argument(
        "--alignment",
        default="results/commect_1s_alignment_audit",
    )
    parser.add_argument(
        "--trace-1s",
        default="data/processed/commect_multiaccess_1s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cadence_qos_availability_audit",
    )
    args = parser.parse_args()

    rolling_1s = _resolve(args.rolling_1s)
    rolling_10s = _resolve(args.rolling_10s)
    alignment = _resolve(args.alignment)
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cadence = build_cadence_comparison(
        (("1s", 1.0, rolling_1s), ("10s", 10.0, rolling_10s))
    )
    cadence.to_csv(output / "cadence_policy_comparison.csv", index=False)

    ablations = []
    for label, directory, keys in (
        ("1s", rolling_1s, ("rolling_fold",)),
        ("10s", rolling_10s, ("rolling_fold",)),
    ):
        ablations.append(
            build_qos_rule_ablation(
                pd.read_csv(directory / "rolling_candidate_predictions.csv"),
                pd.read_csv(directory / "rolling_policy_decisions.csv"),
                cadence_label=label,
                key_columns=keys,
            )
        )
    pd.concat(ablations, ignore_index=True).to_csv(
        output / "qos_filter_rule_ablation.csv",
        index=False,
    )

    full = alignment / "full"
    strict = alignment / "strict"
    full_metadata = json.loads(
        (full / "validation_metadata.json").read_text(encoding="utf-8")
    )
    unavailable = build_unavailability_audit(
        trace=pd.read_csv(_resolve(args.trace_1s)),
        full_metadata=full_metadata,
        full_decisions=pd.read_csv(full / "policy_decisions.csv"),
        strict_decisions=pd.read_csv(strict / "policy_decisions.csv"),
    )
    unavailable.to_csv(output / "scheduled_unavailability_audit.csv", index=False)

    metadata = {
        "frozen_post_test_diagnostic": True,
        "candidate_or_gate_reselection_allowed": False,
        "cadence_claim_boundary": (
            "the matched rolling comparison changes aggregation interval, "
            "physical one-bin horizon, and sample count together"
        ),
        "qos_ablation": (
            "conservative reactive-in-mixed shielding versus predictive ranking "
            "restricted to currently compliant paths"
        ),
        "availability_semantics": (
            "scheduled slots excluded before replay remain in the conservative "
            "service denominator"
        ),
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(cadence.to_string(index=False))
    print(f"audit_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
