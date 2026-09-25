"""Regression tests for the reviewer-facing cadence and QoS-rule audit."""

from __future__ import annotations

import pandas as pd

from scripts.audit_cadence_qos_and_availability import (
    FILTER_RANK,
    REACTIVE,
    SHIELD,
    build_qos_rule_ablation,
    build_unavailability_audit,
)
from scripts.build_operator_collection_plan import build_collection_plan


def test_qos_rule_ablation_counts_protection_and_opportunity() -> None:
    candidates = pd.DataFrame(
        {
            "session_bin_index": [1, 1, 1, 2, 2, 2],
            "relative_path": ["a", "b", "c", "a", "b", "c"],
            "latency_mean_ms": [30.0, 50.0, 90.0, 40.0, 80.0, 90.0],
            "is_feasible_path": [1, 1, 1, 1, 1, 1],
            "qos_shield_mode": ["mixed_qos_safeguard"] * 6,
        }
    )
    rows = []
    for policy, choices, outcomes in (
        (SHIELD, ("a", "a"), (1, 1)),
        (FILTER_RANK, ("b", "a"), (0, 1)),
        (REACTIVE, ("a", "a"), (1, 1)),
    ):
        for epoch, path, success in zip((1, 2), choices, outcomes, strict=True):
            rows.append(
                {
                    "session_bin_index": epoch,
                    "policy_name": policy,
                    "chosen_relative_path": path,
                    "success_under_budget": success,
                    "realized_next_latency_ms": 40.0 if success else 100.0,
                }
            )
    audit = build_qos_rule_ablation(
        candidates,
        pd.DataFrame(rows),
        cadence_label="test",
        key_columns=(),
    )
    mixed_multi = audit.loc[
        audit["scope"].eq("mixed_with_multiple_compliant")
    ].iloc[0]
    assert mixed_multi["decision_count"] == 1
    assert mixed_multi["different_action_count"] == 1
    assert mixed_multi["shield_only_success_count"] == 1
    assert mixed_multi["filter_then_rank_only_success_count"] == 0
    assert mixed_multi["shield_current_violation_count"] == 0
    assert mixed_multi["filter_then_rank_current_violation_count"] == 0


def test_unavailability_audit_partitions_scheduled_slots() -> None:
    trace = pd.DataFrame({"bin_epoch": [1, 2, 4, 5, 6, 7, 8]})
    metadata = {
        "split_boundary_audit": {
            "partitions": {
                "test": {
                    "first_bin_epoch": 1,
                    "last_bin_epoch": 10,
                    "scheduled_decision_epoch_count": 10,
                }
            }
        }
    }

    def decisions(count: int, no_action: int = 0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "policy_name": [REACTIVE] * count,
                "no_feasible_candidate": [1] * no_action + [0] * (count - no_action),
            }
        )

    audit = build_unavailability_audit(
        trace=trace,
        full_metadata=metadata,
        full_decisions=decisions(6, no_action=1),
        strict_decisions=decisions(4),
    ).set_index("alignment_case")
    assert audit.loc["full", "scheduled_raw_gap_epochs"] == 3
    assert (
        audit.loc[
            "full", "observed_but_no_complete_exact_target_or_candidate_set"
        ]
        == 1
    )
    assert audit.loc["strict_100ms_skew", "additional_skew_excluded_epochs"] == 2
    assert audit.loc["full", "retained_evaluable_action_decisions"] == 5
    assert audit.loc["full", "unknown_or_no_action_count"] == 5
    assert audit.loc["strict_100ms_skew", "unknown_or_no_action_count"] == 6


def test_practical_collection_plan_matches_frozen_profile() -> None:
    plan = build_collection_plan()
    one_use = plan.loc[
        plan["planned_gate_uses"].eq(1)
        & plan["assumed_opportunity_group_prevalence"].eq(0.75)
        & plan["parallel_independent_sites_per_day"].eq(50)
    ].iloc[0]
    assert one_use["zero_harm_success_group_floor"] == 228
    assert one_use["best_case_tail_group_floor"] == 721
    assert one_use["joint_opportunity_bearing_group_floor"] == 721
    assert one_use["required_independent_campaign_groups"] == 962
    assert one_use["minimum_collection_days"] == 20
