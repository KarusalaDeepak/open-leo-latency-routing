"""Focused checks for the prospective evidence-gate design artifact."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.audit_gate_design_sensitivity import (
    CANONICAL_OUTPUT_NAME,
    DEFAULT_CONFIG,
    MAIN_OUTPUT_NAME,
    METADATA_OUTPUT_NAME,
    best_case_cvar_group_floor,
    build_canonical_reference_rows,
    build_requested_rows,
    endpoint_alpha,
    generate_artifact,
    grid_contract,
    load_config,
    zero_harm_success_group_floor,
)
from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    _block_weights,
    _bounded_block_cvar_interval,
    _exact_harmful_block_lcb,
    _observation_weights,
    select_opportunity_aware_risk_controlled_policy,
)


def _singleton_blocks(groups: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray([index], dtype=int) for index in range(groups))


def test_requested_table_is_exact_cartesian_product() -> None:
    config = load_config(DEFAULT_CONFIG)
    rows = build_requested_rows(config)
    assert len(rows) == 108
    combinations = {
        (
            row["n_min_opportunity_bearing_groups"],
            row["epsilon_aggregate"],
            row["practical_cvar_gain_ms"],
            row["latency_cap_ms"],
        )
        for row in rows
    }
    assert len(combinations) == 108
    assert {row["n_min_opportunity_bearing_groups"] for row in rows} == {
        5,
        10,
        20,
        50,
    }
    assert {row["epsilon_aggregate"] for row in rows} == {0.01, 0.02, 0.05}
    assert {row["practical_cvar_gain_ms"] for row in rows} == {1.0, 5.0, 10.0}
    assert {row["latency_cap_ms"] for row in rows} == {60.0, 200.0, 1000.0}
    assert all(row["epsilon_aggregate"] == row["epsilon_opportunity"] for row in rows)
    assert all(row["measured_admitted"] is False for row in rows)
    assert all(row["configuration_selection_allowed"] is False for row in rows)


def test_zero_harm_floor_matches_production_exact_bound() -> None:
    config = load_config(DEFAULT_CONFIG)
    for planned_uses in (1, 5):
        alpha_e = endpoint_alpha(config, planned_uses)
        for epsilon in (0.01, 0.02, 0.05):
            floor = zero_harm_success_group_floor(epsilon, alpha_e)
            for groups, expected_pass in ((floor - 1, False), (floor, True)):
                values = np.zeros(groups, dtype=float)
                blocks = _singleton_blocks(groups)
                _, harmful, probability_ucb, lcb = _exact_harmful_block_lcb(
                    values,
                    alpha_e,
                    blocks,
                )
                assert harmful == 0
                assert math.isclose(lcb, -probability_ucb)
                assert (lcb >= -epsilon) is expected_pass


def test_tail_floor_matches_production_bound_at_integer_boundary() -> None:
    config = load_config(DEFAULT_CONFIG)
    q = float(config["statistical_contract"]["cvar_quantile"])
    spacing = float(config["statistical_contract"]["cvar_grid_spacing_ms"])
    for planned_uses in (1, 5):
        alpha_e = endpoint_alpha(config, planned_uses)
        for cap in (60.0, 200.0, 1000.0):
            points, _, correction = grid_contract(cap, spacing, q)
            for delta in (1.0, 5.0, 10.0):
                floor = best_case_cvar_group_floor(
                    delta,
                    latency_cap_ms=cap,
                    quantile=q,
                    grid_points=points,
                    grid_correction_ms=correction,
                    alpha_endpoint=alpha_e,
                )
                for groups, expected_pass in ((floor - 1, False), (floor, True)):
                    blocks = _singleton_blocks(groups)
                    block_weights = _block_weights(blocks)
                    observation_weights = _observation_weights(
                        blocks,
                        block_weights,
                        groups,
                    )
                    reactive = np.full(groups, cap, dtype=float)
                    candidate = np.zeros(groups, dtype=float)
                    _, reactive_lcb, _, _ = _bounded_block_cvar_interval(
                        reactive,
                        q,
                        cap,
                        points,
                        alpha_e,
                        blocks,
                        block_weights,
                        observation_weights,
                    )
                    _, _, candidate_ucb, _ = _bounded_block_cvar_interval(
                        candidate,
                        q,
                        cap,
                        points,
                        alpha_e,
                        blocks,
                        block_weights,
                        observation_weights,
                    )
                    assert ((reactive_lcb - candidate_ucb) >= delta) is expected_pass


def test_single_measured_group_abstains_for_every_requested_n_min() -> None:
    config = load_config(DEFAULT_CONFIG)
    for n_min in config["requested_grid"]["minimum_opportunity_bearing_groups"]:
        selection = select_opportunity_aware_risk_controlled_policy(
            realized_selection_latency={
                "reactive": [60.0],
                "candidate_a": [0.0],
                "candidate_b": [0.0],
            },
            opportunity_mask=np.asarray([True]),
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                alpha=0.05,
                noninferiority_margin=0.05,
                opportunity_noninferiority_margin=0.05,
                minimum_effective_opportunities=float(n_min),
                practical_cvar_gain_ms=1.0,
                cvar_quantile=0.95,
                latency_cap_ms=60.0,
                cvar_grid_points=1201,
                planned_gate_uses=1,
            ),
            reactive_policy="reactive",
            independence_group_ids=np.asarray(["one_drive"]),
        )
        assert selection.selected_policy == "reactive"
        evidence = selection.evidence_frame()
        assert not evidence.loc[
            evidence["policy"].eq("candidate_a"), "opportunity_sufficient"
        ].item()


def test_generated_outputs_and_canonical_reference(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    requested, canonical, metadata = generate_artifact(config, tmp_path)
    assert len(requested) == 108
    assert len(canonical) == 3
    assert {row["latency_cap_ms"] for row in canonical} == {60_000.0}
    assert {row["practical_cvar_gain_ms"] for row in canonical} == {
        1.0,
        5.0,
        10.0,
    }
    assert metadata["requested_cartesian_row_count"] == 108
    assert metadata["canonical_reference_row_count"] == 3
    assert metadata["measured_efficacy_evidence"] is False
    assert metadata["configuration_selection_allowed"] is False
    assert "do not choose" in str(metadata["warning"]).lower()

    with (tmp_path / MAIN_OUTPUT_NAME).open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == 108
    assert {row["measured_admitted"] for row in persisted} == {"false"}
    with (tmp_path / CANONICAL_OUTPUT_NAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 3
    disk_metadata = json.loads(
        (tmp_path / METADATA_OUTPUT_NAME).read_text(encoding="utf-8")
    )
    assert disk_metadata["output_sha256"] == metadata["output_sha256"]


def test_expected_reference_floor_values() -> None:
    config = load_config(DEFAULT_CONFIG)
    rows = build_requested_rows(config)
    index = {
        (row["latency_cap_ms"], row["practical_cvar_gain_ms"]): row
        for row in rows
        if row["n_min_opportunity_bearing_groups"] == 5
        and row["epsilon_aggregate"] == 0.02
    }
    assert index[(60.0, 1.0)]["u1_tail_best_case_group_floor"] == 2704
    assert index[(60.0, 10.0)]["u5_tail_best_case_group_floor"] == 4248
    assert index[(200.0, 5.0)]["u1_tail_best_case_group_floor"] == 2974
    assert index[(1000.0, 10.0)]["u5_tail_best_case_group_floor"] == 3530
    canonical = build_canonical_reference_rows(config)
    assert canonical[0]["u1_tail_best_case_group_floor"] == 3954
    assert canonical[0]["u5_tail_best_case_group_floor"] == 4276

