"""Checks for reviewer-requested campaign power and profile separation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from open_leo_latency_routing.evaluation.campaign_independence import (
    audit_campaign_manifest,
)
from scripts.run_synthetic_operator_power import (
    generate_episodes,
    synthetic_config,
)
from scripts.simulate_campaign_admission_power import (
    PowerScenario,
    practical_config,
    run_power_simulation,
)
from scripts.simulate_structured_campaign_admission import (
    SCENARIOS as STRUCTURED_SCENARIOS,
    generate_campaign_sample,
    run_structured_power,
    wilson_binomial_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_profiles_are_explicit_and_not_pooled() -> None:
    profiles = yaml.safe_load(
        (REPO_ROOT / "configs/evidence_profiles.yaml").read_text(encoding="utf-8")
    )
    assert profiles["cross_profile_pooling"] == "prohibited"
    measured = profiles["profiles"]["commect_measured_60ms"]
    synthetic = profiles["profiles"]["synthetic_operator_150ms"]
    assert measured["latency_target_ms"] == 60.0
    assert synthetic["latency_target_ms"] == 150.0
    assert measured["evidence_role"] != synthetic["evidence_role"]


def test_campaign_manifest_rejects_reused_start() -> None:
    valid = pd.DataFrame(
        {
            "campaign_id": ["a", "b"],
            "terminal_id": ["t1", "t1"],
            "site_id": ["s1", "s1"],
            "collection_date": ["2026-01-01", "2026-01-01"],
            "randomized_start_id": ["r1", "r2"],
        }
    )
    audit = audit_campaign_manifest(valid)
    assert audit.design_valid
    assert audit.repeated_terminal_site_day_count == 2

    invalid = valid.copy()
    invalid.loc[1, "randomized_start_id"] = "r1"
    with pytest.raises(ValueError, match="distinct randomized starts"):
        audit_campaign_manifest(invalid)


def test_small_power_run_is_deterministic_and_fail_closed() -> None:
    scenario = PowerScenario(
        "unsafe_test",
        40.0,
        "Candidate is slower for a regression test.",
    )
    first_detail, first_summary, _ = run_power_simulation(
        repetitions=2,
        seed=77,
        group_counts=(40,),
        scenarios=(scenario,),
    )
    second_detail, second_summary, _ = run_power_simulation(
        repetitions=2,
        seed=77,
        group_counts=(40,),
        scenarios=(scenario,),
    )
    pd.testing.assert_frame_equal(first_detail, second_detail)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert first_summary["admissions"].item() == 0


def test_profile_configs_match_declared_targets() -> None:
    measured = practical_config(1)
    synthetic = synthetic_config(1)
    assert measured.cvar_quantile == synthetic.cvar_quantile == 0.90
    assert measured.latency_cap_ms == synthetic.latency_cap_ms == 250.0
    current, indicator, next_rtt = generate_episodes(
        __import__("numpy").random.default_rng(4),
        3,
    )
    assert current.shape == indicator.shape == next_rtt.shape == (3, 3)


def test_structured_campaign_generator_exercises_declared_conditions() -> None:
    arrays, metadata = generate_campaign_sample(
        __import__("numpy").random.default_rng(123),
        group_count=20,
        scenario=STRUCTURED_SCENARIOS[0],
        base_missing_probability=0.10,
    )
    assert metadata["minimum_campaign_length"] < metadata["maximum_campaign_length"]
    assert metadata["complete_case_count"] < metadata["scheduled_epoch_count"]
    assert metadata["mixed_current_snapshot_count"] > 0
    assert metadata["uniform_current_snapshot_count"] > 0
    assert arrays["generic"].any()
    assert arrays["active"].any()
    assert len(__import__("numpy").unique(arrays["group_ids"])) == 20


def test_structured_power_is_deterministic_and_rejects_unsafe_candidate() -> None:
    unsafe = (STRUCTURED_SCENARIOS[-1],)
    first_detail, first_summary, _ = run_structured_power(
        repetitions=2,
        seed=91,
        group_counts=(30,),
        scenarios=unsafe,
    )
    second_detail, second_summary, _ = run_structured_power(
        repetitions=2,
        seed=91,
        group_counts=(30,),
        scenarios=unsafe,
    )
    pd.testing.assert_frame_equal(first_detail, second_detail)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert first_summary["sign_gate_admission_rate"].item() == 0.0
    assert first_summary["magnitude_gate_admission_rate"].item() == 0.0
    assert first_summary["sign_gate_admissions"].item() == 0
    assert first_summary["sign_gate_admission_ci95_low"].item() == 0.0
    assert 0.0 < first_summary["sign_gate_admission_ci95_high"].item() < 1.0


def test_wilson_interval_reports_uncertainty_at_extreme_admission_counts() -> None:
    zero_low, zero_high = wilson_binomial_interval(0, 30)
    full_low, full_high = wilson_binomial_interval(30, 30)
    assert zero_low == 0.0
    assert zero_high == pytest.approx(0.113513, abs=1e-6)
    assert full_low == pytest.approx(0.886487, abs=1e-6)
    assert full_high == 1.0


def test_fixed_total_design_keeps_scheduled_epoch_budget() -> None:
    detail, _, _ = run_structured_power(
        repetitions=1,
        seed=92,
        group_counts=(25, 100),
        scenarios=(STRUCTURED_SCENARIOS[0],),
        fixed_total_epochs=1_000,
    )
    assert detail["scheduled_epoch_count"].eq(1_000).all()
