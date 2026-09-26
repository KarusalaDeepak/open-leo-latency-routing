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
