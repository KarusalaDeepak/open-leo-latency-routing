from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from open_leo_latency_routing.models.forecast_baselines import select_ridge_alpha
from open_leo_latency_routing.parameter_registry import (
    assess_evidence_tier,
    load_parameter_registry,
)
from open_leo_latency_routing.evaluation.campaign_qualification import (
    load_campaign_qualification,
    require_unique_gate_campaign_id,
    summarize_gate_qualified_evidence,
)
import run_service_path_experiments as service_exp


def test_active_parameter_registry_has_provenance_and_claim_boundaries():
    registry = load_parameter_registry(ROOT / "configs" / "parameter_registry.yaml")
    rows = {row["name"]: row for row in registry["parameters"]}
    required = {
        "ridge_alpha",
        "ensemble_uncertainty_multiplier",
        "service_reply_pressure_weight",
        "service_volatility_weight",
        "consensus_weights",
        "calibrated_fusion_weights",
        "residual_risk_weights",
        "residual_risk_gate_quantile",
        "ridge_development_fraction",
        "disagreement_only_multiplier",
        "gaussian_cvar_proxy_multiplier",
        "age_residual_quantile",
        "practical_cvar_gain_ms",
        "switching_penalty_profiles_ms",
        "stochastic_switching_profile",
        "exact_lag_steps",
        "outer_policy_split",
        "minimum_opportunity_bearing_groups",
    }
    assert required.issubset(rows)
    assert rows["ridge_alpha"]["selection_partition"] == "training_internal_development"
    assert "not an IEEE rule" not in rows["minimum_opportunity_bearing_groups"]["rationale"].lower()
    assert "count alone" in rows["minimum_opportunity_bearing_groups"]["claim_boundary"].lower()


def test_ridge_alpha_is_selected_inside_training_partition():
    frame = pd.DataFrame(
        {
            "bin_epoch": list(range(30)),
            "x": [float(value) for value in range(30)],
            "target_next": [2.0 * value + 1.0 for value in range(30)],
        }
    )
    result = select_ridge_alpha(
        frame,
        ["x"],
        alpha_grid=(0.0, 1.0, 100.0),
        development_fraction=0.20,
    )
    assert result.selected_alpha in {0.0, 1.0, 100.0}
    assert result.fit_rows + result.development_rows == len(frame)
    assert len(result.candidate_scores) == 3


def test_ensemble_multiplier_uses_development_outcomes_not_test_interface():
    frame = pd.DataFrame(
        {
            "session_bin_index": [0, 0, 1, 1],
            "path_state": ["active"] * 4,
            "observed_replies": [10] * 4,
            "target_next": [100.0, 20.0, 100.0, 20.0],
        }
    )
    mean = pd.Series([10.0, 20.0, 10.0, 20.0])
    spread = pd.Series([100.0, 0.0, 100.0, 0.0])
    pressure = pd.Series([0.0] * 4)
    selected, audit = service_exp._select_ensemble_uncertainty_multiplier(
        frame,
        mean,
        spread,
        pressure,
        latency_budget_ms=60.0,
        multiplier_grid=(0.0, 0.5),
    )
    assert selected == pytest.approx(0.5)
    assert {row["multiplier"] for row in audit} == {0.0, 0.5}


def test_campaign_tiers_do_not_turn_five_into_a_sufficiency_rule():
    one = assess_evidence_tier(["commect"])
    assert one.tier == "proof_of_method"
    assert not one.deployment_release_permitted

    two = assess_evidence_tier(
        ["commect", "nuwins_alaska"],
        acquisition_family_ids=["commect", "nuwins"],
        frozen_external_test=True,
        paired_opportunities_present=True,
        campaign_block_precision_passed=True,
        calibration_passed=True,
        shift_check_passed=True,
        execution_cost_check_passed=True,
    )
    assert two.tier == "frozen_external_validation"
    assert two.deployment_release_permitted

    five_without_quality = assess_evidence_tier(
        ["c1", "c2", "c3", "c4", "c5"],
        leave_one_campaign_out=True,
    )
    assert five_without_quality.tier == "conservative_broader_generalization_profile"
    assert not five_without_quality.deployment_release_permitted


def test_only_synchronized_opportunity_audited_campaigns_are_counted():
    manifest = load_campaign_qualification(
        ROOT / "configs" / "campaign_qualification.yaml"
    )
    summary = summarize_gate_qualified_evidence(manifest)
    assert summary["qualified_campaign_count"] == 6
    assert summary["qualified_acquisition_family_count"] == 2
    assert summary["qualified_campaign_ids"] == (
        "commect_denmark_drive",
        "conext23_20230424",
        "conext23_20230425",
        "conext23_20230426",
        "conext23_20230427",
        "conext23_20230428",
    )
    assert {"NUWiNS", "LENS", "WetLinks", "FjordLink"}.issubset(
        set(summary["nonqualified_sources"])
    )

    fjordlink = next(row for row in manifest["sources"] if row["source"] == "FjordLink")
    assert fjordlink["local_audit"]["status"] == "completed"
    assert fjordlink["local_audit"]["starlink_sessions"] == 51
    assert fjordlink["local_audit"]["terrestrial_sessions"] == 15
    assert fjordlink["local_audit"]["qualifying_overlaps"] == 0
    assert fjordlink["local_audit"]["nearest_interval_gap_s"] == pytest.approx(149.379)
    assert not fjordlink["gate_qualified"]


def test_production_campaign_id_is_manifest_verified_and_fail_closed():
    manifest = load_campaign_qualification(
        ROOT / "configs" / "campaign_qualification.yaml"
    )
    assert require_unique_gate_campaign_id(
        manifest,
        source="COMMECT",
    ) == "commect_denmark_drive"

    ambiguous = {**manifest, "sources": [*manifest["sources"], manifest["sources"][0].copy()]}
    with pytest.raises(ValueError, match="exactly one"):
        require_unique_gate_campaign_id(ambiguous, source="COMMECT")
    with pytest.raises(ValueError, match="exactly one"):
        require_unique_gate_campaign_id(manifest, source="UNKNOWN")


def test_nondefault_selection_manifest_propagates_to_runner_kwargs():
    config = {
        "optimization": {
            "hyperparameter_selection": {
                "ridge_alpha_grid": [0.2, 2.0],
                "ridge_development_fraction": 0.25,
                "ensemble_multiplier_grid": [0.1, 0.9],
                "consensus_temporal_weight_grid": [0.4, 0.6],
                "consensus_disagreement_penalty_grid": [0.05, 0.15],
            },
            "disagreement_aware": {
                "gate_validation_quantile": 0.85,
                "calibration_coverage": 0.95,
            },
        }
    }
    kwargs = service_exp.selection_kwargs_from_config(config)
    assert kwargs == {
        "ridge_alpha_grid": (0.2, 2.0),
        "ridge_development_fraction": 0.25,
        "ensemble_multiplier_grid": (0.1, 0.9),
        "consensus_temporal_weight_grid": (0.4, 0.6),
        "consensus_disagreement_penalty_grid": (0.05, 0.15),
        "residual_risk_gate_quantile": 0.85,
        "conformal_coverage": 0.95,
    }
