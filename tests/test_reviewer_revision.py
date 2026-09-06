"""Regression tests for reviewer-driven methodological changes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from open_leo_latency_routing.data.loaders import (
    assign_decision_groups,
    load_compatible_latency_trace,
)
from open_leo_latency_routing.features.temporal import (
    build_forecast_table,
    split_train_calibration_selection_test,
    split_train_val_test,
    split_group_holdout,
)
from open_leo_latency_routing.models.forecast_baselines import default_feature_columns
from open_leo_latency_routing.graphs.snapshots import (
    add_graph_snapshot_features,
    graph_context_feature_columns,
)
from open_leo_latency_routing.optimization.calibrated_risk import (
    CalibratedRiskConfig,
    add_calibrated_mixture_risk_scores,
    fit_expert_calibration,
)
from open_leo_latency_routing.optimization.explainability import summarize_xai_attribution
from open_leo_latency_routing.evaluation.decision_opportunity import (
    build_candidate_opportunity_audit,
    build_opportunity_conditioned_results,
    build_pairwise_success_gap_bounds,
)
from open_leo_latency_routing.evaluation.delayed_execution import replay_delayed_execution
from open_leo_latency_routing.evaluation.risk_metrics import (
    empirical_upper_cvar,
    empirical_weighted_upper_cvar,
)
from open_leo_latency_routing.optimization.policies import (
    add_qos_filter_then_rank_scores,
    add_qos_shielded_scores,
    evaluate_decision_policies,
    select_validation_gated_fallback,
)
from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    select_opportunity_aware_risk_controlled_policy,
)
from open_leo_latency_routing.models.orbital_physics import (
    control_horizon_margin_ms,
    slant_range_km,
)
from scripts.generate_physics_informed_multipath_trace import build_trace
from scripts.build_commect_multiaccess_trace import _load_and_bin
from scripts.build_victoria_multihomed_trace import _pool_boundary_bins
from scripts.run_commect_rolling_origin_validation import _closed_partition
from scripts.run_reviewer_validation import _fit_paired_expert_calibrations
from scripts.run_service_path_experiments import (
    _apply_feasible_snapshot_residual_gate,
    _build_actionable_gate_selection_samples,
    _finite_sample_split_conformal_radius,
)


class ReviewerRevisionTests(unittest.TestCase):
    def test_snapshot_trust_gate_ignores_unavailable_candidate_risk(self) -> None:
        base = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 0, 1, 1],
                "relative_path": ["a", "b", "unavailable", "a", "b"],
                "is_feasible_path": [1, 1, 0, 1, 1],
                "pred_learned_residual_risk": [10.0, 30.0, 1.0, 80.0, 100.0],
            }
        )
        perturbed = base.copy()
        perturbed.loc[
            perturbed["relative_path"].eq("unavailable"),
            "pred_learned_residual_risk",
        ] = 1.0e12

        # The feasible-only median is 20 ms, below the 25 ms threshold. Under
        # the old all-row median, changing the unavailable row from 1 ms to an
        # extreme value would move the median from 10 ms to 30 ms and flip the
        # branch.
        gated_base = _apply_feasible_snapshot_residual_gate(base, 25.0)
        gated_perturbed = _apply_feasible_snapshot_residual_gate(
            perturbed,
            25.0,
        )
        feasible_rows = base["is_feasible_path"].astype(bool)

        pd.testing.assert_series_equal(
            gated_base.loc[feasible_rows, "disagreement_trust_gate"],
            gated_perturbed.loc[feasible_rows, "disagreement_trust_gate"],
        )
        pd.testing.assert_series_equal(
            gated_base.loc[feasible_rows, "snapshot_trust_gate_state"],
            gated_perturbed.loc[feasible_rows, "snapshot_trust_gate_state"],
        )
        self.assertTrue(
            gated_base.loc[
                gated_base["session_bin_index"].eq(0),
                "snapshot_learned_residual_risk_ms",
            ].eq(20.0).all()
        )
        self.assertTrue(
            gated_base.loc[
                gated_base["session_bin_index"].eq(0),
                "snapshot_trust_gate_state",
            ].eq("calibrated_risk").all()
        )
        self.assertTrue(
            gated_base.loc[
                gated_base["session_bin_index"].eq(1),
                "snapshot_trust_gate_state",
            ].eq("risk_fallback").all()
        )

    def test_snapshot_trust_gate_all_infeasible_is_deterministic_no_action(
        self,
    ) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [3, 3, 3],
                "relative_path": ["a", "b", "c"],
                "is_feasible_path": [0, 0, 0],
                "pred_learned_residual_risk": [0.0, 50.0, 1.0e12],
            }
        )
        changed_unavailable_scores = candidates.assign(
            pred_learned_residual_risk=[1.0e12, -100.0, 0.0]
        )

        first = _apply_feasible_snapshot_residual_gate(candidates, 50.0)
        second = _apply_feasible_snapshot_residual_gate(
            changed_unavailable_scores,
            50.0,
        )
        for gated in (first, second):
            self.assertTrue(gated["snapshot_feasible_candidate_count"].eq(0).all())
            self.assertTrue(gated["snapshot_residual_risk_valid"].eq(0).all())
            self.assertTrue(gated["disagreement_trust_gate"].eq(1.0).all())
            self.assertTrue(
                gated["snapshot_trust_gate_state"].eq(
                    "emergency_no_action"
                ).all()
            )
            self.assertTrue(
                gated["snapshot_learned_residual_risk_ms"].isna().all()
            )

        pd.testing.assert_series_equal(
            first["snapshot_trust_gate_state"],
            second["snapshot_trust_gate_state"],
        )
        pd.testing.assert_series_equal(
            first["disagreement_trust_gate"],
            second["disagreement_trust_gate"],
        )

    def test_split_conformal_radius_uses_corrected_order_statistic(self) -> None:
        scores = np.arange(1.0, 20.0)
        # n=19 and 90% coverage gives ceil(20*.9)=18, not an interpolated q90.
        self.assertEqual(
            _finite_sample_split_conformal_radius(scores, 0.90),
            18.0,
        )

    def test_split_conformal_radius_is_unbounded_when_sample_too_small(self) -> None:
        self.assertTrue(
            np.isinf(_finite_sample_split_conformal_radius([1.0, 2.0], 0.90))
        )

    def test_empirical_cvar_uses_fractional_tail_mass(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 100.0]
        # q=.7 gives a 1.5-observation upper tail: all of 100 plus half of 3.
        self.assertAlmostEqual(
            empirical_upper_cvar(values, 0.70),
            (100.0 + 0.5 * 3.0) / 1.5,
        )
        # At q=0 this reduces exactly to the empirical mean.
        self.assertAlmostEqual(empirical_upper_cvar(values, 0.0), 21.2)

    def test_empirical_cvar_does_not_overweight_quantile_ties(self) -> None:
        values = [0.0, 0.0, 10.0, 10.0]
        self.assertEqual(empirical_upper_cvar(values, 0.75), 10.0)

    def test_weighted_empirical_cvar_integrates_probability_mass(self) -> None:
        # Only ten percent of the mixture is at 100.  The upper half therefore
        # contains that full mass plus forty percent mass at zero.
        self.assertAlmostEqual(
            empirical_weighted_upper_cvar(
                [100.0, 0.0],
                [0.10, 0.90],
                0.50,
            ),
            20.0,
        )
        self.assertAlmostEqual(
            empirical_weighted_upper_cvar(
                [100.0, 0.0],
                [0.10, 0.90],
                0.0,
            ),
            10.0,
        )

    def test_four_way_split_keeps_policy_selection_independent(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": [path for path in ("a", "b") for _ in range(20)],
                "bin_epoch": list(range(20)) * 2,
            }
        )
        train, calibration, selection, test = split_train_calibration_selection_test(
            frame,
            train_ratio=0.55,
            calibration_ratio=0.15,
            selection_ratio=0.15,
            test_ratio=0.15,
        )
        for path in ("a", "b"):
            bounds = [
                part.loc[part["relative_path"].eq(path), "bin_epoch"]
                for part in (train, calibration, selection, test)
            ]
            self.assertLess(bounds[0].max(), bounds[1].min())
            self.assertLess(bounds[1].max(), bounds[2].min())
            self.assertLess(bounds[2].max(), bounds[3].min())

    def test_three_way_split_closes_one_step_and_multibin_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * 8,
                "bin_epoch": list(range(8)),
                "target_next_bin_epoch": list(range(1, 9)),
                "target_available_2": [1] * 8,
                "target_cumulative_2": [20.0] * 8,
                "target_mean_2": [10.0] * 8,
                "target_end_bin_epoch_2": list(range(2, 10)),
            }
        )

        train, validation, test = split_train_val_test(
            frame,
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
        )

        self.assertEqual(train["bin_epoch"].tolist(), [0, 1, 2])
        self.assertEqual(validation["bin_epoch"].tolist(), [4])
        self.assertEqual(test["bin_epoch"].tolist(), [6, 7])
        self.assertEqual(train["target_available_2"].tolist(), [1, 1, 0])
        self.assertEqual(validation["target_available_2"].tolist(), [0])
        self.assertTrue(
            train.loc[
                train["target_available_2"].eq(0),
                ["target_cumulative_2", "target_mean_2"],
            ].isna().all().all()
        )

    def test_four_way_split_uses_global_time_boundaries_and_closed_targets(self) -> None:
        # Path b starts late. A path-wise row-count split would put epochs 6--9
        # in different partitions across the two candidate paths.
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * 12 + ["b"] * 8,
                "bin_epoch": list(range(12)) + list(range(4, 12)),
                "target_next_bin_epoch": list(range(1, 13)) + list(range(5, 13)),
            }
        )
        parts = split_train_calibration_selection_test(
            frame,
            train_ratio=0.50,
            calibration_ratio=0.20,
            selection_ratio=0.15,
            test_ratio=0.15,
        )
        def expected_split(epoch: int) -> str:
            if epoch <= 5:
                return "train"
            if epoch <= 7:
                return "calibration"
            if epoch <= 9:
                return "selection"
            return "test"

        for name, part in zip(
            ("train", "calibration", "selection", "test"), parts
        ):
            self.assertTrue(part["bin_epoch"].map(expected_split).eq(name).all())
            self.assertTrue(
                part["target_next_bin_epoch"].map(expected_split).eq(name).all()
            )

    def test_risk_control_rejects_learning_without_opportunities(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": [40.0] * 20,
                "graph": [30.0] * 20,
                "ensemble": [35.0] * 20,
            },
            opportunity_mask=[False] * 20,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=1.0,
                bootstrap_samples=100,
            ),
        )
        self.assertEqual(result.selected_policy, "reactive")
        self.assertIn("insufficient", result.reason)

    def test_risk_control_rejects_malformed_public_inputs(self) -> None:
        base = {"reactive": [50.0, 51.0], "graph": [49.0, 50.0]}
        invalid_calls = (
            ({"reactive": [[50.0], [51.0]], "graph": [[49.0], [50.0]]}, [1, 1], 60.0, RiskControlConfig()),
            (base, [1, np.nan], 60.0, RiskControlConfig()),
            (base, [1, 2], 60.0, RiskControlConfig()),
            (base, [True, False], float("nan"), RiskControlConfig()),
            (base, [True, False], 60.0, RiskControlConfig(noninferiority_margin=float("nan"))),
            (base, [True, False], 60.0, RiskControlConfig(opportunity_noninferiority_margin=1.01)),
            (base, [True, False], 60.0, RiskControlConfig(minimum_effective_opportunities=float("nan"))),
            (base, [True, False], 60.0, RiskControlConfig(practical_cvar_gain_ms=float("nan"))),
            (base, [True, False], 60.0, RiskControlConfig(cvar_grid_points=2.5)),
            (base, [True, False], 60.0, RiskControlConfig(planned_gate_uses=1.5)),
            (base, [True, False], 60.0, RiskControlConfig(gate_use_index=1.5)),
            (base, [True, False], 60.0, RiskControlConfig(block_length=1.5)),
        )
        for arrays, opportunities, budget, config in invalid_calls:
            with self.subTest(config=config, budget=budget):
                with self.assertRaises(ValueError):
                    select_opportunity_aware_risk_controlled_policy(
                        arrays,
                        opportunity_mask=opportunities,
                        latency_budget_ms=budget,
                        config=config,
                    )

    def test_default_cvar_grid_resolves_practical_gain_margin(self) -> None:
        config = RiskControlConfig()
        spacing = config.latency_cap_ms / (config.cvar_grid_points - 1)
        lipschitz = max(
            1.0,
            config.cvar_quantile / (1.0 - config.cvar_quantile),
        )
        between_grid_correction = 0.5 * spacing * lipschitz
        self.assertLess(
            between_grid_correction,
            config.practical_cvar_gain_ms,
        )

    def test_risk_control_admits_noninferior_practical_cvar_gain(self) -> None:
        # A deliberately large paired effect can pass the conservative finite-
        # sample gate.  This is not a test that a small sample proves equality.
        reactive = [90.0] * 500
        graph = [10.0] * 500
        ensemble = [95.0] * 500
        result = select_opportunity_aware_risk_controlled_policy(
            {"reactive": reactive, "graph": graph, "ensemble": ensemble},
            opportunity_mask=[True] * len(reactive),
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=2.0,
                practical_cvar_gain_ms=1.0,
                cvar_quantile=0.5,
                latency_cap_ms=100.0,
                cvar_grid_points=21,
            ),
            independence_group_ids=np.arange(len(reactive)),
        )
        self.assertEqual(result.selected_policy, "graph")
        selected = result.evidence_frame().query("selected").iloc[0]
        self.assertTrue(bool(selected["noninferior"]))
        self.assertTrue(bool(selected["practically_better"]))
        self.assertGreater(float(selected["cvar_gain_lcb_ms"]), 1.0)

    def test_risk_control_all_zero_success_differences_keep_uncertainty(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": [90.0] * 80,
                "graph": [80.0] * 80,
            },
            opportunity_mask=[True] * 80,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                noninferiority_margin=0.02,
                minimum_effective_opportunities=1.0,
                cvar_quantile=0.5,
                latency_cap_ms=100.0,
                cvar_grid_points=11,
            ),
            independence_group_ids=np.arange(80),
        )
        graph = result.evidence_frame().query("policy == 'graph'").iloc[0]
        self.assertEqual(float(graph["success_delta_vs_reactive"]), 0.0)
        self.assertLess(float(graph["success_delta_lcb"]), 0.0)
        self.assertGreater(float(graph["success_bound_radius"]), 0.0)
        # One candidate/use has four protected endpoint families, so the
        # aggregate-success Clopper--Pearson endpoint receives alpha=.0125.
        expected_lcb = -(1.0 - 0.0125 ** (1.0 / 80.0))
        self.assertAlmostEqual(float(graph["success_delta_lcb"]), expected_lcb)
        self.assertAlmostEqual(
            float(graph["opportunity_conditioned_success_delta_lcb"]),
            expected_lcb,
        )
        self.assertFalse(bool(graph["noninferior"]))
        self.assertEqual(result.selected_policy, "reactive")

    def test_opportunity_conditioned_success_bound_is_a_required_endpoint(
        self,
    ) -> None:
        # Two harmful opportunity-bearing groups are diluted by 900 groups
        # with no decision opportunity in the aggregate harmful-group bound.
        # The candidate also has a certified tail gain. The strengthened gate
        # must still reject it because the separately conditioned opportunity
        # endpoint does not clear its pre-declared non-inferiority threshold.
        group_count = 1000
        opportunities = np.zeros(group_count, dtype=bool)
        opportunities[:100] = True
        reactive = np.full(group_count, 90.0)
        candidate = np.full(group_count, 70.0)
        candidate[:98] = 10.0
        reactive[98:100] = 50.0

        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": reactive.tolist(),
                "graph": candidate.tolist(),
            },
            opportunity_mask=opportunities,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                noninferiority_margin=0.02,
                opportunity_noninferiority_margin=0.02,
                minimum_effective_opportunities=5.0,
                cvar_quantile=0.5,
                latency_cap_ms=100.0,
                cvar_grid_points=21,
            ),
            independence_group_ids=np.arange(group_count),
        )
        graph = result.evidence_frame().query("policy == 'graph'").iloc[0]

        self.assertTrue(bool(graph["aggregate_actionable_success_noninferior"]))
        self.assertTrue(bool(graph["practically_better"]))
        self.assertTrue(bool(graph["opportunity_sufficient"]))
        self.assertFalse(
            bool(graph["opportunity_conditioned_success_noninferior"])
        )
        self.assertFalse(bool(graph["success_endpoints_noninferior"]))
        self.assertFalse(bool(graph["noninferior"]))
        self.assertFalse(bool(graph["eligible"]))
        self.assertEqual(result.selected_policy, "reactive")
        self.assertEqual(
            graph["opportunity_conditioned_success_estimand_population"],
            "uniform_opportunity_bearing_independent_group_then_uniform_post_hoc_"
            "decision_opportunity_within_group",
        )
        self.assertEqual(
            int(graph["opportunity_conditioned_inference_group_count"]),
            100,
        )
        self.assertGreaterEqual(
            float(graph["aggregate_actionable_success_delta_lcb"]),
            float(graph["aggregate_actionable_success_lcb_threshold"]),
        )
        self.assertLess(
            float(graph["opportunity_conditioned_success_delta_lcb"]),
            float(graph["opportunity_conditioned_success_lcb_threshold"]),
        )

    def test_risk_control_requires_cvar_confidence_not_point_gain(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": [90.0] * 500,
                "graph": [85.0] * 500,
            },
            opportunity_mask=[True] * 500,
            latency_budget_ms=87.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=1.0,
                practical_cvar_gain_ms=1.0,
                cvar_quantile=0.5,
                latency_cap_ms=100.0,
                cvar_grid_points=11,
            ),
            independence_group_ids=np.arange(500),
        )
        graph = result.evidence_frame().query("policy == 'graph'").iloc[0]
        self.assertGreater(float(graph["cvar_gain_vs_reactive_ms"]), 1.0)
        self.assertLess(float(graph["cvar_gain_lcb_ms"]), 1.0)
        self.assertTrue(bool(graph["noninferior"]))
        self.assertFalse(bool(graph["practically_better"]))
        self.assertEqual(result.selected_policy, "reactive")

    def test_effective_opportunities_count_independent_groups(self) -> None:
        mask = [True, True, False, False, True, False, False, False, True, False, False, False]
        result = select_opportunity_aware_risk_controlled_policy(
            {"reactive": [50.0] * 12, "graph": [40.0] * 12},
            opportunity_mask=mask,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=3.0,
                latency_cap_ms=100.0,
                cvar_quantile=0.5,
                cvar_grid_points=11,
            ),
            independence_group_ids=np.repeat(np.arange(3), 4),
        )
        evidence = result.evidence_frame().iloc[0]
        self.assertEqual(int(evidence["opportunity_count"]), 4)
        self.assertEqual(int(evidence["opportunity_block_count"]), 3)
        self.assertEqual(float(evidence["effective_opportunity_count"]), 3.0)
        self.assertTrue(bool(evidence["opportunity_sufficient"]))

    def test_gate_alpha_is_shared_across_planned_uses_and_endpoints(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": [90.0] * 20,
                "graph": [10.0] * 20,
                "ensemble": [20.0] * 20,
            },
            opportunity_mask=[True] * 20,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                alpha=0.05,
                planned_gate_uses=5,
                gate_use_index=3,
                latency_cap_ms=100.0,
                cvar_quantile=0.5,
                cvar_grid_points=11,
            ),
            independence_group_ids=np.arange(20),
        )
        evidence = result.evidence_frame().iloc[0]
        self.assertAlmostEqual(float(evidence["alpha_per_learned_policy"]), 0.005)
        self.assertAlmostEqual(float(evidence["alpha_success_bound"]), 0.00125)
        self.assertAlmostEqual(
            float(evidence["alpha_aggregate_success_bound"]),
            0.00125,
        )
        self.assertAlmostEqual(
            float(evidence["alpha_opportunity_success_bound"]),
            0.00125,
        )
        self.assertAlmostEqual(float(evidence["alpha_cvar_comparison"]), 0.0025)
        self.assertAlmostEqual(
            float(evidence["alpha_cvar_reactive_interval"]),
            0.00125,
        )
        self.assertAlmostEqual(
            float(evidence["alpha_cvar_candidate_interval"]),
            0.00125,
        )
        self.assertAlmostEqual(
            2.0 * float(evidence["alpha_success_bound"])
            + float(evidence["alpha_cvar_comparison"]),
            float(evidence["alpha_per_learned_policy"]),
        )
        self.assertEqual(
            int(evidence["alpha_endpoint_family_count_per_candidate_use"]),
            4,
        )
        self.assertEqual(int(evidence["gate_use_index"]), 3)

    def test_explicit_session_groups_prevent_bin_level_pseudoreplication(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {"reactive": [90.0] * 40, "graph": [10.0] * 40},
            opportunity_mask=[True] * 40,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=2.0,
                latency_cap_ms=100.0,
                cvar_quantile=0.5,
                cvar_grid_points=11,
            ),
            independence_group_ids=["one_drive"] * 40,
        )
        evidence = result.evidence_frame().iloc[0]
        self.assertEqual(int(evidence["inference_block_count"]), 1)
        self.assertEqual(int(evidence["opportunity_block_count"]), 1)
        self.assertEqual(
            evidence["inference_unit_source"],
            "supplied_session_or_collection_group",
        )
        self.assertFalse(bool(evidence["opportunity_sufficient"]))
        self.assertEqual(result.selected_policy, "reactive")
        graph = result.evidence_frame().query("policy == 'graph'").iloc[0]
        self.assertFalse(
            bool(graph["aggregate_actionable_success_noninferior"])
        )
        self.assertFalse(
            bool(graph["opportunity_conditioned_success_noninferior"])
        )
        self.assertFalse(bool(graph["success_endpoints_noninferior"]))
        self.assertLess(
            float(graph["opportunity_conditioned_success_delta_lcb"]),
            float(graph["opportunity_conditioned_success_lcb_threshold"]),
        )

    def test_gate_estimand_is_invariant_to_within_group_replication(self) -> None:
        config = RiskControlConfig(
            minimum_effective_opportunities=0.0,
            cvar_quantile=0.5,
            latency_cap_ms=100.0,
            cvar_grid_points=21,
        )

        def evaluate(
            values: list[float],
            group_ids: list[str],
        ) -> pd.Series:
            result = select_opportunity_aware_risk_controlled_policy(
                {"reactive": values, "graph": values},
                opportunity_mask=[True] * len(values),
                latency_budget_ms=60.0,
                config=config,
                independence_group_ids=group_ids,
            )
            return result.evidence_frame().query("policy == 'reactive'").iloc[0]

        balanced = evaluate(
            [100.0, 0.0],
            ["high_latency_collection", "low_latency_collection"],
        )
        replicated = evaluate(
            [100.0] + [0.0] * 10,
            ["high_latency_collection"] + ["low_latency_collection"] * 10,
        )

        for field in (
            "success_rate",
            "group_uniform_success_rate",
            "mean_latency_ms",
            "group_uniform_mean_latency_ms",
            "cvar_latency_ms",
            "group_uniform_cvar_latency_ms",
            "bounded_cvar_latency_ms",
            "group_uniform_bounded_cvar_latency_ms",
            "cvar_lcb_ms",
            "group_uniform_cvar_lcb_ms",
            "cvar_ucb_ms",
            "group_uniform_cvar_ucb_ms",
        ):
            self.assertAlmostEqual(
                float(balanced[field]),
                float(replicated[field]),
            )
        self.assertAlmostEqual(float(replicated["success_rate"]), 0.5)
        self.assertAlmostEqual(float(replicated["mean_latency_ms"]), 50.0)
        self.assertAlmostEqual(float(replicated["cvar_latency_ms"]), 100.0)
        self.assertEqual(int(replicated["inference_block_count"]), 2)
        self.assertEqual(replicated["observation_weight_formula"], "1/(G*n_g)")
        self.assertEqual(replicated["primary_gate_point_field_prefix"], "group_uniform_")
        self.assertFalse(
            bool(replicated["epoch_pooled_point_estimates_used_for_gate"])
        )
        self.assertAlmostEqual(
            float(replicated["success_rate"]),
            float(replicated["group_uniform_success_rate"]),
        )
        self.assertEqual(
            replicated["cvar_estimand_population"],
            "uniform_independent_group_then_uniform_epoch_within_group",
        )

    def test_opportunity_success_estimand_is_group_uniform(self) -> None:
        config = RiskControlConfig(
            minimum_effective_opportunities=0.0,
            cvar_quantile=0.5,
            latency_cap_ms=100.0,
            cvar_grid_points=21,
        )

        def candidate_evidence(
            reactive: list[float],
            candidate: list[float],
            groups: list[str],
        ) -> pd.Series:
            result = select_opportunity_aware_risk_controlled_policy(
                {"reactive": reactive, "graph": candidate},
                opportunity_mask=[True] * len(reactive),
                latency_budget_ms=60.0,
                config=config,
                independence_group_ids=groups,
            )
            return result.evidence_frame().query("policy == 'graph'").iloc[0]

        balanced = candidate_evidence(
            [50.0, 70.0],
            [70.0, 50.0],
            ["harmful", "beneficial"],
        )
        replicated = candidate_evidence(
            [50.0] + [70.0] * 10,
            [70.0] + [50.0] * 10,
            ["harmful"] + ["beneficial"] * 10,
        )

        for field in (
            "opportunity_conditioned_success_delta_vs_reactive",
            "opportunity_conditioned_block_success_delta",
            "opportunity_conditioned_success_delta_lcb",
            "opportunity_conditioned_harmful_group_probability_ucb",
        ):
            self.assertAlmostEqual(
                float(balanced[field]),
                float(replicated[field]),
            )
        self.assertEqual(
            int(replicated["opportunity_conditioned_inference_group_count"]),
            2,
        )
        self.assertEqual(
            int(replicated["opportunity_conditioned_harmful_group_count"]),
            1,
        )
        self.assertEqual(
            replicated["opportunity_conditioned_observation_weight_formula"],
            "1/(G_opportunity*m_g)",
        )

    def test_risk_control_rejects_success_regression(self) -> None:
        result = select_opportunity_aware_risk_controlled_policy(
            {
                "reactive": [50.0] * 40,
                "graph": [20.0] * 20 + [80.0] * 20,
                "ensemble": [55.0] * 40,
            },
            opportunity_mask=[True] * 40,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=2.0,
                bootstrap_samples=200,
            ),
            independence_group_ids=np.repeat(np.arange(20), 2),
        )
        self.assertEqual(result.selected_policy, "reactive")

    def test_risk_control_rejects_time_blocks_as_independence(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "cannot establish independent acquisition groups",
        ):
            select_opportunity_aware_risk_controlled_policy(
                {"reactive": [50.0] * 8, "graph": [40.0] * 8},
                opportunity_mask=[True] * 8,
                latency_budget_ms=60.0,
                config=RiskControlConfig(block_length=1),
            )

    def test_qos_filter_then_rank_uses_prediction_within_compliant_set(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 0],
                "relative_path": ["a", "b", "c"],
                "latency_mean_ms": [20.0, 30.0, 80.0],
                "is_feasible_path": [1, 1, 1],
                "prediction": [50.0, 10.0, 1.0],
            }
        )
        output = add_qos_filter_then_rank_scores(
            candidates,
            "prediction",
            latency_budget_ms=60.0,
        )
        selected = output.loc[output["pred_qos_filter_then_rank"].idxmin()]
        self.assertEqual(selected["relative_path"], "b")

    def test_rolling_partition_drops_targets_crossing_the_time_boundary(self) -> None:
        frame = pd.DataFrame(
            {
                "session_bin_index": [0, 1, 2],
                "bin_epoch": [100, 110, 120],
                "target_next_bin_epoch": [110, 120, 130],
            }
        )
        partition = _closed_partition(frame, [0, 1])
        self.assertEqual(partition["session_bin_index"].tolist(), [0])

    def test_rolling_partition_does_not_require_endpoint_forecast_eligibility(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a", "a"],
                "session_bin_index": [0, 3],
                "bin_epoch": [100, 130],
                "target_next_bin_epoch": [110, 140],
                "target_available_2": [1, 1],
                "target_cumulative_2": [20.0, 20.0],
                "target_mean_2": [10.0, 10.0],
            }
        )

        # Epochs 1 and 2 are not eligible decision rows in this hypothetical
        # fold, but their observed timestamps remain inside its declared
        # [100, 130] wall-clock interval.  They may therefore close outcomes
        # for epoch 0 without having to forecast outcomes of their own.
        partition = _closed_partition(frame, [0, 3], (2,))

        self.assertEqual(partition["session_bin_index"].tolist(), [0])
        self.assertEqual(partition["target_available_2"].tolist(), [1])
        self.assertEqual(partition["target_cumulative_2"].tolist(), [20.0])

    def test_rolling_partition_closes_every_configured_future_window(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * 8,
                "session_bin_index": list(range(8)),
                "bin_epoch": list(range(100, 180, 10)),
                "target_next_bin_epoch": list(range(110, 190, 10)),
                "target_available_3": [1] * 8,
                "target_cumulative_3": [60.0] * 8,
                "target_mean_3": [20.0] * 8,
                "target_available_4": [1] * 8,
                "target_cumulative_4": [80.0] * 8,
                "target_mean_4": [20.0] * 8,
                "target_available_5": [1] * 8,
                "target_cumulative_5": [100.0] * 8,
                "target_mean_5": [20.0] * 8,
            }
        )
        partition = _closed_partition(frame, [0, 1, 2, 3, 4], (3, 4, 5))

        # The one-step boundary row is removed. A future window is available
        # only when its exact final epoch remains in the same block.
        self.assertEqual(partition["session_bin_index"].tolist(), [0, 1, 2, 3])
        self.assertEqual(partition["target_available_3"].tolist(), [1, 1, 0, 0])
        self.assertEqual(partition["target_available_4"].tolist(), [1, 0, 0, 0])
        self.assertEqual(partition["target_available_5"].tolist(), [0, 0, 0, 0])
        self.assertTrue(
            partition.loc[
                partition["target_available_4"].eq(0),
                ["target_cumulative_4", "target_mean_4"],
            ].isna().all().all()
        )
        self.assertNotIn("target_end_bin_epoch_4", frame.columns)

    def test_fixed_split_closes_a_nondefault_multibin_horizon(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * 20,
                "bin_epoch": list(range(20)),
                "target_next_bin_epoch": list(range(1, 20)) + [float("nan")],
                "target_available_4": [1] * 16 + [0] * 4,
                "target_cumulative_4": [80.0] * 16 + [float("nan")] * 4,
                "target_mean_4": [20.0] * 16 + [float("nan")] * 4,
                "target_end_bin_epoch_4": list(range(4, 20))
                + [float("nan")] * 4,
            }
        )
        parts = split_train_calibration_selection_test(
            frame,
            train_ratio=0.25,
            calibration_ratio=0.25,
            selection_ratio=0.25,
            test_ratio=0.25,
        )
        for part, (lower, upper) in zip(
            parts,
            ((0, 4), (5, 9), (10, 14), (15, 19)),
        ):
            available = part[part["target_available_4"].eq(1)]
            self.assertTrue(
                available["target_end_bin_epoch_4"].between(
                    lower, upper
                ).all()
            )
            unavailable = part[part["target_available_4"].eq(0)]
            self.assertTrue(
                unavailable[
                    ["target_cumulative_4", "target_mean_4"]
                ].isna().all().all()
            )
        train = parts[0].set_index("bin_epoch")
        self.assertEqual(int(train.loc[0, "target_available_4"]), 1)
        self.assertEqual(int(train.loc[1, "target_available_4"]), 0)

    def test_forecast_table_exports_configured_target_endpoints(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * 8,
                "bin_epoch": list(range(8)),
                "latency_mean_ms": [20.0 + value for value in range(8)],
                "observed_replies": [100] * 8,
                "path_state": ["active"] * 8,
                "window_duration": ["1h"] * 8,
                "probe_interval": ["1000ms"] * 8,
                "session_date": pd.to_datetime(["2026-08-21"] * 8),
            }
        )
        forecast = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1, 2],
            horizon_bins=1,
            decision_cadence_seconds=1,
            multi_bin_horizons=[2, 4],
        )
        self.assertIn("target_end_bin_epoch_2", forecast)
        self.assertIn("target_end_bin_epoch_4", forecast)
        self.assertNotIn("target_available_3", forecast)
        self.assertEqual(float(forecast.loc[0, "target_end_bin_epoch_2"]), 2.0)
        self.assertEqual(float(forecast.loc[0, "target_end_bin_epoch_4"]), 4.0)

    def test_forecast_targets_reject_next_available_rows_across_gaps(self) -> None:
        epochs = [0, 10, 30, 40, 50]
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * len(epochs),
                "session_bin_index": list(range(len(epochs))),
                "bin_epoch": epochs,
                "bin_seconds": [10] * len(epochs),
                "latency_mean_ms": [10.0, 20.0, 30.0, 40.0, 50.0],
                "observed_replies": [100] * len(epochs),
                "path_state": ["active"] * len(epochs),
                "window_duration": ["1h"] * len(epochs),
                "probe_interval": ["1000ms"] * len(epochs),
                "session_date": pd.to_datetime(["2026-08-21"] * len(epochs)),
            }
        )
        forecast = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1, 2],
            horizon_bins=1,
            decision_cadence_seconds=10,
            multi_bin_horizons=[2],
        )
        by_epoch = forecast.set_index("bin_epoch")
        self.assertEqual(set(by_epoch.index), {0, 30, 40})
        self.assertEqual(float(by_epoch.loc[0, "target_next_bin_epoch"]), 10.0)
        self.assertEqual(int(by_epoch.loc[0, "target_available_2"]), 0)
        self.assertTrue(pd.isna(by_epoch.loc[0, "target_end_bin_epoch_2"]))
        self.assertEqual(float(by_epoch.loc[30, "latency_mean_ms_lag_1"]), 0.0)
        self.assertEqual(
            int(by_epoch.loc[30, "latency_mean_ms_lag_1_available"]),
            0,
        )
        self.assertEqual(float(by_epoch.loc[30, "latency_mean_ms_lag_2"]), 20.0)
        self.assertEqual(
            int(by_epoch.loc[30, "latency_mean_ms_lag_2_available"]),
            1,
        )
        self.assertAlmostEqual(
            float(by_epoch.loc[30, "latency_mean_ms_roll_mean_3"]),
            25.0,
        )
        self.assertAlmostEqual(
            float(by_epoch.loc[30, "latency_mean_ms_roll_coverage_3"]),
            2.0 / 3.0,
        )
        portable_features = default_feature_columns(forecast)
        self.assertIn("latency_mean_ms_lag_1_available", portable_features)
        self.assertIn("latency_mean_ms_roll_coverage_3", portable_features)
        self.assertIn("observed_replies_lag_1_available", portable_features)
        self.assertIn("history_lag_coverage_ratio", portable_features)
        audit = forecast.attrs["exact_horizon_audit"]
        self.assertEqual(int(audit["excluded_nonexact_gap_row_count"]), 1)
        self.assertEqual(
            int(
                audit["multi_bin_horizons"]["2"][
                    "excluded_nonexact_gap_row_count"
                ]
            ),
            2,
        )

    def test_forecast_targets_require_exact_multi_bin_sequences(self) -> None:
        epochs = list(range(0, 60, 10))
        frame = pd.DataFrame(
            {
                "relative_path": ["a"] * len(epochs),
                "session_bin_index": list(range(len(epochs))),
                "bin_epoch": epochs,
                "bin_seconds": [10] * len(epochs),
                "latency_mean_ms": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "observed_replies": [100] * len(epochs),
                "path_state": ["active"] * len(epochs),
                "window_duration": ["1h"] * len(epochs),
                "probe_interval": ["1000ms"] * len(epochs),
                "session_date": pd.to_datetime(["2026-08-21"] * len(epochs)),
            }
        )
        forecast = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1],
            horizon_bins=2,
            decision_cadence_seconds=10,
            multi_bin_horizons=[3],
        ).set_index("bin_epoch")
        self.assertEqual(float(forecast.loc[0, "target_next"]), 3.0)
        self.assertEqual(float(forecast.loc[0, "target_next_bin_epoch"]), 20.0)
        self.assertEqual(float(forecast.loc[0, "target_cumulative_3"]), 9.0)
        self.assertEqual(float(forecast.loc[0, "target_mean_3"]), 3.0)
        self.assertEqual(float(forecast.loc[0, "target_end_bin_epoch_3"]), 30.0)

    def test_exact_target_guard_supports_all_evaluated_cadences(self) -> None:
        for cadence in (5, 10, 30, 60):
            with self.subTest(cadence_seconds=cadence):
                epochs = [0, cadence, 2 * cadence, 3 * cadence]
                frame = pd.DataFrame(
                    {
                        "relative_path": ["a"] * 4,
                        "session_bin_index": list(range(4)),
                        "bin_epoch": epochs,
                        "bin_seconds": [cadence] * 4,
                        "latency_mean_ms": [1.0, 2.0, 3.0, 4.0],
                        "observed_replies": [100] * 4,
                        "path_state": ["active"] * 4,
                        "window_duration": ["1h"] * 4,
                        "probe_interval": ["1000ms"] * 4,
                        "session_date": pd.to_datetime(["2026-08-21"] * 4),
                    }
                )
                forecast = build_forecast_table(
                    frame,
                    target_column="latency_mean_ms",
                    lags=[1],
                    horizon_bins=2,
                    decision_cadence_seconds=cadence,
                    multi_bin_horizons=[3],
                ).set_index("bin_epoch")
                self.assertEqual(
                    float(forecast.loc[0, "target_next_bin_epoch"]),
                    float(2 * cadence),
                )
                self.assertEqual(
                    float(forecast.loc[0, "target_end_bin_epoch_3"]),
                    float(3 * cadence),
                )

    def test_policy_forecast_drops_asymmetric_missing_candidate_epoch(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a", "a", "a", "b", "b"],
                "session_bin_index": [0, 1, 2, 0, 2],
                "bin_epoch": [0, 10, 20, 0, 20],
                "bin_seconds": [10] * 5,
                "latency_mean_ms": [10.0, 11.0, 12.0, 20.0, 22.0],
                "observed_replies": [100] * 5,
                "path_state": ["active"] * 5,
                "window_duration": ["1h"] * 5,
                "probe_interval": ["1000ms"] * 5,
                "session_date": pd.to_datetime(["2026-08-21"] * 5),
            }
        )
        forecast = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1],
            horizon_bins=1,
            decision_cadence_seconds=10,
            require_complete_decision_epochs=True,
        )
        self.assertNotIn(0, set(forecast["session_bin_index"]))
        self.assertIn(1, set(forecast["session_bin_index"]))
        audit = forecast.attrs["exact_horizon_audit"]
        self.assertEqual(int(audit["excluded_incomplete_decision_epoch_count"]), 2)
        self.assertEqual(
            int(audit["asymmetric_missing_candidate_epoch_count"]),
            1,
        )

    def test_opportunity_audit_separates_saturation_from_actionable_epochs(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1, 2, 2, 3, 3],
                "relative_path": ["a", "b"] * 4,
                "target_next": [20.0, 30.0, 20.0, 80.0, 70.0, 90.0, 10.0, 20.0],
                "is_feasible_path": [1, 1, 1, 1, 1, 1, 0, 0],
            }
        )
        audit, labels = build_candidate_opportunity_audit(
            candidates,
            thresholds_ms=(60.0,),
        )
        row = audit.iloc[0]
        self.assertEqual(int(row["all_candidates_succeed_count"]), 1)
        self.assertEqual(int(row["mixed_outcome_opportunity_count"]), 1)
        self.assertEqual(int(row["all_candidates_fail_count"]), 1)
        self.assertEqual(int(row["emergency_no_current_feasible_count"]), 1)
        self.assertEqual(int(row["decision_opportunity_count"]), 1)
        self.assertEqual(
            labels.loc[labels["session_bin_index"].eq(3), "decision_opportunity"].iloc[0],
            0,
        )

    def test_gate_samples_exclude_emergency_epochs_from_every_aligned_array(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1, 2, 2],
                "relative_path": ["a", "b"] * 3,
                # Epoch 0 is an outage. Epoch 1 is a routing opportunity.
                # At epoch 2 only path a is actionable; path b's attractive
                # scores and outcome must not enter any policy array.
                "target_next": [5.0, 500.0, 20.0, 80.0, 40.0, 1.0],
                "is_feasible_path": [0, 0, 1, 1, 1, 0],
                "latency_mean_ms": [1.0, 2.0, 50.0, 10.0, 30.0, 1.0],
                "selection_graph_shield": [1.0, 2.0, 1.0, 2.0, 4.0, 0.0],
                "selection_ensemble_shield": [2.0, 1.0, 3.0, 1.0, 4.0, 0.0],
            }
        )
        base_selection = candidates[["session_bin_index", "relative_path"]].copy()
        base_selection["collection_id"] = [
            "outage-a",
            "outage-b",
            "drive-one",
            "drive-one",
            "drive-two",
            "drive-two",
        ]

        samples = _build_actionable_gate_selection_samples(
            candidates,
            base_selection,
            latency_budget_ms=60.0,
            risk_control_group_column="collection_id",
        )

        self.assertEqual(samples.total_epoch_count, 3)
        self.assertEqual(samples.excluded_emergency_epoch_count, 1)
        self.assertEqual(samples.actionable_epoch_count, 2)
        self.assertEqual(samples.opportunity_mask, [True, False])
        self.assertEqual(samples.independence_group_ids, ["drive-one", "drive-two"])
        self.assertEqual(samples.realized_latency["reactive"], [80.0, 40.0])
        self.assertEqual(samples.realized_latency["graph"], [20.0, 40.0])
        self.assertEqual(samples.realized_latency["ensemble"], [80.0, 40.0])

        result = select_opportunity_aware_risk_controlled_policy(
            samples.realized_latency,
            samples.opportunity_mask,
            latency_budget_ms=60.0,
            config=RiskControlConfig(
                minimum_effective_opportunities=1.0,
                latency_cap_ms=100.0,
                cvar_quantile=0.5,
                cvar_grid_points=11,
            ),
            independence_group_ids=samples.independence_group_ids,
        )
        evidence = result.evidence_frame().iloc[0]
        self.assertEqual(int(evidence["decision_count"]), 2)
        self.assertEqual(int(evidence["opportunity_count"]), 1)
        self.assertEqual(int(evidence["inference_block_count"]), 2)

    def test_all_emergency_gate_samples_fail_closed_without_pseudo_actions(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b"] * 2,
                "target_next": [10.0, 90.0, 20.0, 100.0],
                "is_feasible_path": [0, 0, 0, 0],
                "latency_mean_ms": [1.0, 2.0, 3.0, 4.0],
                "selection_graph_shield": [4.0, 3.0, 2.0, 1.0],
                "selection_ensemble_shield": [1.0, 2.0, 3.0, 4.0],
            }
        )
        base_selection = candidates[["session_bin_index", "relative_path"]].copy()
        # Conflicting identifiers demonstrate that excluded outages never
        # become gate inference groups.
        base_selection["collection_id"] = ["a", "b", "c", "d"]
        samples = _build_actionable_gate_selection_samples(
            candidates,
            base_selection,
            latency_budget_ms=60.0,
            risk_control_group_column="collection_id",
        )

        self.assertEqual(samples.total_epoch_count, 2)
        self.assertEqual(samples.excluded_emergency_epoch_count, 2)
        self.assertEqual(samples.actionable_epoch_count, 0)
        self.assertEqual(samples.opportunity_mask, [])
        self.assertEqual(samples.independence_group_ids, [])
        self.assertTrue(all(not values for values in samples.realized_latency.values()))

        result = select_opportunity_aware_risk_controlled_policy(
            samples.realized_latency,
            samples.opportunity_mask,
            latency_budget_ms=60.0,
            independence_group_ids=samples.independence_group_ids,
        )
        self.assertEqual(result.selected_policy, "reactive")
        self.assertIn("no actionable", result.reason)
        evidence = result.evidence_frame()
        self.assertTrue(evidence["empty_actionable_guard"].all())
        self.assertTrue(evidence["decision_count"].eq(0).all())
        self.assertTrue(evidence["opportunity_count"].eq(0).all())
        self.assertTrue(evidence["inference_block_count"].eq(0).all())
        learned = evidence[evidence["policy"].ne("reactive")]
        self.assertFalse(learned["eligible"].any())
        self.assertFalse(
            learned["opportunity_conditioned_success_noninferior"].any()
        )
        self.assertTrue(
            learned["opportunity_conditioned_success_delta_lcb"].eq(-1.0).all()
        )
        self.assertTrue(
            learned["opportunity_conditioned_endpoint_defined"].eq(False).all()
        )

    def test_opportunity_conditioning_uses_only_mixed_candidate_sets(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b", "a", "b"],
                "target_next": [20.0, 80.0, 20.0, 30.0],
            }
        )
        _, labels = build_candidate_opportunity_audit(
            candidates,
            thresholds_ms=(60.0,),
        )
        decisions = pd.DataFrame(
            {
                "session_bin_index": [0, 1, 0, 1],
                "policy_name": ["good", "good", "bad", "bad"],
                "realized_next_latency_ms": [20.0, 20.0, 80.0, 20.0],
                "decision_gap_ms": [0.0, 0.0, 60.0, 0.0],
            }
        )
        output = build_opportunity_conditioned_results(decisions, labels)
        good = output[output["policy_name"].eq("good")].iloc[0]
        bad = output[output["policy_name"].eq("bad")].iloc[0]
        self.assertEqual(int(good["opportunity_count"]), 1)
        self.assertEqual(float(good["opportunity_capture_rate"]), 1.0)
        self.assertEqual(float(bad["opportunity_capture_rate"]), 0.0)

    def test_pairwise_success_gap_respects_discriminative_rate_bound(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b", "a", "b"],
                "target_next": [20.0, 80.0, 20.0, 30.0],
            }
        )
        _, labels = build_candidate_opportunity_audit(
            candidates,
            thresholds_ms=(60.0,),
        )
        decisions = pd.DataFrame(
            {
                "session_bin_index": [0, 1, 0, 1],
                "policy_name": ["a_policy", "a_policy", "b_policy", "b_policy"],
                "realized_next_latency_ms": [20.0, 20.0, 80.0, 30.0],
            }
        )
        bounds = build_pairwise_success_gap_bounds(decisions, labels)
        self.assertAlmostEqual(float(bounds.iloc[0]["absolute_success_rate_gap"]), 0.5)
        self.assertAlmostEqual(
            float(bounds.iloc[0]["discriminative_epoch_rate_bound"]), 0.5
        )
        self.assertTrue(bool(bounds.iloc[0]["bound_holds"]))

    def test_delayed_replay_uses_later_availability_and_latency(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "bin_epoch": [100, 100, 110, 110],
                "relative_path": ["a", "b", "a", "b"],
                "target_next": [40.0, 50.0, 90.0, 30.0],
                "is_feasible_path": [1, 1, 0, 1],
            }
        )
        decisions = pd.DataFrame(
            {
                "session_bin_index": [0],
                "decision_bin_epoch": [100],
                "policy_name": ["test_policy"],
                "chosen_relative_path": ["a"],
            }
        )
        summary, detail = replay_delayed_execution(
            candidates,
            decisions,
            latency_budget_ms=60.0,
            delay_bins=(0, 1),
            decision_cadence_seconds=10,
        )
        zero = summary[summary["delay_bins"].eq(0)].iloc[0]
        stale = summary[summary["delay_bins"].eq(1)].iloc[0]
        self.assertEqual(float(zero["network_qos_success_rate"]), 1.0)
        self.assertEqual(float(stale["execution_availability_rate"]), 0.0)
        self.assertEqual(float(stale["network_qos_success_rate"]), 0.0)
        self.assertEqual(
            int(detail[detail["delay_bins"].eq(1)].iloc[0]["replay_epoch"]),
            1,
        )

    def test_delayed_replay_does_not_treat_wall_clock_gap_as_one_bin(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 1],
                "bin_epoch": [100, 120],
                "bin_seconds": [10, 10],
                "relative_path": ["a", "a"],
                "target_next": [40.0, 30.0],
                "is_feasible_path": [1, 1],
            }
        )
        decisions = pd.DataFrame(
            {
                "session_bin_index": [0],
                "decision_bin_epoch": [100],
                "policy_name": ["test_policy"],
                "chosen_relative_path": ["a"],
            }
        )
        summary, detail = replay_delayed_execution(
            candidates,
            decisions,
            latency_budget_ms=60.0,
            delay_bins=(1, 2),
            decision_cadence_seconds=10,
        )
        one_bin = detail[detail["delay_bins"].eq(1)].iloc[0]
        two_bins = detail[detail["delay_bins"].eq(2)].iloc[0]
        self.assertEqual(float(one_bin["expected_replay_bin_epoch"]), 110.0)
        self.assertEqual(int(one_bin["trace_row_matched"]), 0)
        self.assertEqual(int(one_bin["trace_endpoint_observed"]), 0)
        self.assertEqual(
            one_bin["endpoint_observation_status"],
            "unobserved_acquisition_endpoint",
        )
        self.assertTrue(pd.isna(one_bin["network_qos_success"]))
        self.assertTrue(pd.isna(one_bin["replay_bin_epoch"]))
        self.assertEqual(float(two_bins["expected_replay_bin_epoch"]), 120.0)
        self.assertEqual(int(two_bins["trace_row_matched"]), 1)
        self.assertEqual(int(two_bins["trace_endpoint_observed"]), 1)
        self.assertEqual(float(two_bins["replay_bin_epoch"]), 120.0)
        one_bin_summary = summary[summary["delay_bins"].eq(1)].iloc[0]
        self.assertEqual(
            float(one_bin_summary["trace_endpoint_observability_rate"]),
            0.0,
        )
        self.assertTrue(
            pd.isna(one_bin_summary["execution_feasibility_rate_when_observed"])
        )

    def test_session_holdout_keeps_autocorrelated_groups_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": [
                    path
                    for path in ("a", "b", "c", "d", "e", "f")
                    for _ in range(5)
                ],
                "bin_epoch": list(range(5)) * 6,
            }
        )
        train, val, test = split_group_holdout(
            frame,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=7,
        )
        train_groups = set(train["relative_path"])
        val_groups = set(val["relative_path"])
        test_groups = set(test["relative_path"])
        self.assertFalse(train_groups & val_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(val_groups & test_groups)
        self.assertEqual(train_groups | val_groups | test_groups, set("abcdef"))

    def test_graph_features_are_partition_local(self) -> None:
        train = pd.DataFrame(
            {
                "session_bin_index": [0, 0],
                "relative_path": ["train-a", "train-b"],
                "latency_mean_ms": [10.0, 20.0],
                "path_state": ["active", "active"],
                "target_hint": ["x", "y"],
                "location": ["a", "b"],
                "observed_replies": [10, 10],
                "burst_indicator": [0.0, 0.0],
            }
        )
        test = train.copy()
        test["relative_path"] = ["test-a", "test-b"]
        test["latency_mean_ms"] = [1000.0, 2000.0]
        train_graph = add_graph_snapshot_features(train)
        self.assertEqual(train_graph["peer_latency_mean"].tolist(), [20.0, 10.0])
        self.assertEqual(
            train_graph["peer_latency_observed_count"].tolist(),
            [1, 1],
        )
        self.assertEqual(train_graph["peer_latency_available"].tolist(), [1, 1])

    def test_graph_features_fail_closed_without_observed_peers(self) -> None:
        frame = pd.DataFrame(
            {
                "session_bin_index": [0, 1, 1],
                "relative_path": ["solo", "a", "b"],
                "latency_mean_ms": [123.0, 10.0, 20.0],
                "path_state": ["active", "active", "inactive"],
                "target_hint": ["solo-target", "shared", "shared"],
                "location": ["solo-location", "x", "y"],
                "observed_replies": [7, 8, 9],
                "burst_indicator": [0.4, 0.1, 0.2],
            }
        )

        graph = add_graph_snapshot_features(frame)
        solo = graph.iloc[0]
        moment_columns = [
            "peer_latency_mean",
            "peer_latency_std",
            "state_peer_latency_mean",
            "state_peer_latency_std",
            "target_peer_latency_mean",
            "target_peer_latency_std",
            "peer_reply_mean",
            "peer_reply_std",
            "peer_burst_indicator_mean",
            "peer_burst_indicator_std",
            "peer_latency_gap",
        ]
        self.assertTrue(solo[moment_columns].isna().all())
        count_columns = [
            "peer_latency_observed_count",
            "state_peer_latency_observed_count",
            "target_peer_latency_observed_count",
            "peer_reply_observed_count",
            "peer_burst_indicator_observed_count",
        ]
        availability_columns = [
            "peer_latency_available",
            "state_peer_latency_available",
            "target_peer_latency_available",
            "peer_reply_available",
            "peer_burst_indicator_available",
        ]
        self.assertEqual(solo[count_columns].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(solo[availability_columns].tolist(), [0, 0, 0, 0, 0])

        # Row a has one global/target peer, but no peer in its path-state class.
        row_a = graph.iloc[1]
        self.assertEqual(float(row_a["peer_latency_mean"]), 20.0)
        self.assertEqual(int(row_a["peer_latency_observed_count"]), 1)
        self.assertEqual(int(row_a["peer_latency_available"]), 1)
        self.assertTrue(pd.isna(row_a["state_peer_latency_mean"]))
        self.assertTrue(pd.isna(row_a["state_peer_latency_std"]))
        self.assertEqual(int(row_a["state_peer_latency_observed_count"]), 0)
        self.assertEqual(int(row_a["state_peer_latency_available"]), 0)
        self.assertEqual(float(row_a["target_peer_latency_std"]), 0.0)

        selected = graph_context_feature_columns(graph)
        for indicator in count_columns + availability_columns:
            self.assertIn(indicator, selected)

    def test_nonconcurrent_trace_requires_explicit_counterfactual_mode(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["path-a", "path-a", "path-b", "path-b"],
                "bin_epoch": [1, 2, 11, 12],
            }
        )
        with self.assertRaisesRegex(ValueError, "no simultaneously observed"):
            assign_decision_groups(frame)
        aligned, audit = assign_decision_groups(
            frame,
            allow_normalized_counterfactual=True,
        )
        self.assertEqual(
            audit["decision_alignment"],
            "normalized_stage_counterfactual",
        )
        self.assertFalse(audit["supports_shadow_policy_replay"])
        self.assertFalse(audit["supports_candidate_outcome_shadow_replay"])
        self.assertFalse(audit["supports_literal_single_controller_steering"])
        self.assertFalse(audit["supports_closed_loop_deployment_evidence"])
        self.assertEqual(aligned["session_bin_index"].tolist(), [0, 1, 0, 1])

    def test_concurrent_trace_uses_actual_timestamps(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["path-a", "path-b", "path-a", "path-b"],
                "bin_epoch": [1, 1, 2, 2],
            }
        )
        aligned, audit = assign_decision_groups(frame)
        self.assertEqual(audit["decision_alignment"], "actual_timestamp")
        self.assertTrue(audit["supports_shadow_policy_replay"])
        self.assertTrue(audit["supports_candidate_outcome_shadow_replay"])
        self.assertFalse(audit["supports_literal_single_controller_steering"])
        self.assertEqual(
            audit["controller_topology_scope"],
            "not established by timestamp concurrency alone",
        )
        self.assertFalse(audit["supports_closed_loop_deployment_evidence"])
        self.assertEqual(aligned["session_bin_index"].tolist(), [0, 0, 1, 1])

        _, controller_audit = assign_decision_groups(
            frame,
            literal_single_controller_steering=True,
        )
        self.assertTrue(
            controller_audit["supports_literal_single_controller_steering"]
        )
        self.assertIn(
            "one steering controller",
            controller_audit["controller_topology_scope"],
        )

    def test_orbital_physics_bounds_are_monotonic(self) -> None:
        low_elevation = slant_range_km(550.0, 10.0)
        overhead = slant_range_km(550.0, 90.0)
        self.assertGreater(low_elevation, overhead)
        self.assertAlmostEqual(overhead, 550.0, places=6)
        self.assertEqual(control_horizon_margin_ms(1.0, 100.0), 900.0)

    def test_future_targets_never_enter_model_features(self) -> None:
        frame = pd.DataFrame(
            {
                "latency_mean_ms": [20.0],
                "target_next": [21.0],
                "target_mean_3": [22.0],
                "target_cumulative_5": [110.0],
                "target_available_3": [1],
                "bin_epoch": [100],
            }
        )
        features = default_feature_columns(frame)
        self.assertEqual(features, ["latency_mean_ms"])

    def test_graph_expert_does_not_reuse_temporal_history(self) -> None:
        frame = pd.DataFrame(
            {
                "peer_latency_mean": [20.0],
                "location_degree": [2],
                "latency_mean_ms_lag_1": [21.0],
            }
        )
        features = graph_context_feature_columns(frame)
        self.assertIn("peer_latency_mean", features)
        self.assertNotIn("latency_mean_ms_lag_1", features)

    def test_physics_trace_has_concurrent_paths_without_feature_leakage(self) -> None:
        frame, metadata = build_trace(
            bin_seconds=5,
            duration_hours=0.05,
            satellites=3,
            gateways=2,
            altitude_km=550.0,
            seed=7,
        )
        self.assertTrue(metadata["concurrent_alternative_paths"])
        self.assertTrue(metadata["injected_events_confined_to_test"])
        self.assertEqual(frame["relative_path"].nunique(), 6)
        self.assertTrue((frame.groupby("bin_epoch").size() == 6).all())
        event_rows = frame[frame["attenuation_event"].astype(bool)]
        first_epoch = int(frame["bin_epoch"].min())
        duration_seconds = int(frame["bin_epoch"].max()) - first_epoch + 5
        self.assertGreaterEqual(
            int(event_rows["bin_epoch"].min()),
            first_epoch + int(duration_seconds * 0.85),
        )
        frame["session_bin_index"] = frame.groupby("relative_path").cumcount()
        forecast = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1, 2],
            horizon_bins=1,
            decision_cadence_seconds=5,
        )
        temporal_features = default_feature_columns(forecast)
        graph_features = graph_context_feature_columns(
            add_graph_snapshot_features(forecast)
        )
        for forbidden in (
            "handover_event",
            "attenuation_event",
            "elevation_degrees",
            "propagation_lower_bound_ms",
            "queue_state",
        ):
            self.assertNotIn(forbidden, temporal_features)
            self.assertNotIn(forbidden, graph_features)

    def test_external_trace_adapter_builds_canonical_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "trace.csv"
            pd.DataFrame(
                {
                    "path": ["a", "a", "b", "b"],
                    "timestamp": [100, 110, 100, 110],
                    "rtt": [20.0, 21.0, 30.0, 29.0],
                }
            ).to_csv(source_path, index=False)
            output = load_compatible_latency_trace(
                source_path,
                column_map={
                    "relative_path": "path",
                    "bin_epoch": "timestamp",
                    "latency_mean_ms": "rtt",
                },
                dataset_name="test_trace",
                bin_seconds=10,
            )
            self.assertEqual(len(output), 4)
            self.assertIn("session_bin_index", output)
            self.assertEqual(output["location"].nunique(), 1)

    def test_commect_adapter_preserves_distinct_ten_second_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "Operator_A_RTT.csv"
            pd.DataFrame(
                {
                    "Time": [
                        "'11-Oct-2023 13:11:04.100'",
                        "'11-Oct-2023 13:11:09.900'",
                        "'11-Oct-2023 13:11:10.100'",
                        "'11-Oct-2023 13:11:19.900'",
                    ],
                    "Latency (ms)": [20.0, 22.0, 30.0, 32.0],
                    "RSRP (dBm)": [-90.0, -91.0, -92.0, -93.0],
                }
            ).to_csv(source_path, index=False)
            output = _load_and_bin(
                source_path,
                path_name="operator_a_5g",
                bin_seconds=10,
            )
            self.assertEqual(len(output), 2)
            self.assertEqual(output["bin_epoch"].nunique(), 2)
            self.assertEqual(
                int(output["bin_epoch"].iloc[1] - output["bin_epoch"].iloc[0]),
                10,
            )
            self.assertIn("source_timestamp_median", output.columns)
            self.assertEqual(
                output["source_timestamp_median"].dt.second.tolist(),
                [7, 15],
            )

    def test_victoria_builder_pools_hour_boundary_replies(self) -> None:
        frame = pd.DataFrame(
            {
                "relative_path": ["a", "a", "b"],
                "bin_epoch": [100, 100, 100],
                "observed_replies": [1, 600, 500],
                "window_start": ["hour_0", "hour_1", "hour_1"],
                "latency_mean_ms": [10.0, 20.0, 30.0],
                "latency_std_ms": [0.0, 0.0, 0.0],
                "latency_min_ms": [10.0, 20.0, 30.0],
                "latency_max_ms": [10.0, 20.0, 30.0],
                "ttl_mean": [63.0, 63.0, 64.0],
                "ttl_min": [63, 63, 64],
                "ttl_max": [63, 63, 64],
                "icmp_seq_min": [99, 1, 1],
                "icmp_seq_max": [99, 600, 500],
            }
        )
        output, audit = _pool_boundary_bins(frame)
        self.assertEqual(len(output), 2)
        self.assertEqual(audit["overlapping_boundary_pairs"], 1)
        pooled = output.loc[output["relative_path"].eq("a")].iloc[0]
        self.assertEqual(int(pooled["observed_replies"]), 601)
        self.assertAlmostEqual(
            float(pooled["latency_mean_ms"]),
            (10.0 + 600.0 * 20.0) / 601.0,
        )
        self.assertEqual(audit["input_reply_count"], 1101)
        self.assertEqual(audit["output_reply_count"], 1101)
        self.assertEqual(audit["overshoot_replies_pooled"], 1)
        self.assertEqual(
            float(output.loc[output["relative_path"].eq("a"), "latency_mean_ms"].iloc[0]),
            float(pooled["latency_mean_ms"]),
        )

    def test_calibrated_risk_downweights_noisier_expert(self) -> None:
        temporal = fit_expert_calibration([10, 20, 30], [10, 20, 30])
        graph = fit_expert_calibration([10, 20, 30], [5, 30, 10])
        candidate = pd.DataFrame(
            {"pred_forecast": [20.0], "pred_graph": [40.0], "risk": [0.0]}
        )
        output = add_calibrated_mixture_risk_scores(
            candidate,
            temporal,
            graph,
            CalibratedRiskConfig(),
            service_risk_column="risk",
        )
        self.assertGreater(
            float(output["temporal_expert_weight"].iloc[0]),
            float(output["graph_expert_weight"].iloc[0]),
        )
        self.assertGreater(float(output["pred_mixture_std"].iloc[0]), 0.0)

    def test_fusion_variance_uses_squared_weights_and_residual_covariance(self) -> None:
        temporal = replace(
            fit_expert_calibration([10, 20, 30], [9, 19, 29]),
            residual_scale_ms=2.0,
            residual_variance_ms2=4.0,
            paired_residual_covariance_ms2=1.5,
        )
        graph = replace(
            fit_expert_calibration([10, 20, 30], [8, 18, 28]),
            residual_scale_ms=4.0,
            residual_variance_ms2=16.0,
            paired_residual_covariance_ms2=1.5,
        )
        output = add_calibrated_mixture_risk_scores(
            pd.DataFrame({"pred_forecast": [20.0], "pred_graph": [30.0]}),
            temporal,
            graph,
        )
        temporal_weight = float(output["temporal_expert_weight"].iloc[0])
        graph_weight = float(output["graph_expert_weight"].iloc[0])
        expected_variance = (
            temporal_weight**2 * 2.0**2
            + graph_weight**2 * 4.0**2
            + 2.0 * temporal_weight * graph_weight * 1.5
        )
        self.assertAlmostEqual(
            float(output["pred_fusion_error_std"].iloc[0]) ** 2,
            expected_variance,
        )
        self.assertAlmostEqual(
            float(output["pred_mixture_std"].iloc[0]),
            float(output["pred_fusion_error_std"].iloc[0]),
        )

    def test_reviewer_pair_audit_uses_shared_calibration_covariance(self) -> None:
        truth = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        temporal_prediction = pd.Series([9.0, 22.0, 27.0, 44.0, 45.0])
        graph_prediction = pd.Series([8.0, 16.0, 31.0, 43.0, 50.0])
        temporal, graph = _fit_paired_expert_calibrations(
            truth,
            temporal_prediction,
            graph_prediction,
        )
        temporal_centered = (
            truth - temporal_prediction - temporal.residual_bias_ms
        )
        graph_centered = truth - graph_prediction - graph.residual_bias_ms
        expected_covariance = float(temporal_centered.cov(graph_centered))
        self.assertAlmostEqual(
            temporal.paired_residual_covariance_ms2,
            expected_covariance,
        )
        self.assertAlmostEqual(
            graph.paired_residual_covariance_ms2,
            expected_covariance,
        )

        output = add_calibrated_mixture_risk_scores(
            pd.DataFrame({"pred_forecast": [20.0], "pred_graph": [30.0]}),
            temporal,
            graph,
        )
        temporal_weight = float(output["temporal_expert_weight"].iloc[0])
        graph_weight = float(output["graph_expert_weight"].iloc[0])
        self.assertNotAlmostEqual(temporal_weight, 0.5)
        expected_variance = (
            temporal_weight**2 * float(temporal.residual_variance_ms2)
            + graph_weight**2 * float(graph.residual_variance_ms2)
            + 2.0 * temporal_weight * graph_weight * expected_covariance
        )
        self.assertAlmostEqual(
            float(output["pred_fusion_error_std"].iloc[0]) ** 2,
            expected_variance,
        )

    def test_control_loop_latency_is_counted_in_success(self) -> None:
        candidate = pd.DataFrame(
            {
                "relative_path": ["a", "b"],
                "location": ["x", "x"],
                "path_state": ["active", "active"],
                "session_bin_index": [0, 0],
                "latency_mean_ms": [50.0, 70.0],
                "target_next": [58.0, 70.0],
                "score": [1.0, 2.0],
            }
        )
        summary, decisions = evaluate_decision_policies(
            candidate,
            latency_budget_ms=60.0,
            policy_columns={"test": "score"},
            control_loop_latency_ms=5.0,
            decision_window_seconds=1.0,
        )
        self.assertEqual(int(decisions["success_under_budget"].iloc[0]), 0)
        self.assertEqual(
            float(summary["mean_end_to_end_latency_ms"].iloc[0]), 63.0
        )

    def test_unavailable_path_is_removed_before_policy_scoring(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0],
                "relative_path": ["unavailable", "available"],
                "location": ["a", "b"],
                "path_state": ["degraded", "active"],
                "latency_mean_ms": [1.0, 20.0],
                "target_next": [2.0, 21.0],
                "is_feasible_path": [0, 1],
                "score": [1.0, 20.0],
            }
        )
        summary, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"test": "score"},
        )
        self.assertEqual(
            decisions.iloc[0]["chosen_relative_path"],
            "available",
        )
        self.assertEqual(
            int(decisions.iloc[0]["feasible_candidate_count"]),
            1,
        )
        self.assertEqual(
            float(summary.iloc[0]["no_feasible_candidate_rate"]),
            0.0,
        )

    def test_qos_shield_is_lexicographic_only_for_mixed_snapshots(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1, 2, 2],
                "relative_path": ["a", "b"] * 3,
                "latency_mean_ms": [50.0, 70.0, 30.0, 40.0, 80.0, 90.0],
                "is_feasible_path": [1] * 6,
                "fallback": [100.0, 1.0, 100.0, 1.0, 100.0, 1.0],
            }
        )
        output = add_qos_shielded_scores(candidates, "fallback", 60.0)
        selected = output.loc[
            output.groupby("session_bin_index")[
                "pred_qos_shielded_operational"
            ].idxmin()
        ]
        # Mixed: protect the currently compliant path. All/none compliant:
        # retain the validation-selected predictive fallback.
        self.assertEqual(selected["relative_path"].tolist(), ["a", "b", "b"])
        self.assertEqual(
            selected["qos_shield_mode"].tolist(),
            ["mixed_qos_safeguard", "all_qos_fallback", "no_qos_fallback"],
        )

    def test_qos_threshold_changes_the_executed_shield_branch(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0],
                "relative_path": ["a", "b"],
                "latency_mean_ms": [50.0, 70.0],
                "is_feasible_path": [1, 1],
                "fallback": [100.0, 1.0],
            }
        )
        strict = add_qos_shielded_scores(candidates, "fallback", 60.0)
        relaxed = add_qos_shielded_scores(candidates, "fallback", 80.0)
        strict_choice = strict.loc[
            strict["pred_qos_shielded_operational"].idxmin()
        ]
        relaxed_choice = relaxed.loc[
            relaxed["pred_qos_shielded_operational"].idxmin()
        ]
        self.assertEqual(strict_choice["relative_path"], "a")
        self.assertEqual(strict_choice["qos_shield_mode"], "mixed_qos_safeguard")
        self.assertEqual(relaxed_choice["relative_path"], "b")
        self.assertEqual(relaxed_choice["qos_shield_mode"], "all_qos_fallback")

    def test_validation_gate_abstains_when_reactive_has_best_qos(self) -> None:
        selected = select_validation_gated_fallback(
            {
                "reactive": [40.0, 50.0, 55.0],
                "graph": [30.0, 45.0, 80.0],
                "ensemble": [35.0, 75.0, 85.0],
            },
            latency_budget_ms=60.0,
        )
        self.assertEqual(selected, "reactive")

    def test_validation_gate_uses_latency_only_after_qos_tie(self) -> None:
        selected = select_validation_gated_fallback(
            {
                "reactive": [40.0, 50.0, 90.0],
                "graph": [30.0, 45.0, 80.0],
                "ensemble": [35.0, 55.0, 100.0],
            },
            latency_budget_ms=60.0,
        )
        self.assertEqual(selected, "graph")

    def test_qos_fallback_comparators_share_the_shield_but_isolate_fallbacks(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b", "a", "b"],
                "latency_mean_ms": [50.0, 70.0, 30.0, 40.0],
                "is_feasible_path": [1, 1, 1, 1],
                "context": [100.0, 1.0, 1.0, 100.0],
                "ensemble": [1.0, 100.0, 100.0, 1.0],
            }
        )
        context = add_qos_shielded_scores(
            candidates,
            "context",
            60.0,
            "pred_qos_context",
        )
        ensemble = add_qos_shielded_scores(
            candidates,
            "ensemble",
            60.0,
            "pred_qos_ensemble",
        )
        context_choice = context.loc[
            context.groupby("session_bin_index")["pred_qos_context"].idxmin(),
            "relative_path",
        ].tolist()
        ensemble_choice = ensemble.loc[
            ensemble.groupby("session_bin_index")["pred_qos_ensemble"].idxmin(),
            "relative_path",
        ].tolist()
        # Both variants preserve the compliant path in the mixed snapshot,
        # while the all-compliant snapshot isolates fallback selection.
        self.assertEqual(context_choice, ["a", "a"])
        self.assertEqual(ensemble_choice, ["a", "b"])

    def test_online_switch_penalty_changes_the_executed_decision(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b", "a", "b"],
                "location": ["x", "x", "x", "x"],
                "path_state": ["active"] * 4,
                "latency_mean_ms": [10.0, 20.0, 12.0, 10.0],
                "target_next": [10.0, 20.0, 12.0, 10.0],
                "score": [10.0, 20.0, 12.0, 10.0],
            }
        )
        _, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"switch_aware": "score"},
            online_switch_penalties_ms={"switch_aware": 5.0},
        )
        self.assertEqual(decisions["chosen_relative_path"].tolist(), ["a", "a"])
        self.assertEqual(decisions["switched_path"].sum(), 0)

    def test_switch_state_resets_at_declared_cadence_gap(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 2, 2, 3, 3],
                "bin_epoch": [0.0, 0.0, 20.0, 20.0, 30.0, 30.0],
                "relative_path": ["a", "b"] * 3,
                "location": ["x"] * 6,
                "path_state": ["active"] * 6,
                "latency_mean_ms": [10.0, 20.0, 12.0, 10.0, 2.0, 10.0],
                "target_next": [10.0, 20.0, 12.0, 10.0, 2.0, 10.0],
                "score": [10.0, 20.0, 12.0, 10.0, 2.0, 10.0],
            }
        )
        summary, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"switch_aware": "score"},
            decision_window_seconds=10.0,
            online_switch_penalties_ms={"switch_aware": 5.0},
        )
        # Carrying path a across the missing 10-second slot would select a at
        # the second decision. Resetting state makes b the unbiased choice and
        # does not count the cross-gap path difference as a switch.
        self.assertEqual(
            decisions["chosen_relative_path"].tolist(),
            ["a", "b", "a"],
        )
        self.assertEqual(decisions["switched_path"].tolist(), [0, 0, 1])
        self.assertEqual(
            decisions["switch_transition_eligible"].tolist(),
            [0, 0, 1],
        )
        self.assertEqual(decisions["continuity_reset"].tolist(), [0, 1, 0])
        self.assertIn("epoch_gap", decisions["continuity_reset_reason"].iloc[1])
        self.assertEqual(float(summary["switch_rate"].iloc[0]), 1.0)
        self.assertEqual(int(summary["continuity_reset_count"].iloc[0]), 1)
        self.assertEqual(int(summary["continuity_segment_count"].iloc[0]), 2)

    def test_switch_state_uses_bin_seconds_cadence_when_argument_is_absent(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "bin_epoch": [0.0, 0.0, 20.0, 20.0],
                "bin_seconds": [10.0] * 4,
                "relative_path": ["a", "b", "a", "b"],
                "location": ["x"] * 4,
                "path_state": ["active"] * 4,
                "latency_mean_ms": [10.0, 20.0, 12.0, 10.0],
                "target_next": [10.0, 20.0, 12.0, 10.0],
                "score": [10.0, 20.0, 12.0, 10.0],
            }
        )
        summary, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"switch_aware": "score"},
            online_switch_penalties_ms={"switch_aware": 5.0},
        )
        self.assertEqual(decisions["chosen_relative_path"].tolist(), ["a", "b"])
        self.assertEqual(decisions["continuity_reset"].tolist(), [0, 1])
        self.assertEqual(
            summary["continuity_cadence_source"].iloc[0],
            "column:bin_seconds",
        )

    def test_switch_state_resets_when_campaign_changes(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1, 2, 2],
                "bin_epoch": [0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
                "campaign_id": ["one", "one", "two", "two", "two", "two"],
                "relative_path": ["a", "b"] * 3,
                "location": ["x"] * 6,
                "path_state": ["active"] * 6,
                "latency_mean_ms": [10.0, 20.0, 12.0, 10.0, 12.0, 9.0],
                "target_next": [10.0, 20.0, 12.0, 10.0, 12.0, 9.0],
                "score": [10.0, 20.0, 12.0, 10.0, 12.0, 9.0],
            }
        )
        summary, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"switch_aware": "score"},
            decision_window_seconds=10.0,
            online_switch_penalties_ms={"switch_aware": 5.0},
        )
        self.assertEqual(decisions["chosen_relative_path"].tolist(), ["a", "b", "b"])
        self.assertEqual(decisions["continuity_reset"].tolist(), [0, 1, 0])
        self.assertEqual(
            decisions["continuity_reset_reason"].tolist(),
            ["initial_decision", "continuity_key_change", "continuous"],
        )
        self.assertEqual(decisions["switched_path"].tolist(), [0, 0, 0])
        self.assertEqual(int(summary["switch_transition_count"].iloc[0]), 1)
        self.assertEqual(summary["continuity_key_column"].iloc[0], "campaign_id")

    def test_decision_policy_exports_explainable_ai_fields(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0, 1, 1],
                "relative_path": ["a", "b", "a", "b"],
                "location": ["x", "x", "x", "x"],
                "path_state": ["active"] * 4,
                "latency_mean_ms": [20.0, 30.0, 20.0, 21.0],
                "target_next": [22.0, 33.0, 21.0, 23.0],
                "pred_calibrated_fusion": [20.0, 24.0, 20.0, 19.0],
                "pred_disagreement": [1.0, 6.0, 2.0, 3.0],
                "pred_disagreement_only": [20.5, 27.0, 21.0, 20.5],
                "pred_mixture_std": [0.5, 1.5, 0.4, 1.2],
                "pred_ensemble_std": [0.7, 2.0, 0.6, 1.5],
                "service_risk_ms": [0.1, 3.0, 0.2, 2.0],
                "score": [20.0, 29.0, 20.0, 20.5],
            }
        )
        _, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={"xai_policy": "score"},
            online_switch_penalties_ms={"xai_policy": 2.0},
        )
        required_columns = {
            "xai_latency_component_ms",
            "xai_disagreement_component_ms",
            "xai_uncertainty_component_ms",
            "xai_service_risk_component_ms",
            "xai_switch_component_ms",
            "xai_calibration_component_ms",
            "xai_score_branch",
            "xai_gate_active",
            "xai_fallback_policy",
            "xai_dominant_component",
            "xai_runner_up_relative_path",
            "xai_score_margin_ms",
            "xai_counterfactual_reason",
        }
        self.assertTrue(required_columns.issubset(decisions.columns))
        self.assertEqual(decisions["xai_runner_up_relative_path"].tolist(), ["b", "b"])
        self.assertTrue(decisions["xai_score_margin_ms"].notna().all())
        attribution = summarize_xai_attribution(decisions)
        self.assertEqual(int(attribution["decision_count"].iloc[0]), 2)
        self.assertIn("mean_attr_latency", attribution.columns)

    def test_qos_shield_explanation_reports_the_executed_branch(self) -> None:
        candidates = pd.DataFrame(
            {
                "session_bin_index": [0, 0],
                "relative_path": ["safe", "over_budget"],
                "location": ["x", "x"],
                "path_state": ["active", "active"],
                "latency_mean_ms": [50.0, 70.0],
                "target_next": [52.0, 68.0],
                "is_feasible_path": [1, 1],
                "fallback": [100.0, 1.0],
            }
        )
        candidates = add_qos_shielded_scores(candidates, "fallback", 60.0)
        _, decisions = evaluate_decision_policies(
            candidates,
            policy_columns={
                "qos_shielded": "pred_qos_shielded_operational"
            },
        )
        self.assertEqual(decisions.iloc[0]["qos_shield_mode"], "mixed_qos_safeguard")
        self.assertEqual(decisions.iloc[0]["xai_score_branch"], "mixed_qos_safeguard")
        self.assertEqual(decisions.iloc[0]["xai_fallback_policy"], "current_qos_latency")


if __name__ == "__main__":
    unittest.main()
