"""Regression tests for signed XAI score-reconstruction fidelity."""

from __future__ import annotations

import unittest

import pandas as pd

from open_leo_latency_routing.optimization.explainability import (
    explain_candidate_score,
)
from scripts.build_transactions_evidence import _xai_case_studies


class XaiFidelityTests(unittest.TestCase):
    def test_candidate_explanation_exports_signed_and_absolute_totals(self) -> None:
        explanation = explain_candidate_score(
            pd.Series({"score": -3.0}),
            selected_score=-3.0,
            sort_column="score",
        )

        self.assertEqual(explanation["xai_explained_signed_total_ms"], -3.0)
        self.assertEqual(explanation["xai_explained_abs_total_ms"], 3.0)

    def test_case_study_fidelity_uses_signed_reconstruction(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "policy_name": "qos_shielded_operational_selector",
                    "qos_shield_mode": "mixed_qos_safeguard",
                    "dataset": "negative-score-regression",
                    "evaluation_case": "focused_test",
                    "session_bin_index": 7,
                    "chosen_relative_path": "path-a",
                    "xai_runner_up_relative_path": "path-b",
                    "reactive_latency_ms": 10.0,
                    "selected_online_score": -3.0,
                    "xai_explained_signed_total_ms": -3.0,
                    "xai_explained_abs_total_ms": 3.0,
                    "xai_runner_up_score_ms": -2.0,
                    "xai_score_margin_ms": 1.0,
                    "realized_next_latency_ms": 9.0,
                    "success_under_budget": 1,
                    "xai_fallback_policy": "current_qos_latency",
                    "xai_counterfactual_reason": "selected_lower_latency",
                }
            ]
        )

        case_studies = _xai_case_studies(decisions)

        self.assertEqual(len(case_studies), 1)
        self.assertEqual(
            case_studies.iloc[0]["explanation_fidelity_error_ms"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
