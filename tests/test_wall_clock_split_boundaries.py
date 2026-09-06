from __future__ import annotations

import unittest

import pandas as pd

from open_leo_latency_routing.features.temporal import (
    WALL_CLOCK_SPLIT_AUDIT_ATTR,
    build_forecast_table,
    build_rolling_origin_split_plan,
    build_wall_clock_decision_schedule,
    split_train_calibration_selection_test,
)
from scripts.run_commect_rolling_origin_validation import _closed_partition
from scripts.run_service_path_experiments import _target_retention_protocol


class WallClockSplitBoundaryTests(unittest.TestCase):
    @staticmethod
    def _raw_schedule_frame() -> pd.DataFrame:
        observed = [0, 10, 30, 40, 50, 60, 70, 80, 90]
        return pd.DataFrame(
            {
                "relative_path": ["a"] * len(observed),
                "bin_epoch": observed,
                "bin_seconds": [10] * len(observed),
            }
        )

    @staticmethod
    def _eligible_rows(epochs: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "relative_path": ["a"] * len(epochs),
                "bin_epoch": epochs,
                "target_next_bin_epoch": [epoch + 10 for epoch in epochs],
            }
        )

    @staticmethod
    def _boundary_signature(parts: tuple[pd.DataFrame, ...]) -> tuple[tuple[object, ...], ...]:
        audit = parts[0].attrs[WALL_CLOCK_SPLIT_AUDIT_ATTR]
        return tuple(
            (
                name,
                audit["partitions"][name]["first_schedule_index"],
                audit["partitions"][name]["last_schedule_index"],
                audit["partitions"][name]["first_bin_epoch"],
                audit["partitions"][name]["last_bin_epoch"],
            )
            for name in ("train", "calibration", "selection", "test")
        )

    def test_schedule_includes_raw_cadence_gaps(self) -> None:
        schedule = build_wall_clock_decision_schedule(self._raw_schedule_frame())
        self.assertEqual(schedule["bin_epoch"].tolist(), list(range(0, 100, 10)))
        audit = schedule.attrs["wall_clock_decision_schedule_audit"]
        self.assertEqual(audit["scheduled_decision_epoch_count"], 10)
        self.assertEqual(audit["observed_raw_decision_epoch_count"], 9)
        self.assertEqual(audit["missing_raw_decision_epoch_count"], 1)
        self.assertFalse(audit["target_availability_used_for_schedule"])

    def test_fixed_boundaries_ignore_added_or_deleted_target_rows(self) -> None:
        schedule = build_wall_clock_decision_schedule(self._raw_schedule_frame())
        sparse = self._eligible_rows([0, 30, 40, 60, 70, 80])
        augmented = self._eligible_rows([0, 10, 20, 30, 40, 50, 60, 70, 80])
        kwargs = {
            "train_ratio": 0.4,
            "calibration_ratio": 0.2,
            "selection_ratio": 0.2,
            "test_ratio": 0.2,
            "decision_schedule": schedule,
        }

        sparse_parts = split_train_calibration_selection_test(sparse, **kwargs)
        augmented_parts = split_train_calibration_selection_test(augmented, **kwargs)

        self.assertEqual(
            self._boundary_signature(sparse_parts),
            self._boundary_signature(augmented_parts),
        )
        audit = sparse_parts[0].attrs[WALL_CLOCK_SPLIT_AUDIT_ATTR]
        self.assertTrue(audit["boundaries_declared_before_target_filtering"])
        self.assertFalse(audit["target_availability_used_for_boundary_derivation"])
        self.assertEqual(audit["partitions"]["train"]["last_bin_epoch"], 30.0)
        self.assertEqual(audit["partitions"]["calibration"]["last_bin_epoch"], 50.0)

    def test_rolling_intervals_and_closure_ignore_outcome_row_edges(self) -> None:
        raw = pd.DataFrame(
            {
                "relative_path": ["a"] * 60,
                "bin_epoch": list(range(0, 600, 10)),
                "bin_seconds": [10] * 60,
            }
        )
        schedule = build_wall_clock_decision_schedule(raw)
        plan = build_rolling_origin_split_plan(schedule, fold_count=3)
        first_test = plan["folds"][0]["partitions"]["test"]
        start = int(first_test["first_bin_epoch"])
        end = int(first_test["last_bin_epoch"])
        # No eligible decision row exists at either declared interval edge.
        eligible_epochs = list(range(start + 10, end, 10))
        forecast = pd.DataFrame(
            {
                "relative_path": ["a"] * len(eligible_epochs),
                "session_bin_index": list(range(len(eligible_epochs))),
                "bin_epoch": eligible_epochs,
                "target_next_bin_epoch": [value + 10 for value in eligible_epochs],
            }
        )
        closed = _closed_partition(
            forecast,
            partition_start_epoch=start,
            partition_end_epoch=end,
        )

        self.assertEqual(plan["block_size_scheduled_epochs"], 10)
        self.assertEqual(closed["bin_epoch"].tolist(), eligible_epochs)
        self.assertEqual(float(closed["target_next_bin_epoch"].max()), float(end))

    def test_normalized_lens_retains_per_row_exact_targets(self) -> None:
        protocol = _target_retention_protocol(
            {
                "decision_alignment": "normalized_stage_counterfactual",
                "supports_shadow_policy_replay": False,
            }
        )
        self.assertEqual(protocol["mode"], "per_row_exact_target")
        self.assertFalse(protocol["require_complete_decision_epochs"])
        self.assertFalse(
            protocol["normalized_session_index_used_for_candidate_completeness"]
        )

        frame = pd.DataFrame(
            {
                "relative_path": ["session-a", "session-a", "session-b"],
                "session_bin_index": [0, 1, 0],
                "bin_epoch": [0, 10, 0],
                "bin_seconds": [10, 10, 10],
                "latency_mean_ms": [10.0, 11.0, 20.0],
                "observed_replies": [10, 10, 10],
                "path_state": ["active", "active", "active"],
                "window_duration": ["1h", "1h", "1h"],
                "probe_interval": ["1000ms", "1000ms", "1000ms"],
                "session_date": pd.to_datetime(["2026-08-21"] * 3),
            }
        )
        retained = build_forecast_table(
            frame,
            target_column="latency_mean_ms",
            lags=[1],
            horizon_bins=1,
            decision_cadence_seconds=10,
            require_complete_decision_epochs=False,
        )
        self.assertEqual(retained[["relative_path", "bin_epoch"]].values.tolist(), [["session-a", 0]])
        self.assertFalse(
            retained.attrs["exact_horizon_audit"][
                "require_complete_decision_epochs"
            ]
        )

        concurrent = _target_retention_protocol(
            {
                "decision_alignment": "actual_timestamp",
                "supports_shadow_policy_replay": True,
            }
        )
        self.assertTrue(concurrent["require_complete_decision_epochs"])


if __name__ == "__main__":
    unittest.main()
