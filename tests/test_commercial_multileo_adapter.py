"""Tests for the claim-gated commercial multi-LEO adapter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.build_commercial_multileo_trace import build_commercial_multileo_trace
from scripts.run_commercial_multileo_validation import _require_validation_scope
from scripts.run_commercial_multileo_validation import (
    _four_way_split_audit,
    _resolve_campaign_gate_grouping,
)
from open_leo_latency_routing.features.temporal import (
    split_train_calibration_selection_test,
)


class CommercialMultiLeoAdapterTests(unittest.TestCase):
    def _write_trace(
        self,
        root: Path,
        *,
        days: int = 31,
        include_oneweb: bool = True,
        include_coordinates: bool = True,
        include_controller_id: bool = True,
        oneweb_longitude_offset: float = 0.0,
        conflicting_controller_ids: bool = False,
        include_campaign_id: bool = False,
        conflicting_campaign_ids: bool = False,
        single_campaign_id: bool = False,
    ) -> Path:
        rows = []
        operators = ["Starlink"] + (["Eutelsat OneWeb"] if include_oneweb else [])
        for day in range(days + 1):
            timestamp = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=day)
            for operator_index, operator in enumerate(operators):
                row = {
                    "ts": timestamp.isoformat(),
                    "network": operator,
                    "owd": 40.0 + operator_index,
                    "received": True,
                    "scenario": "urban",
                }
                if include_coordinates:
                    row["latitude"] = 41.8781
                    row["longitude"] = -93.0977 + (
                        oneweb_longitude_offset if operator_index else 0.0
                    )
                if include_controller_id:
                    row["controller_id"] = (
                        f"controller-{operator_index}"
                        if conflicting_controller_ids
                        else "vehicle-controller-1"
                    )
                if include_campaign_id:
                    campaign_index = 0 if single_campaign_id else int(
                        day >= max(1, (days + 1) // 2)
                    )
                    row["campaign_id"] = (
                        f"campaign-{campaign_index}-{operator_index}"
                        if conflicting_campaign_ids
                        else f"campaign-{campaign_index}"
                    )
                rows.append(row)
        path = root / "trace.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _build(self, path: Path, **overrides):
        arguments = {
            "bin_seconds": 10,
            "timeout_ms": 1000.0,
            "minimum_duration_days": 30.0,
            "minimum_concurrent_hours": 0.001,
            "maximum_p95_skew_ms": 100.0,
            "dataset_name": "test_multileo",
            "dataset_url": "https://example.test/dataset",
            "dataset_doi": "10.test/example",
            "license_name": "test-only",
            "independent_provenance": True,
        }
        arguments.update(overrides)
        source_columns = set(pd.read_csv(path, nrows=0).columns)
        column_map = {
            "timestamp": "ts",
            "operator": "network",
            "latency_ms": "owd",
            "packet_received": "received",
            "scenario": "scenario",
        }
        if {"latitude", "longitude"}.issubset(source_columns):
            column_map.update(
                {
                    "latitude": "latitude",
                    "longitude": "longitude",
                }
            )
        if "controller_id" in source_columns:
            column_map["controller_id"] = "controller_id"
        if "campaign_id" in source_columns:
            column_map["campaign_id"] = "campaign_id"
        return build_commercial_multileo_trace(
            path,
            column_map,
            **arguments,
        )

    def test_full_trace_passes_dataset_and_topology_claim_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame, metadata = self._build(
                self._write_trace(Path(temp_dir)),
                bin_seconds=86400,
            )
        self.assertEqual(set(frame["operator"]), {"starlink", "oneweb"})
        self.assertTrue(metadata["all_candidate_outcomes_observed"])
        self.assertTrue(metadata["spatial_colocation_pass"])
        self.assertTrue(metadata["shared_controller_provenance_pass"])
        self.assertTrue(metadata["same_controller_selectable_path_evidence"])
        self.assertTrue(metadata["long_duration_pass"])
        self.assertTrue(metadata["closes_independent_longitudinal_multileo_limitation"])
        self.assertEqual(metadata["complete_concurrent_epoch_fraction"], 1.0)

    def test_short_campaign_cannot_close_longitudinal_limitation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(self._write_trace(Path(temp_dir), days=2))
        self.assertTrue(metadata["all_candidate_outcomes_observed"])
        self.assertFalse(metadata["long_duration_pass"])
        self.assertFalse(metadata["closes_independent_longitudinal_multileo_limitation"])

    def test_time_alignment_without_gps_fails_closed_to_scoped_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(Path(temp_dir), include_coordinates=False),
                bin_seconds=86400,
            )
        self.assertTrue(metadata["all_candidate_outcomes_observed"])
        self.assertFalse(metadata["coordinate_fields_present"])
        self.assertFalse(metadata["spatial_colocation_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])
        self.assertFalse(metadata["concurrent_interchangeable_paths"])
        self.assertTrue(
            metadata["temporal_concurrency_audit"][
                "has_temporally_concurrent_candidates"
            ]
        )
        self.assertTrue(
            metadata["concurrency_audit"][
                "supports_candidate_outcome_shadow_replay"
            ]
        )
        self.assertFalse(
            metadata["concurrency_audit"][
                "supports_literal_single_controller_steering"
            ]
        )
        self.assertFalse(
            metadata["concurrency_audit"]["supports_single_controller_shadow_replay"]
        )
        self.assertFalse(
            metadata["concurrency_audit"]["supports_closed_loop_deployment_evidence"]
        )
        self.assertEqual(
            metadata["evidence_scope"],
            "scoped_location_unverified_time_aligned_comparison",
        )
        self.assertFalse(metadata["closes_independent_longitudinal_multileo_limitation"])

    def test_spatially_separated_operators_cannot_be_selectable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(
                    Path(temp_dir),
                    oneweb_longitude_offset=1.0,
                ),
                bin_seconds=86400,
                maximum_inter_operator_distance_meters=10_000.0,
            )
        self.assertEqual(
            metadata["effective_maximum_inter_operator_distance_meters"],
            100.0,
        )
        self.assertGreater(metadata["maximum_inter_operator_distance_meters"], 50_000.0)
        self.assertFalse(metadata["spatial_colocation_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])
        self.assertEqual(
            metadata["evidence_scope"],
            "scoped_spatially_separated_time_aligned_comparison",
        )

    def test_incomplete_gps_coverage_fails_colocation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir))
            source = pd.read_csv(path)
            source.loc[
                source["network"].eq("Eutelsat OneWeb").idxmax(),
                "latitude",
            ] = None
            source.to_csv(path, index=False)
            _, metadata = self._build(path, bin_seconds=86400)
        self.assertLess(metadata["complete_coordinate_pair_fraction"], 1.0)
        self.assertFalse(metadata["spatial_colocation_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])
        self.assertEqual(
            metadata["evidence_scope"],
            "scoped_incomplete_location_audit_paired_comparison",
        )

    def test_missing_packet_gps_cannot_be_hidden_by_bin_median(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir))
            source = pd.read_csv(path)
            source = pd.concat([source, source.copy()], ignore_index=True)
            first_oneweb = source["network"].eq("Eutelsat OneWeb").idxmax()
            source.loc[first_oneweb, "latitude"] = None
            source.to_csv(path, index=False)
            _, metadata = self._build(path, bin_seconds=86400)
        self.assertEqual(metadata["complete_coordinate_pair_fraction"], 1.0)
        self.assertLess(metadata["complete_source_coordinate_row_fraction"], 1.0)
        self.assertFalse(metadata["spatial_colocation_pass"])
        self.assertEqual(
            metadata["evidence_scope"],
            "scoped_incomplete_location_audit_paired_comparison",
        )

    def test_blank_gps_fields_fail_closed_without_becoming_absent_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir))
            source = pd.read_csv(path)
            source[["latitude", "longitude"]] = None
            source.to_csv(path, index=False)
            _, metadata = self._build(path, bin_seconds=86400)
        self.assertTrue(metadata["coordinate_fields_present"])
        self.assertEqual(metadata["complete_coordinate_pair_count"], 0)
        self.assertFalse(metadata["spatial_colocation_pass"])

    def test_blank_controller_ids_do_not_count_as_shared_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir))
            source = pd.read_csv(path)
            source["controller_id"] = None
            source.to_csv(path, index=False)
            _, metadata = self._build(path, bin_seconds=86400)
        self.assertTrue(metadata["mapped_controller_ids_present"])
        self.assertEqual(metadata["complete_controller_pair_count"], 0)
        self.assertFalse(metadata["shared_controller_provenance_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])

    def test_colocation_without_controller_provenance_is_only_convoy_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(Path(temp_dir), include_controller_id=False),
                bin_seconds=86400,
            )
        self.assertTrue(metadata["spatial_colocation_pass"])
        self.assertFalse(metadata["shared_controller_provenance_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])
        self.assertEqual(
            metadata["evidence_scope"],
            "scoped_near_concurrent_colocated_or_convoy_replay",
        )

    def test_conflicting_controller_ids_override_external_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(
                    Path(temp_dir),
                    conflicting_controller_ids=True,
                ),
                bin_seconds=86400,
                same_controller_provenance=True,
                controller_provenance_note="data-owner confirmation dated 2026-08-21",
            )
        self.assertGreater(metadata["conflicting_controller_pair_count"], 0)
        self.assertFalse(metadata["shared_controller_provenance_pass"])
        self.assertFalse(metadata["same_controller_selectable_path_evidence"])

    def test_documented_controller_provenance_requires_a_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir), include_controller_id=False)
            with self.assertRaisesRegex(ValueError, "controller_provenance_note"):
                self._build(path, same_controller_provenance=True)

    def test_documented_controller_provenance_can_replace_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(Path(temp_dir), include_controller_id=False),
                bin_seconds=86400,
                same_controller_provenance=True,
                controller_provenance_note="campaign topology document, section 2",
            )
        self.assertEqual(
            metadata["controller_provenance_mode"],
            "documented_external_attestation",
        )
        self.assertTrue(metadata["same_controller_selectable_path_evidence"])
        self.assertTrue(metadata["closes_independent_longitudinal_multileo_limitation"])

    def test_campaign_ids_are_not_independence_groups_without_explicit_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame, metadata = self._build(
                self._write_trace(Path(temp_dir), include_campaign_id=True),
                bin_seconds=86400,
            )
        self.assertIn("campaign_id", frame)
        self.assertTrue(metadata["mapped_campaign_id_pass"])
        self.assertEqual(metadata["mapped_campaign_count"], 2)
        self.assertEqual(metadata["audited_campaign_count"], 0)
        self.assertFalse(metadata["independent_campaign_ids_asserted"])
        self.assertFalse(metadata["independent_campaign_grouping_pass"])
        self.assertEqual(
            metadata["campaign_independence_mode"],
            "mapped_ids_not_asserted_independent",
        )

    def test_documented_independent_campaign_ids_enable_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame, metadata = self._build(
                self._write_trace(Path(temp_dir), include_campaign_id=True),
                bin_seconds=86400,
                independent_campaign_ids_audited=True,
                campaign_independence_note=(
                    "data owner confirms separate collection drives and resets"
                ),
            )
        self.assertTrue(metadata["independent_campaign_grouping_pass"])
        self.assertEqual(metadata["audited_campaign_count"], 2)
        self.assertEqual(frame["segment_id"].nunique(), 2)
        group_column, audit = _resolve_campaign_gate_grouping(metadata, frame)
        self.assertEqual(group_column, "campaign_id")
        self.assertTrue(audit["forecast_campaign_ids_audited"])

    def test_campaign_independence_assertion_requires_documentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir), include_campaign_id=True)
            with self.assertRaisesRegex(ValueError, "campaign_independence_note"):
                self._build(path, independent_campaign_ids_audited=True)

    def test_one_campaign_id_still_fails_closed_to_one_inference_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame, metadata = self._build(
                self._write_trace(
                    Path(temp_dir),
                    include_campaign_id=True,
                    single_campaign_id=True,
                ),
                bin_seconds=86400,
                independent_campaign_ids_audited=True,
                campaign_independence_note="one collected campaign",
            )
        self.assertFalse(metadata["independent_campaign_grouping_pass"])
        group_column, audit = _resolve_campaign_gate_grouping(metadata, frame)
        self.assertIsNone(group_column)
        self.assertIn("fail closed", audit["grouping_reason"])

    def test_conflicting_campaign_ids_fail_paired_campaign_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame, metadata = self._build(
                self._write_trace(
                    Path(temp_dir),
                    include_campaign_id=True,
                    conflicting_campaign_ids=True,
                ),
                bin_seconds=86400,
                independent_campaign_ids_audited=True,
                campaign_independence_note="claimed independent campaigns",
            )
        self.assertGreater(metadata["conflicting_campaign_pair_count"], 0)
        self.assertFalse(metadata["mapped_campaign_id_pass"])
        self.assertFalse(metadata["independent_campaign_grouping_pass"])
        group_column, _ = _resolve_campaign_gate_grouping(metadata, frame)
        self.assertIsNone(group_column)

    def test_legacy_campaign_metadata_never_creates_independent_blocks(self) -> None:
        forecast = pd.DataFrame(
            {"campaign_id": ["a", "a", "b", "b"]}
        )
        group_column, audit = _resolve_campaign_gate_grouping(
            {
                "audited_campaign_ids": ["a", "b"],
                "concurrent_interchangeable_paths": True,
            },
            forecast,
        )
        self.assertIsNone(group_column)
        self.assertFalse(audit["declared_campaign_independence_pass"])

    def test_global_four_way_split_audit_closes_future_targets(self) -> None:
        rows = []
        for epoch in range(20):
            for path in ("starlink", "oneweb"):
                rows.append(
                    {
                        "relative_path": path,
                        "bin_epoch": float(epoch),
                        "target_next_bin_epoch": float(epoch + 1),
                        "target_available_3": 1,
                        "target_end_bin_epoch_3": float(epoch + 3),
                        "target_cumulative_3": 30.0,
                        "target_mean_3": 10.0,
                        "campaign_id": "campaign-a" if epoch < 10 else "campaign-b",
                    }
                )
        source = pd.DataFrame(rows)
        ratios = {
            "train": 0.4,
            "calibration": 0.2,
            "selection": 0.2,
            "test": 0.2,
        }
        parts = split_train_calibration_selection_test(
            source,
            train_ratio=ratios["train"],
            calibration_ratio=ratios["calibration"],
            selection_ratio=ratios["selection"],
            test_ratio=ratios["test"],
        )
        audit = _four_way_split_audit(
            dict(zip(("train", "calibration", "selection", "test"), parts)),
            source,
            ratios,
        )
        self.assertTrue(audit["pairwise_epoch_disjoint"])
        self.assertTrue(audit["strict_chronological_order"])
        self.assertTrue(audit["one_step_target_boundary_closed"])
        self.assertTrue(audit["multi_bin_target_boundaries_closed"])
        corrupted = dict(
            zip(("train", "calibration", "selection", "test"), parts)
        )
        corrupted["train"] = corrupted["train"].copy()
        corrupted["train"].loc[
            corrupted["train"].index[0],
            "target_next_bin_epoch",
        ] = 100.0
        corrupted_audit = _four_way_split_audit(corrupted, source, ratios)
        self.assertFalse(corrupted_audit["one_step_target_boundary_closed"])
        corrupted_multi = dict(
            zip(("train", "calibration", "selection", "test"), parts)
        )
        corrupted_multi["train"] = corrupted_multi["train"].copy()
        first_train = corrupted_multi["train"].index[0]
        corrupted_multi["train"].loc[first_train, "target_available_3"] = 1
        corrupted_multi["train"].loc[
            first_train,
            "target_end_bin_epoch_3",
        ] = 100.0
        corrupted_multi["train"].loc[first_train, "target_cumulative_3"] = 30.0
        corrupted_multi["train"].loc[first_train, "target_mean_3"] = 10.0
        corrupted_multi_audit = _four_way_split_audit(
            corrupted_multi,
            source,
            ratios,
        )
        self.assertFalse(
            corrupted_multi_audit["multi_bin_target_boundaries_closed"]
        )

    def test_validation_runner_rejects_scoped_trace_without_explicit_opt_in(self) -> None:
        metadata = {
            "operators": ["starlink", "oneweb"],
            "all_candidate_outcomes_observed": True,
            "same_controller_selectable_path_evidence": False,
        }
        with self.assertRaisesRegex(ValueError, "allow-scoped-paired-replay"):
            _require_validation_scope(
                metadata,
                allow_scoped_paired_replay=False,
            )
        self.assertFalse(
            _require_validation_scope(
                metadata,
                allow_scoped_paired_replay=True,
            )
        )

    def test_legacy_metadata_cannot_imply_same_controller_evidence(self) -> None:
        metadata = {
            "operators": ["starlink", "oneweb"],
            "all_candidate_outcomes_observed": True,
            "concurrent_interchangeable_paths": True,
        }
        with self.assertRaisesRegex(ValueError, "one controller"):
            _require_validation_scope(
                metadata,
                allow_scoped_paired_replay=False,
            )

    def test_claim_floors_cannot_be_weakened_by_command_line_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(Path(temp_dir), days=2),
                minimum_duration_days=0.001,
                minimum_concurrent_hours=0.001,
                maximum_p95_skew_ms=10000.0,
            )
        self.assertEqual(metadata["effective_minimum_duration_days"], 30.0)
        self.assertEqual(metadata["effective_minimum_concurrent_hours"], 24.0)
        self.assertEqual(metadata["effective_maximum_p95_skew_ms"], 100.0)
        self.assertFalse(metadata["closes_independent_longitudinal_multileo_limitation"])

    def test_single_operator_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir), include_oneweb=False)
            with self.assertRaisesRegex(ValueError, "Starlink and OneWeb"):
                self._build(path)

    def test_lost_packets_are_preserved_as_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_trace(Path(temp_dir))
            source = pd.read_csv(path)
            oneweb = source["network"].eq("Eutelsat OneWeb")
            source.loc[oneweb, "received"] = False
            source.loc[oneweb, "owd"] = None
            source.to_csv(path, index=False)
            frame, metadata = self._build(path)
        oneweb_rows = frame[frame["operator"].eq("oneweb")]
        self.assertTrue(oneweb_rows["observed_replies"].eq(0).all())
        self.assertTrue(oneweb_rows["latency_mean_ms"].eq(1000.0).all())
        self.assertTrue(metadata["all_candidate_outcomes_observed"])

    def test_sparse_calendar_span_fails_default_observation_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, metadata = self._build(
                self._write_trace(Path(temp_dir)),
                minimum_concurrent_hours=24.0,
            )
        self.assertTrue(metadata["duration_span_pass"])
        self.assertFalse(metadata["observation_volume_pass"])
        self.assertFalse(metadata["long_duration_pass"])
        self.assertFalse(metadata["closes_independent_longitudinal_multileo_limitation"])

    def test_timestamp_gaps_create_distinct_forecast_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.csv"
            rows = []
            for second in (0, 10, 100, 110):
                for operator in ("Starlink", "OneWeb"):
                    rows.append(
                        {
                            "ts": (
                                pd.Timestamp("2026-01-01T00:00:00Z")
                                + pd.Timedelta(seconds=second)
                            ).isoformat(),
                            "network": operator,
                            "owd": 40.0,
                            "received": True,
                            "scenario": "urban",
                        }
                    )
            pd.DataFrame(rows).to_csv(path, index=False)
            frame, metadata = self._build(
                path,
                minimum_duration_days=0.001,
                minimum_concurrent_hours=0.001,
            )
        self.assertEqual(frame["segment_id"].nunique(), 2)
        self.assertEqual(frame["relative_path"].nunique(), 4)
        self.assertEqual(metadata["continuous_segment_count"], 2)


if __name__ == "__main__":
    unittest.main()
