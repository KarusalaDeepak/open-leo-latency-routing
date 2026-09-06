"""Regression tests for evidence provenance and failure-safe publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import pandas as pd

from scripts.build_commect_multiaccess_trace import (
    _verify_extracted_sources_against_archive,
)
import scripts.build_transactions_evidence as evidence
from scripts.audit_reviewer_readiness import (
    _supports_hypatia_zero_shot_replication,
)
from scripts.run_commect_rolling_origin_validation import _aggregate_decisions
import scripts.rebuild_transactions_artifact as rebuild


_verify_evidence_manifest = evidence._verify_evidence_manifest
_write_evidence_manifest = evidence._write_evidence_manifest


class ArtifactIntegrityTests(unittest.TestCase):
    def test_concurrency_evidence_separates_replay_from_controller_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            victoria = root / "victoria.json"
            victoria.write_text(
                json.dumps(
                    {
                        "measured_concurrent_paths": True,
                        "concurrent_alternative_paths": True,
                        "concurrency_audit": {
                            "concurrent_epoch_count": 12,
                            "max_concurrent_paths": 2,
                            "concurrent_row_fraction": 1.0,
                            "has_temporally_concurrent_candidates": True,
                            "supports_candidate_outcome_shadow_replay": True,
                            "supports_literal_single_controller_steering": False,
                            "supports_closed_loop_deployment_evidence": False,
                            "controller_topology_scope": (
                                "co-located terminals; steering unverified"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit = evidence._dataset_audit(
                [
                    {
                        "dataset": "LENS Victoria holdout",
                        "source_type": "measured LEO terminals",
                        "metadata": victoria,
                    }
                ]
            )

        row = audit.iloc[0]
        self.assertTrue(row["has_temporally_concurrent_candidates"])
        self.assertTrue(row["supports_candidate_outcome_shadow_replay"])
        self.assertFalse(row["supports_literal_single_controller_steering"])
        self.assertFalse(row["supports_closed_loop_deployment_evidence"])
        self.assertNotIn("supports_online_path_selection", audit.columns)

    def test_missing_outcome_envelopes_use_scheduled_test_denominators(
        self,
    ) -> None:
        fixed_summary = pd.DataFrame(
            {
                "policy_name": ["reactive_greedy"],
                "decision_count": [88],
                "success_rate_under_60ms": [57 / 88],
            }
        )
        rolling_summary = pd.DataFrame(
            {
                "policy_name": ["reactive_greedy"],
                "decision_count": [289],
                "success_rate_under_60ms": [196 / 289],
            }
        )
        envelopes = evidence._missing_outcome_envelopes(
            fixed_summary,
            {
                "partitions": {
                    "test": {"scheduled_decision_epoch_count": 95}
                }
            },
            pd.DataFrame({"scheduled_test_decision_epochs": [63] * 5}),
            rolling_summary,
        ).set_index("protocol")

        fixed = envelopes.loc["fixed"]
        self.assertEqual(fixed["unevaluable_scheduled_test_decision_epochs"], 7)
        self.assertAlmostEqual(fixed["worst_case_success_lower_bound"], 57 / 95)
        self.assertAlmostEqual(fixed["best_case_success_upper_bound"], 64 / 95)
        rolling = envelopes.loc["rolling"]
        self.assertEqual(
            rolling["unevaluable_scheduled_test_decision_epochs"],
            26,
        )
        self.assertAlmostEqual(
            rolling["worst_case_success_lower_bound"],
            196 / 315,
        )
        self.assertAlmostEqual(
            rolling["best_case_success_upper_bound"],
            222 / 315,
        )

    def test_switch_audits_use_only_eligible_contiguous_transitions(
        self,
    ) -> None:
        decisions = pd.DataFrame(
            {
                "dataset": ["test"] * 3,
                "evaluation_case": ["continuity_gap"] * 3,
                "policy_name": ["policy"] * 3,
                "success_under_budget": [1, 1, 0],
                "realized_next_latency_ms": [10.0, 20.0, 80.0],
                "decision_gap_ms": [0.0, 5.0, 10.0],
                "retrospective_best_path_match": [1, 1, 0],
                "switched_path": [0, 0, 1],
                "switch_transition_eligible": [0, 0, 1],
                "continuity_reset": [0, 1, 0],
                "continuity_segment_start": [1, 1, 0],
                "model_and_ranking_time_us": [1.0, 1.0, 1.0],
            }
        )
        summary = _aggregate_decisions(decisions)
        summary["dataset"] = "test"
        summary["evaluation_case"] = "continuity_gap"

        # The middle decision begins a new continuity segment.  The only
        # eligible transition is the final one, which switches, so the rate is
        # 1/1 rather than 1/3.
        self.assertEqual(float(summary["switch_rate"].iloc[0]), 1.0)
        self.assertEqual(int(summary["switch_count"].iloc[0]), 1)
        self.assertEqual(
            int(summary["eligible_switch_transition_count"].iloc[0]),
            1,
        )
        self.assertEqual(int(summary["switch_transition_count"].iloc[0]), 1)
        self.assertEqual(int(summary["continuity_reset_count"].iloc[0]), 1)
        self.assertEqual(int(summary["continuity_segment_count"].iloc[0]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evidence, "REPO_ROOT", Path(temporary)):
                audit = evidence._numerical_consistency_audit(
                    summary,
                    decisions,
                    pd.DataFrame(),
                    pd.DataFrame(),
                )
        switch_checks = audit[
            audit["check"].eq("policy_summary_metric_recomputed")
            & audit["observed"].astype(str).str.startswith(
                (
                    "switch_rate=",
                    "switch_count=",
                    "eligible_switch_transition_count=",
                    "switch_transition_count=",
                )
            )
        ]
        self.assertEqual(len(switch_checks), 4)
        self.assertTrue(switch_checks["passed"].all())
        closure = audit[
            audit["check"].eq("switch_transition_t_minus_s_closure")
        ]
        self.assertEqual(len(closure), 1)
        self.assertTrue(closure["passed"].all())

        wrong_count = summary.copy()
        wrong_count["switch_transition_count"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evidence, "REPO_ROOT", Path(temporary)):
                with self.assertRaisesRegex(
                    AssertionError,
                    "switch_transition_count=1",
                ):
                    evidence._numerical_consistency_audit(
                        wrong_count,
                        decisions,
                        pd.DataFrame(),
                        pd.DataFrame(),
                    )

        invalid = decisions.copy()
        invalid.loc[1, "switched_path"] = 1
        with self.assertRaisesRegex(ValueError, "ineligible"):
            _aggregate_decisions(invalid)

        invalid_closure = decisions.copy()
        invalid_closure["continuity_segment_start"] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, "T-S continuity closure"):
            _aggregate_decisions(invalid_closure)

        invalid_rowwise_closure = decisions.copy()
        invalid_rowwise_closure["continuity_segment_start"] = [1, 0, 1]
        with self.assertRaisesRegex(ValueError, "row-wise complement"):
            _aggregate_decisions(invalid_rowwise_closure)

    def test_numerical_audit_fails_closed_on_summary_key_mismatch(self) -> None:
        decisions = pd.DataFrame(
            {
                "dataset": ["test", "test"],
                "evaluation_case": ["case", "case"],
                "policy_name": ["policy", "policy"],
                "success_under_budget": [1, 0],
                "realized_next_latency_ms": [10.0, 80.0],
                "decision_gap_ms": [0.0, 5.0],
                "retrospective_best_path_match": [1, 0],
                "switched_path": [0, 1],
                "switch_transition_eligible": [0, 1],
                "continuity_reset": [0, 0],
                "continuity_segment_start": [1, 0],
                "model_and_ranking_time_us": [1.0, 1.0],
            }
        )
        summary = _aggregate_decisions(decisions)
        summary["dataset"] = "test"
        summary["evaluation_case"] = "case"

        cases = {
            "missing": summary.iloc[0:0].copy(),
            "duplicate": pd.concat([summary, summary], ignore_index=True),
            "extra": pd.concat(
                [
                    summary,
                    summary.assign(policy_name="summary_only_policy"),
                ],
                ignore_index=True,
            ),
        }
        for label, malformed_summary in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    with mock.patch.object(
                        evidence,
                        "REPO_ROOT",
                        Path(temporary),
                    ):
                        with self.assertRaisesRegex(
                            AssertionError,
                            "policy_summary_decision_key_cardinality",
                        ):
                            evidence._numerical_consistency_audit(
                                malformed_summary,
                                decisions,
                                pd.DataFrame(),
                                pd.DataFrame(),
                            )

        legacy_summary = summary.drop(columns=["switch_count"])
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evidence, "REPO_ROOT", Path(temporary)):
                with self.assertRaisesRegex(
                    AssertionError,
                    "policy_summary_switch_audit_schema",
                ):
                    evidence._numerical_consistency_audit(
                        legacy_summary,
                        decisions,
                        pd.DataFrame(),
                        pd.DataFrame(),
                    )

    def test_hypatia_readiness_is_bound_to_the_canonical_zero_shot_target(
        self,
    ) -> None:
        trace_metadata = {
            "dataset_name": "hypatia_service_replica_paths",
            "is_hypatia_output": True,
            "uses_tle_orbital_propagation": True,
            "uses_dynamic_shortest_path_state": True,
        }
        zero_shot_metadata = {
            "zero_shot_transfer": True,
            "target_trace": "data/processed/hypatia_service_paths_10s.csv",
            "target_family": "hypatia_service_replica_paths",
            "target_rows_used_for_training": 0,
            "target_rows_used_for_calibration": 0,
            "target_concurrency_audit": {
                "has_temporally_concurrent_candidates": True,
                "decision_alignment": "actual_timestamp",
                "supports_shadow_policy_replay": True,
            },
        }
        self.assertTrue(
            _supports_hypatia_zero_shot_replication(
                trace_metadata,
                zero_shot_metadata,
            )
        )

        for field, invalid_value in (
            ("target_trace", "data/processed/different_target.csv"),
            ("target_family", "different_simulator"),
            ("target_rows_used_for_training", 1),
            ("target_rows_used_for_calibration", 1),
        ):
            with self.subTest(field=field):
                invalid = {**zero_shot_metadata, field: invalid_value}
                self.assertFalse(
                    _supports_hypatia_zero_shot_replication(
                        trace_metadata,
                        invalid,
                    )
                )

        invalid_concurrency = {
            **zero_shot_metadata,
            "target_concurrency_audit": {
                **zero_shot_metadata["target_concurrency_audit"],
                "decision_alignment": "normalized_stage_counterfactual",
            },
        }
        self.assertFalse(
            _supports_hypatia_zero_shot_replication(
                trace_metadata,
                invalid_concurrency,
            )
        )

    def test_commect_extraction_must_match_verified_zip_member_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "source.zip"
            source_dir = root / "extracted"
            source_dir.mkdir()
            filename = "Operator_A_RTT.csv"
            expected_bytes = b"Time,Latency (ms)\n01-Jan-2025,12\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    f"renamed/archive/root/{filename}",
                    expected_bytes,
                )
            extracted_path = source_dir / filename
            extracted_path.write_bytes(expected_bytes)

            verification = _verify_extracted_sources_against_archive(
                archive_path,
                source_dir,
                {"operator_a_5g": filename},
            )
            self.assertTrue(
                verification["operator_a_5g"]["byte_identity_verified"]
            )
            self.assertEqual(
                verification["operator_a_5g"]["matching_archive_members"],
                [f"renamed/archive/root/{filename}"],
            )

            extracted_path.write_bytes(b"different bytes with same filename")
            with self.assertRaisesRegex(ValueError, "does not match"):
                _verify_extracted_sources_against_archive(
                    archive_path,
                    source_dir,
                    {"operator_a_5g": filename},
                )

    def test_reuse_fingerprint_includes_dependency_environment(self) -> None:
        code = {"aggregate_sha256": "code"}
        inputs = [{"path": "raw.csv", "sha256": "input", "bytes": 1}]
        environment_a = {"aggregate_sha256": "environment-a"}
        environment_b = {"aggregate_sha256": "environment-b"}
        self.assertNotEqual(
            rebuild._reuse_fingerprint(code, inputs, environment_a),
            rebuild._reuse_fingerprint(code, inputs, environment_b),
        )
        observed = rebuild._environment_manifest()
        self.assertEqual(observed["schema_version"], 2)
        self.assertEqual(len(observed["aggregate_sha256"]), 64)
        self.assertTrue(observed["packages"])
        self.assertNotIn("python_executable", observed)
        self.assertNotIn(str(Path.home()), json.dumps(observed))

    def test_runtime_environment_records_exact_direct_and_test_pins(self) -> None:
        expected_runtime = {
            "numpy": "1.24.3",
            "pandas": "2.0.3",
            "PyYAML": "6.0",
            "scikit-learn": "1.3.0",
            "scipy": "1.11.1",
            "matplotlib": "3.7.2",
            "Pillow": "9.4.0",
            "xgboost": "3.1.2",
        }
        expected_test = {"pytest": "8.2.2"}
        runtime = evidence._runtime_environment()
        self.assertEqual(runtime["schema_version"], 2)
        self.assertEqual(
            runtime["package_groups"],
            {"runtime": expected_runtime, "test": expected_test},
        )
        declarations = runtime["dependency_declarations"]
        self.assertFalse(declarations["optional_hypatia_in_canonical_build"])
        self.assertIn(
            "unverified",
            declarations["optional_hypatia_dependency_status"],
        )

        def exact_pins(path: Path) -> dict[str, str]:
            pins: dict[str, str] = {}
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                self.assertEqual(line.count("=="), 1, line)
                name, package_version = line.split("==", maxsplit=1)
                pins[name] = package_version
            return pins

        expected_all = {**expected_runtime, **expected_test}
        self.assertEqual(exact_pins(rebuild.REPO_ROOT / "requirements.txt"), expected_all)
        self.assertEqual(
            exact_pins(rebuild.REPO_ROOT / "requirements-lock.txt"),
            expected_all,
        )

    def test_recorded_provenance_commands_and_paths_are_portable(self) -> None:
        command_log: list[dict[str, object]] = []
        rebuild._run(
            ["scripts/example.py", "--output", "results/example"],
            dry_run=True,
            command_log=command_log,
        )
        self.assertEqual(
            command_log,
            [
                {
                    "argv": [
                        rebuild.PYTHON_COMMAND_TOKEN,
                        "scripts/example.py",
                        "--output",
                        "results/example",
                    ],
                    "cwd": ".",
                }
            ],
        )
        self.assertEqual(
            rebuild._portable_path_record(rebuild.REPO_ROOT / ".cache"),
            ".cache",
        )
        self.assertEqual(
            rebuild._portable_command([str(rebuild.REPO_ROOT / "script.py")]),
            [rebuild.PYTHON_COMMAND_TOKEN, "script.py"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                rebuild._portable_path_record(temporary),
                rebuild.EXTERNAL_PATH_TOKEN,
            )
        self.assertEqual(rebuild.PROVENANCE_SCHEMA_VERSION, 3)

    def test_temporal_resolution_fingerprint_resolves_manifest_raw_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "raw"
            data_root.mkdir()
            for name in ("a.csv", "b.csv", "unused.csv"):
                (data_root / name).write_text(name, encoding="utf-8")
            manifest = root / "candidate_manifest.csv"
            manifest.write_text(
                "relative_path\na.csv\nb.csv\nunused.csv\n",
                encoding="utf-8",
            )
            resolved = rebuild._temporal_resolution_external_inputs(
                manifest_path=manifest,
                data_root=data_root,
                max_files=2,
            )
            self.assertEqual(
                resolved,
                [manifest.resolve(), data_root / "a.csv", data_root / "b.csv"],
            )

    def test_temporal_resolution_inputs_preserve_logical_symlink_paths(
        self,
    ) -> None:
        from open_leo_latency_routing.data.aggregations import aggregate_ping_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            physical_root = root / "physical-raw"
            relative_path = Path(
                "inside-out/active/victoria/2025-03-01/"
                "ping-example-1.1.1.1-10ms-1h-2025-03-01-00-00-00.txt"
            )
            physical_path = physical_root / relative_path
            physical_path.parent.mkdir(parents=True)
            physical_path.write_text(
                "PING example (1.1.1.1) from 10.0.0.1 eth0:\n"
                "[1740787200.000] 64 bytes from 1.1.1.1: "
                "icmp_seq=1 ttl=64 time=20.0 ms\n",
                encoding="utf-8",
            )
            logical_root = root / "logical-raw"
            try:
                logical_root.symlink_to(physical_root, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - platform capability
                self.skipTest(f"directory symlinks unavailable: {exc}")
            manifest = root / "candidate_manifest.csv"
            manifest.write_text(
                f"relative_path\n{relative_path.as_posix()}\n",
                encoding="utf-8",
            )

            targets = rebuild.resolve_temporal_resolution_inputs(
                data_root=logical_root,
                manifest_path=manifest,
                max_files=1,
            )

            self.assertEqual(targets, [logical_root / relative_path])
            rows = aggregate_ping_file(
                path=targets[0],
                data_root=logical_root,
                bin_seconds=60,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], relative_path.as_posix())
            self.assertEqual(rows[0]["location"], "victoria")

    def test_quick_reuse_manifest_rejects_hash_and_stale_extra_drift(self) -> None:
        output_root = rebuild.REPO_ROOT / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="provenance-test-",
            dir=output_root,
        ) as temporary:
            root = Path(temporary) / "secondary"
            root.mkdir()
            reused = root / "summary.csv"
            reused.write_bytes(b"old")
            expected = rebuild._records_for_roots(
                (root,),
                label="test secondary output",
            )

            reused.write_bytes(b"new")
            (root / "stale-extra.csv").write_text("stale", encoding="utf-8")
            observed = rebuild._records_for_roots(
                (root,),
                label="test secondary output",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "extra=.*stale-extra.*hash_drift=.*summary",
            ):
                rebuild._verify_exact_file_manifest(
                    expected,
                    observed,
                    label="reused secondary output",
                )

    def test_evidence_manifest_rejects_missing_extra_size_and_hash_drift(
        self,
    ) -> None:
        def prepared_tree() -> tuple[tempfile.TemporaryDirectory[str], Path]:
            temporary = tempfile.TemporaryDirectory()
            root = Path(temporary.name)
            (root / "table.csv").write_bytes(b"abc")
            _write_evidence_manifest(root)
            return temporary, root

        temporary, root = prepared_tree()
        with temporary:
            verification = _verify_evidence_manifest(root)
            self.assertEqual(verification["verified_file_count"], 1)

        for drift, mutate, message in (
            (
                "missing",
                lambda root: (root / "table.csv").unlink(),
                "missing=.*table.csv",
            ),
            (
                "extra",
                lambda root: (root / "stale.txt").write_text(
                    "stale", encoding="utf-8"
                ),
                "extra=.*stale.txt",
            ),
            (
                "size",
                lambda root: (root / "table.csv").write_bytes(b"longer"),
                "size_drift=.*table.csv",
            ),
            (
                "hash",
                lambda root: (root / "table.csv").write_bytes(b"xyz"),
                "hash_drift=.*table.csv",
            ),
        ):
            with self.subTest(drift=drift):
                temporary, root = prepared_tree()
                with temporary:
                    mutate(root)
                    with self.assertRaisesRegex(ValueError, message):
                        _verify_evidence_manifest(root)

    def test_evidence_publish_preserves_prior_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            live = parent / "evidence"
            stage = parent / ".evidence-stage"
            backup = parent / "backups" / "evidence-old"
            live.mkdir()
            stage.mkdir()
            (live / "version.txt").write_text("old", encoding="utf-8")
            (stage / "version.txt").write_text("new", encoding="utf-8")

            rebuild._publish_staged_evidence(stage, live, backup)

            self.assertEqual(
                (live / "version.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertEqual(
                (backup / "version.txt").read_text(encoding="utf-8"),
                "old",
            )

    def test_evidence_publish_rolls_back_if_second_rename_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            live = parent / "evidence"
            stage = parent / ".evidence-stage"
            backup = parent / "backups" / "evidence-old"
            live.mkdir()
            stage.mkdir()
            (live / "version.txt").write_text("old", encoding="utf-8")
            (stage / "version.txt").write_text("new", encoding="utf-8")
            real_replace = os.replace

            def fail_stage_publish(source: object, destination: object) -> None:
                if Path(source) == stage and Path(destination) == live:
                    raise OSError("injected stage publication failure")
                real_replace(source, destination)

            with mock.patch.object(
                rebuild.os,
                "replace",
                side_effect=fail_stage_publish,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    rebuild._publish_staged_evidence(stage, live, backup)

            self.assertEqual(
                (live / "version.txt").read_text(encoding="utf-8"),
                "old",
            )
            self.assertEqual(
                (stage / "version.txt").read_text(encoding="utf-8"),
                "new",
            )
            self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
