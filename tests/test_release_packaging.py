"""Regression tests for fail-closed Transactions release packaging."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import unittest
from unittest import mock

import scripts.build_transactions_evidence as evidence
import scripts.package_transactions_release as release
import scripts.rebuild_transactions_artifact as rebuild


class ReleasePackagingTests(unittest.TestCase):
    @staticmethod
    def _portable_provenance() -> dict[str, object]:
        return {
            "schema_version": 3,
            "repository_root": ".",
            "published_evidence_directory": "results/transactions_evidence",
            "staged_evidence_directory": (
                "results/.transactions_evidence.staging-example"
            ),
            "python_executable": release.PYTHON_COMMAND_TOKEN,
            "environment": {
                "MPLCONFIGDIR": ".mpl-cache",
                "XDG_CACHE_HOME": release.EXTERNAL_PATH_TOKEN,
            },
            "environment_manifest": {
                "schema_version": 2,
                "packages": [],
            },
            "code_manifest": {
                "files": [
                    {
                        "path": "scripts/example.py",
                        "sha256": "0" * 64,
                        "bytes": 1,
                    }
                ]
            },
            "input_manifest": [
                {
                    "path": "data/external/example.csv",
                    "sha256": "1" * 64,
                    "bytes": 1,
                }
            ],
            "secondary_output_manifest": [
                {
                    "path": "results/example.csv",
                    "sha256": "2" * 64,
                    "bytes": 1,
                }
            ],
            "previous_evidence_archive": (
                "output/build_backups/transactions_evidence_previous"
            ),
            "commands": [
                {
                    "argv": [
                        release.PYTHON_COMMAND_TOKEN,
                        "scripts/example.py",
                        "--output",
                        "results/example",
                    ],
                    "cwd": ".",
                    "returncode": 0,
                }
            ],
            "manifest_refresh_command": [
                release.PYTHON_COMMAND_TOKEN,
                "scripts/build_transactions_evidence.py",
                "--manifest-only",
            ],
            "manifest_verify_command": [
                release.PYTHON_COMMAND_TOKEN,
                "scripts/build_transactions_evidence.py",
                "--verify-manifest",
            ],
            "path_recording_policy": (
                "repository-relative paths; external paths are redacted"
            ),
        }

    def test_release_allowlist_excludes_stale_docs_and_duplicate_gate_tree(
        self,
    ) -> None:
        self.assertEqual(
            release.CURATED_DOC_FILES,
            (
                "docs/THIRD_PARTY_NOTICES.md",
                "docs/closed_loop_field_validation_protocol.md",
                "docs/commercial_multileo_acquisition_status.json",
                "docs/commercial_multileo_validation_protocol.md",
                "docs/model_feature_contract.md",
                "docs/open_leo_latency_routing_scope.md",
                "docs/release_readiness_checklist.md",
                "docs/reproducibility_guide.md",
            ),
        )
        self.assertNotIn("docs", release.TREE_ROOTS)
        self.assertNotIn(
            "results/gate_operating_characteristics",
            release.TREE_ROOTS,
        )
        self.assertNotIn("results/reviewer_readiness", release.TREE_ROOTS)
        superseded = {
            "docs/decision_opportunity_revision.md",
            "docs/final_reviewer_implementation_status.md",
            "docs/reviewer_response_map.md",
            "docs/transactions_final_review_audit.md",
            "docs/transactions_risk_revision.md",
        }
        self.assertTrue(superseded.isdisjoint(release.CURATED_DOC_FILES))

    def test_evidence_preflight_rejects_exact_tree_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results/transactions_evidence"
            output.mkdir(parents=True)
            (output / "table.csv").write_bytes(b"a,b\n1,2\n")
            evidence._write_evidence_manifest(output)
            with mock.patch.object(release, "REPO_ROOT", root):
                verified = release._verify_evidence_tree()
                self.assertEqual(verified["verified_file_count"], 1)
                (output / "stale.csv").write_text("stale\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "extra=.*stale.csv"):
                    release._verify_evidence_tree()

    def test_code_preflight_rejects_modified_executable_specification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            source = scripts / "example.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            with mock.patch.object(rebuild, "REPO_ROOT", root):
                expected = rebuild._code_manifest()
                provenance = {"code_manifest": expected}
                release._verify_current_code_manifest(provenance)
                source.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "differs from build provenance",
                ):
                    release._verify_current_code_manifest(provenance)

    def test_portable_provenance_accepts_schema_three_contract(self) -> None:
        provenance = self._portable_provenance()
        release._verify_portable_provenance(provenance)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results/transactions_evidence"
            output.mkdir(parents=True)
            path = output / "build_provenance.json"
            path.write_text(json.dumps(provenance), encoding="utf-8")
            with mock.patch.object(release, "REPO_ROOT", root):
                self.assertEqual(
                    release._load_build_provenance()["schema_version"],
                    3,
                )

            provenance["schema_version"] = 2
            path.write_text(json.dumps(provenance), encoding="utf-8")
            with mock.patch.object(release, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "schema 3"):
                    release._load_build_provenance()

    def test_portable_provenance_rejects_absolute_home_path_leakage(
        self,
    ) -> None:
        cases = [
            (
                "environment",
                lambda value: value["environment"].update(
                    {"MPLCONFIGDIR": "/Users/alice/project/.mpl-cache"}
                ),
            ),
            (
                "arbitrary_text",
                lambda value: value.update(
                    {"private_note": "built under /home/alice/private/repo"}
                ),
            ),
            (
                "windows_command",
                lambda value: value["commands"][0]["argv"].append(
                    "C:\\Users\\alice\\private.csv"
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                provenance = deepcopy(self._portable_provenance())
                mutate(provenance)
                with self.assertRaisesRegex(
                    ValueError,
                    "absolute|portable path",
                ):
                    release._verify_portable_provenance(provenance)

    def test_portable_provenance_rejects_malformed_commands(self) -> None:
        cases = [
            (
                "absolute_interpreter",
                lambda value: value["commands"][0]["argv"].__setitem__(
                    0,
                    "/Users/alice/bin/python",
                ),
                "must be.*python",
            ),
            (
                "nonportable_cwd",
                lambda value: value["commands"][0].update(
                    {"cwd": "/Users/alice/project"}
                ),
                "cwd must be",
            ),
            (
                "empty_argv",
                lambda value: value["commands"][0].update({"argv": []}),
                "non-empty argv",
            ),
            (
                "parent_traversal",
                lambda value: value["commands"][0]["argv"].append(
                    "../private/input.csv"
                ),
                "parent traversal",
            ),
            (
                "refresh_without_token",
                lambda value: value.update(
                    {"manifest_refresh_command": ["python", "script.py"]}
                ),
                "manifest_refresh_command.*must be",
            ),
        ]
        for label, mutate, message in cases:
            with self.subTest(label=label):
                provenance = deepcopy(self._portable_provenance())
                mutate(provenance)
                with self.assertRaisesRegex(ValueError, message):
                    release._verify_portable_provenance(provenance)

    def test_dependency_preflight_binds_lock_installed_and_build_versions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements.txt").write_text(
                "NumPy>=1.0\npandas>=2.0\n",
                encoding="utf-8",
            )
            (root / "requirements-lock.txt").write_text(
                "numpy==1.2.3\npandas==2.4.5\n",
                encoding="utf-8",
            )
            installed = {"numpy": "1.2.3", "pandas": "2.4.5"}
            with (
                mock.patch.object(release, "REPO_ROOT", root),
                mock.patch.object(
                    release.importlib_metadata,
                    "version",
                    side_effect=lambda name: installed[name],
                ),
            ):
                pins = release._verify_installed_dependency_pins()
                self.assertEqual(
                    pins,
                    {"numpy": "1.2.3", "pandas": "2.4.5"},
                )
                release._verify_build_dependency_versions(
                    {
                        "environment_manifest": {
                            "packages": [
                                {"name": "numpy", "version": "1.2.3"},
                                {"name": "pandas", "version": "2.4.5"},
                            ]
                        }
                    },
                    pins,
                )

                installed["numpy"] = "9.9.9"
                with self.assertRaisesRegex(RuntimeError, "version_drift"):
                    release._verify_installed_dependency_pins()
                installed["numpy"] = "1.2.3"

                with self.assertRaisesRegex(RuntimeError, "build=9.9.9"):
                    release._verify_build_dependency_versions(
                        {
                            "environment_manifest": {
                                "packages": [
                                    {"name": "numpy", "version": "9.9.9"},
                                    {"name": "pandas", "version": "2.4.5"},
                                ]
                            }
                        },
                        pins,
                    )

            (root / "requirements-lock.txt").write_text(
                "numpy>=1.2.3\npandas==2.4.5\n",
                encoding="utf-8",
            )
            with mock.patch.object(release, "REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "exact, unconditional"):
                    release._verify_installed_dependency_pins()

    @staticmethod
    def _write_readiness(
        root: Path,
        rows: list[dict[str, str]],
        *,
        acknowledged: bool = True,
    ) -> None:
        output = root / "results/reviewer_readiness"
        output.mkdir(parents=True)
        fields = [
            "category",
            "check",
            "status",
            "evidence",
            "likely_future_reviewer_comment",
            "required_action",
        ]
        with (output / "reviewer_readiness.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        by_category: dict[str, dict[str, object]] = {}
        for category in sorted({row["category"] for row in rows}):
            selected = [row for row in rows if row["category"] == category]
            pending_count = sum(row["status"] == "PENDING" for row in selected)
            by_category[category] = {
                "pass_count": sum(row["status"] == "PASS" for row in selected),
                "pending_count": pending_count,
                "complete": pending_count == 0,
            }
        pending_count = sum(row["status"] == "PENDING" for row in rows)
        summary = {
            "pass_count": sum(row["status"] == "PASS" for row in rows),
            "pending_count": pending_count,
            "all_review_items_complete": pending_count == 0,
            "artifact_checks_complete": by_category[
                "artifact_integrity"
            ]["complete"],
            "external_or_deployment_evidence_complete": all(
                by_category[category]["complete"]
                for category in (
                    "evaluation_evidence",
                    "deployment_evidence",
                )
            ),
            "allow_pending_acknowledged": acknowledged,
            "by_category": by_category,
        }
        (output / "reviewer_readiness.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        (output / "reviewer_readiness.md").write_text(
            "# Reviewer Readiness Audit\n",
            encoding="utf-8",
        )

    @staticmethod
    def _readiness_rows() -> list[dict[str, str]]:
        common = {
            "evidence": "{}",
            "likely_future_reviewer_comment": "review comment",
            "required_action": "",
        }
        return [
            {
                **common,
                "category": "artifact_integrity",
                "check": "artifact check",
                "status": "PASS",
            },
            {
                **common,
                "category": "evaluation_evidence",
                "check": "evaluation check",
                "status": "PASS",
            },
            {
                **common,
                "category": "deployment_evidence",
                "check": "closed-loop deployment",
                "status": "PENDING",
                "required_action": "run the prospective field trial",
            },
        ]

    def test_readiness_allows_only_documented_deployment_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._readiness_rows()
            self._write_readiness(root, rows)
            with mock.patch.object(release, "REPO_ROOT", root):
                summary = release._verify_reviewer_readiness()
                self.assertEqual(summary["pending_count"], 1)

        variants = {
            "fail": ("status", "FAIL", "FAIL or unknown"),
            "not_deployment": (
                "category",
                "evaluation_evidence",
                "only documented deployment",
            ),
            "undocumented": (
                "required_action",
                "",
                "lacks documentation",
            ),
        }
        for label, (field, value, message) in variants.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    rows = self._readiness_rows()
                    rows[-1][field] = value
                    if label == "not_deployment":
                        rows.append(
                            {
                                **rows[0],
                                "category": "deployment_evidence",
                                "check": "deployment pass",
                            }
                        )
                    self._write_readiness(root, rows)
                    with mock.patch.object(release, "REPO_ROOT", root):
                        with self.assertRaisesRegex(RuntimeError, message):
                            release._verify_reviewer_readiness()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_readiness(
                root,
                self._readiness_rows(),
                acknowledged=False,
            )
            with mock.patch.object(release, "REPO_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "not explicitly"):
                    release._verify_reviewer_readiness()

    def test_archive_is_deterministic_and_has_one_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("release\n", encoding="utf-8")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            with (
                mock.patch.object(release, "REPO_ROOT", root),
                mock.patch.object(release, "ROOT_FILES", ("README.md",)),
                mock.patch.object(release, "CURATED_DOC_FILES", ()),
                mock.patch.object(release, "REVIEWER_READINESS_FILES", ()),
                mock.patch.object(release, "TREE_ROOTS", ()),
                mock.patch.object(release, "EXTRA_FILES", ()),
                mock.patch.object(release, "_preflight_release") as preflight,
            ):
                release._write_archive(first)
                release._write_archive(second)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertTrue(
                    release._verify_archive(first)[
                        "single_top_level_directory"
                    ]
                )
                with tarfile.open(first, mode="r:gz") as archive:
                    roots = {
                        PurePosixPath(member.name).parts[0]
                        for member in archive.getmembers()
                    }
                self.assertEqual(roots, {release.ARCHIVE_ROOT})
                self.assertEqual(preflight.call_count, 2)

    def test_preflight_failure_does_not_replace_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release.tar.gz"
            output.write_bytes(b"prior archive")
            with mock.patch.object(
                release,
                "_preflight_release",
                side_effect=RuntimeError("injected preflight failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    release._write_archive(output)
            self.assertEqual(output.read_bytes(), b"prior archive")


if __name__ == "__main__":
    unittest.main()
