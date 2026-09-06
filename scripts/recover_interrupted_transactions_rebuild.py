#!/usr/bin/env python3
"""Recover only the publication phase of an interrupted full rebuild.

This utility is intentionally conservative.  It does not claim to regenerate
the expensive primary and secondary experiment roots.  Instead, it hashes the
current executable specification, dependencies, canonical inputs, and every
secondary output; regenerates the complete evidence stage from those outputs;
runs reviewer-readiness checks and the full test suite; verifies that none of
the hashed premises changed; and only then publishes the stage atomically.

The resulting provenance is explicitly labelled as a recovered interrupted
full build.  A subsequent canonical ``rebuild_transactions_artifact.py
--quick`` can use its exact manifests to perform the ordinary verified-reuse
workflow and replace the recovery provenance with canonical quick provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rebuild_transactions_artifact import (
    PROVENANCE_SCHEMA_VERSION,
    PYTHON_COMMAND_TOKEN,
    SECONDARY_OUTPUT_ROOTS,
    _code_manifest,
    _display_path,
    _environment_manifest,
    _fresh_evidence_stage,
    _next_evidence_backup_path,
    _portable_command,
    _portable_path_record,
    _primary_input_manifest,
    _publish_staged_evidence,
    _records_for_roots,
    _reuse_fingerprint,
    _run,
    _run_manifest_utility,
    _verify_exact_file_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discard-stage",
        type=Path,
        help=(
            "Optional stale sibling stage to leave untouched but identify in "
            "the recovery note. A new stage is always generated."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    code_manifest = _code_manifest()
    environment_manifest = _environment_manifest()
    input_manifest = _primary_input_manifest()
    secondary_output_manifest = _records_for_roots(
        SECONDARY_OUTPUT_ROOTS,
        label="recovered secondary output",
    )
    reuse_fingerprint = _reuse_fingerprint(
        code_manifest,
        input_manifest,
        environment_manifest,
    )

    evidence_live = REPO_ROOT / "results/transactions_evidence"
    evidence_stage = _fresh_evidence_stage(dry_run=False)
    evidence_stage_arg = evidence_stage.relative_to(REPO_ROOT).as_posix()
    evidence_live_arg = evidence_live.relative_to(REPO_ROOT).as_posix()
    evidence_backup = _next_evidence_backup_path(evidence_live)
    command_log: list[dict[str, object]] = []

    final_commands = [
        [
            "scripts/build_transactions_evidence.py",
            "--output-dir",
            evidence_stage_arg,
        ],
        [
            "scripts/build_risk_control_figures.py",
            "--output-dir",
            f"{evidence_stage_arg}/figures",
        ],
        ["scripts/generate_result_figures.py", "--manuscript-assets-only"],
        [
            "scripts/audit_reviewer_readiness.py",
            "--allow-pending",
            "--evidence-dir",
            evidence_stage_arg,
        ],
        ["-m", "pytest", "-q"],
    ]
    for command in final_commands:
        _run(command, dry_run=False, command_log=command_log)

    final_code_manifest = _code_manifest()
    if final_code_manifest != code_manifest:
        raise RuntimeError("code/configuration changed during recovery")
    final_environment_manifest = _environment_manifest()
    if final_environment_manifest != environment_manifest:
        raise RuntimeError("Python dependency environment changed during recovery")
    final_input_manifest = _primary_input_manifest()
    _verify_exact_file_manifest(
        input_manifest,
        final_input_manifest,
        label="canonical input",
    )
    final_secondary_output_manifest = _records_for_roots(
        SECONDARY_OUTPUT_ROOTS,
        label="recovered secondary output",
    )
    _verify_exact_file_manifest(
        secondary_output_manifest,
        final_secondary_output_manifest,
        label="recovered secondary output",
    )

    manifest_refresh = [
        "scripts/build_transactions_evidence.py",
        "--output-dir",
        evidence_stage_arg,
        "--manifest-only",
    ]
    manifest_verify = [
        "scripts/build_transactions_evidence.py",
        "--output-dir",
        evidence_stage_arg,
        "--verify-manifest",
    ]
    backup_record = (
        evidence_backup.relative_to(REPO_ROOT).as_posix()
        if evidence_backup is not None
        else None
    )
    stale_stage = None
    if args.discard_stage is not None:
        candidate = args.discard_stage.resolve()
        if candidate.parent != evidence_live.parent.resolve():
            raise ValueError("identified stale stage must be a sibling of live evidence")
        stale_stage = _display_path(candidate)

    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rebuild_mode": "recovered_interrupted_full_publication",
        "repository_root": ".",
        "published_evidence_directory": evidence_live_arg,
        "staged_evidence_directory": evidence_stage_arg,
        "python_executable": PYTHON_COMMAND_TOKEN,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "environment": {
            "MPLCONFIGDIR": _portable_path_record(os.environ["MPLCONFIGDIR"]),
            "XDG_CACHE_HOME": _portable_path_record(os.environ["XDG_CACHE_HOME"]),
        },
        "environment_manifest": environment_manifest,
        "code_manifest": code_manifest,
        "input_manifest": input_manifest,
        "secondary_output_manifest": secondary_output_manifest,
        "reuse_fingerprint": reuse_fingerprint,
        "previous_evidence_archive": backup_record,
        "commands": command_log,
        "manifest_refresh_command": _portable_command(manifest_refresh),
        "manifest_verify_command": _portable_command(manifest_verify),
        "path_recording_policy": (
            "repository-relative paths; {python} denotes the recorded Python "
            "implementation/version/environment; external paths are redacted"
        ),
        "publication_protocol": (
            "recover after interrupted full-run publication; exact premise "
            "hashes; fresh sibling stage; regenerate evidence; audit; full "
            "tests; exact manifest refresh and verification; rollback-capable "
            "directory swap"
        ),
        "recovery_note": (
            "The preceding full runner was interrupted after expensive output "
            "generation but before atomic publication. This recovery regenerated "
            "the evidence stage from the extant output roots and independently "
            "verified their complete hashes."
            + (f" The abandoned stage was {stale_stage}." if stale_stage else "")
        ),
    }
    (evidence_stage / "build_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    _run_manifest_utility(manifest_refresh)
    _run_manifest_utility(manifest_verify)
    archived = _publish_staged_evidence(
        evidence_stage,
        evidence_live,
        evidence_backup,
    )
    if archived != backup_record:
        raise RuntimeError("published evidence backup record changed unexpectedly")
    _run_manifest_utility(
        [
            "scripts/build_transactions_evidence.py",
            "--output-dir",
            evidence_live_arg,
            "--verify-manifest",
        ]
    )
    print(f"recovered_transactions_evidence={evidence_live}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
