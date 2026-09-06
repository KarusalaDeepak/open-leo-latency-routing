#!/usr/bin/env python3
"""Rebuild the canonical Transactions evidence from prepared source data.

Downloads are deliberately separate because LENS is large and each external
source has its own license/checksum step.  This runner removes the former
producer/consumer path mismatch: every command writes to the directories read
by ``build_transactions_evidence.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_temporal_resolution_tables import (
    DEFAULT_DATA_ROOT as TEMPORAL_RESOLUTION_DATA_ROOT_VALUE,
    DEFAULT_MANIFEST as TEMPORAL_RESOLUTION_MANIFEST_VALUE,
    DEFAULT_MAX_FILES as TEMPORAL_RESOLUTION_MAX_FILES,
    resolve_temporal_resolution_inputs,
)

TEMPORAL_RESOLUTION_MANIFEST = REPO_ROOT / TEMPORAL_RESOLUTION_MANIFEST_VALUE
TEMPORAL_RESOLUTION_DATA_ROOT = REPO_ROOT / TEMPORAL_RESOLUTION_DATA_ROOT_VALUE
PYTHON_COMMAND_TOKEN = "{python}"
EXTERNAL_PATH_TOKEN = "{external-path-redacted}"
PROVENANCE_SCHEMA_VERSION = 3

SECONDARY_OUTPUT_ROOTS = (
    REPO_ROOT / "results/transactions_seed_matrix",
    REPO_ROOT / "results/transactions_seed_matrix_30_short",
    REPO_ROOT / "results/simulator_parameter_sensitivity",
    REPO_ROOT / "data/processed/temporal_resolutions",
    REPO_ROOT / "results/temporal_resolution_evaluation",
    REPO_ROOT / "results/service_path_reviewer_revision",
    REPO_ROOT / "results/reviewer_validation",
    REPO_ROOT / "results/gate_operating_characteristics",
    REPO_ROOT / "results/physical_feasibility",
    REPO_ROOT / "results/zero_shot_transfer_validation",
)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _portable_path_record(value: str | Path) -> str:
    """Record a path without serializing a user- or host-specific prefix."""

    path = Path(value)
    if not path.is_absolute():
        if ".." in path.parts:
            return EXTERNAL_PATH_TOKEN
        return path.as_posix() or "."
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return EXTERNAL_PATH_TOKEN
    return relative.as_posix() or "."


def _portable_command(arguments: list[str]) -> list[str]:
    """Represent a Python invocation independently of its installation path."""

    recorded_arguments = [
        _portable_path_record(argument)
        if Path(argument).is_absolute()
        else argument
        for argument in arguments
    ]
    return [PYTHON_COMMAND_TOKEN, *recorded_arguments]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _code_manifest() -> dict[str, object]:
    """Hash the complete executable specification, including uncommitted files."""

    files: set[Path] = set()
    for directory, suffixes in (
        (REPO_ROOT / "src", {".py"}),
        (REPO_ROOT / "scripts", {".py", ".sh"}),
        (REPO_ROOT / "tests", {".py"}),
        (REPO_ROOT / "configs", {".yaml", ".yml", ".json"}),
    ):
        if directory.exists():
            files.update(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix in suffixes
                and "__pycache__" not in path.parts
            )
    for name in (
        "pyproject.toml",
        "pytest.ini",
        "requirements.txt",
        "requirements-lock.txt",
        "requirements-hypatia.txt",
    ):
        path = REPO_ROOT / name
        if path.exists():
            files.add(path)
    records = [_file_record(path) for path in sorted(files)]
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"aggregate_sha256": aggregate, "files": records}


def _environment_manifest() -> dict[str, object]:
    """Fingerprint the interpreter and packages without host-specific paths."""

    packages: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages.append(
            {
                "name": name.strip().lower().replace("_", "-"),
                "version": distribution.version,
            }
        )
    packages.sort(key=lambda item: (item["name"], item["version"]))
    specification = {
        "schema_version": 2,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    aggregate = hashlib.sha256(
        json.dumps(
            specification,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {"aggregate_sha256": aggregate, **specification}


def _temporal_resolution_external_inputs(
    *,
    manifest_path: Path = TEMPORAL_RESOLUTION_MANIFEST,
    data_root: Path = TEMPORAL_RESOLUTION_DATA_ROOT,
    max_files: int = TEMPORAL_RESOLUTION_MAX_FILES,
) -> list[Path]:
    """Resolve the exact external rows consumed by the resolution builder."""

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing temporal-resolution candidate manifest: {manifest_path}"
        )
    targets = resolve_temporal_resolution_inputs(
        data_root=data_root,
        manifest_path=manifest_path,
        max_files=max_files,
    )
    return [manifest_path.resolve(), *targets]


def _records_for_roots(
    roots: tuple[Path, ...],
    *,
    label: str,
) -> list[dict[str, object]]:
    """Hash every file under required output roots, with no silent omissions."""

    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {label} roots:\n" + "\n".join(missing)
        )
    paths: set[Path] = set()
    for root in roots:
        if root.is_file():
            paths.add(root)
        else:
            paths.update(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"{label} roots contain no files")
    return [_file_record(path) for path in sorted(paths)]


def _verify_exact_file_manifest(
    expected_records: object,
    observed_records: list[dict[str, object]],
    *,
    label: str,
) -> None:
    """Require identical path sets, byte sizes, and hashes for reused files."""

    if not isinstance(expected_records, list) or not expected_records:
        raise RuntimeError(
            f"cannot verify {label}: prior provenance has no complete manifest"
        )

    def index(
        records: list[dict[str, object]],
        source: str,
    ) -> dict[str, dict[str, object]]:
        indexed: dict[str, dict[str, object]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError(f"invalid {label} record in {source}")
            path = record.get("path")
            sha256 = record.get("sha256")
            size = record.get("bytes")
            if (
                not isinstance(path, str)
                or not isinstance(sha256, str)
                or not isinstance(size, int)
                or isinstance(size, bool)
            ):
                raise RuntimeError(f"invalid {label} record in {source}: {record}")
            if path in indexed:
                raise RuntimeError(
                    f"duplicate {label} path in {source}: {path}"
                )
            indexed[path] = record
        return indexed

    expected = index(expected_records, "prior provenance")
    observed = index(observed_records, "current tree")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    size_drift = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path]["bytes"] != observed[path]["bytes"]
    )
    hash_drift = sorted(
        path
        for path in set(expected) & set(observed)
        if expected[path]["sha256"] != observed[path]["sha256"]
    )
    if missing or extra or size_drift or hash_drift:
        raise RuntimeError(
            f"{label} verification failed: missing={missing}; extra={extra}; "
            f"size_drift={size_drift}; hash_drift={hash_drift}"
        )


def _primary_input_manifest() -> list[dict[str, object]]:
    """Hash the canonical tables actually consumed by validation runners."""

    paths = [
        REPO_ROOT / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency.zip",
        REPO_ROOT
        / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency"
        / "Operator_A_RTT.csv",
        REPO_ROOT
        / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency"
        / "Operator_B_RTT.csv",
        REPO_ROOT
        / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency"
        / "Satellite_RTT.csv",
        REPO_ROOT / "data/processed/commect_multiaccess_10s.csv",
        REPO_ROOT / "data/processed/commect_multiaccess_10s.metadata.json",
        REPO_ROOT / "data/processed/ping_time_bins.csv",
        REPO_ROOT / "data/processed/lens_victoria_multihomed_holdout_10s.csv",
        REPO_ROOT / "data/processed/lens_victoria_multihomed_holdout_10s.metadata.json",
        REPO_ROOT / "data/processed/wetlinks_latency_5min.csv",
        REPO_ROOT / "data/processed/wetlinks_latency_5min.metadata.json",
        REPO_ROOT / "data/processed/physics_informed_orbital_multipath_5s.csv",
        REPO_ROOT / "data/processed/physics_informed_orbital_multipath_5s.metadata.json",
        REPO_ROOT / "data/processed/physics_informed_orbital_multipath_10s.csv",
        REPO_ROOT / "data/processed/physics_informed_orbital_multipath_10s.metadata.json",
        REPO_ROOT / "data/processed/hypatia_service_paths_10s.csv",
        REPO_ROOT / "data/processed/hypatia_service_paths_10s.metadata.json",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "cannot fingerprint missing canonical build inputs:\n" + "\n".join(missing)
        )
    records_by_path = {
        str(record["path"]): record
        for record in (_file_record(path) for path in paths)
    }
    wetlinks_dir = REPO_ROOT / "data/external/wetlinks_dataset/Preprocessed_Data"
    wetlinks_paths = sorted(wetlinks_dir.glob("analysis_data_*.csv"))
    if not wetlinks_paths:
        raise FileNotFoundError(
            f"no WetLinks raw inputs found beneath {wetlinks_dir}"
        )
    for path in wetlinks_paths:
        record = _file_record(path)
        records_by_path[str(record["path"])] = record

    for path in _temporal_resolution_external_inputs():
        record = _file_record(path)
        records_by_path[str(record["path"])] = record
    victoria_metadata = json.loads(
        (
            REPO_ROOT
            / "data/processed/lens_victoria_multihomed_holdout_10s.metadata.json"
        ).read_text(encoding="utf-8")
    )
    lens_root = REPO_ROOT / "data/raw/lens_2025_03/LENS-2025-03"
    for record in victoria_metadata.get("source_files", []):
        path = lens_root / str(record["path"])
        observed = _file_record(path)
        if observed["sha256"] != record["sha256"]:
            raise ValueError(f"Victoria source hash changed during rebuild: {path}")
        records_by_path[str(observed["path"])] = observed
    return [records_by_path[path] for path in sorted(records_by_path)]


def _reuse_fingerprint(
    code_manifest: dict[str, object],
    input_manifest: list[dict[str, object]],
    environment_manifest: dict[str, object],
) -> str:
    payload = {
        "code_sha256": code_manifest["aggregate_sha256"],
        "environment_sha256": environment_manifest["aggregate_sha256"],
        "inputs": input_manifest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run(
    arguments: list[str],
    *,
    dry_run: bool,
    command_log: list[dict[str, object]],
) -> None:
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    entry: dict[str, object] = {
        "argv": _portable_command(arguments),
        "cwd": ".",
    }
    command_log.append(entry)
    if not dry_run:
        entry["started_utc"] = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        entry["elapsed_seconds"] = time.perf_counter() - started
        entry["returncode"] = completed.returncode
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)


def _next_evidence_backup_path(source: Path) -> Path | None:
    if not source.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = REPO_ROOT / "output/build_backups" / f"transactions_evidence_{stamp}"
    suffix = 1
    while destination.exists():
        destination = destination.with_name(
            f"transactions_evidence_{stamp}_{suffix}"
        )
        suffix += 1
    return destination


def _fresh_evidence_stage(*, dry_run: bool) -> Path:
    results_root = REPO_ROOT / "results"
    if dry_run:
        return results_root / ".transactions_evidence.staging-DRYRUN"
    results_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=".transactions_evidence.staging-",
            dir=results_root,
        )
    )


def _publish_staged_evidence(
    stage: Path,
    live: Path,
    backup: Path | None,
) -> str | None:
    """Publish a verified stage and restore the prior tree on swap failure."""

    if not stage.is_dir():
        raise FileNotFoundError(f"missing staged evidence directory: {stage}")
    if stage.parent != live.parent:
        raise ValueError("evidence staging and live directories must be siblings")
    if live.exists() and backup is None:
        raise ValueError("a live evidence tree requires a backup destination")
    if backup is not None and backup.exists():
        raise FileExistsError(f"evidence backup already exists: {backup}")

    if not live.exists():
        os.replace(stage, live)
        return None

    assert backup is not None
    backup.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"~ preserve {_display_path(live)} -> {_display_path(backup)}",
        flush=True,
    )
    os.replace(live, backup)
    try:
        os.replace(stage, live)
    except BaseException:
        # The last known-good tree was moved only after the stage passed every
        # check.  If the second rename fails, immediately restore that tree.
        os.replace(backup, live)
        raise
    return _display_path(backup)


def _run_manifest_utility(arguments: list[str]) -> None:
    """Run a final manifest operation outside the provenance command cycle."""

    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Reuse, and require, existing multi-seed and secondary diagnostic "
            "outputs instead of recomputing them."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command_log: list[dict[str, object]] = []

    previous_provenance_path = (
        REPO_ROOT / "results/transactions_evidence/build_provenance.json"
    )
    previous_provenance = None
    if previous_provenance_path.exists():
        previous_provenance = json.loads(
            previous_provenance_path.read_text(encoding="utf-8")
        )
    code_manifest = _code_manifest()
    environment_manifest = _environment_manifest()

    if args.quick and not args.dry_run:
        if not isinstance(previous_provenance, dict):
            raise RuntimeError(
                "--quick refused: no prior evidence provenance is available; "
                "run a full rebuild without --quick"
            )
        if (
            previous_provenance.get("schema_version")
            != PROVENANCE_SCHEMA_VERSION
            or not isinstance(
                previous_provenance.get("environment_manifest"), dict
            )
            or not isinstance(
                previous_provenance.get("secondary_output_manifest"), list
            )
            or not previous_provenance["secondary_output_manifest"]
        ):
            raise RuntimeError(
                "--quick refused: prior provenance predates strict dependency "
                "and secondary-output manifests; run one full rebuild first"
            )

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    required = [
        REPO_ROOT
        / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency.zip",
        REPO_ROOT
        / "data/external/commect_latency/Mutli-connectivity_KPIs_Latency",
        REPO_ROOT / "data/processed/ping_time_bins.csv",
        REPO_ROOT / "data/raw/lens_2025_03/LENS-2025-03",
        REPO_ROOT / "data/external/wetlinks_dataset/Preprocessed_Data",
        TEMPORAL_RESOLUTION_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing and not args.dry_run:
        raise FileNotFoundError(
            "prepare/download the required sources first:\n" + "\n".join(missing)
        )

    primary = [
        ["scripts/build_commect_multiaccess_trace.py"],
        ["scripts/run_commect_multiaccess_validation.py"],
        ["scripts/run_commect_rolling_origin_validation.py"],
        ["scripts/run_commect_rolling_timestamp_sensitivity.py"],
        ["scripts/run_commect_threshold_gate_sensitivity.py"],
        ["scripts/run_commect_timestamp_sensitivity.py"],
        ["scripts/audit_gate_design_sensitivity.py"],
        [
            "scripts/build_victoria_multihomed_trace.py",
            "--session-count",
            "12",
            "--session-offset",
            "100",
            "--output",
            "data/processed/lens_victoria_multihomed_holdout_10s.csv",
        ],
        ["scripts/run_measured_multihomed_validation.py"],
        ["scripts/build_wetlinks_longitudinal_table.py"],
        ["scripts/run_wetlinks_longitudinal_validation.py"],
        ["scripts/generate_wetlinks_validation_figure.py"],
        ["scripts/generate_physics_informed_multipath_trace.py"],
        ["scripts/run_independent_multipath_validation.py"],
    ]
    for command in primary:
        _run(command, dry_run=args.dry_run, command_log=command_log)

    input_manifest = [] if args.dry_run else _primary_input_manifest()
    current_reuse_fingerprint = (
        "dry-run"
        if args.dry_run
        else _reuse_fingerprint(
            code_manifest,
            input_manifest,
            environment_manifest,
        )
    )
    if args.quick:
        if not args.dry_run:
            previous_fingerprint = (
                previous_provenance or {}
            ).get("reuse_fingerprint")
            if previous_fingerprint != current_reuse_fingerprint:
                raise RuntimeError(
                    "--quick refused: the prior evidence provenance does not "
                    "match the current code, configuration, and canonical inputs; "
                    "run a full rebuild without --quick"
                )
            secondary_output_manifest = _records_for_roots(
                SECONDARY_OUTPUT_ROOTS,
                label="reused secondary output",
            )
            _verify_exact_file_manifest(
                (previous_provenance or {}).get("secondary_output_manifest"),
                secondary_output_manifest,
                label="reused secondary output",
            )
            for record in secondary_output_manifest:
                print(f"= verified reuse {record['path']}", flush=True)
        else:
            secondary_output_manifest = []
    else:
        secondary = [
            [
                "scripts/run_independent_multipath_seed_matrix.py",
                "--max-workers",
                "3",
            ],
            [
                "scripts/run_independent_multipath_seed_matrix.py",
                "--seeds",
                *[str(seed) for seed in range(3001, 3031)],
                "--duration-hours",
                "0.5",
                "--output-dir",
                "results/transactions_seed_matrix_30_short",
                "--max-workers",
                "3",
            ],
            ["scripts/run_simulator_parameter_sensitivity.py"],
            ["scripts/build_temporal_resolution_tables.py"],
            ["scripts/run_temporal_resolution_evaluation.py"],
            [
                "scripts/run_service_path_experiments.py",
                "--config",
                "configs/experiment.yaml",
                "--output-dir",
                "results/service_path_reviewer_revision",
                "--allow-normalized-counterfactual",
            ],
            ["scripts/run_reviewer_validation.py"],
            ["scripts/run_zero_shot_transfer_validation.py"],
            ["scripts/audit_gate_operating_characteristics.py"],
            ["scripts/run_physical_feasibility_analysis.py"],
        ]
        for command in secondary:
            _run(command, dry_run=args.dry_run, command_log=command_log)
        secondary_output_manifest = (
            []
            if args.dry_run
            else _records_for_roots(
                SECONDARY_OUTPUT_ROOTS,
                label="secondary output",
            )
        )

    evidence_live = REPO_ROOT / "results/transactions_evidence"
    evidence_stage = _fresh_evidence_stage(dry_run=args.dry_run)
    evidence_stage_arg = evidence_stage.relative_to(REPO_ROOT).as_posix()
    evidence_live_arg = evidence_live.relative_to(REPO_ROOT).as_posix()
    evidence_backup = _next_evidence_backup_path(evidence_live)
    final = [
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
    for command in final:
        _run(command, dry_run=args.dry_run, command_log=command_log)

    manifest_refresh = [
        "scripts/build_transactions_evidence.py",
        "--output-dir",
        evidence_stage_arg,
        "--manifest-only",
    ]
    manifest_verify_stage = [
        "scripts/build_transactions_evidence.py",
        "--output-dir",
        evidence_stage_arg,
        "--verify-manifest",
    ]
    if args.dry_run:
        print("+", " ".join([sys.executable, *manifest_refresh]), flush=True)
        print(
            "+",
            " ".join([sys.executable, *manifest_verify_stage]),
            flush=True,
        )
        print(
            f"~ publish verified {evidence_stage_arg} -> {evidence_live_arg}",
            flush=True,
        )
        return 0

    # Detect source, dependency, input, or reused-output drift that occurred
    # while the build was running. Publication stops before the live tree is
    # touched if any executable premise changed.
    final_code_manifest = _code_manifest()
    if final_code_manifest["aggregate_sha256"] != code_manifest["aggregate_sha256"]:
        raise RuntimeError("code/configuration changed during rebuild; refusing publication")
    final_environment_manifest = _environment_manifest()
    if (
        final_environment_manifest["aggregate_sha256"]
        != environment_manifest["aggregate_sha256"]
    ):
        raise RuntimeError("Python dependency environment changed during rebuild")
    final_input_manifest = _primary_input_manifest()
    _verify_exact_file_manifest(
        input_manifest,
        final_input_manifest,
        label="canonical input",
    )
    final_secondary_output_manifest = _records_for_roots(
        SECONDARY_OUTPUT_ROOTS,
        label="secondary output",
    )
    _verify_exact_file_manifest(
        secondary_output_manifest,
        final_secondary_output_manifest,
        label="secondary output",
    )

    backup_record = (
        evidence_backup.relative_to(REPO_ROOT).as_posix()
        if evidence_backup is not None
        else None
    )
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rebuild_mode": "quick" if args.quick else "full",
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
        "reuse_fingerprint": current_reuse_fingerprint,
        "previous_evidence_archive": backup_record,
        "commands": command_log,
        "manifest_refresh_command": _portable_command(manifest_refresh),
        "manifest_verify_command": _portable_command(manifest_verify_stage),
        "path_recording_policy": (
            "repository-relative paths; {python} denotes the recorded Python "
            "implementation/version/environment; external paths are redacted"
        ),
        "publication_protocol": (
            "fresh sibling stage; generate; audit; full tests; exact manifest "
            "refresh and verification; rollback-capable directory swap"
        ),
    }
    provenance_path = evidence_stage / "build_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    _run_manifest_utility(manifest_refresh)
    _run_manifest_utility(manifest_verify_stage)
    archived_evidence = _publish_staged_evidence(
        evidence_stage,
        evidence_live,
        evidence_backup,
    )
    if archived_evidence != backup_record:
        raise RuntimeError("published evidence backup record changed unexpectedly")
    _run_manifest_utility(
        [
            "scripts/build_transactions_evidence.py",
            "--output-dir",
            evidence_live_arg,
            "--verify-manifest",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
