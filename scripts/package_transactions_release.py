#!/usr/bin/env python3
"""Build and verify the curated Transactions reproducibility archive.

The archive intentionally excludes raw/processed third-party data, VCS state,
caches, editor metadata, and superseded outputs.  It is therefore suitable for
source/evidence integrity review, but it is not a raw-data-self-contained
reproduction bundle.  External inputs must be obtained under their own terms.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
from importlib import metadata as importlib_metadata
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARCHIVE_ROOT = "opportunity-aware-evidence-gating-artifact"
PROVENANCE_SCHEMA_VERSION = 3
PYTHON_COMMAND_TOKEN = "{python}"
EXTERNAL_PATH_TOKEN = "{external-path-redacted}"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/artifact/Opportunity_Aware_Evidence_Gating_Reproducibility_Artifact.tar.gz"
)

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-hypatia.txt",
)

TREE_ROOTS = (
    "configs",
    "src/open_leo_latency_routing",
    "scripts",
    "tests",
    "results/transactions_evidence",
    "results/wetlinks_longitudinal_validation",
    "results/figures/manuscript_assets",
)

CURATED_DOC_FILES = (
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/closed_loop_field_validation_protocol.md",
    "docs/commercial_multileo_acquisition_status.json",
    "docs/commercial_multileo_validation_protocol.md",
    "docs/model_feature_contract.md",
    "docs/open_leo_latency_routing_scope.md",
    "docs/release_readiness_checklist.md",
    "docs/reproducibility_guide.md",
)

REVIEWER_READINESS_FILES = (
    "results/reviewer_readiness/reviewer_readiness.csv",
    "results/reviewer_readiness/reviewer_readiness.json",
    "results/reviewer_readiness/reviewer_readiness.md",
)

EXTRA_FILES = (
    "results/figures/wetlinks_longitudinal_validation.pdf",
    "results/figures/wetlinks_longitudinal_validation.png",
    "results/transactions_seed_matrix/seed_matrix_metadata.json",
)

FORBIDDEN_PARTS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".mpl-cache",
    "raw",
    "processed",
    "build_backups",
}

FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
    ".bak",
    ".tmp",
}


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    bytes: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_forbidden(relative: Path) -> bool:
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return True
    if any(part.startswith("._") for part in relative.parts):
        return True
    if relative.name in {".DS_Store", "Thumbs.db"}:
        return True
    return relative.suffix.lower() in FORBIDDEN_SUFFIXES


def _active_requirement_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split(" #", 1)[0].strip()
        if line and not line.startswith("#"):
            lines.append((line_number, line))
    return lines


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _verify_installed_dependency_pins() -> dict[str, str]:
    """Require exact direct pins and the interpreter versions they declare."""

    requirements_path = REPO_ROOT / "requirements.txt"
    lock_path = REPO_ROOT / "requirements-lock.txt"
    if not requirements_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(
            "dependency preflight requires requirements.txt and "
            "requirements-lock.txt"
        )

    direct_names: set[str] = set()
    for line_number, line in _active_requirement_lines(requirements_path):
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match is None:
            raise ValueError(
                f"cannot parse direct requirement at "
                f"requirements.txt:{line_number}: {line!r}"
            )
        name = _normalize_distribution_name(match.group(1))
        if name in direct_names:
            raise ValueError(f"duplicate direct requirement: {name}")
        direct_names.add(name)

    pins: dict[str, str] = {}
    pin_pattern = re.compile(
        r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)"
    )
    for line_number, line in _active_requirement_lines(lock_path):
        match = pin_pattern.fullmatch(line)
        if match is None:
            raise ValueError(
                "every direct lock entry must be an exact, unconditional "
                f"name==version pin; requirements-lock.txt:{line_number}: "
                f"{line!r}"
            )
        name = _normalize_distribution_name(match.group(1))
        if name in pins:
            raise ValueError(f"duplicate dependency lock pin: {name}")
        pins[name] = match.group(2)

    missing_pins = sorted(direct_names - set(pins))
    extra_pins = sorted(set(pins) - direct_names)
    if missing_pins or extra_pins:
        raise ValueError(
            "direct requirement and lock sets differ: "
            f"missing_pins={missing_pins}; extra_pins={extra_pins}"
        )

    missing_installs: list[str] = []
    version_drift: list[str] = []
    for name, pinned_version in sorted(pins.items()):
        try:
            installed_version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            missing_installs.append(name)
            continue
        if installed_version != pinned_version:
            version_drift.append(
                f"{name}: pinned={pinned_version}, installed={installed_version}"
            )
    if missing_installs or version_drift:
        raise RuntimeError(
            "direct dependency preflight failed: "
            f"missing={missing_installs}; version_drift={version_drift}"
        )
    return pins


def _verify_evidence_tree() -> dict[str, object]:
    """Use the evidence builder's exact-set manifest verifier."""

    from scripts.build_transactions_evidence import _verify_evidence_manifest

    return _verify_evidence_manifest(
        REPO_ROOT / "results/transactions_evidence"
    )


def _contains_absolute_path(value: str) -> bool:
    """Detect POSIX, Windows, or file-URI paths embedded in a string."""

    if "file://" in value.lower():
        return True
    candidates = re.split(r"[\s\"'=,;()\[\]{}]+", value)
    for candidate in candidates:
        token = candidate.strip().rstrip(".:!?\")")
        if not token:
            continue
        if PurePosixPath(token).is_absolute():
            return True
        if PureWindowsPath(token).is_absolute():
            return True
    return False


def _iter_string_values(
    value: object,
    *,
    location: str = "$",
) -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    if isinstance(value, str):
        strings.append((location, value))
    elif isinstance(value, dict):
        for key, child in value.items():
            strings.extend(
                _iter_string_values(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            strings.extend(
                _iter_string_values(child, location=f"{location}[{index}]")
            )
    return strings


def _verify_portable_path(
    value: object,
    *,
    label: str,
    allow_dot: bool = False,
    allow_redacted: bool = False,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty portable path")
    if value == EXTERNAL_PATH_TOKEN:
        if allow_redacted:
            return value
        raise ValueError(f"{label} must be repository-relative, not redacted")
    if value == ".":
        if allow_dot:
            return value
        raise ValueError(f"{label} must identify a repository-relative file")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
    ):
        raise ValueError(f"{label} is not a canonical portable path: {value!r}")
    return value


def _verify_portable_command(value: object, *, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty argv list")
    if value[0] != PYTHON_COMMAND_TOKEN:
        raise ValueError(
            f"{label}[0] must be {PYTHON_COMMAND_TOKEN!r}, "
            f"not {value[0]!r}"
        )
    for index, argument in enumerate(value):
        if not isinstance(argument, str) or not argument:
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        if _contains_absolute_path(argument):
            raise ValueError(
                f"{label}[{index}] leaks an absolute path: {argument!r}"
            )
        normalized = argument.replace("\\", "/")
        if ".." in PurePosixPath(normalized).parts:
            raise ValueError(
                f"{label}[{index}] contains parent traversal: {argument!r}"
            )


def _verify_portable_provenance(payload: dict[str, object]) -> None:
    """Enforce the schema-3 portable, privacy-safe provenance contract."""

    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "portable provenance contract requires schema "
            f"{PROVENANCE_SCHEMA_VERSION}"
        )
    if payload.get("repository_root") != ".":
        raise ValueError("build provenance repository_root must be '.'")
    if payload.get("python_executable") != PYTHON_COMMAND_TOKEN:
        raise ValueError(
            "build provenance python_executable must be the portable "
            f"token {PYTHON_COMMAND_TOKEN!r}"
        )

    _verify_portable_path(
        payload.get("published_evidence_directory"),
        label="published_evidence_directory",
    )
    _verify_portable_path(
        payload.get("staged_evidence_directory"),
        label="staged_evidence_directory",
    )
    previous_archive = payload.get("previous_evidence_archive")
    if previous_archive is not None:
        _verify_portable_path(
            previous_archive,
            label="previous_evidence_archive",
        )

    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("build provenance lacks portable environment paths")
    for name, value in environment.items():
        _verify_portable_path(
            value,
            label=f"environment.{name}",
            allow_dot=True,
            allow_redacted=True,
        )

    environment_manifest = payload.get("environment_manifest")
    if not isinstance(environment_manifest, dict):
        raise ValueError("build provenance lacks environment_manifest")
    if environment_manifest.get("schema_version") != 2:
        raise ValueError("environment manifest must use portable schema 2")
    if "python_executable" in environment_manifest:
        raise ValueError(
            "portable environment manifest must not record python_executable"
        )

    code_manifest = payload.get("code_manifest")
    if not isinstance(code_manifest, dict):
        raise ValueError("build provenance lacks code_manifest")
    manifest_specs = (
        ("code_manifest.files", code_manifest.get("files")),
        ("input_manifest", payload.get("input_manifest")),
        ("secondary_output_manifest", payload.get("secondary_output_manifest")),
    )
    for label, raw_records in manifest_specs:
        if not isinstance(raw_records, list):
            raise ValueError(f"{label} must be a list")
        for index, record in enumerate(raw_records):
            if not isinstance(record, dict):
                raise ValueError(f"{label}[{index}] must be an object")
            _verify_portable_path(
                record.get("path"),
                label=f"{label}[{index}].path",
                allow_redacted=True,
            )

    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("build provenance commands must be a non-empty list")
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise ValueError(f"commands[{index}] must be an object")
        if command.get("cwd") != ".":
            raise ValueError(f"commands[{index}].cwd must be '.'")
        _verify_portable_command(
            command.get("argv"),
            label=f"commands[{index}].argv",
        )
    for field in ("manifest_refresh_command", "manifest_verify_command"):
        _verify_portable_command(payload.get(field), label=field)

    policy = payload.get("path_recording_policy")
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("build provenance lacks path_recording_policy")

    leaked = [
        f"{location}={value!r}"
        for location, value in _iter_string_values(payload)
        if _contains_absolute_path(value)
    ]
    if leaked:
        raise ValueError(
            "build provenance leaks absolute host paths: "
            + "; ".join(leaked[:5])
        )


def _load_build_provenance() -> dict[str, object]:
    path = REPO_ROOT / "results/transactions_evidence/build_provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical build provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError(
            "release packaging requires portable build provenance schema "
            f"{PROVENANCE_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("code_manifest"), dict):
        raise ValueError("build provenance lacks code_manifest")
    if not isinstance(payload.get("environment_manifest"), dict):
        raise ValueError("build provenance lacks environment_manifest")
    _verify_portable_provenance(payload)
    return payload


def _verify_current_code_manifest(provenance: dict[str, object]) -> None:
    """Bind the release source to the executable specification that built it."""

    from scripts.rebuild_transactions_artifact import _code_manifest

    expected = provenance["code_manifest"]
    observed = _code_manifest()
    if observed != expected:
        expected_digest = (
            expected.get("aggregate_sha256")
            if isinstance(expected, dict)
            else None
        )
        raise RuntimeError(
            "current code/configuration manifest differs from build provenance: "
            f"expected={expected_digest}; "
            f"observed={observed.get('aggregate_sha256')}"
        )


def _verify_build_dependency_versions(
    provenance: dict[str, object],
    pins: dict[str, str],
) -> None:
    """Require the evidence-build environment to match every direct pin."""

    environment = provenance["environment_manifest"]
    raw_packages = environment.get("packages")
    if not isinstance(raw_packages, list):
        raise ValueError("build environment manifest lacks package records")
    build_versions: dict[str, str] = {}
    for index, record in enumerate(raw_packages):
        if not isinstance(record, dict):
            raise ValueError(
                f"build environment package record {index} is not an object"
            )
        raw_name = record.get("name")
        raw_version = record.get("version")
        if not isinstance(raw_name, str) or not isinstance(raw_version, str):
            raise ValueError(
                f"build environment package record {index} is malformed"
            )
        name = _normalize_distribution_name(raw_name)
        if name in build_versions and build_versions[name] != raw_version:
            raise ValueError(
                f"build environment contains conflicting versions for {name}"
            )
        build_versions[name] = raw_version

    missing: list[str] = []
    drift: list[str] = []
    for name, pinned_version in sorted(pins.items()):
        build_version = build_versions.get(name)
        if build_version is None:
            missing.append(name)
        elif build_version != pinned_version:
            drift.append(
                f"{name}: pinned={pinned_version}, build={build_version}"
            )
    if missing or drift:
        raise RuntimeError(
            "build provenance dependency versions differ from direct pins: "
            f"missing={missing}; version_drift={drift}"
        )


def _verify_reviewer_readiness() -> dict[str, object]:
    """Reject failed checks while retaining documented deployment blockers."""

    root = REPO_ROOT / "results/reviewer_readiness"
    csv_path = root / "reviewer_readiness.csv"
    json_path = root / "reviewer_readiness.json"
    markdown_path = root / "reviewer_readiness.md"
    for path in (csv_path, json_path, markdown_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing reviewer-readiness output: {path}")

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {
            "category",
            "check",
            "status",
            "evidence",
            "likely_future_reviewer_comment",
            "required_action",
        }
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError("reviewer-readiness CSV lacks required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("reviewer-readiness CSV contains no checks")

    required_categories = {
        "artifact_integrity",
        "evaluation_evidence",
        "deployment_evidence",
    }
    observed_categories = {str(row["category"]) for row in rows}
    if not required_categories.issubset(observed_categories):
        raise ValueError(
            "reviewer-readiness CSV lacks required categories: "
            f"{sorted(required_categories - observed_categories)}"
        )

    invalid_status = [
        str(row["check"])
        for row in rows
        if row["status"] not in {"PASS", "PENDING"}
    ]
    if invalid_status:
        raise RuntimeError(
            "reviewer-readiness contains FAIL or unknown status: "
            f"{invalid_status}"
        )
    pending = [row for row in rows if row["status"] == "PENDING"]
    for row in pending:
        if row["category"] != "deployment_evidence":
            raise RuntimeError(
                "only documented deployment-evidence PENDING items may be "
                f"packaged: {row['check']}"
            )
        missing_documentation = [
            field
            for field in (
                "evidence",
                "likely_future_reviewer_comment",
                "required_action",
            )
            if not str(row[field]).strip()
        ]
        if missing_documentation:
            raise RuntimeError(
                f"PENDING reviewer item {row['check']!r} lacks documentation: "
                f"{missing_documentation}"
            )

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("reviewer-readiness JSON must be an object")
    pass_count = sum(row["status"] == "PASS" for row in rows)
    pending_count = len(pending)
    if summary.get("pass_count") != pass_count:
        raise ValueError("reviewer-readiness pass_count disagrees with CSV")
    if summary.get("pending_count") != pending_count:
        raise ValueError("reviewer-readiness pending_count disagrees with CSV")
    if summary.get("all_review_items_complete") is not (pending_count == 0):
        raise ValueError(
            "reviewer-readiness all_review_items_complete disagrees with CSV"
        )
    if pending and summary.get("allow_pending_acknowledged") is not True:
        raise RuntimeError(
            "documented PENDING items were not explicitly acknowledged"
        )
    if summary.get("fail_count", 0) != 0:
        raise RuntimeError("reviewer-readiness summary reports failures")

    calculated_by_category: dict[str, dict[str, object]] = {}
    for category in sorted(observed_categories):
        category_rows = [row for row in rows if row["category"] == category]
        category_pending = sum(
            row["status"] == "PENDING" for row in category_rows
        )
        calculated_by_category[category] = {
            "pass_count": sum(
                row["status"] == "PASS" for row in category_rows
            ),
            "pending_count": category_pending,
            "complete": category_pending == 0,
        }
    if summary.get("by_category") != calculated_by_category:
        raise ValueError("reviewer-readiness category summary disagrees with CSV")

    artifact_complete = calculated_by_category["artifact_integrity"]["complete"]
    if artifact_complete is not True:
        raise RuntimeError("artifact-integrity reviewer checks are incomplete")
    if summary.get("artifact_checks_complete") is not artifact_complete:
        raise ValueError(
            "reviewer-readiness artifact_checks_complete disagrees with CSV"
        )
    external_complete = all(
        calculated_by_category[category]["complete"]
        for category in ("evaluation_evidence", "deployment_evidence")
    )
    if (
        summary.get("external_or_deployment_evidence_complete")
        is not external_complete
    ):
        raise ValueError(
            "reviewer-readiness external/deployment completeness disagrees "
            "with CSV"
        )
    if calculated_by_category["evaluation_evidence"]["complete"] is not True:
        raise RuntimeError("evaluation-evidence reviewer checks are incomplete")
    return summary


def _preflight_release() -> dict[str, object]:
    pins = _verify_installed_dependency_pins()
    evidence = _verify_evidence_tree()
    provenance = _load_build_provenance()
    _verify_current_code_manifest(provenance)
    _verify_build_dependency_versions(provenance, pins)
    readiness = _verify_reviewer_readiness()
    return {
        "dependency_pin_count": len(pins),
        "evidence": evidence,
        "reviewer_readiness": readiness,
    }


def _collect_files() -> list[Path]:
    files: set[Path] = set()
    for relative in (
        *ROOT_FILES,
        *CURATED_DOC_FILES,
        *REVIEWER_READINESS_FILES,
    ):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required release file is missing: {relative}")
        files.add(path)

    for relative in TREE_ROOTS:
        root = REPO_ROOT / relative
        if not root.is_dir():
            raise FileNotFoundError(f"required release tree is missing: {relative}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"release tree must not contain symlinks: {path}")
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT)
            if not _is_forbidden(rel):
                files.add(path)

    for relative in EXTRA_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required release evidence is missing: {relative}")
        files.add(path)

    ordered = sorted(files, key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    for path in ordered:
        relative = path.relative_to(REPO_ROOT)
        if _is_forbidden(relative):
            raise ValueError(f"forbidden path entered release allowlist: {relative}")
    return ordered


def _artifact_readme() -> bytes:
    return (
        "# Opportunity-Aware Evidence-Gating Reproducibility Artifact\n\n"
        "This curated archive contains the executable source, configurations, "
        "tests, protocol documentation, canonical aggregate evidence, and exact "
        "evidence/provenance manifests used for the revised manuscript.\n\n"
        "It deliberately excludes raw and processed third-party datasets, VCS "
        "metadata, caches, and superseded outputs. It is integrity-verifiable "
        "but not raw-data self-contained. Obtain COMMECT, LENS, WetLinks, and "
        "other external inputs under their source terms before running the full "
        "canonical rebuild. See `docs/reproducibility_guide.md` and "
        "`docs/THIRD_PARTY_NOTICES.md`.\n\n"
        "No software license is asserted by this archive. License selection, "
        "third-party redistribution confirmation, immutable tagging, and DOI "
        "publication remain author release actions.\n"
    ).encode("utf-8")


def _manifest_payload(files: list[Path], readme: bytes) -> bytes:
    records = [
        FileRecord(
            path=path.relative_to(REPO_ROOT).as_posix(),
            sha256=_sha256_file(path),
            bytes=path.stat().st_size,
        ).__dict__
        for path in files
    ]
    records.append(
        FileRecord(
            path="ARTIFACT_README.md",
            sha256=_sha256_bytes(readme),
            bytes=len(readme),
        ).__dict__
    )
    records.sort(key=lambda record: record["path"])
    payload = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "scope": "source-tests-protocols-and-aggregate-evidence",
        "raw_data_self_contained": False,
        "software_license_included": False,
        "file_count": len(records),
        "files": records,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _tar_directory_info(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _tar_file_info(name: str, size: int, executable: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_archive(output: Path) -> None:
    _preflight_release()
    files = _collect_files()
    readme = _artifact_readme()
    manifest = _manifest_payload(files, readme)
    payloads = {
        "ARTIFACT_README.md": readme,
        "artifact_manifest.json": manifest,
    }
    relative_files = [path.relative_to(REPO_ROOT) for path in files]

    directory_names = {ARCHIVE_ROOT}
    for relative in [*relative_files, *(Path(name) for name in payloads)]:
        parent = PurePosixPath(ARCHIVE_ROOT, relative.as_posix()).parent
        while str(parent) not in {".", "/"}:
            directory_names.add(str(parent))
            if str(parent) == ARCHIVE_ROOT:
                break
            parent = parent.parent

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                compresslevel=9,
                mtime=0,
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.GNU_FORMAT,
                ) as archive:
                    for directory in sorted(
                        directory_names,
                        key=lambda name: (name.count("/"), name),
                    ):
                        archive.addfile(_tar_directory_info(directory))
                    for path in files:
                        relative = path.relative_to(REPO_ROOT).as_posix()
                        member_name = f"{ARCHIVE_ROOT}/{relative}"
                        executable = bool(path.stat().st_mode & 0o111)
                        info = _tar_file_info(
                            member_name,
                            path.stat().st_size,
                            executable,
                        )
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    for relative, payload in sorted(payloads.items()):
                        info = _tar_file_info(
                            f"{ARCHIVE_ROOT}/{relative}",
                            len(payload),
                            executable=False,
                        )
                        archive.addfile(info, BytesIO(payload))
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _safe_member_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive has unsafe member path: {name}")
    if not path.parts or path.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"archive member is outside the single release root: {name}")
    return PurePosixPath(*path.parts[1:])


def _verify_archive(output: Path) -> dict[str, object]:
    seen: set[str] = set()
    extracted: dict[str, bytes] = {}
    with tarfile.open(output, mode="r:gz") as archive:
        for member in archive.getmembers():
            relative = _safe_member_relative(member.name)
            canonical = relative.as_posix().rstrip("/")
            if canonical in seen:
                raise ValueError(f"duplicate archive member: {member.name}")
            seen.add(canonical)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"archive contains unsupported member: {member.name}")
            if member.isfile():
                if _is_forbidden(Path(canonical)):
                    raise ValueError(f"archive contains forbidden path: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"archive file cannot be read: {member.name}")
                extracted[canonical] = stream.read()

    manifest_bytes = extracted.get("artifact_manifest.json")
    if manifest_bytes is None:
        raise ValueError("archive is missing artifact_manifest.json")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    expected_paths = set()
    for record in manifest.get("files", []):
        relative = str(record["path"])
        expected_paths.add(relative)
        payload = extracted.get(relative)
        if payload is None:
            raise ValueError(f"manifest path is absent from archive: {relative}")
        if len(payload) != int(record["bytes"]):
            raise ValueError(f"size mismatch for archive path: {relative}")
        if _sha256_bytes(payload) != record["sha256"]:
            raise ValueError(f"hash mismatch for archive path: {relative}")
    actual_payload_paths = set(extracted).difference({"artifact_manifest.json"})
    if expected_paths != actual_payload_paths:
        raise ValueError("archive payload set differs from artifact manifest")
    return {
        "archive": str(output),
        "archive_sha256": _sha256_file(output),
        "file_count": len(expected_paths),
        "single_top_level_directory": True,
        "raw_data_self_contained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if not args.verify_only:
        _write_archive(output)
    verification = _verify_archive(output)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
