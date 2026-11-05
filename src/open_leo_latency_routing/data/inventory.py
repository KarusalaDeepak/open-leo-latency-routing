"""Dataset inventory and data card helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from open_leo_latency_routing.data.loaders import list_dataset_files


TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".log"}
TABULAR_EXTENSIONS = {".csv", ".tsv"}
LIKELY_TIME_COLUMNS = ("time", "timestamp", "date", "datetime", "epoch")
LIKELY_NETWORK_COLUMNS = ("latency", "delay", "throughput", "bandwidth", "node", "sat", "link", "gateway")
PING_HEADER_PATTERN = re.compile(
    r"^PING\s+(?P<target>\S+)\((?P<target_ip>[^)]+)\)\s+from\s+(?P<source_ip>\S+)\s+(?P<interface>\S+):"
)
PING_SAMPLE_PATTERN = re.compile(
    r"^\[(?P<epoch>[0-9.]+)\]\s+\d+\s+bytes\s+from\s+(?P<reply_ip>\S+):\s+"
    r"icmp_seq=(?P<icmp_seq>\d+)\s+ttl=(?P<ttl>\d+)\s+time=(?P<latency_ms>[0-9.]+)\s+ms$"
)
PING_FILENAME_PATTERN = re.compile(
    r"^ping-(?P<target_hint>.+)-(?P<probe_interval>\d+ms)-(?P<window_duration>\d+h)-"
    r"(?P<window_start>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.txt$"
)


@dataclass
class FileSummary:
    """Compact summary for one dataset file."""

    relative_path: str
    suffix: str
    size_bytes: int
    sampled: bool
    columns: list[str]
    row_sample_count: int
    notes: str


def _human_bytes(size_bytes: int) -> str:
    """Render a file size in a compact human-readable form."""
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _sample_tabular_file(path: Path, max_rows: int) -> tuple[list[str], int, str]:
    """Read a small sample from CSV/TSV files for schema hints."""
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=separator, nrows=max_rows)
    columns = [str(column) for column in frame.columns.tolist()]
    return columns, len(frame.index), ""


def _sample_ping_log(path: Path, max_rows: int) -> tuple[list[str], int, str]:
    """Extract schema-like fields from line-oriented ping logs."""
    columns = ["epoch", "icmp_seq", "ttl", "latency_ms", "reply_ip"]
    sample_count = 0
    header_seen = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if not header_seen and PING_HEADER_PATTERN.match(text):
                header_seen = True
                continue
            if PING_SAMPLE_PATTERN.match(text):
                sample_count += 1
                if sample_count >= max_rows:
                    break

    notes = "ping log parsed" if sample_count > 0 else "text file did not match ping log pattern"
    return columns if sample_count > 0 else [], sample_count, notes


def _summarize_file(path: Path, data_root: Path, max_rows: int) -> FileSummary:
    """Build a summary for one file, sampling schema when appropriate."""
    suffix = path.suffix.lower()
    columns: list[str] = []
    row_sample_count = 0
    sampled = False
    notes = ""

    if suffix in TABULAR_EXTENSIONS:
        try:
            columns, row_sample_count, notes = _sample_tabular_file(path, max_rows)
            sampled = True
        except Exception as exc:  # pragma: no cover - defensive path for unknown files
            notes = f"tabular sample failed: {exc}"
    elif suffix == ".txt":
        try:
            columns, row_sample_count, notes = _sample_ping_log(path, max_rows)
            sampled = row_sample_count > 0
        except Exception as exc:  # pragma: no cover - defensive path for unknown files
            notes = f"text sample failed: {exc}"
    elif suffix in TEXT_EXTENSIONS:
        notes = "text-like file; schema sampling skipped"
    else:
        notes = "non-tabular or unsupported extension"

    return FileSummary(
        relative_path=str(path.relative_to(data_root)),
        suffix=suffix or "<no_ext>",
        size_bytes=path.stat().st_size,
        sampled=sampled,
        columns=columns,
        row_sample_count=row_sample_count,
        notes=notes,
    )


def _extract_path_metadata(relative_path: str) -> dict[str, str]:
    """Recover lightweight metadata from the LENS directory layout and filename."""
    parts = Path(relative_path).parts
    metadata = {
        "measurement_family": parts[0] if len(parts) > 0 else "",
        "path_state": parts[1] if len(parts) > 1 else "",
        "location": parts[2] if len(parts) > 2 else "",
        "session_date": parts[3] if len(parts) > 3 else "",
        "target_hint": "",
        "probe_interval": "",
        "window_duration": "",
        "window_start": "",
    }

    match = PING_FILENAME_PATTERN.match(Path(relative_path).name)
    if match:
        metadata.update(match.groupdict())
    return metadata


def build_inventory(data_root: str | Path, max_rows: int = 25) -> dict[str, Any]:
    """Scan the dataset root and return a structured inventory."""
    root = Path(data_root)
    files = list_dataset_files(root)
    summaries = [_summarize_file(path, root, max_rows) for path in files]
    suffix_counter = Counter(summary.suffix for summary in summaries)
    total_size = sum(summary.size_bytes for summary in summaries)

    return {
        "data_root": str(root),
        "file_count": len(summaries),
        "total_size_bytes": total_size,
        "total_size_human": _human_bytes(total_size),
        "suffix_counts": dict(sorted(suffix_counter.items())),
        "files": [asdict(summary) for summary in summaries],
    }


def write_inventory_json(inventory: dict[str, Any], output_path: str | Path) -> None:
    """Write the inventory as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


def write_data_card_markdown(inventory: dict[str, Any], output_path: str | Path, sample_limit: int = 20) -> None:
    """Write a compact Markdown data card from the inventory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Dataset Data Card",
        "",
        f"- `data_root`: `{inventory['data_root']}`",
        f"- `file_count`: `{inventory['file_count']}`",
        f"- `total_size`: `{inventory['total_size_human']}`",
        "",
        "## File Types",
        "",
        "| Suffix | Count |",
        "| --- | ---: |",
    ]

    for suffix, count in inventory["suffix_counts"].items():
        lines.append(f"| `{suffix}` | {count} |")

    lines.extend(
        [
            "",
            "## Sampled Files",
            "",
            "| Relative Path | Size | Sampled | Columns | Notes |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )

    for file_info in inventory["files"][:sample_limit]:
        columns = ", ".join(file_info["columns"][:8]) if file_info["columns"] else "-"
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                file_info["relative_path"],
                _human_bytes(file_info["size_bytes"]),
                "yes" if file_info["sampled"] else "no",
                columns,
                file_info["notes"] or "-",
            )
        )

    if inventory["file_count"] > sample_limit:
        lines.extend(
            [
                "",
                f"_Only the first {sample_limit} files are shown above. Full details are stored in the inventory JSON._",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_manifest(
    inventory: dict[str, Any],
    top_k: int = 25,
    max_per_location: int = 4,
    max_per_day: int = 2,
) -> dict[str, Any]:
    """Select and rank likely modeling files from the dataset inventory."""
    ranked_files: list[dict[str, Any]] = []

    for file_info in inventory["files"]:
        score = 0
        suffix = file_info["suffix"]
        columns = [str(column).lower() for column in file_info.get("columns", [])]
        size_bytes = int(file_info["size_bytes"])
        metadata = _extract_path_metadata(file_info["relative_path"])

        if any(part.startswith(".") for part in Path(file_info["relative_path"]).parts):
            continue
        if suffix not in TABULAR_EXTENSIONS and not file_info.get("sampled"):
            continue

        if suffix in TABULAR_EXTENSIONS:
            score += 40
        elif suffix == ".txt" and file_info.get("sampled"):
            score += 35
        if size_bytes > 0:
            score += min(30, int(size_bytes / (1024 * 1024 * 50)))
        if any(token in column for column in columns for token in LIKELY_TIME_COLUMNS):
            score += 20
        if any(token in column for column in columns for token in LIKELY_NETWORK_COLUMNS):
            score += 20
        if file_info.get("sampled"):
            score += 10
        if metadata["measurement_family"]:
            score += 5
        if metadata["path_state"] == "active":
            score += 5
        if metadata["window_duration"] == "2h":
            score += 3

        file_copy = dict(file_info)
        file_copy["candidate_score"] = score
        file_copy.update(metadata)
        ranked_files.append(file_copy)

    ranked_files.sort(
        key=lambda item: (
            -item["candidate_score"],
            -int(item["size_bytes"]),
            item["relative_path"],
        )
    )

    grouped_candidates: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for file_info in ranked_files:
        group_key = (
            file_info.get("location", "") or "<unknown>",
            file_info.get("path_state", "") or "<unknown>",
            file_info.get("window_duration", "") or "<unknown>",
        )
        grouped_candidates[group_key].append(file_info)

    ordered_group_keys = sorted(
        grouped_candidates,
        key=lambda key: (
            -grouped_candidates[key][0]["candidate_score"],
            key[0],
            key[1],
            key[2],
        ),
    )

    selected_files: list[dict[str, Any]] = []
    location_counts: Counter[str] = Counter()
    day_counts: Counter[tuple[str, str]] = Counter()
    exhausted_groups: set[tuple[str, str, str]] = set()

    while len(selected_files) < top_k and len(exhausted_groups) < len(ordered_group_keys):
        made_progress = False
        for group_key in ordered_group_keys:
            if group_key in exhausted_groups:
                continue
            candidate_list = grouped_candidates[group_key]
            while candidate_list:
                candidate = candidate_list[0]
                location = candidate.get("location", "") or "<unknown>"
                session_date = candidate.get("session_date", "") or "<unknown>"
                if location_counts[location] >= max_per_location:
                    candidate_list.pop(0)
                    continue
                if day_counts[(location, session_date)] >= max_per_day:
                    candidate_list.pop(0)
                    continue

                selected_files.append(candidate)
                location_counts[location] += 1
                day_counts[(location, session_date)] += 1
                candidate_list.pop(0)
                made_progress = True
                break
            if not candidate_list:
                exhausted_groups.add(group_key)
            if len(selected_files) >= top_k:
                break
        if not made_progress:
            break

    return {
        "data_root": inventory["data_root"],
        "candidate_count": len(ranked_files),
        "top_k": top_k,
        "selection_strategy": "diversity_round_robin",
        "max_per_location": max_per_location,
        "max_per_day": max_per_day,
        "selected_files": selected_files,
    }


def write_manifest_json(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write the selected manifest as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_manifest_csv(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write the selected manifest as a flat CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_info in manifest["selected_files"]:
        rows.append(
            {
                "relative_path": file_info["relative_path"],
                "suffix": file_info["suffix"],
                "size_bytes": file_info["size_bytes"],
                "candidate_score": file_info["candidate_score"],
                "sampled": file_info["sampled"],
                "row_sample_count": file_info["row_sample_count"],
                "measurement_family": file_info.get("measurement_family", ""),
                "path_state": file_info.get("path_state", ""),
                "location": file_info.get("location", ""),
                "session_date": file_info.get("session_date", ""),
                "window_duration": file_info.get("window_duration", ""),
                "columns": "|".join(file_info.get("columns", [])),
                "notes": file_info["notes"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_manifest_markdown(manifest: dict[str, Any], output_path: str | Path) -> None:
    """Write a short Markdown summary of top candidate modeling files."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Candidate File Manifest",
        "",
        f"- `data_root`: `{manifest['data_root']}`",
        f"- `candidate_count`: `{manifest['candidate_count']}`",
        f"- `top_k`: `{manifest['top_k']}`",
        f"- `selection_strategy`: `{manifest.get('selection_strategy', 'rank_only')}`",
        f"- `max_per_location`: `{manifest.get('max_per_location', '-')}`",
        f"- `max_per_day`: `{manifest.get('max_per_day', '-')}`",
        "",
        "| Rank | Relative Path | Location | State | Window | Score | Size | Columns |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]

    for index, file_info in enumerate(manifest["selected_files"], start=1):
        columns = ", ".join(file_info.get("columns", [])[:8]) if file_info.get("columns") else "-"
        lines.append(
            f"| {index} | `{file_info['relative_path']}` | {file_info.get('location', '-') or '-'} | "
            f"{file_info.get('path_state', '-') or '-'} | {file_info.get('window_duration', '-') or '-'} | "
            f"{file_info['candidate_score']} | { _human_bytes(int(file_info['size_bytes'])) } | {columns} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
