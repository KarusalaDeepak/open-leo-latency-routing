#!/usr/bin/env python3
"""Build a ranked manifest of candidate modeling files from the raw dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.inventory import (
    build_candidate_manifest,
    build_inventory,
    write_manifest_csv,
    write_manifest_json,
    write_manifest_markdown,
)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--max-per-location", type=int, default=4)
    parser.add_argument("--max-per-day", type=int, default=2)
    parser.add_argument("--manifest-json-out", default="results/candidate_manifest.json")
    parser.add_argument("--manifest-csv-out", default="results/candidate_manifest.csv")
    parser.add_argument("--manifest-md-out", default="docs/candidate_manifest.md")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    inventory = build_inventory(data_root, max_rows=args.max_rows)
    manifest = build_candidate_manifest(
        inventory,
        top_k=args.top_k,
        max_per_location=args.max_per_location,
        max_per_day=args.max_per_day,
    )

    json_out = _resolve_repo_path(args.manifest_json_out)
    csv_out = _resolve_repo_path(args.manifest_csv_out)
    md_out = _resolve_repo_path(args.manifest_md_out)

    write_manifest_json(manifest, json_out)
    write_manifest_csv(manifest, csv_out)
    write_manifest_markdown(manifest, md_out)

    print(f"manifest_json_written={json_out}")
    print(f"manifest_csv_written={csv_out}")
    print(f"manifest_md_written={md_out}")
    print(f"selected_file_count={len(manifest['selected_files'])}")
    print(f"selection_strategy={manifest['selection_strategy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
