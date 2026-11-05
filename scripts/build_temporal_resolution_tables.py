#!/usr/bin/env python3
"""Re-aggregate the same raw probe sessions at multiple decision resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.aggregations import aggregate_ping_file


DEFAULT_DATA_ROOT = "data/raw/lens_2025_03/LENS-2025-03"
DEFAULT_MANIFEST = "results/candidate_manifest.csv"
DEFAULT_MAX_FILES = 16


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_temporal_resolution_inputs(
    *,
    data_root: Path,
    manifest_path: Path,
    max_files: int,
) -> list[Path]:
    """Resolve the exact raw files selected by the candidate manifest."""

    if max_files <= 0:
        raise ValueError("--max-files must be positive")
    manifest = pd.read_csv(manifest_path)
    if "relative_path" not in manifest:
        raise ValueError("candidate manifest lacks relative_path")
    relative_paths = manifest["relative_path"].head(max_files).tolist()
    if not relative_paths:
        raise ValueError("candidate manifest selects no raw files")
    # Resolve paths only for the containment check.  Keep the manifest-derived
    # path itself lexical so downstream metadata remains relative to the
    # caller's logical data root when that root is a symlink (as in staged
    # artifact rebuilds).
    resolved_root = data_root.resolve()
    target_files: list[Path] = []
    for relative_value in relative_paths:
        relative_path = Path(str(relative_value))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe relative_path in candidate manifest: {relative_value!r}"
            )
        logical_target = data_root / relative_path
        resolved_target = logical_target.resolve()
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(
                f"candidate-manifest path escapes data root: {relative_value!r}"
            )
        target_files.append(logical_target)
    missing = [str(path) for path in target_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing raw files listed by manifest: {missing[:3]}")
    return target_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[5, 10, 30, 60])
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--output-dir", default="data/processed/temporal_resolutions")
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    target_files = resolve_temporal_resolution_inputs(
        data_root=data_root,
        manifest_path=_resolve(args.manifest),
        max_files=args.max_files,
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, int | str]] = []
    for resolution in sorted(set(args.resolutions)):
        if resolution <= 0:
            raise ValueError("all temporal resolutions must be positive")
        rows = []
        for path in target_files:
            rows.extend(
                aggregate_ping_file(
                    path=path,
                    data_root=data_root,
                    bin_seconds=resolution,
                )
            )
        frame = pd.DataFrame(rows)
        output_path = output_dir / f"ping_time_bins_{resolution}s.csv"
        frame.to_csv(output_path, index=False)
        summary_rows.append(
            {
                "resolution_seconds": resolution,
                "rows": len(frame),
                "paths": int(frame["relative_path"].nunique()),
                "locations": int(frame["location"].nunique()),
                "output_path": output_path.relative_to(REPO_ROOT).as_posix(),
            }
        )
        print(f"resolution_table_written={output_path}")

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "temporal_resolution_table_summary.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
