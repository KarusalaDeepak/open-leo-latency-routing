#!/usr/bin/env python3
"""Convert an external LEO latency trace to the repository's canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.loaders import load_compatible_latency_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--bin-seconds", type=int, required=True)
    parser.add_argument(
        "--column-map",
        required=True,
        help='JSON map, e.g. {"relative_path":"path","bin_epoch":"ts","latency_mean_ms":"rtt"}',
    )
    args = parser.parse_args()

    column_map = json.loads(args.column_map)
    frame = load_compatible_latency_trace(
        args.input,
        column_map=column_map,
        dataset_name=args.dataset_name,
        bin_seconds=args.bin_seconds,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"canonical_trace_written={output_path}")
    print(f"rows={len(frame)} paths={frame['relative_path'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
