#!/usr/bin/env python3
"""Inspect the raw dataset tree and print a compact inventory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.loaders import list_dataset_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    files = list_dataset_files(root)
    print(f"data_root={root}")
    print(f"file_count={len(files)}")
    for path in files[:50]:
        print(path)
    if len(files) > 50:
        print("... truncated ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
