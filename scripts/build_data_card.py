#!/usr/bin/env python3
"""Build a machine-readable inventory and Markdown data card for the raw dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.inventory import build_inventory, write_data_card_markdown, write_inventory_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--max-rows", type=int, default=25)
    parser.add_argument(
        "--inventory-out",
        default="results/dataset_inventory.json",
        help="Path for structured JSON inventory output.",
    )
    parser.add_argument(
        "--card-out",
        default="docs/dataset_data_card.md",
        help="Path for Markdown data card output.",
    )
    args = parser.parse_args()

    inventory_out = Path(args.inventory_out)
    if not inventory_out.is_absolute():
        inventory_out = REPO_ROOT / inventory_out

    card_out = Path(args.card_out)
    if not card_out.is_absolute():
        card_out = REPO_ROOT / card_out

    inventory = build_inventory(args.data_root, max_rows=args.max_rows)
    write_inventory_json(inventory, inventory_out)
    write_data_card_markdown(inventory, card_out)

    print(f"inventory_written={inventory_out}")
    print(f"data_card_written={card_out}")
    print(f"file_count={inventory['file_count']}")
    print(f"total_size={inventory['total_size_human']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
