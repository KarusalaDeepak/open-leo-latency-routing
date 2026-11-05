#!/usr/bin/env python3
"""Stream the independent Starlink IRTT archive into canonical time bins."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
import statistics
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IRTT_PATTERN = re.compile(
    rb"^(?P<epoch>\d+\.\d+).*?rtt=(?P<rtt>[\d.]+)ms "
    rb"rd=(?P<rd>[\d.]+)ms sd=(?P<sd>[\d.]+)ms"
)
EXPECTED_SHA256 = "eacb5d3182f89f2b5a0cb8abfe1ff379d44cddeb7ecc7718a6e07a0f9826d338"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default="data/raw/external_starlink_irtt.zip",
    )
    parser.add_argument("--bin-seconds", type=int, default=10)
    parser.add_argument(
        "--max-observations-per-session",
        type=int,
        default=2_000_000,
        help="Bounded reproducible subset; use 0 for every packet.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/external_starlink_irtt_10s.csv",
    )
    args = parser.parse_args()

    archive_path = _resolve(args.archive)
    output_path = _resolve(args.output)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith("/irtt.txt") and not name.startswith("testing-")
        )
        for member in members:
            session_name = member.split("/", 1)[0]
            buckets: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: {"rtt": [], "uplink": [], "downlink": []}
            )
            observations = 0
            with archive.open(member) as handle:
                for line in handle:
                    match = IRTT_PATTERN.match(line)
                    if not match:
                        continue
                    epoch = float(match.group("epoch"))
                    bucket_epoch = int(epoch // args.bin_seconds) * args.bin_seconds
                    bucket = buckets[bucket_epoch]
                    bucket["rtt"].append(float(match.group("rtt")))
                    bucket["uplink"].append(float(match.group("sd")))
                    bucket["downlink"].append(float(match.group("rd")))
                    observations += 1
                    if (
                        args.max_observations_per_session > 0
                        and observations >= args.max_observations_per_session
                    ):
                        break
            for bucket_epoch, values in sorted(buckets.items()):
                rtt = values["rtt"]
                rows.append(
                    {
                        "relative_path": session_name,
                        "measurement_family": "independent_starlink_irtt",
                        "path_state": "available",
                        "location": "independent_starlink_testbed",
                        "session_date": datetime.utcfromtimestamp(
                            bucket_epoch
                        ).date().isoformat(),
                        "target_hint": "130.225.37.208",
                        "probe_interval": "irtt",
                        "window_duration": "bounded_trace",
                        "bin_seconds": args.bin_seconds,
                        "bin_epoch": bucket_epoch,
                        "bin_start_utc": datetime.utcfromtimestamp(
                            bucket_epoch
                        ).isoformat(),
                        "observed_replies": len(rtt),
                        "latency_mean_ms": statistics.mean(rtt),
                        "latency_std_ms": (
                            statistics.pstdev(rtt) if len(rtt) > 1 else 0.0
                        ),
                        "latency_min_ms": min(rtt),
                        "latency_max_ms": max(rtt),
                        "uplink_mean_ms": statistics.mean(values["uplink"]),
                        "downlink_mean_ms": statistics.mean(values["downlink"]),
                    }
                )
            print(
                f"parsed_session={session_name} observations={observations} "
                f"bins={len(buckets)}"
            )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata = pd.DataFrame(
        [
            {
                "dataset": "Statistical Characterization and Prediction of E2E Latency over LEO Satellite Networks",
                "doi": "10.17632/479v4mym7j.2",
                "license": "CC BY 4.0",
                "archive_sha256": EXPECTED_SHA256,
                "sessions": frame["relative_path"].nunique(),
                "rows": len(frame),
                "bin_seconds": args.bin_seconds,
                "max_observations_per_session": args.max_observations_per_session,
            }
        ]
    )
    metadata.to_csv(output_path.with_name("external_starlink_irtt_metadata.csv"), index=False)
    print(f"external_irtt_table_written={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
