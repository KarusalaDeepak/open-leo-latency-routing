#!/usr/bin/env python3
"""Audit inter-path timestamp skew in the frozen 10-second COMMECT replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RESULTS = REPO_ROOT / "results/commect_rolling_active_opportunity"
POLICIES = {
    "reactive_greedy": "reactive_baseline",
    "qos_shielded_operational_selector": "predictive_shield",
    "qos_filter_then_context_selector": "filter_then_rank_context",
    "qos_filter_then_ensemble_selector": "filter_then_rank_ensemble",
    "validation_gated_qos_selector": "evidence_gated_policy",
}


def build_alignment_audit(
    prediction_path: Path,
    decision_path: Path,
    *,
    threshold_ms: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions = pd.read_csv(prediction_path)
    decisions = pd.read_csv(decision_path)
    keys = ["rolling_fold", "bin_epoch"]
    skew = (
        predictions.groupby(keys, as_index=False)["inter_path_skew_ms"]
        .max()
        .rename(columns={"bin_epoch": "decision_bin_epoch"})
    )
    if len(skew) != 289:
        raise AssertionError(f"expected 289 frozen decision epochs, observed {len(skew)}")
    joined = decisions.merge(
        skew,
        on=["rolling_fold", "decision_bin_epoch"],
        how="left",
        validate="many_to_one",
    )
    if joined["inter_path_skew_ms"].isna().any():
        raise AssertionError("one or more decision epochs lack a skew record")

    views = {
        "all_evaluable_10s": joined,
        "skew_at_most_100ms": joined[
            joined["inter_path_skew_ms"].le(threshold_ms)
        ],
    }
    rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    for view_name, frame in views.items():
        epoch_count = int(frame["decision_bin_epoch"].nunique())
        epoch_rows.append(
            {
                "alignment_view": view_name,
                "skew_threshold_ms": (
                    "" if view_name == "all_evaluable_10s" else threshold_ms
                ),
                "decision_epoch_count": epoch_count,
                "excluded_from_289": 289 - epoch_count,
            }
        )
        for source_name, display_name in POLICIES.items():
            policy = frame[frame["policy_name"].eq(source_name)]
            if len(policy) != epoch_count:
                raise AssertionError(
                    f"{source_name} has {len(policy)} rows in {view_name}, "
                    f"expected {epoch_count}"
                )
            latency = policy["realized_next_latency_ms"].astype(float)
            rows.append(
                {
                    "alignment_view": view_name,
                    "skew_threshold_ms": (
                        "" if view_name == "all_evaluable_10s" else threshold_ms
                    ),
                    "policy": display_name,
                    "decision_count": epoch_count,
                    "success_count": int(policy["success_under_budget"].sum()),
                    "success_rate": float(policy["success_under_budget"].mean()),
                    "mean_rtt_ms": float(latency.mean()),
                    "empirical_cvar95_ms": empirical_upper_cvar(
                        latency.to_numpy(),
                        0.95,
                    ),
                }
            )
    metadata = {
        "analysis": "post-specified 10-second inter-path timestamp-skew sensitivity",
        "threshold_ms": threshold_ms,
        "skew_field": "maximum per-epoch inter_path_skew_ms across candidate rows",
        "source_semantics": (
            "the field is the spread of per-path median source timestamps; it is "
            "not verified clock error or packet-level probe matching"
        ),
        "decision_scope": "frozen five-fold 10-second locked-test decisions",
        "claim_boundary": (
            "subset sensitivity within one COMMECT drive; excluded epochs are not "
            "reclassified as successful and no candidate is retrained"
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(epoch_rows), metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=PRIMARY_RESULTS / "rolling_candidate_predictions.csv",
    )
    parser.add_argument(
        "--decision-path",
        type=Path,
        default=PRIMARY_RESULTS / "rolling_policy_decisions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/commect_rolling_10s_alignment",
    )
    parser.add_argument("--threshold-ms", type=float, default=100.0)
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, epochs, metadata = build_alignment_audit(
        args.prediction_path,
        args.decision_path,
        threshold_ms=args.threshold_ms,
    )
    summary.to_csv(output_dir / "alignment_policy_summary.csv", index=False)
    epochs.to_csv(output_dir / "alignment_epoch_summary.csv", index=False)
    (output_dir / "alignment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
