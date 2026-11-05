#!/usr/bin/env python3
"""Run multi-path evaluation across regenerated simulator seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
PROPOSED_POLICIES = (
    "qos_shielded_operational_selector",
    "validation_gated_qos_selector",
)
COMPARATORS = (
    "reactive_greedy",
    "predictive_greedy",
    "predictive_graph_greedy",
    "ensemble_uncertainty_selector",
)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original row order."""

    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Calculate a small-sample Student-t interval for independent seeds."""

    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    return mean, mean - critical * standard_error, mean + critical * standard_error


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(2026, 2036)),
    )
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default="results/transactions_seed_matrix",
    )
    parser.add_argument(
        "--reuse-runs",
        action="store_true",
        help="Rebuild aggregate statistics from completed per-seed runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed seeds and run only missing per-seed outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of independent seed subprocesses to run concurrently.",
    )
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    runs_dir = output_dir / "runs"
    traces_dir = output_dir / "traces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    def run_seed(seed: int) -> None:
        trace_path = traces_dir / f"orbital_multipath_seed_{seed}.csv"
        run_dir = runs_dir / f"seed_{seed}"
        completed = (run_dir / "independent_policy_summary.csv").exists()
        if args.reuse_runs:
            if not completed:
                raise FileNotFoundError(
                    f"missing completed seed run for --reuse-runs: {run_dir}"
                )
            return
        if args.resume and completed:
            return
        if not completed or not args.resume:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "generate_physics_informed_multipath_trace.py"),
                    "--output",
                    str(trace_path),
                    "--duration-hours",
                    str(args.duration_hours),
                    "--seed",
                    str(seed),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "run_independent_multipath_validation.py"),
                    "--trace",
                    str(trace_path),
                    "--output-dir",
                    str(run_dir),
                ],
                check=True,
            )

    workers = max(1, int(args.max_workers))
    if workers == 1:
        for seed in args.seeds:
            run_seed(seed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_seed, seed): seed for seed in args.seeds}
            for future in as_completed(futures):
                seed = futures[future]
                future.result()
                print(f"seed_completed={seed}")

    seed_rows = []
    gate_grid_points_by_seed: dict[str, list[int]] = {}
    gate_selection_reason_by_seed: dict[str, str] = {}
    for seed in args.seeds:
        run_dir = runs_dir / f"seed_{seed}"
        summary = pd.read_csv(run_dir / "independent_policy_summary.csv")
        summary["seed"] = seed
        seed_rows.append(summary)

        run_metadata_path = run_dir / "run_metadata.json"
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        calibration = run_metadata["disagreement_aware_config"][
            "temporal_calibration"
        ]
        gate_evidence = json.loads(calibration["gate_selection_evidence_json"])
        grid_points = sorted(
            {
                int(row["cvar_grid_points"])
                for row in gate_evidence
                if "cvar_grid_points" in row
            }
        )
        if not grid_points:
            raise ValueError(
                f"missing cvar_grid_points in gate evidence for seed {seed}"
            )
        gate_grid_points_by_seed[str(seed)] = grid_points
        gate_selection_reason_by_seed[str(seed)] = str(
            calibration["gate_selection_reason"]
        )

    seed_results = pd.concat(seed_rows, ignore_index=True)
    seed_results.to_csv(output_dir / "per_seed_policy_results.csv", index=False)
    metrics = [
        "mean_realized_latency_ms",
        "mean_decision_gap_ms",
        "success_rate_under_60ms",
        "p95_realized_latency_ms",
    ]
    aggregate_rows = []
    grouped_columns = ["scenario_name", "policy_name"]
    for group_key, frame in seed_results.groupby(grouped_columns, sort=False):
        for metric in metrics:
            values = frame[metric].dropna().to_numpy(dtype=float)
            mean, ci_lower, ci_upper = _mean_ci95(values)
            if metric == "success_rate_under_60ms":
                ci_lower = max(0.0, ci_lower)
                ci_upper = min(1.0, ci_upper)
            aggregate_rows.append(
                {
                    "scenario_name": group_key[0],
                    "policy_name": group_key[1],
                    "metric_name": metric,
                    "seed_count": len(values),
                    "mean_value": mean,
                    "standard_deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                }
            )
    pd.DataFrame(aggregate_rows).to_csv(
        output_dir / "multi_seed_policy_summary.csv",
        index=False,
    )

    delta_rows = []
    for scenario_name, scenario_frame in seed_results.groupby(
        "scenario_name",
        sort=False,
    ):
        for proposed_policy in PROPOSED_POLICIES:
            proposed = scenario_frame[
                scenario_frame["policy_name"].eq(proposed_policy)
            ].set_index("seed")
            if proposed.empty:
                continue
            for comparator in COMPARATORS:
                baseline = scenario_frame[
                    scenario_frame["policy_name"].eq(comparator)
                ].set_index("seed")
                shared_seeds = proposed.index.intersection(baseline.index)
                for metric in metrics:
                    deltas = (
                        proposed.loc[shared_seeds, metric]
                        - baseline.loc[shared_seeds, metric]
                    )
                    favorable_deltas = (
                        deltas.to_numpy(dtype=float)
                        if metric == "success_rate_under_60ms"
                        else -deltas.to_numpy(dtype=float)
                    )
                    mean_favorable, ci_lower, ci_upper = _mean_ci95(
                        favorable_deltas
                    )
                    tolerance = 1e-12
                    tie_mask = np.abs(favorable_deltas) <= tolerance
                    if len(favorable_deltas) > 0 and np.any(
                        np.abs(favorable_deltas) > tolerance
                    ):
                        p_value = float(
                            wilcoxon(favorable_deltas, alternative="two-sided").pvalue
                        )
                    else:
                        p_value = 1.0
                    delta_rows.append(
                        {
                            "scenario_name": scenario_name,
                            "proposed_policy": proposed_policy,
                            "comparator": comparator,
                            "metric_name": metric,
                            "seed_count": len(deltas),
                            "mean_delta_proposed_minus_comparator": float(
                                deltas.mean()
                            ),
                            "mean_favorable_delta": mean_favorable,
                            "favorable_delta_ci95_lower": ci_lower,
                            "favorable_delta_ci95_upper": ci_upper,
                            "win_rate": float(
                                (favorable_deltas > tolerance).mean()
                            ),
                            "tie_rate": float(tie_mask.mean()),
                            "loss_rate": float(
                                (favorable_deltas < -tolerance).mean()
                            ),
                            "paired_standardized_effect": (
                                float(
                                    favorable_deltas.mean()
                                    / favorable_deltas.std(ddof=1)
                                )
                                if len(favorable_deltas) > 1
                                and favorable_deltas.std(ddof=1) > tolerance
                                else 0.0
                            ),
                            "wilcoxon_p_value": p_value,
                        }
                    )
    delta_table = pd.DataFrame(delta_rows)
    # Each reported metric is a separate inferential family. Correcting all
    # heterogeneous outcomes together would mix distinct scientific claims
    # and be needlessly conservative.
    delta_table["holm_adjusted_p_value"] = np.nan
    for _, family in delta_table.groupby(
        ["proposed_policy", "metric_name"], sort=False
    ):
        delta_table.loc[family.index, "holm_adjusted_p_value"] = _holm_adjust(
            family["wilcoxon_p_value"].tolist()
        )
    delta_table.to_csv(
        output_dir / "multi_seed_pairwise_deltas.csv",
        index=False,
    )
    config_path = REPO_ROOT / "configs" / "experiment.yaml"
    common_grid_points = sorted(
        {
            grid_point
            for seed_grid_points in gate_grid_points_by_seed.values()
            for grid_point in seed_grid_points
        }
    )
    metadata = {
        "dataset_name": "physics_informed_orbital_multipath",
        "source_type": "physics-informed concurrent-path simulator",
        "is_measured_dataset": False,
        "is_simulated_dataset": True,
        "seeds": args.seeds,
        "duration_hours_per_seed": args.duration_hours,
        "trace_regenerated_per_seed": True,
        "model_retraining_scope": "retrained from scratch within each seed trace",
        "cross_seed_model_reuse": False,
        "within_seed_partition_protocol": (
            "four disjoint train, calibration, policy-selection, and test blocks"
        ),
        "policy_admission_outcomes": "policy-selection block only; test outcomes excluded",
        "training_protocol": (
            "retrained independently within each simulated seed trace using "
            "four disjoint train/calibration/policy-selection/test blocks"
        ),
        "seed_count": len(args.seeds),
        "max_workers": workers,
        "seed_interval_method": "Student-t 95% confidence interval",
        "paired_test": "two-sided Wilcoxon signed-rank with Holm correction within each metric family",
        "tie_tolerance": 1e-12,
        "experiment_config_path": str(config_path.relative_to(REPO_ROOT)),
        "experiment_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "risk_control_gate_evidence": {
            "common_cvar_grid_points": common_grid_points,
            "per_seed_cvar_grid_points": gate_grid_points_by_seed,
            "per_seed_selection_reason": gate_selection_reason_by_seed,
            "all_seed_gate_evidence_present": (
                len(gate_grid_points_by_seed) == len(args.seeds)
            ),
        },
    }
    (output_dir / "seed_matrix_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"seed_matrix_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
