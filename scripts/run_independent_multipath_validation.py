#!/usr/bin/env python3
"""Run full policy evaluation on a separately generated multi-path trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--trace",
        default="data/processed/physics_informed_orbital_multipath_5s.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/transactions_orbital_validation",
    )
    parser.add_argument("--horizon-seconds", type=int, default=5)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Rebuild transfer/event summaries from an existing completed run.",
    )
    args = parser.parse_args()

    trace_path = _resolve(args.trace)
    output_dir = _resolve(args.output_dir)
    if not trace_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "generate_physics_informed_multipath_trace.py"),
                "--output",
                str(trace_path),
            ],
            check=True,
        )

    if not args.reuse_results:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_service_path_experiments.py"),
                "--config",
                str(_resolve(args.config)),
                "--time-bins",
                str(trace_path),
                "--output-dir",
                str(output_dir),
                "--horizon-seconds",
                str(args.horizon_seconds),
                "--holdout-count",
                "1",
            ],
            check=True,
        )
    elif not (output_dir / "policy_summary.csv").exists():
        raise FileNotFoundError(
            "--reuse-results requires an existing completed policy run"
        )

    summary = pd.read_csv(output_dir / "policy_summary.csv")
    operational = summary[
        summary["scenario_name"].isin(
            [
                "session_holdout",
                "temporal_holdout",
                "site_holdout",
                "operational_mild",
                "operational_moderate",
                "operational_severe",
            ]
        )
    ].copy()
    operational.to_csv(
        output_dir / "independent_policy_summary.csv",
        index=False,
    )
    decisions = pd.read_csv(output_dir / "policy_decisions.csv")
    event_rows = []
    for policy_name, policy_frame in decisions.groupby("policy_name", sort=False):
        for event_name, event_column in (
            ("handover", "chosen_handover_event"),
            ("attenuation", "chosen_attenuation_event"),
        ):
            if event_column not in policy_frame:
                continue
            for event_active, event_frame in policy_frame.groupby(event_column):
                event_rows.append(
                    {
                        "policy_name": policy_name,
                        "event_type": event_name,
                        "event_active": int(event_active),
                        "decision_count": len(event_frame),
                        "mean_realized_latency_ms": float(
                            event_frame["realized_next_latency_ms"].mean()
                        ),
                        "success_rate_under_60ms": float(
                            event_frame["success_under_budget"].mean()
                        ),
                        "mean_decision_gap_ms": float(
                            event_frame["decision_gap_ms"].mean()
                        ),
                    }
                )
    pd.DataFrame(event_rows).to_csv(
        output_dir / "physical_event_policy_summary.csv",
        index=False,
    )
    candidates = pd.read_csv(
        output_dir / "candidate_predictions.csv",
        low_memory=False,
    )
    avoidance_rows = []
    for scenario_name, scenario_candidates in candidates.groupby(
        "scenario_name",
        sort=False,
    ):
        scenario_decisions = decisions[
            decisions["scenario_name"].eq(scenario_name)
        ]
        for event_name, event_column in (
            ("handover", "handover_event"),
            ("attenuation", "attenuation_event"),
        ):
            if event_column not in scenario_candidates:
                continue
            event_by_window = scenario_candidates.groupby("session_bin_index")[
                event_column
            ].agg(["min", "max"])
            avoidable_windows = event_by_window[
                event_by_window["min"].eq(0) & event_by_window["max"].eq(1)
            ].index
            if len(avoidable_windows) == 0:
                continue
            event_lookup = scenario_candidates.set_index(
                ["session_bin_index", "relative_path"]
            )[event_column]
            for policy_name, policy_decisions in scenario_decisions.groupby(
                "policy_name",
                sort=False,
            ):
                exposed = 0
                evaluated = 0
                for row in policy_decisions[
                    policy_decisions["session_bin_index"].isin(
                        avoidable_windows
                    )
                ].itertuples(index=False):
                    key = (
                        int(row.session_bin_index),
                        row.chosen_relative_path,
                    )
                    if key not in event_lookup.index:
                        continue
                    evaluated += 1
                    exposed += int(event_lookup.loc[key] > 0)
                avoidance_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "policy_name": policy_name,
                        "event_type": event_name,
                        "avoidable_event_windows": evaluated,
                        "selected_affected_path_count": exposed,
                        "affected_path_selection_rate": (
                            exposed / evaluated if evaluated else float("nan")
                        ),
                        "affected_path_avoidance_rate": (
                            1.0 - exposed / evaluated
                            if evaluated
                            else float("nan")
                        ),
                    }
                )
    avoidance = pd.DataFrame(avoidance_rows)
    avoidance.to_csv(
        output_dir / "physical_event_avoidance_summary.csv",
        index=False,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_policies = [
        "predictive_greedy",
        "predictive_graph_greedy",
        "ensemble_uncertainty_selector",
        "calibrated_operational_selector",
    ]
    transfer_plot = operational[
        operational["policy_name"].isin(plot_policies)
    ].pivot(
        index="scenario_name",
        columns="policy_name",
        values="success_rate_under_60ms",
    )
    transfer_plot = transfer_plot.rename(
        index={
            "session_holdout": "Path holdout",
            "temporal_holdout": "Temporal holdout",
            "site_holdout": "Gateway holdout",
            "operational_mild": "Injected mild",
            "operational_moderate": "Injected moderate",
            "operational_severe": "Injected severe",
        },
        columns={
            "predictive_greedy": "Temporal",
            "predictive_graph_greedy": "Graph context",
            "ensemble_uncertainty_selector": "Ensemble uncertainty",
            "calibrated_operational_selector": "Calibrated operational",
        },
    )
    ax = transfer_plot.plot(kind="bar", figsize=(10.5, 4.8), width=0.82)
    ax.set_title("Policy evaluation on a physics-informed multi-path trace")
    ax.set_xlabel("Evaluation condition")
    ax.set_ylabel("Success rate under 60 ms")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(
        figures_dir / "independent_policy_success.png",
        dpi=300,
        bbox_inches="tight",
    )
    ax.figure.savefig(
        figures_dir / "independent_policy_success.pdf",
        bbox_inches="tight",
    )
    plt.close(ax.figure)

    base_avoidance = avoidance[
        avoidance["scenario_name"].eq("session_holdout")
        & avoidance["policy_name"].isin(plot_policies)
    ].pivot(
        index="event_type",
        columns="policy_name",
        values="affected_path_avoidance_rate",
    )
    base_avoidance = base_avoidance.rename(
        index={"handover": "Handover", "attenuation": "Attenuation"},
        columns={
            "predictive_greedy": "Temporal",
            "predictive_graph_greedy": "Graph context",
            "ensemble_uncertainty_selector": "Ensemble uncertainty",
            "calibrated_operational_selector": "Calibrated operational",
        },
    )
    ax = base_avoidance.plot(kind="bar", figsize=(9.5, 4.6), width=0.80)
    ax.set_title("Avoidance of affected paths when a healthy alternative exists")
    ax.set_xlabel("Physical event")
    ax.set_ylabel("Affected-path avoidance rate")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Policy", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(
        figures_dir / "physical_event_avoidance.png",
        dpi=300,
        bbox_inches="tight",
    )
    ax.figure.savefig(
        figures_dir / "physical_event_avoidance.pdf",
        bbox_inches="tight",
    )
    plt.close(ax.figure)

    trace_metadata_path = trace_path.with_suffix(".metadata.json")
    trace_metadata = (
        json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        if trace_metadata_path.exists()
        else {}
    )
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    validation_metadata = {
        "trace_metadata": trace_metadata,
        "policy_level_evaluation": True,
        "concurrent_alternative_paths": True,
        "forecast_horizon_seconds": args.horizon_seconds,
        "exact_horizon_audit": run_metadata.get("exact_horizon_audit", {}),
        "training_protocol": "experts retrained within the same generated trace",
        "zero_shot_transfer": False,
        "valid_claim": (
            "complete path-selection evaluation on a separately generated "
            "physics-informed concurrent-path simulator trace"
        ),
        "invalid_claim": (
            "validation on an independent measured multi-path dataset"
        ),
        "legacy_naming_note": (
            "output filenames retain 'independent' for compatibility; the trace "
            "is simulated and is not independent measured deployment evidence"
        ),
    }
    (output_dir / "independent_validation_metadata.json").write_text(
        json.dumps(validation_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"independent_multipath_validation_written={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
