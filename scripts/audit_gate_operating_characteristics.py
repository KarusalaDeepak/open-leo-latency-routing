#!/usr/bin/env python3
"""Synthetic operating-characteristic audit for the production evidence gate.

This audit is a calibration and non-vacuity check.  It uses independently
generated, one-observation collection groups under predeclared synthetic data
generating processes.  It is not measured-path evidence and it does not
establish deployment efficacy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt

from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    risk_control_config_to_dict,
    select_opportunity_aware_risk_controlled_policy,
)


GROUP_COUNTS = (100, 217, 1_000, 5_000)
DEFAULT_REPETITIONS = 50
DEFAULT_SEED = 20_260_821
LATENCY_BUDGET_MS = 150.0
REACTIVE_POLICY = "reactive"
CANDIDATE_POLICY = "candidate"


def _paired_null(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    reactive = np.clip(rng.normal(95.0, 8.0, size), 40.0, 140.0)
    return reactive, reactive.copy()


def _unsafe(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    reactive = np.clip(rng.normal(95.0, 8.0, size), 40.0, 140.0)
    candidate = np.clip(reactive + rng.normal(80.0, 2.0, size), 0.0, 60_000.0)
    return reactive, candidate


def _moderate_safe_tail_benefit(
    rng: np.random.Generator,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    reactive = np.clip(rng.normal(110.0, 6.0, size), 60.0, 140.0)
    candidate = reactive - 20.0
    return reactive, candidate


def _strong_positive_control(
    rng: np.random.Generator,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    reactive = np.clip(rng.normal(59_900.0, 20.0, size), 59_800.0, 59_990.0)
    candidate = np.clip(rng.normal(5.0, 1.0, size), 0.0, 10.0)
    return reactive, candidate


SCENARIOS: tuple[
    tuple[str, str, Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]],
    ...,
] = (
    (
        "paired_null",
        "Candidate equals reactive observation by observation.",
        _paired_null,
    ),
    (
        "unsafe_success",
        "Candidate has materially worse binary QoS success than reactive.",
        _unsafe,
    ),
    (
        "moderate_safe_tail_benefit",
        "Both policies satisfy binary QoS; candidate is exactly 20 ms faster.",
        _moderate_safe_tail_benefit,
    ),
    (
        "strong_positive_control",
        "Reactive is near the predeclared cap and candidate is near zero.",
        _strong_positive_control,
    ),
)


def _default_config(seed: int) -> RiskControlConfig:
    return RiskControlConfig(
        alpha=0.05,
        noninferiority_margin=0.02,
        opportunity_noninferiority_margin=0.02,
        minimum_effective_opportunities=5.0,
        practical_cvar_gain_ms=1.0,
        cvar_quantile=0.95,
        block_length=None,
        latency_cap_ms=60_000.0,
        cvar_grid_points=1_200_001,
        planned_gate_uses=1,
        gate_use_index=1,
        random_seed=seed,
    )


def run_audit(repetitions: int, seed: int) -> tuple[pd.DataFrame, RiskControlConfig]:
    """Run every predeclared scenario/count/replication combination."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = _default_config(seed)
    rows: list[dict[str, object]] = []
    for scenario_index, (scenario, _, generator) in enumerate(SCENARIOS):
        for group_count in GROUP_COUNTS:
            for repetition in range(repetitions):
                seed_sequence = np.random.SeedSequence(
                    [seed, scenario_index, group_count, repetition]
                )
                replication_seed = int(
                    seed_sequence.generate_state(1, dtype=np.uint64)[0]
                )
                rng = np.random.default_rng(replication_seed)
                reactive, candidate = generator(rng, group_count)
                group_ids = np.arange(group_count, dtype=np.int64)
                selection = select_opportunity_aware_risk_controlled_policy(
                    realized_selection_latency={
                        REACTIVE_POLICY: reactive.tolist(),
                        CANDIDATE_POLICY: candidate.tolist(),
                    },
                    opportunity_mask=np.ones(group_count, dtype=bool),
                    latency_budget_ms=LATENCY_BUDGET_MS,
                    config=config,
                    reactive_policy=REACTIVE_POLICY,
                    independence_group_ids=group_ids,
                )
                evidence = selection.evidence_frame().set_index("policy")
                candidate_evidence = evidence.loc[CANDIDATE_POLICY]
                aggregate_noninferiority_pass = bool(
                    candidate_evidence[
                        "aggregate_actionable_success_noninferior"
                    ]
                )
                opportunity_noninferiority_pass = bool(
                    candidate_evidence[
                        "opportunity_conditioned_success_noninferior"
                    ]
                )
                noninferiority_pass = bool(
                    candidate_evidence["success_endpoints_noninferior"]
                )
                cvar_pass = bool(candidate_evidence["practically_better"])
                admitted = selection.selected_policy == CANDIDATE_POLICY
                rows.append(
                    {
                        "scenario": scenario,
                        "group_count": group_count,
                        "repetition": repetition,
                        "replication_seed": replication_seed,
                        "reactive_success_rate": float(
                            np.mean(reactive <= LATENCY_BUDGET_MS)
                        ),
                        "candidate_success_rate": float(
                            np.mean(candidate <= LATENCY_BUDGET_MS)
                        ),
                        "empirical_success_delta": float(
                            candidate_evidence["success_delta_vs_reactive"]
                        ),
                        "empirical_cvar_gain_ms": float(
                            candidate_evidence["cvar_gain_vs_reactive_ms"]
                        ),
                        "success_delta_lcb": float(
                            candidate_evidence["success_delta_lcb"]
                        ),
                        "aggregate_success_delta_lcb": float(
                            candidate_evidence[
                                "aggregate_actionable_success_delta_lcb"
                            ]
                        ),
                        "opportunity_conditioned_success_delta_lcb": float(
                            candidate_evidence[
                                "opportunity_conditioned_success_delta_lcb"
                            ]
                        ),
                        "cvar_gain_lcb_ms": float(
                            candidate_evidence["cvar_gain_lcb_ms"]
                        ),
                        "noninferiority_pass": noninferiority_pass,
                        "aggregate_noninferiority_pass": (
                            aggregate_noninferiority_pass
                        ),
                        "opportunity_noninferiority_pass": (
                            opportunity_noninferiority_pass
                        ),
                        "cvar_pass": cvar_pass,
                        "opportunity_pass": bool(
                            candidate_evidence["opportunity_sufficient"]
                        ),
                        "admitted": admitted,
                        "selected_policy": selection.selected_policy,
                        "selection_reason": selection.reason,
                        "inference_block_count": int(
                            candidate_evidence["inference_block_count"]
                        ),
                        "inference_unit_source": str(
                            candidate_evidence["inference_unit_source"]
                        ),
                    }
                )
    return pd.DataFrame(rows), config


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(["scenario", "group_count"], sort=False)
        .agg(
            repetitions=("repetition", "size"),
            admission_rate=("admitted", "mean"),
            noninferiority_pass_rate=("noninferiority_pass", "mean"),
            aggregate_noninferiority_pass_rate=(
                "aggregate_noninferiority_pass",
                "mean",
            ),
            opportunity_noninferiority_pass_rate=(
                "opportunity_noninferiority_pass",
                "mean",
            ),
            cvar_pass_rate=("cvar_pass", "mean"),
            opportunity_pass_rate=("opportunity_pass", "mean"),
            median_success_delta_lcb=("success_delta_lcb", "median"),
            median_opportunity_conditioned_success_delta_lcb=(
                "opportunity_conditioned_success_delta_lcb",
                "median",
            ),
            median_cvar_gain_lcb_ms=("cvar_gain_lcb_ms", "median"),
            median_empirical_success_delta=("empirical_success_delta", "median"),
            median_empirical_cvar_gain_ms=("empirical_cvar_gain_ms", "median"),
        )
        .reset_index()
    )
    return summary


def _write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    )
    labels = {
        "paired_null": "Paired null",
        "unsafe_success": "Unsafe success",
        "moderate_safe_tail_benefit": "Moderate 20-ms benefit",
        "strong_positive_control": "Strong positive control",
    }
    styles = {
        "paired_null": ("o", "#4d4d4d"),
        "unsafe_success": ("s", "#b2182b"),
        "moderate_safe_tail_benefit": ("^", "#2166ac"),
        "strong_positive_control": ("D", "#1b7837"),
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for scenario, _, _ in SCENARIOS:
        subset = summary.loc[summary["scenario"].eq(scenario)]
        marker, color = styles[scenario]
        ax.plot(
            subset["group_count"],
            subset["admission_rate"],
            marker=marker,
            color=color,
            linewidth=1.5,
            markersize=5,
            label=labels[scenario],
        )
    ax.set_xscale("log")
    ax.set_xticks(GROUP_COUNTS, [f"{value:,}" for value in GROUP_COUNTS])
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Independent one-observation collection groups")
    ax.set_ylabel("Candidate admission rate")
    ax.set_title("Synthetic evidence-gate operating characteristic")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(
        output_dir / "gate_admission_rates.png",
        bbox_inches="tight",
        metadata={"Software": "audit_gate_operating_characteristics.py"},
    )
    fig.savefig(
        output_dir / "gate_admission_rates.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "audit_gate_operating_characteristics.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    config: RiskControlConfig,
    output_dir: Path,
    repetitions: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "gate_operating_characteristics_detailed.csv", index=False)
    summary.to_csv(output_dir / "gate_operating_characteristics_summary.csv", index=False)
    metadata = {
        "artifact_kind": "synthetic_calibration_and_non_vacuity_audit",
        "measured_efficacy_evidence": False,
        "warning": (
            "Synthetic independently generated collection groups; this audit "
            "does not establish measured-path or deployment efficacy."
        ),
        "seed": seed,
        "repetitions_per_cell": repetitions,
        "group_counts": list(GROUP_COUNTS),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "independence_design": (
            "Each generated observation has a unique collection-group ID; "
            "groups are independently generated within every replication."
        ),
        "policies": {
            "reactive": REACTIVE_POLICY,
            "single_learned_candidate": CANDIDATE_POLICY,
        },
        "risk_control_config": risk_control_config_to_dict(config),
        "scenarios": [
            {"name": name, "data_generating_process": description}
            for name, description, _ in SCENARIOS
        ],
        "summary_records": summary.to_dict(orient="records"),
        "outputs": [
            "gate_operating_characteristics_detailed.csv",
            "gate_operating_characteristics_summary.csv",
            "gate_operating_characteristics_metadata.json",
            "gate_admission_rates.png",
            "gate_admission_rates.pdf",
        ],
    }
    (output_dir / "gate_operating_characteristics_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_plot(summary, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a synthetic operating-characteristic audit of the evidence gate."
    )
    parser.add_argument(
        "--output-dir",
        default="results/gate_operating_characteristics",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    detail, config = run_audit(args.repetitions, args.seed)
    summary = summarize(detail)
    write_outputs(
        detail,
        summary,
        config,
        output_path,
        args.repetitions,
        args.seed,
    )
    print(summary.to_string(index=False))
    print(f"outputs={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
