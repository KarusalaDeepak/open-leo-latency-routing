#!/usr/bin/env python3
"""Scenario-based admission power for the measured 60-ms service profile.

Every generated row is one independently initiated campaign-level evaluation
unit. The simulation estimates operating characteristics under explicit data-
generating assumptions; it is not pilot data or measured deployment evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    risk_control_config_to_dict,
    select_opportunity_aware_risk_controlled_policy,
)


PROFILE_ID = "commect_measured_60ms"
LATENCY_TARGET_MS = 60.0
DEFAULT_GROUP_COUNTS = (721, 962, 2_500, 5_000, 7_500, 10_000)
DEFAULT_REPETITIONS = 30
DEFAULT_SEED = 20_260_926


@dataclass(frozen=True)
class PowerScenario:
    name: str
    candidate_shift_ms: float
    interpretation: str


SCENARIOS = (
    PowerScenario(
        "safe_modest_20ms",
        -20.0,
        "Candidate is never slower and has a 20-ms campaign-level benefit.",
    ),
    PowerScenario(
        "safe_strong_80ms",
        -80.0,
        "Candidate is never slower and has an 80-ms campaign-level benefit.",
    ),
    PowerScenario(
        "unsafe_20ms",
        20.0,
        "Candidate is 20 ms slower and can convert QoS successes into misses.",
    ),
)


def practical_config(seed: int) -> RiskControlConfig:
    return RiskControlConfig(
        alpha=0.05,
        noninferiority_margin=0.02,
        opportunity_noninferiority_margin=0.02,
        minimum_effective_opportunities=5.0,
        practical_cvar_gain_ms=5.0,
        cvar_quantile=0.90,
        latency_cap_ms=250.0,
        cvar_grid_points=5_001,
        planned_gate_uses=1,
        gate_use_index=1,
        random_seed=seed,
    )


def _binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if successes == 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    if successes == trials:
        upper = 1.0
    else:
        upper = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def run_power_simulation(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    group_counts: tuple[int, ...] = DEFAULT_GROUP_COUNTS,
    scenarios: tuple[PowerScenario, ...] = SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame, RiskControlConfig]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if any(groups < 1 for groups in group_counts):
        raise ValueError("group counts must be positive")
    config = practical_config(seed)
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for groups in group_counts:
            for repetition in range(repetitions):
                seed_sequence = np.random.SeedSequence(
                    [seed, scenario_index, groups, repetition]
                )
                replication_seed = int(
                    seed_sequence.generate_state(1, dtype=np.uint64)[0]
                )
                rng = np.random.default_rng(replication_seed)
                reactive = np.clip(rng.normal(120.0, 45.0, groups), 0.0, 250.0)
                candidate = np.clip(
                    reactive + scenario.candidate_shift_ms,
                    0.0,
                    250.0,
                )
                reactive_success = reactive <= LATENCY_TARGET_MS
                candidate_success = candidate <= LATENCY_TARGET_MS
                opportunities = reactive_success != candidate_success
                group_ids = np.arange(groups, dtype=np.int64)
                selection = select_opportunity_aware_risk_controlled_policy(
                    realized_selection_latency={
                        "reactive": reactive.tolist(),
                        "candidate": candidate.tolist(),
                    },
                    opportunity_mask=opportunities,
                    latency_budget_ms=LATENCY_TARGET_MS,
                    config=config,
                    reactive_policy="reactive",
                    independence_group_ids=group_ids,
                    policy_active_opportunity_masks={
                        "candidate": opportunities,
                    },
                )
                evidence = selection.evidence_frame().set_index("policy").loc[
                    "candidate"
                ]
                rows.append(
                    {
                        "profile_id": PROFILE_ID,
                        "scenario": scenario.name,
                        "group_count": groups,
                        "repetition": repetition,
                        "replication_seed": replication_seed,
                        "opportunity_group_prevalence": float(opportunities.mean()),
                        "empirical_success_delta": float(
                            evidence["group_uniform_success_delta_vs_reactive"]
                        ),
                        "empirical_cvar_gain_ms": float(
                            evidence["group_uniform_cvar_gain_vs_reactive_ms"]
                        ),
                        "cvar_gain_lcb_ms": float(
                            evidence["group_uniform_cvar_gain_lcb_ms"]
                        ),
                        "aggregate_success_pass": bool(
                            evidence["aggregate_actionable_success_noninferior"]
                        ),
                        "generic_opportunity_pass": bool(
                            evidence["opportunity_conditioned_success_noninferior"]
                        ),
                        "active_opportunity_pass": bool(
                            evidence["policy_active_opportunity_success_noninferior"]
                        ),
                        "tail_gain_pass": bool(evidence["practically_better"]),
                        "admitted": selection.selected_policy == "candidate",
                    }
                )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (scenario, groups), cell in detail.groupby(
        ["scenario", "group_count"], sort=False
    ):
        admitted = int(cell["admitted"].sum())
        trials = int(len(cell))
        ci_low, ci_high = _binomial_interval(admitted, trials)
        summary_rows.append(
            {
                "profile_id": PROFILE_ID,
                "scenario": scenario,
                "group_count": int(groups),
                "repetitions": trials,
                "admissions": admitted,
                "admission_probability": admitted / trials,
                "admission_probability_ci95_low": ci_low,
                "admission_probability_ci95_high": ci_high,
                "median_opportunity_group_prevalence": float(
                    cell["opportunity_group_prevalence"].median()
                ),
                "median_empirical_success_delta": float(
                    cell["empirical_success_delta"].median()
                ),
                "median_empirical_cvar_gain_ms": float(
                    cell["empirical_cvar_gain_ms"].median()
                ),
                "median_cvar_gain_lcb_ms": float(
                    cell["cvar_gain_lcb_ms"].median()
                ),
            }
        )
    return detail, pd.DataFrame(summary_rows), config


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    config: RiskControlConfig,
    output_dir: Path,
    *,
    repetitions: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "campaign_power_detail.csv", index=False)
    summary.to_csv(output_dir / "campaign_power_summary.csv", index=False)
    metadata = {
        "artifact_kind": "scenario_based_campaign_admission_power",
        "profile_id": PROFILE_ID,
        "measured_or_pilot_evidence": False,
        "field_power_guarantee": False,
        "repetitions_per_cell": repetitions,
        "seed": seed,
        "group_counts": sorted(int(value) for value in detail["group_count"].unique()),
        "independence_model": (
            "Each generated value is one independently initiated campaign-level "
            "unit. Field use additionally requires manifest validation and must "
            "account for shared site, weather, gateway, and provider shocks."
        ),
        "simulation_assumptions": {
            "reactive_campaign_latency_ms": "clip(N(120,45^2),0,250)",
            "candidate_latency_ms": "clip(reactive + scenario_shift,0,250)",
            "generic_opportunity": "reactive and candidate 60-ms success differ",
            "candidate_active_opportunity": "equal to generic opportunity",
        },
        "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
        "risk_control_config": risk_control_config_to_dict(config),
        "interpretation": (
            "Admission probability is conditional on the stated synthetic data-"
            "generating process. The 721-group analytical floor is necessary for "
            "best-case non-vacuity but is not sufficient for useful power. A "
            "disjoint field pilot must replace these assumptions before deployment."
        ),
    }
    (output_dir / "campaign_power_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/campaign_admission_power_60ms",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    detail, summary, config = run_power_simulation(
        repetitions=args.repetitions,
        seed=args.seed,
    )
    write_outputs(
        detail,
        summary,
        config,
        output,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    print(summary.to_string(index=False))
    print(f"outputs={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
