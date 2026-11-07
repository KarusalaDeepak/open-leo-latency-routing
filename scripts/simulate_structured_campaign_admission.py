#!/usr/bin/env python3
"""Campaign-structured operating characteristics for the admission gate.

This simulation complements the independent-episode code-path test. Each
synthetic campaign contains a dependent sequence of decisions, campaign
lengths vary, and complete-case retention can depend on high latency. The
experiment is a declared stress model, not measured field evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    risk_control_config_to_dict,
    select_opportunity_aware_risk_controlled_policy,
)


LATENCY_TARGET_MS = 60.0
DEFAULT_GROUP_COUNTS = (100, 200, 400, 800, 1_600, 3_200)
DEFAULT_REPETITIONS = 30
DEFAULT_SEED = 20_261_104


def wilson_binomial_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson interval for a repeated-simulation admission fraction."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return lower, upper


@dataclass(frozen=True)
class StructuredScenario:
    name: str
    lower_future_path_probability: float
    harmful_campaign_probability: float = 0.0
    prefer_high_tail_on_uniform_epochs: bool = False
    interpretation: str = ""


SCENARIOS = (
    StructuredScenario(
        name="strong_benefit",
        lower_future_path_probability=0.995,
        interpretation=(
            "Candidate almost always identifies the lower next-interval RTT."
        ),
    ),
    StructuredScenario(
        name="near_boundary_benefit",
        lower_future_path_probability=0.58,
        interpretation=(
            "Candidate has only a small ranking advantage over current RTT."
        ),
    ),
    StructuredScenario(
        name="success_gain_tail_tradeoff",
        lower_future_path_probability=0.90,
        prefer_high_tail_on_uniform_epochs=True,
        interpretation=(
            "Candidate improves mixed-outcome choices but prefers the slower "
            "path when both paths share the same success label."
        ),
    ),
    StructuredScenario(
        name="rare_severe_campaign_loss",
        lower_future_path_probability=0.98,
        harmful_campaign_probability=0.05,
        interpretation=(
            "Most campaigns benefit, but a small campaign fraction reverses "
            "the ranking and creates severe losses."
        ),
    ),
    StructuredScenario(
        name="unsafe_regression",
        lower_future_path_probability=0.30,
        interpretation="Candidate usually ranks the higher future RTT first.",
    ),
)


def structured_config(
    seed: int,
    *,
    latency_cap_ms: float = 250.0,
) -> RiskControlConfig:
    return RiskControlConfig(
        alpha=0.05,
        noninferiority_margin=0.02,
        opportunity_noninferiority_margin=0.02,
        minimum_effective_opportunities=5.0,
        practical_cvar_gain_ms=5.0,
        cvar_quantile=0.90,
        latency_cap_ms=latency_cap_ms,
        cvar_grid_points=501,
        planned_gate_uses=1,
        gate_use_index=1,
        random_seed=seed,
    )


def _campaign_lengths(
    rng: np.random.Generator,
    group_count: int,
    *,
    total_scheduled_epochs: int | None,
) -> np.ndarray:
    if total_scheduled_epochs is None:
        return rng.integers(18, 43, size=group_count, endpoint=False)
    if total_scheduled_epochs < 6 * group_count:
        raise ValueError("fixed total must provide at least six epochs per campaign")
    remaining = total_scheduled_epochs - 6 * group_count
    weights = rng.dirichlet(np.full(group_count, 5.0))
    return 6 + rng.multinomial(remaining, weights)


def _markov_regimes(
    rng: np.random.Generator,
    length: int,
) -> np.ndarray:
    state = np.empty(length, dtype=float)
    state[0] = rng.normal()
    innovation_scale = math.sqrt(1.0 - 0.78**2)
    for index in range(1, length):
        state[index] = (
            0.78 * state[index - 1]
            + innovation_scale * rng.normal()
        )
    # 0: both future paths pass, 1: generic opportunity, 2: both fail.
    return np.where(state > 0.35, 1, np.where(state < -0.85, 2, 0))


def generate_campaign_sample(
    rng: np.random.Generator,
    *,
    group_count: int,
    scenario: StructuredScenario,
    base_missing_probability: float = 0.05,
    total_scheduled_epochs: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Generate aligned candidate and reactive outcomes for many campaigns."""

    if not 0.0 <= base_missing_probability < 1.0:
        raise ValueError("missing probability must lie in [0, 1)")
    lengths = _campaign_lengths(
        rng,
        group_count,
        total_scheduled_epochs=total_scheduled_epochs,
    )
    retained_reactive: list[np.ndarray] = []
    retained_candidate: list[np.ndarray] = []
    retained_generic: list[np.ndarray] = []
    retained_active: list[np.ndarray] = []
    retained_groups: list[np.ndarray] = []
    scheduled = 0
    complete = 0
    mixed_current = 0
    uniform_current = 0
    high_latency_missing = 0

    for group_id, length in enumerate(lengths):
        length = int(length)
        scheduled += length
        campaign_shift = rng.normal(0.0, 4.0)
        regimes = _markov_regimes(rng, length)
        future = np.empty((length, 2), dtype=float)
        for index, regime in enumerate(regimes):
            if regime == 0:
                future[index] = np.clip(
                    rng.normal(46.0 + campaign_shift, 5.5, size=2),
                    20.0,
                    59.0,
                )
            elif regime == 1:
                good_path = int(rng.integers(0, 2))
                future[index, good_path] = np.clip(
                    rng.normal(46.0 + 0.3 * campaign_shift, 4.5),
                    25.0,
                    58.0,
                )
                future[index, 1 - good_path] = np.clip(
                    rng.normal(175.0 + campaign_shift, 30.0),
                    62.0,
                    250.0,
                )
            else:
                low_path = int(rng.integers(0, 2))
                future[index, low_path] = np.clip(
                    rng.normal(72.0 + 0.2 * campaign_shift, 6.0),
                    61.0,
                    95.0,
                )
                high_path = 1 - low_path
                future[index, high_path] = np.clip(
                    rng.normal(220.0 + campaign_shift, 25.0),
                    120.0,
                    250.0,
                )
            if rng.random() < 0.025:
                future[index, int(rng.integers(0, 2))] = rng.uniform(180.0, 250.0)

        current_state = np.empty((length, 2), dtype=float)
        current_state[0] = rng.normal(60.0 + campaign_shift, 12.0, size=2)
        for index in range(1, length):
            current_state[index] = (
                0.72 * current_state[index - 1]
                + 0.28 * rng.normal(60.0 + campaign_shift, 12.0, size=2)
            )
        current_state = np.clip(current_state, 15.0, 180.0)
        current_success = current_state <= LATENCY_TARGET_MS
        mixed_current += int(np.count_nonzero(current_success.sum(axis=1) == 1))
        uniform_current += int(np.count_nonzero(current_success.sum(axis=1) != 1))

        reactive_choice = np.argmin(current_state, axis=1)
        lower_future = np.argmin(future, axis=1)
        higher_future = 1 - lower_future
        campaign_harmful = (
            rng.random() < scenario.harmful_campaign_probability
        )
        accuracy = 0.02 if campaign_harmful else scenario.lower_future_path_probability
        candidate_choice = np.where(
            rng.random(length) < accuracy,
            lower_future,
            higher_future,
        )
        if scenario.prefer_high_tail_on_uniform_epochs:
            uniform_future = (future <= LATENCY_TARGET_MS).sum(axis=1) != 1
            candidate_choice[uniform_future] = higher_future[uniform_future]

        future_success = future <= LATENCY_TARGET_MS
        generic = future_success[:, 0] != future_success[:, 1]
        active = generic & (candidate_choice != reactive_choice)
        reactive = future[np.arange(length), reactive_choice]
        candidate = future[np.arange(length), candidate_choice]

        missing_probability = (
            base_missing_probability
            + 0.12 * (future > 150.0)
            + 0.04 * (current_state > 110.0)
        )
        path_observed = rng.random((length, 2)) >= np.clip(
            missing_probability,
            0.0,
            0.95,
        )
        complete_case = path_observed.all(axis=1)
        high_latency_missing += int(
            np.count_nonzero((~complete_case) & (future.max(axis=1) > 150.0))
        )
        complete += int(complete_case.sum())
        retained_reactive.append(reactive[complete_case])
        retained_candidate.append(candidate[complete_case])
        retained_generic.append(generic[complete_case])
        retained_active.append(active[complete_case])
        retained_groups.append(
            np.full(int(complete_case.sum()), group_id, dtype=np.int64)
        )

    arrays = {
        "reactive": np.concatenate(retained_reactive),
        "candidate": np.concatenate(retained_candidate),
        "generic": np.concatenate(retained_generic),
        "active": np.concatenate(retained_active),
        "group_ids": np.concatenate(retained_groups),
    }
    metadata: dict[str, float | int] = {
        "initiated_campaign_count": group_count,
        "scheduled_epoch_count": scheduled,
        "complete_case_count": complete,
        "missing_epoch_count": scheduled - complete,
        "complete_case_fraction": complete / scheduled,
        "mixed_current_snapshot_count": mixed_current,
        "uniform_current_snapshot_count": uniform_current,
        "high_latency_missing_count": high_latency_missing,
        "minimum_campaign_length": int(lengths.min()),
        "maximum_campaign_length": int(lengths.max()),
    }
    return arrays, metadata


def empirical_bernstein_lcb(values: np.ndarray, *, alpha: float) -> float:
    """One-sided empirical-Bernstein LCB for values supported on [-1, 1]."""

    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return -1.0
    log_term = math.log(2.0 / alpha)
    variance = float(np.var(values, ddof=1))
    radius = math.sqrt(2.0 * variance * log_term / len(values))
    radius += 14.0 * log_term / (3.0 * (len(values) - 1))
    return max(-1.0, float(np.mean(values)) - radius)


def _policy_active_group_deltas(
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    reactive_success = arrays["reactive"] <= LATENCY_TARGET_MS
    candidate_success = arrays["candidate"] <= LATENCY_TARGET_MS
    paired = candidate_success.astype(float) - reactive_success.astype(float)
    values: list[float] = []
    for group_id in pd.unique(arrays["group_ids"]):
        mask = (arrays["group_ids"] == group_id) & arrays["active"]
        if mask.any():
            values.append(float(np.mean(paired[mask])))
    return np.asarray(values, dtype=float)


def evaluate_sample(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, float | int],
    *,
    config: RiskControlConfig,
) -> dict[str, object]:
    selection = select_opportunity_aware_risk_controlled_policy(
        realized_selection_latency={
            "reactive": arrays["reactive"].tolist(),
            "candidate": arrays["candidate"].tolist(),
        },
        opportunity_mask=arrays["generic"],
        latency_budget_ms=LATENCY_TARGET_MS,
        config=config,
        reactive_policy="reactive",
        independence_group_ids=arrays["group_ids"],
        policy_active_opportunity_masks={"candidate": arrays["active"]},
    )
    evidence = selection.evidence_frame().set_index("policy").loc["candidate"]
    active_deltas = _policy_active_group_deltas(arrays)
    magnitude_lcb = empirical_bernstein_lcb(
        active_deltas,
        alpha=float(evidence["alpha_success_bound"]),
    )
    magnitude_success_pass = bool(
        len(active_deltas) >= config.minimum_effective_opportunities
        and magnitude_lcb >= -config.opportunity_noninferiority_margin
    )
    support_pass = bool(
        evidence["independent_group_sufficient"]
        and evidence["opportunity_sufficient"]
        and evidence["policy_active_opportunity_sufficient"]
    )
    tail_pass = bool(evidence["practically_better"])
    reactive_success = arrays["reactive"] <= LATENCY_TARGET_MS
    candidate_success = arrays["candidate"] <= LATENCY_TARGET_MS
    scheduled = int(metadata["scheduled_epoch_count"])
    return {
        **metadata,
        "retained_campaign_count": int(evidence["inference_block_count"]),
        "generic_opportunity_campaign_count": int(
            evidence["opportunity_block_count"]
        ),
        "policy_active_campaign_count": int(
            evidence["policy_active_opportunity_block_count"]
        ),
        "generic_opportunity_count": int(evidence["opportunity_count"]),
        "policy_active_opportunity_count": int(
            evidence["policy_active_opportunity_count"]
        ),
        "reactive_complete_case_success": float(reactive_success.mean()),
        "candidate_complete_case_success": float(candidate_success.mean()),
        "reactive_scheduled_success_lower": float(reactive_success.sum() / scheduled),
        "candidate_scheduled_success_lower": float(candidate_success.sum() / scheduled),
        "primary_sign_lcb": float(evidence["success_delta_lcb"]),
        "magnitude_lcb": magnitude_lcb,
        "sign_success_pass": bool(evidence["primary_success_noninferior"]),
        "magnitude_success_pass": magnitude_success_pass,
        "tail_gain_point_ms": float(evidence["cvar_gain_vs_reactive_ms"]),
        "tail_gain_lcb_ms": float(evidence["cvar_gain_lcb_ms"]),
        "tail_pass": tail_pass,
        "support_pass": support_pass,
        "sign_gate_admitted": selection.selected_policy == "candidate",
        "magnitude_gate_admitted": bool(
            support_pass and magnitude_success_pass and tail_pass
        ),
        "reactive_clipped_fraction": float(
            selection.evidence_frame()
            .set_index("policy")
            .loc["reactive", "latency_clipped_count"]
            / len(arrays["reactive"])
        ),
        "candidate_clipped_fraction": float(
            evidence["latency_clipped_count"] / len(arrays["candidate"])
        ),
    }


def run_structured_power(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    group_counts: tuple[int, ...] = DEFAULT_GROUP_COUNTS,
    scenarios: tuple[StructuredScenario, ...] = SCENARIOS,
    base_missing_probability: float = 0.05,
    latency_cap_ms: float = 250.0,
    fixed_total_epochs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, RiskControlConfig]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = structured_config(seed, latency_cap_ms=latency_cap_ms)
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for group_count in group_counts:
            for repetition in range(repetitions):
                sequence = np.random.SeedSequence(
                    [seed, scenario_index, group_count, repetition]
                )
                replication_seed = int(
                    sequence.generate_state(1, dtype=np.uint64)[0]
                )
                arrays, metadata = generate_campaign_sample(
                    np.random.default_rng(replication_seed),
                    group_count=group_count,
                    scenario=scenario,
                    base_missing_probability=base_missing_probability,
                    total_scheduled_epochs=fixed_total_epochs,
                )
                rows.append(
                    {
                        "scenario": scenario.name,
                        "group_count": group_count,
                        "repetition": repetition,
                        "replication_seed": replication_seed,
                        "missing_probability": base_missing_probability,
                        "latency_cap_ms": latency_cap_ms,
                        "fixed_total_epochs": fixed_total_epochs or "",
                        **evaluate_sample(arrays, metadata, config=config),
                    }
                )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (scenario, group_count), cell in detail.groupby(
        ["scenario", "group_count"], sort=False
    ):
        repetitions_in_cell = len(cell)
        sign_admissions = int(cell["sign_gate_admitted"].sum())
        magnitude_admissions = int(cell["magnitude_gate_admitted"].sum())
        sign_ci_low, sign_ci_high = wilson_binomial_interval(
            sign_admissions,
            repetitions_in_cell,
        )
        magnitude_ci_low, magnitude_ci_high = wilson_binomial_interval(
            magnitude_admissions,
            repetitions_in_cell,
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "group_count": int(group_count),
                "repetitions": repetitions_in_cell,
                "median_scheduled_epochs": float(
                    cell["scheduled_epoch_count"].median()
                ),
                "median_complete_case_fraction": float(
                    cell["complete_case_fraction"].median()
                ),
                "median_generic_opportunity_campaigns": float(
                    cell["generic_opportunity_campaign_count"].median()
                ),
                "median_policy_active_campaigns": float(
                    cell["policy_active_campaign_count"].median()
                ),
                "median_candidate_minus_reactive_success": float(
                    (
                        cell["candidate_complete_case_success"]
                        - cell["reactive_complete_case_success"]
                    ).median()
                ),
                "sign_success_pass_rate": float(
                    cell["sign_success_pass"].mean()
                ),
                "magnitude_success_pass_rate": float(
                    cell["magnitude_success_pass"].mean()
                ),
                "tail_pass_rate": float(cell["tail_pass"].mean()),
                "sign_gate_admissions": sign_admissions,
                "sign_gate_admission_rate": sign_admissions / repetitions_in_cell,
                "sign_gate_admission_ci95_low": sign_ci_low,
                "sign_gate_admission_ci95_high": sign_ci_high,
                "magnitude_gate_admissions": magnitude_admissions,
                "magnitude_gate_admission_rate": (
                    magnitude_admissions / repetitions_in_cell
                ),
                "magnitude_gate_admission_ci95_low": magnitude_ci_low,
                "magnitude_gate_admission_ci95_high": magnitude_ci_high,
                "median_primary_sign_lcb": float(
                    cell["primary_sign_lcb"].median()
                ),
                "median_magnitude_lcb": float(cell["magnitude_lcb"].median()),
                "median_tail_gain_point_ms": float(
                    cell["tail_gain_point_ms"].median()
                ),
                "median_tail_gain_lcb_ms": float(
                    cell["tail_gain_lcb_ms"].median()
                ),
            }
        )
    return detail, pd.DataFrame(summary_rows), config


def build_sensitivity(
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    scenario = (SCENARIOS[0],)
    for missing_probability in (0.0, 0.05, 0.15):
        for cap in (150.0, 200.0, 250.0):
            detail, _, _ = run_structured_power(
                repetitions=repetitions,
                seed=seed,
                group_counts=(800,),
                scenarios=scenario,
                base_missing_probability=missing_probability,
                latency_cap_ms=cap,
            )
            rows.append(detail)
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(["missing_probability", "latency_cap_ms"], as_index=False)
        .agg(
            repetitions=("repetition", "count"),
            median_complete_case_fraction=("complete_case_fraction", "median"),
            median_reactive_clipped_fraction=("reactive_clipped_fraction", "median"),
            median_candidate_clipped_fraction=("candidate_clipped_fraction", "median"),
            median_tail_gain_lcb_ms=("tail_gain_lcb_ms", "median"),
            sign_gate_admission_rate=("sign_gate_admitted", "mean"),
            magnitude_gate_admission_rate=("magnitude_gate_admitted", "mean"),
        )
    )


def plot_admission_curves(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for scenario, cell in summary.groupby("scenario", sort=False):
        line = axes[0].plot(
            cell["group_count"],
            cell["sign_gate_admission_rate"],
            marker="o",
            label=scenario.replace("_", " "),
        )[0]
        axes[0].fill_between(
            cell["group_count"],
            cell["sign_gate_admission_ci95_low"],
            cell["sign_gate_admission_ci95_high"],
            color=line.get_color(),
            alpha=0.09,
        )
        line = axes[1].plot(
            cell["group_count"],
            cell["magnitude_gate_admission_rate"],
            marker="o",
            label=scenario.replace("_", " "),
        )[0]
        axes[1].fill_between(
            cell["group_count"],
            cell["magnitude_gate_admission_ci95_low"],
            cell["magnitude_gate_admission_ci95_high"],
            color=line.get_color(),
            alpha=0.09,
        )
    axes[0].set_title("Sign-only release rule")
    axes[1].set_title("Magnitude-sensitive sensitivity")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Independent campaigns")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Admission fraction")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    *,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    fixed_total_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    config: RiskControlConfig,
    repetitions: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "structured_campaign_power_detail.csv", index=False)
    summary.to_csv(output_dir / "structured_campaign_power_summary.csv", index=False)
    fixed_total_summary.to_csv(
        output_dir / "structured_fixed_total_summary.csv", index=False
    )
    sensitivity.to_csv(
        output_dir / "structured_cap_missingness_sensitivity.csv", index=False
    )
    plot_admission_curves(summary, output_dir / "structured_admission_curves.pdf")
    metadata = {
        "artifact_kind": "campaign_structured_gate_operating_characteristics",
        "measured_or_field_evidence": False,
        "repetitions_per_cell": repetitions,
        "seed": seed,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "risk_control_config": risk_control_config_to_dict(config),
        "campaign_structure": {
            "unequal_lengths": "integer uniform from 18 through 42 epochs",
            "within_campaign_dependence": "AR(1) state with rho=0.78",
            "current_states": "mixture of uniform and mixed QoS snapshots",
            "future_states": "uniform-pass, generic-opportunity, and uniform-fail regimes",
            "rare_tail_probability_per_epoch": 0.025,
            "missingness": "base probability plus latency-dependent increments",
        },
        "admission_fraction_uncertainty": (
            "Two-sided 95% Wilson binomial intervals across independent simulation "
            "repetitions; these quantify Monte Carlo repetition uncertainty only."
        ),
        "claim_boundary": (
            "Operating characteristics are conditional on this synthetic model. "
            "They demonstrate discrimination and evidence scaling, not field power."
        ),
    }
    (output_dir / "structured_campaign_power_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/structured_campaign_admission",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    detail, summary, config = run_structured_power(
        repetitions=args.repetitions,
        seed=args.seed,
    )
    _, fixed_total_summary, _ = run_structured_power(
        repetitions=args.repetitions,
        seed=args.seed + 1,
        group_counts=(25, 50, 100, 250),
        scenarios=(SCENARIOS[0],),
        fixed_total_epochs=6_000,
    )
    sensitivity = build_sensitivity(
        repetitions=max(10, args.repetitions // 3),
        seed=args.seed + 2,
    )
    write_outputs(
        output_dir,
        detail=detail,
        summary=summary,
        fixed_total_summary=fixed_total_summary,
        sensitivity=sensitivity,
        config=config,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    print(summary.to_string(index=False))
    print(f"outputs={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
