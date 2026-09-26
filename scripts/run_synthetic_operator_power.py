#!/usr/bin/env python3
"""Reproduce the separate 150-ms synthetic gate-admission demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar
from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    risk_control_config_to_dict,
    select_opportunity_aware_risk_controlled_policy,
)


PROFILE_ID = "synthetic_operator_150ms"
LATENCY_TARGET_MS = 150.0
GROUP_COUNTS = (500, 2_500, 5_000, 10_000)
DEVELOPMENT_EPISODES = 5_000
LOCKED_TEST_EPISODES = 5_000
DEFAULT_REPETITIONS = 30
DEFAULT_SEED = 20_260_926


def synthetic_config(seed: int) -> RiskControlConfig:
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


def generate_episodes(
    rng: np.random.Generator,
    episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.clip(rng.normal(75.0, 9.0, (episodes, 3)), 45.0, 110.0)
    indicator = rng.uniform(0.0, 1.0, (episodes, 3))
    path_bias = np.asarray([-3.0, 0.0, 3.0])
    noise = rng.normal(0.0, 1.5, (episodes, 3))
    next_rtt = np.clip(
        30.0 + 0.25 * current + 190.0 * indicator + path_bias + noise,
        0.0,
        250.0,
    )
    return current, indicator, next_rtt


def feature_frame(current: np.ndarray, indicator: np.ndarray) -> pd.DataFrame:
    episodes, paths = current.shape
    return pd.DataFrame(
        {
            "current_rtt_ms": current.reshape(-1),
            "observable_indicator": indicator.reshape(-1),
            "path_id": np.tile(np.arange(paths).astype(str), episodes),
        }
    )


def fit_frozen_candidate(seed: int) -> Pipeline:
    rng = np.random.default_rng(seed)
    current, indicator, next_rtt = generate_episodes(rng, DEVELOPMENT_EPISODES)
    preprocessing = ColumnTransformer(
        [
            (
                "continuous",
                StandardScaler(),
                ["current_rtt_ms", "observable_indicator"],
            ),
            (
                "path",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["path_id"],
            ),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessing),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(feature_frame(current, indicator), next_rtt.reshape(-1))
    return model


def policy_outcomes(
    model: Pipeline,
    current: np.ndarray,
    indicator: np.ndarray,
    next_rtt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = model.predict(feature_frame(current, indicator)).reshape(
        current.shape
    )
    reactive_choice = np.argmin(current, axis=1)
    candidate_choice = np.argmin(predictions, axis=1)
    rows = np.arange(len(current))
    reactive = next_rtt[rows, reactive_choice]
    candidate = next_rtt[rows, candidate_choice]
    candidate_success = next_rtt <= LATENCY_TARGET_MS
    opportunities = candidate_success.max(axis=1) != candidate_success.min(axis=1)
    active_opportunities = opportunities & (candidate_choice != reactive_choice)
    return reactive, candidate, opportunities, active_opportunities


def run_experiment(
    *,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    group_counts: tuple[int, ...] = GROUP_COUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RiskControlConfig]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    model = fit_frozen_candidate(seed)
    config = synthetic_config(seed)
    rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for groups in group_counts:
        for repetition in range(repetitions):
            seed_sequence = np.random.SeedSequence([seed, groups, repetition])
            gate_seed, test_seed = seed_sequence.spawn(2)
            gate_rng = np.random.default_rng(gate_seed)
            test_rng = np.random.default_rng(test_seed)

            current, indicator, next_rtt = generate_episodes(gate_rng, groups)
            reactive, candidate, opportunities, active = policy_outcomes(
                model, current, indicator, next_rtt
            )
            selection = select_opportunity_aware_risk_controlled_policy(
                realized_selection_latency={
                    "reactive": reactive.tolist(),
                    "candidate": candidate.tolist(),
                },
                opportunity_mask=opportunities,
                latency_budget_ms=LATENCY_TARGET_MS,
                config=config,
                reactive_policy="reactive",
                independence_group_ids=np.arange(groups, dtype=np.int64),
                policy_active_opportunity_masks={"candidate": active},
            )
            evidence = selection.evidence_frame().set_index("policy").loc[
                "candidate"
            ]

            test_current, test_indicator, test_next_rtt = generate_episodes(
                test_rng, LOCKED_TEST_EPISODES
            )
            test_reactive, test_candidate, _, _ = policy_outcomes(
                model,
                test_current,
                test_indicator,
                test_next_rtt,
            )
            admitted = selection.selected_policy == "candidate"
            gated = test_candidate if admitted else test_reactive
            rows.append(
                {
                    "profile_id": PROFILE_ID,
                    "group_count": groups,
                    "repetition": repetition,
                    "opportunity_group_count": int(opportunities.sum()),
                    "policy_active_opportunity_group_count": int(active.sum()),
                    "admitted": admitted,
                    "cvar_gain_lcb_ms": float(evidence["cvar_gain_lcb_ms"]),
                    "reactive_test_successes": int(
                        np.count_nonzero(test_reactive <= LATENCY_TARGET_MS)
                    ),
                    "candidate_test_successes": int(
                        np.count_nonzero(test_candidate <= LATENCY_TARGET_MS)
                    ),
                    "gated_test_successes": int(
                        np.count_nonzero(gated <= LATENCY_TARGET_MS)
                    ),
                    "reactive_test_cvar_ms": empirical_upper_cvar(
                        test_reactive, 0.90
                    ),
                    "candidate_test_cvar_ms": empirical_upper_cvar(
                        test_candidate, 0.90
                    ),
                    "gated_test_cvar_ms": empirical_upper_cvar(gated, 0.90),
                }
            )

            if groups == max(group_counts):
                for name, control_candidate in (
                    ("paired_null", reactive.copy()),
                    ("success_unsafe", np.clip(reactive + 80.0, 0.0, 250.0)),
                ):
                    control_opportunity = (
                        (reactive <= LATENCY_TARGET_MS)
                        != (control_candidate <= LATENCY_TARGET_MS)
                    )
                    control = select_opportunity_aware_risk_controlled_policy(
                        realized_selection_latency={
                            "reactive": reactive.tolist(),
                            "candidate": control_candidate.tolist(),
                        },
                        opportunity_mask=control_opportunity,
                        latency_budget_ms=LATENCY_TARGET_MS,
                        config=config,
                        reactive_policy="reactive",
                        independence_group_ids=np.arange(groups, dtype=np.int64),
                        policy_active_opportunity_masks={
                            "candidate": control_opportunity,
                        },
                    )
                    control_rows.append(
                        {
                            "profile_id": PROFILE_ID,
                            "control": name,
                            "group_count": groups,
                            "repetition": repetition,
                            "admitted": control.selected_policy == "candidate",
                        }
                    )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("group_count", sort=True)
        .agg(
            repetitions=("repetition", "size"),
            median_opportunity_groups=("opportunity_group_count", "median"),
            median_policy_active_opportunity_groups=(
                "policy_active_opportunity_group_count",
                "median",
            ),
            admissions=("admitted", "sum"),
            admission_rate=("admitted", "mean"),
            median_cvar_gain_lcb_ms=("cvar_gain_lcb_ms", "median"),
            reactive_test_successes=("reactive_test_successes", "sum"),
            candidate_test_successes=("candidate_test_successes", "sum"),
            gated_test_successes=("gated_test_successes", "sum"),
            median_reactive_test_cvar_ms=("reactive_test_cvar_ms", "median"),
            median_candidate_test_cvar_ms=("candidate_test_cvar_ms", "median"),
            median_gated_test_cvar_ms=("gated_test_cvar_ms", "median"),
        )
        .reset_index()
    )
    summary["locked_test_decisions"] = (
        summary["repetitions"] * LOCKED_TEST_EPISODES
    )
    return detail, summary, pd.DataFrame(control_rows), config


def write_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    config: RiskControlConfig,
    output_dir: Path,
    *,
    repetitions: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "synthetic_operator_power_detail.csv", index=False)
    summary.to_csv(output_dir / "synthetic_operator_power_summary.csv", index=False)
    controls.to_csv(output_dir / "synthetic_operator_controls.csv", index=False)
    metadata = {
        "artifact_kind": "controlled_synthetic_gate_admission_demonstration",
        "profile_id": PROFILE_ID,
        "measured_deployment_evidence": False,
        "cross_profile_pooling": False,
        "seed": seed,
        "repetitions_per_group_count": repetitions,
        "development_episodes": DEVELOPMENT_EPISODES,
        "locked_test_episodes_per_repetition": LOCKED_TEST_EPISODES,
        "group_counts": list(GROUP_COUNTS),
        "generator": (
            "current=clip(N(75,9^2),45,110); indicator=U(0,1); "
            "next=clip(30+0.25*current+190*indicator+path_bias+N(0,1.5^2),0,250)"
        ),
        "candidate": (
            "standardized ridge(alpha=1) using current RTT, observable indicator, "
            "and one-hot path identity; fitted once on disjoint development data"
        ),
        "risk_control_config": risk_control_config_to_dict(config),
    }
    (output_dir / "synthetic_operator_power_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/synthetic_operator_power_150ms",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    detail, summary, controls, config = run_experiment(
        repetitions=args.repetitions,
        seed=args.seed,
    )
    write_outputs(
        detail,
        summary,
        controls,
        config,
        output,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    print(summary.to_string(index=False))
    print(controls.groupby("control")["admitted"].agg(["sum", "count"]))
    print(f"outputs={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
