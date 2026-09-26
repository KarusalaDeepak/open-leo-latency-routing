#!/usr/bin/env python3
"""Translate a frozen gate profile into an operator collection plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_gate_design_sensitivity import (
    best_case_cvar_group_floor,
    grid_contract,
    zero_harm_success_group_floor,
)


def build_collection_plan(
    *,
    familywise_alpha: float = 0.05,
    candidate_count: int = 1,
    planned_uses: tuple[int, ...] = (1, 5),
    permitted_success_deficit: float = 0.02,
    minimum_groups: int = 5,
    cvar_quantile: float = 0.90,
    latency_cap_ms: float = 250.0,
    cvar_gain_ms: float = 5.0,
    grid_spacing_ms: float = 0.05,
    opportunity_prevalences: tuple[float, ...] = (1.0, 0.75, 0.50),
    parallel_sites: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    """Build best-case group floors and prevalence-adjusted campaign counts.

    The result is a non-vacuity plan, not a conventional power calculation.
    It assumes no harmful acquisition groups and therefore must be inflated by
    a prospective effect/variance simulation before field deployment.
    """

    points, _, correction = grid_contract(
        latency_cap_ms,
        grid_spacing_ms,
        cvar_quantile,
    )
    rows: list[dict[str, object]] = []
    for uses in planned_uses:
        endpoint_alpha = familywise_alpha / (5 * candidate_count * uses)
        success_floor = zero_harm_success_group_floor(
            permitted_success_deficit,
            endpoint_alpha,
        )
        tail_floor = best_case_cvar_group_floor(
            cvar_gain_ms,
            latency_cap_ms=latency_cap_ms,
            quantile=cvar_quantile,
            grid_points=points,
            grid_correction_ms=correction,
            alpha_endpoint=endpoint_alpha,
        )
        opportunity_floor = max(minimum_groups, success_floor, tail_floor)
        for prevalence in opportunity_prevalences:
            if not 0.0 < prevalence <= 1.0:
                raise ValueError("opportunity prevalence must be in (0, 1]")
            campaigns = math.ceil(opportunity_floor / prevalence)
            for sites in parallel_sites:
                if sites < 1:
                    raise ValueError("parallel site count must be positive")
                rows.append(
                    {
                        "planned_gate_uses": uses,
                        "candidate_count": candidate_count,
                        "endpoint_alpha": endpoint_alpha,
                        "permitted_success_deficit": permitted_success_deficit,
                        "cvar_quantile": cvar_quantile,
                        "latency_cap_ms": latency_cap_ms,
                        "required_cvar_gain_ms": cvar_gain_ms,
                        "minimum_group_floor": minimum_groups,
                        "zero_harm_success_group_floor": success_floor,
                        "best_case_tail_group_floor": tail_floor,
                        "joint_opportunity_bearing_group_floor": opportunity_floor,
                        "assumed_opportunity_group_prevalence": prevalence,
                        "required_independent_campaign_groups": campaigns,
                        "parallel_independent_sites_per_day": sites,
                        "minimum_collection_days": math.ceil(campaigns / sites),
                        "independence_unit": (
                            "independently initiated terminal-site-day campaign; "
                            "repeated windows from one drive remain one group"
                        ),
                        "planning_boundary": (
                            "best-case non-vacuity floor under zero harmful groups; "
                            "not assurance of power under the field effect distribution"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/operator_collection_plan",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    plan = build_collection_plan()
    plan.to_csv(output / "operator_collection_plan.csv", index=False)
    metadata = {
        "profile_id": "commect_measured_60ms",
        "profile_registry": "configs/evidence_profiles.yaml",
        "settings_frozen_before_collection": True,
        "five_protected_endpoint_families_per_candidate_use": True,
        "independence_warning": (
            "same-drive windows must not be relabeled as independent campaigns"
        ),
        "power_warning": (
            "the exported floor is necessary for a non-vacuous best-case bound, "
            "not sufficient power for an unknown field effect"
        ),
        "simulation_power_artifact": (
            "results/campaign_admission_power_60ms/campaign_power_summary.csv"
        ),
        "independence_scope": (
            "terminal-site-day labels support a prospective design audit but do "
            "not prove independence; shared gateway, provider, weather, and "
            "regional shocks require stratification or higher-level clustering"
        ),
    }
    (output / "operator_collection_plan_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(plan.to_string(index=False))
    print(f"collection_plan_written={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
