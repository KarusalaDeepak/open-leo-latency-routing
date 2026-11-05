#!/usr/bin/env python3
"""Run the frozen five-campaign CoNEXT'23 external evidence-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from open_leo_latency_routing.evaluation.conext23_external_gate import (
    ALL_SOURCE_PROVIDERS,
    DEVELOPMENT_DATE,
    EVIDENCE_DATES,
    PRIMARY_PROVIDERS,
    ROAM_SENSITIVITY_PROVIDERS,
    add_exact_forecast_features,
    audit_provider_date_coverage,
    build_concurrent_seconds,
    fit_frozen_models,
    score_frozen_candidate,
)
from open_leo_latency_routing.evaluation.risk_metrics import empirical_upper_cvar
from open_leo_latency_routing.optimization.risk_control import (
    RiskControlConfig,
    select_opportunity_aware_risk_controlled_policy,
)


PRIMARY_RIDGE_ALPHA = 1.0
RIDGE_ALPHA_SENSITIVITY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_from_decisions(decisions: pd.DataFrame):
    """Apply the frozen gate to already scored decision-level outcomes."""

    required = {
        "campaign_id",
        "reactive_latency_ms",
        "candidate_latency_ms",
        "generic_opportunity",
        "policy_active_opportunity",
    }
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"decision export is missing required columns: {missing}")
    campaign_counts = decisions.groupby("campaign_id").size()
    if len(campaign_counts) != 5 or bool(campaign_counts.le(0).any()):
        raise AssertionError("the external gate requires five non-empty campaigns")
    return select_opportunity_aware_risk_controlled_policy(
        {
            "reactive": decisions["reactive_latency_ms"].tolist(),
            "frozen_external_candidate": decisions[
                "candidate_latency_ms"
            ].tolist(),
        },
        decisions["generic_opportunity"].astype(bool).to_numpy(),
        latency_budget_ms=60.0,
        config=RiskControlConfig(
            alpha=0.05,
            noninferiority_margin=0.02,
            opportunity_noninferiority_margin=0.02,
            minimum_effective_opportunities=5.0,
            practical_cvar_gain_ms=5.0,
            cvar_quantile=0.90,
            latency_cap_ms=250.0,
            cvar_grid_points=5001,
            planned_gate_uses=1,
            gate_use_index=1,
        ),
        reactive_policy="reactive",
        independence_group_ids=decisions["campaign_id"].tolist(),
        policy_active_opportunity_masks={
            "frozen_external_candidate": decisions[
                "policy_active_opportunity"
            ].astype(bool).to_numpy()
        },
    )


def _policy_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    populations = [("pooled", decisions), *decisions.groupby("campaign_id", sort=True)]
    for campaign_id, frame in populations:
        for policy in ("reactive", "candidate"):
            latency = frame[f"{policy}_latency_ms"].to_numpy(dtype=float)
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "policy": policy,
                    "decision_count": int(len(frame)),
                    "success_count": int(frame[f"{policy}_success"].sum()),
                    "success_rate": float(frame[f"{policy}_success"].mean()),
                    "mean_rtt_ms": float(np.mean(latency)),
                    "empirical_cvar90_ms": empirical_upper_cvar(latency, 0.90),
                    "generic_opportunity_count": int(
                        frame["generic_opportunity"].sum()
                    ),
                    "policy_active_opportunity_count": int(
                        frame["policy_active_opportunity"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_profile(
    dataset_root: Path,
    output_dir: Path,
    providers: tuple[str, ...],
    *,
    prefix: str,
    ridge_alpha: float = PRIMARY_RIDGE_ALPHA,
) -> dict[str, object]:
    """Run one frozen provider-set profile and archive its exact outputs."""

    concurrent, source_metadata = build_concurrent_seconds(dataset_root, providers)
    exact = add_exact_forecast_features(concurrent, providers)
    development = exact[exact["collection_date"].eq(DEVELOPMENT_DATE)]
    evidence = exact[exact["collection_date"].isin(EVIDENCE_DATES)]
    observed_dates = tuple(sorted(evidence["collection_date"].unique()))
    if observed_dates != EVIDENCE_DATES:
        raise ValueError(
            f"expected evidence dates {EVIDENCE_DATES}, observed {observed_dates} "
            f"for providers {providers}"
        )

    models, model_manifest = fit_frozen_models(
        development,
        providers,
        ridge_alpha=ridge_alpha,
    )
    decisions = score_frozen_candidate(evidence, models, providers)
    gate = _gate_from_decisions(decisions)

    policy_summary = _policy_summary(decisions)
    gate_evidence = gate.evidence_frame()
    decisions.to_csv(output_dir / f"{prefix}decision_outputs.csv", index=False)
    policy_summary.to_csv(output_dir / f"{prefix}policy_summary.csv", index=False)
    gate_evidence.to_csv(output_dir / f"{prefix}gate_evidence.csv", index=False)
    model_manifest.update(
        {
            "candidate_name": "frozen_calibration_weighted_temporal_context_ridge",
            "provider_set": providers,
            "fusion_weight_selection": (
                "bounded Bates-Granger minimum-variance weights estimated "
                "from paired April 23 calibration residuals only"
            ),
            "latency_target_ms": 60.0,
            "current_state_qos_filter": (
                "rank only currently compliant paths when at least one exists"
            ),
            "evidence_dates": EVIDENCE_DATES,
            "evidence_outcomes_used_for_tuning": False,
            "ridge_alpha_role": (
                "fixed engineering value for the primary audit; not selected "
                "from the evidence campaigns"
            ),
        }
    )
    (output_dir / f"{prefix}model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2), encoding="utf-8"
    )
    return {
        "providers": providers,
        "source_metadata": source_metadata,
        "decisions": decisions,
        "policy_summary": policy_summary,
        "gate_evidence": gate_evidence,
        "gate": gate,
        "model_manifest": model_manifest,
    }


def _run_ridge_alpha_sensitivity(
    dataset_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Report, but never select from, a fixed ridge-alpha robustness grid."""

    concurrent, _ = build_concurrent_seconds(dataset_root, PRIMARY_PROVIDERS)
    exact = add_exact_forecast_features(concurrent, PRIMARY_PROVIDERS)
    development = exact[exact["collection_date"].eq(DEVELOPMENT_DATE)]
    evidence = exact[exact["collection_date"].isin(EVIDENCE_DATES)]
    rows: list[dict[str, object]] = []
    for ridge_alpha in RIDGE_ALPHA_SENSITIVITY_GRID:
        models, manifest = fit_frozen_models(
            development,
            PRIMARY_PROVIDERS,
            ridge_alpha=ridge_alpha,
        )
        decisions = score_frozen_candidate(
            evidence,
            models,
            PRIMARY_PROVIDERS,
        )
        gate = _gate_from_decisions(decisions)
        summary = _policy_summary(decisions)
        reactive = summary[
            summary["campaign_id"].eq("pooled")
            & summary["policy"].eq("reactive")
        ].iloc[0]
        candidate = summary[
            summary["campaign_id"].eq("pooled")
            & summary["policy"].eq("candidate")
        ].iloc[0]
        evidence_row = gate.evidence_frame()[
            lambda frame: frame["policy"].eq("frozen_external_candidate")
        ].iloc[0]
        rows.append(
            {
                "ridge_alpha": ridge_alpha,
                "primary_predeclared_value": ridge_alpha == PRIMARY_RIDGE_ALPHA,
                "decision_count": int(candidate["decision_count"]),
                "reactive_success_count": int(reactive["success_count"]),
                "candidate_success_count": int(candidate["success_count"]),
                "candidate_success_rate": float(candidate["success_rate"]),
                "candidate_mean_rtt_ms": float(candidate["mean_rtt_ms"]),
                "candidate_empirical_cvar90_ms": float(
                    candidate["empirical_cvar90_ms"]
                ),
                "policy_active_harmful_campaign_count": int(
                    evidence_row[
                        "policy_active_opportunity_harmful_group_count"
                    ]
                ),
                "policy_active_campaign_count": int(
                    evidence_row["policy_active_opportunity_block_count"]
                ),
                "policy_active_success_lcb": float(
                    evidence_row[
                        "policy_active_opportunity_success_delta_lcb"
                    ]
                ),
                "capped_cvar_gain_point_ms": float(
                    evidence_row["group_uniform_cvar_gain_vs_reactive_ms"]
                ),
                "capped_cvar_gain_lcb_ms": float(
                    evidence_row["group_uniform_cvar_gain_lcb_ms"]
                ),
                "candidate_gate_state": evidence_row["candidate_gate_state"],
                "gate_failure_codes": evidence_row["gate_failure_codes"],
                "selected_policy": gate.selected_policy,
                "selection_reason_code": gate.reason_code,
                "temporal_fusion_weights": ";".join(
                    f"{provider}:{manifest['providers'][provider]['fusion_weights']['temporal']:.6f}"
                    for provider in PRIMARY_PROVIDERS
                ),
            }
        )
    result = pd.DataFrame.from_records(rows)
    result.to_csv(output_dir / "ridge_alpha_sensitivity.csv", index=False)
    return result


def _pooled_row(profile: dict[str, object], policy: str) -> pd.Series:
    summary = profile["policy_summary"]
    assert isinstance(summary, pd.DataFrame)
    return summary[
        summary["campaign_id"].eq("pooled") & summary["policy"].eq(policy)
    ].iloc[0]


def _recompute_archived_profile(
    output_dir: Path,
    *,
    prefix: str,
) -> dict[str, object]:
    """Recompute only gate evidence from an archived frozen decision export."""

    decision_path = output_dir / f"{prefix}decision_outputs.csv"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decisions = pd.read_csv(decision_path)
    gate = _gate_from_decisions(decisions)
    policy_summary = _policy_summary(decisions)
    gate_evidence = gate.evidence_frame()
    policy_summary.to_csv(output_dir / f"{prefix}policy_summary.csv", index=False)
    gate_evidence.to_csv(output_dir / f"{prefix}gate_evidence.csv", index=False)
    return {
        "decisions": decisions,
        "decision_path": decision_path,
        "decision_sha256": _sha256(decision_path),
        "policy_summary": policy_summary,
        "gate_evidence": gate_evidence,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        help="Path to the CoNEXT'23 UDPPingAll directory",
    )
    parser.add_argument(
        "--output-dir",
        default="results/conext23_five_campaign_gate",
    )
    parser.add_argument(
        "--recompute-gate-from-archived-decisions",
        action="store_true",
        help=(
            "Recompute policy summaries and gate evidence from the frozen "
            "decision-level CSVs already present in --output-dir. This does "
            "not rerun source parsing, model fitting, or path scoring."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recompute_gate_from_archived_decisions:
        primary = _recompute_archived_profile(output_dir, prefix="")
        roam = _recompute_archived_profile(
            output_dir,
            prefix="roam_sensitivity_",
        )
        for metadata_name, profile in (
            ("validation_metadata.json", primary),
            ("roam_sensitivity_validation_metadata.json", roam),
        ):
            metadata_path = output_dir / metadata_name
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            candidate = profile["gate_evidence"]
            assert isinstance(candidate, pd.DataFrame)
            candidate_row = candidate[
                candidate["policy"].eq("frozen_external_candidate")
            ].iloc[0]
            metadata.update(
                {
                    "gate_selected_policy": profile["gate"].selected_policy,
                    "gate_selection_reason": profile["gate"].reason,
                    "gate_selection_reason_code": profile["gate"].reason_code,
                    "candidate_eligible": bool(candidate_row["eligible"]),
                    "gate_recomputation": {
                        "mode": "archived_frozen_decision_outputs",
                        "decision_file": profile["decision_path"].name,
                        "decision_sha256": profile["decision_sha256"],
                        "scope": (
                            "recomputes the revised gate only; source parsing, "
                            "model fitting, and path scoring are unchanged"
                        ),
                    },
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        print(primary["policy_summary"].to_string(index=False))
        print()
        print(primary["gate_evidence"].to_string(index=False))
        print(f"external_gate_outputs={output_dir}")
        return 0

    if not args.dataset_root:
        parser.error(
            "--dataset-root is required unless "
            "--recompute-gate-from-archived-decisions is used"
        )
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    coverage = audit_provider_date_coverage(dataset_root, ALL_SOURCE_PROVIDERS)
    coverage.to_csv(output_dir / "provider_date_coverage.csv", index=False)
    att_coverage = coverage[coverage["provider"].eq("att")].iloc[0]
    if att_coverage["missing_required_dates"] != "2023-04-26":
        raise AssertionError("AT&T date-coverage limitation changed")

    primary = _run_profile(
        dataset_root,
        output_dir,
        PRIMARY_PROVIDERS,
        prefix="",
        ridge_alpha=PRIMARY_RIDGE_ALPHA,
    )
    roam_sensitivity = _run_profile(
        dataset_root,
        output_dir,
        ROAM_SENSITIVITY_PROVIDERS,
        prefix="roam_sensitivity_",
        ridge_alpha=PRIMARY_RIDGE_ALPHA,
    )
    ridge_alpha_sensitivity = _run_ridge_alpha_sensitivity(
        dataset_root,
        output_dir,
    )

    gate = primary["gate"]
    gate_evidence = primary["gate_evidence"]
    assert isinstance(gate_evidence, pd.DataFrame)
    candidate_evidence = gate_evidence[
        gate_evidence["policy"].eq("frozen_external_candidate")
    ].iloc[0]
    sensitivity_gate = roam_sensitivity["gate_evidence"]
    assert isinstance(sensitivity_gate, pd.DataFrame)
    sensitivity_candidate = sensitivity_gate[
        sensitivity_gate["policy"].eq("frozen_external_candidate")
    ].iloc[0]
    sensitivity_reactive = _pooled_row(roam_sensitivity, "reactive")
    sensitivity_policy = _pooled_row(roam_sensitivity, "candidate")
    metadata = {
        "source": "Hu et al., CoNEXT Companion 2023",
        "source_repository": (
            "https://github.com/Starlink-Project/Satellite-vs-Cellular"
        ),
        "source_repository_commit": "79f9412cf6888de1b44a87576c308a9f26539169",
        "source_doi": "10.1145/3624354.3630588",
        "source_files": primary["source_metadata"],
        "development_date": DEVELOPMENT_DATE,
        "evidence_dates": EVIDENCE_DATES,
        "qualified_campaign_count": 5,
        "qualified_acquisition_family_count": 1,
        "qualification_status": "conditional",
        "campaign_definition": (
            "one separately restarted calendar-day road-test collection"
        ),
        "independence_and_exchangeability_assumption": (
            "the five daily collection groups are treated as independent and "
            "exchangeable draws from the declared road-test campaign population"
        ),
        "common_endpoint_limitation": (
            "the release documents one UDP-Ping application used on side-by-side "
            "devices but does not export the destination server identifier"
        ),
        "conditional_qualification_basis": (
            "a common destination is assumed from the shared UDP-Ping collection "
            "description; the result is not a fully endpoint-verified campaign audit"
        ),
        "primary_provider_set": PRIMARY_PROVIDERS,
        "primary_provider_selection_rule": (
            "retrospective coverage-only qualification: use Starlink Mobility as "
            "the primary LEO service and cellular services with timestamped files "
            "on the development date and every declared evidence date; AT&T lacks "
            "April 26 coverage, while Starlink Roam is evaluated in a separately "
            "archived four-path sensitivity"
        ),
        "provider_set_qualification_uses_evidence_date_metadata": True,
        "provider_set_qualification_uses_rtt_values_or_outcomes": False,
        "provider_set_freeze_scope": (
            "model features, coefficients, target, and gate thresholds are frozen "
            "from April 23; provider eligibility is a retrospective file-coverage "
            "audit across the declared dates and is not claimed as prospectively "
            "frozen before April 24"
        ),
        "provider_date_coverage_file": "provider_date_coverage.csv",
        "primary_ridge_alpha": PRIMARY_RIDGE_ALPHA,
        "ridge_alpha_selection_rule": (
            "alpha=1.0 is fixed for the primary audit; held-out evidence outcomes "
            "do not select alpha"
        ),
        "ridge_alpha_sensitivity_grid": RIDGE_ALPHA_SENSITIVITY_GRID,
        "ridge_alpha_sensitivity_file": "ridge_alpha_sensitivity.csv",
        "ridge_alpha_sensitivity_role": (
            "post-specified robustness audit only; no row replaces the primary "
            "alpha=1.0 result"
        ),
        "roam_sensitivity": {
            "provider_set": ROAM_SENSITIVITY_PROVIDERS,
            "decision_count": int(sensitivity_reactive["decision_count"]),
            "reactive_success_count": int(sensitivity_reactive["success_count"]),
            "candidate_success_count": int(sensitivity_policy["success_count"]),
            "reactive_success_rate": float(sensitivity_reactive["success_rate"]),
            "candidate_success_rate": float(sensitivity_policy["success_rate"]),
            "reactive_empirical_cvar90_ms": float(
                sensitivity_reactive["empirical_cvar90_ms"]
            ),
            "candidate_empirical_cvar90_ms": float(
                sensitivity_policy["empirical_cvar90_ms"]
            ),
            "candidate_eligible": bool(sensitivity_candidate["eligible"]),
            "selected_policy": roam_sensitivity["gate"].selected_policy,
        },
        "timeout_limitation": (
            "source timeout rows omit timestamps and cannot be aligned; the "
            "experiment is a complete-case RTT audit over timestamped replies"
        ),
        "access_scope": (
            "cellular--LEO mobility evidence; the UDP-Ping release does not "
            "export a per-sample LTE-versus-5G technology label"
        ),
        "gate_selected_policy": gate.selected_policy,
        "gate_selection_reason": gate.reason,
        "gate_selection_reason_code": gate.reason_code,
        "candidate_eligible": bool(candidate_evidence["eligible"]),
        "valid_claim": (
            "conditional five-group external applicability and admission audit "
            "under the declared common-endpoint and day-level sampling assumptions"
        ),
        "invalid_claims": [
            "five independent acquisition families",
            "5G-only external validation",
            "live steering or deployment benefit",
            "automatic transfer to future routes or operators",
        ],
    }
    (output_dir / "validation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    sensitivity_metadata = {
        "profile": "four_path_starlink_roam_sensitivity",
        "provider_set": ROAM_SENSITIVITY_PROVIDERS,
        "development_date": DEVELOPMENT_DATE,
        "evidence_dates": EVIDENCE_DATES,
        "evidence_outcomes_used_for_tuning": False,
        "decision_count": int(sensitivity_reactive["decision_count"]),
        "gate_selected_policy": roam_sensitivity["gate"].selected_policy,
        "gate_selection_reason_code": roam_sensitivity["gate"].reason_code,
        "candidate_eligible": bool(sensitivity_candidate["eligible"]),
        "scope": (
            "path-set sensitivity under the same conditional common-endpoint and "
            "day-level independence/exchangeability assumptions as the primary audit"
        ),
    }
    (output_dir / "roam_sensitivity_validation_metadata.json").write_text(
        json.dumps(sensitivity_metadata, indent=2), encoding="utf-8"
    )

    policy_summary = primary["policy_summary"]
    assert isinstance(policy_summary, pd.DataFrame)
    print(policy_summary.to_string(index=False))
    print()
    print(
        gate_evidence[
            [
                "policy",
                "inference_block_count",
                "opportunity_block_count",
                "policy_active_opportunity_block_count",
                "group_uniform_success_delta_vs_reactive",
                "aggregate_actionable_success_delta_lcb",
                "opportunity_conditioned_success_delta_lcb",
                "policy_active_opportunity_success_delta_lcb",
                "group_uniform_cvar_gain_lcb_ms",
                "eligible",
                "selected",
                "selection_reason",
            ]
        ].to_string(index=False)
    )
    print("\nRoam-inclusive four-path sensitivity")
    sensitivity_summary = roam_sensitivity["policy_summary"]
    assert isinstance(sensitivity_summary, pd.DataFrame)
    print(
        sensitivity_summary[sensitivity_summary["campaign_id"].eq("pooled")]
        .to_string(index=False)
    )
    print("\nRidge-alpha sensitivity (reporting only; primary alpha remains 1.0)")
    print(ridge_alpha_sensitivity.to_string(index=False))
    print(f"external_gate_outputs={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
