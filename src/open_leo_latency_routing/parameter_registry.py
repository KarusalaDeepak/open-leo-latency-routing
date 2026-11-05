"""Machine-checkable parameter provenance and evidence-tier definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PARAMETER_FIELDS = {
    "name",
    "value",
    "unit",
    "category",
    "selection_partition",
    "rationale",
    "alternatives",
    "claim_boundary",
}


def load_parameter_registry(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the active coefficient registry."""

    with Path(path).open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported parameter-registry schema")
    parameters = registry.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("parameter registry must contain a non-empty list")
    names: set[str] = set()
    for row in parameters:
        if not isinstance(row, dict):
            raise ValueError("parameter-registry rows must be mappings")
        missing = REQUIRED_PARAMETER_FIELDS.difference(row)
        if missing:
            raise ValueError(
                f"parameter {row.get('name', '<unknown>')} is missing {sorted(missing)}"
            )
        name = str(row["name"])
        if name in names:
            raise ValueError(f"duplicate parameter-registry entry: {name}")
        names.add(name)
        if not str(row["rationale"]).strip() or not str(row["claim_boundary"]).strip():
            raise ValueError(f"parameter {name} lacks rationale or claim boundary")
        alternatives = row["alternatives"]
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"parameter {name} lacks bounded alternatives")
    return registry


@dataclass(frozen=True)
class EvidenceTierAssessment:
    tier: str
    campaign_count: int
    family_count: int
    deployment_release_permitted: bool
    reasons: tuple[str, ...]


def assess_evidence_tier(
    campaign_ids: list[object],
    *,
    acquisition_family_ids: list[object] | None = None,
    frozen_external_test: bool = False,
    leave_one_campaign_out: bool = False,
    paired_opportunities_present: bool = False,
    campaign_block_precision_passed: bool = False,
    calibration_passed: bool = False,
    shift_check_passed: bool = False,
    execution_cost_check_passed: bool = False,
) -> EvidenceTierAssessment:
    """Classify evidence without turning campaign count into a guarantee.

    One campaign supports proof-of-method. Two independently acquired
    campaigns can support a frozen external test. Five campaigns are an
    optional conservative broader-generalization profile when leave-one-
    campaign-out analysis is performed. Deployment release additionally
    requires paired opportunities, precision, calibration, shift, and
    execution-cost checks; the count never substitutes for those conditions.
    """

    campaigns = {str(value) for value in campaign_ids}
    families = (
        {str(value) for value in acquisition_family_ids}
        if acquisition_family_ids is not None
        else set()
    )
    campaign_count = len(campaigns)
    family_count = len(families)
    if campaign_count == 0:
        tier = "no_campaign_evidence"
    elif campaign_count == 1:
        tier = "proof_of_method"
    elif frozen_external_test:
        tier = "frozen_external_validation"
    else:
        tier = "multi_campaign_descriptive"
    if campaign_count >= 5 and leave_one_campaign_out:
        tier = "conservative_broader_generalization_profile"

    checks = {
        "paired opportunities": paired_opportunities_present,
        "campaign-block precision": campaign_block_precision_passed,
        "calibration": calibration_passed,
        "distribution-shift check": shift_check_passed,
        "execution-cost check": execution_cost_check_passed,
        "frozen external test": frozen_external_test,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    release = campaign_count >= 2 and not failed
    reasons = (
        ("all deployment-release checks passed",)
        if release
        else tuple(f"missing or failed: {name}" for name in failed)
    )
    return EvidenceTierAssessment(
        tier=tier,
        campaign_count=campaign_count,
        family_count=family_count,
        deployment_release_permitted=release,
        reasons=reasons,
    )
