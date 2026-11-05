"""Strict campaign and acquisition-family accounting for policy evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_campaign_qualification(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported campaign-qualification schema")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("campaign-qualification manifest requires sources")
    required = {
        "source",
        "campaign_id",
        "acquisition_family",
        "concurrent_candidate_paths",
        "paired_next_interval_outcomes",
        "actionable_opportunity_audit",
        "gate_qualified",
        "role",
        "limitation",
    }
    for row in sources:
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"campaign source {row.get('source', '<unknown>')} is missing {sorted(missing)}"
            )
        if row["gate_qualified"]:
            if row["concurrent_candidate_paths"] is not True:
                raise ValueError(
                    "gate-qualified campaigns must have concurrent candidate paths"
                )
            if row["paired_next_interval_outcomes"] is not True:
                raise ValueError("gate-qualified campaigns require paired outcomes")
            if row["actionable_opportunity_audit"] is not True:
                raise ValueError("gate-qualified campaigns require opportunity audit")
            if not row["campaign_id"]:
                raise ValueError("gate-qualified campaigns require a campaign id")
    return manifest


def summarize_gate_qualified_evidence(
    manifest: dict[str, Any],
) -> dict[str, object]:
    """Return campaign and acquisition-family counts without conflating them."""

    qualified = [row for row in manifest["sources"] if row["gate_qualified"]]
    campaign_ids = {str(row["campaign_id"]) for row in qualified}
    family_ids = {str(row["acquisition_family"]) for row in qualified}
    return {
        "qualified_campaign_count": len(campaign_ids),
        "qualified_acquisition_family_count": len(family_ids),
        "qualified_campaign_ids": tuple(sorted(campaign_ids)),
        "qualified_acquisition_families": tuple(sorted(family_ids)),
        "nonqualified_sources": tuple(
            sorted({str(row["source"]) for row in manifest["sources"] if not row["gate_qualified"]})
        ),
    }


def require_unique_gate_campaign_id(
    manifest: dict[str, Any],
    *,
    source: str,
) -> str:
    """Return the sole qualified campaign id for ``source`` or fail closed.

    Production policy runners use this helper instead of deriving inferential
    groups from dates, folds, or time blocks.  A missing or ambiguous manifest
    entry is an applicability failure: it must never silently manufacture
    independent evidence.
    """

    rows = [
        row
        for row in manifest["sources"]
        if str(row["source"]).casefold() == source.casefold()
        and row["gate_qualified"] is True
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one gate-qualified {source} campaign; "
            f"found {len(rows)}"
        )
    campaign_id = rows[0].get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError(
            f"gate-qualified {source} entry requires a non-empty campaign id"
        )
    return campaign_id.strip()
