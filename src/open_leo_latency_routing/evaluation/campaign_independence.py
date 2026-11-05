"""Provenance checks for prospective independent campaign groups.

The checks enforce a collection design; they cannot prove statistical
independence from observed outcomes. Independence remains an experimental
assumption supported by separate terminal/site/day starts and operational
separation, not by splitting one time series into smaller windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_CAMPAIGN_FIELDS = (
    "campaign_id",
    "terminal_id",
    "site_id",
    "collection_date",
    "randomized_start_id",
)


@dataclass(frozen=True)
class CampaignIndependenceAudit:
    campaign_count: int
    unique_terminal_count: int
    unique_site_count: int
    unique_date_count: int
    repeated_terminal_site_day_count: int
    design_valid: bool
    interpretation: str


def audit_campaign_manifest(manifest: pd.DataFrame) -> CampaignIndependenceAudit:
    """Fail closed when campaign provenance cannot support group inference."""

    missing_columns = [
        field for field in REQUIRED_CAMPAIGN_FIELDS if field not in manifest.columns
    ]
    if missing_columns:
        raise ValueError(f"missing campaign provenance fields: {missing_columns}")
    if manifest.empty:
        raise ValueError("campaign manifest must contain at least one campaign")
    if manifest.loc[:, REQUIRED_CAMPAIGN_FIELDS].isna().any().any():
        raise ValueError("campaign provenance fields must not contain missing values")
    if manifest["campaign_id"].duplicated().any():
        raise ValueError("campaign_id must be unique; windows cannot become groups")

    terminal_site_day = ["terminal_id", "site_id", "collection_date"]
    repeated = int(manifest.duplicated(terminal_site_day, keep=False).sum())
    duplicated_start = manifest.duplicated(
        terminal_site_day + ["randomized_start_id"],
        keep=False,
    )
    if duplicated_start.any():
        raise ValueError(
            "terminal-site-day campaigns must have distinct randomized starts"
        )

    return CampaignIndependenceAudit(
        campaign_count=int(len(manifest)),
        unique_terminal_count=int(manifest["terminal_id"].nunique()),
        unique_site_count=int(manifest["site_id"].nunique()),
        unique_date_count=int(manifest["collection_date"].nunique()),
        repeated_terminal_site_day_count=repeated,
        design_valid=True,
        interpretation=(
            "Provenance supports separately initiated campaign groups. This is "
            "a design audit, not an empirical test or proof of independence; "
            "shared weather, gateway, provider, or regional failures must be "
            "handled by stratification, sensitivity analysis, or a higher-level "
            "cluster definition before admission."
        ),
    )
