from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from open_leo_latency_routing.evaluation.conext23_external_gate import (
    add_exact_forecast_features,
    audit_provider_date_coverage,
    build_concurrent_seconds,
    minimum_variance_fusion_weights,
)


def _write_provider(path: Path, provider: str, timestamps: list[str]) -> None:
    frame = pd.DataFrame(
        {
            "time": [*timestamps, "timeout"],
            "RTT": [*(str(40 + index) for index in range(len(timestamps))), "timeo"],
        }
    )
    frame.to_csv(path / f"{provider}_udpping_summary.csv", index=False)


def test_external_adapter_uses_exact_seconds_and_does_not_timestamp_timeouts(
    tmp_path: Path,
) -> None:
    timestamps = [
        "2023-04-24 10:00:00.100 CDT",
        "2023-04-24 10:00:01.100 CDT",
        "2023-04-24 10:00:02.100 CDT",
        "2023-04-24 10:00:04.100 CDT",
        "2023-04-24 10:00:05.100 CDT",
    ]
    for provider in ("mob", "tm", "vz"):
        _write_provider(tmp_path, provider, timestamps)

    concurrent, metadata = build_concurrent_seconds(tmp_path)
    exact = add_exact_forecast_features(concurrent)

    assert len(concurrent) == 5
    assert len(exact) == 0  # No row has two exact lags and an exact next second.
    assert {row["unassignable_timeout_or_malformed_row_count"] for row in metadata} == {1}


def test_external_adapter_assigns_one_campaign_id_per_calendar_date(
    tmp_path: Path,
) -> None:
    timestamps = [
        f"2023-04-{day:02d} 10:00:0{second}.100 CDT"
        for day in range(24, 29)
        for second in range(5)
    ]
    for provider in ("mob", "tm", "vz"):
        _write_provider(tmp_path, provider, timestamps)

    concurrent, _ = build_concurrent_seconds(tmp_path)

    assert concurrent["campaign_id"].nunique() == 5
    assert set(concurrent["campaign_id"]) == {
        f"conext23_202304{day:02d}" for day in range(24, 29)
    }


def test_provider_coverage_audit_reports_missing_required_date(tmp_path: Path) -> None:
    complete = [
        f"2023-04-{day:02d} 10:00:00.100 CDT" for day in range(23, 29)
    ]
    missing_april_26 = [
        timestamp for timestamp in complete if not timestamp.startswith("2023-04-26")
    ]
    _write_provider(tmp_path, "mob", complete)
    _write_provider(tmp_path, "att", missing_april_26)

    coverage = audit_provider_date_coverage(tmp_path, ("mob", "att"))
    indexed = coverage.set_index("provider")

    assert bool(indexed.loc["mob", "all_required_dates_present"])
    assert not bool(indexed.loc["att", "all_required_dates_present"])
    assert indexed.loc["att", "missing_required_dates"] == "2023-04-26"


def test_minimum_variance_weights_use_paired_calibration_residuals() -> None:
    temporal = np.array([2.0, -1.0, 3.0, -2.0, 1.0])
    context = np.array([4.0, -3.0, 1.0, -4.0, 2.0])

    temporal_weight, context_weight, diagnostics = minimum_variance_fusion_weights(
        temporal, context
    )

    temporal_variance = float(np.var(temporal, ddof=1))
    context_variance = float(np.var(context, ddof=1))
    covariance = float(np.cov(temporal, context, ddof=1)[0, 1])
    expected = np.clip(
        (context_variance - covariance)
        / (temporal_variance + context_variance - 2.0 * covariance),
        0.0,
        1.0,
    )
    assert temporal_weight == expected
    assert context_weight == 1.0 - expected
    assert temporal_weight + context_weight == 1.0
    assert diagnostics["method"] == "bounded_minimum_variance"


def test_minimum_variance_weights_have_deterministic_degenerate_fallback() -> None:
    residual = np.array([1.0, -1.0, 2.0, -2.0])

    temporal_weight, context_weight, diagnostics = minimum_variance_fusion_weights(
        residual, residual
    )

    assert temporal_weight == 0.5
    assert context_weight == 0.5
    assert diagnostics["method"] == "inverse_variance_degenerate_fallback"
