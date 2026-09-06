from pathlib import Path

import numpy as np
import pandas as pd

from open_leo_latency_routing.optimization.calibrated_risk import (
    add_calibrated_mixture_risk_scores,
    fit_expert_calibration,
)
from scripts.run_wetlinks_longitudinal_validation import (
    _attach_paired_residual_covariance,
    _time_split,
)


def test_wetlinks_builder_marks_distributed_sites_as_non_policy_data(tmp_path):
    from scripts.build_wetlinks_longitudinal_table import build_table

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    base = {
        "timestamp_end": ["2024-01-01 00:00:49", "2024-01-01 00:05:49"],
        "server": ["example", "example"],
        "ping_packet_loss": [0.0, 4.0],
        "ping_packets_send": [250, 250],
        "ping_avg": [50.0, 55.0],
        "ping_worst": [80.0, 90.0],
        "ping_stddev": [5.0, 6.0],
        "download": [100e6, 90e6],
        "upload": [10e6, 9e6],
    }
    for site in ("site_a", "site_b"):
        frame = pd.DataFrame(
            {
                **base,
                "site_name": [site, site],
                "timestamp_start": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:05:00",
                ],
            }
        )
        frame.to_csv(input_dir / f"analysis_data_{site}.csv", index=False)

    table, metadata = build_table(Path(input_dir), bin_minutes=5)

    assert len(table) == 4
    assert metadata["shared_epoch_count"] == 2
    assert metadata["has_temporally_concurrent_candidates"] is False
    assert metadata["supports_candidate_outcome_shadow_replay"] is False
    assert metadata["supports_literal_single_controller_steering"] is False
    assert metadata["candidate_set_semantics"] == (
        "distributed_observations_not_interchangeable_paths"
    )
    assert table["observed_replies"].min() == 240.0
    assert sorted(table["bin_epoch"].unique().tolist()) == [
        1704067200,
        1704067500,
    ]
    assert table.groupby("relative_path")["bin_epoch"].diff().dropna().eq(300).all()


def test_wetlinks_fusion_uses_paired_calibration_residual_covariance():
    truth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    temporal_prediction = truth - np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    context_prediction = truth - np.array([2.0, 4.0, 3.0, 7.0, 8.0])
    temporal = fit_expert_calibration(truth, temporal_prediction)
    context = fit_expert_calibration(truth, context_prediction)
    temporal, context = _attach_paired_residual_covariance(
        temporal,
        context,
        truth,
        temporal_prediction,
        context_prediction,
    )

    covariance = temporal.paired_residual_covariance_ms2
    assert covariance != 0.0
    assert context.paired_residual_covariance_ms2 == covariance

    output = add_calibrated_mixture_risk_scores(
        pd.DataFrame({"pred_forecast": [25.0], "pred_graph": [35.0]}),
        temporal,
        context,
    )
    temporal_weight = float(output["temporal_expert_weight"].iloc[0])
    context_weight = float(output["graph_expert_weight"].iloc[0])
    expected_variance = (
        temporal_weight**2 * float(temporal.residual_variance_ms2)
        + context_weight**2 * float(context.residual_variance_ms2)
        + 2.0 * temporal_weight * context_weight * covariance
    )
    assert np.isclose(
        float(output["pred_fusion_error_std"].iloc[0]) ** 2,
        expected_variance,
    )


def test_wetlinks_time_split_closes_global_future_target_boundaries():
    rows = []
    for relative_path in ("site_a", "site_b"):
        for epoch in range(20):
            rows.append(
                {
                    "relative_path": relative_path,
                    "bin_epoch": epoch,
                    "target_next_bin_epoch": epoch + 1,
                    "target_available_2": 1,
                    "target_cumulative_2": 20.0,
                    "target_mean_2": 10.0,
                    "target_end_bin_epoch_2": epoch + 2,
                }
            )
    frame = pd.DataFrame(rows)

    train, calibration, selection, test, metadata = _time_split(frame)

    # The last current epoch in each internal block is excluded because its
    # one-step label belongs to the following wall-clock block.
    assert not train["bin_epoch"].eq(11).any()
    assert not calibration["bin_epoch"].eq(14).any()
    assert not selection["bin_epoch"].eq(16).any()
    assert test["bin_epoch"].min() == 17
    # A longer endpoint crossing the same boundary is cleared without deleting
    # the row when its one-step label remains closed.
    train_epoch_10 = train["bin_epoch"].eq(10)
    assert train.loc[train_epoch_10, "target_available_2"].eq(0).all()
    assert train.loc[
        train_epoch_10,
        ["target_cumulative_2", "target_mean_2"],
    ].isna().all().all()
    assert metadata["global_wall_clock_partitions"] is True
    assert metadata["future_target_boundary_guard"] is True
    assert metadata["one_step_boundary_crossing_rows_removed"] == 6
