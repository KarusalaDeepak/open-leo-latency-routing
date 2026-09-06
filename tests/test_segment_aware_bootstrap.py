"""Regression tests for segment-safe circular moving-block resampling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from open_leo_latency_routing.evaluation.confidence_intervals import (
    build_bootstrap_policy_intervals,
)
from open_leo_latency_routing.evaluation.significance import (
    SEGMENTED_CIRCULAR_BLOCK_METHOD,
    _moving_block_means,
    build_bootstrap_segment_ids,
    build_paired_policy_significance,
    sample_segmented_circular_block_indices,
)


class SegmentAwareBootstrapTests(unittest.TestCase):
    def test_blocks_and_circular_wraps_remain_inside_each_segment(self) -> None:
        segment_ids = ["first"] * 3 + ["second"] * 4
        sampled = sample_segmented_circular_block_indices(
            segment_ids,
            n_bootstrap=128,
            random_state=7,
            block_length=3,
        )

        self.assertEqual(sampled.shape, (128, 7))
        self.assertTrue(np.isin(sampled[:, :3], [0, 1, 2]).all())
        self.assertTrue(np.isin(sampled[:, 3:], [3, 4, 5, 6]).all())

        # The first segment is exactly one length-three block. It advances
        # cyclically inside [0, 1, 2], including observed 2 -> 0 wraps.
        first = sampled[:, :3]
        self.assertTrue((((first[:, 1:] - first[:, :-1]) % 3) == 1).all())
        self.assertTrue(((first[:, :-1] == 2) & (first[:, 1:] == 0)).any())

        # The first complete block from the second segment likewise advances
        # only inside its own local four-position circle.
        second_first_block = sampled[:, 3:6] - 3
        self.assertTrue(
            (((second_first_block[:, 1:] - second_first_block[:, :-1]) % 4) == 1).all()
        )
        self.assertTrue(
            (
                (second_first_block[:, :-1] == 3)
                & (second_first_block[:, 1:] == 0)
            ).any()
        )

    def test_noncontiguous_repeated_segment_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one contiguous run"):
            sample_segmented_circular_block_indices(
                ["a", "a", "b", "b", "a"],
                n_bootstrap=10,
                random_state=1,
                block_length=2,
            )

    def test_metric_missingness_splits_an_otherwise_constant_segment(self) -> None:
        frame = pd.DataFrame(
            {
                "validated_segment": ["only"] * 5,
                "metric": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )
        valid = np.isfinite(frame["metric"].to_numpy(dtype=float))
        segment_ids = build_bootstrap_segment_ids(
            frame,
            ("validated_segment",),
            valid_mask=valid,
        )
        self.assertEqual(segment_ids.tolist(), [0, 0, 1, 1])

        sampled = sample_segmented_circular_block_indices(
            segment_ids,
            n_bootstrap=32,
            random_state=3,
            block_length=2,
        )
        self.assertTrue(np.isin(sampled[:, :2], [0, 1]).all())
        self.assertTrue(np.isin(sampled[:, 2:], [2, 3]).all())

    def test_unsegmented_default_is_rejected_but_explicit_single_segment_works(
        self,
    ) -> None:
        decisions = pd.DataFrame(
            {
                "policy_name": ["a"] * 4,
                "session_bin_index": [0, 1, 2, 3],
                "validated_segment": ["one"] * 4,
                "latency": [10.0, 20.0, 30.0, 40.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "segment_columns is required"):
            build_bootstrap_policy_intervals(
                decisions,
                ["latency"],
                n_bootstrap=20,
            )

        result = build_bootstrap_policy_intervals(
            decisions,
            ["latency"],
            n_bootstrap=20,
            segment_columns=("validated_segment",),
        )
        self.assertEqual(result["bootstrap_segment_count"].tolist(), [1])
        self.assertEqual(
            result["bootstrap_method"].tolist(),
            [SEGMENTED_CIRCULAR_BLOCK_METHOD],
        )

    def test_explicit_single_segment_preserves_legacy_draws(self) -> None:
        values = np.asarray([2.0, 4.0, 8.0, 16.0, 32.0])
        draw_count = 17
        block_length = 2
        random_state = 19

        # Reconstruct the prior one-series circular moving-block algorithm.
        rng = np.random.default_rng(random_state)
        blocks_per_draw = int(np.ceil(len(values) / block_length))
        starts = rng.integers(
            0,
            len(values),
            size=(draw_count, blocks_per_draw),
        )
        offsets = np.arange(block_length)
        legacy_indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % len(values)
        legacy_indices = legacy_indices.reshape(draw_count, -1)[:, : len(values)]
        legacy_means = values[legacy_indices].mean(axis=1)

        observed = _moving_block_means(
            values,
            segment_ids=["validated-one-segment"] * len(values),
            n_bootstrap=draw_count,
            random_state=random_state,
            block_length=block_length,
        )
        np.testing.assert_array_equal(observed, legacy_means)

    def test_rolling_fold_and_continuity_segments_are_both_audited(self) -> None:
        base = pd.DataFrame(
            {
                "session_bin_index": [0, 1, 2, 3, 4, 5],
                "rolling_fold": [1, 1, 1, 2, 2, 2],
                # The evaluator restarts this local counter in every fold.
                "continuity_segment_id": [1, 1, 2, 1, 1, 1],
                "latency": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        decisions = pd.concat(
            [
                base.assign(policy_name="left"),
                base.assign(policy_name="right", latency=base["latency"] + 1.0),
            ],
            ignore_index=True,
        )
        result = build_paired_policy_significance(
            decisions,
            [("left_vs_right", "left", "right")],
            ["latency"],
            n_bootstrap=30,
            block_length=2,
            segment_columns=("rolling_fold", "continuity_segment_id"),
        )
        self.assertEqual(result["bootstrap_segment_count"].tolist(), [3])
        self.assertEqual(
            result["bootstrap_segment_columns"].tolist(),
            ["rolling_fold|continuity_segment_id"],
        )

    def test_paired_segment_assignment_mismatch_fails_closed(self) -> None:
        decisions = pd.DataFrame(
            {
                "policy_name": ["left"] * 3 + ["right"] * 3,
                "session_bin_index": [0, 1, 2, 0, 1, 2],
                "continuity_segment_id": [1, 1, 1, 1, 2, 2],
                "latency": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "identical ordered decision keys"):
            build_paired_policy_significance(
                decisions,
                [("left_vs_right", "left", "right")],
                ["latency"],
                n_bootstrap=20,
                segment_columns=("continuity_segment_id",),
            )

    def test_low_level_sampler_validates_lengths_missing_ids_and_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "length must equal"):
            _moving_block_means(
                np.asarray([1.0, 2.0]),
                segment_ids=["one"],
                n_bootstrap=10,
                random_state=1,
                block_length=1,
            )
        with self.assertRaisesRegex(ValueError, "must not contain missing"):
            sample_segmented_circular_block_indices(
                ["one", None],
                n_bootstrap=10,
                random_state=1,
                block_length=1,
            )
        with self.assertRaisesRegex(ValueError, "values must be finite"):
            _moving_block_means(
                np.asarray([1.0, np.inf]),
                segment_ids=["one", "one"],
                n_bootstrap=10,
                random_state=1,
                block_length=1,
            )


if __name__ == "__main__":
    unittest.main()
