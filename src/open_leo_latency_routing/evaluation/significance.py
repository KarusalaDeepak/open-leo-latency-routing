"""Statistical comparison helpers for paired policy evaluations.

The descriptive bootstrap is segment-stratified: every resampled circular
moving block is contained within one explicitly supplied continuity segment.
This prevents a block (including its circular wrap) from crossing a telemetry
gap, session/campaign change, or rolling-fold boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEGMENTED_CIRCULAR_BLOCK_METHOD = (
    "segment_stratified_circular_moving_block"
)


@dataclass
class PairedSignificanceResult:
    """Stores one paired statistical comparison."""

    comparison_name: str
    metric_name: str
    sample_count: int
    mean_delta: float
    median_delta: float
    ci_lower: float
    ci_upper: float
    statistic: float
    p_value: float
    holm_adjusted_p_value: float
    effect_size_dz: float
    win_rate: float
    loss_rate: float
    block_bootstrap_p_value: float
    bootstrap_method: str
    bootstrap_segment_columns: str
    bootstrap_segment_count: int


def _normalize_segment_columns(
    segment_columns: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return a nonempty, duplicate-free tuple of segment column names."""

    if segment_columns is None:
        raise ValueError(
            "segment_columns is required for moving-block inference; pass "
            "an explicit constant segment column only after validating that "
            "the input is one uninterrupted telemetry segment"
        )
    if isinstance(segment_columns, str):
        normalized = (segment_columns,)
    else:
        normalized = tuple(segment_columns)
    if not normalized or any(
        not isinstance(column, str) or not column.strip()
        for column in normalized
    ):
        raise ValueError("segment_columns must contain nonempty column names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("segment_columns must not contain duplicates")
    return normalized


def _validated_segment_runs(
    segment_ids: Sequence[object] | np.ndarray,
    *,
    expected_length: int,
) -> list[np.ndarray]:
    """Validate ordered segment IDs and return their contiguous row indices.

    Each ID must occupy exactly one contiguous run. Reappearance after another
    ID is rejected because circular wrapping would otherwise join two pieces
    separated by an unrepresented boundary.
    """

    ids = pd.Series(list(segment_ids), dtype="object")
    if len(ids) != expected_length:
        raise ValueError(
            "segment_ids length must equal the number of bootstrap values: "
            f"{len(ids)} != {expected_length}"
        )
    if expected_length == 0:
        raise ValueError("moving-block bootstrap requires at least one value")
    if ids.isna().any():
        raise ValueError("segment_ids must not contain missing values")
    try:
        codes, _ = pd.factorize(ids, sort=False)
    except TypeError as exc:
        raise ValueError("segment_ids must contain hashable scalar values") from exc
    if (codes < 0).any():
        raise ValueError("segment_ids must not contain missing values")

    starts = np.flatnonzero(
        np.r_[True, codes[1:] != codes[:-1]]
    )
    ends = np.r_[starts[1:], expected_length]
    run_codes = codes[starts]
    if len(np.unique(run_codes)) != len(run_codes):
        raise ValueError(
            "each segment_id must form one contiguous run; a repeated "
            "noncontiguous ID would make circular wrapping cross a boundary"
        )
    return [
        np.arange(start, end, dtype=np.int64)
        for start, end in zip(starts, ends)
    ]


def build_bootstrap_segment_ids(
    frame: pd.DataFrame,
    segment_columns: str | Sequence[str] | None,
    *,
    valid_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Build validated ordered segment IDs from explicit DataFrame columns.

    `segment_columns` must encode every boundary relevant to the analysis. In
    this project, fixed protocols pass ``continuity_segment_id`` (which is
    reset on telemetry gaps and session/campaign changes), while pooled rolling
    validation additionally passes ``rolling_fold``.

    If `valid_mask` removes metric rows, each resulting positional gap starts a
    new bootstrap segment. Thus metric-specific missingness cannot reconnect
    observations that were not adjacent in the original decision sequence.
    Passing no segment columns is intentionally rejected. Legacy single-series
    behavior remains available by passing one explicit constant column after
    validating that the frame contains one uninterrupted segment.
    """

    columns = _normalize_segment_columns(segment_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"bootstrap segment columns are missing: {missing}")
    key_frame = frame.loc[:, list(columns)]
    if key_frame.isna().any().any():
        raise ValueError("bootstrap segment columns must not contain missing values")

    keys = pd.MultiIndex.from_frame(key_frame, names=list(columns))
    codes, _ = pd.factorize(keys, sort=False)
    # Validate the unfiltered source first. This catches a segment label that
    # disappears and later reappears before metric-specific filtering.
    _validated_segment_runs(codes, expected_length=len(frame))

    if valid_mask is None:
        mask = np.ones(len(frame), dtype=bool)
    else:
        mask = np.asarray(valid_mask)
        if mask.ndim != 1 or len(mask) != len(frame):
            raise ValueError(
                "valid_mask must be one-dimensional and match the frame length"
            )
        if mask.dtype != np.bool_:
            if not np.isin(mask, (0, 1)).all():
                raise ValueError("valid_mask must contain only boolean values")
            mask = mask.astype(bool)
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return np.asarray([], dtype=np.int64)
    filtered_codes = codes[positions]
    new_run = np.r_[
        True,
        (filtered_codes[1:] != filtered_codes[:-1])
        | (positions[1:] != positions[:-1] + 1),
    ]
    # Consecutive run numbers are safe even when a source segment is split by
    # missing metric values; each piece gets a distinct ID.
    segment_ids = np.cumsum(new_run, dtype=np.int64) - 1
    _validated_segment_runs(segment_ids, expected_length=len(segment_ids))
    return segment_ids


def _validate_bootstrap_parameters(
    *,
    n_bootstrap: int,
    block_length: int | None,
) -> None:
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap:
        raise ValueError("n_bootstrap must be a positive integer")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if block_length is not None:
        if isinstance(block_length, bool) or int(block_length) != block_length:
            raise ValueError("block_length must be a positive integer or None")
        if int(block_length) < 1:
            raise ValueError("block_length must be a positive integer or None")


def _sample_segment_runs(
    runs: list[np.ndarray],
    *,
    draw_count: int,
    rng: np.random.Generator,
    block_length: int | None,
    total_length: int,
) -> np.ndarray:
    """Sample fixed-size circular blocks independently within each run."""

    requested_block_length = int(
        block_length
        if block_length is not None
        else max(2, round(total_length ** (1.0 / 3.0)))
    )
    sampled = np.empty((draw_count, total_length), dtype=np.int64)
    output_start = 0
    for run in runs:
        run_length = len(run)
        effective_block_length = min(requested_block_length, run_length)
        blocks_per_draw = int(np.ceil(run_length / effective_block_length))
        starts = rng.integers(
            0,
            run_length,
            size=(draw_count, blocks_per_draw),
        )
        offsets = np.arange(effective_block_length)
        local_indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % run_length
        local_indices = local_indices.reshape(draw_count, -1)[:, :run_length]
        sampled[:, output_start : output_start + run_length] = run[local_indices]
        output_start += run_length
    return sampled


def sample_segmented_circular_block_indices(
    segment_ids: Sequence[object] | np.ndarray,
    *,
    n_bootstrap: int,
    random_state: int,
    block_length: int | None,
) -> np.ndarray:
    """Return segment-safe circular moving-block source indices.

    Each output draw contains the original number of observations from every
    segment. Blocks start uniformly and wrap only within that segment. The
    returned indices are primarily useful for audits and regression tests;
    production mean estimation uses the same sampler in bounded batches.
    """

    _validate_bootstrap_parameters(
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )
    ids = list(segment_ids)
    runs = _validated_segment_runs(ids, expected_length=len(ids))
    rng = np.random.default_rng(random_state)
    return _sample_segment_runs(
        runs,
        draw_count=int(n_bootstrap),
        rng=rng,
        block_length=block_length,
        total_length=len(ids),
    )


def _moving_block_means(
    values: np.ndarray,
    *,
    segment_ids: Sequence[object] | np.ndarray,
    n_bootstrap: int,
    random_state: int,
    block_length: int | None,
) -> np.ndarray:
    """Return segment-stratified circular moving-block bootstrap means."""

    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    if not np.isfinite(numeric).all():
        raise ValueError("bootstrap values must be finite")
    _validate_bootstrap_parameters(
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )
    ids = list(segment_ids)
    runs = _validated_segment_runs(ids, expected_length=len(numeric))
    rng = np.random.default_rng(random_state)
    bootstrap_means: list[float] = []
    batch_size = min(200, int(n_bootstrap))
    for batch_start in range(0, int(n_bootstrap), batch_size):
        current_size = min(batch_size, int(n_bootstrap) - batch_start)
        indices = _sample_segment_runs(
            runs,
            draw_count=current_size,
            rng=rng,
            block_length=block_length,
            total_length=len(numeric),
        )
        bootstrap_means.extend(numeric[indices].mean(axis=1).tolist())
    return np.asarray(bootstrap_means)


def _bootstrap_delta_interval(
    deltas: pd.Series,
    *,
    segment_ids: Sequence[object] | np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    ci: float = 0.95,
    block_length: int | None = None,
) -> tuple[float, float]:
    """Return a segment-safe moving-block interval for a paired mean delta."""

    if not 0.0 < float(ci) < 1.0:
        raise ValueError("ci must be strictly between zero and one")
    values = deltas.to_numpy(dtype=float)
    if len(values) <= 1:
        # Still validate the explicit segmentation contract for degenerate
        # intervals; no code path may silently fall back to an unsegmented run.
        _validated_segment_runs(segment_ids, expected_length=len(values))
        value = float(values[0]) if len(values) == 1 else 0.0
        return value, value
    alpha = 1.0 - ci
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    bootstrap_means = _moving_block_means(
        values,
        segment_ids=segment_ids,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        block_length=block_length,
    )
    return (
        float(np.percentile(bootstrap_means, lower_q)),
        float(np.percentile(bootstrap_means, upper_q)),
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Apply Holm-Bonferroni correction to a vector of p-values."""

    if p_values.empty:
        return p_values
    order = p_values.sort_values().index.tolist()
    m = len(order)
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running = 0.0
    for rank, index in enumerate(order, start=1):
        candidate = min(1.0, float(p_values.loc[index]) * (m - rank + 1))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted


def build_paired_policy_significance(
    decisions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    metric_columns: list[str],
    group_column: str = "session_bin_index",
    n_bootstrap: int = 2000,
    random_state: int = 42,
    block_length: int | None = None,
    segment_columns: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run paired policy comparisons with segment-safe descriptive bootstrap.

    Policies are paired on ``group_column`` *and* all ``segment_columns``.
    Segment labels must be nonmissing, contiguous, and aligned between the two
    policies. Circular moving blocks are sampled separately inside each
    segment, keeping its observation count fixed in every draw. Callers must
    explicitly pass boundary columns; an implicit all-rows segment is rejected.

    The Wilcoxon result remains an ordinary paired-window diagnostic. The
    segment-aware bootstrap controls block construction but does not by itself
    establish independent sampling groups or a causal estimand.
    """

    columns = _normalize_segment_columns(segment_columns)
    if group_column in columns:
        raise ValueError("group_column must be distinct from segment_columns")
    required = ["policy_name", group_column, *columns]
    missing = [column for column in required if column not in decisions.columns]
    if missing:
        raise KeyError(f"paired significance columns are missing: {missing}")
    _validate_bootstrap_parameters(
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )

    rows: list[PairedSignificanceResult] = []
    pair_columns = [*columns, group_column]
    segment_label = "|".join(columns)
    for comparison_name, left_policy, right_policy in comparisons:
        left = decisions[decisions["policy_name"] == left_policy].copy()
        right = decisions[decisions["policy_name"] == right_policy].copy()
        if left.empty or right.empty:
            continue
        # Validate each source sequence before pairing can hide a malformed or
        # unmatched segment boundary.
        build_bootstrap_segment_ids(left, columns)
        build_bootstrap_segment_ids(right, columns)
        if left.duplicated(pair_columns).any() or right.duplicated(pair_columns).any():
            raise ValueError(
                "paired policy keys must be unique within bootstrap segments"
            )
        left_keys = pd.MultiIndex.from_frame(left[pair_columns])
        right_keys = pd.MultiIndex.from_frame(right[pair_columns])
        if not left_keys.equals(right_keys):
            raise ValueError(
                "paired policies must have identical ordered decision keys "
                "and bootstrap segment assignments"
            )
        paired = left.merge(
            right,
            on=pair_columns,
            suffixes=("_left", "_right"),
            how="inner",
            sort=False,
            validate="one_to_one",
        )

        for metric_name in metric_columns:
            left_column = f"{metric_name}_left"
            right_column = f"{metric_name}_right"
            if left_column not in paired.columns or right_column not in paired.columns:
                continue
            left_values = pd.to_numeric(paired[left_column], errors="coerce")
            right_values = pd.to_numeric(paired[right_column], errors="coerce")
            valid_mask = np.isfinite(left_values.to_numpy(dtype=float)) & np.isfinite(
                right_values.to_numpy(dtype=float)
            )
            if not valid_mask.any():
                continue
            segment_ids = build_bootstrap_segment_ids(
                paired,
                columns,
                valid_mask=valid_mask,
            )
            deltas = pd.Series(
                left_values.to_numpy(dtype=float)[valid_mask]
                - right_values.to_numpy(dtype=float)[valid_mask]
            )
            nonzero = deltas[deltas != 0]
            if nonzero.empty:
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(deltas)
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            ci_lower, ci_upper = _bootstrap_delta_interval(
                deltas,
                segment_ids=segment_ids,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                block_length=block_length,
            )
            centered = deltas - deltas.mean()
            null_means = _moving_block_means(
                centered.to_numpy(dtype=float),
                segment_ids=segment_ids,
                n_bootstrap=n_bootstrap,
                random_state=random_state + 1,
                block_length=block_length,
            )
            observed_mean = abs(float(deltas.mean()))
            block_bootstrap_p_value = float(
                (1 + np.sum(np.abs(null_means) >= observed_mean))
                / (n_bootstrap + 1)
            )
            std_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
            effect_size_dz = float(deltas.mean() / std_delta) if std_delta > 1e-9 else 0.0
            win_rate = float((deltas < 0).mean())
            loss_rate = float((deltas > 0).mean())

            rows.append(
                PairedSignificanceResult(
                    comparison_name=comparison_name,
                    metric_name=metric_name,
                    sample_count=len(deltas),
                    mean_delta=float(deltas.mean()),
                    median_delta=float(deltas.median()),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    statistic=statistic,
                    p_value=p_value,
                    holm_adjusted_p_value=p_value,
                    effect_size_dz=effect_size_dz,
                    win_rate=win_rate,
                    loss_rate=loss_rate,
                    block_bootstrap_p_value=block_bootstrap_p_value,
                    bootstrap_method=SEGMENTED_CIRCULAR_BLOCK_METHOD,
                    bootstrap_segment_columns=segment_label,
                    bootstrap_segment_count=int(len(np.unique(segment_ids))),
                )
            )

    result = pd.DataFrame([asdict(item) for item in rows])
    if result.empty:
        return result
    for metric_name, metric_frame in result.groupby("metric_name"):
        adjusted = _holm_adjust(metric_frame["p_value"])
        result.loc[metric_frame.index, "holm_adjusted_p_value"] = adjusted
    return result
