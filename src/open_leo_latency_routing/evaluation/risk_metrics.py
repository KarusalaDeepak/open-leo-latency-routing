"""Finite-sample tail-risk metrics with explicit empirical definitions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def empirical_weighted_upper_cvar(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float = 0.95,
) -> float:
    r"""Return weighted empirical upper-tail RU CVaR.

    Weights are normalized to unit mass.  The upper ``1 - quantile`` mass is
    integrated exactly, with fractional mass assigned at the tail boundary.
    This supports clustered designs in which every independent group receives
    equal mass and observations within a group divide that mass equally.
    """

    q = float(quantile)
    if not 0.0 <= q < 1.0:
        raise ValueError("quantile must lie in [0, 1)")
    array = np.asarray(values, dtype=float).reshape(-1)
    raw_weights = np.asarray(weights, dtype=float).reshape(-1)
    if not array.size:
        raise ValueError("CVaR requires at least one observation")
    if len(raw_weights) != len(array):
        raise ValueError("CVaR weights must align with observations")
    if not np.isfinite(array).all():
        raise ValueError("CVaR observations must be finite")
    if not np.isfinite(raw_weights).all() or (raw_weights < 0.0).any():
        raise ValueError("CVaR weights must be finite and non-negative")
    total_weight = float(raw_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("CVaR weights must have positive total mass")
    normalized_weights = raw_weights / total_weight

    order = np.argsort(array, kind="stable")[::-1]
    descending = array[order]
    descending_weights = normalized_weights[order]
    tail_mass = 1.0 - q
    cumulative_before = np.cumsum(descending_weights) - descending_weights
    included_mass = np.clip(
        tail_mass - cumulative_before,
        0.0,
        descending_weights,
    )
    return float(np.dot(included_mass, descending) / tail_mass)


def empirical_upper_cvar(
    values: Sequence[float] | np.ndarray,
    quantile: float = 0.95,
) -> float:
    r"""Return empirical upper-tail Rockafellar--Uryasev CVaR.

    For empirical observations :math:`x_1,\ldots,x_n`, this evaluates

    .. math::

       \min_\eta \left\{\eta + \frac{1}{(1-q)n}
       \sum_i (x_i-\eta)_+\right\}.

    Equivalently, it averages the largest ``(1-q) * n`` observations while
    assigning the boundary observation fractional mass when that tail size is
    not an integer.  This is intentionally different from averaging every
    value greater than or equal to a library-specific sample quantile, which
    overweights tied boundary observations and changes with quantile
    interpolation conventions.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    return empirical_weighted_upper_cvar(
        array,
        np.ones(len(array), dtype=float),
        quantile,
    )
