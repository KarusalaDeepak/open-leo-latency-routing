"""Conservative, opportunity-aware policy admission before final testing.

The confidence statements in this module are conditional on pre-declared,
independently acquired collection groups: distinct groups are independent and
exchangeable, while observations inside a group may be arbitrarily dependent.
This is an assumption that an experiment must justify from acquisition
provenance (for example with independently collected sessions); it is never
inferred by cutting one time series into consecutive blocks.

Success uses two exact harmful-block binomial lower bounds: one over the full
actionable population and one conditioned on post-hoc decision opportunities
inside opportunity-bearing groups. Tail latency uses simultaneous Hoeffding
bounds for the Rockafellar--Uryasev representation of CVaR on a pre-declared
bounded latency scale. Bonferroni allocation covers every learned candidate,
both success endpoints, both CVaR policy intervals, the CVaR threshold grid,
and every planned invocation of the gate. Consequently, an all-zero paired
sample retains a strict uncertainty penalty instead of producing a spurious
zero-width bound. The aggregate success and CVaR endpoints draw an independent
group uniformly and then an actionable epoch uniformly within that group. The
opportunity-conditioned endpoint instead draws an opportunity-bearing group
uniformly and then a post-hoc decision opportunity uniformly within that group.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import beta

from open_leo_latency_routing.evaluation.risk_metrics import (
    empirical_weighted_upper_cvar,
)


@dataclass(frozen=True)
class RiskControlConfig:
    """Pre-declared admission, multiplicity, and bounded-risk parameters."""

    alpha: float = 0.05
    noninferiority_margin: float = 0.02
    opportunity_noninferiority_margin: float = 0.02
    # Legacy public/configuration name.  The value is an integer-like floor on
    # independently acquired opportunity-bearing groups, never raw epochs or a
    # block-bootstrap effective sample size.  Evidence also exports the clearer
    # ``minimum_opportunity_bearing_groups`` alias.
    minimum_effective_opportunities: float = 5.0
    practical_cvar_gain_ms: float = 1.0
    cvar_quantile: float = 0.95
    # Retained only so archived configuration files still parse.  A time-block
    # length cannot establish independent acquisition groups and is rejected by
    # the production gate; callers must supply ``independence_group_ids``.
    block_length: int | None = None
    latency_cap_ms: float = 60_000.0
    # A 0.05-ms grid over the default 60-s cap limits the deterministic
    # between-grid CVaR correction to 0.475 ms at q=0.95, below the 1-ms
    # practical-gain margin.  The implementation evaluates this dense grid in
    # bounded-memory chunks rather than materializing a decision-by-grid array.
    cvar_grid_points: int = 1_200_001
    planned_gate_uses: int = 1
    gate_use_index: int = 1
    # Retained for configuration compatibility with archived experiments.  The
    # finite-sample procedure below does not use bootstrap resampling or a seed.
    bootstrap_samples: int = 4000
    random_seed: int = 2026


@dataclass(frozen=True)
class RiskControlSelection:
    """Frozen policy choice and its complete pre-test evidence."""

    selected_policy: str
    reason: str
    evidence: tuple[dict[str, object], ...]

    def evidence_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.evidence)


def _cvar(
    values: np.ndarray,
    quantile: float,
    observation_weights: np.ndarray,
) -> float:
    return empirical_weighted_upper_cvar(
        values,
        observation_weights,
        quantile,
    )


def _independence_group_indices(
    group_ids: list[object] | np.ndarray,
    size: int,
) -> tuple[np.ndarray, ...]:
    """Return one inference block per explicitly supplied session/group."""

    groups = np.asarray(group_ids, dtype=object)
    if len(groups) != size:
        raise ValueError("independence groups must align with policy outcomes")
    if bool(pd.isna(groups).any()):
        raise ValueError("independence groups must not contain missing values")
    codes, uniques = pd.factorize(groups, sort=False)
    return tuple(np.flatnonzero(codes == code) for code in range(len(uniques)))


def _block_weights(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Give every declared independent group equal population mass."""

    if not blocks:
        raise ValueError("at least one inference block is required")
    return np.full(len(blocks), 1.0 / len(blocks), dtype=float)


def _observation_weights(
    blocks: tuple[np.ndarray, ...],
    block_weights: np.ndarray,
    size: int,
) -> np.ndarray:
    """Divide each group's equal mass uniformly across its epochs."""

    if len(blocks) != len(block_weights):
        raise ValueError("block weights must align with inference blocks")
    observation_weights = np.zeros(size, dtype=float)
    assignment_count = np.zeros(size, dtype=int)
    for block, block_weight in zip(blocks, block_weights):
        if not len(block):
            raise ValueError("inference blocks must be non-empty")
        observation_weights[block] = float(block_weight) / len(block)
        assignment_count[block] += 1
    if not np.all(assignment_count == 1):
        raise ValueError("inference blocks must partition policy outcomes")
    if not np.isclose(observation_weights.sum(), 1.0):
        raise AssertionError("observation weights must sum to one")
    return observation_weights


def _opportunity_conditioned_blocks(
    blocks: tuple[np.ndarray, ...],
    opportunities: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Restrict each opportunity-bearing group to its opportunity epochs."""

    conditioned: list[np.ndarray] = []
    for block in blocks:
        opportunity_epochs = block[opportunities[block]]
        if len(opportunity_epochs):
            conditioned.append(opportunity_epochs)
    return tuple(conditioned)


def _group_uniform_mean(
    values: np.ndarray,
    blocks: tuple[np.ndarray, ...],
) -> float:
    """Average within groups first and then give every group equal mass."""

    if not blocks:
        return math.nan
    return float(np.mean([np.mean(values[block]) for block in blocks]))


def _exact_harmful_block_lcb(
    values: np.ndarray,
    alpha: float,
    blocks: tuple[np.ndarray, ...],
) -> tuple[float, int, float, float]:
    """Exact conservative LCB for the mean paired success difference.

    A block is harmful when its mean paired difference is negative.  Since a
    block mean lies in ``[-1, 1]``, its expectation is at least minus the
    harmful-block probability.  A one-sided Clopper--Pearson upper bound on
    that probability therefore gives a finite-sample lower bound.  Benefits
    are deliberately ignored, making this a sufficient non-inferiority test.
    Blocks must be independent and identically distributed inference units.
    """

    block_means = np.asarray(
        [float(np.mean(values[block])) for block in blocks],
        dtype=float,
    )
    harmful_count = int(np.count_nonzero(block_means < 0.0))
    trials = len(block_means)
    if harmful_count == trials:
        harmful_probability_ucb = 1.0
    else:
        harmful_probability_ucb = float(
            beta.ppf(
                1.0 - alpha,
                harmful_count + 1,
                trials - harmful_count,
            )
        )
    cluster_point = float(np.mean(block_means))
    return (
        cluster_point,
        harmful_count,
        harmful_probability_ucb,
        -harmful_probability_ucb,
    )


def _bounded_block_cvar_interval(
    values: np.ndarray,
    quantile: float,
    latency_cap_ms: float,
    grid_points: int,
    alpha: float,
    blocks: tuple[np.ndarray, ...],
    block_weights: np.ndarray,
    observation_weights: np.ndarray,
) -> tuple[float, float, float, int]:
    """Simultaneous interval for group-uniform, bounded empirical CVaR.

    For each fixed threshold ``eta``, the Rockafellar--Uryasev objective is

        eta + E[(X - eta)_+] / (1 - q).

    Its empirical value is formed inside each group and then combined using
    equal group weights.  Hence observation ``i`` in group ``g`` has weight
    ``1 / (G * n_g)``. Hoeffding bounds are simultaneous over a fixed grid.
    The lower endpoint includes the worst-case Lipschitz correction between
    neighboring grid values; the upper endpoint needs no grid correction.
    """

    clipped = np.clip(np.asarray(values, dtype=float), 0.0, latency_cap_ms)
    clipped_count = int(np.count_nonzero(clipped != values))
    point = _cvar(clipped, quantile, observation_weights)
    if len(blocks) != len(block_weights):
        raise ValueError("block weights must align with inference blocks")
    if len(observation_weights) != len(clipped):
        raise ValueError("observation weights must align with CVaR outcomes")
    grid_points = int(grid_points)
    # Both sides of the interval are exported.  Split alpha across the lower
    # and upper event as well as across every fixed threshold.
    per_grid_alpha = alpha / (2.0 * grid_points)
    tail_probability = 1.0 - quantile
    # Sorting values together with their 1/(G*n_g) weights lets every hinge
    # expectation be evaluated from weighted suffix sums. Complexity is
    # O(n log n + M log n) time and O(n + chunk_size) memory.
    order = np.argsort(clipped, kind="stable")
    sorted_values = clipped[order]
    sorted_weights = observation_weights[order]
    suffix_weights = np.empty(len(sorted_values) + 1, dtype=float)
    suffix_weighted_values = np.empty(len(sorted_values) + 1, dtype=float)
    suffix_weights[-1] = 0.0
    suffix_weighted_values[-1] = 0.0
    suffix_weights[:-1] = np.cumsum(sorted_weights[::-1])[::-1]
    suffix_weighted_values[:-1] = np.cumsum(
        (sorted_weights * sorted_values)[::-1]
    )[::-1]
    concentration_factor = math.sqrt(
        float(np.square(block_weights).sum())
        * math.log(1.0 / per_grid_alpha)
        / 2.0
    )
    lower_min = math.inf
    upper_min = math.inf
    chunk_size = 100_000
    denominator = float(grid_points - 1)
    for start in range(0, grid_points, chunk_size):
        stop = min(start + chunk_size, grid_points)
        grid_indices = np.arange(start, stop, dtype=float)
        thresholds = latency_cap_ms * grid_indices / denominator
        first_greater = np.searchsorted(
            sorted_values,
            thresholds,
            side="right",
        )
        excess_sums = (
            suffix_weighted_values[first_greater]
            - suffix_weights[first_greater] * thresholds
        )
        empirical_objective = (
            thresholds
            + excess_sums / tail_probability
        )
        objective_range = (
            latency_cap_ms - thresholds
        ) / tail_probability
        radius = objective_range * concentration_factor
        lower_min = min(
            lower_min,
            float(np.min(empirical_objective - radius)),
        )
        upper_min = min(
            upper_min,
            float(np.min(empirical_objective + radius)),
        )

    spacing = latency_cap_ms / (grid_points - 1)
    objective_lipschitz = max(1.0, quantile / tail_probability)
    grid_correction = 0.5 * spacing * objective_lipschitz
    lower_bound = max(0.0, lower_min - grid_correction)
    upper_bound = min(latency_cap_ms, upper_min)
    if lower_bound > upper_bound:
        # Numerical roundoff is the only legitimate way this can occur.
        lower_bound = upper_bound
    return point, lower_bound, upper_bound, clipped_count


def _no_actionable_epoch_selection(
    arrays: dict[str, np.ndarray],
    cfg: RiskControlConfig,
    reactive_policy: str,
) -> RiskControlSelection:
    """Fail closed when the selection interval contains only outage epochs."""

    reason = (
        "no actionable policy-selection epochs; all epochs were outage/no-action"
    )
    learned = sorted(set(arrays).difference({reactive_policy}))
    learned_count = max(1, len(learned))
    alpha_per_candidate_per_use = (
        cfg.alpha / cfg.planned_gate_uses / learned_count
    )
    # Four simultaneously protected endpoint families per candidate/use:
    # aggregate success, opportunity-conditioned success, reactive CVaR, and
    # candidate CVaR. The two CVaR intervals jointly form the gain comparison.
    endpoint_alpha = alpha_per_candidate_per_use / 4.0
    aggregate_success_alpha = endpoint_alpha
    opportunity_success_alpha = endpoint_alpha
    cvar_policy_alpha = endpoint_alpha
    cvar_comparison_alpha = 2.0 * cvar_policy_alpha
    common_evidence: dict[str, object] = {
        "decision_count": 0,
        "opportunity_count": 0,
        "opportunity_block_count": 0,
        "block_length": 0,
        "block_length_source": "no_actionable_epochs",
        "inference_unit_source": "no_actionable_epochs",
        "inference_block_count": 0,
        "effective_block_count": 0.0,
        "effective_opportunity_count": 0.0,
        "opportunity_sufficient": False,
        "minimum_opportunity_bearing_groups": (
            cfg.minimum_effective_opportunities
        ),
        "confidence_method": "not_computed_no_actionable_epochs",
        "success_confidence_method": "not_computed_no_actionable_epochs",
        "aggregate_success_confidence_method": (
            "not_computed_no_actionable_epochs"
        ),
        "opportunity_conditioned_success_confidence_method": (
            "not_computed_no_actionable_epochs"
        ),
        "independence_assumption": (
            "not invoked because the actionable selection population is empty"
        ),
        "cvar_bound_method": "not_computed_no_actionable_epochs",
        "bounded_latency_cap_ms": cfg.latency_cap_ms,
        "cvar_quantile": cfg.cvar_quantile,
        "cvar_grid_points": cfg.cvar_grid_points,
        "gate_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "success_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "aggregate_success_estimand_population": (
            "uniform_independent_group_then_uniform_actionable_epoch_within_group"
        ),
        "opportunity_conditioned_success_estimand_population": (
            "uniform_opportunity_bearing_independent_group_then_uniform_post_hoc_"
            "decision_opportunity_within_group"
        ),
        "opportunity_conditioning": (
            "post_hoc_mixed_feasible_candidate_QoS_outcomes"
        ),
        "opportunity_conditioned_inference_group_count": 0,
        "opportunity_conditioned_endpoint_defined": False,
        "opportunity_conditioned_group_weight": math.nan,
        "minimum_opportunities_per_bearing_group": 0,
        "maximum_opportunities_per_bearing_group": 0,
        "opportunity_conditioned_observation_weight_formula": (
            "1/(G_opportunity*m_g)"
        ),
        "cvar_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "group_population_weighting": "equal_mass_per_independent_group",
        "within_group_population_weighting": "uniform_epoch_within_group",
        "observation_weight_formula": "1/(G*n_g)",
        "primary_gate_point_field_prefix": "group_uniform_",
        "unprefixed_point_field_semantics": (
            "compatibility aliases of group_uniform point estimates; not "
            "epoch-pooled estimates"
        ),
        "noninferior_alias_semantics": (
            "conjunction_of_aggregate_and_opportunity_conditioned_success_"
            "noninferiority"
        ),
        "epoch_pooled_point_estimates_used_for_gate": False,
        "group_weight": math.nan,
        "minimum_group_size": 0,
        "maximum_group_size": 0,
        "minimum_observation_weight": math.nan,
        "maximum_observation_weight": math.nan,
        "cvar_computational_complexity": "not_computed_no_actionable_epochs",
        "joint_coverage_level": 1.0 - cfg.alpha,
        "joint_coverage_family": (
            "all learned candidates x aggregate-success/opportunity-success/"
            "reactive-CVaR/candidate-CVaR endpoints x planned gate uses"
        ),
        "joint_coverage_theorem": (
            "under the declared independent-group, bounded-latency, and fixed-"
            "configuration assumptions, Bonferroni allocation gives joint "
            "coverage at least 1-alpha across every learned candidate, both "
            "success endpoints, both CVaR policy intervals, and all planned "
            "gate uses"
        ),
        "success_gate_requires_both_endpoints": True,
        "alpha_endpoint_family_count_per_candidate_use": 4,
        "learned_candidate_count": len(learned),
        "alpha_allocation_formula": (
            "familywise_alpha/(planned_gate_uses*max(1,learned_candidate_count)*4)"
        ),
        "gate_use_index": cfg.gate_use_index,
        "planned_gate_uses": cfg.planned_gate_uses,
        "empty_actionable_guard": True,
    }
    evidence: list[dict[str, object]] = []
    for name in [reactive_policy, *learned]:
        is_reactive = name == reactive_policy
        evidence.append(
            {
                "policy": name,
                "success_rate": math.nan,
                "group_uniform_success_rate": math.nan,
                "success_delta_vs_reactive": 0.0 if is_reactive else math.nan,
                "group_uniform_success_delta_vs_reactive": (
                    0.0 if is_reactive else math.nan
                ),
                "bounded_block_success_delta": 0.0 if is_reactive else math.nan,
                "success_bound_radius": 0.0 if is_reactive else 1.0,
                "harmful_block_count": 0,
                "harmful_block_probability_ucb": 0.0 if is_reactive else 1.0,
                "simultaneous_lcb": 0.0 if is_reactive else -1.0,
                "success_delta_lcb": 0.0 if is_reactive else -1.0,
                "aggregate_actionable_success_delta_vs_reactive": (
                    0.0 if is_reactive else math.nan
                ),
                "aggregate_actionable_block_success_delta": (
                    0.0 if is_reactive else math.nan
                ),
                "aggregate_actionable_harmful_group_count": 0,
                "aggregate_actionable_harmful_group_probability_ucb": (
                    0.0 if is_reactive else 1.0
                ),
                "aggregate_actionable_success_delta_lcb": (
                    0.0 if is_reactive else -1.0
                ),
                "aggregate_actionable_success_noninferior": is_reactive,
                "opportunity_conditioned_success_rate": math.nan,
                "group_uniform_opportunity_conditioned_success_rate": math.nan,
                "opportunity_conditioned_success_delta_vs_reactive": (
                    0.0 if is_reactive else math.nan
                ),
                "opportunity_conditioned_block_success_delta": (
                    0.0 if is_reactive else math.nan
                ),
                "opportunity_conditioned_success_bound_radius": (
                    0.0 if is_reactive else 1.0
                ),
                "opportunity_conditioned_harmful_group_count": 0,
                "opportunity_conditioned_harmful_group_probability_ucb": (
                    0.0 if is_reactive else 1.0
                ),
                "opportunity_conditioned_success_delta_lcb": (
                    0.0 if is_reactive else -1.0
                ),
                "opportunity_conditioned_success_noninferior": is_reactive,
                "success_endpoints_noninferior": is_reactive,
                "mean_latency_ms": math.nan,
                "group_uniform_mean_latency_ms": math.nan,
                "cvar_latency_ms": math.nan,
                "group_uniform_cvar_latency_ms": math.nan,
                "bounded_cvar_latency_ms": math.nan,
                "group_uniform_bounded_cvar_latency_ms": math.nan,
                "cvar_lcb_ms": math.nan,
                "group_uniform_cvar_lcb_ms": math.nan,
                "cvar_ucb_ms": math.nan,
                "group_uniform_cvar_ucb_ms": math.nan,
                "raw_cvar_gain_vs_reactive_ms": 0.0 if is_reactive else math.nan,
                "group_uniform_raw_cvar_gain_vs_reactive_ms": (
                    0.0 if is_reactive else math.nan
                ),
                "cvar_gain_vs_reactive_ms": 0.0 if is_reactive else math.nan,
                "group_uniform_cvar_gain_vs_reactive_ms": (
                    0.0 if is_reactive else math.nan
                ),
                "cvar_gain_lcb_ms": (
                    0.0 if is_reactive else -cfg.latency_cap_ms
                ),
                "group_uniform_cvar_gain_lcb_ms": (
                    0.0 if is_reactive else -cfg.latency_cap_ms
                ),
                "latency_clipped_count": 0,
                "noninferior": is_reactive,
                "practically_better": False,
                "eligible": is_reactive,
                "selected": is_reactive,
                **common_evidence,
                "selection_reason": reason,
                "alpha_familywise": cfg.alpha,
                "alpha_per_learned_policy": alpha_per_candidate_per_use,
                "alpha_per_candidate_per_gate_use": (
                    alpha_per_candidate_per_use
                ),
                "alpha_success_bound": aggregate_success_alpha,
                "alpha_aggregate_success_bound": aggregate_success_alpha,
                "alpha_opportunity_success_bound": opportunity_success_alpha,
                "alpha_cvar_comparison": cvar_comparison_alpha,
                "alpha_cvar_policy_interval": cvar_policy_alpha,
                "alpha_cvar_reactive_interval": cvar_policy_alpha,
                "alpha_cvar_candidate_interval": cvar_policy_alpha,
                "alpha_cvar_per_grid_bound": (
                    cvar_policy_alpha / (2.0 * cfg.cvar_grid_points)
                ),
                "noninferiority_margin": cfg.noninferiority_margin,
                "aggregate_actionable_noninferiority_margin": (
                    cfg.noninferiority_margin
                ),
                "aggregate_actionable_success_lcb_threshold": (
                    -cfg.noninferiority_margin
                ),
                "opportunity_noninferiority_margin": (
                    cfg.opportunity_noninferiority_margin
                ),
                "opportunity_conditioned_noninferiority_margin": (
                    cfg.opportunity_noninferiority_margin
                ),
                "opportunity_conditioned_success_lcb_threshold": (
                    -cfg.opportunity_noninferiority_margin
                ),
                "minimum_effective_opportunities": (
                    cfg.minimum_effective_opportunities
                ),
                "practical_cvar_gain_ms": cfg.practical_cvar_gain_ms,
                "bootstrap_samples_used": 0,
            }
        )
    return RiskControlSelection(
        selected_policy=reactive_policy,
        reason=reason,
        evidence=tuple(dict(row) for row in evidence),
    )


def select_opportunity_aware_risk_controlled_policy(
    realized_selection_latency: dict[str, list[float]],
    opportunity_mask: list[bool] | np.ndarray,
    latency_budget_ms: float,
    config: RiskControlConfig | None = None,
    reactive_policy: str = "reactive",
    independence_group_ids: list[object] | np.ndarray | None = None,
) -> RiskControlSelection:
    """Admit learning only with simultaneous bounded group evidence.

    The returned guarantees concern the policy-selection population under the
    independently acquired-group assumption stated at module level.  They do
    not imply a guarantee after distribution shift.  Group identifiers must
    come from acquisition provenance; ``block_length`` is rejected because
    slicing one trace cannot manufacture independent replications.  The CVaR
    endpoint is explicitly the CVaR of latency clipped to ``latency_cap_ms``;
    the cap must therefore be fixed from the measurement protocol before
    outcomes are inspected.
    An aligned zero-length population is permitted for the all-outage case and
    returns a fail-closed reactive selection with no inferential claims.
    """

    cfg = config or RiskControlConfig()
    if reactive_policy not in realized_selection_latency:
        raise KeyError(f"missing reactive policy: {reactive_policy}")
    if not 0.0 < cfg.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if (
        not math.isfinite(cfg.noninferiority_margin)
        or not 0.0 <= cfg.noninferiority_margin <= 1.0
    ):
        raise ValueError("noninferiority margin must be finite and in [0, 1]")
    if (
        not math.isfinite(cfg.opportunity_noninferiority_margin)
        or not 0.0 <= cfg.opportunity_noninferiority_margin <= 1.0
    ):
        raise ValueError(
            "opportunity noninferiority margin must be finite and in [0, 1]"
        )
    if (
        not math.isfinite(cfg.minimum_effective_opportunities)
        or cfg.minimum_effective_opportunities < 0.0
    ):
        raise ValueError(
            "minimum effective opportunities must be finite and non-negative"
        )
    if (
        not math.isfinite(cfg.practical_cvar_gain_ms)
        or cfg.practical_cvar_gain_ms < 0.0
    ):
        raise ValueError("practical CVaR gain must be finite and non-negative")
    if not 0.0 < cfg.cvar_quantile < 1.0:
        raise ValueError("CVaR quantile must be in (0, 1)")
    if not math.isfinite(cfg.latency_cap_ms) or cfg.latency_cap_ms <= 0.0:
        raise ValueError("latency cap must be finite and positive")
    if (
        isinstance(cfg.cvar_grid_points, bool)
        or int(cfg.cvar_grid_points) != cfg.cvar_grid_points
        or cfg.cvar_grid_points < 2
    ):
        raise ValueError("CVaR grid size must be an integer of at least two")
    if (
        isinstance(cfg.planned_gate_uses, bool)
        or int(cfg.planned_gate_uses) != cfg.planned_gate_uses
        or cfg.planned_gate_uses < 1
    ):
        raise ValueError("planned gate uses must be a positive integer")
    if (
        isinstance(cfg.gate_use_index, bool)
        or int(cfg.gate_use_index) != cfg.gate_use_index
        or not 1 <= cfg.gate_use_index <= cfg.planned_gate_uses
    ):
        raise ValueError("gate use index must be within the planned gate family")
    if cfg.block_length is not None and (
        isinstance(cfg.block_length, bool)
        or int(cfg.block_length) != cfg.block_length
        or cfg.block_length < 1
    ):
        raise ValueError("block length must be a positive integer when supplied")
    if cfg.block_length is not None:
        raise ValueError(
            "block_length cannot establish independent acquisition groups; "
            "supply independence_group_ids or omit block_length to fail "
            "closed as one collection"
        )
    if not math.isfinite(latency_budget_ms) or latency_budget_ms < 0.0:
        raise ValueError("latency budget must be finite and non-negative")

    arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in realized_selection_latency.items()
    }
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("policy latency arrays must be one-dimensional")
    sizes = {len(values) for values in arrays.values()}
    if len(sizes) != 1 or not sizes:
        raise ValueError("policy latency arrays must be aligned")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("policy latency arrays must contain only finite values")
    if any((values < 0.0).any() for values in arrays.values()):
        raise ValueError("policy latency arrays must be non-negative")
    n = next(iter(sizes))
    raw_opportunities = np.asarray(opportunity_mask)
    if raw_opportunities.ndim != 1 or len(raw_opportunities) != n:
        raise ValueError("opportunity mask must align with policy outcomes")
    if raw_opportunities.dtype.kind not in "biuf":
        raise ValueError("opportunity mask must contain only Boolean or 0/1 values")
    numeric_opportunities = raw_opportunities.astype(float)
    if (
        not np.isfinite(numeric_opportunities).all()
        or not np.isin(numeric_opportunities, (0.0, 1.0)).all()
    ):
        raise ValueError("opportunity mask must contain only Boolean or 0/1 values")
    opportunities = numeric_opportunities.astype(bool)
    if independence_group_ids is not None and len(independence_group_ids) != n:
        raise ValueError("independence groups must align with policy outcomes")
    if n == 0:
        return _no_actionable_epoch_selection(arrays, cfg, reactive_policy)

    if independence_group_ids is not None:
        blocks = _independence_group_indices(independence_group_ids, n)
        block_length = max(len(block) for block in blocks)
        block_length_source = "explicit_independence_groups"
        inference_unit_source = "supplied_session_or_collection_group"
    else:
        # A time-series length cannot establish independent replications.  In
        # the absence of explicit collection/session identifiers, fail closed
        # by treating the complete selection interval as one collection.
        block_length = n
        block_length_source = "single_collection_default"
        inference_unit_source = "no_independent_groups_declared"
        blocks = (np.arange(n, dtype=int),)
    block_weights = _block_weights(blocks)
    observation_weights = _observation_weights(blocks, block_weights, n)
    effective_block_count = float(1.0 / np.square(block_weights).sum())
    opportunity_count = int(opportunities.sum())
    opportunity_blocks = _opportunity_conditioned_blocks(
        blocks,
        opportunities,
    )
    opportunity_block_count = len(opportunity_blocks)
    # An opportunity-bearing block is an inference unit.  No division by block
    # length is used: five opportunities in one block still provide one unit.
    effective_opportunities = float(opportunity_block_count)
    opportunity_sufficient = (
        effective_opportunities >= cfg.minimum_effective_opportunities
    )

    learned = sorted(set(arrays).difference({reactive_policy}))
    learned_count = max(1, len(learned))
    alpha_per_candidate_per_use = (
        cfg.alpha / cfg.planned_gate_uses / learned_count
    )
    # Bonferroni allocation protects four endpoint families for every learned
    # candidate and planned use: aggregate actionable success,
    # opportunity-conditioned success, the reactive CVaR interval, and the
    # candidate CVaR interval. The two CVaR intervals jointly certify gain.
    endpoint_alpha = alpha_per_candidate_per_use / 4.0
    aggregate_success_alpha = endpoint_alpha
    opportunity_success_alpha = endpoint_alpha
    cvar_policy_alpha = endpoint_alpha
    cvar_comparison_alpha = 2.0 * cvar_policy_alpha

    reactive = arrays[reactive_policy]
    reactive_success = (reactive <= latency_budget_ms).astype(float)
    reactive_opportunity_success = _group_uniform_mean(
        reactive_success,
        opportunity_blocks,
    )
    reactive_raw_cvar = _cvar(
        reactive,
        cfg.cvar_quantile,
        observation_weights,
    )
    (
        reactive_bounded_cvar,
        reactive_cvar_lcb,
        reactive_cvar_ucb,
        reactive_clipped_count,
    ) = _bounded_block_cvar_interval(
        reactive,
        cfg.cvar_quantile,
        cfg.latency_cap_ms,
        cfg.cvar_grid_points,
        cvar_policy_alpha,
        blocks,
        block_weights,
        observation_weights,
    )
    evidence: list[dict[str, object]] = []

    common_evidence: dict[str, object] = {
        "decision_count": n,
        "opportunity_count": opportunity_count,
        "opportunity_block_count": opportunity_block_count,
        "block_length": block_length,
        "block_length_source": block_length_source,
        "inference_unit_source": inference_unit_source,
        "inference_block_count": len(blocks),
        "effective_block_count": effective_block_count,
        "effective_opportunity_count": effective_opportunities,
        "opportunity_sufficient": opportunity_sufficient,
        "minimum_opportunity_bearing_groups": (
            cfg.minimum_effective_opportunities
        ),
        "confidence_method": (
            "independent_acquisition_group_finite_sample_bounds"
        ),
        "success_confidence_method": (
            "one_sided_Clopper_Pearson_harmful_group_probability"
        ),
        "aggregate_success_confidence_method": (
            "one_sided_Clopper_Pearson_harmful_group_probability"
        ),
        "opportunity_conditioned_success_confidence_method": (
            "one_sided_Clopper_Pearson_harmful_opportunity_bearing_group_"
            "probability"
        ),
        "independence_assumption": (
            "inference blocks are mutually independent and exchangeable; "
            "arbitrary dependence is allowed within blocks"
        ),
        "cvar_bound_method": (
            "bounded_group_uniform_Rockafellar_Uryasev_grid_Hoeffding"
        ),
        "gate_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "success_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "aggregate_success_estimand_population": (
            "uniform_independent_group_then_uniform_actionable_epoch_within_group"
        ),
        "opportunity_conditioned_success_estimand_population": (
            "uniform_opportunity_bearing_independent_group_then_uniform_post_hoc_"
            "decision_opportunity_within_group"
        ),
        "opportunity_conditioning": (
            "post_hoc_mixed_feasible_candidate_QoS_outcomes"
        ),
        "opportunity_conditioned_inference_group_count": (
            opportunity_block_count
        ),
        "opportunity_conditioned_endpoint_defined": bool(opportunity_blocks),
        "opportunity_conditioned_group_weight": (
            1.0 / opportunity_block_count
            if opportunity_block_count
            else math.nan
        ),
        "minimum_opportunities_per_bearing_group": (
            min(len(block) for block in opportunity_blocks)
            if opportunity_blocks
            else 0
        ),
        "maximum_opportunities_per_bearing_group": (
            max(len(block) for block in opportunity_blocks)
            if opportunity_blocks
            else 0
        ),
        "opportunity_conditioned_observation_weight_formula": (
            "1/(G_opportunity*m_g)"
        ),
        "cvar_estimand_population": (
            "uniform_independent_group_then_uniform_epoch_within_group"
        ),
        "group_population_weighting": "equal_mass_per_independent_group",
        "within_group_population_weighting": "uniform_epoch_within_group",
        "observation_weight_formula": "1/(G*n_g)",
        "primary_gate_point_field_prefix": "group_uniform_",
        "unprefixed_point_field_semantics": (
            "compatibility aliases of group_uniform point estimates; not "
            "epoch-pooled estimates"
        ),
        "noninferior_alias_semantics": (
            "conjunction_of_aggregate_and_opportunity_conditioned_success_"
            "noninferiority"
        ),
        "epoch_pooled_point_estimates_used_for_gate": False,
        "group_weight": float(block_weights[0]),
        "minimum_group_size": min(len(block) for block in blocks),
        "maximum_group_size": max(len(block) for block in blocks),
        "minimum_observation_weight": float(observation_weights.min()),
        "maximum_observation_weight": float(observation_weights.max()),
        "cvar_computational_complexity": (
            "O(n log n + M log n) time; O(n + grid_chunk_size) memory"
        ),
        "joint_coverage_level": 1.0 - cfg.alpha,
        "joint_coverage_family": (
            "all learned candidates x aggregate-success/opportunity-success/"
            "reactive-CVaR/candidate-CVaR endpoints x planned gate uses"
        ),
        "joint_coverage_theorem": (
            "under the declared independent-group, bounded-latency, and fixed-"
            "configuration assumptions, Bonferroni allocation gives joint "
            "coverage at least 1-alpha across every learned candidate, both "
            "success endpoints, both CVaR policy intervals, and all planned "
            "gate uses"
        ),
        "success_gate_requires_both_endpoints": True,
        "alpha_endpoint_family_count_per_candidate_use": 4,
        "learned_candidate_count": len(learned),
        "alpha_allocation_formula": (
            "familywise_alpha/(planned_gate_uses*max(1,learned_candidate_count)*4)"
        ),
        "bounded_latency_cap_ms": cfg.latency_cap_ms,
        "cvar_quantile": cfg.cvar_quantile,
        "cvar_grid_points": cfg.cvar_grid_points,
        "gate_use_index": cfg.gate_use_index,
        "planned_gate_uses": cfg.planned_gate_uses,
        "empty_actionable_guard": False,
    }

    evidence.append(
        {
            "policy": reactive_policy,
            "success_rate": float(np.dot(observation_weights, reactive_success)),
            "group_uniform_success_rate": float(
                np.dot(observation_weights, reactive_success)
            ),
            "success_delta_vs_reactive": 0.0,
            "group_uniform_success_delta_vs_reactive": 0.0,
            "bounded_block_success_delta": 0.0,
            "success_bound_radius": 0.0,
            "harmful_block_count": 0,
            "harmful_block_probability_ucb": 0.0,
            "simultaneous_lcb": 0.0,
            "success_delta_lcb": 0.0,
            "aggregate_actionable_success_delta_vs_reactive": 0.0,
            "aggregate_actionable_block_success_delta": 0.0,
            "aggregate_actionable_harmful_group_count": 0,
            "aggregate_actionable_harmful_group_probability_ucb": 0.0,
            "aggregate_actionable_success_delta_lcb": 0.0,
            "aggregate_actionable_success_noninferior": True,
            "opportunity_conditioned_success_rate": (
                reactive_opportunity_success
            ),
            "group_uniform_opportunity_conditioned_success_rate": (
                reactive_opportunity_success
            ),
            "opportunity_conditioned_success_delta_vs_reactive": 0.0,
            "opportunity_conditioned_block_success_delta": 0.0,
            "opportunity_conditioned_success_bound_radius": 0.0,
            "opportunity_conditioned_harmful_group_count": 0,
            "opportunity_conditioned_harmful_group_probability_ucb": 0.0,
            "opportunity_conditioned_success_delta_lcb": 0.0,
            "opportunity_conditioned_success_noninferior": True,
            "success_endpoints_noninferior": True,
            "mean_latency_ms": float(np.dot(observation_weights, reactive)),
            "group_uniform_mean_latency_ms": float(
                np.dot(observation_weights, reactive)
            ),
            "cvar_latency_ms": reactive_raw_cvar,
            "group_uniform_cvar_latency_ms": reactive_raw_cvar,
            "bounded_cvar_latency_ms": reactive_bounded_cvar,
            "group_uniform_bounded_cvar_latency_ms": reactive_bounded_cvar,
            "cvar_lcb_ms": reactive_cvar_lcb,
            "group_uniform_cvar_lcb_ms": reactive_cvar_lcb,
            "cvar_ucb_ms": reactive_cvar_ucb,
            "group_uniform_cvar_ucb_ms": reactive_cvar_ucb,
            "raw_cvar_gain_vs_reactive_ms": 0.0,
            "group_uniform_raw_cvar_gain_vs_reactive_ms": 0.0,
            "cvar_gain_vs_reactive_ms": 0.0,
            "group_uniform_cvar_gain_vs_reactive_ms": 0.0,
            "cvar_gain_lcb_ms": 0.0,
            "group_uniform_cvar_gain_lcb_ms": 0.0,
            "latency_clipped_count": reactive_clipped_count,
            "noninferior": True,
            "practically_better": False,
            "eligible": True,
            "selected": False,
            **common_evidence,
        }
    )

    eligible_learned: list[tuple[float, float, str]] = []
    for name in learned:
        values = arrays[name]
        success = (values <= latency_budget_ms).astype(float)
        differences = success - reactive_success
        (
            block_delta,
            harmful_block_count,
            harmful_probability_ucb,
            success_lcb,
        ) = _exact_harmful_block_lcb(
            differences,
            aggregate_success_alpha,
            blocks,
        )
        opportunity_success_rate = _group_uniform_mean(
            success,
            opportunity_blocks,
        )
        if opportunity_blocks:
            (
                opportunity_block_delta,
                opportunity_harmful_group_count,
                opportunity_harmful_probability_ucb,
                opportunity_success_lcb,
            ) = _exact_harmful_block_lcb(
                differences,
                opportunity_success_alpha,
                opportunity_blocks,
            )
        else:
            # The conditional population does not exist in this selection
            # interval. A learned policy cannot pass an undefined endpoint.
            opportunity_block_delta = math.nan
            opportunity_harmful_group_count = 0
            opportunity_harmful_probability_ucb = 1.0
            opportunity_success_lcb = -1.0
        raw_cvar = _cvar(
            values,
            cfg.cvar_quantile,
            observation_weights,
        )
        bounded_cvar, cvar_lcb, cvar_ucb, clipped_count = (
            _bounded_block_cvar_interval(
                values,
                cfg.cvar_quantile,
                cfg.latency_cap_ms,
                cfg.cvar_grid_points,
                cvar_policy_alpha,
                blocks,
                block_weights,
                observation_weights,
            )
        )
        raw_cvar_gain = reactive_raw_cvar - raw_cvar
        cvar_gain = reactive_bounded_cvar - bounded_cvar
        cvar_gain_lcb = reactive_cvar_lcb - cvar_ucb
        aggregate_success_noninferior = bool(
            success_lcb >= -cfg.noninferiority_margin
        )
        opportunity_success_noninferior = bool(
            opportunity_blocks
            and opportunity_success_lcb
            >= -cfg.opportunity_noninferiority_margin
        )
        success_endpoints_noninferior = bool(
            aggregate_success_noninferior
            and opportunity_success_noninferior
        )
        practically_better = bool(
            cvar_gain_lcb >= cfg.practical_cvar_gain_ms
        )
        eligible = bool(
            opportunity_sufficient
            and success_endpoints_noninferior
            and practically_better
        )
        if eligible:
            eligible_learned.append(
                (
                    cvar_ucb,
                    float(np.dot(observation_weights, values)),
                    name,
                )
            )
        evidence.append(
            {
                "policy": name,
                "success_rate": float(np.dot(observation_weights, success)),
                "group_uniform_success_rate": float(
                    np.dot(observation_weights, success)
                ),
                "success_delta_vs_reactive": float(
                    np.dot(observation_weights, differences)
                ),
                "group_uniform_success_delta_vs_reactive": float(
                    np.dot(observation_weights, differences)
                ),
                "bounded_block_success_delta": block_delta,
                "success_bound_radius": block_delta - success_lcb,
                "harmful_block_count": harmful_block_count,
                "harmful_block_probability_ucb": harmful_probability_ucb,
                "simultaneous_lcb": success_lcb,
                "success_delta_lcb": success_lcb,
                "aggregate_actionable_success_delta_vs_reactive": float(
                    np.dot(observation_weights, differences)
                ),
                "aggregate_actionable_block_success_delta": block_delta,
                "aggregate_actionable_harmful_group_count": (
                    harmful_block_count
                ),
                "aggregate_actionable_harmful_group_probability_ucb": (
                    harmful_probability_ucb
                ),
                "aggregate_actionable_success_delta_lcb": success_lcb,
                "aggregate_actionable_success_noninferior": (
                    aggregate_success_noninferior
                ),
                "opportunity_conditioned_success_rate": (
                    opportunity_success_rate
                ),
                "group_uniform_opportunity_conditioned_success_rate": (
                    opportunity_success_rate
                ),
                "opportunity_conditioned_success_delta_vs_reactive": (
                    opportunity_block_delta
                ),
                "opportunity_conditioned_block_success_delta": (
                    opportunity_block_delta
                ),
                "opportunity_conditioned_success_bound_radius": (
                    opportunity_block_delta - opportunity_success_lcb
                    if math.isfinite(opportunity_block_delta)
                    else 1.0
                ),
                "opportunity_conditioned_harmful_group_count": (
                    opportunity_harmful_group_count
                ),
                "opportunity_conditioned_harmful_group_probability_ucb": (
                    opportunity_harmful_probability_ucb
                ),
                "opportunity_conditioned_success_delta_lcb": (
                    opportunity_success_lcb
                ),
                "opportunity_conditioned_success_noninferior": (
                    opportunity_success_noninferior
                ),
                "success_endpoints_noninferior": (
                    success_endpoints_noninferior
                ),
                "mean_latency_ms": float(np.dot(observation_weights, values)),
                "group_uniform_mean_latency_ms": float(
                    np.dot(observation_weights, values)
                ),
                "cvar_latency_ms": raw_cvar,
                "group_uniform_cvar_latency_ms": raw_cvar,
                "bounded_cvar_latency_ms": bounded_cvar,
                "group_uniform_bounded_cvar_latency_ms": bounded_cvar,
                "cvar_lcb_ms": cvar_lcb,
                "group_uniform_cvar_lcb_ms": cvar_lcb,
                "cvar_ucb_ms": cvar_ucb,
                "group_uniform_cvar_ucb_ms": cvar_ucb,
                "raw_cvar_gain_vs_reactive_ms": raw_cvar_gain,
                "group_uniform_raw_cvar_gain_vs_reactive_ms": raw_cvar_gain,
                "cvar_gain_vs_reactive_ms": cvar_gain,
                "group_uniform_cvar_gain_vs_reactive_ms": cvar_gain,
                "cvar_gain_lcb_ms": cvar_gain_lcb,
                "group_uniform_cvar_gain_lcb_ms": cvar_gain_lcb,
                "latency_clipped_count": clipped_count,
                # Compatibility alias: after the strengthened gate this means
                # that both pre-declared success endpoints are non-inferior.
                "noninferior": success_endpoints_noninferior,
                "practically_better": practically_better,
                "eligible": eligible,
                "selected": False,
                **common_evidence,
            }
        )

    if eligible_learned:
        selected = min(eligible_learned)[2]
        reason = (
            "learned policy passed minimum opportunity-bearing groups, both "
            "simultaneous success bounds, and the bounded-CVaR gain gate"
        )
    elif not opportunity_sufficient:
        selected = reactive_policy
        reason = "insufficient independent opportunity-bearing blocks"
    else:
        selected = reactive_policy
        reason = (
            "no learned policy passed both simultaneous success bounds and "
            "the bounded-CVaR confidence gate"
        )

    for row in evidence:
        row["selected"] = row["policy"] == selected
        row["selection_reason"] = reason
        row["alpha_familywise"] = cfg.alpha
        # Kept for downstream table compatibility; this now also accounts for
        # all planned invocations of the gate.
        row["alpha_per_learned_policy"] = alpha_per_candidate_per_use
        row["alpha_per_candidate_per_gate_use"] = (
            alpha_per_candidate_per_use
        )
        row["alpha_success_bound"] = aggregate_success_alpha
        row["alpha_aggregate_success_bound"] = aggregate_success_alpha
        row["alpha_opportunity_success_bound"] = opportunity_success_alpha
        row["alpha_cvar_comparison"] = cvar_comparison_alpha
        row["alpha_cvar_policy_interval"] = cvar_policy_alpha
        row["alpha_cvar_reactive_interval"] = cvar_policy_alpha
        row["alpha_cvar_candidate_interval"] = cvar_policy_alpha
        row["alpha_cvar_per_grid_bound"] = (
            cvar_policy_alpha / (2.0 * cfg.cvar_grid_points)
        )
        row["noninferiority_margin"] = cfg.noninferiority_margin
        row["aggregate_actionable_noninferiority_margin"] = (
            cfg.noninferiority_margin
        )
        row["aggregate_actionable_success_lcb_threshold"] = (
            -cfg.noninferiority_margin
        )
        row["opportunity_noninferiority_margin"] = (
            cfg.opportunity_noninferiority_margin
        )
        row["opportunity_conditioned_noninferiority_margin"] = (
            cfg.opportunity_noninferiority_margin
        )
        row["opportunity_conditioned_success_lcb_threshold"] = (
            -cfg.opportunity_noninferiority_margin
        )
        row["minimum_effective_opportunities"] = (
            cfg.minimum_effective_opportunities
        )
        row["practical_cvar_gain_ms"] = cfg.practical_cvar_gain_ms
        row["bootstrap_samples_used"] = 0

    return RiskControlSelection(
        selected_policy=selected,
        reason=reason,
        evidence=tuple(dict(row) for row in evidence),
    )


def risk_control_config_to_dict(config: RiskControlConfig) -> dict[str, object]:
    return asdict(config)
