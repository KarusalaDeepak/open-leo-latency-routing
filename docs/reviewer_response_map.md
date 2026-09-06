> **SUPERSEDED (2026-08-21).** This historical review note is not a source of current claims or counts. Use [`README.md`](../README.md), the `main.tex` in the `leo-conf2-transactions-revision-2026-08-21` submission bundle, and [`results/transactions_evidence/`](../results/transactions_evidence/) as the canonical sources.

# Reviewer-to-Code Evidence Map

This map identifies the executable artifact behind every response. The
canonical paper tables and figures are assembled by
`scripts/build_transactions_evidence.py` in
`results/transactions_evidence/`.

## Reviewer 1

### R1.1: System figure and notation

- `scripts/generate_result_figures.py` builds the system assets.
- `results/figures/manuscript_assets/` contains the generated source assets.
- The final manuscript uses `leo_system_model.png`; notation is defined in
  Table II, and Algorithms 1--2 show calibration and execution explicitly.

### R1.2: Generalization beyond one LENS release

- WetLinks adds 116,954 usable source rows across two independent European
  Starlink sites and 148--179 days of coverage.
- The five-minute adapter yields 30,221 shared observation epochs and retains
  only exact 300-second forecast targets.
- A chronological 60/15/10/15 train/calibration/selection/test protocol tests
  late-period behavior without choosing a model on the test interval.
- Bidirectional unseen-site transfer fits no labels from the target site.
- Machine-readable metadata marks the geographically distinct sites as invalid
  for online service-path selection, preventing a stronger claim than the data
  supports.

- `scripts/build_commect_multiaccess_trace.py` converts the independent
  COMMECT campaign with two 5G links and Starlink into timestamp-concurrent
  alternatives.
- `scripts/run_commect_multiaccess_validation.py` uses strict chronological
  train/calibration/test partitions and exports all per-decision outcomes.
- `scripts/build_victoria_multihomed_trace.py` and
  `scripts/run_measured_multihomed_validation.py` evaluate a previously unused
  concurrent two-terminal LENS interval.
- `scripts/generate_physics_informed_multipath_trace.py` creates 18 concurrent
  orbital alternatives from geometric propagation, queue, handover,
  attenuation, and service-event state.
- `scripts/run_independent_multipath_seed_matrix.py` regenerates and retrains on
  ten independent orbital traces.
- `scripts/generate_hypatia_service_trace.py` provides an established orbital
  dynamic-state replication; it is not mislabeled as measured policy evidence.
- `scripts/build_external_irtt_table.py` evaluates portable prediction fields
  on independent Starlink IRTT measurements; the metadata prevents this
  single-path source from being reported as policy-level generalization.

### R1.3: Synthetic degradation disclosure

- Moderate and severe perturbations are injected only into simulator test rows
  after clean model fitting and calibration.
- Every output records `source_type`, `evaluation_case`, and `scenario_name`.
- Measured COMMECT and Victoria rows receive no injected perturbation.

### R1.4: Lightweight operational selector

- `optimization/policies.py:add_qos_shielded_scores` implements the final
  shield score, and `select_validation_gated_fallback` implements deployment
  abstention.
- Mixed-feasibility snapshots use a current-state QoS safeguard.
- Uniformly feasible or infeasible snapshots use the clean-validation-selected
  reactive/context/ensemble fallback. Reactive is a recorded abstention.
- `qos_fallback_comparison.csv` evaluates fixed-context and fixed-ensemble
  fallbacks behind the same shield, so fallback selection is not conflated
  with the safeguard.
- `qos_branch_frequency.csv` records how often each executable branch is used
  in every measured and injected case.
- `qos_branch_outcomes.csv` records next-epoch success, current-to-next
  violation/recovery rates, tail latency, and decision gap for each branch.
- `operational_secondary_metrics.csv` compares reactive, predictive, and
  QoS-shielded selectors with equal 10-ms hysteresis controls.
- `mean_model_scoring_time_us` and `mean_model_and_ranking_time_us` separate
  inference cost from ranking cost.

### R1.5: Ensemble behavior

- `results/reviewer_validation/disagreement_diagnostics.csv` compares raw
  disagreement and ensemble spread against realized error.
- `results/transactions_evidence/component_ablation.csv` compares temporal,
  context, fusion, disagreement, ensemble, calibrated risk, and QoS shield.
- Fallback selection is based on clean validation rather than a test-time claim
  that one uncertainty signal always wins.

## Reviewer 2

### R2.1: Physical and mathematical grounding

- `optimization/calibrated_risk.py` implements bias correction,
  inverse-residual-variance weighting, mixture variance, and non-negative
  residual-risk calibration.
- Squared expert disagreement is the between-expert term in the exact
  law-of-total-variance decomposition; it is not coded as a guaranteed shift
  label.
- `models/orbital_physics.py` and
  `scripts/run_physical_feasibility_analysis.py` generate propagation and
  control-horizon bounds.
- Shared-failure diagnostics quantify cases in which both experts are wrong but
  raw disagreement is small.
- The code exports branch-conditioned persistence rates used by the
  manuscript's conditional next-epoch proposition; no unconditional guarantee
  is asserted.

### R2.2: Dataset dependence

- `scripts/build_wetlinks_longitudinal_table.py` provides a second independent
  commercial Starlink adapter with source-file SHA-256 hashes.
- `scripts/run_wetlinks_longitudinal_validation.py` exports late-period,
  unseen-site, risk-diagnostic, and paired day-block evidence.
- The result reproduces a useful negative finding outside LENS: disagreement
  high-error AUROC is 0.511, so the manuscript does not treat it as a universal
  detector.

- `data/loaders.py` enforces a portable canonical schema and audits timestamp
  concurrency.
- A trace without at least two actual-time alternatives fails by default.
- Normalized-session alignment is available only through an explicit
  counterfactual flag and is marked non-deployable in metadata.
- Policy claims are restricted to COMMECT, LENS Victoria, and the concurrent
  orbital trace.

### R2.3: Control-loop latency

- Policy evaluation adds collection, model scoring, ranking, and dissemination
  delay to network latency.
- `results/transactions_evidence/control_loop_sensitivity.csv` evaluates the
  resulting QoS and P95 degradation.
- Stale-state sensitivity shifts decisions by complete forecast bins.
- `evaluation/delayed_execution.py` replays the selected path against the later
  trace state, including later path availability and measured/generated latency.
- `delayed_state_replay.csv` and
  `commect_rolling_delayed_state_replay.csv` separate network-only stale-state
  degradation from end-to-end waiting-time infeasibility.
- A decision is flagged stale when the control loop reaches the associated
  decision horizon.

### R2.4: OLS/XGBoost capacity mismatch

- The default temporal and context experts are both Ridge regressors.
- Their diversity is created by disjoint information views, not capacity.
- `matched_model_pair_audit.csv` and `predictor_combination_audit.csv` repeat
  the analysis across linear, Ridge, tree, and small-MLP families.

### R2.5: Temporal resolution

- `scripts/build_temporal_resolution_tables.py` rebuilds 5-, 10-, 30-, and
  60-second inputs without future filling.
- `scripts/run_temporal_resolution_evaluation.py` reruns each matching
  next-bin horizon.
- Final policy evidence uses 5-second orbital and 10-second measured decisions;
  60 seconds is not presented as the only short horizon.
- `timestamp_alignment_audit.csv` reports COMMECT's source-time skew after
  binning, preventing bin alignment from being mislabeled as simultaneous
  packet probing.

### R2.6: Weaker standalone context expert

- Expert biases and variances are estimated on clean validation.
- Inverse-variance weighting prevents a noisier expert from receiving equal
  trust by assumption.
- The model audit reports standalone MAE and disagreement/error correlation;
  complementarity is tested rather than inferred from model complexity.

## Reviewer 3

### R3.1: Motivation

The feature pipeline models two failure-relevant views: path-local persistence
and simultaneous peer context. Diagnostics separately report raw disagreement,
ensemble spread, and shared failures. The final selector retains a QoS safeguard
because no uncertainty statistic is universally proactive.

### R3.2: Predictor-family portability

`matched_model_pair_audit.csv` covers four matched families and
`predictor_combination_audit.csv` covers their complete 4-by-4 cross-product.
The default method does not depend on linear-versus-XGBoost divergence.

### R3.3: Related-work attribution

This is addressed in manuscript Sec. II and Table I. Code cannot establish
literature positioning, so no software-only claim is made here.

### R3.4: Contribution separation

- Algorithm: `src/open_leo_latency_routing/optimization/`.
- Data and evaluation: canonical adapters and `scripts/run_*` programs.
- Reproducibility: configurations, checksums/metadata, tests, result builders,
  and `docs/reproducibility_guide.md`.

### R3.5: Weight selection

- Model fitting uses train rows only.
- Bias, variance, fallback, risk coefficients, and threshold use disjoint clean
  validation rows only.
- Injected test labels and next-epoch test outcomes never tune a parameter.
- Regression tests protect partition disjointness and exclude all `target_*`
  fields from online features.

### R3.6: Ablation

The canonical ablation includes temporal-only, context-only, simple fusion,
disagreement-only, ensemble uncertainty, split conformal, calibrated risk/gate,
fixed-context shield, fixed-ensemble shield, predictive-only QoS shield, and
validation-gated QoS shield. All variants use the same split and evaluation
rows.

## Decision Identifiability Bound

`evaluation/decision_opportunity.py` exports the finite-sample bound that the
absolute success-rate gap between any two policies cannot exceed the fraction
of epochs containing both successful and failed runtime candidates. The
canonical audit checks 2,400 policy-pair/threshold/case combinations and all
bounds hold. This is an evaluation limit, not a safety or optimality claim.

## Exact Explainability

`optimization/explainability.py` exports the actual executed branch, selected
path, nearest rejected path, score margin, current QoS class, and
counterfactual component differences. The final branch values are:

- `mixed_qos_safeguard`
- `all_qos_fallback`
- `no_qos_fallback`

For the calibrated-risk ablation, latency, disagreement, ensemble uncertainty,
service risk, switching, and calibration components sum exactly to its online
score. For the final shield, the explanation describes the real lexicographic
branch rather than applying a post-hoc surrogate.

`xai_case_studies.csv` exports one representative frozen-test decision for
each branch, including selected and runner-up paths, score margin, realized
next latency, outcome, counterfactual reason, and score-reconstruction error.

## Statistical Evidence

- `paired_block_significance.csv` contains paired Wilcoxon tests, Holm-adjusted
  p-values, effect sizes, win/loss rates, and segment-stratified circular
  moving-block bootstrap p-values.
- `block_bootstrap_confidence_intervals.csv` preserves within-segment temporal
  dependence without allowing blocks or circular wraps to cross telemetry-gap,
  session/campaign, continuity-segment, or rolling-fold boundaries. The output
  records the segment columns and segment count used for each metric.
- `multiseed_orbital_summary.csv` reports Student-t intervals over ten
  independently regenerated traces.
- `short_30_seed_policy_summary.csv` and `short_30_seed_pairwise_deltas.csv`
  report the separate 30-seed short-trace extension without pooling protocols.
- Holm correction is performed within a metric family, not across unrelated
  outcomes.
- Exact ties are retained as ties rather than discarded from win-rate claims.

## Reproducibility Checks

- `pytest.ini` fixes test discovery and local package resolution.
- `tests/` contains 26 passing leakage, concurrency, policy, calibration,
  validation-gating, delayed-replay, opportunity-bound, XAI, and statistics
  regressions.
- `scripts/build_transactions_evidence.py` is the canonical manuscript evidence
  builder.
- `evidence_manifest.json` stores a SHA-256 digest and byte count for every
  canonical table and figure.
- `docs/final_reviewer_implementation_status.md` records the final numerical
  results and remaining evidence boundary.
