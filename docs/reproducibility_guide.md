# Reproducibility Guide

This guide lists the exact repo-level commands used to regenerate the primary
results discussed in the manuscript.

The canonical evidence build additionally writes
`results/transactions_evidence/validation_gate_selection_audit.csv`, which
records the predictive-only and evidence-gated fallback frozen before every
fixed or rolling test interval. Its numerical audit recomputes policy summaries
from decision-level outputs and fails closed on any fold-order or denominator
inconsistency. Gate-selection evidence separately exports group-uniform lower
bounds for aggregate-actionable success over all declared independent groups
and opportunity-conditioned success over independent groups containing at
least one decision opportunity; both must clear their configured
non-inferiority thresholds before admission.

### Canonical artifact terminology

Some machine-readable column names are retained for backward compatibility
with the rejected-paper artifact. In those files, `graph`, `pred_graph`, and
`graph_expert_weight` denote the tabular peer-context ridge view; they do not
denote a graph neural network. The legacy configuration/output key
`minimum_effective_opportunities` is the same quantity exported canonically as
`minimum_opportunity_bearing_groups`: a floor on independently acquired groups
that contain at least one post-hoc decision opportunity, not a count of epochs
or analyst-created time blocks. Likewise, the reported `block_length` field is
only the observed maximum size of an explicit acquisition group. The
production gate rejects a configured time-block length as evidence of
independence and defaults to one collection if provenance identifiers are
absent.

## Environment

- Repository root: `leo-conf2-open-leo`
- Main configuration: `configs/experiment.yaml`
- Tested direct/test dependency pins: `requirements-lock.txt` (CPython 3.11.5,
  macOS arm64). The schema-3 build provenance records the complete installed
  environment; this compact pin file is not a cross-platform, hash-locked
  transitive environment.
- Default random seeds:
  - In-distribution split: `42`
  - Operational mild shift: `101`
  - Operational moderate shift: `102`
  - Operational severe shift: `103`
  - Bootstrap confidence intervals: `42`
  - Stochastic switching Monte Carlo: `42`

## Data Preparation

Build the processed time-bin table from the raw LENS release:

```bash
python3 scripts/build_ping_tables.py --config configs/experiment.yaml
```

The default configuration uses 60-second bins for the legacy LENS prediction
diagnostic. Final policy evidence uses the dedicated 10-second measured and
5-second orbital adapters. If a compatible processed table is available at
another resolution, update `dataset.time_bins_path` and the matching
graph/forecast horizon settings in `configs/experiment.yaml`.

Inspect the dataset inventory and generate the data card:

```bash
python3 scripts/build_manifest.py
python3 scripts/build_data_card.py
```

## Core Forecasting and Structural-Shift Results

Run temporal forecasting baselines:

```bash
python3 scripts/run_temporal_forecasting.py --config configs/experiment.yaml
```

Run graph-context forecasting:

```bash
python3 scripts/run_graph_forecasting.py --config configs/experiment.yaml
```

Run the main path-selection structural-shift evaluation:

```bash
python3 scripts/run_service_path_experiments.py \
  --config configs/experiment.yaml \
  --output-dir results/service_path_structural_shift_analysis
```

The command above regenerates:

- `policy_summary.csv`
- `policy_decisions.csv`
- `policy_significance.csv`
- `policy_confidence_intervals.csv`
- `disagreement_calibration_summary.csv`
- `stratified_disagreement_analysis.csv`
- `switching_cost_summary.csv`
- `stochastic_switching_summary.csv`
- `multi_bin_summary.csv`
- `run_metadata.json`

Important interpretation note:

- the mild/moderate/severe scenarios are synthetic evaluation perturbations;
- they are applied only to the evaluation split;
- they should be described as stress tests, not as claims that the raw LENS
  release contains those outage traces.

## Extended Robustness Analyses

Run sensitivity sweeps and lightweight temporal-model comparisons:

```bash
python3 scripts/run_path_selection_sensitivity.py \
  --config configs/experiment.yaml \
  --output-dir results/path_selection_sensitivity_analysis
```

This command regenerates:

- `hyperparameter_sensitivity.csv`
- `temporal_model_comparison.csv`
- `horizon_sensitivity.csv`
- `ensemble_member_sensitivity.csv`
- `path_selection_sensitivity_metadata.json`

## Larger-Subset Check

Run the 64-log scale-consistency analysis:

```bash
python3 scripts/run_larger_lens_subset.py --config configs/experiment.yaml
```

## External-Source Measured Multi-Access Shadow Replay

Download the public Aalborg University COMMECT archive, verify its published
MD5, aggregate timestamp-aligned Operator A 5G, Operator B 5G, and Starlink
measurements into common 10-second bins without interpolation, and run the
policy matrix plus its linked sensitivity and design audits:

```bash
./scripts/download_commect_multiaccess_data.sh
python3 scripts/build_commect_multiaccess_trace.py
python3 scripts/run_commect_multiaccess_validation.py
python3 scripts/run_commect_rolling_origin_validation.py
python3 scripts/run_commect_rolling_timestamp_sensitivity.py
python3 scripts/run_commect_threshold_gate_sensitivity.py
python3 scripts/run_commect_timestamp_sensitivity.py
python3 scripts/audit_gate_design_sensitivity.py
```

The builder first verifies the published archive MD5 and then requires each
extracted CSV to be byte-identical (SHA-256 and size) to a member of that
verified ZIP; basename matching does not assume a particular archive-root
folder. The generated metadata records those member matches, DOI
`10.5281/zenodo.14620779`, CC BY-SA 4.0 licensing, source-row counts,
timestamp assumptions, and the actual-time concurrency audit. Boundaries are
first frozen on the raw 633-slot exact-cadence wall-clock schedule (603
observed slots plus 30 gaps), before target availability, completeness, or
skew filtering. They separate training, calibration, policy selection, and
testing; any row whose exact next-bin target crosses a boundary is removed.
The fixed protocol retains 312/89/86/88 decision epochs in those four blocks.
COMMECT is one continuous drive, so `session_date`
defines one statistical collection unit. The conditional evidence gate must
therefore abstain even though the trace contains many raw decision
opportunities. This is external-source measured shadow-policy replay from one
drive, not zero-shot transfer, installed-action validation, or independent
replication across drives.

The rolling timestamp-skew command rebuilds the complete five-fold protocol at
0.5-, 1-, 2-, and 5-second maximum inter-path skew and on the full validated
data. These overlapping rebuilds are post-specified robustness views, not
independent replications or a family from which to select a favorable policy.
The gate-design audit evaluates the declared 108-setting planning grid and its
analytical independent-group requirements. It does not tune the released gate
from test outcomes; every measured COMMECT setting remains limited to one
opportunity-bearing collection and therefore abstains.

Descriptive confidence intervals and centered-null bootstrap diagnostics use a
segment-stratified circular moving-block bootstrap. Fixed evaluations pass the
audited `continuity_segment_id`, which resets at telemetry gaps and explicit
session/campaign changes. The pooled rolling evaluation passes both
`rolling_fold` and `continuity_segment_id`. Segment sample sizes are held fixed,
wraps remain inside a segment, and metric-specific missing rows create another
break, so no resampled block can bridge any of those boundaries. These
intervals remain descriptive and do not manufacture independent collections.

## Co-located Victoria holdout

Build the non-overlapping 12-session holdout and evaluate it:

```bash
python3 scripts/build_victoria_multihomed_trace.py \
  --session-count 12 --session-offset 100 \
  --output data/processed/lens_victoria_multihomed_holdout_10s.csv
python3 scripts/run_measured_multihomed_validation.py
```

The builder resolves duplicate path--epoch rows introduced by overlapping
hourly-file boundary samples using reply count only, then asserts uniqueness.
The final test contains 648 two-path decisions.
The co-located terminals support candidate-outcome shadow replay, but they do
not establish literal single-controller steering authority or closed-loop
deployment evidence; those claim fields are exported separately.

## Independent WetLinks Longitudinal Validation

Download the open WetLinks release, retain the compact merged analysis tables,
build exact five-minute bins, and run the late-period and unseen-site audits:

```bash
./scripts/download_wetlinks_data.sh
python3 scripts/build_wetlinks_longitudinal_table.py
python3 scripts/run_wetlinks_longitudinal_validation.py
python3 scripts/generate_wetlinks_validation_figure.py
```

The adapter records source-file SHA-256 hashes, the CC BY-SA 4.0 license,
148--179-day coverage, and 30,221 shared observation epochs. Only exact
300-second targets are retained. Unique epochs are divided chronologically into
60% training, 15% residual calibration, 10% predictor selection, and 15% final
evaluation. The two sites are geographically distinct; metadata therefore
marks this as prediction and calibration transfer rather than online
service-path policy evidence.

## Optional, Unverified Hypatia Adapter Regeneration

In a separate environment, install the provisional optional dependencies and
clone the exact official Hypatia revision:

```bash
./scripts/setup_hypatia.sh
python3 scripts/generate_hypatia_service_trace.py
python3 scripts/run_zero_shot_transfer_validation.py
```

The canonical rebuild consumes a hashed, pre-generated Hypatia trace; it does
not execute this optional dependency stack. The adapter uses Hypatia `satgenpy` TLE orbital propagation, plus-grid
inter-satellite links, and dynamic shortest-path state. The zero-shot runner
fits and calibrates only on the separate physics-informed source and replays
the frozen policy on the three concurrent Hypatia service replicas; this
avoids manufacturing a four-way path-group split from three replicas. It does
not run the optional Hypatia ns-3 packet simulator, and the metadata explicitly
prevents that stronger claim. Treat setup completion as source/dependency
preparation only; end-to-end adapter compatibility remains unverified until
trace generation and the relevant tests succeed on the target platform.

## Notes

Run the claim-readiness audit after regenerating the experiment artifacts:

```bash
python3 scripts/audit_reviewer_readiness.py --allow-pending
```

The `--allow-pending` flag does not convert missing evidence into a pass. It
keeps the command usable while the report explicitly retains the two
data-dependent blockers: synchronized commercial multi-LEO replay and
closed-loop action/queue validation.

To reproduce the synthetic calibration/non-vacuity check for the evidence
gate, run:

```bash
python3 scripts/audit_gate_operating_characteristics.py
```

This audit evaluates declared independent groups under synthetic null,
unsafe, moderate-benefit, and overwhelming-benefit cases. It is not measured
policy efficacy and must not be pooled with COMMECT or LENS results.

To reproduce the separate prospective gate-design and minimum-sample audit,
run:

```bash
python3 scripts/audit_gate_design_sensitivity.py
```

This command evaluates the predeclared Cartesian planning grid and analytical
best-case group-count floors. The settings are not candidates for retrospective
test-time selection, and the output is not empirical coverage or practical
power evidence.

For a single canonical rebuild after all source datasets have been prepared,
run:

```bash
python3 scripts/rebuild_transactions_artifact.py
```

Use `--dry-run` to inspect every resolved command. `--quick` skips the long
seed and secondary diagnostic matrices only when the prior provenance matches
the current code/configuration hash, Python dependency fingerprint, every
canonical external input (including the candidate manifest and the exact raw
files it selects), and every file under every reused secondary-output tree.
Missing, stale-extra, size-changed, or hash-changed reuse is rejected and
requires a full rebuild.

The runner builds `transactions_evidence` in a fresh sibling directory. It
runs the reviewer audit, full test suite, refreshes the evidence manifest, and
verifies the exact file set, sizes, and SHA-256 values before swapping the
stage into the published location. The prior evidence tree is moved to
`output/build_backups/` only after those checks pass, and is restored if the
directory swap fails. An already-published tree can be checked without
regeneration:

```bash
python3 scripts/build_transactions_evidence.py \
  --output-dir results/transactions_evidence --verify-manifest
```

- All operational-shift perturbations are injected only into evaluation data.
- The retrospective best path is used only for post-hoc evaluation metrics such
  as decision gap; it is not an online deployable baseline.
- Figure-generation scripts read the CSV outputs above and do not alter the
  underlying metrics.
- The calibrated multi-signal selector is a lightweight operational rule. It is not
  positioned as a new predictor family.
- The ensemble uncertainty selector is kept as a comparator because it can
  become more conservative under moderate/severe degradation when predictor
  spread grows.

The required prospective logging, causal ordering, failure accounting, and
claim gate are specified in `docs/closed_loop_field_validation_protocol.md`.
Author-controlled licensing, tagging, DOI, disclosure, and venue checks are
listed in `docs/release_readiness_checklist.md`.
