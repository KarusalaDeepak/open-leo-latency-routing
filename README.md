# Open LEO Latency Routing

This repository contains the reproducible code for opportunity-aware,
evidence-gated latency-QoS path selection on open LEO and heterogeneous-access
network measurements. The primary replay endpoint is next-epoch network RTT;
controller timing is reported separately and is not relabeled as end-to-end
service latency.

## Scope

The project scope is:

1. time-series forecasting of short-horizon LEO network state,
2. graph-based learning on temporal network snapshots,
3. decision optimization on predicted network state,
4. conservative pre-test evidence gating with explicit abstention.

The implementation is schema-driven. Any other open LEO measurement table can be
used if it exposes the same processed time-bin columns as the current LENS
release and the `time_bins_path` entry in `configs/experiment.yaml` is updated.

The default experts use the same standardized, fixed-alpha Ridge estimator
family but separate information views and feature dimensionalities:
the temporal expert receives path history, while the context expert receives
only concurrent neighborhood features. Their deterministic linear pool is fit
from paired calibration residual variances and covariance. Predictor
disagreement remains a separately calibrated diagnostic and ranking feature;
it is not treated as an independent expert or a distribution-free shift
detector.

The proposed rule is a lightweight evidence-gated QoS shield, not a new
prediction family. When an epoch mixes currently QoS-compliant and
non-compliant paths, the shield chooses the lowest-current-latency compliant
path. Outside that branch, a disjoint pre-test selection interval may admit one
frozen predictive fallback only if it passes the declared opportunity floor,
separate exact harmful-group QoS non-inferiority bounds for the aggregate
actionable population and for post-hoc decision opportunities within
opportunity-bearing groups, and a bounded CVaR-improvement test with
simultaneous multiplicity control. Familywise alpha covers both success
endpoints, the reactive and candidate CVaR intervals, all learned candidates,
and all planned gate uses. Otherwise the rule uses reactive fallback, which is
explicit abstention from prediction. Independent inference groups must be
supplied by the collection protocol; if none are declared, the entire
selection interval is conservatively treated as one group. The predictive-only
context/ensemble shield remains an ablation. No evaluation outcome or injected
test label tunes either rule.

The clipped-CVaR endpoint uses a predeclared 60-second cap and a 1,200,001-point
grid (0.05-ms spacing), whose worst between-grid correction is 0.475 ms at
q=0.95. Canonical copies of the synthetic operating-characteristic audit are
provided under `results/transactions_evidence/`. They confirm that the implementation
can admit an overwhelming positive control, while also showing that the
distribution-free cap has very low power for a 20-ms effect even with 5,000
independent groups. This is a calibration/non-vacuity audit, not measured
efficacy.

The ensemble uncertainty selector is an additional uncertainty-aware comparator.
It trains small bootstrapped temporal models, then scores each path using the
ensemble mean, ensemble spread, and observed service-risk features.

The component ablation separately evaluates temporal-only, graph-only, plain
fusion, calibrated fusion, disagreement-only, learned risk without
disagreement, ungated learned risk, ensemble-only, and the full conditional
rule.

The mild, moderate, and severe structural-shift settings are injected only into
evaluation data. They are controlled stress tests, not raw outage traces.

Key decision metrics reported by the repo include:

- mean realized next-epoch network RTT,
- mean decision gap against a retrospective best-path benchmark,
- `retrospective_best_path_match_rate`, which measures how often a policy chooses the same path as the post-hoc reference path,
- success rate under the configured latency budget,
- P95 and CVaR95 realized latency,
- switch rate and equal-information hysteresis controls,
- model-scoring and total controller runtime in microseconds, reported
  separately from network RTT,
- exact branch-conditioned next-epoch outcomes and explanation fidelity,
- finite-sample pairwise success-gap bounds based on decision opportunity,
- delayed-state trace replay over zero to three execution-delay bins.

Stateful selectors reset their previous-path state, hysteresis penalty, and
switch counter at telemetry gaps or session/campaign boundaries. Exported
decision rows record the continuity segment and reset reason; switch rates use
only eligible transitions within a continuous segment. Policy summaries export
`switch_count` as the numerator and `eligible_switch_transition_count` as the
denominator. The legacy `switch_transition_count` field is retained only as a
compatibility alias for that denominator; it is not the number of switches.

The manuscript assets generated by `scripts/generate_result_figures.py` are
written under `results/figures/` and `results/figures/manuscript_assets/`.
Those include a methodology overview, a physical LEO service-path/control-loop
model, and a notation table draft for the Section III revision requested by
reviewers.

## Dataset Plan

Datasets:

- LENS 2025-03 open measurement release, DOI
  `10.5281/zenodo.15331299`
- independent Starlink IRTT measurements, DOI `10.17632/479v4mym7j.2`,
  CC BY 4.0
- independent WetLinks longitudinal Starlink measurements from Enschede and
  Osnabrueck, approximately six months, CC BY-SA 4.0
- independent concurrent COMMECT measurements with two 5G operators and one
  Starlink access link, DOI `10.5281/zenodo.14620779`, CC BY-SA 4.0
- schema-driven support for an authorized independent commercial Starlink--
  OneWeb trace; the importer enforces hard 30-day, 24-concurrent-hour, 95%
  complete-epoch, and 100-ms P95 synchronization claim floors
- separately generated physics-informed concurrent-path simulator trace with
  elevation-dependent propagation, handovers, queueing, gateway attenuation,
  and satellite service incidents
- official Hypatia TLE-derived dynamic routing state at commit
  `0ac531c313eba2335f6344b46347140c3a0d4230`

Expected local dataset path after download and extraction:

- `data/raw/lens_2025_03/`

### LENS acquisition and provenance

Acquire the LENS 2025-03 release from its official Zenodo record,
<https://doi.org/10.5281/zenodo.15331299>, and extract the release beneath
`data/raw/lens_2025_03/` so that the dataset root is
`data/raw/lens_2025_03/LENS-2025-03/`. Before extraction, record the archive
filename, byte size, and checksum published by the release, compute the local
archive checksum, and require an exact match. This repository does not invent
or hard-code an unverified archive checksum; release checksums must be verified
against the Zenodo record at acquisition time. Generated artifact hashes are
recorded separately in `results/transactions_evidence/evidence_manifest.json`;
the canonical rebuild verifies that manifest against the exact file set,
sizes, and SHA-256 values before publishing the staged evidence tree.

## Repository Layout

- `configs/`: experiment configuration files
- `docs/`: project notes, scope, and data cards
- `scripts/`: runnable entrypoints for inspection and experiments
- `src/open_leo_latency_routing/`: source package
- `data/raw/`: raw downloaded and extracted data
- `data/processed/`: cleaned intermediate artifacts
- `results/`: outputs, figures, metrics, and logs

Raw LENS downloads and processed intermediate data tables are ignored by Git.
The release package intentionally tracks compact result CSV/JSON files and
generated figures that support the manuscript tables and plots. It is therefore
source-, protocol-, and evidence-verifiable but not raw-data self-contained;
full recomputation requires separately acquiring and checksum-verifying the
third-party inputs under their own terms.

The candidate synchronized Starlink--OneWeb campaign is not currently exposed
through an identified public raw archive. Its access status and the prepared
author request are recorded in the working repository. The curated submission
archive includes `docs/commercial_multileo_acquisition_status.json` but excludes
email correspondence and drafts. Until authorized traces are
received and pass the temporal, complete-GPS co-location (hard maximum 100 m),
and shared-controller topology gates, the independent longitudinal commercial
multi-LEO path-selection claim remains disabled. Timestamp overlap alone is
accepted only as a scoped paired comparison. Commercial gate inference also
defaults to one complete campaign; multiple inference groups require complete
paired campaign IDs and an explicit documented independence audit.

## Actual Ingestion Outputs

The LENS release is primarily a large collection of line-oriented `ping` logs rather than ready-made CSV tables. The repo converts those logs into three compact artifacts:

1. `ping_session_summary.csv`: one row per measurement session,
2. `ping_observations_sample.csv`: sparse observation-level samples for inspection,
3. `ping_time_bins.csv`: fixed-width time-bin aggregates for forecasting and graph snapshots.

## Reproducibility Guide

For the manuscript-level commands and expected outputs, see
`docs/reproducibility_guide.md`.

## Modeling Pipeline

After the raw logs are parsed, the repo supports four direct experiment stages:

1. next-bin forecasting and decisions at 5/10/30/60-second cadences,
2. graph-aware learning on normalized session snapshots,
3. decision optimization using reactive and predictive policies,
4. robustness evaluation under burst and outage shifts.

The current study version also includes five reviewer-driven
analyses:

1. disagreement-bin validation to test whether predictor disagreement behaves
   like an uncertainty signal,
2. a correlated structural-shift stress that degrades multiple location groups
   across aligned decision windows,
3. an ensemble uncertainty baseline for comparison against the proposed
   calibrated multi-signal operational selector,
4. conformal uncertainty, switching-cost, stochastic handover, multi-bin, and
   sensitivity analyses,
5. a larger-subset runner that rebuilds the time-bin table from more LENS ping
   files when raw logs are available locally.

## Quick Start

```bash
cd opportunity-aware-evidence-gating-artifact
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
./scripts/download_commect_multiaccess_data.sh
python scripts/run_commect_multiaccess_validation.py
python scripts/run_commect_rolling_origin_validation.py
python scripts/build_commercial_multileo_trace.py --help
python scripts/run_commercial_multileo_validation.py --help
python scripts/inspect_dataset.py --data-root data/raw/lens_2025_03
python scripts/build_data_card.py --data-root data/raw/lens_2025_03
python scripts/build_manifest.py --data-root data/raw/lens_2025_03/LENS-2025-03 --top-k 24 --max-per-location 4 --max-per-day 2
python scripts/build_ping_tables.py --data-root data/raw/lens_2025_03/LENS-2025-03 --max-files 16 --time-bin-seconds 60
python scripts/run_temporal_forecasting.py
python scripts/run_graph_forecasting.py
python scripts/run_decision_policy_evaluation.py
python scripts/run_robustness_evaluation.py
python scripts/generate_result_figures.py
python scripts/run_service_path_experiments.py --config configs/experiment.yaml --output-dir results/service_path_reviewer_revision --allow-normalized-counterfactual
python scripts/build_temporal_resolution_tables.py
python scripts/run_temporal_resolution_evaluation.py
python scripts/build_external_irtt_table.py
python scripts/run_external_dataset_validation.py
./scripts/download_wetlinks_data.sh
python scripts/build_wetlinks_longitudinal_table.py
python scripts/run_wetlinks_longitudinal_validation.py
python scripts/generate_wetlinks_validation_figure.py
python scripts/generate_physics_informed_multipath_trace.py
python scripts/run_independent_multipath_validation.py
python scripts/run_independent_multipath_seed_matrix.py
python scripts/run_simulator_parameter_sensitivity.py
python scripts/run_reviewer_validation.py
python scripts/audit_gate_operating_characteristics.py
python scripts/audit_reviewer_readiness.py --allow-pending
python scripts/run_physical_feasibility_analysis.py
python scripts/rebuild_transactions_artifact.py
```

The selected normalized LENS sessions have no simultaneously measured
alternative paths. Their policy comparison is therefore an explicitly labeled
counterfactual diagnostic. Literal single-controller shadow replay is reported
only for traces that pass both the timestamp-concurrency and controller-topology
claim gates. COMMECT supports
external-source measured multi-access shadow replay; Victoria supports only a
co-located candidate-outcome counterfactual because common steering authority
is unverified. Simulator studies are identified separately. None is a
closed-loop deployment trial.

The COMMECT result is heterogeneous-access shadow replay of one continuous
drive, not independent commercial multi-LEO validation, repeated-drive
replication, or universal dominance. Split boundaries are frozen on the raw
633-slot exact-cadence schedule before target filtering. With globally closed
chronological partitions, reactive and evidence-gated policies obtain 0.648
success (57/88) on the fixed holdout. Across five expanding folds, reactive and
evidence-gated success are both 0.678 (196/289); the unadmitted predictive
shield obtains 0.671 (194/289).
The campaign contains only one independent collection unit, so the evidence
gate abstains in every fold. This negative
result is the intended fail-closed behavior, not post-hoc test selection.

The Hypatia adapter is optional and non-canonical. Its provisional dependency
set is not exercised by the canonical rebuild, which consumes a hashed prepared
trace. Run `scripts/setup_hypatia.sh`, trace generation, and adapter validation
only in a separate environment after validating those optional dependencies on
the target platform.

The canonical Victoria command builds the 12-session holdout at offset 100.
Overlapping hourly-file boundary bins are resolved before evaluation, and an
assertion enforces one row per terminal and timestamp. The resulting holdout is
nearly saturated at 60 ms and therefore supports a co-located candidate-outcome
counterfactual and non-regression check, not literal steering or learned-policy
superiority.

The Transactions evidence builder also re-executes the shield at 40, 60, 100,
and 200 ms with frozen experts and fallback selection. Its generated numerical
audit fails the build if policy denominators, opportunity partitions, or
rolling-fold boundaries are inconsistent. It also recomputes decision counts,
mean latency, decision gap, P95, CVaR, and switch rate from decision-level rows.
The evidence builder fails on numerical-consistency violations, and the
regression suite is executed by the canonical rebuild rather than represented
by a stale hard-coded count. The canonical build includes an explicit
`validation_gate_selection_audit.csv`, delayed-state replay, and pairwise
success-gap evidence. A separate 30-seed
short-trace extension is reported as a saturated replication/non-regression
audit and is not pooled with the principal ten-seed 2-h protocol.

WetLinks is used for a separate five-minute longitudinal audit. Its two sites
are geographically distinct, so overlapping timestamps support a distributed
context-prediction test but never an online path-selection claim. The adapter
records this restriction in machine-readable metadata and evaluates both a
late-period holdout and bidirectional unseen-site temporal transfer.

Prospective evidence and release actions are specified in
`docs/closed_loop_field_validation_protocol.md`,
`docs/commercial_multileo_validation_protocol.md`, and
`docs/release_readiness_checklist.md`. A protocol is not counted as completed
deployment evidence by the reviewer-readiness audit.
