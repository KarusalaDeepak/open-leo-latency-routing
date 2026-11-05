# Campaign-aware evidence gating for cellular–LEO path selection

This repository contains the current scientific implementation, configuration
files, data adapters and tests for Paper 2. It provides RTT forecasting,
counterfactual multi-access replay and a conservative campaign-level evidence
gate for comparing a frozen predictive candidate with a reactive baseline.

The repository contains source code only. Measurement archives, generated
results, manuscript files, LaTeX sources, presentations and earlier repository
versions are excluded. Experiments create ignored local output directories.

## What the gate decides

The reactive policy selects the available path with the lowest current RTT.
The evidence gate checks whether a frozen predictive alternative has enough
independent acquisition groups, adequate active-policy support, acceptable
success uncertainty and sufficient improvement in capped tail RTT.

| State | Meaning |
| --- | --- |
| `INSUFFICIENT_SUPPORT` | Required acquisition-group or opportunity counts are missing. |
| `INSUFFICIENT_PRECISION` | Observed performance meets its requirement, but the simultaneous confidence bound does not. |
| `ADVERSE_PERFORMANCE` | Observed performance misses a declared requirement. |
| `AUTHORIZED` | The declared offline support and statistical requirements pass. |

An offline pass can support eligibility for a separately specified monitored
trial. The replay code does not establish installed routing actions or live
application benefit. A failed gate retains the predefined reactive baseline.

The measured endpoint is next-interval mean RTT on exact complete-case
decisions. It is distinct from packet-delivery success, scheduled availability
and the latency of an implemented controller. Telemetry gaps and excluded
outcomes are audited separately.

## Install and test

Use Python 3.10 or later. The reference direct dependency pins were tested on
CPython 3.11.5; they are not a hash-locked transitive environment for every
platform.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-lock.txt
PYTHONPATH=src python3 -m pytest -q
```

The test suite constructs small fixtures in temporary directories and does not
require the external measurement archives. `requirements-hypatia.txt` is an
optional compatibility path; it is separate from the reference environment.

## Run the analyses

Every entrypoint supports `--help`. Acquire the source data separately and
configure its local path before running measured evaluations.

```sh
python3 scripts/audit_gate_design_sensitivity.py --help
python3 scripts/run_conext23_external_gate.py --help
python3 scripts/run_commect_multiaccess_validation.py --help
python3 scripts/run_commect_rolling_origin_validation.py --help
python3 scripts/run_wetlinks_longitudinal_validation.py --help
python3 scripts/simulate_structured_campaign_admission.py --help
```

`configs/evidence_profiles.yaml` declares the source-specific latency budget,
success margin, tail quantile, cap and practical tail-improvement requirement.
`configs/campaign_qualification.yaml` records acquisition eligibility.
`configs/parameter_registry.yaml` identifies fitted, selected, statistical,
application-defined and reference quantities. `configs/experiment.yaml`
contains executable model settings. Freeze development choices before using
gate evidence; do not tune on final evaluation outcomes.

The archived COMMECT profile uses a 60-ms target, a 60,000-ms statistical cap,
the slowest 5% tail and a 1-ms required gain. The prospective profile and the
CoNEXT audit separately use a 250-ms cap, the slowest 10% tail and a 5-ms gain.
The cap does not impute missing observations. CoNEXT fixes the ridge penalty at
one and fits its temporal/context fusion weights using only the April 23
calibration partition. These source-specific settings must not be substituted
or pooled. Regression tests compare the declared profiles with execution.

The canonical evidence rebuild and integrity utilities remain in `scripts/`.
They require separately acquired inputs and generate local outputs; they do
not contain or build the manuscript. Plotting and table utilities produce
scientific experiment assets from those outputs.

## Data and interpretation

COMMECT contributes one acquisition campaign. The CoNEXT’23 audit uses April 23
for development and April 24–28 as five conditional daily evidence groups from
one acquisition family. Common endpoint identity and day-level independence
and exchangeability are assumptions, not independently verified fields. If the
five days are treated as one family-level group, the evidence floor fails.
LENS and WetLinks support prediction diagnostics; they do not supply paired
selectable terrestrial alternatives for the gate.

The measured audits retain the reactive baseline. Controlled synthetic
admission checks exercise the gate when its requirements are satisfied; they
do not constitute field authorization. More intervals from one collection do
not create independent campaigns, and five campaigns alone do not imply
statistical sufficiency.

See [data and provenance](docs/DATA_AND_PROVENANCE.md), the
[feature and target contract](docs/model_feature_contract.md), and
[third-party notices](docs/THIRD_PARTY_NOTICES.md).

## Layout

- `src/open_leo_latency_routing/`: forecasting, features, policies and inference.
- `scripts/`: adapters, experiments, diagnostics and evidence integrity tools.
- `tests/`: regression tests with generated temporary fixtures.
- `configs/`: experiment, parameter, evidence and deployment profiles.
- `docs/`: data provenance and implementation contracts.

The repository does not assert a project source-code license. Third-party
software and datasets retain their original terms.
