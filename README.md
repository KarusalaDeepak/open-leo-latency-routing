# Open LEO Latency Routing

This repository contains the reproducible code for predictive latency-aware path
selection on open LEO network measurements.

## Scope

The project scope is:

1. time-series forecasting of short-horizon LEO network state,
2. graph-based learning on temporal network snapshots,
3. decision optimization on predicted network state.

The paper-facing comparison keeps only the methods used in the manuscript:

- temporal baselines: `persistence`, `linear_regression`
- graph-aware predictor: `graph_xgboost`
- decision policies: `random`, `reactive_greedy`, `predictive_greedy`, `predictive_graph_greedy`, `predictive_consensus_greedy`

The consensus policy is a lightweight hybrid score:

- `pred_consensus = 0.65 * pred_forecast + 0.35 * pred_graph + 0.30 * |pred_graph - pred_forecast|`

This keeps the temporal model as the anchor, borrows graph context when it helps,
and penalizes strong disagreement between the two predictors as a simple
uncertainty signal.

Key decision metrics reported by the repo include:

- mean realized latency,
- mean regret,
- `best_path_match_rate`, which measures how often a policy chooses the same path as the hindsight-best path,
- success rate under the configured latency budget,
- mean per-decision runtime in microseconds.

## Dataset Plan

Primary dataset:

- LENS 2025-03 open measurement release

Expected local dataset path after download and extraction:

- `data/raw/lens_2025_03/`

## Repository Layout

- `configs/`: experiment configuration files
- `docs/`: project notes, scope, and data cards
- `scripts/`: runnable entrypoints for inspection and experiments
- `src/open_leo_latency_routing/`: source package
- `data/raw/`: raw downloaded and extracted data
- `data/processed/`: cleaned intermediate artifacts
- `results/`: outputs, figures, metrics, and logs

Generated raw data extracts, processed data tables, and result files are ignored
by Git by default. This keeps the public repository lightweight while preserving
the exact commands needed to regenerate the reported artifacts from the LENS
release.

## Actual Ingestion Outputs

The LENS release is primarily a large collection of line-oriented `ping` logs rather than ready-made CSV tables. The repo converts those logs into three compact artifacts:

1. `ping_session_summary.csv`: one row per measurement session,
2. `ping_observations_sample.csv`: sparse observation-level samples for inspection,
3. `ping_time_bins.csv`: fixed-width time-bin aggregates for forecasting and graph snapshots.

## First Milestones

1. inspect extracted dataset files and create a data card,
2. build a diverse candidate manifest instead of overfitting to one location,
3. build reactive and predictive optimization baselines,
4. compare temporal-only and graph-aware decision policies.

## Modeling Pipeline

After the raw logs are parsed, the repo supports three direct experiment stages:

1. forecasting on minute-level latency bins,
2. graph-aware learning on normalized session snapshots,
3. decision optimization using reactive and predictive policies,
4. robustness evaluation under burst and outage shifts.

The current study version also includes three novelty-strengthening
analyses:

1. disagreement-bin validation to test whether predictor disagreement behaves
   like an uncertainty signal,
2. a correlated structural-shift stress that degrades multiple location groups
   across aligned decision windows,
3. a consensus-penalty sweep over the disagreement regularization coefficient.

## Quick Start

```bash
cd open-leo-latency-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/inspect_dataset.py --data-root data/raw/lens_2025_03
python scripts/build_data_card.py --data-root data/raw/lens_2025_03
python scripts/build_manifest.py --data-root data/raw/lens_2025_03/LENS-2025-03 --top-k 24 --max-per-location 4 --max-per-day 2
python scripts/build_ping_tables.py --data-root data/raw/lens_2025_03/LENS-2025-03 --max-files 16 --time-bin-seconds 60
python scripts/run_temporal_forecasting.py
python scripts/run_graph_forecasting.py
python scripts/run_decision_policy_evaluation.py
python scripts/run_robustness_evaluation.py
python scripts/generate_result_figures.py
```
