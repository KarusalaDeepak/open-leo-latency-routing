# Reproducibility Guide

This guide lists the exact repo-level commands used to regenerate the primary
results discussed in the manuscript.

## Environment

- Repository root: `leo-conf2-open-leo`
- Main configuration: `configs/experiment.yaml`
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

## Notes

- All operational-shift perturbations are injected only into evaluation data.
- The retrospective best path is used only for post-hoc evaluation metrics such
  as decision gap; it is not an online deployable baseline.
- Figure-generation scripts read the CSV outputs above and do not alter the
  underlying metrics.
