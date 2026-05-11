# Baseline Results Summary

This note records the final end-to-end conference baseline built from the extracted LENS 2025-03 dataset.

## Processed Dataset Slice

- candidate manifest size: `24` sessions
- parsed working subset: `16` sessions
- sampled observations: `1920`
- minute-level time bins: `1087`
- session states represented: `13 active`, `3 inactive`

The processed tables used by the pipeline are:

- `data/processed/ping_session_summary.csv`
- `data/processed/ping_observations_sample.csv`
- `data/processed/ping_time_bins.csv`

## Forecasting Setup

- target: next-bin `latency_mean_ms`
- snapshot interval: `60 s`
- requested prediction horizon: `10 s`
- effective supervised horizon: `1` time bin
- lag features: `1, 2, 3, 6, 12`
- split rule: chronological split inside each session to avoid future leakage

## Forecasting Results

| Model | MAE | RMSE | MAPE | Test Rows |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 6.4712 | 9.4286 | 0.1544 | 162 |
| Linear Regression | 4.7105 | 7.4679 | 0.1106 | 162 |

Current best pure temporal forecaster: `linear_regression`

## Graph-Aware Result

Graph snapshots are normalized by `session_bin_index`, allowing different sessions to contribute to the same decision stage even when recorded on different calendar dates. The graph features currently include:

- peer latency mean and standard deviation,
- same-state peer latency mean,
- same-target peer latency mean,
- location degree,
- target degree,
- snapshot candidate count.

| Model | MAE | RMSE | MAPE | Test Rows |
| --- | ---: | ---: | ---: | ---: |
| Graph XGBoost Regressor | 4.9449 | 7.5467 | 0.1199 | 162 |

The final conference version keeps `graph_xgboost` as the single graph-aware model because it is the strongest graph-based predictor in this repo and is the method used by the downstream graph-aware decision policy.

## Optimization Results

Decision windows are defined by normalized `session_bin_index`. For each window, the policy selects one candidate path and is evaluated on the realized next-bin latency.

| Policy | Decisions | Mean Realized Latency (ms) | Mean Regret (ms) | Best-Path Match Rate | Success Under 60 ms | Mean Decision Time (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 28 | 43.2027 | 4.3333 | 0.4643 | 0.9643 | 194.15 |
| Reactive Greedy | 28 | 43.4266 | 4.5572 | 0.3214 | 0.8929 | 203.53 |
| Predictive Greedy | 28 | 41.5787 | 2.7093 | 0.5000 | 0.9286 | 222.57 |
| Predictive Graph Greedy | 28 | 41.9660 | 3.0967 | 0.4286 | 0.9286 | 206.13 |
| Predictive Simple Fusion Greedy | 28 | 41.7712 | 2.9018 | 0.5000 | 0.9286 | 191.92 |
| Predictive Consensus Greedy | 28 | 41.9269 | 3.0575 | 0.4643 | 0.9286 | 186.10 |

On the base split, `predictive_greedy` is the strongest decision rule by mean realized latency and regret. `predictive_simple_fusion_greedy` remains intermediate, while `predictive_consensus_greedy` is slightly behind in the stable condition. This is a useful negative result: the disagreement penalty is not meant to dominate in every nominal case.

Best-path match rate should be interpreted only as a secondary metric. It uses perfect hindsight and is not deployable; the value of this column is simply to show how often a practical policy happens to match the true best path that can only be identified after observing the future.

Random achieves the highest `Success Under 60 ms` in the base setting, while `predictive_greedy` wins on mean latency and regret. This highlights the classic latency-vs.-tail tradeoff in LEO scheduling: a policy can improve average decision quality without always dominating the tail threshold metric.

## Statistical Significance

Wilcoxon signed-rank tests were run across the `28` paired decision windows.

| Comparison | Metric | Mean Delta (ms) | p-value | Interpretation |
| --- | --- | ---: | ---: | --- |
| Graph vs Reactive | Realized Latency | -1.4605 | 0.1688 | Improvement trend, not significant on the base split |
| Graph vs Reactive | Regret | -1.4605 | 0.1688 | Improvement trend, not significant on the base split |
| Graph vs Predictive-only | Realized Latency | 0.3873 | 0.4236 | No significant difference |
| Graph vs Predictive-only | Regret | 0.3873 | 0.4236 | No significant difference |
| Fusion vs Temporal | Realized Latency | 0.1925 | 0.2850 | Naive blending does not significantly beat temporal-only prediction |
| Fusion vs Temporal | Regret | 0.1925 | 0.2850 | Same conclusion for regret |
| Consensus vs Fusion | Realized Latency | 0.1557 | 0.5930 | No significant difference on the stable base split |
| Consensus vs Fusion | Regret | 0.1557 | 0.5930 | Same conclusion for regret |

## Current Interpretation

The final cleaned baseline supports the conference direction:

1. Predictive selection outperforms purely reactive selection on realized latency and regret.
2. `linear_regression` is the strongest pure temporal forecaster on the current working subset, while `graph_xgboost` is the graph-aware predictor used for the proposed graph-based policy.
3. The stable base split does not justify claiming universal dominance for either graph-only or hybrid decision rules, because `predictive_greedy` remains best in this condition.
4. The stronger conference claim is therefore robustness: graph-aware decision-making becomes more valuable in shifted conditions such as outage windows, and the disagreement-aware consensus rule should be judged against the simple-fusion ablation rather than against temporal-only forecasting error.
