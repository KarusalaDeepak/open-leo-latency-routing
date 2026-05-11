# Robustness Evaluation Summary

This report evaluates train-on-base, test-on-shift robustness under burst, outage, and correlated structural-shift conditions.

## Forecast Metrics

| Scenario | Model | MAE | RMSE | MAPE | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| base | persistence | 6.4712 | 9.4286 | 0.1544 | 162 |
| base | linear_regression | 4.7105 | 7.4679 | 0.1106 | 162 |
| burst | persistence | 9.1263 | 13.3162 | 0.2074 | 162 |
| burst | linear_regression | 6.8304 | 10.4285 | 0.1484 | 162 |
| outage | persistence | 6.6990 | 9.8372 | 0.1599 | 162 |
| outage | linear_regression | 5.5470 | 8.6211 | 0.1290 | 162 |
| structural | persistence | 6.5696 | 9.5582 | 0.1522 | 162 |
| structural | linear_regression | 6.2389 | 10.3762 | 0.1272 | 162 |

## Graph Metrics

| Scenario | Model | MAE | RMSE | MAPE | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| base | graph_xgboost | 5.0842 | 7.7873 | 0.1237 | 162 |
| burst | graph_xgboost | 7.4267 | 10.3649 | 0.1722 | 162 |
| outage | graph_xgboost | 5.8583 | 8.6063 | 0.1435 | 162 |
| structural | graph_xgboost | 5.6577 | 8.5634 | 0.1269 | 162 |

## Consensus Hybrid Calibration

| Temporal Weight | Graph Weight | Disagreement Penalty | Mean Validation Gap (ms) |
| ---: | ---: | ---: | ---: |
| 0.65 | 0.35 | 0.30 | -0.6255 |

## Policy Summary

| Scenario | Policy | Decisions | Mean Realized Latency (ms) | Mean Regret (ms) | Best-Path Match Rate | Success Under 60 ms | Mean Decision Time (us) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | random | 28 | 43.2027 | 4.3333 | 0.4643 | 0.9643 | 269.04 |
| base | reactive_greedy | 28 | 43.4266 | 4.5572 | 0.3214 | 0.8929 | 315.11 |
| base | predictive_greedy | 28 | 41.5787 | 2.7093 | 0.5000 | 0.9286 | 287.57 |
| base | predictive_graph_greedy | 28 | 40.6557 | 1.7863 | 0.5714 | 0.9643 | 307.99 |
| base | predictive_simple_fusion_greedy | 28 | 41.1929 | 2.3235 | 0.5000 | 0.9643 | 266.63 |
| base | predictive_consensus_greedy | 28 | 40.7455 | 1.8761 | 0.6429 | 0.9643 | 294.96 |
| burst | random | 28 | 44.5169 | 4.9632 | 0.4643 | 0.9643 | 327.73 |
| burst | reactive_greedy | 28 | 45.8579 | 6.3042 | 0.4286 | 0.8571 | 298.52 |
| burst | predictive_greedy | 28 | 44.6901 | 5.1364 | 0.5000 | 0.8929 | 262.16 |
| burst | predictive_graph_greedy | 28 | 45.9807 | 6.4270 | 0.4643 | 0.8571 | 251.72 |
| burst | predictive_simple_fusion_greedy | 28 | 45.9389 | 6.3853 | 0.4286 | 0.8571 | 281.98 |
| burst | predictive_consensus_greedy | 28 | 45.8330 | 6.2793 | 0.4286 | 0.8571 | 298.67 |
| outage | random | 28 | 45.9272 | 7.0578 | 0.4643 | 0.8929 | 289.07 |
| outage | reactive_greedy | 28 | 43.4266 | 4.5572 | 0.3214 | 0.8929 | 296.94 |
| outage | predictive_greedy | 28 | 41.5787 | 2.7093 | 0.5000 | 0.9286 | 276.70 |
| outage | predictive_graph_greedy | 28 | 40.9901 | 2.1208 | 0.5357 | 0.9643 | 289.22 |
| outage | predictive_simple_fusion_greedy | 28 | 41.1929 | 2.3235 | 0.5000 | 0.9643 | 289.00 |
| outage | predictive_consensus_greedy | 28 | 40.9088 | 2.0394 | 0.6071 | 0.9643 | 288.00 |
| structural | random | 28 | 48.8898 | 4.2025 | 0.4286 | 0.7500 | 304.43 |
| structural | reactive_greedy | 28 | 49.1523 | 4.4650 | 0.2857 | 0.7143 | 262.37 |
| structural | predictive_greedy | 28 | 47.4615 | 2.7743 | 0.4643 | 0.7500 | 284.42 |
| structural | predictive_graph_greedy | 28 | 47.3827 | 2.6954 | 0.4286 | 0.7500 | 292.96 |
| structural | predictive_simple_fusion_greedy | 28 | 47.6447 | 2.9574 | 0.4643 | 0.7500 | 254.60 |
| structural | predictive_consensus_greedy | 28 | 47.4115 | 2.7243 | 0.5357 | 0.7500 | 288.79 |

Best-path match rate is reported only as a secondary metric because it uses perfect hindsight and cannot be deployed online.

## Policy Significance

| Scenario | Comparison | Metric | Mean Delta (ms) | p-value |
| --- | --- | --- | ---: | ---: |
| base | graph_vs_reactive | realized_next_latency_ms | -2.7709 | 0.0044 |
| base | graph_vs_reactive | regret_ms | -2.7709 | 0.0044 |
| base | graph_vs_predictive_only | realized_next_latency_ms | -0.9230 | 0.4328 |
| base | graph_vs_predictive_only | regret_ms | -0.9230 | 0.4328 |
| base | fusion_vs_temporal | realized_next_latency_ms | -0.3858 | 1.0000 |
| base | fusion_vs_temporal | regret_ms | -0.3858 | 1.0000 |
| base | fusion_vs_graph | realized_next_latency_ms | 0.5372 | 0.1394 |
| base | fusion_vs_graph | regret_ms | 0.5372 | 0.1394 |
| base | consensus_vs_fusion | realized_next_latency_ms | -0.4474 | 0.0679 |
| base | consensus_vs_fusion | regret_ms | -0.4474 | 0.0679 |
| base | consensus_vs_temporal | realized_next_latency_ms | -0.8332 | 0.3980 |
| base | consensus_vs_temporal | regret_ms | -0.8332 | 0.3980 |
| base | consensus_vs_graph | realized_next_latency_ms | 0.0898 | 0.9165 |
| base | consensus_vs_graph | regret_ms | 0.0898 | 0.9165 |
| burst | graph_vs_reactive | realized_next_latency_ms | 0.1228 | 0.7532 |
| burst | graph_vs_reactive | regret_ms | 0.1228 | 0.7532 |
| burst | graph_vs_predictive_only | realized_next_latency_ms | 1.2906 | 0.6949 |
| burst | graph_vs_predictive_only | regret_ms | 1.2906 | 0.6949 |
| burst | fusion_vs_temporal | realized_next_latency_ms | 1.2488 | 0.2489 |
| burst | fusion_vs_temporal | regret_ms | 1.2488 | 0.2489 |
| burst | fusion_vs_graph | realized_next_latency_ms | -0.0418 | 0.4990 |
| burst | fusion_vs_graph | regret_ms | -0.0418 | 0.4990 |
| burst | consensus_vs_fusion | realized_next_latency_ms | -0.1060 | 0.6547 |
| burst | consensus_vs_fusion | regret_ms | -0.1060 | 0.6547 |
| burst | consensus_vs_temporal | realized_next_latency_ms | 1.1429 | 0.4008 |
| burst | consensus_vs_temporal | regret_ms | 1.1429 | 0.4008 |
| burst | consensus_vs_graph | realized_next_latency_ms | -0.1477 | 0.6858 |
| burst | consensus_vs_graph | regret_ms | -0.1477 | 0.6858 |
| outage | graph_vs_reactive | realized_next_latency_ms | -2.4365 | 0.0208 |
| outage | graph_vs_reactive | regret_ms | -2.4365 | 0.0208 |
| outage | graph_vs_predictive_only | realized_next_latency_ms | -0.5886 | 0.8589 |
| outage | graph_vs_predictive_only | regret_ms | -0.5886 | 0.8589 |
| outage | fusion_vs_temporal | realized_next_latency_ms | -0.3858 | 1.0000 |
| outage | fusion_vs_temporal | regret_ms | -0.3858 | 1.0000 |
| outage | fusion_vs_graph | realized_next_latency_ms | 0.2027 | 0.5147 |
| outage | fusion_vs_graph | regret_ms | 0.2027 | 0.5147 |
| outage | consensus_vs_fusion | realized_next_latency_ms | -0.2841 | 0.1088 |
| outage | consensus_vs_fusion | regret_ms | -0.2841 | 0.1088 |
| outage | consensus_vs_temporal | realized_next_latency_ms | -0.6699 | 0.4631 |
| outage | consensus_vs_temporal | regret_ms | -0.6699 | 0.4631 |
| outage | consensus_vs_graph | realized_next_latency_ms | -0.0813 | 0.7532 |
| outage | consensus_vs_graph | regret_ms | -0.0813 | 0.7532 |
| structural | graph_vs_reactive | realized_next_latency_ms | -1.7697 | 0.0284 |
| structural | graph_vs_reactive | regret_ms | -1.7697 | 0.0284 |
| structural | graph_vs_predictive_only | realized_next_latency_ms | -0.0789 | 0.9292 |
| structural | graph_vs_predictive_only | regret_ms | -0.0789 | 0.9292 |
| structural | fusion_vs_temporal | realized_next_latency_ms | 0.1831 | 0.3173 |
| structural | fusion_vs_temporal | regret_ms | 0.1831 | 0.3173 |
| structural | fusion_vs_graph | realized_next_latency_ms | 0.2620 | 0.5337 |
| structural | fusion_vs_graph | regret_ms | 0.2620 | 0.5337 |
| structural | consensus_vs_fusion | realized_next_latency_ms | -0.2332 | 0.1441 |
| structural | consensus_vs_fusion | regret_ms | -0.2332 | 0.1441 |
| structural | consensus_vs_temporal | realized_next_latency_ms | -0.0500 | 0.6858 |
| structural | consensus_vs_temporal | regret_ms | -0.0500 | 0.6858 |
| structural | consensus_vs_graph | realized_next_latency_ms | 0.0289 | 0.8658 |
| structural | consensus_vs_graph | regret_ms | 0.0289 | 0.8658 |

In base evaluation, random selection retains the strongest `Success Under 60 ms` tail score while the graph-aware policy improves mean latency and regret. This is a classic latency-vs.-tail tradeoff rather than a contradiction.

## Disagreement as an Uncertainty Signal

| Scenario | Disagreement Bin | Policy | Mean Latency (ms) | Mean Regret (ms) | Match Rate | Success Under 60 ms | Decisions |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | Low | predictive_consensus_greedy | 43.6684 | 2.2452 | 0.7000 | 1.0000 | 10 |
| base | Low | predictive_graph_greedy | 43.3109 | 1.8877 | 0.7000 | 1.0000 | 10 |
| base | Low | predictive_greedy | 43.1294 | 1.7062 | 0.7000 | 1.0000 | 10 |
| base | Low | predictive_simple_fusion_greedy | 44.1258 | 2.7026 | 0.6000 | 1.0000 | 10 |
| base | Low | random | 43.6067 | 2.1835 | 0.6000 | 1.0000 | 10 |
| base | Low | reactive_greedy | 46.9313 | 5.5082 | 0.5000 | 0.9000 | 10 |
| base | Medium | predictive_consensus_greedy | 36.0927 | 0.4608 | 0.6667 | 1.0000 | 9 |
| base | Medium | predictive_graph_greedy | 36.3180 | 0.6860 | 0.4444 | 1.0000 | 9 |
| base | Medium | predictive_greedy | 39.1090 | 3.4770 | 0.3333 | 0.8889 | 9 |
| base | Medium | predictive_simple_fusion_greedy | 36.8015 | 1.1696 | 0.4444 | 1.0000 | 9 |
| base | Medium | random | 41.1807 | 5.5487 | 0.4444 | 1.0000 | 9 |
| base | Medium | reactive_greedy | 40.3770 | 4.7451 | 0.1111 | 0.8889 | 9 |
| base | High | predictive_consensus_greedy | 42.1505 | 2.8813 | 0.5556 | 0.8889 | 9 |
| base | High | predictive_graph_greedy | 42.0432 | 2.7739 | 0.5556 | 0.8889 | 9 |
| base | High | predictive_greedy | 42.3255 | 3.0562 | 0.4444 | 0.8889 | 9 |
| base | High | predictive_simple_fusion_greedy | 42.3255 | 3.0562 | 0.4444 | 0.8889 | 9 |
| base | High | random | 44.7758 | 5.5065 | 0.3333 | 0.8889 | 9 |
| base | High | reactive_greedy | 42.5820 | 3.3127 | 0.3333 | 0.8889 | 9 |
| burst | Low | predictive_consensus_greedy | 49.2279 | 4.2308 | 0.5000 | 0.9000 | 10 |
| burst | Low | predictive_graph_greedy | 49.2279 | 4.2308 | 0.5000 | 0.9000 | 10 |
| burst | Low | predictive_greedy | 51.6760 | 6.6789 | 0.3000 | 0.8000 | 10 |
| burst | Low | predictive_simple_fusion_greedy | 49.5246 | 4.5275 | 0.5000 | 0.9000 | 10 |
| burst | Low | random | 47.0316 | 2.0345 | 0.7000 | 1.0000 | 10 |
| burst | Low | reactive_greedy | 54.6180 | 9.6209 | 0.4000 | 0.8000 | 10 |
| burst | Medium | predictive_consensus_greedy | 40.3657 | 7.9112 | 0.3333 | 0.8889 | 9 |
| burst | Medium | predictive_graph_greedy | 41.6502 | 9.1957 | 0.5556 | 0.8889 | 9 |
| burst | Medium | predictive_greedy | 40.1280 | 7.6734 | 0.4444 | 0.8889 | 9 |
| burst | Medium | predictive_simple_fusion_greedy | 40.3657 | 7.9112 | 0.3333 | 0.8889 | 9 |
| burst | Medium | random | 39.4597 | 7.0052 | 0.1111 | 1.0000 | 9 |
| burst | Medium | reactive_greedy | 38.4797 | 6.0251 | 0.3333 | 0.8889 | 9 |
| burst | High | predictive_consensus_greedy | 47.5281 | 6.9235 | 0.4444 | 0.7778 | 9 |
| burst | High | predictive_graph_greedy | 46.7031 | 6.0986 | 0.3333 | 0.7778 | 9 |
| burst | High | predictive_greedy | 41.4901 | 0.8856 | 0.7778 | 1.0000 | 9 |
| burst | High | predictive_simple_fusion_greedy | 47.5281 | 6.9235 | 0.4444 | 0.7778 | 9 |
| burst | High | random | 46.7800 | 6.1755 | 0.5556 | 0.8889 | 9 |
| burst | High | reactive_greedy | 43.5025 | 2.8980 | 0.5556 | 0.8889 | 9 |
| outage | Low | predictive_consensus_greedy | 45.5834 | 2.0647 | 0.7000 | 1.0000 | 10 |
| outage | Low | predictive_graph_greedy | 45.5834 | 2.0647 | 0.7000 | 1.0000 | 10 |
| outage | Low | predictive_greedy | 45.6561 | 2.1374 | 0.7000 | 1.0000 | 10 |
| outage | Low | predictive_simple_fusion_greedy | 46.1397 | 2.6210 | 0.6000 | 1.0000 | 10 |
| outage | Low | random | 45.4385 | 1.9198 | 0.7000 | 1.0000 | 10 |
| outage | Low | reactive_greedy | 48.6315 | 5.1128 | 0.5000 | 0.9000 | 10 |
| outage | Medium | predictive_consensus_greedy | 39.1639 | 1.3594 | 0.5556 | 1.0000 | 9 |
| outage | Medium | predictive_graph_greedy | 38.4217 | 0.6171 | 0.6667 | 1.0000 | 9 |
| outage | Medium | predictive_greedy | 41.1673 | 3.3627 | 0.2222 | 0.8889 | 9 |
| outage | Medium | predictive_simple_fusion_greedy | 39.4296 | 1.6250 | 0.3333 | 1.0000 | 9 |
| outage | Medium | random | 42.7483 | 4.9438 | 0.4444 | 1.0000 | 9 |
| outage | Medium | reactive_greedy | 42.1192 | 4.3146 | 0.2222 | 0.8889 | 9 |
| outage | High | predictive_consensus_greedy | 37.4597 | 2.6914 | 0.5556 | 0.8889 | 9 |
| outage | High | predictive_graph_greedy | 38.4550 | 3.6867 | 0.2222 | 0.8889 | 9 |
| outage | High | predictive_greedy | 37.4597 | 2.6914 | 0.5556 | 0.8889 | 9 |
| outage | High | predictive_simple_fusion_greedy | 37.4597 | 2.6914 | 0.5556 | 0.8889 | 9 |
| outage | High | random | 49.6490 | 14.8807 | 0.2222 | 0.6667 | 9 |
| outage | High | reactive_greedy | 38.9508 | 4.1825 | 0.2222 | 0.8889 | 9 |
| structural | Low | predictive_consensus_greedy | 39.5015 | 1.7615 | 0.8000 | 1.0000 | 10 |
| structural | Low | predictive_graph_greedy | 39.2682 | 1.5282 | 0.7000 | 1.0000 | 10 |
| structural | Low | predictive_greedy | 39.5278 | 1.7878 | 0.6000 | 1.0000 | 10 |
| structural | Low | predictive_simple_fusion_greedy | 40.0406 | 2.3006 | 0.6000 | 1.0000 | 10 |
| structural | Low | random | 41.7160 | 3.9760 | 0.6000 | 1.0000 | 10 |
| structural | Low | reactive_greedy | 42.9702 | 5.2303 | 0.4000 | 0.9000 | 10 |
| structural | Medium | predictive_consensus_greedy | 37.2245 | 1.5845 | 0.3333 | 1.0000 | 9 |
| structural | Medium | predictive_graph_greedy | 36.9667 | 1.3267 | 0.3333 | 1.0000 | 9 |
| structural | Medium | predictive_greedy | 37.3994 | 1.7594 | 0.2222 | 1.0000 | 9 |
| structural | Medium | predictive_simple_fusion_greedy | 37.3994 | 1.7594 | 0.2222 | 1.0000 | 9 |
| structural | Medium | random | 40.2235 | 4.5835 | 0.2222 | 1.0000 | 9 |
| structural | Medium | reactive_greedy | 38.5631 | 2.9231 | 0.1111 | 1.0000 | 9 |
| structural | High | predictive_consensus_greedy | 66.3875 | 4.9337 | 0.4444 | 0.2222 | 9 |
| structural | High | predictive_graph_greedy | 66.8147 | 5.3609 | 0.2222 | 0.2222 | 9 |
| structural | High | predictive_greedy | 66.3390 | 4.8852 | 0.5556 | 0.2222 | 9 |
| structural | High | predictive_simple_fusion_greedy | 66.3390 | 4.8852 | 0.5556 | 0.2222 | 9 |
| structural | High | random | 65.5269 | 4.0731 | 0.4444 | 0.2222 | 9 |
| structural | High | reactive_greedy | 66.6105 | 5.1567 | 0.3333 | 0.2222 | 9 |

## Consensus Penalty Sweep

| Scenario | Disagreement Penalty | Mean Latency (ms) | Mean Regret (ms) | Match Rate | Runtime (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.00 | 41.1929 | 2.3235 | 0.5000 | 291.45 |
| base | 0.10 | 41.1637 | 2.2943 | 0.5357 | 286.22 |
| base | 0.20 | 40.7455 | 1.8761 | 0.6429 | 277.76 |
| base | 0.30 | 40.7455 | 1.8761 | 0.6429 | 282.23 |
| base | 0.50 | 40.6789 | 1.8095 | 0.6429 | 287.31 |
| burst | 0.00 | 45.9389 | 6.3853 | 0.4286 | 302.74 |
| burst | 0.10 | 46.0316 | 6.4780 | 0.3929 | 330.16 |
| burst | 0.20 | 45.8330 | 6.2793 | 0.4286 | 295.19 |
| burst | 0.30 | 45.8330 | 6.2793 | 0.4286 | 285.28 |
| burst | 0.50 | 45.6602 | 6.1066 | 0.4643 | 260.39 |
| outage | 0.00 | 41.1929 | 2.3235 | 0.5000 | 250.79 |
| outage | 0.10 | 41.1637 | 2.2943 | 0.5357 | 314.54 |
| outage | 0.20 | 40.9088 | 2.0394 | 0.6071 | 287.21 |
| outage | 0.30 | 40.9088 | 2.0394 | 0.6071 | 270.59 |
| outage | 0.50 | 40.7979 | 1.9285 | 0.6429 | 295.92 |
| structural | 0.00 | 47.6447 | 2.9574 | 0.4643 | 266.05 |
| structural | 0.10 | 47.6155 | 2.9282 | 0.5000 | 255.17 |
| structural | 0.20 | 47.4115 | 2.7243 | 0.5357 | 251.41 |
| structural | 0.30 | 47.4115 | 2.7243 | 0.5357 | 287.73 |
| structural | 0.50 | 47.4058 | 2.7185 | 0.5000 | 269.52 |
