# Model Feature and Target Contract

This document is the paper-facing contract for the default standardized
ridge/ridge comparison. It lists the online inputs selected by
`default_feature_columns()` and `graph_context_feature_columns()`. A field is
used only when it exists in the source schema and is numeric. The fitted
`StandardScaler` and ridge coefficients use training rows only.

## Decision and target timing

- The current row represents aggregate bin `B_t = [b_t, b_t + H)`.
- `target_next` is the arithmetic mean of successful RTT replies in the exact
  adjacent bin `B_{t+1}` whose start is `b_t + H`.
- A later observed bin never substitutes for a missing adjacent target.
- Zero replies make a path unavailable. Missing/zero-reply targets are not
  assigned a finite timeout value.
- Target, event, queue, injected-shift, and other future-only fields are never
  model features.

## Temporal view

| Field or family | Unit | Availability and construction |
| --- | --- | --- |
| `latency_mean_ms`, `latency_std_ms`, `latency_max_ms` | ms | Successful replies in the just-closed current bin. |
| `observed_replies` | count | Reply count in the current bin. |
| `path_state_flag` | binary | Current source state mapped to a numeric availability flag. |
| `window_duration_hours` | h | Source metadata, when present. |
| `probe_interval_ms` | ms | Source metadata, when present. |
| `session_day_of_month` | day number | Current timestamp metadata, when present. |
| `latency_mean_ms_lag_{1,2,3,6,12}` | ms | Exact scheduled lookup at `b_t-kH`; absent if that slot is missing. |
| `latency_mean_ms_lag_{1,2,3,6,12}_available` | binary | One iff the exact lag slot exists. |
| `latency_mean_ms_roll_mean_{3,5}` | ms | Mean over the exact scheduled window. |
| `latency_mean_ms_roll_std_{3,5}` | ms | Standard deviation over the exact scheduled window. |
| `latency_mean_ms_roll_observed_count_{3,5}` | count | Number of observed slots in the scheduled window. |
| `latency_mean_ms_roll_coverage_{3,5}` | fraction | Observed slots divided by scheduled slots. |
| `latency_mean_ms_roll_complete_{3,5}` | binary | One iff every scheduled slot exists. |
| `observed_replies_lag_1` | count | Exact one-step reply-count lag. |
| `observed_replies_lag_1_available` | binary | One iff the exact reply lag exists. |
| `observed_replies_roll_mean_3`, `observed_replies_roll_std_3` | count | Exact scheduled three-slot reply moments. |
| `observed_replies_roll_observed_count_3` | count | Observed reply slots in that window. |
| `observed_replies_roll_coverage_3` | fraction | Observed reply slots divided by three. |
| `observed_replies_roll_complete_3` | binary | One iff all three reply slots exist. |
| `history_lag_coverage_ratio` | fraction | Mean of exact latency-lag availability indicators. |
| `latency_delta_1`, `latency_delta_roll3` | ms | Current mean minus the exact lag/rolling mean. |
| `latency_jump_ratio`, `latency_volatility_ratio` | ratio | Current-to-rolling mean and rolling std-to-mean. |
| `reply_delta_1`, `reply_gap_roll3` | count | Current replies minus lag/rolling replies. |
| `reply_pressure_score`, `burst_indicator` | dimensionless | Current-only derived diagnostics defined in `features/temporal.py`. |

Missing numeric feature values are zero-imputed only at the model interface;
the corresponding availability, count, coverage, and completeness fields
remain in the feature vector. Path identity is not a feature.

## Peer-context view

Every peer statistic is computed within the current bin and excludes the row's
own path. A peer moment is missing when no other row has an observed value;
the implementation never substitutes the candidate's own value. Missing
moments are zero-imputed only at the model interface, where the paired count
and availability fields preserve the distinction between “no peer evidence”
and a genuine zero-valued statistic.

| Field | Unit | Meaning |
| --- | --- | --- |
| `peer_latency_mean`, `peer_latency_std` | ms | Current RTT moments over peer paths. |
| `peer_latency_observed_count`, `peer_latency_available` | count, binary | Number of peers contributing an observed RTT and one iff that count is positive. |
| `state_peer_latency_mean`, `state_peer_latency_std` | ms | Peer moments within current path-state class. |
| `state_peer_latency_observed_count`, `state_peer_latency_available` | count, binary | Observed RTT peer count and availability within the current path-state class. |
| `target_peer_latency_mean`, `target_peer_latency_std` | ms | Peer moments within the current remote-endpoint metadata class; `target` here is a source endpoint label, not a future outcome. |
| `target_peer_latency_observed_count`, `target_peer_latency_available` | count, binary | Observed RTT peer count and availability within the current endpoint class. |
| `peer_reply_mean`, `peer_reply_std` | count | Current reply-count moments over peers. |
| `peer_reply_observed_count`, `peer_reply_available` | count, binary | Number of peers contributing an observed reply count and one iff positive. |
| `peer_burst_indicator_mean`, `peer_burst_indicator_std` | dimensionless | Current burst-diagnostic moments over peers. |
| `peer_burst_indicator_observed_count`, `peer_burst_indicator_available` | count, binary | Number of peers contributing an observed burst diagnostic and one iff positive. |
| `peer_latency_gap` | ms | Current path mean minus peer mean. |
| `location_degree`, `target_degree`, `snapshot_candidate_count` | count | Current same-bin topology/cardinality metadata. |

## Service-risk score

For the canonical configuration, the causal online service-risk penalty is

`service_risk_ms = 2.5 max(reply_pressure_score, 0) + 1.5 latency_std_ms / s_cal`,

where `reply_pressure_score = 1 - observed_replies /
max(observed_replies_roll_mean_3, 1e-6)`. Missing current latency standard
deviation is treated as zero. The normalization scale is
`s_cal = max(1 ms, median(latency_std_ms))`, computed once on the clean
calibration block and then frozen for policy selection and test scoring. The
weights 2.5 and 1.5 are the defaults recorded under
`optimization.service_risk` in `configs/experiment.yaml`; neither is fitted on
test outcomes.

## Temporal uncertainty ensemble

The canonical ensemble has nine standardized ridge-regression members
(`alpha=1.0`). For each member, a NumPy generator initialized once with seed
314 draws, with
replacement, `max(2, floor(0.82 n_train))` training rows and, without
replacement, `max(1, floor(0.78 p))` temporal features. A new draw is made for
each member from that same deterministic generator stream. The member count
and row/feature fractions are the defaults in `configs/experiment.yaml`; seed
314 is fixed by `_fit_temporal_uncertainty_ensemble()` in
`scripts/run_service_path_experiments.py`. Ensemble spread is the population
standard deviation (`ddof=0`) of the nine member predictions.

## Calibration score

The empirical residual-risk model is pooled across paths on the calibration
block. Its target is the absolute residual of the covariance-aware fused
prediction. Inputs are normalized expert disagreement, temporal-ensemble
spread, and the current service-risk score. A non-negative linear model fits
the three slopes; the intercept is unconstrained and the predicted score is
clipped at zero. The same calibration block supplies the 0.90 quantile of the
fitted scores, so this score has no in-sample or marginal coverage claim. A
separate policy-selection block evaluates fallback policies and gate evidence.
