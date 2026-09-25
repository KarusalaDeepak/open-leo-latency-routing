# Reviewer Follow-Up Evidence Map

This map links the six follow-up comments to executable evidence. The
manuscript uses only the compact CSV results listed below. The large row-level
1-second decision and prediction exports are reproducible from the documented
rolling-validation command and are intentionally not versioned.

| Comment | Implemented analysis | Evidence and safeguards |
|---|---|---|
| Decision-interval reversal | Reran the same five-fold rolling protocol at 1 s and 10 s. The audit reports aggregation width, physical next-bin horizon, train rows, scheduled epochs, retained decisions, success, and latency. | `scripts/audit_cadence_qos_and_availability.py`; `results/cadence_qos_availability_audit/cadence_policy_comparison.csv`; `results/commect_rolling_1s_active_opportunity/rolling_fold_manifest.csv`. The interpretation explicitly treats cadence, horizon, and sample count as co-varying, not as a causal decomposition. |
| Practical evidence collection | Converted the frozen practical gate profile into zero-harm success and bounded-tail non-vacuity floors, then inflated them by assumed opportunity prevalence and site throughput. | `scripts/build_operator_collection_plan.py`; `results/operator_collection_plan/operator_collection_plan.csv`. Same-drive windows remain one group, and every output states that the values are planning floors rather than power guarantees. |
| Exact QoS-filtering rule | Added a frozen post-test comparison between the conservative reactive-in-mixed shield and ensemble ranking among currently compliant paths. Reports different actions, policy-only successes, realized RTT, and current-threshold violations at both cadences. | `results/cadence_qos_availability_audit/qos_filter_rule_ablation.csv`. Both rules have zero current-threshold violations in the mixed/multiple-compliant scope; the preferred rule reverses between 1 s and 10 s, so no variant is selected from test outcomes. |
| Missing-outcome prominence | Partitioned all 948 scheduled fixed-test epochs into raw gaps, incomplete exact-target/concurrent outcomes, alignment exclusions, retained decisions, and retained no-action decisions. | `results/cadence_qos_availability_audit/scheduled_unavailability_audit.csv`. Complete-case success is reported beside the worst-case scheduled-epoch lower bound; exclusions are labeled evaluation unobservability rather than measured outage. |
| Protection-setting choice | Encoded the latency target, non-inferiority margin, tail quantile, latency cap, required tail gain, multiplicity, and their operational bases in a machine-readable prospective profile. | `configs/prospective_deployment_protocol.yaml`. Settings are frozen from the application SLO and cost model before gate evidence; they are not test-tuned. |
| Concrete deployment experiment | Specified instrumentation, independent shadow collection, one-shot admission, 5% canary, staged expansion, immediate and statistical rollback, and revalidation triggers. | `configs/prospective_deployment_protocol.yaml`. Required timestamps cover sampling through first packet after installation, and required outcomes include intended/installed paths, queues, outages, loss/jitter, and unselected-path probes. |

## Reproduction

```bash
python scripts/run_commect_rolling_origin_validation.py \
  --trace data/processed/commect_multiaccess_1s.csv \
  --output-dir results/commect_rolling_1s_active_opportunity
python scripts/audit_cadence_qos_and_availability.py
python scripts/build_operator_collection_plan.py
python -m pytest -q
```

The audit is read-only with respect to learned models and policy admission: it
loads frozen rolling outputs, performs no refit, and cannot reselect a candidate.
