> **SUPERSEDED (2026-08-21).** This historical review note is not a source of current claims or counts. Use [`README.md`](../README.md), the `main.tex` in the `leo-conf2-transactions-revision-2026-08-21` submission bundle, and [`results/transactions_evidence/`](../results/transactions_evidence/) as the canonical sources.

# Transactions Risk-Control Revision

This note records the executable response to the latest Transactions-level
review. The revision does not manufacture predictive superiority. It converts
the deployment rule into an opportunity-aware risk-control procedure and
reports both successful abstention and failed post-admission transfer.

| Review risk | Implemented correction | Canonical evidence | Status |
| --- | --- | --- | --- |
| Gate was ordinary model selection | Learned ranking now requires independent opportunity-bearing groups, separate exact harmful-group success bounds for the aggregate actionable population and the post-hoc opportunity-conditioned population, and a simultaneous bounded-CVaR interval with a pre-declared practical gain. | `src/open_leo_latency_routing/optimization/risk_control.py`; `results/transactions_evidence/commect_fixed_gate_selection_evidence.csv` | Complete |
| Calibration data reused for admission | Every principal experiment uses chronological train/calibration/policy-selection/test blocks. Calibration fits residual quantities; the disjoint selection interval alone admits a policy; test is read once. | `src/open_leo_latency_routing/features/temporal.py`; split manifests and leakage tests | Complete |
| Opportunity audit was post-hoc | The post-hoc mixed-outcome label is explicit. Admission requires both enough opportunity-bearing groups and a separately multiplicity-protected success LCB conditional on those groups; exact ties and insufficient evidence default to reactive selection. | `risk_control.py`; Table V in the manuscript | Complete |
| No measured learned-policy admission | The full COMMECT pipeline is reselected at 40, 60, 100, and 200 ms. All four invocations abstain because one continuous drive supplies only one inference group; threshold-specific shield differences are descriptive and unadmitted. | `results/commect_threshold_gate_sensitivity/` | Correctly unresolved; no learned deployment claim |
| Admission might fail after shift | All five rolling folds abstain. Pooled gate and reactive success are 0.678 (196/289), versus 0.671 (194/289) for the unadmitted shield. | `results/commect_validation_gated_rolling/` | Explicitly measured; no future-distribution guarantee claimed |
| Title and scope were too broad | The method and manuscript use `access-path`, `latency-QoS`, and shadow-policy terminology. The contribution is risk-controlled evaluation and admission, not a new predictor or commercial multi-LEO validation. | Manuscript title, abstract, introduction, discussion, conclusion | Complete |
| Timestamp alignment hid age | COMMECT exports observation age and inter-path skew. The rolling protocol is rebuilt at maximum skew 0.5, 1, 2, and 5 s and with the full validated data; the fixed diagnostic is exported separately. A calibration-derived age-margin control is reported as vacuous rather than tuned after test. | `scripts/run_commect_rolling_timestamp_sensitivity.py`; `scripts/run_commect_timestamp_sensitivity.py`; corresponding result directories | Complete within available timestamp precision |
| 5--10 s horizon conflicted with 60 ms QoS | The model separates forecast horizon from collection/inference/dissemination/switch delay. Additive delay and later-state replay are separate tests; whole-bin waiting is explicitly infeasible for a 60-ms end-to-end contract. | `results/transactions_evidence/control_loop_sensitivity.csv`; `delayed_state_replay.csv` | Complete within offline replay |
| Offline replay implied causal deployment | The loader and manuscript identify evaluation as shadow-policy replay under a no-action-feedback assumption. Queue and transport feedback are not claimed. | System model; `docs/open_leo_latency_routing_scope.md` | Complete claim boundary; closed-loop validation remains future work |
| Baselines were mostly predictor variants | Decision controls now include age-aware reactive, robust persistence, CVaR proxy, filter-then-context, and the predictive shield, in addition to temporal/context/ensemble controls. | `results/transactions_evidence/operational_secondary_metrics.csv` | Complete |
| Theory was elementary | The revision defines group-uniform aggregate and opportunity-conditioned estimands, exact harmful-group success bounds, simultaneous bounded-CVaR intervals, and a prospective conditional coverage and sample-requirement audit. These are presented as a fail-closed evidence screen, not a next-epoch safety theorem or a present one-drive coverage claim. | Manuscript method and gate-design appendix; `optimization/risk_control.py` | Complete within stated assumptions |
| Statistical and practical effects were conflated | Admission pre-declares separate 0.02 aggregate and opportunity-conditioned success non-inferiority margins and a 1-ms clipped-CVaR improvement. Familywise alpha covers both success bounds, both CVaR policy intervals, every candidate, and every planned gate use. | Config; risk-control evidence; statistical tables | Complete |
| Evidence and manuscript numbers could drift | The evidence builder recomputes tables and fails on split leakage, denominator mismatch, gate inconsistency, non-exhaustive opportunity partitions, or invalid bounds. | `results/transactions_evidence/numerical_consistency_audit.csv` | 1,503/1,503 checks pass |

## Remaining External Boundary

No open source used here provides a long-duration commercial multi-LEO trace
with synchronized interchangeable paths, controller-visible state, and
post-selection outcomes. COMMECT gives measured heterogeneous 5G--Starlink
shadow-policy evidence from one continuous drive; LENS Victoria gives same-provider co-located LEO
terminals but is threshold-saturated; WetLinks gives prediction transfer only;
and the 18-path orbital evidence is simulated. This limitation cannot be fixed
honestly in code without acquiring new measurements.
