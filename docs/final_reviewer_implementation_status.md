> **SUPERSEDED (2026-08-21).** This historical review note is not a source of current claims or counts. Use [`README.md`](../README.md), the `main.tex` in the `leo-conf2-transactions-revision-2026-08-21` submission bundle, and [`results/transactions_evidence/`](../results/transactions_evidence/) as the canonical sources.

# Final Reviewer Implementation Status

This map covers the three original reviewers and the subsequent
Transactions-level critique. `Complete within scope` means that code,
generated evidence, and manuscript claims agree; it does not guarantee
acceptance.

## Original Reviews

| Reviewer comment | Implemented response | Status |
| --- | --- | --- |
| R1.1 system figure and notation | Generated concurrent access model and timing diagram; manuscript includes principal notation and two executable algorithms. | Complete |
| R1.2 generalization beyond LENS | Added external-source COMMECT policy replay, LENS Victoria holdout, WetLinks longitudinal/unseen-site transfer, physics-informed multi-path traces, and Hypatia compatibility. Dataset roles are not pooled. | Substantially addressed; commercial multi-LEO outcome data remain unavailable |
| R1.3 injected degradation | Injected events occur only in simulator evaluation; labels, captions, tables, and manifests distinguish measured, simulated, and injected evidence. | Complete |
| R1.4 operational rather than predictive novelty | Final contribution is an optional learned ranker behind an opportunity-aware risk-control gate. Predictive-only behavior is an ablation. | Complete |
| R1.5 ensemble uncertainty | Shared-failure, dispersion/error, bootstrap-spread, and coverage diagnostics explain why ensemble spread can outperform raw disagreement. | Complete |
| R2.1 physical/theoretical disagreement grounding | Disagreement is only the between-expert term in total mixture variance and is admitted only after residual calibration. WetLinks near-chance AUROC is retained as a boundary. | Complete for the bounded, non-universal claim |
| R2.2 LENS dependence | Added independent sources and portable adapters, while explicitly separating policy evidence from prediction transfer. | Substantially addressed |
| R2.3 control-loop latency | Added collection, inference, dissemination, switching, additive-delay, and later-state replay models. | Complete within shadow replay |
| R2.4 OLS/XGBoost mismatch | Replaced the default pair with standardized, matched-estimator-family ridge experts using distinct information views; added linear/ridge/tree/MLP family audits. | Complete |
| R2.5 60-s resolution | Main concurrent experiments use 5- or 10-s cadence; 5/10/30/60-s diagnostics are supported. Timing semantics state that the horizon is not control-loop waiting. | Complete |
| R2.6 weaker context model | Validation bias and inverse-variance calibration account for unequal expert quality; complementarity is audited rather than assumed. | Complete |
| R3.1 motivation | Introduction and diagnostics explain overconfidence, common-mode failure, and why independent information views can be complementary. | Complete |
| R3.2 predictor-pair generality | Added matched-family and 4-by-4 predictor-pair audits. | Complete |
| R3.3 related-work attribution | Rewritten reference-by-reference comparison and literature table. | Complete in manuscript |
| R3.4 contribution separation | Algorithmic, evaluation, and reproducibility contributions are distinguished. | Complete |
| R3.5 manual score weights | Biases, residual variances, fusion weights, risk coefficients, and policy admission are fitted only on pre-test blocks. | Complete |
| R3.6 ablation completeness | Added temporal-only, context-only, fusion, disagreement, ensemble, conformal, calibrated-risk, filter-then-learned, predictive shield, and risk-controlled gate variants. | Complete |

## Transactions-Level Follow-Up

| Concern | Final implementation | Evidence/result | Status |
| --- | --- | --- | --- |
| Gate too close to reactive | The paper now treats safe abstention as the primary mechanism. Learned ranking is optional and must satisfy independent evidence constraints. | Algorithms 1--2 | Complete framing and implementation |
| Same data used for calibration and admission | Four chronological blocks: 0.55 train, 0.15 calibration, 0.15 policy selection, 0.15 test. | Split manifests and leakage tests | Complete |
| Opportunity audit not operational | Admission requires at least five independent opportunity-bearing groups and a separately protected harmful-group success LCB conditional on post-hoc opportunities; ties and inadequate opportunity evidence default to reactive. | Gate evidence | Complete |
| Point-estimate admission | Independent-group finite-sample bounds must clear separate -0.02 aggregate actionable and opportunity-conditioned success margins plus a 1-ms clipped-CVaR gain margin simultaneously. | `optimization/risk_control.py` | Complete |
| No positive measured admission | The complete pipeline is rerun at 40/60/100/200 ms, but one drive remains one inference group and the gate abstains at every threshold. | Threshold gate sensitivity | Correctly unresolved; no learned deployment claim |
| Admission robustness | All five rolling folds abstain; pooled gate equals reactive at 0.678 (196/289), while the unadmitted shield obtains 0.671 (194/289). | Rolling-origin evaluation | Negative evidence retained; no unsupported guarantee |
| Timestamp skew/AoI | The five-fold rolling policy is rebuilt at 0.5/1/2/5-s maximum skew and on the full validated data; observation age is exported. | Rolling timestamp-skew sensitivity | Complete |
| Strong decision controls | Added age-aware reactive, robust persistence, CVaR proxy, and filter-then-context. | Operational controls table | Complete |
| Practical significance | Pre-declared success and CVaR margins; descriptive block intervals and independent opportunity-bearing group counts reported. | Statistical protocol | Complete |
| Causal replay | Shadow-policy counterfactual assumption is explicit; no action-induced queue effect is claimed. | System model and Threats to Validity | Complete claim boundary |
| Multi-metric QoS overclaim | Method is described as latency-QoS; threshold-specific complete reselection is reported at 40/60/100/200 ms. | Threshold experiment | Complete for latency scope |
| Dense or ambiguous terminology | Policies standardized to Reactive baseline, Predictive shield, and Risk-controlled gate; title uses access-path. | Manuscript and generated figures | Complete |

## Current Headline Evidence

| Protocol | Reactive | Predictive shield | Risk-controlled gate | Interpretation |
| --- | ---: | ---: | ---: | --- |
| COMMECT fixed, 60-ms success | 0.648 (57/88) | 0.648 (57/88) | 0.648 (57/88) | Gate abstains; shield mean is descriptively lower but its block interval includes zero; 72/88 test epochs are decision opportunities |
| COMMECT rolling, 60-ms success | 0.678 (196/289) | 0.671 (194/289) | 0.678 (196/289) | Every fold abstains because the drive supplies one inference group; 236/289 test epochs are decision opportunities |
| COMMECT reselected, 100-ms success | 0.795 (70/88) | 0.807 (71/88) | 0.795 (70/88) | Unadmitted one-success shield difference; gate remains reactive |
| COMMECT reselected, 200-ms success | 0.875 (77/88) | 0.898 (79/88) | 0.875 (77/88) | Unadmitted two-success shield difference; gate remains reactive |
| LENS Victoria, 60-ms success | 1.000 | 1.000 | 1.000 | Gate defaults to reactive because there is only one opportunity-bearing collection |

## Verification

- Automated tests: 148/148 pass (one third-party deprecation warning).
- Numerical consistency checks: 1,503/1,503 pass.
- All principal policy folds use disjoint chronological train, calibration,
  policy-selection, and test intervals.
- The release procedure regenerates all figures and tables from canonical
  result artifacts before freezing checksums.
- The paper reports the remaining lack of commercial concurrent multi-LEO
  closed-loop evidence rather than substituting simulated data for it.
