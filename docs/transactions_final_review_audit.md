> **SUPERSEDED (2026-08-21).** This historical review note is not a source of current claims or counts. Use [`README.md`](../README.md), the `main.tex` in the `leo-conf2-transactions-revision-2026-08-21` submission bundle, and [`results/transactions_evidence/`](../results/transactions_evidence/) as the canonical sources.

# Transactions Final Review Audit

This audit maps the final pre-submission concerns to executable evidence and
the corresponding manuscript treatment. It is not a claim of acceptance.

| Concern | Code-side evidence | Manuscript treatment | Status |
|---|---|---|---|
| Recommended policy can collapse to reactive | `validation_gate_selection_audit.csv` exports every pre-test choice; COMMECT is reactive in the fixed holdout and all five rolling folds | Abstract, Table V, and Results distinguish predictive-only shield, gated policy, and reactive baseline | Addressed |
| Test-time policy selection | Unit tests verify abstention and tie-breaking; the evidence audit verifies one allowed frozen fallback per case/fold | Four ordered blocks state train, calibrate, select and freeze, then test once | Addressed |
| Broad shield guarantee | Tests verify only the implemented current-state exclusion branch | Rule statements are labeled properties; next-epoch preservation is explicitly conditional and not distribution-free | Addressed |
| Disagreement presented as a shift detector | Matched-capacity audit, residual calibration, and WetLinks AUROC/correlation diagnostics are exported | Disagreement is a validation-dependent dispersion component; WetLinks chance-level result is retained | Addressed |
| COMMECT overstated as multi-LEO | Dataset metadata encodes valid and invalid claim boundaries | Every policy-level COMMECT claim is qualified as external-source heterogeneous 5G--Starlink access | Addressed |
| Victoria saturation | Opportunity audit reconstructs 647 all-pass epochs and one mixed-outcome opportunity among 648 decisions | Victoria is described only as concurrency and non-regression evidence at 60 ms | Addressed |
| Injected stress presented as measured outage evidence | Simulator metadata marks the trace unmeasured and records injected event windows | Abstract, dataset table, captions, Results, and Discussion call these controlled injected-shift stress tests | Addressed |
| Ambiguous Hypatia role | Hypatia adapter outputs remain available but are excluded from the canonical primary source list | Hypatia is limited to adapter/dynamic-state compatibility and has no primary policy result | Addressed |
| Fixed/rolling COMMECT values conflated | Separate fixed and rolling result directories and non-overlap checks | Abstract and Results report 88 fixed decisions (57/88 success for reactive, shield, and gate; 72 opportunities) separately from 289 rolling decisions (196/289 reactive/gate, 194/289 shield; 236 opportunities) | Addressed |
| Tail-latency overclaim | Segment-confined block intervals and centered-null diagnostics are generated from decision rows | Mean, P95, and CVaR differences without resolved inference are labeled descriptive | Addressed |
| Threshold dependence hidden | Objective-specific policies are retrained, recalibrated, reselected, and re-executed at 40/60/100/200 ms | Paper reports reactive/shield/gate success of 70/71/70 at 100 ms and 77/79/77 at 200 ms, labels the differences unadmitted, and states that the gate is objective-dependent | Addressed |
| Conformal/risk threshold reuse | Run metadata records separately fitted calibration artifacts and the disjoint policy-selection evidence | Methodology separates residual calibration from evidence-gated policy admission | Addressed |
| Per-seed retraining mistaken for transfer | Run metadata records the within-trace training protocol | Multi-seed section limits the claim to reproducibility with per-seed calibration | Addressed |
| Table denominators or summaries inconsistent | The canonical build performs 1,503 checks, including summary metrics recomputed from decision rows | Reported table values are sourced from the canonical evidence directory | Addressed |
| Typesetting/extraction artifacts | Not applicable | The final PDF is rendered page by page; clipping, overfull boxes, and broken references are checked before release | Addressed |

The unresolved evidence boundary is external rather than an implementation
defect: no open long-duration commercial multi-LEO trace currently supplies
timestamp-aligned interchangeable candidates and next-outcome labels suitable
for policy replay. The manuscript does not claim that validation.
