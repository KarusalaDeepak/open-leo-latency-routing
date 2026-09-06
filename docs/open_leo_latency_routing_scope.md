# Open LEO Latency Routing Scope

## Working Title

Opportunity-Aware Evidence Gating for Latency-QoS Multi-Access Selection in
LEO--Terrestrial Networks

## Core Question

When concurrently observable access alternatives are evaluated by
next-epoch network RTT, is there enough held-out, independently replicated
evidence to admit a learned ranking policy over a reactive baseline?

The current contribution is an evaluation-and-abstention methodology, not a
new forecasting family, an unconditional safety guarantee, or evidence of
commercial multi-LEO deployment benefit. Predictor disagreement is one
calibrated diagnostic and ranking feature; it is not assumed to be an
independent uncertainty estimator or a reliable stand-alone shift detector.

## Required Comparisons and Audits

- Reactive selection versus context, ensemble, shielded, and evidence-gated
  policies under equal test information.
- Four globally ordered train, residual-calibration, policy-selection, and
  test intervals in the primary fixed and rolling policy protocols.
- Explicit separation of raw decision opportunities from independent,
  opportunity-bearing collection groups.
- Simultaneous aggregate actionable-population and post-hoc
  opportunity-conditioned success non-inferiority, plus clipped-CVaR benefit
  evidence, before a learned fallback can be admitted. The conditional
  success estimand draws an opportunity-bearing independent group uniformly
  and then a decision-discriminating epoch uniformly within that group.
- Current-state QoS branch, timestamp-skew, delayed-state, switching, and
  controller-runtime audits.
- Clear separation of measured shadow replay, prediction-only transfer,
  simulator stress tests, and implementation-compatibility checks.

## Supported Evidence Boundary

- COMMECT supports heterogeneous 5G--Starlink shadow replay from one drive.
- The LENS Victoria trace supports a co-located same-provider candidate-outcome
  counterfactual, but public metadata do not verify one steering controller and
  the result is nearly saturated at the 60-ms threshold.
- WetLinks and the independent Starlink trace support prediction transfer, not
  online path-selection claims between geographically separated sites.
- The physics-informed trace supports controlled simulated stress analysis;
  Hypatia supports adapter/dynamic-state compatibility only.

All measured evidence gates currently abstain because the available campaigns
do not contain enough independent opportunity-bearing collections. This is the
intended fail-closed result. Additional independent drives, synchronized
commercial multi-LEO alternatives, and closed-loop action/queue measurements
remain prerequisites for a learned-deployment claim.

## Reviewer Risks Retained Explicitly

- one-drive dependence and low evidence-gate power for modest tail gains;
- bin-aligned rather than packet-level synchronization;
- shadow replay without action-induced queue feedback;
- network RTT rather than end-to-end service completion time;
- synthetic orbital impairments and absence of commercial multi-LEO replay.
