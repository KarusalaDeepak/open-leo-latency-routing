# Prospective Closed-Loop Field Validation Protocol

## Status and Claim Boundary

This document is an implementation-ready protocol, not evidence that a field
trial has occurred. The repository must continue to label closed-loop
deployment evidence as pending until the claim gate at the end of this document
passes and the signed run metadata is archived.

## Objective

Test whether the frozen evidence-gated controller can install real access-path
actions before their telemetry expires, without treating shadow replay as a
causal deployment result. The trial separates network RTT, controller compute,
action-installation latency, queueing, and outage/no-action events.

## Required Testbed

- At least two simultaneously usable access interfaces connected to the same
  client and application endpoint.
- Independent probe traffic on every candidate path, including paths not chosen
  for application traffic, so candidate outcomes remain observable without
  assuming the chosen action caused them.
- A flow-steering mechanism whose acknowledgement can be timestamped, such as
  policy routing, an SD-WAN rule, or a multipath proxy.
- Monotonic clocks on collection, controller, steering, and receiver processes,
  plus a documented UTC synchronization-error bound.
- An append-only log and packet capture sufficient to reconstruct the order
  `telemetry cutoff -> frozen-policy decision -> installation request ->
  installation acknowledgement -> first steered packet -> outcome`.

## Frozen Trial Configuration

Before the first evaluation drive, archive the following values and their
SHA-256 digest:

- latency-QoS threshold and packet-loss/timeout convention;
- telemetry cadence, state-age ceiling, and action deadline;
- trained experts, calibration state, candidate scores, and admitted fallback;
- switching penalty, evidence-gate configuration, loss cap, CVaR grid, and
  planned gate-use count;
- decision-opportunity and independent collection-group definitions;
- primary and secondary endpoints and every exclusion rule.

No threshold, model, timeout, group definition, or exclusion rule may be
changed after an evaluation outcome is inspected. A changed configuration
starts a new, separately identified trial.

## Trial Design

1. Use complete drives or predeclared day-by-route campaigns as collection
   groups; consecutive telemetry bins are not independent replications.
2. Assign complete evaluation groups to frozen reactive or evidence-gated
   execution using a pre-generated randomized crossover schedule. Include
   washout intervals after interface changes.
3. Preserve a deterministic safety monitor outside learned ranking. When no
   currently feasible path exists, record outage/no-action rather than forcing
   an emergency choice into the routing estimand.
4. Log all failed installations, late installations, controller crashes,
   missing candidate probes, and operator outages. Do not silently discard
   them from availability or total-latency analyses.
5. Stop learned execution and revert to reactive on a predeclared safety-monitor
   violation. Report the stop as an outcome, not as missing data.

## Endpoints

The primary endpoint is per-group network-RTT success under the frozen latency
threshold. Required secondary endpoints are clipped CVaR network RTT,
end-to-end success including installation delay, action-installation latency,
state age at installation, switch rate, failed-installation rate, outage rate,
packet loss, and queue occupancy when observable. Network RTT and controller
timing must remain separate columns and may be combined only in a separately
labelled total-latency endpoint.

## Minimum Logged Schema

| Field | Meaning |
| --- | --- |
| `trial_id`, `collection_group_id` | Immutable trial and independent group |
| `decision_id`, `candidate_path_id` | Decision and candidate identifiers |
| `telemetry_timestamp_ns` | Last candidate telemetry admitted to the decision |
| `decision_timestamp_ns` | Frozen-policy decision completion |
| `install_request_ns`, `install_ack_ns` | Steering request and acknowledgement |
| `first_steered_packet_ns` | First packet verified on the selected path |
| `selected_path_id`, `installed_path_id` | Requested and observed action |
| `path_state`, `observed_replies` | Availability inputs for every candidate |
| `candidate_rtt_ms`, `candidate_loss` | Concurrent probe outcomes for all paths |
| `application_rtt_ms`, `application_loss` | Installed application-path outcome |
| `outage_no_action`, `late_install`, `install_failed` | Required failure flags |
| `policy_artifact_sha256`, `config_sha256` | Frozen artifact identity |

## Integrity and Analysis Checks

- Reject a run if decision or installation timestamps are not monotonic, the
  clock-error bound exceeds the action deadline, or the installed path cannot
  be verified.
- Report candidate-probe coverage and application-traffic coverage separately.
- Reconstruct every decision from the archived policy artifact and require
  exact path/branch agreement.
- Keep descriptive moving-block intervals separate from the independent-group
  evidence gate.
- Apply the planned-use multiplicity allocation across every candidate,
  endpoint, grid point, and interim gate invocation.
- Publish all randomized assignments, exclusions, stops, missingness counts,
  and negative outcomes.

## Claim Gate

`results/closed_loop_deployment/deployment_metadata.json` may set
`closed_loop_field_trial`, `policy_installed_before_outcomes`, and
`network_actions_executed` to true only when timestamped logs, packet captures,
frozen hashes, reconstruction audit, exclusions, and the primary analysis are
present. Until then, the valid claim is only that the artifact provides a
prospective field-validation protocol.
