# Independent Commercial Multi-LEO Validation Protocol

## Candidate Dataset

The strongest identified candidate is the Aalborg University campaign reported
in *Toward Reliable Connectivity: Measurement-Driven Assessment of Starlink and
OneWeb Non-Terrestrial and 5G Terrestrial Networks* (IEEE OJ-COMS, 2026, DOI
`10.1109/OJCOMS.2026.3682548`). The paper reports synchronized Starlink,
OneWeb, and terrestrial measurements and 7.7 million packet observations. No
public raw-data archive was identified on 21 August 2026, so the repository
does not present this campaign as completed validation.

Authoritative records checked on that date were the
[Aalborg University publication record](https://vbn.aau.dk/en/publications/toward-reliable-connectivity-measurement-driven-assessment-of-sta/),
the [publisher DOI](https://doi.org/10.1109/OJCOMS.2026.3682548), and the
[corresponding-author profile](https://vbn.aau.dk/da/persons/araar/). None of
those records exposed a raw-data or supplementary-trace archive. The absence
of a public link is recorded as an access limitation, not interpreted as
evidence that the data do not exist.

The public LENS repository was also rechecked because it now exposes a OneWeb
archive. Its OneWeb terminal is documented in Ames, Iowa, while the Starlink
Iowa installation is documented in Iowa City. These geographically separated
installations are useful single-provider measurement sources, but they are not
interchangeable alternatives at one controller. Timestamp overlap between
them would therefore not close the synchronized commercial multi-LEO policy
validation gap.

## Required Data Contract

The requested raw table must contain one row per packet attempt or equivalent
fine-grained observation. It must provide:

| Field | Requirement |
| --- | --- |
| Timestamp | Timezone-aware and shared across both operators |
| Operator | Starlink or Eutelsat OneWeb |
| Latency | One-way delay or RTT in milliseconds |
| Packet outcome | Received/lost indicator when lost packets are retained |
| Scenario | Urban, suburban, forest, static, or equivalent context |
| Direction | Uplink/downlink when available |
| Position | Per-packet latitude and longitude for both terminals; required for any selectable-path claim |
| Controller topology | A shared `controller_id` for both interfaces in every decision epoch, or documented data-owner confirmation that one controller could select both paths |
| Campaign identity | Optional `campaign_id`; use only for physically and administratively separate collection campaigns, never files, dates, windows, sessions, vehicles within one campaign, or constructed time blocks |

The timeout used by the original campaign must be supplied to the importer. It
is not estimated from evaluation outcomes.

## Claim Gate

`build_commercial_multileo_trace.py` checks the following dataset and topology
criteria independently:

1. documented independence from LENS;
2. measured commercial Starlink and OneWeb observations;
3. both candidates observed in the same decision epochs;
4. outcomes retained for all candidates, enabling unchosen-path replay; and
5. at least 30 elapsed days and 24 hours of complete concurrent observations
   by default;
6. valid GPS coordinates on every retained packet row in each complete decision
   epoch;
7. a maximum inter-terminal separation of 100 m in every complete epoch; and
8. evidence that both paths were simultaneously selectable by one controller.

The default synchronization gate also requires the 95th-percentile timestamp
skew between operators to be at most 100 ms. The spatial audit computes WGS84
great-circle distance between the median Starlink and OneWeb terminal
coordinates in each complete bin. It requires 100% raw-coordinate coverage and
applies the 100 m threshold to the maximum per-bin representative distance, not
the campaign mean or percentile. Gaps create new
sequence segments, preventing the forecasting code from treating separated
measurements as adjacent next-bin outcomes.

GPS co-location is necessary but not sufficient for a same-controller claim.
The topology gate additionally requires matching non-empty `controller_id`
values in every complete epoch, or an explicit provenance assertion with a
non-empty citation or data-owner-confirmation note. A mapped controller-ID
conflict overrides an external assertion and fails closed. Use
`--same-controller-provenance` only when the campaign documentation establishes
literal selectability; a shared vehicle, convoy, site, or timestamp does not by
itself establish a shared controller.

These values are hard lower/upper claim floors. Command-line arguments may make
the audit stricter, but cannot turn a shorter, less synchronized, or spatially
separated campaign into limitation-closing evidence. Missing GPS also fails the
topology gate rather than being interpreted as co-location.

The output separates the following scopes:

- `literal_same_controller_selectable_path_replay` is available only after the
  spatial and controller-topology gates pass;
- `scoped_near_concurrent_colocated_or_convoy_replay` means the terminals were
  within the distance threshold but shared-controller selectability is not
  established; and
- a restricted time-aligned comparison is used when location is absent,
  incomplete, or the terminals are spatially separated. Measurements that also
  fail the paired-outcome/synchronization audit are labelled
  `non_counterfactual_dual_operator_observations`.

`temporal_concurrency_audit` preserves the generic same-timestamp calculation
as `has_temporally_concurrent_candidates`; this is never topology evidence.
The claim-safe `concurrency_audit` separately records candidate-outcome replay
support and `supports_literal_single_controller_steering`, with the latter true
only when the spatial and shared-controller gates pass. The field
`supports_closed_loop_deployment_evidence` remains false for imported traces.

The validation runner fails closed on the two scoped categories. They can be
run only with `--allow-scoped-paired-replay`, and the resulting metadata marks
`policy_level_evaluation` and `measured_concurrent_paths` false.

## Chronology and Gate-Inference Units

Commercial replay uses the same global shared-wall-clock four-way protocol as
the principal evaluation: model fitting on `train`, residual/uncertainty
fitting on `calibration`, policy admission on `selection`, and one-time scoring
on `test`. Forecasting targets that cross any boundary are removed. The runner
exports partition epochs, disjointness, strict ordering, and target-closure
checks in `split_audit`; it does not use the legacy per-path three-way split.

A time series does not create independent replications. Unless the importer
receives a complete paired `campaign_id` field with at least two distinct IDs
and `--independent-campaign-ids-audited` plus a provenance note, the complete
selection interval is one gate-inference group. The risk-control block length
is forced to `None` in this case, so dates, gaps, files, sequence segments, and
arbitrary blocks cannot inflate the effective opportunity count. When audited
campaign IDs are enabled, both operators must carry the same non-empty ID in
every complete epoch; conflicts or missing values fail closed. The manual
independence assertion must be supported by campaign documentation or explicit
data-owner confirmation. The runner also verifies that an unaudited single-
campaign import selects the reactive fallback and exposes at most one inference
block (zero only when every selection epoch is an outage); otherwise it stops
without writing a validation claim.

The generated metadata field
`closes_independent_longitudinal_multileo_limitation` is true only when every
dataset and topology criterion passes. A time-aligned AAU campaign without
complete co-location and shared-controller evidence supports only the scoped
comparison named in its metadata, even if it is long-running.

## Reproduction Commands

After receiving the authorized raw CSV, update the example column map and run:

```bash
python scripts/build_commercial_multileo_trace.py \
  --input data/external/aau_multileo/raw_packet_outcomes.csv \
  --column-map configs/commercial_multileo_columns.json \
  --output data/processed/commercial_multileo_10s.csv \
  --bin-seconds 10 \
  --timeout-ms ORIGINAL_CAMPAIGN_TIMEOUT \
  --minimum-duration-days 30 \
  --minimum-concurrent-hours 24 \
  --maximum-p95-skew-ms 100 \
  --maximum-inter-operator-distance-meters 100 \
  --dataset-name aau_starlink_oneweb \
  --dataset-url https://doi.org/10.1109/OJCOMS.2026.3682548 \
  --dataset-doi 10.1109/OJCOMS.2026.3682548 \
  --license DATA_OWNER_LICENSE \
  --independent-of-lens \
  --same-controller-provenance \
  --controller-provenance-note "DATA_OWNER_CONFIRMATION_OR_PRIMARY_SOURCE"

python scripts/run_commercial_multileo_validation.py
```

If the raw table supplies matching `controller_id` values for both paths in
every epoch, the last two controller-provenance arguments may be omitted. Do
not assert independence or shared-controller topology without documented
provenance. To intentionally run a non-selectable paired trace, add
`--allow-scoped-paired-replay` to the validation command and retain the emitted
claim restriction verbatim.

Only when the mapped table contains at least two genuinely independent
campaigns, append the following importer arguments:

```bash
  --independent-campaign-ids-audited \
  --campaign-independence-note "DATA_OWNER_CONFIRMATION_OR_PRIMARY_SOURCE"
```

Without those arguments, even a populated `campaign_id` column remains
descriptive and the gate treats the import as one campaign.

## Data-Access Request

The ready-to-send request is stored in
`docs/commercial_multileo_data_request_email.txt`; an email-client draft with
the `X-Unsent` marker is stored in
`docs/commercial_multileo_data_request.eml`. The corresponding author
listed by Aalborg University is Dr. Alejandro Ramirez-Arroyo
(`araar@es.aau.dk`). The repository records the request as a draft until an
email client confirms transmission; preparing a local message is not counted
as author contact.

Subject: Request for synchronized Starlink--OneWeb measurement traces (IEEE OJ-COMS 2026)

Dear Dr. Ramirez-Arroyo,

I am K Deepak Chowdary, a PhD researcher at IIIT Dharwad, India. I am
evaluating a lightweight, evidence-gated QoS-shielded service-path selector
using synchronized commercial LEO measurements. Your article, "Toward Reliable
Connectivity: Measurement-Driven Assessment of Starlink and OneWeb
Non-Terrestrial and 5G Terrestrial Networks" (IEEE OJ-COMS, 2026), is uniquely
suitable because the reported campaign measures Starlink and OneWeb
simultaneously. Would it be possible to obtain anonymized, time-aligned raw or
processed per-interface traces, subject to your preferred licence and
acknowledgement conditions?

For reproducible chronological train/calibration/test evaluation, the useful
fields are timestamps, operator/interface identity, one-way delay or RTT,
packet-received/lost outcome, throughput, availability, scenario, direction,
and per-terminal GPS information where release is permitted. Could you also
confirm the campaign dates and duration, packet timeout/failure convention,
timestamp synchronization accuracy between Starlink and OneWeb, whether every
interface was probed continuously (including paths not selected by an online
policy), whether the terminals were co-located throughout the campaign, and
whether one controller could literally select between both interfaces at each
decision epoch? If the release spans multiple collection campaigns, could you
also identify them and confirm whether they are independent replications rather
than files, sessions, days, or windows from one campaign?

We will cite the article (DOI 10.1109/OJCOMS.2026.3682548), preserve the
original provenance, publish only permitted derived or aggregate results, and
share our evaluation code. If a public archive is not possible, we are happy
to follow a research-only data agreement.

Kind regards,

K Deepak Chowdary
PhD Researcher, Department of Computer Science and Engineering
Indian Institute of Information Technology Dharwad, India
deepak.24phdcs02@iiitdwd.ac.in
