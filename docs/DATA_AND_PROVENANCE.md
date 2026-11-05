# Data and provenance

External data are acquired separately. Keep their archives and extracted data
under ignored local directories. Verify source terms and published checksums
before extraction. Source-specific adapters record input paths, sizes, hashes,
pairing rules and eligible decision counts in their generated metadata.

| Source | Acquisition reference | Role and limitation |
| --- | --- | --- |
| COMMECT rural transport measurements | [Zenodo 14620779](https://doi.org/10.5281/zenodo.14620779) | Concurrent 5G and Starlink replay; one acquisition campaign. Folds and bins do not create independent campaigns. |
| CoNEXT Companion 2023 satellite/cellular road tests | [Study DOI](https://doi.org/10.1145/3624354.3630588) | April 23 development; April 24–28 conditional daily gate audit. Five dates belong to one acquisition family. |
| LENS 2025-03 | [Zenodo 15331299](https://doi.org/10.5281/zenodo.15331299) | Prediction and QoS diagnostics; paired selectable terrestrial alternatives are not established by this adapter. |
| WetLinks | [Study DOI](https://doi.org/10.23919/TMA62044.2024.10558998) and `scripts/download_wetlinks_data.sh` | Longitudinal and cross-site prediction diagnostics. Sites are not installed selectable alternatives. |
| Independent IRTT measurements | [Mendeley Data v2](https://doi.org/10.17632/479v4mym7j.2) | External RTT characterization under its own source schema. |

The CoNEXT release lacks a destination-server identifier and a per-sample
LTE-versus-5G label. The audit assumes a common UDP-Ping endpoint and independent,
exchangeable daily groups; it does not verify these properties. Provider-set
qualification uses retrospective coverage metadata. Untimestamped timeouts
cannot be inserted into exact scheduled intervals and are excluded from the
complete-case RTT endpoint. These limitations must accompany reported results.

## Selection boundary

- COMMECT ridge penalties use the chronological development tail inside the
  outer training block. CoNEXT fixes the ridge penalty at 1; its five-value
  sensitivity is reporting-only.
- Bias correction and temporal/context fusion use their calibration block.
- Candidate and application/statistical profiles are frozen before gate evidence.
- Source datasets and statistical profiles are not pooled into a campaign count
  merely because they contain many rows or sites.
- If independent group identifiers are absent, the selection interval is one
  group. Eligibility count is a necessary condition, not a power guarantee.

`configs/campaign_qualification.yaml` and `configs/evidence_profiles.yaml`
provide the executable source roles and profiles. Additional optional adapters
remain available for explicitly authorized datasets; their presence is not
evidence that those datasets have passed acquisition or gate requirements.

## Reproduction outputs

Scientific scripts write local CSV/JSON/figure outputs beneath ignored
`results/`, `data/processed/` or `output/` directories. Evidence-manifest helpers
can verify generated file sets, byte sizes and SHA-256 values. No generated
result, measurement archive or manuscript is distributed in this source-only
repository.
