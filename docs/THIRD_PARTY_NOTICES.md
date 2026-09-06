# Third-Party Data and Software Notices

The reproducibility archive contains project source code and generated
evidence, but it does not redistribute the raw measurement archives or the
vendored Hypatia repository. Rebuild users must obtain each source from its
official record and comply with the source-specific terms.

## Measurement sources

- **LENS 2025-03**, J. Zhao and J. Pan, Zenodo record
  [10.5281/zenodo.15331299](https://doi.org/10.5281/zenodo.15331299). Verify
  the release license and published archive checksum on the official record at
  download time.
- **Radio KPI & Latency Measurement of Cellular and Satellite Networks for
  Evaluating Multi-Connectivity Solutions in Livestock Transport Monitoring in
  Rural Areas**, Aalborg University, Zenodo record
  [10.5281/zenodo.14620779](https://doi.org/10.5281/zenodo.14620779), identified
  by the source metadata as CC BY-SA 4.0.
- **WetLinks**, D. Laniewski et al., distributed under CC BY-SA 4.0 by the
  source project; see the paper DOI
  [10.23919/TMA62044.2024.10558998](https://doi.org/10.23919/TMA62044.2024.10558998)
  and the acquisition script for the official archive location.
- **Statistical characterization and prediction of E2E latency over LEO
  satellite networks dataset**, A. Casparsen, Mendeley Data version 2,
  [10.17632/479v4mym7j.2](https://doi.org/10.17632/479v4mym7j.2), CC BY 4.0.

Generated CSV and figure paths, sizes, and SHA-256 values are recorded in
`results/transactions_evidence/evidence_manifest.json`. Canonical input paths,
sizes, and hashes are recorded separately in
`results/transactions_evidence/build_provenance.json` and source-specific
metadata. Those records do not replace the original dataset terms.

## Optional software

The Hypatia compatibility path uses the upstream Hypatia source at the commit
recorded in the repository README and setup script. Hypatia is not bundled in
the reproducibility archive; obtain it from its upstream repository and follow
its license. Canonical direct/test pins are listed in
`requirements-lock.txt`. The separate `requirements-hypatia.txt` is optional,
non-canonical, and not end-to-end verified. All dependencies retain their
respective upstream licenses.

## Project release status

No license for this project's own source code is asserted by this notice. The
authors must choose and add an explicit repository license before a public
software release. They must also create the immutable release tag and archival
DOI referenced as pending in the manuscript.
