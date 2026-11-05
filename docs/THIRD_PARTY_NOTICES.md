# Third-party data and software notices

This repository contains project source code and tests. It does not redistribute
third-party measurement archives, a vendored simulator, or generated results.
Acquire each input from its official source under the source-specific terms.

- **LENS 2025-03**, J. Zhao and J. Pan:
  [Zenodo record](https://doi.org/10.5281/zenodo.15331299). Verify the current
  release license and archive checksum at acquisition time.
- **COMMECT rural cellular/satellite measurements**, Aalborg University:
  [Zenodo record](https://doi.org/10.5281/zenodo.14620779). The adapter's source
  metadata identifies CC BY-SA 4.0; retain the original attribution and terms.
- **WetLinks**, D. Laniewski et al.:
  [study DOI](https://doi.org/10.23919/TMA62044.2024.10558998). The source project
  distributes its data under CC BY-SA 4.0; consult its archive terms.
- **Statistical characterization and prediction of E2E latency over LEO
  satellite networks**, A. Casparsen:
  [Mendeley Data version 2](https://doi.org/10.17632/479v4mym7j.2), CC BY 4.0.
- **CoNEXT Companion 2023 satellite/cellular road tests**:
  [study DOI](https://doi.org/10.1145/3624354.3630588). Obtain the associated
  source release and follow its stated redistribution and attribution terms.

The optional Hypatia compatibility path obtains upstream software separately
through `scripts/setup_hypatia.sh`. It is not bundled here. All dependencies
retain their upstream licenses. Direct reference environment pins are in
`requirements-lock.txt`; optional Hypatia dependencies are separate.

These notices do not grant a license for the project's own source code. No
project source-code license has been supplied with this repository.
