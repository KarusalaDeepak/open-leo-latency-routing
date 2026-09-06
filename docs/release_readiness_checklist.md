# Public Release and Submission Checklist

These actions require an author decision or an external account and are not
silently completed by the reproducibility scripts.

## Before Submission

- Confirm the exact target transaction, page limit, anonymization policy,
  supplementary-material policy, and generative-AI disclosure requirements.
- Verify author order, affiliations, corresponding author, funding,
  acknowledgments, conflicts of interest, and dataset-access statements.
- Choose an explicit project-source license. Apache-2.0 is often suitable when
  a permissive license with an express patent grant is desired; MIT is shorter;
  GPL-3.0 imposes reciprocal source obligations. This is a rights decision for
  the authors, not an automatic build choice.
- Confirm that raw third-party measurement archives are excluded from the
  project release and that derived outputs comply with their licenses.
- Run the canonical rebuild and retain the test, numerical-consistency,
  success-gap, readiness, and PDF-render reports.

## Immutable Artifact Release

1. Commit the exact manuscript and artifact snapshot.
2. Create a signed semantic-version tag and publish the tag to the public
   repository.
3. Connect the repository release to an archival service such as Zenodo and
   mint a version-specific DOI.
4. Add the DOI and release date to `CITATION.cff`, the manuscript results-
   availability statement, and the submission form.
5. Recompute and publish `SHA256SUMS.txt`; verify the archived object against
   those checksums after download.
6. Record the source commit, tag, DOI, dependency lockfile, configuration hash,
   and generated-evidence manifest in one release note.
7. Run `python scripts/build_transactions_evidence.py --output-dir
   results/transactions_evidence --verify-manifest`; do not package a tree
   with missing, extra, size-drifted, or hash-drifted files.

## Claim Discipline

- Do not convert either pending reviewer-readiness item into a pass because a
  protocol, simulator, or replay exists.
- Commercial multi-LEO validation requires authorized, synchronized,
  candidate-level Starlink/OneWeb outcomes from one controller setting.
- Closed-loop deployment requires timestamped installed actions and
  post-installation outcomes; replay and emulation remain separate evidence.
