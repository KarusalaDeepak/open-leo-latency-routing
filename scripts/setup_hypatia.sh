#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HYPATIA_ROOT="${REPO_ROOT}/external/hypatia"
HYPATIA_COMMIT="0ac531c313eba2335f6344b46347140c3a0d4230"

if [[ ! -d "${HYPATIA_ROOT}/.git" ]]; then
  git clone https://github.com/snkas/hypatia.git "${HYPATIA_ROOT}"
  git -C "${HYPATIA_ROOT}" checkout --detach "${HYPATIA_COMMIT}"
fi

actual_commit="$(git -C "${HYPATIA_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${HYPATIA_COMMIT}" ]]; then
  printf 'Hypatia commit mismatch: expected %s, found %s\n' \
    "${HYPATIA_COMMIT}" "${actual_commit}" >&2
  printf 'Use a separate clone or explicitly check out the recorded commit.\n' >&2
  exit 1
fi

python3 -m pip install -r "${REPO_ROOT}/requirements-hypatia.txt"
printf 'Hypatia source and optional dependencies prepared at commit %s.\n' \
  "${actual_commit}"
printf '%s\n' \
  'This non-canonical dependency set remains unverified until the adapter generation and its tests succeed.'
