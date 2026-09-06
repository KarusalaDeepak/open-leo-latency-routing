#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data/external/commect_latency"
ARCHIVE="${DATA_ROOT}/Mutli-connectivity_KPIs_Latency.zip"
URL="https://zenodo.org/records/14620779/files/Mutli-connectivity_KPIs_Latency.zip?download=1"

mkdir -p "${DATA_ROOT}"
if [[ ! -f "${ARCHIVE}" ]]; then
  curl -L --fail --retry 3 "${URL}" -o "${ARCHIVE}"
fi
unzip -o "${ARCHIVE}" -d "${DATA_ROOT}" >/dev/null

# The Python adapter verifies the published MD5 before reading any source file.
python3 "${REPO_ROOT}/scripts/build_commect_multiaccess_trace.py"
