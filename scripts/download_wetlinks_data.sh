#!/usr/bin/env bash
set -euo pipefail

# Download only the compact merged tables required by the longitudinal audit.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repo_root}/data/external/wetlinks_dataset"
temporary_root="$(mktemp -d)"
archive="${temporary_root}/WetLinks-main.zip"
trap 'rm -rf "${temporary_root}"' EXIT

curl -L --fail --retry 5 --retry-delay 3 \
  -o "${archive}" \
  https://codeload.github.com/sys-uos/WetLinks/zip/refs/heads/main
unzip -q "${archive}" -d "${temporary_root}"

mkdir -p "${destination}/Preprocessed_Data"
cp "${temporary_root}/WetLinks-main/Preprocessed_Data/analysis_data_Enschede.csv" \
  "${destination}/Preprocessed_Data/"
cp "${temporary_root}/WetLinks-main/Preprocessed_Data/analysis_data_Osnabr++ck.csv" \
  "${destination}/Preprocessed_Data/"
cp "${temporary_root}/WetLinks-main/README.md" \
  "${destination}/DATASET_README.md"

printf 'WetLinks analysis tables written to %s\n' "${destination}"
