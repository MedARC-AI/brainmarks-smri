#!/usr/bin/env bash
# Download SOOP (OpenNeuro ds004889, snapshot 1.1.2) into datasets/soop/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=soop
OUT=$DATA_ROOT/$NAME
FILES=(uv run --no-project python "$(dirname "$0")/snapshot_files.py")

check_budget $((80 * 10**9))
# The whole snapshot is in scope: anat/ (T1w, FLAIR), dwi/ (TRACE + ADC only, no raw
# multi-direction DWI) and derivatives/lesion_masks/ (the segmentation targets, the only
# derivative). There is no fMRI.
# openneuro-py fails on this dataset (HEAD on GET-presigned S3 URLs -> 403; see
# snapshot_files.py), so we list the snapshot via the OpenNeuro API and fetch each pinned
# S3 object version with curl. Only missing or incomplete files are listed, so re-runs resume.
mkdir -p "$OUT/source"
"${FILES[@]}" curl-config ds004889 1.1.2 "$OUT/source" > "$OUT/curl.cfg"
if [[ -s $OUT/curl.cfg ]]; then
    curl --parallel --parallel-max 8 -fsSL --retry 5 --create-dirs -C - -K "$OUT/curl.cfg"
fi
rm "$OUT/curl.cfg"
write_manifest "$NAME"
# Check every file against the snapshot's own checksums (git-annex SHA256 / git blob SHA1).
"${FILES[@]}" verify ds004889 1.1.2 "$OUT/source" "$OUT/manifest.sha256"
