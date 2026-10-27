#!/usr/bin/env bash
# Build datasets/adni/source/ from your own ADNI download (LONI IDA). See README.md.
# Not redistributable: apply for ADNI access, then download from IDA
#   - the DICOM series listed in selected_images.tsv (image IDs), as zips, into one folder;
#   - the ADNIMERGE2 R package (Study Data), version 0.1.1.
# and run
#   ADNI_ZIPS=<folder of zips> ADNIMERGE2=<ADNIMERGE2 tarball> bash scripts/adni/download.sh
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=adni
OUT=$DATA_ROOT/$NAME

[[ -d "${ADNI_ZIPS:-}" ]] || die "set ADNI_ZIPS to the folder of IDA image zips"
[[ -f "${ADNIMERGE2:-}" ]] || die "set ADNIMERGE2 to the ADNIMERGE2 tarball"
mkdir -p "$OUT/source"
# ADNIMERGE2 0.1.1 (packaged 2025-12-17, data pulled 2025-11-12), as downloaded 2026-06-18.
echo "c2b2973e3216a6da2fb22b6c25200ca25283a007a4dfc2ad4814e4333b18fb13  $ADNIMERGE2" \
    | sha256sum -c --quiet || die "ADNIMERGE2: checksum mismatch"
[[ -d "$OUT/source/ADNIMERGE2" ]] || tar -xzf "$ADNIMERGE2" -C "$OUT/source"
# The images: one T1w and FLAIR per session (y0, y2, y4), chosen by select_images.py.
cp "$(dirname "$0")/selected_images.tsv" "$OUT/source/"
uv run -q --project "$REPO" --inexact python "$(dirname "$0")/convert.py" \
    "$ADNI_ZIPS" "$OUT/source/selected_images.tsv" "$OUT/source" --jobs "${JOBS:-8}"
write_manifest "$NAME"
