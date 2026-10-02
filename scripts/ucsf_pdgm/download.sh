#!/usr/bin/env bash
# Download UCSF-PDGM (TCIA, collection version 5, 2025-05-30) into datasets/ucsf_pdgm/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=ucsf_pdgm
OUT=$DATA_ROOT/$NAME
URL=https://www.cancerimagingarchive.net/wp-content/uploads
# Public link to Faspex package 1065 ("UCSF-PDGM Version 5"), from the collection page.
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEwNjUiLCJwYXNzY29kZSI6IjYxZmRlZDkzYzYzM2JhODIxNWYwZjBjMDU2YWRiNjk3Y2IwOTM4MDIiLCJwYWNrYWdlX2lkIjoiMTA2NSIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=='

check_budget $((20 * 10**9))  # kept images ~15 GB
# Version 5 clinical table (labels + BraTS 2021 ID mapping) and its glossary, as linked
# from the collection page's v5 "Data Access" table. The DTI bval/bvec zips are skipped
# because we don't fetch the raw 4D DTI they describe.
mkdir -p "$OUT/source"
for f in UCSF-PDGM-metadata_v5.csv UCSF-PDGM-metadata_glossary.csv; do
    [[ -s $OUT/source/$f ]] && continue
    curl -fsSL -o "$OUT/source/$f.part" "$URL/$f" && mv "$OUT/source/$f.part" "$OUT/source/$f"
done
# Images: the exam folders over FASP into source/UCSF-PDGM-v5/, keeping 13 of the 24 files per
# exam (structural + bias-corrected copies, DWI, ADC, segmentations). Left out: DWI_bias,
# DTI_eddy_* (fits, raw 4D, bvecs), SWI, SWI_bias, ASL, ASL_M0. The package has no checksum file.
uv run --project "$REPO" python -m brain_datasets.tcia_faspex get "$LINK" "$OUT/source" /UCSF-PDGM-v5 \
    --exclude='*_DWI_bias.nii.gz' --exclude='*_DTI_eddy*' --exclude='*_SWI*' --exclude='*_ASL*'
write_manifest "$NAME"
