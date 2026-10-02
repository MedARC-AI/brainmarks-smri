#!/usr/bin/env bash
# Download UPENN-GBM (TCIA collection, Version 2, 2022/10/24) into datasets/upenn_gbm/:
# the supporting tables (HTTPS) and the structural NIfTI images + tumor segmentations
# (Aspera Faspex package 604, "UPENN-GBM-NIfTI"). See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=upenn_gbm
OUT=$DATA_ROOT/$NAME
URL=https://www.cancerimagingarchive.net/wp-content/uploads
# Public download link for the NIfTI package, from the collection page.
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjYwNCIsInBhc3Njb2RlIjoiYzJiMjI2Mzg5ZjljYWE0NWNkYjc4MzM4NWE4Yzc2MjBjNGU1NDY1MiIsInBhY2thZ2VfaWQiOiI2MDQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0='
FASPEX=(uv run --with requests python "$REPO/scripts/tcia_faspex.py")
NIFTI=UPENN-GBM/NIfTI-files
KEEP=(images_structural images_structural_unstripped automated_segm images_segm)

check_budget $((75 * 10**9))

# The Version 2 supporting files from the collection page (clinical + molecular labels,
# per-subject data availability, acquisition parameters, radiomic features, NBIA manifest).
# All plain HTTPS. Version 1-only files (clinical_info_v1.0, manifest_20210923,
# captk_parameter_file_ispy1) are left out.
mkdir -p "$OUT/source"
for f in \
    UPENN-GBM_clinical_info_v2.1.csv \
    UPENN-GBM_data_availability.csv \
    UPENN-GBM_acquisition.csv \
    radiology_mapping.csv \
    UPENN-GBM_CaPTk_radiomic_features_list.csv \
    UPENN-GBM_CaPTk_fe_params.csv \
    radiomic_features_CaPTk.zip \
    UPENN-GBM_DownloadManifest20221129.tcia \
    UPENN-GBM_DownloadManifest20221129-nbia-digest.xlsx
do
    [[ -e $OUT/source/$f ]] && continue
    curl -fsSL --retry 3 -o "$OUT/source/$f.part" "$URL/$f" && mv "$OUT/source/$f.part" "$OUT/source/$f"
done

# NIfTI package over FASP, keeping the package tree. Structural images (skull-stripped and
# unstripped) and both segmentation sets; DTI fits (images_DTI) and DSC perfusion
# (images_DSC) are left out. ascp skips complete files on a re-run.
"${FASPEX[@]}" get "$LINK" "$OUT/source" /UPENN-GBM.sums
"${FASPEX[@]}" get "$LINK" "$OUT/source/$NIFTI" "${KEEP[@]/#//$NIFTI/}"

# Verify the kept files against the package's md5 list (format: `md5 relpath`).
(cd "$OUT/source" && grep -E " $NIFTI/($(IFS='|'; echo "${KEEP[*]}"))/" UPENN-GBM.sums | md5sum -c --quiet)
write_manifest "$NAME"
