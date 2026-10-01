#!/usr/bin/env bash
# Download UPENN-GBM (TCIA collection, Version 2, 2022/10/24) supporting data into
# datasets/upenn_gbm/. Imaging is NOT fetched yet; see README.md ("Imaging: set aside").
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=upenn_gbm
OUT=$DATA_ROOT/$NAME
URL=https://www.cancerimagingarchive.net/wp-content/uploads

check_budget $((100 * 10**6))
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
write_manifest "$NAME"
