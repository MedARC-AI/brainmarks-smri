#!/usr/bin/env bash
# Download BraTS 2021 (TCIA analysis result RSNA-ASNR-MICCAI-BraTS-2021, Version 1, 2023/08/25)
# into datasets/brats2021/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=brats2021
OUT=$DATA_ROOT/$NAME
URL=https://www.cancerimagingarchive.net/wp-content/uploads

check_budget $((10 * 10**6))
# Everything linked from the TCIA page as a plain file (~2 MB): the BraTS<->TCIA ID crosswalk
# (which also holds the Task 1/Task 2 cohorts and the MGMT labels), the list of BraTS files with
# no TCIA DICOM equivalent, and the NBIA/CRDC manifests of the original DICOM series.
FILES=(
    BraTS2021_MappingToTCIA.xlsx
    NotPreviouslyInTCIA.csv
    GC_manifest_RSNA-ASNR-MICCAI-BRATS-2021_sources.csv
    RSNA-ASNR-MICCAI-BraTS-2021_UPENN-GBM_manifest.tcia
    RSNA-ASNR-MICCAI-BraTS-2021_UPENN-GBM_manifes-nbia-digest-1.xlsx
)
for src in ACRIN-FMISO-Brain CPTAC-GBM IvyGAP TCGA-GBM TCGA-LGG TCIAderived UPENN-GBM; do
    for task in Seg Class; do
        for split in Training Validation; do
            # Not every source has every task/split.
            case $src-$task-$split in ACRIN-FMISO-Brain-*-Validation|ACRIN-FMISO-Brain-Class-*|TCGA-LGG-Class-*) continue ;; esac
            FILES+=("BraTS2021_${src}_${task}-Task-${split}.tcia")
        done
    done
done
mkdir -p "$OUT/source/metadata"
for f in "${FILES[@]}"; do
    dst=$OUT/source/metadata/$f
    [[ -s $dst ]] && continue
    curl -fsSL --retry 5 --retry-all-errors -o "$dst.part" "$URL/$f" && mv "$dst.part" "$dst"
    sleep 1  # be polite to TCIA
done

# Imaging: Faspex package 636, only distributed over Aspera FASP. Public link = "Challenge data
# both tasks" on the TCIA page. Fetch the Task 1 NIfTI (training + validation, 15.8 GB) and the
# package's md5 list; skip the Task 2 `_dcm` folders (~127 GB). Paths keep the package tree.
LINK='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjYzNiIsInBhc3Njb2RlIjoiNDM5YTVhZjM3NGRhYjk3OGExYjExMzA4MTcyZDhlMDdkY2Q5OWMzMSIsInBhY2thZ2VfaWQiOiI2MzYiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0='
PKG=RSNA-ASNR-MICCAI-BraTS-2021
faspex() { uv run --project "$REPO" python -m brainmarks_smri.utils.tcia_faspex get "$LINK" "$@"; }
check_budget $((17 * 10**9))
faspex "$OUT/source" "/$PKG.sums"
faspex "$OUT/source/$PKG" "/$PKG/BraTS2021_TrainingSet" "/$PKG/BraTS2021_ValidationSet"
# Verify the NIfTI against the package md5 list (paths in .sums are relative to source/).
(cd "$OUT/source" && grep -E " $PKG/BraTS2021_(Training|Validation)Set/" "$PKG.sums" | md5sum -c --quiet)
write_manifest "$NAME"
