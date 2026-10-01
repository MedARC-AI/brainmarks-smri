#!/usr/bin/env bash
# Download ABIDE I (INDI s3://fcp-indi/data/Projects/ABIDE, RawDataBIDS as listed 2026-10-01)
# into datasets/abide1/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=abide1
OUT=$DATA_ROOT/$NAME
S3=s3://fcp-indi/data/Projects/ABIDE

check_budget $((9 * 10**9))
# The bucket is the release and has no version tags (images last modified 2016-12-16,
# sidecards/ 2022-04-13); the manifest is the version record.
# Structural only, from the BIDS form (RawData/ holds the same T1w images in the old layout):
# per-site metadata, sub-*/anat/ T1w, and per-subject T1w sidecars from sidecards/.
# Skips rs-fMRI (func/, task-rest_bold.json) and .DS_Store files.
aws s3 sync --no-sign-request --only-show-errors "$S3/RawDataBIDS/" "$OUT/source/RawDataBIDS" \
    --exclude '*' \
    --include '*/dataset_description.json' \
    --include '*/participants.tsv' \
    --include '*/T1w.json' \
    --include '*/sub-*/anat/*'
# Composite phenotypic file (the "Composite Phenotypic File" on the ABIDE I page) and its
# legend. Single files, so fetch only if missing. (Syncing ABIDE/ with filters would list
# the huge Derivatives/ tree.)
[[ -f "$OUT/source/Phenotypic_V1_0b.csv" ]] ||
    aws s3 cp --no-sign-request --only-show-errors "$S3/Phenotypic_V1_0b.csv" "$OUT/source/"
[[ -f "$OUT/source/ABIDE_LEGEND_V1.02.pdf" ]] ||
    { curl -fsSL -o "$OUT/source/ABIDE_LEGEND_V1.02.pdf.part" \
        https://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf &&
      mv "$OUT/source/ABIDE_LEGEND_V1.02.pdf.part" "$OUT/source/ABIDE_LEGEND_V1.02.pdf"; }
write_manifest "$NAME"
