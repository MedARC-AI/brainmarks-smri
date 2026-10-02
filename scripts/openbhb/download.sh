#!/usr/bin/env bash
# Download OpenBHB (HuggingFace benoit-dufumier/openBHB, commit 8508cda) into datasets/openbhb/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=openbhb
OUT=$DATA_ROOT/$NAME
REV=8508cda68fea74f217926acbf46ee5863f8879d1

# Original T1w (rawdata/, 32 GB), metadata tables, resource/ and the concatenated ROI
# feature CSVs. The preprocessed per-subject arrays (quasi-raw 230 GB, CAT12 VBM 68 GB,
# FreeSurfer xhemi 42 GB) are left out.
INCLUDES=(--include README.md --include '*.tsv' --include 'resource/*'
          --include '*/derivatives/*_roi/*' --include '*/rawdata/*')

check_budget $((35 * 10**9))
uvx --from huggingface_hub@2.1.1 hf download benoit-dufumier/openBHB --repo-type dataset \
    --revision "$REV" --local-dir "$OUT/source" "${INCLUDES[@]}" --format quiet
# hf keeps per-file download metadata (with timestamps) in source/.cache/. Drop it so
# the manifest only covers the release; a re-run re-hashes local files and skips them.
rm -rf "$OUT/source/.cache"
write_manifest "$NAME"
