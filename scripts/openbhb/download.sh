#!/usr/bin/env bash
# Download OpenBHB (HuggingFace benoit-dufumier/openBHB, commit 8508cda) into datasets/openbhb/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=openbhb
OUT=$DATA_ROOT/$NAME
REV=8508cda68fea74f217926acbf46ee5863f8879d1

# Default: original T1w (rawdata/, 32 GB), metadata tables, resource/ and the
# concatenated ROI feature CSVs. The preprocessed 3D arrays are left out: quasi-raw
# (230 GB, over the 200 GB check-with-user threshold), CAT12 VBM (68 GB), FreeSurfer
# xhemi (42 GB). Set OPENBHB_QUASIRAW=1 to also fetch the quasi-raw arrays.
INCLUDES=(--include README.md --include '*.tsv' --include 'resource/*'
          --include '*/derivatives/*_roi/*' --include '*/rawdata/*')
NEED=$((35 * 10**9))
if [[ ${OPENBHB_QUASIRAW:-0} == 1 ]]; then
    INCLUDES+=(--include '*_preproc-quasiraw_T1w.npy')
    NEED=$((NEED + 235 * 10**9))
fi

check_budget $NEED
uvx --from huggingface_hub@2.1.1 hf download benoit-dufumier/openBHB --repo-type dataset \
    --revision "$REV" --local-dir "$OUT/source" "${INCLUDES[@]}" --format quiet
# hf keeps per-file download metadata (with timestamps) in source/.cache/. Drop it so
# the manifest only covers the release; a re-run re-hashes local files and skips them.
rm -rf "$OUT/source/.cache"
write_manifest "$NAME"
