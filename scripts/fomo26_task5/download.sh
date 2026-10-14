#!/usr/bin/env bash
# Download FOMO26 finetuning Task 5 (ERDA share, files dated 2026-05-01) into
# datasets/fomo26_task5/. See README.md. Not redistributable: get the share link from the
# organizers and run it yourself:
#   FOMO26_URL=<share link> bash scripts/fomo26_task5/download.sh
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=fomo26_task5
OUT=$DATA_ROOT/$NAME

[[ -n "${FOMO26_URL:-}" ]] || die "set FOMO26_URL to the FOMO26 share link from the organizers"
check_budget $((3 * 10**9))
# The share is unversioned: the sha256s below pin the bytes (first download, 2026-10-13).
declare -A FILES=(
    [Task_5_extract.py]=fbcce4ce2b7fdf1a1efd6da24fce9c57c2437ae7aba36108f3353b5c7b6ec378
    [Zhang_Lingfeng_2022_PPMR_Dataset.zip]=cddcc3796136238da27b76fe8a082798f4ce1278171cd81e900e80c5932c0517
)
# The organizers ship the source zip (coronal JPEG slices) and a script that converts it to
# NIfTI. We run the script, keep its output and drop the zip.
if [[ ! -d "$OUT/source" ]]; then
    WORK=$OUT/.tmp-source
    rm -rf "$WORK"
    mkdir -p "$WORK"
    for file in "${!FILES[@]}"; do
        curl -fL --retry 5 -sS -o "$WORK/$file" "$FOMO26_URL/$file"
        echo "${FILES[$file]}  $WORK/$file" | sha256sum -c --quiet || die "$file: checksum mismatch"
    done
    (cd "$WORK" && uv run -q --project "$REPO" --inexact python -I Task_5_extract.py)
    mv "$WORK/Task_5_extract.py" "$WORK/Task_5/"
    mv "$WORK/Task_5" "$OUT/source"
    rm -rf "$WORK"
fi
write_manifest "$NAME"
