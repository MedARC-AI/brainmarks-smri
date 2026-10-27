#!/usr/bin/env bash
# Build datasets/a4learn/source/ from your own A4/LEARN download (LONI IDA). See README.md.
# Not redistributable: apply for A4 access, then download from IDA the T1 image zips
# ("ImageData - T1 - NN of 16.zip") and the clinical data (Clinical.zip) into one folder, and the
# documentation (data dictionaries, guides, methods) into another, and run
#   A4_ZIPS=<zip folder> A4_DOCS=<documentation folder> bash scripts/a4learn/download.sh
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=a4learn
OUT=$DATA_ROOT/$NAME

[[ -d "${A4_ZIPS:-}" ]] || die "set A4_ZIPS to the folder of IDA zips"
[[ -d "${A4_DOCS:-}/Data Dictionaries" ]] || die "set A4_DOCS to the documentation folder"
mkdir -p "$OUT/source"
# The T1w images (NIfTI + JSON) of the sessions in selected_images.tsv (bl, wk84, wk240).
cp "$(dirname "$0")/selected_images.tsv" "$OUT/source/"
WANTED=$(mktemp)
tail -n +2 "$OUT/source/selected_images.tsv" | cut -f5 | sed 'p; s/\.nii\.gz$/.json/' > "$WANTED"
for zip in "$A4_ZIPS"/ImageData\ -\ T1\ -\ *.zip; do
    unzip -Z1 "$zip" | grep -F -x -f "$WANTED" | xargs -r -d '\n' unzip -n -q "$zip" -d "$OUT/source"
done
rm "$WANTED"
n_images=$(find "$OUT/source/ImageData" -name '*.nii.gz' | wc -l)
(( n_images == $(tail -n +2 "$OUT/source/selected_images.tsv" | wc -l) )) || die "found $n_images images"
# Clinical data, without the safety ECG and lab tables (clrm_ecg.csv, clrm_lab.csv; 270 MB).
unzip -n -q "$A4_ZIPS/Clinical.zip" -d "$OUT/source" -x 'Clinical/External Data/clrm_*'
mkdir -p "$OUT/source/Clinical/Documents"
cp -r "$A4_DOCS"/. "$OUT/source/Clinical/Documents/"
write_manifest "$NAME"
