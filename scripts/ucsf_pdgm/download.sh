#!/usr/bin/env bash
# Download UCSF-PDGM (TCIA, collection version 5, 2025-05-30) into datasets/ucsf_pdgm/. See README.md.
#
# PARTIAL: only the clinical metadata is fetched. The images (142 GB NIfTI) are only
# distributed as an Aspera Faspex package, and Aspera is blocked here. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=ucsf_pdgm
OUT=$DATA_ROOT/$NAME
URL=https://www.cancerimagingarchive.net/wp-content/uploads

check_budget $((1 * 10**6))
# Version 5 clinical table (labels + BraTS 2021 ID mapping) and its glossary, as linked
# from the collection page's v5 "Data Access" table. The DTI bval/bvec zips are skipped
# because we don't fetch the raw 4D DTI they describe.
mkdir -p "$OUT/source"
for f in UCSF-PDGM-metadata_v5.csv UCSF-PDGM-metadata_glossary.csv; do
    [[ -s $OUT/source/$f ]] && continue
    curl -fsSL -o "$OUT/source/$f.part" "$URL/$f" && mv "$OUT/source/$f.part" "$OUT/source/$f"
done
write_manifest "$NAME"
echo "warning: images not fetched (Aspera only; see scripts/$NAME/README.md)" >&2
