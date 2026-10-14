#!/usr/bin/env bash
# Download OASIS-2 (longitudinal, download.nrg.wustl.edu, archives dated 2009-03) into
# datasets/oasis2/. See README.md. Not redistributable: accept the OASIS terms and run it yourself.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=oasis2
OUT=$DATA_ROOT/$NAME
BASE=https://download.nrg.wustl.edu/data
SHEET=https://sites.wustl.edu/oasisbrains/files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx

check_budget $((60 * 10**9))
mkdir -p "$OUT/source"
# No checksums are published: the sha256s below pin the bytes (first download, 2026-10-13).
declare -A PARTS=(
    [OAS2_RAW_PART1]=6df8fa570698c1f65769709275ca946f076838fb6b3c305c042ae6957897ac64
    [OAS2_RAW_PART2]=50bc39b6d911d7fca665bae496c54f92a18e9c96c548df3e5e35a9f763f0fd5f
)
# Each archive is downloaded next to source/, verified, extracted to source/<part>/, and deleted.
# The NIfTI pairs (.nifti.hdr/.img) are rewritten as .nii.gz with the same header and voxels.
for part in "${!PARTS[@]}"; do
    [[ -d "$OUT/source/$part" ]] && continue
    tarball=$OUT/$part.tar.gz
    curl -fL -C - --retry 5 -sS -o "$tarball" "$BASE/$part.tar.gz"
    echo "${PARTS[$part]}  $tarball" | sha256sum -c --quiet || die "$part: checksum mismatch"
    rm -rf "$OUT/source/.tmp-$part"
    mkdir "$OUT/source/.tmp-$part"
    tar -xzf "$tarball" -C "$OUT/source/.tmp-$part"
    uv run -q --project "$REPO" --inexact \
        python "$(dirname "$0")/convert_images.py" "$OUT/source/.tmp-$part"
    mv "$OUT/source/.tmp-$part/$part" "$OUT/source/$part"
    rmdir "$OUT/source/.tmp-$part"
    rm "$tarball"
done
# Demographics and clinical data per visit (group, age, MMSE, CDR, ...).
if [[ ! -s "$OUT/source/oasis_longitudinal_demographics.xlsx" ]]; then
    curl -fL --retry 5 -sS -o "$OUT/source/oasis_longitudinal_demographics.xlsx.part" "$SHEET"
    mv "$OUT/source/oasis_longitudinal_demographics.xlsx.part" "$OUT/source/oasis_longitudinal_demographics.xlsx"
fi
write_manifest "$NAME"
