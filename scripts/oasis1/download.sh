#!/usr/bin/env bash
# Download OASIS-1 (cross-sectional, download.nrg.wustl.edu, discs dated 2007-06) into
# datasets/oasis1/. See README.md. Not redistributable: accept the OASIS terms and run it yourself.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=oasis1
OUT=$DATA_ROOT/$NAME
BASE=https://download.nrg.wustl.edu/data
SHEETS=https://sites.wustl.edu/oasisbrains/files/2024

check_budget $((52 * 10**9))
mkdir -p "$OUT/source"
# No checksums are published: the disc sha256s below pin the bytes (first download, 2026-10-13).
# The FreeSurfer discs (oasis_cs_freesurfer_disc*.tar.gz, ~105 GB) are left out.
declare -A DISCS=(
    [disc1]=dba4e59335b9df889ad9dade2a560af6cf87e2000012c70b438b585ca7d95073
    [disc2]=6b65979d07b76e906eb2d81ee84d834844061461396f1c9fc7d284a9c03bed5b
    [disc3]=7a9eba83b4a155262a938a47330d99c907e070ee845d0ad664cb9d66db62375f
    [disc4]=466b1a873a2f4a3d98f3901c7d06a459d65f674b2a996de9919baebeaa965ba7
    [disc5]=78d77f200bd55dda1756b6b2313c787ab0c4e9a065768f5bd15811a6d9d5b9b3
    [disc6]=ff4403850c3119e94559e2b4c803b9c69a7dcf31fccef2168c0d35f480adc08e
    [disc7]=cc2635f0864d57c417a115f7ec4d8c00b94f2b00a10bba43b49e2f9f67086cee
    [disc8]=394e38f15dd14f721385d5a5ee694ed47ed0447e7727bd49885780b0d7c2a67b
    [disc9]=a31876e00322e5e9a9bede9053c49626fed56aa9a72f97aa3b370f4d1e5d20a7
    [disc10]=24bece6af5b7076c3da8a4dceff18ab3cafaa13d07fc5ce6ae419b42bf25edc6
    [disc11]=3e3023b7a229fcc04b830acb7ffd0a2625c15ed3183b2623f429c0e95122b12e
    [disc12]=b0189de4bd922148a7a7956d8d3056278f0420f523a264d7a5cf799640cacacd
)
# Each disc is downloaded next to source/, verified, extracted to source/discN/, and deleted.
# The Analyze images (.hdr/.img) are converted to .nii.gz with the same voxels (see convert_images.py).
for disc in "${!DISCS[@]}"; do
    [[ -d "$OUT/source/$disc" ]] && continue
    tarball=$OUT/oasis_cross-sectional_$disc.tar.gz
    curl -fL -C - --retry 5 -sS -o "$tarball" "$BASE/oasis_cross-sectional_$disc.tar.gz"
    echo "${DISCS[$disc]}  $tarball" | sha256sum -c --quiet || die "$disc: checksum mismatch"
    rm -rf "$OUT/source/.tmp-$disc"
    mkdir "$OUT/source/.tmp-$disc"
    tar -xzf "$tarball" -C "$OUT/source/.tmp-$disc"
    uv run -q --project "$REPO" --inexact \
        python "$(dirname "$0")/convert_images.py" "$OUT/source/.tmp-$disc"
    mv "$OUT/source/.tmp-$disc/$disc" "$OUT/source/$disc"
    rmdir "$OUT/source/.tmp-$disc"
    rm "$tarball"
done
# Clinical data (age, sex, MMSE, CDR, ...), the reliability repeats, and the fact sheet.
declare -A FILES=(
    [oasis_cross-sectional.xlsx]=04/oasis_cross-sectional-5708aa0a98d82080.xlsx
    [oasis_cross-sectional-reliability.xlsx]=04/oasis_cross-sectional-reliability-063c8642b909ee76.xlsx
    [oasis_cross-sectional_facts.pdf]=03/oasis_cross-sectional_facts-bcc7a002dfb104f4.pdf
)
for file in "${!FILES[@]}"; do
    [[ -s "$OUT/source/$file" ]] && continue
    curl -fL --retry 5 -sS -o "$OUT/source/$file.part" "$SHEETS/${FILES[$file]}"
    mv "$OUT/source/$file.part" "$OUT/source/$file"
done
write_manifest "$NAME"
