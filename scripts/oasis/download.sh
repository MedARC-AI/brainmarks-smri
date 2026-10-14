#!/usr/bin/env bash
# Download OASIS-1 (cross-sectional, 2007) and OASIS-2 (longitudinal, 2009) into
# datasets/oasis/source/oasis1/ and source/oasis2/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=oasis
OUT=$DATA_ROOT/$NAME
BASE=https://download.nrg.wustl.edu/data
SITE=https://sites.wustl.edu/oasisbrains

# Peak: the extracted source (91 GB) plus 4 tarballs in flight (up to 10.9 GB each).
check_budget $((140 * 10**9))
mkdir -p "$OUT/source/oasis1" "$OUT/source/oasis2"
# No checksums are published: README.md records Last-Modified/ETag, and the tarball sha256s
# below pin the bytes (computed at the first download, 2026-10-09).
# OASIS-1 image discs: each tarball holds disc<N>/OAS1_xxxx_MRy/, extracted to source/oasis1/disc<N>/.
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
# OASIS-2: each tarball holds <name>/OAS2_xxxx_MRy/RAW/, extracted to source/oasis2/<name>/.
declare -A PARTS=(
    [OAS2_RAW_PART1]=6df8fa570698c1f65769709275ca946f076838fb6b3c305c042ae6957897ac64
    [OAS2_RAW_PART2]=50bc39b6d911d7fca665bae496c54f92a18e9c96c548df3e5e35a9f763f0fd5f
)

# fetch_tar <url> <sha256> <dest>: download next to source/, verify, extract to <dest>/, delete.
# Each tarball has a single top-level folder, which becomes <dest>.
fetch_tar() {
    local url=$1 sha=$2 dest=$3 tar=$OUT/${1##*/} tmp
    [[ -d $dest ]] && return
    curl -fL -C - --retry 5 -sS -o "$tar" "$url"
    echo "$sha  $tar" | sha256sum -c --quiet || die "${url##*/}: checksum mismatch"
    tmp=$(dirname "$dest")/.tmp-$(basename "$dest")
    rm -rf "$tmp"
    mkdir -p "$tmp"
    tar -xzf "$tar" -C "$tmp"
    mv "$tmp"/* "$dest"
    rmdir "$tmp"
    rm "$tar"
}
# The server gives each connection ~10 MB/s, so fetch 4 tarballs at a time.
export OUT
export -f fetch_tar die
{
    for disc in "${!DISCS[@]}"; do
        echo "$BASE/oasis_cross-sectional_$disc.tar.gz ${DISCS[$disc]} $OUT/source/oasis1/$disc"
    done
    for part in "${!PARTS[@]}"; do
        echo "$BASE/$part.tar.gz ${PARTS[$part]} $OUT/source/oasis2/$part"
    done
} | xargs -n 3 -P 4 bash -c 'set -euo pipefail; fetch_tar "$@"' _
# Demographic and clinical spreadsheets, saved without the WordPress hash suffix.
declare -A FILES=(
    [oasis1/oasis_cross-sectional.xlsx]=files/2024/04/oasis_cross-sectional-5708aa0a98d82080.xlsx
    [oasis2/oasis_longitudinal_demographics.xlsx]=files/2024/03/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx
)
for f in "${!FILES[@]}"; do
    curl -fL --retry 5 -sS -o "$OUT/source/$f" "$SITE/${FILES[$f]}"
done
write_manifest "$NAME"
