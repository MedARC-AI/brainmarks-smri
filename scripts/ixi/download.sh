#!/usr/bin/env bash
# Download IXI (brain-development.org, files dated 2014-10-07) into datasets/ixi/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=ixi
OUT=$DATA_ROOT/$NAME
BASE=https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI

check_budget $((18 * 10**9))
mkdir -p "$OUT/source/docs"
# No versioning or checksums are published: README.md records Last-Modified/ETag, and the
# tarball sha256s below pin the bytes (computed at the first download, 2026-10-01).
# MRA (IXI-MRA.tar, 12.4 GB) is excluded: angiography, not used by brain-age benchmarks.
declare -A TARS=(
    [IXI-T1]=56bbe911fbe967f7d1b830a552995263bdba281e0a6eb78e62f1c38d88a946a9
    [IXI-T2]=39af50f87be8f7d0be2f166a64f5f59b5e43577255eb5613553dae37c348c2e4
    [IXI-PD]=c00124d2409b68f9800938ede547cff71cafb0d51e19bd5138d85a4c4e144130
    [IXI-DTI]=ddffe4189bd159b0be42d0472b9ea17f9a72f21296ebc85f382f3fa3ef55d15f
)
# Each tarball is downloaded next to source/, verified, extracted to source/<name>/, and deleted.
for name in "${!TARS[@]}"; do
    [[ -d "$OUT/source/$name" ]] && continue
    curl -fL -C - --retry 5 -sS -o "$OUT/$name.tar" "$BASE/$name.tar"
    echo "${TARS[$name]}  $OUT/$name.tar" | sha256sum -c --quiet || die "$name.tar: checksum mismatch"
    rm -rf "$OUT/source/.tmp-$name"
    mkdir "$OUT/source/.tmp-$name"
    tar -xf "$OUT/$name.tar" -C "$OUT/source/.tmp-$name"
    mv "$OUT/source/.tmp-$name" "$OUT/source/$name"
    rm "$OUT/$name.tar"
done
for f in bvals.txt bvecs.txt IXI.xls marital.xls; do
    curl -fL -C - --retry 5 -sS -o "$OUT/source/$f" "$BASE/$f"
done
# Documentation pages (dataset page with license, scanner parameters). They're dynamic
# WordPress pages, so fetch once only; re-fetching would churn the manifest.
for page in ixi-dataset scanner-philips-medical-systems-intera-3t scanner-philips-medical-systems-gyroscan-intera-1-5t; do
    [[ -s "$OUT/source/docs/$page.html" ]] ||
        curl -fL -sS -o "$OUT/source/docs/$page.html" "https://brain-development.org/$page/"
done
write_manifest "$NAME"
