#!/usr/bin/env bash
# Download IXI (brain-development.org, files dated 2014-10-07) into datasets/ixi/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=ixi
OUT=$DATA_ROOT/$NAME
BASE=https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI

check_budget $((18 * 10**9))
mkdir -p "$OUT/source/docs"
# Tarballs kept as distributed (not unpacked). No versioning or checksums are published;
# README.md records Last-Modified/ETag, and the manifest pins the bytes.
# MRA (IXI-MRA.tar, 12.4 GB) is excluded: angiography, not used by brain-age benchmarks.
for f in IXI-T1.tar IXI-T2.tar IXI-PD.tar IXI-DTI.tar bvals.txt bvecs.txt IXI.xls marital.xls; do
    curl -fL -C - --retry 5 -sS -o "$OUT/source/$f" "$BASE/$f"
done
# Documentation pages (dataset page with license, scanner parameters). They're dynamic
# WordPress pages, so fetch once only; re-fetching would churn the manifest.
for page in ixi-dataset scanner-philips-medical-systems-intera-3t scanner-philips-medical-systems-gyroscan-intera-1-5t; do
    [[ -s "$OUT/source/docs/$page.html" ]] ||
        curl -fL -sS -o "$OUT/source/docs/$page.html" "https://brain-development.org/$page/"
done
write_manifest "$NAME"
