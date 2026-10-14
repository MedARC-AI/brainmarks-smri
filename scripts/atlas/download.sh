#!/usr/bin/env bash
# Download ATLAS R2.1 (INDI S3, minimally processed, 2026-04-22) into datasets/atlas/. See README.md.
# Not redistributable: apply for the decryption key yourself (see README.md), then run
#   ATLAS_KEY=<key> bash scripts/atlas/download.sh
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=atlas
OUT=$DATA_ROOT/$NAME
FILE=atlas21_training_raw.tar.gz
URL=https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/ATLAS/R2.1/$FILE
SHA256=dd22217c150011f913c69ad8cde36abddea6e5c16b0ff9b161e079fab9ed76f0

[[ -n "${ATLAS_KEY:-}" ]] || die "set ATLAS_KEY to the key you received from the ATLAS team"
check_budget $((10 * 10**9))
# The fully normalized version (atlas21_training_preprocessed.tar.gz, 9.1 GiB) is left out.
if [[ ! -d "$OUT/source" ]]; then
    mkdir -p "$OUT"
    if [[ ! -f "$OUT/$FILE" ]]; then
        curl -fL -C - --retry 5 -sS -o "$OUT/$FILE.part" "$URL"
        echo "$SHA256  $OUT/$FILE.part" | sha256sum -c --quiet || die "$FILE: checksum mismatch"
        mv "$OUT/$FILE.part" "$OUT/$FILE"
    fi
    rm -rf "$OUT/.tmp-source"
    mkdir "$OUT/.tmp-source"
    openssl aes-256-cbc -d -a -md sha256 -pass env:ATLAS_KEY -in "$OUT/$FILE" |
        tar -xz --exclude=.DS_Store -C "$OUT/.tmp-source"
    mv "$OUT/.tmp-source" "$OUT/source"
    rm "$OUT/$FILE"
fi
write_manifest "$NAME"
