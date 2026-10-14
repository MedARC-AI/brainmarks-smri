#!/usr/bin/env bash
# Download FOMO26 finetuning Task 4 (ERDA share, Task_4.zip dated 2026-04-12) into
# datasets/fomo26_task4/. See README.md. Not redistributable: get the share link from the
# organizers and run it yourself:
#   FOMO26_URL=<share link> bash scripts/fomo26_task4/download.sh
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=fomo26_task4
OUT=$DATA_ROOT/$NAME
# The share is unversioned: the sha256 pins the bytes (first download, 2026-10-13).
SHA256=3b15b7d97ea6d1e889cab05a8c1c3912c976af19937d1f7a095dbedac7445fc0

[[ -n "${FOMO26_URL:-}" ]] || die "set FOMO26_URL to the FOMO26 share link from the organizers"
check_budget $((5 * 10**9))
if [[ ! -d "$OUT/source" ]]; then
    mkdir -p "$OUT"
    curl -fL -C - --retry 5 -sS -o "$OUT/Task_4.zip.part" "$FOMO26_URL/Task_4.zip"
    echo "$SHA256  $OUT/Task_4.zip.part" | sha256sum -c --quiet || die "Task_4.zip: checksum mismatch"
    mv "$OUT/Task_4.zip.part" "$OUT/Task_4.zip"
    rm -rf "$OUT/.tmp-source"
    mkdir "$OUT/.tmp-source"
    unzip -q "$OUT/Task_4.zip" -d "$OUT/.tmp-source"
    mv "$OUT/.tmp-source/Task_4" "$OUT/source"
    rmdir "$OUT/.tmp-source"
    rm "$OUT/Task_4.zip"
fi
write_manifest "$NAME"
