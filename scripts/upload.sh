#!/usr/bin/env bash
# Upload datasets/ to the Hugging Face mirror, with README_hf.md as the dataset card.
# Usage: bash scripts/upload.sh
# Checks that every dataset matches its tracked README, tables and manifest first.
# Resumable: re-run the same command after an interruption.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

REPO_ID=medarc/brainmarks-smri
HF=(uvx --from huggingface_hub@2.1.1 hf)

for dir in "$DATA_ROOT"/*/; do
    name=$(basename "$dir")
    echo "checking $name"
    cmp -s "$REPO/scripts/$name/README.md" "$dir/README.md" || die "$name: README.md differs from scripts/$name/"
    diff -r "$REPO/scripts/$name/tables" "$dir/tables" > /dev/null || die "$name: tables/ differ from scripts/$name/tables/"
    cmp -s "$REPO/scripts/$name/manifest.sha256" "$dir/manifest.sha256" || die "$name: manifest differs from scripts/$name/"
    (cd "$dir" && sha256sum -c --quiet manifest.sha256) || die "$name: manifest check failed"
done
cp "$REPO/README_hf.md" "$DATA_ROOT/README.md"

# Public repo, gated (automatic approval) before any data goes up.
"${HF[@]}" repos create "$REPO_ID" --repo-type dataset --exist-ok
"${HF[@]}" repos settings "$REPO_ID" --repo-type dataset --gated auto
"${HF[@]}" upload "$REPO_ID" "$DATA_ROOT" . --repo-type dataset
