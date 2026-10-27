#!/usr/bin/env bash
# Upload the datasets to the Hugging Face mirror, with README_hf.md as the dataset card.
# Usage: bash scripts/upload.sh
# Hosted datasets: README.md, tables/ and shards/. Manual datasets: README.md and the tables
# tracked in scripts/<name>/tables/ (no samples.tsv). FOMO300K: README.md, tables/ and shards/.
# Resumable: re-run the same command after an interruption.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

REPO_ID=medarc/brainmarks-smri
HF=(uvx --from huggingface_hub@2.1.1 hf)
HOSTED=(abide1 adhd200 brats2021 cnp ixi openbhb soop ucsf_pdgm upenn_gbm)
MANUAL=(a4learn adni atlas fomo26_task4 fomo26_task5 oasis1 oasis2)

for name in "${HOSTED[@]}"; do
    diff <(tail -n +2 "$DATA_ROOT/$name/tables/shards.tsv" | cut -f2 | sort -u) \
        <(cd "$DATA_ROOT/$name" && ls shards/*.zip) > /dev/null ||
        die "$name: shards/ does not match tables/shards.tsv"
done

# Public repo, gated (automatic approval) before any data goes up.
"${HF[@]}" repos create "$REPO_ID" --repo-type dataset --exist-ok
"${HF[@]}" repos settings "$REPO_ID" --repo-type dataset --gated auto
"${HF[@]}" upload "$REPO_ID" "$REPO/README_hf.md" README.md --repo-type dataset

for name in "${HOSTED[@]}" "${MANUAL[@]}" fomo300k; do
    "${HF[@]}" upload "$REPO_ID" "$REPO/scripts/$name" "$name" --repo-type dataset \
        --include README.md --include summary.png --include "tables/*"
done
for name in "${HOSTED[@]}" fomo300k; do
    "${HF[@]}" upload "$REPO_ID" "$DATA_ROOT/$name/shards" "$name/shards" --repo-type dataset
done
"${HF[@]}" upload "$REPO_ID" "$DATA_ROOT/fomo300k/tables" fomo300k/tables --repo-type dataset
