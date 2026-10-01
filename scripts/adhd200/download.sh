#!/usr/bin/env bash
# Download ADHD-200 (INDI s3://fcp-indi/data/Projects/ADHD200, unversioned; S3 listing
# 2026-10-01) into datasets/adhd200/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=adhd200
OUT=$DATA_ROOT/$NAME
S3=s3://fcp-indi/data/Projects/ADHD200
NITRC=https://fcon_1000.projects.nitrc.org/indi/adhd200

check_budget $((10 * 10**9))
# BIDS release, structural only: T1w + sidecars, participants.tsv, phenotypic CSVs.
# Skip the rs-fMRI (func/, *_bold.json; 79 GB) and macOS .DS_Store junk.
aws s3 sync --no-sign-request --only-show-errors "$S3/RawDataBIDS/" "$OUT/source/RawDataBIDS/" \
    --exclude '*/func/*' --exclude '*_bold.json' --exclude '*.DS_Store'
# The original (pre-BIDS) phenotypic CSVs only; RawData/ images duplicate the BIDS ones.
# These include NYU/NeuroIMAGE/WashU CSVs that RawDataBIDS/ lacks.
aws s3 sync --no-sign-request --only-show-errors "$S3/RawData/" "$OUT/source/RawData/" \
    --exclude '*' --include '*_phenotypic.csv'
# Docs and released test-set labels, public (no login) on the NITRC project site.
mkdir -p "$OUT/source/nitrc"
for f in json/intro.json json/sites.json results.html general/ADHD-200_PhenotypicKey.pdf \
         general/allSubs_testSet_phenotypic_dx.csv general/ADHD-200_CompetitionScoring.pdf \
         fixes/ADHD-200.PhenotypicFix.csv fixes/DeobliqueFixAffectedSubs.txt; do
    dest=$OUT/source/nitrc/$f
    [[ -s $dest ]] || { curl -fsSL --create-dirs -o "$dest.part" "$NITRC/$f" && mv "$dest.part" "$dest"; }
done
write_manifest "$NAME"
