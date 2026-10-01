#!/usr/bin/env bash
# Download Pixar (OpenNeuro ds000228, snapshot 1.1.1) into datasets/pixar/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=pixar
OUT=$DATA_ROOT/$NAME

check_budget $((2 * 10**9))
# Note: the public S3 mirror (s3://openneuro.org/ds000228) is stale. Its metadata files
# date from 1.1.0, so we use the official client, which fetches the tagged snapshot.
# Structural only: skip the fMRI (func/, task-*_bold.json) and derivatives/ (fMRI
# preprocessing and MRIQC outputs).
uvx openneuro-py@2026.9.1 download --dataset ds000228 --tag 1.1.1 --target-dir "$OUT/source" \
    --exclude 'sub-*/func' --exclude 'task-*' --exclude derivatives
write_manifest "$NAME"
