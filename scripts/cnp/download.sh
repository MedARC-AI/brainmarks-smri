#!/usr/bin/env bash
# Download CNP / LA5c (OpenNeuro ds000030, snapshot 1.0.0) into datasets/cnp/. See README.md.
set -euo pipefail
source "$(dirname "$0")/../lib.sh"

NAME=cnp
OUT=$DATA_ROOT/$NAME

check_budget $((15 * 10**9))
# Structural + DWI only: skip fMRI (func/, task-*_bold.json) and the per-subject
# stop-signal training behavioural logs (beh/, task-* files). The snapshot has no derivatives/.
uvx openneuro-py@2026.9.1 download --dataset ds000030 --tag 1.0.0 --target-dir "$OUT/source" \
    --exclude 'sub-*/func' --exclude 'sub-*/beh' --exclude 'task-*'
write_manifest "$NAME"
