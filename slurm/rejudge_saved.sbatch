#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SFT_IMAGE:?SFT_IMAGE is required}"
: "${SFT_PROJECT_DIR:?SFT_PROJECT_DIR is required}"
: "${SFT_CONFIG:?SFT_CONFIG is required}"
: "${SFT_SAVED_CANDIDATES:?SFT_SAVED_CANDIDATES is required}"
: "${SFT_REJUDGE_OUTPUT:?SFT_REJUDGE_OUTPUT is required}"

srun --environment="$SFT_PROJECT_DIR/container/cscs.toml" \
  bash -lc "cd '$SFT_PROJECT_DIR' && python scripts/rejudge_saved.py \
    '$SFT_CONFIG' '$SFT_SAVED_CANDIDATES' '$SFT_REJUDGE_OUTPUT' \
    '${SFT_CALIBRATION_CASES:-}'"
