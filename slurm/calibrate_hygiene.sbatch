#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SFT_IMAGE:?SFT_IMAGE is required}"
: "${SFT_PROJECT_DIR:?SFT_PROJECT_DIR is required}"
: "${SFT_CONFIG:?SFT_CONFIG is required}"
: "${SFT_CALIBRATION_CASES:?SFT_CALIBRATION_CASES is required}"
: "${SFT_CALIBRATION_OUTPUT:?SFT_CALIBRATION_OUTPUT is required}"

srun --environment="$SFT_PROJECT_DIR/container/cscs.toml" \
  bash -lc "cd '$SFT_PROJECT_DIR' && python scripts/calibrate_hygiene.py \
    '$SFT_CONFIG' '$SFT_CALIBRATION_CASES' '$SFT_CALIBRATION_OUTPUT'"
