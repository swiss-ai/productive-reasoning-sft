#!/usr/bin/env bash
set -euo pipefail

: "${SFT_CONFIG:?SFT_CONFIG is required}"
: "${SFT_ENVIRONMENT:?SFT_ENVIRONMENT is required}"
: "${SFT_PROJECT_DIR:?SFT_PROJECT_DIR is required}"
: "${SFT_IMAGE:?SFT_IMAGE is required}"

mkdir -p "$RAY_PORT_BROADCAST_DIR"

srun \
  --ntasks-per-node=1 \
  --environment="$SFT_ENVIRONMENT" \
  bash -lc "cd '$SFT_PROJECT_DIR' && python -m synthetic_sft.cli run '$SFT_CONFIG'"

