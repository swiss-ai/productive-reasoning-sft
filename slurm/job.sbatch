#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SFT_CONFIG:?SFT_CONFIG is required}"
: "${SFT_ENVIRONMENT:?SFT_ENVIRONMENT is required}"
: "${SFT_PROJECT_DIR:?SFT_PROJECT_DIR is required}"
: "${SFT_IMAGE:?SFT_IMAGE is required}"

mkdir -p "$RAY_PORT_BROADCAST_DIR"

requeue_before_timeout() {
  echo "Received the pre-timeout signal; requeueing job $SLURM_JOB_ID for a resumable continuation."
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1

srun \
  --ntasks-per-node=1 \
  --environment="$SFT_ENVIRONMENT" \
  bash -lc "cd '$SFT_PROJECT_DIR' && python -m synthetic_sft.cli run '$SFT_CONFIG'" &
srun_pid=$!
wait "$srun_pid"
