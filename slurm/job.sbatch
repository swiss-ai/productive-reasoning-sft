#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

: "${SFT_CONFIG:?SFT_CONFIG is required}"
: "${SFT_ENVIRONMENT:?SFT_ENVIRONMENT is required}"
: "${SFT_PROJECT_DIR:?SFT_PROJECT_DIR is required}"
: "${SFT_IMAGE:?SFT_IMAGE is required}"

mkdir -p "$RAY_PORT_BROADCAST_DIR"

requeue_before_timeout() {
  trap - USR1
  echo "Received the pre-timeout signal; requeueing job $SLURM_JOB_ID for a resumable continuation."
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_timeout USR1

watchdog_pid=""
if [[ -n "${SFT_REQUEUE_AFTER_SECONDS:-}" ]]; then
  (
    sleep "$SFT_REQUEUE_AFTER_SECONDS"
    kill -USR1 "$$"
  ) &
  watchdog_pid=$!
fi

srun \
  --ntasks-per-node=1 \
  --environment="$SFT_ENVIRONMENT" \
  bash -lc "cd '$SFT_PROJECT_DIR' && python -m synthetic_sft.cli run '$SFT_CONFIG'" &
srun_pid=$!
set +e
wait "$srun_pid"
srun_status=$?
set -e

if [[ -n "$watchdog_pid" ]]; then
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
fi
exit "$srun_status"
