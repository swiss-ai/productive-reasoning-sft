#!/bin/bash
# Run the SML service and the CPU pipeline client in one bounded allocation.
set -euo pipefail
ulimit -c 0
config=$1
master=$2
seeds=$3
run_dir=$4
samples=$5
inflight=$6
prefill=$7
cd "$SFT_PROJECT_DIR"
mkdir -p "$run_dir/logs"
bash "$master" &
serving_pid=$!
client_pid=""
cleanup() {
    if [[ -n "$client_pid" ]]; then kill "$client_pid" 2>/dev/null || true; fi
    kill "$serving_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT
sml_head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
sml_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$sml_head" hostname -i)
export SFT_API_BASE="http://$sml_ip:8080/v1"
client_step=(srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --cpu-bind=none
    --nodelist="$sml_head" --environment="$SFT_CLIENT_ENVIRONMENT")
run_client() {
    "${client_step[@]}" python scripts/probe_endpoint.py "$config" "$run_dir/probes" \
        --wait-seconds 3300
    "${client_step[@]}" python scripts/benchmark_pipeline.py "$config" "$seeds" \
        "$run_dir/benchmark" --execution endpoint --samples "$samples" \
        --batch-sizes "$samples" --inflight "$inflight" --max-num-batched-tokens "$prefill"
}
run_client > "$run_dir/logs/client.out" 2> "$run_dir/logs/client.err" &
client_pid=$!
wait -n "$serving_pid" "$client_pid"
if kill -0 "$client_pid" 2>/dev/null; then
    echo "SML service exited before the pipeline client completed" >&2
    exit 1
fi
