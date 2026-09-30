"""Render a private Swiss Model Launcher deployment from a pipeline config.

Run in SML's environment. Uses its public rendering API, without gateway
registration or credentials. The generated master.sh is submitted with sbatch.
"""

import argparse
import json
import shlex
from pathlib import Path

import yaml
from swiss_ai_model_launch.launchers.framework import render_master, render_rank_scripts
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.launchers.topology import Topology
from swiss_ai_model_launch.launchers.utils import render_sbatch_header


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model, slurm = config["model"], config["slurm"]
    framework_args = shlex.join([
        "--model", model["model_source"],
        "--served-model-name", model["endpoint"]["served_model_name"],
        "--tensor-parallel-size", str(model["tensor_parallel_size"]),
        "--reasoning-parser", "deepseek_v41",
        "--max-model-len", str(model["max_model_len"]),
        "--max-num-seqs", str(model["max_num_seqs"]),
        "--max-num-batched-tokens", str(model["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(model["gpu_memory_utilization"]),
        "--generation-config", "vllm",
        "--seed", str(config["run"]["seed"]),
    ])
    launch = LaunchArgs(
        job_name=config["run"]["run_id"],
        served_model_name=model["endpoint"]["served_model_name"],
        account=slurm["account"], partition=slurm["partition"], time=slurm["time"],
        topology=Topology(replicas=1, nodes_per_replica=slurm["nodes"]),
        environment=str(args.environment.resolve()),
        framework="vllm", framework_args=framework_args,
        disable_opentela=True, disable_metrics=True, disable_dcgm_exporter=True,
        pre_launch_cmds=(
            'export VLLM_CACHE_ROOT="$SCRATCH/.cache/synthetic-sft/sml-vllm"\n'
            'export TRITON_CACHE_DIR="$SCRATCH/.cache/synthetic-sft/sml-triton"'
        ),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in render_rank_scripts(launch).items():
        (args.output / name).write_text(content)
    master = render_sbatch_header(launch) + "\n" + render_master(launch)
    (args.output / "master.sh").write_text(master)
    (args.output / "launch.json").write_text(json.dumps(launch.model_dump(), indent=2))
    print(args.output / "master.sh")


if __name__ == "__main__":
    main()
