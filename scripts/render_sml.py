"""Render a private Swiss Model Launcher deployment from a pipeline config.

Run in SML's environment. Uses its public rendering API, without gateway
registration or credentials. The generated master.sh is submitted with sbatch.
"""

import argparse
import json
import os
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
    parser.add_argument("--seeds", type=Path, help="Also render a single-job pipeline pilot")
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
    if model.get("safetensors_load_strategy"):
        framework_args += " --safetensors-load-strategy " + shlex.quote(
            model["safetensors_load_strategy"]
        )
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
    header = render_sbatch_header(launch)
    if slurm.get("mem") is not None:
        header += f"\n#SBATCH --mem={shlex.quote(str(slurm['mem']))}\n"
    master = header + "\n" + render_master(launch)
    (args.output / "master.sh").write_text(master)
    (args.output / "launch.json").write_text(json.dumps(launch.model_dump(), indent=2))
    if args.seeds:
        project = Path(__file__).resolve().parent.parent
        run_dir = (
            args.config.parent / config["run"]["output_dir"] / config["run"]["run_id"]
        ).resolve()
        environment = (args.config.parent / slurm["environment"]).resolve()
        command = shlex.join([
            "bash", str(project / "slurm/sml-pilot.sh"),
            str(args.config.resolve()), str((args.output / "master.sh").resolve()),
            str(args.seeds.resolve()), str(run_dir), str(config["source"]["num_samples"]),
            str(model["inflight_candidates"]), str(model["max_num_batched_tokens"]),
        ])
        exports = "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in {
                "SFT_PROJECT_DIR": str(project),
                "SFT_IMAGE": os.path.expandvars(slurm["image"]),
                "SFT_CLIENT_ENVIRONMENT": str(environment),
            }.items()
        )
        (args.output / "pilot.sh").write_text(header + "\n" + exports + "\nexec " + command + "\n")
    print(args.output / "master.sh")


if __name__ == "__main__":
    main()
