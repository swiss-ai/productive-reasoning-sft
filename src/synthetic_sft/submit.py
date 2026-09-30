from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from synthetic_sft.config import PipelineConfig


def build_sbatch_command(config: PipelineConfig, config_path: Path) -> list[str]:
    if config.model.execution == "endpoint":
        raise ValueError("Deploy endpoint pilots with scripts/render_sml.py, not the Ray launcher")
    slurm = config.slurm
    project_dir = _project_root()
    job_script = project_dir / "slurm" / "job.sbatch"
    if not job_script.exists():
        raise FileNotFoundError(f"Slurm job script not found: {job_script}")

    log_dir = config.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launch_config = _write_launch_config(config)
    exported = {
        "SFT_CONFIG": str(launch_config),
        "SFT_ENVIRONMENT": str(slurm.environment),
        # Execute the source baked into the versioned image. The host repository is
        # intentionally not on the runtime import path during a long resumable run.
        "SFT_PROJECT_DIR": "/opt/synthetic-sft",
        "SFT_IMAGE": os.path.expandvars(slurm.image),
        "RAY_PORT_BROADCAST_DIR": str(config.run_dir / "cluster"),
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exported.items())
    command = [
        "sbatch",
        f"--job-name=sft-{config.run.run_id}",
        f"--nodes={slurm.nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={slurm.cpus_per_node}",
        f"--gpus-per-node={slurm.gpus_per_node}",
        f"--time={slurm.time}",
        f"--output={log_dir}/slurm-%j.out",
        f"--export={export_arg}",
    ]
    if slurm.partition:
        command.append(f"--partition={slurm.partition}")
    if slurm.account:
        command.append(f"--account={slurm.account}")
    if slurm.qos:
        command.append(f"--qos={slurm.qos}")
    if slurm.exclusive:
        command.append("--exclusive")
    if slurm.mem is not None:
        command.append(f"--mem={slurm.mem}")
    if slurm.requeue_before_timeout_seconds is not None:
        if not slurm.requeue:
            raise ValueError("requeue_before_timeout_seconds requires slurm.requeue: true")
        command.append(f"--signal=B:USR1@{slurm.requeue_before_timeout_seconds}")
    command.append("--requeue" if slurm.requeue else "--no-requeue")
    command.append(str(job_script))
    return command


def submit(config: PipelineConfig, config_path: Path, *, dry_run: bool = False) -> str:
    command = build_sbatch_command(config, config_path)
    if dry_run:
        return " ".join(command)
    environment = os.environ.copy()
    # A stale shell setting from the old reservation must not change this launch.
    environment.pop("SBATCH_RESERVATION", None)
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "unknown Slurm error"
        raise RuntimeError(f"sbatch failed: {message}")
    return result.stdout.strip()


def _project_root() -> Path:
    candidate = Path.cwd().resolve()
    if (candidate / "slurm" / "job.sbatch").exists():
        return candidate
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "slurm" / "job.sbatch").exists():
        return candidate
    raise FileNotFoundError("run submission from the synthetic-sft repository root")


def _write_launch_config(config: PipelineConfig) -> Path:
    path = config.run_dir / "manifests" / "launch.resolved.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    if path.exists():
        previous = yaml.safe_load(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(
                f"run_id {config.run.run_id!r} already has a different launch configuration"
            )
        return path
    temporary = path.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path
