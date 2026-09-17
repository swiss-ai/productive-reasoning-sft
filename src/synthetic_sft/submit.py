from __future__ import annotations

import os
import subprocess
from pathlib import Path

from synthetic_sft.config import PipelineConfig


def build_sbatch_command(config: PipelineConfig, config_path: Path) -> list[str]:
    slurm = config.slurm
    project_dir = _project_root()
    job_script = project_dir / "slurm" / "job.sbatch"
    if not job_script.exists():
        raise FileNotFoundError(f"Slurm job script not found: {job_script}")

    log_dir = config.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    exported = {
        "SFT_CONFIG": str(config_path.resolve()),
        "SFT_ENVIRONMENT": str(slurm.environment),
        "SFT_PROJECT_DIR": str(project_dir),
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
    command.append(str(job_script))
    return command


def submit(config: PipelineConfig, config_path: Path, *, dry_run: bool = False) -> str:
    command = build_sbatch_command(config, config_path)
    if dry_run:
        return " ".join(command)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
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
