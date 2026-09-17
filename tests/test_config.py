from __future__ import annotations

from pathlib import Path

from synthetic_sft.config import load_config
from synthetic_sft.submit import build_sbatch_command


def test_example_config_and_single_sbatch_command() -> None:
    path = Path("configs/reasoning-gym-smoke.yaml")
    config = load_config(path)
    command = build_sbatch_command(config, path)
    assert command[0] == "sbatch"
    assert "--nodes=1" in command
    assert "--gpus-per-node=4" in command
    assert command[-1].endswith("slurm/job.sbatch")
