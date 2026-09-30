from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from synthetic_sft.schemas import StrictModel


class RunConfig(StrictModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    output_dir: Path = Path("runs")
    seed: int = 42
    resume: bool = True


class SourceConfig(StrictModel):
    adapter: str
    num_samples: int = Field(gt=0)
    params: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None


class SamplingConfig(StrictModel):
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    max_tokens: int = Field(default=8192, gt=0)
    presence_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    reasoning_effort: str | None = Field(default="medium", pattern=r"^(low|medium|high|xhigh)$")


class EndpointConfig(StrictModel):
    served_model_name: str
    base_url_env: str = "SFT_API_BASE"
    api_key_env: str = "SFT_API_KEY"
    timeout_seconds: float = Field(default=1800, gt=0)


class ModelConfig(StrictModel):
    execution: Literal["waves", "continuous", "endpoint"] = "waves"
    endpoint: EndpointConfig | None = None
    inflight_candidates: int = Field(default=32, gt=0)
    model_source: str = "Qwen/Qwen3.8-27B"
    revision: str | None = None
    dtype: str = "bfloat16"
    tensor_parallel_size: int = Field(default=1, gt=0)
    max_model_len: int = Field(default=16384, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, lt=1.0)
    max_num_seqs: int = Field(default=256, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    enable_prefix_caching: bool = True
    batch_size: int = Field(default=64, gt=0)
    trust_remote_code: bool = False
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


class GenerationConfig(StrictModel):
    rollouts_per_prompt: int = Field(default=1, gt=0)
    polish: bool = True
    polish_max_tokens: int = Field(default=4096, gt=0)


class JudgeConfig(StrictModel):
    enabled: bool = True
    model_source: str | None = None
    batch_size: int = Field(default=64, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    analysis_samples: int = Field(default=2, ge=1, le=4)
    analysis_max_tokens: int = Field(default=2048, gt=0)
    hygiene_max_tokens: int = Field(default=768, gt=0)
    reasoning_effort: str | None = Field(default="medium", pattern=r"^(low|medium|high|xhigh)$")
    rubric_version: int = Field(default=4, ge=1)


class QualityConfig(StrictModel):
    verifier_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class OutputConfig(StrictModel):
    checkpoint_rows: int = Field(default=32, gt=0)
    checkpoint_seconds: float = Field(default=60.0, gt=0)
    parquet_target_mb: int = Field(default=384, ge=64)
    compression: str = "zstd"
    prepared_shards_per_gpu: int = Field(default=4, gt=0)


class SlurmConfig(StrictModel):
    requeue: bool = True
    nodes: int = Field(default=1, gt=0)
    gpus_per_node: int = Field(default=4, gt=0)
    cpus_per_node: int = Field(default=288, gt=0)
    time: str = Field(default="02:00:00", pattern=r"^\d{1,3}:\d{2}:\d{2}$")
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    exclusive: bool = True
    environment: Path = Path("container/cscs.toml")
    image: str = "nvcr.io#nvidia/vllm:26.08-py3"


class PipelineConfig(StrictModel):
    run: RunConfig
    source: SourceConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

    @property
    def run_dir(self) -> Path:
        return (self.run.output_dir / self.run.run_id).resolve()

    @property
    def seed_path(self) -> Path:
        return self.run_dir / "prepared" / "seeds"


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    config = PipelineConfig.model_validate(raw)
    base = config_path.parent
    config.run.output_dir = (base / config.run.output_dir).resolve()
    config.slurm.environment = (base / config.slurm.environment).resolve()
    return config
