from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from synthetic_sft.cluster import RayCluster, available_gpu_replicas
from synthetic_sft.config import PipelineConfig
from synthetic_sft.generation import FanOutCandidates, build_vllm_processor, can_fuse_judge
from synthetic_sft.prepare import prepare_source
from synthetic_sft.quality import FinalizeQuality, ParseAndVerify
from synthetic_sft.schemas import quality_details_schema

SFT_COLUMNS = [
    "sample_id",
    "candidate_id",
    "candidate_index",
    "system_prompt",
    "user_prompt",
    "reasoning",
    "response",
    "source",
    "model",
    "quality_score",
    "quality_details_json",
    "provenance_json",
]


def run_pipeline(config: PipelineConfig, *, force_prepare: bool = False) -> Path:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RayCluster(
        run_dir=config.run_dir,
        cpus_per_node=config.slurm.cpus_per_node,
        gpus_per_node=config.slurm.gpus_per_node,
    ):
        import ray

        # Only the Slurm node-0 process returns from RayCluster.__enter__.
        _write_run_metadata(config)
        seeds = prepare_source(config, force=force_prepare)
        context = ray.data.DataContext.get_current()
        context.target_max_block_size = config.output.parquet_target_mb * 1024 * 1024
        replicas = available_gpu_replicas(config.model.tensor_parallel_size)
        inference_blocks = replicas * config.output.prepared_shards_per_gpu
        generated_path = config.run_dir / "intermediate" / "generated"
        if not _stage_complete(config, "generated"):
            stage_started = time.time()
            _rotate_incomplete(generated_path)
            dataset = ray.data.read_parquet(
                str(seeds), override_num_blocks=inference_blocks
            )
            dataset = dataset.flat_map(
                FanOutCandidates(
                    config.generation.rollouts_per_prompt,
                    config.run.seed,
                    config.model.model_source,
                )
            )
            dataset = dataset.repartition(inference_blocks, shuffle=False)
            generated = build_vllm_processor(config, judge=False, concurrency=replicas)(dataset)
            generated.write_parquet(str(generated_path), compression=config.output.compression)
            _mark_stage(config, "generated", generated_path, stage_started)

        candidates_path = config.run_dir / "candidates"
        if not _stage_complete(config, "candidates"):
            stage_started = time.time()
            _rotate_incomplete(candidates_path)
            candidates = ray.data.read_parquet(
                str(generated_path), override_num_blocks=inference_blocks
            )
            candidates = candidates.repartition(inference_blocks, shuffle=False)
            if not can_fuse_judge(config):
                candidates = candidates.map(
                    ParseAndVerify(
                        config.source.adapter,
                        config.source.params,
                        config.quality.verifier_threshold,
                    )
                )
                if config.quality.judge.enabled:
                    candidates = build_vllm_processor(
                        config, judge=True, concurrency=replicas
                    )(candidates)
                candidates = candidates.map(FinalizeQuality(config.model_dump(mode="json")))
            candidates.write_parquet(str(candidates_path), compression=config.output.compression)
            _mark_stage(config, "candidates", candidates_path, stage_started)

        sft_path = config.run_dir / "sft"
        if not _stage_complete(config, "sft"):
            stage_started = time.time()
            _rotate_incomplete(sft_path)
            sft = ray.data.read_parquet(str(candidates_path)).select_columns(SFT_COLUMNS)
            sft.write_parquet(str(sft_path), compression=config.output.compression)
            _mark_stage(config, "sft", sft_path, stage_started)
    return sft_path


def _manifest_dir(config: PipelineConfig) -> Path:
    path = config.run_dir / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_complete(config: PipelineConfig, stage: str) -> bool:
    if not config.run.resume:
        return False
    marker = _manifest_dir(config) / f"{stage}.json"
    if not marker.exists():
        return False
    output = Path(json.loads(marker.read_text(encoding="utf-8"))["output"])
    if not output.exists() or not any(output.rglob("*.parquet")):
        raise RuntimeError(f"completed stage {stage!r} is missing its Parquet output: {output}")
    return True


def _mark_stage(
    config: PipelineConfig, stage: str, output: Path, started_at_unix: float
) -> None:
    if not any(output.rglob("*.parquet")):
        raise RuntimeError(f"stage {stage!r} produced no Parquet files in {output}")
    marker = _manifest_dir(config) / f"{stage}.json"
    temporary = marker.with_suffix(f".tmp.{time.time_ns()}")
    completed_at_unix = time.time()
    temporary.write_text(
        json.dumps(
            {
                "stage": stage,
                "output": str(output),
                "started_at_unix": started_at_unix,
                "completed_at_unix": completed_at_unix,
                "duration_seconds": completed_at_unix - started_at_unix,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(marker)


def _rotate_incomplete(path: Path) -> None:
    if path.exists():
        path.rename(path.with_name(f"{path.name}.incomplete.{time.time_ns()}"))
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_run_metadata(config: PipelineConfig) -> None:
    manifest_dir = _manifest_dir(config)
    snapshot = manifest_dir / "config.resolved.yaml"
    resolved = config.model_dump(mode="json")
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        if previous != resolved:
            raise RuntimeError(
                f"run_id {config.run.run_id!r} already has a different resolved configuration"
            )
    else:
        snapshot.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    schema_path = manifest_dir / "quality-details.schema.json"
    if not schema_path.exists():
        schema_path.write_text(
            json.dumps(quality_details_schema(), indent=2, sort_keys=True), encoding="utf-8"
        )
