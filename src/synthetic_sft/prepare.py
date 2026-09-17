from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from synthetic_sft.adapters import create_adapter
from synthetic_sft.config import PipelineConfig
from synthetic_sft.json_utils import canonical_json, stable_id

SEED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("user_prompt", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("provenance_json", pa.string(), nullable=False),
        pa.field("system_prompt", pa.string()),
        pa.field("verification_json", pa.string()),
    ]
)


def prepare_source(config: PipelineConfig, *, force: bool = False) -> Path:
    output = config.seed_path
    success = output / "_SUCCESS.json"
    fingerprint = _source_fingerprint(config)
    if success.exists() and not force:
        metadata = json.loads(success.read_text(encoding="utf-8"))
        if metadata.get("source_fingerprint") != fingerprint:
            raise RuntimeError(
                "prepared source does not match this configuration; use --force or a new run_id"
            )
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.rename(output.with_name(f"{output.name}.incomplete.{time.time_ns()}"))
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.mkdir(parents=True)

    adapter = create_adapter(config.source.adapter, config.source.params)
    rows_per_shard = _rows_per_shard(config)
    buffer: list[dict[str, object]] = []
    count = 0
    part = 0
    try:
        for record in adapter.prepare(
            num_samples=config.source.num_samples,
            seed=config.run.seed,
            system_prompt=config.source.system_prompt,
        ):
            buffer.append(record.model_dump())
            count += 1
            if len(buffer) >= rows_per_shard:
                _write_part(temporary, part, buffer, config.output.compression)
                buffer.clear()
                part += 1
        if buffer:
            _write_part(temporary, part, buffer, config.output.compression)
            part += 1
        if count != config.source.num_samples:
            raise RuntimeError(
                f"source adapter produced {count} records, expected {config.source.num_samples}"
            )
        if part == 0:
            raise ValueError("source adapter produced no records")
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(
                {
                    "records": count,
                    "parts": part,
                    "rows_per_shard": rows_per_shard,
                    "source_fingerprint": fingerprint,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.rename(temporary.with_name(f"{output.name}.incomplete.{time.time_ns()}"))
        raise
    return output


def _rows_per_shard(config: PipelineConfig) -> int:
    target_parts = (
        config.slurm.nodes * config.slurm.gpus_per_node * config.output.prepared_shards_per_gpu
    )
    calculated = math.ceil(config.source.num_samples / target_parts)
    return min(100_000, max(1_000, calculated))


def _write_part(path: Path, part: int, rows: list[dict[str, object]], compression: str) -> None:
    table = pa.Table.from_pylist(rows, schema=SEED_SCHEMA)
    pq.write_table(
        table,
        path / f"part-{part:06d}.parquet",
        compression=compression,
        row_group_size=min(len(rows), 10_000),
    )


def _source_fingerprint(config: PipelineConfig) -> str:
    material = {
        "seed": config.run.seed,
        "source": config.source.model_dump(mode="json"),
    }
    return stable_id(canonical_json(material))
