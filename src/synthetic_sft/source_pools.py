from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, snapshot_download

from synthetic_sft.json_utils import canonical_json, stable_id, to_jsonable

PROMPT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("user_prompt", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("provenance_json", pa.string(), nullable=False),
        pa.field("system_prompt", pa.string()),
        pa.field("verification_json", pa.string()),
        pa.field("difficulty_band", pa.string(), nullable=False),
        pa.field("difficulty_value", pa.float64()),
        pa.field("subset", pa.string()),
        pa.field("domain", pa.string()),
    ]
)

REFERENCE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("reference_id", pa.string(), nullable=False),
        pa.field("reasoning", pa.string()),
        pa.field("response", pa.string()),
        pa.field("source_model", pa.string()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("provenance_json", pa.string(), nullable=False),
    ]
)


def build_source_pools(config_path: str | Path, *, names: set[str] | None = None) -> list[Path]:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError("source-pool configuration requires a sources list")
    root = Path(os.path.expandvars(str(raw.get("output_dir", "source-pools")))).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    raw_root = Path(os.path.expandvars(str(raw.get("raw_dir", root / "raw")))).expanduser()
    if not raw_root.is_absolute():
        raw_root = (path.parent / raw_root).resolve()
    outputs = []
    for source in raw["sources"]:
        name = str(source["name"])
        if names is not None and name not in names:
            continue
        raw_name = str(source.get("raw_name", name))
        outputs.append(_build_one(source, root / name, raw_root / raw_name))
    return outputs


def _build_one(config: Mapping[str, Any], output: Path, raw_dir: Path) -> Path:
    success = output / "_SUCCESS.json"
    if success.exists():
        return output
    kind = str(config["kind"])
    repo_id = str(config.get("repo_id", "reasoning-gym"))
    if kind == "reasoning_gym":
        resolved_revision = package_version("reasoning-gym")
    else:
        patterns = list(map(str, config.get("allow_patterns", [])))
        revision = config.get("revision")
        resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            allow_patterns=patterns or None,
            local_dir=raw_dir,
            max_workers=int(config.get("download_workers", 8)),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.mkdir(parents=True)
    prompt_writer = _ShardWriter(temporary / "prompts", PROMPT_SCHEMA)
    reference_writer = _ShardWriter(temporary / "references", REFERENCE_SCHEMA)
    builder = {
        "deepscaler": _deepscaler_rows,
        "deepmath": _deepmath_rows,
        "openmath": _openmath_rows,
        "openthoughts": _openthoughts_rows,
        "reasoning_gym": _reasoning_gym_rows,
    }.get(kind)
    if builder is None:
        raise ValueError(f"unknown source-pool kind: {kind}")

    seen: set[str] = set()
    prompt_count = 0
    reference_count = 0
    try:
        for prompt, references in builder(raw_dir, repo_id, resolved_revision, config):
            if prompt["sample_id"] not in seen:
                seen.add(prompt["sample_id"])
                prompt_writer.append(prompt)
                prompt_count += 1
            for reference in references:
                reference_writer.append(reference)
                reference_count += 1
        prompt_writer.close()
        reference_writer.close()
        if prompt_count == 0:
            raise RuntimeError(f"source {repo_id} produced no prompts")
        (temporary / "_SUCCESS.json").write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": resolved_revision,
                    "prompts": prompt_count,
                    "references": reference_count,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        prompt_writer.close()
        reference_writer.close()
        if temporary.exists():
            temporary.rename(temporary.with_name(f"{output.name}.incomplete.{time.time_ns()}"))
        raise
    return output


class _ShardWriter:
    def __init__(self, path: Path, schema: pa.Schema, rows_per_shard: int = 10_000):
        self.path = path
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.buffer: list[dict[str, Any]] = []
        self.part = 0
        self.closed = False
        path.mkdir(parents=True)

    def append(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        pq.write_table(
            pa.Table.from_pylist(self.buffer, schema=self.schema),
            self.path / f"part-{self.part:06d}.parquet",
            compression="zstd",
            row_group_size=min(10_000, len(self.buffer)),
        )
        self.buffer.clear()
        self.part += 1

    def close(self) -> None:
        if not self.closed:
            self._flush()
            self.closed = True


def _parquet_rows(paths: Iterable[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in paths:
        dataset = pads.dataset(path, format="parquet")
        for batch in dataset.scanner(batch_size=2_000).to_batches():
            for row in batch.to_pylist():
                yield path.name, row


def _make_prompt(
    *,
    repo_id: str,
    revision: str,
    prompt: str,
    source: str,
    answer: Any,
    band: str,
    difficulty: float | None,
    subset: str | None,
    domain: str | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = stable_id("source_pool", repo_id, prompt)
    provenance = {
        "adapter": "pool",
        "dataset": repo_id,
        "dataset_revision": revision,
        "difficulty_band": band,
        "difficulty_value": difficulty,
        "subset": subset,
        "domain": domain,
        "metadata": to_jsonable(metadata),
    }
    verification = None
    if answer is not None and str(answer).strip():
        verification = canonical_json(
            {"type": "reference_answer", "entry": {"answer": to_jsonable(answer)}}
        )
    return {
        "sample_id": sample_id,
        "user_prompt": prompt,
        "source": source,
        "provenance_json": canonical_json(provenance),
        "system_prompt": None,
        "verification_json": verification,
        "difficulty_band": band,
        "difficulty_value": difficulty,
        "subset": subset,
        "domain": domain,
    }


def _make_reference(
    prompt: Mapping[str, Any],
    raw: str | None,
    answer: Any,
    model: str | None,
    source: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    reasoning, response = _split_response(raw, answer)
    return {
        "sample_id": prompt["sample_id"],
        "reference_id": stable_id(prompt["sample_id"], source, model, raw),
        "reasoning": reasoning,
        "response": response,
        "source_model": model,
        "source": source,
        "provenance_json": canonical_json(to_jsonable(metadata)),
    }


def _split_response(raw: str, answer: Any) -> tuple[str | None, str | None]:
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        response = text.split("</think>", 1)[1].strip()
        return reasoning or None, response or (str(answer).strip() if answer is not None else None)
    return text, str(answer).strip() if answer is not None and str(answer).strip() else None


def _deepscaler_rows(raw_dir: Path, repo_id: str, revision: str, config: Mapping[str, Any]):
    rows = json.loads((raw_dir / "deepscaler.json").read_text(encoding="utf-8"))
    for index, row in enumerate(rows):
        prompt = _make_prompt(
            repo_id=repo_id,
            revision=revision,
            prompt=str(row["problem"]),
            source=f"{repo_id}:train",
            answer=row.get("answer"),
            band="unknown",
            difficulty=None,
            subset="train",
            domain="math",
            metadata={"source_index": index},
        )
        reference = _make_reference(
            prompt, row.get("solution"), row.get("answer"), "official", "solution", {}
        )
        yield prompt, [reference] if reference else []


def _deepmath_rows(raw_dir: Path, repo_id: str, revision: str, config: Mapping[str, Any]):
    paths = sorted((raw_dir / "data").glob("*.parquet"))
    for filename, row in _parquet_rows(paths):
        difficulty = float(row["difficulty"]) if row.get("difficulty") is not None else None
        band = (
            "unknown"
            if difficulty is None
            else ("easy" if difficulty <= 4 else "medium" if difficulty <= 6.5 else "hard")
        )
        prompt = _make_prompt(
            repo_id=repo_id,
            revision=revision,
            prompt=str(row["question"]),
            source=f"{repo_id}:train",
            answer=row.get("final_answer"),
            band=band,
            difficulty=difficulty,
            subset="train",
            domain="math",
            metadata={"topic": row.get("topic"), "raw_file": filename},
        )
        references = []
        for index in range(1, 4):
            reference = _make_reference(
                prompt,
                row.get(f"r1_solution_{index}"),
                row.get("final_answer"),
                "DeepSeek-R1",
                f"r1_solution_{index}",
                {"solution_index": index},
            )
            if reference:
                references.append(reference)
        yield prompt, references


def _openmath_rows(raw_dir: Path, repo_id: str, revision: str, config: Mapping[str, Any]):
    paths = list((raw_dir / "data").glob("*.parquet"))
    priority = {"cot": 0, "tir": 1, "genselect": 2, "additional_problems": 3}
    paths.sort(key=lambda path: (priority.get(path.name.split("-", 1)[0], 9), path.name))
    for filename, row in _parquet_rows(paths):
        subset = filename.split("-", 1)[0]
        value = _float_or_none(row.get("pass_rate_72b_tir"))
        band = (
            "unknown"
            if value is None
            else ("easy" if value >= 0.75 else "medium" if value > 0.25 else "hard")
        )
        answer = row.get("expected_answer")
        prompt = _make_prompt(
            repo_id=repo_id,
            revision=revision,
            prompt=str(row["problem"]),
            source=f"{repo_id}:{row.get('problem_source') or subset}",
            answer=answer,
            band=band,
            difficulty=value,
            subset=subset,
            domain="math",
            metadata={
                "problem_type": row.get("problem_type"),
                "problem_source": row.get("problem_source"),
                "used_in_kaggle": row.get("used_in_kaggle"),
                "raw_file": filename,
            },
        )
        reference = _make_reference(
            prompt,
            row.get("generated_solution"),
            answer,
            row.get("generation_model"),
            subset,
            {"inference_mode": row.get("inference_mode")},
        )
        yield prompt, [reference] if reference else []


def _openthoughts_rows(raw_dir: Path, repo_id: str, revision: str, config: Mapping[str, Any]):
    paths = sorted((raw_dir / "data").glob("*.parquet"))
    for filename, row in _parquet_rows(paths):
        conversations = row.get("conversations") or []
        user = next(
            (item.get("value") for item in conversations if item.get("from") == "human"), None
        )
        assistant = next(
            (item.get("value") for item in conversations if item.get("from") == "gpt"), None
        )
        if not user:
            continue
        difficulty = float(row["difficulty"]) if row.get("difficulty") is not None else None
        band = (
            "unknown"
            if difficulty is None
            else ("easy" if difficulty <= 6 else "medium" if difficulty <= 8 else "hard")
        )
        prompt = _make_prompt(
            repo_id=repo_id,
            revision=revision,
            prompt=str(user),
            source=f"{repo_id}:{row.get('source')}",
            answer=None,
            band=band,
            difficulty=difficulty,
            subset=str(row.get("source")),
            domain=str(row.get("domain")),
            metadata={"source": row.get("source"), "raw_file": filename},
        )
        reference = _make_reference(
            prompt, assistant, None, "QwQ-32B", "conversation", {"difficulty": difficulty}
        )
        yield prompt, [reference] if reference else []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reasoning_gym_rows(raw_dir: Path, repo_id: str, revision: str, config: Mapping[str, Any]):
    groups = config.get("task_groups")
    tasks: list[str] = []
    task_weights: list[float] = []
    if isinstance(groups, Mapping):
        for group in groups.values():
            group_tasks = list(map(str, group.get("tasks", [])))
            group_weight = float(group.get("weight", 1.0))
            tasks.extend(group_tasks)
            task_weights.extend([group_weight / len(group_tasks)] * len(group_tasks))
    else:
        tasks = list(map(str, config.get("tasks", [])))
        task_weights = [1.0] * len(tasks)
    if not tasks:
        raise ValueError("reasoning_gym pool requires non-empty tasks or task_groups")
    total = int(config.get("num_samples", 1_000_000))
    seed = int(config.get("seed", 42))
    difficulty_weights = config.get(
        "difficulty_weights", {"easy": 0.15, "medium": 0.35, "hard": 0.5}
    )
    task_counts = _allocate_counts(total, task_weights)
    chunk_size = int(config.get("task_chunk_size", 1_000))
    jobs = []
    for task_index, (task, task_count) in enumerate(zip(tasks, task_counts, strict=True)):
        for chunk_index, chunk_start in enumerate(range(0, task_count, chunk_size)):
            jobs.append(
                (
                    task_index,
                    task,
                    min(chunk_size, task_count - chunk_start),
                    chunk_index,
                    chunk_start,
                    seed,
                    dict(difficulty_weights),
                    repo_id,
                    revision,
                )
            )
    workers = min(int(config.get("workers", 16)), len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_reasoning_gym_task_rows, job) for job in jobs]
        for future in as_completed(futures):
            yield from future.result()


def _reasoning_gym_task_rows(args: tuple[Any, ...]) -> list[tuple[dict[str, Any], list]]:
    import reasoning_gym
    from reasoning_gym.coaching.base_curriculum import (
        DefaultCurriculumContext,
        RangeAttributeMode,
    )
    from reasoning_gym.factory import create_curriculum, has_curriculum

    (
        task_index,
        task,
        task_count,
        chunk_index,
        chunk_start,
        seed,
        difficulty_weights,
        repo_id,
        revision,
    ) = args
    fractions = {"easy": 0.2, "medium": 0.6, "hard": 1.0}
    band_names = list(difficulty_weights)
    band_counts = _allocate_counts(
        task_count, [float(difficulty_weights[band]) for band in band_names]
    )
    rows = []
    for band, band_count in zip(band_names, band_counts, strict=True):
        if band_count == 0:
            continue
        task_seed = (
            seed
            + task_index * 1_000_003
            + band_names.index(band) * 100_003
            + chunk_index * 10_007
        )
        kwargs: dict[str, Any] = {"size": band_count, "seed": task_seed}
        if has_curriculum(task):
            curriculum = create_curriculum(task)
            fraction = fractions.get(band, 0.6)
            for attribute in curriculum.attributes.values():
                level = round(fraction * (len(attribute.levels) - 1))
                curriculum.set_attr_level(attribute.name, level)
            generated = curriculum.generate_configuration(
                context=DefaultCurriculumContext(mode=RangeAttributeMode.LAST_K, k=1)
            )
            kwargs.update(
                {
                    key: value
                    for key, value in generated.__dict__.items()
                    if key not in {"size", "seed"}
                }
            )
        dataset = reasoning_gym.create_dataset(task, **kwargs)
        for local_index, entry in enumerate(dataset):
            answer = to_jsonable(entry.get("answer"))
            metadata = to_jsonable(entry.get("metadata", {}))
            prompt = _make_prompt(
                repo_id=repo_id,
                revision=revision,
                prompt=str(entry["question"]),
                source=f"reasoning_gym:{task}",
                answer=None,
                band=band,
                difficulty=fractions.get(band),
                subset=task,
                domain="reasoning",
                metadata={
                    "task": task,
                    "task_seed": task_seed,
                    "task_index": chunk_start + local_index,
                    "task_chunk": chunk_index,
                    "task_config": to_jsonable(kwargs),
                    "metadata": metadata,
                },
            )
            source_dataset = metadata.get("source_dataset", task)
            prompt["verification_json"] = canonical_json(
                {
                    "type": "reasoning_gym",
                    "entry": {
                        "question": str(entry["question"]),
                        "answer": answer,
                        "metadata": metadata,
                    },
                    "source_dataset": source_dataset,
                }
            )
            rows.append((prompt, []))
    return rows


def _allocate_counts(total: int, weights: list[float]) -> list[int]:
    exact = [total * weight / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in exact]
    for index in sorted(
        range(len(weights)), key=lambda item: exact[item] - counts[item], reverse=True
    )[: total - sum(counts)]:
        counts[index] += 1
    return counts
