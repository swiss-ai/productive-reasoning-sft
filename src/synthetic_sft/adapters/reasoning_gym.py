from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

from synthetic_sft.adapters.base import SourceAdapter, VerificationResult
from synthetic_sft.json_utils import canonical_json, stable_id, to_jsonable
from synthetic_sft.schemas import SeedRecord


class ReasoningGymAdapter(SourceAdapter):
    name = "reasoning_gym"

    def _task_specs(self) -> list[dict[str, Any]]:
        tasks = self.params.get("tasks")
        if tasks is None:
            return [
                {
                    "name": self.params.get("dataset", "leg_counting"),
                    "weight": 1.0,
                    "config": self.params.get("config", {}),
                }
            ]
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("reasoning_gym params.tasks must be a non-empty list")
        normalized = []
        for task in tasks:
            if not isinstance(task, dict) or not task.get("name"):
                raise ValueError("each reasoning_gym task requires a name")
            weight = float(task.get("weight", 1.0))
            if weight <= 0:
                raise ValueError("reasoning_gym task weights must be positive")
            normalized.append(
                {"name": str(task["name"]), "weight": weight, "config": task.get("config", {})}
            )
        return normalized

    @staticmethod
    def _allocate(total: int, specs: list[dict[str, Any]]) -> list[int]:
        weight_sum = sum(spec["weight"] for spec in specs)
        exact = [total * spec["weight"] / weight_sum for spec in specs]
        counts = [math.floor(value) for value in exact]
        remainder = total - sum(counts)
        order = sorted(
            range(len(specs)), key=lambda index: exact[index] - counts[index], reverse=True
        )
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    def prepare(
        self, *, num_samples: int, seed: int, system_prompt: str | None
    ) -> Iterator[SeedRecord]:
        import reasoning_gym

        specs = self._task_specs()
        counts = self._allocate(num_samples, specs)
        ordinal = 0
        for task_index, (spec, count) in enumerate(zip(specs, counts, strict=True)):
            if count == 0:
                continue
            task_seed = seed + task_index * 1_000_003
            dataset = reasoning_gym.create_dataset(
                spec["name"], size=count, seed=task_seed, **dict(spec["config"])
            )
            for local_index, entry in enumerate(dataset):
                question = str(entry["question"])
                answer = to_jsonable(entry.get("answer"))
                metadata = to_jsonable(entry.get("metadata", {}))
                source_dataset = metadata.get("source_dataset", spec["name"])
                provenance = {
                    "adapter": self.name,
                    "dataset": spec["name"],
                    "dataset_seed": task_seed,
                    "dataset_index": local_index,
                    "task_config": to_jsonable(spec["config"]),
                    "metadata": metadata,
                    "reference_answer": answer,
                }
                verification = {
                    "entry": {"question": question, "answer": answer, "metadata": metadata},
                    "source_dataset": source_dataset,
                }
                yield SeedRecord(
                    sample_id=stable_id(self.name, seed, ordinal, spec["name"], question),
                    system_prompt=system_prompt,
                    user_prompt=question,
                    source=f"reasoning_gym:{spec['name']}",
                    provenance_json=canonical_json(provenance),
                    verification_json=canonical_json(verification),
                )
                ordinal += 1

    def supports_verification(self, record: Mapping[str, Any]) -> bool:
        return bool(record.get("verification_json"))

    def verifier_name(self, record: Mapping[str, Any]) -> str | None:
        return "reasoning_gym" if self.supports_verification(record) else None

    def verify(self, record: Mapping[str, Any], response: str) -> VerificationResult:
        from reasoning_gym import get_score_answer_fn

        payload = record.get("verification_json")
        if not payload:
            return VerificationResult(score=None)
        try:
            verification = json.loads(str(payload))
            scorer = get_score_answer_fn(verification["source_dataset"])
            score = float(scorer(response, verification["entry"]))
            return VerificationResult(score=min(1.0, max(0.0, score)))
        except Exception as exc:  # A broken source scorer must not lose the generated row.
            return VerificationResult(score=None, error=f"{type(exc).__name__}: {exc}")
