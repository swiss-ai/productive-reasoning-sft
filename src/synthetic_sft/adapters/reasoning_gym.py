from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from typing import Any

from synthetic_sft.adapters.base import SourceAdapter, VerificationResult
from synthetic_sft.json_utils import canonical_json, stable_id, to_jsonable
from synthetic_sft.schemas import SeedRecord


class ReasoningGymAdapter(SourceAdapter):
    name = "reasoning_gym"

    def _task_specs(self) -> list[dict[str, Any]]:
        if self.params.get("all_tasks"):
            from reasoning_gym.factory import DATASETS

            excluded = {"composite", *map(str, self.params.get("exclude_tasks", []))}
            return [
                {"name": name, "weight": 1.0, "config": {}}
                for name in sorted(DATASETS)
                if name not in excluded
            ]
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
        if self.params.get("all_tasks") and num_samples != len(specs):
            raise ValueError(
                f"all_tasks requires num_samples={len(specs)} for exactly one example per task; "
                f"got {num_samples}"
            )
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
            score = max(
                float(scorer(candidate, verification["entry"]))
                for candidate in _answer_candidates(response)
            )
            return VerificationResult(score=min(1.0, max(0.0, score)))
        except Exception as exc:  # A broken source scorer must not lose the generated row.
            return VerificationResult(score=None, error=f"{type(exc).__name__}: {exc}")


def _answer_candidates(response: str) -> list[str]:
    """Offer source scorers plausible final-answer spans without knowing answer semantics."""
    text = response.strip()
    plain = re.sub(r"[*_`#$]", "", text)
    plain = plain.replace(r"\(", "").replace(r"\)", "")
    candidates = [text, plain]
    candidates.extend(re.findall(r"\\boxed\{([^{}]+)\}", text))
    candidates.extend(
        f"{numerator}/{denominator}"
        for numerator, denominator in re.findall(
            r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", text
        )
    )
    candidates.extend(
        match.strip()
        for match in re.findall(
            r"(?im)(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", plain
        )
    )
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        last = re.sub(r"[*_`#]", "", nonempty_lines[-1]).strip()
        candidates.append(last)
        if "=" in last:
            candidates.append(last.rsplit("=", 1)[-1].strip().rstrip("."))
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", last.replace(",", ""))
        if numbers:
            candidates.append(numbers[-1])
            try:
                numeric = float(numbers[-1])
                if numeric.is_integer():
                    candidates.append(str(int(numeric)))
            except ValueError:
                pass
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))
