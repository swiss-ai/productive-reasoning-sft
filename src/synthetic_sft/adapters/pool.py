from __future__ import annotations

import heapq
import json
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads

from synthetic_sft.adapters.base import SourceAdapter, VerificationResult
from synthetic_sft.json_utils import stable_id
from synthetic_sft.schemas import SeedRecord


class PoolAdapter(SourceAdapter):
    """Draw a deterministic, stratified mixture from persistent source pools."""

    name = "pool"

    def prepare(
        self, *, num_samples: int, seed: int, system_prompt: str | None
    ) -> Iterator[SeedRecord]:
        sources = self.params.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("pool adapter requires a non-empty params.sources list")
        source_counts = _allocate(num_samples, [float(item.get("weight", 1.0)) for item in sources])
        selected: list[tuple[int, dict[str, Any]]] = []
        seen_prompts: set[str] = set()

        for source, source_count in zip(sources, source_counts, strict=True):
            if source_count == 0:
                continue
            strata = _sample_source(source, source_count, seed)
            for band, (target, rows) in strata.items():
                accepted = 0
                for rank, row in rows:
                    normalized = " ".join(str(row["user_prompt"]).split()).casefold()
                    if normalized in seen_prompts:
                        continue
                    seen_prompts.add(normalized)
                    selected.append((rank, row))
                    accepted += 1
                    if accepted == target:
                        break
                if accepted != target:
                    raise RuntimeError(
                        f"pool {source.get('path')} supplied {accepted} unique {band!r} rows, "
                        f"but {target} were requested"
                    )

        # Hash order makes the final seed files independent of input shard order.
        for _, row in sorted(selected, key=lambda item: item[0]):
            yield SeedRecord(
                sample_id=str(row["sample_id"]),
                user_prompt=str(row["user_prompt"]),
                source=str(row["source"]),
                provenance_json=str(row["provenance_json"]),
                system_prompt=system_prompt
                if system_prompt is not None
                else row.get("system_prompt"),
                verification_json=row.get("verification_json"),
            )

    def supports_verification(self, record: Mapping[str, Any]) -> bool:
        return bool(record.get("verification_json"))

    def verifier_name(self, record: Mapping[str, Any]) -> str | None:
        if not self.supports_verification(record):
            return None
        payload = json.loads(str(record["verification_json"]))
        return str(payload.get("type", "reference_answer"))

    def verify(self, record: Mapping[str, Any], response: str) -> VerificationResult:
        payload = record.get("verification_json")
        if not payload:
            return VerificationResult(score=None)
        try:
            verification = json.loads(str(payload))
            if verification.get("source_dataset"):
                from synthetic_sft.adapters.reasoning_gym import ReasoningGymAdapter

                return ReasoningGymAdapter().verify(record, response)
            answer = verification.get("entry", {}).get("answer")
            if answer is None or str(answer).strip() == "":
                return VerificationResult(score=None)
            candidate = _extracted_answer(record.get("answer_json"))
            reference = _extracted_answer(record.get("reference_answer_json"))
            if candidate is None or reference is None:
                return VerificationResult(
                    score=None,
                    details={"reason": "structured_answer_unavailable"},
                )
            result = _verify_extracted(reference, candidate)
            if result is not None:
                return result
            return VerificationResult(
                score=None,
                details={
                    "reason": "answer_type_requires_model_review",
                    "candidate_type": candidate.get("answer_type"),
                    "reference_type": reference.get("answer_type"),
                },
            )
        except Exception as exc:
            return VerificationResult(score=None, error=f"{type(exc).__name__}: {exc}")


def _extracted_answer(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("status") != "extracted":
        return None
    return value


def _verify_extracted(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> VerificationResult | None:
    exact_types = {"boolean", "choice"}
    reference_type = str(reference.get("answer_type"))
    candidate_type = str(candidate.get("answer_type"))
    if reference_type in exact_types and candidate_type in exact_types:
        gold = _normalized_text(str(reference.get("value") or ""))
        predicted = _normalized_text(str(candidate.get("value") or ""))
        return VerificationResult(
            score=1.0 if gold == predicted else 0.0,
            details={"method": "normalized_exact", "gold": gold, "candidate": predicted},
        )

    math_types = {"number", "expression", "equation", "set", "interval"}
    if reference_type not in math_types or candidate_type not in math_types:
        return None
    reference_unit = _normalized_text(str(reference.get("unit") or ""))
    candidate_unit = _normalized_text(str(candidate.get("unit") or ""))
    if reference_unit and not candidate_unit:
        return VerificationResult(score=None, details={"reason": "candidate_unit_missing"})
    if reference_unit and reference_unit != candidate_unit:
        return VerificationResult(
            score=0.0,
            details={
                "method": "typed_math",
                "reason": "unit_mismatch",
                "gold_unit": reference_unit,
                "candidate_unit": candidate_unit,
            },
        )
    gold_items = _answer_items(reference)
    candidate_items = _answer_items(candidate)
    if not gold_items or not candidate_items:
        return VerificationResult(score=None, details={"reason": "empty_math_answer"})
    if len(gold_items) != len(candidate_items):
        return VerificationResult(
            score=0.0,
            details={
                "method": "typed_math",
                "reason": "different_answer_count",
                "gold_count": len(gold_items),
                "candidate_count": len(candidate_items),
            },
        )

    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    extraction = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())

    def parsed(value: str):
        # Both sides are extractor-produced answer spans, never arbitrary prose. A display
        # environment gives LaTeX unambiguous boundaries while ExprExtractionConfig covers
        # plain values such as 0.99.
        return parse(
            f"\\[\\boxed{{{value}}}\\]",
            extraction_config=extraction,
            parsing_timeout=5,
        )

    remaining = [parsed(item) for item in candidate_items]
    if any(not item for item in remaining):
        return VerificationResult(score=None, details={"reason": "candidate_parse_failed"})
    for gold_text in gold_items:
        gold = parsed(gold_text)
        if not gold:
            return VerificationResult(score=None, details={"reason": "reference_parse_failed"})
        match = next(
            (
                index
                for index, predicted in enumerate(remaining)
                if verify(gold, predicted, timeout_seconds=5)
            ),
            None,
        )
        if match is None:
            return VerificationResult(
                score=0.0,
                details={
                    "method": "typed_math",
                    "gold": gold_items,
                    "candidate": candidate_items,
                },
            )
        remaining.pop(match)
    return VerificationResult(
        score=1.0,
        details={"method": "typed_math", "gold": gold_items, "candidate": candidate_items},
    )


def _answer_items(answer: Mapping[str, Any]) -> list[str]:
    values = answer.get("values")
    if isinstance(values, list) and values:
        return [str(value).strip() for value in values if str(value).strip()]
    value = str(answer.get("value") or "").strip()
    return [value] if value else []


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(".").split())


def _sample_source(
    source: Mapping[str, Any], count: int, seed: int
) -> dict[str, tuple[int, list[tuple[int, dict]]]]:
    path = Path(os.path.expandvars(str(source["path"]))).expanduser()
    dataset = pads.dataset(str(path), format="parquet", exclude_invalid_files=True)
    weights = source.get("difficulty_weights", {"hard": 0.5, "medium": 0.35, "easy": 0.15})
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError(f"difficulty_weights must be a non-empty mapping for {path}")
    bands = list(weights)
    targets = dict(
        zip(bands, _allocate(count, [float(weights[band]) for band in bands]), strict=True)
    )
    oversample = float(source.get("oversample_factor", 1.25))
    capacities = {band: max(targets[band], math.ceil(targets[band] * oversample)) for band in bands}
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {band: [] for band in bands}
    serial = 0
    filters = {
        "subset": set(map(str, source.get("subsets", []))),
        "domain": set(map(str, source.get("domains", []))),
    }
    columns = [
        "sample_id",
        "user_prompt",
        "source",
        "provenance_json",
        "system_prompt",
        "verification_json",
        "difficulty_band",
        "subset",
        "domain",
    ]
    available = set(dataset.schema.names)
    columns = [column for column in columns if column in available]
    for batch in dataset.scanner(columns=columns, batch_size=20_000).to_batches():
        for row in batch.to_pylist():
            band = str(row.get("difficulty_band", "unknown"))
            if band not in heaps:
                continue
            if any(
                allowed and str(row.get(field, "")) not in allowed
                for field, allowed in filters.items()
            ):
                continue
            rank = int(stable_id(seed, row["sample_id"])[:16], 16)
            item = (-rank, serial, row)
            serial += 1
            heap = heaps[band]
            capacity = capacities[band]
            if capacity == 0:
                continue
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    strata: dict[str, tuple[int, list[tuple[int, dict]]]] = {}
    for band, target in targets.items():
        available_rows = sorted([(-rank, row) for rank, _, row in heaps[band]])
        if len(available_rows) < target:
            raise RuntimeError(
                f"pool {path} has only {len(available_rows)} eligible {band!r} rows; "
                f"requested {target}"
            )
        strata[band] = (target, available_rows)
    return strata


def _allocate(total: int, weights: list[float]) -> list[int]:
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("sampling weights must be non-negative with a positive sum")
    exact = [total * weight / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in exact]
    for index in sorted(
        range(len(weights)), key=lambda item: exact[item] - counts[item], reverse=True
    )[: total - sum(counts)]:
        counts[index] += 1
    return counts
