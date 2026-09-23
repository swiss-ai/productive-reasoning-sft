"""Run the production focused judges on hand-reviewed saved rollouts only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import yaml

from synthetic_sft.config import load_config
from synthetic_sft.generation import VLLMBatchPredictor
from synthetic_sft.hygiene import SEMANTIC_CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))["cases"]
    rows = _load_cases(cases)
    predictor = VLLMBatchPredictor(config.model_dump(mode="json"), judge=True)
    predictor._run_hygiene(rows)

    comparisons: Counter[tuple[str, str, str]] = Counter()
    results: list[dict[str, Any]] = []
    for case, row in zip(cases, rows, strict=True):
        findings = json.loads(row["hygiene_findings_json"])
        by_category = {item["category"]: item for item in findings}
        expected = case.get("expected") or {}
        for category, label in expected.items():
            comparisons[(category, label, by_category[category]["verdict"])] += 1
        results.append(
            {
                "id": case["id"],
                "source_dataset": case["dataset"],
                "sample_id": row["sample_id"],
                "field": case["field"],
                "expected": expected,
                "note": case.get("note"),
                "findings": findings,
                "raw_outputs": json.loads(row["hygiene_raw_outputs_json"]),
            }
        )
    payload = {
        "model": config.quality.judge.model_source or config.model.model_source,
        "hygiene_max_tokens": config.quality.judge.hygiene_max_tokens,
        "comparisons": [
            {"category": category, "expected": expected, "observed": observed, "count": count}
            for (category, expected, observed), count in sorted(comparisons.items())
        ],
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    for comparison in payload["comparisons"]:
        print(comparison)


def _load_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache: dict[str, list[dict[str, Any]]] = {}
    rows = []
    for case in cases:
        path = str(Path(case["dataset"]).resolve())
        if path not in cache:
            cache[path] = ds.dataset(path, format="parquet").to_table().to_pylist()
        matching = [
            row
            for row in cache[path]
            if str(row.get("sample_id", "")).startswith(case["sample_id_prefix"])
        ]
        if len(matching) != 1:
            raise ValueError(f"{case['id']}: expected one saved rollout, found {len(matching)}")
        original = matching[0]
        reasoning = original.get(case["field"])
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(f"{case['id']}: no reasoning in {case['field']}")
        unknown = set(case.get("expected") or {}) - set(SEMANTIC_CATEGORIES)
        if unknown:
            raise ValueError(f"{case['id']}: unknown categories {sorted(unknown)}")
        rows.append(
            {
                "sample_id": original["sample_id"],
                "candidate_id": f"{original['candidate_id']}:calibration:{case['id']}",
                "user_prompt": original.get("user_prompt"),
                "reasoning": reasoning,
                "response": original.get("response"),
            }
        )
    return rows


if __name__ == "__main__":
    main()
