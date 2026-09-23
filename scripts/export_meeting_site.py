"""Export retained Parquet rollouts into the small static site under docs/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


def _json(value: object, fallback: object) -> object:
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except ValueError:
        return fallback


def _record(row: dict, cohort: str) -> dict:
    quality = _json(row.get("quality_details_json"), {})
    judge = quality.get("judge") or {}
    scores = judge.get("scores") or {}
    return {
        "id": row.get("candidate_id"),
        "sample_id": row.get("sample_id"),
        "cohort": cohort,
        "source": row.get("source"),
        "question": row.get("user_prompt") or "",
        "reasoning": row.get("reasoning") or "",
        "response": row.get("response") or "",
        "score": row.get("quality_score"),
        "selected": bool(row.get("productivity_filtered_eligible")),
        "correctness_only": bool(row.get("correctness_only_eligible")),
        "correctness": row.get("correctness_verdict"),
        "hygiene": row.get("hygiene_status"),
        "verification": row.get("verification_status"),
        "verifier": row.get("verifier_name"),
        "finish": row.get("finish_reason"),
        "reasoning_tokens": row.get("reasoning_num_tokens"),
        "response_tokens": row.get("response_num_tokens"),
        "exclusions": _json(row.get("exclusion_reasons_json"), []),
        "findings": [
            finding
            for finding in (quality.get("hygiene") or {}).get("findings", [])
            if finding.get("verdict") != "clear"
        ],
        "judge": {
            label: {
                "score": (scores.get(label) or {}).get("score"),
                "verdict": (scores.get(label) or {}).get("verdict"),
                "feedback": (scores.get(label) or {}).get("feedback"),
            }
            for label in ("correctness", "reasoning", "response")
        },
    }


def _load(path: Path, cohort: str) -> list[dict]:
    return [
        _record(row, cohort)
        for row in ds.dataset(str(path), format="parquet").to_table().to_pylist()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--curation", type=Path, default=Path("docs/curation.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/data.js"))
    args = parser.parse_args()

    current = _load(args.current, "New 50-rollout batch")
    earlier = _load(args.earlier, "Earlier 24-prompt pilot")
    by_id = {item["id"]: item for item in current + earlier}
    if len(by_id) != len(current) + len(earlier):
        parser.error("Candidate IDs overlap across the two batches")

    annotations = json.loads(args.curation.read_text(encoding="utf-8"))
    if len({item["id"] for item in annotations}) != len(annotations):
        parser.error("Curated candidate IDs must be unique")
    cases = []
    for annotation in annotations:
        candidate = by_id.get(annotation["id"])
        if candidate is None:
            parser.error(f"Curated candidate not found: {annotation['id']}")
        if annotation["cohort"] != candidate["cohort"]:
            parser.error(f"Wrong cohort label: {annotation['id']}")
        quote = annotation.get("quote")
        field = annotation.get("quote_field", "reasoning")
        if quote and (field not in ("reasoning", "response") or quote not in candidate[field]):
            parser.error(f"Quoted passage does not occur in {field}: {annotation['id']}")
        cases.append({**candidate, **annotation})

    payload = {
        "batch": {
            "count": len(current),
            "passed": sum(item["selected"] for item in current),
            "rejected": sum(not item["selected"] for item in current),
            "earlier_count": len(earlier),
        },
        "cases": cases,
        "audit": current,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"window.SFT_REVIEW_DATA = {encoded};\n", encoding="utf-8")
    print(f"{args.output}: {len(cases)} annotated cases, {len(current)} audit rows")


if __name__ == "__main__":
    main()
