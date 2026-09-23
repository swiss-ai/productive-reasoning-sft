"""Build a self-contained, searchable HTML review from candidate Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("--title", default="Productive reasoning · pilot review")
    parser.add_argument(
        "--curation",
        type=Path,
        help="JSON list of {candidate_id, note, label}; show only these manually reviewed cases",
    )
    args = parser.parse_args()

    rows = ds.dataset(str(args.candidates), format="parquet").to_table().to_pylist()
    review = [_review_row(row) for row in rows]
    if args.curation:
        choices = json.loads(args.curation.read_text(encoding="utf-8"))
        if not isinstance(choices, list):
            parser.error("Curation must be a JSON list")
        by_id = {row["candidate_id"]: row for row in review}
        missing = [item["candidate_id"] for item in choices if item["candidate_id"] not in by_id]
        if missing:
            parser.error(f"Curated candidate IDs absent from Parquet: {missing}")
        ids = [item["candidate_id"] for item in choices]
        if len(ids) != len(set(ids)):
            parser.error("Curated candidate IDs must be unique")
        for item in choices:
            if not item.get("note") or not item.get("label"):
                parser.error(f"Curation needs a label and human note: {item['candidate_id']}")
            excerpt = item.get("excerpt")
            if excerpt and excerpt not in (by_id[item["candidate_id"]]["reasoning"] or ""):
                parser.error(f"Curated passage is not in reasoning: {item['candidate_id']}")
        review = [dict(by_id[item["candidate_id"]], annotation=item) for item in choices]
    data = json.dumps(review, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    title = _html_text(args.title)
    template = (Path(__file__).resolve().parents[1] / "review_site" / "template.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("__REVIEW_TITLE__", title).replace("__REVIEW_DATA__", data)
    html = html.replace(
        "__REVIEW_SUBTITLE__",
        "Individually inspected examples from the retained rollouts. This is a curated demonstration, not a random sample or an accuracy estimate."
        if args.curation
        else "Every rollout is retained. Selection is the pipeline decision, not human approval.",
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(f"{args.output_html}: {len(review)} candidates")


def _review_row(row: dict) -> dict:
    return {
        "sample_id": row.get("sample_id"),
        "candidate_id": row.get("candidate_id"),
        "source": row.get("source"),
        "model": row.get("model"),
        "question": row.get("user_prompt"),
        "system_prompt": row.get("system_prompt"),
        "reasoning": row.get("reasoning"),
        "response": row.get("response"),
        "draft": row.get("draft_generation"),
        "score": row.get("quality_score"),
        "selected": bool(row.get("productivity_filtered_eligible")),
        "correctness_only": bool(row.get("correctness_only_eligible")),
        "correctness": row.get("correctness_verdict"),
        "hygiene_status": row.get("hygiene_status"),
        "verification_status": row.get("verification_status"),
        "verifier_name": row.get("verifier_name"),
        "verifier_details": _parse_json(row.get("verifier_details_json"), {}),
        "reasoning_tokens": row.get("reasoning_num_tokens"),
        "response_tokens": row.get("response_num_tokens"),
        "finish_reason": row.get("finish_reason"),
        "exclusion_reasons": _parse_json(row.get("exclusion_reasons_json"), []),
        "quality": _parse_json(row.get("quality_details_json"), {}),
        "provenance": _parse_json(row.get("provenance_json"), {}),
        "critics": _parse_json(row.get("critic_analyses_json"), []),
        "annotation": None,
    }


def _parse_json(value: object, fallback: object) -> object:
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _html_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
