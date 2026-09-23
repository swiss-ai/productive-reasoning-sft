"""Export readable pass/reject reviews from a saved candidate Parquet dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="saved candidate snapshot")
    args = parser.parse_args()

    rows = ds.dataset(str(args.candidates), format="parquet").to_table().to_pylist()
    selected = [row for row in rows if row.get("productivity_filtered_eligible")]
    rejected = [row for row in rows if not row.get("productivity_filtered_eligible")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, group in (("selected", selected), ("rejected", rejected)):
        path = args.output_dir / f"{name}.md"
        path.write_text(_render(name, group, len(rows), args.label), encoding="utf-8")
        print(f"{path}: {len(group)} rows")


def _render(name: str, rows: list[dict], total: int, label: str) -> str:
    heading = (
        "Selected by the strict policy" if name == "selected" else "Rejected by the strict policy"
    )
    lines = [
        f"# {heading}",
        "",
        f"{len(rows)} of {total} retained rollouts from {label}. "
        "Machine decisions are not human approval; saved reasoning and responses are unchanged.",
        "",
    ]
    ordered = sorted(
        rows,
        key=lambda item: (-int(item.get("quality_score") or 0), item["sample_id"]),
    )
    for index, row in enumerate(ordered, 1):
        quality = json.loads(row["quality_details_json"])
        judge = quality["judge"].get("scores") or {}
        correctness = judge.get("correctness") or {}
        reasoning = judge.get("reasoning") or {}
        response = judge.get("response") or {}
        findings = [
            finding for finding in quality["hygiene"]["findings"] if finding["verdict"] != "clear"
        ]
        lines.extend(
            [
                f"## {index}. {row['sample_id'][:12]} — {row['source']}",
                "",
                f"- Candidate: `{row['candidate_id']}`",
                f"- Score: {row.get('quality_score')}; "
                f"correctness: {row.get('correctness_verdict')}",
                f"- Source verifier: {row.get('verification_status')}; "
                f"hygiene: {row.get('hygiene_status')}",
                f"- Verifier details: `{row.get('verifier_details_json') or '{}'}`",
                f"- Reasoning tokens: {row.get('reasoning_num_tokens')}; "
                f"response tokens: {row.get('response_num_tokens')}",
                f"- Finish: {row.get('finish_reason')}",
                f"- Correctness-only eligible: {row.get('correctness_only_eligible')}; "
                f"productivity-filtered eligible: {row.get('productivity_filtered_eligible')}",
                f"- Reasons: {_joined(json.loads(row.get('exclusion_reasons_json') or '[]'))}",
                f"- Judge scores: reasoning {reasoning.get('score', '—')}, "
                f"response {response.get('score', '—')}",
                f"- Judge correctness: {correctness.get('verdict', '—')}",
                f"- Judge rubric: {quality['judge'].get('rubric_version')}; "
                f"retries: {quality['judge'].get('retries', 0)}; "
                f"error: {quality['judge'].get('error') or 'none'}",
                f"- Extracted answer: {_answer(quality['answer'].get('candidate'))}; "
                f"source answer: {_answer(quality['answer'].get('reference'))}",
                "",
                "Question:",
                "",
                str(row.get("user_prompt") or "*(missing)*"),
                "",
                "Reasoning:",
                "",
                str(row.get("reasoning") or "*(missing)*"),
                "",
                "Final response:",
                "",
                str(row.get("response") or "*(missing)*"),
                "",
                "Judge feedback:",
                "",
                f"- Correctness: {correctness.get('feedback') or '—'}",
                f"- Reasoning: {reasoning.get('feedback') or '—'}",
                f"- Response: {response.get('feedback') or '—'}",
                "",
                "Focused findings:",
                "",
            ]
        )
        if findings:
            for finding in findings:
                lines.append(
                    f"- {finding['category']} — {finding['verdict']}: {finding['explanation']}"
                )
                for excerpt in finding.get("evidence") or []:
                    lines.append(f"  - Evidence: `{excerpt.replace('`', chr(39))}`")
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "<details><summary>Raw teacher draft</summary>",
                "",
                str(row.get("draft_generation") or "*(not saved)*"),
                "",
                "</details>",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def _joined(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _answer(value: dict | None) -> str:
    if not value or value.get("status") not in {"extracted", "conditional"}:
        return "unavailable"
    return str(value.get("value") or value.get("values") or "unavailable")


if __name__ == "__main__":
    main()
