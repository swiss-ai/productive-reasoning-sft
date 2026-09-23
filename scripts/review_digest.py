"""Write a compact index for manual review of candidate rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _json(value: object, fallback: object) -> object:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = ds.dataset(str(args.candidates), format="parquet").to_table().to_pylist()
    rows.sort(
        key=lambda row: (
            not row.get("productivity_filtered_eligible"),
            row.get("source") or "",
            row.get("candidate_id") or "",
        )
    )
    lines = [
        "# Candidate review index",
        "",
        "This is a navigation aid, not a human validation or filter-accuracy estimate. Read the full trace before endorsing a verdict.",
        "",
    ]
    for index, row in enumerate(rows, 1):
        details = _json(row.get("quality_details_json"), {})
        scores = details.get("judge", {}).get("scores") or {}
        salient = [
            finding for finding in details.get("hygiene", {}).get("findings", [])
            if finding.get("verdict") != "clear"
        ]
        findings_text = "; ".join(
            f"{item.get('category')}: {item.get('verdict')} — "
            f"{_text(item.get('explanation'))[:200]}"
            for item in salient
        ) or "none"
        lines.extend(
            [
                f"## {index}. {'PASS' if row.get('productivity_filtered_eligible') else 'REJECT'} · {row.get('candidate_id')}",
                "",
                f"- Source: {row.get('source')} · score {row.get('quality_score')} · correctness {row.get('correctness_verdict')} · hygiene {row.get('hygiene_status')}",
                f"- Question: {_text(row.get('user_prompt'))[:500]}",
                f"- Final: {_text(row.get('response'))[:300]}",
                f"- Verification: {row.get('verification_status')} ({row.get('verifier_name')}) · {_text(row.get('verifier_details_json'))[:300]}",
                f"- Judge correctness: {_text((scores.get('correctness') or {}).get('feedback'))[:400]}",
                f"- Judge reasoning: {_text((scores.get('reasoning') or {}).get('feedback'))[:400]}",
                f"- Judge response: {_text((scores.get('response') or {}).get('feedback'))[:300]}",
                f"- Focused findings: {findings_text}",
                f"- Exclusion: {', '.join(_json(row.get('exclusion_reasons_json'), [])) or 'none'}",
                "",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"{args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
