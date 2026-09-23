"""Re-run the production quality stages on saved real rollouts without generating again."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from calibrate_hygiene import _load_cases

from synthetic_sft.config import load_config
from synthetic_sft.generation import VLLMBatchPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("source_candidates", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("hygiene_cases", nargs="?", default="")
    args = parser.parse_args()

    output_path = args.output_dir / "candidates.parquet"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing review: {output_path}")
    config = load_config(args.config)
    rows = ds.dataset(str(args.source_candidates), format="parquet").to_table().to_pylist()
    previous = {
        row["candidate_id"]: {
            "score": row.get("quality_score"),
            "selected": row.get("productivity_filtered_eligible"),
        }
        for row in rows
    }
    predictor = VLLMBatchPredictor(config.model_dump(mode="json"), judge=True)
    predictor._quality(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    complete_rows = [{key: row.get(key) for key in columns} for row in rows]
    pq.write_table(pa.Table.from_pylist(complete_rows), output_path, compression="zstd")
    summary = {
        "source_candidates": str(args.source_candidates.resolve()),
        "config": str(args.config.resolve()),
        "count": len(rows),
        "scores": dict(sorted(Counter(row.get("quality_score") for row in rows).items())),
        "selected": sum(bool(row.get("productivity_filtered_eligible")) for row in rows),
        "changes": [
            {
                "sample_id": row["sample_id"],
                "candidate_id": row["candidate_id"],
                "old": previous[row["candidate_id"]],
                "new": {
                    "score": row.get("quality_score"),
                    "selected": row.get("productivity_filtered_eligible"),
                    "correctness": row.get("correctness_verdict"),
                    "hygiene": row.get("hygiene_status"),
                    "exclusion_reasons": json.loads(row.get("exclusion_reasons_json") or "[]"),
                },
            }
            for row in rows
            if previous[row["candidate_id"]]["score"] != row.get("quality_score")
            or previous[row["candidate_id"]]["selected"]
            != row.get("productivity_filtered_eligible")
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.hygiene_cases:
        cases = yaml.safe_load(Path(args.hygiene_cases).read_text(encoding="utf-8"))["cases"]
        case_rows = _load_cases(cases)
        predictor._run_hygiene(case_rows)
        hygiene_review = [
            {
                "id": case["id"],
                "expected": case.get("expected") or {},
                "observed": {
                    finding["category"]: finding["verdict"]
                    for finding in json.loads(row["hygiene_findings_json"])
                    if finding["category"] in (case.get("expected") or {})
                },
                "findings": json.loads(row["hygiene_findings_json"]),
                "raw_outputs": json.loads(row["hygiene_raw_outputs_json"]),
            }
            for case, row in zip(cases, case_rows, strict=True)
        ]
        (args.output_dir / "hygiene_calibration.json").write_text(
            json.dumps(hygiene_review, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"focused calibration: {len(hygiene_review)} real traces")
    print(json.dumps({key: value for key, value in summary.items() if key != "changes"}))
    print(f"{len(summary['changes'])} score or selection changes; {output_path}")


if __name__ == "__main__":
    main()
