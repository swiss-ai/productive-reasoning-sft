"""Re-run deterministic verification and quality selection on saved rollouts.

The original candidate dataset is never modified. Model judgments and hygiene findings
are reused; only source verification and the derived decision are recalculated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from synthetic_sft.config import PipelineConfig
from synthetic_sft.quality import FinalizeQuality, ParseAndVerify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("config", type=Path, help="run's manifests/config.resolved.yaml")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = PipelineConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    verifier = ParseAndVerify(
        config.source.adapter,
        config.source.params,
        config.quality.verifier_threshold,
    )
    finalize = FinalizeQuality(config.model_dump(mode="json"))
    rows = ds.dataset(str(args.candidates), format="parquet").to_table().to_pylist()
    old_errors = sum(row.get("verification_status") == "error" for row in rows)
    before = [
        (
            row.get("verification_status"),
            row.get("quality_score"),
            row.get("productivity_filtered_eligible"),
        )
        for row in rows
    ]
    for row in rows:
        verifier.verify(row)
        finalize(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    new_errors = sum(row.get("verification_status") == "error" for row in rows)
    changed = sum(
        (
            row.get("verification_status"),
            row.get("quality_score"),
            row.get("productivity_filtered_eligible"),
        )
        != prior
        for row, prior in zip(rows, before, strict=True)
    )
    print(
        f"{args.output}: {len(rows)} rows; verification errors {old_errors} -> {new_errors}; "
        f"changed decisions/statuses {changed}"
    )


if __name__ == "__main__":
    main()
