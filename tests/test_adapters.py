from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from synthetic_sft.adapters.base import SourceAdapter
from synthetic_sft.adapters.registry import create_adapter, register_adapter
from synthetic_sft.quality import ParseAndVerify
from synthetic_sft.schemas import SeedRecord


class NoAnswerAdapter(SourceAdapter):
    name = "test_no_answer"

    def prepare(self, *, num_samples: int, seed: int, system_prompt: str | None):
        for index in range(num_samples):
            yield SeedRecord(
                sample_id=f"id-{index}",
                user_prompt=f"prompt {index}",
                source=self.name,
                provenance_json=json.dumps({"index": index}),
                system_prompt=system_prompt,
            )


def _register_no_answer() -> None:
    try:
        register_adapter(NoAnswerAdapter.name, NoAnswerAdapter)
    except ValueError:
        pass


def test_adapter_without_answer_has_natural_seed_contract() -> None:
    _register_no_answer()
    adapter = create_adapter(NoAnswerAdapter.name)
    record = next(adapter.prepare(num_samples=1, seed=1, system_prompt=None))
    assert record.verification_json is None
    assert not adapter.supports_verification(record.model_dump())


def test_parse_and_verify_skips_missing_capability() -> None:
    _register_no_answer()
    annotator = ParseAndVerify(NoAnswerAdapter.name, {}, 1.0)
    row = annotator(
        {
            "sample_id": "id",
            "raw_generation": "a direct answer",
            "verification_json": None,
        }
    )
    assert row["response"] == "a direct answer"
    assert row["verifier_available"] is False
    assert row["verifier_score"] is None


def test_parquet_adapter_requires_no_answer(tmp_path) -> None:
    path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": "x", "question": "Why?", "difficulty": "open-ended"}]),
        path,
    )
    adapter = create_adapter(
        "parquet",
        {
            "path": str(path),
            "prompt_field": "question",
            "id_field": "id",
            "provenance_fields": ["difficulty"],
        },
    )
    row = next(adapter.prepare(num_samples=1, seed=42, system_prompt=None))
    assert row.sample_id == "x"
    assert row.verification_json is None
    assert json.loads(row.provenance_json)["difficulty"] == "open-ended"
