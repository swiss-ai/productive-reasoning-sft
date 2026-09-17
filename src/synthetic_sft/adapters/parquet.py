from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.dataset as pads

from synthetic_sft.adapters.base import SourceAdapter
from synthetic_sft.json_utils import canonical_json, json_object, stable_id, to_jsonable
from synthetic_sft.schemas import SeedRecord


class ParquetAdapter(SourceAdapter):
    """Map an arbitrary Parquet dataset into answer-free generic SFT seeds."""

    name = "parquet"

    def prepare(
        self, *, num_samples: int, seed: int, system_prompt: str | None
    ) -> Iterator[SeedRecord]:
        path = Path(self._required("path")).expanduser()
        prompt_field = str(self.params.get("prompt_field", "user_prompt"))
        id_field = self.params.get("id_field")
        system_field = self.params.get("system_field")
        provenance_field = self.params.get("provenance_field")
        provenance_fields = list(self.params.get("provenance_fields", []))
        source = str(self.params.get("source", f"parquet:{path.name}"))

        columns = [prompt_field]
        columns.extend(
            str(field) for field in (id_field, system_field, provenance_field) if field is not None
        )
        columns.extend(str(field) for field in provenance_fields)
        columns = list(dict.fromkeys(columns))

        dataset = pads.dataset(str(path), format="parquet")
        ordinal = 0
        for batch in dataset.scanner(columns=columns, batch_size=10_000).to_batches():
            for row in batch.to_pylist():
                prompt = str(row[prompt_field])
                if provenance_field is not None and row.get(str(provenance_field)) is not None:
                    provenance = json_object(
                        row[str(provenance_field)], field=str(provenance_field)
                    )
                else:
                    provenance = {
                        str(field): to_jsonable(row.get(str(field))) for field in provenance_fields
                    }
                provenance.setdefault("adapter", self.name)
                provenance.setdefault("input_path", str(path))
                sample_id = (
                    str(row[str(id_field)])
                    if id_field is not None and row.get(str(id_field)) is not None
                    else stable_id(self.name, path, seed, ordinal, prompt)
                )
                row_system = (
                    str(row[str(system_field)])
                    if system_field is not None and row.get(str(system_field)) is not None
                    else system_prompt
                )
                yield SeedRecord(
                    sample_id=sample_id,
                    user_prompt=prompt,
                    source=source,
                    provenance_json=canonical_json(provenance),
                    system_prompt=row_system,
                    verification_json=None,
                )
                ordinal += 1
                if ordinal >= num_samples:
                    return

    def _required(self, key: str) -> Any:
        value = self.params.get(key)
        if value is None:
            raise ValueError(f"parquet adapter requires params.{key}")
        return value
