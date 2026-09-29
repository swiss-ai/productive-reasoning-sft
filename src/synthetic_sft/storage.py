"""Typed, atomic candidate checkpoints, including failures and absent optional values."""

from __future__ import annotations

import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# A fixed schema prevents a first all-null shard from hiding later error messages,
# answer extractions, or verifier scores when reading the Parquet directory.
_STRINGS = """
sample_id candidate_id user_prompt system_prompt source provenance_json verification_json model
draft_generation draft_finish_reason raw_generation finish_reason reasoning response
generation_status generation_error batch_generation_error judge_generation_error
batch_judge_generation_error candidate_answer_raw_output candidate_answer_method
candidate_answer_error
reference_answer_raw_output reference_answer_method reference_answer_error answer_json
reference_answer_json verifier_details_json verifier_name verifier_error verification_status
critic_analyses_json judge_raw_output judge_first_raw_output judge_finish_reason
hygiene_findings_json
hygiene_raw_outputs_json judge_status correctness_verdict hygiene_status exclusion_reasons_json
quality_details_json
""".split()
_INTEGERS = """
candidate_index draft_num_input_tokens draft_num_generated_tokens
num_input_tokens num_generated_tokens
reasoning_num_tokens response_num_tokens judge_num_generated_tokens judge_retry_count quality_score
""".split()
_BOOLEANS = """
reasoning_parsed response_present generation_complete verifier_available correctness_only_eligible
productivity_filtered_eligible
""".split()
CANDIDATE_SCHEMA = pa.schema(
    [(name, pa.string()) for name in _STRINGS]
    + [(name, pa.int64()) for name in _INTEGERS]
    + [(name, pa.bool_()) for name in _BOOLEANS]
    + [(name, pa.float64()) for name in ("verifier_score", "verifier_threshold")]
)


def candidate_table(records):
    unknown = set().union(*(row.keys() for row in records)) - set(CANDIDATE_SCHEMA.names)
    if unknown:
        raise ValueError(f"candidate checkpoint schema is missing fields: {sorted(unknown)}")
    return pa.Table.from_pylist(records, schema=CANDIDATE_SCHEMA)


def write_candidates(records, directory: Path, compression="zstd") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"part-{uuid.uuid4().hex}.parquet"
    temporary = path.with_suffix(".pending")
    pq.write_table(candidate_table(records), temporary, compression=compression)
    temporary.replace(path)
    return path
