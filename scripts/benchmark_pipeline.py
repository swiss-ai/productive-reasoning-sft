"""Measure the complete production pipeline on real prepared prompts, retaining every result."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from synthetic_sft.config import load_config
from synthetic_sft.generation import FanOutCandidates, VLLMBatchPredictor


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("seeds", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--execution", choices=["waves", "continuous"], default="waves")
    parser.add_argument("--inflight", type=int, default=32)
    parser.add_argument("--model")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "none"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config.model.max_num_seqs = args.max_num_seqs
    config.model.execution = args.execution
    config.model.inflight_candidates = args.inflight
    if args.model:
        config.model.model_source = args.model
    if args.tp:
        config.model.tensor_parallel_size = args.tp
    if args.effort:
        config.model.sampling.reasoning_effort = None if args.effort == "none" else args.effort
        if args.effort == "none":
            config.quality.judge.reasoning_effort = None
    config.model.max_num_batched_tokens = args.max_num_batched_tokens
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    seeds = ds.dataset(args.seeds, format="parquet").to_table().to_pylist()
    if not 0 <= args.shard_index < args.shards:
        parser.error("shard-index must be in [0, shards)")
    seeds = sorted(seeds, key=lambda row: row["sample_id"])[: args.samples * args.shards][
        args.shard_index :: args.shards
    ]
    fanout = FanOutCandidates(1, config.run.seed, config.model.model_source)
    records = [fanout(seed)[0] for seed in seeds]
    (args.output / "sample_ids.json").write_text(json.dumps([row["sample_id"] for row in records]))
    started = time.perf_counter()
    predictor_type = VLLMBatchPredictor
    if args.execution == "continuous":
        from synthetic_sft.continuous import ContinuousVLLMPredictor

        predictor_type = ContinuousVLLMPredictor
    predictor = predictor_type(config.model_dump(mode="json"), judge=False)
    startup_seconds = time.perf_counter() - started
    import vllm

    print(
        json.dumps(
            {"event": "startup", "seconds": startup_seconds, "vllm_version": vllm.__version__}
        ),
        flush=True,
    )
    for batch_size in args.batch_sizes:
        target = args.output / f"batch-{batch_size}"
        target.mkdir()
        started = time.perf_counter()
        started_at = time.time()
        all_rows = []
        for offset in range(0, len(records), batch_size):
            frame = pd.DataFrame.from_records(records[offset : offset + batch_size])
            results = predictor(frame)
            if isinstance(results, pd.DataFrame):
                results = [results]
            for index, result in enumerate(results):
                pq.write_table(
                    pa.Table.from_pandas(result, preserve_index=False),
                    target / f"part-{offset:06d}-{index:06d}.parquet",
                    compression="zstd",
                )
                all_rows.extend(result.to_dict(orient="records"))
        elapsed = time.perf_counter() - started
        eligible = sum(bool(row["productivity_filtered_eligible"]) for row in all_rows)
        generated = sum(bool(row.get("draft_generation")) for row in all_rows)
        complete = sum(row.get("generation_status") == "ok" for row in all_rows)
        summary = {
            "event": "benchmark",
            "valid": generated > 0,
            "model": config.model.model_source,
            "execution": config.model.execution,
            "gpus": config.model.tensor_parallel_size,
            "startup_seconds": startup_seconds,
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
            "batch_size": batch_size,
            "samples": len(all_rows),
            "seconds": elapsed,
            "samples_per_hour": len(all_rows) * 3600 / elapsed,
            "generated": generated,
            "complete": complete,
            "complete_per_hour": complete * 3600 / elapsed,
            "eligible": eligible,
            "eligible_per_hour": eligible * 3600 / elapsed,
            "scores": dict(Counter(str(row["quality_score"]) for row in all_rows)),
            "hygiene": dict(Counter(row["hygiene_status"] for row in all_rows)),
        }
        (target / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if not generated:
            raise RuntimeError(
                "no successful generations; this is not a valid throughput benchmark"
            )
    if hasattr(predictor, "close"):
        predictor.close()


if __name__ == "__main__":
    main()
