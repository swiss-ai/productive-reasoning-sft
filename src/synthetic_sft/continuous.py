"""Keep one engine busy with independent trajectories at different pipeline stages.

The existing synchronous stage code runs in a bounded pool of orchestration threads.
All inference is submitted to one AsyncLLM event loop: no engine is shared unsafely
between threads, and no second copy of the quality policy is needed. A completed
trajectory immediately frees a slot for another prompt, without a stage-wide barrier.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd

from synthetic_sft.generation import VLLMBatchPredictor
from synthetic_sft.quality import ParseAndVerify
from synthetic_sft.storage import candidate_table, write_candidates


class ContinuousVLLMPredictor(VLLMBatchPredictor):
    def __init__(self, config, judge, checkpoint_dir=None):
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        super().__init__(config, judge)

    def _create_engine(self, engine_type, kwargs):
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        async def create():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))

        try:
            return asyncio.run_coroutine_threadsafe(create(), self._loop).result()
        except BaseException:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            raise

    def _chat(self, records, messages, sampling, template_kwargs, *, phase):
        if not records:
            return []
        started = time.perf_counter()
        tokenizer = self.llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, **template_kwargs
            )
            for conversation in messages
        ]

        async def request(row, prompt, params):
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.exceptions import EngineDeadError

            params.output_kind = RequestOutputKind.FINAL_ONLY
            request_id = f"{row['candidate_id']}:{phase}:{uuid.uuid4().hex}"
            try:
                result = None
                async for output in self.llm.generate(
                    {"prompt_token_ids": prompt}, params, request_id=request_id
                ):
                    result = output
                if result is None:
                    raise RuntimeError("engine returned no final output")
                return row, result
            except EngineDeadError:
                raise
            except Exception as exc:
                prefix = "generation" if phase in {"generation", "polish"} else "judge_generation"
                row[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
                return None

        async def wave():
            return await asyncio.gather(
                *(
                    request(row, prompt, params)
                    for row, prompt, params in zip(records, prompts, sampling, strict=True)
                )
            )

        results = asyncio.run_coroutine_threadsafe(wave(), self._loop).result()
        results = [result for result in results if result is not None]
        self._record_timing(phase, started, len(records), results)
        return results

    def _process_one(self, row):
        # Each worker owns its adapter state; only AsyncLLM and its event loop are shared.
        worker = copy.copy(self)
        if self.judge or self.fused_judge:
            worker._parse_and_verify = ParseAndVerify(
                self.config.source.adapter,
                self.config.source.params,
                self.config.quality.verifier_threshold,
            )
        return VLLMBatchPredictor.__call__(worker, pd.DataFrame.from_records([row]))

    def __call__(self, frame):
        """Yield completed rows while bounded workers continuously replenish the engine."""
        records = iter(frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"))
        completed = []
        flushed_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.config.model.inflight_candidates) as pool:
            pending = set()
            for _ in range(self.config.model.inflight_candidates):
                row = next(records, None)
                if row is None:
                    break
                pending.add(pool.submit(self._process_one, row))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    row = next(records, None)
                    if row is not None:
                        pending.add(pool.submit(self._process_one, row))
                    # Use records instead of inferred one-row Arrow types: optional fields
                    # must have identical types across accepted and rejected candidates.
                    completed.extend(
                        result.astype(object).where(pd.notna(result), None).to_dict(
                            orient="records"
                        )
                    )
                if (
                    len(completed) >= self.config.output.checkpoint_rows
                    or time.monotonic() - flushed_at >= self.config.output.checkpoint_seconds
                    or not pending
                ):
                    if self._checkpoint_dir is not None:
                        write_candidates(
                            completed, self._checkpoint_dir, self.config.output.compression
                        )
                        yield pd.DataFrame({"candidate_id": [r["candidate_id"] for r in completed]})
                    else:
                        yield candidate_table(completed).to_pandas()
                    completed = []
                    flushed_at = time.monotonic()

    def close(self):
        async def shutdown():
            self.llm.shutdown()

        asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
