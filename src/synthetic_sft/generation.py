from __future__ import annotations

import json
from typing import Any

import pandas as pd

from synthetic_sft.config import PipelineConfig
from synthetic_sft.json_utils import stable_id
from synthetic_sft.quality import judge_prompt


class FanOutCandidates:
    def __init__(self, rollouts_per_prompt: int, run_seed: int, model: str) -> None:
        self.rollouts_per_prompt = rollouts_per_prompt
        self.run_seed = run_seed
        self.model = model

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "candidate_index": index,
                "candidate_id": stable_id(row["sample_id"], self.run_seed, index),
                "model": self.model,
            }
            for index in range(self.rollouts_per_prompt)
        ]


class VLLMBatchPredictor:
    """One long-lived vLLM engine actor processing Ray Data batches."""

    def __init__(self, config: dict[str, Any], judge: bool) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is unavailable; run generation in the NVIDIA vLLM container"
            ) from exc

        self._sampling_type = SamplingParams
        self.config = PipelineConfig.model_validate(config)
        self.judge = judge
        model = self.config.model
        judge_config = self.config.quality.judge
        model_source = (
            judge_config.model_source if judge and judge_config.model_source else model.model_source
        )
        kwargs: dict[str, Any] = {
            "model": model_source,
            "dtype": model.dtype,
            "tensor_parallel_size": model.tensor_parallel_size,
            "max_model_len": model.max_model_len,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "trust_remote_code": model.trust_remote_code,
            "enable_chunked_prefill": True,
            "seed": self.config.run.seed,
            "generation_config": "vllm",
            "disable_log_stats": False,
        }
        if model.revision:
            kwargs["revision"] = model.revision
        self.llm = LLM(**kwargs)

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        records = frame.to_dict(orient="records")
        messages = [self._messages(row) for row in records]
        sampling = [self._sampling(row, phase="judge" if self.judge else "draft") for row in records]
        template_kwargs = (
            {"enable_thinking": False}
            if self.judge
            else {
                "enable_thinking": True,
                "reasoning_effort": self.config.model.sampling.reasoning_effort,
            }
        )
        results = self._chat(records, messages, sampling, template_kwargs)
        for row, output in results:
            candidate = output.outputs[0]
            if self.judge:
                row["judge_raw_output"] = candidate.text
            else:
                row["draft_generation"] = candidate.text
                row["draft_finish_reason"] = candidate.finish_reason
                row["draft_num_input_tokens"] = len(output.prompt_token_ids or [])
                row["draft_num_generated_tokens"] = len(candidate.token_ids or [])
        if not self.judge:
            self._polish(records)
        return pd.DataFrame.from_records(records)

    def _chat(self, records, messages, sampling, template_kwargs):
        try:
            outputs = self.llm.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=False,
                chat_template_kwargs=template_kwargs,
            )
            return list(zip(records, outputs, strict=True))
        except Exception as batch_error:
            results = []
            for row, row_messages, row_sampling in zip(records, messages, sampling, strict=True):
                try:
                    output = self.llm.chat(
                        row_messages,
                        sampling_params=row_sampling,
                        use_tqdm=False,
                        chat_template_kwargs=template_kwargs,
                    )[0]
                    results.append((row, output))
                except Exception as exc:
                    prefix = "judge_generation" if self.judge else "generation"
                    row[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
                    row[f"batch_{prefix}_error"] = f"{type(batch_error).__name__}: {batch_error}"
            return results

    def _polish(self, records: list[dict[str, Any]]) -> None:
        ready = [row for row in records if row.get("draft_generation")]
        if not self.config.generation.polish:
            for row in ready:
                row["raw_generation"] = row["draft_generation"]
                row["finish_reason"] = row["draft_finish_reason"]
                row["num_input_tokens"] = row["draft_num_input_tokens"]
                row["num_generated_tokens"] = row["draft_num_generated_tokens"]
            return
        messages = [self._polish_messages(row) for row in ready]
        sampling = [self._sampling(row, phase="polish") for row in ready]
        results = self._chat(ready, messages, sampling, {"enable_thinking": False})
        for row, output in results:
            candidate = output.outputs[0]
            row["raw_generation"] = candidate.text
            row["finish_reason"] = candidate.finish_reason
            row["num_input_tokens"] = len(output.prompt_token_ids or [])
            row["num_generated_tokens"] = len(candidate.token_ids or [])

    def _messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        if self.judge:
            return [
                {
                    "role": "user",
                    "content": judge_prompt(row, self.config.quality.judge.rubric_version),
                }
            ]
        messages = []
        if row.get("system_prompt"):
            messages.append({"role": "system", "content": str(row["system_prompt"])})
        messages.append({"role": "user", "content": str(row["user_prompt"])})
        return messages

    def _polish_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        reference = _reference_answer(row.get("verification_json"))
        prompt = f"""Rewrite scratch work into expert-quality supervised fine-tuning data.

Solve and check the problem yourself. Use the scratch work only when it is sound; discard it when
it is long, confused, or contradictory. The trusted verification target may help detect an error,
but the reasoning must independently establish the result. Never mention scratch work, editing,
the prompt, a reference answer, a target, a checker, or formatting instructions. Remove self-talk,
backtracking, repeated calculations, and failed attempts. Give a concise derivation with every
necessary logical step and no unnecessary ones; do not narrate exhaustive search. The final
response must obey the user's requested answer format exactly.

Return exactly these two tagged sections with no text before or after them:
<reasoning>
clean derivation
</reasoning>
<response>
final response
</response>

User request:
{row.get("user_prompt", "")}

Scratch work:
{row.get("draft_generation", "")}

Trusted verification target (never mention this in the output):
{reference}
"""
        return [{"role": "user", "content": prompt}]

    def _sampling(self, row: dict[str, Any], *, phase: str):
        if phase == "judge":
            return self._sampling_type(
                temperature=0.0,
                max_tokens=self.config.quality.judge.max_tokens,
                seed=_candidate_seed(str(row["candidate_id"]), suffix="judge"),
            )
        if phase == "polish":
            return self._sampling_type(
                temperature=0.0,
                max_tokens=self.config.generation.polish_max_tokens,
                seed=_candidate_seed(str(row["candidate_id"]), suffix="polish"),
            )
        params = self.config.model.sampling.model_dump(exclude={"reasoning_effort"})
        return self._sampling_type(
            **params,
            seed=_candidate_seed(str(row["candidate_id"]), suffix="generate"),
        )


def _reference_answer(raw: Any) -> str:
    if not raw:
        return "(none)"
    try:
        payload = json.loads(str(raw))
        answer = payload.get("entry", {}).get("answer")
        return "(none)" if answer is None else json.dumps(answer, ensure_ascii=False)
    except (TypeError, ValueError, AttributeError):
        return "(unreadable)"


def build_vllm_processor(config: PipelineConfig, *, judge: bool, concurrency: int):
    batch_size = config.quality.judge.batch_size if judge else config.model.batch_size
    config_payload = config.model_dump(mode="json")

    def process(dataset):
        return dataset.map_batches(
            VLLMBatchPredictor,
            batch_format="pandas",
            batch_size=batch_size,
            concurrency=(concurrency, concurrency),
            num_cpus=1,
            num_gpus=config.model.tensor_parallel_size,
            zero_copy_batch=False,
            fn_constructor_kwargs={"config": config_payload, "judge": judge},
        )

    return process


def _candidate_seed(candidate_id: str, *, suffix: str) -> int:
    return int(stable_id(candidate_id, suffix)[:8], 16)
