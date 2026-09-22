from __future__ import annotations

import json
from typing import Any

import pandas as pd

from synthetic_sft.config import PipelineConfig
from synthetic_sft.hygiene import (
    SEMANTIC_CATEGORIES,
    hygiene_prompt,
    parse_hygiene_assessment,
    repeated_span_signal,
)
from synthetic_sft.json_utils import canonical_json, stable_id
from synthetic_sft.quality import (
    FinalizeQuality,
    ParseAndVerify,
    answer_extraction_prompt,
    arbitration_prompt,
    critic_prompt,
    parse_answer_extraction,
    split_reasoning,
)
from synthetic_sft.schemas import AnswerExtraction, JudgeScores, ModelHygieneAssessment


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


class ExcludeCandidateIds:
    """Skip candidates already durably written by an interrupted generation stage."""

    def __init__(self, candidate_ids: set[str]) -> None:
        self.candidate_ids = candidate_ids

    def __call__(self, row: dict[str, Any]) -> bool:
        return str(row["candidate_id"]) not in self.candidate_ids


class VLLMBatchPredictor:
    """One long-lived vLLM engine actor processing Ray Data batches."""

    def __init__(self, config: dict[str, Any], judge: bool) -> None:
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is unavailable; run generation in the NVIDIA vLLM container"
            ) from exc

        self._sampling_type = SamplingParams
        self._structured_outputs_type = StructuredOutputsParams
        self.config = PipelineConfig.model_validate(config)
        self.judge = judge
        self.fused_judge = not judge and can_fuse_judge(self.config)
        if self.fused_judge or judge:
            self._parse_and_verify = ParseAndVerify(
                self.config.source.adapter,
                self.config.source.params,
                self.config.quality.verifier_threshold,
            )
            self._finalize_quality = FinalizeQuality(config)
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
            "max_num_seqs": model.max_num_seqs,
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
        if self.judge:
            self._quality(records)
            return pd.DataFrame.from_records(records)
        messages = [self._messages(row) for row in records]
        sampling = [self._sampling(row, phase="draft") for row in records]
        template_kwargs = {
            "enable_thinking": True,
            "reasoning_effort": self.config.model.sampling.reasoning_effort,
        }
        results = self._chat(records, messages, sampling, template_kwargs, phase="generation")
        for row, output in results:
            candidate = output.outputs[0]
            row["draft_generation"] = candidate.text
            row["draft_finish_reason"] = candidate.finish_reason
            row["draft_num_input_tokens"] = len(output.prompt_token_ids or [])
            row["draft_num_generated_tokens"] = len(candidate.token_ids or [])
        self._polish(records)
        tokenizer = self.llm.get_tokenizer()
        for row in records:
            reasoning, response, _ = split_reasoning(row.get("raw_generation"))
            row["reasoning_num_tokens"] = (
                len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0
            )
            row["response_num_tokens"] = (
                len(tokenizer.encode(response, add_special_tokens=False)) if response else 0
            )
        if self.fused_judge:
            self._quality(records)
        return pd.DataFrame.from_records(records)

    def _chat(self, records, messages, sampling, template_kwargs, *, phase: str):
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
                    prefix = (
                        "judge_generation"
                        if phase in {"answer_extraction", "critique", "arbitration", "hygiene"}
                        else "generation"
                    )
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
        results = self._chat(
            ready,
            messages,
            sampling,
            {"enable_thinking": False},
            phase="polish",
        )
        for row, output in results:
            candidate = output.outputs[0]
            row["raw_generation"] = candidate.text
            row["finish_reason"] = candidate.finish_reason
            row["num_input_tokens"] = len(output.prompt_token_ids or [])
            row["num_generated_tokens"] = len(candidate.token_ids or [])

    def _quality(self, records: list[dict[str, Any]]) -> None:
        for row in records:
            self._parse_and_verify.parse(row)
        self._extract_answers(records)
        # Structured extraction makes heterogeneous source answers safe for deterministic
        # comparison. Native source checkers may also use the extracted answer as a candidate.
        for row in records:
            self._parse_and_verify.verify(row)
        self._run_critiques(records)
        self._arbitrate(records)
        self._run_hygiene(records)
        for row in records:
            self._finalize_quality(row)

    def _run_hygiene(self, records: list[dict[str, Any]]) -> None:
        """One batched wave: a narrow, structured request per semantic failure category."""
        requests: list[dict[str, Any]] = []
        messages = []
        sampling = []
        max_tokens = self.config.quality.judge.hygiene_max_tokens
        for index, row in enumerate(records):
            signal = repeated_span_signal(str(row.get("reasoning") or ""))
            signal_text = signal[0][:500] if signal is not None else ""
            for category in SEMANTIC_CATEGORIES:
                request = {
                    "candidate_id": f"{row['candidate_id']}:hygiene:{category}",
                    "row_index": index,
                    "category": category,
                }
                prompt = hygiene_prompt(
                    row, category, signal_text if category == "repetition" else ""
                )
                fitted = self._fit_prompt(prompt, max_tokens)
                request["prompt_truncated"] = fitted != prompt
                requests.append(request)
                messages.append([{"role": "user", "content": fitted}])
                sampling.append(
                    self._structured_sampling(
                        request, ModelHygieneAssessment, max_tokens, "hygiene"
                    )
                )
        results = self._chat(
            requests,
            messages,
            sampling,
            {"enable_thinking": False},
            phase="hygiene",
        )
        findings: list[list[dict[str, Any]]] = [[] for _ in records]
        for request, output in results:
            index = int(request["row_index"])
            finding = parse_hygiene_assessment(
                output.outputs[0].text,
                category=request["category"],
                reasoning=str(records[index].get("reasoning") or ""),
                response=str(records[index].get("response") or ""),
                truncated=bool(request["prompt_truncated"]),
            )
            findings[index].append(finding.model_dump(mode="json"))
        for index, row in enumerate(records):
            row["hygiene_findings_json"] = canonical_json(findings[index])

    def _extract_answers(self, records: list[dict[str, Any]]) -> None:
        requests: list[dict[str, Any]] = []
        messages = []
        sampling = []
        for index, row in enumerate(records):
            response = str(row.get("response") or "")
            request = {
                "candidate_id": f"{row['candidate_id']}:answer:candidate",
                "row_index": index,
                "target": "candidate",
            }
            requests.append(request)
            prompt = answer_extraction_prompt(row, response, reference=False)
            messages.append([{"role": "user", "content": self._fit_prompt(prompt, 512)}])
            sampling.append(self._structured_sampling(request, AnswerExtraction, 512, "answer"))
            reference = _reference_text(row.get("verification_json"))
            if reference is not None:
                request = {
                    "candidate_id": f"{row['candidate_id']}:answer:reference",
                    "row_index": index,
                    "target": "reference",
                }
                requests.append(request)
                prompt = answer_extraction_prompt(row, reference, reference=True)
                messages.append([{"role": "user", "content": self._fit_prompt(prompt, 512)}])
                sampling.append(
                    self._structured_sampling(request, AnswerExtraction, 512, "reference")
                )
        results = self._chat(
            requests,
            messages,
            sampling,
            {"enable_thinking": False},
            phase="answer_extraction",
        )
        seen: set[tuple[int, str]] = set()
        for request, output in results:
            index, target = int(request["row_index"]), str(request["target"])
            raw = output.outputs[0].text
            extracted, error = parse_answer_extraction(raw)
            records[index][f"{target}_answer_raw_output"] = raw
            if extracted is not None:
                field = "answer_json" if target == "candidate" else "reference_answer_json"
                records[index][field] = canonical_json(extracted.model_dump(mode="json"))
                records[index][f"{target}_answer_method"] = "model"
            if error is not None:
                records[index][f"{target}_answer_error"] = error
            seen.add((index, target))
        for request in requests:
            key = (int(request["row_index"]), str(request["target"]))
            if key not in seen:
                records[key[0]][f"{key[1]}_answer_error"] = request.get(
                    "judge_generation_error", "answer extraction failed"
                )
        for row in records:
            fallback = _atomic_reference_fallback(row)
            if fallback is not None:
                row["reference_answer_json"] = canonical_json(fallback.model_dump(mode="json"))
                row["reference_answer_method"] = "atomic_source_fallback"

    def _run_critiques(self, records: list[dict[str, Any]]) -> None:
        requests: list[dict[str, Any]] = []
        messages = []
        sampling = []
        judge = self.config.quality.judge
        for index, row in enumerate(records):
            prompt = self._fit_prompt(critic_prompt(row), judge.analysis_max_tokens)
            for sample_index in range(judge.analysis_samples):
                request = {
                    "candidate_id": f"{row['candidate_id']}:critique:{sample_index}",
                    "row_index": index,
                    "sample_index": sample_index,
                }
                requests.append(request)
                messages.append([{"role": "user", "content": prompt}])
                sampling.append(
                    self._sampling_type(
                        temperature=0.3,
                        top_p=0.95,
                        max_tokens=judge.analysis_max_tokens,
                        seed=_candidate_seed(str(request["candidate_id"]), suffix="critic"),
                    )
                )
        results = self._chat(
            requests,
            messages,
            sampling,
            {
                "enable_thinking": True,
                "reasoning_effort": judge.reasoning_effort,
            },
            phase="critique",
        )
        analyses: list[list[tuple[int, str]]] = [[] for _ in records]
        for request, output in results:
            analyses[int(request["row_index"])].append(
                (int(request["sample_index"]), output.outputs[0].text)
            )
        for index, values in enumerate(analyses):
            ordered = [text for _, text in sorted(values)]
            records[index]["critic_analyses_json"] = canonical_json(ordered)
            if len(ordered) != judge.analysis_samples:
                records[index]["judge_generation_error"] = (
                    f"received {len(ordered)} of {judge.analysis_samples} critical analyses"
                )

    def _arbitrate(self, records: list[dict[str, Any]]) -> None:
        messages = []
        sampling = []
        judge = self.config.quality.judge
        for row in records:
            analyses = _json_string_list(row.get("critic_analyses_json"))
            prompt = arbitration_prompt(row, analyses, judge.rubric_version)
            messages.append(
                [{"role": "user", "content": self._fit_prompt(prompt, judge.max_tokens)}]
            )
            sampling.append(self._structured_sampling(row, JudgeScores, judge.max_tokens, "judge"))
        results = self._chat(
            records,
            messages,
            sampling,
            {"enable_thinking": False},
            phase="arbitration",
        )
        for row, output in results:
            row["judge_raw_output"] = output.outputs[0].text

    def _messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        messages = []
        if row.get("system_prompt"):
            messages.append({"role": "system", "content": str(row["system_prompt"])})
        messages.append({"role": "user", "content": str(row["user_prompt"])})
        return messages

    def _polish_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        prefix = f"""Rewrite scratch work into expert-quality supervised fine-tuning data.

Solve and check the problem yourself. Use the scratch work only when it is sound; discard it when
it is long, confused, or contradictory. Independently establish every claim from the user request;
do not infer an answer merely because it appears in the scratch work. Never mention scratch work,
editing, the prompt, a reference answer, a target, a checker, or formatting instructions. Remove
self-talk, backtracking, repeated calculations, and failed attempts. Give a concise derivation with
every necessary logical step and no unnecessary ones. Do not narrate plans, candidate approaches,
or exhaustive search unless the search itself is the shortest proof. Do not appeal to tables,
software, external sources, or "known" values without deriving the needed result. Prefer the
shortest rigorous explanation that teaches the solution. The final response must obey the user's
requested answer format exactly. If the request is inconsistent or underdetermined, explain the
specific defect and stop; never invent a correction, add assumptions, or solve speculative variants.

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
"""
        output_tokens = self.config.generation.polish_max_tokens
        reserve = self.config.model.max_model_len - output_tokens - self._token_count(prefix) - 256
        scratch = self._truncate_text(str(row.get("draft_generation") or ""), max(0, reserve))
        prompt = f"{prefix}{scratch}"
        return [{"role": "user", "content": prompt}]

    def _structured_sampling(self, row, schema, max_tokens: int, suffix: str):
        return self._sampling_type(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=_candidate_seed(str(row["candidate_id"]), suffix=suffix),
            structured_outputs=self._structured_outputs_type(json=schema.model_json_schema()),
        )

    def _token_count(self, text: str) -> int:
        return len(self.llm.get_tokenizer().encode(text, add_special_tokens=False))

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokenizer = self.llm.get_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        marker = "\n\n[content truncated to fit model context]\n\n"
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        keep = max(0, max_tokens - len(marker_ids))
        left = keep * 2 // 5
        right = keep - left
        tail = tokenizer.decode(token_ids[-right:]) if right else ""
        return tokenizer.decode(token_ids[:left]) + marker + tail

    def _fit_prompt(self, prompt: str, output_tokens: int) -> str:
        budget = max(1, self.config.model.max_model_len - output_tokens - 256)
        return self._truncate_text(prompt, budget)

    def _sampling(self, row: dict[str, Any], *, phase: str):
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


def can_fuse_judge(config: PipelineConfig) -> bool:
    judge = config.quality.judge
    return judge.enabled and (
        judge.model_source is None or judge.model_source == config.model.model_source
    )


def _candidate_seed(candidate_id: str, *, suffix: str) -> int:
    return int(stable_id(candidate_id, suffix)[:8], 16)


def _reference_text(raw: Any) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
        answer = (payload.get("entry") or {}).get("answer")
        return None if answer is None else str(answer)
    except (TypeError, ValueError, AttributeError):
        return None


def _json_string_list(raw: Any) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
        return [str(item) for item in value] if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _atomic_reference_fallback(row: dict[str, Any]) -> AnswerExtraction | None:
    """Recover a bare source answer when the model incorrectly calls it absent."""
    raw_reference = _reference_text(row.get("verification_json"))
    if raw_reference is None:
        return None
    text = raw_reference.strip()
    if not text or len(text) > 200 or len(text.splitlines()) > 2:
        return None
    try:
        candidate = AnswerExtraction.model_validate_json(str(row.get("answer_json") or ""))
        reference = AnswerExtraction.model_validate_json(
            str(row.get("reference_answer_json") or "")
        )
    except Exception:
        return None
    comparable = {"number", "expression", "equation", "boolean", "choice"}
    if (
        candidate.status != "extracted"
        or candidate.answer_type not in comparable
        or candidate.value is None
        or candidate.values
        or reference.status != "no_answer"
    ):
        return None
    return AnswerExtraction(
        status="extracted",
        answer_type=candidate.answer_type,
        value=text,
    )
