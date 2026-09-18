from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from synthetic_sft.adapters import create_adapter
from synthetic_sft.config import PipelineConfig
from synthetic_sft.schemas import (
    JudgeDetails,
    JudgeScores,
    QualityDecision,
    QualityDetails,
    ValidityDetails,
    VerifierDetails,
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_REASONING_PREFIX = re.compile(r"^<reasoning>?\s*", re.IGNORECASE)


def split_reasoning(raw: str | None) -> tuple[str | None, str | None, bool]:
    """Read polished JSON or split Qwen-style thinking, preserving malformed output."""
    if not raw:
        return None, None, False
    text = raw.strip()
    if "<reasoning>" in text and "</reasoning>" in text and "<response>" in text:
        reasoning = text.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0].strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, bool(reasoning and response)
    if "</reasoning>" in text and "<response>" in text:
        reasoning = text.split("</reasoning>", 1)[0]
        reasoning = _REASONING_PREFIX.sub("", reasoning, count=1).strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, bool(reasoning and response)
    if "<reasoning>" in text and "<response>" in text:
        reasoning = text.split("<reasoning>", 1)[1].split("<response>", 1)[0]
        reasoning = reasoning.strip().removesuffix("</think>").strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, bool(reasoning and response)
    json_text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        polished = json.loads(json_text)
        if isinstance(polished, dict):
            reasoning = polished.get("reasoning")
            response = polished.get("response")
            if isinstance(reasoning, str) and reasoning.strip() and isinstance(response, str):
                return reasoning.strip(), response.strip() or None, True
    except (TypeError, ValueError):
        pass
    if "<think>" in text:
        _, _, after_start = text.partition("<think>")
        if "</think>" not in after_start:
            return after_start.strip() or None, None, False
        reasoning, _, response = after_start.partition("</think>")
        return reasoning.strip() or None, response.strip() or None, True
    if "</think>" in text:
        reasoning, _, response = text.partition("</think>")
        return reasoning.strip() or None, response.strip() or None, True
    return None, text, False


class ParseAndVerify:
    def __init__(
        self,
        adapter_name: str,
        adapter_params: Mapping[str, Any],
        verifier_threshold: float,
    ) -> None:
        self.adapter = create_adapter(adapter_name, adapter_params)
        self.threshold = verifier_threshold

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        raw = row.get("raw_generation")
        reasoning, response, parsed = split_reasoning(str(raw) if raw is not None else None)
        finish_reason = row.get("finish_reason")
        generation_complete = response is not None and finish_reason not in {"length", "abort"}
        if raw is None:
            generation_status = "failed"
            generation_error = row.get("generation_error") or "generation returned no output"
        elif not generation_complete:
            generation_status = "incomplete"
            generation_error = row.get("generation_error") or "generation has no complete response"
        else:
            generation_status = "ok"
            generation_error = row.get("generation_error")
        row.update(
            reasoning=reasoning,
            response=response,
            reasoning_parsed=parsed,
            response_present=response is not None,
            generation_complete=generation_complete,
            generation_status=generation_status,
            generation_error=generation_error,
        )

        available = self.adapter.supports_verification(row)
        score: float | None = None
        error: str | None = None
        name: str | None = None
        if available:
            name = self.adapter.verifier_name(row)
            if response is None:
                error = "response is missing"
            else:
                result = self.adapter.verify(row, response)
                score, error = result.score, result.error
        if not available:
            verification_status = "unavailable"
        elif error is not None or score is None:
            verification_status = "error"
        elif score >= self.threshold:
            verification_status = "passed"
        else:
            verification_status = "failed"
        row.update(
            verifier_available=available,
            verifier_name=name,
            verifier_score=score,
            verifier_error=error,
            verifier_threshold=self.threshold if available else None,
            verification_status=verification_status,
        )
        return row


def judge_prompt(row: Mapping[str, Any], rubric_version: int) -> str:
    reasoning = row.get("reasoning") or "(no separate reasoning trace)"
    reference = _reference_answer(row.get("verification_json"))
    return f"""You are a strict SFT data editor. Evaluate the reasoning trace and final response
independently. Return exactly one JSON object and no prose.

Rubric version: {rubric_version}
Use only integer scores 1 through 5. A score measures the editing required before this text is
safe and useful as training data, not its length or confidence.
Keep each feedback string concrete and no longer than 30 words.

5 — TRAINING-READY: exceptional, correct, rigorous, direct, self-contained, and needs no edit.
    Every material reasoning step is justified and necessary; there is no planning narration,
    self-talk, answer-format chatter, backtracking, repetition, avoidable enumeration, unsupported
    lookup, or needless restatement. The final response follows the requested format exactly.
4 — LIGHT EDIT: fully correct and reliable, with one minor clarity, style, or harmless redundancy
    issue. A small edit makes it training-ready.
3 — SUBSTANTIVE LOCAL EDIT: the core approach/conclusion is mostly correct, but there is a
    meaningful gap, imprecision, distracting meta-commentary, repeated recomputation, or local
    reasoning defect. It needs a focused rewrite, not just copyediting.
2 — MAJOR REWRITE: some useful progress, but a serious logical gap, contradiction, unsupported
    conclusion, or wrong result means most of the component must be rewritten.
1 — UNUSABLE: absent when required, mostly wrong, incoherent, irrelevant, fabricated, or not
    recoverable without replacement.

Calibration examples:
- Direct necessary steps followed by one useful check: reasoning 5.
- Correct clean derivation with one harmless repeated sentence: reasoning 4.
- Correct derivation that is substantially longer than necessary: reasoning 4 or lower.
- Correct answer reached through repeated self-talk/recomputation or an unexplained key leap:
  reasoning 3, even though the answer is correct.
- Reliance on an asserted table, external calculation, or "known value" for a material step:
  reasoning 3 or lower unless that value is established in the trace.
- Correct answer apparently reached by invalid reasoning: reasoning 2.
- Wrong or unrelated work: reasoning 1.
- Exact requested short answer with no extra material: response 5; concise is not a defect.
- Correct answer with a small presentational blemish: response 4.
- Mostly correct answer needing a meaningful localized correction: response 3.
- Wrong result with some relevant content: response 2; wholly unusable response: response 1.

Audit the reasoning adversarially, step by step, and identify the earliest material defect. A 5 is
appropriate only after finding no factual, logical, relevance, style, or self-containment defect;
when in doubt between 4 and 5, use 4. A correct reference response does not prove that the reasoning
is sound. Treat a supplied reference as authoritative evidence about the result, but do not copy
source metadata into feedback.

User request:
{row.get("user_prompt", "")}

Reasoning trace:
{reasoning}

Final response:
{row.get("response", "")}

Reference material:
{reference}

Required JSON shape:
{{"reasoning":{{"score":1,"issues":["meta_commentary"],
"feedback":"brief concrete reason"}},"response":{{"score":1,
"issues":["incorrect"],"feedback":"brief concrete reason"}}}}

Allowed reasoning issues: absent, incomplete, factual_or_logical_error, unsupported_step,
missing_critical_step, contradiction, meandering, repetition, meta_commentary, poor_structure,
unverifiable.
Allowed response issues: incorrect, incomplete, instruction_violation, format_violation,
irrelevant, unclear, oververbose, unsupported_claim, meta_commentary. Use an empty list when there
is no issue.
"""


def _reference_answer(raw: Any) -> str:
    if not raw:
        return "(no reference answer is available)"
    try:
        payload = json.loads(str(raw))
        answer = payload.get("entry", {}).get("answer")
        if answer is None:
            return "(the source provides a verifier but no textual reference answer)"
        return json.dumps(answer, ensure_ascii=False)
    except (TypeError, ValueError, AttributeError):
        return "(reference metadata could not be decoded; rely on the problem itself)"


def parse_judge_scores(raw: str | None) -> tuple[JudgeScores | None, str | None]:
    if not raw:
        return None, "judge returned an empty response"
    match = _JSON_BLOCK.search(raw)
    if match is None:
        return None, "judge response did not contain a JSON object"
    try:
        payload = json.loads(match.group(0))
        return JudgeScores.model_validate(payload), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


class FinalizeQuality:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = PipelineConfig.model_validate(config)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        quality = self.config.quality
        verifier = VerifierDetails(
            available=bool(row.get("verifier_available")),
            name=row.get("verifier_name"),
            score=row.get("verifier_score"),
            threshold=row.get("verifier_threshold"),
            error=row.get("verifier_error"),
        )

        judge_config = quality.judge
        if judge_config.enabled:
            scores, error = parse_judge_scores(row.get("judge_raw_output"))
            error = row.get("judge_generation_error") or error
            judge = JudgeDetails(
                enabled=True,
                model=judge_config.model_source or self.config.model.model_source,
                rubric_version=judge_config.rubric_version,
                scores=scores,
                aggregate=scores.effective() if scores is not None else None,
                error=error,
            )
            row["judge_status"] = "passed" if error is None else "error"
        else:
            judge = JudgeDetails(enabled=False)
            row["judge_status"] = "disabled"

        validity = ValidityDetails(
            reasoning_parsed=bool(row.get("reasoning_parsed")),
            response_present=bool(row.get("response_present")),
            finish_reason=row.get("finish_reason"),
            generation_complete=bool(row.get("generation_complete")),
        )
        raw_aggregate = judge.aggregate
        zero_reasons = []
        if not validity.generation_complete:
            zero_reasons.append("generation_incomplete")
        if verifier.available:
            if verifier.error is not None or verifier.score is None:
                zero_reasons.append("verifier_error")
            elif verifier.threshold is not None and verifier.score < verifier.threshold:
                zero_reasons.append("verifier_failed")
        if judge.enabled and (judge.error is not None or judge.aggregate is None):
            zero_reasons.append("judge_error")
        aggregate = 0 if zero_reasons else raw_aggregate
        details = QualityDetails(
            aggregate_score=aggregate,
            decision=QualityDecision(
                raw_aggregate_score=raw_aggregate,
                zeroed=bool(zero_reasons),
                zero_reasons=zero_reasons,
            ),
            verifier=verifier,
            judge=judge,
            validity=validity,
        )
        row["quality_score"] = aggregate
        row["quality_details_json"] = details.as_json()
        return row
