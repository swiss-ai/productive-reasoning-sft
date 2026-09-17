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


def split_reasoning(raw: str | None) -> tuple[str | None, str | None, bool]:
    """Split Qwen-style thinking while preserving malformed output for auditing."""
    if not raw:
        return None, None, False
    text = raw.strip()
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
    return f"""Evaluate this proposed SFT answer. Return exactly one JSON object and no prose.

Rubric version: {rubric_version}
Each score must be a number from 0.0 to 1.0.
- correctness: likely factual/logical correctness relative to the user request
- clarity: clear, direct, readable final response
- pedagogy: useful explanation appropriate for supervised fine-tuning
- reasoning_consistency: reasoning supports and does not contradict the final response

User request:
{row.get("user_prompt", "")}

Reasoning trace:
{reasoning}

Final response:
{row.get("response", "")}

Required JSON shape:
{{"correctness":0.0,"clarity":0.0,"pedagogy":0.0,"reasoning_consistency":0.0}}
"""


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
                aggregate=scores.mean() if scores is not None else None,
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
        raw_aggregate = aggregate_quality(verifier.score, judge.aggregate, quality.verifier_weight)
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
        aggregate = 0.0 if zero_reasons else raw_aggregate
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


def aggregate_quality(
    verifier_score: float | None, judge_score: float | None, verifier_weight: float
) -> float | None:
    if verifier_score is not None and judge_score is not None:
        return verifier_weight * verifier_score + (1.0 - verifier_weight) * judge_score
    if verifier_score is not None:
        return verifier_score
    return judge_score
