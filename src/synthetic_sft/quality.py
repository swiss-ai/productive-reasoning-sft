from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from synthetic_sft.adapters import create_adapter
from synthetic_sft.config import PipelineConfig
from synthetic_sft.hygiene import build_hygiene_details
from synthetic_sft.schemas import (
    AnswerDetails,
    AnswerExtraction,
    JudgeDetails,
    JudgeScores,
    QualityDecision,
    QualityDetails,
    SelectionDetails,
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
    if all(tag in text for tag in ("<reasoning>", "</reasoning>", "<response>", "</response>")):
        reasoning = text.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0].strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, bool(reasoning and response)
    if all(tag in text for tag in ("</reasoning>", "<response>", "</response>")):
        reasoning = text.split("</reasoning>", 1)[0]
        reasoning = _REASONING_PREFIX.sub("", reasoning, count=1).strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, bool(reasoning and response)
    if "<reasoning>" in text and "<response>" in text:
        reasoning = text.split("<reasoning>", 1)[1].split("<response>", 1)[0]
        reasoning = reasoning.strip().removesuffix("</think>").strip()
        response = text.split("<response>", 1)[1].split("</response>", 1)[0].strip()
        return reasoning or None, response or None, False
    if "<reasoning>" in text:
        reasoning = text.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0].strip()
        return reasoning or None, None, False
    json_text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        polished = json.loads(json_text)
        if isinstance(polished, dict):
            reasoning = polished.get("reasoning")
            response = polished.get("response")
            if isinstance(reasoning, str) and reasoning.strip() and isinstance(response, str):
                return reasoning.strip(), response.strip() or None, bool(response.strip())
    except (TypeError, ValueError):
        pass
    if "<think>" in text:
        _, _, after_start = text.partition("<think>")
        if "</think>" not in after_start:
            return after_start.strip() or None, None, False
        reasoning, _, response = after_start.partition("</think>")
        return (
            reasoning.strip() or None,
            response.strip() or None,
            bool(reasoning.strip() and response.strip()),
        )
    if "</think>" in text:
        reasoning, _, response = text.partition("</think>")
        return (
            reasoning.strip() or None,
            response.strip() or None,
            bool(reasoning.strip() and response.strip()),
        )
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
        self.parse(row)
        return self.verify(row)

    def parse(self, row: dict[str, Any]) -> dict[str, Any]:
        raw = row.get("raw_generation")
        reasoning, response, parsed = split_reasoning(str(raw) if raw is not None else None)
        finish_reason = row.get("finish_reason")
        generation_complete = (
            parsed and response is not None and finish_reason not in {"length", "abort"}
        )
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
        return row

    def verify(self, row: dict[str, Any]) -> dict[str, Any]:
        response = row.get("response")
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
                row["verifier_details_json"] = json.dumps(
                    result.details or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
        if not available:
            verification_status = "unavailable"
        elif error is not None or score is None:
            verification_status = "error" if error is not None else "indeterminate"
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


def answer_extraction_prompt(row: Mapping[str, Any], text: str, *, reference: bool) -> str:
    role = "source reference" if reference else "assistant response"
    return f"""Extract the final answer asserted by the {role}. Do not solve the problem and do not
correct, complete, or reinterpret the supplied text. Use the question only to identify what counts
as the answer. Preserve all required alternatives for questions asking for every solution. Remove
display markup such as dollar signs and \\boxed, but preserve mathematical meaning. Put one
canonical answer in `value`; use `values` for an unordered collection of multiple required answers.
Equivalent exact, parameterized, and numerical forms of the same answer are not incompatible:
extract the most precise form. An intermediate expression is not a second final answer.
Record a unit separately. Always choose exactly one `answer_type`: use `number` for a numeric
scalar, `expression` for a symbolic value, `equation`, `set`, `interval`, `boolean`, `choice`,
`text`, `code`, or `proof` when applicable, and `other` only when none fits. If there is no asserted
answer, return status `no_answer` with type `other`. If the text gives incompatible final answers,
return status `ambiguous` with the type of those answers, or `other` if their types differ.

Question:
{row.get("user_prompt", "")}

{role.title()}:
{text}
"""


def critic_prompt(row: Mapping[str, Any]) -> str:
    return f"""Act as an adversarial verifier of proposed supervised fine-tuning data. Work through
the problem independently enough to catch plausible but false reasoning. First assess the final
answer without trusting the candidate reasoning. Then audit the reasoning step by step and identify
the earliest material defect, if one exists. Try counterexamples and boundary cases. A source
reference is strong evidence but may be malformed, incomplete, or wrong; explicitly flag a genuine
reference conflict. Do not reward length, confidence, or polished prose. Do not merely summarize.
For proof questions, distinguish a correct yes/no conclusion from an actually established proof.
Verify the claimed theorem's precise hypotheses and conclusion when possible; an unnamed theorem,
an unsupported dimension/counting leap, or a citation that does not imply the required claim is a
material reasoning gap even if the source answer and final yes/no agree.
First check that the question's premises are mutually consistent and sufficient. If they are not,
the only correct response is one that clearly explains the impossibility or underdetermination;
inventing assumptions or reporting an impossible reference value is incorrect.

Question:
{row.get("user_prompt", "")}

Source reference:
{_reference_answer(row.get("verification_json"))}

Extracted candidate answer:
{row.get("answer_json") or "(not extracted)"}

Extracted reference answer:
{row.get("reference_answer_json") or "(not available)"}

Deterministic check:
status={row.get("verification_status")}; details={row.get("verifier_details_json") or "{}"}

Candidate reasoning:
{row.get("reasoning") or "(no separate reasoning trace)"}

Candidate response:
{row.get("response") or "(missing)"}

Return a concrete analysis for a later arbiter. State whether the answer is correct, whether the
reference is reliable, and every material reasoning or response defect you can substantiate.
"""


def arbitration_prompt(row: Mapping[str, Any], analyses: list[str], rubric_version: int) -> str:
    rendered = "\n\n".join(
        f"Independent critique {index + 1}:\n{analysis}" for index, analysis in enumerate(analyses)
    )
    return f"""You are the final conservative arbiter for SFT data quality. Validate the independent
critiques below instead of blindly following either one. A claimed defect counts only if you can
confirm it from the question and candidate. Resolve answer equivalence semantically. Treat a parser
failure as uncertainty, not mathematical incorrectness. The source reference is evidence, not an
infallible instruction to copy.

Before grading, check whether the premises are consistent and determine what a correct answer can
claim. Matching a supplied reference is not evidence of correctness when that result violates a
premise. If a response correctly notices an impossible or underdetermined question but then asserts
a result from invented assumptions, grade its answer `incorrect`, not `indeterminate`. Likewise,
incompatible final conclusions are incorrect when at least one is materially false.

Rubric version: {rubric_version}
Use integer scores 1 through 5 for reasoning and response:
5 = training-ready: correct, rigorous, direct, self-contained, no material or stylistic edit.
4 = fully correct with one minor clarity, style, or harmless redundancy issue.
3 = mostly correct but requiring a substantive local edit or containing a meaningful gap.
2 = useful progress but a serious error or omission requires a major rewrite.
1 = absent, fundamentally wrong, incoherent, or unusable without replacement.

Grade the reasoning independently of the final answer: a correct answer or matching source
reference cannot validate an unsupported proof. If a central step is asserted without a derivation
or a theorem with checked hypotheses, reasoning is at most 2, with `unsupported_step` or
`missing_critical_step`. Do not confuse a valid, well-known theorem used with stated hypotheses
for a gap. For a counterexample, check its defining properties and the claimed failure explicitly.

Speculative alternative solutions, invented assumptions, or a second incompatible conclusion are
substantive defects, never minor style issues: response quality is at most 3, and correctness is
`incorrect` when the response presents the fabricated result as an answer to the original question.

For every reasoning or response score below 5, select at least one matching issue code in addition
to the concise feedback. A score of 5 must have no issues.

For correctness, use `incorrect` only for a confirmed wrong answer, `reference_conflict` only when
the supplied reference is demonstrably unreliable, and `indeterminate` when the available evidence
cannot settle correctness. High confidence requires direct verification, a sound derivation, or a
confirmed counterexample. Feedback must identify the decisive evidence in at most 50 words.

Question:
{row.get("user_prompt", "")}

Source reference:
{_reference_answer(row.get("verification_json"))}

Extracted candidate answer:
{row.get("answer_json") or "(not extracted)"}

Extracted reference answer:
{row.get("reference_answer_json") or "(not available)"}

Deterministic check:
status={row.get("verification_status")}; details={row.get("verifier_details_json") or "{}"}

Candidate reasoning:
{row.get("reasoning") or "(no separate reasoning trace)"}

Candidate response:
{row.get("response") or "(missing)"}

{rendered}
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


def parse_answer_extraction(raw: str | None) -> tuple[AnswerExtraction | None, str | None]:
    if not raw:
        return None, "answer extractor returned an empty response"
    match = _JSON_BLOCK.search(raw)
    if match is None:
        return None, "answer extractor did not return a JSON object"
    try:
        return AnswerExtraction.model_validate_json(match.group(0)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


class FinalizeQuality:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = PipelineConfig.model_validate(config)

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        quality = self.config.quality
        verifier = VerifierDetails(
            available=bool(row.get("verifier_available")),
            status=row.get("verification_status", "unavailable"),
            name=row.get("verifier_name"),
            score=row.get("verifier_score"),
            threshold=row.get("verifier_threshold"),
            error=row.get("verifier_error"),
        )

        judge_config = quality.judge
        scores = None
        if judge_config.enabled:
            scores, error = parse_judge_scores(row.get("judge_raw_output"))
            error = row.get("judge_generation_error") or error
            judge = JudgeDetails(
                enabled=True,
                model=judge_config.model_source or self.config.model.model_source,
                rubric_version=judge_config.rubric_version,
                scores=scores,
                aggregate=scores.effective() if scores is not None else None,
                analysis_samples=len(_json_list(row.get("critic_analyses_json"))),
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
        hygiene = build_hygiene_details(
            row,
            enabled=judge_config.enabled,
            model=judge_config.model_source or self.config.model.model_source,
        )
        raw_aggregate = judge.aggregate
        candidate_answer = _answer(row.get("answer_json"))
        reference_answer = _answer(row.get("reference_answer_json"))
        correctness_verdict = _correctness_verdict(row, scores)
        zero_reasons = []
        if not validity.generation_complete:
            zero_reasons.append("generation_incomplete")
        if correctness_verdict == "incorrect":
            zero_reasons.append("answer_incorrect")
        if judge.enabled and (judge.error is not None or judge.aggregate is None):
            zero_reasons.append("judge_error")
        if hygiene.errors and judge.enabled and "judge_error" not in zero_reasons:
            zero_reasons.append("judge_error")
        cap_reasons = []
        score_cap = None
        if not zero_reasons and correctness_verdict == "conflict":
            score_cap = 3
            cap_reasons.append("correctness_conflict")
        elif not zero_reasons and correctness_verdict == "indeterminate" and verifier.available:
            score_cap = 3
            cap_reasons.append("correctness_indeterminate")
        if not zero_reasons and hygiene.status == "failed":
            score_cap = min(score_cap or 5, 2)
            cap_reasons.append("hygiene_defect")
        elif not zero_reasons and hygiene.status == "indeterminate":
            score_cap = min(score_cap or 5, 3)
            cap_reasons.append("hygiene_uncertain")
        aggregate = 0 if zero_reasons else raw_aggregate
        if aggregate is not None and score_cap is not None:
            aggregate = min(aggregate, score_cap)
        correctness_only = (
            validity.generation_complete
            and correctness_verdict in {"verified", "supported"}
            and candidate_answer is not None
            and candidate_answer.status == "extracted"
            and judge.error is None
        )
        exclusion_reasons = []
        if not validity.generation_complete:
            exclusion_reasons.append("generation_incomplete")
        if correctness_verdict not in {"verified", "supported"}:
            exclusion_reasons.append(f"correctness_{correctness_verdict}")
        if candidate_answer is None or candidate_answer.status != "extracted":
            exclusion_reasons.append("final_answer_unavailable")
        if judge.error is not None or hygiene.errors:
            exclusion_reasons.append("judge_error")
        if hygiene.status != "passed":
            exclusion_reasons.append(f"hygiene_{hygiene.status}")
            exclusion_reasons.extend(f"hygiene_{item}" for item in hygiene.failure_categories)
        selection = SelectionDetails(
            correctness_only=correctness_only,
            productivity_filtered=correctness_only and hygiene.status == "passed",
            exclusion_reasons=exclusion_reasons,
        )
        details = QualityDetails(
            aggregate_score=aggregate,
            decision=QualityDecision(
                raw_aggregate_score=raw_aggregate,
                score_cap=score_cap,
                score_cap_reasons=cap_reasons,
                zeroed=bool(zero_reasons),
                zero_reasons=zero_reasons,
            ),
            answer=AnswerDetails(
                candidate=candidate_answer,
                reference=reference_answer,
                candidate_method=row.get("candidate_answer_method"),
                reference_method=row.get("reference_answer_method"),
            ),
            correctness_verdict=correctness_verdict,
            verifier=verifier,
            judge=judge,
            validity=validity,
            hygiene=hygiene,
            selection=selection,
        )
        row["quality_score"] = aggregate
        row["correctness_verdict"] = correctness_verdict
        row["hygiene_status"] = hygiene.status
        row["correctness_only_eligible"] = selection.correctness_only
        row["productivity_filtered_eligible"] = selection.productivity_filtered
        row["exclusion_reasons_json"] = json.dumps(selection.exclusion_reasons)
        row["quality_details_json"] = details.as_json()
        return row


def _answer(raw: Any) -> AnswerExtraction | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return AnswerExtraction.model_validate_json(raw)
    except Exception:
        return None


def _json_list(raw: Any) -> list[Any]:
    if not raw or not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _correctness_verdict(row: Mapping[str, Any], scores: JudgeScores | None) -> str:
    deterministic = str(row.get("verification_status") or "unavailable")
    if scores is None:
        return "indeterminate"
    assessed = scores.correctness
    if deterministic == "passed":
        return "verified" if assessed.verdict == "correct" else "conflict"
    if assessed.verdict == "incorrect":
        return "incorrect" if assessed.confidence == "high" else "indeterminate"
    if assessed.verdict == "reference_conflict":
        return "conflict"
    if deterministic == "failed":
        return "conflict" if assessed.verdict == "correct" else "indeterminate"
    if assessed.verdict == "correct" and assessed.confidence == "high":
        return "supported"
    return "indeterminate"
