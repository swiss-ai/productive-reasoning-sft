from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synthetic_sft.json_utils import canonical_json, json_object


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedRecord(StrictModel):
    sample_id: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    source: str = Field(min_length=1)
    provenance_json: str
    system_prompt: str | None = None
    verification_json: str | None = None

    @field_validator("provenance_json")
    @classmethod
    def provenance_is_object(cls, value: str) -> str:
        return canonical_json(json_object(value, field="provenance_json"))

    @field_validator("verification_json")
    @classmethod
    def verification_is_object(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return canonical_json(json_object(value, field="verification_json"))


class VerifierDetails(StrictModel):
    available: bool
    status: Literal["passed", "failed", "indeterminate", "error", "unavailable"]
    name: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    error: str | None = None

    @model_validator(mode="after")
    def availability_is_consistent(self) -> VerifierDetails:
        if not self.available and any(
            item is not None for item in (self.name, self.score, self.threshold, self.error)
        ):
            raise ValueError("unavailable verifier must not contain verifier data")
        return self


AnswerType = Literal[
    "number",
    "expression",
    "equation",
    "set",
    "interval",
    "boolean",
    "choice",
    "text",
    "code",
    "proof",
    "other",
]


class AnswerExtraction(StrictModel):
    status: Literal["extracted", "conditional", "no_answer", "ambiguous"]
    answer_type: AnswerType
    value: str | None = None
    values: list[str] = Field(default_factory=list)
    unit: str | None = None
    evidence: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def extracted_answer_has_value(self) -> AnswerExtraction:
        if self.status == "extracted" and not (self.value or self.values):
            raise ValueError("an extracted answer requires value or values")
        if self.status == "conditional" and (
            self.answer_type != "text" or not self.value or self.values
        ):
            raise ValueError("a conditional answer needs one text value and no values list")
        return self


class AnswerDetails(StrictModel):
    candidate: AnswerExtraction | None = None
    reference: AnswerExtraction | None = None
    candidate_method: Literal["model"] | None = None
    reference_method: Literal["model", "atomic_source_fallback"] | None = None


class CorrectnessAssessment(StrictModel):
    verdict: Literal["correct", "incorrect", "indeterminate", "reference_conflict"]
    confidence: Literal["high", "medium", "low"]
    feedback: str = Field(min_length=1, max_length=500)


ReasoningIssue = Literal[
    "absent",
    "incomplete",
    "factual_or_logical_error",
    "unsupported_step",
    "missing_critical_step",
    "contradiction",
    "meandering",
    "repetition",
    "meta_commentary",
    "poor_structure",
    "unverifiable",
]
ResponseIssue = Literal[
    "incorrect",
    "incomplete",
    "instruction_violation",
    "format_violation",
    "irrelevant",
    "unclear",
    "oververbose",
    "unsupported_claim",
    "meta_commentary",
]


class ReasoningAssessment(StrictModel):
    score: int = Field(ge=1, le=5)
    issues: list[ReasoningIssue] = Field(default_factory=list)
    feedback: str = Field(min_length=1, max_length=300)

    @field_validator("issues", mode="before")
    @classmethod
    def issues_are_unique(cls, value: Any) -> Any:
        return list(dict.fromkeys(value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def lower_score_identifies_issue(self) -> ReasoningAssessment:
        if self.score < 5 and not self.issues:
            raise ValueError("reasoning scores below 5 require at least one issue")
        if self.score == 5 and self.issues:
            raise ValueError("reasoning score 5 cannot contain issues")
        return self


class ResponseAssessment(StrictModel):
    score: int = Field(ge=1, le=5)
    issues: list[ResponseIssue] = Field(default_factory=list)
    feedback: str = Field(min_length=1, max_length=300)

    @field_validator("issues", mode="before")
    @classmethod
    def issues_are_unique(cls, value: Any) -> Any:
        return list(dict.fromkeys(value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def lower_score_identifies_issue(self) -> ResponseAssessment:
        if self.score < 5 and not self.issues:
            raise ValueError("response scores below 5 require at least one issue")
        if self.score == 5 and self.issues:
            raise ValueError("response score 5 cannot contain issues")
        return self


class JudgeScores(StrictModel):
    correctness: CorrectnessAssessment
    reasoning: ReasoningAssessment
    response: ResponseAssessment

    def effective(self) -> int:
        """The weaker component determines training readiness."""
        return min(self.reasoning.score, self.response.score)


class JudgeDetails(StrictModel):
    enabled: bool
    model: str | None = None
    rubric_version: int | None = Field(default=None, ge=1)
    scores: JudgeScores | None = None
    aggregate: int | None = Field(default=None, ge=1, le=5)
    analysis_samples: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    generated_tokens: int | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def state_is_consistent(self) -> JudgeDetails:
        if not self.enabled:
            if any(
                item is not None
                for item in (
                    self.model,
                    self.rubric_version,
                    self.scores,
                    self.aggregate,
                    self.analysis_samples or None,
                    self.retries or None,
                    self.finish_reason,
                    self.generated_tokens,
                    self.error,
                )
            ):
                raise ValueError("disabled judge must not contain judge data")
            return self
        if self.model is None or self.rubric_version is None:
            raise ValueError("enabled judge requires model and rubric_version")
        if self.scores is not None:
            expected = self.scores.effective()
            if self.aggregate != expected:
                raise ValueError("judge aggregate must equal the weaker rubric score")
        elif self.aggregate is not None:
            raise ValueError("judge aggregate requires rubric scores")
        return self


class ValidityDetails(StrictModel):
    reasoning_parsed: bool
    response_present: bool
    finish_reason: str | None = None
    generation_complete: bool


HygieneCategory = Literal[
    "repetition",
    "circular_rechecking",
    "no_new_progress",
    "unresolved_branch",
    "reasoning_limit_stop",
    "final_answer_missing_or_malformed",
]


class ModelHygieneAssessment(StrictModel):
    verdict: Literal["defect", "clear", "uncertain"]
    evidence: list[str] = Field(max_length=2)
    explanation: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def defect_has_evidence(self) -> ModelHygieneAssessment:
        if self.verdict == "defect" and not self.evidence:
            raise ValueError("a confirmed defect requires an exact excerpt")
        return self


class HygieneFinding(StrictModel):
    category: HygieneCategory
    verdict: Literal["defect", "clear", "uncertain"]
    method: Literal["deterministic", "model"]
    evidence: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    error: str | None = None


class HygieneDetails(StrictModel):
    status: Literal["passed", "failed", "indeterminate", "not_run"]
    model: str | None = None
    findings: list[HygieneFinding] = Field(default_factory=list)
    failure_categories: list[HygieneCategory] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SelectionDetails(StrictModel):
    correctness_only: bool
    productivity_filtered: bool
    exclusion_reasons: list[str] = Field(default_factory=list)


QualityZeroReason = Literal[
    "generation_incomplete",
    "answer_incorrect",
    "judge_error",
]


class QualityDecision(StrictModel):
    raw_aggregate_score: int | None = Field(default=None, ge=1, le=5)
    score_cap: int | None = Field(default=None, ge=1, le=5)
    score_cap_reasons: list[
        Literal[
            "correctness_conflict",
            "correctness_indeterminate",
            "material_judge_issue",
            "hygiene_defect",
            "hygiene_uncertain",
        ]
    ] = Field(default_factory=list)
    zeroed: bool
    zero_reasons: list[QualityZeroReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def zero_state_is_consistent(self) -> QualityDecision:
        if self.zeroed != bool(self.zero_reasons):
            raise ValueError("zeroed must match whether zero_reasons is non-empty")
        return self


class QualityDetails(StrictModel):
    schema_version: Literal[8] = 8
    aggregate_score: int | None = Field(default=None, ge=0, le=5)
    decision: QualityDecision
    answer: AnswerDetails
    correctness_verdict: Literal["verified", "supported", "incorrect", "conflict", "indeterminate"]
    verifier: VerifierDetails
    judge: JudgeDetails
    validity: ValidityDetails
    hygiene: HygieneDetails
    selection: SelectionDetails

    @model_validator(mode="after")
    def decision_is_consistent(self) -> QualityDetails:
        if self.decision.zeroed:
            if self.aggregate_score != 0.0:
                raise ValueError("zeroed quality must have aggregate_score 0")
        else:
            expected = self.decision.raw_aggregate_score
            if expected is not None and self.decision.score_cap is not None:
                expected = min(expected, self.decision.score_cap)
            if self.aggregate_score != expected:
                raise ValueError("aggregate must equal the raw score after applying its cap")
        return self

    def as_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def quality_details_schema() -> dict[str, Any]:
    return QualityDetails.model_json_schema()
