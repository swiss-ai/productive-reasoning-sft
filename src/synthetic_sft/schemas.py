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


class ResponseAssessment(StrictModel):
    score: int = Field(ge=1, le=5)
    issues: list[ResponseIssue] = Field(default_factory=list)
    feedback: str = Field(min_length=1, max_length=300)


class JudgeScores(StrictModel):
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


QualityZeroReason = Literal[
    "generation_incomplete",
    "verifier_failed",
    "verifier_error",
    "judge_error",
]


class QualityDecision(StrictModel):
    raw_aggregate_score: int | None = Field(default=None, ge=1, le=5)
    zeroed: bool
    zero_reasons: list[QualityZeroReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def zero_state_is_consistent(self) -> QualityDecision:
        if self.zeroed != bool(self.zero_reasons):
            raise ValueError("zeroed must match whether zero_reasons is non-empty")
        return self


class QualityDetails(StrictModel):
    schema_version: Literal[3] = 3
    aggregate_score: int | None = Field(default=None, ge=0, le=5)
    decision: QualityDecision
    verifier: VerifierDetails
    judge: JudgeDetails
    validity: ValidityDetails

    @model_validator(mode="after")
    def decision_is_consistent(self) -> QualityDetails:
        if self.decision.zeroed:
            if self.aggregate_score != 0.0:
                raise ValueError("zeroed quality must have aggregate_score 0")
        elif self.aggregate_score != self.decision.raw_aggregate_score:
            raise ValueError("nonzeroed aggregate must equal raw_aggregate_score")
        return self

    def as_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


def quality_details_schema() -> dict[str, Any]:
    return QualityDetails.model_json_schema()
