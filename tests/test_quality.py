from __future__ import annotations

import json

import pytest

from synthetic_sft.quality import aggregate_quality, parse_judge_scores, split_reasoning
from synthetic_sft.schemas import (
    JudgeDetails,
    JudgeScores,
    QualityDecision,
    QualityDetails,
    ValidityDetails,
    VerifierDetails,
)


def test_split_reasoning_with_both_tags() -> None:
    assert split_reasoning("<think>work</think>answer") == ("work", "answer", True)


def test_split_reasoning_when_open_tag_is_in_prompt() -> None:
    assert split_reasoning("work</think>answer") == ("work", "answer", True)


def test_split_reasoning_preserves_direct_answer() -> None:
    assert split_reasoning("answer") == (None, "answer", False)


def test_unclosed_reasoning_is_not_a_response() -> None:
    assert split_reasoning("<think>unfinished") == ("unfinished", None, False)


def test_quality_details_are_structured_and_consistent() -> None:
    scores = JudgeScores(
        correctness=1.0,
        clarity=0.8,
        pedagogy=0.9,
        reasoning_consistency=0.9,
    )
    judge = JudgeDetails(
        enabled=True,
        model="judge",
        rubric_version=1,
        scores=scores,
        aggregate=0.9,
    )
    details = QualityDetails(
        aggregate_score=0.98,
        decision=QualityDecision(raw_aggregate_score=0.98, zeroed=False),
        verifier=VerifierDetails(available=True, name="exact", score=1.0, threshold=1.0),
        judge=judge,
        validity=ValidityDetails(
            reasoning_parsed=True,
            response_present=True,
            finish_reason="stop",
            generation_complete=True,
        ),
    )
    encoded = json.loads(details.as_json())
    assert encoded["schema_version"] == 2
    assert encoded["judge"]["scores"]["pedagogy"] == 0.9
    assert encoded["decision"] == {
        "raw_aggregate_score": 0.98,
        "zero_reasons": [],
        "zeroed": False,
    }


def test_disabled_judge_rejects_stale_fields() -> None:
    with pytest.raises(ValueError):
        JudgeDetails(enabled=False, model="stale")


def test_quality_details_reject_inconsistent_zero_decision() -> None:
    with pytest.raises(ValueError, match="zeroed quality"):
        QualityDetails(
            aggregate_score=0.5,
            decision=QualityDecision(
                raw_aggregate_score=0.5,
                zeroed=True,
                zero_reasons=["verifier_failed"],
            ),
            verifier=VerifierDetails(available=False),
            judge=JudgeDetails(enabled=False),
            validity=ValidityDetails(
                reasoning_parsed=True,
                response_present=True,
                generation_complete=True,
            ),
        )


def test_judge_parser_validates_fixed_rubric() -> None:
    raw = '{"correctness":1,"clarity":0.8,"pedagogy":0.7,"reasoning_consistency":0.9}'
    scores, error = parse_judge_scores(raw)
    assert error is None
    assert scores is not None and scores.mean() == pytest.approx(0.85)


def test_aggregate_quality_handles_optional_components() -> None:
    assert aggregate_quality(1.0, 0.5, 0.8) == pytest.approx(0.9)
    assert aggregate_quality(0.75, None, 0.8) == 0.75
    assert aggregate_quality(None, 0.6, 0.8) == 0.6
    assert aggregate_quality(None, None, 0.8) is None
