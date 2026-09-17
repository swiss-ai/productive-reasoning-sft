from __future__ import annotations

import math

import pandas as pd

from synthetic_sft.selection import rank_candidate_groups


def _frame(scores: list[float | None], *, verifier: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(len(scores))],
            "generation_complete": [True] * len(scores),
            "verifier_available": [verifier] * len(scores),
            "verifier_score": [1.0 if verifier else math.nan] * len(scores),
            "verifier_threshold": [1.0 if verifier else math.nan] * len(scores),
            "quality_score": [math.nan if score is None else score for score in scores],
        }
    )


def test_no_quality_signal_keeps_all_valid_candidates() -> None:
    ranked = rank_candidate_groups(_frame([None, None]), keep_per_prompt=1)
    assert ranked["selected"].tolist() == [True, True]
    assert ranked["selection_rank"].isna().all()


def test_quality_signal_selects_top_k() -> None:
    ranked = rank_candidate_groups(_frame([0.2, 0.9, 0.8]), keep_per_prompt=2)
    selected = ranked.loc[ranked["selected"], "candidate_id"].tolist()
    assert selected == ["c1", "c2"]
    assert ranked.loc[ranked["candidate_id"] == "c1", "selection_rank"].item() == 1.0


def test_zero_quality_rollout_is_ranked_and_retained() -> None:
    frame = _frame([0.0, 0.8], verifier=True)
    ranked = rank_candidate_groups(frame, keep_per_prompt=1)
    assert ranked.loc[0, "selected"] == False  # noqa: E712
    assert ranked.loc[0, "selection_rank"] == 2
    assert ranked.loc[1, "selected"] == True  # noqa: E712


def test_all_zero_quality_rollouts_are_not_selected() -> None:
    ranked = rank_candidate_groups(_frame([0.0, 0.0], verifier=True), keep_per_prompt=1)
    assert ranked["selection_rank"].notna().all()
    assert not ranked["selected"].any()
