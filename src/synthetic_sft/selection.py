from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def rank_candidate_groups(frame: pd.DataFrame, *, keep_per_prompt: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["selection_rank"] = float("nan")
    frame["selected"] = False
    ranked = frame.copy()
    if ranked.empty:
        return frame

    if ranked["quality_score"].notna().any():
        ranked = ranked.sort_values(
            ["quality_score", "candidate_id"], ascending=[False, True], na_position="last"
        )
        ranks = list(range(1, len(ranked) + 1))
        frame.loc[ranked.index, "selection_rank"] = [float(rank) for rank in ranks]
        nonzero = ranked.loc[ranked["quality_score"] > 0.0]
        frame.loc[nonzero.index[:keep_per_prompt], "selected"] = True
    else:
        # Without a meaningful quality signal there is no defensible top-k ordering.
        frame.loc[ranked.index, "selected"] = True
    return frame


def content_hash(row: dict[str, Any]) -> dict[str, Any]:
    payload = f"{row.get('user_prompt', '')}\x1f{row.get('response', '')}".encode()
    row["content_hash"] = hashlib.sha256(payload).hexdigest()
    return row


def choose_duplicate(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 1:
        return frame
    ordered = frame.sort_values(
        ["quality_score", "candidate_id"], ascending=[False, True], na_position="last"
    )
    return ordered.head(1)


SELECTED_COLUMNS = [
    "sample_id",
    "system_prompt",
    "user_prompt",
    "reasoning",
    "response",
    "source",
    "model",
    "quality_score",
    "quality_details_json",
    "provenance_json",
]
