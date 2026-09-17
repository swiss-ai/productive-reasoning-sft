from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from synthetic_sft.config import load_config
from synthetic_sft.generation import VLLMBatchPredictor


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_template_kwargs = None

    def chat(
        self,
        messages,
        *,
        sampling_params,
        use_tqdm,
        chat_template_kwargs,
    ):
        self.last_template_kwargs = chat_template_kwargs
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[
                    SimpleNamespace(
                        text="reasoning</think>answer",
                        finish_reason="stop",
                        token_ids=[3, 4, 5],
                    )
                ],
            )
            for _ in messages
        ]


def test_offline_vllm_actor_contract(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    config = load_config("configs/reasoning-gym-smoke.yaml")
    predictor = VLLMBatchPredictor(config.model_dump(mode="json"), judge=False)
    output = predictor(
        pd.DataFrame.from_records(
            [
                {
                    "candidate_id": "abc",
                    "user_prompt": "question",
                    "system_prompt": None,
                }
            ]
        )
    ).iloc[0]
    assert output["raw_generation"] == "reasoning</think>answer"
    assert output["finish_reason"] == "stop"
    assert output["num_input_tokens"] == 2
    assert output["num_generated_tokens"] == 3
    assert predictor.llm.last_template_kwargs["enable_thinking"] is True
