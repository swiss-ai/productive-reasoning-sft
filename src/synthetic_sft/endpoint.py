"""Private OpenAI-compatible serving transport; reuses every local quality stage.

The server owns chat rendering (important for DeepSeek's specialized encoder).
Usage counts are represented by ranges, not claimed to be actual token IDs.
No automatic transport retries: an infrastructure error must not become a math reject.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from synthetic_sft.continuous import ContinuousVLLMPredictor


class EndpointPredictor(ContinuousVLLMPredictor):
    def _backend_types(self):
        return None, SimpleNamespace, SimpleNamespace

    def _create_engine(self, engine_type, kwargs):
        import httpx
        from tokenizers import Tokenizer

        endpoint = self.config.model.endpoint
        if endpoint is None:
            raise ValueError("model.endpoint is required for endpoint execution")
        if self.judge or not self.fused_judge:
            raise ValueError("Endpoint pilot requires enabled same-model judging")
        base_url = os.environ.get(endpoint.base_url_env, "").rstrip("/")
        if not base_url:
            raise ValueError(f"Set {endpoint.base_url_env} to the private server's /v1 URL")
        headers = {}
        api_key = os.environ.get(endpoint.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        concurrency = self.config.model.inflight_candidates * 4
        self._client = httpx.Client(
            base_url=base_url + "/", headers=headers, trust_env=False,
            timeout=endpoint.timeout_seconds,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )
        models = self._client.get("models")
        models.raise_for_status()
        if endpoint.served_model_name not in {model["id"] for model in models.json()["data"]}:
            self._client.close()
            raise ValueError("Configured served model is absent from endpoint /models")
        tokenizer = Tokenizer.from_file(str(Path(kwargs["model"]) / "tokenizer.json"))
        self._requests = ThreadPoolExecutor(max_workers=concurrency)
        # Text-only counts/truncation; never apply a local chat template.
        counter = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            ).ids,
            decode=tokenizer.decode,
        )
        return SimpleNamespace(get_tokenizer=lambda: counter)

    def _chat(self, records, messages, sampling, template_kwargs, *, phase):
        if not records:
            return []
        started = time.perf_counter()

        def request(row, conversation, params):
            payload = dict(vars(params))
            structured = payload.pop("structured_outputs", None)
            if structured is not None:
                payload["structured_outputs"] = {"json": structured.json}
            payload.update(
                model=self.config.model.endpoint.served_model_name,
                messages=conversation, chat_template_kwargs=template_kwargs,
            )
            response = self._client.post("chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning:
                # A capped thinking-only result must remain visibly unfinished.
                content = "<think>" + reasoning + ("</think>" + content if content else "")
            usage = result["usage"]
            output = SimpleNamespace(
                prompt_token_ids=range(usage["prompt_tokens"]),
                outputs=[SimpleNamespace(
                    text=content, finish_reason=choice["finish_reason"],
                    token_ids=range(usage["completion_tokens"]),
                )],
            )
            return row, output

        futures = [
            self._requests.submit(request, row, conversation, params)
            for row, conversation, params in zip(records, messages, sampling, strict=True)
        ]
        results = [future.result() for future in futures]
        self._record_timing(phase, started, len(records), results)
        return results

    def close(self):
        self._requests.shutdown(wait=True, cancel_futures=True)
        self._client.close()
