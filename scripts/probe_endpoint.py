"""Exercise thinking, nonthinking and structured output on a real private endpoint."""

import argparse
import json
import os
import time
from pathlib import Path

import httpx

from synthetic_sft.config import load_config
from synthetic_sft.schemas import AnswerExtraction


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=1200)
    args = parser.parse_args()
    config = load_config(args.config)
    endpoint = config.model.endpoint
    args.output.mkdir(parents=True, exist_ok=False)
    headers = {}
    key = os.environ.get(endpoint.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(
        base_url=os.environ[endpoint.base_url_env].rstrip("/") + "/",
        timeout=endpoint.timeout_seconds, trust_env=False, headers=headers,
    ) as client:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            try:
                response = client.get("models", timeout=5)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(5)
        (args.output / "models.json").write_text(response.text)
        probes = [
            ("thinking", "Solve 17 * 19. Give a short derivation and the answer.",
             {"enable_thinking": True,
              "reasoning_effort": config.model.sampling.reasoning_effort}, None),
            ("nonthinking", "Reply with exactly: ready", {"enable_thinking": False}, None),
            ("structured", 'Extract the final answer from "17 * 19 = 323". '
             'Return the numeric answer using the supplied JSON schema.',
             {"enable_thinking": False}, AnswerExtraction.model_json_schema()),
        ]
        for name, prompt, template, schema in probes:
            payload = dict(
                model=endpoint.served_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=2048, chat_template_kwargs=template,
            )
            if schema:
                payload["structured_outputs"] = {"json": schema}
            (args.output / f"{name}-request.json").write_text(json.dumps(payload, indent=2))
            started = time.perf_counter()
            response = client.post("chat/completions", json=payload)
            (args.output / f"{name}.json").write_text(response.text)
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if choice["finish_reason"] != "stop" or not message.get("content"):
                raise RuntimeError(f"{name}: incomplete response; inspect saved output")
            if name == "thinking" and not reasoning:
                raise RuntimeError("Thinking request did not return separate reasoning")
            if name != "thinking" and reasoning:
                raise RuntimeError("Nonthinking request unexpectedly returned reasoning")
            if schema:
                AnswerExtraction.model_validate_json(message["content"])
            print(json.dumps({
                "probe": name, "seconds": time.perf_counter() - started,
                "usage": result["usage"], "content": message["content"],
            }), flush=True)


if __name__ == "__main__":
    main()
