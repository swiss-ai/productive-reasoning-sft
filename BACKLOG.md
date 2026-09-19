# Backlog

## Swiss Model Launcher and stronger teacher models

Adopt [Swiss Model Launcher](https://github.com/swiss-ai/model-launch) as the
intended long-term model-serving path, with GLM-5.2 for the harder question
sets. This may eventually replace the current vLLM execution path.

Postpone this migration until the core pipeline is settled: dataset preparation,
separate reasoning and final-answer capture, correctness verification, structured
1–5 judging, provenance, resumability, and Parquet output. Swiss Model Launcher
runs are comparatively compute-heavy, so the current vLLM path remains the fast
iteration backend in the meantime.

Before switching production generation:

- add Swiss Model Launcher as a selectable backend without changing the dataset
  or output interfaces;
- prepare a GLM-5.2 run profile for hard questions;
- confirm reasoning and final-answer extraction independently;
- validate correctness and judge-output compatibility on a representative hard
  sample;
- compare quality and useful throughput against the strongest practical local
  Qwen profile.
