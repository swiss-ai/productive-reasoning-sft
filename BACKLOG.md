# Backlog

## Critique-guided reasoning rewrite (only if filtering is too destructive)

The default pipeline **evaluates and selects** teacher rollouts; it does not rewrite them in
response to judge findings. If calibration shows that otherwise correct traces frequently contain
small, repairable loops—or the productivity filter rejects too much useful long reasoning—evaluate
a separate repair stage. Keep the judge's verdict and supporting passages separate from the
candidate; ask a generator to revise only the substantiated defect, then re-run independent
correctness and hygiene checks on the revised trace. Preserve the original and revised versions
for a paired quality comparison. Do not enable this merely to inflate filter yield: a rewrite that
hides errors, removes necessary exploration, or changes the final answer is worse than exclusion.

First decide from manually reviewed real rollouts whether prompt calibration or a different
critical judge is sufficient. A rewrite stage is justified only if it improves verified quality
without materially increasing cost or flattening the difficulty-dependent reasoning length.

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
