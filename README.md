# Productive Reasoning SFT

Generate and evaluate large-scale synthetic SFT data on one or many Slurm nodes. The current focus
is **productive math/reasoning trajectories**: reasoning that reaches a correct final answer without
getting stuck in repeated checking or continuing without progress. This is meant to test whether
cleaner cold-start SFT improves the starting point and early efficiency of a later RL climb. It is
not an assumption that shorter reasoning is always better or that RL cannot learn to stop itself.

For the full project context, pipeline design, pilot mix, and backlog, see [AGENTS.md](AGENTS.md).

1. A varied math/reasoning source produces prompts and provenance.
2. A large teacher model solves each prompt, then rewrites its scratch work into clean reasoning
   and a final answer; both are stored separately.
3. The final answer is checked where possible. Critical reviewers judge correctness and general
   quality; focused checks separately flag repeated steps, circular checking, stalled progress,
   unresolved branches, reasoning-limit stops, and missing or malformed final answers.
4. Every rollout is written to Parquet. Each row says whether it qualifies for a correctness-only
   or productivity-filtered SFT dataset (passing hygiene and quality ≥4), with explicit reasons
   when it does not. No top-k cutoff
   forces good or bad samples into either dataset.

Filtering criteria are distinct: repeated spans/steps, circular re-checking, continuation without
new progress, unresolved contradictions or branches, reasoning-limit stops, and missing or malformed
final answers. A long trace can pass when its steps are useful and it finishes. The focused model
checks currently use Qwen and must be calibrated by manual review before their labels are trusted
for a large training run.

## Run it

Prepare the reusable source pools once. Supplied solutions are preserved separately from prompts:

```bash
uv run synthetic-sft build-pools configs/source-pools.yaml
```

Build the image, then submit a run. The example draws a deterministic 1,000-prompt mixture with
50% of each difficulty-aware source allocated to its hard band:

```bash
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.5.sqsh"
uv run synthetic-sft submit configs/reasoning-productivity-1000.yaml
```

For a different run size, change `source.num_samples`; the source mixture is sampled from the same
pools without downloading or rebuilding them. A single-source Reasoning Gym smoke run remains
available:

```bash
uv run synthetic-sft submit configs/reasoning-gym-smoke.yaml
```

The YAML controls the source, model, number of rollouts, judging, output location, and
Slurm resources. A single submission uses every requested node and GPU.

## Results

Each run is written under `output_dir/run_id`:

- `candidates/` contains every rollout plus operational and verification details.
- `sft/` contains every rollout with stable training columns and both selection flags.
- `manifests/` records the exact run configuration and quality-details schema.

SFT rows contain `sample_id`, `candidate_id`, `system_prompt`, `user_prompt`, `reasoning`, `response`,
`reasoning_num_tokens`, `response_num_tokens`, `answer_json`, `correctness_verdict`, `source`, `model`,
`quality_score`, `quality_details_json`, `hygiene_status`, `correctness_only_eligible`,
`productivity_filtered_eligible`,
`exclusion_reasons_json`, and `provenance_json`.

The Parquet datasets can be queried directly:

```sql
SELECT source, correctness_only_eligible, productivity_filtered_eligible, count(*)
FROM read_parquet('runs/reasoning-productivity-1000-v1/sft/**/*.parquet')
GROUP BY source, correctness_only_eligible, productivity_filtered_eligible;
```

For the planned ablation, match prompts, effort, and training budget between the correctness-only
and productivity-filtered SFT selections. Evaluate both checkpoints before RL and through the same
short RL climb. Compare completed correct answers, reasoning-limit hits, repetition, reasoning
tokens per correct answer, early reward coverage, and hard-problem solution coverage. The filter is
useful only if cleaner stopping does not merely remove valuable exploration.

## Using another source

`reasoning_gym` creates tasks with known answers. `parquet` reads existing prompts and works
naturally when no answer is available; verification is simply unavailable and judging supplies
the quality signal. Source-specific fields are preserved inside `provenance_json`.

Use a new `run_id` whenever you change the model, prompts, sampling, or quality policy. Restarting
an interrupted run keeps every completed rollout and generates only the missing rows.
