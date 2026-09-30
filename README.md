# Productive Reasoning SFT

Generate and evaluate large-scale synthetic SFT data on one or many Slurm nodes. The current focus
is **productive math/reasoning trajectories**: reasoning that reaches a correct final answer without
getting stuck in repeated checking or continuing without progress. This is meant to test whether
cleaner cold-start SFT improves the starting point and early efficiency of a later RL climb. It is
not an assumption that shorter reasoning is always better or that RL cannot learn to stop itself.

For the full project context, measured results, launch plan, and backlog, see [AGENTS.md](AGENTS.md).
The [pilot review site](docs/index.html) shows pass/reject examples from an earlier development
batch. It is useful for understanding the output, not for estimating filter accuracy.

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

Build the image, then run the 128-sample calibration. It uses the same model, sources, generation,
and review policy as the large launch:

```bash
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.19.sqsh"
uv run synthetic-sft submit configs/reasoning-production-calibration-128.yaml
```

The production candidate is `configs/reasoning-easy-production-1p84m.yaml`. It contains 230,000
unique, elementary prompts and produces eight independent rollouts per prompt: 1.84 million
candidates in total. The mix is intentionally easy and unambiguous because its purpose is to prime
a policy for RL, not to use difficult synthesis as a substitute for RL. Submit it only after the
calibration has been manually reviewed:

```bash
uv run synthetic-sft submit --dry-run configs/reasoning-easy-production-1p84m.yaml
uv run synthetic-sft submit configs/reasoning-easy-production-1p84m.yaml
```

One submission uses all requested nodes and GPUs. Completed candidates are checkpointed while the
job is running. Preemption or a planned 24-hour restart resumes from those checkpoints and skips
finished candidate IDs.

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
FROM read_parquet('runs/reasoning-productive-easy-qwen36-1p84m-v4/sft/**/*.parquet')
GROUP BY source, correctness_only_eligible, productivity_filtered_eligible;
```

To inspect a pilot, make a self-contained review page and a compact index:

```bash
uv run python scripts/build_review_site.py runs/<run_id>/candidates runs/<run_id>/review/all.html
uv run python scripts/review_digest.py runs/<run_id>/candidates runs/<run_id>/review/digest.md
```

The page shows the prompt, final answer, focused findings, verifier basis, judge feedback, and
expandable full reasoning. For a meeting gallery, pass `--curation review.json` to the page builder;
that JSON is a list of `{ "candidate_id": "...", "label": "...", "note": "..." }` entries, with
an optional exact `excerpt` from the reasoning. A curated gallery is illustrative, not a measure
of filter accuracy. Keep the full run for auditing false passes and false rejections.

For the planned ablation, match prompts, effort, and training budget between the correctness-only
and productivity-filtered SFT selections. Evaluate both checkpoints before RL and through the same
short RL climb. Compare completed correct answers, reasoning-limit hits, repetition, reasoning
tokens per correct answer, early reward coverage, and hard-problem solution coverage. The filter is
useful only if cleaner stopping does not merely remove valuable exploration.

## Adding another source

New sources are first normalized into prompt and reference artifacts. A source answer is optional:
when it is absent, deterministic verification stays unavailable and the critical reviews provide
the quality signal. Source-specific metadata is preserved inside `provenance_json`.

Use a new `run_id` whenever you change the model, prompts, sampling, or quality policy. Restarting
an interrupted run keeps every completed rollout and generates only the missing rows.
