# synthetic-sft

Generate and judge synthetic SFT data on one or many Slurm nodes.

1. The configured source produces prompts and provenance.
2. The model solves each prompt, then rewrites its scratch work into clean reasoning and a final
   answer; both are stored separately.
3. Known answers are verified when available. A judge grades reasoning and the final response
   separately from 1–5 using fixed, edit-readiness milestones; the lower grade is the quality
   score. Failed generation or verification is marked `0` with a structured reason.
4. Every rollout is written to the SFT dataset. Filtering is left to the downstream query, so an
   unusually good or bad batch is never distorted by a fixed top-k rule.

## Run it

Edit `configs/reasoning-gym-smoke.yaml`, then submit one job:

```bash
uv run synthetic-sft submit configs/reasoning-gym-smoke.yaml
```

The YAML controls the source, model, number of rollouts, judging, output location, and
Slurm resources. A single submission uses every requested node and GPU.

To build the image first:

```bash
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.1.sqsh"
```

## Results

Each run is written under `output_dir/run_id`:

- `candidates/` contains every rollout plus operational and verification details.
- `sft/` contains every rollout with stable training columns, ready to query by quality.
- `manifests/` records the exact run configuration and quality-details schema.

SFT rows contain `sample_id`, `candidate_id`, `system_prompt`, `user_prompt`, `reasoning`, `response`,
`source`, `model`, `quality_score`, `quality_details_json`, and `provenance_json`.

The Parquet datasets can be queried directly:

```sql
SELECT source, quality_score, count(*)
FROM read_parquet('runs/reasoning-gym-all-tasks-quality-v4/sft/**/*.parquet')
GROUP BY source, quality_score;
```

## Using another source

`reasoning_gym` creates tasks with known answers. `parquet` reads existing prompts and works
naturally when no answer is available; verification is simply unavailable and judging supplies
the quality signal. Source-specific fields are preserved inside `provenance_json`.

Use a new `run_id` whenever you change the model, prompts, sampling, or quality policy. Completed
stages resume automatically; incomplete outputs are preserved for inspection.
