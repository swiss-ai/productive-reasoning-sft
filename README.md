# synthetic-sft

Generate and rank synthetic SFT data on one or many Slurm nodes.

1. The configured source produces prompts and provenance.
2. The model produces one or more rollouts per prompt, with reasoning and the final answer stored
   separately.
3. Known answers are verified when available, and the configured judge scores every rollout.
   Incomplete or failed rollouts remain in the data with quality `0` and a structured reason.
4. The best nonzero-scored rollouts per prompt are written to a clean SFT dataset. Every
   rollout—including zero-score and unselected rows—remains available for analysis.

## Run it

Edit `configs/reasoning-gym-smoke.yaml`, then submit one job:

```bash
uv run synthetic-sft submit configs/reasoning-gym-smoke.yaml
```

The YAML controls the source, model, number of rollouts, judging, selection, output location, and
Slurm resources. A single submission uses every requested node and GPU.

To build the image first:

```bash
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.1.sqsh"
```

For a quick local check that does not use GPUs:

```bash
uv sync --extra dev
uv run pytest
```

## Results

Each run is written under `output_dir/run_id`:

- `candidates/` contains every rollout, its quality score and details, errors, rank, and selection
  decision.
- `selected/` contains only the chosen SFT rows with stable training columns.
- `manifests/` records the exact run configuration and quality-details schema.

Selected rows contain `sample_id`, `system_prompt`, `user_prompt`, `reasoning`, `response`,
`source`, `model`, `quality_score`, `quality_details_json`, and `provenance_json`.

The Parquet datasets can be queried directly:

```sql
SELECT selected, quality_score, count(*)
FROM read_parquet('runs/reasoning-gym-judged-smoke/candidates/**/*.parquet')
GROUP BY selected, quality_score;
```

## Using another source

`reasoning_gym` creates tasks with known answers. `parquet` reads existing prompts and works
naturally when no answer is available; verification is simply unavailable and judging supplies
the quality signal. Source-specific fields are preserved inside `provenance_json`.

Use a new `run_id` whenever you change the model, prompts, sampling, or quality policy. Completed
stages resume automatically; incomplete outputs are preserved for inspection.
