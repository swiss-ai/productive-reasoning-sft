# Project handoff: productive reasoning SFT

Read this file first when working in this repository. `README.md` is the short user guide. The
working research proposal is `/iopsstor/scratch/cscs/tchu/post-training-sync/proposal.md`. The
GitHub repository is `swiss-ai/productive-reasoning-sft`; the Python package and CLI remain named
`synthetic-sft`.

## Goal and research claim

This repository generates and evaluates large-scale synthetic SFT data. The current experiment
filters **unproductive math/reasoning trajectories** before cold-start SFT for Apertus. The claim
to test is narrow: a correctness-preserving productivity filter may give a reasoning RL climb a
better step-zero policy and reduce early rollout waste. It is not assumed to improve the final RL
policy, and this repository does not train SFT or RL checkpoints.

The motivation comes from small Apertus 1.5 8B reasoning-RL runs: many failures exhausted the
reasoning budget without a final answer, often after repeated checking. Doubling the limit helped
only partly at roughly 3.7x wall time. The complementary
[effort-conditioning experiment](https://github.com/swiss-ai/apertus-program/issues/1061) found
that effort labels control length, but longer output is not consistently more accurate and can
truncate more. Yixuan's
[70B RL study](https://claude.ai/code/artifact/97f49e29-a52a-46bf-ba17-81a2b4884445) found that RL
can improve completion and length penalties can shorten responses, while too much pressure can
lose hard-problem coverage. Filtering is therefore a hypothesis to ablate, not a replacement for
RL's learned stopping behavior.

Do not present this project as fixing safety, refusal calibration, general instruction following,
or language drift. Math/reasoning is the starting domain because the observed failure is there and
answers are often verifiable. Transfer to other domains is unproven.

## What counts as productive

A trajectory should finish within its budget and give a complete final response. Each substantial
step should advance the solution, resolve real uncertainty, or add one useful independent check.
The amount of explanation should match the problem: long proofs are welcome when needed, while a
routine calculation should not become a tutorial, an exhaustive search, or a second derivation.
Valid exploration, self-correction, and one independent check are not defects.

Correctness, completion, and productivity are separate judgments:

| Category | Failure | Do not confuse with |
|---|---|---|
| Repetition | A span, calculation, or step repeats without useful change | Brief recap or recurring notation |
| Circular re-checking | A settled result is repeatedly validated with the same argument | One independent check |
| No new progress | A substantial continuation resolves nothing | Necessary exploration |
| Unresolved branch | A contradiction remains open, or the answer relies on an abandoned branch | An explicitly corrected mistake |
| Reasoning-limit stop | Generation hits its cap before clean completion | A long trace that finishes |
| Missing/malformed final | No separate, asserted final answer | Equivalent or conditional answers needing review |

The broad quality arbiter also grades proportionality. Unnecessary input restatement, construction
of unused data, irrelevant branch enumeration, and redundant verification use the `meandering`
issue. A substantive instance is capped below the productivity-filtered threshold.

## Pipeline

```text
normalized prompts + optional source answers
                 │
                 ▼
        sampled teacher solution
                 │
                 ▼
  deterministic polish into separate
       reasoning + final response
                 │
        answer extraction
      + typed/native verification
                 │
        ┌────────┴────────┐
        ▼                 ▼
 deep correctness     short direct local/
 critique             productivity critique
        └────────┬────────┘
                 ▼
        conservative arbiter
                 │
     four focused hygiene checks
        in one batched wave
                 │
                 ▼
 all candidates + clean SFT rows in Parquet
 with correctness-only and productivity-filtered flags
```

The teacher runs with one thinking/effort configuration per launch; never vary effort sample by
sample. The polish call must remove scratch-work artifacts and produce natural training text. It
does not add required answer tags or `\\boxed{}` notation. The raw draft remains available for
audit. Candidate and source answers are extracted independently. The candidate extractor never
sees the expected answer or reasoning trace.

Verification is typed. Numeric expressions, equations, and sets use normalized symbolic
equivalence; Boolean/choice answers use normalized exact match; Reasoning Gym can use its native
scorer. Free text and proofs are not forced through brittle exact matching. Missing source answers
remain unavailable rather than becoming failures. A source answer is strong but fallible evidence:
the arbiter can mark a demonstrated `reference_conflict`.

The global correctness critic keeps thinking enabled. The local mathematical/productivity critic
runs as a separate direct batched wave: with thinking enabled it repeatedly spent its entire token
budget restating inputs before reaching a verdict. The arbiter sees both reviews. Four narrow model
calls then check the semantic hygiene categories; deterministic code checks truncation and
final-answer presence. Semantic findings use `defect`/`clear`/`uncertain` and exact quoted evidence.
A truncated judge input cannot establish `clear`. Failed or malformed checks remain explicit
uncertainty rather than silently passing.

Every rollout is retained. `correctness_only_eligible` requires a complete rollout, an extractable
answer, and verified or high-confidence supported correctness. `productivity_filtered_eligible`
also requires passed hygiene and quality at least 4. There is no top-k selection. Exclusion reasons
and the full structured decision are stored in each row.

Quality scores are:

- 5: training-ready.
- 4: correct with one minor edit.
- 3: needs a substantive local repair.
- 2: useful progress but a major rewrite.
- 1: unusable.
- 0: confirmed incorrectness, incomplete generation, or failure of the core arbiter.

A confirmed hygiene defect caps quality at 2. Hygiene uncertainty caps it at 3. A known-answer
conflict or indeterminate verification caps it at 3. Material issue codes such as meandering,
repetition, or a logical error also prevent an inconsistent score of 4 from entering the strict
view. `quality_details_json` follows schema version 9; the current prompt policy is rubric 19.

## Code map

- `configs/*.yaml`: one resolved launch description covering source, teacher, review, output, and
  Slurm resources.
- `source_pools.py`: download or generate normalized prompt pools; source solutions are archived
  separately and never silently treated as new teacher rollouts.
- `adapters/` and `prepare.py`: deterministic sampling into resumable seed Parquet.
- `generation.py`: teacher generation, polish, answer extraction, critiques, arbitration, and
  hygiene review.
- `continuous.py`: one asynchronous vLLM engine per replica with bounded independent trajectories;
  ready requests progress without waiting for unrelated long ones.
- `quality.py`, `hygiene.py`, `schemas.py`: typed verification, critical prompts, structured
  judgments, score policy, and eligibility.
- `storage.py`: fixed Parquet schemas and atomic candidate checkpoints.
- `pipeline.py`, `cluster.py`, `submit.py`, `slurm/job.sbatch`: one Slurm allocation, a Ray cluster
  across all nodes, automatic pre-timeout requeue, resume, and final Parquet export.
- `scripts/benchmark_pipeline.py`: full-pipeline measurements with per-stage request/token timing.
- `scripts/rejudge_saved.py`: apply another judge to saved real rollouts without regenerating them.
- `scripts/review_digest.py` and `scripts/build_review_site.py`: human inspection artifacts.

## Production candidate

The production launch is `configs/reasoning-easy-production-6m.yaml`:

- teacher/judge: Qwen3.6-35B-A3B-FP8, four independent TP1 replicas per 4xGH200 node;
- no effort label: this model supports thinking but its template does not expose Qwen effort
  controls;
- 600,000 unique elementary prompts, ten stochastic rollouts each, 6 million candidates;
- 20 preemptible nodes, one Slurm submission, automatic restart before each 24-hour limit;
- continuous scheduling with 32 active trajectories and 1,024 queued candidates per GPU actor;
- draft limit 8,192, polish limit 4,096, two 1,024-token critics, and four 768-token focused checks;
- atomic checkpoints every 256 completed rows or 300 seconds.

The source mix is 0.5% Numina GSM8K and 99.5% a purpose-built Reasoning Gym pool. The reusable
`reasoning-gym-production-v4` pool has 910,921 unique prompts. The production seeds use this exact
composition:

| Source/task | Unique prompts | Candidates after 10 rollouts | Intended role |
|---|---:|---:|---|
| Numina GSM8K | 3,000 | 30,000 | external elementary-math anchor |
| calendar arithmetic | 24,000 | 240,000 | short modular/date reasoning |
| fraction simplification | 35,000 | 350,000 | elementary exact arithmetic |
| symbolic GSM | 270,000 | 2,700,000 | varied short word problems |
| knights and knaves | 172,000 | 1,720,000 | compact logic |
| time intervals | 96,000 | 960,000 | simple clock subtraction |

Time-interval generation is restricted to minute- and second-resolution clock subtraction. Date,
datetime, millisecond, and timezone forms were removed after real calibration exposed parsing,
rounding, and offset conflicts. Path-star remains in the reusable pool but is excluded from this
launch: 10/17 calibration samples were rejected, while some clearly exhaustive searches still
passed, so its productivity labels were not reliable enough. Orca Math was removed because
ambiguity and repaired-premise behavior dominated its failures. Decimal arithmetic and
gimmicky/challenge Reasoning Gym tasks are also excluded. The prepared production seeds contain
exactly 600,000 unique IDs with the counts above and no path-star rows under
`runs/reasoning-productive-easy-qwen36-6m-v1/prepared/seeds/`.

The source is intentionally easy and unambiguous. This dataset is meant to teach clean completion
and stopping before RL, not to replace RL with hard synthetic SFT. Difficulty and source-specific
fields live in `provenance_json`, not task-specific top-level columns.

Reusable but currently inactive pools include NVIDIA OpenMathReasoning, DeepMath-103K,
DeepScaleR Preview, and OpenThoughts3-1.2M. Preserve their supplied solutions as references. Hard
math needs a separate teacher/effort configuration and its own quality calibration.

## Measured systems evidence

The original barriered implementation starved the GPUs. Continuous per-candidate scheduling,
larger actor queues, fixed actor assignment, and durable in-actor checkpoints removed that
bottleneck. Do not sum overlapping request durations as GPU wall time.

- A one-node, 1,024-prompt full-pipeline run reached 1,266 candidates/hour after engines were
  ready and 1,214/hour including startup. It used the older noisy source mix and policy, so its
  useful-yield rate is not a production quality estimate.
- The exact four-node production-path soak was Slurm job 3549813. It completed all 8,192 unique
  candidates and final exports in 21:21. The generation/review stage took 1,172.7 seconds including
  engine startup and tail drain: 6,287 candidates/node-hour. After engines were ready, the bounded
  run including its tail was about 6,986 candidates/node-hour. All 16 actors stayed fed.
- The production source weights are about 7% more output-heavy than that soak. Conservatively
  scaling the warm rate gives about 6,529 candidates/node-hour. Six million candidates therefore
  project to about 46.0 wall-clock hours on 20 nodes, or about 920 node-hours, plus only the small
  final CPU export. This is an evidence-based projection with roughly 8% budget margin, not a
  guarantee against queue delays or unusual preemption churn.
- The full topology gate was Slurm job 3549885: exactly 20 nodes, 80 TP1 actors, and 10,240 unique
  candidates. It completed all manifests with exit `0:0` in 10:50. The generation/review stage took
  414.1 seconds. Its 128 rows per actor intentionally test topology, not steady-state rate; startup
  and tail drain dominate this bounded run.
- Critics consumed most non-draft output. Reducing each critic from 2,048 to 1,024 tokens roughly
  halved critic tokens and reduced median critic latency from about 28 to 15 seconds. The small
  bounded comparison improved end-to-end time only about 7% because requests overlap. Keep the
  current 1,024-token budget until quality evidence supports a further reduction.
- Qwen engine startup is about 90-100 seconds with the shared compilation caches warm.

Swiss Model Launcher was also exercised with DeepSeek V4.1 Flash through the same pipeline. Its
aligned 64-prompt run reached about 865 candidates/node-hour on two TP8 nodes, substantially below
Qwen's node efficiency. On saved Qwen rollouts it was more critical about ambiguous correctness,
but it missed at least one long redundant recalculation that Qwen rejected. It is not the current
production judge. SML remains useful for future long-lived model services and stronger hard-math
teachers; it is a deployment layer over engines such as vLLM/SGLang, not an automatic throughput
fix.

The automatic restart path was tested as Slurm job 3549672. The first attempt checkpointed work,
received the pre-timeout signal, and requeued. The restarted two-node allocation recovered exactly
16 unique candidates, finalized all manifests, and exited `COMPLETED` with code `0:0`. Ray
rendezvous and completion sentinels include the Slurm restart count, so workers cannot attach to a
stale head process.

The v0.21 image adds an unconstrained, schema-validated recovery only after both structured arbiter
attempts fail. On the 20-node topology gate, 148/10,240 rows reached this fallback and 147 recovered;
one malformed result remained an explicit conservative judge error (0.01%). This reduced the prior
1.37% structured-decoding failure rate without relaxing any validator or eligibility rule.

## Calibration evidence and inspection

Rubrics 16-18 exposed overlong graph/date reasoning and insufficiently problem-relative judging.
Rubric 19 is the current policy: one deep correctness critique plus one short direct productivity
critique. Graph paths and unstable time variants were removed rather than trusting noisy labels.

The final exact-source calibration is `reasoning-production-calibration-128-v6`, Slurm job 3549984.
It used the v4 pool, production weights, v0.21 image, generation settings, and rubric 19:

- 128/128 unique candidates completed with no incomplete generations or judge errors;
- scores were `2:3, 3:27, 5:98`; 122 rows were correctness-only eligible and 98 entered the strict
  view;
- correctness was 123 verified and five conservative reference conflicts; hygiene was 124 passed,
  one failed, and three indeterminate;
- reasoning length was 192 tokens at the median, 607 at p90, 672 at p95, and 1,781 maximum;
- manual review covered every rejection and the 15 longest strict passes. It found useful rejection
  of an inconsistent source problem, circular date checking, unnecessary second derivations, and
  disproportionate routine arithmetic. The retained long tail was substantive three-person logic
  with at most one consistency check. No obvious unsafe false pass was found.

Several correct rows are deliberately excluded for borderline redundant checking. That is a
conservative yield loss, not deletion: every candidate remains queryable. A malformed Numina
reference extraction also caused one correct candidate to be excluded. The strict flag is not a
statistically calibrated classifier; the matched SFT/RL ablation remains the research test.

The owner approved the production launch. Slurm job **3551775** was submitted on 2026-09-30 at
13:09 CEST with the resolved 6M-candidate configuration above. It was preempted six times: after
6:54:06, then after 2:08, 4:14, 26:27, 11:12, and 14:41. Slurm then placed it in `REQUEUE_HOLD`
with reason `launch_failure_limit_exceeded_requeued_held`; accounting records every ended attempt
as `PREEMPTED` with exit `0:0`, and the final log shows all 80 actors serving requests immediately
before `SIGTERM`. This is scheduler preemption churn, not a pipeline exception.

The checkpoints contain 1,088,297 unique candidates with zero duplicates: 1,047,856 are
correctness-only eligible and 869,484 are productivity-filtered eligible. The six allocations used
about 157.6 node-hours and achieved roughly 6,905 candidates/node-hour including repeated startup
losses. At that realized rate, the remaining 4,911,703 candidates need about 35.6 active wall-clock
hours on 20 nodes, keeping the projected total near 869 node-hours. The held job 3551775 was
cancelled, without touching its checkpoints, and replaced by Slurm job **3558380** on 2026-10-01.
The replacement initially entered `PENDING (Priority)` with `Restarts=0`, `Requeue=1`, and the same
20-node/80-GPU request. Job 3558380 was subsequently preempted five times on October 1
(57:54, 3:07:09, 14:17, 1:07:52, 1:35:06) and held again. On October 2, the owner requested
status/recovery; the held job was cancelled and replaced by **3567665**, using the same config
and checkpoints. It started on 20 nodes with `Restarts=0`. Job 3567665 is authoritative;
do not revive either cancelled predecessor or create a concurrent duplicate. This recovery
does not remove the cluster's automatic requeue limit; further preemptions can cause another hold.

The October 2 recovery audit counted 2,034,955 durable rows, all unique (33.9% of the 6M target).
Of these, 1,959,848 are correctness-only eligible and 1,622,949 are productivity-filtered eligible.
Scores 0–5: 27,280 / 3,723 / 15,235 / 365,768 / 233 / 1,622,716. These are pipeline labels,
not a fresh manual quality audit. The two completed jobs consumed about 298.4 node-hours;
the remaining work projects to roughly 29.1 active hours on 20 nodes at their combined rate,
excluding future queue delays and changes in preemption frequency.

Inspect a completed run with:

```bash
uv run python scripts/review_digest.py \
  runs/<run-id>/candidates runs/<run-id>/review/digest.md
uv run python scripts/build_review_site.py \
  runs/<run-id>/candidates runs/<run-id>/review/all.html
```

Review every rejected, failed, or uncertain sample, then the longest strict passes and a random
pass sample. Report scores 0-5, correctness, completion, hygiene/category rates, token-length
distributions, and useful yield by source. Counts alone do not validate the filter. `docs/` is an
older illustrative meeting gallery, not a calibration result.

## Launch procedure

There is no reservation. Use ordinary `preemptable` or `normal` Slurm capacity and never assume
elevated QOS. Build pools and the versioned image, run and inspect the calibration, then dry-run the
large launch:

```bash
uv run synthetic-sft build-pools configs/source-pools.yaml
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.21.sqsh"
uv run synthetic-sft submit configs/reasoning-production-calibration-128.yaml
uv run synthetic-sft submit --dry-run configs/reasoning-easy-production-6m.yaml
```

Production was submitted with explicit owner approval. The original job 3551775 was cancelled
after exhausting its automatic requeue allowance; job 3558380 exhausted its allowance too.
Current replacement job 3567665 resumes the same run.
Do not submit another copy. Use a new `run_id` whenever the model, source, sampling, or quality
policy changes; completed stages have manifests tied to the resolved configuration. A resumed run
skips durable candidate IDs. In-flight requests can be lost at preemption, but completed groups
cannot.

Each run contains `prepared/seeds/`, `intermediate/generated/`, `candidates/`, `sft/`, `logs/`, and
`manifests/`. The SFT export contains every candidate with clean training columns, provenance,
quality details, and both eligibility flags.

## Downstream decision experiment

The actual research test is a matched correctness-only versus productivity-filtered SFT ablation.
Match prompt/source composition, base checkpoint, training-token budget, chat template, trainer,
and later RL settings. Run the same short reasoning RL climb from both checkpoints, including the
same length objective. Compare step zero and learning curves for completed correct answers, parse
failures, limit hits, repetition, tokens per correct completed answer, pass@1/pass@k, non-zero
reward coverage, all-zero rollout groups, and hard-problem coverage. A better step zero with the
same eventual policy may still save rollout compute; lower hard-problem coverage means the filter
is too aggressive. Final response length alone is not the success metric.

## Backlog and boundaries

- Measure filter error on a larger manually labeled sample before treating the strict flag as a
  calibrated classifier. Switch judges only if narrow Qwen checks remain too lenient or unreliable.
- Do not enable critique-guided rewriting by default. Consider paired revision only if rejection
  exceeds 80% on representative data; preserve originals and independently reverify revisions.
- Keep source trajectories as archived references. A source-solution SFT arm requires an explicit
  experiment, not silent mixing.
- GLM-5.2 or another stronger SML-served teacher is a future option for genuinely hard math.
- Candidate future pools include `PrimeIntellect/SYNTHETIC-2-Base-v2`,
  `open-r1/OpenR1-Math-220k`, and `nvidia/OpenCodeReasoning`; deduplicate and verify them first.

## Working conventions

Preserve user data and unrelated changes. Commit each reasonable checkpoint using `task: details`
subjects. Prefer real Slurm runs and manual trace review over tests that only assert counts or
mirror implementation logic. Report failures and uncertainty plainly; a completed job is not proof
that a teacher or filter is calibrated. Update this file whenever the verified run procedure or a
production decision changes.
