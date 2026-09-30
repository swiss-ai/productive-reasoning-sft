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

The production launch is `configs/reasoning-easy-production-1p89m.yaml`:

- teacher/judge: Qwen3.6-35B-A3B-FP8, four independent TP1 replicas per 4xGH200 node;
- no effort label: this model supports thinking but its template does not expose Qwen effort
  controls;
- 210,000 unique elementary prompts, nine stochastic rollouts each, 1.89 million candidates;
- 20 preemptible nodes, one Slurm submission, automatic restart before each 24-hour limit;
- continuous scheduling with 32 active trajectories and 1,024 queued candidates per GPU actor;
- draft limit 8,192, polish limit 4,096, two 1,024-token critics, and four 768-token focused checks;
- atomic checkpoints every 256 completed rows or 300 seconds.

The source mix is 1.5% Numina GSM8K and 98.5% a purpose-built Reasoning Gym pool. The full pool has
251,205 unique prompts; the launch samples only these five calibrated tasks:

| Task | Pool size | Intended role |
|---|---:|---|
| calendar arithmetic | 27,449 | short modular/date reasoning |
| fraction simplification | 39,199 | elementary exact arithmetic |
| symbolic GSM | 53,823 | varied short word problems |
| time intervals | 55,734 | simple clock subtraction |
| knights and knaves | 37,500 | compact logic |

Time-interval generation is restricted to minute- and second-resolution clock subtraction. Date,
datetime, millisecond, and timezone forms were removed after real calibration exposed parsing,
rounding, and offset conflicts. Path-star remains in the reusable pool but is excluded from this
launch: 10/17 calibration samples were rejected, while some clearly exhaustive searches still
passed, so its productivity labels were not reliable enough. Orca Math was removed because
ambiguity and repaired-premise behavior dominated its failures. Decimal arithmetic and
gimmicky/challenge Reasoning Gym tasks are also excluded. The prepared production seeds contain
210,000 unique IDs with no path-star rows under
`runs/reasoning-productive-easy-qwen36-1p89m-v1/prepared/seeds/`.

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
- A later two-node, 2,048-candidate soak reached about 3,450 candidates/node-hour including
  startup and roughly 4,250/node-hour warm. At that measured rate, 1.89 million candidates on 20
  nodes require about 27.4 wall-clock hours (about 548 node-hours), before queue/preemption overhead. This is comfortably
  inside a 1,000-node-hour budget but remains a projection until the large launch runs.
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

## Calibration evidence and inspection

The rubric-16 production-source calibration (`reasoning-production-calibration-128-v1`) completed
all 128 candidates in 4:08. It produced 121 strict rows, 122 correctness-only rows, and scores
`0:2, 2:1, 3:4, 5:121`; all generation calls completed. Manual review found:

- one genuinely wrong weekday and one genuinely wrong day-count were conservatively rejected;
- three correct time answers conflicted with faulty millisecond/timezone references and were kept
  out of the strict view;
- one correct timezone answer was also conservatively rejected after an arbiter arithmetic error;
- the focused checks did not fabricate a confirmed hygiene defect, but two unanchored/unsupported
  findings remained uncertainty;
- several accepted graph/date traces were correct yet disproportionally long, motivating rubric 17
  and the tighter polish prompt.

The rubric-17 follow-up reduced median reasoning from 324 to 236 tokens and the longest strict pass
from 2,209 to 1,135. It also exposed two accepted graph traces that still copied the input and
enumerated dead branches. Rubric 18 therefore makes the second critique explicitly audit
problem-relative length, with concrete graph/date/arithmetic criteria; generic concision wording
was not sufficient.

Rubric 19 separates the critics into a deep thinking correctness wave and a direct productivity
wave. In the exact no-path production mix, Slurm job 3549808 completed 128/128 candidates in 3:35:

- scores were `0:1, 3:34, 5:93`; 125 rows were correctness-only eligible and 93 entered the strict
  view;
- all generations completed; correctness was 125 verified, two reference/extractor conflicts,
  and one confirmed incorrect answer;
- hygiene was 126 passed, one failed, and one indeterminate;
- reasoning length was 177 tokens at the median, 530 at p90, 599 at p95, and 835 maximum;
- the filter caught a same-day flight incorrectly treated as crossing midnight, redundant second
  methods, full calendar enumerations, unnecessary primality proofs, and repeated logic checks;
- manual review covered every rejection and the 15 longest strict passes. The retained long tail
  consisted of substantive logic derivations and self-contained arithmetic, not graph-search
  rabbit holes.

This gate demonstrates useful conservative behavior, not a statistical estimate of filter error.
Several correct rows were deliberately excluded for modest redundancy, and one malformed answer
extraction created a conservative reference conflict. Every row remains queryable, so those false
positives reduce strict yield rather than delete data. The production candidate is ready for owner
approval, but the 20-node job has not been submitted.

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
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.20.sqsh"
uv run synthetic-sft submit configs/reasoning-production-calibration-128.yaml
uv run synthetic-sft submit --dry-run configs/reasoning-easy-production-1p89m.yaml
```

Do not submit the 20-node launch without explicit owner approval. Use a new `run_id` whenever the
model, source, sampling, or quality policy changes; completed stages have manifests tied to the
resolved configuration. A resumed run skips durable candidate IDs. In-flight requests can be lost
at preemption, but completed groups cannot.

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
