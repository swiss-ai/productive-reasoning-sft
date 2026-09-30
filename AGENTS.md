# Project handoff: synthetic SFT for productive reasoning

Read this file first when working in this repository. It consolidates the project direction,
pipeline design, launch plan, and backlog; `README.md` is the short user guide. The working issue
text is `/iopsstor/scratch/cscs/tchu/post-training-sync/proposal.md`. The GitHub repository is
`swiss-ai/productive-reasoning-sft`; the Python package and CLI remain named `synthetic-sft`.

## Goal and motivation

Build a readable, scalable pipeline for large-scale, high-quality synthetic SFT data. The current
research focus is **filtering unproductive reasoning** from math/reasoning trajectories before
cold-start SFT for Apertus. The intended downstream question is whether this improves the policy
at RL step zero and early RL efficiency, without reducing correctness or hard-problem solution
coverage. This repository prepares/evaluates SFT data; it does not train SFT or RL checkpoints.

Our small Apertus 1.5 8B reasoning-RL ablation found that many failed trajectories exhausted the
reasoning limit without a final answer; repeated checking was a leading cause. Doubling the limit
helped only partly at roughly 3.7× wall time, and more RL without a length penalty improved reward
without removing degeneration. The complementary [effort-conditioning ablation](https://github.com/swiss-ai/apertus-program/issues/1061)
showed that effort labels control length but longer output is not consistently more accurate and
can truncate more. Yixuan's [Apertus 1.5 70B RL study](https://claude.ai/code/artifact/97f49e29-a52a-46bf-ba17-81a2b4884445)
showed that RL can itself improve completion and a length penalty can shorten responses further;
too much length pressure can lose hard-problem coverage. Therefore **SFT filtering is a hypothesis
to ablate, not an assumed substitute for RL's stopping behavior**.

Do not frame this as fixing general instruction following, refusal calibration, safety, or language
drift. Math/reasoning is the starting domain because the failure was observed there and answers
are often verifiable. Transfer to other tasks remains unproven. Difficulty and source-specific
parameters live in provenance JSON, not task-specific top-level output columns.

## What productive reasoning means

A trajectory should close within its budget and produce a complete final response. Each
substantial step should advance the solution, resolve genuine uncertainty, or add a useful check.
Checking should stop when the answer is adequately supported. Long solutions are welcome when the
problem needs them; a universal short-trace threshold would be counterproductive. Valid
self-correction and one genuinely independent check must not be mistaken for failure.

Correctness, completion, and productivity are separate judgments. The six hygiene categories are:

| Category | Specific failure | Avoid false positives on |
|---|---|---|
| Repetition | Repeated span, calculation, or step without useful change | Harmless recap or notation |
| Circular re-checking | Revalidating a settled result with essentially the same argument | One useful independent check |
| No new progress | Long continuation resolving nothing, even without repeated wording | Necessary exploration |
| Unresolved branch | Contradiction left unresolved or final answer relying on an abandoned branch | Explicitly corrected mistakes |
| Reasoning-limit stop | Token cap reached before clean completion | A long trace that does finish |
| Missing/malformed final | Separate response absent or no answer asserted | Equivalent or conditional answers that need review |

Semantic checks use narrow critical prompts with structured `defect`/`clear`/`uncertain` results,
quoted evidence, and explanations. Exact quote anchoring prevents invented evidence but does not
guarantee a correct verdict. Deterministic checks handle observable stops and answer presence; a
long repeated-span signal assists the model, with only extreme repetition auto-confirmed. A
truncated judge input cannot justify `clear`. Calibrate false positives and false negatives on
manually reviewed **real rollouts**, including long successful reasoning and repeated-checking
failures. Qwen is the initial judge for fast iteration; replace it if its narrow checks are still
too lenient or unreliable.

## Data path and code map

```text
source pools / prepared prompts
  → one or more teacher rollouts per prompt (thinking/effort set per run)
  → polish scratch work into separate reasoning + final response
  → extract final answer and source answer, when available
  → native or typed deterministic verification + two critical analyses + arbiter
  → one batched wave of four focused semantic hygiene checks
  → deterministic length-stop/final-answer checks
  → all candidates and SFT rows in Parquet, with quality and eligibility metadata
  → later matched correctness-only vs productivity-filtered SFT/RL ablation
```

- `src/synthetic_sft/source_pools.py`: downloads/normalizes the source datasets. Prompt and
  reference trajectories are separate pool artifacts. Archived source traces are **not** silently
  added as newly generated SFT candidates.
- `src/synthetic_sft/adapters/`: generic source interfaces, Reasoning Gym, Parquet, and stratified
  pool sampling. Sources may have no answer; verification then remains unavailable rather than
  treating the sample as wrong.
- `prepare.py`: deterministic, resumable seed Parquet preparation.
- `config.py`: the one YAML run configuration: source, model/effort, generation, judge, output,
  and Slurm resources. Use one effort setting per run, never sample-wise effort switching.
- `generation.py`: batched Ray/vLLM actor. It performs teacher generation, polishing, answer
  extraction, parallel critiques, arbitration, and the four focused checks. It records exact
  teacher-tokenizer counts for reasoning and final response. Same-model judging shares the loaded
  actor; a different judge `model_source` uses a separate actor.
- `continuous.py`: optional `model.execution: continuous` execution using one AsyncLLM per
  replica and bounded independent trajectories. It reuses every generation/quality stage and
  lets ready requests enter the engine without waiting for unrelated long trajectories.
- `storage.py`: fixed candidate Parquet schema and atomic completion checkpoints. Continuous
  generation writes completed groups (`output.checkpoint_rows`, or `checkpoint_seconds`)
  directly, avoiding Ray's large output-block buffering. Actor retries are disabled; resume
  skips durable candidate IDs. In-flight work can still be lost on preemption.
- `scripts/benchmark_pipeline.py`: full-pipeline measurements on saved real prompts, including
  polishing and all judges. Saves outputs and per-stage request/token timings. Continuous
  request durations overlap; do not sum them as GPU wall time.
- `quality.py`: parse reasoning/response, check answer with the source adapter, combine the
  correctness verdict, 1–5 quality, hygiene, and selection flags.
- `hygiene.py`: category-specific critical prompts, exact-evidence validation, bounded repeated
  span scan, deterministic checks, and combined hygiene status.
- `schemas.py`: strict JSON schemas for answers, quality, hygiene findings, decisions, and flags.
- `pipeline.py`, `cluster.py`, `submit.py`, `slurm/job.sbatch`: one Slurm allocation starts a Ray
  cluster across requested nodes; GPU actors scale across nodes. Stages write Parquet and completion
  manifests; interrupted generation skips already durable candidate IDs.
- `cli.py`: `build-pools`, `prepare`, `submit`, `run`, and `quality-schema` entry points.

The teacher initially uses thinking mode and a configured effort/token budget. A deterministic
polish call produces natural `reasoning` and `response` text; training output has no required
`\\boxed{}` or answer tags. The raw draft is retained for audit. The candidate answer extractor
sees only the question and final response, never the expected answer or trace. Native Reasoning
Gym scoring or typed math verification is used where possible; parse failures are indeterminate,
not mathematical failures. Two complementary critiques inspect global correctness and local
mathematical steps; the arbiter returns correctness plus separate reasoning and response scores.
Four narrower hygiene requests are batched in one wave, not sent sequentially per sample.

The current 1–5 rubric is: 5 training-ready; 4 correct with a minor edit; 3 substantive local
repair; 2 useful progress but major rewrite; 1 unusable. Score 0 means confirmed incorrectness,
incomplete generation, or failure of the core arbiter. A confirmed material hygiene defect caps
the score at 2; hygiene uncertainty, including a failed focused check, caps it at 3 rather than
mislabeling the candidate incorrect. Known-answer conflicts/indeterminate verification are capped
at 3. A substantive issue code also caps a rubric-inconsistent score of 4 at 3. All causes are
explicit in `quality_details_json` (schema version 9). Malformed arbiter JSON gets one bounded
retry; a focused-check parsing failure remains an explicit uncertainty.

Deterministic verification is deliberately typed: numeric expressions, equations, and sets use
normalized symbolic equivalence; Boolean/choice answers use normalized exact match; Reasoning
Gym can use its native scorer. Free text or proofs are not forced through a brittle exact checker.
Candidate and reference answers are extracted independently before comparison, with math markup
added only inside the verifier; the training response is unchanged. Equivalent forms and clearly
conditional answers are not treated as missing; conditional answers require semantic judging.
Mixed equation/expression forms are likewise indeterminate rather than automatically conflicting.
A reference answer is strong but fallible evidence, not an instruction to copy. The judge should
distinguish an impossible or underdetermined question from a candidate that invents assumptions
to answer it. Broad critiques look for counterexamples and the earliest material defect;
arbitration confirms rather than
blindly aggregating the critiques. The hybrid rule/model design follows ideas in
[Qwen3](https://arxiv.org/html/2505.09388), [Kimi k1.5](https://arxiv.org/html/2501.12599v4),
[DeepSeekMath-V2](https://arxiv.org/html/2511.22570v1), and
[GLM-4.5](https://arxiv.org/html/2508.06471).

**No rollout is deleted by selection.** `sft/` contains every candidate with clean training
columns and metadata. `correctness_only_eligible` requires a complete rollout, extractable final
answer, and verified or high-confidence supported correctness.
`productivity_filtered_eligible` adds a passing hygiene status and quality score of at least 4.
`exclusion_reasons_json` and
`quality_details_json.hygiene.findings` explain why a candidate did not enter the stricter view.
The same Parquet rows support alternative downstream thresholds; there is no per-batch top-k rule.

## Initial sources, composition, and profiles

**Current sourcing decision (September 30): use easy tasks only for the next RL-priming
dataset, with no demanding-task slice.** The objective is demonstrating useful reasoning,
appropriate checking, a complete answer, and stopping—not maximizing SFT math capability.
Selected NuminaMath-CoT school-math problems are the proposed new math pool; retain manageable,
verifiable logic/puzzles from Reasoning Gym. Existing worked solutions must still be checked.
Do not enforce shortness at the expense of necessary steps. This supersedes the hard-tilted
sampling recommendation below; the existing configs and active throughput benchmark still use
the historical mix and have not been changed to implement the new source selection.

The reusable pools include [Reasoning Gym](https://github.com/open-thought/reasoning-gym),
[NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning),
[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K),
[DeepScaleR Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset),
and [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M).
The historical 1,000-prompt pilot weights are 25% Reasoning Gym, 35% OpenMath, 25% DeepMath, 10%
DeepScaleR, and 5% OpenThoughts math. Within sources that have useful difficulty bands, sample
15% easy, 35% medium, and 50% hard. This is a mild hard tilt, not a hard-only curriculum;
source-native difficulty labels are not globally comparable. The small debug config uses the same
mix at 24 prompts. Existing source solutions stay in `references/` for later comparisons.

Reasoning Gym should not be sampled uniformly across all tasks. Target roughly 60% broadly useful
math/logic/probability/graph reasoning, 25% algorithmic/state tracking, 10% compact constraint
puzzles, and 5% experimental challenge tasks. Core examples include arithmetic, algebra,
geometry, probability, graph paths, logic, and scheduling. Algorithmic examples include binary
matrix, graph coloring, jugs, and matrix manipulation; compact puzzles include kakurasu, Sudoku,
n-queens, and tower of Hanoi. Challenge tasks (`arc_1d`, `game_of_life_halting`, `letter_jumble`,
`modulo_grid`, `number_sequence`) should stay small until verified. Quarantine tasks whose
earlier small runs failed verification across efforts (notably `arc_agi`, `rearc`, `bf`, `codeio`,
`circuit_logic`, `figlet_font`, `rubiks_cube`, `sokoban`, `sudoku`, `tsumego`) rather than scaling
gimmicky or unreliable formats.

For OpenMath, deduplicate to unique problems, prefer answer-extractable math for the first run,
stratify by source pass rate, and archive DeepSeek-R1/QwQ solutions as references. For DeepMath,
use its answer, topic, difficulty, and three R1 solutions; the current pipeline generates a new
trace and can later compare it with the best archived source solution. For DeepScaleR, use source
answers/official solutions where available; it lacks a native difficulty field. OpenThoughts has
many traces per underlying question, so sample unique math questions; code/science are outside
this initial filter experiment until credible verification is available.

The near-term quality profile is Qwen3.8-Flash-Next-FP8, TP4 on one 4×GH200 node, xhigh effort,
32K context, and 16K output limit. A Qwen3.8-27B BF16 TP1 profile is a throughput baseline, not
automatically an approved teacher. Effort comparisons use separate runs. Long-context or hard
tail GLM-5.2 via Swiss Model Launcher remains future work, not a dependency for the current pilot.

## Run and inspect

There is no reservation now. Use ordinary Slurm allocations (`normal`, or `preemptable` when
appropriate); do not assume high-priority/non-preemptible access. The September 30 throughput
investigation uses the current debug allocation and writes measurements under
`runs/throughput-20260930-*`. The continuous execution path is initially opt-in while being
validated on real rollouts. Original waves remain the comparison path.
Slurm submissions request requeue by default (`slurm.requeue`); Ray rendezvous filenames include
the Slurm restart count so a requeued worker cannot read the previous head's address.

The initial systems audit found eight-candidate batches and whole-stage barriers starving the
four-GPU teacher. Profiling reserved ~35 GiB/GPU of peak activations, leaving ~4.5 GiB/GPU for
KV cache in the first 64-sequence benchmark. Disabling unused image/video inputs raised that to
only ~5 GiB, and reducing prefill from 8,192 to 4,096 tokens raised it to ~5.3 GiB. Neither change
explains or eliminates the large reservation; do not claim a major memory gain from them.
A real no-reference prompt also exposed Pandas converting null
references to NaN; generation now normalizes missing scalar values before calling verifiers.

The smaller-teacher pilot is `configs/reasoning-throughput-128.yaml`: Qwen3.6-35B-A3B-FP8,
four TP1 replicas on one node, with all quality checks. Its template supports thinking but does
**not** support effort labels; use `reasoning_effort: null` rather than claiming xhigh control.
Its quality must be reviewed before replacing the current teacher.

The first valid continuous large-teacher measurement completed 64 real mixed-source candidates
in 705.6 seconds after 306.8 seconds of engine startup: 326.5 candidates/hour/node, with 40
pipeline-eligible rows (204.1 eligible/hour). Including startup gives 227.6 candidates/hour.
Results: `runs/throughput-20260930-continuous-v2/batch-64/summary.json`. All inference requests
returned; two polished outputs lacked a complete response. Output-token shares were ~60% draft,
28% broad critiques, 7.5% polish, and 1.8% focused hygiene. The 23 capped drafts and 93/128 capped
critique calls make budgets and critic behavior worth investigating, not automatically shortening.
These are warm finite-batch measurements, not a matched speedup or a calibrated useful-yield
estimate. The earlier `continuous` directory is explicitly invalid (a prompt-tokenization bug
caused every request to fail); never report its apparent samples/hour as throughput.

Continuous actors now log engine statistics every ten seconds. Small pilots and resumed tails
bound the Ray batch size to leave work for every active replica; otherwise Ray can combine all
small blocks into one large batch and leave the other GPUs idle. The 128-prompt pilot uses
32-row actor batches; production needs larger batches to amortize long-tail draining.
Use a fixed `ActorPoolStrategy` with `max_tasks_in_flight_per_actor=1`. In the first real Ray
pilot, default task prefetch assigned four batches to two actors and shut down the other two
just after initialization. That interrupted run is not a four-GPU throughput measurement.
The retry uses one actor batch in flight; this does not limit the concurrent model requests
inside each batch. Engine-ready and phase metrics include timestamps for startup/warm-rate analysis.
The container's home cache is ephemeral: the first two Qwen3.6 starts each recompiled the same
825 DeepGEMM warmup kernels. `container/cscs.toml` now places vLLM/DeepGEMM, Triton, FlashInfer,
and CUDA caches under scratch. This targets restart overhead, not steady-state throughput;
cache reuse still needs a measured subsequent launch.

The corrected four-GPU Ray pilot completed all 128 prompts. Generation plus all fused checks
took 414.6 seconds after all engines were ready: 1,111 candidates/hour/node, 60 eligible
(521 eligible/hour). Including actor startup and input setup gives 587 candidates/hour; the
original CPU export added 3.3 seconds. Scores 0–5 were 20, 1, 19, 28, 0, 60; hygiene passed/failed/
uncertain was 73/36/19. There were 67 capped drafts, 10 capped polishes and 11 incomplete final
generations. Artifacts: `runs/reasoning-throughput-qwen36-128-v1/`, with an investigation report
at `runs/throughput-20260930-report.md`. These rates include rejects and all review calls, but
do not count the earlier interrupted actor-pool attempt or export repair as steady-state work.

On the 35 shared prompts, large/small teacher configurations had 25/17 eligible rows and both
had 23 verifier passes. Mean draft lengths were 7,922/11,367 tokens. Sampling and judges differ;
this is not a controlled teacher-quality comparison. Manual review found a useful rejection of
an incorrect syllogism classification, but also an extractor that changed the literal answer
`7` into `,  `. Do not approve the smaller teacher/judge for production just from its speed.

The final CPU transforms initially reintroduced null-only Arrow columns. Final candidate export
now restores the fixed nullable schema, and the repaired candidate/SFT directories read directly
as Arrow datasets, preserving every candidate field. The old exports are retained as
`*.incomplete.*`; generations were not rerun. Ray log deduplication is disabled for future jobs
because it otherwise suppresses distinct numeric phase metrics.

`configs/reasoning-throughput-1024.yaml` is the sustained-throughput follow-up on `preemptable`:
same model/sampling/checks, four TP1 replicas, 32 active trajectories and a 256-prompt queue per
replica. It measures replenishment instead of draining after only 32 prompts/GPU. The large-teacher
1,000-prompt profile now uses continuous scheduling with a fresh `reasoning-productivity-1000-v2`
run ID. No million-sample production run is authorized by these pilot results.
The sustained follow-up was submitted as Slurm job **3548372** on September 30, 2026;
it was pending for priority at 02:46 CEST. Inspect that job and
`runs/reasoning-throughput-qwen36-1024-v1/` before submitting a duplicate.

Update: that job completed all 1,024 prompts successfully at about 03:38 CEST. Warm generation
and all checks took 2,911.2 seconds: **1,266 candidates/hour/node**, with 522 pipeline-eligible
rows (**645 eligible/hour**). Including input/actor startup gives 1,214 candidates/hour.
All four engines started in about 100 seconds each (versus roughly 370 in the prior pilot;
this is consistent with cache reuse, not an isolated cache ablation). Scores 0–5: 198, 7,
125, 172, 0, 522. Hygiene passed/failed/indeterminate: 649/243/132. There were 523 capped
drafts, 44 capped polishes, and 68 incomplete final generations. All rows are readable in
`candidates/` and `sft/`. These are pipeline-policy yields, not human-calibrated filter accuracy.

Swiss Model Launcher manages vLLM/SGLang deployments; it is not a separate inference engine.
The [serving leaderboard](https://serving.swissai.svc.cscs.ch/leaderboard) ranks token usage, and
the performance page reports per-query speed, not sustained samples/GPU-hour. The published
[DeepSeek V4.1 Flash recipe](https://github.com/swiss-ai/model-launch/blob/main/mfa_examples/clariden/deepseek-ai/DeepSeek-V4.1-Flash/vllm/DeepSeek-V4.1-Flash-vllm.sh)
uses TP8/two nodes, with its throughput benchmark explicitly not run. Its locally available
weights total ~510 GB. The BF16 Qwen3.5-397B weights total ~807 GB and the published recipe uses
four nodes. Do not multiply a multi-GPU replica's sample rate by every GPU again. SML is a useful
future deployment option, but changing serving frameworks does not fix an underfed client.

Use a fresh `run_id` whenever the model, source, sampling, or quality policy changes; completed
stages have manifests tied to the resolved config. `configs/reasoning-productivity-debug-24.yaml`
is the small real-rollout validation; `configs/reasoning-productivity-1000.yaml` is the first pilot.
Both use the v0.6 image and a single Slurm job (no job arrays).

```bash
uv run synthetic-sft build-pools configs/source-pools.yaml
./container/build.sh "$SCRATCH/images/synthetic-sft-v0.6.sqsh"
uv run synthetic-sft submit configs/reasoning-productivity-debug-24.yaml
```

`output_dir/run_id` contains `prepared/seeds/`, `intermediate/generated/`, `candidates/`,
`sft/`, logs, and `manifests/`. Query all SFT rows by source, correctness, hygiene, quality, and
the two eligibility flags. Inspect both accepted and rejected examples manually; a correct answer
and a model's quoted evidence are not by themselves proof of a productive trace. Report the full
0–5 score distribution, per-category defect/uncertainty rates, answer correctness, completion,
reasoning-token distributions, and useful yield by source/difficulty. Do not infer filter quality
from counts alone.

The main decision experiment is **matched correctness-only versus productivity-filtered SFT**:
match prompts/source/difficulty where possible, base checkpoint, training-token budget, chat
template, effort, and trainer. Run the same short RL climb (including the same length objective)
from both checkpoints. Compare at step zero and during RL: completed correct answers, parse
failures, limit hits, repetition, tokens per correct completed answer, pass@1/pass@k, prompts with
non-zero reward, all-zero rollout groups, reward distribution, early learning speed, and final
quality. A better step zero but same eventual RL may still save compute; cleaner but lower solution
coverage means the filter is too aggressive. RL can learn to stop, so final response length alone
is not the success metric.

The paired 24-prompt debug runs (v1 and v2, seed 43) used identical prepared prompts. The v2
pilot retained all 24 rows: scores were 13 at 5, 7 at 3, 1 at 2, and 3 at 0; 17 passed hygiene,
6 were uncertain, and 1 failed. Focused-judge parse errors fell from 9 to 1 after raising its
JSON budget and saving raw outputs. Manual review found a previously overrated yes/no proof with
an unsupported central step; the v2 broad judge named the gap but rated it 4, while a focused
finding excluded the row. The current rubric-6 postprocessing also caps substantive issue codes
at 3; recalculating the 24 rows with that rule leaves 13 in the strict view. These are pilot
diagnostics, **not** calibration of false-positive/false-negative rates or approval for a large
generation run. In particular, the focused finding's explanation for the weak proof was partly
unsound, so the judge still needs manual calibration.

`docs/index.html` is the meeting review site. It shows the first 50 durable rollouts from the
interrupted 200-prompt run (35 pass and 15 reject by pipeline policy), plus two clearly labeled
reasoning-quality rejects from the earlier 24-prompt pilot. `docs/curation.json` holds the human
notes; `docs/data.js` is generated by `scripts/export_meeting_site.py` from the saved Parquet.
The site is illustrative, not a filter-accuracy estimate. The 50 new rollouts do not contain a
convincing repeated-checking rejection; do not present one as such.

## Active SML / DeepSeek pilot (September 30)

**Current attempt: job 3548993**, `configs/reasoning-sml-deepseek41-prefetch-64.yaml`,
run `reasoning-sml-deepseek41-64-v3`, on two preemptible nodes with 800 GiB RAM/node.
It uses explicit prefetch, effort 100/75, and the automatic single-job 64-prompt client below.
Check this job and its `logs/client.{out,err}` before launching anything else. Neither earlier
attempt produced inference results. The history below explains the loading changes.

The private DeepSeek V4.1 Flash deployment is Slurm **3548849**, two debug nodes
`nid007170,nid007183`, started about 03:39 CEST. Debug allows 90 node-minutes, so the
two-node allocation is limited to 45 minutes. Do not register this pilot on the public gateway.
The existing Qwen 1,024-prompt job 3548372 completed successfully at about 03:38; its outputs
still need the sustained-throughput/quality summary.

`scripts/render_sml.py` uses SML's rendering API, with OpenTela/telemetry disabled, and its
official vLLM 0.30 ARM64 environment. SML checkout:
`/iopsstor/scratch/cscs/tchu/.cache/synthetic-sft/model-launch`, commit
`1b3f59974cf593946faffe83fb4710f177ae82f1` (separate `uv sync --no-dev` environment).
Render with that environment, supplying the config, a fresh deployment directory, and
`--environment <sml-checkout>/src/swiss_ai_model_launch/assets/envs/vllm_0.30.0.toml`.
Submit the resulting `master.sh` with `sbatch`; serving logs are `logs/<job-id>/replica_0.*`.
Current generated recipe: `runs/reasoning-sml-deepseek41-64-v1/deployment-v2/` (its original
90-minute request was changed to 45 through Slurm; the checked-in config is corrected).

`model.execution: endpoint` is currently supported only by `scripts/benchmark_pipeline.py`,
not the production Ray `run`/`submit` commands (those reject it explicitly). Its HTTP transport
reuses the exact continuous orchestration and all generation/quality stages. The serving
engine owns chat rendering; local `tokenizers` is used only to count/truncate text. Set
`SFT_API_BASE=http://<head-ip>:8080/v1`; the configured model name is checked against `/models`.
`uv sync --extra endpoint` supplies lightweight client dependencies, with no local vLLM needed.
The v0.6 container can also run the client on a compute node. Use `srun --jobid=<id> --overlap`
on the head node, and set `SLURM_JOB_ID` explicitly when invoking its EDF from a login shell.

First run `scripts/probe_endpoint.py CONFIG OUTPUT`: it waits for readiness and saves real
thinking/nonthinking/structured-answer calls, failing on a broken serving contract. Then use
`scripts/benchmark_pipeline.py CONFIG SAVED_SEEDS OUTPUT --execution endpoint --samples 64
--batch-sizes 64 --inflight 32 --max-num-batched-tokens 4096`. Saved real prompts are reused;
this benchmark does not rebuild pools. Inspection of the actual server image's
`deepseek_v41_encoding.py` found low/high/xhigh/max map to 25/50/75/100, unlike the hosted
API's aliases. The config therefore uses numeric draft effort **100** and critique effort **75**.
`medium` is unsupported. Numeric efforts 1–100 and `max` are accepted by our config; only use
these with teachers that support them. Polishing/JSON checks
disable thinking. Context/output are bounded at 32K/16K for this pilot, not the model card's
full-capacity evaluation settings. Transport errors fail the run, never become math failures;
completed groups remain in Parquet. Benchmark output directories are fresh-only, not resumable.
Rates must count all eight GPUs/two nodes and distinguish server startup from warm generation.
Keep the serving job available across experiments within its allocation; do not restart the
model after each bounded measurement. Cancel when the experiment session is finished. No DeepSeek throughput or
quality claim is established yet.

The first SML launch stalled in weight materialization from Lustre. The running vLLM image
defaults to mmap loading. Its actual `weight_utils.py` recognizes **both NFS and Lustre** for
automatic prefetch, gated on checkpoint size fitting within 90% of available RAM; the
`config/load.py` docstring mentioning only NFS is incomplete. The driver's logs do not establish
whether that heuristic activated. The retry uses `--safetensors-load-strategy eager`, reading
each shard into CPU RAM up front, following [vLLM's network-filesystem guidance](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/load.py).
This has not yet been measured here and must not be reported as an achieved speedup. The two
Engram shards are about 102 GB each, so monitor host-memory peaks during eager deserialization.

Update at ~04:12 CEST: the debug attempt was cancelled deliberately after over 30 minutes
without a ready endpoint. No inference result exists from it. Its shard iterator reached 100%
after 17.4 minutes, but that was **not** model-ready: stack snapshots found the workers still
inside Engram/expert `copy_` loaders, blocked on Lustre I/O, with GPUs idle. Job memory events
showed no limit hits or OOMs. Two bounded read-only cache-warming steps read the Engram shards
(~1.4–1.9 GB/s per node) and then the other shards; they did not make the endpoint ready before
cancellation. Do not present this attempt as an inference throughput measurement.

The regular-queue retry is **3548976**, using
`configs/reasoning-sml-deepseek41-eager-64.yaml` and run `reasoning-sml-deepseek41-64-v2`.
It requests 800 GiB host RAM per node and 75 minutes, with eager loading; Slurm rejected
`--mem=0`, while the explicit 800G request was accepted. This larger allowance avoids the
default 450 GiB limit while multiple workers stage large shards in CPU RAM. The actual recipe
is `runs/reasoning-sml-deepseek41-64-v2/deployment-mem800/`. It was submitted with an afterany
dependency on the cancelled debug job; check its state before submitting anything else.
At 04:13, `normal` was blocked by `QOSGrpNodeLimit`; the pending job was moved in place to
`preemptable`, and the checked-in config now requests that partition. The generated historical
SBATCH header still says normal; the live Slurm allocation is authoritative. No reservation
or elevated QOS was requested.
The retry started at **04:14:16 CEST** on `nid006144,nid007009`; Slurm confirms 800 GiB RAM
per node and all eight GPUs allocated. No DeepSeek inference result was available at 04:17.
It then failed during initialization with `KeyError: 'F8_E8M0'` in
`safetensors.torch.load`/`_view2torch`: the **eager** loader in this image cannot deserialize
the model's scale dtype. The single-job wrapper detected the serving failure and released
the allocation. Do not use the eager profile as a working launch recipe. Also, the DeepSeek
loader calls `sorted(...)` on the entire weight iterator, so eager deserialization would
retain a full checkpoint per worker, not merely a shard; patching only the dtype mapping
would create a serious host-memory problem. Explicit prefetch retains the compatible mmap
reader and shared OS cache. Job 3548993 is that corrected, bounded retry, submitted at ~04:18.

The renderer now accepts `--seeds SAVED_SEEDS` and emits a `pilot.sh` allocation script as well
as `master.sh`. Submit **pilot.sh** for a single-job trial: `slurm/sml-pilot.sh` starts SML and
a CPU client in the same allocation, runs the real API probes followed by the 64-prompt
benchmark, and keeps serving after the initial client completes or fails so subsequent clients
can reuse the endpoint. A serving failure or the allocation time limit still ends the job.
This lifecycle change applies to newly launched wrappers; job 3548993 started with the old
client-completion shutdown and must not be assumed to have adopted a live script edit.
Client logs are `runs/<id>/logs/client.{out,err}`; serving logs
remain `logs/<job>/replica_0.*`; raw probes and completed candidate groups stay under the run.
The initial endpoint probes were restarted before inference to use explicit effort 100.

## Backlog and boundaries

- Calibrate the narrow Qwen judge on real traces. If false negatives remain high, tune prompts or
  switch to a stronger, low-hallucination judge via `quality.judge.model_source`.
- **Do not enable critique-guided rewriting by default.** If strict-view rejection exceeds 80% on
  a representative batch, evaluate a separate revision stage for correct, repairable traces. Feed
  the judge's structured decision and explanation to a refactor, then independently re-run the
  same correctness and hygiene checks on the revision. Keep original and revised traces paired;
  reject any repair that hides errors, changes the answer, removes useful exploration, or costs
  more than it helps. The paired 24-prompt pilot rejected 11/24 (46%), below this trigger.
- Keep source-provided solutions as archived references/baselines; a paired source-solution
  comparison or source-trace SFT arm needs explicit implementation, not an undocumented assumption.
- Swiss Model Launcher is the longer-term serving path; GLM-5.2 is a possible stronger teacher
  for hard math, but requires model-support and cost/quality validation. Avoid making it a blocker
  for the vLLM pilot.
- Candidate future pools include `PrimeIntellect/SYNTHETIC-2-Base-v2`, `open-r1/OpenR1-Math-220k`,
  and `nvidia/OpenCodeReasoning`; deduplicate and verify their relevant domains first.

## Working conventions

Preserve user data and unrelated worktree changes. Create a commit at each reasonable checkpoint
(the project owner explicitly requested this). Prefer actual small Slurm runs and manual trace
review over tests that merely assert counts or repeat implementation logic. Report failures and
uncertainty clearly; do not claim that a filter or teacher is validated merely because the job
completed. Update this file when design decisions, configs, or the verified run procedure change.
