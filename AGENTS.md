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
explicit in `quality_details_json` (schema version 7).

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

The reusable pools include [Reasoning Gym](https://github.com/open-thought/reasoning-gym),
[NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning),
[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K),
[DeepScaleR Preview](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset),
and [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M).
The current 1,000-prompt pilot weights are 25% Reasoning Gym, 35% OpenMath, 25% DeepMath, 10%
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
