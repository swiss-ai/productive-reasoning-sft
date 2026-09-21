# Reasoning data launch plan

This is the working plan for choosing the data mix and generation profiles before a large run.
The first run is deliberately small: it should tell us what deserves scale, not produce the final
dataset.

## Proposed pilot

Start with **1,000 unique prompts** and one new rollout per prompt. Where a source already provides
a solution, retain that solution as a separate candidate rather than replacing or discarding it.
All candidates are judged and written, including failures.

| Source | Prompts | Share | Initial treatment |
|---|---:|---:|---|
| Reasoning Gym | 250 | 25% | Generate; exact source verifier |
| NVIDIA OpenMathReasoning | 300 | 30% | Generate; retain supplied traces as references/baselines |
| DeepMath-103K | 200 | 20% | Generate; retain all three supplied R1 solutions |
| DeepScaleR Preview | 100 | 10% | Usually retain the official solution; regenerate a paired subset |
| OpenThoughts3-1.2M | 150 | 15% | Retain supplied trace; regenerate a paired subset |

This is intentionally math-heavy for the first pilot because those sources have the strongest
correctness signals. It is not the intended composition of a general-purpose final SFT mixture.
Before the full launch, add verified code and broader science/general reasoning only after their
verification paths are credible.

Use a mild difficulty tilt, not a hard-only distribution:

- 15% easy/foundational
- 35% medium/reasonable
- 50% hard

Difficulty bands are source-native and remain in provenance. They are sampling strata, never a
global claim that difficulty values from different sources are comparable.

## Reasoning Gym composition

Do not sample all 105 tasks uniformly. The initial 600-prompt mix should be:

- 60% broadly useful math, logic, probability, and graph reasoning
- 25% algorithmic and state-tracking reasoning
- 10% compact constraint puzzles
- 5% experimental challenge tasks

Within each group, use the same 15/35/50 easy/medium/hard split. Reasoning Gym has curricula for
102 of 105 tasks, so difficulty should be selected from normalized curriculum levels rather than
maintaining dozens of unrelated numeric knobs by hand.

### Core tasks (60%)

`complex_arithmetic`, `intermediate_integration`, `polynomial_equations`,
`polynomial_multiplication`, `advanced_geometry`, `simple_geometry`, `calendar_arithmetic`,
`decimal_arithmetic`, `dice`, `fraction_simplification`, `gsm_symbolic`, `number_format`,
`power_function`, `prime_factorization`, `time_intervals`, `course_schedule`,
`family_relationships`, `largest_island`, `path_star`, `shortest_path`, `acre`,
`list_functions`, `knights_knaves`, `propositional_logic`, `self_reference`, `syllogism`,
`zebra_puzzles`, and `coin_flip`.

### Algorithmic/state tasks (25%)

`binary_matrix`, `count_bits`, `cryptarithm`, `game_of_life`, `graph_color`, `jugs`,
`manipulate_matrix`, `rotate_matrix`, `spiral_matrix`, `string_manipulation`, and
`string_synthesis`.

### Compact puzzles (10%)

`boxnet`, `kakurasu`, `maze`, `mini_sudoku`, `n_queens`, `puzzle24`, `survo`, and
`tower_of_hanoi`.

### Challenge lane (5%)

`arc_1d`, `game_of_life_halting`, `letter_jumble`, `modulo_grid`, and `number_sequence`.
This lane is intentionally small because the one-example-per-task runs were unstable.

Exclude trivial formatting/reversal/counting exercises from the first launch. Quarantine tasks
that failed verification in all three earlier effort runs—especially `arc_agi`, `rearc`, `bf`,
`codeio`, `circuit_logic`, `figlet_font`, `rubiks_cube`, `sokoban`, `sudoku`, and `tsumego`—until a
small dedicated profile demonstrates that their output format and token budget are reliable. This
does not mean every quarantined task is intrinsically bad; it means the current generation path is
not trustworthy enough to scale it.

## Source-specific policy

### OpenMathReasoning

- Deduplicate to unique problems before sampling; do not sample its millions of repeated traces as
  if they were unique prompts.
- Prefer `has_answer_extracted` in the first pilot. Hold converted proofs and the recovered
  `additional_problems` split back: NVIDIA reports that adding the recovered proof questions
  regressed SFT performance.
- Stratify by `pass_rate_72b_tir`: 15% `[0.75, 1]`, 35% `(0.25, 0.75)`, and 50% `[0, 0.25]`.
  Keep unavailable values in the reusable pool but outside the first run. Record the exact value.
- Preserve the DeepSeek-R1/QwQ solution and model name. Use the expected answer for verification;
  use the old solution as judge reference and a retained baseline, not as hidden input to the new
  generator.

### DeepMath-103K

- This is the main hard-math source: it has answers, topics, numeric difficulty, and three R1
  solutions per problem.
- Sample 15% difficulty `<=4`, 35% `(4, 6.5]`, and 50% `>6.5`, while balancing the topic hierarchy.
- Retain all three source solutions. Generate a new solution, verify its final answer, and compare
  reasoning quality against the best source solution.

### DeepScaleR Preview

- It contains about 40K problem/answer pairs and often an official solution, but no native
  difficulty field.
- Prefer non-empty official solutions and use answer verification. Derive pilot strata from problem
  source, statement length, and a small baseline pass-rate probe; do not mistake length alone for
  difficulty.
- Preserve official solutions by default. Regenerate only a paired subset until the new teacher
  demonstrably improves rigor or language.

### OpenThoughts3-1.2M

- It is 75K unique questions annotated 16 times with QwQ-32B, yielding 850K math, 250K code, and
  100K science traces. Group by the underlying question before sampling.
- Pilot 120 unique prompts from each domain. Retain the supplied traces, and regenerate one trace
  for a paired quality comparison.
- Treat its difficulty label as a sampling signal. Since the published rows expose conversations
  rather than a general exact-answer field, correctness is judge/reference based unless a source
  record supplies a usable verifier.
- Do not scale its code portion until execution-based checking exists. Judge-only code is too easy
  to score confidently while being subtly wrong.

## Other candidates worth retaining

- [`PrimeIntellect/SYNTHETIC-2-Base-v2`](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-Base-v2):
  promising mixed math/code/general-reasoning prompts with `verification_info`; evaluate after the
  verifier payloads can be executed safely.
- [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k):
  strong verified math baseline, but likely overlaps substantially with the selected math sources.
  Keep as a comparison source rather than adding more duplicate math immediately.
- [`nvidia/OpenCodeReasoning`](https://huggingface.co/datasets/nvidia/OpenCodeReasoning): useful
  code source, but OpenThoughts already includes part of it. Prefer the original only after
  deduplication and executable verification are ready.

## Model and environment profiles

Run effort levels as separate configurations; never vary effort sample by sample inside one run.

### Throughput baseline

- Model: `Qwen/Qwen3.8-27B`, BF16
- Layout: four tensor-parallel-1 replicas per 4xGH200 node
- Effort: `medium` for the easy band, `xhigh` for medium/hard bands
- Context/output: 32K context, 8K output initially
- Use: establish cost and useful-throughput baselines only. It must pass manual trace review before
  it is approved as a production teacher.

### Near-term quality challenger

- Model: `Qwen/Qwen3.8-Flash-Next`, FP8, one tensor-parallel-4 replica per node
- Effort: `xhigh`
- Context/output: 32K context, 16K output for hard math
- Use: paired comparison on the same prompts. Its official card reports stronger reasoning results
  than the 27B model and supports the same thinking controls, but fit and throughput must be
  confirmed on GH200 before adopting it.

### Future hard-question profile

- Model: GLM-5.2 through Swiss Model Launcher, with the same model also judging its outputs
- Use only for the hard tail if its quality gain justifies multi-node inference
- Keep postponed: SML's GLM-5.2 model-support issue is currently open and estimates 3 nodes per
  replica as the functional minimum, 4 for full-context/high-concurrency FP8 serving.

The current vLLM container remains the pilot environment. SML should become a selectable serving
backend later, not a prerequisite for deciding the data mix.

## Decision after the pilot

Inspect a stratified sample of reasoning and final responses from every source, task group,
difficulty band, and model profile. Then compare:

- exact-verification pass rate where available
- score counts from 0 through 5, separately for reasoning and response
- completion/degeneration rate and token-length distribution
- useful throughput: verified and quality `>=4` candidates per GPU-hour
- paired preference: retained source solution versus newly generated solution
- duplicate rate across all math sources

Scale only strata whose samples are correct and genuinely training-ready. Keep every generated row
in Parquet regardless of score; the launch mix controls what we generate, while downstream users
remain free to choose their own quality threshold.

## Dataset references

- [Reasoning Gym dataset catalog](https://github.com/EnvCommons/ReasoningGym/blob/main/DATASETS.md)
- [DeepScaleR Preview Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [NVIDIA OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Swiss Model Launcher GLM-5.2 support request](https://github.com/swiss-ai/model-launch/issues/162)
