# `packages/evals/`: Dataset A evaluation pipeline

End-to-end pipeline to grade an LLM (or the Brick router) on the 5,504-query Dataset A and produce the per-dimension accuracy + cost numbers from the paper.

## What's in here

```
packages/evals/
├── scripts/                    # Numbered pipeline (run in order)
│   ├── 00_setup_check.py            # validate env (HF token, API keys, tooling)
│   ├── 10_download.py               # bootstrap pulled benchmark sources (MMLU-Pro, BFCL, …)
│   ├── 10b_download_bfcl.py
│   ├── 10c_clone_bfcl_multi_turn.py
│   ├── 10c_download_livecodebench.py
│   ├── 11_clone_taubench.py
│   ├── 12_fetch_eqbench.py
│   ├── 13_planning_custom_generate.py
│   ├── 14_planning_custom_validate.py
│   ├── 20_normalize.py              # canonical schema (uses brick_evals.normalize.NORMALIZERS)
│   ├── 30_fewshot_extract.py        # few-shot pool extraction
│   ├── 30b_fewshot_regolo_generate.py
│   ├── 30c_fewshot_quality_regen.py
│   ├── 30d_fewshot_ifeval_realpool.py
│   ├── 31_creative_custom_generate.py
│   ├── 32_creative_custom_validate.py
│   ├── 40_assemble_eval_params.py   # merge per-query inference parameters
│   ├── 50_tokenize_triplo.py        # tokenize for 3 backend tokenizers
│   ├── 60_stratify_report.py        # stratification audit per dimension
│   ├── 70_push_hub.py               # push to massaindustries/dataset-A-routing-eval (legacy)
│   ├── 71_push_hub_multiconfig.py
│   ├── 72_push_dataset_a_routing.py # push to regolo/brick-dataset-A-routing-eval (canonical)
│   ├── 72_validate_routing_db.py
│   ├── 80_verify_dataset.py
│   ├── 99_unmask_gated.py
│   ├── 100_run_inference.py         # run inference for one model OR Brick router
│   ├── 110_grade_inference.py       # per-grader scoring (LCB, BFCL, IFEval, math-equiv, …)
│   ├── 115_aggregate_panel.py       # 3-judge majority vote
│   ├── 120_run_bfcl_multi_turn.py
│   ├── 130_aggregate_results.py     # final accuracy / cost / latency table
│   ├── 131_panel_report.py
│   └── 140_extract_model_skill_profiles.py
├── src/brick_evals/            # Python package (importable: `from brick_evals import ...`)
│   ├── clients/                # regolo_client.py, openrouter_client.py
│   ├── judge.py + openrouter_judge_client.py
│   ├── graders/                # bfcl_grader, ifeval_grader, lcb_grader, math_equiv_grader, rubric_judge_grader, simpleqa_grader
│   ├── normalize/              # NORMALIZERS registry per benchmark source
│   ├── tokenizers.py, schema.py, dedup.py, contamination.py, fewshot.py, io_utils.py
│   └── __init__.py
├── configs/
│   ├── models.yaml             # pool definition (qwen3.5-9b, deepseek-v4-flash, kimi2.6, ...)
│   ├── protocols.yaml          # per-dimension grader assignment
│   ├── prompts.yaml            # system / user templates
│   ├── sources.yaml            # benchmark provenance + version pins
│   └── judges.yaml             # 3-judge panel (gpt-5.4-mini + Mistral + GLM)
├── tests/                      # pytest (smoke + grader unit + schema + κ + PII)
├── baselines/                  # RouteLLM / FrugalGPT / cascade-routing → see baselines/README.md
└── pyproject.toml              # name = brick-evals
```

## Most common entry points

After `uv sync` from repo root, the typical evaluation flow is just 3 scripts:

```bash
# 1. Inference (Brick router or any single backend)
uv run python packages/evals/scripts/100_run_inference.py \
  --dataset ./data/dataset_a \
  --endpoint http://localhost:18000/v1/chat/completions \
  --out ./runs/brick_run.jsonl

# 2. Grade (3-judge panel)
uv run python packages/evals/scripts/110_grade_inference.py \
  --inputs ./runs/brick_run.jsonl \
  --judges packages/evals/configs/judges.yaml \
  --out ./runs/brick_graded.jsonl

# 3. Aggregate (final table)
uv run python packages/evals/scripts/130_aggregate_results.py \
  --in ./runs/brick_graded.jsonl | tee results.txt
```

Full quickstart with expected output: [`docs/quickstart/eval.md`](../../docs/quickstart/eval.md).

## The 3-judge panel

Used for `rubric_judge` (planning + creative_synthesis) and `llm_judge_factual` (world_knowledge subset). Configured in `configs/judges.yaml`:

| Judge | Model | Role |
|---|---|---|
| Judge 1 | `openai/gpt-5.4-mini` (via OpenRouter) | tie-breaker reference |
| Judge 2 | `mistralai/mistral-small-2603` | semantic precision |
| Judge 3 | `zai/glm-5-turbo` | factual recall |

Aggregation: 2-of-3 majority vote on a 4-point rubric (`fail`/`partial`/`pass`/`exceptional`), with structured output parser per judge to handle variant phrasings. See `src/brick_evals/judge.py` and `src/brick_evals/graders/rubric_judge_grader.py`.

Inter-rater κ on 5,504 graded queries: **0.761**. Cost: ~$30–50 for the full panel run via OpenRouter.

## Per-dimension graders

| Dimension | Grader | Method |
|---|---|---|
| `coding` | `lcb_grader` | Unit-test execution (LiveCodeBench harness) |
| `math_reasoning` | `math_equiv_grader` | Symbolic equivalence (sympy + final-answer extractor) |
| `instruction_following` | `ifeval_grader` | Constraint-satisfaction checks (IFEval) |
| `planning_agentic` | `rubric_judge_grader` | 3-judge panel on rubric (planning quality) |
| `creative_synthesis` | `rubric_judge_grader` | 3-judge panel on rubric (creative quality) |
| `world_knowledge` | `simpleqa_grader` | LLM-judge factuality check |
| `world_knowledge` (sub) | `bfcl_grader` | Function-calling correctness (BFCL) |

## Running tests

```bash
uv run pytest packages/evals/tests -q
# or just the smoke suite:
uv run pytest packages/evals/tests/test_smoke_load.py -q
```

## Skip-the-pipeline shortcut

If you just want the final accuracy table from the **published** graded run (no inference + grading needed):

```bash
python packages/datasets/scripts/download_dataset_a.py --out ./data/dataset_a
# The HF snapshot includes the `results` config with already-graded verdicts:
uv run python packages/evals/scripts/130_aggregate_results.py \
  --in ./data/dataset_a/results/train.jsonl.gz
```
