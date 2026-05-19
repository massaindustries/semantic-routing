# Quickstart C. Reproduce the paper evaluation

Goal: download Dataset A + the trained classifiers from HuggingFace, run inference + 3-judge grading, aggregate, and confirm Brick reaches ~76.98% accuracy.

This is the closed-loop replication path used to produce the numbers in the paper. It does **not** rebuild Dataset A from raw sources: that flow is out of scope for this quickstart (the published HF dataset is the canonical artifact). For training the classifiers from scratch, see [`packages/training/README.md`](../../packages/training/README.md).

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/) (recommended): `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- A HuggingFace token if the datasets/models are gated for you (`export HF_TOKEN=...`).
- An OpenRouter or Regolo API key for the judge panel (`OPENROUTER_API_KEY` or `REGOLO_API_KEY`).
- ~10 GB free disk for the dataset + model snapshots.
- ~6 hours of judge time if you grade all 5,504 queries with the 3-judge panel (cost: ~$30-50 depending on judge models).

## Install workspaces

```bash
git clone https://github.com/regolo-ai/brick-SR1 && cd brick-SR1
uv sync                                  # creates .venv and installs packages/*
```

## Download artifacts

```bash
# Dataset A (5,504 queries with per-model verdicts)
python packages/datasets/scripts/download_dataset_a.py --out ./data/dataset_a

# Trained classifiers (capability + complexity)
python packages/datasets/scripts/download_models.py --out ./models
```

After download:

```
./data/dataset_a/                          # HF snapshot of regolo/brick-dataset-A-routing-eval
./models/capability-classifier/             # ModernBERT-base (6-label sigmoid)
./models/complexity-classifier/             # Qwen3.5-0.8B + LoRA (3-class)
```

## Run inference

The router (or any of the 3 backend models directly) needs to produce a response for every query. Use the existing run script:

```bash
# Inference via Brick router (recommended for the paper's exact metric):
docker run -d --name brick -p 18000:18000 -e REGOLO_API_KEY=$REGOLO_API_KEY \
  ghcr.io/regolo-ai/brick:latest

uv run python packages/evals/scripts/100_run_inference.py \
  --dataset ./data/dataset_a \
  --endpoint http://localhost:18000/v1/chat/completions \
  --config packages/evals/configs/protocols.yaml \
  --out ./runs/brick_run.jsonl
```

For per-model baseline runs (always-Qwen, always-DS4, always-Kimi), pass `--model <model_id>` and set `--endpoint` to the relevant provider.

## Grade with the 3-judge panel

The panel uses `gpt-5.4-mini` + `mistral-small-2603` + `glm-5-turbo` via OpenRouter. Judge templates live in `packages/evals/configs/judges.yaml`; per-dimension graders (LCB, BFCL, IFEval, math-equiv, rubric-judge, SimpleQA factual) are in `packages/evals/src/brick_evals/graders/`.

```bash
uv run python packages/evals/scripts/110_grade_inference.py \
  --inputs ./runs/brick_run.jsonl \
  --judges packages/evals/configs/judges.yaml \
  --out ./runs/brick_graded.jsonl

# Aggregate the panel into majority-vote verdicts
uv run python packages/evals/scripts/115_aggregate_panel.py \
  --in ./runs/brick_graded.jsonl \
  --out ./runs/brick_final.jsonl
```

## Aggregate results

```bash
uv run python packages/evals/scripts/130_aggregate_results.py \
  --in ./runs/brick_final.jsonl | tee results.txt
```

Expected output (Brick max-quality profile):

```
Overall accuracy                : 76.98%   (κ vs panel: 0.761)
Per-dimension:
  coding                         : 80.4%
  creative_synthesis             : 72.1%
  instruction_following          : 78.9%
  math_reasoning                 : 74.6%
  planning_agentic               : 79.3%
  world_knowledge                : 76.5%
Cost (× cheapest)               : 1.5×
Latency (avg, s)                : 22.8
Oracle bound (3-model pool)     : 83.25%
```

## Compare with baselines (RouteLLM, FrugalGPT, cascade-routing)

```bash
cd packages/evals/baselines
uv run python run_routellm.py    --dataset ../../../data/dataset_a --out ../../../runs/routellm.jsonl
uv run python run_frugalgpt.py   --dataset ../../../data/dataset_a --out ../../../runs/frugalgpt.jsonl
uv run python run_cascade_routing.py --dataset ../../../data/dataset_a --out ../../../runs/cascade.jsonl

# Cross-comparison: accuracy × cost × latency frontier
uv run python compare_routers.py \
  ../../../runs/brick_final.jsonl \
  ../../../runs/routellm.jsonl \
  ../../../runs/frugalgpt.jsonl \
  ../../../runs/cascade.jsonl
```

Results table format matches `packages/evals/baselines/RESULTS.md` and paper Section 6.

## Cost / quality Pareto sweep

```bash
uv run python packages/evals/baselines/eval_knob_full_dataset.py \
  --dataset ./data/dataset_a \
  --knob-range -1.0 1.0 0.05 \
  --out ./runs/brick_pareto.jsonl
```

Sweeps the preference knob `r ∈ [-1, 1]` and plots the resulting cost/quality frontier (figure in paper Section 6).

## Troubleshooting

- **Judge cost too high** → grade a stratified subsample first (e.g. `--limit 500` to `100_run_inference.py`) before scaling to the full 5,504.
- **Rate limits from OpenRouter** → bump `judges.yaml` retry settings or distribute across multiple keys.
- **Memory pressure on capability classifier** → the ModernBERT-base inference fits in <2 GB; if you must run on CPU, set `--device cpu` and expect ~10× slower.

## Where to go next

- Audit how the classifiers were trained: [`packages/training/README.md`](../../packages/training/README.md).
- Understand the routing math: paper Section 3 and [`apps/router/README.md`](../../apps/router/README.md).
