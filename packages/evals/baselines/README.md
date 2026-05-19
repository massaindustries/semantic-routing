# `packages/evals/baselines/`: Zero-shot router baselines

Reproduction of the three router baselines compared against Brick in the paper:

- **RouteLLM** (Ong et al., 2024): binary preference-aware router (Causal-LLM judge + matrix factorization).
- **FrugalGPT** (Chen et al., 2023): cost-aware cascade with calibrated thresholds.
- **Cascade-routing** (Aggarwal et al., 2024): sequential model cascade with uncertainty thresholds.

All three are run **zero-shot** on Dataset A: no training or fitting on the eval set. Each baseline ships with its own runner script (`run_<name>.py`) plus a shared aggregation harness.

See the published numbers and the per-dimension breakdown in [`RESULTS.md`](RESULTS.md).

## Layout

```
packages/evals/baselines/
├── RouteLLM/                   # Upstream RouteLLM source (vendored, Apache-2.0)
├── FrugalGPT/                  # Upstream FrugalGPT source (vendored, MIT)
├── cascade-routing/            # Upstream cascade-routing source (vendored, MIT)
├── 00_load_dataset.py          # shared Dataset A loader (uses brick_evals.schema)
├── run_routellm.py             # zero-shot RouteLLM run on Dataset A
├── run_frugalgpt.py            # zero-shot FrugalGPT run
├── run_cascade_routing.py
├── run_brick.py                # Brick run (calls the router HTTP endpoint)
├── run_brick_debug.py          # offline replay against saved verdicts
├── compare_routers.py          # cross-router accuracy × cost × latency table
├── eval_brick_3way.py          # 3-way comparison helper (paper Section 6)
├── eval_knob_full_dataset.py   # cost/quality Pareto sweep over Brick knob r ∈ [-1, 1]
├── fit_brick_risk_adjusted.py  # risk-adjusted optimization (paper §6.3)
├── apply_best_v2.py            # apply best knob to held-out split
├── apply_best_v3.py
├── build_comparison_subset.py
├── select_brick_profiles.py
├── sweep_*.py                  # ablation sweeps (W&B-backed)
├── sweep_complexity_models.sh
├── wandb_sweep_brick_bayes.yaml
└── RESULTS.md                  # canonical numbers reported in the paper
```

## Run

Each baseline needs Dataset A pre-downloaded (see [`packages/datasets/`](../../datasets/)) and a provider API key for the backend pool calls.

```bash
cd packages/evals/baselines

# Brick (against a running router)
uv run python run_brick.py \
  --dataset ../../../data/dataset_a \
  --endpoint http://localhost:18000/v1/chat/completions \
  --out ../../../runs/brick.jsonl

# RouteLLM (zero-shot, uses public RouteLLM checkpoints)
uv run python run_routellm.py \
  --dataset ../../../data/dataset_a \
  --threshold 0.5 \
  --out ../../../runs/routellm.jsonl

# FrugalGPT (zero-shot, calibrated thresholds from paper Table 4)
uv run python run_frugalgpt.py \
  --dataset ../../../data/dataset_a \
  --budget 1.5 \
  --out ../../../runs/frugalgpt.jsonl

# Cascade routing (zero-shot)
uv run python run_cascade_routing.py \
  --dataset ../../../data/dataset_a \
  --out ../../../runs/cascade.jsonl

# Cross-comparison
uv run python compare_routers.py \
  ../../../runs/brick.jsonl \
  ../../../runs/routellm.jsonl \
  ../../../runs/frugalgpt.jsonl \
  ../../../runs/cascade.jsonl
```

## Methodology notes

- **Zero-shot, no fit-on-eval.** None of the baseline routers are trained or calibrated on Dataset A. RouteLLM uses its public Causal-LLM checkpoint; FrugalGPT uses thresholds from the original paper; cascade-routing uses uncertainty thresholds from the published reproduction.
- **Same backend pool.** All routers dispatch to the same 3-model pool (Qwen3.5-9b, DeepSeek-v4-flash, Kimi2.6) for apples-to-apples comparison. The pool definition is in [`../configs/models.yaml`](../configs/models.yaml).
- **Same judges.** Each run is graded by the same 3-judge panel (`gpt-5.4-mini` + Mistral + GLM) using `../scripts/110_grade_inference.py` so the final accuracy numbers are directly comparable.

## Caveat: cascade inefficiency

The cascade-based baselines (FrugalGPT, Cascade Routing) are intrinsically less efficient than pure routers: they spend tokens/latency/cost on a first attempt before deciding whether to escalate. The paper's narrative (Section 6) discusses this in detail: these baselines are included for completeness, not as recommended alternatives.

## Where the numbers in the paper come from

The exact verdict JSONL files used in the paper are too large to commit. To regenerate them from scratch follow the recipe above; total wall-clock is ~2 hours per router on a single workstation (most time is judge calls, not router decisions).
