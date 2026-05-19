# `packages/training/`: Training the Brick classifiers

Two trained models power Brick's routing decision:

1. **Capability classifier**: ModernBERT-base, 6-label sigmoid head, predicts `p(x) ∈ Δ⁶` over the 6 capability dimensions. Published as [`regolo/brick-modernbert-capability-classifier`](https://huggingface.co/regolo/brick-modernbert-capability-classifier).
2. **Complexity classifier**: Qwen3.5-0.8B + LoRA adapter, 3-class (easy/medium/hard) predicts `τ`. Published as [`regolo/brick-complexity-2-eco`](https://huggingface.co/regolo/brick-complexity-2-eco).

**Both models are already trained and on HF.** For day-to-day use, download them via [`packages/datasets/scripts/download_models.py`](../datasets/scripts/download_models.py). This README is for **auditing and re-training** the models.

## What's in here

```
packages/training/
├── dataset_b/                  # Build Dataset B (training set) from scratch
│   ├── scripts/
│   │   ├── 01_generate_queries.py     # synthesize candidate queries per capability dim
│   │   ├── 02_run_judge.py            # judge each query for capability label (vLLM panel)
│   │   ├── 03_aggregate.py            # aggregate judge votes
│   │   ├── 04_human_eval_prep.py      # stratified subset for human validation
│   │   ├── 05_push_hub.py             # push to massaindustries/dataset-B-modernbert-train
│   │   ├── 06_annotate_human_eval.py
│   │   └── 07_compute_kappa.py        # Cohen's κ on human vs panel labels
│   ├── configs/                # judges.yaml + generation specs
│   ├── prompts/                # query-generation prompts per dim
│   └── sky/                    # SkyPilot YAML for cloud GPU runs
└── modernbert/                 # Train the capability classifier (the headline model)
    ├── scripts/
    │   ├── train_modernbert.py        # fine-tune ModernBERT-base/large, 6-label BCE
    │   ├── sanity_check.py            # smoke test on a fresh model+dataset
    │   ├── select_top3.py             # pick top-3 sweep runs by human_eval Pearson
    │   ├── push_winner.py             # push the winner to HF Hub
    │   └── sweep_email_monitor.py     # emails progress during multi-hour sweeps
    ├── configs/                # sweep.yaml (hyperparam grid for wandb agent)
    ├── sky/                    # SkyPilot YAML for sweep runs
    └── README.md               # detailed training run notes
```

## Reference run: capability classifier

The winning configuration (selected against the human-eval split):

| Hyperparameter | Value |
|---|---|
| Base model | `answerdotai/ModernBERT-base` |
| Loss | Binary cross-entropy (multi-label, 6 outputs) |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01) |
| Learning rate | 5e-5, warmup 5%, linear decay |
| Batch size | 32 (gradient accumulation 1) |
| Epochs | 4 |
| Max length | 1024 tokens |
| Hardware | 4× L40S (SkyPilot, ~2 h wall clock) |
| Result | Pearson macro = 0.87, MAE = 0.14 on `human_eval` split |

The full hyperparameter sweep (search space + per-run metrics) is publicly readable on Weights & Biases:

> 🔗 **W&B sweep audit**: [`wandb.ai/massa-industries/sweeps/dataset-b-modernbert/0srgzjrg`](https://wandb.ai/massa-industries/sweeps/dataset-b-modernbert/0srgzjrg)

Reproduce the sweep locally:

```bash
cd packages/training/modernbert
uv pip install -e ../../..        # install brick-training workspace
wandb login                       # uses your wandb token
wandb sweep configs/sweep.yaml    # prints SWEEP_ID
wandb agent <entity>/<project>/<SWEEP_ID>
```

Push winner to HF:

```bash
python scripts/select_top3.py --sweep <SWEEP_ID>
python scripts/push_winner.py --run-id <RUN_ID> --repo regolo/brick-modernbert-capability-classifier-replicated
```

## Reference run: complexity classifier

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (Qwen3.5-0.8B in paper) |
| Adapter | LoRA (r=16, α=32, dropout=0.05, target: q/k/v/o + gate/up/down) |
| Loss | Cross-entropy (3-class) with asymmetric penalty (`λ_over=0.7`) and `label_smoothing=0.08` |
| Optimizer | AdamW, LR=3e-4 (LoRA-only) |
| Batch size | 64 |
| Epochs | 3 |
| Hardware | Single L40S (~45 min) |

Training script: `dataset_b/scripts/train_complexity_lora.py` (TODO: add separate doc; current canonical version lives under `modernbert/scripts/` for backward-compat with the published `brick-complexity-2-eco`).

## Rebuild Dataset B from scratch (advanced)

If you want to regenerate the training set rather than download it:

```bash
cd packages/training/dataset_b
uv pip install -e ../../..
# 1. Generate candidate queries per capability dim
python scripts/01_generate_queries.py --config configs/generation_specs.yaml
# 2. Spin up judge panel (vLLM) and label each query
python scripts/02_run_judge.py --judges configs/judges.yaml
# 3. Aggregate panel votes (majority + tie-break)
python scripts/03_aggregate.py
# 4-7. Human-eval split, push, annotate, κ
```

Expect ~12 h of judge time on 4-GPU node + ~4 h human annotation for the κ split.

## Email notifier (optional)

Long sweeps can email progress via Gmail SMTP. Configure `~/.gmail_app_password` and run:

```bash
python scripts/sweep_email_monitor.py --sweep <SWEEP_ID> --to you@example.com --every 30m
```

## Caveats

- Training scripts are checked in as-is from the paper run; they are not packaged behind a stable Python API. Treat them as reproducible recipes, not as a library.
- Dataset B is published with judge labels only (no judge raw chains). The judge raw outputs are too large for the Hub.
- The Qwen complexity LoRA adapter is published merged with its base model. To fine-tune your own LoRA on top, start from `Qwen/Qwen2.5-0.5B-Instruct` and follow the recipe above.
