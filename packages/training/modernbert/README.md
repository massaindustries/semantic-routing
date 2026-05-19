# ModernBERT capability classifier: Training pipeline

Fine-tune ModernBERT-{base,large} on `massaindustries/dataset-B-modernbert-train` as a 6-output sigmoid multi-label classifier for the Brick router.

## Pipeline (10 step)

1. **Preconditions**: `sky status` shows qwen-bench UP; HF dataset accessible.
2. **Manual annotation**: `python scripts/manual_annotate_200.py` → Claude-judged `sample_200_filled.csv`. Then `python ../scripts/07_compute_kappa.py` → target κ ≥ 0.6.
3. **Cluster setup**: `sky launch sky/train.yaml -c qwen-bench --no-setup` (already UP, just sync). Installs deps + auth.
4. **Smoke train**:
   ```bash
   sky exec qwen-bench "cd training && python scripts/train_modernbert.py --smoke --model-size base"
   ```
5. **W&B Sweep**:
   ```bash
   SWEEP=$(wandb sweep --project dataset-b-modernbert configs/sweep.yaml | tail -1 | awk '{print $NF}')
   for i in 0 1 2 3; do
       sky exec qwen-bench "CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP" &
   done
   ```
6. **Top-3 select**: `python scripts/select_top3.py --project dataset-b-modernbert --sweep $SWEEP`
7. **Eval on human_eval (200 Claude)**: `for r in rank1 rank2 rank3; do python scripts/eval_human.py --ckpt outputs/top3/$r/best --output outputs/top3/$r/eval_human.json; done`
8. **Sanity check winner**: `python scripts/sanity_check.py --ckpt outputs/top3/rank1/best`
9. **Export Candle-ready**: `python scripts/export_for_candle.py --ckpt outputs/top3/rank1/best --output outputs/modernbert-winner/best`
10. **Push HF + bench**: `python scripts/push_winner.py --ckpt outputs/modernbert-winner/best --repo massaindustries/modernbert-capability-classifier && python scripts/bench_latency.py --ckpt outputs/modernbert-winner/best`

## Pass criteria

| Step | Threshold |
|---|---|
| Manual annotation κ | macro ≥ 0.6 |
| Sweep best run val | pearson_macro ≥ 0.55 |
| Winner human_eval | pearson_macro ≥ 0.60, mae_macro ≤ 0.15, no dim < 0.30 |
| Sanity semantic | 6/6 match |
| Latency base | p50 < 5ms, p99 < 25ms |

## Hyperparams fixed (sweep variables in `configs/sweep.yaml`)

- Adam β1=0.9, β2=0.98, ε=1e-6 (AnswerDotAI recipe)
- bf16 + FlashAttention-2 (no fp16: issue #35988)
- `lr_scheduler_type=linear`, `optim=adamw_torch_fused`
- max_seq_length=512, max_grad_norm=1.0, seed=42
- per_device_bs: 32 (base), 16 (large, gradacc=2) → effective 128
- EarlyStoppingCallback(patience=2) on pearson_macro
