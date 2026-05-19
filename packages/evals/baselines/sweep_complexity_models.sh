#!/bin/bash
# Run V3 honest 3way multi-seed for each complexity model debug file
# Usage: ./sweep_complexity_models.sh
set -e

cd /root/forkGO
export WANDB_API_KEY=$(cat /root/.wandb_key)

for MODEL in eco max extractor; do
  if [ "$MODEL" = "eco" ]; then
    INPUT="external_comparison/predictions/brick_debug_gpu.jsonl"
  else
    INPUT="external_comparison/predictions/brick_debug_${MODEL}.jsonl"
  fi
  if [ ! -f "$INPUT" ]; then
    echo "[skip] $MODEL: input $INPUT missing"
    continue
  fi
  echo "[run] complexity_model=$MODEL input=$INPUT"
  for SEED in alpha beta gamma delta epsilon; do
    nohup python3 external_comparison/eval_brick_3way.py \
      --input "$INPUT" \
      --mode v3_random --trials 30000 \
      --seed brick-3way-${MODEL}-${SEED} \
      --run-name 3way-v3-${MODEL}-${SEED} \
      --out external_comparison/predictions/brick_3way_v3_${MODEL}_${SEED}.json \
      > /tmp/brick_3way_v3_${MODEL}_${SEED}.log 2>&1 &
    echo "  ${MODEL}-${SEED} PID: $!"
  done
done
echo "all launched"
