#!/usr/bin/env bash
# Eval gemma-4-31b con vLLM (SGLang 0.5.10 non supporta gemma4)
set -euo pipefail

WORKDIR=$HOME/sky_workdir
RESULTS=/data/results/eval_brick
PORT=30000
MODEL="google/gemma-4-31B-it"
SLUG="gemma-4-31b"
PYTHON=$HOME/venv/bin/python

echo "=== [$SLUG] check vLLM ==="
"$PYTHON" -c "import vllm; print('vllm', vllm.__version__)" || {
    echo "[FAIL] vllm not installed"; exit 1
}

echo "=== [$SLUG] serve start $(date) ==="
T0=$(date +%s)
mkdir -p "$RESULTS/$SLUG"
LOG="$RESULTS/${SLUG}_vllm_serve.log"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --trust-remote-code \
    --host 127.0.0.1 --port "$PORT" \
    > "$LOG" 2>&1 &
SRV_PID=$!
echo "[serve] pid=$SRV_PID"

TIMEOUT=1800
ELAPSED=0
while ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    sleep 5; ELAPSED=$((ELAPSED+5))
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "[serve] process died"; tail -30 "$LOG"; exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[serve] timeout ${TIMEOUT}s"; kill "$SRV_PID" 2>/dev/null; exit 1
    fi
done
echo "[serve] ready after ${ELAPSED}s"

"$PYTHON" "$WORKDIR/evaluate_brick.py" \
    --source-jsonl /data/source/brick_large.jsonl \
    --url "http://127.0.0.1:$PORT/v1" \
    --model-name "$MODEL" \
    --output-dir "$RESULTS/$SLUG" \
    --num-workers 32 --max-tokens 8192 --retry 2 \
    2>&1 | tee "$RESULTS/$SLUG/eval.log"

pkill -f "vllm.entrypoints" || true
sleep 15
pkill -9 -f vllm 2>/dev/null || true
sleep 5

T1=$(date +%s)
echo "{\"slug\":\"$SLUG\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/$SLUG/timing.json"
echo "=== [$SLUG] done in $((T1-T0))s ==="

rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-31B-it 2>/dev/null || true

# Rigenera report
"$PYTHON" "$WORKDIR/aggregate_brick_results.py" \
    --results-dir "$RESULTS" \
    --models-tsv "$WORKDIR/models.tsv" \
    --cost-per-hour 3.40 \
    --output-csv "$RESULTS/brick_large_results.csv" \
    --output-md  "$RESULTS/brick_large_results.md" \
    2>&1 | tee "$RESULTS/gemma_aggregate.log"

echo "=== REPORT UPDATED ==="
