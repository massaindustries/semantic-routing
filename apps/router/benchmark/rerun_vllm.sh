#!/usr/bin/env bash
# Eval Devstral-2-123B + Mistral-Small-4-119B + Gemma-4-31B con vLLM
# SGLang non supporta MoE Triton kernel su L40S (shared memory limit)
set -euo pipefail

WORKDIR=$HOME/sky_workdir
RESULTS=/data/results/eval_brick
PORT=30000
PYTHON=$HOME/venv/bin/python

kill_server() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f vllm 2>/dev/null || true
    sleep 15
    echo "[kill] done"
}

serve_vllm() {
    local MODEL="$1"
    local LOG="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo "[serve-vllm] $MODEL"
    NCCL_P2P_DISABLE=1 \
    NCCL_SHM_DISABLE=0 \
    NCCL_DEBUG=WARN \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        "${EXTRA_ARGS[@]}" \
        --gpu-memory-utilization 0.82 \
        --max-model-len 8192 \
        --max-num-seqs 128 \
        --enforce-eager \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --host 127.0.0.1 --port "$PORT" \
        > "$LOG" 2>&1 &
    local SRV_PID=$!
    echo "[serve-vllm] pid=$SRV_PID"

    local TIMEOUT=1800 ELAPSED=0
    while ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED+10))
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "[serve-vllm] process died. Last 20 lines:"; tail -20 "$LOG"; return 1
        fi
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "[serve-vllm] timeout ${TIMEOUT}s"; kill "$SRV_PID" 2>/dev/null; return 1
        fi
    done
    echo "[serve-vllm] ready after ${ELAPSED}s"
}

# Installa vLLM se non presente
if ! "$PYTHON" -c "import vllm" 2>/dev/null; then
    echo "[install] installing vLLM..."
    "$PYTHON" -m pip install -q vllm 2>&1 | tail -5
fi

# --- Devstral-2-123B ---
echo "=== devstral-2-123b start $(date) ==="
T0=$(date +%s)
mkdir -p "$RESULTS/devstral-2-123b"

serve_vllm "mistralai/Devstral-2-123B-Instruct-2512" \
    "$RESULTS/devstral-2-123b_vllm_serve.log" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --quantization fp8 || {
    echo "[FAIL] devstral vllm serve failed"
    kill_server
    DEVSTRAL_FAIL=1
}

if [ "${DEVSTRAL_FAIL:-0}" != "1" ]; then
    "$PYTHON" "$WORKDIR/evaluate_brick.py" \
        --source-jsonl /data/source/brick_large.jsonl \
        --url "http://127.0.0.1:$PORT/v1" \
        --model-name "mistralai/Devstral-2-123B-Instruct-2512" \
        --output-dir "$RESULTS/devstral-2-123b" \
        --num-workers 32 --max-tokens 8192 --retry 2 \
        2>&1 | tee "$RESULTS/devstral-2-123b/eval.log"
    kill_server
    T1=$(date +%s)
    echo "{\"slug\":\"devstral-2-123b\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/devstral-2-123b/timing.json"
    echo "=== devstral-2-123b done in $((T1-T0))s ==="
    rm -rf ~/.cache/huggingface/hub/models--mistralai--Devstral-2-123B-Instruct-2512 2>/dev/null || true
fi

# --- Mistral-Small-4-119B ---
echo "=== mistral-small-4-119b start $(date) ==="
T0=$(date +%s)
mkdir -p "$RESULTS/mistral-small-4-119b"
> "$RESULTS/mistral-small-4-119b/results.jsonl"
rm -f "$RESULTS/mistral-small-4-119b/summary.json"

serve_vllm "mistralai/Mistral-Small-4-119B-2603" \
    "$RESULTS/mistral-small-4-119b_vllm_serve.log" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --quantization fp8 || {
    echo "[FAIL] mistral vllm serve failed"
    kill_server
    MISTRAL_FAIL=1
}

if [ "${MISTRAL_FAIL:-0}" != "1" ]; then
    "$PYTHON" "$WORKDIR/evaluate_brick.py" \
        --source-jsonl /data/source/brick_large.jsonl \
        --url "http://127.0.0.1:$PORT/v1" \
        --model-name "mistralai/Mistral-Small-4-119B-2603" \
        --output-dir "$RESULTS/mistral-small-4-119b" \
        --num-workers 32 --max-tokens 8192 --retry 2 \
        2>&1 | tee "$RESULTS/mistral-small-4-119b/eval.log"
    kill_server
    T1=$(date +%s)
    echo "{\"slug\":\"mistral-small-4-119b\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/mistral-small-4-119b/timing.json"
    echo "=== mistral-small-4-119b done in $((T1-T0))s ==="
    rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603 2>/dev/null || true
fi

# --- Gemma-4-31B ---
echo "=== gemma-4-31b start $(date) ==="
T0=$(date +%s)
mkdir -p "$RESULTS/gemma-4-31b"

serve_vllm "google/gemma-4-31B-it" \
    "$RESULTS/gemma-4-31b_vllm_serve.log" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 || {
    echo "[FAIL] gemma vllm serve failed"
    kill_server
    GEMMA_FAIL=1
}

if [ "${GEMMA_FAIL:-0}" != "1" ]; then
    "$PYTHON" "$WORKDIR/evaluate_brick.py" \
        --source-jsonl /data/source/brick_large.jsonl \
        --url "http://127.0.0.1:$PORT/v1" \
        --model-name "google/gemma-4-31B-it" \
        --output-dir "$RESULTS/gemma-4-31b" \
        --num-workers 32 --max-tokens 8192 --retry 2 \
        2>&1 | tee "$RESULTS/gemma-4-31b/eval.log"
    kill_server
    T1=$(date +%s)
    echo "{\"slug\":\"gemma-4-31b\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/gemma-4-31b/timing.json"
    echo "=== gemma-4-31b done in $((T1-T0))s ==="
    rm -rf ~/.cache/huggingface/hub/models--google--gemma-4-31B-it 2>/dev/null || true
fi

echo "=== ALL VLLM DONE $(date) ==="

"$PYTHON" "$WORKDIR/aggregate_brick_results.py" \
    --results-dir "$RESULTS" \
    --models-tsv "$WORKDIR/models.tsv" \
    --cost-per-hour 3.40 \
    --output-csv "$RESULTS/brick_large_results.csv" \
    --output-md  "$RESULTS/brick_large_results.md" \
    2>&1 | tee "$RESULTS/vllm_aggregate.log"

echo "=== REPORT UPDATED ==="
