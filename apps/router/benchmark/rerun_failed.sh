#!/usr/bin/env bash
# Re-run Devstral-2-123B e Mistral-Small-4-119B
set -euo pipefail

WORKDIR=$HOME/sky_workdir
RESULTS=/data/results/eval_brick
PORT=30000
PYTHON=$HOME/venv/bin/python

kill_server() {
    $HOME/venv/bin/python -c "import subprocess; subprocess.run(['pkill','-f','sglang.launch_server'], capture_output=True)" 2>/dev/null || true
    pkill -f sglang.launch_server 2>/dev/null || true
    sleep 15
    pkill -9 -f sglang 2>/dev/null || true
    sleep 5
    echo "[kill] done"
}

serve_wait() {
    local MODEL="$1"
    local LOG="$2"
    shift 2
    local EXTRA_ARGS=("$@")

    echo "[serve] $MODEL"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL" \
        "${EXTRA_ARGS[@]}" \
        --host 127.0.0.1 --port "$PORT" \
        --max-running-requests 128 \
        --chunked-prefill-size 4096 \
        --mem-fraction-static 0.82 \
        --schedule-policy lpm \
        --trust-remote-code \
        > "$LOG" 2>&1 &
    local SRV_PID=$!
    echo "[serve] pid=$SRV_PID"

    local TIMEOUT=1800 ELAPSED=0
    while ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 5; ELAPSED=$((ELAPSED+5))
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "[serve] process died. Last lines:"; tail -20 "$LOG"; return 1
        fi
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "[serve] timeout"; kill "$SRV_PID" 2>/dev/null; return 1
        fi
    done
    echo "[serve] ready after ${ELAPSED}s"
}

# --- Devstral-2-123B ---
echo "=== devstral-2-123b start $(date) ==="
T0=$(date +%s)
mkdir -p "$RESULTS/devstral-2-123b"

serve_wait "mistralai/Devstral-2-123B-Instruct-2512" \
    "$RESULTS/devstral-2-123b_serve2.log" \
    --tp-size 4 --dp-size 1 --quantization fp8 || {
    echo "[FAIL] devstral serve failed — skip to mistral"
    kill_server
    DEVSTRAL_FAILED=1
}

if [ "${DEVSTRAL_FAILED:-0}" != "1" ]; then
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

# Svuota results precedenti (tutti pred=None)
> "$RESULTS/mistral-small-4-119b/results.jsonl"
rm -f "$RESULTS/mistral-small-4-119b/summary.json"

serve_wait "mistralai/Mistral-Small-4-119B-2603" \
    "$RESULTS/mistral-small-4-119b_serve2.log" \
    --tp-size 4 --dp-size 1 --quantization fp8 || {
    echo "[FAIL] mistral serve failed"
    exit 1
}

"$PYTHON" "$WORKDIR/evaluate_brick.py" \
    --source-jsonl /data/source/brick_large.jsonl \
    --url "http://127.0.0.1:$PORT/v1" \
    --model-name "mistralai/Mistral-Small-4-119B-2603" \
    --output-dir "$RESULTS/mistral-small-4-119b" \
    --num-workers 32 --max-tokens 8192 --retry 2 \
    2>&1 | tee "$RESULTS/mistral-small-4-119b/rerun_eval.log"

kill_server
T1=$(date +%s)
echo "{\"slug\":\"mistral-small-4-119b\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/mistral-small-4-119b/timing.json"
echo "=== mistral-small-4-119b done in $((T1-T0))s ==="
rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603 2>/dev/null || true

echo "=== ALL RERUN DONE $(date) ==="

"$PYTHON" "$WORKDIR/aggregate_brick_results.py" \
    --results-dir "$RESULTS" \
    --models-tsv "$WORKDIR/models.tsv" \
    --cost-per-hour 3.40 \
    --output-csv "$RESULTS/brick_large_results.csv" \
    --output-md  "$RESULTS/brick_large_results.md" \
    2>&1 | tee "$RESULTS/rerun_aggregate.log"

echo "=== REPORT UPDATED ==="
