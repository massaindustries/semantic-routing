#!/usr/bin/env bash
# Eval Devstral + Mistral + Gemma via Docker vLLM containers
# vLLM 0.12.0: supporta ministral3 (Devstral, Mistral-Small-4)
# vLLM 0.19.1: supporta gemma4 (Gemma-4-31B)
set -euo pipefail

WORKDIR=$HOME/sky_workdir
RESULTS=/data/results/eval_brick
PORT=30000
HF_CACHE=$HOME/.cache/huggingface
PYTHON=$HOME/venv/bin/python

stop_container() {
    sudo docker stop vllm_serve 2>/dev/null || true
    sudo docker rm vllm_serve 2>/dev/null || true
    sleep 10
}

wait_health() {
    local TIMEOUT=1800 ELAPSED=0
    while ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 10; ELAPSED=$((ELAPSED+10))
        if ! sudo docker ps | grep -q vllm_serve; then
            echo "[serve] container died. Last log:"
            sudo docker logs --tail 30 vllm_serve 2>&1 || true
            return 1
        fi
        if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
            echo "[serve] timeout ${TIMEOUT}s"
            sudo docker stop vllm_serve 2>/dev/null
            return 1
        fi
    done
    echo "[serve] ready after ${ELAPSED}s"
}

run_model() {
    local IMAGE="$1"
    local MODEL="$2"
    local SLUG="$3"
    local TP="$4"
    shift 4
    local EXTRA=("$@")

    echo "=== [$SLUG] start $(date) ==="
    local T0=$(date +%s)
    mkdir -p "$RESULTS/$SLUG"
    > "$RESULTS/$SLUG/results.jsonl"
    rm -f "$RESULTS/$SLUG/summary.json"

    stop_container

    echo "[serve] starting $IMAGE for $MODEL"
    sudo docker run -d --name vllm_serve \
        --gpus all --runtime=nvidia \
        --ipc=host \
        -e NCCL_P2P_DISABLE=1 \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -p $PORT:$PORT \
        "$IMAGE" \
        --model "$MODEL" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --max-num-seqs 128 \
        --trust-remote-code \
        --host 0.0.0.0 --port $PORT \
        "${EXTRA[@]}" \
        > "$RESULTS/${SLUG}_docker.log" 2>&1

    if ! wait_health; then
        echo "[FAIL] $SLUG serve failed"
        sudo docker logs --tail 60 vllm_serve > "$RESULTS/${SLUG}_docker_serve.log" 2>&1 || true
        stop_container
        return 1
    fi

    "$PYTHON" "$WORKDIR/evaluate_brick.py" \
        --source-jsonl /data/source/brick_large.jsonl \
        --url "http://127.0.0.1:$PORT/v1" \
        --model-name "$MODEL" \
        --output-dir "$RESULTS/$SLUG" \
        --num-workers 32 --max-tokens 8192 --retry 2 \
        2>&1 | tee "$RESULTS/$SLUG/eval.log"

    sudo docker logs vllm_serve > "$RESULTS/${SLUG}_docker_serve.log" 2>&1 || true
    stop_container

    local T1=$(date +%s)
    echo "{\"slug\":\"$SLUG\",\"wall_seconds\":$((T1-T0))}" > "$RESULTS/$SLUG/timing.json"
    echo "=== [$SLUG] done in $((T1-T0))s ==="
}

# --- Devstral-2-123B (vLLM 0.12, ministral3) ---
run_model "vllm/vllm-openai:v0.12.0" \
    "mistralai/Devstral-2-123B-Instruct-2512" \
    "devstral-2-123b" 4 \
    --quantization fp8 || echo "[SKIP] devstral failed"

# --- Mistral-Small-4-119B (vLLM 0.12, ministral3) ---
run_model "vllm/vllm-openai:v0.12.0" \
    "mistralai/Mistral-Small-4-119B-2603" \
    "mistral-small-4-119b" 4 \
    --quantization fp8 || echo "[SKIP] mistral failed"

# --- Gemma-4-31B (vLLM 0.19.1, gemma4) ---
run_model "vllm/vllm-openai:v0.19.1" \
    "google/gemma-4-31B-it" \
    "gemma-4-31b" 4 \
    --dtype bfloat16 || echo "[SKIP] gemma failed"

echo "=== ALL DOCKER DONE $(date) ==="

"$PYTHON" "$WORKDIR/aggregate_brick_results.py" \
    --results-dir "$RESULTS" \
    --models-tsv "$WORKDIR/models.tsv" \
    --cost-per-hour 3.40 \
    --output-csv "$RESULTS/brick_large_results.csv" \
    --output-md "$RESULTS/brick_large_results.md" \
    2>&1 | tee "$RESULTS/docker_aggregate.log"

echo "=== REPORT UPDATED ==="
