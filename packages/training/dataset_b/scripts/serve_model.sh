#!/usr/bin/env bash
# Run inside tmux on cluster.
# Args: $1 = HF repo id, $2 = vllm docker image tag (e.g. v0.19.1), $3... = extra vllm args
set -e
REPO="$1"; shift
IMAGE_TAG="$1"; shift
LOG="$HOME/sgl_serve.log"
echo "[serve] $(date) launching vllm/vllm-openai:$IMAGE_TAG for $REPO" | tee -a "$LOG"

HF_TOKEN_VAL=""
if [ -f "$HOME/.hf_token_regolo" ]; then HF_TOKEN_VAL=$(cat "$HOME/.hf_token_regolo"); fi
if [ -f /root/.hf_token_regolo ]; then HF_TOKEN_VAL=$(cat /root/.hf_token_regolo); fi

sudo docker stop vllm_serve 2>/dev/null || true
sudo docker rm vllm_serve 2>/dev/null || true
sleep 5

sudo docker run -d --name vllm_serve \
  --gpus all --runtime=nvidia --ipc=host \
  -e NCCL_P2P_DISABLE=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_TOKEN="$HF_TOKEN_VAL" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 30000:30000 \
  "vllm/vllm-openai:$IMAGE_TAG" \
  --model "$REPO" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --trust-remote-code \
  --host 0.0.0.0 --port 30000 \
  "$@" 2>&1 | tee -a "$LOG"

# follow container logs into LOG
sudo docker logs -f vllm_serve 2>&1 | tee -a "$LOG"
