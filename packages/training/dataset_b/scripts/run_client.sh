#!/usr/bin/env bash
# Run inside tmux on cluster.
# Args: full python command appended to "$HOME/venv/bin/python"
set -e
LOG="$HOME/client.log"
echo "[client] $(date) cmd: $@" | tee -a "$LOG"
cd "$HOME/dataset_b"
exec "$HOME/venv/bin/python" "$@" 2>&1 | tee -a "$LOG"
