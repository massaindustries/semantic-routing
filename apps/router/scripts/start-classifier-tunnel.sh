#!/usr/bin/env bash
# Open SSH tunnel from local 127.0.0.1:18094 -> l40s-dev:8094 (brick classifier).
# mymodel reads complexity_service.base_url=http://127.0.0.1:18094 and authenticates
# with the bearer in /root/.brick_classifier_token.
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-18094}"
REMOTE_HOST="${REMOTE_HOST:-l40s-dev}"
REMOTE_PORT="${REMOTE_PORT:-8094}"

pkill -f "ssh.*-L *${LOCAL_PORT}:.*${REMOTE_HOST}" 2>/dev/null || true

ssh -fN \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${REMOTE_HOST}"

echo "Tunnel up: 127.0.0.1:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
