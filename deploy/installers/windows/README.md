# Brick → Claude Code on Windows

Run the Brick router locally on Windows so Claude Code transparently routes
each prompt to Haiku / Sonnet / Opus by complexity, saving subscription quota.

## What runs where

```
Claude Code (Windows host)
        │  HTTPS /v1/messages
        ▼
mymodel container (CPU)  ──► api.anthropic.com
        │
        │  /classify  (loopback Docker network)
        ▼
classifier container (NVIDIA GPU, ~3-5s startup)
```

The classifier picks easy/medium/hard. mymodel rewrites the `model` field of
each Anthropic Messages request before forwarding. Your subscription bearer
token never leaves your machine except to reach `api.anthropic.com` directly.

## Prerequisites

| | |
|---|---|
| Windows | 10 22H2 or 11 |
| GPU | Any NVIDIA card with ≥4 GB VRAM (Qwen3.5-0.8B fp16 ~2 GB) |
| Driver | NVIDIA driver 535+ for Windows ([download](https://www.nvidia.com/Download/index.aspx)) |
| Docker Desktop | 4.30+ with WSL2 backend enabled |
| Disk | ~6 GB free (image + model cache) |

GPU passthrough on Windows uses CUDA-on-WSL bundled with the modern NVIDIA
Windows driver. No separate `nvidia-container-toolkit` install is needed —
Docker Desktop wires it up automatically when WSL2 is the backend.

Verify GPU passthrough works **before** running setup:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If you get `unknown flag: --gpus` or a CUDA
error, fix that first (see Troubleshooting).

## Install

```powershell
git clone https://github.com/regolo-ai/brick-cc.git
cd brick-cc\deploy\installers\windows
.\setup.ps1
```

The script:
1. Checks Docker / Compose / GPU passthrough.
2. Generates a 64-char hex bearer token, writes `.env` (next to compose file).
3. Pulls images from GHCR (`-Build` to build locally instead).
4. Brings up the stack with `docker compose up -d`.
5. Waits up to 3 min for health.
6. Sets `ANTHROPIC_BASE_URL=http://localhost:18000` as a User env var.

Reopen your terminal so the env var picks up, then:

```powershell
claude
```

## Verify

```powershell
mymodel claude status
```

Expected output (approximate):

```
Connection
  ANTHROPIC_BASE_URL  http://localhost:18000     ✓ attached
  mymodel             :18000                     ✓ healthy (uptime 12m)
  classifier          http://classifier:8094     ✓ healthy (cuda, last 87ms)

Routing since restart (12m)
  Total requests       18
  easy   → haiku-4-5    7  (39%)
  medium → sonnet-4-6  10  (56%)
  hard   → opus-4-7     1  ( 5%)
  Classifier p50/p95   94ms / 312ms
  Fallback rate         0.0%
```

## Operate

| | |
|---|---|
| Stop      | `docker compose -f deploy/docker-compose/docker-compose.brick-cc.yml down` |
| Start     | `docker compose -f deploy/docker-compose/docker-compose.brick-cc.yml up -d` |
| Logs      | `docker compose -f ... logs -f mymodel` (or `classifier`) |
| Update    | `git pull && docker compose -f ... pull && ... up -d` |
| Uninstall | `.\uninstall.ps1` |

## Troubleshooting

**`docker run --gpus all` errors with "unknown flag" or CUDA error.**
Docker Desktop is not running on the WSL2 backend, or the NVIDIA Windows
driver predates CUDA-on-WSL. Update the driver and toggle Settings → General
→ "Use the WSL 2 based engine".

**Setup hangs at "Waiting for stack to become healthy".**
Open another PowerShell and watch logs:
`docker compose -f deploy/docker-compose/docker-compose.brick-cc.yml logs -f`.
First start takes longer because the classifier downloads ~2 GB of weights.
Ctrl-C the watcher when you see `Model loaded ... on cuda` and `MyModel proxy
server starting on port 8000`.

**Port 18000 already in use.**
`.\setup.ps1 -HostPort 19000` (anything free). The env var is updated to match.

**`claude` says "Extra usage is required for 1M context".**
Brick strips the `context-1m-*` beta header by default — you should not see
this. If you do, your subscription request used a different code path.
Restart `claude`. If persistent, file an issue with `mymodel claude status`
output and the last 50 lines of `mymodel` logs.

**Classifier always returns "medium".**
Look for `Fallback to medium: HTTP request failed: ... timeout` in mymodel
logs — means the classifier is too slow (CPU fallback?) or unreachable.
Confirm with `docker compose ps` that `classifier` shows `Up (healthy)` and
that GPU is being used: `docker compose logs classifier | grep "on cuda"`.

## Privacy / security notes

- The bearer token in `.env` only protects the local classifier endpoint
  inside the Docker network. It is not your Anthropic credential.
- Your Anthropic OAuth bearer is forwarded verbatim to `api.anthropic.com`
  by mymodel; nothing logs or persists it.
- Prompts you send to Claude Code transit the local classifier on their way
  out. Classifier inference happens entirely on your machine; no prompt text
  ever leaves the box for routing purposes.
