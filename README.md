# Regolo Semantic Routing

A system for intelligent routing to Regolo API endpoints with vLLM Semantic Router.

## Architecture

- **Container 1**: Filter Router (`core.py`) - Port 8000
  - Analyzes modality (text/image/audio)
  - Forwards text-only requests to vLLM SR

- **Container 2**: vLLM Semantic Router - Port 8888
  - Routing based on keywords, embedding, domain signals
  - Routes text queries to appropriate Regolo model

## Quick Start

```bash
# 1. Install Docker and setup
./install-docker.sh

# 2. Set environment variables
echo "REGOLO_API_KEY=your_key" > .env
echo "HF_TOKEN=your_hf_token" >> .env

# 3. Start services
cd docker
docker compose up -d
```

## Test Endpoint

```bash
# Test filter router (port 8000)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"Hello"}]}'

# Test vLLM SR (port 8888)
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"Hello"}]}'
```

## Quickstart for my-model

### Install

```bash
pip install -e .
```

### Initialize

```bash
my-model init
# Follow wizard to create alias <ALIAS>
```

### Run gateway

```bash
my-model serve --alias <ALIAS> --host 127.0.0.1 --port 8000
```

### Test (non-stream)

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","messages":[{"role":"user","content":"ciao"}]}'
```

### Test (stream)

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","stream":true,"messages":[{"role":"user","content":"scrivi 5 parole"}]}'
```

![quickstart screenshot](docs/quickstart.png)

## Configuration

- **vLLM SR Config**: `docker/config/config.yaml`
- **Models**: `vLLMsr_model.py`

## Files

| File | Purpose |
|------|---------|
| `core.py` | Filter router (modality detection + forwarding) |
| `vLLMsr_model.py` | Model definitions for Regolo |
| `docker/config/config.yaml` | vLLM SR routing configuration |
| `install-docker.sh` | Ubuntu 24.04 Docker setup script |

## Requirements

- **Ubuntu 24.04 LTS**
- **8 GB RAM minimum** (16 GB recommended)
- **Docker & Docker Compose**

## Zero to gateway in 2 minutes

### Install
```bash
pip install -e .
```

### Init (wizard)
Run the interactive wizard to create a new virtual model alias and configure router and providers:
```bash
my-model init
```
Follow the prompts to:
- Choose an alias for your virtual model (e.g., `brick`)
- Provide the vLLM Semantic Router URL and mode
- Add one or more providers (OpenAI‑compatible, Regolo, etc.) and their API keys
- Register backend models you want to expose

A workspace configuration will be saved in `~/.my-model/<ALIAS>/config.yaml`.

### Serve
Start the local gateway:
```bash
my-model serve --alias <ALIAS> --host 127.0.0.1 --port 8000
```
The server exposes OpenAI‑compatible endpoints on the given host/port.

### Curl examples
#### Non‑streaming request
```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<ALIAS>","messages":[{"role":"user","content":"ciao"}]}'
```

#### Streaming request
```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<ALIAS>","stream":true,"messages":[{"role":"user","content":"scrivi 5 parole"}]}'
```

### Override header `x-selected-model`
If you want to bypass the router and call a specific backend model directly, add the header `x-selected-model` with the backend model ID:
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-selected-model: my-regolo-model" \
  -d '{"model":"<ALIAS>","messages":[{"role":"user","content":"hello"}]}'
```

### Config file reference
The workspace config (`config.yaml`) contains:

- `alias`: virtual model name
- `gateway`: `host`, `port`, `log_level`, `cors`
- `router`: `vsr_url`, `mode`, `timeout`
- `providers`: list of provider definitions (`id`, `type`, `api_key`, `base_url`, …)
- `models`: list of backend models (`backend_id`, `provider_id`, `model_id`, `tags`)
- `routing`: optional static mapping from VSR model names to backend IDs

Edit this file manually if you need to fine‑tune routing or add additional models.

### Done
You should now have a fully functional local OpenAI‑compatible gateway in under two minutes.
