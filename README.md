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
