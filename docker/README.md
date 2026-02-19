# Docker Compose (optional)

This directory provides Docker Compose configurations for running the **my-model** gateway locally.

## 1️⃣ Standard composition (real VLLM‑SR)

The repository already contains a `docker-compose.yml` that starts:
- **filter-router** – the FastAPI gateway (based on `core.py`).
- **vllm‑sr** – the official vLLM Semantic Router image.

```bash
cd docker
docker compose up -d   # starts both services
```

The gateway expects the router at the URL defined by the `VLLM_SR_URL` environment variable
(default `http://vllm-sr:8888/v1/chat/completions`).  Once the containers are up you can
test the gateway:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"brick","messages":[{"role":"user","content":"ciao"}]}'
```

## 2️⃣ Mock composition (offline / CI friendly)

For environments without access to the real vLLM router you can use the **mock router** which
simply returns a fixed model name (`mock-model`).  This is useful for CI pipelines or quick local
debugging.

A separate compose file `docker-compose.mock.yml` is provided.  It starts the gateway and a
minimal mock router implemented in `mock_router.py`.

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.mock.yml up -d
```

The mock router listens on port `8888` inside the network and the gateway is configured
automatically via the `VLLM_SR_URL` environment variable to point at `http://mock-router:8888/v1/chat/completions`.

You can test the same way as above – the response will always contain the model `mock-model`.

## Environment variables

- `REGOLO_API_KEY` – optional, required only if you want the gateway to forward calls to
  the real Regolo provider.  When using the mock router you can omit it.
- `HF_TOKEN` – required for the real vLLM‑SR image (see the main `README.md`).

## Clean‑up

```bash
docker compose down --remove-orphans
```
