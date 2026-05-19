# Quickstart A. Run the gateway (Docker)

Goal: pull the published image, run it, and confirm the OpenAI-compatible API works. Total time: ~1 minute (after image pull).

## Prerequisites

- Docker (any recent version).
- A `REGOLO_API_KEY` (sign up at [regolo.ai](https://regolo.ai)). The router uses this to call the backend model pool. You can also bring your own provider keys via a custom `config.yaml`.

## Run

```bash
docker run --rm -p 18000:18000 \
  -e REGOLO_API_KEY=$REGOLO_API_KEY \
  ghcr.io/regolo-ai/brick:latest
```

The container starts the Brick HTTP proxy on port 18000 (host).

## Verify

```bash
# Health probe
curl -fs http://localhost:18000/health && echo OK

# Models list: exposes the virtual model "brick" plus the backend pool
curl -s http://localhost:18000/v1/models | jq '.data[].id'
```

## Send a request

```bash
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REGOLO_API_KEY" \
  -d '{
    "model": "brick",
    "messages": [
      {"role": "user", "content": "Explain Mixture-of-Models routing in two sentences."}
    ]
  }' | jq '.choices[0].message.content, .usage'
```

Inspect which backend handled the request:

```bash
curl -s -D - -o /dev/null http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REGOLO_API_KEY" \
  -d '{"model":"brick","messages":[{"role":"user","content":"hi"}]}' \
  | grep -i '^x-selected-model'
```

## Override config

The image bakes in `apps/router/config/config.yaml` as default. To use a custom config, mount it as a volume:

```bash
docker run --rm -p 18000:18000 \
  -v $(pwd)/config.yaml:/app/config/config.yaml:ro \
  -e REGOLO_API_KEY=$REGOLO_API_KEY \
  ghcr.io/regolo-ai/brick:latest
```

See [`config.yaml`](../../config.yaml) for the default and [`apps/router/README.md`](../../apps/router/README.md) for every config knob.

## Bypass the router for a specific model

Pass the backend model directly via `x-selected-model` header:

```bash
curl http://localhost:18000/v1/chat/completions \
  -H "x-selected-model: qwen3.5-9b" \
  -H "Authorization: Bearer $REGOLO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"brick","messages":[{"role":"user","content":"hi"}]}'
```

Brick validates the model exists in the pool, then forwards without running the routing pipeline.

## Next

- Install the CLI for easier day-to-day use: [serve.md](serve.md).
- Re-run the Brick paper evaluation: [eval.md](eval.md).
