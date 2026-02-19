# Project Definition

## Objective
Build a Python package `my-model` that provides a CLI (`my-model`) and a local OpenAI-compatible gateway. The gateway forwards requests to multiple provider backends (OpenAI-compatible, Regolo) based on decisions made by the vLLM Semantic Router (VSR). Users can create a virtual model alias, configure routing, and run the service locally.

## MVP Scope
- Installable via `pip install my-model` (editable mode allowed) exposing the `my-model` command.
- `my-model init` wizard creates a virtual model alias and persistent workspace config (`~/.my-model/<ALIAS>/config.yaml`).
- `my-model serve` starts a FastAPI server exposing:
  - `POST /v1/chat/completions` (supports streaming via SSE)
  - `GET /v1/models` (returns the virtual alias)
  - `GET /health`
- Routing logic:
  - Header `x-selected-model` overrides VSR.
  - Otherwise call VSR (via `x-vsr-selected-model` or response model) to pick a backend model.
- Provider adapters:
  - Generic OpenAI-compatible provider (httpx async client).
  - Regolo provider (default base URL).
- Configuration schema using Pydantic; secrets stored in OS keyring or `.env` fallback, never logged.
- CLI built with Typer; includes `init`, `serve`, `status`, `provider add|list|remove`, `model add|list|remove`, `doctor`.
- README includes a quick-start with curl examples and a screenshot of output.

## Definition of Done
- `pip install -e .` creates the `my-model` console script.
- `my-model init` completes wizard and produces `~/.my-model/<ALIAS>/config.{yaml,json}`.
- `my-model serve --alias <ALIAS>` runs without errors, logs routing decisions, and respects log level.
- Manual tests succeed:
  - `curl -s http://127.0.0.1:8000/v1/models`
  - `curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","messages":[{"role":"user","content":"ciao"}]}'`
  - `curl -N http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","stream":true,"messages":[{"role":"user","content":"scrivi 5 parole"}]}'`
- README contains reproducible quick‑start section with these curl commands and a screenshot of the non‑streaming response.
- All unit and integration tests in `tests/` pass (`pytest` green).
- No secrets are printed in logs; API keys are either stored in keyring or `.env`.
- Code follows linting/style conventions.
