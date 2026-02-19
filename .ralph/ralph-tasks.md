# Ralph Tasks
# Progetto: "my-model" — CLI + Gateway locale (OpenAI-compatible) con routing via vLLM Semantic Router (VSR)

- [x] Definire obiettivo, scope MVP e “definition of done”
  - Output atteso MVP:
    - `pip install my-model` (o equivalente) -> comando `my-model`
    - `my-model init` crea un "modello virtuale" (alias) + config provider/modelli
    - `my-model serve` avvia un gateway locale con endpoint `/v1/chat/completions` e `/v1/models`
    - Chiamando l’endpoint con `"model": "<ALIAS>"` ottengo routing e risposta dal backend selezionato
  - Vincoli:
    - API OpenAI-compatible (request/response e SSE streaming)
    - Routing deciso da VSR (leggendo header `x-vsr-selected-model` o fallback equivalente)
    - Provider supportati (MVP -> OpenAI-compatible + Regolo; poi OpenAI/Anthropic/Google/xAI)
  - Acceptance:
    - README con quickstart riproducibile (curl) + screenshot output

- [x] Scelta naming e struttura repo
  - Decidere:
    - Nome package python: `my_model` (import) e CLI: `my-model` (entrypoint)
    - Directory:
      - `my_model/` (core)
      - `my_model/providers/` (adapters)
      - `my_model/router/` (VSR client)
      - `my_model/gateway/` (FastAPI)
      - `my_model/cli/` (Typer)
      - `examples/` (curl + python)
      - `docker/` (compose opzionale)
  - Acceptance:
    - Repo bootstrappato con `pyproject.toml`, lint/test config, `__init__.py`

- [x] Definire schema config (Pydantic) e formato di persistenza
  - File config “workspace” (per modello virtuale):
    - Path: `~/.my-model/<ALIAS>/config.yaml` (o json) + override via env
    - Contenuti minimi:
      - `alias`: nome modello virtuale
      - `gateway`: host/port, log level, CORS
      - `router`: `vsr_url`, `mode` (headers|json), timeout
      - `providers`: elenco provider configurati (senza segreti in chiaro se possibile)
      - `models`: elenco backend model registrati (provider_id + model_id + tags/capabilities)
      - `routing`: mapping/strategie (opzionale, se VSR usa nomi custom)
  - Segreti:
    - Preferibile: OS keyring (fallback: `.env` locale)
    - Regola: mai stampare segreti nei log
  - Acceptance:
    - `my-model config show` maschera le API key
    - `my-model config validate` segnala errori

- [x] Definire interfaccia provider (adapter) “OpenAI Chat Completions”
  - Contratto minimo:
    - `chat_completions(messages, model, stream, temperature, extra) -> JSON | AsyncIterator[bytes]`
    - Normalizzazione messaggi (text-only MVP)
    - Streaming SSE passthrough compatibile OpenAI
  - Standardizzare error model (status_code + error.message)
  - Acceptance:
    - Provider finto `MockProvider` per test end-to-end senza rete

- [x] Implementare provider: OpenAI-compatible (generico)
  - Parametri:
    - `base_url` (es. `https://api.openai.com/v1` o endpoint compatibile)
    - `api_key`
    - `models` recuperabili via `/v1/models` (opzionale)
  - Implementazione:
    - httpx AsyncClient con retry/backoff per 429/5xx
    - Streaming: passthrough SSE
  - Acceptance:
    - Test: mock server -> verify streaming e non-stream

- [x] Implementare provider: Regolo (come OpenAI-compatible specializzato)
  - Default:
    - `base_url=https://api.regolo.ai/v1`
  - Extra (P1, non MVP):
    - endpoint audio transcriptions
    - OCR model (se esiste) via chat completions multimodale
  - Acceptance:
    - `my-model doctor` verifica connessione e key valida (senza loggare il token)

- [x] Implementare “Model Registry” (backend models)
  - CLI deve supportare:
    - `my-model model add` (provider, model_id, tags)
    - `my-model model list`
    - `my-model model remove <id>`
  - Schema backend model:
    - `backend_id` (string univoca)
    - `provider_id`
    - `model_id` (es. `gpt-4o-mini`, `claude-...`, `qwen...`)
    - `capabilities/tags` (es. `code`, `math`, `vision`, `fast`, `cheap`)
  - Acceptance:
    - Dati persistenti nel config workspace

- [x] Implementare client Router (VSR) come “decision engine"
  - Input:
    - prompt/messages normalizzati
    - “virtual model” fisso lato router (es. `MoM` o `auto`) se serve
  - Output:
    - `selected_backend_id` (string) determinato da:
      - header `x-vsr-selected-model` (primario)
      - fallback su `response.model` se VSR risponde OpenAI-like
      - fallback su mapping statico `routing.map[vsr_model_name] -> backend_id`
  - Gestione:
    - timeouts separati (connect/read)
    - log: decision + selected (senza contenuti sensibili)
  - Acceptance:
    - Unit test: header parsing + fallback path

- [x] Progettare UX CLI: flusso “init” come da processo utente
  - `my-model init`
    1) chiede nome alias (es. `brick`, `francesco-gw`, ecc.)
    2) chiede router config (VSR URL + modalità)
    3) entra in wizard providers:
       - seleziona provider: regolo, openai, anthropic, google, xai, openai-compatible
       - chiede API key (store in keyring o env-file)
       - chiede modelli da aggiungere (multi-select o input ripetuto)
    4) salva workspace e mostra comando `my-model serve`
  - Acceptance:
    - Wizard ripetibile: `my-model provider add`, `my-model model add`
    - `my-model init --non-interactive` (P1)

- [x] Implementare CLI con Typer (o Click)
  - Comandi MVP:
    - `my-model init`
    - `my-model serve`
    - `my-model status` (config + check router/provider)
    - `my-model provider add|list|remove`
    - `my-model model add|list|remove`
  - Flags:
    - `--workspace <path>` o `--alias <name>`
    - `--host/--port`
    - `--log-level`
  - Acceptance:
    - `my-model --help` completo + esempi in README

- [x] Implementare Gateway FastAPI (server locale)
  - Endpoints MVP:
    - `POST /v1/chat/completions`
    - `GET /v1/models`
    - `GET /health`
  - Comportamento:
    - se request.model != alias -> 400 con messaggio chiaro
    - supporto `stream=true` (SSE)
    - header override: `x-selected-model` (se presente) bypass router e usa backend specifico
  - Acceptance:
    - `curl` non-stream e stream funzionano

- [x] Integrare routing nel Gateway (text-only MVP)
  - Pipeline:
    1) parse request JSON
    2) normalizza messages (solo testo)
    3) if `x-selected-model`: valida e usa diretto
    4) else: chiama VSR e ottieni `selected_backend_id`
    5) chiama provider selezionato con model_id corrispondente
    6) ritorna la risposta OpenAI-compatible
  - Acceptance:
    - Log include: alias, modality=text, selected_backend_id, provider_id, model_id

- [x] Implementare streaming SSE robusto (passthrough)
  - Requisiti:
    - proxy di bytes SSE senza buffering eccessivo
    - gestione CancelledError pulita
    - errori in streaming: inviare evento `data: { "error": ... }`
  - Acceptance:
    - test manuale con client SSE (curl) + chiusura connessione senza crash

- [x] Implementare `/v1/models` coerente con modello virtuale
  - Deve restituire SOLO l’alias come modello disponibile (MVP)
  - (P1) opzionale: includere anche backend models con `owned_by=provider`
  - Acceptance:
    - `GET /v1/models` -> lista con `id=<ALIAS>`

- [x] Logging, telemetry locale e sicurezza base
  - Logging:
    - request-id
    - tempi (router decision, provider latency)
    - no PII esplicita nei log (best-effort)
  - Sicurezza:
    - CORS configurabile (default: localhost)
    - rate limit (P1)
  - Acceptance:
    - `my-model serve --log-level debug` mostra routing senza dumpare segreti

  - [x] “Doctor” e troubleshooting guidato
  - `my-model doctor`:
    - verifica workspace valido
    - ping router VSR
    - verifica provider keys (senza esporle)
    - test call “hello” dry-run (opzionale)
  - Acceptance:
    - output deterministico e utile (exit code != 0 se fail)

- [x] Packaging e distribuzione
  - `pyproject.toml` con entrypoints console_script
  - versioning semver
  - build: `python -m build`
  - publish (P1): PyPI + GitHub Releases
  - Acceptance:
    - install locale: `pip install -e .` crea comando `my-model`

- [x] Documentazione: README “da zero a gateway in 2 minuti”
  - Sezioni:
    - Install
    - Init (wizard)
    - Serve
    - Curl examples (stream e non-stream)
    - Override header `x-selected-model`
    - Config file reference
  - Esempio curl:
    - `POST /v1/chat/completions` con `"model":"<ALIAS>"` e `"messages":[...]`
  - Acceptance:
    - Un utente “fresh” riesce a farlo partire senza leggere codice

- [x] Test suite minima (CI-ready)
  - Unit:
    - config validation
    - VSR header parsing
    - provider adapter (mock)
  - Integration:
    - avvia gateway + mock provider + mock router
    - verifica end-to-end su `/v1/chat/completions`
  - Acceptance:
    - `pytest` verde in CI

- [x] Docker Compose opzionale (P1)
  - Obiettivo:
    - `docker compose up` avvia gateway + (opzionale) vsr + mock upstream
  - Nota:
    - VSR “vero” può richiedere componenti extra (envoy/stack). Fornire compose “dev” semplificato.
  - Acceptance:
    - esempio funzionante documentato (anche solo mock router)

- [x] Estensione P1: provider Anthropic
  - Implementare adapter (Messages API) e mapping OpenAI<->Anthropic
  - Streaming conversione (se serve)
  - Acceptance:
    - aggiungibile da wizard e selezionabile dal routing

- [x] Estensione P1: provider Google (Gemini)
  - Adapter Gemini + mapping a chat completions
  - Acceptance:
    - aggiungibile da wizard e selezionabile dal routing

- [x] Estensione P1: provider xAI
  - Verificare modalità (OpenAI-compatible o custom)
  - Adapter coerente
  - Acceptance:
    - aggiungibile da wizard e selezionabile dal routing

- [x] Estensione P2: multimodale (riprendere tuo codice, ma “pulito”)
  - Modalità:
    - text-only (già MVP)
    - audio-only -> trascrizione -> routing su testo
    - image-only -> OCR/vision -> routing su testo o vision model
    - combinazioni (text+image, text+audio, image+audio, text+image+audio)
  - Architettura:
    - pipeline per “preprocessing modality” prima del router
    - `capabilities` per backend models (es. `vision=true`, `audio=true`)
  - Acceptance:
    - almeno: audio-only funzionante con provider che supporta trascrizione

- [ ] Estensione P2: policy e controlli (guardrails)
  - Blocco su prompt pericolosi / PII (opzionale)
  - Audit log locale
  - Acceptance:
    - flag `--enable-safety` + test

- [ ] Open Questions (da chiudere prima di P1/P2)
  - [ ] VSR: lo usiamo “solo decision” o come gateway completo a monte?
    - Nota: MVP assume “decision engine” + proxy verso provider esterni
  - [ ] Normalizzazione OpenAI schema:
    - accettiamo input “content” come string o array blocks?
  - [ ] Storage segreti:
    - keyring obbligatorio o fallback `.env` per tutti?
  - [ ] Naming:
    - preferisci `my-model` o nome reale prodotto?

- [ ] Checklist finale “DONE”
  - [ ] Installazione pulita in venv + avvio gateway OK
  - [ ] Wizard init completato e config persistente
  - [ ] Routing VSR -> backend provider funzionante
  - [ ] Streaming SSE OK
  - [ ] README + esempi curl OK
  - [ ] Test base OK
