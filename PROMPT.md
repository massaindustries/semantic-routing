# PRD — my-model (CLI + Local Gateway OpenAI-compatible con routing via vLLM Semantic Router)

## Goal
Costruire una libreria Python installabile che espone una CLI (`my-model`) per creare un **modello virtuale (alias)** e avviare un **gateway locale** compatibile OpenAI (`/v1/chat/completions`, `/v1/models`) che instrada le richieste verso più provider/modelli scelti dall’utente, usando **vLLM Semantic Router** come decision engine.

## Reference Implementation (obbligatoria)
Usa come riferimento tecnico principale questo file locale:
- `/home/rdseeweb/regolo-semantic-routing/core.py`

Regole:
- Prima di implementare qualcosa, apri e studia `core.py`.
- Replica i comportamenti chiave (request parsing, streaming SSE, header override, normalizzazione messaggi, routing via VSR) ma rifattorizza in modo pulito a moduli/libreria.
- Se c’è ambiguità, preferisci il comportamento di `core.py` (è la “source of truth”).

## Scope
### In scope (MVP)
1) `pip install my-model` (o install editable) che crea il comando `my-model`.
2) `my-model init` (wizard) che:
   - chiede nome alias del modello virtuale
   - chiede configurazione router (VSR URL + timeouts)
   - chiede provider e API keys
   - chiede modelli backend da aggiungere dietro al router
   - salva workspace config in `~/.my-model/<ALIAS>/config.yaml` (o json)
3) `my-model serve` avvia un gateway locale FastAPI:
   - `POST /v1/chat/completions` (supporta `stream=true` con SSE)
   - `GET /v1/models` (restituisce almeno l’alias)
   - `GET /health`
4) Routing:
   - se header `x-selected-model` è presente: bypass router e usa direttamente il backend selezionato
   - altrimenti: chiama VSR e usa il modello selezionato (via header VSR o fallback)
5) Provider minimo: `openai-compatible` + `regolo` (che è comunque openai-compatible con default base_url regolo)

### Out of scope (MVP)
- Multimodale (audio/image), guardrails, rate-limit, docker full stack (possono essere P1/P2).
- Auto-discovery completo dei modelli provider (opzionale).

## Requirements (testabili)
1) OpenAI compatibility:
   - accetta payload stile OpenAI: `{model, messages, stream, temperature, ...}`
   - risponde in formato OpenAI-like (incluse scelte/usage dove possibile)
2) Streaming:
   - se `stream=true`, ritorna `text/event-stream` con SSE e passthrough robusto (no crash su disconnect)
3) Alias model:
   - se request `model` != alias workspace -> 400 con errore chiaro
4) Config e segreti:
   - config persistente per alias
   - API keys mai in chiaro nei log
   - preferire keyring; fallback `.env` nel workspace se necessario
5) CLI UX:
   - `my-model --help` completo
   - `my-model status` / `doctor` (minimo ping router + sanity config)

## Constraints (tech stack)
- Python >= 3.11
- FastAPI + Uvicorn
- httpx AsyncClient
- Pydantic per schema config
- Typer (o Click) per CLI
- Struttura modulare (no monolite unico stile script)

## Acceptance criteria (Definition of Done)
- Install:
  - `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Init:
  - `my-model init` completa wizard e crea `~/.my-model/<ALIAS>/config.*`
- Serve:
  - `my-model serve --alias <ALIAS> --host 127.0.0.1 --port 8000`
- Test manuale (non-stream):
  - `curl -s http://127.0.0.1:8000/v1/models`
  - `curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","messages":[{"role":"user","content":"ciao"}]}'`
- Test manuale (stream):
  - `curl -N http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"<ALIAS>","stream":true,"messages":[{"role":"user","content":"scrivi 5 parole"}]}'`

## How to work (Tasks Mode — obbligatorio)
- Usa `.ralph/ralph-tasks.md` come unica fonte di verità del piano.
- Esegui **una task per iterazione**:
  1) scegli la prima task `[ ]`
  2) portala a `[/]` mentre lavori
  3) quando soddisfa acceptance, marcala `[x]`
  4) aggiorna anche eventuali sotto-task
- Se scopri nuove attività necessarie, aggiungile in fondo con checkbox `[ ]` mantenendo lo stesso layout.
- Prima di dichiarare completamento, verifica che i comandi di Acceptance criteria funzionino.

## Implementation notes (dal reference core.py)
- Mantieni semantica:
  - env `VLLM_SR_URL` e comportamento di selezione modello come in `core.py`
  - header override `x-selected-model`
  - gestione streaming SSE simile (yield linee `data: ...\n\n`)
- Rifattorizza in moduli:
  - `gateway/` (FastAPI app)
  - `router/` (client VSR + parsing header)
  - `providers/` (adapter OpenAI-compatible + Regolo)
  - `config/` (schema + persistence + secret store)
  - `cli/` (wizard + comandi)

<promise>COMPLETE</promise>
