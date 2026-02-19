"""FastAPI gateway for my_model.

Provides endpoints:
- POST /v1/chat/completions
- GET /v1/models
- GET /health

The gateway uses the workspace configuration to validate the alias, selects the appropriate backend model via the VSR client (or overrides via the `x-selected-model` header), and forwards the request to the corresponding provider.
"""

from fastapi import FastAPI, Request, HTTPException, Header
import uuid
import time
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import json
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from my_model.modality import detect_modality, extract_audio_url, transcribe_audio
from typing import Optional, AsyncIterator

from my_model.config import WorkspaceConfig, ModelConfig, ProviderConfig
from my_model.router.client import select_backend_id
from my_model.providers import OpenAIProvider, RegoloProvider, MockProvider, AnthropicProvider, GoogleGeminiProvider, XAIProvider, BaseProvider

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(f"[Gateway] Incoming request {request.method} {request.url.path} - ID {request_id}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

async def _wrap_sse_stream(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Wrap an async SSE byte stream, handling client disconnects and errors.

    - Stops iteration cleanly on ``asyncio.CancelledError`` (client closed).
    - On any other exception, logs the error and yields a single SSE error
      event in OpenAI format before terminating.
    """
    try:
        async for chunk in stream:
            yield chunk
    except asyncio.CancelledError:
        logger.info("[Gateway] SSE client disconnected")
        # Silently stop iteration.
        return
    except Exception as exc:
        logger.error(f"[Gateway] Streaming error: {exc}")
        err = {"error": {"message": str(exc)}}
        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
        return

app = FastAPI()
app.add_middleware(RequestIdMiddleware)

# Global config set by the CLI before server start.
_workspace_config: Optional[WorkspaceConfig] = None

def set_workspace_config(config: WorkspaceConfig) -> None:
    """Set the workspace configuration used by the gateway.

    This must be called before the FastAPI app starts handling requests.
    """
    global _workspace_config
    _workspace_config = config

    # Configure logger level based on workspace config
    log_level_name = getattr(config.gateway, "log_level", "info").upper()
    numeric_level = getattr(logging, log_level_name, None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    else:
        logger.setLevel(logging.INFO)

    # Configure CORS middleware
    origins = [config.gateway.cors_origin] if config.gateway.cors_origin else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
def _get_provider_instance(provider_cfg: ProviderConfig):
    """Instantiate a provider based on its configuration.

    Supported providers: ``openai``, ``regolo``, ``anthropic``, ``google`` and ``xai``.
    """
    provider_id = provider_cfg.provider_id.lower()
    api_key = provider_cfg.api_key.get_secret_value() if provider_cfg.api_key else None
    if provider_id == "openai":
        return OpenAIProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "regolo":
        return RegoloProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "anthropic":
        return AnthropicProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "mock":
        return MockProvider()
    if provider_id == "google":
        return GoogleGeminiProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "xai":
        return XAIProvider(base_url=provider_cfg.base_url, api_key=api_key)
    raise ValueError(f"Unsupported provider_id '{provider_cfg.provider_id}'")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_selected_model: Optional[str] = Header(None)):
    if _workspace_config is None:
        raise HTTPException(status_code=500, detail="Workspace configuration not set")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    model_requested = payload.get("model", "")
    if model_requested != _workspace_config.alias:
        raise HTTPException(status_code=400, detail=f"Model '{model_requested}' not supported. Use alias '{_workspace_config.alias}'.")
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)
    # Detect modality (audio, image, text)
    modality = detect_modality(messages)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    router_start = time.monotonic()
    # Header override takes precedence over any multimodal handling.
    if x_selected_model:
        backend_id = x_selected_model
        logger.info(f"[Gateway] [{request_id}] Using explicit backend from header: {backend_id}")
        # For header override we keep the original normalisation (plain‑text).
        normalized = BaseProvider.normalize_messages(messages)
    elif modality["audio"] and not modality["image"] and not modality["text"]:
        # Audio‑only request: transcribe then route based on the transcription.
        logger.info(f"[Gateway] [{request_id}] Audio‑only request – performing transcription")
        audio_url = None
        for msg in messages:
            url = extract_audio_url(msg.get("content", ""))
            if url:
                audio_url = url
                break
        if not audio_url:
            raise HTTPException(status_code=400, detail="No audio URL found in request")
        transcription_res = await transcribe_audio(audio_url)
        if transcription_res.get("error"):
            return JSONResponse(content=transcription_res)
        # Extract transcription text (different payload shapes)
        transcription = transcription_res.get("text", "")
        if not transcription:
            choices = transcription_res.get("choices", [])
            if choices:
                transcription = choices[0].get("message", {}).get("content", "")
        if not transcription:
            raise HTTPException(status_code=500, detail="Unable to transcribe audio")
        # Route based on transcription text
        backend_id = await select_backend_id([{"role": "user", "content": transcription}], _workspace_config)
        # Normalise the transcription for the provider call.
        normalized = BaseProvider.normalize_messages([{"role": "user", "content": transcription}])
    else:
        # Default text‑only (or other multimodal) processing – normalise messages.
        normalized = BaseProvider.normalize_messages(messages)
        if not _workspace_config.models:
            raise HTTPException(status_code=500, detail="No models configured in workspace")
        try:
            backend_id = await select_backend_id(normalized, _workspace_config)
        except Exception as exc:
            logger.warning(f"[Gateway] [{request_id}] Routing failed ({exc}), falling back to first configured backend")
            backend_id = _workspace_config.models[0].backend_id
    router_elapsed = time.monotonic() - router_start
    logger.info(f"[Gateway] [{request_id}] Routing selected backend: {backend_id} (t={router_elapsed:.3f}s)")

    # Locate matching model configuration
    model_cfg: Optional[ModelConfig] = next((m for m in _workspace_config.models if m.backend_id == backend_id), None)
    if model_cfg is None:
        raise HTTPException(status_code=400, detail=f"Backend '{backend_id}' not found in workspace configuration")
    # Locate provider configuration for this model
    provider_cfg: Optional[ProviderConfig] = next((p for p in _workspace_config.providers if p.provider_id == model_cfg.provider_id), None)
    if provider_cfg is None:
        raise HTTPException(status_code=500, detail=f"Provider '{model_cfg.provider_id}' not configured")
    # Build provider instance
    logger.info(f"[Gateway] [{request_id}] Alias: {_workspace_config.alias}, modality=text, selected_backend_id: {backend_id}, provider_id: {provider_cfg.provider_id}, model_id: {model_cfg.model_id}")
    provider = _get_provider_instance(provider_cfg)
    # Forward request to provider
    provider_start = time.monotonic()
    result = await provider.chat_completions(normalized, model=model_cfg.model_id, stream=stream)
    provider_elapsed = time.monotonic() - provider_start
    logger.info(f"[Gateway] [{request_id}] Provider '{provider_cfg.provider_id}' latency: {provider_elapsed:.3f}s")
    if stream:
        # Provider returns an async iterator of SSE bytes
        return StreamingResponse(_wrap_sse_stream(result), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})  # type: ignore[arg-type]
    else:
        # Provider returns a JSON‑compatible dict
        return JSONResponse(content=result)

@app.get("/v1/models")
async def list_models():
    if _workspace_config is None:
        raise HTTPException(status_code=500, detail="Workspace configuration not set")
    payload = {
        "object": "list",
        "data": [
            {
                "id": _workspace_config.alias,
                "object": "model",
                "created": 0,
                "owned_by": "my-model",
                "description": f"Virtual model alias '{_workspace_config.alias}'",
            }
        ],
    }
    return JSONResponse(content=payload)

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})

