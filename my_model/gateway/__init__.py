"""FastAPI gateway for my_model.

Provides endpoints:
- POST /v1/chat/completions
- GET /v1/models
- GET /health

The gateway uses the workspace configuration to validate the alias, selects the appropriate backend model via the VSR client (or overrides via the `x-selected-model` header), and forwards the request to the corresponding provider.
"""

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from typing import Optional

from my_model.config import WorkspaceConfig, ModelConfig, ProviderConfig
from my_model.router.client import select_backend_id
from my_model.providers import OpenAIProvider, RegoloProvider, MockProvider, BaseProvider

logger = logging.getLogger(__name__)

app = FastAPI()

# Global config set by the CLI before server start.
_workspace_config: Optional[WorkspaceConfig] = None

def set_workspace_config(config: WorkspaceConfig) -> None:
    """Set the workspace configuration used by the gateway.

    This must be called before the FastAPI app starts handling requests.
    """
    global _workspace_config
    _workspace_config = config

def _get_provider_instance(provider_cfg: ProviderConfig):
    """Instantiate a provider based on its configuration.

    Supported providers: ``openai``, ``regolo`` and ``mock``.
    """
    provider_id = provider_cfg.provider_id.lower()
    api_key = provider_cfg.api_key.get_secret_value() if provider_cfg.api_key else None
    if provider_id == "openai":
        return OpenAIProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "regolo":
        return RegoloProvider(base_url=provider_cfg.base_url, api_key=api_key)
    if provider_id == "mock":
        return MockProvider()
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
    # Normalise messages to plain‑text only (MVP behaviour)
    normalized = BaseProvider.normalize_messages(messages)
    # Determine backend identifier
    if x_selected_model:
        backend_id = x_selected_model
        logger.info(f"[Gateway] Using explicit backend from header: {backend_id}")
    else:
        # For MVP without routing, select the first configured backend
        if not _workspace_config.models:
            raise HTTPException(status_code=500, detail="No models configured in workspace")
        backend_id = _workspace_config.models[0].backend_id
        logger.info(f"[Gateway] Using first configured backend: {backend_id}")
    # Locate matching model configuration
    model_cfg: Optional[ModelConfig] = next((m for m in _workspace_config.models if m.backend_id == backend_id), None)
    if model_cfg is None:
        raise HTTPException(status_code=400, detail=f"Backend '{backend_id}' not found in workspace configuration")
    # Locate provider configuration for this model
    provider_cfg: Optional[ProviderConfig] = next((p for p in _workspace_config.providers if p.provider_id == model_cfg.provider_id), None)
    if provider_cfg is None:
        raise HTTPException(status_code=500, detail=f"Provider '{model_cfg.provider_id}' not configured")
    # Build provider instance
    provider = _get_provider_instance(provider_cfg)
    # Forward request to provider
    result = await provider.chat_completions(normalized, model=model_cfg.model_id, stream=stream)
    if stream:
        # Provider returns an async iterator of SSE bytes
        return StreamingResponse(result, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
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

