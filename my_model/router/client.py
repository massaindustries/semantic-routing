"""VSR (vLLM Semantic Router) client implementation.

This module provides a thin async client that talks to a VSR service and
extracts the selected backend model identifier.  The client supports two
operation modes configured via :class:`my_model.config.schema.RouterConfig`:

* ``headers`` – The VSR returns the selected model in the HTTP response
  header ``x-vsr-selected-model`` (primary) and optionally also in the JSON
  body ``model`` as a fallback.
* ``json`` – The VSR returns a standard OpenAI‑compatible JSON payload with a
  ``model`` field; no header is expected.

The client is deliberately lightweight and does not depend on any external
packages other than ``httpx`` which is already used elsewhere in the project.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import httpx

logger = logging.getLogger(__name__)


class VSRClient:
    """Async client for the vLLM Semantic Router.

    Parameters
    ----------
    vsr_url: str
        Full URL of the VSR endpoint (e.g. ``http://localhost:8888/v1/chat/completions``).
    mode: str, optional
        ``"headers"`` (default) reads the selected model from the header
        ``x-vsr-selected-model``; ``"json"`` reads it from the JSON body.
    timeout: int, optional
        Request timeout in seconds (applies to both connect and read).
    """

    def __init__(self, vsr_url: str, mode: str = "headers", timeout: int = 30):
        if mode not in {"headers", "json"}:
            raise ValueError("mode must be 'headers' or 'json'")
        self.vsr_url = vsr_url.rstrip('/')
        self.mode = mode
        self.timeout = timeout
        # httpx client reuse for performance – it will be closed by the caller
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout))

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    async def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        """Internal helper that performs the POST request.

        It raises ``httpx.HTTPStatusError`` for non‑2xx responses; callers can
        catch it and convert to a domain‑specific error if required.
        """
        headers = {"Content-Type": "application/json"}
        response = await self._client.post(self.vsr_url, json=payload, headers=headers)
        response.raise_for_status()
        return response

    async def get_selected_model(self, messages: List[Dict[str, Any]]) -> str:
        """Send *messages* to the VSR and return the selected model name.

        The request payload mirrors the one used in ``core.py`` because the VSR
        expects an OpenAI‑compatible request.  Only the ``model`` field is a
        placeholder – the router does not use it.
        """
        payload = {
            "model": "generic",
            "messages": messages,
            "temperature": 0.0,
        }
        try:
            response = await self._post(payload)
        except Exception as exc:
            logger.error(f"[VSR] request failed: {exc}")
            raise

        selected = ""
        if self.mode == "headers":
            selected = response.headers.get("x-vsr-selected-model", "")
            if selected:
                logger.info(f"[VSR] selected model from header: {selected}")
            else:
                logger.debug("[VSR] header 'x-vsr-selected-model' missing, falling back to JSON body")
        if not selected:
            try:
                body = response.json()
                selected = body.get("model", "")
                if selected:
                    logger.info(f"[VSR] selected model from JSON body: {selected}")
            except Exception:
                logger.debug("[VSR] failed to parse JSON body for model selection")
        return selected


async def select_backend_id(messages: List[Dict[str, Any]], workspace_config) -> str:
    """Determine the backend identifier for *messages* using the workspace router.

    The function creates a temporary :class:`VSRClient` based on the router
    configuration stored in ``workspace_config.router``.  After obtaining the
    VSR‑selected model name it resolves the final ``backend_id`` using the
    following precedence:

    1. If the VSR model name exists in ``workspace_config.routing.mapping``
       return the mapped ``backend_id``.
    2. If a backend with ``model_id`` equal to the VSR model name exists, return
       its ``backend_id``.
    3. Fallback – return the raw VSR model name (the caller may treat it as a
       backend identifier directly).
    """
    router_cfg = workspace_config.router
    client = VSRClient(vsr_url=router_cfg.vsr_url, mode=router_cfg.mode, timeout=router_cfg.timeout)
    try:
        vsr_model = await client.get_selected_model(messages)
    finally:
        await client.close()

    if not vsr_model:
        raise ValueError("VSR did not return a selected model")

    mapping = getattr(workspace_config.routing, "mapping", {}) or {}
    if vsr_model in mapping:
        backend_id = mapping[vsr_model]
        logger.info(f"[VSR] routing map used: {vsr_model} -> {backend_id}")
        return backend_id

    for model_cfg in getattr(workspace_config, "models", []):
        if model_cfg.model_id == vsr_model:
            logger.info(f"[VSR] backend matched by model_id: {vsr_model} -> {model_cfg.backend_id}")
            return model_cfg.backend_id

    logger.info(f"[VSR] fallback to raw model name: {vsr_model}")
    return vsr_model
