# Router package

"""Expose VSR routing client utilities.

The :class:`my_model.router.client.VSRClient` communicates with a vLLM
Semantic Router (VSR) instance and returns the model identifier selected by the
router.  The helper function :func:`my_model.router.client.select_backend_id`
adds a thin layer that resolves the identifier to a backend configuration
defined in the workspace.
"""

from .client import VSRClient, select_backend_id

__all__ = ["VSRClient", "select_backend_id"]
