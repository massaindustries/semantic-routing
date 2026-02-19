from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, Field, SecretStr, model_validator

class GatewayConfig(BaseModel):
    """Configuration for the local FastAPI gateway server."""
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origin: Optional[str] = "*"

class RouterConfig(BaseModel):
    """Configuration for the vLLM Semantic Router client."""
    vsr_url: str = "http://localhost:8888/v1/chat/completions"
    mode: str = "headers"
    timeout: int = 30

    @model_validator(mode="after")
    def check_mode(self):
        if self.mode not in {"headers", "json"}:
            raise ValueError("mode must be one of 'headers' or 'json'")
        return self

class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    provider_id: str
    base_url: str
    api_key: Optional[SecretStr] = None

class ModelConfig(BaseModel):
    """Representation of a backend model."""
    backend_id: str
    provider_id: str
    model_id: str
    tags: List[str] = Field(default_factory=list)

class RoutingConfig(BaseModel):
    """Optional static routing map."""
    mapping: Dict[str, str] = Field(default_factory=dict)

class WorkspaceConfig(BaseModel):
    """Root configuration persisted per virtual model alias."""
    alias: str
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    providers: List[ProviderConfig] = Field(default_factory=list)
    models: List[ModelConfig] = Field(default_factory=list)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @property
    def _workspace_dir(self) -> Path:
        return Path.home() / ".my-model" / self.alias

    @property
    def _config_file(self) -> Path:
        yaml_path = self._workspace_dir / "config.yaml"
        json_path = self._workspace_dir / "config.json"
        return yaml_path if yaml_path.is_file() else json_path

    def save(self) -> None:
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self._config_file
        data = self.model_dump(exclude_none=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError("YAML output requested but PyYAML is not installed") from exc
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, alias: str) -> "WorkspaceConfig":
        base_dir = Path.home() / ".my-model" / alias
        yaml_path = base_dir / "config.yaml"
        json_path = base_dir / "config.json"
        if yaml_path.is_file():
            try:
                import yaml
            except Exception as exc:
                raise RuntimeError("YAML config file found but PyYAML is not installed") from exc
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif json_path.is_file():
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"Config file not found for alias '{alias}' in {base_dir}")
        return cls.model_validate(data)

    def masked_dict(self) -> dict:
        raw = self.model_dump()
        for provider in raw.get("providers", []):
            if provider.get("api_key"):
                provider["api_key"] = "*****"
        return raw
