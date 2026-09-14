# -*- coding: utf-8 -*-
"""Shared adapter contract for every LLM provider (WP-A).

Every adapter implements the same small async surface so director_provider.py
and pipeline.py can talk to any provider without provider-specific branches.
Secrets (API keys/tokens) are only ever held in-memory on the adapter
instance for the duration of one call; they are never logged, returned in
list_models()/test()/chat() results, or written to config.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """kind: auth|network|model|parse|unsupported."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class ProviderSpec:
    id: str
    label: str
    kind: str  # "local" | "external"
    default_base_url: str = ""
    url_editable: bool = False
    needs_key: bool = False
    key_name: str = ""
    supports_model_list: bool = False
    supports_unload: bool = False
    vision_detection: str = "manual"  # native|list|heuristic|manual
    fields: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "default_base_url": self.default_base_url,
            "url_editable": self.url_editable,
            "needs_key": self.needs_key,
            # The credstore entry NAME (never its value): the settings UI
            # shows the API key / token row only for providers that have one.
            "key_name": self.key_name,
            "supports_model_list": self.supports_model_list,
            "supports_unload": self.supports_unload,
            "vision_detection": self.vision_detection,
            "fields": self.fields,
        }


class BaseAdapter:
    """Common async surface. Subclasses set the class attribute `spec`."""

    spec: ProviderSpec = None  # type: ignore[assignment]

    def __init__(self, *, base_url: str = "", model: str = "", token: str = "",
                extra: dict | None = None,
                vision_models: list[str] | None = None):
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "")
        self.token = str(token or "")
        self.extra: dict[str, Any] = dict(extra or {})
        self.vision_models = list(vision_models or [])

    def target(self) -> str:
        return self.base_url or (self.spec.label if self.spec else "")

    async def list_models(self) -> dict:
        """Returns {"models":[{"id","display","vision","loaded"}],
        "supported": bool, "error": str}."""
        return {"models": [], "supported": False, "error": ""}

    async def chat(self, *, system: str = "", user: str = "",
                   images: list[str] | None = None, max_tokens: int = 1024,
                   temperature: float | None = 0.7) -> tuple[str, dict]:
        """Returns (text, info). info keys match local_llm.chat()."""
        raise LLMError("unsupported", "このプロバイダーは未対応です。")

    async def vision_capability(self, model: str) -> bool | None:
        if model and model in self.vision_models:
            return True
        return None

    async def test(self, model: str) -> dict:
        """Default: one lightweight text probe + a vision_capability check."""
        import time as _time
        self.model = model or self.model
        started = _time.monotonic()
        try:
            await self.chat(user="Reply with exactly: OK.", max_tokens=8,
                            temperature=0.0)
        except LLMError as exc:
            return {"ok": False, "target": self.target(), "model": self.model,
                    "text_ok": False, "image_ok": None,
                    "latency_ms": int((_time.monotonic() - started) * 1000),
                    "error": str(exc)}
        try:
            vision = await self.vision_capability(self.model)
        except Exception:                                        # noqa: BLE001
            vision = None
        return {"ok": True, "target": self.target(), "model": self.model,
                "text_ok": True, "image_ok": vision,
                "latency_ms": int((_time.monotonic() - started) * 1000),
                "error": ""}

    async def release(self, models: list[str]) -> dict:
        """Best-effort VRAM unload. Never raises. Providers that cannot
        unload (or are external) MUST return supported=False WITHOUT making
        any network request."""
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
