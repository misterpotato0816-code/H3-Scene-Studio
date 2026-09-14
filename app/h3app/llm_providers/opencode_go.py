# -*- coding: utf-8 -*-
"""OpenCode Go adapter: thin wrapper around the existing opencode_go.py
transport (never moved - Go safety guards live there). Credential name kept
as "opencode_go" for backward compatibility.
"""
from __future__ import annotations

from .base import BaseAdapter, LLMError, ProviderSpec
from .. import opencode_go as _go


class OpenCodeGoAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="opencode_go", label="OpenCode Go", kind="external",
        default_base_url=_go.GO_BASE, url_editable=False, needs_key=True,
        key_name="opencode_go", supports_model_list=True,
        supports_unload=False, vision_detection="manual", fields=[])

    async def list_models(self) -> dict:
        try:
            models = await _go.fetch_models(api_key=self.token)
        except _go.GoError as exc:
            return {"models": [], "supported": True, "error": str(exc)}
        return {"models": [{"id": m["id"], "display": m.get("display") or m["id"],
                            "vision": None, "loaded": None} for m in models],
                "supported": True, "error": ""}

    async def chat(self, *, system: str = "", user: str = "",
                   images: list[str] | None = None, max_tokens: int = 1024,
                   temperature: float | None = 0.7) -> tuple[str, dict]:
        if not self.token:
            raise LLMError("auth", "API Keyが設定されていません。")
        endpoint = str(self.extra.get("endpoint") or "responses")
        session_id = str(self.extra.get("session_id") or "h3-director")
        try:
            return await _go.generate_text_full(
                api_key=self.token, model=self.model, endpoint=endpoint,
                system=system, user=user, images=images or [],
                max_tokens=max_tokens, temperature=temperature,
                session_id=session_id)
        except _go.GoError as exc:
            kind = exc.kind if exc.kind in ("auth", "provider", "model") else "network"
            raise LLMError(kind, str(exc)) from exc

    async def test(self, model: str) -> dict:
        import time as _time
        self.model = model or self.model
        started = _time.monotonic()
        try:
            # 256 tokens: reasoning models spend their first tokens thinking
            # and a 32-token probe came back "incomplete" on the real service.
            await self.chat(user="Reply with exactly: OK.", max_tokens=256,
                            temperature=None)
        except LLMError as exc:
            if "回答が途中で切れました" in str(exc):
                # The service answered (just truncated): connectivity, key
                # and model routing are all proven; only the probe was short.
                return {"ok": True, "target": _go.GO_BASE, "model": self.model,
                        "text_ok": True, "image_ok": None,
                        "latency_ms": int((_time.monotonic() - started) * 1000),
                        "error": "", "note": "応答は届きましたが途中で切れました（接続は有効）。"}
            return {"ok": False, "target": _go.GO_BASE, "model": self.model,
                    "text_ok": False, "image_ok": None,
                    "latency_ms": int((_time.monotonic() - started) * 1000),
                    "error": str(exc)}
        return {"ok": True, "target": _go.GO_BASE, "model": self.model,
                "text_ok": True, "image_ok": None,
                "latency_ms": int((_time.monotonic() - started) * 1000),
                "error": ""}

    async def vision_capability(self, model: str) -> bool | None:
        return None

    async def release(self, models: list[str]) -> dict:
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
