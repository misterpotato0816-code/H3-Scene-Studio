# -*- coding: utf-8 -*-
"""Anthropic adapter: https://api.anthropic.com/v1/messages
(x-api-key, anthropic-version: 2023-06-01, images as base64 source blocks),
GET /v1/models."""
from __future__ import annotations

import json as _json

from .base import BaseAdapter, LLMError, ProviderSpec

BASE = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
USER_AGENT = "h3-video-studio/1.0"


class AnthropicAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="anthropic", label="Anthropic", kind="external",
        default_base_url=BASE, url_editable=False, needs_key=True,
        key_name="llm:anthropic", supports_model_list=True,
        supports_unload=False, vision_detection="manual", fields=[])

    def _headers(self) -> dict:
        return {"x-api-key": self.token, "anthropic-version": API_VERSION,
                "Content-Type": "application/json", "User-Agent": USER_AGENT}

    async def list_models(self) -> dict:
        import aiohttp
        if not self.token:
            return {"models": [], "supported": True,
                    "error": "API Keyが設定されていません。"}
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE}/models",
                                       headers=self._headers(),
                                       timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status in (401, 403):
                        return {"models": [], "supported": True,
                                "error": "API Keyが正しくないか認証に失敗しました。"}
                    if resp.status != 200:
                        return {"models": [], "supported": True,
                                "error": f"モデル一覧を取得できませんでした（HTTP {resp.status}）。"}
                    payload = _json.loads(text)
        except Exception:                                        # noqa: BLE001
            return {"models": [], "supported": True,
                    "error": "APIへ接続できませんでした。"}
        data = payload.get("data") if isinstance(payload, dict) else None
        models = [{"id": e.get("id"), "display": e.get("display_name") or e.get("id"),
                  "vision": None, "loaded": None}
                  for e in (data or []) if isinstance(e, dict) and e.get("id")]
        return {"models": models, "supported": True, "error": ""}

    async def chat(self, *, system: str = "", user: str = "",
                   images: list[str] | None = None, max_tokens: int = 1024,
                   temperature: float | None = 0.7) -> tuple[str, dict]:
        import aiohttp
        import time as _time
        if not self.token:
            raise LLMError("auth", "API Keyが設定されていません。")
        if not self.model:
            raise LLMError("model", "モデルIDを指定してください。")
        content: list[dict] = []
        for url in images or []:
            media, _sep, data = str(url).partition(";base64,")
            media_type = media.split(":")[-1] if ":" in media else "image/jpeg"
            content.append({"type": "image",
                            "source": {"type": "base64",
                                      "media_type": media_type or "image/jpeg",
                                      "data": data or url}})
        if user:
            content.append({"type": "text", "text": user})
        body: dict = {"model": self.model, "max_tokens": int(max_tokens or 1024),
                     "messages": [{"role": "user", "content": content}]}
        if temperature is not None:
            body["temperature"] = float(temperature)
        if system:
            body["system"] = system
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=15)
        started = _time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{BASE}/messages", headers=self._headers(),
                        data=_json.dumps(body), timeout=timeout) as resp:
                    text = await resp.text()
                    elapsed_ms = int((_time.monotonic() - started) * 1000)
                    if resp.status in (401, 403):
                        raise LLMError(
                            "auth", "API Keyが正しくないか認証に失敗しました。")
                    if resp.status != 200:
                        raise LLMError(
                            "network", f"APIがエラーを返しました（HTTP {resp.status}）。")
        except LLMError:
            raise
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError("network", "APIへ接続できませんでした。") from exc
        try:
            payload = _json.loads(text)
            parts = [str(b.get("text", "")) for b in payload.get("content", [])
                     if isinstance(b, dict) and b.get("type") == "text"]
            raw_text = "\n".join(parts).strip()
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError("parse", "応答を解釈できませんでした。") from exc
        usage = payload.get("usage") or {}
        info = {"http_status": 200, "elapsed_ms": elapsed_ms, "retry_after": "",
                "usage": {"input_tokens": int(usage.get("input_tokens") or 0),
                          "output_tokens": int(usage.get("output_tokens") or 0),
                          "reasoning_tokens": 0},
                "incomplete": "", "endpoint": "anthropic"}
        return raw_text, info

    async def vision_capability(self, model: str) -> bool | None:
        if model and model in self.vision_models:
            return True
        return None

    async def release(self, models: list[str]) -> dict:
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
