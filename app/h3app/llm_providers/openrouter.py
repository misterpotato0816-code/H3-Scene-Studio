# -*- coding: utf-8 -*-
"""OpenRouter adapter: https://openrouter.ai/api/v1 chat.completions
(Bearer + optional HTTP-Referer/X-Title), GET /models (vision detected from
architecture.input_modalities)."""
from __future__ import annotations

import json as _json

from .base import BaseAdapter, LLMError, ProviderSpec

BASE = "https://openrouter.ai/api/v1"
USER_AGENT = "h3-video-studio/1.0"


class OpenRouterAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="openrouter", label="OpenRouter", kind="external",
        default_base_url=BASE, url_editable=False, needs_key=True,
        key_name="llm:openrouter", supports_model_list=True,
        supports_unload=False, vision_detection="list",
        fields=[{"name": "referer", "label": "HTTP-Referer（任意）",
                "type": "text", "placeholder": ""},
                {"name": "title", "label": "X-Title（任意）", "type": "text",
                "placeholder": ""}])

    def _headers(self) -> dict:
        headers = {"Authorization": f"Bearer {self.token}",
                  "Content-Type": "application/json", "User-Agent": USER_AGENT}
        referer = str(self.extra.get("referer") or "").strip()
        title = str(self.extra.get("title") or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    async def list_models(self) -> dict:
        import aiohttp
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
        models: list[dict] = []
        for entry in (payload.get("data") if isinstance(payload, dict) else None) or []:
            if not isinstance(entry, dict):
                continue
            model_id = str(entry.get("id") or "").strip()
            if not model_id:
                continue
            arch = entry.get("architecture") or {}
            modalities = arch.get("input_modalities") if isinstance(arch, dict) else None
            vision = "image" in modalities if isinstance(modalities, list) else None
            models.append({"id": model_id, "display": entry.get("name") or model_id,
                           "vision": vision, "loaded": None})
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
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: list[dict] = [{"type": "text", "text": user}] if user else []
            for url in images:
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})
        body = {"model": self.model, "messages": messages,
                "max_tokens": int(max_tokens)}
        if temperature is not None:
            body["temperature"] = float(temperature)
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=15)
        started = _time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{BASE}/chat/completions", headers=self._headers(),
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
            raw_text = payload["choices"][0]["message"]["content"]
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError("parse", "応答を解釈できませんでした。") from exc
        usage = payload.get("usage") or {}
        info = {"http_status": 200, "elapsed_ms": elapsed_ms, "retry_after": "",
                "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                          "output_tokens": int(
                              usage.get("completion_tokens") or 0),
                          "reasoning_tokens": 0},
                "incomplete": "", "endpoint": "openrouter"}
        return str(raw_text or "").strip(), info

    async def vision_capability(self, model: str) -> bool | None:
        result = await self.list_models()
        for entry in result.get("models", []):
            if entry.get("id") == model:
                return entry.get("vision")
        if model and model in self.vision_models:
            return True
        return None

    async def release(self, models: list[str]) -> dict:
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
