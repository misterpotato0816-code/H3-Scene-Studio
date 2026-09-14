# -*- coding: utf-8 -*-
"""Google Gemini adapter: https://generativelanguage.googleapis.com/v1beta
/models/{model}:generateContent (x-goog-api-key, inline_data images),
GET /v1beta/models."""
from __future__ import annotations

import json as _json

from .base import BaseAdapter, LLMError, ProviderSpec

BASE = "https://generativelanguage.googleapis.com/v1beta"
USER_AGENT = "h3-video-studio/1.0"


class GeminiAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="gemini", label="Google Gemini", kind="external",
        default_base_url=BASE, url_editable=False, needs_key=True,
        key_name="llm:gemini", supports_model_list=True,
        supports_unload=False, vision_detection="manual", fields=[])

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.token, "Content-Type": "application/json",
                "User-Agent": USER_AGENT}

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
        models: list[dict] = []
        for entry in payload.get("models", []) if isinstance(payload, dict) else []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            model_id = name.split("/")[-1] if name else ""
            if model_id:
                models.append({"id": model_id,
                              "display": entry.get("displayName") or model_id,
                              "vision": None, "loaded": None})
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
        parts: list[dict] = []
        if user:
            parts.append({"text": user})
        for url in images or []:
            media, _sep, data = str(url).partition(";base64,")
            mime = media.split(":")[-1] if ":" in media else "image/jpeg"
            parts.append({"inline_data": {"mime_type": mime or "image/jpeg",
                                          "data": data or url}})
        body: dict = {"contents": [{"role": "user", "parts": parts}],
                     "generationConfig": {"maxOutputTokens": int(max_tokens or 1024)}}
        if temperature is not None:
            body["generationConfig"]["temperature"] = float(temperature)
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=15)
        started = _time.monotonic()
        url = f"{BASE}/models/{self.model}:generateContent"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self._headers(),
                                        data=_json.dumps(body),
                                        timeout=timeout) as resp:
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
            candidates = payload.get("candidates") or []
            content = (candidates[0] or {}).get("content") or {}
            texts = [str(p.get("text", "")) for p in content.get("parts", [])
                     if isinstance(p, dict) and "text" in p]
            raw_text = "\n".join(texts).strip()
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError("parse", "応答を解釈できませんでした。") from exc
        usage = payload.get("usageMetadata") or {}
        info = {"http_status": 200, "elapsed_ms": elapsed_ms, "retry_after": "",
                "usage": {"input_tokens": int(usage.get("promptTokenCount") or 0),
                          "output_tokens": int(
                              usage.get("candidatesTokenCount") or 0),
                          "reasoning_tokens": 0},
                "incomplete": "", "endpoint": "gemini"}
        return raw_text, info

    async def vision_capability(self, model: str) -> bool | None:
        if model and model in self.vision_models:
            return True
        return None

    async def release(self, models: list[str]) -> dict:
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
