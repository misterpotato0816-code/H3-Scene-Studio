# -*- coding: utf-8 -*-
"""Generic OpenAI-compatible LOCAL server adapter (chat.completions, GET
/models, optional Bearer token). Used directly for the "openai_compat"
provider, and subclassed by llamacpp/vllm/localai (unload unsupported).
"""
from __future__ import annotations

import json as _json

from .base import BaseAdapter, LLMError, ProviderSpec

USER_AGENT = "h3-video-studio/1.0"


def _headers(token: str) -> dict:
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class OpenAICompatAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="openai_compat", label="OpenAI互換ローカルサーバー", kind="local",
        default_base_url="http://127.0.0.1:8000/v1", url_editable=True,
        needs_key=False, key_name="llm:openai_compat",
        supports_model_list=True, supports_unload=False,
        vision_detection="manual",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:8000/v1"}])

    async def list_models(self) -> dict:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.base_url}/models", headers=_headers(self.token),
                        timeout=timeout, allow_redirects=False) as resp:
                    text = await resp.text()
                    if resp.status in (401, 403):
                        return {"models": [], "supported": True,
                                "error": "認証が必要です。トークンを設定してください。"}
                    if resp.status != 200:
                        return {"models": [], "supported": True,
                                "error": f"モデル一覧を取得できませんでした（HTTP {resp.status}）。"}
                    payload = _json.loads(text)
        except Exception:                                        # noqa: BLE001
            return {"models": [], "supported": True,
                    "error": f"サーバーに接続できません（{self.base_url}）。"}
        data = payload.get("data") if isinstance(payload, dict) else None
        models: list[dict] = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    model_id = str(entry.get("id") or "").strip()
                elif isinstance(entry, str):
                    model_id = entry.strip()
                else:
                    model_id = ""
                if model_id:
                    models.append({"id": model_id, "display": model_id,
                                  "vision": None, "loaded": None})
        return {"models": models, "supported": True, "error": ""}

    async def chat(self, *, system: str = "", user: str = "",
                   images: list[str] | None = None, max_tokens: int = 1024,
                   temperature: float | None = 0.7) -> tuple[str, dict]:
        import aiohttp
        import time as _time
        if not self.model:
            raise LLMError("model", "モデルIDを指定してください。")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        content: list[dict] | str
        if images:
            content = []
            if user:
                content.append({"type": "text", "text": user})
            for url in images:
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content = user
        messages.append({"role": "user", "content": content})
        body = {"model": self.model, "messages": messages,
                "max_tokens": int(max_tokens)}
        if temperature is not None:
            body["temperature"] = float(temperature)
        timeout = aiohttp.ClientTimeout(total=600, sock_connect=15)
        started = _time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=_headers(self.token), data=_json.dumps(body),
                        timeout=timeout, allow_redirects=False) as resp:
                    text = await resp.text()
                    elapsed_ms = int((_time.monotonic() - started) * 1000)
                    if resp.status in (401, 403):
                        raise LLMError("auth", "認証が必要です。トークンを設定してください。")
                    if resp.status != 200:
                        raise LLMError(
                            "network",
                            f"サーバーがエラーを返しました（HTTP {resp.status}）。")
        except LLMError:
            raise
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError(
                "network", f"サーバーに接続できません（{self.base_url}）。") from exc
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
                "incomplete": "", "endpoint": self.spec.id}
        return str(raw_text or "").strip(), info

    async def vision_capability(self, model: str) -> bool | None:
        if model and model in self.vision_models:
            return True
        return None
