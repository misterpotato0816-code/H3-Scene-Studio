# -*- coding: utf-8 -*-
"""Ollama native API adapter: /api/tags, /api/show (capabilities), /api/chat
(images as base64), unload via POST /api/generate {"keep_alive": 0} for the
models H3 itself used (never a blanket unload of unrelated models).
"""
from __future__ import annotations

import json as _json

from .base import BaseAdapter, LLMError, ProviderSpec


class OllamaAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="ollama", label="Ollama", kind="local",
        default_base_url="http://127.0.0.1:11434", url_editable=True,
        needs_key=False, key_name="llm:ollama",
        supports_model_list=True, supports_unload=True,
        vision_detection="heuristic",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:11434"}])

    async def list_models(self) -> dict:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags",
                                       timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {"models": [], "supported": True,
                                "error": f"モデル一覧を取得できませんでした（HTTP {resp.status}）。"}
                    payload = _json.loads(text)
        except Exception:                                        # noqa: BLE001
            return {"models": [], "supported": True,
                    "error": f"サーバーに接続できません（{self.base_url}）。"}
        models: list[dict] = []
        entries = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or entry.get("model") or "").strip()
                if name:
                    models.append({"id": name, "display": name,
                                  "vision": None, "loaded": None})
        return {"models": models, "supported": True, "error": ""}

    async def _capabilities(self, model: str) -> list[str]:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base_url}/api/show",
                        data=_json.dumps({"model": model}),
                        timeout=timeout) as resp:
                    if resp.status != 200:
                        return []
                    payload = _json.loads(await resp.text())
                    caps = payload.get("capabilities") if isinstance(
                        payload, dict) else None
                    return list(caps) if isinstance(caps, list) else []
        except Exception:                                        # noqa: BLE001
            return []

    async def vision_capability(self, model: str) -> bool | None:
        if not model:
            return None
        caps = await self._capabilities(model)
        if caps:
            return "vision" in caps
        return True if model in self.vision_models else None

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
        msg: dict = {"role": "user", "content": user}
        if images:
            encoded: list[str] = []
            for url in images:
                _media, _sep, data = str(url).partition(";base64,")
                encoded.append(data or url)
            msg["images"] = encoded
        messages.append(msg)
        options: dict = {"num_predict": int(max_tokens)}
        if temperature is not None:
            options["temperature"] = float(temperature)
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": options}
        timeout = aiohttp.ClientTimeout(total=600, sock_connect=15)
        started = _time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base_url}/api/chat", data=_json.dumps(body),
                        timeout=timeout) as resp:
                    text = await resp.text()
                    elapsed_ms = int((_time.monotonic() - started) * 1000)
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
            raw_text = (payload.get("message") or {}).get("content", "")
        except Exception as exc:                                 # noqa: BLE001
            raise LLMError("parse", "応答を解釈できませんでした。") from exc
        info = {"http_status": 200, "elapsed_ms": elapsed_ms, "retry_after": "",
                "usage": {"input_tokens": int(payload.get(
                              "prompt_eval_count") or 0),
                          "output_tokens": int(
                              payload.get("eval_count") or 0),
                          "reasoning_tokens": 0},
                "incomplete": "", "endpoint": "ollama"}
        return str(raw_text or "").strip(), info

    async def release(self, models: list[str]) -> dict:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=10, sock_connect=5)
        unloaded: list[str] = []
        error = ""
        for model in models:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            f"{self.base_url}/api/generate",
                            data=_json.dumps({"model": model, "keep_alive": 0}),
                            timeout=timeout) as resp:
                        if resp.status == 200:
                            unloaded.append(model)
                        else:
                            error = f"HTTP {resp.status}"
            except Exception as exc:                             # noqa: BLE001
                error = str(exc)
        return {"ok": not error, "supported": True, "unloaded": unloaded,
                "error": error}
