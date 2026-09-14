# -*- coding: utf-8 -*-
"""LM Studio adapter: reuses the existing local_llm.py transport (OpenAI
compatible chat.completions + native /api/v1/models probe + native unload).

Credential name kept as "openai_compat_local" for backward compatibility
with settings/credentials saved before this provider registry existed.
"""
from __future__ import annotations

from .base import BaseAdapter, LLMError, ProviderSpec
from .. import local_llm as _local

_STATUS_TO_KIND = {
    "auth_required": "auth", "model_not_loaded": "model",
    "model_not_found": "model", "invalid_url": "network",
    "unreachable": "network", "timeout": "network",
    "vision_unsupported": "model", "bad_response": "parse",
}


class LMStudioAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="lmstudio", label="LM Studio", kind="local",
        default_base_url="http://127.0.0.1:1234/v1", url_editable=True,
        needs_key=False, key_name="openai_compat_local",
        supports_model_list=True, supports_unload=True,
        vision_detection="native",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:1234/v1"}])

    async def list_models(self) -> dict:
        try:
            result = await _local.list_models(self.base_url, token=self.token)
        except _local.LocalLLMError as exc:
            return {"models": [], "supported": True, "error": str(exc)}
        models = [{"id": m["id"], "display": m.get("display") or m["id"],
                  "vision": m.get("vision"), "loaded": m.get("loaded")}
                  for m in result.get("models", [])]
        return {"models": models, "supported": True, "error": ""}

    async def chat(self, *, system: str = "", user: str = "",
                   images: list[str] | None = None, max_tokens: int = 1024,
                   temperature: float | None = 0.7) -> tuple[str, dict]:
        if not self.model:
            raise LLMError("model", "モデルIDを指定してください。")
        try:
            return await _local.chat(
                self.base_url, model=self.model, system=system, user=user,
                images=images or [], max_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.0,
                token=self.token)
        except _local.LocalLLMError as exc:
            raise LLMError(_STATUS_TO_KIND.get(exc.status, "network"),
                          str(exc)) from exc

    async def vision_capability(self, model: str) -> bool | None:
        return await _local.vision_capability(
            self.base_url, model, vision_models=self.vision_models,
            token=self.token)

    async def release(self, models: list[str]) -> dict:
        unloaded: list[str] = []
        error = ""
        ok = True
        for model in models:
            result = await _local.release_model(
                self.base_url, model, token=self.token)
            if result.get("ok"):
                unloaded.extend(result.get("unloaded") or [])
            else:
                ok = False
                error = result.get("error") or error
        return {"ok": ok, "supported": True, "unloaded": unloaded,
                "error": error}
