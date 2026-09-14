# -*- coding: utf-8 -*-
"""ComfyUI built-in Gemma (llama-cpp) "provider" placeholder.

The real generate/unload path is director_provider.GemmaDirectorProvider,
which needs the live ComfyUI client and stays untouched (see WP-A design
notes). This adapter only exists so the provider REGISTRY, the settings UI
and release() dispatch can treat "comfy_gemma" uniformly with every other
provider id; chat() is intentionally unsupported here.
"""
from __future__ import annotations

from .base import BaseAdapter, LLMError, ProviderSpec


class ComfyGemmaAdapter(BaseAdapter):
    spec = ProviderSpec(
        id="comfy_gemma", label="ComfyUI内ローカルGemma（任意・要 ComfyUI-llama-cpp）",
        kind="local",
        default_base_url="", url_editable=False, needs_key=False,
        key_name="", supports_model_list=False, supports_unload=False,
        vision_detection="native", fields=[])

    async def list_models(self) -> dict:
        return {"models": [], "supported": False, "error": ""}

    async def vision_capability(self, model: str) -> bool | None:
        return True

    async def test(self, model: str) -> dict:
        return {"ok": True, "target": "ComfyUI", "model": model or "",
                "text_ok": True, "image_ok": True, "latency_ms": 0,
                "error": ""}

    async def chat(self, **kwargs):
        raise LLMError(
            "unsupported",
            "ComfyUI内ローカルGemmaはこのAPI経由では呼び出せません。")

    async def release(self, models: list[str]) -> dict:
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
