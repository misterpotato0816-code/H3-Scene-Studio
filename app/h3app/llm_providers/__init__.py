# -*- coding: utf-8 -*-
"""Provider registry for every WP-A LLM adapter (WP-A).

All provider ids are registered here so the settings UI, server routes,
ai_settings and pipeline share one source of truth. Key-name compatibility:
lmstudio keeps "openai_compat_local", opencode_go keeps "opencode_go".
"""
from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import BaseAdapter, LLMError, ProviderSpec
from .comfy_gemma import ComfyGemmaAdapter
from .gemini import GeminiAdapter
from .llamacpp import LlamaCppAdapter
from .lmstudio import LMStudioAdapter
from .localai import LocalAIAdapter
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter
from .openai_compat import OpenAICompatAdapter
from .openai_compat_external import OpenAICompatExternalAdapter
from .opencode_go import OpenCodeGoAdapter
from .openrouter import OpenRouterAdapter
from .vllm import VLLMAdapter

_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "comfy_gemma": ComfyGemmaAdapter,
    "lmstudio": LMStudioAdapter,
    "openai_compat": OpenAICompatAdapter,
    "ollama": OllamaAdapter,
    "llamacpp": LlamaCppAdapter,
    "vllm": VLLMAdapter,
    "localai": LocalAIAdapter,
    "opencode_go": OpenCodeGoAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "openrouter": OpenRouterAdapter,
    "openai_compat_external": OpenAICompatExternalAdapter,
}

REGISTRY: dict[str, ProviderSpec] = {
    pid: cls.spec for pid, cls in _ADAPTERS.items()
}

__all__ = ["REGISTRY", "get", "for_kind", "build_adapter",
           "validate_provider_url", "key_name", "adapter_class",
           "BaseAdapter", "LLMError", "ProviderSpec"]


def get(provider_id: str) -> ProviderSpec | None:
    """Return the ProviderSpec for id, or None when unknown."""
    return REGISTRY.get(str(provider_id or ""))


def for_kind(kind: str) -> list[ProviderSpec]:
    """Specs filtered by kind ("local"|"external"), registry order."""
    return [spec for spec in REGISTRY.values() if spec.kind == kind]


def adapter_class(provider_id: str) -> type[BaseAdapter] | None:
    """Return the adapter class for id, or None when unknown."""
    return _ADAPTERS.get(str(provider_id or ""))


def key_name(provider_id: str) -> str:
    """Credential-store name for a provider ("" when it needs no key)."""
    spec = get(provider_id)
    return str(spec.key_name or "") if spec else ""


def build_adapter(provider_id: str, *, base_url: str = "",
                  model: str = "", token: str = "",
                  extra: dict | None = None,
                  vision_models: list[str] | None = None) -> BaseAdapter:
    """Instantiate the adapter for provider_id.

    Fixed-URL providers (url_editable=False) always use their spec default
    base URL. Raises ValueError for an unknown provider id.
    """
    cls = adapter_class(provider_id)
    if cls is None:
        raise ValueError(f"未知のプロバイダーです: {provider_id}")
    spec = cls.spec
    resolved_base = spec.default_base_url if not spec.url_editable \
        else (str(base_url or "").strip() or spec.default_base_url)
    return cls(base_url=resolved_base, model=str(model or ""),
               token=str(token or ""), extra=dict(extra or {}),
               vision_models=list(vision_models or []))


def validate_provider_url(provider_id: str,
                           url: object) -> tuple[str, str]:
    """Validate a base URL according to the provider kind.

    Local providers use ai_settings.validate_base_url (loopback/private
    only). The external user-URL provider uses its own http/https rule.
    Fixed-URL external providers accept only their constant ("" or equal).
    Returns (normalized, error); error is "" when valid.
    """
    spec = get(provider_id)
    if spec is None:
        return "", "未知のプロバイダーです。"
    if spec.kind == "local":
        if spec.id == "comfy_gemma":
            return "", ""
        from .. import ai_settings as _ai
        return _ai.validate_base_url(url)
    # External.
    if not spec.url_editable:
        return spec.default_base_url, ""
    from .openai_compat_external import validate_external_user_url
    return validate_external_user_url(url)
