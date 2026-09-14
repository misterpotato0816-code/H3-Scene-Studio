# -*- coding: utf-8 -*-
"""llama.cpp server: OpenAI-compatible /v1 surface. No unload endpoint."""
from __future__ import annotations

from .base import ProviderSpec
from .openai_compat import OpenAICompatAdapter


class LlamaCppAdapter(OpenAICompatAdapter):
    spec = ProviderSpec(
        id="llamacpp", label="llama.cpp server", kind="local",
        default_base_url="http://127.0.0.1:8080/v1", url_editable=True,
        needs_key=False, key_name="llm:llamacpp",
        supports_model_list=True, supports_unload=False,
        vision_detection="manual",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:8080/v1"}])
