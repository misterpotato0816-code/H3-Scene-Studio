# -*- coding: utf-8 -*-
"""vLLM OpenAI-compatible server. No unload endpoint."""
from __future__ import annotations

from .base import ProviderSpec
from .openai_compat import OpenAICompatAdapter


class VLLMAdapter(OpenAICompatAdapter):
    spec = ProviderSpec(
        id="vllm", label="vLLM", kind="local",
        default_base_url="http://127.0.0.1:8000/v1", url_editable=True,
        needs_key=False, key_name="llm:vllm",
        supports_model_list=True, supports_unload=False,
        vision_detection="manual",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:8000/v1"}])
