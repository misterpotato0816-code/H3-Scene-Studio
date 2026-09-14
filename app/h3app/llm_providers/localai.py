# -*- coding: utf-8 -*-
"""LocalAI OpenAI-compatible server. No unload endpoint."""
from __future__ import annotations

from .base import ProviderSpec
from .openai_compat import OpenAICompatAdapter


class LocalAIAdapter(OpenAICompatAdapter):
    spec = ProviderSpec(
        id="localai", label="LocalAI", kind="local",
        default_base_url="http://127.0.0.1:8080/v1", url_editable=True,
        needs_key=False, key_name="llm:localai",
        supports_model_list=True, supports_unload=False,
        vision_detection="manual",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                "placeholder": "http://127.0.0.1:8080/v1"}])
