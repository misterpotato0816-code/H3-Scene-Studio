# -*- coding: utf-8 -*-
"""External user-URL OpenAI-compatible adapter (WP-A).

Any http/https URL (userinfo forbidden). Bearer token. Follows the external
common rule set: fixed external providers are https-only by construction
(their base URL is a constant), while this user-URL provider accepts
http/https and rejects userinfo/query/fragment. Unload is never supported
(external providers always report supported=False without touching the
network).
"""
from __future__ import annotations

from urllib.parse import urlsplit

from .base import ProviderSpec
from .openai_compat import OpenAICompatAdapter


def validate_external_user_url(url: object) -> tuple[str, str]:
    """Validate an arbitrary external OpenAI-compatible base URL."""
    raw = str(url or "").strip()
    bad = "URLの形式が正しくありません（例: https://api.example.com/v1）。"
    if not raw:
        return "", bad
    try:
        parts = urlsplit(raw)
    except Exception:                                        # noqa: BLE001
        return "", bad
    if parts.scheme not in ("http", "https"):
        return "", bad
    if not parts.hostname:
        return "", bad
    if parts.username or parts.password or parts.query or parts.fragment:
        return "", "URLに認証情報やクエリを含められません。"
    host = parts.hostname or ""
    netloc = host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    if ":" in host and not host.startswith("["):
        if parts.port is not None:
            netloc = f"[{host}]:{parts.port}"
        else:
            netloc = f"[{host}]"
    path = (parts.path or "").rstrip("/")
    return f"{parts.scheme}://{netloc}{path}", ""


class OpenAICompatExternalAdapter(OpenAICompatAdapter):
    spec = ProviderSpec(
        id="openai_compat_external", label="外部OpenAI互換",
        kind="external", default_base_url="", url_editable=True,
        needs_key=True, key_name="llm:openai_compat_external",
        supports_model_list=True, supports_unload=False,
        vision_detection="manual",
        fields=[{"name": "base_url", "label": "サーバーURL", "type": "text",
                 "placeholder": "https://api.example.com/v1"}])

    async def release(self, models: list[str]) -> dict:
        # External providers never unload: no network, always unsupported.
        return {"ok": True, "supported": False, "unloaded": [], "error": ""}
