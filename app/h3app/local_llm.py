# -*- coding: utf-8 -*-
"""OpenAI-compatible local LLM server adapter (e.g. LM Studio, port 1234).

Talks only to a loopback/private-IP server (see ai_settings.validate_base_url
for the boundary the caller must enforce before reaching here). Adds the
native LM Studio `/api/v1/models` probe (type/vision/loaded) on top of the
plain OpenAI `/models` list. Never sends credentials anywhere but the
configured base_url; the token (if any) is only ever an Authorization header
on requests THIS module makes.
"""
from __future__ import annotations

import json as _json
import re as _re
from urllib.parse import urlsplit, urlunsplit

USER_AGENT = "h3-video-studio/1.0"


class LocalLLMError(Exception):
    """status: unreachable|auth_required|model_not_loaded|model_not_found|
    invalid_url|vision_unsupported|timeout|bad_response|error."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _origin(base_url: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _headers(token: str = "") -> dict:
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _strip_think(text: str) -> str:
    """Remove <think>...</think> (and similar) reasoning blocks."""
    cleaned = _re.sub(r"<think>.*?</think>", "", text or "",
                      flags=_re.S | _re.I)
    cleaned = _re.sub(r"<\|?channel\|?>", " ", cleaned)
    return cleaned.strip()


async def _get_json(session, url: str, *, headers: dict, timeout):
    async with session.get(url, headers=headers, timeout=timeout,
                           allow_redirects=False) as resp:
        text = await resp.text()
        return resp.status, text


async def list_models(base_url: str, *, token: str = "") -> dict:
    """OpenAI `/models` + native LM Studio `/api/v1/models` (best effort).

    Returns {"models": [{"id","display","type","vision","loaded"}],
             "native": bool, "excluded": int}. Embeddings are excluded from
    the LLM list (excluded counts them).
    """
    import aiohttp
    base = str(base_url or "").rstrip("/")
    timeout = aiohttp.ClientTimeout(total=20, sock_connect=10)
    openai_ids: list[str] = []
    async with aiohttp.ClientSession() as session:
        try:
            status, text = await _get_json(
                session, f"{base}/models", headers=_headers(token),
                timeout=timeout)
        except Exception as exc:                                 # noqa: BLE001
            raise LocalLLMError(
                "unreachable",
                f"ローカルLLMサーバーに接続できません（{base_url}）。"
                "LM Studio等でサーバーを起動してから再テストしてください。") from exc
        if status in (401, 403):
            raise LocalLLMError("auth_required",
                                "認証が必要です。トークンを設定してください。")
        if status != 200:
            raise LocalLLMError(
                "bad_response",
                f"モデル一覧を取得できませんでした（HTTP {status}）。")
        try:
            payload = _json.loads(text)
        except Exception as exc:                                 # noqa: BLE001
            raise LocalLLMError("bad_response",
                                "モデル一覧を解釈できませんでした。") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    model_id = str(entry.get("id") or "").strip()
                elif isinstance(entry, str):
                    model_id = entry.strip()
                else:
                    model_id = ""
                if model_id:
                    openai_ids.append(model_id)

        native = False
        native_by_id: dict[str, dict] = {}
        try:
            n_status, n_text = await _get_json(
                session, f"{_origin(base)}/api/v1/models",
                headers=_headers(token), timeout=timeout)
            if n_status == 200:
                n_payload = _json.loads(n_text)
                entries = n_payload.get("models") if isinstance(
                    n_payload, dict) else None
                if isinstance(entries, list):
                    native = True
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        key = str(entry.get("key") or "").strip()
                        caps = entry.get("capabilities") or {}
                        vision = None
                        if isinstance(caps, dict) and "vision" in caps:
                            vision = bool(caps.get("vision"))
                        entry_type = entry.get("type")
                        loaded_instances = entry.get(
                            "loaded_instances") or []
                        loaded_ids = set()
                        if isinstance(loaded_instances, list):
                            for inst in loaded_instances:
                                if isinstance(inst, dict) and inst.get("id"):
                                    loaded_ids.add(str(inst["id"]))
                        is_loaded = bool(loaded_ids) if isinstance(
                            loaded_instances, list) else None
                        info = {"type": entry_type, "vision": vision,
                                "loaded": is_loaded,
                                "display": str(
                                    entry.get("display_name") or key)}
                        if key:
                            native_by_id[key] = info
                        for lid in loaded_ids:
                            native_by_id[lid] = info
        except Exception:                                        # noqa: BLE001
            native = False

    excluded = 0
    models: list[dict] = []
    seen: set[str] = set()
    for model_id in openai_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        info = native_by_id.get(model_id, {})
        if info.get("type") == "embedding":
            excluded += 1
            continue
        models.append({
            "id": model_id,
            "display": info.get("display") or model_id,
            "type": info.get("type"),
            "vision": info.get("vision"),
            "loaded": info.get("loaded"),
        })
    # Native-only entries (not reported by /models, e.g. downloaded but
    # unloaded) are still useful for the settings UI's model picker.
    for key, info in native_by_id.items():
        if key in seen or info.get("type") == "embedding":
            continue
        seen.add(key)
        models.append({"id": key, "display": info.get("display") or key,
                       "type": info.get("type"), "vision": info.get("vision"),
                       "loaded": info.get("loaded")})
    return {"models": models, "native": native, "excluded": excluded}


async def release_model(base_url: str, model: str, *, token: str = "") -> dict:
    """Best-effort unload of `model` from the local server's VRAM.

    Native LM Studio API only (the plain OpenAI-compatible surface has no
    unload verb): GET /api/v1/models to find the loaded instance id(s) for
    `model`, then POST /api/v1/models/unload {"instance_id": ...} for each.

    Never raises: this only ever runs opportunistically before a ComfyUI
    graph so an idle-but-resident local LLM does not compete for VRAM/GPU
    time with the video job, and it must never block or fail generation on
    a slow/unreachable/non-native server.

    Returns {"ok": bool, "native": bool, "unloaded": [instance_id, ...],
             "error": str}. ok is True when nothing needed unloading too.
    """
    import aiohttp
    from . import ai_settings as ai_mod

    normalized, error = ai_mod.validate_base_url(base_url)
    if error:
        return {"ok": False, "native": False, "unloaded": [], "error": error}

    origin = _origin(normalized)
    timeout = aiohttp.ClientTimeout(total=5, sock_connect=5)
    try:
        async with aiohttp.ClientSession() as session:
            try:
                status, text = await _get_json(
                    session, f"{origin}/api/v1/models",
                    headers=_headers(token), timeout=timeout)
            except Exception as exc:                             # noqa: BLE001
                return {"ok": False, "native": False, "unloaded": [],
                        "error": f"ローカルLLMサーバーに接続できません（{exc}）。"}
            if status != 200:
                return {"ok": False, "native": False, "unloaded": [],
                        "error": f"モデル一覧を取得できませんでした（HTTP {status}）。"}
            try:
                payload = _json.loads(text)
            except Exception:                                    # noqa: BLE001
                return {"ok": False, "native": False, "unloaded": [],
                        "error": "モデル一覧を解釈できませんでした。"}
            entries = payload.get("models") if isinstance(payload, dict) \
                else None
            if not isinstance(entries, list):
                return {"ok": False, "native": False, "unloaded": [],
                        "error": "ネイティブAPIが利用できません。"}

            instance_ids: list[str] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("key") or "").strip()
                entry_id = str(entry.get("id") or "").strip()
                if model not in (key, entry_id):
                    continue
                loaded_instances = entry.get("loaded_instances") or []
                if isinstance(loaded_instances, list):
                    for inst in loaded_instances:
                        if isinstance(inst, dict) and inst.get("id"):
                            instance_ids.append(str(inst["id"]))

            unloaded: list[str] = []
            for instance_id in instance_ids:
                try:
                    async with session.post(
                            f"{origin}/api/v1/models/unload",
                            headers=_headers(token),
                            data=_json.dumps({"instance_id": instance_id}),
                            timeout=timeout) as resp:
                        if resp.status == 200:
                            unloaded.append(instance_id)
                except Exception:                                # noqa: BLE001
                    continue
            return {"ok": True, "native": True, "unloaded": unloaded,
                    "error": ""}
    except Exception as exc:                                     # noqa: BLE001
        return {"ok": False, "native": False, "unloaded": [],
                "error": str(exc)}


def _classify_chat_error(status: int, body: str) -> LocalLLMError:
    low = (body or "").lower()
    if status in (401, 403):
        return LocalLLMError("auth_required",
                             "認証が必要です。トークンを設定してください。")
    if "no model" in low or "no models loaded" in low or \
            "not loaded" in low or "failed to load" in low:
        return LocalLLMError(
            "model_not_loaded",
            "モデルが読み込まれていません。LM Studioでモデルをロードするか、"
            "JIT読み込みを有効にしてください。")
    if status == 404 or "not found" in low:
        return LocalLLMError("model_not_found", "モデルが見つかりません。")
    return LocalLLMError("bad_response",
                         f"ローカルLLMがエラーを返しました（HTTP {status}）。")


async def test_connection(base_url: str, *, model: str = "",
                          token: str = "") -> dict:
    """Never raises. Returns {"ok","status","message","models_count",
    "model_state": {"loaded","vision","type"}|None}."""
    try:
        listing = await list_models(base_url, token=token)
    except LocalLLMError as exc:
        return {"ok": False, "status": exc.status, "message": str(exc),
                "models_count": 0, "model_state": None}
    models_count = len(listing["models"])
    model_state = None
    if model:
        for entry in listing["models"]:
            if entry["id"] == model:
                model_state = {"loaded": entry.get("loaded"),
                               "vision": entry.get("vision"),
                               "type": entry.get("type")}
                break
    if not model:
        return {"ok": True, "status": "connected",
                "message": f"接続できました（モデル {models_count} 件）。",
                "models_count": models_count, "model_state": model_state}

    import aiohttp
    body = {"model": model,
            "messages": [{"role": "user",
                         "content": "Reply with exactly: OK."}],
            "max_tokens": 16, "temperature": 0}
    timeout = aiohttp.ClientTimeout(total=120, sock_connect=15)
    base = str(base_url or "").rstrip("/")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{base}/chat/completions", headers=_headers(token),
                    data=_json.dumps(body), timeout=timeout,
                    allow_redirects=False) as resp:
                text = await resp.text()
                if resp.status != 200:
                    err = _classify_chat_error(resp.status, text)
                    return {"ok": False, "status": err.status,
                            "message": str(err), "models_count": models_count,
                            "model_state": model_state}
    except _TimeoutErrors():
        return {"ok": False, "status": "timeout",
                "message": "応答がタイムアウトしました。",
                "models_count": models_count, "model_state": model_state}
    except Exception as exc:                                     # noqa: BLE001
        return {"ok": False, "status": "unreachable",
                "message": f"ローカルLLMサーバーに接続できません（{base_url}）。"
                          "LM Studio等でサーバーを起動してから再テストしてください。",
                "models_count": models_count, "model_state": model_state}
    return {"ok": True, "status": "connected",
            "message": f"モデル「{model}」に接続できました。",
            "models_count": models_count, "model_state": model_state}


def _TimeoutErrors():
    import asyncio
    import aiohttp
    return (asyncio.TimeoutError, aiohttp.ServerTimeoutError)


async def chat(base_url: str, *, model: str, system: str = "",
               user: str = "", images: list[str] | None = None,
               max_tokens: int = 1024, temperature: float = 0.7,
               token: str = "", timeout: int = 600) -> tuple[str, dict]:
    """One OpenAI-compatible chat call. Returns (text, info).

    images are data URLs (already prepared by the caller); never fetched by
    this module. Reasoning blocks (e.g. <think>...</think>) are stripped.
    """
    import aiohttp
    import time as _time
    if not model:
        raise LocalLLMError("error", "モデルIDを指定してください。")
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
    body = {"model": model, "messages": messages,
            "max_tokens": int(max_tokens), "temperature": float(temperature)}
    base = str(base_url or "").rstrip("/")
    started = _time.monotonic()
    to = aiohttp.ClientTimeout(total=timeout, sock_connect=15)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{base}/chat/completions", headers=_headers(token),
                    data=_json.dumps(body), timeout=to,
                    allow_redirects=False) as resp:
                text = await resp.text()
                elapsed_ms = int((_time.monotonic() - started) * 1000)
                if resp.status != 200:
                    raise _classify_chat_error(resp.status, text)
    except LocalLLMError:
        raise
    except _TimeoutErrors() as exc:
        raise LocalLLMError("timeout", "応答がタイムアウトしました。") from exc
    except Exception as exc:                                     # noqa: BLE001
        raise LocalLLMError(
            "unreachable",
            f"ローカルLLMサーバーに接続できません（{base_url}）。"
            "LM Studio等でサーバーを起動してから再テストしてください。") from exc
    try:
        payload = _json.loads(text)
        raw_text = payload["choices"][0]["message"]["content"]
    except Exception as exc:                                     # noqa: BLE001
        raise LocalLLMError("bad_response",
                            "応答を解釈できませんでした。") from exc
    usage = payload.get("usage") or {}
    info = {"http_status": 200, "elapsed_ms": elapsed_ms, "retry_after": "",
            "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                      "output_tokens": int(
                          usage.get("completion_tokens") or 0),
                      "reasoning_tokens": 0},
            "incomplete": "", "endpoint": "local"}
    return _strip_think(str(raw_text or "")), info


async def vision_capability(base_url: str, model: str, *,
                            vision_models: list[str] | None = None,
                            token: str = "") -> bool | None:
    """True/False from native capability info, else True only when the user
    explicitly marked the model, else None (unknown -> treated as False)."""
    try:
        listing = await list_models(base_url, token=token)
    except LocalLLMError:
        listing = {"models": []}
    for entry in listing.get("models", []):
        if entry["id"] == model and entry.get("vision") is not None:
            return bool(entry["vision"])
    if model in (vision_models or []):
        return True
    return None
