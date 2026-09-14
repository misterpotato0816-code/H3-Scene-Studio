# -*- coding: utf-8 -*-
"""OpenCode Go adapter (subscription quota only - never Zen pay-as-you-go).

Hard rules enforced here, not by convention:
- Every URL must contain the Go path segment. Anything else raises
  before sending.
- No fallback to Zen pay-as-you-go exists anywhere in this module.
- Quota/rate errors fall back to another Go model or local Gemma only.
- API keys never appear in logs, errors, specs or stored JSON.

Endpoint kinds follow the official Go docs (per-model):
  "responses"        -> POST {base}/responses            (OpenAI Responses shape)
  "chat.completions" -> POST {base}/chat/completions     (OpenAI chat shape)
  "messages"         -> POST {base}/messages             (Anthropic shape)
"""
from __future__ import annotations

import json as _json

GO_BASE = "https://opencode.ai/zen/go/v1"
USER_AGENT = "h3-video-studio/1.0"
TIMEOUT_TOTAL = 300

_KINDS = ("responses", "chat.completions", "messages")


class GoError(Exception):
    """kind: auth|quota|rate|model|network|parse|endpoint."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _check_url(url: str) -> str:
    from urllib.parse import urlsplit
    parsed = urlsplit(str(url or ""))
    if (parsed.scheme != "https" or parsed.netloc != "opencode.ai"
            or parsed.path not in ("/zen/go/v1/models", "/zen/go/v1/responses",
                                   "/zen/go/v1/chat/completions", "/zen/go/v1/messages")
            or parsed.query or parsed.fragment):
        raise GoError("endpoint",
                      "Go専用endpointではありません。")
    return url


def _headers(api_key: str, session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "x-opencode-session": session_id or "h3-director",
    }


def _scrub(text: str) -> str:
    """Remove anything resembling a credential from an error body."""
    import re as _re
    text = _re.sub(r"[Bb]earer\s+[\w\-.~+/=]+", "Bearer ***", text)
    text = _re.sub(r"(api[_-]?key\s*[:=]\s*)[\w\-.~+/=]+", r"\1***",
                   text, flags=_re.I)
    return text


def _provider_message(body: str) -> tuple[str, str, str]:
    """Extract (param, type, message) from a Go structured error body."""
    try:
        payload = _json.loads(body or "")
    except Exception:                                        # noqa: BLE001
        return "", "", ""
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return "", "", ""
    return (str(error.get("param") or ""), str(error.get("type") or ""),
            _scrub(str(error.get("message") or ""))[:300])


def _redacted_error(status: int, body: str) -> GoError:
    text = (body or "")[:2000]
    low = text.lower()
    if status in (401, 403) or "unauthorized" in low or "invalid api key" in low \
            or "authentication" in low:
        return GoError("auth", "API Keyが正しくないか、認証に失敗しました。")
    if status == 429 or "rate limit" in low or "too many requests" in low:
        return GoError("rate", "リクエストが多すぎます。しばらく待って再試行してください。")
    if "quota" in low or "usage limit" in low or "limit reached" in low \
            or "insufficient" in low or "balance" in low or status == 402:
        return GoError("quota",
                       "OpenCode Goの利用枠に達しました。ローカルGemmaへ切り替えます。")
    if status in (400, 422):
        param, _etype, message = _provider_message(body)
        if "max_output_tokens" in param or "max_output_tokens" in message:
            floor = ""
            import re as _re2
            match = _re2.search(r">=\s*(\d+)", message)
            if match:
                floor = f"（{match.group(1)}以上が必要です）"
            return GoError(
                "model",
                f"max_output_tokens の値が小さすぎます{floor}。"
                "接続テストは32以上で送信してください。")
        detail = f"（{message}）" if message else ""
        return GoError(
            "model",
            f"指定モデルでリクエスト形式が正しくありません{detail}。"
            "モデル一覧を更新してください。")
    if status == 404 or "model" in low and "not found" in low:
        return GoError("model", "モデルが見つかりません。一覧を更新してください。")
    if status >= 500:
        # Reached the service and it failed on ITS side (real-machine
        # finding 2026-09-14: glm-* returned 500 "Internal server error"
        # while other models on the same key answered). Not a key or
        # connectivity problem, so do not describe it as one.
        _param, _etype, message = _provider_message(body)
        detail = f": {message}" if message else ""
        return GoError(
            "provider",
            f"OpenCode Go 側でサーバーエラーが発生しました（HTTP {status}{detail}）。"
            "API Key と接続は有効です。指定モデルまたはサービス側の一時的な障害の"
            "可能性があります。別のモデルを選ぶか、時間をおいて再試行してください。")
    return GoError("network", f"APIへ接続できませんでした（HTTP {status}）。")


def normalize_models(payload: object) -> list[dict]:
    """Turn a /models response into id/display/endpoint entries."""
    items: object = payload
    if isinstance(payload, dict):
        for key in ("models", "data", "items"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for entry in items:
        if isinstance(entry, str):
            out.append({"id": entry, "display": entry, "endpoint": ""})
            continue
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("model") or
                       entry.get("name") or "").strip()
        if not model_id:
            continue
        endpoint = ""
        for key in ("endpoint", "api", "sdk", "kind", "type"):
            value = str(entry.get(key) or "")
            for kind in _KINDS:
                if kind in value:
                    endpoint = kind
                    break
            if endpoint:
                break
        out.append({"id": model_id,
                    "display": str(entry.get("display") or entry.get("name")
                                   or model_id),
                    "endpoint": endpoint})
    return out


def _responses_body(*, model: str, system: str, user: str,
                    images: list[str], max_tokens: int,
                    temperature: float | None = 0.7) -> dict:
    content: list[dict] = []
    if user:
        content.append({"type": "input_text", "text": user})
    for image_url in images:
        content.append({"type": "input_image", "image_url": image_url,
                        "detail": "auto"})
    body: dict = {"model": model,
                  "input": [{"role": "user", "content": content}]}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if system:
        body["instructions"] = system
    if max_tokens:
        body["max_output_tokens"] = int(max_tokens)
    return body


def _responses_text(payload: dict) -> str:
    parts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(
                            block.get("text"), str):
                        parts.append(block["text"])
            elif isinstance(item.get("text"), str):
                parts.append(item["text"])
    if parts:
        return "\n".join(parts).strip()
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    return ""


def _responses_incomplete_reason(payload: dict) -> str:
    """Non-empty when the model stopped for limits (not an error)."""
    if payload.get("status") != "incomplete":
        return ""
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else ""
    if reason == "max_output_tokens":
        return ("回答が途中で切れました（出力トークン不足）。"
                "より大きいmax_output_tokensで再試行してください。")
    if reason:
        return f"回答が途中で切れました（{reason}）。"
    return "回答が途中で切れました。"


def _chat_body(*, model: str, system: str, user: str,
               images: list[str], max_tokens: int,
               temperature: float | None = 0.7) -> dict:
    content: list[dict] = [{"type": "text", "text": user}] if user else []
    for image_url in images:
        content.append({"type": "image_url",
                        "image_url": {"url": image_url}})
    messages = ([{"role": "system", "content": system}] if system else []) + \
        ([{"role": "user", "content": content}] if content else [])
    body: dict = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
    return body


def _chat_text(payload: dict) -> str:
    try:
        choices = payload.get("choices") or []
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                str(b.get("text", "")) for b in content
                if isinstance(b, dict)).strip()
    except Exception:                                        # noqa: BLE001
        pass
    return ""


def _messages_body(*, model: str, system: str, user: str,
                   images: list[str], max_tokens: int,
                   temperature: float | None = 0.7) -> dict:
    content: list[dict] = []
    for image_url in images:
        media, _, data = image_url.partition(";base64,")
        media_type = media.split(":")[-1] if ":" in media else "image/jpeg"
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": media_type or "image/jpeg",
                                   "data": data or image_url}})
    if user:
        content.append({"type": "text", "text": user})
    body: dict = {"model": model, "max_tokens": int(max_tokens or 2048),
                  "messages": [{"role": "user", "content": content}]}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if system:
        body["system"] = system
    return body


def _messages_text(payload: dict) -> str:
    parts: list[str] = []
    try:
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    except Exception:                                        # noqa: BLE001
        pass
    return "\n".join(parts).strip()


_BUILDERS = {
    "responses": (_responses_body, _responses_text),
    "chat.completions": (_chat_body, _chat_text),
    "messages": (_messages_body, _messages_text),
}


async def _post_json(url: str, *, api_key: str, session_id: str,
                     payload: dict) -> tuple[dict, dict]:
    """POST JSON. Returns (data, meta) where meta holds status/elapsed info.

    meta = {"http_status": int, "elapsed_ms": int, "retry_after": str}.
    """
    import time as _time
    import aiohttp
    _check_url(url)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_TOTAL, sock_connect=20)
    started = _time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                    url, headers=_headers(api_key, session_id),
                    data=_json.dumps(payload), allow_redirects=False) as resp:
                text = await resp.text()
                meta = {"http_status": int(resp.status),
                        "elapsed_ms": int(
                            (_time.monotonic() - started) * 1000),
                        "retry_after": str(
                            resp.headers.get("Retry-After") or "")}
                if resp.status != 200:
                    err = _redacted_error(resp.status, text)
                    err.meta = meta  # type: ignore[attr-defined]
                    raise err
                try:
                    data = _json.loads(text)
                except Exception as exc:                     # noqa: BLE001
                    raise GoError("parse",
                                  "API応答を解釈できませんでした。") from exc
                if not isinstance(data, dict):
                    raise GoError("parse", "API応答を解釈できませんでした。")
                return data, meta
    except GoError:
        raise
    except Exception as exc:                                 # noqa: BLE001
        raise GoError("network",
                      "APIへ接続できませんでした。ネットワークを確認してください。") from exc


def _extract_usage(kind: str, payload: dict) -> dict:
    """Best-effort token accounting. Missing keys stay 0 (never fail)."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    info = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    if not isinstance(usage, dict):
        return info
    try:
        if kind == "messages":
            info["input_tokens"] = int(usage.get("input_tokens") or 0)
            info["output_tokens"] = int(usage.get("output_tokens") or 0)
        else:
            info["input_tokens"] = int(
                usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            info["output_tokens"] = int(
                usage.get("output_tokens") or usage.get("completion_tokens")
                or 0)
        details = usage.get("output_tokens_details") or {}
        if isinstance(details, dict):
            info["reasoning_tokens"] = int(
                details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        pass
    return info


async def generate_text_full(*, api_key: str, model: str, endpoint: str,
                             system: str = "", user: str = "",
                             images: list[str] | None = None,
                             max_tokens: int = 2048,
                             temperature: float | None = 0.7,
                             session_id: str = "") -> tuple[str, dict]:
    """One text (+optional image) call. Returns (text, info).

    info = {"http_status", "elapsed_ms", "retry_after", "usage": {...},
            "incomplete": str, "endpoint": kind}. Secrets never included.
    temperature=None omits the parameter (minimal probe payloads).
    """
    if not api_key:
        raise GoError("auth", "API Keyが設定されていません。")
    kind = (endpoint or "responses").strip()
    if kind not in _BUILDERS:
        raise GoError("model", "モデルのendpoint種別が不明です。一覧を更新してください。")
    build, parse = _BUILDERS[kind]
    url = _check_url(f"{GO_BASE}/{kind}")
    payload, meta = await _post_json(
        url, api_key=api_key, session_id=session_id or "h3-director",
        payload=build(model=model, system=system, user=user,
                      images=list(images or [])[:4], max_tokens=max_tokens,
                      temperature=temperature))
    text = parse(payload)
    info = {"http_status": meta.get("http_status", 0),
            "elapsed_ms": meta.get("elapsed_ms", 0),
            "retry_after": meta.get("retry_after", ""),
            "usage": _extract_usage(kind, payload),
            "incomplete": "",
            "endpoint": kind}
    if not text:
        if kind == "responses":
            reason = _responses_incomplete_reason(payload)
            if reason:
                info["incomplete"] = reason
                raise GoError("model", reason)
        raise GoError("parse", "API応答が空でした。")
    return text, info


async def generate_text(*, api_key: str, model: str, endpoint: str,
                        system: str = "", user: str = "",
                        images: list[str] | None = None,
                        max_tokens: int = 2048,
                        temperature: float | None = 0.7,
                        session_id: str = "") -> str:
    """One text (+optional image) call. Returns the assistant text.

    temperature=None omits the parameter (minimal probe payloads).
    """
    text, _info = await generate_text_full(
        api_key=api_key, model=model, endpoint=endpoint, system=system,
        user=user, images=images, max_tokens=max_tokens,
        temperature=temperature, session_id=session_id)
    return text


async def fetch_models(*, api_key: str = "",
                       session_id: str = "h3-settings") -> list[dict]:
    """GET the Go model list. Public endpoint: works without a key."""
    import aiohttp
    url = _check_url(f"{GO_BASE}/models")
    timeout = aiohttp.ClientTimeout(total=60, sock_connect=15)
    headers = {"User-Agent": USER_AGENT,
               "x-opencode-session": session_id}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=False) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise _redacted_error(resp.status, text)
                try:
                    payload = _json.loads(text)
                except Exception as exc:                     # noqa: BLE001
                    raise GoError("parse",
                                  "モデル一覧を解釈できませんでした。") from exc
                models = normalize_models(payload)
                if not models:
                    raise GoError("parse",
                                  "モデル一覧を解釈できませんでした。")
                return models
    except GoError:
        raise
    except Exception as exc:                                 # noqa: BLE001
        raise GoError("network",
                      "APIへ接続できませんでした。ネットワークを確認してください。") from exc


def image_to_data_url(path, *, max_px: int = 1024) -> str:
    """Local image -> data URL for vision calls. Raises GoError on failure."""
    from pathlib import Path as _Path
    import base64 as _base64
    import io as _io
    src = _Path(str(path))
    if not src.is_file():
        raise GoError("model", "画像ファイルが見つかりません。")
    try:
        from PIL import Image as _Image
        with _Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = _io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            raw = buf.getvalue()
    except Exception:
        try:
            raw = src.read_bytes()
        except Exception as exc:                             # noqa: BLE001
            raise GoError("model", "画像を読めませんでした。") from exc
    return "data:image/jpeg;base64," + _base64.b64encode(raw).decode("ascii")
