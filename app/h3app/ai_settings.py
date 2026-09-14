# -*- coding: utf-8 -*-
"""Shared AI settings: which provider/model serves Director and Profile work.

Stored shape (config["ai_settings"], schema 2, secrets NEVER included):
  {"schema": 2,
   "connection": {"kind": "local"|"external", "provider": "<provider_id>"},
   "roles": {"director": {"override": bool, "provider": "", "model": ""},
             "character_profile": {"override": bool, "provider": "", "model": ""}},
   "providers": {"<provider_id>": {"base_url": "", "model": "",
                                   "vision_models": [], "extra": {}}},
   "gemma_fallback": False, "max_attempts": 3, "favorites": []}

`connection` is the default target. A role with override=false uses the
connection (model from providers[connection.provider]); override=true uses
its own provider/model. Switching providers never erases the other
providers' saved values.

Old shape (no schema, director.provider in gemma|openai_compat|opencode_go,
local_server.base_url) is auto-migrated by normalize():
gemma->comfy_gemma, openai_compat->lmstudio, opencode_go->opencode_go.

Fallback order: primary -> local Gemma (only when gemma_fallback is
explicitly enabled; it is OFF by default). A local failure never reaches
an external provider. External auth errors stop immediately (no Gemma
fallback: the key itself is wrong).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

# Legacy provider ids (old shape). Kept for migration + old callers.
PROVIDER_GEMMA = "gemma"
PROVIDER_LOCAL = "openai_compat"
PROVIDER_GO = "opencode_go"

LEGACY_TO_NEW = {
    PROVIDER_GEMMA: "comfy_gemma",
    PROVIDER_LOCAL: "lmstudio",
    PROVIDER_GO: "opencode_go",
}

SCHEMA_VERSION = 2

# Fallback kind table when the registry cannot be imported.
_LOCAL_IDS = frozenset({
    "comfy_gemma", "lmstudio", "openai_compat", "ollama", "llamacpp",
    "vllm", "localai",
})

ROLE_NAMES = ("director", "character_profile")


def _known_provider_ids() -> list[str]:
    try:
        from .llm_providers import REGISTRY
        return list(REGISTRY)
    except Exception:                                        # noqa: BLE001
        return ["comfy_gemma", "lmstudio", "openai_compat", "ollama",
                "llamacpp", "vllm", "localai", "opencode_go", "openai",
                "anthropic", "gemini", "openrouter",
                "openai_compat_external"]


def _default_base_url(provider_id: str) -> str:
    try:
        from .llm_providers import get as _get
        spec = _get(provider_id)
        if spec is not None:
            return str(spec.default_base_url or "")
    except Exception:                                        # noqa: BLE001
        pass
    return {
        "lmstudio": "http://127.0.0.1:1234/v1",
        "openai_compat": "http://127.0.0.1:8000/v1",
        "ollama": "http://127.0.0.1:11434",
        "llamacpp": "http://127.0.0.1:8080/v1",
        "vllm": "http://127.0.0.1:8000/v1",
        "localai": "http://127.0.0.1:8080/v1",
    }.get(provider_id, "")


def kind_of(provider_id: str) -> str:
    """Registry kind for a provider id ("local"|"external")."""
    try:
        from .llm_providers import get as _get
        spec = _get(provider_id)
        if spec is not None:
            return str(spec.kind or "local")
    except Exception:                                        # noqa: BLE001
        pass
    if str(provider_id) in _LOCAL_IDS:
        return "local"
    return "external"


def _default_provider_entry(provider_id: str) -> dict:
    return {"base_url": _default_base_url(provider_id), "model": "",
            "vision_models": [], "extra": {}}


def _default_providers() -> dict:
    return {pid: _default_provider_entry(pid)
            for pid in _known_provider_ids()}


DEFAULT_CONNECTION = {"kind": "local", "provider": "lmstudio"}

DEFAULT_SETTINGS = {
    "schema": SCHEMA_VERSION,
    "connection": dict(DEFAULT_CONNECTION),
    "roles": {
        "director": {"override": False, "provider": "", "model": ""},
        "character_profile": {"override": False, "provider": "",
                              "model": ""},
    },
    "providers": _default_providers(),
    "gemma_fallback": False,
    "max_attempts": 3,
    "favorites": [],
}

# Secret detection walks dict KEY NAMES only (never values), so model ids and
# URLs (which may legitimately contain the substring "token" etc.) are never
# rejected.
_FORBIDDEN_KEY_NAMES = ("key", "token", "secret", "password", "credential")


def defaults() -> dict:
    out = copy.deepcopy(DEFAULT_SETTINGS)
    out["providers"] = _default_providers()
    return out


def validate_base_url(url: object) -> tuple[str, str]:
    """Normalize + validate a LOCAL server base URL.

    Returns (normalized, error). error is "" when valid. Only loopback or
    literal private/link-local IP hosts are accepted (never a public host),
    any port, no userinfo/query/fragment. Local providers only; external
    URLs follow their own rule (see llm_providers.openai_compat_external).
    """
    import ipaddress
    from urllib.parse import urlsplit

    raw = str(url or "").strip()
    bad_format = "URLの形式が正しくありません（例: http://127.0.0.1:1234/v1）。"
    not_local = "ローカル（このPCまたはLAN内のIPアドレス）のサーバーだけ指定できます。"
    if not raw:
        return "", bad_format
    try:
        parts = urlsplit(raw)
    except Exception:                                            # noqa: BLE001
        return "", bad_format
    if parts.scheme not in ("http", "https"):
        return "", bad_format
    if not parts.hostname:
        return "", bad_format
    if parts.username or parts.password or parts.query or parts.fragment:
        return "", bad_format
    host = parts.hostname
    host_ok = host == "localhost"
    if not host_ok:
        try:
            ip = ipaddress.ip_address(host)
            host_ok = ip.is_loopback or ip.is_private or ip.is_link_local
        except ValueError:
            host_ok = False
    if not host_ok:
        return "", not_local
    path = (parts.path or "").rstrip("/")
    netloc = parts.hostname
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    if ":" in host and parts.port is not None:
        netloc = f"[{host}]:{parts.port}"
    elif ":" in host:
        netloc = f"[{host}]"
    normalized = f"{parts.scheme}://{netloc}{path}"
    return normalized, ""


def _map_provider(value: object) -> str:
    """Map a possibly-legacy provider id to a registry id ("" if unknown)."""
    text = str(value or "").strip()
    if not text:
        return ""
    if text in LEGACY_TO_NEW:
        return LEGACY_TO_NEW[text]
    if text in _known_provider_ids():
        return text
    return ""


def _normalize_new(raw: dict) -> dict:
    merged = defaults()
    known = _known_provider_ids()

    connection = raw.get("connection") if isinstance(
        raw.get("connection"), dict) else {}
    provider = _map_provider(connection.get("provider"))
    if not provider:
        provider = "lmstudio"
    merged["connection"] = {"kind": kind_of(provider),
                            "provider": provider}

    for role_name in ROLE_NAMES:
        value = raw.get("roles", {}).get(role_name) if isinstance(
            raw.get("roles"), dict) else None
        entry = {"override": False, "provider": "", "model": ""}
        if isinstance(value, dict):
            provider_id = _map_provider(value.get("provider"))
            entry["override"] = bool(value.get("override", False))
            entry["provider"] = provider_id
            entry["model"] = str(value.get("model") or "").strip()
            if entry["override"] and not provider_id:
                entry["provider"] = ""
        merged["roles"][role_name] = entry

    raw_providers = raw.get("providers") if isinstance(
        raw.get("providers"), dict) else {}
    for pid in known:
        entry = _default_provider_entry(pid)
        value = raw_providers.get(pid)
        if isinstance(value, dict):
            base_url = str(value.get("base_url") or "").strip()
            # Invalid URLs are kept as the raw string (not silently reset)
            # so the caller (save endpoint) can validate and reject; runtime
            # callers must validate again before use.
            if base_url:
                entry["base_url"] = base_url
            elif pid in raw_providers and isinstance(value, dict) and \
                    "base_url" in value:
                entry["base_url"] = ""
            entry["model"] = str(value.get("model") or "").strip()
            vision = value.get("vision_models")
            if isinstance(vision, list):
                entry["vision_models"] = [
                    str(m).strip() for m in vision[:100]
                    if str(m or "").strip()]
            extra = value.get("extra")
            if isinstance(extra, dict):
                entry["extra"] = copy.deepcopy(extra)
        merged["providers"][pid] = entry

    merged["gemma_fallback"] = bool(raw.get("gemma_fallback", False))
    try:
        merged["max_attempts"] = min(5, max(
            1, int(raw.get("max_attempts", 3))))
    except (TypeError, ValueError):
        merged["max_attempts"] = 3
    favorites = []
    if isinstance(raw.get("favorites"), list):
        for item in raw["favorites"][:50]:
            if not isinstance(item, dict):
                continue
            pid = _map_provider(item.get("provider"))
            if not pid:
                continue
            favorites.append({
                "provider": pid,
                "model": str(item.get("model") or "").strip(),
                "endpoint": str(item.get("endpoint") or "").strip()})
    merged["favorites"] = favorites
    return merged


def _migrate_old(raw: dict) -> dict:
    """Migrate the old shape (director/character_profile/local_server)."""
    merged = defaults()

    def _old_role(value: object) -> dict:
        role = {"provider": "", "model": "",
                "endpoint": "", "gemma_fallback": False}
        if isinstance(value, dict):
            legacy = str(value.get("provider") or PROVIDER_GEMMA)
            role["provider"] = LEGACY_TO_NEW.get(legacy, "comfy_gemma")
            role["model"] = str(value.get("model") or "").strip()
            role["endpoint"] = str(value.get("endpoint") or "").strip()
            role["gemma_fallback"] = bool(value.get("gemma_fallback", False))
        return role

    director = _old_role(raw.get("director"))
    profile = _old_role(raw.get("character_profile"))

    raw_local = raw.get("local_server")
    if isinstance(raw_local, dict):
        base_url = str(raw_local.get("base_url") or "").strip()
        if base_url:
            merged["providers"]["lmstudio"]["base_url"] = base_url
        vision = raw_local.get("vision_models")
        if isinstance(vision, list):
            merged["providers"]["lmstudio"]["vision_models"] = [
                str(m).strip() for m in vision[:100]
                if str(m or "").strip()]

    # Go endpoint kinds live per provider now.
    endpoint = director.get("endpoint") or profile.get("endpoint")
    if endpoint:
        merged["providers"]["opencode_go"]["extra"] = {"endpoint": endpoint}

    for pid in ("lmstudio", "opencode_go"):
        role_model = ""
        if director["provider"] == pid and director["model"]:
            role_model = director["model"]
        elif profile["provider"] == pid and profile["model"]:
            role_model = profile["model"]
        if role_model:
            merged["providers"][pid]["model"] = role_model

    same_target = (director["provider"] == profile["provider"]
                   and director["model"] == profile["model"])
    conn_provider = director["provider"] or profile["provider"] or "lmstudio"
    merged["connection"] = {"kind": kind_of(conn_provider),
                            "provider": conn_provider}
    if same_target:
        merged["roles"]["director"] = {"override": False, "provider": "",
                                       "model": ""}
        merged["roles"]["character_profile"] = {"override": False,
                                                "provider": "",
                                                "model": ""}
    else:
        merged["roles"]["director"] = {"override": False, "provider": "",
                                       "model": ""}
        merged["roles"]["character_profile"] = {
            "override": True, "provider": profile["provider"],
            "model": profile["model"]}

    merged["gemma_fallback"] = bool(
        director.get("gemma_fallback", False)
        and profile.get("gemma_fallback", False))
    try:
        merged["max_attempts"] = min(5, max(
            1, int(raw.get("max_attempts", 3))))
    except (TypeError, ValueError):
        merged["max_attempts"] = 3
    favorites = []
    if isinstance(raw.get("favorites"), list):
        for item in raw["favorites"][:50]:
            if not isinstance(item, dict):
                continue
            legacy = str(item.get("provider") or "")
            pid = LEGACY_TO_NEW.get(legacy)
            if pid is None and legacy in _known_provider_ids():
                pid = legacy
            if not pid:
                continue
            favorites.append({
                "provider": pid,
                "model": str(item.get("model") or "").strip(),
                "endpoint": str(item.get("endpoint") or "").strip()})
    merged["favorites"] = favorites
    return merged


def normalize(raw: object) -> dict:
    """Merge user data over defaults; migrate the old shape automatically."""
    if not isinstance(raw, dict):
        return defaults()
    if raw.get("schema") == SCHEMA_VERSION or "connection" in raw \
            or "roles" in raw or "providers" in raw:
        return _normalize_new(raw)
    if "director" in raw or "character_profile" in raw \
            or "local_server" in raw:
        return _migrate_old(raw)
    if not raw:
        return defaults()
    # Unknown dict without any recognized keys: treat as a new-shape
    # fragment (e.g. a partial save payload) rather than dropping it.
    return _normalize_new(raw)


def effective_selection(settings: dict, role_name: str) -> dict:
    """Resolve the effective provider target for a role.

    Returns {"kind", "provider", "model", "base_url", "extra",
    "vision_models"}. Roles with override=false (or an empty override
    provider) use the connection; otherwise the role's own provider/model.
    """
    clean = settings if isinstance(settings, dict) and \
        settings.get("schema") == SCHEMA_VERSION else normalize(settings)
    connection = clean.get("connection") or {}
    pid = str(connection.get("provider") or "lmstudio")
    if pid not in _known_provider_ids():
        pid = "lmstudio"
    model = str((clean.get("providers") or {}).get(pid, {}).get("model")
                or "").strip()
    role = (clean.get("roles") or {}).get(role_name) or {}
    if role.get("override") and _map_provider(role.get("provider")):
        pid = _map_provider(role.get("provider"))
        model = str(role.get("model") or "").strip()
    entry = (clean.get("providers") or {}).get(pid) or {}
    return {
        "kind": kind_of(pid),
        "provider": pid,
        "model": model,
        "base_url": str(entry.get("base_url") or "").strip(),
        "extra": copy.deepcopy(entry.get("extra") or {}),
        "vision_models": list(entry.get("vision_models") or []),
    }


def role_adapter(settings: dict, role_name: str, *, token: str = ""):
    """Build the adapter for a role's effective selection."""
    from .llm_providers import build_adapter
    selection = effective_selection(settings, role_name)
    return build_adapter(
        selection["provider"], base_url=selection["base_url"],
        model=selection["model"], token=token, extra=selection["extra"],
        vision_models=selection["vision_models"])


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, sub in value.items():
            yield str(key)
            yield from _walk_keys(sub)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _assert_no_secrets(settings: dict) -> None:
    """Reject only if a KEY NAME looks like a secret (never checks values,
    so model ids/URLs containing e.g. "token" as a word are never rejected)."""
    for name in _walk_keys(settings):
        low = name.lower()
        for forbidden in _FORBIDDEN_KEY_NAMES:
            if forbidden in low:
                raise ValueError("設定に秘密情報を含められません。")


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def preview_merge(current: dict | None,
                  incoming: dict | None) -> dict:
    """Normalize(current deep-merged with incoming) without writing."""
    return normalize(_deep_merge_dict(
        current if isinstance(current, dict) else {},
        incoming if isinstance(incoming, dict) else {}))


def save_to_config_file(config_path: str | Path, settings: dict,
                        *, current: dict | None = None) -> dict:
    """Persist ai_settings into config.json (atomic, secrets rejected).

    `settings` is deep-merged over `current` (or the value already on disk)
    before normalization, so a partial payload never erases the rest of the
    saved settings. Per-provider `providers[id]` entries merge per id, so
    switching providers keeps the other providers' saved values. Returns the
    normalized settings actually written.
    """
    from .config import update_config_file
    path = Path(config_path)
    if current is None:
        current = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded.get("ai_settings") or {}
            except Exception:                                    # noqa: BLE001
                current = {}
    merged_raw = _deep_merge_dict(
        current if isinstance(current, dict) else {},
        settings if isinstance(settings, dict) else {})
    clean = normalize(merged_raw)
    _assert_no_secrets(clean)
    update_config_file(path, {"ai_settings": clean})
    return clean


def _primary_kind(primary: dict) -> str:
    """Attempt kind for resolve_chain: "gemma"|"local"|"external"|"go".

    The comfy_gemma provider is ALWAYS "gemma" here, even when the caller
    pre-set kind="local" (effective_selection()/kind_of() report comfy_gemma
    as kind "local" - ProviderSpec.kind is only ever "local"|"external",
    never "gemma" - so a pre-set kind field can never be trusted to already
    say "gemma" for it). This must run before the kind-field shortcut below,
    otherwise an explicit "ComfyUI内ローカルGemma" selection would be treated
    as an unconfigured local provider (empty model -> empty chain) instead
    of the Gemma-only attempt the user actually chose.
    """
    provider = str(primary.get("provider") or "")
    if provider in (PROVIDER_GEMMA, "comfy_gemma"):
        return "gemma"
    kind = str(primary.get("kind") or "").strip()
    if kind in ("gemma", "local", "external", "go"):
        return kind
    # Legacy role dict {provider: openai_compat|opencode_go} (gemma/comfy_gemma
    # already handled above).
    if provider == PROVIDER_LOCAL:
        return "local"
    if provider == PROVIDER_GO:
        return "go"
    if provider:
        return "external"
    return "gemma"


def _primary_model(primary: dict) -> str:
    return str(primary.get("model") or "").strip()


def _primary_provider(primary: dict, kind: str) -> str:
    provider = str(primary.get("provider") or "").strip()
    if kind in ("gemma", "local", "external", "go") and provider in (
            PROVIDER_GEMMA, PROVIDER_LOCAL, PROVIDER_GO):
        return LEGACY_TO_NEW.get(provider, provider)
    if provider and provider not in (PROVIDER_GEMMA,):
        return provider
    if kind == "gemma":
        return "comfy_gemma"
    return provider


def resolve_chain(primary: dict, max_attempts: int = 3,
                  *, gemma_fallback: bool | None = None) -> list[dict]:
    """Ordered attempts: [{"kind", "provider", "model"}].

    kind is "gemma"|"local"|"external" for new-shape primaries ("go" is
    preserved verbatim for legacy opencode_go role dicts so old callers keep
    working). A local primary never gains an external attempt; an external
    primary only ever falls back to Gemma (auth aborts are handled in
    run_with_fallback, which stops before Gemma on auth errors).
    """
    if not isinstance(primary, dict):
        primary = {}
    kind = _primary_kind(primary)
    # Legacy Go fallback lists only exist on old role dicts.
    legacy_fallbacks: list[dict] = []
    if kind == "go" and isinstance(primary.get("fallbacks"), list):
        for entry in primary["fallbacks"][:5]:
            if not isinstance(entry, dict):
                continue
            if entry.get("provider") != PROVIDER_GO:
                continue  # Go models only; Zen is never a fallback
            model = str(entry.get("model") or "").strip()
            if model:
                legacy_fallbacks.append({
                    "kind": "go", "provider": PROVIDER_GO, "model": model,
                    "endpoint": str(entry.get("endpoint") or "")})
    if gemma_fallback is None:
        gemma_fallback = bool(primary.get("gemma_fallback", False))
    # Note: legacy "go" is preserved (not normalized to "external") so old
    # op callbacks matching kind == "go" keep working; new callers pass kind
    # "external" explicitly.
    provider = _primary_provider(primary, kind)
    model = _primary_model(primary)
    chain: list[dict] = []
    if kind == "gemma":
        chain.append({"kind": "gemma", "provider": "comfy_gemma",
                      "model": ""})
    elif model:
        entry: dict = {"kind": kind, "provider": provider, "model": model}
        endpoint = str(primary.get("endpoint") or "").strip()
        if endpoint:
            entry["endpoint"] = endpoint
        chain.append(entry)
        chain.extend(legacy_fallbacks)
        if gemma_fallback:
            chain.append({"kind": "gemma", "provider": "comfy_gemma",
                          "model": ""})
    # else: no model configured and primary is not "gemma" -> empty chain
    # (the caller must not silently fall back to another provider; see
    # run_with_fallback, which raises a DirectorError before calling op).
    # Dedupe, keep order, cap attempts.
    seen: set[str] = set()
    unique: list[dict] = []
    for attempt in chain:
        key = attempt["kind"] + "\x00" + attempt.get("provider", "") + \
            "\x00" + attempt.get("model", "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return unique[:max(1, min(5, int(max_attempts or 3)))]


def resolve_role_chain(settings: dict, role_name: str,
                       *, max_attempts: int | None = None) -> list[dict]:
    """Resolve the attempt chain for one role of normalized settings."""
    clean = settings if isinstance(settings, dict) and \
        settings.get("schema") == SCHEMA_VERSION else normalize(settings)
    selection = effective_selection(clean, role_name)
    attempts = max_attempts
    if attempts is None:
        try:
            attempts = int(clean.get("max_attempts", 3))
        except (TypeError, ValueError):
            attempts = 3
    return resolve_chain(
        {"kind": selection["kind"], "provider": selection["provider"],
         "model": selection["model"],
         "endpoint": str((selection["extra"] or {}).get("endpoint") or "")},
        attempts, gemma_fallback=bool(clean.get("gemma_fallback", False)))


def _load_token(loader, provider_id: str) -> str:
    if loader is None:
        return ""
    try:
        return str(loader(provider_id) or "")
    except TypeError:
        return str(loader() or "")


def _is_auth_error(exc: BaseException) -> bool:
    kind = getattr(exc, "kind", "")
    if kind == "auth":
        return True
    # GoError/DirectorError style without .kind: never guess from text.
    return False


async def run_with_fallback(primary: dict, op, *, max_attempts: int = 3,
                            gemma_fallback: bool | None = None,
                            key_loader=None, local_key_loader=None):
    """Try op(kind, model_cfg, api_key) along the chain.

    Returns (result, used) where used = {"kind","provider","model",
    "fallback": bool, "errors": [...]}. A local failure only ever falls
    through to Gemma (never to an external provider). An external auth
    error aborts immediately (not even Gemma: the key itself is wrong).
    key_loader(provider_id) supplies tokens; a zero-arg loader also works.
    local_key_loader is the legacy second loader for local providers.
    """
    try:
        from . import opencode_go as go_mod
    except Exception:                                        # noqa: BLE001
        go_mod = None
    chain = resolve_chain(primary, max_attempts,
                          gemma_fallback=gemma_fallback if
                          gemma_fallback is not None else None)
    # Legacy role dicts carry their own gemma_fallback when the caller did
    # not pass one explicitly.
    if gemma_fallback is None and isinstance(primary, dict) and \
            "gemma_fallback" in primary:
        gemma_fallback = bool(primary.get("gemma_fallback", False))
    if not chain:
        from . import director_provider as provider_mod
        raise provider_mod.DirectorError(
            "LLM接続が未設定です。設定 > LLM接続 で接続先とモデルを指定してください。",
            kind="ai_config")
    last_error: Exception | None = None
    trail: list[dict] = []
    for position, attempt in enumerate(chain):
        kind = attempt["kind"]
        if kind == "gemma":
            try:
                return await op("gemma", {}, ""), {
                    "kind": "gemma", "provider": "comfy_gemma", "model": "",
                    "fallback": position > 0, "errors": trail}
            except Exception as exc:                             # noqa: BLE001
                last_error = exc
                trail.append({"kind": "gemma", "model": "",
                              "error": type(exc).__name__ + ": " +
                              str(exc)[:160]})
                continue
        provider = str(attempt.get("provider") or "")
        loader = (local_key_loader
                  if (kind == "local" and local_key_loader is not None)
                  else key_loader)
        if kind == "local" and loader is None and local_key_loader is None \
                and key_loader is not None:
            loader = key_loader
        token = _load_token(loader, provider)
        op_kind = kind
        try:
            return await op(op_kind, attempt, token), {
                "kind": op_kind, "provider": provider,
                "model": attempt.get("model", ""),
                "fallback": position > 0, "errors": trail}
        except Exception as exc:                                 # noqa: BLE001
            last_error = exc
            trail.append({"kind": op_kind,
                          "model": attempt.get("model", ""),
                          "error": (getattr(exc, "kind", "") + ": "
                                    if getattr(exc, "kind", "") else "") +
                          str(exc)[:160]})
            if kind in ("external", "go") and _is_auth_error(exc):
                break  # another model/key pair cannot fix auth
            continue
    from . import director_provider as provider_mod
    if isinstance(last_error, provider_mod.DirectorError):
        raise last_error
    if go_mod is not None and isinstance(last_error, go_mod.GoError):
        raise provider_mod.DirectorError(str(last_error)) from last_error
    try:
        from .llm_providers import LLMError as _LLMError
        if isinstance(last_error, _LLMError):
            raise provider_mod.DirectorError(
                str(last_error)) from last_error
    except Exception:                                            # noqa: BLE001
        pass
    raise provider_mod.DirectorError(
        "AI処理に失敗しました。") from last_error
