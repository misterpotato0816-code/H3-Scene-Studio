# -*- coding: utf-8 -*-
"""Windows Credential Manager storage for H3 secrets (no new dependency).

Uses pywin32 (already in the production venv) directly. API keys never touch
JSON, logs, specs or exports. All functions never raise and never log values.
"""
from __future__ import annotations

TARGET_PREFIX = "H3VideoStudio/"


def _target(name: str) -> str:
    safe = "".join(c for c in str(name or "") if c.isalnum() or c in "-_")
    return f"{TARGET_PREFIX}{safe or 'default'}"


def available() -> bool:
    try:
        import win32cred  # noqa: F401,PLC0415
        return True
    except Exception:                                        # noqa: BLE001
        return False


def save(name: str, secret: str) -> bool:
    try:
        import win32cred
        credential = {
            "Type": win32cred.CRED_TYPE_GENERIC,
            "TargetName": _target(name),
            "CredentialBlob": str(secret or ""),
            "Persist": win32cred.CRED_PERSIST_LOCAL_MACHINE,
        }
        win32cred.CredWrite(credential, 0)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def load(name: str) -> str:
    try:
        import win32cred
        cred = win32cred.CredRead(_target(name),
                                  win32cred.CRED_TYPE_GENERIC, 0)
        blob = cred.get("CredentialBlob", "")
        if isinstance(blob, bytes):
            blob = blob.decode("utf-16-le", errors="replace").rstrip("\x00")
        return str(blob or "")
    except Exception:                                        # noqa: BLE001
        return ""


def delete(name: str) -> bool:
    try:
        import win32cred
        win32cred.CredDelete(_target(name),
                             win32cred.CRED_TYPE_GENERIC, 0)
        return True
    except Exception:                                        # noqa: BLE001
        return False


def has(name: str) -> bool:
    return bool(load(name))
