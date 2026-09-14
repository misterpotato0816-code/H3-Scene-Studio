# -*- coding: utf-8 -*-
"""Process identity: who is actually running this app process.

Explains the class of failure seen when the app runs under a sandboxed
Windows account with a write-restricted token (POST /api/media/open ->
os.startfile -> WinError 5). Never raises; every field degrades to None on
a platform/API that does not support it.
"""
from __future__ import annotations

import getpass
import os


def process_identity() -> dict:
    result = {
        "user": "",
        "session_id": None,
        "console_session_id": None,
        "interactive": None,
        "note": "",
    }
    try:
        result["user"] = getpass.getuser()
    except Exception:                                        # noqa: BLE001
        result["user"] = ""

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            pid = kernel32.GetCurrentProcessId()
            session_id = ctypes.c_uint32(0)
            if kernel32.ProcessIdToSessionId(pid, ctypes.byref(session_id)):
                result["session_id"] = int(session_id.value)
        except Exception:                                    # noqa: BLE001
            pass
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            console_id = kernel32.WTSGetActiveConsoleSessionId()
            # 0xFFFFFFFF means "no active console session".
            if console_id not in (None, 0xFFFFFFFF):
                result["console_session_id"] = int(console_id)
        except Exception:                                    # noqa: BLE001
            pass

    session_id = result["session_id"]
    console_id = result["console_session_id"]
    if session_id not in (None, 0) and console_id is not None:
        result["interactive"] = session_id == console_id
    elif session_id in (None, 0):
        result["interactive"] = False if session_id == 0 else None

    note = ""
    user = result["user"] or ""
    if result["interactive"] is False:
        note = (
            f"H3アプリがデスクトップとは別のセッション（ユーザー: {user}）で"
            "動いている可能性があります。RUN_H3_V2.bat からH3を起動し直してください。")
    elif user.startswith("CodexSandbox"):
        note = (
            f"H3アプリがサンドボックスアカウント（{user}）で動いています。"
            "RUN_H3_V2.bat からH3を起動し直してください。")
    result["note"] = note
    return result
