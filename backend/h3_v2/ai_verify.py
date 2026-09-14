# -*- coding: utf-8 -*-
"""Live AI settings verification: roundtrip, restart persistence, key API,
Director regression on Gemma defaults (temp)."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP_ROOT = Path(r"E:\AI-Projects\H3\app")
VENV_PY = r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"
BASE = "http://127.0.0.1:8791"


def post(path, body, timeout=120):
    import urllib.error
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "message": f"HTTP {e.code}"}


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def api_delete(path, timeout=30):
    req = urllib.request.Request(BASE + path, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def start_server():
    import socket as _sock
    with _sock.socket() as _s:
        if _s.connect_ex(("127.0.0.1", 8791)) == 0:
            print("abort: busy")
            sys.exit(4)
    srv = subprocess.Popen(
        [VENV_PY, str(APP_ROOT / "server.py"), "--port", "8791"],
        cwd=str(APP_ROOT),
        stdout=open(APP_ROOT / "_debug" / "ai_verify.log", "wb"),
        stderr=open(APP_ROOT / "_debug" / "ai_verify.err", "wb"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.monotonic()
    while time.monotonic() - t0 < 150:
        try:
            get("/api/config", timeout=5)
            return srv
        except Exception:
            time.sleep(3)
    raise RuntimeError("server did not come up")


def stop_server(srv):
    try:
        srv.terminate()
        srv.wait(timeout=30)
    except Exception:
        try:
            srv.kill()
        except Exception:
            pass
    import socket as _sock2
    t0 = time.monotonic()
    while time.monotonic() - t0 < 90:
        with _sock2.socket() as _s:
            if _s.connect_ex(("127.0.0.1", 8791)) != 0:
                return
        time.sleep(3)


if __name__ == "__main__":
    # 1. settings roundtrip on a live server
    srv = start_server()
    try:
        g = get("/api/ai/settings")
        print(json.dumps({"ev": "settings_get", "ok": g.get("ok"),
                          "director": (g.get("settings") or {}).get(
                              "director", {}).get("provider"),
                          "key_present": g.get("key_present"),
                          "cred": g.get("cred_available")}), flush=True)
        save_body = {"settings": {
            "director": {"provider": "gemma", "model": "",
                         "endpoint": "", "fallbacks": [],
                         "gemma_fallback": True},
            "character_profile": {"provider": "gemma", "model": "",
                                  "endpoint": "", "fallbacks": [],
                                  "gemma_fallback": True},
            "favorites": [{"provider": "opencode_go",
                           "model": "muse-spark-1.3-contributor",
                           "endpoint": "responses"}],
            "max_attempts": 3}}
        s = post("/api/ai/settings", save_body)
        print(json.dumps({"ev": "settings_save", "ok": s.get("ok"),
                          "fav": len((s.get("settings") or {}).get(
                              "favorites", []))}), flush=True)
        k = post("/api/ai/key", {"key": "DUMMY-KEY-FOR-TEST"})
        print(json.dumps({"ev": "key_save", "ok": k.get("ok"),
                          "present": k.get("key_present")}), flush=True)
        g2 = get("/api/ai/settings")
        print(json.dumps({"ev": "key_present_flag",
                          "present": g2.get("key_present")}), flush=True)
        t = post("/api/ai/test-connection", {})
        print(json.dumps({"ev": "test_conn_no_real_key",
                          "ok": t.get("ok"),
                          "msg": str(t.get("message"))[:60]}), flush=True)
        api_delete("/api/ai/key")
        g3 = get("/api/ai/settings")
        print(json.dumps({"ev": "key_deleted",
                          "present": g3.get("key_present")}), flush=True)
    finally:
        stop_server(srv)
    # 2. restart persistence: fresh process reads config.json
    srv = start_server()
    try:
        g4 = get("/api/ai/settings")
        favs = (g4.get("settings") or {}).get("favorites", [])
        print(json.dumps({"ev": "restart_persist",
                          "favs": [f.get("model") for f in favs]}), flush=True)
        # restore defaults (leave no test residue)
        post("/api/ai/settings", {"settings": {
            "director": {"provider": "gemma", "model": "",
                         "endpoint": "", "fallbacks": [],
                         "gemma_fallback": True},
            "character_profile": {"provider": "gemma", "model": "",
                                  "endpoint": "", "fallbacks": [],
                                  "gemma_fallback": True},
            "favorites": [], "max_attempts": 3}})
        print(json.dumps({"ev": "settings_restored"}), flush=True)
    finally:
        stop_server(srv)
    print(json.dumps({"event": "server_stopped"}), flush=True)
