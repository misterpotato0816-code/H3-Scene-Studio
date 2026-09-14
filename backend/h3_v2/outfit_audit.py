# -*- coding: utf-8 -*-
"""Outfit audit driver: character + director outfit change + 1 generation.
Captures every stage. No code changes to the app.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP_ROOT = Path(r"E:\AI-Projects\H3\app")
VENV_PY = r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe"
BASE = "http://127.0.0.1:8791"
DBG = APP_ROOT / "_debug"


def post(path, body, timeout=180):
    import urllib.error
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return {"__http_error__": e.code,
                    "body": json.loads(e.read().decode("utf-8"))}
        except Exception:
            return {"__http_error__": e.code}


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


if __name__ == "__main__":
    import socket as _sock
    with _sock.socket() as _s:
        if _s.connect_ex(("127.0.0.1", 8791)) == 0:
            print("abort: busy")
            sys.exit(4)
    server = subprocess.Popen(
        [VENV_PY, str(APP_ROOT / "server.py"), "--port", "8791"],
        cwd=str(APP_ROOT),
        stdout=open(DBG / "outfit_audit.log", "wb"),
        stderr=open(DBG / "outfit_audit.err", "wb"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 150:
            try:
                get("/api/config", timeout=5)
                break
            except Exception:
                time.sleep(3)
        # 1. save character with a known outfit
        r = post("/api/character/save",
                 {"name": "AuditSakura",
                  "images": ["h3app_ref_overnight.png"],
                  "profile_text": "FACE: round face\nHAIR: pink bob\n"
                                  "BUILD: slim\nSKIN: fair skin\n"
                                  "OUTFIT: white dress\n"
                                  "ACCESSORIES: ribbon\n",
                  "memo": "outfit audit"})
        cid = r["character"]["character_id"]
        print(json.dumps({"ev": "char_saved", "id": cid,
                          "canon": r["character"].get("canon"),
                          "outfit_ref": r["character"].get("outfit_ref")},
                         ensure_ascii=False), flush=True)
        # 2. apply
        ap = post("/api/character/apply", {"character_id": cid})
        snap = ap["snapshot"]
        print(json.dumps({"ev": "applied", "images": ap["images"]}), flush=True)
        # 3. director single create
        cr = post("/api/director/single/create",
                  {"idea": "夜の夏祭りで恋人を撮影", "duration_sec": 10,
                   "request": ""}, timeout=600)
        spec = cr["spec"]
        print(json.dumps({"ev": "spec_created",
                          "outfit": (spec["items"]["outfit"] or {}).get("value")},
                         ensure_ascii=False), flush=True)
        # 4. change outfit via item regen
        spec["items"]["outfit"] = {
            "value": (spec["items"]["outfit"] or {}).get("value", ""),
            "locked": False, "request": "浴衣に変更。白いドレスは着ない。"}
        rg = post("/api/director/single/regenerate",
                  {"spec": spec, "scope_item": "outfit"}, timeout=600)
        spec2 = rg.get("spec", spec)
        print(json.dumps({"ev": "outfit_regen", "ok": rg.get("ok"),
                          "outfit": (spec2["items"]["outfit"] or {}).get("value"),
                          "warnings": rg.get("warnings", [])},
                         ensure_ascii=False), flush=True)
        (DBG / "outfit_final_spec.json").write_text(
            json.dumps(spec2, ensure_ascii=False, indent=2), encoding="utf-8")
        # 5. generate one video (FAST 124f)
        g = post("/api/director/generate",
                 {"spec": spec2, "images": ap["images"],
                  "mode": "FAST", "aspect": "portrait",
                  "advanced": {"frames_mode": "manual", "frames": 124,
                               "width": 480, "height": 864},
                  "character_snapshot": snap}, timeout=180)
        pid = g.get("project_id", "")
        print(json.dumps({"ev": "gen_started", "ok": g.get("ok"),
                          "pid": pid}), flush=True)
        if g.get("ok"):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1500:
                time.sleep(25)
                try:
                    post("/api/heartbeat", {}, timeout=10)
                except Exception:
                    pass
                st = get(f"/api/project/{pid}")
                pr = st.get("project", {})
                if pr.get("status") in ("done", "error") or pr.get("clips"):
                    print(json.dumps({"ev": "gen_done",
                                      "status": pr.get("status"),
                                      "clips": len(pr.get("clips") or [])}), flush=True)
                    break
        print(json.dumps({"ev": "audit_driver_done"}), flush=True)
    finally:
        try:
            server.terminate()
            server.wait(timeout=60)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass
        print(json.dumps({"event": "server_stopped"}), flush=True)
