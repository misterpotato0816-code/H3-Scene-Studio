# -*- coding: utf-8 -*-
"""Live Director verification driver (stdlib only, single call).

Starts the app server, runs Director API checks that need the VLM, prints
JSON evidence lines. GPU-heavy steps are gated by --steps flag subset.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_ROOT = PROJECT_ROOT / "app"
VENV_PY = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe")
APP_PORT = 8791


def req(base, path, body=None, timeout=90, method=None):
    data = json.dumps(body or {}).encode("utf-8") if body is not None else None
    r = urllib.request.Request(base + path, data=data,
                               headers={"Content-Type": "application/json"},
                               method=method or ("POST" if body is not None else "GET"))
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_http(base, deadline_s=150):
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        try:
            get(base, "/api/config", timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("app server did not come up")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma subset: single,regen,staging,story30,sregen,single_gen,regress,st30gen")
    args = ap.parse_args()
    want = set(s for s in args.only.split(",") if s) or None
    # Stage dependencies (a generate step needs its spec first).
    if want is not None:
        if "single_gen" in want:
            want.add("single")
        if "st30gen" in want:
            want.update(("story30", "sregen"))

    def run(name):
        return want is None or name in want

    base = f"http://127.0.0.1:{APP_PORT}"
    import socket as _sock
    with _sock.socket() as _s:
        if _s.connect_ex(("127.0.0.1", APP_PORT)) == 0:
            print(json.dumps({"event": "abort_port_busy"}), flush=True)
            return 4
    server = subprocess.Popen(
        [str(VENV_PY), str(APP_ROOT / "server.py"), "--port", str(APP_PORT)],
        cwd=str(APP_ROOT),
        stdout=open(APP_ROOT / "_debug" / "director_verify.log", "wb"),
        stderr=open(APP_ROOT / "_debug" / "director_verify.err", "wb"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(json.dumps({"event": "server_started", "pid": server.pid}), flush=True)
    saved = {}
    try:
        wait_http(base)
        if run("single"):
            r = req(base, "/api/director/single/create",
                    {"idea": "夜の夏祭りで恋人を撮影", "duration_sec": 10,
                     "request": "会話多め"}, timeout=600)
            spec = r.get("spec", {})
            filled = sum(1 for v in (spec.get("items") or {}).values()
                         if isinstance(v, dict) and str(v.get("value") or "").strip())
            print(json.dumps({"event": "single_create", "ok": r.get("ok"),
                              "filled": filled,
                              "warnings": r.get("warnings", [])[:3]}), flush=True)
            saved["single"] = spec
            Path(APP_ROOT / "_debug" / "director_single.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        if run("regen") and "single" in saved:
            spec = saved["single"]
            spec["items"]["outfit"] = {"value": "浴衣", "locked": True, "request": ""}
            spec["items"]["dialogue"] = {"value": "ねえ、何食べる？",
                                         "locked": False,
                                         "request": "もっと台詞を多く。少し照れた感じ"}
            r = req(base, "/api/director/single/regenerate",
                    {"spec": spec, "scope_item": "dialogue"}, timeout=600)
            s2 = r.get("spec", {})
            print(json.dumps({"event": "single_regen", "ok": r.get("ok"),
                              "outfit": (s2.get("items") or {}).get("outfit", {}).get("value"),
                              "dialogue": ((s2.get("items") or {}).get("dialogue", {}).get("value") or "")[:80],
                              "enforced": r.get("enforced")}), flush=True)
            saved["single"] = s2
        if run("story30"):
            r = req(base, "/api/director/story30/create",
                    {"idea": "夜の夏祭りで恋人を撮影", "request": "会話多め"},
                    timeout=900)
            spec = r.get("spec", {})
            clips = spec.get("clips", [])
            per = [sum(1 for v in (c.get("items") or {}).values()
                       if isinstance(v, dict) and str(v.get("value") or "").strip())
                   for c in clips]
            print(json.dumps({"event": "story30_create", "ok": r.get("ok"),
                              "master_filled": sum(1 for v in ((spec.get("master") or {}).get("items") or {}).values()
                                                   if isinstance(v, dict) and str(v.get("value") or "").strip()),
                              "clips_filled": per,
                              "continuity": r.get("continuity", [])[:4]}), flush=True)
            saved["story30"] = spec
        if run("sregen") and "story30" in saved:
            spec = saved["story30"]
            # lock all of clip0 + outfit everywhere, regen clip2 only
            for c in spec["clips"]:
                if c.get("index") == 0:
                    for it in (c.get("items") or {}).values():
                        if isinstance(it, dict):
                            it["locked"] = True
            r = req(base, "/api/director/story30/regenerate",
                    {"spec": spec, "scope": {"clips": [2]}}, timeout=900)
            s2 = r.get("spec", {})
            c0 = [c for c in s2.get("clips", []) if c.get("index") == 0][0]
            c0vals = {k: (v.get("value") if isinstance(v, dict) else v)
                      for k, v in (c0.get("items") or {}).items()}
            old0 = [c for c in spec["clips"] if c.get("index") == 0][0]
            same = all((old0.get("items") or {}).get(k, {}).get("value") == v
                       for k, v in c0vals.items())
            print(json.dumps({"event": "story30_regen_clip2", "ok": r.get("ok"),
                              "clip0_untouched": same,
                              "enforced_n": len(r.get("enforced", []))}), flush=True)
            saved["story30"] = s2
            # item-only regen: dialogue across clips
            r2 = req(base, "/api/director/story30/regenerate",
                     {"spec": s2, "scope": {"items": ["dialogue"]}}, timeout=900)
            print(json.dumps({"event": "story30_regen_dialogue",
                              "ok": r2.get("ok")}), flush=True)
            saved["story30"] = r2.get("spec", s2)
            Path(APP_ROOT / "_debug" / "director_story30.json").write_text(
                json.dumps(saved["story30"], ensure_ascii=False, indent=2),
                encoding="utf-8")
        if run("single_gen") and "single" in saved:
            r = req(base, "/api/director/generate",
                    {"spec": saved["single"],
                     "images": ["h3app_ref_overnight.png"],
                     "mode": "FAST", "aspect": "portrait",
                     "advanced": {"frames_mode": "manual", "frames": 124}},
                    timeout=120)
            print(json.dumps({"event": "single_gen_started", "ok": r.get("ok"),
                              "project_id": r.get("project_id")}), flush=True)
            if r.get("ok"):
                pid = r["project_id"]
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1500:
                    time.sleep(20)
                    try:
                        post(base, "/api/heartbeat", {}, timeout=10)
                    except Exception:
                        pass
                    st = get(base, f"/api/project/{pid}")
                    pr = (st.get("project") or {})
                    if (pr.get("status") in ("done", "error") or
                            (pr.get("clips") or [])):
                        clips = pr.get("clips") or []
                        print(json.dumps({"event": "single_gen_done",
                                          "status": pr.get("status"),
                                          "clips": len(clips)}), flush=True)
                        break
        if run("regress"):
            # existing normal story path: 1-seg LONG 124f start→merge,
            # plus clip lock + regenerate API smoke.
            cr = req(base, "/api/story/create",
                     {"images": ["h3app_ref_overnight.png"], "mode": "LONG",
                      "name": "regress-1seg",
                      "segments": [{"prompt": "カメラに向かって話す。",
                                    "speech": "こんにちは。"}],
                      "advanced": {"frames_mode": "manual", "frames": 124,
                                   "width": 480, "height": 864}}, timeout=120)
            print(json.dumps({"event": "regress_created", "ok": cr.get("ok")}), flush=True)
            if cr.get("ok"):
                sid = cr["story_id"]
                req(base, "/api/story/start", {"story_id": sid}, timeout=120)
                t0 = time.monotonic()
                done = False
                while time.monotonic() - t0 < 1500:
                    time.sleep(25)
                    try:
                        post(base, "/api/heartbeat", {}, timeout=10)
                    except Exception:
                        pass
                    st = get(base, f"/api/story/{sid}")
                    s = (st.get("story") or {})
                    if s.get("status") in ("done", "error", "merged"):
                        done = s.get("status") in ("done", "merged")
                        break
                print(json.dumps({"event": "regress_story_done",
                                  "done": done}), flush=True)
                try:
                    lk = req(base, "/api/story/clip-lock",
                             {"story_id": sid, "index": 0, "action": "lock"},
                             timeout=60)
                    print(json.dumps({"event": "regress_lock",
                                      "ok": lk.get("ok"),
                                      "locked": lk.get("locked")}), flush=True)
                    req(base, "/api/story/clip-lock",
                        {"story_id": sid, "index": 0, "action": "unlock"},
                        timeout=60)
                except Exception as e:
                    print(json.dumps({"event": "regress_lock_error",
                                      "err": str(e)[:200]}), flush=True)
                try:
                    mg = req(base, "/api/story/merge", {"story_id": sid},
                             timeout=300)
                    print(json.dumps({"event": "regress_merge",
                                      "ok": mg.get("ok")}), flush=True)
                except Exception as e:
                    print(json.dumps({"event": "regress_merge_error",
                                      "err": str(e)[:200]}), flush=True)
        if run("st30gen") and "story30" in saved:
            r = req(base, "/api/director/story30/generate",
                    {"spec": saved["story30"],
                     "images": ["h3app_ref_overnight.png"],
                     "name": "director-30s",
                     "aspect": "portrait",
                     "advanced": {"width": 480, "height": 864}}, timeout=180)
            print(json.dumps({"event": "st30gen_started", "ok": r.get("ok"),
                              "story_id": r.get("story_id")}), flush=True)
            if r.get("ok"):
                sid = r["story_id"]
                t0 = time.monotonic()
                while time.monotonic() - t0 < 2700:
                    time.sleep(25)
                    try:
                        post(base, "/api/heartbeat", {}, timeout=10)
                    except Exception:
                        pass
                    st = get(base, f"/api/story/{sid}")
                    s = (st.get("story") or {})
                    if s.get("status") in ("done", "error", "merged"):
                        print(json.dumps({"event": "st30gen_final",
                                          "status": s.get("status")}), flush=True)
                        break
        return 0
    finally:
        try:
            server.terminate()
            server.wait(timeout=30)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass
        print(json.dumps({"event": "server_stopped"}), flush=True)


def post(base, path, body, timeout=30):
    return req(base, path, body, timeout=timeout)


if __name__ == "__main__":
    sys.exit(main())
