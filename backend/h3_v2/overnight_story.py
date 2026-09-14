# -*- coding: utf-8 -*-
"""Overnight headless story driver (stdlib only).

Starts the H3 app server as a child process, creates a story via HTTP,
starts it, heartbeats while polling to completion, then stops the server.
All output is printed as JSON lines for the night log.

Usage: <venv python> backend/h3_v2/overnight_story.py --mode LONG_FAST ...
"""
from __future__ import annotations

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
APP_PORT = 8790


def post(base: str, path: str, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(base: str, path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_http(base: str, deadline_s: int = 120) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        try:
            get(base, "/api/config", timeout=5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError("app server did not come up")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="LONG_FAST")
    ap.add_argument("--frames", type=int, default=362)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--tag", default="overnight")
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--timeout-min", type=int, default=45)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{APP_PORT}"
    # Freshness guard (overnight 2026-09-08 lesson): a stale app server on the
    # port would serve old code while the new process exits silently. Never
    # attach blindly - abort loudly instead.
    import socket as _sock
    with _sock.socket() as _s:
        if _s.connect_ex(("127.0.0.1", APP_PORT)) == 0:
            print(json.dumps({"event": "abort_port_busy",
                              "port": APP_PORT,
                              "hint": "a previous app server is still alive; "
                                      "retire it before starting"}), flush=True)
            return 4
    server = subprocess.Popen(
        [str(VENV_PY), str(APP_ROOT / "server.py"), "--port", str(APP_PORT)],
        cwd=str(APP_ROOT),
        stdout=open(APP_ROOT / "_debug" / "overnight_appserver.log", "wb"),  # noqa: PTH123
        stderr=open(APP_ROOT / "_debug" / "overnight_appserver.err", "wb"),  # noqa: PTH123
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(json.dumps({"event": "server_started", "pid": server.pid}), flush=True)
    try:
        wait_http(base)
        print(json.dumps({"event": "server_ready"}), flush=True)

        prompts = [
            ("A woman in a cream jacket stands on a pool deck, talking to the camera.",
             "Hello. This is a measurement take."),
            ("The same woman walks along the pool deck, continuing to talk.",
             "And this is the second take."),
        ][:args.segments]
        create_body = {
            "images": ["h3app_ref_overnight.png"],
            "mode": args.mode,
            "name": f"overnight-{args.tag}",
            "segments": [{"prompt": p, "speech": s} for p, s in prompts],
            "advanced": {"frames_mode": "manual", "frames": args.frames,
                         "width": args.width, "height": args.height},
        }
        created = post(base, "/api/story/create", create_body, timeout=90)
        if not created.get("ok"):
            print(json.dumps({"event": "create_failed", "resp": created}), flush=True)
            return 2
        sid = created["story_id"]
        print(json.dumps({"event": "story_created", "story_id": sid}), flush=True)

        started = post(base, "/api/story/start", {"story_id": sid}, timeout=90)
        print(json.dumps({"event": "story_started", "resp": started}), flush=True)

        deadline = time.monotonic() + args.timeout_min * 60
        last_beat = 0.0
        final = None
        while time.monotonic() < deadline:
            if time.monotonic() - last_beat > 25:
                try:
                    post(base, "/api/heartbeat", {}, timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                last_beat = time.monotonic()
            try:
                cur = get(base, f"/api/story/{sid}")
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "poll_error", "err": str(exc)}), flush=True)
                time.sleep(15)
                continue
            story = cur.get("story", {})
            running = cur.get("running", True)
            try:
                queue = get(base, "/queue", timeout=10)
                qinfo = {"running": len(queue.get("queue_running", [])),
                         "pending": len(queue.get("queue_pending", []))}
            except Exception:  # noqa: BLE001
                qinfo = {"running": "?", "pending": "?"}
            print(json.dumps({"event": "poll",
                              "status": story.get("status"),
                              "cursor": story.get("cursor"),
                              "running": running,
                              "queue": qinfo,
                              "segerr": [s.get("error") for s in
                                         story.get("segments", [])]}), flush=True)
            if story.get("status") in ("done", "error", "merged"):
                final = story
                break
            time.sleep(20)
        print(json.dumps({"event": "story_final",
                          "story": (final or {}).get("status"),
                          "segments": [
                              {"index": i,
                               "status": s.get("status"),
                               "timings": s.get("timings"),
                               "clip": s.get("clip")}
                              for i, s in enumerate((final or {}).get("segments", []))]}),
              flush=True)
        return 0 if (final or {}).get("status") == "done" else 3
    finally:
        try:
            server.terminate()
            server.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                server.kill()
            except Exception:  # noqa: BLE001
                pass
        print(json.dumps({"event": "server_stopped"}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
