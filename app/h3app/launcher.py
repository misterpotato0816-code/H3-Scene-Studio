# -*- coding: utf-8 -*-
"""H3 launcher: `python -X utf8 -m h3app.launcher start|stop|status`.

A single, importable entry point (future EXE-friendly: no reliance on being
run as `__main__`, everything reachable via `main(argv)`), meant to be called
by the thin RUN_H3.bat / STOP_H3.bat wrappers at the repo root.

start   - preflight checks, then either open the browser on an already-
          running H3 (identity-matched), warn about an unrelated port
          occupant (never killed), reclaim a stale H3-owned ComfyUI from a
          previous crashed run, or launch the app detached and wait for it.
stop    - POST /api/shutdown on the running app; on failure, fall back to
          terminating only identity-matched processes recorded in the state
          file. Idempotent: nothing running -> "already stopped", exit 0.
status  - compare the state file against what is actually alive right now.

All output is Japanese (the process itself runs with -X utf8 / UTF-8 stdio);
this module is plain, synchronous, and side-effect-light so it can be unit
tested with psutil/subprocess/network entirely mocked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from . import procstate

# app/h3app/launcher.py -> parents[0]=app/h3app, [1]=app, [2]=repo root.
H3APP_DIR = Path(__file__).resolve().parent
APP_DIR = H3APP_DIR.parent
REPO_ROOT = APP_DIR.parent

READY_TIMEOUT_S = 60.0
STOP_WAIT_S = 20.0

try:
    import psutil
except Exception:                                                # noqa: BLE001
    psutil = None  # type: ignore[assignment]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _load_config():
    from . import config as config_mod
    return config_mod.load_config()


# --------------------------------------------------------------- preflight --
def preflight(cfg) -> list[str]:
    """Human-readable problems (Japanese), or an empty list when launchable."""
    problems: list[str] = []
    server_py = APP_DIR / "server.py"
    config_json = APP_DIR / "config.json"
    paths_yaml = cfg.paths_yaml
    barrier_dir = REPO_ROOT / "custom_nodes" / "H3-Device-Barrier"

    if not cfg.comfy_python.is_file():
        problems.append(
            f"ComfyUIのPythonが見つかりません: {cfg.comfy_python}\n"
            "  対処: config.json の comfy_python を確認するか、"
            f"{cfg.comfy_dir}\\.venv\\Scripts\\python.exe を用意してください。")
    if not server_py.is_file():
        problems.append(f"app/server.py が見つかりません: {server_py}\n"
                        "  対処: リポジトリを正しく取得できているか確認してください。")
    if not config_json.is_file():
        problems.append(f"app/config.json が見つかりません: {config_json}\n"
                        "  対処: app/config.example.json をコピーして作成してください。")
    if not paths_yaml.is_file():
        problems.append(f"comfy_paths.yaml が見つかりません: {paths_yaml}\n"
                        "  対処: app/comfy_paths.example.yaml をコピーして作成してください。")
    if not barrier_dir.is_dir():
        problems.append(f"H3-Device-Barrier が見つかりません: {barrier_dir}\n"
                        "  対処: custom_nodes フォルダの配置を確認してください。")
    return problems


# ------------------------------------------------------------------- ports --
def _http_get_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as resp:      # noqa: S310
            return 200 <= resp.status < 300
    except Exception:                                            # noqa: BLE001
        return False


def _describe_listener(port: int) -> Optional[dict]:
    return procstate.find_listener(port)


def _check_app_port(cfg, session_id: str) -> str:
    """Returns "free" | "own" | "foreign". Never kills anything here."""
    listener = _describe_listener(cfg.app_port)
    if listener is None:
        return "free"
    state = procstate.read_state()
    recorded = state.get(procstate.ROLE_APP)
    if recorded and listener.get("pid") == recorded.get("pid"):
        live = procstate.identify(recorded["pid"])
        if procstate.matches(recorded, live):
            return "own"
    _print(f"ポート {cfg.app_port} は既に他のプロセスが使用しています "
          f"(PID={listener.get('pid')}, 実行ファイル={listener.get('exe') or listener.get('name')})。"
          "このプロセスは終了させません。config.json の app_port を変更するか、"
          "そのプロセスを手動で終了してから RUN_H3.bat を再実行してください。")
    return "foreign"


def _check_comfy_port(cfg) -> None:
    listener = _describe_listener(cfg.comfy_port)
    if listener is None:
        return
    state = procstate.read_state()
    recorded = state.get(procstate.ROLE_COMFYUI)
    if recorded and listener.get("pid") == recorded.get("pid") and \
            recorded.get("owned_by_h3") and \
            procstate.matches(recorded, procstate.identify(recorded["pid"])):
        _print(f"ポート {cfg.comfy_port} には前回H3が起動したComfyUIが残っています "
              f"(PID={recorded.get('pid')})。回収します。")
        _reclaim_process(recorded)
        return
    _print(f"ポート {cfg.comfy_port} は既にプロセスが使用しています "
          f"(PID={listener.get('pid')}, 実行ファイル={listener.get('exe') or listener.get('name')})。"
          "H3はこのプロセスに接続を試みます（終了はしません）。")


def _reclaim_process(entry: dict) -> None:
    if psutil is None:
        return
    pid = entry.get("pid")
    if not pid:
        return
    try:
        proc = psutil.Process(int(pid))
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:                                        # noqa: BLE001
            proc.kill()
        procstate.remove_entry(procstate.ROLE_COMFYUI)
    except Exception as exc:                                     # noqa: BLE001
        _print(f"回収に失敗しました: {exc}")


# -------------------------------------------------------------------- start --
def cmd_start(argv: list[str]) -> int:
    cfg = _load_config()
    problems = preflight(cfg)
    if problems:
        _print("H3を起動できません。以下を確認してください:")
        for p in problems:
            _print(f"  - {p}")
        return 2

    session_id = procstate.new_session_id()
    status = _check_app_port(cfg, session_id)
    if status == "own":
        _print(f"H3は既に起動しています。ブラウザーを開きます: http://127.0.0.1:{cfg.app_port}/")
        webbrowser.open(f"http://127.0.0.1:{cfg.app_port}/")
        return 0
    if status == "foreign":
        return 3

    _check_comfy_port(cfg)

    log_path = procstate.RUNTIME_DIR / "app.out.log"
    procstate.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    server_py = APP_DIR / "server.py"
    python_exe = str(cfg.comfy_python)
    args = [python_exe, "-X", "utf8", str(server_py)]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) |
                         getattr(subprocess, "CREATE_NO_WINDOW", 0))
    # Unbuffered stdio: with stdout redirected to a file Python would
    # block-buffer and the log would stay empty until exit, which is useless
    # for diagnosing a start-up that hangs.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "ab") as log_file:
        proc = subprocess.Popen(
            args, cwd=str(APP_DIR), stdout=log_file, stderr=log_file,
            creationflags=creationflags, env=env)

    procstate.record_entry(procstate.ROLE_APP, proc.pid, session_id=session_id,
                           port=cfg.app_port, owned_by_h3=True)

    _print(f"H3を起動しています... (PID={proc.pid})")
    url = f"http://127.0.0.1:{cfg.app_port}/"
    deadline = time.monotonic() + READY_TIMEOUT_S
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _print(f"H3の起動に失敗しました（プロセスが終了しました, code={proc.returncode}）。"
                  f"ログを確認してください: {log_path}")
            return 4
        if _http_get_ok(url + "api/config"):
            ready = True
            break
        time.sleep(1.0)
    if not ready:
        _print(f"H3が {int(READY_TIMEOUT_S)} 秒以内に応答しませんでした。"
              f"ログを確認してください: {log_path}")
        return 5

    _print(f"H3を起動しました: {url}")
    webbrowser.open(url)
    return 0


# --------------------------------------------------------------------- stop --
def _post_shutdown(cfg, *, interrupt: bool) -> Optional[dict]:
    url = f"http://127.0.0.1:{cfg.app_port}/api/shutdown"
    body = json.dumps({"interrupt": interrupt}).encode("utf-8")
    req = urlrequest.Request(url, data=body, method="POST",
                             headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:           # noqa: S310
            return {"status": resp.status,
                   "body": json.loads(resp.read().decode("utf-8") or "{}")}
    except urlerror.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:                                        # noqa: BLE001
            payload = {}
        return {"status": exc.code, "body": payload}
    except Exception:                                            # noqa: BLE001
        return None


def _fallback_stop_from_state() -> None:
    # Only the app itself and a ComfyUI H3 launched: LM Studio termination
    # needs the exe-name check and the shutdown.stop_lm_studio setting, which
    # only ShutdownSequence applies, so it is intentionally not duplicated
    # here.
    state = procstate.read_state()
    for role in (procstate.ROLE_APP, procstate.ROLE_COMFYUI):
        entry = state.get(role)
        if not entry or not entry.get("owned_by_h3"):
            continue
        live = procstate.identify(entry.get("pid"))
        if not procstate.matches(entry, live):
            # Recorded process is gone (or a different one reuses the PID):
            # nothing to stop, and the stale record can go.
            procstate.remove_entry(role)
            continue
        if psutil is None:
            continue
        try:
            proc = psutil.Process(int(entry["pid"]))
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:                                    # noqa: BLE001
                proc.kill()
            procstate.remove_entry(role)
        except Exception as exc:                                 # noqa: BLE001
            _print(f"{role} の終了に失敗しました: {exc}")
    # Whatever is left (e.g. a still-running process this launcher could not
    # stop, or an LM Studio record) stays on file: the record is the only
    # thing that lets the next run identify it. No whole-file delete here.


def cmd_stop(argv: list[str]) -> int:
    cfg = _load_config()
    if psutil is None:
        # Without psutil no listener can be seen: claiming "not running"
        # here would be a false negative, so refuse instead of lying.
        _print("プロセスの確認に必要な psutil がこの Python にありません。"
              "ComfyUI の .venv の Python で実行してください "
              "(RUN_H3.bat / STOP_H3.bat が自動選択します)。")
        return 3
    listener = _describe_listener(cfg.app_port)
    if listener is None:
        _print("H3は起動していません（終了済み）。")
        return 0

    res = _post_shutdown(cfg, interrupt=False)
    if res is not None and res["status"] == 409:
        _print("生成中です。中止して終了しますか？ [y/N]: ")
        answer = ""
        try:
            answer = input().strip().lower()
        except Exception:                                        # noqa: BLE001
            answer = ""
        if answer != "y":
            _print("終了をキャンセルしました。")
            return 1
        res = _post_shutdown(cfg, interrupt=True)

    if res is not None and res["status"] in (200, 202):
        body = res["body"]
        for step in body.get("steps", []):
            mark = "OK" if step.get("ok") else "NG"
            if step.get("skipped"):
                mark = "--"
            _print(f"  [{mark}] {step.get('name')}: {step.get('message', '')}")
        # The app exits ~0.5 s AFTER answering; wait for the port to actually
        # free so a STOP immediately followed by RUN does not see "another
        # process" on the app port (observed 2026-09-13).
        deadline = time.monotonic() + STOP_WAIT_S
        while time.monotonic() < deadline:
            if _describe_listener(cfg.app_port) is None:
                break
            time.sleep(0.5)
        still = _describe_listener(cfg.app_port)
        if still is not None:
            _print(f"終了処理は完了しましたが、ポート {cfg.app_port} がまだ使用中です "
                  f"(PID={still.get('pid')}, 実行ファイル={still.get('exe') or still.get('name')})。"
                  "数秒待ってから再度 STOP_H3.bat を実行してください。")
            return 6
        _print("H3は終了しました。再起動は RUN_H3.bat")
        return 0

    _print("H3に応答がないため、記録済みプロセスから終了を試みます。")
    _fallback_stop_from_state()

    time.sleep(1.0)
    still = _describe_listener(cfg.app_port)
    if still is not None:
        _print(f"ポート {cfg.app_port} がまだ使用中です "
              f"(PID={still.get('pid')}, 実行ファイル={still.get('exe') or still.get('name')})。"
              "H3が管理していないプロセスの可能性があります。手動で確認してください。")
        return 6
    _print("H3は終了しました。再起動は RUN_H3.bat")
    return 0


# ------------------------------------------------------------------- status --
def cmd_status(argv: list[str]) -> int:
    cfg = _load_config()
    state = procstate.read_state()
    if not state:
        _print("記録された状態はありません。")
    for role, entry in state.items():
        live = procstate.identify(entry.get("pid"))
        alive = procstate.matches(entry, live)
        owner = "H3起動" if entry.get("owned_by_h3") else "外部"
        _print(f"{role}: PID={entry.get('pid')} port={entry.get('port')} "
              f"{owner} 生存={'はい' if alive else 'いいえ'}")
    for label, port in (("app", cfg.app_port), ("comfyui", cfg.comfy_port)):
        listener = _describe_listener(port)
        if listener is None:
            _print(f"port {port} ({label}): 空き")
        else:
            _print(f"port {port} ({label}): 使用中 "
                  f"PID={listener.get('pid')} {listener.get('name') or listener.get('exe')}")
    return 0


# --------------------------------------------------------------------- main --
def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("start", "stop", "status"):
        _print("使い方: python -X utf8 -m h3app.launcher start|stop|status")
        return 1
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "start":
        return cmd_start(rest)
    if cmd == "stop":
        return cmd_stop(rest)
    return cmd_status(rest)


if __name__ == "__main__":
    sys.exit(main())
