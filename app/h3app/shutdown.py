# -*- coding: utf-8 -*-
"""Safe shutdown sequence: release models, stop what H3 started, clean state.

Runs the same eight-step order whether it was triggered by POST
/api/shutdown, Ctrl+C / Ctrl+Break, or aiohttp's on_cleanup: only the last
step (scheduling the app's own process exit) is skipped for the latter two,
since the process is already on its way down through another path.

Every step is try/except and recorded regardless of outcome, so one failing
step (e.g. LM Studio not reachable) never stops the rest from running. No
step here ever calls an external/network LLM provider and no step here ever
uses `taskkill /IM`: process termination is always psutil, scoped to a PID
this app itself identified.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import Any, Callable, Optional

from . import ai_settings as ai_settings_mod
from . import credstore as credstore_mod
from . import local_llm as local_llm_mod
from . import procstate

try:
    import psutil
except Exception:                                                # noqa: BLE001
    psutil = None  # type: ignore[assignment]


def _step(name: str, ok: bool, *, skipped: bool = False, message: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "skipped": bool(skipped), "message": message}


def lmstudio_target(cfg) -> Optional[dict]:
    """base_url/port/models of LM Studio, IF a role actually uses it.

    Reads schema-2 settings (ai_settings.normalize() migrates the legacy
    shape first): LM Studio is targeted only when the effective connection
    or a role override points at provider "lmstudio". Any other local
    server (Ollama, llama.cpp, ...) or external provider yields None, so
    H3 never stops a server it does not manage for this run.
    """
    try:
        settings = ai_settings_mod.normalize(cfg.get("ai_settings"))
    except Exception:                                            # noqa: BLE001
        return None
    providers = settings.get("providers") or {}
    lmstudio = providers.get("lmstudio") or {}
    base_url = str(lmstudio.get("base_url") or "").strip()
    if not base_url:
        return None
    models: list[str] = []
    used = False
    connection = settings.get("connection") or {}
    effective = [str(connection.get("provider") or "")]
    for role in (settings.get("roles") or {}).values():
        if isinstance(role, dict) and role.get("override"):
            effective.append(str(role.get("provider") or ""))
    for pid in effective:
        if pid != "lmstudio":
            continue
        used = True
    if not used:
        return None
    for source in [lmstudio.get("model")] + [
            (r.get("model") if isinstance(r, dict) else "")
            for r in (settings.get("roles") or {}).values()
            if isinstance(r, dict) and r.get("override")
            and r.get("provider") == "lmstudio"]:
        model = str(source or "").strip()
        if model and model not in models:
            models.append(model)
    from urllib.parse import urlsplit
    parts = urlsplit(base_url)
    port = parts.port
    if not port:
        return None
    return {"base_url": base_url, "port": int(port), "models": models}


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def target_is_local(target: Optional[dict]) -> bool:
    """True only when the configured LM Studio URL points at THIS machine.
    Anything else (a LAN address, a hostname) is another computer's server:
    H3 may unload the models it used there, but never runs a local CLI stop
    or terminates a local process on its behalf."""
    if not target:
        return False
    from urllib.parse import urlsplit
    try:
        host = (urlsplit(str(target.get("base_url") or "")).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def _looks_like_lmstudio(name: str, exe: str) -> bool:
    return "lm studio" in (name or "").lower() or "lm studio" in (exe or "").lower()


def record_lmstudio_target(cfg, session_id: str) -> Optional[dict]:
    """Record the LM Studio process H3 is about to use (or has just used) so
    the shutdown step can later prove it is still the same process. Only a
    loopback target is recorded (a remote server has no local process);
    the listener must be an LM Studio executable. Best-effort: None when
    nothing was recorded, and never raises."""
    try:
        target = lmstudio_target(cfg)
        if target is None or not target_is_local(target):
            return None
        listener = procstate.find_listener(int(target["port"]))
        if not listener or not listener.get("pid"):
            return None
        if not _looks_like_lmstudio(str(listener.get("name") or ""),
                                    str(listener.get("exe") or "")):
            return None
        existing = procstate.read_state().get(procstate.ROLE_LMSTUDIO)
        if isinstance(existing, dict) and existing.get("session_id") == session_id \
                and procstate.is_alive_and_matching(existing) \
                and int(existing.get("pid") or 0) == int(listener["pid"]):
            return existing
        return procstate.record_entry(
            procstate.ROLE_LMSTUDIO, int(listener["pid"]),
            session_id=session_id, port=int(target["port"]), owned_by_h3=False)
    except Exception:                                            # noqa: BLE001
        return None


class ShutdownSequence:
    def __init__(self, cfg, pipeline, client, story_pipeline=None, *,
                session_id: str = "", log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.pipeline = pipeline
        self.client = client
        self.story_pipeline = story_pipeline
        self.session_id = session_id
        self._log = log or (lambda m: None)

    # ------------------------------------------------------------ helpers --
    def _running_story_ids(self) -> list[str]:
        sp = self.story_pipeline
        if sp is None:
            return []
        try:
            return [sid for sid in list(sp.runners) if sp.is_running(sid)]
        except Exception:                                        # noqa: BLE001
            return []

    def is_busy(self) -> bool:
        try:
            if self.pipeline.is_busy:
                return True
        except Exception:                                        # noqa: BLE001
            pass
        return bool(self._running_story_ids())

    # -------------------------------------------------------------- steps --
    async def _step_interrupt(self) -> dict:
        try:
            await self.client.interrupt()
        except Exception as exc:                                 # noqa: BLE001
            self._log(f"[shutdown] interrupt failed: {exc}")
        for sid in self._running_story_ids():
            try:
                story = self.story_pipeline.store.get(sid)
                if story is not None:
                    await self.story_pipeline.request_stop(story)
            except Exception as exc:                             # noqa: BLE001
                self._log(f"[shutdown] story stop failed for {sid}: {exc}")
        return _step("interrupt", True, message="実行中の生成を中止しました")

    async def _step_release_llm(self) -> dict:
        try:
            await self.pipeline._release_local_llm()
            return _step("release_llm", True, message="ローカルLLMを解放しました")
        except Exception as exc:                                 # noqa: BLE001
            return _step("release_llm", False, message=str(exc))

    async def _step_free_comfy_models(self) -> dict:
        try:
            if not await self.client.is_reachable():
                # Nothing to free is not a failure (the launcher showed it
                # as NG when stopping an app that never launched ComfyUI).
                return _step("comfy_free_models", True, skipped=True,
                             message="ComfyUIは起動していないため解放するモデルはありません")
            result = await self.client.free_models()
            return _step("comfy_free_models", True,
                         message=f"ComfyUIのモデルを解放しました ({result})")
        except Exception as exc:                                 # noqa: BLE001
            return _step("comfy_free_models", False, message=str(exc))

    def _step_stop_comfy(self) -> dict:
        try:
            if not self.client.owned:
                return _step("stop_comfyui", True, skipped=True,
                             message="ComfyUIはH3が起動したものではないため終了しません")
            self.client.shutdown()
            procstate.remove_entry(procstate.ROLE_COMFYUI, self.session_id or None)
            return _step("stop_comfyui", True, message="H3が起動したComfyUIを終了しました")
        except Exception as exc:                                 # noqa: BLE001
            return _step("stop_comfyui", False, message=str(exc))

    async def _step_stop_lmstudio(self) -> dict:
        """Terminate LM Studio only when ALL of these hold: the setting is on,
        the configured target is on this machine, this session recorded the
        LM Studio process it connected to, and that record (pid, create
        time, exe, cmdline) still matches the live process - checked once
        before the network/CLI work and again right before terminate(). A
        port number or an executable name alone never selects a victim
        (review finding on PR #1: a LAN target could kill an unrelated local
        LM Studio on the same port)."""
        if not bool((self.cfg.get("shutdown") or {}).get("stop_lm_studio", True)):
            return _step("stop_lmstudio", True, skipped=True,
                         message="設定によりLM Studioの終了はスキップされました")
        target = lmstudio_target(self.cfg)
        if target is None:
            return _step("stop_lmstudio", True, skipped=True,
                         message="LM Studioは使用されていません")
        if not target_is_local(target):
            return _step("stop_lmstudio", True, skipped=True,
                         message=f"接続先 {target['base_url']} はこのPCではないため、"
                                 "ローカルのLM Studioは終了しません")
        entry = procstate.read_state().get(procstate.ROLE_LMSTUDIO)
        if not isinstance(entry, dict) or not entry.get("pid"):
            return _step("stop_lmstudio", True, skipped=True,
                         message="このセッションで接続したLM Studioの記録がないため終了しません")
        if entry.get("session_id") != (self.session_id or ""):
            return _step("stop_lmstudio", True, skipped=True,
                         message="記録されたLM Studioは別のH3セッションのものなので終了しません")
        if int(entry.get("port") or 0) != int(target["port"]):
            return _step("stop_lmstudio", True, skipped=True,
                         message="記録されたLM Studioのポートが接続先と一致しないため終了しません")
        pid = int(entry["pid"])
        live = procstate.identify(pid)
        if not procstate.matches(entry, live):
            return _step("stop_lmstudio", True, skipped=True,
                         message=f"PID {pid} は記録されたLM Studioと一致しません"
                                 "（終了済みか別プロセス）。終了しません")
        exe = str((live or {}).get("exe") or "")
        if not _looks_like_lmstudio("", exe):
            return _step("stop_lmstudio", True, skipped=True,
                         message=f"PID {pid} の実行ファイルはLM Studioではないため終了しません"
                                 f"（{exe or '不明'}）")
        if psutil is None:
            return _step("stop_lmstudio", False,
                         message="psutilが利用できないためプロセスを終了できません")

        token = credstore_mod.load("openai_compat_local")
        for model in target.get("models") or []:
            try:
                await local_llm_mod.release_model(target["base_url"], model, token=token)
            except Exception as exc:                             # noqa: BLE001
                self._log(f"[shutdown] lmstudio unload failed for {model}: {exc}")
        lms_path = shutil.which("lms")
        if lms_path:
            try:
                subprocess.run([lms_path, "server", "stop"], timeout=10,
                               capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception as exc:                             # noqa: BLE001
                self._log(f"[shutdown] `lms server stop` failed: {exc}")

        # Re-check right before terminating: the unload and the CLI call
        # took time, and the recorded process may have exited (or its PID
        # been reused) meanwhile.
        live = procstate.identify(pid)
        if not procstate.matches(entry, live):
            procstate.remove_entry(procstate.ROLE_LMSTUDIO, self.session_id or None)
            return _step("stop_lmstudio", True, skipped=True,
                         message=f"LM Studio (PID {pid}) は終了処理中に既に停止したか"
                                 "別プロセスになったため、終了操作は行いません")
        try:
            proc = psutil.Process(pid)
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except Exception:                                # noqa: BLE001
                    pass
            proc.terminate()
            gone, alive = psutil.wait_procs([proc, *children], timeout=5)
            for p in alive:
                try:
                    p.kill()
                except Exception:                                # noqa: BLE001
                    pass
            procstate.remove_entry(procstate.ROLE_LMSTUDIO, self.session_id or None)
            return _step("stop_lmstudio", True, message="LM Studioを終了しました")
        except Exception as exc:                                 # noqa: BLE001
            return _step("stop_lmstudio", False, message=str(exc))

    def _ports_report(self) -> list[dict]:
        state = procstate.read_state()
        ports: list[dict] = []
        seen = set()
        candidates: list[tuple[int, str]] = []
        try:
            candidates.append((int(self.cfg.app_port), procstate.ROLE_APP))
        except Exception:                                        # noqa: BLE001
            pass
        try:
            candidates.append((int(self.cfg.comfy_port), procstate.ROLE_COMFYUI))
        except Exception:                                        # noqa: BLE001
            pass
        stop_lm_studio = bool(
            (self.cfg.get("shutdown") or {}).get("stop_lm_studio", True))
        if stop_lm_studio:
            target = lmstudio_target(self.cfg)
            if target is not None:
                candidates.append(
                    (int(target["port"]), procstate.ROLE_LMSTUDIO))
        for port, role in candidates:
            if port in seen:
                continue
            seen.add(port)
            listener = procstate.find_listener(port)
            entry = state.get(role) or {}
            managed = False
            if listener and listener.get("pid") and entry:
                # Identify the live process for real: copying the recorded
                # create_time/cmdline into `live` would make matches()
                # trivially true and defeat recycled-PID protection.
                live = procstate.identify(int(listener["pid"]))
                managed = bool(entry.get("owned_by_h3")) and \
                    procstate.matches(entry, live)
            ports.append({
                "port": port,
                "free": listener is None,
                "pid": listener.get("pid") if listener else None,
                "name": listener.get("name") if listener else "",
                "managed": bool(managed),
            })
        return ports

    # --------------------------------------------------------------- run --
    async def run(self, *, interrupt: bool = False, self_exit: bool = True) -> dict:
        steps: list[dict] = []

        if self.is_busy():
            if not interrupt:
                return {"ok": False, "busy": True, "steps": [
                    _step("stop_accepting", True, skipped=True,
                          message="生成中のため受付停止は行いませんでした")],
                    "failed": [], "ports": []}
            steps.append(_step("stop_accepting", True,
                               message="新規ジョブの受付を停止しました"))
            steps.append(await self._step_interrupt())
        else:
            steps.append(_step("stop_accepting", True,
                               message="新規ジョブの受付を停止しました"))
            steps.append(_step("interrupt", True, skipped=True,
                               message="実行中のジョブはありません"))

        steps.append(await self._step_release_llm())
        steps.append(await self._step_free_comfy_models())
        steps.append(self._step_stop_comfy())
        steps.append(await self._step_stop_lmstudio())

        if self_exit:
            loop = asyncio.get_running_loop()

            def _exit_now() -> None:
                # State is already cleaned above; the HTTP response was sent
                # before this fires, so a hard exit cannot strand anything.
                os._exit(0)

            loop.call_later(0.5, _exit_now)
            steps.append(_step("schedule_exit", True, message="H3を終了します"))
        else:
            steps.append(_step("schedule_exit", True, skipped=True,
                               message="呼び出し元がプロセスの終了を担当します"))

        ports = self._ports_report()
        # Only THIS session's records: an older instance finishing its
        # cleanup after a newer one started must not erase the newer one's
        # state (that was how h3_state.json went missing).
        if procstate.clear_state(self.session_id or None):
            steps.append(_step("clear_state", True, message="状態ファイルを整理しました"))
        else:
            steps.append(_step("clear_state", False,
                               message="状態ファイルのロックを取得できなかったため、"
                                       "記録は変更していません（次回起動時に整理されます）"))

        failed = [s["name"] for s in steps if not s["ok"] and not s["skipped"]]
        return {"ok": not failed, "busy": False, "steps": steps,
               "failed": failed, "ports": ports}
