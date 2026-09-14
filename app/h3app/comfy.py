# -*- coding: utf-8 -*-
"""ComfyUI process driver: launch-or-attach, gate, submit, follow, cancel.

Driving pattern (HTTP + websocket, /object_info gate, stderr marker scan) is the
same one _hetero_test/opt_ref_match/run_opt_ref_match.py uses, because that is the
version that actually completed a 217.61 s run.
"""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import aiohttp

from . import comfy_inputs
from . import gpu as gpu_mod
from .errors import PipelineError, UserCancelled, pack_for

# Markers that mean "the server is dying", scanned in the stderr tail.
ERROR_PATTERNS = (
    "!!! Exception during processing !!!",
    "Exception in thread Thread-1 (prompt_worker)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "Windows fatal exception",
    "GGML_ASSERT",
)

READY_TIMEOUT_SECONDS = 360      # cold start loads a lot of node packs
STALL_SECONDS = 300


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class ComfyClient:
    def __init__(self, config, log: Callable[[str], None] | None = None):
        self.config = config
        self.port = config.comfy_port
        self.base = f"http://127.0.0.1:{self.port}"
        self.ws_url = f"ws://127.0.0.1:{self.port}/ws"
        self.process: subprocess.Popen | None = None
        self.attached = False
        # WP-D: the owning H3 session id (set by create_app) used when the
        # launched server is recorded in / removed from procstate.
        self.session_id: str = ""
        self.object_info: dict | None = None
        # Launch profile of the currently running server ("base" | "fast" | None).
        # The fast profile appends comfy_launch_args_fast (default: ["--fast"]).
        # Parity: --fast changes LEGACY output bytes, so LEGACY-family jobs must
        # never run on the fast profile (see modes_v2.launch_profile).
        self._profile: str | None = None
        self._log = log or (lambda m: print(m, flush=True))
        self._err_path = config.debug_dir / "comfy.err"
        self._out_path = config.debug_dir / "comfy.log"
        self._start_lock = asyncio.Lock()
        # GPU allocation of the server WE launched (None if attached/not launched).
        self.launched_gpu_signature: tuple | None = None
        self.gpu_plan: dict | None = None
        # (attached, signature) pairs already verified against /system_stats,
        # so a hot loop of jobs against a foreign server does not re-fetch
        # /system_stats before every submit.
        self._verified_gpu_signatures: set[tuple] = set()
        # Delivers h3app_* staged files to ComfyUI via /upload/image right
        # before submit; the app process itself never writes into ComfyUI's
        # own input directory.
        self.inputs = comfy_inputs.InputUploader(config, self.base)

    # ------------------------------------------------------------- lifecycle --
    def launch_args_for(self, profile: str = "base") -> list[str]:
        """Full ComfyUI argv tail for a profile. Pure function, no side effects."""
        cfg = self.config
        plan = gpu_mod.current_plan(cfg)
        if not plan.get("ok"):
            raise PipelineError(
                "GPU設定に問題があるため生成エンジンを起動できません: "
                + "; ".join(plan.get("errors") or []),
                kind="gpu_config")
        args = [str(cfg.comfy_python), "-X", "utf8", str(cfg.comfy_main),
                "--port", str(self.port)]
        args += gpu_mod.strip_device_flags(cfg.get("comfy_launch_args", []))
        args += [str(a) for a in plan["launch_args"]]
        if profile == "fast":
            extra = cfg.get("comfy_launch_args_fast", [])
            args += [str(a) for a in (extra if extra else ["--fast"])]
        return args

    @property
    def owned(self) -> bool:
        """True when `process` is alive and this app started it (not attached)."""
        return (self.process is not None and self.process.poll() is None
                and not self.attached)

    async def ensure_ready(self, profile: str | None = "base") -> None:
        """Make sure a server is up. profile None = don't care (VLM traffic).

        A concrete profile restarts a server WE started when it mismatches.
        Restarting waits for the old port to actually close first: without
        that, the liveness check can attach to our own dying process and all
        later submits vanish into the void (measured overnight 2026-09-08).
        """
        async with self._start_lock:
            if self.object_info is not None and port_is_open(self.port):
                if profile is not None and self._profile is not None and \
                        self._profile != profile:
                    if self.process is not None and not self.attached:
                        self._log(f"[comfy] profile switch {self._profile} -> {profile}: "
                                  "restarting server we started")
                        self.shutdown()
                        await self._wait_port_closed()
                        self._launch(profile)
                        await self._wait_ready()
                        return
                    # Attached to a foreign server: never kill what we did
                    # not start. Proceed on the running profile instead.
                    self._log(f"[comfy] attached server runs profile "
                              f"{self._profile}; requested {profile} ignored")
                    await self._wait_ready()
                    if profile is not None:
                        await self._verify_or_restart_gpu(profile)
                    return
                if profile is not None:
                    await self._verify_or_restart_gpu(profile)
                return
            if port_is_open(self.port):
                # Never kill something we did not start. Attaching is the correct
                # behaviour when a ComfyUI is already serving this port.
                self.attached = True
                self._profile = None  # unknown flags on a foreign server
                self.inputs.reset()
                self._log(f"[comfy] attached to existing server on port {self.port}")
            elif self.config.get("auto_launch_comfy", True):
                self._launch(profile or "base")
            else:
                raise PipelineError(
                    f"生成エンジンが起動していません（port {self.port}）。"
                    "config.json の auto_launch_comfy が false になっています。",
                    kind="comfy_launch")
            await self._wait_ready()
            if profile is not None:
                await self._verify_or_restart_gpu(profile)

    def _gpu_mismatch_message(self, applied: dict | None = None) -> str:
        applied_desc = "不明"
        if applied is not None:
            applied_desc = json.dumps(
                {k: applied.get(k) for k in
                 ("known", "video_uuid", "aux_uuid", "visible_uuids",
                  "reserve_vram_gb", "note")},
                ensure_ascii=False)
        return (
            f"起動中のComfyUI（port {self.port}、H3が起動したものではありません）は、"
            f"GPU割り当てが保存済み設定と異なるか確認できません（適用中: {applied_desc}）。"
            "H3はこのComfyUIを停止しません。"
            "そのComfyUIを終了してから生成すると、H3が保存済み設定で起動します。")

    async def _verify_or_restart_gpu(self, profile: str) -> None:
        """Called only for concrete profiles (video jobs). Gemma/VLM traffic
        (profile None) never reaches here and is never gated on GPU plan."""
        plan = gpu_mod.current_plan(self.config)
        if not plan.get("ok"):
            raise PipelineError(
                "GPU設定に問題があるため生成エンジンを起動できません: "
                + "; ".join(plan.get("errors") or []),
                kind="gpu_config")
        signature = tuple(plan["launch_args"])

        if self.owned:
            if self.launched_gpu_signature != signature:
                self._log("[comfy] gpu plan changed: restarting server we started")
                self.shutdown()
                await self._wait_port_closed()
                self._launch(profile)
                await self._wait_ready()
            return

        if not self.attached:
            return  # not running yet / not our concern here

        cache_key = ("attached", signature)
        if cache_key in self._verified_gpu_signatures:
            return
        try:
            stats = await self.system_stats()
        except Exception as exc:                                    # noqa: BLE001
            raise PipelineError(self._gpu_mismatch_message(), kind="gpu_mismatch") from exc
        applied = gpu_mod.applied_from_stats(stats, gpu_mod.detect())
        ok = bool(applied.get("known")) and \
            (applied.get("visible_uuids") or [None])[0] == plan["video"]["uuid"]
        if ok and plan["effective_mode"] == "dual":
            aux = plan.get("aux") or {}
            aux_uuid = aux.get("uuid") if aux.get("kind") == "gpu" else None
            visible = applied.get("visible_uuids") or []
            ok = len(visible) > 1 and visible[1] == aux_uuid
        if not ok:
            raise PipelineError(self._gpu_mismatch_message(applied), kind="gpu_mismatch")
        self._verified_gpu_signatures.add(cache_key)

    async def _wait_port_closed(self, timeout_s: float = 30.0) -> None:
        """Wait until our old server actually releases the port."""
        import asyncio as _asyncio
        deadline = time.monotonic() + timeout_s
        while port_is_open(self.port):
            if time.monotonic() > deadline:
                raise PipelineError(
                    "生成エンジンの再起動待機がタイムアウトしました。",
                    kind="comfy_launch")
            await _asyncio.sleep(0.5)

    def _launch(self, profile: str = "base") -> None:
        cfg = self.config
        cfg.ensure_dirs()
        db = str(cfg.debug_dir / "h3app.db").replace("\\", "/")
        plan = gpu_mod.current_plan(cfg)
        if not plan.get("ok"):
            raise PipelineError(
                "GPU設定に問題があるため生成エンジンを起動できません: "
                + "; ".join(plan.get("errors") or []),
                kind="gpu_config")
        args = self.launch_args_for(profile)
        args += ["--extra-model-paths-config", str(cfg.paths_yaml),
                 "--database-url", "sqlite:///" + db]
        whitelist = cfg.get("whitelist_custom_nodes") or []
        if whitelist:
            # Empty whitelist means "the documented Stable environment", i.e. all
            # custom nodes. Only narrow it when the user explicitly asked.
            args += ["--disable-all-custom-nodes", "--whitelist-custom-nodes"]
            args += [str(w) for w in whitelist]

        env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        import os
        full_env = dict(os.environ)
        # Never let an inherited CUDA_VISIBLE_DEVICES from this app's own
        # process environment override the explicit --cuda-device list.
        full_env.pop("CUDA_VISIBLE_DEVICES", None)
        full_env.update(env)

        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(self._out_path, "wb")
        err = open(self._err_path, "wb")
        try:
            self.process = subprocess.Popen(
                args, cwd=str(cfg.comfy_dir), stdout=out, stderr=err, env=full_env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as exc:
            out.close()
            err.close()
            raise PipelineError(f"生成エンジンを起動できませんでした: {exc}",
                                kind="comfy_launch") from exc
        self._profile = profile
        self.launched_gpu_signature = tuple(plan["launch_args"])
        self.gpu_plan = plan
        self._verified_gpu_signatures.clear()
        self.inputs.reset()
        self._log(f"[comfy] launched pid={self.process.pid} port={self.port} "
                  f"profile={profile} gpu={plan.get('launch_args')}")
        # WP-D: record the server this app started (owned_by_h3=True) so the
        # launcher can identify and reclaim it after an app crash instead of
        # attaching to an orphan that still holds the models in VRAM.
        try:
            from . import procstate                       # noqa: PLC0415
            procstate.record_entry(
                procstate.ROLE_COMFYUI, int(self.process.pid),
                session_id=str(getattr(self, "session_id", "") or ""),
                port=int(self.port), owned_by_h3=True)
        except Exception:                                    # noqa: BLE001
            pass

    async def _wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise PipelineError(
                        "生成エンジンが起動直後に終了しました。"
                        f"詳細は {self._err_path} を確認してください。",
                        kind="comfy_launch", detail=self.stderr_tail())
                try:
                    async with session.get(f"{self.base}/object_info", timeout=aiohttp.ClientTimeout(total=20)) as r:
                        if r.status == 200:
                            self.object_info = await r.json()
                            self._log("[comfy] server ready")
                            return
                except Exception:
                    pass
                await asyncio.sleep(2)
        raise PipelineError(
            f"生成エンジンが {READY_TIMEOUT_SECONDS} 秒以内に応答しませんでした。",
            kind="timeout", detail=self.stderr_tail())

    def shutdown(self) -> None:
        """Stop only a server this app started."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except Exception:
                self.process.kill()
            try:
                from . import procstate                   # noqa: PLC0415
                procstate.remove_entry(
                    procstate.ROLE_COMFYUI,
                    str(getattr(self, "session_id", "") or "") or None)
            except Exception:                                # noqa: BLE001
                pass
        self.process = None
        self.object_info = None
        self._profile = None
        self.launched_gpu_signature = None
        self.gpu_plan = None
        self._verified_gpu_signatures.clear()
        self.inputs.reset()

    async def applied_state(self) -> dict:
        """Best-effort snapshot of what GPU allocation is actually running.

        Never raises: any failure to reach the server is reported as
        `running: False` / `known: False` rather than propagated.
        """
        running = False
        try:
            running = await self.is_reachable()
        except Exception:                                            # noqa: BLE001
            running = False
        if not running:
            return {"known": False, "running": False, "owned": self.owned,
                    "launch_args": [], "video_uuid": "", "aux_uuid": "",
                    "visible_uuids": [], "reserve_vram_gb": None,
                    "devices": [], "note": "ComfyUIは停止しています。"}
        try:
            stats = await self.system_stats()
            applied = gpu_mod.applied_from_stats(stats, gpu_mod.detect())
        except Exception as exc:                                      # noqa: BLE001
            applied = {"known": False, "launch_args": [], "video_uuid": "",
                       "aux_uuid": "", "visible_uuids": [],
                       "reserve_vram_gb": None, "devices": [],
                       "note": f"GPUの状態を取得できませんでした: {exc}"}
        applied["running"] = True
        applied["owned"] = self.owned
        return applied

    # ------------------------------------------------------------------ gate --
    def gate(self, graph: dict) -> None:
        if self.object_info is None:
            raise PipelineError("生成エンジンのノード一覧を取得できていません。", kind="comfy_launch")
        needed = {n["class_type"] for n in graph.values()}
        missing = sorted(needed - set(self.object_info))
        if missing:
            lines = [f"  - {c}  →  {pack_for(c)}" for c in missing]
            raise PipelineError(
                "必要なノードが導入されていません:\n" + "\n".join(lines),
                kind="missing_nodes", detail="missing=" + ", ".join(missing))

        if "SelectCLIPDevice" in needed:
            schema = self.object_info["SelectCLIPDevice"]["input"]["required"]["device"]
            options = list(schema[1].get("options", schema[0]) if len(schema) > 1 else schema[0])
            for node in graph.values():
                if node.get("class_type") != "SelectCLIPDevice":
                    continue
                device = node.get("inputs", {}).get("device")
                if device == "cpu" or not isinstance(device, str) or not device.startswith("gpu:"):
                    continue
                if device not in options:
                    raise PipelineError(
                        f"生成エンジンが {device}（保存済みGPU設定）を認識していません（選択肢: "
                        f"{options}）。設定 > GPU でGPUの割り当てを確認するか、"
                        "生成エンジンを保存済み設定で再起動してください。",
                        kind="no_gpu1")

    # ----------------------------------------------------------------- submit --
    def stderr_tail(self, lines: int = 40) -> str:
        if not self._err_path.exists():
            return ""
        try:
            data = self._err_path.read_bytes()[-60000:]
        except Exception:
            return ""
        return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])

    def _raise_if_server_failed(self, offset: int, node: str | None) -> None:
        if not self._err_path.exists():
            return
        try:
            with self._err_path.open("rb") as f:
                f.seek(offset)
                segment = f.read().decode("utf-8", errors="replace")
        except Exception:
            return
        hits = [m for m in ERROR_PATTERNS if m.lower() in segment.lower()]
        if hits:
            tail = "\n".join(segment.splitlines()[-35:])
            raise PipelineError(
                f"生成エンジンが停止しました（{node or '不明なノード'}）: {', '.join(hits)}",
                node=node or "", detail=tail)

    async def run_graph(self, graph: dict, *, on_event: Callable[[dict], None],
                        cancel: asyncio.Event | None = None,
                        salvage_nodes: list[str] | None = None,
                        profile: str | None = "base") -> dict:
        """Submit, follow to completion, return /history[prompt_id].

        on_event receives {"type": "executing"|"progress", ...} dicts.

        salvage_nodes: node ids whose output is worth more than the failure.
        ComfyUI stops the whole prompt at the first failing node, so a secondary
        node dying (the extra segment save, say) would otherwise throw away a
        main video that already cost 3.6 minutes of GPU time and is sitting on
        disk. If any of these nodes produced output before the error, the run is
        returned with "_partial_error" set instead of raising, and the caller
        decides what to keep.

        profile: launch profile ("base" | "fast" | None). The whole video
        job must pass one concrete profile; switching mid-job restarts the
        server. None means "don't care" (VLM/director traffic): use whatever
        server is up, never restart for it.
        """
        await self.ensure_ready(profile)
        await self.inputs.ensure(comfy_inputs.graph_input_names(graph))
        self.gate(graph)

        offset = self._err_path.stat().st_size if self._err_path.exists() else 0
        client_id = str(uuid.uuid4())
        current: str | None = None
        prompt_id: str | None = None
        pending: PipelineError | None = None

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}", heartbeat=30) as ws:
                async with session.post(f"{self.base}/prompt",
                                        json={"prompt": graph, "client_id": client_id}) as r:
                    payload = await r.json(content_type=None)
                    if r.status != 200 or "prompt_id" not in payload:
                        raise PipelineError(
                            f"生成エンジンがリクエストを拒否しました (HTTP {r.status})",
                            kind="prompt_rejected",
                            detail=json.dumps(payload, ensure_ascii=False)[:4000])
                    prompt_id = payload["prompt_id"]

                last_progress = time.monotonic()
                while True:
                    if cancel is not None and cancel.is_set():
                        await self.interrupt()
                        raise UserCancelled(node=current or "")
                    try:
                        msg = await ws.receive(timeout=20)
                    except asyncio.TimeoutError:
                        self._raise_if_server_failed(offset, current)
                        stalled = time.monotonic() - last_progress
                        if stalled > STALL_SECONDS:
                            await self._on_stall(stalled, current)
                        continue

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        kind = data.get("type")
                        body = data.get("data", {}) or {}
                        if body.get("prompt_id") not in (None, prompt_id):
                            continue
                        if kind == "executing":
                            last_progress = time.monotonic()
                            current = body.get("node")
                            if current is None:
                                break
                            on_event({"type": "executing", "node": current})
                        elif kind == "progress":
                            last_progress = time.monotonic()
                            on_event({"type": "progress", "node": current,
                                      "value": body.get("value", 0),
                                      "max": body.get("max", 1)})
                        elif kind == "executed":
                            last_progress = time.monotonic()
                            on_event({"type": "executed", "node": body.get("node")})
                        elif kind == "execution_error":
                            # Deliberately not raised here: /history has to be read
                            # first to find out what already finished. A node
                            # exception also trips the stderr marker scan, so that
                            # check has to wait for the salvage decision too.
                            node = body.get("node_type") or current or ""
                            pending = PipelineError(
                                body.get("exception_message") or "生成エンジンでエラーが発生しました",
                                node=node,
                                detail=json.dumps(body, ensure_ascii=False)[:6000])
                            break
                        elif kind == "execution_interrupted":
                            raise UserCancelled(node=current or "")
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        self._raise_if_server_failed(offset, current)
                        raise PipelineError("生成エンジンとの接続が切れました。",
                                            node=current or "", detail=self.stderr_tail())

            async with session.get(f"{self.base}/history/{prompt_id}") as r:
                history = await r.json(content_type=None)

        entry = history.get(prompt_id) or {}
        status = (entry.get("status") or {})
        if pending is None and status.get("status_str") == "error":
            pending = PipelineError("生成エンジンでエラーが発生しました。",
                                    node=current or "",
                                    detail=json.dumps(status, ensure_ascii=False)[:6000])

        if pending is not None:
            produced = [n for n in (salvage_nodes or [])
                        if (entry.get("outputs") or {}).get(n)]
            if produced:
                self._log(f"[comfy] partial failure, keeping output of {produced}: {pending}")
                entry["_partial_error"] = {
                    "message": str(pending),
                    "node": pending.node,
                    "salvaged": produced,
                }
                return entry
            self._raise_if_server_failed(offset, current)
            raise pending

        self._raise_if_server_failed(offset, current)
        return entry

    async def interrupt(self) -> None:
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{self.base}/interrupt",
                                   timeout=aiohttp.ClientTimeout(total=10))
        except Exception:
            pass

    async def _on_stall(self, stalled: float, current: str | None) -> None:
        """A job stopped reporting progress for STALL_SECONDS: give up on it.

        Real-machine finding (2026-09-13): a stalled sampler was left running
        at 100% GPU after the app raised PipelineError(kind="timeout") here,
        because nothing ever told ComfyUI to stop. interrupt() is best-effort
        (it already swallows its own errors) and must run BEFORE raising, so
        the server actually frees the GPU instead of continuing the doomed
        job in the background.
        """
        await self.interrupt()
        raise PipelineError(
            f"{stalled:.0f} 秒間まったく進みませんでした"
            f"（処理: {current or '不明'}）",
            kind="timeout", node=current or "")

    # ------------------------------------------------------------- free VRAM --
    async def system_stats(self) -> dict:
        """GET /system_stats - ComfyUI's own per-device VRAM report.

        Reports every torch device multigpu can see, so GPU0 and GPU1 both show
        up. Used only to measure; it changes nothing.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base}/system_stats",
                                   timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    raise PipelineError(f"GPU の状態を取得できませんでした (HTTP {r.status})",
                                        kind="network")
                return await r.json(content_type=None)

    @staticmethod
    def vram_summary(stats: dict) -> list[dict]:
        out = []
        for d in (stats.get("devices") or []):
            if d.get("type") != "cuda":
                continue
            total = int(d.get("vram_total") or 0)
            free = int(d.get("vram_free") or 0)
            out.append({
                "index": d.get("index"),
                "name": d.get("name", ""),
                "total_mib": round(total / 1048576),
                "free_mib": round(free / 1048576),
                "used_mib": round((total - free) / 1048576),
            })
        return out

    async def _unload_vlm_best_effort(self) -> bool:
        """Ask the llama-cpp node pack to drop its model, via its own node.

        ComfyUI's model manager does not own the llama.cpp VLM, so /free cannot
        reach it. Rather than touching the node pack's internals from here, this
        submits a three-node prompt that calls the pack's OWN unload node -
        exactly what graph B does at the end of a normal run. Best effort: if
        the pack is absent or the prompt fails, /free still runs.
        """
        needed = ("PrimitiveInt", "llama_cpp_unload_model", "PreviewAny")
        if not self.object_info or any(n not in self.object_info for n in needed):
            return False
        graph = {
            "seed": {"class_type": "PrimitiveInt", "inputs": {"value": 0}},
            "unload": {"class_type": "llama_cpp_unload_model",
                       "inputs": {"any": ["seed", 0]}},
            # llama_cpp_unload_model is not an OUTPUT_NODE, and ComfyUI rejects a
            # prompt that produces no outputs, so terminate on PreviewAny.
            "done": {"class_type": "PreviewAny", "inputs": {"source": ["unload", 0]}},
        }
        try:
            await self.run_graph(graph, on_event=lambda ev: None,
                                 profile=self._profile or "base")
            return True
        except Exception as exc:
            self._log(f"[comfy] VLM unload skipped: {exc}")
            return False

    async def free_models(self) -> dict:
        """Unload everything ComfyUI has on the GPUs, keeping the server alive.

        Uses ComfyUI's official POST /free. That sets the prompt queue's
        unload_models / free_memory flags; set_flag notifies the queue condition
        variable, so the idle prompt_worker wakes immediately, calls
        model_management.unload_all_models() (which loops over every torch
        device, i.e. GPU0 and GPU1) and then gc.collect() + soft_empty_cache().

        Nothing here reaches into model objects, and no disk cache is touched.
        """
        if not await self.is_reachable():
            raise PipelineError(
                "ComfyUI が起動していません。解放するモデルはありません。",
                kind="comfy_not_running")

        # Populates object_info if this app process only just attached to an
        # already-running server. The port is open, so this attaches and never
        # launches anything. Keeps the running profile: freeing VRAM must never
        # restart the server into another flag set.
        await self.ensure_ready(self._profile or "base")

        before = self.vram_summary(await self.system_stats())
        vlm = await self._unload_vlm_best_effort()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.base}/free",
                        json={"unload_models": True, "free_memory": True},
                        timeout=aiohttp.ClientTimeout(total=30)) as r:
                    if r.status != 200:
                        raise PipelineError(
                            f"ComfyUI がモデル解放を受け付けませんでした (HTTP {r.status})",
                            kind="network")
        except aiohttp.ClientError as exc:
            raise PipelineError("ComfyUI との通信に失敗しました。", kind="network",
                                detail=str(exc)) from exc

        # The worker frees asynchronously; wait until the numbers stop moving
        # rather than guessing a sleep duration.
        after = before
        stable = 0
        for _ in range(20):
            await asyncio.sleep(0.4)
            try:
                current = self.vram_summary(await self.system_stats())
            except Exception:
                break
            if current == after:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            after = current

        freed = 0
        by_index = {d["index"]: d for d in before}
        for d in after:
            prev = by_index.get(d["index"])
            if prev:
                freed += max(0, prev["used_mib"] - d["used_mib"])

        self._log(f"[comfy] free_models: freed ~{freed} MiB (vlm_unloaded={vlm})")
        return {"before": before, "after": after, "freed_mib": freed,
                "vlm_unloaded": vlm}

    async def is_reachable(self) -> bool:
        if not port_is_open(self.port):
            return False
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base}/object_info",
                                       timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------- extractors --
def text_output(history: dict, node_id: str) -> str:
    """Pull a PreviewAny node's text back out of /history."""
    outputs = (history.get("outputs") or {}).get(node_id) or {}
    texts = outputs.get("text") or []
    if not texts:
        raise PipelineError("文章生成の結果を取得できませんでした。",
                            node=node_id,
                            detail=json.dumps(history.get("outputs", {}), ensure_ascii=False)[:4000])
    value = texts[0]
    return value if isinstance(value, str) else str(value)


def saved_video(history: dict, node_id: str) -> dict | None:
    """SaveVideo's ui payload -> {"filename", "subfolder", "type"}."""
    outputs = (history.get("outputs") or {}).get(node_id) or {}
    for key in ("images", "video", "videos", "gifs", "result"):
        items = outputs.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            item = items[0]
            if "filename" in item:
                return {"filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output")}
    return None


def output_path(comfy_output: Path, saved: dict) -> Path:
    sub = saved.get("subfolder") or ""
    return comfy_output / sub / saved["filename"] if sub else comfy_output / saved["filename"]
