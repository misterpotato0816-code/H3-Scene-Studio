# -*- coding: utf-8 -*-
"""Submit dumped app graphs to an isolated server with stage timing.

Usage: <venv python> backend/h3_v2/stage_time.py --graph A.json [--graph B.json ...]
       --port 8402 --tag X [--reuse] [--server-args="--fast"]
Multiple --graph files run IN ORDER on ONE server (state reproduction:
e.g. director dump then generate dump). A sampler thread records
nvidia-smi GPU state every 15 s (in-process, no survivors).
Prints per-graph per-stage seconds as JSON lines.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(BACKEND_ROOT.parent))

import aiohttp  # noqa: E402

LOAD_PREFIX = ("load_", "scale_", "restore", "clip", "vae_", "unet", "lora",
               "sigshift", "sage", "lowvram", "chunkff", "batch", "grid", "sheet",
               "vlm", "params")


def stage_of(node_id: str) -> str:
    n = node_id
    if n.startswith(LOAD_PREFIX):
        return "load"
    if n in ("h3", "last_frame", "instruct", "preview"):
        return "conditioning" if n in ("h3", "last_frame") else "vlm"
    if n in ("noise", "guider", "sampler_select", "scheduler", "sampler"):
        return "sampling"
    if n == "decode_video":
        return "vae"
    if n == "decode_audio":
        return "audio"
    if n in ("prev_video", "prev_components", "tail_images", "tail_audio",
             "voice_master", "last_frame_image", "unload", "purge"):
        return "relay" if n not in ("unload", "purge") else "purge"
    if n.startswith(("concat_", "create_", "save_")):
        return "merge"
    return "other"


def gpu_snapshot() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks.sm,"
             "power.draw,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return " | ".join(out.splitlines())
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def sampler_thread(stop: threading.Event, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        while not stop.is_set():
            f.write(f"{time.strftime('%H:%M:%S')} {gpu_snapshot()}\n")
            f.flush()
            stop.wait(15)


async def submit(base: str, graph: dict, timeout_s: int = 1800) -> dict:
    t0 = time.monotonic()
    cur_stage: str | None = None
    cur_t = t0
    totals: dict[str, float] = {}
    first_exec: float | None = None
    client_id = str(uuid.uuid4())
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(f"{base}/ws?clientId={client_id}",
                                      heartbeat=30) as ws:
            async with session.post(f"{base}/prompt",
                                    json={"prompt": graph,
                                          "client_id": client_id}) as r:
                payload = await r.json(content_type=None)
                if r.status != 200 or "prompt_id" not in payload:
                    raise RuntimeError(f"prompt rejected: {r.status} {payload}")
                prompt_id = payload["prompt_id"]
            while True:
                try:
                    msg = await ws.receive(timeout=20)
                except asyncio.TimeoutError:
                    totals["_stall_note"] = totals.get("_stall_note", 0) + 20  # type: ignore[assignment]
                    if time.monotonic() - t0 > timeout_s:
                        raise RuntimeError("stall: no events, giving up")
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                body = data.get("data", {}) or {}
                if body.get("prompt_id") not in (None, prompt_id):
                    continue
                now = time.monotonic()
                if data.get("type") == "executing":
                    if cur_stage is not None:
                        totals[cur_stage] = totals.get(cur_stage, 0.0) + now - cur_t
                    node = body.get("node")
                    if node is None:
                        break
                    if first_exec is None:
                        first_exec = now - t0
                    cur_stage = stage_of(str(node))
                    cur_t = now
    totals["_total"] = round(time.monotonic() - t0, 1)
    totals["_queue_to_first_exec"] = round(first_exec or -1, 1)
    return {k: (round(v, 1) if isinstance(v, float) else v)
            for k, v in totals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, action="append",
                    help="graph JSON file; repeatable, run in order on one server")
    ap.add_argument("--port", type=int, default=8402)
    ap.add_argument("--tag", default="stage")
    ap.add_argument("--reuse", action="store_true",
                    help="do not start a server; use the one on --port")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--full-nodes", action="store_true",
                    help="no whitelist: reproduce the production app server")
    ap.add_argument("--free-between", action="store_true",
                    help="POST /free (unload+gc, RAM copies kept) between graphs")
    args = ap.parse_args()
    graphs = [json.loads(Path(g).read_text(encoding="utf-8")) for g in args.graph]
    base = f"http://127.0.0.1:{args.port}"
    proc = None
    if not args.reuse:
        from h3_v2.ab_run import launch_server  # noqa: E402
        import h3_v2.ab_run as abr  # noqa: E402
        abr.SERVER_EXTRA_ARGS = [a for a in args.server_args.split()] \
            if args.server_args else []
        bench = PROJECT_ROOT / "benchmarks"
        proc = launch_server(args.port, bench / f"stage_{args.tag}.log",
                             bench / f"stage_{args.tag}.err",
                             full_nodes=args.full_nodes)
        time.sleep(5)
        # wait ready
        import socket as _s
        t0 = time.monotonic()
        while time.monotonic() - t0 < 360:
            try:
                with _s.socket() as s:
                    s.settimeout(1)
                    s.connect(("127.0.0.1", args.port))
                break
            except OSError:
                time.sleep(3)
        time.sleep(10)
    stop = threading.Event()
    sampler = threading.Thread(target=sampler_thread,
                               args=(stop, PROJECT_ROOT / "benchmarks" /
                                     f"gpu_split_{args.tag}.log"),
                               daemon=True)
    sampler.start()
    rc = 0
    try:
        for i, graph in enumerate(graphs):
            if i > 0 and args.free_between:
                t_free = time.monotonic()
                async def _free() -> None:
                    timeout = aiohttp.ClientTimeout(total=120)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                                f"{base}/free",
                                json={"unload_models": True,
                                      "free_memory": True}) as r:
                            if r.status != 200:
                                raise RuntimeError(f"/free HTTP {r.status}")
                asyncio.run(_free())
                print(json.dumps({"tag": args.tag, "seq": f"free{i}",
                                  "free_s": round(time.monotonic() - t_free, 1)}),
                      flush=True)
                time.sleep(5)
            try:
                res = asyncio.run(submit(base, graph))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"tag": args.tag, "seq": i,
                                  "error": str(exc)}), flush=True)
                rc = 1
                break
            res["tag"] = args.tag
            res["seq"] = i
            print(json.dumps(res, ensure_ascii=False), flush=True)
        return rc
    finally:
        stop.set()
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            try:
                proc.terminate()
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    sys.exit(main())
