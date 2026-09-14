from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import aiohttp
import psutil

PORT = 8402
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/ws"
PROJECT = Path(r"E:\AI-Projects\H3")
ROOT = PROJECT / "_hetero_test" / "opt_ref_match"
BASELINE_GRAPH = PROJECT / "_hetero_test" / "final_gpu1_e2e" / "final_prompt.json"
RESULT = ROOT / "opt_result.json"
EVENTS = ROOT / "opt_events.jsonl"
SERVER_ERR = ROOT / "comfy_opt.err"
CANDIDATE_GRAPH = ROOT / "candidate_prompt.json"

EXPECTED_BASELINE_FILE_SHA256 = "8E126ECBA480FA6F9DD6470B51BDAB9C28FC7F9BE29DBA1508FDB925012F70E1"
EXPECTED_BASELINE_CANONICAL_SHA256 = "33116CCAAC4AAB5E102C982775E6368460BA60D344CDADDC6FE846C3CD5608F0"
CURRENT_E2E_SECONDS = 308.36541569999827
CURRENT_REF_SECONDS = 80.90253220000886
CURRENT_SAMPLER_SECONDS = 192.7951810999948
CURRENT_VIDEO_DECODE_SECONDS = 27.97136280000268
CURRENT_AUDIO_DECODE_SECONDS = 1.575199300001259

ALLOWED_DIFFS = {
    ("7", "inputs", "ref_image_size"),
    ("20", "inputs", "filename_prefix"),
}
ERROR_PATTERNS = (
    "!!! Exception during processing !!!", "Exception in thread Thread-1 (prompt_worker)",
    "CUDA out of memory", "OutOfMemoryError", "Windows fatal exception",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def canonical_sha(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(raw)


def emit(event: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"wall_time": time.time(), **event}, ensure_ascii=False) + "\n")


def diff_paths(a, b, path=()):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(path + (str(k),))
            else:
                out.extend(diff_paths(a[k], b[k], path + (str(k),)))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + ("<len>",)]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, path + (str(i),)))
        return out
    return [] if a == b else [path]


def build_candidate() -> dict:
    if not BASELINE_GRAPH.exists():
        raise FileNotFoundError(f"Successful baseline graph is missing: {BASELINE_GRAPH}")
    raw = BASELINE_GRAPH.read_bytes()
    file_sha = sha256_bytes(raw)
    if file_sha != EXPECTED_BASELINE_FILE_SHA256:
        raise RuntimeError(f"Baseline graph file SHA drift: {file_sha}")
    baseline = json.loads(raw.decode("utf-8"))
    canon = canonical_sha(baseline)
    if canon != EXPECTED_BASELINE_CANONICAL_SHA256:
        raise RuntimeError(f"Baseline graph canonical SHA drift: {canon}")
    if baseline["7"]["class_type"] != "MiniMaxH3ReferenceToVideo":
        raise RuntimeError("Baseline node 7 is no longer MiniMaxH3ReferenceToVideo")
    if baseline["7"]["inputs"].get("ref_image_size") != "max":
        raise RuntimeError("Baseline no longer uses ref_image_size=max")
    if baseline["9"]["inputs"] != {"clip": ["4", 0], "device": "gpu:1"}:
        raise RuntimeError("Baseline GPU1 TE routing drifted")
    if baseline["16"]["inputs"].get("latent_image") != ["7", 1]:
        raise RuntimeError("Baseline sampler latent routing drifted")
    if baseline["13"]["inputs"].get("conditioning") != ["7", 0]:
        raise RuntimeError("Baseline conditioning routing drifted")

    candidate = copy.deepcopy(baseline)
    candidate["7"]["inputs"]["ref_image_size"] = "match"
    candidate["20"]["inputs"]["filename_prefix"] = "H3/20260814_OPT/ref_match"

    diffs = set(diff_paths(baseline, candidate))
    if diffs != ALLOWED_DIFFS:
        raise RuntimeError(f"Candidate diff gate failed: {sorted(diffs)}")
    return candidate


def query_resources() -> dict:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpu = {}
    for line in proc.stdout.splitlines():
        idx, name, memory, util = [part.strip() for part in line.split(",", 3)]
        gpu[idx] = {"name": name, "memory_mib": int(memory), "util_percent": int(util)}
    return {"time": time.time(), "gpu": gpu, "ram_used_bytes": int(psutil.virtual_memory().used)}


async def monitor(samples: list, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            samples.append(query_resources())
        except Exception as exc:
            emit({"event": "monitor_error", "error": str(exc)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


async def wait_server(session: aiohttp.ClientSession) -> dict:
    for _ in range(180):
        try:
            async with session.get(f"{BASE}/object_info", timeout=5) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("ComfyUI server did not become ready")


def log_segment(offset: int) -> str:
    if not SERVER_ERR.exists():
        return ""
    with SERVER_ERR.open("rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def raise_if_server_failed(offset: int, current: str | None) -> None:
    segment = log_segment(offset)
    hits = [m for m in ERROR_PATTERNS if m.lower() in segment.lower()]
    if hits:
        tail = "\n".join(segment.splitlines()[-35:])
        raise RuntimeError(
            f"ComfyUI failed while running node {current or 'unknown'}; markers={hits}.\n{tail}"
        )


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULT.unlink(missing_ok=True)
    EVENTS.unlink(missing_ok=True)
    graph = build_candidate()
    CANDIDATE_GRAPH.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    samples: list[dict] = []
    stop = asyncio.Event()
    monitor_task = asyncio.create_task(monitor(samples, stop))
    node_start: dict[str, float] = {}
    node_duration: dict[str, float] = {}
    current: str | None = None
    prompt_id = None
    server_wait_started = time.perf_counter()
    prompt_started = None
    log_offset = SERVER_ERR.stat().st_size if SERVER_ERR.exists() else 0

    try:
        async with aiohttp.ClientSession() as session:
            object_info = await wait_server(session)
            server_ready_seconds = time.perf_counter() - server_wait_started
            required_classes = {n["class_type"] for n in graph.values()}
            missing = sorted(required_classes - set(object_info))
            if missing:
                raise RuntimeError(f"required nodes missing: {missing}")
            print(f"ComfyUI ready in {server_ready_seconds:.1f}s", flush=True)

            device_schema = object_info["SelectCLIPDevice"]["input"]["required"]["device"]
            options = device_schema[1].get("options", device_schema[0])
            if "gpu:1" not in options:
                raise RuntimeError(f"gpu:1 unavailable: {options}")

            client_id = str(uuid.uuid4())
            async with session.ws_connect(f"{WS}?clientId={client_id}", heartbeat=30) as ws:
                prompt_started = time.perf_counter()
                async with session.post(f"{BASE}/prompt", json={"prompt": graph, "client_id": client_id}) as r:
                    payload = await r.json()
                    if r.status != 200 or "prompt_id" not in payload:
                        raise RuntimeError(f"prompt rejected: HTTP {r.status}: {payload}")
                    prompt_id = payload["prompt_id"]
                emit({"event": "prompt_submitted", "prompt_id": prompt_id})

                last_progress = time.perf_counter()
                while True:
                    try:
                        msg = await ws.receive(timeout=20)
                    except asyncio.TimeoutError:
                        raise_if_server_failed(log_offset, current)
                        stalled = time.perf_counter() - last_progress
                        if stalled > 300:
                            raise TimeoutError(f"no workflow progress for {stalled:.0f}s at node {current or 'unknown'}")
                        continue
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        kind = data.get("type")
                        body = data.get("data", {})
                        if body.get("prompt_id") not in (None, prompt_id):
                            continue
                        if kind == "executing":
                            now = time.perf_counter()
                            last_progress = now
                            if current is not None and current in node_start:
                                node_duration[current] = now - node_start[current]
                            current = body.get("node")
                            if current is None:
                                break
                            node_start[current] = now
                            labels = {
                                "7": "GPU1 conditioning — MATCH reference size",
                                "16": "GPU0 H3 sampling (same 8 steps)",
                                "17": "GPU0 video VAE decode",
                                "18": "GPU0 audio VAE decode",
                                "20": "Save candidate video",
                            }
                            print(f"Running node {current}: {labels.get(current, graph[current]['class_type'])}", flush=True)
                            emit({"event": "executing", "node": current})
                        elif kind in ("execution_error", "execution_interrupted"):
                            raise_if_server_failed(log_offset, current)
                            raise RuntimeError(f"{kind}: {body}")
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError(f"websocket ended early: {msg.type}")

            prompt_seconds = time.perf_counter() - prompt_started
            async with session.get(f"{BASE}/history/{prompt_id}") as r:
                history = await r.json()
    finally:
        stop.set()
        await monitor_task

    raise_if_server_failed(log_offset, current)
    peaks = {}
    for idx in ("0", "1"):
        vals = [s for s in samples if idx in s.get("gpu", {})]
        if vals:
            peaks[idx] = {
                "peak_memory_mib": max(s["gpu"][idx]["memory_mib"] for s in vals),
                "peak_util_percent": max(s["gpu"][idx]["util_percent"] for s in vals),
                "active_samples": sum(s["gpu"][idx]["util_percent"] > 0 for s in vals),
            }

    result = {
        "status": "success",
        "candidate": "ref_image_size=match",
        "quality_risk": "Reference resolution is reduced by the official H3 MATCH mode; human A/B is mandatory.",
        "baseline_untouched": True,
        "baseline_e2e_seconds": CURRENT_E2E_SECONDS,
        "candidate_e2e_seconds": prompt_seconds,
        "speedup_seconds": CURRENT_E2E_SECONDS - prompt_seconds,
        "speedup_percent": (1.0 - prompt_seconds / CURRENT_E2E_SECONDS) * 100.0,
        "node_seconds": node_duration,
        "reference_to_video_seconds": node_duration.get("7"),
        "sampler_seconds": node_duration.get("16"),
        "video_decode_seconds": node_duration.get("17"),
        "audio_decode_seconds": node_duration.get("18"),
        "current_reference_to_video_seconds": CURRENT_REF_SECONDS,
        "current_sampler_seconds": CURRENT_SAMPLER_SECONDS,
        "current_video_decode_seconds": CURRENT_VIDEO_DECODE_SECONDS,
        "current_audio_decode_seconds": CURRENT_AUDIO_DECODE_SECONDS,
        "resource_peaks": peaks,
        "ram_peak_bytes": max((s["ram_used_bytes"] for s in samples), default=None),
        "both_gpus_active": all(peaks.get(i, {}).get("active_samples", 0) > 0 for i in ("0", "1")),
        "fixed_settings": {
            "steps": 8,
            "seed": 123456789,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "width": 576,
            "height": 1024,
            "length": 124,
            "clip_device": "gpu:1",
            "dit_device": "gpu:0",
            "reserve_vram_gib": 1.0,
            "only_changed_quality_input": "MiniMaxH3ReferenceToVideo.ref_image_size: max -> match",
        },
        "baseline_graph_file_sha256": EXPECTED_BASELINE_FILE_SHA256,
        "baseline_graph_canonical_sha256": EXPECTED_BASELINE_CANONICAL_SHA256,
        "candidate_graph_canonical_sha256": canonical_sha(graph),
        "prompt_id": prompt_id,
        "server_ready_seconds": server_ready_seconds,
        "history_status": history.get(prompt_id, {}).get("status"),
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SPEED RESULT ===", flush=True)
    print(f"CURRENT   max   : {CURRENT_E2E_SECONDS:.2f} s ({CURRENT_E2E_SECONDS/60:.2f} min)", flush=True)
    print(f"CANDIDATE match : {prompt_seconds:.2f} s ({prompt_seconds/60:.2f} min)", flush=True)
    print(f"DELTA           : {CURRENT_E2E_SECONDS-prompt_seconds:+.2f} s  ({result['speedup_percent']:+.2f}%)", flush=True)
    print("Do NOT adopt from speed alone. Compare the video/contact sheet first.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        ROOT.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "wall_time": time.time()}
        RESULT.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"event": "runner_error", **failure})
        raise
