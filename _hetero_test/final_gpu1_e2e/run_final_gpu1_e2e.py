from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import aiohttp
import psutil

PORT = 8401
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/ws"
PROJECT = Path(r"E:\AI-Projects\H3")
ROOT = PROJECT / "_hetero_test" / "final_gpu1_e2e"
PROMPT_PATH = PROJECT / "_hetero_test" / "gold" / "gold_final_prompt.txt"
RESULT = ROOT / "final_result.json"
EVENTS = ROOT / "final_events.jsonl"
SERVER_ERR = ROOT / "comfy_final.err"
GOLD_PROMPT_SHA = "62CCB107A37FA1D6EED26E5892D6BDDE2A3EBB0446CB294145997029EAD059B3"
REF_FILE = "h3test_face_ref.png"
REF_SHA = "885D9BE9D35B6041959A2DA1D544F83F95BFA1BC32BA57413532A687473E4562"
STEPS = 8
SEED = 123456789

CPU_REFERENCE_TO_VIDEO_SECONDS = 327.5359712999925
GPU1_REFERENCE_TO_VIDEO_SECONDS_PREVIOUS = 76.40710499999113
CPU_CACHED_POSTPROCESS_SECONDS_PREVIOUS = 222.83469139999943
GPU1_CACHED_POSTPROCESS_SECONDS_PREVIOUS = 177.61561449999863

ALLOWED_CLASSES = {
    "H3GoldFixedPrompt", "LoadImage", "LayerUtility: ImageScaleByAspectRatio V2",
    "CLIPLoader", "SelectCLIPDevice", "VAELoader", "MiniMaxH3ReferenceToVideo",
    "UNETLoader", "LoraLoaderModelOnly", "RandomNoise", "BasicGuider",
    "KSamplerSelect", "BasicScheduler", "SamplerCustomAdvanced",
    "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
}

ERROR_PATTERNS = (
    "!!! Exception", "Traceback", "CUDA error", "OutOfMemoryError",
    "CUDA out of memory", "Windows fatal exception",
)


def emit(event: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"wall_time": time.time(), **event}, ensure_ascii=False) + "\n")


def fixed_prompt() -> str:
    raw = PROMPT_PATH.read_bytes()
    text = raw.decode("utf-8")
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith(("\r", "\n")):
        text = text[:-1]
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest().upper()
    if actual != GOLD_PROMPT_SHA:
        raise RuntimeError(f"GOLD prompt hash mismatch: {actual}")
    return text


def build_graph() -> dict:
    return {
        "1": {
            "class_type": "H3GoldFixedPrompt",
            "inputs": {
                "prompt_path": str(PROMPT_PATH).replace("\\", "/"),
                "expected_gold_prompt_sha256": GOLD_PROMPT_SHA,
            },
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": REF_FILE}},
        "3": {
            "class_type": "LayerUtility: ImageScaleByAspectRatio V2",
            "inputs": {
                "aspect_ratio": "original", "proportional_width": 1,
                "proportional_height": 1, "fit": "crop", "method": "lanczos",
                "round_to_multiple": "32", "scale_to_side": "longest",
                "scale_to_length": 1536, "background_color": "#000000",
                "image": ["2", 0],
            },
        },
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                "type": "minimax", "device": "default",
            },
        },
        "9": {"class_type": "SelectCLIPDevice", "inputs": {"clip": ["4", 0], "device": "gpu:1"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "7": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["9", 0], "vae": ["5", 0], "audio_vae": ["6", 0],
                "prompt": ["1", 0], "width": 576, "height": 1024, "length": 124,
                "ref_image_size": "max", "ref_images.ref_image_0": ["3", 0],
            },
        },
        "10": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"},
        },
        "11": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": "minimax_h3_turbo_v4_step600_ema.safetensors",
                "strength_model": 1.0, "model": ["10", 0],
            },
        },
        "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["11", 0], "conditioning": ["7", 0]}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "15": {
            "class_type": "BasicScheduler",
            "inputs": {"scheduler": "simple", "steps": STEPS, "denoise": 1.0, "model": ["11", 0]},
        },
        "16": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["12", 0], "guider": ["13", 0], "sampler": ["14", 0],
                "sigmas": ["15", 0], "latent_image": ["7", 1],
            },
        },
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["5", 0]}},
        "18": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["16", 0], "vae": ["6", 0]}},
        "19": {
            "class_type": "CreateVideo",
            "inputs": {"fps": 24.0, "bit_depth": 8, "images": ["17", 0], "audio": ["18", 0]},
        },
        "20": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": "H3/20260814_FINAL/gpu1_live_e2e",
                "format": "auto", "codec": "auto", "video": ["19", 0],
            },
        },
    }


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
    markers = (
        "!!! Exception during processing !!!",
        "Exception in thread Thread-1 (prompt_worker)",
        "Windows fatal exception",
        "CUDA out of memory",
        "OutOfMemoryError",
    )
    hits = [m for m in markers if m.lower() in segment.lower()]
    if hits:
        tail = "\n".join(segment.splitlines()[-30:])
        raise RuntimeError(
            f"ComfyUI failed while running node {current or 'unknown'}; "
            f"markers={hits}. Recent server log:\n{tail}"
        )


async def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULT.unlink(missing_ok=True)
    EVENTS.unlink(missing_ok=True)
    fixed_prompt()
    graph = build_graph()
    (ROOT / "final_prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    if set(node["class_type"] for node in graph.values()) != ALLOWED_CLASSES:
        raise RuntimeError("final graph class gate failed")
    if graph["9"]["inputs"] != {"clip": ["4", 0], "device": "gpu:1"}:
        raise RuntimeError("GPU1 TE routing gate failed")
    if graph["16"]["inputs"]["latent_image"] != ["7", 1] or graph["13"]["inputs"]["conditioning"] != ["7", 0]:
        raise RuntimeError("live ReferenceToVideo is not feeding the sampler")

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
            missing = sorted(ALLOWED_CLASSES - set(object_info))
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
                            raise TimeoutError(
                                f"no workflow progress for {stalled:.0f}s at node {current or 'unknown'}"
                            )
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
                                "7": "GPU1 text/reference conditioning",
                                "16": "GPU0 H3 sampling (8 steps)",
                                "17": "GPU0 video VAE decode",
                                "18": "GPU0 audio VAE decode",
                                "19": "CreateVideo",
                                "20": "SaveVideo",
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

    segment = log_segment(log_offset)
    error_hits = [p for p in ERROR_PATTERNS if p.lower() in segment.lower()]
    if error_hits:
        raise RuntimeError(f"raw ComfyUI error markers: {error_hits}")

    peaks = {}
    for idx in ("0", "1"):
        vals = [s for s in samples if idx in s.get("gpu", {})]
        if vals:
            peaks[idx] = {
                "peak_memory_mib": max(s["gpu"][idx]["memory_mib"] for s in vals),
                "peak_util_percent": max(s["gpu"][idx]["util_percent"] for s in vals),
                "active_samples": sum(s["gpu"][idx]["util_percent"] > 0 for s in vals),
            }

    ref_s = node_duration.get("7")
    sample_s = node_duration.get("16")
    result = {
        "status": "success",
        "prompt_id": prompt_id,
        "server_ready_seconds": server_ready_seconds,
        "generation_e2e_seconds": prompt_seconds,
        "node_seconds": node_duration,
        "reference_to_video_seconds": ref_s,
        "sampler_seconds": sample_s,
        "sampler_seconds_per_step": (sample_s / STEPS) if sample_s else None,
        "video_decode_seconds": node_duration.get("17"),
        "audio_decode_seconds": node_duration.get("18"),
        "create_video_seconds": node_duration.get("19"),
        "save_video_seconds": node_duration.get("20"),
        "resource_peaks": peaks,
        "ram_peak_bytes": max((s["ram_used_bytes"] for s in samples), default=None),
        "both_gpus_active": all(peaks.get(i, {}).get("active_samples", 0) > 0 for i in ("0", "1")),
        "fixed_settings": {
            "steps": STEPS, "seed": SEED, "sampler": "res_multistep", "scheduler": "simple",
            "width": 576, "height": 1024, "length": 124,
            "clip_device": "gpu:1", "dit_device": "gpu:0",
        },
        "archived_baselines": {
            "cpu_reference_to_video_seconds": CPU_REFERENCE_TO_VIDEO_SECONDS,
            "previous_gpu1_reference_to_video_seconds": GPU1_REFERENCE_TO_VIDEO_SECONDS_PREVIOUS,
            "cpu_cached_sampling_decode_total_seconds": CPU_CACHED_POSTPROCESS_SECONDS_PREVIOUS,
            "gpu1_cached_sampling_decode_total_seconds": GPU1_CACHED_POSTPROCESS_SECONDS_PREVIOUS,
        },
        "history_status": history.get(prompt_id, {}).get("status"),
    }
    if ref_s:
        result["reference_to_video_speedup_vs_cpu_percent"] = (1.0 - ref_s / CPU_REFERENCE_TO_VIDEO_SECONDS) * 100.0
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        ROOT.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "wall_time": time.time()}
        RESULT.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"event": "runner_error", **failure})
        raise
