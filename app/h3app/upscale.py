# -*- coding: utf-8 -*-
"""WP-B: post-generation video upscale (ESRGAN / SeedVR2) via ComfyUI.

Design: docs/DESIGN_2026-09-13_ADDITIONS.md, section "WP-B".

This module is entirely additive: it reads pipeline.Pipeline / comfy.ComfyClient
/ comfy_inputs / media / gpu but never changes their behaviour. It never writes
into app/config.json, app/comfy_paths.yaml, or ComfyUI's own input/models
folders, and it never downloads a model - missing SeedVR2 weights are only
ever reported (source_url/license/size) for the user to install themselves.

Job lifecycle reuses the existing single-slot Pipeline/Runner machinery:
UpscaleRunner IS a pipeline.Runner (own stage names only), scheduled through
pipeline.start_runner() so it shares pipeline.active with video generation and
story runs (same 409 "busy" gate, same /api/events/{job_id} SSE stream, same
Pipeline.cancel()/interrupt() path).
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from . import comfy as comfy_mod
from . import comfy_inputs
from . import media as media_mod
from .errors import PipelineError, UserCancelled, classify, pack_for
from .pipeline import Pipeline, Runner

# --------------------------------------------------------------- constants --
PRESETS = ("1080x1920", "x2")
METHODS = ("lanczos", "esrgan", "seedvr2")

# "standard" needs no extra nodes/models (plain ffmpeg scale); "ai" methods
# run through ComfyUI and may report missing nodes/models.
METHOD_KINDS = {"lanczos": "standard", "esrgan": "ai", "seedvr2": "ai"}
METHOD_LABELS = {
    "lanczos": "標準拡大（Lanczos, FFmpeg）",
    "esrgan": "AI高画質化（RealESRGAN）",
    "seedvr2": "AI高画質化（SeedVR2 7B）",
}

ESRGAN_MODEL = "RealESRGAN_x4plus.pth"
ESRGAN_SOURCE_URL = "https://github.com/xinntao/Real-ESRGAN"
ESRGAN_LICENSE = "BSD-3-Clause"
ESRGAN_SIZE_BYTES = 67_040_989  # published release size (approximate)

SEEDVR2_DIT_FILE = "seedvr2_ema_7b_fp16.safetensors"
SEEDVR2_VAE_FILE = "ema_vae_fp16.safetensors"
SEEDVR2_SOURCE_URL = "https://huggingface.co/numz/SeedVR2_comfyUI"
SEEDVR2_LICENSE = "Apache-2.0"
SEEDVR2_DIT_SIZE_BYTES = 16_500_000_000  # ~16.5 GB, per the real-machine note
SEEDVR2_VAE_SIZE_BYTES = 1_300_000_000   # best-effort estimate when absent

STAGE_INPUT, STAGE_MODEL, STAGE_UPSCALE, STAGE_EXPORT, STAGE_DONE = range(5)
STAGE_NAMES = ["入力を準備", "モデルを読み込み", "アップスケール", "動画を書き出し", "完了"]

_NODE_STAGE = {
    "load_video": STAGE_INPUT, "components": STAGE_INPUT,
    "upscale_model": STAGE_MODEL, "dit": STAGE_MODEL, "vae": STAGE_MODEL,
    "upscale": STAGE_UPSCALE, "scale": STAGE_UPSCALE,
    "create_video": STAGE_EXPORT, "save_video": STAGE_EXPORT,
}

MAX_OOM_RETRIES = 2
MAX_BLOCKS_TO_SWAP = 36
# SeedVR2 7B fp16 on 12 GB cards - measured 2026-09-14 (5-frame probes,
# 576x1024 -> 1080x1920): with ALL blocks swapped to CPU plus
# swap_io_components and 512 px tiled VAE encode/decode, batch 5 finished
# with peak VRAM 2.1 GB (DiT GPU) / 1.6 GB (VAE GPU); the previous plan
# (blocks 20, no io swap, encode NOT tiled) OOMed on the VAE GPU during
# encoding (9.5 GiB allocated + 2.5 GiB requested), a stage that raising
# blocks_to_swap can never fix - which is why all three earlier retries
# failed the same way.
SEEDVR2_BLOCKS_TO_SWAP = 36
SEEDVR2_TILE = 512
SEEDVR2_TILE_OVERLAP = 64
SEEDVR2_MIN_TILE = 256


# ------------------------------------------------------------------ ffprobe --
def find_ffprobe() -> str:
    """PATH first; else the ffprobe next to whatever ffmpeg merge.py resolved."""
    found = shutil.which("ffprobe")
    if found:
        return found
    from . import merge as merge_mod
    ffmpeg = merge_mod.find_ffmpeg()
    candidate = Path(ffmpeg).with_name(Path(ffmpeg).name.replace("ffmpeg", "ffprobe"))
    if candidate.is_file():
        return str(candidate)
    raise PipelineError(
        "動画を解析するための ffprobe が見つかりませんでした。"
        "ffmpeg一式（ffprobeを含む）をインストールしてPATHに追加してください。",
        kind="missing_model", stage=STAGE_NAMES[STAGE_INPUT])


def _parse_rate(text) -> float | None:
    if not text:
        return None
    try:
        if "/" in str(text):
            num, den = str(text).split("/")
            den = float(den)
            return float(num) / den if den else None
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def probe(path: str | Path) -> dict:
    """{"width","height","fps","frames","duration","has_audio"} via ffprobe."""
    exe = find_ffprobe()
    args = [exe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path)]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:                                    # noqa: BLE001
            pass
        raise PipelineError("動画の解析が時間内に終わりませんでした。",
                            kind="timeout", stage=STAGE_NAMES[STAGE_INPUT])
    if proc.returncode != 0:
        raise PipelineError(
            "動画を解析できませんでした（ffprobe）: "
            + (err or b"").decode("utf-8", errors="replace")[:300],
            kind="missing_model", stage=STAGE_NAMES[STAGE_INPUT])
    data = json.loads((out or b"{}").decode("utf-8", errors="replace") or "{}")
    streams = data.get("streams") or []
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if vstream is None:
        raise PipelineError("動画にビデオストリームが見つかりません。",
                            kind="missing_model", stage=STAGE_NAMES[STAGE_INPUT])
    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    fps = _parse_rate(vstream.get("r_frame_rate") or vstream.get("avg_frame_rate"))
    frames = _int_or_none(vstream.get("nb_frames"))
    duration = _float_or_none(vstream.get("duration")) or \
        _float_or_none((data.get("format") or {}).get("duration"))
    if not frames and fps and duration:
        frames = int(round(fps * duration))
    return {"width": width, "height": height, "fps": fps or 0.0,
           "frames": frames or 0, "duration": duration or 0.0,
           "has_audio": astream is not None}


# ------------------------------------------------------------------ geometry --
def _round_even(x: float) -> int:
    n = int(round(x))
    if n % 2:
        n += 1
    return max(2, n)


def target_dims(width: int, height: int, preset: str) -> tuple[int, int]:
    """Aspect-ratio-preserving target size, even-rounded on every axis."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid source dimensions")
    if preset == "x2":
        return _round_even(width * 2), _round_even(height * 2)
    if preset == "1080x1920":
        short = 1080
        if width <= height:
            return short, _round_even(short * height / width)
        return _round_even(short * width / height), short
    raise ValueError(f"unknown preset: {preset!r}")


def _short_long(w: int, h: int) -> tuple[int, int]:
    return (w, h) if w <= h else (h, w)


# --------------------------------------------------------------- GPU devices --
def _video_device(_gpu_plan: dict | None) -> str:
    """DiT / ESRGAN device: always ComfyUI's cuda:0 (the video GPU's slot)."""
    return "cuda:0"


def _seedvr2_devices(gpu_plan: dict | None) -> tuple[str, str, str]:
    """(dit_device, vae_device, offload_device).

    No GPU UUIDs here: only the ComfyUI-side device index that gpu.build_plan's
    `effective_mode` puts the video GPU / aux GPU into (cuda:0 / cuda:1).
    """
    dual = (gpu_plan or {}).get("effective_mode") == "dual"
    return "cuda:0", ("cuda:1" if dual else "cuda:0"), "cpu"


# --------------------------------------------------------- filesystem probes --
def seedvr2_search_dirs(cfg) -> list[Path]:
    """Where SeedVR2 weights might live: default + comfy_paths.yaml (best effort).

    Mirrors pipeline.Pipeline._loras_dir's regex approach - never raises, and
    never invents a directory that is not either the ComfyUI default or an
    explicit entry already present in comfy_paths.yaml.
    """
    dirs = [Path(cfg.comfy_dir) / "models" / "SEEDVR2"]
    try:
        text = Path(cfg.paths_yaml).read_text(encoding="utf-8")
        base_m = re.search(r"base_path:\s*['\"]([^'\"]+)['\"]", text)
        dir_m = re.search(r"SEEDVR2:\s*['\"]([^'\"]+)['\"]", text)
        if base_m and dir_m:
            cand = Path(base_m.group(1)) / dir_m.group(1)
            if cand not in dirs:
                dirs.append(cand)
    except Exception:                                        # noqa: BLE001
        pass
    return dirs


def seedvr2_model_files(cfg) -> dict:
    """Best-effort filesystem probe: {"dit": Path|None, "vae": Path|None}.

    Deliberately filesystem-only (never the ComfyUI model registry): the
    registry lists names that are not on disk and would auto-download.
    """
    dit = vae = None
    for d in seedvr2_search_dirs(cfg):
        try:
            if dit is None and (d / SEEDVR2_DIT_FILE).is_file():
                dit = d / SEEDVR2_DIT_FILE
            if vae is None and (d / SEEDVR2_VAE_FILE).is_file():
                vae = d / SEEDVR2_VAE_FILE
        except OSError:
            continue
    return {"dit": dit, "vae": vae}


def esrgan_model_path(cfg) -> Path | None:
    """Best-effort: the shared upscale_models folder from comfy_paths.yaml."""
    try:
        text = Path(cfg.paths_yaml).read_text(encoding="utf-8")
        base_m = re.search(r"base_path:\s*['\"]([^'\"]+)['\"]", text)
        dir_m = re.search(r"upscale_models:\s*['\"]([^'\"]+)['\"]", text)
        if base_m and dir_m:
            cand = Path(base_m.group(1)) / dir_m.group(1) / ESRGAN_MODEL
            if cand.is_file():
                return cand
    except Exception:                                        # noqa: BLE001
        pass
    return None


def model_files_for(cfg) -> dict:
    """One filesystem probe covering both methods (used by /api/upscale/options)."""
    seedvr2 = seedvr2_model_files(cfg)
    return {"esrgan": esrgan_model_path(cfg),
           "seedvr2_dit": seedvr2["dit"], "seedvr2_vae": seedvr2["vae"]}


def _combo_options(object_info: dict | None, class_type: str, field: str) -> list:
    node = (object_info or {}).get(class_type) or {}
    required = ((node.get("input") or {}).get("required") or {})
    schema = required.get(field)
    if not schema:
        return []
    if len(schema) > 1 and isinstance(schema[1], dict) and "options" in schema[1]:
        return list(schema[1]["options"])
    if schema and isinstance(schema[0], list):
        return list(schema[0])
    return []


def _missing_node(name: str) -> dict:
    return {"kind": "node", "name": name, "source_url": pack_for(name),
           "license": "", "size_bytes": 0}


def _missing_model(name: str, source_url: str, license_: str, size_bytes: int) -> dict:
    return {"kind": "model", "name": name, "source_url": source_url,
           "license": license_, "size_bytes": size_bytes}


def _missing_input(name: str) -> dict:
    return {"kind": "input", "name": name, "source_url": "", "license": "", "size_bytes": 0}


# -------------------------------------------------------------------- plan() --
def plan(source_probe: dict, target: dict, gpu_plan: dict,
         object_info: dict | None, model_files: dict) -> dict:
    """Pure planning: no filesystem, no network - `model_files` is the caller's
    own probe (see model_files_for). Never raises; unusable input becomes a
    "missing" entry and ok=False."""
    result: dict[str, Any] = {
        "ok": False, "target_w": 0, "target_h": 0, "scale": 0.0,
        "method": target.get("method"), "preset": target.get("preset"),
        "graph_inputs": {}, "missing": [], "warnings": [],
    }
    width = int(source_probe.get("width") or 0)
    height = int(source_probe.get("height") or 0)
    preset = str(target.get("preset") or "")
    method = str(target.get("method") or "")
    if width <= 0 or height <= 0:
        result["missing"].append(_missing_input("動画の解像度を取得できませんでした。"))
        return result
    if preset not in PRESETS:
        result["missing"].append(_missing_input(f"不明な解像度指定です: {preset}"))
        return result
    if method not in METHODS:
        result["missing"].append(_missing_input(f"不明な方式です: {method}"))
        return result

    target_w, target_h = target_dims(width, height, preset)
    result["target_w"], result["target_h"] = target_w, target_h
    result["scale"] = round(((target_w * target_h) / float(width * height)) ** 0.5, 4)

    missing: list[dict] = []
    object_info = object_info or {}
    model_files = model_files or {}

    if method == "lanczos":
        # Plain ffmpeg scale: no ComfyUI nodes, no models, nothing to install.
        result["graph_inputs"] = {
            "target_w": target_w, "target_h": target_h, "filter": "lanczos",
        }
    elif method == "esrgan":
        # Always the core node: it upscales in 512 px tiles and halves the
        # tile on OOM. kjnodes' ImageUpscaleWithModelBatched runs whole
        # frames through the network - measured on the real machine, a
        # 576x1024 frame x4 asked for 11.25 GiB at per_batch 5 and still
        # OOMed at per_batch 1 on a 12 GB card.
        upscale_node = "ImageUpscaleWithModel"
        for n in ("LoadVideo", "GetVideoComponents", "UpscaleModelLoader",
                 upscale_node, "ImageScale", "CreateVideo", "SaveVideo"):
            if n not in object_info:
                missing.append(_missing_node(n))
        options = _combo_options(object_info, "UpscaleModelLoader", "model_name")
        if not model_files.get("esrgan") and ESRGAN_MODEL not in options:
            missing.append(_missing_model(
                ESRGAN_MODEL, ESRGAN_SOURCE_URL, ESRGAN_LICENSE, ESRGAN_SIZE_BYTES))
        result["graph_inputs"] = {
            "upscale_node": upscale_node, "upscale_model": ESRGAN_MODEL,
            "per_batch": 5, "device": _video_device(gpu_plan),
            "target_w": target_w, "target_h": target_h,
        }
    else:  # seedvr2
        for n in ("SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler",
                 "LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo"):
            if n not in object_info:
                missing.append(_missing_node(n))
        if not model_files.get("seedvr2_dit"):
            missing.append(_missing_model(
                SEEDVR2_DIT_FILE, SEEDVR2_SOURCE_URL, SEEDVR2_LICENSE, SEEDVR2_DIT_SIZE_BYTES))
        if not model_files.get("seedvr2_vae"):
            missing.append(_missing_model(
                SEEDVR2_VAE_FILE, SEEDVR2_SOURCE_URL, SEEDVR2_LICENSE, SEEDVR2_VAE_SIZE_BYTES))
        dit_device, vae_device, offload_device = _seedvr2_devices(gpu_plan)
        short_edge, long_edge = _short_long(target_w, target_h)
        result["graph_inputs"] = {
            "dit_model": SEEDVR2_DIT_FILE, "vae_model": SEEDVR2_VAE_FILE,
            "dit_device": dit_device, "vae_device": vae_device,
            "offload_device": offload_device, "blocks_to_swap": SEEDVR2_BLOCKS_TO_SWAP,
            "swap_io_components": True,
            "encode_tiled": True, "decode_tiled": True,
            "tile_size": SEEDVR2_TILE, "tile_overlap": SEEDVR2_TILE_OVERLAP,
            "batch_size": 5, "temporal_overlap": 2,
            "color_correction": "lab", "resolution": short_edge,
            "max_resolution": long_edge, "target_w": target_w, "target_h": target_h,
        }

    result["missing"] = missing
    result["ok"] = not missing
    return result


def _relax_for_oom(plan_result: dict) -> dict:
    """seedvr2: everything the DiT side can give is already given (all blocks
    swapped), so an OOM is a VAE/activation problem: halve the VAE tile (min
    256) and drop the batch to 1 (4n+1 rule). esrgan: per_batch halved (min 1)."""
    updated = json.loads(json.dumps(plan_result))
    gi = updated["graph_inputs"]
    if updated.get("method") == "seedvr2":
        gi["blocks_to_swap"] = MAX_BLOCKS_TO_SWAP
        gi["swap_io_components"] = True
        gi["encode_tiled"] = True
        gi["decode_tiled"] = True
        gi["tile_size"] = max(SEEDVR2_MIN_TILE, int(gi.get("tile_size", SEEDVR2_TILE)) // 2)
        gi["batch_size"] = 1
    else:
        gi["per_batch"] = max(1, int(gi.get("per_batch", 5)) // 2)
    return updated


def _is_oom(exc: BaseException) -> bool:
    if getattr(exc, "kind", "") == "oom":
        return True
    info = classify(exc)
    return info.get("kind") == "oom"


# --------------------------------------------------------------- build_graph --
def build_graph(plan_result: dict, source_video_name: str, prefix: str) -> dict:
    gi = plan_result["graph_inputs"]
    method = plan_result["method"]
    graph: dict[str, Any] = {
        "load_video": {"class_type": "LoadVideo", "inputs": {"file": source_video_name}},
        "components": {"class_type": "GetVideoComponents",
                       "inputs": {"video": ["load_video", 0]}},
    }
    if method == "esrgan":
        graph["upscale_model"] = {"class_type": "UpscaleModelLoader",
                                  "inputs": {"model_name": gi["upscale_model"]}}
        # kjnodes' batched node names its frames input "images"; the core
        # ImageUpscaleWithModel names it "image" (real-machine finding: the
        # graph was rejected with required_input_missing "images").
        if gi["upscale_node"] == "ImageUpscaleWithModelBatched":
            upscale_inputs = {"upscale_model": ["upscale_model", 0],
                              "images": ["components", 0], "per_batch": gi["per_batch"]}
        else:
            upscale_inputs = {"upscale_model": ["upscale_model", 0],
                              "image": ["components", 0]}
        graph["upscale"] = {"class_type": gi["upscale_node"], "inputs": upscale_inputs}
        graph["scale"] = {"class_type": "ImageScale",
                          "inputs": {"image": ["upscale", 0],
                                    "width": gi["target_w"], "height": gi["target_h"],
                                    "upscale_method": "lanczos", "crop": "disabled"}}
        graph["create_video"] = {"class_type": "CreateVideo",
                                 "inputs": {"images": ["scale", 0], "fps": ["components", 2]}}
    else:
        tile = int(gi.get("tile_size", SEEDVR2_TILE))
        overlap = int(gi.get("tile_overlap", SEEDVR2_TILE_OVERLAP))
        graph["dit"] = {"class_type": "SeedVR2LoadDiTModel",
                        "inputs": {"model": gi["dit_model"], "device": gi["dit_device"],
                                  "blocks_to_swap": gi["blocks_to_swap"],
                                  "swap_io_components": bool(gi.get("swap_io_components", True)),
                                  "offload_device": gi["offload_device"], "cache_model": False}}
        graph["vae"] = {"class_type": "SeedVR2LoadVAEModel",
                        "inputs": {"model": gi["vae_model"], "device": gi["vae_device"],
                                  "encode_tiled": bool(gi.get("encode_tiled", True)),
                                  "encode_tile_size": tile, "encode_tile_overlap": overlap,
                                  "decode_tiled": gi["decode_tiled"],
                                  "decode_tile_size": tile, "decode_tile_overlap": overlap,
                                  "offload_device": gi["offload_device"]}}
        graph["upscale"] = {"class_type": "SeedVR2VideoUpscaler",
                            "inputs": {"image": ["components", 0], "dit": ["dit", 0],
                                      "vae": ["vae", 0], "seed": 0,
                                      "resolution": gi["resolution"],
                                      "max_resolution": gi["max_resolution"],
                                      "batch_size": gi["batch_size"],
                                      "uniform_batch_size": True,
                                      "color_correction": gi["color_correction"],
                                      "temporal_overlap": gi["temporal_overlap"],
                                      "offload_device": gi["offload_device"]}}
        graph["create_video"] = {"class_type": "CreateVideo",
                                 "inputs": {"images": ["upscale", 0], "fps": ["components", 2]}}
    # format/codec are required by SaveVideo.execute() even though the
    # prompt validator accepts their absence (real-machine finding: the
    # whole 13-minute ESRGAN pass was lost at the save step). Same values
    # as the generation graph in graphs.py.
    graph["save_video"] = {"class_type": "SaveVideo",
                           "inputs": {"video": ["create_video", 0], "filename_prefix": prefix,
                                     "format": "auto", "codec": "auto"}}
    return graph


# --------------------------------------------------------------------- Runner --
class UpscaleJob:
    """Minimal Project-shaped placeholder: .id / .data / .save() only.

    Deliberately never persisted to disk - upscale jobs are not projects and
    must not appear under app/projects/ or app/story_projects/.
    """

    def __init__(self, job_id: str):
        self.id = job_id
        self.data: dict = {"status": "running"}

    def save(self) -> None:
        return None


class UpscaleRunner(Runner):
    """pipeline.Runner with upscale-specific stage names and job_type."""

    def __init__(self, pipeline: Pipeline, job: UpscaleJob):
        super().__init__(pipeline, job)
        self.state["job_type"] = "upscale"
        self.state["stages"] = [{"name": n, "status": "待機"} for n in STAGE_NAMES]
        self.state["result"] = None

    def fail(self, exc: BaseException, stage_index: int) -> None:
        stages = self.state["stages"]
        if not (0 <= stage_index < len(stages)):
            stage_index = len(stages) - 1
        stage_name = stages[stage_index]["name"]
        stages[stage_index]["status"] = "失敗"
        self.state["error"] = classify(exc, stage=stage_name)
        self.state["finished"] = True
        self.emit()


# ------------------------------------------------------------------- verify --
def _verify(src_probe: dict, out_probe: dict, plan_result: dict) -> str:
    """"" on pass, else a Japanese reason (never raises)."""
    target_w = plan_result.get("target_w") or 0
    target_h = plan_result.get("target_h") or 0
    out_w, out_h = int(out_probe.get("width") or 0), int(out_probe.get("height") or 0)
    if target_w and target_h:
        if out_w <= 0 or out_h <= 0:
            return "書き出された動画の解像度を確認できませんでした。"
        if abs((out_w / out_h) - (target_w / target_h)) > 0.02:
            return (f"アスペクト比が一致しません（期待: {target_w}x{target_h}、"
                   f"実際: {out_w}x{out_h}）。")
    src_fps = float(src_probe.get("fps") or 0)
    out_fps = float(out_probe.get("fps") or 0)
    if src_fps and out_fps and abs(src_fps - out_fps) > 0.1:
        return f"フレームレートが一致しません（元: {src_fps:.2f}、書き出し: {out_fps:.2f}）。"
    src_frames = int(src_probe.get("frames") or 0)
    out_frames = int(out_probe.get("frames") or 0)
    if src_frames and out_frames and abs(src_frames - out_frames) > 1:
        return f"フレーム数が一致しません（元: {src_frames}、書き出し: {out_frames}）。"
    src_dur = float(src_probe.get("duration") or 0)
    out_dur = float(out_probe.get("duration") or 0)
    if src_dur and out_dur and abs(src_dur - out_dur) > max(0.5, src_dur * 0.05):
        return f"動画の長さが一致しません（元: {src_dur:.1f}秒、書き出し: {out_dur:.1f}秒）。"
    if src_probe.get("has_audio") and not out_probe.get("has_audio"):
        return "音声が含まれていません。"
    return ""


async def _run_ffmpeg(args: list[str], *, timeout: float = 600) -> tuple[int, str]:
    """Local subprocess runner (own copy - merge.py's _run is WP-C-owned)."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:                                    # noqa: BLE001
            pass
        return 1, "timeout"
    return proc.returncode, (out or b"").decode("utf-8", errors="replace")


async def _run_ffmpeg_cancellable(args: list[str], cancel: asyncio.Event, *,
                                  timeout: float = 1800) -> tuple[int, str]:
    """Same as _run_ffmpeg but polls `cancel` and kills the process on it.

    Never calls Process.communicate() more than once (repeated calls on a
    partially-consumed pipe are unsafe) - output is read once, after wait()
    settles one way or another.
    """
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    cancel_task = asyncio.ensure_future(cancel.wait())
    wait_task = asyncio.ensure_future(proc.wait())
    try:
        done, pending = await asyncio.wait(
            {cancel_task, wait_task}, timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if wait_task in done:
            out = await proc.stdout.read() if proc.stdout else b""
            return proc.returncode, out.decode("utf-8", errors="replace")
        if cancel_task in done:
            try:
                proc.kill()
            except Exception:                                # noqa: BLE001
                pass
            await proc.wait()
            raise UserCancelled(stage=STAGE_NAMES[STAGE_UPSCALE])
        # Neither finished before the timeout.
        try:
            proc.kill()
        except Exception:                                    # noqa: BLE001
            pass
        await proc.wait()
        return 1, "timeout"
    finally:
        for t in (cancel_task, wait_task):
            if not t.done():
                t.cancel()


# -------------------------------------------------------- run(): lanczos ----
async def _run_lanczos(pipeline: Pipeline, src: Path, plan_result: dict,
                       runner: UpscaleRunner) -> dict:
    """Standard upscale: `ffmpeg -vf scale=...:flags=lanczos`, no ComfyUI.

    Never calls client.ensure_ready/run_graph/free_models or
    pipeline._release_local_llm, and never stages the source into
    comfy_inputs - this path exists precisely so a user without any upscale
    model installed can still get a resolution change.
    """
    cfg = pipeline.config
    runner.set_stage(STAGE_INPUT)
    src_probe = await probe(src)

    runner.set_stage(STAGE_UPSCALE)
    gi = plan_result["graph_inputs"]
    target_w, target_h = int(gi["target_w"]), int(gi["target_h"])
    tag = "1080p" if plan_result.get("preset") == "1080x1920" else "x2"
    job_tag = uuid.uuid4().hex[:8]
    work_dir = cfg.debug_dir / "upscale_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / f"{src.stem}_{tag}_{job_tag}.mp4"

    from . import merge as merge_mod
    ffmpeg = merge_mod.find_ffmpeg()
    # -nostats/-v error: the output pipe is only drained after ffmpeg exits
    # (see _run_ffmpeg_cancellable), so per-frame progress spam on a long
    # clip could fill the pipe and stall the process.
    args = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-v", "error", "-y",
            "-i", str(src),
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p"]
    args += ["-c:a", "copy"] if src_probe.get("has_audio") else ["-an"]
    args += ["-movflags", "+faststart", str(out_path)]

    code, log = await _run_ffmpeg_cancellable(args, runner.cancel)

    runner.set_stage(STAGE_EXPORT)
    if code != 0:
        raise PipelineError(
            "標準拡大（ffmpeg）でエラーが発生しました: " + log[-2000:],
            kind="unknown", stage=STAGE_NAMES[STAGE_UPSCALE])

    stem = f"{src.stem}_upscaled_{tag}"
    export = media_mod.export_video(
        out_path, dest_dir=cfg.media_videos_final, filename=stem,
        media_root=cfg.media_root, kind="final")

    result: dict[str, Any] = {
        "ok": False, "output": str(out_path), "export": export,
        "method": "lanczos",
        "elapsed_s": round(time.monotonic() - runner.started, 1),
        "peak_vram_mib": 0, "debug": {"retries": []}, "error": "",
    }
    if not export.get("ok"):
        result["error"] = f"完成動画フォルダへの保存に失敗しました（{export.get('error')}）。"
        runner.state["result"] = result
        runner.finish()
        return result

    out_probe = await probe(export["path"])
    reason = _verify(src_probe, out_probe, plan_result)
    if reason:
        result["error"] = reason
        runner.state["result"] = result
        runner.finish()
        return result

    result["ok"] = True
    runner.state["result"] = result
    runner.finish()
    return result


# ----------------------------------------------------------------------- run --
async def run(pipeline: Pipeline, source_path: str | Path, plan_result: dict,
              runner: UpscaleRunner) -> dict:
    """Execute one upscale job end to end.

    Raises PipelineError only for genuine execution failures that survive
    every retry (missing ffmpeg/ffprobe, ComfyUI errors, cancellation). A
    verification mismatch after a successful export is reported as
    result["ok"]=False with result["error"] set - it is never raised, and it
    is never shown to the user as a success.
    """
    src = Path(source_path)
    if not src.is_file():
        raise PipelineError("元の動画が見つかりません。", kind="input_missing",
                            stage=STAGE_NAMES[STAGE_INPUT])

    if plan_result.get("method") == "lanczos":
        # Standard upscale: plain ffmpeg, no ComfyUI, no models, no staging
        # into comfy_inputs. Never falls back to an AI method on failure.
        return await _run_lanczos(pipeline, src, plan_result, runner)

    cfg = pipeline.config
    client = pipeline.client

    runner.set_stage(STAGE_INPUT)
    src_probe = await probe(src)

    # Same staging path the story runner uses for h3app_story_*.mp4: the app
    # never writes into ComfyUI's own input folder, only into the staging
    # dir, and run_graph()'s InputUploader delivers it via /upload/image.
    staged_name = f"h3app_upscale_{uuid.uuid4().hex}.mp4"
    staged_path = comfy_inputs.stage_path(cfg, staged_name)
    shutil.copy2(src, staged_path)

    await pipeline._release_local_llm()
    try:
        await client.free_models()
    except PipelineError:
        pass  # best effort: "nothing to free" must never fail the job

    debug: dict[str, Any] = {"retries": []}
    current_plan = json.loads(json.dumps(plan_result))
    job_tag = uuid.uuid4().hex[:8]
    prefix = f"h3upscale/{runner.project.id}_{job_tag}"

    peak_vram_mib = 0

    async def _poll_vram(stop: asyncio.Event) -> None:
        nonlocal peak_vram_mib
        while not stop.is_set():
            try:
                stats = await client.system_stats()
                summary = comfy_mod.ComfyClient.vram_summary(stats)
                used = sum(int(d.get("used_mib") or 0) for d in summary)
                peak_vram_mib = max(peak_vram_mib, used)
            except Exception:                                # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

    def on_event(ev: dict) -> None:
        node = ev.get("node") or ""
        if ev["type"] == "executing" and node:
            idx = _NODE_STAGE.get(node)
            if idx is not None:
                runner.set_stage(idx)
        elif ev["type"] == "progress":
            idx = _NODE_STAGE.get(node)
            if idx == STAGE_UPSCALE and ev.get("max"):
                runner.state["percent"] = int(100 * ev["value"] / max(1, ev["max"]))
                runner.emit()

    history: dict | None = None
    attempt = 0
    while True:
        graph = build_graph(current_plan, staged_name, prefix)
        stop_poll = asyncio.Event()
        poll_task = asyncio.get_running_loop().create_task(_poll_vram(stop_poll))
        try:
            history = await client.run_graph(
                graph, on_event=on_event, cancel=runner.cancel, profile=None)
            break
        except PipelineError as exc:
            if getattr(exc, "kind", "") != "cancelled" and _is_oom(exc) and \
                    attempt < MAX_OOM_RETRIES:
                attempt += 1
                current_plan = _relax_for_oom(current_plan)
                debug["retries"].append({
                    "attempt": attempt, "reason": "oom",
                    "graph_inputs": dict(current_plan["graph_inputs"]),
                })
                continue
            if debug["retries"]:
                # Retries exhausted (or a non-retryable error after some
                # retries): keep the retry trail on the surfaced error so
                # the failure display can show what was already attempted.
                tried = ", ".join(
                    f"#{r['attempt']} {r['reason']}" for r in debug["retries"])
                exc.detail = f"{exc.detail}\n[upscale retries exhausted: {tried}]"
            raise
        finally:
            stop_poll.set()
            await poll_task

    runner.set_stage(STAGE_EXPORT)
    saved = comfy_mod.saved_video(history, "save_video")
    if saved is None:
        raise PipelineError("アップスケール結果を取得できませんでした。",
                            kind="unknown", stage=STAGE_NAMES[STAGE_EXPORT])
    comfy_output_path = comfy_mod.output_path(cfg.comfy_output, saved)

    tag = "1080p" if current_plan.get("preset") == "1080x1920" else "x2"
    work_dir = cfg.debug_dir / "upscale_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    muxed_path = comfy_output_path
    if src_probe.get("has_audio"):
        from . import merge as merge_mod
        try:
            ffmpeg = merge_mod.find_ffmpeg()
            candidate = work_dir / f"{src.stem}_{tag}_{job_tag}.mp4"
            code, _log = await _run_ffmpeg([
                ffmpeg, "-hide_banner", "-nostdin", "-y",
                "-i", str(comfy_output_path), "-i", str(src),
                "-map", "0:v", "-map", "1:a", "-c", "copy", "-shortest",
                str(candidate)])
            if code == 0 and candidate.is_file() and candidate.stat().st_size > 0:
                muxed_path = candidate
        except PipelineError:
            pass  # fall back to the video-only ComfyUI output

    stem = f"{src.stem}_upscaled_{tag}"
    export = media_mod.export_video(
        muxed_path, dest_dir=cfg.media_videos_final, filename=stem,
        media_root=cfg.media_root, kind="final")

    result: dict[str, Any] = {
        "ok": False, "output": str(muxed_path), "export": export,
        "method": current_plan.get("method"),
        "elapsed_s": round(time.monotonic() - runner.started, 1),
        "peak_vram_mib": peak_vram_mib, "debug": debug, "error": "",
    }
    if not export.get("ok"):
        result["error"] = f"完成動画フォルダへの保存に失敗しました（{export.get('error')}）。"
        runner.state["result"] = result
        runner.finish()
        return result

    out_probe = await probe(export["path"])
    reason = _verify(src_probe, out_probe, current_plan)
    if reason:
        result["error"] = reason
        runner.state["result"] = result
        runner.finish()
        return result

    result["ok"] = True
    runner.state["result"] = result
    runner.finish()
    return result


# ------------------------------------------------------------- source lookup --
def resolve_source(cfg, store, story_store, source: dict) -> tuple[Path | None, str]:
    """Validate a requested source; never raises. Returns (path, error)."""
    kind = str((source or {}).get("kind") or "")
    if kind == "file":
        name = str(source.get("name") or "")
        base = Path(name).name
        if not base or base != name or base in (".", ".."):
            return None, "不正なファイル名です。"
        for root in (cfg.media_videos_final, cfg.media_videos_clips):
            candidate = root / base
            try:
                resolved = candidate.resolve()
                root_resolved = root.resolve()
                if resolved == root_resolved or root_resolved in resolved.parents:
                    if resolved.is_file():
                        return resolved, ""
            except (OSError, ValueError):
                continue
        return None, "指定された動画が見つかりません。"

    if kind in ("project", "story"):
        obj_id = str(source.get("id") or "")
        obj = (store.get(obj_id) if kind == "project" else story_store.get(obj_id)) \
            if (store if kind == "project" else story_store) is not None else None
        if obj is None:
            return None, ("プロジェクトが見つかりません。" if kind == "project"
                          else "ストーリーが見つかりません。")
        if kind == "story" and source.get("final"):
            final = str(obj.data.get("final_video") or "")
            path = Path(final) if final else None
            if path is not None and path.is_file():
                return path, ""
            return None, "完成動画が見つかりません。"
        step = source.get("step")
        clip = None
        clips = obj.clips
        if step is not None:
            clip = next((c for c in clips if int(c.get("step", -1)) == int(step)), None)
        elif clips:
            clip = clips[-1]
        if clip is None:
            return None, "対象のクリップが見つかりません。"
        export = clip.get("export") or {}
        path_str = export["path"] if export.get("ok") and export.get("path") else \
            (clip.get("local_video") or clip.get("video") or "")
        path = Path(path_str) if path_str else None
        if path is None or not path.is_file():
            return None, "対象の動画ファイルが見つかりません。"
        return path, ""

    return None, "不明な指定です。"
