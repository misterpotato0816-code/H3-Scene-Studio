# -*- coding: utf-8 -*-
"""Phase 2/9 harness: A/B generation benchmark on an isolated ComfyUI.

Mirrors the verified v1 runner (ports 8401/8402, whitelist, never kills
existing processes). One clip per arm, identical prompt/seed/refs:

  <comfy venv python> backend/h3_v2/ab_run.py [--arm legacy|fast|both]

Writes BenchmarkRecord JSONL + built graphs. No v1 file is touched.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(BACKEND_ROOT.parent))

import aiohttp  # noqa: E402  (comfy venv provides it; fail loudly otherwise)

from h3_v2.backend import build  # noqa: E402
from h3_v2.benchmark import BenchmarkRecord, append_jsonl  # noqa: E402
from h3_v2.feature_flags import FeatureFlags  # noqa: E402
from h3_v2.lora_bind import count_bindable_keys, read_tensor_names  # noqa: E402
from h3_v2.presets import PRESETS  # noqa: E402

COMFY = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI")
COMFY_PY = COMFY / ".venv" / "Scripts" / "python.exe"
COMFY_MAIN = COMFY / "main.py"
PATHS_YAML = APP_ROOT / "comfy_paths.yaml"  # app copy: shared models + H3-Device-Barrier
PATHS_RND = BACKEND_ROOT / "paths_rnd.yaml"  # + h3v2pipe R&D nodes (flag-gated)
WHITELIST = ["H3-Device-Barrier", "comfyui_layerstyle", "comfyui-kjnodes"]
WHITELIST_RND_EXTRA = ["h3v2pipe"]
LORAS_DIR = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models\loras")
PROMPT_FILE = PROJECT_ROOT / "_hetero_test" / "gold" / "gold_final_prompt.txt"
BENCH_DIR = PROJECT_ROOT / "benchmarks"
REF_LONGEST = 1536  # harness default; --ref-longest overrides (loop tool)
GEN_DIMS: tuple[int, int] | None = None  # --dims WxH overrides preset dims
PROMPT_FILE_OVERRIDE: str = ""  # --prompt-file (probe tool, recorded in tag)
SAGE_OFF: bool = False  # --sage-off probe (spec mandates sage; off = experiment)
CLIP_OVERRIDE: str = ""  # --clip FILE (TE swap experiment, recorded in tag)
NO_LASTFRAME: bool = False  # --no-lastframe (0f without motion-anchor image)
STEPS_OVERRIDE: int = 0  # --steps N (off-recipe probe, recorded in tag)
SAVE_LATENT: bool = False  # --save-latent (custom raw saver for pipeline R&D)
USE_CNODES: bool = False  # --cnode (load h3v2pipe on the isolated server)
SERVER_EXTRA_ARGS: list[str] = []  # --server-args: extra ComfyUI launch flags
PROBE_LATENT: bool = False  # --probe-latent (attach structure probe node)
VOICE_MODE: str = "ext"  # ext|auto|off (external VM preferred, auto fallback)
EXT_VM_ASSET = BACKEND_ROOT / "assets" / "VoiceMaster_v1.wav"
EXT_VM_INPUT = "V2_VM_EXT.wav"

# v2 descriptive graph ids -> benchmark stage. EVERY builder node id must map
# somewhere: unmapped nodes land in "other" and are recorded as t_other.
# Relay nodes (tail chain) are their own stage, never mixed into conditioning.
STAGE_OF = {
    "restore": "load", "clip": "load", "clip_report_base": "load",
    "clip_report_target": "load", "clip_select": "load",
    "vae_video": "load", "vae_audio": "load", "unet": "load", "lora": "load",
    "lowvram": "load", "chunkff": "load", "sigshift": "load", "sage": "load",
    "h3": "conditioning", "noise": "sampling", "guider": "sampling",
    "sampler_select": "sampling", "scheduler": "sampling", "sampler": "sampling",
    "decode_video": "vae", "decode_audio": "audio",
    "prev_video": "relay", "prev_components": "relay",
    "tail_images": "relay", "tail_audio": "relay", "voice_master": "relay",
    "last_frame": "conditioning",
    "concat_images": "merge", "concat_audio": "merge",
    "create_video": "merge", "create_video_seg": "merge",
    "save_video": "merge", "save_video_seg": "merge",
    "save_latent_raw": "merge", "h3_latent_probe": "other",
    "anchor_slice": "relay", "anchor_dec": "vae", "anchor_save": "merge",
    "voice_extract": "relay",
}
# Reference loaders are conditioning input prep (per-image work, scales with
# reference count) — classified explicitly, never "other".
STAGE_PREFIX = (("load_", "conditioning"), ("scale_", "conditioning"))


def stage_of(node_id: str) -> str:
    if node_id in STAGE_OF:
        return STAGE_OF[node_id]
    for prefix, stage in STAGE_PREFIX:
        if node_id.startswith(prefix):
            return stage
    return "other"

ARMS = ("legacy", "fast", "balanced", "quality", "lowvram")
ARM_PRESET = {"legacy": "LEGACY_V1", "fast": "FAST",
              "balanced": "BALANCED", "quality": "QUALITY",
              "lowvram": "LOW_VRAM"}
UNVERIFIED_ARMS = ("balanced", "quality", "lowvram")  # measured, then flipped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def poll_smi(samples: list, stop: asyncio.Event):
    async def _run():
        while not stop.is_set():
            try:
                p = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE)
                out, _ = await p.communicate()
                mem = {}
                for line in out.decode().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = float(parts[1].strip())
                samples.append({"t": time.time(), "gpu": mem})
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
    return _run()


async def wait_server(session: aiohttp.ClientSession, base: str) -> dict:
    for _ in range(150):
        try:
            async with session.get(f"{base}/object_info", timeout=5) as r:
                if r.status == 200:
                    return await r.json()
        except Exception:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("ComfyUI server did not become ready")


def count_lora_warnings(err_path: Path, offset: int) -> tuple[int, list[str]]:
    if not err_path.is_file():
        return 0, []
    with err_path.open("rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if "lora key not loaded" in l]
    return len(lines), lines[:5]


async def run_arm(arm: str, port: int, log_path: Path, err_path: Path) -> BenchmarkRecord:
    preset_id = ARM_PRESET[arm]
    preset = PRESETS[preset_id]
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    images = ["h3test_face_ref.png"]
    longest = [REF_LONGEST, 1152, 1152, 896]
    out_prefix = f"H3/V2_AB_{arm}"

    graph = build(preset_id, images, prompt_text, longest, out_prefix,
                  flags=FeatureFlags(),
                  allow_unverified=(arm in ("fast",) + UNVERIFIED_ARMS))
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    (BENCH_DIR / f"graph_{arm}.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    lora_hash, bindable = _check_lora(arm, preset, images)
    if isinstance(lora_hash, BenchmarkRecord):  # bind proof failed -> record
        return lora_hash
    return await _submit(arm, preset, graph, port, err_path, bindable,
                         lora_hash, ref_count=len(images))


async def run_long(port: int, log_path: Path, err_path: Path,
                 frames: int = 124, clips: int = 2,
                 tag: str = "long", preset_id: str = "LONG",
                 tail_n: int = 56, voice_master: bool = False) -> list[BenchmarkRecord]:
    """Phase 8 / V3: N clips chained by Tail Relay.

    Tail relay only (concat_previous=False, like story mode); the final
    assembly is an ffmpeg concat afterwards.
    V3 LONG_FAST additions: tail_n (0 = last-frame-image relay only),
    voice_master (fixed voice ref from clip 0 as ref_audio_0 on clips 1+,
    tail audio disconnected — violation is a GraphError, never silent).
    """
    """Phase 8 / finalization: N LONG clips chained by Tail Relay.

    Tail relay only (concat_previous=False, like story mode); the final
    assembly is an ffmpeg concat afterwards. Uses LONG preset as defined
    (base 8-step). Proves the v2 backend drives continuation graphs and
    records relay/drift evidence.
    """
    from h3_v2.presets import PresetError
    import dataclasses
    preset = PRESETS[preset_id]
    override = None
    if SAGE_OFF and preset.sage:
        override = dataclasses.replace(preset, sage=False)
        print(f"[{tag}] SAGE-OFF probe (spec mandates sage; experiment only)",
              flush=True)
    if CLIP_OVERRIDE:
        override = dataclasses.replace(override or preset, clip=CLIP_OVERRIDE)
        print(f"[{tag}] CLIP override -> {CLIP_OVERRIDE} (experiment, recorded)",
              flush=True)
    if STEPS_OVERRIDE:
        override = dataclasses.replace(override or preset, steps=STEPS_OVERRIDE)
        print(f"[{tag}] STEPS override -> {STEPS_OVERRIDE} (off-recipe probe)",
              flush=True)
    if STEPS_OVERRIDE:
        import dataclasses as _dc
        preset = _dc.replace(preset, steps=STEPS_OVERRIDE)
        print(f"[{tag}] STEPS override -> {STEPS_OVERRIDE} (off-recipe probe)",
              flush=True)
    if preset_id == "LONG_FAST":
        flags = FeatureFlags(long_fast=True)
    else:
        flags = FeatureFlags(long_video=True)
    try:
        from h3_v2.presets import get_preset
        get_preset(preset_id, flags)
    except PresetError as exc:
        print(f"[{tag}] gated as expected pre-test: {exc}", flush=True)
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    if PROMPT_FILE_OVERRIDE:
        prompt_text = Path(PROMPT_FILE_OVERRIDE).read_text(encoding="utf-8")
        print(f"[{tag}] prompt override: {PROMPT_FILE_OVERRIDE} "
              f"({len(prompt_text)} chars, recorded)", flush=True)
    images = ["h3test_face_ref.png"]
    longest = [REF_LONGEST, 1152, 1152, 896]
    t_wall0 = time.perf_counter()
    # External Voice Master (preferred): one frozen file for every clip,
    # making the voice reference fully invariant. Auto-extract is fallback.
    ext_vm: str | None = None
    if VOICE_MODE == "ext" and preset_id in ("LONG_FAST",):
        if not EXT_VM_ASSET.is_file():
            raise RuntimeError(f"external Voice Master missing: {EXT_VM_ASSET}")
        dest = COMFY / "input" / EXT_VM_INPUT
        dest.write_bytes(EXT_VM_ASSET.read_bytes())
        ext_vm = EXT_VM_INPUT
        print(f"[{tag}] external Voice Master staged: {EXT_VM_INPUT}", flush=True)
    # Header-level LoRA bind proof (same gate as single arms; a missing LoRA
    # must fail here, never as a silent slow run).
    eff = override or preset
    lora_hash, bindable = _check_lora(tag, eff, images)
    if isinstance(lora_hash, BenchmarkRecord):
        return [lora_hash]
    recs: list[BenchmarkRecord] = []
    prev_name: str | None = None
    prev_frames = frames
    for k in range(clips):
        label = f"{tag}_{chr(97 + k)}"
        # Resume: an existing output file for this clip is reused, never rebuilt.
        existing = sorted((COMFY / "output" / "H3").glob(f"V2_AB_{label}_*.mp4"),
                          key=lambda p: p.stat().st_mtime)
        if existing:
            src = existing[-1]
            dest = COMFY / "input" / f"V2_AB_{label}.mp4"
            dest.write_bytes(src.read_bytes())
            prev_name, prev_frames = dest.name, frames
            if voice_master and VOICE_MODE == "auto" and k == 0 and \
                    not (COMFY / "input" / f"V2_VM_{tag}.wav").is_file():
                # Resume of clip 0 must still mint the Voice Master.
                _extract_voice_master(tag, 0)
            rec = BenchmarkRecord(preset=preset.id, model=preset.model,
                                  sampler=preset.sampler,
                                  scheduler=preset.scheduler, steps=preset.steps,
                                  width=preset.width, height=preset.height,
                                  frames=frames,
                                  ref_count=len(images), seed=123456789,
                                  backend="h3_v2", comfyui_version="0.34.5",
                                  success=True, error="reused prior run",
                                  output_path=str(src))
            recs.append(rec)
            print(f"[{tag}] clip {label} reused: {src.name}", flush=True)
            continue
        cont = _longfast_continuation(tag, k, prev_name, prev_frames, tail_n,
                                      voice_master and ext_vm is None)
        graph = build(preset_id, images, prompt_text, longest,
                      f"H3/V2_AB_{label}", cont, concat_previous=False,
                      flags=flags, allow_unverified=True, frames=frames,
                      preset_obj=override, voice_master=ext_vm)
        if GEN_DIMS:
            graph["h3"]["inputs"]["width"], graph["h3"]["inputs"]["height"] = GEN_DIMS
            print(f"[{tag}] dims override -> {GEN_DIMS} (recorded, preset untouched)",
                  flush=True)
        if preset_id == "LONG_FAST" and SAVE_LATENT:
            # Latent handoff for the GPU1 VAE pipeline (opt-in: stock
            # SaveLatent cannot handle the H3 NestedTensor; the custom
            # H3SaveLatentRaw node in backend/h3_v2/h3v2pipe/ is required).
            graph["save_latent_raw"] = {
                "class_type": "H3SaveLatentRaw",
                "inputs": {"samples": ["sampler", 0],
                           "filename_prefix": f"H3/V2_AB_{label}"},
            }
        if preset_id == "LONG_FAST" and PROBE_LATENT:
            graph["h3_latent_probe"] = {
                "class_type": "H3LatentProbe",
                "inputs": {"samples": ["sampler", 0]},
            }
        (BENCH_DIR / f"graph_{label}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        rec = await _submit(label, eff, graph, port, err_path, bindable,
                            lora_hash, ref_count=len(images))
        # _submit does not know preset dims/overridden frames; fix the record.
        rec.frames = frames
        rec.width, rec.height = (GEN_DIMS or (eff.width, eff.height))
        recs.append(rec)
        if not rec.success or not rec.output_path:
            return recs
        src = Path(rec.output_path)
        if not src.is_absolute():
            src = COMFY / "output" / "H3" / src.name
        dest = COMFY / "input" / f"V2_AB_{label}.mp4"
        dest.write_bytes(src.read_bytes())
        prev_name, prev_frames = dest.name, frames
        if voice_master and VOICE_MODE == "auto" and k == 0:
            _extract_voice_master(tag, 0)
        if tail_n == 0:
            _extract_last_frame(tag, label, src)
    # Assemble the full-length video (ffmpeg concat, no re-encode).
    lst = BENCH_DIR / f"{tag}_concat.txt"
    lst.write_text("".join(
        f"file '{COMFY / 'input' / f'V2_AB_{tag}_{chr(97 + k)}.mp4'}'\n"
        for k in range(clips)), encoding="utf-8")
    final = COMFY / "output" / "H3" / f"V2_AB_{tag}_full.mp4"
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(final))
    await proc.communicate()
    print(f"[{tag}] assembled {final} exists={final.is_file()}", flush=True)
    wall = time.perf_counter() - t_wall0
    orch = round(wall - sum(r.t_total for r in recs), 1)
    (BENCH_DIR / f"orch_{tag}.json").write_text(json.dumps({
        "tag": tag, "wall_clock": round(wall, 1),
        "submit_total": round(sum(r.t_total for r in recs), 1),
        "t_orchestration": orch,
        "clips": len(recs),
        "final": str(final),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] WALL={wall:.1f}s orchestration={orch:.1f}s", flush=True)
    return recs


async def run_anchor(port: int, log_path: Path, err_path: Path,
                     frames: int = 124, tag: str = "anchor") -> BenchmarkRecord:
    """Phase 2A: one submit decodes full (GT) + tail slices C in {0,4,8}.

    Proves the minimal temporal context for the anchor fast path against the
    full-decode ground truth. No new downloads, one GPU run for all variants.
    """
    preset = PRESETS["LONG_FAST"]
    flags = FeatureFlags(long_fast=True)
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    images = ["h3test_face_ref.png"]
    longest = [REF_LONGEST, 1152, 1152, 896]
    label = f"{tag}_a"
    graph = build("LONG_FAST", images, prompt_text, longest,
                  f"H3/V2_AB_{label}", None, concat_previous=False,
                  flags=flags, allow_unverified=True, frames=frames)
    if GEN_DIMS:
        graph["h3"]["inputs"]["width"], graph["h3"]["inputs"]["height"] = GEN_DIMS
    # Anchor branches: K=1 latent frame + C context frames, decoded in-graph
    # next to the full GT decode. Same submit, same conditions.
    for c in (0, 4, 8):
        graph[f"slice_c{c}"] = {
            "class_type": "H3LatentTailSlice",
            "inputs": {"samples": ["sampler", 0],
                       "tail_latent_frames": 1,
                       "context_latent_frames": c},
        }
        graph[f"dec_c{c}"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [f"slice_c{c}", 0],
                       "vae": ["vae_video", 0]},
        }
        graph[f"save_c{c}"] = {
            "class_type": "SaveImage",
            "inputs": {"images": [f"dec_c{c}", 0],
                       "filename_prefix": f"H3/V2_AB_{label}_c{c}"},
        }
    (BENCH_DIR / f"graph_{label}.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    lora_hash, bindable = _check_lora(tag, preset, images)
    if isinstance(lora_hash, BenchmarkRecord):
        return lora_hash
    return await _submit(label, preset, graph, port, err_path, bindable,
                         lora_hash, ref_count=len(images))


def _longfast_continuation(tag: str, k: int, prev_name: str | None,
                           prev_frames: int, tail_n: int,
                           voice_master: bool) -> dict | None:
    """Clip-k continuation for the V3 recipe. Clip 0 (k=0) is always fresh."""
    if k == 0 or not prev_name:
        return None
    cont: dict = {"video": prev_name, "prev_total_frames": prev_frames,
                  "tail_frames": tail_n, "tail_audio": False}
    if tail_n == 0 and not NO_LASTFRAME:
        # 0f relay: previous last frame rides as an extra reference image.
        cont["last_frame_image"] = f"V2_LF_{tag}_{chr(97 + k - 1)}.png"
    if voice_master:
        # Fixed voice from clip 0 as <Audio 1>; tail audio stays disconnected
        # (the builder raises GraphError on any violation — never silent).
        cont["voice_master"] = f"V2_VM_{tag}.wav"
    return cont


def _extract_voice_master(tag: str, k: int) -> Path:
    """Fixed Voice Master: first 5 s of clip audio as ref_audio_0 material."""
    src = COMFY / "input" / f"V2_AB_{tag}_{chr(97 + k)}.mp4"
    dest = COMFY / "input" / f"V2_VM_{tag}.wav"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src), "-ss", "0", "-t", "5", "-vn",
                    "-acodec", "pcm_s16le", "-ar", "32000", "-ac", "2",
                    str(dest)], check=True)
    return dest


def _extract_last_frame(tag: str, label: str, src: Path) -> Path:
    """Previous last frame PNG for 0f relay (motion anchor image)."""
    dest = COMFY / "input" / f"V2_LF_{label}.png"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                    "-sseof", "-0.1", "-i", str(src),
                    "-frames:v", "1", str(dest)], check=True)
    return dest


def measure_audio(path: Path) -> dict:
    """Per-clip audio evidence: duration + loudness (no ears needed for this)."""
    proc = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True)
    out = proc.stderr or ""
    info: dict = {"file": Path(path).name}
    for line in out.splitlines():
        if "Duration:" in line:
            info["duration"] = line.split("Duration:")[1].split(",")[0].strip()
        if "mean_volume:" in line:
            info["mean_db"] = line.split("mean_volume:")[1].split("dB")[0].strip()
        if "max_volume:" in line:
            info["max_db"] = line.split("max_volume:")[1].split("dB")[0].strip()
    return info


def strip_decode_for_pipeline(graph: dict, label: str) -> dict:
    """Decode-free submit for server A: sampling + anchor + raw latent only.

    Full VAE decode + mp4 assembly move to server B (GPU1). The anchor
    (K=1/C=0 partial decode, ~1s, GT-equivalent) unblocks the next clip.
    """
    for nid in ("decode_video", "decode_audio", "create_video", "save_video",
                "create_video_seg", "save_video_seg",
                "save_latent_raw", "h3_latent_probe"):
        graph.pop(nid, None)
    graph["anchor_slice"] = {
        "class_type": "H3LatentTailSlice",
        "inputs": {"samples": ["sampler", 0],
                   "tail_latent_frames": 1, "context_latent_frames": 0},
    }
    graph["anchor_dec"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["anchor_slice", 0], "vae": ["vae_video", 0]},
    }
    graph["anchor_save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["anchor_dec", 0],
                   "filename_prefix": f"H3/V2_AB_{label}_anchor"},
    }
    graph["save_latent_raw"] = {
        "class_type": "H3SaveLatentRaw",
        "inputs": {"samples": ["sampler", 0],
                   "filename_prefix": f"H3/V2_AB_{label}"},
    }
    return graph


def build_decode_graph(latent_path: str, label: str) -> dict:
    """Server-B graph (GPU1): raw latent -> full decode -> mp4. No DiT/TE."""
    return {
        "load_raw": {"class_type": "H3LoadLatentRaw",
                     "inputs": {"filepath": latent_path}},
        "vae_video": {"class_type": "VAELoader",
                      "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "vae_audio": {"class_type": "VAELoader",
                      "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "decode_video": {"class_type": "VAEDecode",
                         "inputs": {"samples": ["load_raw", 0],
                                    "vae": ["vae_video", 0]}},
        "decode_audio": {"class_type": "VAEDecodeAudio",
                         "inputs": {"samples": ["load_raw", 0],
                                    "vae": ["vae_audio", 0]}},
        "create_video": {"class_type": "CreateVideo",
                         "inputs": {"fps": 24.0, "bit_depth": 8,
                                    "images": ["decode_video", 0],
                                    "audio": ["decode_audio", 0]}},
        "save_video": {"class_type": "SaveVideo",
                       "inputs": {"filename_prefix": f"H3/V2_AB_{label}",
                                  "format": "auto", "codec": "auto",
                                  "video": ["create_video", 0]}},
    }


async def wait_for_file(path: Path, timeout_s: int = 120) -> Path:
    for _ in range(timeout_s * 2):
        if path.is_file() and path.stat().st_size > 0:
            return path
        await asyncio.sleep(0.5)
    raise TimeoutError(f"anchor/output never appeared: {path}")


async def run_pipeline(port_a: int, port_b: int, log_path: Path, err_path: Path,
                       frames: int = 362, clips: int = 2,
                       tag: str = "pipe") -> list[BenchmarkRecord]:
    """Phase-2 2-server pipeline (ANCHOR-FIRST DECODE).

    Server A (GPU0): cond + sampling + anchor + raw latent (no full decode).
    Server B (GPU1): full VAE decode + mp4, overlapped with A's next submit:
    while A samples clip k+1, B decodes clip k. Full decoded frames never
    return to GPU0; B completes files on disk. The last decode + ffmpeg
    assembly stay exposed (honest wall clock).
    """
    preset = PRESETS["LONG_FAST"]
    flags = FeatureFlags(long_fast=True)
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    images = ["h3test_face_ref.png"]
    longest = [REF_LONGEST, 1152, 1152, 896]
    t_wall0 = time.perf_counter()
    lora_hash, bindable = _check_lora(tag, preset, images)
    if isinstance(lora_hash, BenchmarkRecord):
        return [lora_hash]
    if VOICE_MODE == "ext":
        if not EXT_VM_ASSET.is_file():
            raise RuntimeError(f"external Voice Master missing: {EXT_VM_ASSET}")
        (COMFY / "input" / EXT_VM_INPUT).write_bytes(
            EXT_VM_ASSET.read_bytes())
        ext_vm: str | None = EXT_VM_INPUT
    else:
        ext_vm = None
    err_b = BENCH_DIR / "ab_server_b.err"
    recs: list[BenchmarkRecord] = []
    latents: list[str] = []
    prev_name: str | None = None
    overlap_saved = 0.0

    async def submit_a(k: int, prev: str | None):
        label = f"{tag}_{chr(97 + k)}"
        if k == 0 or not prev:
            cont, vm = None, ext_vm
        else:
            cont = {"tail_frames": 0, "tail_audio": False,
                    "last_frame_image": prev}
            vm = ext_vm
        graph = build("LONG_FAST", images, prompt_text, longest,
                      f"H3/V2_AB_{label}", cont, concat_previous=False,
                      flags=flags, allow_unverified=True, frames=frames,
                      voice_master=vm)
        if GEN_DIMS:
            graph["h3"]["inputs"]["width"], graph["h3"]["inputs"]["height"] = GEN_DIMS
        graph = strip_decode_for_pipeline(graph, label)
        (BENCH_DIR / f"graph_{label}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        rec = await _submit(label, preset, graph, port_a, err_path, bindable,
                            lora_hash, ref_count=len(images))
        rec.frames = frames
        rec.width, rec.height = (GEN_DIMS or (preset.width, preset.height))
        rec.backend = "h3_v2-pipe-A"
        return rec, label

    async def submit_b_decode(k: int):
        label = f"{tag}_{chr(97 + k)}"
        latent = sorted((COMFY / "output" / "latents").glob(
            f"V2_AB_{label}_*.h3latent.pt"),
            key=lambda p: p.stat().st_mtime)[-1]
        graph = build_decode_graph(str(latent), label)
        (BENCH_DIR / f"graph_{label}_dec.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        rec = await _submit(label, preset, graph, port_b, err_b, bindable,
                            lora_hash, ref_count=len(images))
        rec.frames = frames
        rec.width, rec.height = (GEN_DIMS or (preset.width, preset.height))
        rec.backend = "h3_v2-pipe-B"
        return rec

    def stage_anchor(label: str) -> str:
        anchor = sorted((COMFY / "output" / "H3").glob(f"V2_AB_{label}_anchor_*.png"),
                        key=lambda p: p.stat().st_mtime)[-1]
        (COMFY / "input" / f"V2_LF_{label}.png").write_bytes(anchor.read_bytes())
        return f"V2_LF_{label}.png"

    async def submit_b_logged(k: int):
        rec = await submit_b_decode(k)
        append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
        return rec

    # Clip 0 on A (fresh). Nothing to overlap with yet.
    rec_a0, label_a0 = await submit_a(0, None)
    append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec_a0)
    recs.append(rec_a0)
    if not rec_a0.success:
        return recs
    latents.append(str(sorted((COMFY / "output" / "latents").glob(
        f"V2_AB_{label_a0}_*.h3latent.pt"),
        key=lambda p: p.stat().st_mtime)[-1]))
    prev_name = stage_anchor(label_a0)

    # Middle clips: A-clip-k OVERLAPS B-decode-(k-1) (asyncio.gather).
    for k in range(1, clips):
        (rec_a, label_a), rec_b = await asyncio.gather(
            submit_a(k, prev_name),
            submit_b_logged(k - 1),
        )
        recs.append(rec_a)
        recs.append(rec_b)
        if not rec_a.success or not rec_b.success:
            return recs
        overlap_saved += min(rec_b.t_total, rec_a.t_total)
        latents.append(str(sorted((COMFY / "output" / "latents").glob(
            f"V2_AB_{label_a}_*.h3latent.pt"),
            key=lambda p: p.stat().st_mtime)[-1]))
        prev_name = stage_anchor(label_a)

    # Last clip's decode stays exposed (honest wall clock).
    rec_blast = await submit_b_logged(clips - 1)
    recs.append(rec_blast)

    # Final assembly: ffmpeg concat of B's mp4s, stream copy when safe.
    parts = []
    for k in range(clips):
        cands = sorted((COMFY / "output" / "H3").glob(f"V2_AB_{tag}_{chr(97 + k)}_*.mp4"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise RuntimeError(f"[{tag}] B mp4 missing for clip {k}")
        parts.append(cands[-1])
    lst = BENCH_DIR / f"{tag}_concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    final = COMFY / "output" / "H3" / f"V2_AB_{tag}_full.mp4"
    proc = await asyncio.create_subprocess_exec(
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(final))
    await proc.communicate()
    wall = time.perf_counter() - t_wall0
    submit_sum = sum(r.t_total for r in recs)
    (BENCH_DIR / f"orch_{tag}.json").write_text(json.dumps({
        "tag": tag, "wall_clock": round(wall, 1),
        "submit_total": round(submit_sum, 1),
        "t_orchestration": round(wall - submit_sum, 1),
        "overlap_saved_est": round(overlap_saved, 1),
        "clips": clips, "final": str(final),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] WALL={wall:.1f}s overlap_saved~{overlap_saved:.1f}s",
          flush=True)
    return recs


def _ffmpeg() -> str:
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        package_root = (Path(local_appdata) / "Microsoft" / "WinGet" /
                        "Packages" /
                        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe")
        candidates = sorted(
            package_root.glob("ffmpeg-*-full_build/bin/ffmpeg.exe"),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(
        "ffmpeg was not found on PATH or in the standard WinGet package directory.")


def _check_lora(arm: str, preset, ref_count_imgs: list):
    """Header-level bind proof. Returns (lora_hash, bindable) or a fail record."""
    lora_hash, bindable = "", 0
    if preset.lora:
        lp = LORAS_DIR / preset.lora
        bindable = count_bindable_keys(read_tensor_names(str(lp)))
        print(f"[{arm}] LoRA header: {bindable} bindable keys "
              f"(min required {preset.min_bindable_keys})", flush=True)
        if bindable < preset.min_bindable_keys:
            return BenchmarkRecord(
                preset=preset.id, model=preset.model, lora=preset.lora,
                success=False, error=f"TURBO_UNAVAILABLE: {bindable} bindable keys",
                seed=123456789, width=576, height=1024, frames=124,
                steps=preset.steps, sampler=preset.sampler,
                scheduler=preset.scheduler,
                ref_count=len(ref_count_imgs)), 0
        lora_hash = sha256_file(lp)
        print(f"[{arm}] LoRA sha256: {lora_hash[:16]}...", flush=True)
    return lora_hash, bindable


async def _submit(label: str, preset, graph: dict, port: int,
                  err_path: Path, bindable: int, lora_hash: str,
                  ref_count: int = 1) -> BenchmarkRecord:
    arm = label  # prints + output glob use the run label

    base = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}/ws"
    err_offset = err_path.stat().st_size if err_path.is_file() else 0
    samples: list = []
    stop = asyncio.Event()
    node_dur: dict[str, float] = {}
    node_start: dict[str, float] = {}
    current: str | None = None
    prompt_id = None
    rec = BenchmarkRecord(
        preset=preset.id, model=preset.model, lora=preset.lora,
        lora_hash=lora_hash, lora_strength=preset.lora_strength,
        lora_bindable_keys=bindable, sampler=preset.sampler,
        scheduler=preset.scheduler, steps=preset.steps,
        shift_video=preset.shift_video, shift_audio=preset.shift_audio,
        width=576, height=1024, frames=124, ref_count=ref_count,
        seed=123456789, backend="h3_v2", comfyui_version="0.34.5")
    try:
        async with aiohttp.ClientSession() as session:
            await wait_server(session, base)
            client_id = str(uuid.uuid4())
            async with session.ws_connect(f"{ws_url}?clientId={client_id}",
                                          heartbeat=30) as ws:
                t0 = time.perf_counter()
                async with session.post(
                        f"{base}/prompt",
                        json={"prompt": graph, "client_id": client_id}) as r:
                    payload = await r.json()
                    if r.status != 200 or "prompt_id" not in payload:
                        raise RuntimeError(f"prompt rejected: {r.status}: {payload}")
                    prompt_id = payload["prompt_id"]
                print(f"[{arm}] submitted {prompt_id}", flush=True)
                smi_task = asyncio.create_task(poll_smi(samples, stop))
                last_progress = time.perf_counter()
                try:
                    while True:
                        try:
                            msg = await ws.receive(timeout=20)
                        except asyncio.TimeoutError:
                            if time.perf_counter() - last_progress > 600:
                                raise TimeoutError(f"stalled at {current}")
                            continue
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            # ANY message (incl. progress heartbeats during long
                            # sampling) proves the server is alive.
                            last_progress = time.perf_counter()
                            if data.get("data", {}).get("prompt_id") not in (None, prompt_id):
                                continue
                            if data.get("type") == "executing":
                                now = last_progress
                                if current in node_start:
                                    node_dur[current] = now - node_start[current]
                                current = data["data"].get("node")
                                if current is None:
                                    break
                                node_start[current] = now
                                print(f"[{arm}] node {current}: "
                                      f"{graph.get(current, {}).get('class_type')}",
                                      flush=True)
                            elif data.get("type") in ("execution_error",
                                                      "execution_interrupted"):
                                raise RuntimeError(f"{data.get('type')}: {data}")
                        elif msg.type in (aiohttp.WSMsgType.ERROR,
                                          aiohttp.WSMsgType.CLOSED):
                            raise RuntimeError("websocket ended early")
                finally:
                    stop.set()
                    await smi_task
                rec.t_total = time.perf_counter() - t0
                async with session.get(f"{base}/history/{prompt_id}") as r:
                    history = await r.json()
                outs = (history.get(prompt_id, {}).get("outputs", {}) or {})
                for node_out in outs.values():
                    for f in (node_out.get("gifs") or []) + (node_out.get("videos") or []):
                        if f.get("filename"):
                            rec.output_path = f["filename"]
                if not rec.output_path:
                    # Fallback: newest file matching this arm's prefix in the
                    # default ComfyUI output dir (history shape varies).
                    out_dir = COMFY / "output" / "H3"
                    cands = sorted(out_dir.glob(f"V2_AB_{arm}_*.mp4"),
                                   key=lambda p: p.stat().st_mtime)
                    if cands:
                        rec.output_path = str(cands[-1])
    except Exception as exc:  # noqa: BLE001 - harness must record, not crash
        rec.success = False
        rec.error = f"{type(exc).__name__}: {exc}"
        stop.set()
        return rec
    by_stage: dict[str, float] = {}
    unmapped: dict[str, float] = {}
    for node, dur in node_dur.items():
        st = stage_of(node)
        by_stage[st] = by_stage.get(st, 0.0) + dur
        if st == "other":
            unmapped[node] = round(dur, 1)
    if unmapped:
        # Never judge performance with unclassified nodes: loud, recorded.
        print(f"[{arm}] UNMAPPED NODES -> t_other: {unmapped}", flush=True)
    rec.t_model_load = round(by_stage.get("load", 0.0), 1)
    rec.t_conditioning = round(by_stage.get("conditioning", 0.0), 1)
    rec.t_sampling = round(by_stage.get("sampling", 0.0), 1)
    rec.t_vae_decode = round(by_stage.get("vae", 0.0), 1)
    rec.t_audio_decode = round(by_stage.get("audio", 0.0), 1)
    rec.t_other = round(by_stage.get("other", 0.0), 1)
    rec.t_merge = round(by_stage.get("merge", 0.0), 1)
    rec.t_relay = round(by_stage.get("relay", 0.0), 1)
    if unmapped:
        rec.error = (rec.error + " " if rec.error else "") + \
            f"unmapped_nodes={sorted(unmapped)}"
    g0 = [s["gpu"].get("0", 0) for s in samples if "0" in s.get("gpu", {})]
    g1 = [s["gpu"].get("1", 0) for s in samples if "1" in s.get("gpu", {})]
    rec.gpu0_peak_vram_mb = max(g0) if g0 else 0
    rec.gpu1_peak_vram_mb = max(g1) if g1 else 0
    warn_count, warn_sample = count_lora_warnings(err_path, err_offset)
    print(f"[{arm}] 'lora key not loaded' warnings since submit: {warn_count}",
          flush=True)
    # Verdict vs the preset's own proof requirement: LEGACY_V1 EXPECTS the
    # no-op (min 0, 518 warnings = parity confirmation). FAST requires binding.
    if bindable >= preset.min_bindable_keys:
        rec.success = True
    else:
        rec.success = False
        rec.error = (f"TURBO_UNAVAILABLE: {bindable} bindable keys "
                     f"< required {preset.min_bindable_keys} "
                     f"({warn_count} unbound-key warnings at runtime)")
    if warn_sample:
        print(f"[{arm}] sample: {warn_sample[0][:120]}", flush=True)
    return rec


def launch_server(port: int, log_path: Path, err_path: Path,
                   rnd_nodes: bool = False,
                   default_device: str = "0",
                   full_nodes: bool = False):
    import socket
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use - refusing to kill anything")
    wl = WHITELIST + (WHITELIST_RND_EXTRA if rnd_nodes else [])
    import os as _os
    # Overnight 2026-09-07 W1 finding: post-VLM VRAM fragmentation spills the
    # video VAE decode onto GPU1 (124f vae 61s -> 159s). expandable_segments
    # reduces allocator fragmentation. Opt-in per call, default off.
    extra_env: dict[str, str] = {}
    if _os.environ.get("H3_CUDA_ALLOC_CONF"):
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = _os.environ["H3_CUDA_ALLOC_CONF"]
    args = [str(COMFY_PY), "-X", "utf8", str(COMFY_MAIN),
            "--port", str(port), "--default-device", default_device,
            "--reserve-vram", "1.0",
            *SERVER_EXTRA_ARGS,
            "--extra-model-paths-config",
            str(PATHS_RND if rnd_nodes else PATHS_YAML)]
    if not full_nodes:
        # Bench isolation (default): only the packs the measured graphs need.
        # full_nodes=True reproduces the production app server (all packs).
        args += ["--disable-all-custom-nodes",
                 "--whitelist-custom-nodes", *wl]
    log_f = log_path.open("w", encoding="utf-8")
    err_f = err_path.open("w", encoding="utf-8")
    import os as _os2
    env = dict(_os2.environ)
    env.update(extra_env)
    return subprocess.Popen(args, cwd=str(COMFY), stdout=log_f, stderr=err_f,
                            env=env)


async def amain(arms: list[str], port: int) -> int:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BENCH_DIR / "ab_server.log"
    err_path = BENCH_DIR / "ab_server.err"
    procs: list = []
    try:
        if arms == ["pipe362"]:
            # Phase-2 pipeline: server A (GPU0) + server B (GPU1, decode).
            port_b = 8403
            busy = __import__("socket").socket()
            try:
                busy.bind(("127.0.0.1", port_b))
                busy.close()
            except OSError:
                raise RuntimeError(f"Port {port_b} is already in use - refusing to kill anything")
            print("launching isolated ComfyUI A (:8402/GPU0) + B (:8403/GPU1)...",
                  flush=True)
            procs.append(launch_server(port, log_path, err_path,
                                       rnd_nodes=True))
            procs.append(launch_server(
                port_b, BENCH_DIR / "ab_server_b.log",
                BENCH_DIR / "ab_server_b.err",
                rnd_nodes=True, default_device="1"))
            print("===== Phase-2 pipeline 2x362f @480x864 =====", flush=True)
            global GEN_DIMS
            GEN_DIMS = (480, 864)  # production resolution per directive
            ok = True
            for rec in await run_pipeline(port, port_b, log_path, err_path,
                                          frames=362, clips=2, tag="pipe362"):
                print(f"[pipe362] {rec.backend} total={rec.t_total:.1f}s "
                      f"samp={rec.t_sampling:.1f}s vae={rec.t_vae_decode:.1f}s "
                      f"success={rec.success} out={rec.output_path} {rec.error}",
                      flush=True)
                ok = ok and rec.success
            return 0 if ok else 1
        print("launching isolated ComfyUI...", flush=True)
        procs.append(launch_server(port, log_path, err_path,
                                   rnd_nodes=USE_CNODES))
        if arms == ["longfastanchor"]:
            print("===== Phase 2A anchor GT comparison =====", flush=True)
            rec = await run_anchor(port, log_path, err_path)
            append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
            print(f"[anchor] total={rec.t_total:.1f}s success={rec.success} "
                  f"out={rec.output_path} {rec.error}", flush=True)
            return 0 if rec.success else 1
        if arms == ["long"]:
            print("===== ARM long (2-clip tail relay) =====", flush=True)
            for rec in await run_long(port, log_path, err_path):
                append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
                print(f"[long] total={rec.t_total:.1f}s sampling={rec.t_sampling:.1f}s "
                      f"success={rec.success} out={rec.output_path} {rec.error}",
                      flush=True)
                if not rec.success:
                    return 1
            return 0
        if arms == ["long30a"]:
            print("===== 30s shootout A: 10s x 3 (243f) =====", flush=True)
            ok = True
            for rec in await run_long(port, log_path, err_path, frames=243,
                                      clips=3, tag="long30a"):
                append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
                print(f"[long30a] total={rec.t_total:.1f}s sampling={rec.t_sampling:.1f}s "
                      f"success={rec.success} out={rec.output_path} {rec.error}",
                      flush=True)
                ok = ok and rec.success
            return 0 if ok else 1
        if arms == ["long30b"]:
            print("===== 30s shootout B: 15s x 2 (362f) =====", flush=True)
            ok = True
            for rec in await run_long(port, log_path, err_path, frames=362,
                                      clips=2, tag="long30b"):
                append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
                print(f"[long30b] total={rec.t_total:.1f}s sampling={rec.t_sampling:.1f}s "
                      f"success={rec.success} out={rec.output_path} {rec.error}",
                      flush=True)
                ok = ok and rec.success
            return 0 if ok else 1
        lf = {"longfast0f": (124, 2, 0, "lf0f"),
              "longfast5f": (124, 2, 5, "lf5f"),
              "longfast22f": (124, 2, 22, "lf22f"),
              "longfast243": (243, 2, 0, "lf243"),
              "longfast362": (362, 2, 0, "lf362"),
              "longfast243x3": (243, 3, 0, "lf243x3"),
              "longfast124x6": (124, 6, 0, "lf124x6"),
              "longfastprobe": (124, 1, 0, "probe")}.get(arms[0]) \
            if len(arms) == 1 else None
        if lf:
            frames, clips, tail_n, tag = lf
            # Loop-tool overrides get their own evidence namespace so they
            # can never collide with (or silently reuse) other runs.
            if REF_LONGEST != 1536:
                tag = f"{tag}_rl{REF_LONGEST}"
            if GEN_DIMS:
                tag = f"{tag}_{GEN_DIMS[0]}x{GEN_DIMS[1]}"
            if PROMPT_FILE_OVERRIDE:
                tag = f"{tag}_pshort"
            if SAGE_OFF:
                tag = f"{tag}_nosage"
            if CLIP_OVERRIDE:
                tag = f"{tag}_te4b"
            if NO_LASTFRAME:
                tag = f"{tag}_nolf"
            if STEPS_OVERRIDE:
                tag = f"{tag}_s{STEPS_OVERRIDE}"
            print(f"===== V3 LONG_FAST {tag}: {frames}f x {clips}, "
                  f"tail={tail_n}f + voice master =====", flush=True)
            ok = True
            for rec in await run_long(port, log_path, err_path, frames=frames,
                                      clips=clips, tag=tag,
                                      preset_id="LONG_FAST", tail_n=tail_n,
                                      voice_master=(VOICE_MODE != "off")):
                append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
                print(f"[{tag}] total={rec.t_total:.1f}s sampling={rec.t_sampling:.1f}s "
                      f"relay={rec.t_relay:.1f}s other={rec.t_other:.1f}s "
                      f"success={rec.success} out={rec.output_path} {rec.error}",
                      flush=True)
                ok = ok and rec.success
            return 0 if ok else 1
        for arm in arms:
            print(f"===== ARM {arm} =====", flush=True)
            rec = await run_arm(arm, port, log_path, err_path)
            append_jsonl(BENCH_DIR / "v2_bench.jsonl", rec)
            print(f"[{arm}] total={rec.t_total:.1f}s sampling={rec.t_sampling:.1f}s "
                  f"success={rec.success} out={rec.output_path} {rec.error}",
                  flush=True)
            if not rec.success:
                print(f"[{arm}] STOPPING A/B: arm failed", flush=True)
                return 1
        return 0
    finally:
        for proc in procs:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()


def print_report(store: Path) -> int:
    from h3_v2.benchmark import ab_table, load_jsonl
    rows = ab_table([r for r in load_jsonl(store) if r.success])
    if not rows:
        print("no successful records")
        return 1
    print(f"{'preset':<10} {'total':>8} {'samp':>7} {'vae':>6} {'other':>6} {'relay':>6} {'gpu0':>7}  output")
    for r in rows:
        print(f"{r['preset']:<10} {r['total_sec']:>7.1f}s {r['sampling_sec']:>6.1f}s "
              f"{r.get('t_vae_decode', 0):>5.1f}s {r.get('t_other', 0):>5.1f}s "
              f"{r.get('t_relay', 0):>5.1f}s "
              f"{r['gpu0_peak_mb']:>6.0f}M  {r['output']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="both",
                    choices=("both", "legacy", "fast", "balanced", "quality",
                             "lowvram", "long", "long30a", "long30b",
                             "longfast0f", "longfast5f", "longfast22f",
                             "longfast243", "longfast362", "longfastprobe",
                             "longfast243x3", "longfast124x6", "longfastanchor",
                             "pipe362",
                             "phase3", "phase4", "report"))
    ap.add_argument("--port", type=int, default=8402)
    ap.add_argument("--store", default=str(BENCH_DIR / "v2_bench.jsonl"))
    ap.add_argument("--ref-longest", type=int, default=1536)
    ap.add_argument("--dims", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--sage-off", action="store_true")
    ap.add_argument("--clip", default="")
    ap.add_argument("--no-lastframe", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--save-latent", action="store_true")
    ap.add_argument("--cnode", action="store_true",
                    help="load h3v2pipe R&D nodes (isolated server only)")
    ap.add_argument("--probe-latent", action="store_true",
                    help="attach H3LatentProbe to LONG_FAST graphs (needs --cnode)")
    ap.add_argument("--voice", default="ext",
                    choices=("ext", "auto", "off"),
                    help="ext=frozen external Voice Master (both clips), "
                         "auto=extract from clip 0, off=no voice master")
    ap.add_argument("--server-args", default="",
                    help="extra flags appended to the isolated ComfyUI launch "
                         "(e.g. --server-args=--fast). Recorded in server log.")
    args = ap.parse_args()
    global REF_LONGEST, GEN_DIMS, PROMPT_FILE_OVERRIDE, SAGE_OFF, CLIP_OVERRIDE
    global NO_LASTFRAME, STEPS_OVERRIDE, VOICE_MODE, SAVE_LATENT
    global USE_CNODES, PROBE_LATENT, SERVER_EXTRA_ARGS
    REF_LONGEST = int(args.ref_longest)
    PROMPT_FILE_OVERRIDE = args.prompt_file
    SAGE_OFF = bool(args.sage_off)
    CLIP_OVERRIDE = args.clip
    NO_LASTFRAME = bool(args.no_lastframe)
    STEPS_OVERRIDE = int(args.steps)
    VOICE_MODE = args.voice
    SAVE_LATENT = bool(args.save_latent)
    USE_CNODES = bool(args.cnode)
    PROBE_LATENT = bool(args.probe_latent)
    SERVER_EXTRA_ARGS = [a for a in args.server_args.split()] if args.server_args else []
    if PROBE_LATENT and not USE_CNODES:
        raise RuntimeError("--probe-latent needs --cnode")
    if args.dims:
        w, h = args.dims.lower().split("x")
        GEN_DIMS = (int(w), int(h))
    if args.arm == "report":
        return print_report(Path(args.store))
    if args.arm == "both":
        arms = ["legacy", "fast"]
    elif args.arm == "phase3":
        arms = ["balanced", "quality"]
    elif args.arm == "phase4":
        arms = ["lowvram"]
    elif args.arm == "long":
        arms = ["long"]
    elif args.arm in ("long30a", "long30b", "longfast0f", "longfast5f",
                        "longfast22f", "longfast243", "longfast362",
                        "longfastprobe", "longfast243x3", "longfast124x6",
                        "longfastanchor", "pipe362"):
        arms = [args.arm]
    else:
        arms = [args.arm]
    for p in (COMFY_PY, COMFY_MAIN, PATHS_YAML, PROMPT_FILE):
        if not Path(p).exists():
            raise RuntimeError(f"Required file not found: {p}")
    return asyncio.run(amain(list(arms), args.port))


if __name__ == "__main__":
    raise SystemExit(main())
