# -*- coding: utf-8 -*-
"""音声テスト / 発音テスト (pronunciation test): an INDEPENDENT H3 entry point.

Audio-only inference does not exist for MiniMax H3 (video+audio share one
NestedTensor latent and one attention sequence), so this still runs the full
MiniMaxH3ReferenceToVideo + SamplerCustomAdvanced chain - it simply never
decodes/saves the VIDEO half. VAEDecodeAudio.execute() does
`latent.unbind()[-1]`, so it works without VAEDecode ever running, and
PreviewAudio writes a lossless flac to ComfyUI's OWN temp dir (never
comfy_output), which this module converts to a canonical wav.

This module owns:
  - the two fixed profiles (QUICK / FULL)
  - the graph builder (its OWN builder - graphs.build_generate_graph is
    never touched or branched)
  - the six-section English prompt builder
  - on-disk storage of one test per directory, and safe deletion
  - the ComfyUI /view download + flac->wav conversion + temp cleanup

Nothing here changes behaviour of single generation, story30, AI Director, the
production prompt compiler, Character handling, outfit override or any
performance preset - it only READS shared, already-verified building blocks
(`graphs._ref_chain`, `graphs.validate_resolution`, `graphs.NODE_H3`,
`modes_v2.resolve`/`apply_patches`, `director_lexicon._subject_definitions_line`,
`canon.parse_profile`, `merge.find_ffmpeg`).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from . import canon as canon_mod
from . import director_lexicon as director_lexicon_mod
from . import graphs
from . import merge as merge_mod
from . import modes_v2
from .errors import PipelineError

Graph = dict[str, dict[str, Any]]

# ------------------------------------------------------------------ profiles --
# No lower-resolution profile yet (explicitly out of scope).
PROFILES: dict[str, dict] = {
    "quick": {"width": 576, "height": 1024, "frames": 56, "max_refs": 1},
    "full": {"width": 576, "height": 1024, "frames": 124, "max_refs": 4},
}

# Node ids the graph builder assigns (mirrors graphs.py's NODE_* convention).
NODE_DECODE_AUDIO = "decode_audio"
NODE_PREVIEW_AUDIO = "preview_audio"

_SAFE_TEST_ID = re.compile(r"^[0-9a-fA-F]{6,32}$")


class AudioTestError(RuntimeError):
    """A refused request. `message` is user-facing Japanese. `status` is HTTP."""

    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = int(status)


# -------------------------------------------------------------- profile/mode --
def resolve_profile(profile: str, defaults: dict, models: dict, *,
                    loras_dir: Path | None = None) -> dict:
    """QUICK/FULL width/height/frames + the shared FAST turbo recipe.

    Always resolves via modes_v2.resolve("FAST", ...) so the turbo 4-step
    recipe (steps=4, euler, simple, turbo LoRA, int8 video VAE,
    SigmaShift 12/3) can never silently diverge from production FAST.
    TURBO_BIND_FAILED propagates - a broken LoRA must refuse, not degrade.
    """
    key = str(profile or "").strip().lower()
    if key not in PROFILES:
        raise AudioTestError(f"不明なプロファイルです: {profile}")
    spec = PROFILES[key]
    settings = dict(defaults)
    settings.update({"width": spec["width"], "height": spec["height"],
                     "frames": spec["frames"]})
    resolved = modes_v2.resolve("FAST", settings, models, loras_dir=loras_dir)
    return {"key": key, "spec": spec, "resolved": resolved}


def select_images_for_profile(profile: str, images: list[str]) -> list[str]:
    spec = PROFILES[str(profile).strip().lower()]
    return list(images or [])[:spec["max_refs"]]


# --------------------------------------------------------------- graph build --
def build_audio_test_graph(*, images: list[str], en_prompt: str, resolved: dict,
                           ref_longest: list[int],
                           voice_master_file: str | None = None,
                           clip_device: str = "gpu:1") -> Graph:
    """The audio-test chain: identical model/device plumbing to production's
    build_generate_graph, but NO VAEDecode / CreateVideo / SaveVideo - only
    VAEDecodeAudio -> PreviewAudio. This is its OWN builder; production's
    build_generate_graph is never branched or modified.
    """
    settings = resolved["settings"]
    models = resolved["models"]
    width, height, frames = int(settings["width"]), int(settings["height"]), int(settings["frames"])
    graphs.validate_resolution(width, height)

    graph: Graph = {}
    scale_ids = graphs._ref_chain(graph, images, ref_longest)

    # ---- CUDA phase barrier (same reasoning as build_generate_graph) -------
    graph["restore"] = {
        "class_type": "H3CudaPhaseRestore",
        "inputs": {
            "anything": [scale_ids[0], 0],
            "target_gpu": 0,
            "label": "app-h3-audio-test-restore",
            "synchronize": True,
        },
    }
    graph["clip"] = {
        "class_type": "H3GatedCLIPLoader",
        "inputs": {
            "barrier": ["restore", 0],
            "clip_name": models["clip"],
            "type": "minimax",
            "device": "default",
        },
    }
    graph["clip_report_base"] = {
        "class_type": "H3ClipDeviceReport",
        "inputs": {"clip": ["clip", 0], "label": "clip-base"},
    }
    graph["clip_select"] = {
        "class_type": "SelectCLIPDevice",
        "inputs": {"clip": ["clip_report_base", 0], "device": clip_device},
    }
    graph["clip_report_target"] = {
        "class_type": "H3ClipDeviceReport",
        "inputs": {"clip": ["clip_select", 0], "label": "clip-target"},
    }

    # ---- models -------------------------------------------------------------
    # Video VAE stays connected: MiniMaxH3ReferenceToVideo uses it to encode
    # the reference images even though the video half is never decoded.
    graph["vae_video"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["vae_video"]}}
    graph["vae_audio"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["vae_audio"]}}
    graph["unet"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": models["unet"], "weight_dtype": "default"},
    }
    graph["lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": models["lora"],
            "strength_model": float(models.get("lora_strength", 1.0)),
            "model": ["unet", 0],
        },
    }

    # ---- H3 reference to video (still the only way to reach audio) --------
    h3_inputs: dict[str, Any] = {
        "clip": ["clip_report_target", 0],
        "vae": ["vae_video", 0],
        "audio_vae": ["vae_audio", 0],
        "prompt": en_prompt,
        "width": width,
        "height": height,
        "length": frames,
        "ref_image_size": settings["ref_image_size"],
    }
    for i, sid in enumerate(scale_ids):
        h3_inputs[f"ref_images.ref_image_{i}"] = [sid, 0]

    if voice_master_file:
        graph["voice_master"] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": voice_master_file},
        }
        h3_inputs["ref_audios.ref_audio_0"] = ["voice_master", 0]

    graph[graphs.NODE_H3] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": h3_inputs}

    # ---- sampling -----------------------------------------------------------
    graph["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(settings["seed"])}}
    graph["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": ["lora", 0], "conditioning": [graphs.NODE_H3, 0]},
    }
    graph["sampler_select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": settings["sampler"]},
    }
    graph["scheduler"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "scheduler": settings["scheduler"],
            "steps": int(settings["steps"]),
            "denoise": 1.0,
            "model": ["lora", 0],
        },
    }
    graph[graphs.NODE_SAMPLER] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["scheduler", 0],
            "latent_image": [graphs.NODE_H3, 1],
        },
    }

    # ---- audio-only output: no VAEDecode / CreateVideo / SaveVideo --------
    graph[NODE_DECODE_AUDIO] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": [graphs.NODE_SAMPLER, 0], "vae": ["vae_audio", 0]},
    }
    graph[NODE_PREVIEW_AUDIO] = {
        "class_type": "PreviewAudio",
        "inputs": {"audio": [NODE_DECODE_AUDIO, 0]},
    }

    if resolved.get("patches"):
        modes_v2.apply_patches(graph, resolved["patches"])

    return graph


def assert_audio_only_graph(graph: Graph) -> None:
    """Structural backstop: this must never grow a video decode/save path."""
    class_types = {n.get("class_type") for n in graph.values()}
    for banned in ("VAEDecode", "CreateVideo", "SaveVideo"):
        if banned in class_types:
            raise AudioTestError(
                f"音声テストのグラフに {banned} が含まれています（禁止）。", status=500)
    for required in ("VAEDecodeAudio", "PreviewAudio"):
        if required not in class_types:
            raise AudioTestError(
                f"音声テストのグラフに {required} がありません。", status=500)


# -------------------------------------------------------------- prompt build --
_NEUTRAL_SUBJECT_LINE = "<Subject 1> the main speaker, facing the camera."


def _subject_line_for(character_snapshot: dict | None) -> str:
    """Labelled canon line, built the same way the production compiler does.

    Prefers the structured canon/outfit_ref dict (director_lexicon's own
    subject-definitions helper - identical rendering to compile_direct_prompt),
    falls back to canon.parse_profile(profile_text) when only raw profile text
    is available, and finally degrades to a neutral line when there is no
    Character at all.
    """
    snap = character_snapshot or {}
    identity = dict(snap.get("canon") or {})
    outfit_ref = dict(snap.get("outfit_ref") or {})
    if any(str(v or "").strip() for v in identity.values()) or \
            any(str(v or "").strip() for v in outfit_ref.values()):
        outfit_en = str(outfit_ref.get("outfit", "") or "")
        return director_lexicon_mod._subject_definitions_line(
            identity, outfit_en, False, outfit_ref)

    profile_text = str(snap.get("profile_text") or "").strip()
    if profile_text:
        parsed = canon_mod.parse_profile(profile_text)
        line = parsed.subject_line("<Subject 1>")
        if line:
            return line

    return _NEUTRAL_SUBJECT_LINE


def build_audio_test_prompt(*, character_snapshot: dict | None, n_pictures: int,
                            voice_master_connected: bool, language: str,
                            spoken_text: str, voice_note: str = "") -> str:
    """The official six sections, English prose, dialogue only inside <d>.

    Raises AudioTestError on bad input (empty spoken text / unknown language) -
    those are 400s at the HTTP layer, never silently coerced.
    """
    if language not in ("Japanese", "English"):
        raise AudioTestError(f"language は Japanese/English のみ対応です: {language}")
    spoken = str(spoken_text or "").strip()
    if not spoken:
        raise AudioTestError("読み上げテキストが空です。")
    if any(c in spoken for c in "<>") or len(spoken) > 2000:
        raise AudioTestError("台詞は2000文字以内で、制御タグを含めずに入力してください。")
    n_pictures = max(0, int(n_pictures))

    subject_line = _subject_line_for(character_snapshot)

    sections: dict[str, list[str]] = {name: [] for name in director_lexicon_mod.SECTIONS}

    sections["subject_definitions"].append(subject_line)
    for i in range(n_pictures):
        sections["subject_definitions"].append(
            f"<Picture {i + 1}> reference image {i + 1} anchors <Subject 1>'s "
            "appearance.")
    if voice_master_connected:
        sections["subject_definitions"].append(
            "<Audio 1> is a fixed voice reference defining the speaker's "
            "timbre for this clip; it is not the soundtrack of any video.")

    sections["summary"].append(
        "[reference generation] a short close-up clip of <Subject 1> speaking "
        "to the camera.")

    sections["retention_analysis"].append(
        "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, "
        "build and outfit.")

    sections["detailed_description"].extend([
        "A vertical 9:16 close-up of <Subject 1> facing the camera.",
        "[Shot 1] <Subject 1> faces the camera with a natural expression.",
        "<Subject 1> (S1) uses relaxed conversational phrasing, natural breath pauses "
        "and sentence-final intonation, and says,",
        f"<d>[{language}] {spoken}</d>",
        "Natural synchronized mouth movement matching the spoken dialogue, "
        "with accurate lip timing and clear articulation.",
    ])

    soundscape: list[str] = []
    note = str(voice_note or "").strip()
    if note:
        soundscape.append(note if note.endswith((".", "!", "?")) else note + ".")
    soundscape.append("Quiet room tone. No narration, no additional voices.")
    sections["overall_soundscape"].append(" ".join(soundscape))

    sections["non_diegetic_music"].append("N/A")

    rendered: list[str] = []
    for name in director_lexicon_mod.SECTIONS:
        rendered.append(name)
        rendered.extend(sections[name])
        rendered.append("")
    return "\n".join(rendered).strip() + "\n"


# ---------------------------------------------------------------- extractors --
def preview_audio_output(history: dict, node_id: str = NODE_PREVIEW_AUDIO) -> dict | None:
    """PreviewAudio's ui payload -> {"filename", "subfolder", "type"}.

    comfy_api/latest/_ui.py: PreviewAudio.as_dict() -> {"audio": [SavedResult]},
    SavedResult == {"filename", "subfolder", "type"} (type == "temp" here).
    """
    outputs = (history.get("outputs") or {}).get(node_id) or {}
    items = outputs.get("audio")
    if isinstance(items, list) and items and isinstance(items[0], dict) and \
            "filename" in items[0]:
        item = items[0]
        return {"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "temp")}
    return None


# ---------------------------------------------------------------------- I/O --
def audio_tests_root(config) -> Path:
    return Path(config.app_dir) / "audio_tests"


def is_safe_test_id(test_id) -> bool:
    return bool(test_id) and bool(_SAFE_TEST_ID.match(str(test_id)))


def _real(path) -> Path:
    """realpath, never raises (modelled on reveal.py's containment style)."""
    try:
        return Path(os.path.realpath(str(path)))
    except (OSError, ValueError):
        return Path(str(path))


def resolve_test_dir(root, test_id) -> Path:
    """The one test's directory, proven to be strictly inside `root`.

    Rejects `..`, absolute paths, symlinks/junctions escaping the root, and
    any id failing is_safe_test_id - all BEFORE the filesystem is touched.
    """
    if not is_safe_test_id(test_id):
        raise AudioTestError("不正なテストIDです。", status=400)
    root_real = _real(root)
    candidate_real = _real(Path(root) / str(test_id))
    if candidate_real == root_real or not candidate_real.is_relative_to(root_real):
        raise AudioTestError("不正なテストIDです。", status=400)
    return candidate_real


def delete_test(root, test_id) -> bool:
    """Delete exactly one test directory. Never touches anything outside root."""
    target = resolve_test_dir(root, test_id)
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


def delete_all_tests(root) -> int:
    """Delete every test directory directly under root. Returns the count."""
    root_real = _real(root)
    if not root_real.is_dir():
        return 0
    count = 0
    for child in list(root_real.iterdir()):
        if not child.is_dir() or not is_safe_test_id(child.name):
            continue
        child_real = _real(child)
        if child_real == root_real or not child_real.is_relative_to(root_real):
            continue
        shutil.rmtree(child_real)
        count += 1
    return count


def list_tests(config) -> list[dict]:
    """Every test's meta.json, newest first."""
    root = audio_tests_root(config)
    if not root.is_dir():
        return []
    metas: list[dict] = []
    for child in root.iterdir():
        if not child.is_dir() or not is_safe_test_id(child.name):
            continue
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:                                        # noqa: BLE001
            continue
    metas.sort(key=lambda m: float(m.get("created_at") or 0), reverse=True)
    return metas


def cleanup_comfy_temp(config, comfy_temp: dict | None) -> str:
    """Delete ONLY the exact ComfyUI temp file this test's own history entry
    reported. Never scans or bulk-deletes temp. Returns "" on success/skip-ok,
    or a Japanese note to record in meta.json["notes"] when ownership of the
    file could not be established (the file is then left alone).
    """
    comfy_temp = comfy_temp or {}
    filename = str(comfy_temp.get("filename") or "")
    if not filename:
        return ""
    subfolder = str(comfy_temp.get("subfolder") or "")
    temp_root = _real(config.comfy_temp)
    candidate = Path(config.comfy_temp) / subfolder / filename if subfolder \
        else Path(config.comfy_temp) / filename
    real = _real(candidate)
    if real == temp_root or not real.is_relative_to(temp_root):
        return "一時ファイルの後片付けを見送りました（安全なパスと確認できませんでした）。"
    if not real.is_file():
        return ""
    try:
        real.unlink()
        return ""
    except Exception as exc:                                    # noqa: BLE001
        return f"一時ファイルの後片付けに失敗しました: {exc}"


# --------------------------------------------------------------------- run --
async def _run_ffmpeg(args: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, (out or b"").decode("utf-8", errors="replace")


async def run_audio_test(pipeline, client, config, *, text: str, spoken_text: str,
                         language: str, profile: str, voice_master_file: str | None,
                         images: list[str], character_snapshot: dict | None,
                         notes: str, want_mp3: bool) -> dict:
    """Build, submit (through the shared job slot), download, convert, store.

    Returns the meta.json record. Raises AudioTestError (400/500-ish, message
    already Japanese) or PipelineError (kind="unknown" when the pipeline is
    busy - the SAME busy error single-shot generation uses) on failure.
    """
    spoken = str(spoken_text or text or "").strip()
    images = [str(n) for n in (images or [])]
    if not images:
        raise AudioTestError("参照画像が1枚もありません。少なくとも1枚必要です。")

    loras_dir = pipeline._loras_dir()
    picked = resolve_profile(profile, config.defaults, config.models, loras_dir=loras_dir)
    spec, resolved = picked["spec"], picked["resolved"]
    profile_key = picked["key"]
    sel_images = select_images_for_profile(profile_key, images)

    en_prompt = build_audio_test_prompt(
        character_snapshot=character_snapshot, n_pictures=len(sel_images),
        voice_master_connected=bool(voice_master_file), language=language,
        spoken_text=spoken)

    # GPU plan: an invalid saved GPU configuration must never reach ComfyUI.
    from . import gpu as gpu_mod
    plan = gpu_mod.current_plan(config)
    if not plan.get("ok"):
        raise PipelineError(
            "GPU設定に問題があるため生成できません: " + "; ".join(plan.get("errors") or []),
            kind="gpu_config")
    clip_device = plan.get("clip_device") or "gpu:1"

    graph = build_audio_test_graph(
        images=sel_images, en_prompt=en_prompt, resolved=resolved,
        ref_longest=config.defaults["ref_longest"],
        voice_master_file=voice_master_file, clip_device=clip_device)
    assert_audio_only_graph(graph)
    graphs.assert_reference_labels(graph, en_prompt)

    async def _job():
        t0 = time.monotonic()
        history = await client.run_graph(
            graph, on_event=lambda ev: None, profile=modes_v2.launch_profile("FAST"))
        return history, round(time.monotonic() - t0, 1)

    history, gen_seconds = await pipeline.run_audio_test(_job())

    saved = preview_audio_output(history, NODE_PREVIEW_AUDIO)
    if saved is None:
        raise AudioTestError("音声の生成結果を取得できませんでした。", status=500)

    params = {"filename": saved["filename"], "subfolder": saved.get("subfolder", ""),
             "type": saved.get("type", "temp")}
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{client.base}/view", params=params) as r:
            if r.status != 200:
                raise AudioTestError(
                    f"音声ファイルを取得できませんでした (HTTP {r.status})", status=500)
            flac_bytes = await r.read()

    test_id = uuid.uuid4().hex[:12]
    test_dir = audio_tests_root(config) / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    flac_path = test_dir / "_audio.flac"
    flac_path.write_bytes(flac_bytes)
    wav_path = test_dir / "audio.wav"
    ffmpeg = merge_mod.find_ffmpeg()
    code, log = await _run_ffmpeg([
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(flac_path),
        "-acodec", "pcm_s16le", "-ar", "32000", "-ac", "2", str(wav_path)])
    if code != 0 or not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise AudioTestError(f"音声の変換(flac->wav)に失敗しました: {log[-500:]}",
                             status=500)
    try:
        flac_path.unlink()
    except Exception:                                            # noqa: BLE001
        pass

    mp3_path: Path | None = None
    if want_mp3:
        candidate_mp3 = test_dir / "audio.mp3"
        code2, _log2 = await _run_ffmpeg([
            ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-qscale:a", "2", str(candidate_mp3)])
        if code2 == 0 and candidate_mp3.is_file() and candidate_mp3.stat().st_size > 0:
            mp3_path = candidate_mp3

    prompt_path = test_dir / "prompt.txt"
    prompt_path.write_text(en_prompt, encoding="utf-8")

    reference = {
        "images": sel_images,
        "voice_master_file": voice_master_file or "",
        "comfy_temp": saved,
    }
    (test_dir / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")

    cleanup_note = cleanup_comfy_temp(config, saved)
    notes_text = str(notes or "").strip()
    if cleanup_note:
        notes_text = f"{notes_text} {cleanup_note}".strip() if notes_text else cleanup_note

    models = resolved["models"]
    meta = {
        "test_id": test_id,
        "created_at": time.time(),
        "profile": profile_key,
        "text": str(text or ""),
        "spoken_text": spoken,
        "language": language,
        "character_id": str((character_snapshot or {}).get("character_id") or ""),
        "voice_master_enabled": bool(voice_master_file),
        "voice_source": voice_master_file or "",
        "provider": "local_h3",
        "model": f"{models['unet']} + {models.get('lora') or ''}".strip(" +"),
        "width": spec["width"], "height": spec["height"], "frames": spec["frames"],
        "duration_sec": graphs.seconds_for_frames(spec["frames"]),
        "seed": int(resolved["settings"]["seed"]),
        "wav_path": str(wav_path),
        "mp3_path": str(mp3_path) if mp3_path else "",
        "prompt_dump_path": str(prompt_path),
        "generation_seconds": gen_seconds,
        "quick_validated": None,
        "notes": notes_text,
    }
    (test_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
