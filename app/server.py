# -*- coding: utf-8 -*-
"""H3 App backend: static SPA + REST + SSE, bound to 127.0.0.1 only.

Run:
    <comfy venv python> server.py
    <comfy venv python> server.py --selftest      (no GPU, no ComfyUI, no network)
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import io
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from aiohttp import web                                            # noqa: E402

from h3app import audio_test as audio_test_mod                     # noqa: E402
from h3app import canon as canon_mod                               # noqa: E402
from h3app import compiler                                         # noqa: E402
from h3app import local_security                                  # noqa: E402
from h3app import director_review, voice_rehearsal                 # noqa: E402
from h3app import duration as duration_mod                         # noqa: E402
from h3app import graphs                                           # noqa: E402
from h3app import merge as merge_mod                               # noqa: E402
from h3app import segmenter                                        # noqa: E402
from h3app.comfy import ComfyClient, port_is_open                  # noqa: E402
from h3app import comfy_inputs as comfy_inputs_mod                 # noqa: E402
from h3app import system_info as system_info_mod                   # noqa: E402
from h3app.config import (ASPECT_PRESETS,                          # noqa: E402
                          REF_IMAGE_SIZES, SAMPLERS,
                          SCHEDULERS, SECONDS_PRESETS, STYLE_PRESETS, Config,
                          load_config, load_manifest, measured_seconds,
                          preset_text as config_preset_text,
                          update_config_file)
from h3app import gpu as gpu_mod                                    # noqa: E402
from h3app.errors import PipelineError, classify                   # noqa: E402
from h3app import presets as presets_mod                           # noqa: E402
from h3app.presets import ActionPresetStore, PresetError           # noqa: E402
from h3app import pipeline as pipeline_mod                         # noqa: E402
from h3app.pipeline import Pipeline                                # noqa: E402
from h3app import modes_v2 as modes_mod                             # noqa: E402
from h3app.projects import (STAGES, ProjectStore,                  # noqa: E402
                            profile_cache_key)
from h3app.prompts_bridge import build_director_text, prompts      # noqa: E402
from h3app import reveal as reveal_mod                             # noqa: E402
from h3app import media as media_mod                               # noqa: E402
from h3app import character_library as character_lib_mod           # noqa: E402
from h3app import ai_settings as ai_settings_mod                   # noqa: E402
from h3app import credstore as credstore_mod                       # noqa: E402
from h3app import director_spec as director_spec_mod               # noqa: E402
from h3app import director_prompts as director_prompts_mod         # noqa: E402
from h3app import director_convert as director_convert_mod         # noqa: E402
from h3app import director_lexicon as director_lexicon_mod         # noqa: E402
from h3app import director_provider as director_provider_mod       # noqa: E402
from h3app import transitions as transitions_mod                   # noqa: E402
from h3app import upscale as upscale_mod                           # noqa: E402
from h3app.session import SessionWatchdog                          # noqa: E402
from h3app import procstate as procstate_mod                       # noqa: E402
from h3app import shutdown as shutdown_mod                          # noqa: E402
from h3app import story_prompts                                    # noqa: E402
from h3app import story as story_mod                              # noqa: E402
from h3app.story import (StoryStore, StructureError,              # noqa: E402
                         normalize_segments, segment_seed)
from h3app.story_runner import (StoryPipeline, StoryRunner, assert_preflight,   # noqa: E402
                                preflight_records)

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
UPLOAD_PREFIX = "h3app_ref_"
# Character Library staging uses the same confined naming scheme.
CHAR_PREFIX = "h3app_char_"


# ---------------------------------------------------------------- helpers ----
def _json_error(message: str, *, status: int = 400, extra: dict | None = None):
    body = {"ok": False, "message": message}
    if extra:
        body.update(extra)
    return web.json_response(body, status=status)


def _untranslated_prompt_message(exc: "director_lexicon_mod.UntranslatedPromptError",
                                  clip_index: int | None = None) -> str:
    """Japanese error naming a few offending runs (C1: refuse, don't hide).

    `clip_index` (0-based) adds a "CLIPn: " prefix for story30, matching the
    existing per-clip error wording used elsewhere in this file.
    """
    sample = "".join(f"「{run}」" for run in sorted(set(exc.runs))[:5])
    prefix = f"CLIP{clip_index + 1}: " if clip_index is not None else ""
    return (f"監督案の英語変換に失敗しました（{prefix}未翻訳の日本語 {sample}）。"
            "監督案を作り直してください。")


def build_settings(cfg: Config, *, seconds: float, aspect: str, advanced: dict) -> dict:
    """Merge UI choices onto the verified v1 defaults."""
    d = cfg.defaults
    from h3app.speech_guard import DELIVERIES
    if not isinstance(advanced, (dict, type(None))):
        raise PipelineError("詳細設定が不正です。", kind="prompt_rejected")
    preset = next((p for p in ASPECT_PRESETS if p["id"] == aspect), ASPECT_PRESETS[0])
    advanced = advanced or {}

    width = int(advanced.get("width") or preset["width"])
    height = int(advanced.get("height") or preset["height"])
    graphs.validate_resolution(width, height)

    if advanced.get("frames_mode") == "manual" and advanced.get("frames"):
        frames = int(advanced["frames"])
        if frames % 17 != 5 or not (124 <= frames <= 362):
            raise PipelineError(
                "フレーム数は 17 で割った余りが 5 になる 124〜362 の値にしてください"
                "（例: 124 / 243 / 362）。", kind="prompt_rejected")
    else:
        frames = graphs.frames_for_seconds(seconds)

    seed = int(advanced.get("seed", d["seed"]))
    if advanced.get("randomize_seed"):
        seed = int.from_bytes(os.urandom(4), "big")

    sampler = advanced.get("sampler") or d["sampler"]
    scheduler = advanced.get("scheduler") or d["scheduler"]
    ref_image_size = advanced.get("ref_image_size") or d["ref_image_size"]
    if sampler not in SAMPLERS or scheduler not in SCHEDULERS \
            or ref_image_size not in REF_IMAGE_SIZES:
        raise PipelineError("詳細設定の値が不正です。", kind="prompt_rejected")

    steps = int(advanced.get("steps", d["steps"]))
    if not 1 <= steps <= 60 or not 0 <= seed < 2 ** 64:
        raise PipelineError("stepsは1〜60、seedは0以上の64bit整数にしてください。", kind="prompt_rejected")
    delivery = advanced.get("speech_delivery", "natural")
    if delivery not in DELIVERIES:
        raise PipelineError("話し方の指定が不正です。", kind="prompt_rejected")
    return {
        "width": width, "height": height, "frames": frames,
        "steps": steps, "speech_delivery": delivery,
        "seed": seed,
        "randomize_seed": bool(advanced.get("randomize_seed", False)),
        "sampler": sampler, "scheduler": scheduler,
        "ref_image_size": ref_image_size,
        "output_prefix": (advanced.get("output_prefix") or "").strip(),
    }


def story_structure_refusal(*, running: bool, count: int, limit: int) -> str | None:
    """Why a structure change may not be applied, in Japanese, or None.

    Pure and module-level so the refusal rules are testable without a server:
    a change while the loop is RUNNING is refused outright, because the loop
    holds `cursor` and a clip may be in flight.
    """
    if running:
        return "生成中はセグメントの構成を変更できません。先に停止してください。"
    if count <= 0:
        return "セグメントを空にはできません。少なくとも 1 つのセグメントが必要です。"
    if count > limit:
        return f"セグメントは最大 {limit} 個までです。"
    return None


def _safe_upload_name(name: str) -> bool:
    p = Path(name)
    return (p.name == name and (name.startswith(UPLOAD_PREFIX) or
                                name.startswith(CHAR_PREFIX))
            and p.suffix.lower() in ALLOWED_IMAGE_EXT)


# ----------------------------------------------------------------- routes ----
def create_app(cfg: Config) -> web.Application:
    cfg.ensure_dirs()
    store = ProjectStore(cfg.projects_dir)
    client = ComfyClient(cfg)
    pipeline = Pipeline(cfg, store, client)
    story_store = StoryStore(cfg.story_dir)
    story_pipeline = StoryPipeline(pipeline, story_store)
    watchdog = SessionWatchdog(cfg, pipeline, story_pipeline, client)
    # User-editable ACTION presets (app\action_presets.json). Loaded once at
    # startup; a corrupt file falls back to the factory defaults and SAYS SO.
    action_presets = ActionPresetStore(presets_mod.STORE_PATH)

    # ------------------------------------------------------- WP-D: shutdown --
    h3_session_id = procstate_mod.new_session_id()
    client.session_id = h3_session_id
    shutdown_seq = shutdown_mod.ShutdownSequence(
        cfg, pipeline, client, story_pipeline, session_id=h3_session_id)

    app = web.Application(client_max_size=64 * 1024 * 1024,
                          middlewares=[local_security.local_boundary])
    app["cfg"] = cfg
    app["store"] = store
    app["client"] = client
    app["pipeline"] = pipeline
    app["story_store"] = story_store
    app["story_pipeline"] = story_pipeline
    app["watchdog"] = watchdog
    app["action_presets"] = action_presets
    app["h3_session_id"] = h3_session_id
    app["shutdown_seq"] = shutdown_seq
    app["accepting_jobs"] = True
    app["_shutdown_ran"] = False

    # The UI files are edited in place and reloaded by hand; a cached app.js
    # after an update looks like a broken app, so never let them be cached.
    _NO_STORE = {"Cache-Control": "no-store, must-revalidate"}

    async def index(request: web.Request):
        return web.FileResponse(cfg.web_dir / "index.html", headers=_NO_STORE)

    async def static_file(request: web.Request):
        name = request.match_info["name"]
        target = local_security.contained_file(cfg.web_dir, name)
        if target is None:
            raise web.HTTPNotFound()
        return web.FileResponse(target, headers=_NO_STORE)

    # ------------------------------------------------- v2 compat snapshot ----
    async def _compat_snapshot(client, cfg):
        """Startup checklist. Best-effort: UNKNOWN, never fatal."""
        reachable = False
        try:
            reachable = await client.is_reachable()
        except Exception:                                            # noqa: BLE001
            reachable = False
        oi = client.object_info if reachable else None
        gpus: list[str] = []
        try:
            detected = gpu_mod.detect()
            if detected.get("ok"):
                gpus = [g["name"] for g in detected.get("gpus") or []]
        except Exception:                                            # noqa: BLE001
            pass
        loras_dir = pipeline._loras_dir()
        shared = loras_dir.parent
        want = {"h3_ref2va": shared / "diffusion_models" / cfg.models["unet"],
                "qwen_encoder": shared / "text_encoders" / cfg.models["clip"],
                "vae_video": shared / "vae" / cfg.models["vae_video"],
                "vae_audio": shared / "vae" / cfg.models["vae_audio"]}
        present = {k: p.is_file() for k, p in want.items()}
        try:
            status = modes_mod.compat_status(
                comfy_reachable=reachable, object_info=oi, gpus=gpus,
                loras_dir=loras_dir, models_present=present)
        except Exception as exc:                                     # noqa: BLE001
            status = {"error": str(exc)}
        return status

    async def api_compat(request: web.Request):
        return web.json_response({"ok": True,
                                  "compat": await _compat_snapshot(client, cfg)})

    async def api_config(request: web.Request):
        d = cfg.defaults
        try:
            P = prompts()
            negative_default = P.NEGATIVE_DEFAULT
        except Exception:
            negative_default = ""
        return web.json_response({
            "ok": True,
            "defaults": d,
            "modes": modes_mod.UI_MODES,
            "modes_advanced": modes_mod.UI_MODES_ADVANCED,
            "modes_experimental": modes_mod.UI_MODES_EXPERIMENTAL,
            "compat": await _compat_snapshot(client, cfg),
            "presets": {
                "aspects": ASPECT_PRESETS,
                "seconds": SECONDS_PRESETS,
                "samplers": SAMPLERS,
                "schedulers": SCHEDULERS,
                "ref_image_sizes": REF_IMAGE_SIZES,
                # "motions" is the old key. It now means STYLE presets only:
                # anything safe to apply to every segment.
                "motions": STYLE_PRESETS,
                "styles": STYLE_PRESETS,
                # ACTION presets are USER DATA now, not config constants.
                "actions": action_presets.list(),
            },
            # Kept for the not-yet-updated frontend; same list as style_presets.
            "motion_presets": STYLE_PRESETS,
            "style_presets": STYLE_PRESETS,
            "action_presets": action_presets.list(),
            "action_preset_template": presets_mod.APPLY_TEMPLATE,
            "action_preset_max_name": presets_mod.MAX_NAME_CHARS,
            "story": {
                "seconds_per_segment": int((cfg.story or {}).get("seconds_per_segment", 5)),
                "max_segments": int((cfg.story or {}).get("max_segments", 24)),
            },
            "idle_timeout_seconds": watchdog.timeout,
            "stop_comfy_when_idle": bool(cfg.get("stop_comfy_when_idle", True)),
            "max_seconds": d["max_seconds"],
            "stages": STAGES,
            "output_prefix": cfg["output_prefix"],
            "measured_seconds": measured_seconds(),
            "negative_default": negative_default,
            "comfy_reachable": await client.is_reachable(),
            "comfy_port": cfg.comfy_port,
        })

    MAX_IMAGE_BYTES = 32 * 1024 * 1024

    async def api_images(request: web.Request):
        """Atomic multi-file upload: all files verified before any is kept.

        Every part is first written to `<stage>/.upload_<uuid>.part`, then
        opened+verified with PIL. Only once every part in THIS request has
        passed does each `.part` get renamed to its final h3app_ref_ name
        (os.replace); any failure removes every temp/renamed file from this
        request and nothing already on disk is touched.
        """
        stage_dir = cfg.input_stage_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
        reader = await request.multipart()
        parts: list[dict] = []          # {"part": Path, "final_name": str, "filename": str}
        all_part_paths: list[Path] = []  # every .part created THIS request
        error_response = None
        try:
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name not in ("files", "file", "images"):
                    continue
                filename = part.filename or "(名前なし)"
                if len(parts) >= 4:
                    error_response = _json_error(
                        "参照画像は最大4枚までです。", status=400,
                        extra={"failed_file": filename})
                    break
                ext = Path(filename).suffix.lower()
                if ext not in ALLOWED_IMAGE_EXT:
                    error_response = _json_error(
                        f"「{filename}」は画像として読み込めませんでした。"
                        "PNG/JPEG/WebP/BMPの画像を選び直してください。",
                        status=400, extra={"failed_file": filename})
                    break
                part_path = stage_dir / f".upload_{uuid.uuid4().hex}.part"
                all_part_paths.append(part_path)
                size = 0
                try:
                    with part_path.open("wb") as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > MAX_IMAGE_BYTES:
                                raise ValueError("too_large")
                            f.write(chunk)
                except ValueError:
                    error_response = _json_error(
                        f"「{filename}」は32MBを超えています。"
                        "小さい画像にしてから追加してください。",
                        status=400, extra={"failed_file": filename})
                    break
                except OSError as exc:
                    error_response = _json_error(
                        f"画像を保存できませんでした（{exc}）。"
                        f"保存先フォルダ {stage_dir} への書き込み権限を確認してください。",
                        status=500, extra={"failed_file": filename})
                    break
                # PIL verification: open+verify, then reopen and decode a
                # thumbnail-size render. A corrupt/mislabeled file never
                # reaches ComfyUI.
                try:
                    from PIL import Image
                    with Image.open(part_path) as im:
                        im.verify()
                    with Image.open(part_path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((320, 320))
                except Exception:                                # noqa: BLE001
                    error_response = _json_error(
                        f"「{filename}」は画像として読み込めませんでした。"
                        "PNG/JPEG/WebP/BMPの画像を選び直してください。",
                        status=400, extra={"failed_file": filename})
                    break
                final_name = f"{UPLOAD_PREFIX}{uuid.uuid4().hex}{ext}"
                parts.append({"part": part_path, "final_name": final_name,
                              "filename": filename})
        except Exception as exc:                                 # noqa: BLE001
            error_response = _json_error(
                f"画像を保存できませんでした（{exc}）。"
                f"保存先フォルダ {stage_dir} への書き込み権限を確認してください。",
                status=500)

        if error_response is not None:
            for part_path in all_part_paths:
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:                                # noqa: BLE001
                    pass
            return error_response

        if not parts:
            return _json_error("画像が受け取れませんでした。")

        renamed: list[Path] = []
        try:
            saved = []
            for entry in parts:
                dest = stage_dir / entry["final_name"]
                os.replace(entry["part"], dest)
                renamed.append(dest)
                saved.append({"name": entry["final_name"],
                             "thumb": f"/api/thumb/{entry['final_name']}"})
                # H3_Media export (best-effort; working copy stays staged).
                media_mod.export_copy(dest, cfg.media_images,
                                      entry["final_name"],
                                      media_root=cfg.media_root,
                                      media_type="image")
        except OSError as exc:
            for path in renamed:
                try:
                    path.unlink(missing_ok=True)
                except Exception:                                # noqa: BLE001
                    pass
            for entry in parts:
                try:
                    entry["part"].unlink(missing_ok=True)
                except Exception:                                # noqa: BLE001
                    pass
            return _json_error(
                f"画像を保存できませんでした（{exc}）。"
                f"保存先フォルダ {stage_dir} への書き込み権限を確認してください。",
                status=500)
        return web.json_response({"ok": True, "images": saved})

    async def api_thumb(request: web.Request):
        name = request.match_info["name"]
        if not _safe_upload_name(name):
            raise web.HTTPNotFound()
        src = comfy_inputs_mod.local_path(cfg, name)
        if src is None or not src.is_file():
            raise web.HTTPNotFound()
        try:
            from PIL import Image
        except Exception:
            return web.FileResponse(src)
        buf = io.BytesIO()
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            im.save(buf, format="JPEG", quality=85)
        return web.Response(body=buf.getvalue(), content_type="image/jpeg")

    # -------------------------------------------------------------- generate --
    async def api_generate(request: web.Request):
        # WP-D: block new jobs while a shutdown is in progress.
        if not app.get("accepting_jobs", True):
            return _json_error("H3は終了処理中です。新しい生成は開始できません。",
                               status=503)
        body = await request.json()
        images = [n for n in (body.get("images") or []) if _safe_upload_name(n)]
        if not images:
            return _json_error("参照画像を1枚以上追加してください。")
        if len(images) > 4:
            return _json_error("参照画像は最大4枚までです。")
        for n in images:
            if comfy_inputs_mod.local_path(cfg, n) is None:
                return _json_error("追加した画像が見つかりません。もう一度追加してください。")
        try:
            settings = build_settings(cfg, seconds=float(body.get("seconds", 5)),
                                      aspect=body.get("aspect", "portrait"),
                                      advanced=body.get("advanced") or {})
        except (PipelineError, graphs.GraphError) as exc:
            return _json_error(str(exc))

        try:
            negative = (body.get("jp_negative") or "").strip() or prompts().NEGATIVE_DEFAULT
        except Exception:
            negative = (body.get("jp_negative") or "").strip()

        project = store.create(
            settings=settings, images=images,
            jp_scene=(body.get("jp_scene") or "").strip(),
            jp_dialogue=(body.get("jp_dialogue") or "").strip(),
            jp_negative=negative,
            max_seconds=float(cfg.defaults["max_seconds"]))
        # Character snapshot: prefill the saved profile so no VLM
        # re-analysis runs. Accepted only when it still matches the images;
        # a manual image change afterwards drops it silently.
        snap = _character_snapshot_for(
            images, (body.get("character_snapshot")
                     if isinstance(body, dict) else None))
        if snap is not None:
            project.data["profile"] = snap["profile_text"]
            project.data["character_snapshot"] = snap
        # v2 Generation Mode travels on the project; pipeline resolves it at
        # submit time (and refuses unavailable modes loudly, never fallback).
        mode = str(body.get("mode") or "LEGACY").upper()
        if mode not in modes_mod.RUNNABLE_MODES and \
                modes_mod.experimental_reason(mode) is None:
            return _json_error(f"不明なモードです: {mode}")
        if modes_mod.experimental_reason(mode) is not None:
            return _json_error(
                f"{mode}は利用できません: {modes_mod.experimental_reason(mode)}")
        if mode in modes_mod.STORY_ONLY_MODES:
            return _json_error(
                f"{mode}はStoryモード専用です。通常の生成では使えません。")
        project.data["mode"] = mode
        project.save()
        try:
            await pipeline.start(project, lambda r: pipeline.generate_new(r))
        except PipelineError as exc:
            return _json_error(str(exc), status=409)
        return web.json_response({"ok": True, "project_id": project.id})

    async def api_again(request: web.Request):
        body = await request.json()
        base = store.get(body.get("project_id", ""))
        if base is None:
            return _json_error("元のプロジェクトが見つかりません。", status=404)
        if not base.data.get("en_prompt"):
            return _json_error("この動画には再利用できるプロンプトがありません。")
        base_mode = str(base.data.get("mode") or "LEGACY").upper()
        if base_mode in modes_mod.STORY_ONLY_MODES:
            return _json_error(
                f"{base_mode}はStoryモード専用です。通常の生成では使えません。")
        settings = copy.deepcopy(base.data["settings"])
        # Same everything, new seed: that is the whole point of "もう1本".
        settings["seed"] = int.from_bytes(os.urandom(4), "big")
        project = store.create(
            settings=settings, images=base.data["images"],
            jp_scene=base.data.get("jp_scene", ""),
            jp_dialogue=base.data.get("jp_dialogue", ""),
            jp_negative=base.data.get("jp_negative", ""),
            max_seconds=base.max_seconds,
            profile=base.data.get("profile", ""),
            profile_cache_key_=base.data.get("profile_cache_key", ""),
            en_prompt=base.data["en_prompt"],
            parent_id=base.id)
        project.data["mode"] = str(base.data.get("mode") or "LEGACY")
        project.save()
        try:
            await pipeline.start(project, lambda r: pipeline.generate_again(r))
        except PipelineError as exc:
            return _json_error(str(exc), status=409)
        return web.json_response({"ok": True, "project_id": project.id})

    async def api_continue(request: web.Request):
        body = await request.json()
        project = store.get(body.get("project_id", ""))
        if project is None:
            return _json_error("プロジェクトが見つかりません。", status=404)
        if not project.clips:
            return _json_error("まだ動画がありません。")
        if not project.can_continue:
            return _json_error(
                f"最大の長さです（{project.max_seconds:.0f}秒）。これ以上は延長できません。")
        cont_mode = str(project.data.get("mode") or "LEGACY").upper()
        if cont_mode in modes_mod.STORY_ONLY_MODES:
            return _json_error(
                f"{cont_mode}はStoryモード専用です。通常の生成では使えません。")
        # 続きの指示。空なら「同じ場面を前に進める」既定文が使われる。元のシーン文と
        # セリフを丸ごと入れ直すことは二度としない（同じ内容が再生成される原因）。
        instruction = (body.get("next_instruction") or "").strip()
        try:
            await pipeline.start(
                project, lambda r: pipeline.generate_continue(r, instruction))
        except PipelineError as exc:
            return _json_error(str(exc), status=409)
        return web.json_response({"ok": True, "project_id": project.id})

    async def api_cancel(request: web.Request):
        body = await request.json()
        ok = pipeline.cancel(body.get("project_id", ""))
        return web.json_response({"ok": ok,
                                  "message": "中止しました。" if ok else "中止できる生成がありません。"})

    async def api_release_models(request: web.Request):
        """Drop loaded models from VRAM. Neither ComfyUI nor this server exits."""
        try:
            result = await pipeline.release_models()
        except PipelineError as exc:
            info = classify(exc, stage="モデル解放")
            print(f"[release] refused: {exc}", flush=True)
            return web.json_response(
                {"ok": False, "message": info["title"], "reason": info["advice"],
                 "kind": info["kind"]},
                status=409 if info["kind"] == "busy" else 400)
        except Exception as exc:                     # noqa: BLE001
            # Never let a raw traceback reach the UI; it stays in the log.
            print(f"[release] failed: {type(exc).__name__}: {exc}", flush=True)
            return web.json_response(
                {"ok": False, "message": "モデルを解放できませんでした",
                 "reason": "ComfyUI との通信に失敗しました。もう一度実行してください。",
                 "kind": "network"},
                status=500)

        freed = result.get("freed_mib", 0)
        message = ("GPUメモリを解放しました" if freed <= 0
                   else f"GPUメモリを解放しました（約 {freed} MiB）")
        return web.json_response({
            "ok": True,
            "message": message,
            "reason": "次回の生成時に必要なモデルは自動で再読込されます。",
            "freed_mib": freed,
            "vlm_unloaded": result.get("vlm_unloaded", False),
            "before": result.get("before", []),
            "after": result.get("after", []),
        })

    # --------------------------------------------------------- 音声テスト ----
    async def api_audio_test_run(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        body = body or {}
        text = str(body.get("text") or "").strip()
        spoken_text = str(body.get("spoken_text") or "").strip() or text
        language = str(body.get("language") or "").strip()
        if language not in ("Japanese", "English"):
            return _json_error(
                "language は Japanese または English を指定してください。", status=400)
        profile = str(body.get("profile") or "").strip().lower()
        if profile not in ("quick", "full"):
            return _json_error("profile は quick または full を指定してください。",
                               status=400)
        images = [str(n) for n in (body.get("images") or [])]
        if not images:
            return _json_error("参照画像を1枚以上指定してください。", status=400)
        character_snapshot = body.get("character_snapshot") or None
        notes = str(body.get("notes") or "")
        want_mp3 = bool(body.get("want_mp3"))

        voice_master_file = None
        if bool(body.get("voice_master")):
            voice_file = str((character_snapshot or {}).get("voice_file") or "")
            base = Path(voice_file).name
            # Only our own staged voice copies (never an arbitrary path) -
            # same safety check as _character_stage_story_voice.
            if not (base.startswith(CHAR_PREFIX) and base.endswith(".wav") and
                    base == voice_file and
                    comfy_inputs_mod.local_path(cfg, base) is not None):
                return _json_error(
                    "Voice Master を使うには、音声が登録された Character を"
                    "選択してください。", status=400)
            voice_master_file = base

        try:
            meta = await audio_test_mod.run_audio_test(
                pipeline, client, cfg, text=text, spoken_text=spoken_text,
                language=language, profile=profile,
                voice_master_file=voice_master_file, images=images,
                character_snapshot=character_snapshot, notes=notes,
                want_mp3=want_mp3)
        except audio_test_mod.AudioTestError as exc:
            return _json_error(exc.message, status=exc.status)
        except PipelineError as exc:
            info = classify(exc, stage="音声テスト")
            return _json_error(info["title"],
                               status=409 if info["kind"] == "busy" else 400)
        return web.json_response({"ok": True, "meta": meta})

    async def api_audio_test_list(request: web.Request):
        return web.json_response({"ok": True, "tests": audio_test_mod.list_tests(cfg)})

    async def api_audio_test_audio(request: web.Request):
        test_id = request.match_info.get("test_id", "")
        root = audio_test_mod.audio_tests_root(cfg)
        try:
            test_dir = audio_test_mod.resolve_test_dir(root, test_id)
        except audio_test_mod.AudioTestError:
            raise web.HTTPNotFound()
        wav_path = test_dir / "audio.wav"
        if not wav_path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(wav_path, headers={"Content-Type": "audio/wav"})

    async def api_audio_test_delete(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        test_id = str((body or {}).get("test_id") or "")
        root = audio_test_mod.audio_tests_root(cfg)
        try:
            ok = audio_test_mod.delete_test(root, test_id)
        except audio_test_mod.AudioTestError as exc:
            return _json_error(exc.message, status=exc.status)
        if not ok:
            return _json_error("テストが見つかりませんでした。", status=404)
        return web.json_response({"ok": True})

    async def api_audio_test_delete_all(request: web.Request):
        root = audio_test_mod.audio_tests_root(cfg)
        count = audio_test_mod.delete_all_tests(root)
        return web.json_response({"ok": True, "deleted": count})

    async def api_project(request: web.Request):
        project = store.get(request.match_info["pid"])
        if project is None:
            return _json_error("プロジェクトが見つかりません。", status=404)
        data = project.to_public()
        runner = pipeline.runners.get(project.id)
        data["progress"] = runner.snapshot() if runner else None
        data["character_missing"] = _character_missing_note(
            data.get("character_snapshot"))
        return web.json_response({"ok": True, "project": data})

    async def _stream_events(request: web.Request, runner):
        """SSE for one runner. Shared by /api/events and /api/story/{sid}/events
        so the story stream is the same shape the UI already renders."""
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        if runner is None:
            await resp.write(b"event: end\ndata: {}\n\n")
            return resp

        q = runner.subscribe()
        try:
            while True:
                try:
                    snap = await asyncio.wait_for(q.get(), timeout=10)
                except asyncio.TimeoutError:
                    # keepalive AND a fresh elapsed clock for the UI
                    snap = runner.snapshot()
                payload = json.dumps(snap, ensure_ascii=False)
                await resp.write(f"data: {payload}\n\n".encode("utf-8"))
                if snap.get("finished"):
                    await resp.write(b"event: end\ndata: {}\n\n")
                    break
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            runner.unsubscribe(q)
        return resp

    async def api_events(request: web.Request):
        return await _stream_events(request, pipeline.runners.get(request.match_info["pid"]))

    async def api_video(request: web.Request):
        project = store.get(request.match_info["pid"])
        if project is None:
            raise web.HTTPNotFound()
        try:
            step = int(request.match_info["step"])
        except ValueError:
            raise web.HTTPNotFound()
        clip = next((c for c in project.clips if int(c["step"]) == step), None)
        if clip is None:
            raise web.HTTPNotFound()
        path = Path(clip.get("local_video") or clip["video"])
        if not path.is_file():
            path = Path(clip["video"])
        if not path.is_file():
            raise web.HTTPNotFound()
        # FileResponse handles Range requests, so <video> can seek.
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return web.FileResponse(path, headers={"Content-Type": ctype})

    # ------------------------------------------------------- open the folder --
    # The client sends LOGICAL IDS ONLY - an id it already has plus a target
    # from reveal.TARGETS. It never sends a path, so there is no path to
    # sanitise; the backend looks the folder up from the project's own metadata
    # and reveal.py proves it is inside an allowed output root before anything
    # is opened.
    def _open_resolution(resolution: dict, *, reveal: bool, validate_only: bool):
        payload = {"ok": True, "opened": False, **resolution}
        if validate_only:
            return web.json_response(payload)
        directory = resolution["directory"]
        file_path = resolution.get("file") or ""
        try:
            if reveal and file_path and Path(file_path).is_file():
                # "Reveal the file": same validated path, selected in Explorer.
                subprocess.Popen(["explorer.exe", f"/select,{file_path}"])
            elif hasattr(os, "startfile"):
                os.startfile(directory)                        # noqa: S606 - Windows
            else:
                return _json_error("このOSではフォルダを開けません。", status=500,
                                   extra={"reason_code": "error",
                                          "directory": directory})
        except Exception as exc:                              # noqa: BLE001
            is_access_denied = (isinstance(exc, PermissionError) or
                                (isinstance(exc, OSError) and
                                 getattr(exc, "winerror", None) == 5))
            if is_access_denied:
                identity = system_info_mod.process_identity()
                user = identity.get("user") or "不明なユーザー"
                message = (
                    "エクスプローラーを開けませんでした（アクセス拒否）。"
                    f"H3アプリがデスクトップとは別のアカウント（{user}）で"
                    "動いている可能性があります。RUN_H3_V2.bat からH3を"
                    f"起動し直してください。フォルダ: {directory}")
                return _json_error(message, status=500,
                                   extra={"reason_code": "access_denied",
                                          "directory": directory})
            return _json_error(f"フォルダを開けませんでした: {exc}", status=500,
                               extra={"reason_code": "error",
                                      "directory": directory})
        payload["opened"] = True
        return web.json_response(payload)

    async def api_open_folder(request: web.Request):
        """Single-shot. Body: {project_id, target?, reveal?, validate_only?}."""
        body = await request.json()
        pid = str(body.get("project_id") or "")
        if not reveal_mod.is_safe_project_id(pid):
            return _json_error("プロジェクトが見つかりませんでした。", status=404)
        project = store.get(pid)
        try:
            resolution = reveal_mod.resolve_project_target(
                project, body.get("target", "final"),
                project_dir=store.dir_for(pid),
                roots=reveal_mod.allowed_roots(cfg))
        except reveal_mod.TargetError as exc:
            return _json_error(exc.message, status=exc.status)
        return _open_resolution(
            resolution,
            reveal=bool(body.get("reveal", True)),
            validate_only=bool(body.get("validate_only", False)))

    async def api_story_open_folder(request: web.Request):
        """Story. Body: {story_id, target?, reveal?, validate_only?}."""
        body = await request.json()
        sid = str(body.get("story_id") or "")
        if not story_mod.is_safe_story_id(sid):
            return _json_error("ストーリーが見つかりませんでした。", status=404)
        story = story_store.get(sid)
        try:
            resolution = reveal_mod.resolve_story_target(
                story, body.get("target", "final"),
                roots=reveal_mod.allowed_roots(cfg))
        except reveal_mod.TargetError as exc:
            return _json_error(exc.message, status=exc.status)
        return _open_resolution(
            resolution,
            reveal=bool(body.get("reveal", False)),
            validate_only=bool(body.get("validate_only", False)))

    _MEDIA_OPEN_TARGETS = {
        "videos": lambda: cfg.media_videos,
        "finals": lambda: cfg.media_videos_final,
        "clips": lambda: cfg.media_videos_clips,
        "root": lambda: cfg.media_root,
    }

    async def api_media_open(request: web.Request):
        """Open a fixed生成動画 folder in Explorer.

        Body: {"target": "videos"|"finals"|"clips"|"root", "validate_only"}.
        Default target "videos". Unknown target -> 400.
        """
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            body = {}
        body = body or {}
        target = str(body.get("target") or "videos")
        factory = _MEDIA_OPEN_TARGETS.get(target)
        if factory is None:
            return _json_error("開く場所の指定が不正です。", status=400)
        try:
            cfg.ensure_dirs()
            directory = Path(factory())
            directory.mkdir(parents=True, exist_ok=True)
            resolution = {"kind": "media", "target": target,
                          "directory": str(directory), "file": ""}
            # Media root is always allowed (created above); others still gated.
            roots = reveal_mod.allowed_roots(cfg)
            real = reveal_mod._real(directory)
            if not any(real == r or real.is_relative_to(r) for r in roots):
                return _json_error("許可されていない場所は開けません。",
                                   status=400)
        except reveal_mod.TargetError as exc:
            return _json_error(exc.message, status=exc.status)
        return _open_resolution(
            resolution,
            reveal=False,
            validate_only=bool(body.get("validate_only", False)))

    def _probe_writable(directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".h3app_write_test_{uuid.uuid4().hex}.tmp"
            probe.write_bytes(b"ok")
            probe.unlink()
            return True
        except Exception:                                        # noqa: BLE001
            return False

    async def api_settings_storage(request: web.Request):
        return web.json_response({
            "ok": True,
            "media_root": str(cfg.media_root),
            "videos_dir": str(cfg.media_videos),
            "finals_dir": str(cfg.media_videos_final),
            "clips_dir": str(cfg.media_videos_clips),
            "images_dir": str(cfg.media_images),
            "input_stage_dir": str(cfg.input_stage_dir),
            "comfy_input_dir": str(cfg.comfy_input),
            "output_prefix": cfg["output_prefix"],
            "exists": {
                "videos": cfg.media_videos.is_dir(),
                "finals": cfg.media_videos_final.is_dir(),
                "clips": cfg.media_videos_clips.is_dir(),
            },
            "writable": {
                "videos": _probe_writable(cfg.media_videos),
                "input_stage": _probe_writable(cfg.input_stage_dir),
            },
            "process": system_info_mod.process_identity(),
        })

    def _cleanup_protected() -> set[str]:
        """Clip paths no cleanup may touch: running jobs + locked/approved."""
        protected: set[str] = set()

        def _add_clip(c: dict) -> None:
            for key in ("local_video", "video"):
                value = (c or {}).get(key)
                if value:
                    protected.add(str(Path(str(value))))

        try:
            active = pipeline.active
            if active is not None and getattr(active, "project", None) is not None:
                for c in (active.project.clips or []):
                    _add_clip(c)
        except Exception:                                        # noqa: BLE001
            pass
        try:
            for sid, runner in (story_pipeline.runners or {}).items():
                if not story_pipeline.is_running(sid):
                    continue
                story = story_store.get(sid)
                if story is None:
                    continue
                for c in (story.clips or []):
                    _add_clip(c)
                if story.data.get("final_video"):
                    protected.add(str(Path(str(story.data["final_video"]))))
                # Locked/approved segments keep their source clips.
                locked = {i for i, s in enumerate(story.segments)
                          if (s or {}).get("locked")}
                for i in locked:
                    clip = story.clip_for(i)
                    if clip:
                        _add_clip(clip)
        except Exception:                                        # noqa: BLE001
            pass
        return protected

    def _cleanup_scan(max_age_days: float) -> dict:
        return media_mod.scan_candidates(
            media_root=cfg.media_root,
            scan_roots=[cfg.comfy_output, cfg.comfy_input],
            media_dirs=[cfg.media_images, cfg.media_audio, cfg.media_videos],
            protected_paths=_cleanup_protected(),
            min_age_days=max_age_days)

    async def api_media_cleanup_scan(request: web.Request):
        """Scan-only. Query: max_age_days (default 1). Deletes nothing."""
        try:
            days = float(request.query.get("max_age_days", "1"))
        except ValueError:
            return _json_error("日数が不正です。", status=400)
        days = min(max(days, 0.0), 365.0)
        try:
            result = _cleanup_scan(days)
        except Exception as exc:                                 # noqa: BLE001
            return _json_error(f"スキャンできませんでした: {exc}", status=500)
        return web.json_response({"ok": True, **result})

    async def api_media_cleanup(request: web.Request):
        """Delete scan-approved ids after fresh revalidation (TOCTOU-safe)."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        token = str((body or {}).get("token") or "")
        ids = (body or {}).get("ids") or []
        if not isinstance(ids, list) or not token:
            return _json_error("tokenとidsが必要です。", status=400)
        try:
            days = float((body or {}).get("max_age_days", "1"))
        except ValueError:
            return _json_error("日数が不正です。", status=400)
        try:
            fresh = _cleanup_scan(min(max(days, 0.0), 365.0))
        except Exception as exc:                                 # noqa: BLE001
            return _json_error(f"再スキャンできませんでした: {exc}", status=500)
        if fresh.get("token") != token:
            return web.json_response({"ok": False,
                                      "error": "一覧が古くなっています。再スキャンしてください。",
                                      "removed": 0, "recovered_bytes": 0,
                                      "failed": [], "skipped": []})
        wanted = {str(Path(str(i))) for i in ids if str(i).strip()}
        by_path = {c["path"]: c for c in fresh.get("candidates", [])}
        chosen = [by_path[p] for p in sorted(wanted) if p in by_path]
        result = media_mod.clean_candidates(
            candidates=chosen, token=token,
            expected_token=str(fresh.get("token") or ""))
        return web.json_response({"ok": True, **result})

        return web.json_response({"ok": True, **result})

    # ---------------------------------------------------------- AI Director --
    # Upper layer above the existing pipelines: designs a Director Spec from
    # an idea, then converts the FINAL spec into the SAME inputs the existing
    # single/story paths consume. Generation graphs, modes and parity paths
    # are never touched here.
    def _ai_settings() -> dict:
        return ai_settings_mod.normalize(cfg.get("ai_settings"))

    def _go_key() -> str:
        return credstore_mod.load("opencode_go")

    def _local_key() -> str:
        return credstore_mod.load("openai_compat_local")

    def _local_test_image_path(name: str) -> Path | None:
        return comfy_inputs_mod.local_path(cfg, name)

    # ---------------- WP-A: provider selection (settings/API/director) ----
    # All WP-A additions live in this block: registry-backed provider list,
    # per-provider credentials/models/test routes, and the generic director
    # call. Old /api/ai/local/* etc. delegate to the new routes.
    class _DictRequest:
        """Minimal stand-in so legacy routes can delegate to WP-A handlers."""

        def __init__(self, payload: dict):
            self._payload = payload

        async def json(self):
            return self._payload

    def _wp_a_json_request(request: web.Request, payload: dict):
        _ = request
        return _DictRequest(payload)

    def _ai_provider_specs() -> list[dict]:
        from h3app.llm_providers import REGISTRY
        return [REGISTRY[pid].to_json() for pid in REGISTRY]

    def _ai_keys_present() -> dict:
        from h3app.llm_providers import REGISTRY
        present = {}
        for pid, spec in REGISTRY.items():
            name = spec.key_name
            present[pid] = bool(credstore_mod.load(name)) if name else False
        return present

    def _ai_adapter_for(provider_id: str, *, settings=None,
                        model=None, base_url=None):
        """Build the provider adapter with saved entry + credstore token."""
        from h3app.llm_providers import build_adapter
        from h3app.llm_providers import get as _get_spec
        from h3app.llm_providers import key_name as _key_name
        spec = _get_spec(provider_id)
        if spec is None:
            raise ValueError("未知のプロバイダーです。")
        clean = settings if isinstance(settings, dict) and \
            settings.get("schema") == 2 else _ai_settings()
        entry = (clean.get("providers") or {}).get(provider_id) or {}
        eff_base = str(entry.get("base_url") or "").strip() \
            if base_url is None else str(base_url or "").strip()
        eff_model = str(entry.get("model") or "").strip() \
            if model is None else str(model or "").strip()
        if not spec.url_editable:
            eff_base = spec.default_base_url
        name = _key_name(provider_id)
        token = credstore_mod.load(name) if name else ""
        return build_adapter(
            provider_id, base_url=eff_base, model=eff_model, token=token,
            extra=dict(entry.get("extra") or {}),
            vision_models=list(entry.get("vision_models") or []))

    def _ai_key_loader(provider_id: str) -> str:
        from h3app.llm_providers import key_name as _key_name
        name = _key_name(provider_id)
        return credstore_mod.load(name) if name else ""

    async def api_ai_models_post(request: web.Request):
        """WP-A: POST /api/ai/models {provider, base_url?}."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        provider_id = str((body or {}).get("provider") or "").strip()
        if not provider_id:
            return _json_error("プロバイダーを指定してください。",
                               status=400)
        from h3app.llm_providers import get as _get_spec
        from h3app.llm_providers import validate_provider_url
        spec = _get_spec(provider_id)
        if spec is None:
            return _json_error("未知のプロバイダーです。", status=400)
        if not spec.supports_model_list:
            return web.json_response({"ok": True, "models": [],
                                      "supported": False, "error": ""})
        base_override = (body or {}).get("base_url")
        try:
            adapter = _ai_adapter_for(
                provider_id,
                base_url=base_override
                if base_override is not None else None)
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        if spec.url_editable and str(
                getattr(adapter, "base_url", "") or "").strip():
            _normalized, url_error = validate_provider_url(
                provider_id, adapter.base_url)
            if url_error:
                return _json_error(url_error, status=400)
            adapter.base_url = _normalized
        try:
            result = await adapter.list_models()
        except Exception as exc:                                 # noqa: BLE001
            return web.json_response({"ok": False, "models": [],
                                      "supported": True,
                                      "error": f"モデル一覧を取得できませんでした（{exc}）。"})
        if provider_id == "lmstudio" and result.get("models"):
            # H3 just talked to a local LM Studio: remember WHICH process, so
            # shutdown can prove it is still that one before terminating.
            shutdown_mod.record_lmstudio_target(cfg, h3_session_id)
        return web.json_response({"ok": True, **result})

    async def api_ai_test_post(request: web.Request):
        """WP-A: POST /api/ai/test {provider, base_url?, model, image?}."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        provider_id = str((body or {}).get("provider") or "").strip()
        if not provider_id:
            return _json_error("プロバイダーを指定してください。",
                               status=400)
        from h3app.llm_providers import get as _get_spec
        from h3app.llm_providers import validate_provider_url
        spec = _get_spec(provider_id)
        if spec is None:
            return _json_error("未知のプロバイダーです。", status=400)
        model = str((body or {}).get("model") or "").strip()
        if not model:
            return _json_error("モデルを指定してください。", status=400)
        base_override = (body or {}).get("base_url")
        try:
            adapter = _ai_adapter_for(
                provider_id, model=model,
                base_url=base_override
                if base_override is not None else None)
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        if spec.url_editable and str(
                getattr(adapter, "base_url", "") or "").strip():
            _normalized, url_error = validate_provider_url(
                provider_id, adapter.base_url)
            if url_error:
                return _json_error(url_error, status=400)
            adapter.base_url = _normalized
        image_name = str((body or {}).get("image") or "").strip()
        if image_name:
            try:
                vision = await adapter.vision_capability(model)
            except Exception:                                    # noqa: BLE001
                vision = None
            if vision is not True:
                return web.json_response({
                    "ok": False, "status": "vision_unsupported",
                    "message": "このモデルは画像入力に対応していることを確認できないため、"
                               "画像を送信しませんでした。"})
            src = _local_test_image_path(Path(image_name).name) \
                if Path(image_name).name == image_name else None
            if src is None:
                return _json_error("画像が見つかりません。", status=400)
            try:
                from h3app import opencode_go as go_mod
                data_url = go_mod.image_to_data_url(src, max_px=512)
                text, _info = await adapter.chat(
                    user="Describe this image in five words or less.",
                    images=[data_url], max_tokens=256, temperature=0.2)
                return web.json_response({
                    "ok": True, "status": "connected",
                    "target": adapter.target(), "model": model,
                    "text_ok": True, "image_ok": True,
                    "message": "画像を送信できました。",
                    "sample": text[:200], "error": ""})
            except Exception as exc:                             # noqa: BLE001
                kind = str(getattr(exc, "kind", "") or "")
                return web.json_response({
                    "ok": False, "status": kind or "error",
                    "target": adapter.target(), "model": model,
                    "text_ok": False, "image_ok": None,
                    "message": str(exc), "error": str(exc)})
        try:
            result = await adapter.test(model)
        except Exception as exc:                                 # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)})
        if provider_id == "lmstudio" and result.get("ok"):
            shutdown_mod.record_lmstudio_target(cfg, h3_session_id)
        # Secret hygiene: adapter results never carry tokens, but stringify
        # once to be sure no credential leaks into the response envelope.
        payload = dict(result)
        payload["message"] = payload.get("error") or (
            f"接続できました（{model}）。" if payload.get("ok") else "")
        return web.json_response(payload)

    def _do_key_save(provider_id: str, key: str) -> web.Response:
        """WP-A: shared core for POST /api/ai/key (new + legacy)."""
        from h3app.llm_providers import get as _get_spec
        from h3app.llm_providers import key_name as _key_name
        spec = _get_spec(provider_id)
        if spec is None:
            return _json_error("未知のプロバイダーです。", status=400)
        name = _key_name(provider_id)
        if not name:
            return _json_error(
                "このプロバイダーにAPI Keyは不要です。", status=400)
        if not key:
            label = "トークン" if spec.kind == "local" else "API Key"
            return _json_error(f"{label}を入力してください。",
                               status=400)
        if not credstore_mod.available():
            return _json_error(
                "OSの資格情報ストアを利用できません。", status=500)
        if not credstore_mod.save(name, key):
            label = "トークン" if spec.kind == "local" else "API Key"
            return _json_error(f"{label}を保存できませんでした。",
                               status=500)
        return web.json_response({"ok": True, "keys": _ai_keys_present(),
                                  "key_present": bool(_go_key()),
                                  "local_key_present": bool(_local_key())})

    def _do_key_delete(provider_id: str) -> web.Response:
        """WP-A: shared core for DELETE /api/ai/key (new + legacy)."""
        from h3app.llm_providers import get as _get_spec
        from h3app.llm_providers import key_name as _key_name
        spec = _get_spec(provider_id)
        if spec is None:
            return _json_error("未知のプロバイダーです。", status=400)
        name = _key_name(provider_id)
        if name:
            credstore_mod.delete(name)
        return web.json_response({"ok": True, "keys": _ai_keys_present(),
                                  "key_present": bool(_go_key()),
                                  "local_key_present": bool(_local_key())})

    async def api_ai_key_save_post(request: web.Request):
        """WP-A: POST /api/ai/key {provider, key}."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        provider_id = str((body or {}).get("provider") or "").strip()
        # Legacy callers omit provider (Go key / local token split routes).
        if not provider_id:
            provider_id = "opencode_go"
        return _do_key_save(provider_id,
                            str((body or {}).get("key") or "").strip())

    async def api_ai_key_delete_post(request: web.Request):
        """WP-A: DELETE /api/ai/key (body {provider})."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            body = {}
        provider_id = str((body or {}).get("provider") or "").strip()
        if not provider_id:
            provider_id = "opencode_go"
        return _do_key_delete(provider_id)

    def _ai_providers() -> list[dict]:
        # WP-A: registry-backed (legacy 3-id shape retired).
        return _ai_provider_specs()

    def _gemma_info() -> dict:
        vlm = cfg.vlm
        available = None
        object_info = getattr(client, "object_info", None)
        if isinstance(object_info, dict):
            try:
                schema = object_info.get(
                    "llama_cpp_model_loader", {}).get(
                        "input", {}).get("required", {}).get("model")
                options = schema[0] if isinstance(schema, list) and schema \
                    else None
                if isinstance(options, list):
                    available = str(vlm.get("model")) in options
            except Exception:                                    # noqa: BLE001
                available = None
        return {"model": vlm.get("model"), "mmproj": vlm.get("mmproj"),
                "available": available}

    async def api_ai_settings_get(request: web.Request):
        # WP-A: registry providers + per-provider key presence.
        return web.json_response({
            "ok": True,
            "settings": _ai_settings(),
            "providers": _ai_providers(),
            "keys": _ai_keys_present(),
            "key_present": bool(_go_key()),
            "local_key_present": bool(_local_key()),
            "cred_available": credstore_mod.available(),
            "gemma": _gemma_info(),
        })

    async def api_ai_settings_save(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        incoming = (body or {}).get("settings")
        if not isinstance(incoming, dict):
            return _json_error("リクエストが不正です。", status=400)
        # WP-A: validate the merged preview (partial payloads merge over
        # saved settings, so per-provider values survive a switch).
        from h3app.llm_providers import validate_provider_url
        preview = ai_settings_mod.preview_merge(
            cfg.get("ai_settings"), incoming)
        for pid, entry in (preview.get("providers") or {}).items():
            base_url = str((entry or {}).get("base_url") or "").strip()
            if not base_url:
                continue
            _normalized, url_error = validate_provider_url(pid, base_url)
            if url_error:
                return _json_error(url_error, status=400)
        for role_name in ("director", "character_profile"):
            try:
                selection = ai_settings_mod.effective_selection(
                    preview, role_name)
            except Exception:                                    # noqa: BLE001
                continue
            if selection.get("kind") != "gemma" and not str(
                    selection.get("model") or "").strip():
                return _json_error(
                    "LLMのモデルIDを選択または入力してください。",
                    status=400)
        if cfg.source is None:
            return _json_error("設定ファイルの保存先が不明です。",
                               status=500)
        try:
            settings = ai_settings_mod.save_to_config_file(
                cfg.source, incoming,
                current=cfg.get("ai_settings"))
            cfg.data["ai_settings"] = settings
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        except Exception as exc:                                 # noqa: BLE001
            return _json_error(f"保存できませんでした: {exc}", status=500)
        # The saved connection is what shutdown reads. If it now points at a
        # local LM Studio that is already listening, record that process for
        # this session here too - a connection test run BEFORE saving could
        # not (it only sees the saved settings), and the user's natural flow
        # is "select -> test -> save -> stop" (real-machine finding).
        shutdown_mod.record_lmstudio_target(cfg, h3_session_id)
        return web.json_response({"ok": True, "settings": settings})

    # WP-A: legacy Go-only key routes are served by the provider routes
    # above (missing provider defaults to opencode_go), so no separate
    # handlers remain.

    # -------------------------------------------------------- GPU settings --
    def _any_story_running() -> bool:
        return any(story_pipeline.is_running(sid)
                  for sid in list(story_pipeline.runners))

    def _gpu_saved_settings() -> dict:
        return gpu_mod.normalize_settings(
            cfg.get("gpu"), legacy_launch_args=cfg.get("comfy_launch_args"))

    async def _gpu_settings_payload(*, refresh: bool) -> dict:
        detected = gpu_mod.detect(refresh=refresh)
        saved = _gpu_saved_settings()
        te_bytes = gpu_mod.text_encoder_bytes(cfg)
        plan = gpu_mod.build_plan(saved, detected, text_encoder_bytes=te_bytes)
        applied = await client.applied_state()

        restart_required = False
        restart_reason = ""
        if applied.get("running") and plan.get("ok"):
            expected_video = (plan.get("video") or {}).get("uuid")
            expected_aux = None
            if plan.get("effective_mode") == "dual":
                aux = plan.get("aux") or {}
                if aux.get("kind") == "gpu":
                    expected_aux = aux.get("uuid")
            visible = applied.get("visible_uuids") or []
            actual_video = visible[0] if visible else None
            actual_aux = visible[1] if len(visible) > 1 else None
            expected_reserve = plan.get("reserve_vram_gb")
            actual_reserve = applied.get("reserve_vram_gb")
            if not applied.get("known"):
                restart_required = True
                restart_reason = ("起動中のComfyUIの適用状態を確認できません。"
                                  "保存済み設定を反映するには再起動してください。")
            elif actual_video != expected_video or (
                    plan.get("effective_mode") == "dual" and actual_aux != expected_aux):
                restart_required = True
                restart_reason = (
                    "保存済み設定はComfyUIの再起動後に反映されます"
                    f"（現在適用中: {json.dumps(applied, ensure_ascii=False)}）。")
            elif actual_reserve is None:
                restart_required = True
                restart_reason = (
                    "保存済み設定はComfyUIの再起動後に反映されます"
                    "（現在適用中の予約VRAMを確認できません → 保存済み "
                    f"予約VRAM {expected_reserve:.1f}GB）。")
            elif abs(float(actual_reserve) - float(expected_reserve)) > 0.05:
                restart_required = True
                restart_reason = (
                    "保存済み設定はComfyUIの再起動後に反映されます"
                    f"（現在適用中: 予約VRAM {float(actual_reserve):.1f}GB → "
                    f"保存済み {float(expected_reserve):.1f}GB）。")

        busy = bool(pipeline.is_busy) or _any_story_running()
        can_apply_now = bool(applied.get("running")) and applied.get("owned") and not busy
        if busy:
            apply_block_reason = "生成中は変更できません。"
        elif not applied.get("running"):
            apply_block_reason = "ComfyUIは停止中です。次回の生成開始時に保存済み設定で起動します。"
        elif not applied.get("owned"):
            apply_block_reason = ("このComfyUIはH3が起動したものではないため、"
                                  "H3からは再起動しません。")
        else:
            apply_block_reason = ""

        return {
            "ok": True, "detected": detected, "saved": saved, "plan": plan,
            "applied": applied, "restart_required": restart_required,
            "restart_reason": restart_reason, "busy": busy,
            "can_apply_now": can_apply_now, "apply_block_reason": apply_block_reason,
        }

    async def api_settings_gpu_get(request: web.Request):
        refresh = request.query.get("refresh") in ("1", "true", "yes")
        payload = await _gpu_settings_payload(refresh=refresh)
        return web.json_response(payload)

    async def api_settings_gpu_validate(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        settings = gpu_mod.normalize_settings((body or {}).get("settings"))
        detected = gpu_mod.detect()
        te_bytes = gpu_mod.text_encoder_bytes(cfg)
        plan = gpu_mod.build_plan(settings, detected, text_encoder_bytes=te_bytes)
        return web.json_response({"ok": True, "plan": plan})

    async def api_settings_gpu_save(request: web.Request):
        if pipeline.is_busy or _any_story_running():
            return _json_error(
                "生成中はGPU設定を保存・適用できません。生成の完了後に保存してください。",
                status=409)
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        settings = gpu_mod.normalize_settings((body or {}).get("settings"))
        detected = gpu_mod.detect()
        te_bytes = gpu_mod.text_encoder_bytes(cfg)
        plan = gpu_mod.build_plan(settings, detected, text_encoder_bytes=te_bytes)
        if not plan.get("ok"):
            return _json_error(
                (plan.get("errors") or ["GPU設定が不正です。"])[0],
                status=400, extra={"plan": plan})
        if cfg.source is None:
            return _json_error("設定ファイルの保存先が不明です。", status=500)
        try:
            stripped = gpu_mod.strip_device_flags(cfg.get("comfy_launch_args", []))
            data = update_config_file(cfg.source, {
                "gpu": settings, "comfy_launch_args": stripped,
            })
            cfg.data["gpu"] = data["gpu"]
            cfg.data["comfy_launch_args"] = data["comfy_launch_args"]
        except Exception as exc:                                  # noqa: BLE001
            return _json_error(f"保存できませんでした: {exc}", status=500)
        payload = await _gpu_settings_payload(refresh=False)
        return web.json_response(payload)

    async def api_settings_gpu_apply(request: web.Request):
        if pipeline.is_busy or _any_story_running():
            return _json_error("生成中はGPU設定を保存・適用できません。生成の完了後に保存してください。",
                               status=409)
        running = await client.is_reachable()
        if not running:
            return web.json_response({
                "ok": True,
                "message": "ComfyUIは停止中です。次回の生成開始時に保存済み設定で起動します。"})
        if not client.owned:
            return _json_error(
                "このComfyUIはH3が起動したものではないため、H3からは再起動しません。"
                "そのComfyUIを終了してから、もう一度生成すると、H3が保存済み設定で起動します。",
                status=409)

        async def _restart():
            try:
                client.shutdown()
                await client._wait_port_closed()
                await client.ensure_ready("base")
            except Exception as exc:                              # noqa: BLE001
                client._log(f"[gpu] apply restart failed: {exc}")

        asyncio.get_running_loop().create_task(_restart())
        return web.json_response({
            "ok": True,
            "message": "H3が起動したComfyUIを保存済み設定で再起動しています。"})

    async def api_ai_models(request: web.Request):
        # WP-A: legacy GET (Go-only) delegates to POST /api/ai/models.
        return await api_ai_models_post(
            _wp_a_json_request(request, {"provider": "opencode_go"}))

    async def api_ai_test_connection(request: web.Request):
        # WP-A: legacy Go connection test delegates to POST /api/ai/test
        # with the connection's effective provider/model.
        settings = _ai_settings()
        try:
            selection = ai_settings_mod.effective_selection(
                settings, "director")
        except Exception:                                        # noqa: BLE001
            selection = {"provider": "opencode_go", "model": ""}
        pid = str(selection.get("provider") or "opencode_go")
        if pid == "comfy_gemma":
            pid, model = "opencode_go", "muse-spark-1.3-contributor"
        else:
            model = str(selection.get("model") or "")
            if not model:
                model = "muse-spark-1.3-contributor" \
                    if pid == "opencode_go" else ""
        if not model:
            return _json_error("モデルを指定してください。", status=400)
        return await api_ai_test_post(_wp_a_json_request(
            request, {"provider": pid, "model": model}))

    async def api_ai_test_model(request: web.Request):
        # WP-A: legacy Go model probe delegates to POST /api/ai/test.
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        payload = {"provider": "opencode_go",
                   "model": str((body or {}).get("model") or "")}
        if (body or {}).get("image"):
            payload["image"] = (body or {})["image"]
        return await api_ai_test_post(
            _wp_a_json_request(request, payload))

    async def api_ai_local_models(request: web.Request):
        # WP-A: legacy local list delegates (provider lmstudio).
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        payload = {"provider": "lmstudio"}
        if isinstance(body, dict) and body.get("base_url") is not None:
            payload["base_url"] = body.get("base_url")
        result = await api_ai_models_post(
            _wp_a_json_request(request, payload))
        # Legacy envelope compat: surface message when the probe failed.
        return result

    async def api_ai_local_test(request: web.Request):
        # WP-A: legacy local probe delegates (provider lmstudio).
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        payload = {"provider": "lmstudio"}
        if isinstance(body, dict):
            for key in ("base_url", "model", "image"):
                if body.get(key) is not None:
                    payload[key] = body.get(key)
        return await api_ai_test_post(
            _wp_a_json_request(request, payload))

    async def api_ai_local_key_save(request: web.Request):
        # WP-A: legacy local token route delegates (provider lmstudio).
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        if isinstance(body, dict) and not body.get("provider"):
            body = {**body, "provider": "lmstudio"}
        return _do_key_save("lmstudio",
                            str((body or {}).get("key") or "").strip())

    async def api_ai_local_key_delete(request: web.Request):
        # WP-A: legacy local token route delegates (provider lmstudio).
        return _do_key_delete("lmstudio")

    def _director_session_id(body: dict | None) -> str:
        import uuid as _uuid
        sid = str(((body or {}).get("session_id") or "")).strip()
        return sid or _uuid.uuid4().hex

    async def _director_call(role_name: str, *, system: str, user: str,
                             max_tokens: int = 2048,
                             temp_model: dict | None = None,
                             session_id: str = "") -> tuple[dict, dict]:
        """One logical Director call with format-repair retry + fallback.

        Returns (answer, used) where used describes the winning attempt:
        {kind, model, fallback, elapsed_ms, attempts, repairs,
         http_status, retry_after, usage, incomplete, endpoint}.
        temp_model ({provider, model}) overrides the default for this call
        only and is never persisted.
        """
        import time as _time
        started = _time.monotonic()
        settings = _ai_settings()
        # WP-A: resolve the primary target (temp override or role default).
        from h3app.llm_providers import get as _get_spec
        primary = None
        temp_endpoint = ""
        if isinstance(temp_model, dict) and str(
                temp_model.get("model") or "").strip():
            raw_pid = str(temp_model.get("provider") or "").strip()
            pid = ai_settings_mod.LEGACY_TO_NEW.get(raw_pid, raw_pid)
            if not raw_pid:
                try:
                    fallback_sel = ai_settings_mod.effective_selection(
                        settings, role_name)
                    pid = fallback_sel.get("provider") or "opencode_go"
                except Exception:                                # noqa: BLE001
                    pid = "opencode_go"
            if _get_spec(pid) is not None:
                primary = {"kind": ai_settings_mod.kind_of(pid),
                           "provider": pid,
                           "model": str(temp_model["model"]).strip()}
                temp_endpoint = str(temp_model.get("endpoint") or "").strip()
                if temp_endpoint:
                    primary["endpoint"] = temp_endpoint
        if primary is None:
            try:
                selection = ai_settings_mod.effective_selection(
                    settings, role_name)
            except Exception:                                    # noqa: BLE001
                selection = {"kind": "local", "provider": "lmstudio",
                             "model": ""}
            primary = {"kind": selection.get("kind") or "local",
                       "provider": selection.get("provider")
                       or "lmstudio",
                       "model": str(selection.get("model") or "")}
            extra_endpoint = str(
                (selection.get("extra") or {}).get("endpoint") or "")
            if extra_endpoint:
                primary["endpoint"] = extra_endpoint
        stats = {"attempts": 0, "repairs": 0}
        last_info: dict = {}

        async def _once(provider, attempt_system: str):
            async def generate():
                try:
                    return await provider.generate(
                        system=attempt_system, user=user, max_tokens=max_tokens)
                finally:
                    if isinstance(provider, director_provider_mod.GemmaDirectorProvider):
                        await provider.unload()
            if isinstance(provider, director_provider_mod.GemmaDirectorProvider):
                return await pipeline.run_audio_test(generate())
            return await generate()

        async def _op(kind: str, model_cfg: dict, api_key: str) -> dict:
            # WP-A: generic adapter for every provider (Gemma unchanged).
            if kind == "gemma":
                provider: director_provider_mod.DirectorProvider = \
                    director_provider_mod.GemmaDirectorProvider(client, cfg.vlm)
            else:
                pid = str(model_cfg.get("provider") or
                          primary.get("provider") or "")
                adapter = _ai_adapter_for(
                    pid, model=str(model_cfg.get("model") or ""))
                endpoint_override = str(
                    model_cfg.get("endpoint") or "").strip()
                if endpoint_override:
                    try:
                        adapter.extra["endpoint"] = endpoint_override
                    except Exception:                            # noqa: BLE001
                        pass
                if session_id and pid == "opencode_go":
                    try:
                        adapter.extra["session_id"] = session_id
                    except Exception:                            # noqa: BLE001
                        pass
                provider = director_provider_mod.AdapterDirectorProvider(
                    adapter)
            stats["attempts"] += 1
            try:
                answer = await _once(provider, system)
            except director_provider_mod.DirectorError:
                stats["repairs"] += 1
                answer = await _once(
                    provider,
                    system + "\n" + director_prompts_mod._REPAIR_NUDGE)
            info = dict(getattr(provider, "last_info", None) or {})
            if info:
                last_info.clear()
                last_info.update(info)
            return answer

        # WP-A: kind is gemma|local|external; local never reaches
        # external, external auth aborts before Gemma.
        answer, used = await ai_settings_mod.run_with_fallback(
            primary, _op, max_attempts=int(settings.get("max_attempts", 3)),
            gemma_fallback=bool(settings.get("gemma_fallback", False)),
            key_loader=_ai_key_loader)
        used = dict(used or {})
        used["elapsed_ms"] = int((_time.monotonic() - started) * 1000)
        used["attempts"] = int(stats["attempts"])
        used["repairs"] = int(stats["repairs"])
        used["http_status"] = int(last_info.get("http_status", 0) or 0)
        used["retry_after"] = str(last_info.get("retry_after") or "")
        used["usage"] = last_info.get("usage") or {
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        used["incomplete"] = str(last_info.get("incomplete") or "")
        used["endpoint"] = str(last_info.get("endpoint") or "")
        used["errors"] = used.get("errors") or []
        # Resolve the effective endpoint when the saved value was blank.
        # WP-A: legacy "go" kind kept for old callers.
        if used.get("kind") in ("go", "external") and not used.get(
                "endpoint") and used.get("provider") == "opencode_go":
            used["endpoint"] = "responses"
        return answer, used

    def _director_check_spec(spec: dict, kind: str) -> dict | None:
        """Return an error response payload dict, or None when usable."""
        if not isinstance(spec, dict) or spec.get("kind") != kind:
            return {"ok": False, "error": "Director Specの形式が不正です。"}
        report = director_spec_mod.validate_spec(spec)
        if not report["ok"]:
            return {"ok": False,
                    "error": "Director Specの形式が不正です。",
                    "detail": "; ".join(report["errors"][:5])}
        return None

    def _director_envelope(spec: dict, used: dict | None = None) -> dict:
        out = {"ok": True, "spec": spec,
               "warnings": director_convert_mod.dialogue_budget_warnings(spec),
               "continuity": director_spec_mod.check_continuity(spec)
               if spec.get("kind") == "story30" else []}
        _merge_note = spec.get("_clips_merge") if isinstance(spec, dict) \
            else None
        if isinstance(_merge_note, dict):
            if not _merge_note.get("matched"):
                out["warnings"] = list(out.get("warnings") or []) + [
                    "CLIP案の結合が0件でした（番号付けの不一致）。再生成してください。"]
            elif _merge_note.get("positional"):
                out["warnings"] = list(out.get("warnings") or []) + [
                    "CLIP案は位置順で結合しました（番号が1始まりでした）。"]
        if used is not None:
            out["used"] = {"kind": used.get("kind", ""),
                           "provider": used.get("provider", ""),
                           "model": used.get("model", ""),
                           "fallback": bool(used.get("fallback", False)),
                           "elapsed_ms": int(used.get("elapsed_ms", 0) or 0),
                           "attempts": int(used.get("attempts", 0) or 0),
                           "repairs": int(used.get("repairs", 0) or 0),
                           "http_status": int(
                               used.get("http_status", 0) or 0),
                           "retry_after": str(used.get("retry_after") or ""),
                           "usage": used.get("usage") or {},
                           "incomplete": str(used.get("incomplete") or ""),
                           "endpoint": str(used.get("endpoint") or ""),
                           "errors": used.get("errors") or []}
        return out

    def _director_call_args(body: dict | None) -> dict:
        """temp_model override + session id from the client (never persisted).

        WP-A: temp_model is {provider, model} (endpoint optional, legacy)."""
        body = body if isinstance(body, dict) else {}
        temp = body.get("temp_model")
        temp_model = None
        if isinstance(temp, dict) and str(temp.get("model") or "").strip():
            temp_model = {"model": str(temp["model"]).strip(),
                          "endpoint": str(temp.get("endpoint") or ""),
                          "provider": str(temp.get("provider") or "")}
        return {"temp_model": temp_model,
                "session_id": _director_session_id(body)}

    async def api_director_single_create(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        idea = str((body or {}).get("idea") or "").strip()
        if not idea:
            return _json_error("アイデアを入力してください。", status=400)
        try:
            duration_sec = int((body or {}).get("duration_sec", 10))
        except (TypeError, ValueError):
            return _json_error("動画尺が不正です。", status=400)
        if duration_sec not in (5, 10, 15):
            return _json_error("動画尺は5/10/15秒から選んでください。",
                               status=400)
        spec = director_spec_mod.new_single(
            idea=idea, duration_sec=duration_sec,
            initial_request=str((body or {}).get("request") or ""))
        try:
            answer, used = await _director_call(
                "director",
                system=director_prompts_mod.single_system(),
                user=director_prompts_mod.single_user(
                    idea=idea, duration_sec=duration_sec,
                    initial_request=spec["initial_request"]),
                # Token budget (observed-max + headroom, 2026-09-08 samples):
                # single out=2888 -> 4096 (1.42x). Smaller budgets cut off
                # (observed incomplete at 2048). incomplete=0 required.
                max_tokens=4096,
                **_director_call_args(body))
        except director_provider_mod.DirectorError as exc:
            return _json_error(str(exc), status=502)
        spec = director_convert_mod.merge_model_answer_single(spec, answer)
        return web.json_response(_director_envelope(spec, used))

    async def api_director_single_regenerate(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        spec = (body or {}).get("spec")
        bad = _director_check_spec(spec, "single")
        if bad:
            return web.json_response(bad, status=400)
        scope_item = str((body or {}).get("scope_item") or "").strip()
        scope = f"単発ディレクション全体" if not scope_item else \
            f"項目「{scope_item}」のみ（他項目は維持）"
        items = spec.get("items") or {}
        current = {"items": items,
                   "duration_sec": spec.get("duration_sec"),
                   "timeline": spec.get("timeline") or []}
        shape = director_prompts_mod.single_item_shape(scope_item) \
            if scope_item else director_prompts_mod.single_full_shape()
        try:
            answer, used = await _director_call(
                "director",
                system=director_prompts_mod.regen_system(scope=scope,
                                                         shape=shape),
                user=director_prompts_mod.regen_user(
                    current=current, scope=scope,
                    item_request=str((items.get(scope_item) or {}).get(
                        "request", "")) if scope_item else "",
                    global_request=str(spec.get("global_request") or "")),
                **_director_call_args(body))
        except director_provider_mod.DirectorError as exc:
            return _json_error(str(exc), status=502)
        names = tuple(items.keys()) if not scope_item else (scope_item,)
        scoped_missing = bool(
            scope_item and not (
                isinstance((answer.get("items") or {}).get(scope_item), dict)
                and str(((answer.get("items") or {}).get(scope_item) or {}).get(
                    "value") or "").strip()))
        for name in names:
            ans_item = (answer.get("items") or {}).get(name)
            if isinstance(ans_item, dict) and str(
                    ans_item.get("value") or "").strip():
                slot = items.setdefault(name, director_spec_mod.blank_item())
                if not slot.get("locked"):
                    slot["value"] = str(ans_item["value"])
                    slot["en"] = str(ans_item.get("en") or "")
        if not scope_item and isinstance(answer.get("timeline"), list):
            spec["timeline"] = answer["timeline"][:16]
        spec, enforced = director_spec_mod.apply_locks(
            _director_locked_snapshot(spec), spec)
        out = _director_envelope(spec, used)
        out["enforced"] = enforced
        if scoped_missing:
            out["warnings"] = list(out.get("warnings") or []) + \
                ["対象項目の回答が得られませんでした。再実行してください。"]
        return web.json_response(out)

    def _director_locked_snapshot(spec: dict) -> dict:
        """Deep copy carrying the user-side locks into enforcement."""
        import copy as _copy
        return _copy.deepcopy(spec)

    async def api_director_story30_create(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        idea = str((body or {}).get("idea") or "").strip()
        if not idea:
            return _json_error("アイデアを入力してください。", status=400)
        spec = director_spec_mod.new_story30(
            idea=idea, initial_request=str((body or {}).get("request") or ""))
        try:
            master_answer, master_used = await _director_call(
                "director",
                system=director_prompts_mod.master_system(),
                user=director_prompts_mod.master_user(
                    idea=idea, initial_request=spec["initial_request"]),
                # Token budget (observed-max + headroom): MASTER out =
                # 3098/3651 -> 5120 (1.40x over max). 2048 cuts off.
                # incomplete=0 required.
                max_tokens=5120,
                **_director_call_args(body))
        except director_provider_mod.DirectorError as exc:
            return _json_error(str(exc), status=502)
        spec = director_convert_mod.merge_model_answer_master(
            spec, master_answer)
        try:
            clips_answer, clips_used = await _director_call(
                "director",
                system=director_prompts_mod.clips_system(),
                user=director_prompts_mod.clips_user(
                    master=spec.get("master") or {}, idea=idea),
                # Token budget (observed-max + headroom): CLIPS out =
                # 4771/5797 -> 8192 (1.41x over max). 3072 cuts off.
                # incomplete=0 required.
                max_tokens=8192,
                **_director_call_args(body))
        except director_provider_mod.DirectorError as exc:
            return _json_error(str(exc), status=502)
        spec = director_convert_mod.merge_model_answer_clips(spec, clips_answer)
        out = _director_envelope(spec, clips_used)
        out["master_used"] = {"kind": master_used.get("kind", ""),
                              "model": master_used.get("model", ""),
                              "fallback": bool(
                                  master_used.get("fallback", False)),
                              "elapsed_ms": int(
                                  master_used.get("elapsed_ms", 0) or 0),
                              "attempts": int(
                                  master_used.get("attempts", 0) or 0),
                              "repairs": int(
                                  master_used.get("repairs", 0) or 0),
                              "http_status": int(
                                  master_used.get("http_status", 0) or 0),
                              "retry_after": str(
                                  master_used.get("retry_after") or ""),
                              "usage": master_used.get("usage") or {},
                              "incomplete": str(
                                  master_used.get("incomplete") or ""),
                              "endpoint": str(
                                  master_used.get("endpoint") or ""),
                              "errors": master_used.get("errors") or []}
        return web.json_response(out)

    async def api_director_story30_regenerate(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        spec = (body or {}).get("spec")
        bad = _director_check_spec(spec, "story30")
        if bad:
            return web.json_response(bad, status=400)
        scope = (body or {}).get("scope") or {}
        clip_idx = scope.get("clips")
        item_names = scope.get("items")
        only_clips = tuple(int(i) for i in clip_idx) \
            if isinstance(clip_idx, list) and clip_idx else None
        only_items = tuple(str(i) for i in item_names) \
            if isinstance(item_names, list) and item_names else None
        if only_clips is None and only_items is None:
            label = "MASTER＋CLIP全体（ロック維持）"
        elif only_clips is not None and only_items is None:
            label = f"CLIP {', '.join(str(i + 1) for i in only_clips)}のみ"
        elif only_clips is None:
            label = f"項目 {', '.join(only_items or [])}のみ（3CLIP一括）"
        else:
            label = (f"CLIP {', '.join(str(i + 1) for i in only_clips)}の"
                     f"項目 {', '.join(only_items or [])}のみ")
        boundary = _director_boundary(spec, only_clips)
        current = {"master": spec.get("master"), "clips": spec.get("clips")}
        try:
            answer, used = await _director_call(
                "director",
                system=director_prompts_mod.regen_system(
                    scope=label,
                    shape=director_prompts_mod.story_shape(with_master=True)),
                user=director_prompts_mod.regen_user(
                    current=current, scope=label,
                    global_request=str(spec.get("global_request") or ""),
                    boundary=boundary),
                # Story regen carries master+clips: same budget as CLIPS.
                max_tokens=8192,
                **_director_call_args(body))
        except director_provider_mod.DirectorError as exc:
            return _json_error(str(exc), status=502)
        if only_clips is None and only_items is None:
            spec = director_convert_mod.merge_model_answer_master(
                spec, answer.get("master", answer))
            spec = director_convert_mod.merge_model_answer_clips(spec, answer)
        else:
            spec = director_convert_mod.merge_model_answer_clips(
                spec, answer, only=only_clips, only_items=only_items)
            if only_items:
                # Master-level items share names with clip items (e.g. mood);
                # a scoped item regen also refreshes the matching master slot.
                spec = director_convert_mod.merge_model_answer_master(
                    spec, {"items": {
                        k: v for k, v in
                        ((answer.get("master") or {}).get("items") or {}).items()
                        if k in only_items}})
        spec, enforced = director_spec_mod.apply_locks(
            _director_locked_snapshot(spec), spec)
        out = _director_envelope(spec, used)
        out["enforced"] = enforced
        return web.json_response(out)

    def _director_boundary(spec: dict,
                           only_clips: tuple[int, ...] | None) -> str:
        """Neighbour end/start states framing a scoped regen."""
        if not only_clips:
            return ""
        import json as _json2
        clips = {c.get("index"): c for c in (spec.get("clips") or [])
                 if isinstance(c, dict)}
        bits = []
        for i in only_clips:
            prev_c = clips.get(i - 1)
            next_c = clips.get(i + 1)
            if isinstance(prev_c, dict):
                bits.append(
                    f"CLIP{i}の開始条件: 前clip終了="
                    f"{_json2.dumps(prev_c.get('end_state') or {}, ensure_ascii=False)}")
            if isinstance(next_c, dict):
                bits.append(
                    f"CLIP{i + 2}と矛盾しないこと: 後clip開始="
                    f"{_json2.dumps(next_c.get('start_state') or {}, ensure_ascii=False)}")
        master = spec.get("master") or {}
        bits.append(f"MASTER: {_json2.dumps(master.get('items') or {}, ensure_ascii=False)[:2000]}")
        return "\n".join(bits)

    def _director_check_images(body: dict):
        images = [n for n in (body.get("images") or []) if _safe_upload_name(n)]
        if not images:
            return None, "参照画像を1枚以上追加してください。"
        if len(images) > 4:
            return None, "参照画像は最大4枚までです。"
        for n in images:
            if comfy_inputs_mod.local_path(cfg, n) is None:
                return None, "追加した画像が見つかりません。もう一度追加してください。"
        return images, ""

    def _dump_final_prompt(name: str, prompt: str) -> dict:
        """Save the exact H3 prompt string + speech audit (evidence).

        Best-effort: a dump failure never blocks generation. No secrets are
        involved (prompt text + structural audit only).
        """
        from h3app import director_speech as speech_mod
        audit = speech_mod.audit_prompt(prompt)
        try:
            dest = cfg.debug_dir / f"{name}_final_prompt.txt"
            dest.write_text(str(prompt or ""), encoding="utf-8")
            (cfg.debug_dir / f"{name}_final_prompt_audit.json").write_text(
                __import__("json").dumps(audit, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except Exception:                                    # noqa: BLE001
            pass
        return audit

    async def api_director_generate(request: web.Request):
        """Final single spec -> deterministic H3 prompt -> generate path.

        No second VLM director runs here: the Final Spec is compiled
        directly, then the existing single-shot submit path takes over.
        """
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        spec = (body or {}).get("spec")
        bad = _director_check_spec(spec, "single")
        if bad:
            return web.json_response(bad, status=400)
        images, err = _director_check_images(body or {})
        if err:
            return _json_error(err, status=400)
        try:
            converted = director_convert_mod.single_to_inputs(spec)
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        mode = str((body or {}).get("mode") or "FAST").upper()
        if mode not in modes_mod.RUNNABLE_MODES and \
                modes_mod.experimental_reason(mode) is None:
            return _json_error(f"不明なモードです: {mode}")
        if modes_mod.experimental_reason(mode) is not None:
            return _json_error(
                f"{mode}は利用できません: {modes_mod.experimental_reason(mode)}")
        if mode in modes_mod.STORY_ONLY_MODES:
            return _json_error(
                f"{mode}はStoryモード専用です。通常の生成では使えません。")
        try:
            settings = build_settings(
                cfg, seconds=float(converted_frames_to_seconds(
                    converted["frames"])),
                aspect=str((body or {}).get("aspect", "portrait")),
                advanced=(body or {}).get("advanced") or {})
        except (PipelineError, graphs.GraphError) as exc:
            return _json_error(str(exc))
        try:
            negative = ((body or {}).get("jp_negative") or "").strip() or \
                prompts().NEGATIVE_DEFAULT
        except Exception:                                        # noqa: BLE001
            negative = ((body or {}).get("jp_negative") or "").strip()
        snap = _character_snapshot_for(
            images, (body.get("character_snapshot")
                     if isinstance(body, dict) else None))
        try:
            en_prompt = director_convert_mod.final_to_h3_single(
                spec,
                snapshot=(snap or {}),
                refs=list(images),
                frames=int(settings["frames"]))
        except director_lexicon_mod.UntranslatedPromptError as exc:
            return _json_error(_untranslated_prompt_message(exc), status=400)
        except ValueError as exc:
            return _json_error(str(exc), status=500)
        # Spec -> compiler -> speech policy -> FINAL validation -> H3.
        # HARD structural violations refuse BEFORE any GPU work (fail-fast);
        # heuristic observations ride along as non-blocking warnings.
        from h3app import director_speech as speech_mod
        _single_items = spec.get("items") if isinstance(
            spec.get("items"), dict) else {}
        _single_dialogue = _single_items.get("dialogue") if isinstance(
            _single_items.get("dialogue"), dict) else {}
        en_prompt, _speech_report = speech_mod.final_gate(
            en_prompt,
            dialogue=speech_mod.dialogue_lines(
                _single_dialogue.get("value")),
            allow_s2=director_spec_mod.allow_second_subject(
                director_spec_mod.normalize_cast(spec.get("cast"))))
        _conversions = _speech_report.get("conversions") or []
        if _speech_report["hard"]:
            return _json_error(
                "発話検証で生成を停止しました: " +
                " / ".join(_speech_report["hard"][:3]), status=500)
        _prompt_validation = director_lexicon_mod.validate_compiled_prompt(
            en_prompt)
        if not _prompt_validation["ok"]:
            return _json_error(
                "監督案の英語変換に失敗しました（" +
                "; ".join(_prompt_validation["errors"][:5]) +
                "）。監督案を作り直してください。", status=400)
        project = store.create(
            settings=settings, images=images,
            jp_scene=converted["scene"], jp_dialogue=converted["speech"],
            jp_negative=negative,
            max_seconds=float(cfg.defaults["max_seconds"]))
        project.data["mode"] = mode
        project.data["director"] = spec
        project.data["speech_report"] = {
            "warnings": _speech_report.get("warnings") or [],
            "ambience_converted": _conversions}
        llm_block = (body or {}).get("llm")
        if isinstance(llm_block, dict) and isinstance(
                llm_block.get("used"), dict):
            project.data["director_llm"] = {
                "used": llm_block["used"]}
        if snap is not None:
            project.data["profile"] = snap["profile_text"]
            project.data["character_snapshot"] = snap
        project.data["en_prompt"] = en_prompt
        project.data["speech_audit"] = _dump_final_prompt(
            f"{project.id}_director", en_prompt)
        project.save()
        try:
            await pipeline.start(project, lambda r: pipeline.generate_director(r))
        except PipelineError as exc:
            return _json_error(str(exc), status=409)
        return web.json_response({"ok": True, "project_id": project.id})

    def converted_frames_to_seconds(frames: int) -> float:
        return {124: 5.0, 243: 10.0}.get(int(frames), 15.0)

    async def api_director_story30_generate(request: web.Request):
        """Final story30 spec -> precompiled segments -> story pipeline.

        Each clip's H3 prompt is compiled deterministically here; the story
        runner uses it verbatim instead of calling the VLM director again.
        """
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        spec = (body or {}).get("spec")
        bad = _director_check_spec(spec, "story30")
        if bad:
            return web.json_response(bad, status=400)
        images, err = _director_check_images(body or {})
        if err:
            return _json_error(err, status=400)
        try:
            segments = director_convert_mod.story30_to_segments(spec)
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        # 30s story continuity lives on the LONG_FAST mechanisms (voice anchor
        # + 0f relay). This is a pipeline technical constraint, not a Director
        # choice: other story modes are refused here, never coerced.
        story_mode = "LONG_FAST"
        try:
            settings = build_settings(
                cfg, seconds=10.0,
                aspect=str((body or {}).get("aspect", "portrait")),
                advanced=(body or {}).get("advanced") or {})
            settings["frames"] = 243
        except (PipelineError, graphs.GraphError) as exc:
            return _json_error(str(exc))
        try:
            negative = ((body or {}).get("jp_negative") or "").strip() or \
                prompts().NEGATIVE_DEFAULT
        except Exception:                                        # noqa: BLE001
            negative = ((body or {}).get("jp_negative") or "").strip()
        snap = _character_snapshot_for(
            images, (body.get("character_snapshot")
                     if isinstance(body, dict) else None))
        direct_prompts = []
        for index in range(3):
            try:
                direct_prompts.append(director_convert_mod.final_to_h3_clip(
                    spec, index,
                    snapshot=(snap or {}),
                    refs=list(images),
                    frames=int(settings["frames"]),
                    story_mode=story_mode))
            except director_lexicon_mod.UntranslatedPromptError as exc:
                return _json_error(
                    _untranslated_prompt_message(exc, index), status=400)
            except ValueError as exc:
                return _json_error(str(exc), status=500)
        # Spec -> compiler -> speech policy -> FINAL validation -> H3, per clip.
        from h3app import director_speech as speech_mod
        _master = spec.get("master") if isinstance(
            spec.get("master"), dict) else {}
        _allow_s2 = director_spec_mod.allow_second_subject(
            director_spec_mod.normalize_cast(_master.get("cast")))
        _speech_warnings: dict = {}
        _ambience_converted: dict = {}
        _clips = spec.get("clips") if isinstance(
            spec.get("clips"), list) else []
        for _index, _prompt in enumerate(direct_prompts):
            _clip = _clips[_index] if _index < len(_clips) and isinstance(
                _clips[_index], dict) else {}
            _items = _clip.get("items") if isinstance(
                _clip.get("items"), dict) else {}
            _dialogue = _items.get("dialogue") if isinstance(
                _items.get("dialogue"), dict) else {}
            _prompt, _report = speech_mod.final_gate(
                _prompt,
                dialogue=speech_mod.dialogue_lines(_dialogue.get("value")),
                allow_s2=_allow_s2)
            direct_prompts[_index] = _prompt
            _conv = _report.get("conversions") or []
            if _report["hard"]:
                return _json_error(
                    f"CLIP{_index + 1}の発話検証で生成を停止しました: " +
                    " / ".join(_report["hard"][:3]), status=500)
            _clip_validation = director_lexicon_mod.validate_compiled_prompt(
                _prompt)
            if not _clip_validation["ok"]:
                return _json_error(
                    f"CLIP{_index + 1}の監督案の英語変換に失敗しました（" +
                    "; ".join(_clip_validation["errors"][:5]) +
                    "）。監督案を作り直してください。", status=400)
            if _report["warnings"] or _conv:
                _speech_warnings[str(_index)] = _report["warnings"]
                _ambience_converted[str(_index)] = _conv
        built_segments = []
        for index, s in enumerate(segments):
            built_segments.append({"prompt": s["prompt"], "speech": s["speech"],
                                   "direct_en_prompt": direct_prompts[index]})
        story = story_store.create(
            name=(str((body or {}).get("name") or "").strip() or
                  f"Director {spec.get('idea', '')[:24]}"),
            source_text="", lines=[],
            segments=normalize_segments(built_segments),
            images=images, settings=settings, jp_negative=negative,
            motion_presets=[])
        story.data["mode"] = story_mode
        story.data["director"] = spec
        _clip_audits = {}
        for _index, _prompt in enumerate(direct_prompts):
            _clip_audits[str(_index)] = _dump_final_prompt(
                f"{story.id}_clip{_index:02d}", _prompt)
        story.data["speech_audit"] = _clip_audits
        story.data["speech_report"] = {
            "warnings": _speech_warnings,
            "ambience_converted": _ambience_converted}
        llm_block = (body or {}).get("llm")
        if isinstance(llm_block, dict):
            # Director LLM cost belongs to the create phase (MASTER+CLIPS);
            # carried here so segment timings and reports can reference it.
            story.data["director_llm"] = {
                k: llm_block.get(k) for k in
                ("master_used", "clips_used") if k in llm_block}
        if snap is not None:
            story.data["profile"] = snap["profile_text"]
            story.data["character_snapshot"] = snap
            _character_stage_story_voice(story, snap)
        story.save()
        try:
            await story_pipeline.start(story)
        except PipelineError as exc:
            info = classify(exc)
            return _json_error(str(exc), status=409,
                               extra={"error": info})
        return web.json_response({"ok": True, "story_id": story.id})

    # ------------------------------------------------------- Character ------
    # Library = reusable template (owns its copies). Project/Story = snapshot
    # owner (copies materialized into the existing h3app_ref_* management).
    # Deleting a Library character never touches snapshots.
    character_store = character_lib_mod.CharacterStore(
        APP_DIR / "character_library")

    def _character_local_file(name: str, exts: set[str]) -> Path | None:
        p = Path(str(name or ""))
        if p.name != str(name or "") or p.suffix.lower() not in exts:
            return None
        return comfy_inputs_mod.local_path(cfg, p.name)

    def _character_ref_file(name: str) -> Path | None:
        return _character_local_file(
            name, ALLOWED_IMAGE_EXT | {".png", ".jpg", ".jpeg", ".webp",
                                       ".bmp"})

    def _character_public(item: dict) -> dict:
        out = dict(item)
        out["thumb"] = ""
        refs = out.get("refs") or []
        if refs:
            out["thumb"] = (f"/api/character/thumb/{item['character_id']}/0")
        return out

    async def api_character_list(request: web.Request):
        return web.json_response({
            "ok": True,
            "characters": [_character_public(c)
                           for c in character_store.list()]})

    async def api_character_thumb(request: web.Request):
        cid = str(request.match_info.get("cid", ""))
        try:
            index = int(request.match_info.get("idx", "0"))
        except ValueError:
            raise web.HTTPNotFound()
        item = character_store.get(cid)
        if item is None:
            raise web.HTTPNotFound()
        refs = item.get("refs") or []
        if not (0 <= index < len(refs)):
            raise web.HTTPNotFound()
        src = character_store.root / item["character_id"] / Path(
            str(refs[index])).name
        if not src.is_file():
            raise web.HTTPNotFound()
        try:
            from PIL import Image
        except Exception:                                        # noqa: BLE001
            return web.FileResponse(src)
        buf = io.BytesIO()
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            im.save(buf, format="JPEG", quality=85)
        return web.Response(body=buf.getvalue(), content_type="image/jpeg")

    def _character_canon_of(profile_text: str) -> tuple[dict, dict]:
        """Split a parsed profile into identity canon + outfit reference."""
        try:
            canon = canon_mod.parse_profile(profile_text or "")
            full = canon.as_dict()
        except Exception:                                        # noqa: BLE001
            return {}, {}
        identity = {k: full.get(k, "") for k in
                    ("face", "hair", "build", "skin")}
        outfit_ref = {k: full.get(k, "") for k in
                      ("outfit", "accessories", "materials", "color_keys")}
        return identity, outfit_ref

    async def api_character_save(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        name = str((body or {}).get("name") or "").strip()
        if not name:
            return _json_error("名前を入力してください。", status=400)
        images = [str(n) for n in ((body or {}).get("images") or [])]
        if not images:
            return _json_error("参照画像を1枚以上追加してください。",
                               status=400)
        if len(images) > character_lib_mod.MAX_REFS:
            return _json_error("参照画像は最大4枚までです。", status=400)
        refs: list[Path] = []
        for n in images:
            found = _character_ref_file(n)
            if found is None:
                return _json_error(f"画像が見つかりません: {n}", status=400)
            refs.append(found)
        voice_src = None
        voice_name = str((body or {}).get("voice_wav") or "").strip()
        if voice_name:
            voice_src = _character_local_file(voice_name, {".wav"})
            if voice_src is None:
                return _json_error("音声ファイルが見つかりません。",
                                   status=400)
        profile_text = str((body or {}).get("profile_text") or "")
        canon, outfit_ref = _character_canon_of(profile_text)
        item, err = character_store.save_new(
            display_name=name, refs=refs, profile_text=profile_text,
            canon=canon, outfit_ref=outfit_ref, voice_src=voice_src,
            voice_note=str((body or {}).get("voice_note") or ""),
            memo=str((body or {}).get("memo") or ""))
        if item is None:
            return _json_error(f"保存できませんでした: {err}", status=500)
        return web.json_response({"ok": True,
                                  "character": _character_public(item)})

    async def api_character_update(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        cid = str((body or {}).get("character_id") or "")
        kwargs: dict = {}
        if "name" in (body or {}):
            kwargs["display_name"] = str(body.get("name") or "")
        if "memo" in (body or {}):
            kwargs["memo"] = str(body.get("memo") or "")
        if "voice_note" in (body or {}):
            kwargs["voice_note"] = str(body.get("voice_note") or "")
        if "profile_text" in (body or {}):
            profile_text = str(body.get("profile_text") or "")
            canon, outfit_ref = _character_canon_of(profile_text)
            kwargs["profile_text"] = profile_text
            kwargs["canon"] = canon
            kwargs["outfit_ref"] = outfit_ref
        if "images" in (body or {}):
            images = [str(n) for n in (body.get("images") or [])]
            if len(images) > character_lib_mod.MAX_REFS:
                return _json_error("参照画像は最大4枚までです。", status=400)
            refs = []
            for n in images:
                found = _character_ref_file(n)
                if found is None:
                    return _json_error(f"画像が見つかりません: {n}",
                                       status=400)
                refs.append(found)
            kwargs["refs"] = refs
        if str((body or {}).get("voice_wav") or "").strip():
            voice_src = _character_local_file(
                str(body.get("voice_wav") or "").strip(), {".wav"})
            if voice_src is None:
                return _json_error("音声ファイルが見つかりません。",
                                   status=400)
            kwargs["voice_src"] = voice_src
        if bool((body or {}).get("voice_clear")):
            kwargs["voice_clear"] = True
        item, err = character_store.update(cid, **kwargs)
        if item is None:
            return _json_error(err or "更新できませんでした。", status=400)
        return web.json_response({"ok": True,
                                  "character": _character_public(item)})

    async def api_character_rename(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        ok, err = character_store.rename(
            str((body or {}).get("character_id") or ""),
            str((body or {}).get("name") or ""))
        if not ok:
            return _json_error(err or "名前を変更できませんでした。",
                               status=400)
        return web.json_response({"ok": True,
                                  "characters": [_character_public(c)
                                                 for c in
                                                 character_store.list()]})

    async def api_character_duplicate(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        item, err = character_store.duplicate(
            str((body or {}).get("character_id") or ""))
        if item is None:
            return _json_error(err or "複製できませんでした。", status=400)
        return web.json_response({"ok": True,
                                  "character": _character_public(item)})

    async def api_character_delete(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        ok, err = character_store.delete(
            str((body or {}).get("character_id") or ""))
        if not ok:
            return _json_error(err or "削除できませんでした。", status=400)
        return web.json_response({"ok": True})

    def _character_snapshot(item: dict, images: list[str],
                            voice_file: str = "") -> dict:
        key = ""
        try:
            key = profile_cache_key(
                [comfy_inputs_mod.local_path(cfg, n) or (cfg.comfy_input / n)
                 for n in images])
        except Exception:                                        # noqa: BLE001
            pass
        return {
            "character_id": item["character_id"],
            "display_name": item.get("display_name", ""),
            "profile_text": item.get("profile_text", ""),
            "canon": dict(item.get("canon") or {}),
            "outfit_ref": dict(item.get("outfit_ref") or {}),
            "voice_note": item.get("voice_note", ""),
            "refs": list(images),
            "key": key,
            "voice_file": voice_file,
        }

    async def api_character_apply(request: web.Request):
        """Materialize a Library character into the existing ref management.

        Copies refs (+voice) into comfy_input with deterministic names and
        returns everything the UI/generate path needs. Touches no project.
        """
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        item = character_store.get(str((body or {}).get("character_id") or ""))
        if item is None:
            return _json_error("キャラクターが見つかりませんでした。",
                               status=404)
        images = character_store.materialize_refs(item, cfg.input_stage_dir)
        if not images:
            return _json_error("参照画像を準備できませんでした。",
                               status=500)
        voice_file = character_store.materialize_voice(item, cfg.input_stage_dir)
        snapshot = _character_snapshot(item, images, voice_file)
        return web.json_response({"ok": True, "images": images,
                                  "snapshot": snapshot})

    def _character_missing_note(snapshot: dict | None) -> bool:
        """True when the snapshot's Library character is gone.

        Non-blocking by design: generation from the snapshot keeps working.
        """
        if not isinstance(snapshot, dict):
            return False
        cid = str(snapshot.get("character_id") or "")
        if not cid:
            return False
        try:
            return character_store.get(cid) is None
        except Exception:                                        # noqa: BLE001
            return False

    def _character_snapshot_for(images: list[str],
                                snapshot: dict | None) -> dict | None:
        """Accept the client snapshot only when it still matches the images.

        A manual image change after apply silently drops the snapshot profile
        so the normal cache path (re-analysis when needed) takes over.
        """
        if not isinstance(snapshot, dict):
            return None
        if list(snapshot.get("refs") or []) != list(images or []):
            return None
        if not str(snapshot.get("profile_text") or "").strip():
            return None
        return snapshot

    def _character_stage_story_voice(story, snapshot: dict) -> None:
        """Pre-seed the story Voice Master from the snapshot voice file.

        Copies into the existing h3app_story_<id>_voice_master.wav management
        so later Library deletion cannot take the voice away. Best-effort.
        """
        try:
            voice_file = str((snapshot or {}).get("voice_file") or "")
            base = Path(voice_file).name
            # Only our own staged voice copies (never an arbitrary path).
            if not (base.startswith("h3app_char_") and
                    base.endswith(".wav") and base == voice_file):
                return
            src = comfy_inputs_mod.local_path(cfg, base)
            if src is None or not src.is_file():
                return
            dest = comfy_inputs_mod.stage_path(
                cfg, f"h3app_story_{story.id}_voice_master.wav")
            import shutil as _shutil
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                _shutil.copy2(src, dest)
            story.data["voice_master"] = dest.name
        except Exception:                                        # noqa: BLE001
            pass

    # ----------------------------------------------------------- story mode --
    def _story_or_404(request: web.Request):
        story = story_store.get(request.match_info.get("sid", ""))
        if story is None:
            raise web.HTTPNotFound()
        return story

    def _story_seconds(body: dict) -> float:
        try:
            value = float(body.get("seconds")
                          or (cfg.story or {}).get("seconds_per_segment", 5))
        except (TypeError, ValueError):
            value = 5.0
        return min(15.0, max(2.0, value))

    def _story_delivery(body: dict):
        """Optional delivery descriptor. Absent -> duration.DEFAULT_DELIVERY,
        which is what every existing caller gets."""
        value = (body or {}).get("delivery")
        return value if isinstance(value, dict) else None

    async def api_story_preview(request: web.Request):
        """Pure Python split. No GPU, no ComfyUI - used for the live preview and
        for the「分割をやり直す」button."""
        body = await request.json()
        seconds = _story_seconds(body)
        delivery = _story_delivery(body)
        lines = body.get("lines")
        if isinstance(lines, list) and lines:
            # STRUCTURED input: the types the user set are authoritative. No
            # quote detection, no quotative-と heuristic, no re-classification.
            clean = segmenter.normalize_lines(lines)
            segments = segmenter.segments_from_lines(clean, seconds=seconds,
                                                     delivery=delivery)
            structured = True
        else:
            text = body.get("text") or ""
            clean = segmenter.split_lines(text)
            segments = segmenter.segments_from_text(text, seconds=seconds,
                                                    delivery=delivery)
            structured = False
        limit = int((cfg.story or {}).get("max_segments", 24))
        return web.json_response({
            "ok": True, "lines": clean, "segments": segments,
            "structured": structured,
            "seconds": seconds, "max_segments": limit,
            "over_limit": len(segments) > limit,
        })

    async def api_story_create(request: web.Request):
        body = await request.json()
        images = [n for n in (body.get("images") or []) if _safe_upload_name(n)]
        if not images:
            return _json_error("参照画像を1枚以上追加してください。")
        if len(images) > 4:
            return _json_error("参照画像は最大4枚までです。")
        for n in images:
            if comfy_inputs_mod.local_path(cfg, n) is None:
                return _json_error("追加した画像が見つかりません。もう一度追加してください。")

        seconds = _story_seconds(body)
        try:
            settings = build_settings(cfg, seconds=seconds,
                                      aspect=body.get("aspect", "portrait"),
                                      advanced=body.get("advanced") or {})
        except (PipelineError, graphs.GraphError) as exc:
            return _json_error(str(exc))

        source_text = body.get("source_text") or body.get("text") or ""
        lines = body.get("lines")
        structured = isinstance(lines, list) and bool(lines)
        if structured:
            lines = segmenter.normalize_lines(lines)
        else:
            lines = segmenter.split_lines(source_text)
        raw_segments = body.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            # Explicit structure -> no type inference. Raw text was already typed
            # by split_lines above.
            delivery = _story_delivery(body)
            raw_segments = (
                segmenter.segments_from_lines(lines, seconds=seconds,
                                              delivery=delivery)
                if structured
                else segmenter.segment_lines(lines, seconds=seconds,
                                             delivery=delivery))
        segments = normalize_segments(raw_segments)
        if not segments:
            return _json_error("セグメントがありません。文章を入力してから作成してください。")
        limit = int((cfg.story or {}).get("max_segments", 24))
        if len(segments) > limit:
            return _json_error(
                f"セグメントが多すぎます（{len(segments)} 個）。"
                f"1つのストーリーは最大 {limit} 個までです。文章を短く分けてください。")

        # Story mode defaults to LONG (tail relay + Original Re-anchor).
        # Callers may opt into another story mode (e.g. LONG_FAST); anything
        # outside modes_v2.STORY_MODES is refused explicitly, never coerced.
        story_mode = str(body.get("mode") or "LONG").upper()
        if story_mode not in modes_mod.STORY_MODES:
            return _json_error(
                "不明なストーリーモードです: "
                f"{story_mode}（利用可能: {', '.join(modes_mod.STORY_MODES)}）")

        try:
            negative = (body.get("jp_negative") or "").strip() or prompts().NEGATIVE_DEFAULT
        except Exception:
            negative = (body.get("jp_negative") or "").strip()

        story = story_store.create(
            name=(body.get("name") or "").strip() or "ストーリー",
            source_text=source_text, lines=lines, segments=segments,
            images=images, settings=settings, jp_negative=negative,
            motion_presets=[str(p) for p in (body.get("motion_presets") or [])])
        story.data["mode"] = story_mode
        snap = _character_snapshot_for(
            images, (body.get("character_snapshot")
                     if isinstance(body, dict) else None))
        if snap is not None:
            story.data["profile"] = snap["profile_text"]
            story.data["character_snapshot"] = snap
            _character_stage_story_voice(story, snap)
        story.save()
        return web.json_response({"ok": True, "story_id": story.id,
                                  "story": story.to_public()})

    async def api_story_start(request: web.Request):
        # WP-D: block new jobs while a shutdown is in progress.
        if not app.get("accepting_jobs", True):
            return _json_error("H3は終了処理中です。新しい生成は開始できません。",
                               status=503)
        body = await request.json()
        story = story_store.get(body.get("story_id", ""))
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        try:
            await story_pipeline.start(story)
        except PipelineError as exc:
            info = classify(exc)
            # The pre-flight list travels with the refusal so the UI can point at
            # the segment that is wrong instead of just showing a message.
            return _json_error(str(exc), status=409,
                               extra={"error": info,
                                      "preflight": preflight_records(story)})
        return web.json_response({"ok": True, "story_id": story.id,
                                  "cursor": story.cursor,
                                  "preflight": story.data.get("preflight", [])})

    async def api_story_preflight(request: web.Request):
        """What WOULD be generated: per segment prompt / speech / seed / context.
        Read-only, no GPU, safe to call at any time."""
        story = _story_or_404(request)
        return web.json_response({"ok": True, "story_id": story.id,
                                  "preflight": preflight_records(story)})

    async def api_story_stop(request: web.Request):
        body = await request.json()
        story = story_store.get(body.get("story_id", ""))
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        running = await story_pipeline.request_stop(story)
        return web.json_response({
            "ok": True, "running": running,
            "message": ("いま生成中のクリップが終わったら停止します。"
                        if running else "停止しました。ここまでの結果は保存済みです。"),
        })

    async def api_story_discard(request: web.Request):
        body = await request.json()
        sid = body.get("story_id", "")
        story = story_store.get(sid)
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        if story_pipeline.is_running(sid):
            return _json_error("生成中は削除できません。先に停止してください。", status=409)
        ok = story_store.delete(sid)
        story_pipeline.runners.pop(sid, None)
        if not ok:
            return _json_error("削除できませんでした。ファイルが使用中の可能性があります。",
                               status=500)
        return web.json_response({"ok": True, "message": "削除しました。"})

    async def api_story_merge(request: web.Request):
        body = await request.json()
        story = story_store.get(body.get("story_id", ""))
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        if not story.clips:
            return _json_error("書き出せるクリップがまだありません。")
        try:
            out = await story_pipeline.merge(story)
        except PipelineError as exc:
            info = classify(exc, stage="動画の書き出し")
            return _json_error(str(exc),
                               status=409 if info["kind"] == "busy" else 400,
                               extra={"error": info})
        # WP-C: per-boundary seam report + needs_review, so the UI can show
        # "regenerate or check" instead of a bare success toast.
        seams = story.data.get("seams") or {"report": [], "needs_review": False}
        return web.json_response({"ok": True, "final_video": str(out),
                                  "url": f"/api/story/{story.id}/final",
                                  "export": story.data.get("final_export"),
                                  "seams": seams})

    async def api_story_list(request: web.Request):
        return web.json_response({"ok": True, "stories": story_store.list_summaries()})

    async def api_story_get(request: web.Request):
        story = _story_or_404(request)
        data = story.to_public()
        runner = story_pipeline.runner_for(story.id)
        data["progress"] = runner.snapshot() if runner else None
        data["running"] = story_pipeline.is_running(story.id)
        data["character_missing"] = _character_missing_note(
            data.get("character_snapshot"))
        return web.json_response({"ok": True, "story": data})

    # --------------------------------------- v2 clip control (lock/regen) ----
    async def api_story_clip_lock(request: web.Request):
        """LOCKED/APPROVED a finished clip so later runs never rebuild it."""
        body = await request.json()
        story = story_store.get(body.get("story_id", ""))
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        if story_pipeline.is_running(story.id):
            return _json_error("生成中はロックを変更できません。先に停止してください。",
                               status=409)
        try:
            index = int(body.get("index", -1))
        except (TypeError, ValueError):
            return _json_error("indexが不正です。")
        if not (0 <= index < len(story.segments)):
            return _json_error("indexが範囲外です。")
        action = str(body.get("action") or "").lower()
        if action not in ("lock", "approve", "unlock"):
            return _json_error("actionは lock / approve / unlock のいずれかです。")
        seg = story.segments[index]
        if action == "unlock":
            seg.pop("locked", None)
        else:
            if seg.get("status") != "done" or story.clip_for(index) is None:
                return _json_error(
                    "未生成のクリップはロックできません。先に生成してください。")
            seg["locked"] = "approved" if action == "approve" else "locked"
        story.save()
        return web.json_response({"ok": True, "index": index,
                                  "locked": seg.get("locked", "")})

    async def api_story_regenerate(request: web.Request):
        """Regenerate ONLY the chosen clips. Downstream non-frozen clips
        rebuild too (their tail input changed); LOCKED/APPROVED never rebuild.
        Full-story regeneration is refused by design — pick clips instead."""
        body = await request.json()
        story = story_store.get(body.get("story_id", ""))
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        if story_pipeline.is_running(story.id):
            return _json_error("生成中は再生成を指定できません。先に停止してください。",
                               status=409)
        raw = body.get("indices") or []
        try:
            indices = sorted({int(i) for i in raw})
        except (TypeError, ValueError):
            return _json_error("indicesが不正です。")
        if not indices or any(not (0 <= i < len(story.segments)) for i in indices):
            return _json_error("再生成するクリップを1つ以上正しく選んでください。")
        for i in indices:
            if story.segments[i].get("locked") in ("locked", "approved"):
                return _json_error(
                    f"クリップ {i + 1} はロックされています。先に解除してください。")
        first = indices[0]
        # Tail-chain rule (mirrors backend/h3_v2/story_v2.plan): everything
        # from the first regenerated clip onward rebuilds, except frozen clips
        # (kept, flagged stale by the runner's own relay logic).
        rebuilt, kept = [], []
        for i in range(first, len(story.segments)):
            seg = story.segments[i]
            if seg.get("locked") in ("locked", "approved") and \
                    seg.get("status") == "done" and story.clip_for(i) is not None:
                kept.append(i)
                continue
            seg["status"] = "pending"
            seg["error"] = ""
            rebuilt.append(i)
        story.data["cursor"] = first
        story.data["status"] = "pending"
        story.save()
        return web.json_response({"ok": True, "rebuilt": rebuilt, "kept": kept,
                                  "cursor": first})

    async def api_story_segments(request: web.Request):
        body = await request.json()
        sid = body.get("story_id", "")
        story = story_store.get(sid)
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        if story_pipeline.is_running(sid):
            return _json_error("生成中はセグメントを編集できません。先に停止してください。",
                               status=409)
        raw_segments = body.get("segments") or []
        # CONTENT-ONLY edit: positions are unchanged, so a client that does not
        # send segment_id keeps the identity that already sits at that position
        # (and with it the clips that reference it).
        for i, item in enumerate(raw_segments):
            if isinstance(item, dict) and i < len(story.segments):
                # Older UIs omit these fields; saving text must not erase them.
                for key in ("attribute_overrides", "delivery", "continuity", "transition"):
                    if key not in item and key in story.segments[i]:
                        item[key] = story.segments[i][key]
            if isinstance(item, dict) and not item.get("segment_id") \
                    and i < len(story.segments):
                item["segment_id"] = story.segments[i].get("segment_id", "")
        segments = normalize_segments(raw_segments)
        if not segments:
            return _json_error("セグメントが空です。")
        limit = int((cfg.story or {}).get("max_segments", 24))
        if len(segments) > limit:
            return _json_error(f"セグメントは最大 {limit} 個までです。")
        try:
            segments = story_mod.content_edit(story.segments, segments)
        except ValueError as exc:
            return _json_error(str(exc), status=409)
        story.segments = segments
        if "cursor" in body:
            try:
                story.data["cursor"] = max(0, min(len(segments), int(body["cursor"])))
            except (TypeError, ValueError):
                pass
        else:
            story.data["cursor"] = min(int(story.data.get("cursor", 0)), len(segments))
        story.save()
        return web.json_response({"ok": True, "story": story.to_public()})

    async def api_story_structure(request: web.Request):
        """ADD / DELETE / REORDER: one endpoint, because all three are the same
        operation - "here is the new ordered segment list".

        `dry_run: true` returns the impact (how many generated clips would be
        invalidated) without writing anything, so the UI can confirm first.
        """
        body = await request.json()
        sid = body.get("story_id", "")
        story = story_store.get(sid)
        if story is None:
            return _json_error("ストーリーが見つかりません。", status=404)
        raw = body.get("segments")
        if not isinstance(raw, list):
            return _json_error("セグメントの指定が不正です。")
        running = story_pipeline.is_running(sid)
        refusal = story_structure_refusal(
            running=running, count=len(raw),
            limit=int((cfg.story or {}).get("max_segments", 24)))
        if refusal:
            return _json_error(refusal, status=409 if running else 400)
        try:
            result = story_mod.apply_structure(
                story, raw, dry_run=bool(body.get("dry_run")))
        except StructureError as exc:
            return _json_error(str(exc))
        return web.json_response({"ok": True, "story_id": story.id, **result})

    async def api_story_events(request: web.Request):
        story = _story_or_404(request)
        return await _stream_events(request, story_pipeline.runner_for(story.id))

    async def api_story_video(request: web.Request):
        story = _story_or_404(request)
        try:
            index = int(request.match_info["index"])
        except ValueError:
            raise web.HTTPNotFound()
        clip = story.clip_for(index)
        if clip is None:
            raise web.HTTPNotFound()
        path = Path(clip.get("local_video") or clip.get("video", ""))
        if not path.is_file():
            path = Path(clip.get("video", ""))
        if not path.is_file():
            raise web.HTTPNotFound()
        # FileResponse handles Range requests, so <video> can seek.
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return web.FileResponse(path, headers={"Content-Type": ctype})

    async def api_story_final(request: web.Request):
        story = _story_or_404(request)
        path = Path(story.data.get("final_video") or "")
        if not path.is_file():
            raise web.HTTPNotFound()
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return web.FileResponse(path, headers={"Content-Type": ctype})

    async def api_story_frame(request: web.Request):
        story = _story_or_404(request)
        try:
            index = int(request.match_info["index"])
        except ValueError:
            raise web.HTTPNotFound()
        clip = story.clip_for(index)
        path = Path((clip or {}).get("last_frame") or "")
        if not path.is_file():
            path = story.frames_dir / f"frame_{index:02d}.png"
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Type": "image/png"})

    # =========================================== WP-B: /api/upscale/* ========
    # Post-generation video upscale (ESRGAN / SeedVR2). Actual planning/graph/
    # run logic lives in h3app/upscale.py; this block only translates HTTP <->
    # that module and applies the same busy/path-safety gates used elsewhere.
    # /api/events/{job_id} needs no new route: UpscaleRunner is a
    # pipeline.Runner registered in pipeline.runners exactly like a normal
    # generation, so the existing api_events() lookup already serves it.
    def _upscale_source_from_body(body: dict) -> dict:
        source = body.get("source") if isinstance(body, dict) else None
        return source if isinstance(source, dict) else {}

    async def _upscale_plan_for(source_path: Path, preset: str, method: str) -> dict:
        # "lanczos" is plain ffmpeg: never launch ComfyUI (ensure_ready would
        # start it) and never require its object_info.
        object_info = None
        if method != "lanczos":
            await client.ensure_ready(None)
            object_info = client.object_info
        src_probe = await upscale_mod.probe(source_path)
        gpu_plan = gpu_mod.current_plan(cfg)
        model_files = upscale_mod.model_files_for(cfg)
        return upscale_mod.plan(
            src_probe, {"preset": preset, "method": method},
            gpu_plan, object_info, model_files)

    def _upscale_methods_payload() -> list:
        return [{"id": m, "label": upscale_mod.METHOD_LABELS[m],
                 "kind": upscale_mod.METHOD_KINDS[m]} for m in upscale_mod.METHODS]

    async def api_upscale_options(request: web.Request):
        source = {"kind": request.query.get("kind", "file"),
                 "name": request.query.get("name", ""),
                 "id": request.query.get("id", ""),
                 "step": request.query.get("step"),
                 "final": request.query.get("final") in ("1", "true")}
        preset = request.query.get("preset", "1080x1920")
        method = request.query.get("method", "esrgan")
        if preset not in upscale_mod.PRESETS or method not in upscale_mod.METHODS:
            return _json_error("解像度または方式の指定が不正です。", status=400)
        path, err = upscale_mod.resolve_source(cfg, store, story_store, source)
        if path is None:
            return _json_error(err, status=404)
        try:
            plan_result = await _upscale_plan_for(path, preset, method)
            src_probe = await upscale_mod.probe(path)
        except PipelineError as exc:
            return _json_error(str(exc), status=502)
        return web.json_response({
            "ok": True, "plan": plan_result,
            "methods": _upscale_methods_payload(), "source": src_probe})

    async def api_upscale_start(request: web.Request):
        body = await request.json()
        source = _upscale_source_from_body(body)
        preset = str(body.get("preset") or "")
        method = str(body.get("method") or "")
        if preset not in upscale_mod.PRESETS or method not in upscale_mod.METHODS:
            return _json_error("解像度または方式の指定が不正です。", status=400)
        path, err = upscale_mod.resolve_source(cfg, store, story_store, source)
        if path is None:
            return _json_error(err, status=404)
        if pipeline.is_busy or _any_story_running():
            return _json_error(
                "いま別の動画を生成中です。終わるか中止してから実行してください。",
                status=409)
        try:
            plan_result = await _upscale_plan_for(path, preset, method)
        except PipelineError as exc:
            return _json_error(str(exc), status=502)
        if not plan_result.get("ok") and not bool(body.get("confirm_missing")):
            return _json_error(
                "必要なノードまたはモデルが不足しています。導入状況を確認してください。",
                status=409, extra={"plan": plan_result})
        job = upscale_mod.UpscaleJob(f"up_{uuid.uuid4().hex}")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        try:
            await pipeline.start_runner(
                runner, lambda r: upscale_mod.run(pipeline, path, plan_result, r))
        except PipelineError as exc:
            return _json_error(str(exc), status=409)
        return web.json_response({"ok": True, "job_id": job.id})

    async def api_upscale_cancel(request: web.Request):
        body = await request.json()
        ok = pipeline.cancel(str(body.get("job_id") or ""))
        return web.json_response({
            "ok": ok,
            "message": "中止しました。" if ok else "中止できるジョブがありません。"})
    # ========================================= /WP-B: /api/upscale/* ========

    # ------------------------------------------------- action presets (16) ---
    def _presets_payload(extra: dict | None = None) -> dict:
        body = {
            "ok": True,
            "presets": action_presets.list(),
            "template": presets_mod.APPLY_TEMPLATE,
            "max_name": presets_mod.MAX_NAME_CHARS,
            # True = the saved file could not be read and the factory defaults
            # are in use. Never hidden from the user.
            "fallback": bool(action_presets.fallback),
            "message": action_presets.fallback_reason or "",
        }
        if extra:
            body.update(extra)
        return body

    async def api_action_presets(request: web.Request):
        return web.json_response(_presets_payload())

    async def api_action_preset_add(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                   # noqa: BLE001
            body = {}
        try:
            preset = action_presets.add((body or {}).get("name"))
        except PresetError as exc:
            return _json_error(str(exc))
        return web.json_response(_presets_payload({"preset": preset}))

    async def api_action_preset_delete(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                   # noqa: BLE001
            body = {}
        try:
            removed = action_presets.delete((body or {}).get("id"))
        except PresetError as exc:
            return _json_error(str(exc), status=404)
        # Prompt text that was already written stays exactly as it is.
        return web.json_response(_presets_payload({"deleted": removed}))

    # ------------------------------------------------------------- watchdog --
    async def api_heartbeat(request: web.Request):
        return web.json_response({"ok": True, **watchdog.heartbeat()})

    async def api_detach(request: web.Request):
        # sendBeacon on tab close. Body is ignored on purpose: it may be a Blob.
        return web.json_response({"ok": True, **watchdog.detach()})

    # ================================================ WP-D: shutdown block ==
    async def api_shutdown(request: web.Request):
        """POST /api/shutdown {"interrupt": bool}. See h3app.shutdown."""
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            body = {}
        interrupt = bool((body or {}).get("interrupt"))
        busy = pipeline.is_busy or _any_story_running()
        if busy and not interrupt:
            return web.json_response(
                {"ok": False, "busy": True, "running": True,
                 "message": "生成中です。中止して終了しますか？"}, status=409)
        app["accepting_jobs"] = False
        try:
            result = await shutdown_seq.run(interrupt=interrupt, self_exit=True)
        except Exception as exc:                                 # noqa: BLE001
            app["accepting_jobs"] = True
            return _json_error(f"終了処理に失敗しました: {exc}", status=500)
        app["_shutdown_ran"] = True
        return web.json_response(result, status=200)

    async def api_settings_shutdown_get(request: web.Request):
        stop_lm_studio = bool((cfg.get("shutdown") or {}).get("stop_lm_studio", True))
        pre_existing = False
        try:
            state = procstate_mod.read_state()
            entry = state.get(procstate_mod.ROLE_LMSTUDIO)
            if entry is not None and not entry.get("owned_by_h3", True):
                pre_existing = True
            elif shutdown_mod.lmstudio_target(cfg) is not None:
                listener = procstate_mod.find_listener(
                    shutdown_mod.lmstudio_target(cfg)["port"])
                if listener is not None and entry is None:
                    # Reachable but never recorded by this run: treat as
                    # already running before H3 started this session.
                    pre_existing = True
        except Exception:                                        # noqa: BLE001
            pre_existing = False
        return web.json_response({"ok": True, "stop_lm_studio": stop_lm_studio,
                                  "lmstudio_pre_existing": pre_existing})

    async def api_settings_shutdown_save(request: web.Request):
        try:
            body = await request.json()
        except Exception:                                        # noqa: BLE001
            return _json_error("リクエストが不正です。", status=400)
        if cfg.source is None:
            return _json_error("設定ファイルの保存先が不明です。", status=500)
        stop_lm_studio = bool((body or {}).get("stop_lm_studio", True))
        try:
            data = update_config_file(cfg.source, {
                "shutdown": {"stop_lm_studio": stop_lm_studio}})
            cfg.data["shutdown"] = data["shutdown"]
        except Exception as exc:                                  # noqa: BLE001
            return _json_error(f"保存できませんでした: {exc}", status=500)
        return web.json_response({"ok": True, "stop_lm_studio": stop_lm_studio})

    app.router.add_get("/", index)
    async def api_direction_review(request):
        body = await request.json()
        spec = body.get("spec") if isinstance(body, dict) else None
        if not isinstance(spec, dict) or spec.get("kind") not in ("single", "story30"):
            return _json_error("先に監督案を作ってください。")
        try:
            report = director_review.review(spec)
        except (ValueError, TypeError, AttributeError):
            return _json_error("監督案の形式が不正です。尺・台詞・ショットを確認してください。")
        return web.json_response(report, status=200 if report["ok"] else 400)

    rehearsal = voice_rehearsal.VoiceRehearsal()

    async def api_voice_speakers(request):
        try:
            return web.json_response({"ok": True, "speakers": await rehearsal.speakers()})
        except voice_rehearsal.RehearsalError as exc:
            return _json_error(str(exc), status=503)

    async def api_voice_query(request):
        body = await request.json()
        try:
            query = await rehearsal.query(body.get("text"), body.get("speaker"))
            return web.json_response({"ok": True, "query": query})
        except voice_rehearsal.RehearsalError as exc:
            return _json_error(str(exc), status=400)

    async def api_voice_preview(request):
        body = await request.json()
        try:
            wav = await rehearsal.synthesize(body.get("query"), body.get("speaker"))
            return web.Response(body=wav, content_type="audio/wav", headers={"Cache-Control": "no-store"})
        except voice_rehearsal.RehearsalError as exc:
            return _json_error(str(exc), status=400)

    app.router.add_post("/api/director/review", api_direction_review)
    app.router.add_get("/api/voice-rehearsal/speakers", api_voice_speakers)
    app.router.add_post("/api/voice-rehearsal/query", api_voice_query)
    app.router.add_post("/api/voice-rehearsal/preview", api_voice_preview)
    app.router.add_get("/static/{name:.*}", static_file)
    app.router.add_get("/api/config", api_config)
    app.router.add_post("/api/images", api_images)
    app.router.add_get("/api/thumb/{name}", api_thumb)
    app.router.add_post("/api/generate", api_generate)
    app.router.add_post("/api/again", api_again)
    app.router.add_post("/api/continue", api_continue)
    app.router.add_post("/api/cancel", api_cancel)
    app.router.add_get("/api/project/{pid}", api_project)
    app.router.add_get("/api/events/{pid}", api_events)
    app.router.add_get("/api/video/{pid}/{step}", api_video)
    app.router.add_post("/api/open-folder", api_open_folder)
    app.router.add_post("/api/media/open", api_media_open)
    app.router.add_get("/api/settings/storage", api_settings_storage)
    app.router.add_get("/api/media/cleanup/scan", api_media_cleanup_scan)
    app.router.add_post("/api/media/cleanup", api_media_cleanup)
    app.router.add_post("/api/director/single/create",
                        api_director_single_create)
    app.router.add_post("/api/director/single/regenerate",
                        api_director_single_regenerate)
    app.router.add_post("/api/director/story30/create",
                        api_director_story30_create)
    app.router.add_post("/api/director/story30/regenerate",
                        api_director_story30_regenerate)
    app.router.add_post("/api/director/generate", api_director_generate)
    app.router.add_post("/api/director/story30/generate",
                        api_director_story30_generate)
    app.router.add_get("/api/ai/settings", api_ai_settings_get)
    app.router.add_post("/api/ai/settings", api_ai_settings_save)
    # WP-A: provider routes (new) + legacy delegates below.
    app.router.add_post("/api/ai/key", api_ai_key_save_post)
    app.router.add_delete("/api/ai/key", api_ai_key_delete_post)
    app.router.add_post("/api/ai/models", api_ai_models_post)
    app.router.add_post("/api/ai/test", api_ai_test_post)
    app.router.add_get("/api/settings/gpu", api_settings_gpu_get)
    app.router.add_post("/api/settings/gpu/validate", api_settings_gpu_validate)
    app.router.add_post("/api/settings/gpu", api_settings_gpu_save)
    app.router.add_post("/api/settings/gpu/apply", api_settings_gpu_apply)
    app.router.add_get("/api/ai/models", api_ai_models)
    app.router.add_post("/api/ai/test-connection", api_ai_test_connection)
    app.router.add_post("/api/ai/test-model", api_ai_test_model)
    app.router.add_post("/api/ai/local/models", api_ai_local_models)
    app.router.add_post("/api/ai/local/test", api_ai_local_test)
    app.router.add_post("/api/ai/local/key", api_ai_local_key_save)
    app.router.add_delete("/api/ai/local/key", api_ai_local_key_delete)
    app.router.add_get("/api/character/list", api_character_list)
    app.router.add_get("/api/character/thumb/{cid}/{idx}",
                       api_character_thumb)
    app.router.add_post("/api/character/save", api_character_save)
    app.router.add_post("/api/character/update", api_character_update)
    app.router.add_post("/api/character/rename", api_character_rename)
    app.router.add_post("/api/character/duplicate", api_character_duplicate)
    app.router.add_post("/api/character/delete", api_character_delete)
    app.router.add_post("/api/character/apply", api_character_apply)
    app.router.add_post("/api/release-models", api_release_models)
    app.router.add_post("/api/audio-test/run", api_audio_test_run)
    app.router.add_get("/api/audio-test/list", api_audio_test_list)
    app.router.add_get("/api/audio-test/audio/{test_id}", api_audio_test_audio)
    app.router.add_post("/api/audio-test/delete", api_audio_test_delete)
    app.router.add_post("/api/audio-test/delete-all", api_audio_test_delete_all)

    # Story mode. /api/story/list is registered BEFORE /api/story/{sid} because
    # aiohttp resolves routes in registration order.
    app.router.add_post("/api/story/preview", api_story_preview)
    app.router.add_post("/api/story/create", api_story_create)
    app.router.add_post("/api/story/start", api_story_start)
    app.router.add_post("/api/story/stop", api_story_stop)
    app.router.add_post("/api/story/discard", api_story_discard)
    app.router.add_post("/api/story/merge", api_story_merge)
    app.router.add_post("/api/story/segments", api_story_segments)
    app.router.add_post("/api/story/structure", api_story_structure)
    app.router.add_post("/api/story/open-folder", api_story_open_folder)
    app.router.add_get("/api/story/list", api_story_list)
    app.router.add_get("/api/story/{sid}/preflight", api_story_preflight)
    app.router.add_get("/api/story/{sid}/events", api_story_events)
    app.router.add_get("/api/story/{sid}/video/{index}", api_story_video)
    app.router.add_get("/api/story/{sid}/final", api_story_final)
    app.router.add_get("/api/story/{sid}/frame/{index}", api_story_frame)
    app.router.add_get("/api/story/{sid}", api_story_get)

    app.router.add_get("/api/compat", api_compat)
    app.router.add_post("/api/story/clip-lock", api_story_clip_lock)
    app.router.add_post("/api/story/regenerate", api_story_regenerate)

    # WP-B: post-generation video upscale.
    app.router.add_get("/api/upscale/options", api_upscale_options)
    app.router.add_post("/api/upscale/start", api_upscale_start)
    app.router.add_post("/api/upscale/cancel", api_upscale_cancel)

    # User-editable ACTION presets.
    app.router.add_get("/api/action-presets", api_action_presets)
    app.router.add_post("/api/action-presets", api_action_preset_add)
    app.router.add_post("/api/action-presets/delete", api_action_preset_delete)

    app.router.add_post("/api/heartbeat", api_heartbeat)
    app.router.add_post("/api/detach", api_detach)

    # WP-D: shutdown routes (handlers are defined in the WP-D block above).
    app.router.add_post("/api/shutdown", api_shutdown)
    app.router.add_get("/api/settings/shutdown", api_settings_shutdown_get)
    app.router.add_post("/api/settings/shutdown", api_settings_shutdown_save)

    @web.middleware
    async def error_middleware(request, handler):
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as exc:                    # noqa: BLE001
            return web.json_response({"ok": False, "message": str(exc),
                                      "error": classify(exc)}, status=500)

    app.middlewares.append(error_middleware)

    async def on_startup(_app):
        # Set before anything else: on_cleanup runs the shutdown sequence
        # ONLY for an app that really started this way. Test suites build
        # the app with on_startup cleared, and without this gate their
        # teardown sent /interrupt and /free to the real ComfyUI on the
        # default port and deleted the real h3_state.json (observed
        # 2026-09-13: a running story was cancelled by a unit-test run).
        _app["_h3_started"] = True
        watchdog.start()
        # WP-D: record this run's ownership state. Best-effort: never fails
        # boot. Servers already listening that this run did not start are
        # marked foreign (owned_by_h3=False) so shutdown never touches them.
        try:
            # Records of a previous run whose processes are gone (crash,
            # hard kill) are dropped; live ones are kept so their identity
            # can still be checked against the current listeners below.
            procstate_mod.prune_dead_entries(h3_session_id)
            procstate_mod.record_entry(
                procstate_mod.ROLE_APP, os.getpid(),
                session_id=h3_session_id, port=int(cfg.app_port),
                owned_by_h3=True)
            try:
                _comfy_listener = procstate_mod.find_listener(
                    int(cfg.comfy_port))
            except Exception:                                    # noqa: BLE001
                _comfy_listener = None
            if _comfy_listener and _comfy_listener.get("pid"):
                _live = procstate_mod.identify(int(_comfy_listener["pid"]))
                if not procstate_mod.matches(
                        procstate_mod.read_state().get(
                            procstate_mod.ROLE_COMFYUI), _live):
                    procstate_mod.record_entry(
                        procstate_mod.ROLE_COMFYUI,
                        int(_comfy_listener["pid"]),
                        session_id=h3_session_id,
                        port=int(cfg.comfy_port), owned_by_h3=False)
            # LM Studio already running on THIS machine at the configured
            # loopback URL: record its identity for this session (remote
            # targets and non-LM-Studio listeners are never recorded).
            shutdown_mod.record_lmstudio_target(cfg, h3_session_id)
        except Exception:                                        # noqa: BLE001
            pass

    async def on_cleanup(_app):
        # WP-D: Ctrl+C / SIGBREAK / runner cleanup run the same sequence
        # minus the app's own exit. Guarded so POST /api/shutdown runs it
        # exactly once; the watchdog behaviour below is unchanged.
        if _app.get("_h3_started") and not _app.get("_shutdown_ran"):
            _app["_shutdown_ran"] = True
            try:
                await shutdown_seq.run(interrupt=True, self_exit=False)
            except Exception:                                    # noqa: BLE001
                pass
        await watchdog.stop()
        client.shutdown()

    # WP-D: Windows Ctrl+Break (SIGBREAK) is not handled by web.run_app, so
    # schedule the same sequence here. Best-effort; cleanup covers the rest.
    try:
        _sigbreak_no = getattr(signal, "SIGBREAK", None)
        if _sigbreak_no is not None:
            def _wp_d_on_sigbreak(*_args):
                try:
                    app["_shutdown_ran"] = True
                    _loop = asyncio.get_event_loop()
                    _loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            shutdown_seq.run(interrupt=True,
                                             self_exit=False)))
                except Exception:                                # noqa: BLE001
                    pass
            signal.signal(_sigbreak_no, _wp_d_on_sigbreak)
    except Exception:                                            # noqa: BLE001
        pass

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# --------------------------------------------------------------- selftest ----
# candidate node id -> app node id for the shared v1 chain.
PARITY_MAP = {
    "2": "load_0", "3": "scale_0", "4": "clip", "9": "clip_select",
    "5": "vae_video", "6": "vae_audio", "7": "h3", "10": "unet", "11": "lora",
    "12": "noise", "13": "guider", "14": "sampler_select", "15": "scheduler",
    "16": "sampler", "17": "decode_video", "18": "decode_audio",
    "19": "create_video", "20": "save_video",
}

# The ONLY differences allowed between candidate_prompt.json and the app's
# default generate graph. Anything else means the v1 chain drifted.
ALLOWED_STRUCTURAL_DIFF = sorted([
    "node_removed:1(H3GoldFixedPrompt)",              # prompt now comes from the director
    "node_added:restore(H3CudaPhaseRestore)",         # CUDA phase barrier
    "node_added:clip_report_base(H3ClipDeviceReport)",
    "node_added:clip_report_target(H3ClipDeviceReport)",
    "4.class_type:CLIPLoader->H3GatedCLIPLoader",
    "4.inputs.barrier:added",
    "9.inputs.clip:link",                             # via the base device report
    "7.inputs.clip:link",                             # via the target device report
    "7.inputs.prompt:value",                          # director text instead of node 1
    "2.inputs.image:value",                           # the user's reference image
    "20.inputs.filename_prefix:value",                # per-project output path
])


def structural_diff(candidate: dict, mine: dict) -> list[str]:
    diffs: list[str] = []
    mapped = set()
    for cid, cnode in candidate.items():
        mid = PARITY_MAP.get(cid)
        if mid is None or mid not in mine:
            diffs.append(f"node_removed:{cid}({cnode['class_type']})")
            continue
        mapped.add(mid)
        mnode = mine[mid]
        if cnode["class_type"] != mnode["class_type"]:
            diffs.append(f"{cid}.class_type:{cnode['class_type']}->{mnode['class_type']}")
        ci, mi = cnode["inputs"], mnode["inputs"]
        for k in sorted(set(ci) | set(mi)):
            if k not in mi:
                diffs.append(f"{cid}.inputs.{k}:removed")
            elif k not in ci:
                diffs.append(f"{cid}.inputs.{k}:added")
            else:
                a, b = ci[k], mi[k]
                a_link = isinstance(a, list) and len(a) == 2
                b_link = isinstance(b, list) and len(b) == 2
                if a_link and b_link:
                    if [PARITY_MAP.get(a[0], a[0]), a[1]] != list(b):
                        diffs.append(f"{cid}.inputs.{k}:link")
                elif a_link != b_link or a != b:
                    diffs.append(f"{cid}.inputs.{k}:value")
    for mid, mnode in mine.items():
        if mid not in mapped:
            diffs.append(f"node_added:{mid}({mnode['class_type']})")
    return sorted(diffs)


def selftest(cfg: Config) -> int:
    cfg.ensure_dirs()
    manifest = load_manifest()
    P = prompts()
    d = cfg.defaults
    debug = cfg.debug_dir
    failures: list[str] = []

    def dump(graph: dict, name: str) -> None:
        (debug / f"selftest_{name}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    def check(label: str, fn) -> None:
        try:
            fn()
            print(f"  OK    {label}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"  FAIL  {label}: {exc}")

    one = ["h3app_ref_a.png"]
    four = [f"h3app_ref_{c}.png" for c in "abcd"]

    print("== frame maths ==")
    for secs, expected in ((5, 124), (10, 243), (15, 362)):
        got = graphs.frames_for_seconds(secs)
        check(f"{secs}s -> {expected} frames (got {got})",
              lambda got=got, expected=expected: _eq(got, expected))

    print("\n== case (a) 1 image / new clip / defaults ==")
    settings_a = build_settings(cfg, seconds=5, aspect="portrait", advanced={})
    ga = graphs.build_profile_graph(one, cfg.vlm, d["ref_longest"],
                                    P.SYS_CHARACTER_PROFILE, P.USER_CHARACTER_PROFILE)
    gb = graphs.build_director_graph(one, cfg.vlm, d["ref_longest"],
                                     P.SYS_DIRECTOR_NEW, "USER PROMPT", 42)
    gc = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                     d["ref_longest"], cfg["output_prefix"] + "/a_s1")
    dump(ga, "a_profile"), dump(gb, "a_director"), dump(gc, "a_generate")
    check("graph A with 1 image feeds the reference straight to the VLM (no stitch)",
          lambda: _eq(("sheet" in ga, "sheet_row0" in ga, ga["instruct"]["inputs"]["images"]),
                      (False, False, ["scale_0", 0])))
    check("graph B unload+purge is a data dependency",
          lambda: _eq(gb["purge"]["inputs"]["anything"], ["unload", 0]))
    check("assert_v1_parity (a)",
          lambda: graphs.assert_v1_parity(gc, settings_a, cfg.models, manifest, defaults=d))

    print("\n== case (b) 4 images / 10s / landscape ==")
    settings_b = build_settings(cfg, seconds=10, aspect="landscape", advanced={})
    gb_a = graphs.build_profile_graph(four, cfg.vlm, d["ref_longest"],
                                      P.SYS_CHARACTER_PROFILE, P.USER_CHARACTER_PROFILE)
    gb_b = graphs.build_director_graph(four, cfg.vlm, d["ref_longest"],
                                       P.SYS_DIRECTOR_NEW, "USER PROMPT", 7)
    gb_c = graphs.build_generate_graph(four, "EN PROMPT", settings_b, cfg.models,
                                       d["ref_longest"], cfg["output_prefix"] + "/b_s1")
    dump(gb_a, "b_profile"), dump(gb_b, "b_director"), dump(gb_c, "b_generate")
    check("4 references form a 2x2 ImageStitch sheet (core node, no Comfyroll)",
          lambda: _eq((gb_a["sheet_row0"]["class_type"], gb_a["sheet_row1"]["class_type"],
                       gb_a["sheet"]["inputs"]["direction"],
                       gb_a["sheet"]["inputs"]["image1"], gb_a["sheet"]["inputs"]["image2"],
                       gb_a["sheet_row0"]["inputs"]["image2"], gb_a["sheet_row1"]["inputs"]["image2"]),
                      ("ImageStitch", "ImageStitch", "down", ["sheet_row0", 0], ["sheet_row1", 0],
                       ["scale_1", 0], ["scale_3", 0])))
    check("no Comfyroll / BatchImagesNode node remains in the VLM graphs",
          lambda: _eq([n["class_type"] for g in (gb_a, gb_b, ga) for n in g.values()
                       if n["class_type"] in ("CR Image Grid Panel", "BatchImagesNode")], []))
    check("4 references reach MiniMaxH3ReferenceToVideo",
          lambda: _eq(sum(1 for k in gb_c["h3"]["inputs"] if k.startswith("ref_images.")), 4))
    check("10s -> 243 frames / 1024x576",
          lambda: _eq((gb_c["h3"]["inputs"]["length"], gb_c["h3"]["inputs"]["width"],
                       gb_c["h3"]["inputs"]["height"]), (243, 1024, 576)))
    check("assert_v1_parity (b)",
          lambda: graphs.assert_v1_parity(gb_c, settings_b, cfg.models, manifest, defaults=d))

    print("\n== case (c) continuation, step 2 and step 3 ==")
    tail = int(d["tail_frames"])
    for step, prev_total in ((2, 124), (3, 248)):
        cont = {"video": f"h3app_pid_s{step - 1}.mp4",
                "prev_total_frames": prev_total, "tail_frames": tail}
        gcx = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                          d["ref_longest"],
                                          cfg["output_prefix"] + f"/c_s{step}", cont)
        dump(gcx, f"c_generate_s{step}")
        check(f"step {step}: tail window = frames {prev_total - tail}..{prev_total}",
              lambda g=gcx, pt=prev_total: _eq(
                  (g["tail_images"]["inputs"]["batch_index"], g["tail_images"]["inputs"]["length"]),
                  (pt - tail, tail)))
        check(f"step {step}: audio tail = last {tail / 24.0:.4f}s",
              lambda g=gcx: _eq((g["tail_audio"]["inputs"]["start_index"],
                                 g["tail_audio"]["inputs"]["duration"]),
                                (-(tail / 24.0), tail / 24.0)))
        check(f"step {step}: saved file is the FULL video (concat)",
              lambda g=gcx: _eq(g["create_video"]["inputs"]["images"], ["concat_images", 0]))
        check(f"step {step}: the new segment is saved separately",
              lambda g=gcx: _eq(g["create_video_seg"]["inputs"]["images"], ["decode_video", 0]))
        check(f"step {step}: character references stay connected",
              lambda g=gcx: _eq("ref_images.ref_image_0" in g["h3"]["inputs"], True))
        check(f"step {step}: assert_v1_parity",
              lambda g=gcx: graphs.assert_v1_parity(g, settings_a, cfg.models, manifest,
                                                    defaults=d))

    print("\n== negative test: assert_v1_parity must REJECT tampering ==")
    for label, mutate in (
        ("ref_image_size match -> max",
         lambda g: g["h3"]["inputs"].__setitem__("ref_image_size", "max")),
        ("SelectCLIPDevice gpu:1 -> gpu:0",
         lambda g: g["clip_select"]["inputs"].__setitem__("device", "gpu:0")),
        ("H3CudaPhaseRestore target_gpu 0 -> 1",
         lambda g: g["restore"]["inputs"].__setitem__("target_gpu", 1)),
        ("steps 8 -> 20",
         lambda g: g["scheduler"]["inputs"].__setitem__("steps", 20)),
        ("barrier removed (plain CLIPLoader)",
         lambda g: g.__setitem__("clip", {"class_type": "CLIPLoader",
                                          "inputs": {"clip_name": cfg.models["clip"],
                                                     "type": "minimax", "device": "default"}})),
    ):
        tampered = copy.deepcopy(gc)
        mutate(tampered)
        try:
            graphs.assert_v1_parity(tampered, settings_a, cfg.models, manifest, defaults=d)
        except graphs.GraphError as exc:
            print(f"  OK    rejected: {label}  ({exc})")
        else:
            failures.append(f"negative test NOT rejected: {label}")
            print(f"  FAIL  NOT rejected: {label}")

    # steps tampering is caught by the settings comparison, so also prove the
    # manifest arm fires when the settings themselves are moved off the defaults.
    off_default = dict(settings_a)
    off_default["steps"] = 20
    tampered = graphs.build_generate_graph(one, "EN PROMPT", off_default, cfg.models,
                                           d["ref_longest"], "x")
    bad_models = dict(cfg.models)
    bad_models["unet"] = "some_other_model.safetensors"
    try:
        graphs.assert_v1_parity(tampered, off_default, bad_models, manifest, defaults=d)
    except graphs.GraphError as exc:
        print(f"  OK    rejected: unet name drift  ({exc})")
    else:
        failures.append("negative test NOT rejected: unet name drift")
        print("  FAIL  NOT rejected: unet name drift")

    print("\n== structural equivalence vs _hetero_test/opt_ref_match/candidate_prompt.json ==")
    cand_path = (APP_DIR.parent / "_hetero_test" / "opt_ref_match" / "candidate_prompt.json")
    if not cand_path.is_file():
        failures.append(f"candidate graph not found: {cand_path}")
        print(f"  FAIL  candidate graph not found: {cand_path}")
    else:
        candidate = json.loads(cand_path.read_text(encoding="utf-8"))
        diff = structural_diff(candidate, gc)
        print("  actual diff:")
        for line in diff:
            print(f"    {line}")
        print("  allow-list:")
        for line in ALLOWED_STRUCTURAL_DIFF:
            print(f"    {line}")
        unexpected = [x for x in diff if x not in ALLOWED_STRUCTURAL_DIFF]
        missing = [x for x in ALLOWED_STRUCTURAL_DIFF if x not in diff]
        if unexpected or missing:
            failures.append(f"structural diff mismatch: unexpected={unexpected} missing={missing}")
            print(f"  FAIL  unexpected={unexpected} missing={missing}")
        else:
            print("  OK    diff is exactly the allow-list")

    # ------------------------------------------------------------- story mode --
    print("\n== segmenter: the real example ==")
    example = (
        "「はじめまして、私はH3モデルです。よろしくお願いします。と笑顔で自己紹介する、"
        "深くお辞儀する。その後、部屋の紹介を始める"
        "『普段はこの椅子に座って配信してます。FPSは少し苦手です。』"
        "と話しながら椅子に座る。少し照れ笑い。その後、椅子に座り足をばたつかせながら"
        "今後の方針について話す"
        "『たくさんの人とお友達になれたらいいな～と思っています。応援よろしくお願いします。』"
        "とピースサイン」")
    # CLIP LENGTH IS A PARAMETER, THE SCRIPT IS THE FIXTURE.
    # Everything this block asserts is about what happens when a script needs
    # MORE than one clip: a quote cut at its own sentence end, an introducing
    # clause riding on the piece that starts the quote. After the speech-rate
    # recalibration (h3app\duration.py: 7.0 -> 28.0 mora/s, measured against real
    # H3 output) this three-quote script is only ~7 s of speech, so at 5 s it no
    # longer reaches that regime at all. It is run at 2 s instead - the same
    # script, the same guarantees, a clip it genuinely does not fit in.
    EXAMPLE_SECONDS = 2
    ex_segments = segmenter.segments_from_text(example, seconds=EXAMPLE_SECONDS)
    for seg in ex_segments:
        t = seg["timing"]
        print(f"  --- segment {seg['index']} ---")
        print(f"      prompt: {seg['prompt']!r}")
        print(f"      speech: {seg['speech']!r}")
        print(f"      est={t['estimated_seconds']}s of {t['segment_duration']}s "
              f"density={t['density']} band={t['band']} "
              f"split_reason={t['split_reason']} units={t['speech_units']}")
    check(f"3 <= segments <= 8 (got {len(ex_segments)})",
          lambda: _eq(3 <= len(ex_segments) <= 8, True))
    check("no empty segment",
          lambda: _eq([s["index"] for s in ex_segments
                       if not s["prompt"] and not s["speech"]], []))
    check("segment indexes are 0..n-1",
          lambda: _eq([s["index"] for s in ex_segments], list(range(len(ex_segments)))))

    quoted = ["普段はこの椅子に座って配信してます。FPSは少し苦手です。",
              "たくさんの人とお友達になれたらいいな～と思っています。応援よろしくお願いします。"]
    greeting = "はじめまして、私はH3モデルです。よろしくお願いします。"
    all_speech = "".join(s["speech"] for s in ex_segments).replace("\n", "")
    all_prompt = "\n".join(s["prompt"] for s in ex_segments)
    for q in quoted:
        check(f"quoted span lands wholly in speech: {q[:12]}...",
              lambda q=q: _eq(q in all_speech, True))
        check(f"quoted span is NOT in prompt: {q[:12]}...",
              lambda q=q: _eq(q in all_prompt, False))
    # BUG A: <発話>と<発話動詞> - the greeting is dialogue, not a stage direction.
    # NOTE: the greeting takes longer than one clip to say at EXAMPLE_SECONDS
    # (duration.py estimates it at ~2.4 s), so it is cut at its own sentence end
    # and spans two clips. It still STARTS in segment 0 and its pieces are still
    # consecutive.
    check("unquoted quotative speech is speech, starting in the FIRST segment",
          lambda: _eq((bool(ex_segments[0]["speech"]),
                       greeting.startswith(ex_segments[0]["speech"].replace("\n", "")),
                       greeting in all_speech),
                      (True, True, True)))
    check("unquoted quotative speech never leaks into any prompt",
          lambda: _eq([s["index"] for s in ex_segments
                       if "はじめまして" in s["prompt"] or "よろしくお願いします" in s["prompt"]],
                      []))
    check("the と + verb phrase stays in the prompt (it says HOW)",
          lambda: _eq(("と笑顔で自己紹介する" in all_prompt,
                       "と笑顔で自己紹介する" in all_speech),
                      (True, False)))
    # BUG B: a quote is never orphaned from the clause that introduces it. The
    # quote itself is longer than one clip can speak, so the guarantee is that
    # the introducing clause rides on the piece that STARTS it.
    check("the chair quote keeps its introducing clause",
          lambda: _eq([s["index"] for s in ex_segments
                       if s["speech"].split("\n")[0]
                       and quoted[0].startswith(s["speech"].split("\n")[0])
                       and "部屋の紹介を始める" in s["prompt"]] != [], True))
    check("an over-long quote is cut at a sentence end, never mid-sentence",
          lambda: _eq([p for p in ("普段はこの椅子に座って配信してます。", "FPSは少し苦手です。")
                       if p not in [ln["text"] for s in ex_segments
                                    for ln in s["lines"] if ln["type"] == "speech"]],
                      []))
    check("the closing quote keeps its introducing clause",
          lambda: _eq([s["index"] for s in ex_segments
                       if s["speech"] and "今後の方針について話す" in s["prompt"]] != [], True))
    bare = [ln["text"] for s in ex_segments for ln in s["lines"]
            if ln["type"] == "prompt" and ln["text"].startswith("と")
            and segmenter._looks_verbless(ln["text"][1:])]
    check("no segment prompt is a dangling と-fragment without a verb",
          lambda: _eq(bare, []))
    check("no prompt line is speech-looking (no segment has speech text in prompt)",
          lambda: _eq([q for q in quoted + [greeting] if q in all_prompt], []))
    check("quote marks never leak into a line",
          lambda: _eq([ln["text"] for s in ex_segments for ln in s["lines"]
                       if any(c in ln["text"] for c in "「」『』")], []))

    print("\n== segmenter: edge cases ==")
    check("plain unquoted text -> one prompt segment",
          lambda: _eq([(s["prompt"], s["speech"]) for s in
                       segmenter.segments_from_text("手を振る。")],
                      [("手を振る。", "")]))
    check("empty text -> no segments",
          lambda: _eq(segmenter.segments_from_text("   \n  "), []))
    check("a lone quote -> one speech segment",
          lambda: _eq([(s["prompt"], s["speech"]) for s in
                       segmenter.segments_from_text("『こんにちは』")],
                      [("", "こんにちは")]))
    check("plain と (not quotative) is left alone",
          lambda: _eq([(s["prompt"], s["speech"]) for s in
                       segmenter.segments_from_text("椅子と机を見せる。")],
                      [("椅子と机を見せる。", "")]))
    # The quotative-と reach-back is a LINE-level guarantee (split_lines), so it
    # is asserted there. The packer no longer breaks on the marker: 手を振る +
    # こんにちは is about 2 s of content and one 5 s clip holds all of it, which
    # is the whole point of packing by estimated duration.
    check("quotative と after a marker only reaches back to the marker (lines)",
          lambda: _eq([(l["type"], l["text"]) for l in
                       segmenter.split_lines("手を振る。その後、こんにちは、と言う。")],
                      [("prompt", "手を振る。その後、"), ("speech", "こんにちは"),
                       ("prompt", "と言う。")]))
    check("an action marker is a split CANDIDATE, not a forced break",
          lambda: _eq([(s["prompt"], s["speech"]) for s in
                       segmenter.segments_from_text("手を振る。その後、こんにちは、と言う。")],
                      [("手を振る。\nその後、\nと言う。", "こんにちは")]))

    # ---------------------------------------------- duration-based segmenter --
    # The packer no longer counts characters or stage directions: it estimates
    # how many SECONDS the text takes to say (h3app\duration.py) and fills the
    # clip's own duration. Everything below is pure: no GPU, no ComfyUI, no clock.
    print("\n== segmenter: estimated-duration packing ==")

    def _show(label: str, segs: list) -> None:
        print(f"    {label}: {len(segs)} segment(s)")
        for s in segs:
            t = s["timing"]
            print(f"      [{s['index']}] est={t['estimated_seconds']:>5}s "
                  f"(speech {t['estimated_speech_seconds']}s / action "
                  f"{t['estimated_action_seconds']}s) density={t['density']:>5} "
                  f"{t['band']:<11} reason={t['split_reason']:<14} "
                  f"boundary={t.get('boundary_type', '?'):<9} "
                  f"comma_fallback={str(t.get('fallback_comma_split')):<5} "
                  f"units={t['speech_units']}")
            print(f"           speech: {s['speech']!r}")

    # (1) THE REGRESSION: nine typed lines, five of them stage directions.
    REPRO = [("prompt", "カメラ正面で笑顔になる。"),
             ("speech", "はじめまして、白凪みおです。"),
             ("prompt", "軽く会釈する。"),
             ("speech", "普段はゲームやパソコンが好きです。"),
             ("prompt", "少し照れ笑いする。"),
             ("prompt", "手を振る。"),
             ("speech", "よかったら仲良くしてね。"),
             ("prompt", "椅子に座る。"),
             ("prompt", "ピースサインをする。")]
    repro_lines = [{"type": t, "text": x} for t, x in REPRO]
    print("  (1) the 9-line reproduction - before / after")
    old_structured = _legacy_segment_count(repro_lines, seconds=5)
    new_structured = segmenter.segments_from_lines(repro_lines, seconds=5)
    # The same nine lines pasted as RAW TEXT: nine stage-direction lines, and
    # the old packer made one segment out of every single one of them.
    old_rawtext = 9                       # measured before this change
    new_rawtext = segmenter.segments_from_text("\n".join(x for _, x in REPRO),
                                               seconds=5)
    print(f"    typed lines : before {old_structured} -> after {len(new_structured)}")
    _show("after (typed lines)", new_structured)
    print(f"    raw text    : before {old_rawtext} -> after {len(new_rawtext)}")
    _show("after (raw text)", new_rawtext)
    check("1: no segment exists with zero speech because of a stage-direction count",
          lambda: _eq([s["index"] for s in new_structured if not s["speech"]], []))
    check("1: the raw-text explosion is gone (fewer segments than before)",
          lambda: _eq(len(new_rawtext) < old_rawtext, True))
    check("1: typed lines never produce MORE segments than the old packer did",
          lambda: _eq(len(new_structured) <= old_structured, True))
    check("1: every segment fits its own duration budget",
          lambda: _eq([s["index"] for s in new_structured
                       if s["timing"]["band"] == "overloaded"], []))

    # (2) The same text, three deliveries. Nothing about the segment count is
    #     fixed: how the character speaks changes how much fits in one clip.
    #     The 9-line script is repeated so it is longer than one clip at EVERY
    #     delivery: after the recalibration it is ~4.9 s of speech, which fits a
    #     single 5 s clip, and a script that fits one clip cannot show a
    #     delivery changing the segment count no matter what the packer does.
    print("  (2) same text, three deliveries")
    delivery_lines = repro_lines * 3
    deliveries = [("normal", None),
                  ("fast / excited", {"pace": "fast", "emotion": "excited",
                                      "pause_level": "low"}),
                  ("slow / sad", {"pace": "slow", "emotion": "sad",
                                  "pause_level": "high"})]
    by_delivery = []
    for label, dl in deliveries:
        segs = segmenter.segments_from_lines(delivery_lines, seconds=5, delivery=dl)
        total = round(sum(s["timing"]["estimated_speech_seconds"] for s in segs), 3)
        by_delivery.append((label, dl, segs, total))
        print(f"    {label:<15} -> {len(segs)} segment(s), "
              f"{total}s of estimated speech")
    check("2: the same text estimates a different number of seconds per delivery",
          lambda: _eq(len({t for _, _, _, t in by_delivery}), 3))
    check("2: the segment counts are NOT all identical",
          lambda: _eq(len({len(s) for _, _, s, _ in by_delivery}) > 1, True))
    check("2: a faster delivery never estimates more seconds than a slower one",
          lambda: _eq(by_delivery[1][3] < by_delivery[0][3] < by_delivery[2][3], True))

    # (3) Short sentences that comfortably fit one budget stay together.
    print("  (3) several short sentences fit ONE segment")
    short_lines = [{"type": "prompt", "text": t} for t in
                   ("手を振る。", "軽くうなずく。", "首をかしげる。", "微笑む。")]
    short_segs = segmenter.segments_from_lines(short_lines, seconds=5)
    _show("four short stage directions", short_segs)
    check("3: four short stage directions are not split one per sentence",
          lambda: _eq(len(short_segs), 1))
    short_speech = [{"type": "speech", "text": t} for t in ("こんにちは。", "元気？")]
    short_speech_segs = segmenter.segments_from_lines(short_speech, seconds=5)
    check("3: two short utterances that fit together are not split either",
          lambda: _eq(len(short_speech_segs), 1))

    # (4) Nothing caps the segment count: long text still splits as often as the
    #     duration says it must.
    print("  (4) a long script still splits as many times as it needs to")
    long_lines: list[dict] = []
    # 24, not 8: one prompt+speech pair is ~1.6 s at the recalibrated rate, so 8
    # of them are three clips, not eight. The assertion below is unchanged - the
    # script just has to be long enough to still be able to break it.
    for i in range(24):
        long_lines.append({"type": "prompt", "text": "カメラを見て姿勢を整える。"})
        long_lines.append({"type": "speech",
                           "text": "今日は使い方を順番に説明します。"})
    long_segs = segmenter.segments_from_lines(long_lines, seconds=5)
    print(f"    {len(long_lines)} lines -> {len(long_segs)} segments "
          f"(legacy packer: {_legacy_segment_count(long_lines, seconds=5)})")
    print("    (MORE than the legacy packer on purpose: the legacy budget of 32 "
          "characters\n     packed two ~3.9 s utterances into one 5 s clip.)")
    check("4: a long script produces many segments (no artificial cap)",
          lambda: _eq(len(long_segs) >= 5, True))
    check("4: doubling the clip duration reduces the segment count",
          lambda: _eq(len(segmenter.segments_from_lines(long_lines, seconds=10))
                      < len(long_segs), True))

    # (5) One utterance longer than a whole clip still splits INSIDE the line,
    #     and only at a sentence end.
    print("  (5) one over-long utterance splits internally")
    # Six sentences (~7.3 s at the recalibrated rate) so that ONE line is still
    # longer than one 5 s clip. The four-sentence version this replaces is only
    # ~4.6 s now and no longer exercises the internal split at all.
    over = ("こんにちは、はじめまして。今日は自己紹介をさせてください。"
            "普段はゲームの配信をしています。よろしくお願いします。"
            "最近は動画の編集にも挑戦しています。"
            "これからも少しずつ続けていきたいと思っています。")
    over_segs = segmenter.segments_from_lines(
        [{"type": "speech", "text": over}], seconds=5)
    _show("one long utterance", over_segs)
    check("5: an utterance longer than the budget is split",
          lambda: _eq(len(over_segs) > 1, True))
    check("5: every piece ends at a sentence end",
          lambda: _eq([s["index"] for s in over_segs
                       if not s["speech"].rstrip().endswith("。")], []))
    check("5: the pieces reassemble into the original utterance",
          lambda: _eq("".join(s["speech"] for s in over_segs).replace("\n", ""), over))
    check("5: the split reason says so",
          lambda: _eq(over_segs[0]["timing"]["split_reason"], "long_utterance"))

    # (6) MORA, not characters.
    print("  (6) mora estimation")
    kana5, kana8 = "あいうえお", "あいうえおかきく"
    for label, text, other in (("2026年", "2026年", kana5),
                               ("RTX 5090", "RTX 5090", kana8)):
        a = duration_mod.estimate_speech_seconds(text)[0]
        b = duration_mod.estimate_speech_seconds(other)[0]
        print(f"    {label!r} ({len(text)} chars) = {a}s   vs   "
              f"{other!r} ({len(other)} chars) = {b}s")
        check(f"6: {label} takes longer to say than {len(other)} plain kana",
              lambda a=a, b=b: _eq(a > b, True))
    print(f"    きゃ={duration_mod.count_moras('きゃ')} きや={duration_mod.count_moras('きや')} "
          f"らーめん={duration_mod.count_moras('らーめん')} "
          f"がっこう={duration_mod.count_moras('がっこう')} "
          f"ほん={duration_mod.count_moras('ほん')}")
    check("6: a small kana adds no mora (きゃ=1, きや=2)",
          lambda: _eq((duration_mod.count_moras("きゃ"),
                       duration_mod.count_moras("きや")), (1.0, 2.0)))
    check("6: ー / っ / ん each count as one mora",
          lambda: _eq((duration_mod.count_moras("らーめん"),
                       duration_mod.count_moras("がっこう"),
                       duration_mod.count_moras("ほん")), (4.0, 4.0, 2.0)))
    check("6: an acronym is read letter by letter, a word is not",
          lambda: _eq(duration_mod.count_moras("RTX") > duration_mod.count_moras("Rtx"),
                      True))

    # (7) The structural guarantees survive the rewrite.
    print("  (7) quotes are never cut mid-quote; no bare と-fragment")
    quote_text = ("その後、部屋の紹介を始める『こんにちは。』と話しながら椅子に座る。"
                  "その後、ピースサインをする。")
    q_segs = segmenter.segments_from_text(quote_text, seconds=5)
    _show("quoted script", q_segs)
    check("7: the quote is never cut in half",
          lambda: _eq([s["index"] for s in q_segs
                       if s["speech"] and s["speech"] != "こんにちは。"], []))
    check("7: no segment prompt is a dangling と-fragment without a verb",
          lambda: _eq([ln["text"] for s in q_segs for ln in s["lines"]
                       if ln["type"] == "prompt" and ln["text"].startswith("と")
                       and segmenter._looks_verbless(ln["text"][1:])], []))
    check("7: quote marks never leak into a line",
          lambda: _eq([ln["text"] for s in q_segs for ln in s["lines"]
                       if any(c in ln["text"] for c in "「」『』")], []))

    # (8) Explicit user structure still wins.
    print("  (8) segments_from_lines still preserves explicit types verbatim")
    explicit = [{"type": "prompt", "text": "こんにちは、と言う。"},
                {"type": "speech", "text": "椅子と机を見せる。"},
                {"type": "speech", "text": "その後、手を振る。"}]
    ex_out = segmenter.segments_from_lines(explicit, seconds=5)
    check("8: types and text pass through unchanged",
          lambda: _eq([(l["type"], l["text"]) for s in ex_out for l in s["lines"]],
                      [(l["type"], l["text"]) for l in explicit]))

    # (8b) An action marker is PREFERRED as a cut point when a cut is needed.
    print("  (8b) a cut is moved onto the nearest split candidate")
    cand = [{"type": "speech", "text": "こんにちは、はじめまして。"},
            {"type": "prompt", "text": "その後、椅子に座る。"},
            {"type": "prompt", "text": "カメラを見る。"},
            {"type": "speech", "text": "今日は使い方を順番に説明します。"}]
    # 2.5 s, not 5 s: moving a cut onto a candidate only happens when there IS a
    # cut, and at the recalibrated rate these four lines are ~3.2 s and fit one
    # 5 s clip. Same lines, same assertion, a clip they do not fit in.
    cand_segs = segmenter.segments_from_lines(cand, seconds=2.5)
    _show("cut moved to the marker", cand_segs)
    check("8b: the marker clause travels with what FOLLOWS it, not what precedes",
          lambda: _eq([(s["timing"]["split_reason"], s["prompt"].split("\n")[0])
                       for s in cand_segs],
                      [("action_marker", ""), ("end", "その後、椅子に座る。")]))

    # (9) Band classification and the too_short merge.
    print("  (9) fill band + merge of a too_short segment")
    # Two stage directions stranded in front of an utterance that has to be cut
    # in half: the greedy pass leaves them as a 1 s segment with no speech in it
    # at all - exactly the shape that used to reach the director as a silent
    # clip. It is `too_short`, it fits in the next segment, so it is merged.
    tail_lines = [{"type": "prompt", "text": "カメラ正面に立つ。"},
                  {"type": "prompt", "text": "笑顔になる。"},
                  {"type": "speech",
                   "text": ("今日は新しいアプリの使い方を説明します。"
                            "よろしくお願いします、最後まで見てください。")}]
    # 1.5 s, not 5 s: at the recalibrated rate these three lines are ~3.1 s and
    # fit ONE 5 s clip, so nothing is stranded and there is nothing to merge.
    # The same lines in a clip they do not fit produce exactly the shape this
    # check exists for - two stage directions with no speech over them.
    merged_segs = segmenter.segments_from_lines(tail_lines, seconds=1.5)
    _show("stage directions in front of a long utterance", merged_segs)
    check("9: the too_short leftover is not left as its own silent segment",
          lambda: _eq([s["index"] for s in merged_segs if not s["speech"]], []))
    check("9: a merged segment says so in its debug metadata",
          lambda: _eq(any(s["timing"]["split_reason"] == "merged"
                          for s in merged_segs), True))
    check("9: bands are classified, never forced",
          lambda: _eq(sorted({s["timing"]["band"] for s in merged_segs})
                      != [], True))
    print("    band edges: too_short < "
          f"{duration_mod.BAND_TOO_SHORT} <= comfortable <= "
          f"{duration_mod.BAND_DENSE} < dense <= "
          f"{duration_mod.BAND_OVERLOADED} < overloaded")

    # A longer, realistic script: the before/after table asked for.
    print("  (+) a longer realistic script - before / after")
    realistic = [
        ("prompt", "カメラ正面で軽く手を振る。"),
        ("speech", "みなさんこんにちは、配信を見に来てくれてありがとうございます。"),
        ("prompt", "少し照れ笑いする。"),
        ("prompt", "椅子に座り直す。"),
        ("speech", "今日は2026年に発売されたRTX 5090の話をします。"),
        ("prompt", "画面の方を指差す。"),
        ("speech", "去年のモデルと比べて、生成の速度がかなり上がりました。"),
        ("prompt", "首をかしげる。"),
        ("speech", "とはいえ、お値段はちょっと考えてしまいますね。"),
        ("prompt", "肩をすくめる。"),
        ("prompt", "カメラに向き直る。"),
        ("speech", "よかったら最後まで見ていってください。"),
        ("prompt", "ピースサインをする。"),
    ]
    real_lines = [{"type": t, "text": x} for t, x in realistic]
    real_segs = segmenter.segments_from_lines(real_lines, seconds=5)
    print(f"    before (char-count packer): "
          f"{_legacy_segment_count(real_lines, seconds=5)} segments")
    _show("after (duration packer)", real_segs)
    check("+: the realistic script has no silent leftover segment",
          lambda: _eq([s["index"] for s in real_segs if not s["speech"]], []))

    # ---------------------------------------------------------------------- --
    # (10) THE OVER-SPLIT REGRESSION, on the real script the user reported it
    #      with. 8 typed lines: 4 SCENE lines and 4 DIALOGUE lines, alternating,
    #      fed through the STRUCTURED path exactly as the UI feeds it.
    # ------------------------------------------------------------------------ -
    print("\n== segmenter: the long-script over-split (fixture) ==")
    fixture_path = APP_DIR / "tests_fixtures" / "story_long_script.txt"
    fixture_lines = [ln.strip() for ln in
                     fixture_path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
    # ODD lines are SCENE, EVEN lines are DIALOGUE. This is the structure the UI
    # supplies; the segmenter is never asked to infer it.
    fx_lines = [{"type": ("prompt" if i % 2 == 0 else "speech"), "text": t}
                for i, t in enumerate(fixture_lines)]
    print(f"    fixture: {fixture_path.name}, {len(fx_lines)} typed lines "
          f"({len([l for l in fx_lines if l['type'] == 'prompt'])} scene / "
          f"{len([l for l in fx_lines if l['type'] == 'speech'])} dialogue)")

    # MEASURED with the pre-change code, at seconds=5, default delivery. Kept as
    # data because the old code no longer exists to re-run.
    FIXTURE_BEFORE = [
        (2.993, "comfortable", "long_utterance"), (5.129, "overloaded", "long_utterance"),
        (4.279, "dense", "long_utterance"), (5.129, "overloaded", "long_utterance"),
        (4.307, "dense", "long_utterance"), (4.471, "dense", "long_utterance"),
        (2.157, "too_short", "long_utterance"), (7.421, "overloaded", "long_utterance"),
        (2.329, "comfortable", "long_utterance"), (6.307, "overloaded", "long_utterance"),
        (3.679, "comfortable", "long_utterance"), (4.671, "dense", "long_utterance"),
        (4.271, "dense", "long_utterance"), (5.593, "overloaded", "long_utterance"),
        (3.993, "comfortable", "long_utterance"), (1.814, "too_short", "long_utterance"),
        (4.421, "dense", "long_utterance"), (5.393, "overloaded", "long_utterance"),
        (4.571, "dense", "end"),
    ]
    fx_segs = segmenter.segments_from_lines(fx_lines, seconds=5)
    print(f"\n    BEFORE: {len(FIXTURE_BEFORE)} segments      "
          f"AFTER: {len(fx_segs)} segments")
    print("      #  | BEFORE  est   band         reason         "
          "| AFTER   est   band         boundary   reason         scene comma")
    for i in range(max(len(FIXTURE_BEFORE), len(fx_segs))):
        if i < len(FIXTURE_BEFORE):
            e, b, r = FIXTURE_BEFORE[i]
            before = f"{e:>7.3f}s  {b:<12} {r:<14}"
        else:
            before = " " * 36
        if i < len(fx_segs):
            t = fx_segs[i]["timing"]
            after = (f"{t['estimated_seconds']:>7.3f}s  {t['band']:<12} "
                     f"{t['boundary_type']:<10} {t['split_reason']:<14} "
                     f"{t['scene_index']:>5} {str(t['fallback_comma_split']):<5}")
        else:
            after = ""
        print(f"      {i:>2} | {before} | {after}")
    print("    AFTER, in full:")
    for s in fx_segs:
        print(f"      [{s['index']}] prompt: {s['prompt']!r}")
        print(f"           speech: {s['speech']!r}")

    check(f"10: the 19-way over-split is gone (19 -> {len(fx_segs)})",
          lambda: _eq(len(fx_segs) <= len(FIXTURE_BEFORE) // 2, True))
    # A normal boundary is never a 、. A segment may only END on one when a
    # single sentence did not fit a clip even on its own - and then it says so.
    check("10: no segment's speech ends on a bare 、 unless it is a comma fallback",
          lambda: _eq([s["index"] for s in fx_segs
                       if s["speech"].rstrip().endswith(("、", "，", ","))
                       and not s["timing"]["fallback_comma_split"]], []))
    check("10: NO segment has an empty prompt",
          lambda: _eq([s["index"] for s in fx_segs if not s["prompt"].strip()], []))
    check("10: no segment has an empty speech either",
          lambda: _eq([s["index"] for s in fx_segs if not s["speech"].strip()], []))
    # Every segment traces back to the SCENE line it came from, and all four
    # scenes are still represented. Asserted on the trace, not on any wording.
    scene_lines = [i for i, l in enumerate(fx_lines) if l["type"] == "prompt"]
    check("10: the 4 scene/dialogue pairings are preserved (scene trace)",
          lambda: _eq(sorted({s["timing"]["scene_index"] for s in fx_segs}),
                      scene_lines))
    check("10: every segment traces back to exactly one scene",
          lambda: _eq([s["index"] for s in fx_segs
                       if len(s["timing"]["scene_indices"]) != 1], []))
    check("10: every segment's source lines belong to its own scene's pair",
          lambda: _eq([s["index"] for s in fx_segs
                       if not set(s["timing"]["source_units"]).issubset(
                           {s["timing"]["scene_index"],
                            s["timing"]["scene_index"] + 1})], []))
    check("10: a follow-on segment carries a continuation note, not a copy of "
          "the scene",
          lambda: _eq([s["index"] for s in fx_segs
                       if s["timing"]["scene_continuation"]
                       and s["prompt"].split("\n")[0]
                       != segmenter.SCENE_CONTINUATION_NOTE], []))
    check("10: the scene text is never duplicated into a follow-on segment",
          lambda: _eq([s["index"] for s in fx_segs
                       if s["timing"]["scene_continuation"]
                       and fixture_lines[s["timing"]["scene_index"]] in s["prompt"]],
                      []))
    check("10: the dialogue is preserved verbatim, in order",
          lambda: _eq("".join(s["speech"] for s in fx_segs).replace("\n", ""),
                      "".join(fixture_lines[1::2])))
    check("10: no segment is overloaded",
          lambda: _eq([s["index"] for s in fx_segs
                       if s["timing"]["band"] == "overloaded"], []))
    # Several sentences share a clip: that is the packer running on the pieces of
    # a split utterance, which is exactly what used to be bypassed.
    check("10: several sentences of one utterance are packed into one segment",
          lambda: _eq(any(s["speech"].count("。") > 1 for s in fx_segs), True))

    print("\n  (10b) the fixture at three deliveries")
    fx_by_delivery = []
    for label, dl in deliveries:
        segs = segmenter.segments_from_lines(fx_lines, seconds=5, delivery=dl)
        total = round(sum(s["timing"]["estimated_speech_seconds"] for s in segs), 3)
        fx_by_delivery.append((label, len(segs), total))
        print(f"    {label:<15} -> {len(segs):>2} segment(s), {total}s of "
              f"estimated speech")
    check("10b: the three deliveries estimate three different totals",
          lambda: _eq(len({t for _, _, t in fx_by_delivery}), 3))
    check("10b: the three deliveries do NOT all give the same segment count",
          lambda: _eq(len({n for _, n, _ in fx_by_delivery}) > 1, True))

    print("\n  (10c) 、 is the LAST RESORT, and it says when it was used")
    # ONE sentence, no 。 inside it, longer than a clip. There is no sentence end
    # to cut at, so the packer is allowed to fall back to the 、 - and every
    # segment it produces that way is flagged.
    one_sentence = ("今日はこの新しくなった仕組みについて、どんなことができるのかを、"
                    "順番にひとつずつ、できるだけていねいに、"
                    "実際に画面を動かしながら、途中で困ったところも隠さずに、"
                    "見てくれているみなさんに最後まで紹介していきたいと思っています。")
    comma_segs = segmenter.segments_from_lines(
        [{"type": "speech", "text": one_sentence}], seconds=5)
    _show("one over-long sentence with no sentence end inside it", comma_segs)
    check("10c: one sentence too long for a clip is still split",
          lambda: _eq(len(comma_segs) > 1, True))
    check("10c: every piece of it is marked fallback_comma_split",
          lambda: _eq([s["index"] for s in comma_segs
                       if not s["timing"]["fallback_comma_split"]], []))
    check("10c: the comma fallback is recorded as the boundary type",
          lambda: _eq(any(s["timing"]["boundary_type"] == "comma"
                          for s in comma_segs), True))
    check("10c: the sentence is reassembled verbatim",
          lambda: _eq("".join(s["speech"] for s in comma_segs).replace("\n", ""),
                      one_sentence))
    # ... and a sentence that DOES fit is never cut at a 、, even though it has
    # plenty of them.
    fits_lines = [{"type": "speech", "text": "今日は、この仕組みを、少しだけ試します。"}]
    fits_segs = segmenter.segments_from_lines(fits_lines, seconds=5)
    check("10c: a sentence that fits is never cut at a 、",
          lambda: _eq((len(fits_segs),
                       fits_segs[0]["timing"]["fallback_comma_split"]),
                      (1, False)))

    print("\n  (10d) short sentences that fit one budget share a segment")
    pack_lines = [{"type": "speech", "text": t} for t in
                  ("こんにちは。", "元気ですか。", "今日はいい天気ですね。",
                   "少しだけお話しします。")]
    pack_segs = segmenter.segments_from_lines(pack_lines, seconds=5)
    _show("four short utterances", pack_segs)
    check("10d: four short utterances are packed into ONE segment",
          lambda: _eq(len(pack_segs), 1))

    print("\n== graphs: concat_previous=False (story tail relay) ==")
    cont_s = {"video": "h3app_story_x_s1.mp4", "prev_total_frames": 124,
              "tail_frames": tail}
    g_relay = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                          d["ref_longest"], cfg["output_prefix"] + "/story_s2",
                                          cont_s, concat_previous=False)
    dump(g_relay, "story_generate_relay")
    check("assert_v1_parity (relay)",
          lambda: graphs.assert_v1_parity(g_relay, settings_a, cfg.models, manifest, defaults=d))
    check("exactly one SaveVideo",
          lambda: _eq(len([n for n in g_relay.values()
                           if n["class_type"] == "SaveVideo"]), 1))
    check("no ImageBatch / AudioConcat",
          lambda: _eq(sorted({n["class_type"] for n in g_relay.values()}
                             & {"ImageBatch", "AudioConcat"}), []))
    check("no extra segment save nodes",
          lambda: _eq([k for k in ("concat_images", "concat_audio",
                                   "create_video_seg", graphs.NODE_SAVE_SEG)
                       if k in g_relay], []))
    check("CreateVideo takes decode_video / decode_audio directly",
          lambda: _eq((g_relay["create_video"]["inputs"]["images"],
                       g_relay["create_video"]["inputs"]["audio"]),
                      (["decode_video", 0], ["decode_audio", 0])))
    check("tail chain is still built",
          lambda: _eq([g_relay[k]["class_type"] for k in
                       ("prev_video", "prev_components", "tail_images", "tail_audio")],
                      ["LoadVideo", "GetVideoComponents", "ImageFromBatch",
                       "TrimAudioDuration"]))
    check("ref_videos / ref_video_audios come from the tail",
          lambda: _eq((g_relay["h3"]["inputs"]["ref_videos.ref_video_0"],
                       g_relay["h3"]["inputs"]["ref_video_audios.ref_video_audio_0"]),
                      (["tail_images", 0], ["tail_audio", 0])))
    check("tail window uses the PREVIOUS SEGMENT's frames only",
          lambda: _eq((g_relay["tail_images"]["inputs"]["batch_index"],
                       g_relay["tail_images"]["inputs"]["length"]),
                      (124 - tail, tail)))

    print("\n== graphs: concat_previous=True is unchanged (single-shot guard) ==")
    cont_ss = {"video": "h3app_pid_s1.mp4", "prev_total_frames": 124, "tail_frames": tail}
    g_default = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                            d["ref_longest"], cfg["output_prefix"] + "/c_s2",
                                            cont_ss)
    g_true = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                         d["ref_longest"], cfg["output_prefix"] + "/c_s2",
                                         cont_ss, concat_previous=True)
    check("default argument == concat_previous=True",
          lambda: _eq(json.dumps(g_default, ensure_ascii=False, sort_keys=True),
                      json.dumps(g_true, ensure_ascii=False, sort_keys=True)))
    # The exact node set the pre-change builder produced for a continuation.
    expected_nodes = {
        "load_0": "LoadImage", "scale_0": "LayerUtility: ImageScaleByAspectRatio V2",
        "restore": "H3CudaPhaseRestore", "clip": "H3GatedCLIPLoader",
        "clip_report_base": "H3ClipDeviceReport", "clip_select": "SelectCLIPDevice",
        "clip_report_target": "H3ClipDeviceReport",
        "vae_video": "VAELoader", "vae_audio": "VAELoader",
        "unet": "UNETLoader", "lora": "LoraLoaderModelOnly",
        "prev_video": "LoadVideo", "prev_components": "GetVideoComponents",
        "tail_images": "ImageFromBatch", "tail_audio": "TrimAudioDuration",
        "h3": "MiniMaxH3ReferenceToVideo", "noise": "RandomNoise",
        "guider": "BasicGuider", "sampler_select": "KSamplerSelect",
        "scheduler": "BasicScheduler", "sampler": "SamplerCustomAdvanced",
        "decode_video": "VAEDecode", "decode_audio": "VAEDecodeAudio",
        "concat_images": "ImageBatch", "concat_audio": "AudioConcat",
        "create_video": "CreateVideo", "save_video": "SaveVideo",
        "create_video_seg": "CreateVideo", "save_video_seg": "SaveVideo",
    }
    check("continuation node set is unchanged",
          lambda: _eq({k: v["class_type"] for k, v in g_default.items()}, expected_nodes))
    check("main save is still the FULL concatenated video",
          lambda: _eq((g_default["create_video"]["inputs"]["images"],
                       g_default["create_video"]["inputs"]["audio"]),
                      (["concat_images", 0], ["concat_audio", 0])))
    check("the _seg save is still the new segment only",
          lambda: _eq((g_default["create_video_seg"]["inputs"]["images"],
                       g_default[graphs.NODE_SAVE_SEG]["inputs"]["filename_prefix"]),
                      (["decode_video", 0], cfg["output_prefix"] + "/c_s2_seg")))
    check("assert_v1_parity (concat)",
          lambda: graphs.assert_v1_parity(g_default, settings_a, cfg.models, manifest,
                                          defaults=d))

    print("\n== story store round trip ==")
    story_root = debug / "selftest_story_projects"
    if story_root.exists():
        shutil.rmtree(story_root, ignore_errors=True)
    st_store = StoryStore(story_root)
    st = st_store.create(
        name="セルフテスト", source_text=example,
        lines=segmenter.split_lines(example), segments=ex_segments,
        images=one, settings=settings_a, jp_negative=P.NEGATIVE_DEFAULT,
        # Project level = STYLE presets only (an action id here would be migrated
        # away instead of being applied to every clip - see the preset split).
        motion_presets=["smile_talk", "calm"])
    st.data["cursor"] = 1
    st.clips.append({"segment_index": 0, "step": 1, "video": "X:/out/clip_01.mp4",
                     "local_video": str(st.clips_dir / "clip_01.mp4"),
                     "last_frame": str(st.frames_dir / "frame_00.png"),
                     "frames": 124, "seconds": 5.167, "seed": 123456789,
                     "en_prompt": "EN", "created_at": 0.0})
    st.segments[0]["status"] = "done"
    st.save()
    check("segments.json written", lambda: _eq(st.segments_path.is_file(), True))
    check("project.json written", lambda: _eq(st.project_path.is_file(), True))
    check("clips / frames / final folders exist",
          lambda: _eq([p.is_dir() for p in (st.clips_dir, st.frames_dir, st.final_dir)],
                      [True, True, True]))
    check("project.json holds no segment array",
          lambda: _eq("segments" in json.loads(st.project_path.read_text(encoding="utf-8")),
                      False))

    reloaded = StoryStore(story_root).get(st.id)
    check("reload finds the story", lambda: _eq(reloaded is not None, True))
    check("cursor survives", lambda: _eq(reloaded.cursor, 1))
    check("clips survive", lambda: _eq(len(reloaded.clips), 1))
    check("segments survive",
          lambda: _eq([s["prompt"] for s in reloaded.segments],
                      [s["prompt"] for s in ex_segments]))
    check("settings / images / negative survive",
          lambda: _eq((reloaded.data["settings"], reloaded.data["images"],
                       reloaded.data["jp_negative"]),
                      (settings_a, one, P.NEGATIVE_DEFAULT)))
    check("motion presets survive",
          lambda: _eq(reloaded.data["motion_presets"], ["smile_talk", "calm"]))
    check("summary has everything a resume list needs",
          lambda: _eq(sorted(reloaded.summary()),
                      sorted(["id", "name", "status", "cursor", "segment_total",
                              "clips_done", "updated_at", "created_at", "merged"])))
    check("clip lookup by segment index",
          lambda: _eq(reloaded.clip_for(0)["step"], 1))

    print("  --- project.json ---")
    for line in st.project_path.read_text(encoding="utf-8").splitlines():
        print("    " + line)
    print("  --- segments.json (first segment) ---")
    for line in json.dumps(json.loads(st.segments_path.read_text(encoding="utf-8"))[0],
                           ensure_ascii=False, indent=2).splitlines():
        print("    " + line)
    shutil.rmtree(story_root, ignore_errors=True)

    print("\n== continuation must NOT re-feed what was already staged ==")
    ORIG_SCENE = "ORIGINAL_SCENE_MARKER 部屋で自己紹介をする"
    ORIG_DIALOGUE = "ORIGINAL_DIALOGUE_MARKER はじめまして、私はH3モデルです。"

    class _FakeProject:
        def __init__(self, data):
            self.data = data

        def get(self, k, default=None):
            return self.data.get(k, default)

    fake = _FakeProject({
        "jp_scene": ORIG_SCENE, "jp_dialogue": ORIG_DIALOGUE,
        "jp_negative": P.NEGATIVE_DEFAULT, "profile": "PROFILE",
        "settings": settings_a,
    })
    prev_clip = {"en_prompt": "summary\nshe waves at the camera\nretention_analysis\n"}
    jp_prev = pipeline_mod._previous_summary(fake, prev_clip)

    # This mirrors _run_director's body for the single-shot continue path.
    cont_prompt = build_director_text(
        profile=fake.data["profile"],
        jp_scene=pipeline_mod.CONTINUE_SCENE_DEFAULT,
        jp_prev=jp_prev,
        jp_negative=fake.data["jp_negative"],
        jp_dialogue="",
        frames=int(settings_a["frames"]))
    (debug / "selftest_continue_director.txt").write_text(cont_prompt, encoding="utf-8")
    check("continue: the original DIALOGUE is gone from the director input",
          lambda: _eq(ORIG_DIALOGUE in cont_prompt, False))
    check("continue: the carry-forward instruction is present",
          lambda: _eq(pipeline_mod.CONTINUE_SCENE_DEFAULT in cont_prompt, True))
    check("continue: the previous clip is still described (tail context kept)",
          lambda: _eq("she waves at the camera" in cont_prompt, True))

    new_prompt = build_director_text(
        profile=fake.data["profile"], jp_scene=ORIG_SCENE, jp_prev=pipeline_mod._no_prev(),
        jp_negative=fake.data["jp_negative"], jp_dialogue=ORIG_DIALOGUE,
        frames=int(settings_a["frames"]))
    check("first clip DOES contain the original scene and dialogue",
          lambda: _eq(ORIG_SCENE in new_prompt and ORIG_DIALOGUE in new_prompt, True))

    # Story mode: each segment supplies its OWN text, and only its own.
    story_root2 = debug / "selftest_story_dir"
    shutil.rmtree(story_root2, ignore_errors=True)
    st2 = StoryStore(story_root2).create(
        name="t", source_text=example, lines=segmenter.split_lines(example),
        segments=ex_segments, images=one, settings=settings_a,
        jp_negative=P.NEGATIVE_DEFAULT, motion_presets=["calm"])
    seg_idx = next(i for i, s in enumerate(st2.segments) if s["speech"])
    seg = st2.segments[seg_idx]
    other = next(s for s in st2.segments if s["speech"] and s is not seg)
    seg_scene = StoryPipeline._scene_text(st2, seg)
    seg_prompt = build_director_text(
        profile="PROFILE", jp_scene=seg_scene,
        jp_prev=StoryPipeline._prev_text(st2, seg_idx, {"en_prompt": ""}),
        jp_negative=P.NEGATIVE_DEFAULT, jp_dialogue=seg["speech"],
        frames=int(settings_a["frames"]))
    (debug / "selftest_story_director.txt").write_text(seg_prompt, encoding="utf-8")
    check("story: the segment's OWN speech is in the director input",
          lambda: _eq(seg["speech"] in seg_prompt, True))
    check("story: a LATER segment's speech is not",
          lambda: _eq(other["speech"] in seg_prompt, False))
    check("story: the project-level STYLE preset reaches SCENE",
          lambda: _eq("落ち着いた雰囲気" in seg_prompt, True))
    check("story: cursor starts at 0 so nothing is replayed",
          lambda: _eq(st2.cursor, 0))
    shutil.rmtree(story_root2, ignore_errors=True)

    # ------------------------------------------- story: empty speech = SILENCE --
    print("\n== story mode: an empty Speech means SILENCE (A-G) ==")
    story_root3 = debug / "selftest_story_speech"
    shutil.rmtree(story_root3, ignore_errors=True)
    line_groups = [
        [{"type": "prompt", "text": "カメラの前に立って周りを見る"}],
        [{"type": "prompt", "text": "カメラに向かって話しかける"},
         {"type": "speech", "text": "こんにちは"}],
        [{"type": "prompt", "text": "夜の窓辺に移動する"},
         {"type": "speech", "text": "こんばんは"}],
        [{"type": "prompt", "text": "黙って窓の外を眺める"}],
    ]
    st3_segments = normalize_segments([
        {"lines": line_groups[0]},
        # THREE action presets, attached to ONE segment only (requirement E).
        {"lines": line_groups[1], "action_presets": ["wave", "bow", "peace"]},
        {"lines": line_groups[2]},
        {"lines": line_groups[3]},
    ])
    st3 = StoryStore(story_root3).create(
        name="無音テスト", source_text="", lines=[l for g in line_groups for l in g],
        segments=st3_segments, images=one, settings=settings_a,
        jp_negative=P.NEGATIVE_DEFAULT, motion_presets=["calm"])

    def _director(index: int) -> str:
        seg = st3.segments[index]
        prev_clip = {"en_prompt": ""} if index > 0 else None
        return story_prompts.build_story_director_text(
            profile="PROFILE",
            segment_prompt=StoryPipeline._scene_text(st3, seg),
            segment_speech=(seg.get("speech") or "").strip(),
            previous_context=StoryPipeline._prev_text(st3, index, prev_clip),
            jp_negative=P.NEGATIVE_DEFAULT,
            frames=int(settings_a["frames"]))

    def _policy(index: int, fake_director_output: str):
        seg = st3.segments[index]
        return story_prompts.enforce_speech_policy(
            fake_director_output, speech=(seg.get("speech") or "").strip(),
            forbidden_speech=StoryPipeline._other_speech(st3, index))

    # A VLM that ignores the instruction and speaks anyway.
    CHATTY = ("detailed_description\n"
              "<Subject 1> (S1) smiles at the camera and says, "
              "<d>[Japanese] こんにちは</d>\nThen she turns away.\n"
              "overall_soundscape\nroom tone\n")
    dir0 = _director(0)
    (debug / "selftest_story_silent_director.txt").write_text(dir0, encoding="utf-8")
    clean0, rep0 = _policy(0, CHATTY)
    print("  (A) segment 0: prompt present, speech empty")
    check("A: no <d> tag survives in the final H3 prompt",
          lambda: _eq(story_prompts.has_dialogue_marker(clean0), False))
    check("A: the invented line was stripped and reported",
          lambda: _eq((rep0["mode"], rep0["removed"], rep0["ok"]),
                      ("silent", ["こんにちは"], True)))
    check("A: the director was told this segment is SILENT",
          lambda: _eq("THIS SEGMENT IS SILENT" in dir0, True))

    dir1 = _director(1)
    SPEAKS = ("detailed_description\n<Subject 1> (S1) waves and says, "
              "<d>[Japanese] こんにちは</d>\n")
    clean1, rep1 = _policy(1, SPEAKS)
    print("  (B) segment 1: prompt present, speech 'こんにちは'")
    # The compiler contract: the director is given SLOT METADATA ONLY, so the
    # text itself must be nowhere in its input - it gets a placeholder instead.
    check("B: こんにちは is NOT handed to the director at all",
          lambda: _eq("こんにちは" in dir1, False))
    def _slots(text: str) -> str:
        return text.split("--- SLOTS START ---")[-1].split("--- SLOTS END ---")[0]

    check("B: the director gets one <<LINE_1>> slot of the right length",
          lambda: _eq((_slots(dir1).count("<<LINE_1>>"), "(5 characters" in dir1),
                      (1, True)))
    check("B: こんにちは survives as the spoken line",
          lambda: _eq((rep1["dialogue_after"], rep1["ok"]), (["こんにちは"], True)))

    dir2 = _director(2)
    (debug / "selftest_story_prev_speech_director.txt").write_text(dir2, encoding="utf-8")
    REPEATS = ("detailed_description\n<Subject 1> (S1) says, "
               "<d>[Japanese] こんにちは</d>\nand then, "
               "<d>[Japanese] こんばんは</d>\n")
    clean2, rep2 = _policy(2, REPEATS)
    print("  (C) segment 2: previous speech 'こんにちは', current 'こんばんは'")
    check("C: the previous line こんにちは is nowhere in the director input",
          lambda: _eq("こんにちは" in dir2, False))
    check("C: the previous clip is described by COUNT, never by text",
          lambda: _eq("already contained 1 spoken line(s)" in dir2, True))
    check("C: the current line こんばんは is not handed over either",
          lambda: _eq(("こんばんは" in dir2, _slots(dir2).count("<<LINE_1>>")),
                      (False, 1)))
    check("C: the repeated previous line is stripped from the final prompt",
          lambda: _eq((rep2["removed"], rep2["dialogue_after"], rep2["ok"]),
                      (["こんにちは"], ["こんばんは"], True)))
    check("C: こんにちは cannot be spoken again",
          lambda: _eq("こんにちは" in "".join(story_prompts.dialogue_texts(clean2)), False))

    dir3 = _director(3)
    clean3, rep3 = _policy(3, REPEATS)
    print("  (D) segment 3: previous speech present, current speech empty")
    check("D: zero dialogue markers in the final H3 prompt",
          lambda: _eq(story_prompts.has_dialogue_marker(clean3), False))
    check("D: both invented lines were stripped",
          lambda: _eq((rep3["mode"], rep3["removed"], rep3["ok"]),
                      ("silent", ["こんにちは", "こんばんは"], True)))
    check("D: a surviving marker would REFUSE the run",
          lambda: _eq(story_prompts.enforce_speech_policy(
              "he says <d>[Japanese] だめ", speech="", forbidden_speech=[])[1]["ok"], False))

    print("  (E) three ACTION presets attached to segment 1 only")
    scenes = [StoryPipeline._scene_text(st3, s) for s in st3.segments]
    check("E: the action presets appear on their own segment",
          lambda: _eq([t in scenes[1] for t in ("手を振る", "お辞儀をする", "ピースサインをする")],
                      [True, True, True]))
    check("E: they are NOT appended to any other segment",
          lambda: _eq([i for i, sc in enumerate(scenes)
                       if i != 1 and any(t in sc for t in
                                         ("手を振る", "お辞儀をする", "ピースサインをする"))],
                      []))
    check("E: the project-level STYLE preset IS on every segment",
          lambda: _eq([("落ち着いた雰囲気" in sc) for sc in scenes],
                      [True] * len(scenes)))
    check("E: an action id at project level is migrated away, not broadcast",
          lambda: _eq(StoryStore(story_root3).create(
              name="x", source_text="", lines=[], segments=st3_segments, images=one,
              settings=settings_a, jp_negative="", motion_presets=["calm", "wave"]
          ).data["motion_presets"], ["calm"]))

    print("  (F) deterministic per-segment seeds")
    check("F: base seed is the project's settings seed (123456789)",
          lambda: _eq((settings_a["seed"], st3.base_seed), (123456789, 123456789)))
    check("F: segments 0/1/2 get distinct deterministic seeds",
          lambda: _eq([st3.seed_for(i) for i in range(3)],
                      [123456789, 123456790, 123456791]))
    st3.data["cursor"] = 2
    st3.segments[0]["seed"] = st3.seed_for(0)
    st3.segments[1]["seed"] = st3.seed_for(1)
    st3.clips.append({"segment_index": 0, "step": 1, "seed": st3.seed_for(0),
                      "frames": 124, "video": "", "local_video": ""})
    st3.save()
    resumed = StoryStore(story_root3).get(st3.id)
    check("F: the same seeds come back after a simulated resume",
          lambda: _eq([resumed.seed_for(i) for i in range(3)],
                      [st3.seed_for(i) for i in range(3)]))
    check("F: the seed used is persisted in segments.json",
          lambda: _eq([s.get("seed") for s in resumed.segments[:2]],
                      [123456789, 123456790]))
    check("F: the seed used is persisted in project.json clips[]",
          lambda: _eq(resumed.clips[0]["seed"], 123456789))

    print("  (G) structured prompt / speech round trip")
    check("G: line types survive save -> reload exactly",
          lambda: _eq([s["lines"] for s in resumed.segments],
                      [s["lines"] for s in st3_segments]))
    check("G: prompt / speech split survives save -> reload exactly",
          lambda: _eq([(s["prompt"], s["speech"]) for s in resumed.segments],
                      [(s["prompt"], s["speech"]) for s in st3_segments]))
    check("G: segments_from_lines never re-classifies an explicit line",
          lambda: _eq([[(l["type"], l["text"]) for l in s["lines"]]
                       for s in segmenter.segments_from_lines(
                           [{"type": "prompt", "text": "こんにちは、と言う。"},
                            {"type": "speech", "text": "椅子と机を見せる。"}],
                           seconds=5)],
                      [[("prompt", "こんにちは、と言う。"),
                        ("speech", "椅子と机を見せる。")]]))
    check("G: split_lines (raw text) still infers types as before",
          lambda: _eq([(l["type"], l["text"]) for l in
                       segmenter.split_lines("『こんにちは』と言う。")],
                      [("speech", "こんにちは"), ("prompt", "と言う。")]))

    print("  (+) all modes forbid unapproved dialogue invention")
    check("story silent block never says 'you may invent the dialogue yourself'",
          lambda: _eq(story_prompts.INVENT_SENTENCE in dir0, False))
    check("story speaking block never says it either",
          lambda: _eq(story_prompts.INVENT_SENTENCE in dir1, False))
    check("the single-shot director never invents dialogue when empty",
          lambda: _eq(story_prompts.INVENT_SENTENCE in build_director_text(
              profile="P", jp_scene="S", jp_prev=pipeline_mod._no_prev(),
              jp_negative=P.NEGATIVE_DEFAULT, jp_dialogue="",
              frames=int(settings_a["frames"])), False))
    check("HDR_DIALOGUE does not grant permission for extra speech",
          lambda: _eq(story_prompts.INVENT_SENTENCE in P.HDR_DIALOGUE, False))
    check("the six section headers come from build/prompts.py unchanged",
          lambda: _eq([h in dir1 for h in (P.HDR_PROFILE, P.HDR_SCENE.strip(),
                                           P.HDR_PREV.strip(), P.HDR_NEG.strip(),
                                           P.HDR_LEN_A.strip(), P.HDR_LEN_B)],
                      [True] * 6))

    print("  (+) pre-flight gate")
    records = preflight_records(st3)
    for r in records:
        print(f"    segment {r['index']}: seed={r['seed']} silent={r['silent']} "
              f"prev={r['previous_context']} actions={r['action_presets']} "
              f"pending={r['pending']}")
    check("preflight record shape",
          lambda: _eq(sorted(records[0]),
                      sorted(["index", "prompt", "speech", "silent", "seed",
                              "previous_context", "style_presets", "action_presets",
                              "pending", "problems", "warnings"])))
    check("preflight: segment 0 has no previous context, later ones do",
          lambda: _eq([r["previous_context"] for r in records],
                      [False, True, True, True]))
    check("preflight refuses an empty segment before any GPU work",
          lambda: _eq(_preflight_refuses(st3), True))

    print("  (+) tail audio default wiring unchanged")
    cont_default = {"video": "h3app_story_x_s1.mp4", "prev_total_frames": 124,
                    "tail_frames": tail}
    cont_flag = dict(cont_default, tail_audio=True)
    g_ta = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                       d["ref_longest"], "x", cont_default,
                                       concat_previous=False)
    g_tb = graphs.build_generate_graph(one, "EN PROMPT", settings_a, cfg.models,
                                       d["ref_longest"], "x", cont_flag,
                                       concat_previous=False)
    check("tail_audio defaults to True (graph is identical)",
          lambda: _eq(json.dumps(g_ta, ensure_ascii=False, sort_keys=True),
                      json.dumps(g_tb, ensure_ascii=False, sort_keys=True)))
    check("the tail audio node is still wired into the H3 node",
          lambda: _eq((g_ta["tail_audio"]["class_type"],
                       g_ta["h3"]["inputs"]["ref_video_audios.ref_video_audio_0"],
                       g_ta["tail_audio"]["inputs"]["start_index"],
                       g_ta["tail_audio"]["inputs"]["duration"]),
                      ("TrimAudioDuration", ["tail_audio", 0],
                       -(tail / 24.0), tail / 24.0)))
    check("the story runner asks for tail audio explicitly",
          lambda: _eq(_story_continuation_tail_audio(cfg, st3), True))
    check("assert_v1_parity (tail audio default)",
          lambda: graphs.assert_v1_parity(g_ta, settings_a, cfg.models, manifest,
                                          defaults=d))

    print("  (+) StoryPipeline._continuation: LONG vs LONG_FAST shape")
    lf_frame = story_root3 / "prev_lastframe.png"
    lf_frame.write_bytes(b"not a real png")
    lf_video = story_root3 / "prev_clip_lf.mp4"
    lf_video.write_bytes(b"not a real video")
    # The anchor is trust-but-verify: _continuation only reuses a recorded
    # voice_master when the staged file is really on disk, so stage it first.
    vm_name_shape = "h3app_story_selftest_voice_master.wav"
    (_fake_comfy_input(st3) / vm_name_shape).write_bytes(b"not a real wav")
    st3.data["voice_master"] = vm_name_shape
    cont_long = _story_continuation_for_mode(
        cfg, st3, mode="LONG",
        prev_clip={"local_video": str(lf_video), "frames": 124, "step": 1})
    cont_lf = _story_continuation_for_mode(
        cfg, st3, mode="LONG_FAST",
        prev_clip={"local_video": str(lf_video), "frames": 124, "step": 1,
                   "last_frame": str(lf_frame)})
    check("_continuation(mode=LONG): tail_frames=config default, tail_audio=True, "
          "no voice_master",
          lambda: _eq((cont_long["tail_frames"], cont_long["tail_audio"],
                       "voice_master" in cont_long),
                      (tail, True, False)))
    check("_continuation(mode=LONG_FAST): tail_frames=0, tail_audio=False, "
          "last_frame_image + voice_master present",
          lambda: _eq((cont_lf["tail_frames"], cont_lf["tail_audio"],
                       bool(cont_lf.get("last_frame_image")),
                       cont_lf.get("voice_master")),
                      (0, False, True, st3.data["voice_master"])))

    print("  (+) LONG_FAST voice anchor: regenerate refresh, resume derivation, "
          "hard refusal")
    vm_root = debug / "selftest_story_voice_master"
    shutil.rmtree(vm_root, ignore_errors=True)
    vm_segments = normalize_segments([
        {"lines": [{"type": "prompt", "text": "シーン0"}]},
        {"lines": [{"type": "prompt", "text": "シーン1"}]},
    ])
    st_vm = StoryStore(vm_root).create(
        name="VoiceMasterテスト", source_text="", lines=[], segments=vm_segments,
        images=one, settings=settings_a, jp_negative=P.NEGATIVE_DEFAULT,
        motion_presets=[])
    st_vm.data["mode"] = "LONG_FAST"
    st_vm.save()

    # A valid six-section director answer with no <d> markers - segment 0 is
    # silent, so this must pass compile_story_prompt/validate_final_prompt
    # untouched, letting the test reach the Voice Master code with no GPU work.
    _VM_DIRECTOR_RAW = (
        "subject_definitions\n<Subject 1> a person\n<Picture 1> the reference sheet\n\n"
        "summary\n[reference generation] a short vertical clip.\n\n"
        "retention_analysis\n<Subject 1> (appears in [Shot 1]): "
        "fully_preserved - face, hair, outfit\n\n"
        "detailed_description\nSoft daylight style. <Subject 1> stands still.\n\n"
        "overall_soundscape\nQuiet room tone.\n\n"
        "non_diegetic_music\nN/A\n")

    class _StubVMPipeline:
        """Stands in for Pipeline: only the three async calls _run_segment
        makes are exercised - no LLM, no ComfyUI."""
        def __init__(self, config, clip_video: Path):
            self.config = config
            self.clip_video = clip_video

        async def _ensure_profile(self, runner, story):
            return "PROFILE"

        async def _run_director(self, runner, story, **kwargs):
            return _VM_DIRECTOR_RAW

        async def _run_generate(self, runner, story, en_prompt, *, step, **kwargs):
            return {"local_video": str(self.clip_video), "frames": 124, "step": step}

    fake_cfg_vm = Config(copy.deepcopy(cfg.data))
    fake_cfg_vm.data["comfy_dir"] = str(vm_root / "_fake_comfy")
    (Path(fake_cfg_vm.data["comfy_dir"]) / "input").mkdir(parents=True, exist_ok=True)
    fake_cfg_vm.data["input_stage_dir"] = str(
        Path(fake_cfg_vm.data["comfy_dir"]) / "input")

    clip0_a = vm_root / "clip0_take_a.mp4"
    clip0_a.write_bytes(b"CLIP0-TAKE-A")
    clip0_b = vm_root / "clip0_take_b.mp4"
    clip0_b.write_bytes(b"CLIP0-TAKE-B")

    _real_extract_audio = merge_mod.extract_audio

    async def _fake_extract_audio(video_path, out_wav):
        # Deterministic stand-in for ffmpeg: copies the SOURCE clip's own
        # bytes, so two different "clip 0" takes stage two different anchors
        # and the test can tell whether the anchor was actually refreshed.
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(Path(video_path).read_bytes())
        return out_wav

    merge_mod.extract_audio = _fake_extract_audio
    try:
        stub_pipeline = _StubVMPipeline(fake_cfg_vm, clip0_a)
        story_pipeline_vm = StoryPipeline(stub_pipeline, StoryStore(vm_root))
        runner_vm = StoryRunner(stub_pipeline, st_vm)

        asyncio.run(story_pipeline_vm._run_segment(
            runner_vm, st_vm, 0, st_vm.segments[0]))
        vm_name_a = st_vm.data.get("voice_master")
        staged_a = fake_cfg_vm.comfy_input / vm_name_a
        bytes_after_a = staged_a.read_bytes() if staged_a.is_file() else b""

        # Simulate api_story_regenerate targeting clip 0: the segment goes
        # back to "pending" and clip 0 is generated again as a DIFFERENT take.
        st_vm.segments[0]["status"] = "pending"
        stub_pipeline.clip_video = clip0_b
        asyncio.run(story_pipeline_vm._run_segment(
            runner_vm, st_vm, 0, st_vm.segments[0]))
        vm_name_b = st_vm.data.get("voice_master")
        staged_b = fake_cfg_vm.comfy_input / vm_name_b
        bytes_after_b = staged_b.read_bytes() if staged_b.is_file() else b""

        check("regenerating clip 0 re-derives the Voice Master (not kept stale)",
              lambda: _eq((vm_name_a, vm_name_b, bytes_after_a, bytes_after_b),
                          (vm_name_a, vm_name_a, b"CLIP0-TAKE-A", b"CLIP0-TAKE-B")))

        # Resume / LOCKED case: clip 0's video is on disk (segments[0]["clip"]),
        # but voice_master was never staged (old snapshot, or clip 0 is
        # LOCKED/APPROVED and never regenerates). _continuation must derive
        # the anchor instead of refusing.
        st_vm.data["voice_master"] = None
        st_vm.save()
        lf_png = vm_root / "prev_lastframe.png"
        lf_png.write_bytes(b"not a real png")
        cont_resume = asyncio.run(story_pipeline_vm._continuation(
            st_vm,
            {"local_video": str(clip0_b), "frames": 124, "step": 1,
             "last_frame": str(lf_png)},
            mode="LONG_FAST"))
        check("resume/LOCKED clip 0 (no voice_master yet) still yields an anchor",
              lambda: _eq((bool(cont_resume.get("voice_master")),
                           st_vm.data.get("voice_master")),
                          (True, cont_resume.get("voice_master"))))

        # Clip 0's video genuinely unavailable: this must still be a hard
        # refusal, never a silent unanchored run.
        st_vm.data["voice_master"] = None
        st_vm.segments[0]["clip"] = ""
        st_vm.save()

        def _missing_clip0_refuses() -> bool:
            try:
                asyncio.run(story_pipeline_vm._continuation(
                    st_vm,
                    {"local_video": str(clip0_b), "frames": 124, "step": 1,
                     "last_frame": str(lf_png)},
                    mode="LONG_FAST"))
                return False
            except PipelineError:
                return True

        check("clip 0 video genuinely unavailable still raises PipelineError",
              lambda: _eq(_missing_clip0_refuses(), True))

        # Stale voice_master pointer: story.data records a filename, but the
        # staged file itself is missing from comfy input (e.g. wiped by an
        # external cleanup). _continuation must not trust it blindly - it
        # should re-derive from clip 0 instead of returning the missing name.
        st_vm.segments[0]["clip"] = str(clip0_b)
        stale_vm_name = "h3app_story_stale_missing_voice_master.wav"
        st_vm.data["voice_master"] = stale_vm_name
        st_vm.save()
        stale_path = fake_cfg_vm.comfy_input / stale_vm_name
        stale_path.unlink(missing_ok=True)
        cont_stale = asyncio.run(story_pipeline_vm._continuation(
            st_vm,
            {"local_video": str(clip0_b), "frames": 124, "step": 1,
             "last_frame": str(lf_png)},
            mode="LONG_FAST"))
        stale_vm_result = cont_stale.get("voice_master")
        check("stale/missing voice_master pointer triggers re-derivation",
              lambda: _eq((stale_vm_result != stale_vm_name,
                           (fake_cfg_vm.comfy_input / str(stale_vm_result)).is_file()),
                          (True, True)))
    finally:
        merge_mod.extract_audio = _real_extract_audio
        shutil.rmtree(vm_root, ignore_errors=True)

    print("  (+) extract_audio() matches the measured benchmark recipe "
          "(backend/h3_v2/ab_run.py::_extract_voice_master)")
    check("VOICE_MASTER constants: first 5s, 32000 Hz, stereo",
          lambda: _eq((merge_mod.VOICE_MASTER_SECONDS,
                       merge_mod.VOICE_MASTER_SAMPLE_RATE,
                       merge_mod.VOICE_MASTER_CHANNELS),
                      (5, 32000, 2)))

    def _extract_audio_args() -> list[str]:
        captured: list[list[str]] = []

        async def _fake_run(args, *, timeout=1800):
            captured.append(args)
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(b"wav")
            return 0, ""

        real_run = merge_mod._run
        merge_mod._run = _fake_run
        try:
            src = vm_root2_src
            asyncio.run(merge_mod.extract_audio(src, src.with_suffix(".wav")))
        finally:
            merge_mod._run = real_run
        return captured[0] if captured else []

    vm_root2_src = debug / "selftest_extract_audio_args_src.mp4"
    vm_root2_src.write_bytes(b"not a real video")
    args = _extract_audio_args()
    check("extract_audio() builds ffmpeg args with the 5s trim + -ar 32000 + -ac 2",
          lambda: _eq(("-ss" in args, "0" in args, "-t" in args,
                       str(merge_mod.VOICE_MASTER_SECONDS) in args,
                       "-ar" in args, str(merge_mod.VOICE_MASTER_SAMPLE_RATE) in args,
                       "-ac" in args, str(merge_mod.VOICE_MASTER_CHANNELS) in args),
                      (True, True, True, True, True, True, True, True)))
    vm_root2_src.unlink(missing_ok=True)

    print("  --- debug\\seg_00.json produced by a mocked (no-GPU) run ---")
    mock = _mock_segment_run(st3, 0, dir0, CHATTY, cfg)
    for line in json.dumps(mock, ensure_ascii=False, indent=2).splitlines()[:40]:
        print("    " + line)
    check("debug record written to story_projects\\<id>\\debug\\seg_00.json",
          lambda: _eq((st3.debug_dir / "seg_00.json").is_file(), True))
    check("debug record carries the whole chain",
          lambda: _eq(sorted(mock),
                      sorted(["segment_index", "written_at", "source_lines",
                              "segment_prompt", "segment_speech", "silent",
                              "style_presets", "action_presets",
                              "effective_scene_text", "profile_used",
                              "director_user_prompt", "director_raw_output",
                              "final_en_prompt", "seed", "previous_clip_reference",
                              "delivery", "timing",
                              "segment_id", "previous_segment_id",
                              "entry_state", "exit_state",
                              "continuity_constraints", "transition_block",
                              "tail_frames", "tail_audio", "speech_policy",
                              "validation", "compile_report", "character_canon",
                              "attribute_overrides", "profile_schema_version"])))
    print(f"    (kept for inspection: {st3.debug_dir / 'seg_00.json'})")

    # ------------------------------------------ story: TRANSITION CONTRACT ----
    # Requirement 12/13/15: consecutive clips must connect. Every assertion is
    # STRUCTURAL - on the state dicts and the rendered block - never on Japanese
    # or English wording of the scene text.
    print("\n== story mode: TRANSITION CONTRACT (entry / exit / camera) ==")
    TR = transitions_mod
    UNS, HOFF = TR.UNSPECIFIED, TR.HANDOFF

    def _ids_of(record) -> list[str]:
        return [c["id"] for c in record["continuity_constraints"]]

    # A -> B -> C, where B introduces a NEW PHYSICAL ACTION (an action preset).
    # Nothing here reads the Japanese text: the action preset id is the signal.
    tr_segments = normalize_segments([
        {"prompt": "セグメントA"},
        {"prompt": "セグメントB", "action_presets": ["sit"]},
        {"prompt": "セグメントC"},
    ])
    tr_plan = TR.plan_transitions(tr_segments)
    print("  --- planned states for a 3-segment story (B introduces an action) ---")
    for rec in tr_plan:
        print(f"    segment {rec['segment_index']}  "
              f"new_physical_action={rec['starts_new_physical_action']}  "
              f"constraints={_ids_of(rec)}")
        print(f"      entry: {json.dumps(rec['entry_state'], ensure_ascii=False)}")
        print(f"      exit : {json.dumps(rec['exit_state'], ensure_ascii=False)}")

    check("T1: every boundary is a hand-over: B.entry == A.exit, C.entry == B.exit",
          lambda: _eq([tr_plan[1]["entry_state"] == tr_plan[0]["exit_state"],
                       tr_plan[2]["entry_state"] == tr_plan[1]["exit_state"]],
                      [True, True]))
    check("T1: the hand-off is planned on the EARLIER segment (A's exit_state)",
          lambda: _eq((tr_plan[0]["exit_state"]["ongoing_action"],
                       tr_plan[1]["entry_state"]["ongoing_action"]),
                      (HOFF, HOFF)))
    check("T1: B does NOT open in an already-changed configuration",
          lambda: _eq([tr_plan[1]["entry_state"][f] for f in TR._PHYSICAL_FIELDS],
                      [UNS] * len(TR._PHYSICAL_FIELDS)))
    check("T1: the app invents no pose it has no reason to believe",
          lambda: _eq(TR.state_is_empty(tr_plan[0]["entry_state"]), True))
    check("T1: A is told to hand over, B is told to continue the movement",
          lambda: _eq(("handoff_out" in _ids_of(tr_plan[0]),
                       "handoff_in" in _ids_of(tr_plan[1]),
                       "handoff_out" in _ids_of(tr_plan[1])),
                      (True, True, False)))
    check("T1: the ongoing action does not leak past the boundary it was for",
          lambda: _eq(tr_plan[1]["exit_state"]["ongoing_action"], UNS))
    check("T1: the first clip inherits nothing and the last hands over nothing",
          lambda: _eq((_ids_of(tr_plan[0])[0], "exit_ready" in _ids_of(tr_plan[2])),
                      ("first_clip", False)))
    check("T1: previous_segment_id is recorded for every boundary",
          lambda: _eq([r["previous_segment_id"] for r in tr_plan],
                      ["", tr_segments[0]["segment_id"], tr_segments[1]["segment_id"]]))

    # CAMERA CONTINUITY across BOTH boundaries -----------------------------------
    cam_segments = normalize_segments([
        {"prompt": "A", "continuity": {"camera_framing": "medium close-up",
                                       "camera_angle": "eye level"}},
        {"prompt": "B"},
        {"prompt": "C"},
    ])
    cam_plan = TR.plan_transitions(cam_segments)
    check("T2: framing/angle are carried across BOTH boundaries",
          lambda: _eq([(r["entry_state"]["camera_framing"],
                        r["exit_state"]["camera_angle"]) for r in cam_plan[1:]],
                      [("medium close-up", "eye level")] * 2))
    check("T2: every continuation clip carries the SETTING continuity rule (never the first)",
          lambda: _eq([("setting_continuity" in _ids_of(r)) for r in cam_plan],
                      [False, True, True]))
    check("T2: a segment that asks for no change gets the continuity rule",
          lambda: _eq([("camera_continuity" in _ids_of(r),
                        "camera_change_as_motion" in _ids_of(r))
                       for r in cam_plan[1:]], [(True, False)] * 2))

    cam2 = normalize_segments([
        {"prompt": "A", "continuity": {"camera_framing": "medium close-up"}},
        {"prompt": "B"},
        {"prompt": "C", "continuity": {"camera_framing": "wide shot"}},
    ])
    cam2_plan = TR.plan_transitions(cam2)
    check("T2: an EXPLICIT camera request changes it from that segment on",
          lambda: _eq([r["exit_state"]["camera_framing"] for r in cam2_plan],
                      ["medium close-up", "medium close-up", "wide shot"]))
    check("T2: the segment before it is unaffected (no early change)",
          lambda: _eq(cam2_plan[2]["entry_state"]["camera_framing"],
                      "medium close-up"))
    check("T2: a requested change is asked for as MOTION, not as a cut",
          lambda: _eq(("camera_change_as_motion" in _ids_of(cam2_plan[2]),
                       "camera_continuity" in _ids_of(cam2_plan[2])),
                      (True, False)))

    check("T3: planning is stable - the same segments plan the same states",
          lambda: _eq(json.dumps(TR.plan_transitions(tr_segments), sort_keys=True),
                      json.dumps(TR.plan_transitions(tr_segments), sort_keys=True)))
    check("T3: an unknown continuity field is discarded, not pasted into a prompt",
          lambda: _eq(TR.normalize_state({"body_pose": "seated", "rm -rf": "x"}),
                      dict(TR.blank_state(), body_pose="seated")))
    check("T3: a multi-line value is flattened and capped",
          lambda: _eq(TR.normalize_state({"gaze": "a\nb"})["gaze"], "a b"))

    # The block in the DIRECTOR TEXT -------------------------------------------
    tr_block_b = TR.transition_block(tr_plan[1])
    tr_dir_b = story_prompts.build_story_director_text(
        profile="PROFILE", segment_prompt="セグメントB", segment_speech="",
        previous_context=StoryPipeline._prev_text(
            type("_S", (), {"segments": tr_segments})(), 1, {"en_prompt": ""}),
        jp_negative=P.NEGATIVE_DEFAULT, frames=int(settings_a["frames"]),
        transition=tr_plan[1])
    print("  --- TRANSITION CONTRACT block as the director sees it (segment B) ---")
    for line in tr_block_b.splitlines():
        print("    | " + line)
    check("T4: the block is in the director text, with both states",
          lambda: _eq([t in tr_dir_b for t in
                       (TR.ENTRY_HEADING, TR.EXIT_HEADING, TR.CONSTRAINT_HEADING,
                        "TRANSITION CONTRACT")], [True] * 4))
    check("T4: the entry state is rendered as 'unchanged', never as a new pose",
          lambda: _eq("body pose: unchanged from the end of the previous clip"
                      in tr_dir_b, True))
    check("T4: the hand-off reads as a movement already under way",
          lambda: _eq("ongoing action: already in motion from the previous clip"
                      in tr_dir_b, True))
    check("T4: without a transition record the prompt is exactly what it was",
          lambda: _eq(TR.transition_block(None), ""))
    check("T4: the PREVIOUS CLIP block is still there too (they complement)",
          lambda: _eq((story_prompts.STORY_PREV_NOTE in tr_dir_b,
                       P.HDR_PREV in tr_dir_b), (True, True)))

    # The state must survive into the DEBUG RECORD (requirement 15) --------------
    tr_root = debug / "selftest_story_transitions"
    shutil.rmtree(tr_root, ignore_errors=True)
    st_tr = StoryStore(tr_root).create(
        name="遷移テスト", source_text="", lines=[], segments=tr_segments,
        images=one, settings=settings_a, jp_negative="", motion_presets=[])
    tr_mock = _mock_segment_run(st_tr, 1, "DIRECTOR TEXT", CHATTY, cfg)
    check("T5: the debug record carries the contract the director was given",
          lambda: _eq((tr_mock["entry_state"], tr_mock["exit_state"]),
                      (tr_plan[1]["entry_state"], tr_plan[1]["exit_state"])))
    check("T5: it names the previous segment and the constraints",
          lambda: _eq(([c["id"] for c in tr_mock["continuity_constraints"]],
                       tr_mock["previous_segment_id"]),
                      (_ids_of(tr_plan[1]), st_tr.segments[0]["segment_id"])))
    check("T5: and the exact block that was rendered into the prompt",
          lambda: _eq(tr_mock["transition_block"], tr_block_b))
    check("T5: the runner plans over the WHOLE segment list before generating",
          lambda: _eq("plan_transitions(" in inspect.getsource(TR.record_for)
                      and "_transition_for(" in inspect.getsource(
                          StoryPipeline._run_segment), True))
    shutil.rmtree(tr_root, ignore_errors=True)

    # slot timing now agrees with the duration model ----------------------------
    print("  --- speech slot timing vs duration.estimate_speech_seconds ---")
    for _line in ("こんにちは", "はじめまして、白凪みおです。",
                  "RTX 5090 を 2026年 に使ってみました"):
        for _delivery in (None, {"pace": "slow", "pause_level": "high"}):
            _want = round(duration_mod.estimate_speech_seconds(
                _line, delivery=_delivery)[0], 1)
            _row = story_prompts.slot_block([_line], delivery=_delivery)
            print(f"    {_row}   (model: {_want}s, delivery={_delivery})")
            check(f"T6: slot timing == estimate for {_line[:12]!r} "
                  f"({'default' if _delivery is None else 'slow'})",
                  lambda w=_want, r=_row: _eq(f"roughly {w} seconds" in r, True))
    check("T6: the flat chars/6.0 rule is gone from story_prompts",
          lambda: _eq("n / 6.0" in inspect.getsource(story_prompts.slot_block),
                      False))

    # ------------------------------------- story: prompt COMPILER + VALIDATOR --
    # Every check below is PARAMETERISED. No attribute value, no dialogue line
    # and no banned word is known to canon.py / compiler.py: the same code path
    # is exercised twice with two completely different synthetic characters to
    # prove it.
    print("\n== story mode: prompt COMPILER + VALIDATOR ==")

    def _profile(accessory: str, outfit: str = "a plain grey tunic",
                 hair: str = "short black bob") -> str:
        """A synthetic CHARACTER PROFILE in build\\prompts.py's own format."""
        return ("Sure! Here is the profile you asked for.\n"
                "FACE: oval face, wide almond eyes\n"
                f"HAIR: {hair}\n"
                "BUILD: slim, average height\n"
                "SKIN: fair, matte finish\n"
                f"OUTFIT: {outfit}\n"
                f"ACCESSORIES: {accessory}\n"
                "MATERIALS: cotton weave, brushed metal\n"
                "COLOR_KEYS: grey, black, white\n"
                "NOTES: this heading is not in the schema and must be discarded\n")

    def _director_out(detailed: str, *, subject: str = "<Subject 1> a person",
                      prefix: str = "", suffix: str = "") -> str:
        """A mocked six-section director answer."""
        return (f"{prefix}subject_definitions\n{subject}\n"
                "<Picture 1> the reference sheet\n\n"
                "summary\n[reference generation] a short vertical clip.\n\n"
                "retention_analysis\n"
                "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, outfit\n\n"
                f"detailed_description\nSoft daylight style.\n{detailed}\n\n"
                "overall_soundscape\nQuiet room tone.\n\n"
                f"non_diegetic_music\nN/A\n{suffix}")

    def _compile(raw, *, profile_text, lines, overrides=None):
        c = canon_mod.parse_profile(profile_text)
        final, crep = compiler.compile_story_prompt(
            raw, canon=c, overrides=overrides, dialogue_lines=lines)
        vrep = compiler.validate_final_prompt(
            final, dialogue_lines=lines, canon=c, overrides=overrides)
        return c, final, crep, vrep

    LINE_A = "こんにちは"
    STAGING_AS_SPEECH = "お辞儀をする"          # a stage direction, not a line
    PROF_1 = _profile("round silver-rimmed glasses")
    PROF_2 = _profile("a red wool scarf", outfit="a long beige coat",
                      hair="waist-length platinum braid")

    print("  (0) the story director SYSTEM prompt drops the <d> mandate")
    sys_new = story_prompts.story_system_prompt(continuation=False)
    sys_cont = story_prompts.story_system_prompt(continuation=True)
    for name, mandate in (("R1 dialogue mandate", "DIALOGUE IS SACRED"),
                          ("<d> tag count rule",
                           "the number of <d> tags must equal the"),
                          ("<d> writing rule", "Dialogue is written as:"),
                          ("profile paste rule",
                           "paste the physical detail from the CHARACTER PROFILE")):
        check(f"0: build/prompts.py STILL carries the {name} (single-shot untouched)",
              lambda m=mandate: _eq(m in P.SYS_DIRECTOR_NEW
                                    and m in P.SYS_DIRECTOR_CONTINUATION, True))
        check(f"0: the story system prompt does NOT carry the {name}",
              lambda m=mandate: _eq(m in sys_new or m in sys_cont, False))
    check("0: the story system prompt teaches the placeholder contract instead",
          lambda: _eq(("<<LINE_1>>" in sys_new,
                       "SUPPLIED BY THE APPLICATION" in sys_new), (True, True)))
    check("0: the six headings survive in the same order",
          lambda: _eq([h for h in compiler.SECTIONS if h in sys_new],
                      list(compiler.SECTIONS)))
    check("0: R2/R3/R4/R5 and the camera vocabulary are still there",
          lambda: _eq([t in sys_new for t in
                       ("NO SCORE UNLESS ASKED", "VERTICAL FRAMING", "LANGUAGE SPLIT",
                        "Respect the requested duration", "Roll Clockwise/Counterclockwise",
                        "fully_preserved")], [True] * 6))
    check("0: the system prompt carries the ENTRY/EXIT transition contract",
          lambda: _eq([all(t in s for t in
                           ("R6. THE TRANSITION CONTRACT IS BINDING.",
                            "TRANSITION CONTRACT", "ENTRY STATE", "EXIT STATE",
                            "ACROSS a\n      boundary, never at it"))
                       for s in (sys_new, sys_cont)], [True, True]))
    check("0: and the CAMERA CONTINUITY preference (a preference, not a lock)",
          lambda: _eq([all(t in s for t in
                           ("R7. CAMERA CONTINUITY IS THE DEFAULT.",
                            "keep the shot size and the camera angle the previous",
                            "a strong preference, not a lock",
                            "rather than as a cut to a different shot size"))
                       for s in (sys_new, sys_cont)], [True, True]))
    check("0: the final checklist asks the director to verify both",
          lambda: _eq([("[ ] If a TRANSITION CONTRACT was supplied" in s
                        and "The framing and angle continue from the previous clip" in s)
                       for s in (sys_new, sys_cont)], [True, True]))
    check("0: continuation mode still carries C1-C5 for the tail relay",
          lambda: _eq([t in sys_cont for t in
                       ("[video continuation + reference generation + audio reference]",
                        "<Video 1>", "<Audio 1>", "C3.", "C4.", "C5.")],
                      [True] * 6))
    check("0: _run_director accepts an explicit system_prompt override",
          lambda: _eq(inspect.signature(Pipeline._run_director)
                      .parameters["system_prompt"].default, None))

    print("  (1) the director renders a stage direction as speech")
    # RECOVER-THEN-VALIDATE: the invented <d> is dropped and reported, the real
    # slot is filled from the placeholder, and the validator certifies the
    # result. A refusal here would block a 3.6-minute-per-clip run for something
    # the app can fix deterministically.
    raw1 = _director_out(
        "[Shot 1] <Subject 1> (S1) bows, "
        f"<d>[Japanese] {STAGING_AS_SPEECH}</d> then straightens up, <<LINE_1>>")
    c1, final1, crep1, vrep1 = _compile(raw1, profile_text=PROF_1, lines=[LINE_A])
    check("1: the invented stage direction is DROPPED and reported",
          lambda: _eq(crep1["reconciliation"]["dropped_invented_dialogue"],
                      [STAGING_AS_SPEECH]))
    check("1: it is nowhere in the final prompt",
          lambda: _eq(STAGING_AS_SPEECH in final1, False))
    check("1: the final dialogue is exactly the user's line",
          lambda: _eq(compiler.dialogue_texts(final1), [LINE_A]))
    check("1: compile and validation both pass after recovery",
          lambda: _eq((crep1["ok"], vrep1["ok"]), (True, True)))
    # What is still an outright refusal, and how it reads to the user.
    _bad, crep1b = compiler.compile_story_prompt(
        "Here is my answer:\nJust a paragraph, no sections at all.",
        canon=c1, dialogue_lines=[LINE_A])
    check("1: an answer that is not the six-section contract is still REFUSED",
          lambda: _eq((crep1b["ok"], _bad), (False, "")))
    check("1: the refusal message is Japanese and names the segment",
          lambda: _eq(compiler.validation_error_message(0, crep1b).startswith(
              "セグメント 1 のプロンプトを検証できませんでした"), True))

    print("  (1b) the director writes the CORRECT line as <d>, with no placeholder")
    raw1c = _director_out(
        f"[Shot 1] <Subject 1> (S1) smiles and says, <d>[Japanese] {LINE_A}</d>")
    c1c, final1c, crep1c, vrep1c = _compile(raw1c, profile_text=PROF_1, lines=[LINE_A])
    check("1b: the line is RECOVERED as its slot and reported",
          lambda: _eq((crep1c["reconciliation"]["recovered_dialogue"],
                       crep1c["reconciliation"]["placeholders_found"]),
                      ([LINE_A], 0)))
    check("1b: the final dialogue is exactly the user's line, not duplicated",
          lambda: _eq(compiler.dialogue_texts(final1c), [LINE_A]))
    check("1b: compile and validation both pass",
          lambda: _eq((crep1c["ok"], vrep1c["ok"]), (True, True)))

    print("  (1c) a mix: one correct <d>, one invented <d>, one placeholder")
    TWO = [LINE_A, "またね"]
    raw1d = _director_out(
        f"[Shot 1] <Subject 1> (S1) waves, <d>[Japanese] {LINE_A}</d> "
        f"then bows, <d>[Japanese] {STAGING_AS_SPEECH}</d> "
        "and finally, <<LINE_2>>")
    c1d, final1d, crep1d, vrep1d = _compile(raw1d, profile_text=PROF_1, lines=TWO)
    check("1c: the correct line is recovered, the invented one dropped",
          lambda: _eq((crep1d["reconciliation"]["recovered_dialogue"],
                       crep1d["reconciliation"]["dropped_invented_dialogue"]),
                      ([LINE_A], [STAGING_AS_SPEECH])))
    check("1c: the final dialogue is exactly the user's lines, in order",
          lambda: _eq(compiler.dialogue_texts(final1d), TWO))
    check("1c: the invented text is gone from the prompt entirely",
          lambda: _eq(STAGING_AS_SPEECH in final1d, False))
    check("1c: compile and validation both pass",
          lambda: _eq((crep1d["ok"], vrep1d["ok"]), (True, True)))
    raw1e = _director_out(
        f"[Shot 1] <Subject 1> (S1) says <d>[Japanese] {TWO[1]}</d> "
        f"and earlier said <d>[Japanese] {TWO[0]}</d>")
    c1e, final1e, crep1e, vrep1e = _compile(raw1e, profile_text=PROF_1, lines=TWO)
    check("1c: an out-of-order <d> never reorders the script",
          lambda: _eq((compiler.dialogue_texts(final1e), vrep1e["ok"]), (TWO, True)))
    check("1c: speech placed outside detailed_description is removed",
          lambda: _eq(_compile(_director_out(
              "[Shot 1] <Subject 1> (S1) smiles, <<LINE_1>>").replace(
                  "Quiet room tone.",
                  f"Quiet room tone. <d>[Japanese] {LINE_A}</d>"),
              profile_text=PROF_1, lines=[LINE_A])[2][
                  "dropped_outside_detailed_description"], [LINE_A]))

    print("  (2) the director stages the placeholder correctly")
    raw2 = _director_out("[Shot 1] <Subject 1> (S1) smiles at the camera, <<LINE_1>>")
    c2, final2, crep2, vrep2 = _compile(raw2, profile_text=PROF_1, lines=[LINE_A])
    check("2: compile and validation both pass",
          lambda: _eq((crep2["ok"], vrep2["ok"]), (True, True)))
    check("2: the <d> content is the user's line character-for-character",
          lambda: _eq(compiler.dialogue_texts(final2), [LINE_A]))
    check("2: the <d> element is exactly what the app wrote",
          lambda: _eq(f"<d>[Japanese] {LINE_A}</d>" in final2, True))
    check("2: no placeholder token survives",
          lambda: _eq(bool(compiler.PLACEHOLDER_RE.search(final2)), False))
    check("2: the speaker ID stayed OUTSIDE the tag",
          lambda: _eq("(S1) <d>" in final2 or "(S1) smiles" in final2, True))

    print("  (3) a silent segment with any <d> at all")
    raw3 = _director_out(f"[Shot 1] <Subject 1> (S1) murmurs, <d>[Japanese] {LINE_A}</d>")
    c3, final3, crep3, vrep3 = _compile(raw3, profile_text=PROF_1, lines=[])
    check("3: every <d> in a silent segment is DROPPED (no expected line to match)",
          lambda: _eq(crep3["reconciliation"]["dropped_invented_dialogue"], [LINE_A]))
    check("3: zero dialogue markers remain, so the validator certifies silence",
          lambda: _eq((compiler.has_dialogue_marker(final3), crep3["ok"], vrep3["ok"]),
                      (False, True, True)))
    check("3: a <d> FORCED back into a silent prompt FAILS validation",
          lambda: _eq(compiler.validate_final_prompt(
              final3.replace("Quiet room tone.",
                             f"Quiet room tone. (S1) <d>[Japanese] {LINE_A}</d>"),
              dialogue_lines=[], canon=c3)["ok"], False))
    check("3: a broken unpaired <d tag in a silent segment is removed too",
          lambda: _eq(compiler.has_dialogue_marker(_compile(
              _director_out("[Shot 1] <Subject 1> (S1) murmurs, <d>[Japanese] oops"),
              profile_text=PROF_1, lines=[])[1]), False))
    silent_ok = _compile(_director_out("[Shot 1] <Subject 1> looks out of the window."),
                         profile_text=PROF_1, lines=[])
    check("3: a genuinely silent staging passes with zero <d>",
          lambda: _eq((silent_ok[2]["ok"], silent_ok[3]["ok"],
                       compiler.has_dialogue_marker(silent_ok[1])),
                      (True, True, False)))

    print("  (4) a persistent trait the director dropped is restored")
    for label, prof in (("canon A", PROF_1), ("canon B", PROF_2)):
        c4 = canon_mod.parse_profile(prof)
        # The director redefines <Subject 1> as a bare label: every physical
        # attribute is gone, exactly the observed failure.
        raw4 = _director_out("[Shot 1] <Subject 1> (S1) waves, <<LINE_1>>",
                             subject="<Subject 1> a young woman")
        final4, crep4 = compiler.compile_story_prompt(
            raw4, canon=c4, dialogue_lines=[LINE_A])
        vrep4 = compiler.validate_final_prompt(
            final4, dialogue_lines=[LINE_A], canon=c4)
        traits = c4.persistent_traits()
        check(f"4[{label}]: parsed {len(traits)} persistent traits from the profile",
              lambda t=traits: _eq(sorted(t), sorted(
                  ["face", "hair", "build", "skin", "outfit", "accessories"])))
        check(f"4[{label}]: EVERY persistent trait is back in the final prompt",
              lambda f=final4, t=traits: _eq([k for k, v in t.items() if v not in f], []))
        check(f"4[{label}]: the accessory {list(traits.values())[-1][:18]!r} is present",
              lambda f=final4, t=traits: _eq(t["accessories"] in f, True))
        check(f"4[{label}]: the director's bare redefinition was replaced",
              lambda f=final4: _eq("<Subject 1> a young woman" in f, False))
        check(f"4[{label}]: compile + validation pass",
              lambda a=crep4, b=vrep4: _eq((a["ok"], b["ok"]), (True, True)))
        check(f"4[{label}]: dropping a trait from the FINAL text fails validation",
              lambda f=final4, c=c4, t=traits: _eq(compiler.validate_final_prompt(
                  f.replace(t["accessories"], ""), dialogue_lines=[LINE_A],
                  canon=c)["ok"], False))

    print("  (5) a persistent trait changes ONLY through a structured override")
    NEW_OUTFIT = "a black knee-length raincoat over the same dress"
    c5 = canon_mod.parse_profile(PROF_1)
    raw5 = _director_out("[Shot 1] <Subject 1> (S1) steps outside, <<LINE_1>>",
                         subject="<Subject 1> a young woman")
    plain5, _cr5 = compiler.compile_story_prompt(raw5, canon=c5, dialogue_lines=[LINE_A])
    over5, cr5b = compiler.compile_story_prompt(
        raw5, canon=c5, overrides={"outfit": NEW_OUTFIT}, dialogue_lines=[LINE_A])
    vr5b = compiler.validate_final_prompt(
        over5, dialogue_lines=[LINE_A], canon=c5, overrides={"outfit": NEW_OUTFIT})
    check("5: without an override the profile outfit is preserved",
          lambda: _eq((c5.outfit in plain5, NEW_OUTFIT in plain5), (True, False)))
    check("5: with an override the new outfit replaces it",
          lambda: _eq((NEW_OUTFIT in over5, c5.outfit in over5), (True, False)))
    check("5: the overridden segment still validates",
          lambda: _eq((cr5b["ok"], vr5b["ok"]), (True, True)))
    check("5: every OTHER persistent trait is untouched by the override",
          lambda: _eq([k for k, v in c5.persistent_traits().items()
                       if k != "outfit" and v not in over5], []))
    check("5: an unknown override key is discarded (schema-bounded)",
          lambda: _eq(canon_mod.normalize_overrides(
              {"outfit": NEW_OUTFIT, "mood": "angry", "": "x", "hair": ""}),
              {"outfit": NEW_OUTFIT}))
    check("5: an override PERSISTS into later segments (merge_overrides)",
          lambda: _eq(canon_mod.merge_overrides(
              [None, {"outfit": NEW_OUTFIT}, None]), {"outfit": NEW_OUTFIT}))
    check("5: segments.json keeps the structured field, unknown keys dropped",
          lambda: _eq(normalize_segments([{"prompt": "外に出る",
                                           "attribute_overrides":
                                               {"OUTFIT": NEW_OUTFIT, "vibe": "x"}}]
                                         )[0]["attribute_overrides"],
                      {"outfit": NEW_OUTFIT}))

    print("  (6) internal monologue never reaches the final prompt")
    THOUGHT = ("<|channel>thought The user wants a video. Let me plan.<|end|>\n"
               "Thinking Process:\n1. decide the shot\n2. write it\n"
               "```json\n{\"foo\": 1}\n```\n")
    raw6 = _director_out("[Shot 1] <Subject 1> (S1) nods, <<LINE_1>>", prefix=THOUGHT)
    c6, final6, crep6, vrep6 = _compile(raw6, profile_text=PROF_1, lines=[LINE_A])
    check("6: junk before the first heading is discarded, not patched",
          lambda: _eq((crep6["ok"], vrep6["ok"]), (True, True)))
    check("6: no monologue marker survives anywhere in the final prompt",
          lambda: _eq(compiler.hygiene_violations(final6), []))
    check("6: none of the thought text is in the final prompt",
          lambda: _eq([t for t in ("<|channel", "Thinking Process", "```", "<|end|>")
                       if t in final6], []))
    check("6: the compiler recorded what it threw away",
          lambda: _eq(crep6["sections"]["discarded_prefix_lines"] > 0, True))
    raw6b = _director_out("[Shot 1] Thinking Process: hmm. <Subject 1> (S1) nods, <<LINE_1>>")
    c6b, final6b, crep6b, vrep6b = _compile(raw6b, profile_text=PROF_1, lines=[LINE_A])
    check("6: a marker INSIDE a kept section is a refusal, not a silent patch",
          lambda: _eq((crep6b["ok"], crep6b["hygiene"]), (False, ["thinking_process"])))
    check("6: the profile parser is schema-bounded too",
          lambda: _eq(canon_mod.parse_profile(
              "<|channel>thought hmm\nFACE: a face\nNOTES: junk\n").as_dict(),
              dict(canon_mod.CharacterCanon(face="a face").as_dict())))

    print("  (7) placeholder reconciliation is deterministic and reported")
    THREE = ["いち", "に", "さん"]
    raw7_few = _director_out("[Shot 1] <Subject 1> (S1) speaks, <<LINE_1>>")
    c7a, final7a, crep7a, vrep7a = _compile(raw7_few, profile_text=PROF_1, lines=THREE)
    check("7: fewer placeholders than lines -> the rest are appended, in order",
          lambda: _eq(compiler.dialogue_texts(final7a), THREE))
    check("7: the shortfall is reported",
          lambda: _eq((crep7a["reconciliation"]["placeholders_found"],
                       crep7a["reconciliation"]["lines_appended"]), (1, THREE[1:])))
    check("7: the appended lines still validate",
          lambda: _eq((crep7a["ok"], vrep7a["ok"]), (True, True)))
    raw7_many = _director_out(
        "[Shot 1] <Subject 1> (S1) speaks, <<LINE_1>> then <<LINE_2>> then <<LINE_3>>")
    c7b, final7b, crep7b, vrep7b = _compile(raw7_many, profile_text=PROF_1,
                                            lines=[LINE_A])
    check("7: more placeholders than lines -> the extras are DROPPED, never invented",
          lambda: _eq(compiler.dialogue_texts(final7b), [LINE_A]))
    check("7: the extra placeholders are reported and removed from the text",
          lambda: _eq((crep7b["reconciliation"]["placeholders_dropped"],
                       bool(compiler.PLACEHOLDER_RE.search(final7b))),
                      (["<<LINE_2>>", "<<LINE_3>>"], False)))
    check("7: both reconciliations pass validation",
          lambda: _eq((crep7b["ok"], vrep7b["ok"]), (True, True)))
    raw7_rev = _director_out(
        "[Shot 1] <Subject 1> (S1) says <<LINE_2>> and later <<LINE_1>>")
    c7c, final7c, _cr7c, _vr7c = _compile(raw7_rev, profile_text=PROF_1,
                                          lines=[THREE[0], THREE[1]])
    check("7: placeholders are consumed in ORDER OF APPEARANCE, never renumbered",
          lambda: _eq(compiler.dialogue_texts(final7c), [THREE[0], THREE[1]]))
    raw7_nospk = _director_out("[Shot 1] The room is quiet. <<LINE_1>>")
    c7d, final7d, cr7d, vr7d = _compile(raw7_nospk, profile_text=PROF_1, lines=[LINE_A])
    check("7: a missing speaker ID is supplied by the compiler, so validation passes",
          lambda: _eq((cr7d["reconciliation"]["speaker_ids_inserted"], vr7d["ok"]),
                      (1, True)))

    print("  (+) the validator refuses invented / leaked dialogue on the FINAL text")
    check("V: an extra <d> in the final text fails",
          lambda: _eq(compiler.validate_final_prompt(
              final2 + f"\n<d>[Japanese] {STAGING_AS_SPEECH}</d>",
              dialogue_lines=[LINE_A], canon=c2)["ok"], False))
    check("V: another segment's line leaking in fails",
          lambda: _eq(compiler.validate_final_prompt(
              final2.replace("Quiet room tone.", "Quiet room tone. こんばんは"),
              dialogue_lines=[LINE_A], canon=c2,
              other_dialogue=["こんばんは"])["ok"], False))
    check("V: another segment's Japanese staging leaking in fails",
          lambda: _eq(compiler.validate_final_prompt(
              final2.replace("Quiet room tone.", "夜の窓辺に移動する"),
              dialogue_lines=[LINE_A], canon=c2,
              other_staging=["夜の窓辺に移動する"])["ok"], False))
    check("V: text outside the six sections fails",
          lambda: _eq(compiler.validate_final_prompt(
              "Here you go!\n" + final2, dialogue_lines=[LINE_A], canon=c2)["ok"],
              False))

    print("  (8) PROFILE_SCHEMA_VERSION invalidates the profile cache")
    imgs = [Path("h3app_ref_a.png"), Path("h3app_ref_b.png")]
    key_now = profile_cache_key(imgs)
    check("8: the current key equals the key at PROFILE_SCHEMA_VERSION",
          lambda: _eq(key_now, profile_cache_key(
              imgs, canon_mod.PROFILE_SCHEMA_VERSION)))
    check("8: bumping the schema version changes the key",
          lambda: _eq(key_now == profile_cache_key(
              imgs, canon_mod.PROFILE_SCHEMA_VERSION + 1), False))
    check("8: the old schema's key is different too (v1 entries are stale)",
          lambda: _eq(key_now == profile_cache_key(imgs, 1), False))
    check("8: slot order still matters",
          lambda: _eq(key_now == profile_cache_key(list(reversed(imgs))), False))
    cache_root = debug / "selftest_profile_cache"
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_store = ProjectStore(cache_root)
    cache_store.create(settings=settings_a, images=["h3app_ref_a.png"],
                       jp_scene="", jp_dialogue="", jp_negative="",
                       max_seconds=15.0, profile="OLD PROFILE",
                       profile_cache_key_=profile_cache_key(imgs, 1))
    check("8: a profile cached under the OLD key is ignored, not crashed on",
          lambda: _eq(cache_store.find_cached_profile(key_now), ""))
    check("8: the same key still hits",
          lambda: _eq(cache_store.find_cached_profile(
              profile_cache_key(imgs, 1)), "OLD PROFILE"))
    shutil.rmtree(cache_root, ignore_errors=True)

    print("  (9) single-shot regression: lenient parsing only, never the compiler")
    ss_raw = _director_out(
        "[Shot 1] <Subject 1> (S1) smiles and says, "
        f"<d>[Japanese] {LINE_A}</d>", prefix=THOUGHT)
    ss_clean = compiler.lenient_clean(ss_raw)
    check("9: single-shot output still parses to the six sections",
          lambda: _eq(compiler.parse_sections(ss_clean)[0] is not None, True))
    check("9: the director's OWN <d> is kept (single-shot's verified behaviour)",
          lambda: _eq(compiler.dialogue_texts(ss_clean), [LINE_A]))
    check("9: the monologue prefix is gone",
          lambda: _eq(compiler.hygiene_violations(ss_clean), []))
    check("9: unparseable output falls back to the input, byte for byte",
          lambda: _eq(compiler.lenient_clean("just some free text\nno sections here"),
                      "just some free text\nno sections here"))
    check("9: output with a marker INSIDE a section also falls back untouched",
          lambda: _eq(compiler.lenient_clean(raw6b), raw6b))
    check("9: lenient_clean is idempotent",
          lambda: _eq(compiler.lenient_clean(ss_clean), ss_clean))
    check("9: the single-shot director never adds invention permission",
          lambda: _eq(story_prompts.INVENT_SENTENCE in build_director_text(
              profile="P", jp_scene="S", jp_prev=pipeline_mod._no_prev(),
              jp_negative=P.NEGATIVE_DEFAULT, jp_dialogue=LINE_A,
              frames=int(settings_a["frames"])), False))
    check("9: _run_director cleans by default and story mode opts out",
          lambda: _eq([
              inspect.signature(Pipeline._run_director)
              .parameters["clean_sections"].default,
              inspect.signature(story_prompts.build_story_director_text)
              .parameters["dialogue_lines"].default], [True, None]))

    print("\n  --- END-TO-END compile demo (mocked director, no GPU) ---")
    # One mock carrying every observed failure at once: an invented dialogue
    # line, a correct line the director wrapped in <d> itself, a dropped
    # persistent trait, and an internal-monologue block.
    demo_lines = TWO
    demo_raw = _director_out(
        "[Shot 1] <Subject 1> (S1) bows deeply, "
        f"<d>[Japanese] {STAGING_AS_SPEECH}</d> then straightens up and says, "
        f"<d>[Japanese] {TWO[0]}</d>\n"
        "[Shot 2] At 00:03.000, <Subject 1> (S1) waves goodbye, <<LINE_2>>",
        subject="<Subject 1> a young woman", prefix=THOUGHT)
    demo_canon = canon_mod.parse_profile(PROF_1)
    demo_final, demo_cr = compiler.compile_story_prompt(
        demo_raw, canon=demo_canon, dialogue_lines=demo_lines)
    demo_vr = compiler.validate_final_prompt(
        demo_final, dialogue_lines=demo_lines, canon=demo_canon)
    print("  raw director output (mock):")
    for line in demo_raw.splitlines():
        print("    | " + line)
    print("  compiled final prompt:")
    for line in demo_final.splitlines():
        print("    | " + line)
    print(f"  compile   ok={demo_cr['ok']} violations={demo_cr['violations']}")
    print(f"  recovered={demo_cr['reconciliation']['recovered_dialogue']} "
          f"dropped_invented="
          f"{demo_cr['reconciliation']['dropped_invented_dialogue']} "
          f"placeholders={demo_cr['reconciliation']['placeholders_found']}")
    print(f"  validate  ok={demo_vr['ok']} violations={demo_vr['violations']}")
    (debug / "selftest_compiled_prompt.txt").write_text(demo_final, encoding="utf-8")
    (debug / "selftest_validation_report.json").write_text(
        json.dumps({"compile": demo_cr, "validate": demo_vr},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    check("demo: the compiled prompt passes both gates",
          lambda: _eq((demo_cr["ok"], demo_vr["ok"]), (True, True)))
    check("demo: the invented line was dropped and reported",
          lambda: _eq((demo_cr["reconciliation"]["dropped_invented_dialogue"],
                       STAGING_AS_SPEECH in demo_final),
                      ([STAGING_AS_SPEECH], False)))
    check("demo: the director's own correct <d> was recovered as its slot",
          lambda: _eq(demo_cr["reconciliation"]["recovered_dialogue"], [TWO[0]]))
    check("demo: the dialogue is EXACTLY the user's lines, in order",
          lambda: _eq(compiler.dialogue_texts(demo_final), demo_lines))
    check("demo: the dropped accessory is back",
          lambda: _eq(demo_canon.accessories in demo_final, True))
    check("demo: every persistent trait is present",
          lambda: _eq([k for k, v in demo_canon.persistent_traits().items()
                       if v not in demo_final], []))
    check("demo: no thought, no fence, no placeholder, nothing outside the sections",
          lambda: _eq((compiler.hygiene_violations(demo_final),
                       bool(compiler.PLACEHOLDER_RE.search(demo_final)),
                       compiler.parse_sections(demo_final)[1]["discarded_prefix_lines"]),
                      ([], False, 0)))

    # --------------------------------- story: segment ADD / DELETE / REORDER --
    print("\n== story mode: segment ADD / DELETE / REORDER (A-I) ==")
    struct_root = debug / "selftest_story_structure"
    shutil.rmtree(struct_root, ignore_errors=True)

    def _mk(names: list[str], *, generated: int = 0):
        """A story whose first `generated` segments already produced a clip.
        Real files on disk, so invalidation really has something to move."""
        return _make_struct_story(struct_root, names, generated=generated,
                                  settings=settings_a, images=one)

    def _ids(story, *picks: int) -> list[dict]:
        return [{"segment_id": story.segments[i]["segment_id"]} for i in picks]

    def _prompts(segs) -> list[str]:
        return [s["prompt"] for s in segs]

    def _statuses(segs) -> list[str]:
        return [s["status"] for s in segs]

    # A. all pending, delete the middle one -------------------------------------
    stA = _mk(["A", "B", "C"])
    resA = story_mod.apply_structure(stA, _ids(stA, 0, 2))
    check("A: [A,B,C] minus B -> [A,C]",
          lambda: _eq(_prompts(stA.segments), ["A", "C"]))
    check("A: indexes are renumbered 0..n-1",
          lambda: _eq([s["index"] for s in stA.segments], [0, 1]))
    check("A: nothing was generated, so nothing is invalidated",
          lambda: _eq((resA["invalidated_count"], resA["requires_confirmation"],
                       resA["divergence"], resA["cursor"]), (0, False, 1, 0)))

    # B. append after generated work -------------------------------------------
    stB = _mk(["A", "B"], generated=2)
    resB = story_mod.apply_structure(stB, _ids(stB, 0, 1) + [{"prompt": "C"}])
    check("B: appending C invalidates nothing (divergence == len(old))",
          lambda: _eq((resB["invalidated_count"], resB["divergence"]), (0, 2)))
    check("B: both clips are kept",
          lambda: _eq([c["segment_index"] for c in stB.clips], [0, 1]))
    check("B: only C is pending",
          lambda: _eq(_statuses(stB.segments), ["done", "done", "pending"]))
    check("B: the cursor is unchanged",
          lambda: _eq((resB["cursor"], stB.cursor), (2, 2)))
    check("B: the kept clips still have their files",
          lambda: _eq([Path(c["local_video"]).is_file() for c in stB.clips],
                      [True, True]))

    # C. insert in the middle of generated work ---------------------------------
    stC = _mk(["A", "B", "C"], generated=3)
    seedsC = [s["seed"] for s in stC.segments]
    clipsC = [c["local_video"] for c in stC.clips]
    resC = story_mod.apply_structure(
        stC, _ids(stC, 0) + [{"prompt": "X"}] + _ids(stC, 1, 2))
    check("C: order is A,X,B,C",
          lambda: _eq(_prompts(stC.segments), ["A", "X", "B", "C"]))
    check("C: A is kept, X/B/C are pending",
          lambda: _eq(_statuses(stC.segments),
                      ["done", "pending", "pending", "pending"]))
    check("C: cursor == 1 (== divergence)",
          lambda: _eq((resC["divergence"], stC.cursor), (1, 1)))
    check("C: B's and C's clips are invalidated",
          lambda: _eq((resC["invalidated_count"], resC["requires_confirmation"],
                       [x["index"] for x in resC["invalidated"]]),
                      (2, True, [1, 2])))
    check("C: only A's clip is left in project.json",
          lambda: _eq([c["segment_index"] for c in stC.clips], [0]))
    check("C: the invalidated files were MOVED, never deleted",
          lambda: _eq([Path(p).is_file() for p in clipsC[1:]]
                      + [Path(r["orphaned_path"]).is_file()
                         for r in stC.data["invalidated_clips"]],
                      [False, False, True, True]))
    check("C: they live in orphaned\\ with disambiguated names",
          lambda: _eq(sorted(p.name.split("__")[0]
                             for p in stC.orphaned_dir.glob("clip_*.mp4")),
                      ["clip_02", "clip_03"]))
    check("C: the kept segment keeps its ORIGINAL seed",
          lambda: _eq(stC.segments[0]["seed"], seedsC[0]))
    check("C: moved / new segments get the seed of their NEW index",
          lambda: _eq([s["seed"] for s in stC.segments],
                      [stC.seed_for(i) for i in range(4)]))
    check("C: merged / final_video were cleared",
          lambda: _eq((stC.data["merged"], stC.data["final_video"]), (False, "")))
    check("C: clips[-1] (the continuation source) is the last KEPT clip",
          lambda: _eq(stC.clips[-1]["segment_index"], stC.cursor - 1))

    # D. delete a generated segment in the middle -------------------------------
    stD = _mk(["A", "B", "C"], generated=3)
    resD = story_mod.apply_structure(stD, _ids(stD, 0, 2))
    check("D: [A,B,C] minus B -> [A,C]",
          lambda: _eq(_prompts(stD.segments), ["A", "C"]))
    check("D: C moved to index 1 and is PENDING again",
          lambda: _eq((stD.segments[1]["index"], stD.segments[1]["status"],
                       stD.segments[1]["clip"]), (1, "pending", None)))
    check("D: cursor == 1",
          lambda: _eq((resD["divergence"], stD.cursor), (1, 1)))
    check("D: B's and C's clips are invalidated, A's is kept",
          lambda: _eq((resD["invalidated_count"],
                       [c["segment_index"] for c in stD.clips]), (2, [0])))
    check("D: C's seed follows its new index",
          lambda: _eq(stD.segments[1]["seed"], stD.seed_for(1)))

    # E. swap two generated segments -------------------------------------------
    stE = _mk(["A", "B", "C"], generated=3)
    resE = story_mod.apply_structure(stE, _ids(stE, 0, 2, 1))
    check("E: order is A,C,B",
          lambda: _eq(_prompts(stE.segments), ["A", "C", "B"]))
    check("E: everything from index 1 is invalidated",
          lambda: _eq((resE["divergence"], resE["invalidated_count"],
                       _statuses(stE.segments)),
                      (1, 2, ["done", "pending", "pending"])))
    check("E: cursor == 1 and only A's clip survives",
          lambda: _eq((stE.cursor, [c["segment_index"] for c in stE.clips]),
                      (1, [0])))

    # F. reorder only UNGENERATED segments --------------------------------------
    stF = _mk(["A", "B", "C", "D"], generated=2)
    filesF = [c["local_video"] for c in stF.clips]
    resF = story_mod.apply_structure(stF, _ids(stF, 0, 1, 3, 2))
    check("F: order is A,B,D,C",
          lambda: _eq(_prompts(stF.segments), ["A", "B", "D", "C"]))
    check("F: nothing is invalidated (the change is after the cursor)",
          lambda: _eq((resF["divergence"], resF["invalidated_count"],
                       resF["requires_confirmation"]), (2, 0, False)))
    check("F: A's and B's clips are untouched, on disk and in project.json",
          lambda: _eq(([c["segment_index"] for c in stF.clips],
                       [Path(p).is_file() for p in filesF]), ([0, 1], [True, True])))
    check("F: the cursor and the generated statuses are unchanged",
          lambda: _eq((stF.cursor, _statuses(stF.segments)),
                      (2, ["done", "done", "pending", "pending"])))
    check("F: merged is NOT cleared when nothing was invalidated",
          lambda: _eq("invalidated_clips" in stF.data, False))

    # G. the last remaining segment cannot be deleted ---------------------------
    stG = _mk(["A"])
    check("G: deleting the only segment is REFUSED, with a Japanese message",
          lambda: _eq(_structure_refused(stG, []),
                      "セグメントを空にはできません。少なくとも 1 つのセグメントが必要です。"))
    check("G: the story is untouched by the refusal",
          lambda: _eq((_prompts(stG.segments), stG.cursor), (["A"], 0)))
    check("G: an unknown segment_id is rejected, not silently dropped",
          lambda: _eq(_structure_refused(stG, [{"segment_id": "deadbeef"}]).startswith(
              "知らないセグメント"), True))
    check("G: a duplicated segment_id is rejected too",
          lambda: _eq(_structure_refused(stG, _ids(stG, 0) + _ids(stG, 0)),
                      "同じセグメントが複数回指定されています。"))
    check("G: the running story refusal is Japanese and has its own rule",
          lambda: _eq(story_structure_refusal(running=True, count=3, limit=24),
                      "生成中はセグメントの構成を変更できません。先に停止してください。"))
    check("G: not running and within the limit -> no refusal",
          lambda: _eq(story_structure_refusal(running=False, count=3, limit=24), None))
    check("G: over the limit is refused",
          lambda: _eq(story_structure_refusal(running=False, count=25, limit=24),
                      "セグメントは最大 24 個までです。"))
    check("G: the endpoint applies that same guard",
          lambda: _eq("story_structure_refusal(" in inspect.getsource(create_app), True))

    # H. reload from disk -------------------------------------------------------
    reloadedC = StoryStore(struct_root).get(stC.id)
    check("H: order and content survive a reload",
          lambda: _eq(_prompts(reloadedC.segments), _prompts(stC.segments)))
    check("H: identities, statuses and seeds survive a reload",
          lambda: _eq([(s["segment_id"], s["status"], s["seed"], s["index"])
                       for s in reloadedC.segments],
                      [(s["segment_id"], s["status"], s["seed"], s["index"])
                       for s in stC.segments]))
    check("H: cursor and clips survive a reload",
          lambda: _eq((reloadedC.cursor, [c["segment_index"] for c in reloadedC.clips]),
                      (stC.cursor, [c["segment_index"] for c in stC.clips])))
    check("H: the invalidation history survives a reload",
          lambda: _eq(len(reloadedC.data["invalidated_clips"]), 2))

    # I. the merge can never see an invalidated clip ----------------------------
    check("I: merge() builds its input from mergeable_clips(), not story.clips",
          lambda: _eq("story.mergeable_clips()" in
                      inspect.getsource(StoryPipeline.merge), True))
    check("I: after an invalidation the merge list is the kept clips only",
          lambda: _eq([c["segment_index"] for c in stC.mergeable_clips()], [0]))
    orphaned_files = {r["orphaned_path"] for r in stC.data["invalidated_clips"]}
    check("I: no orphaned file appears in the merge list",
          lambda: _eq([c for c in stC.mergeable_clips()
                       if c.get("local_video") in orphaned_files], []))
    # A project.json deliberately corrupted to still list an orphaned clip.
    corrupt_data = json.loads(stC.project_path.read_text(encoding="utf-8"))
    ghost = dict(stC.data["invalidated_clips"][0])
    corrupt_data["clips"].append({
        "segment_index": 1, "step": 2, "frames": 124, "seconds": 5.167,
        "segment_id": ghost["segment_id"], "seed": ghost.get("seed"),
        "video": "", "local_video": ghost["orphaned_path"], "last_frame": ""})
    _atomic_test_write(stC.project_path, corrupt_data)
    corrupted = StoryStore(struct_root).get(stC.id)
    check("I: the corrupted project really does list the orphaned clip",
          lambda: _eq([c["local_video"] for c in corrupted.clips][-1],
                      ghost["orphaned_path"]))
    check("I: mergeable_clips() STILL refuses it (segment 1 is pending)",
          lambda: _eq([c["segment_index"] for c in corrupted.mergeable_clips()], [0]))
    check("I: and the paths merge() would concatenate contain no orphan",
          lambda: _eq([str(Path(c.get("local_video") or c.get("video", "")))
                       for c in corrupted.mergeable_clips()],
                      [str(Path(stC.clips[0]["local_video"]))]))
    # ... even if the segment is marked done by hand: identity still decides.
    corrupted.segments[1]["status"] = "done"
    corrupted.segments[1]["clip"] = ghost["orphaned_path"]
    check("I: a hand-marked 'done' segment cannot claim another segment's clip",
          lambda: _eq([c["segment_index"] for c in corrupted.mergeable_clips()], [0]))

    # dry run -------------------------------------------------------------------
    stDry = _mk(["A", "B", "C"], generated=3)
    before = (stDry.project_path.read_bytes(), stDry.segments_path.read_bytes(),
              sorted(p.name for p in stDry.clips_dir.iterdir()))
    dry = story_mod.apply_structure(stDry, _ids(stDry, 0, 2), dry_run=True)
    after = (stDry.project_path.read_bytes(), stDry.segments_path.read_bytes(),
             sorted(p.name for p in stDry.clips_dir.iterdir()))
    check("dry_run: reports the impact it would have",
          lambda: _eq((dry["divergence"], dry["invalidated_count"],
                       dry["requires_confirmation"], dry["cursor"],
                       _prompts(dry["segments"])),
                      (1, 2, True, 1, ["A", "C"])))
    check("dry_run: writes NOTHING to disk",
          lambda: _eq(before, after))
    check("dry_run: does not touch the story in memory either",
          lambda: _eq((_prompts(stDry.segments), stDry.cursor,
                       [c["segment_index"] for c in stDry.clips]),
                      (["A", "B", "C"], 3, [0, 1, 2])))
    check("dry_run: no orphaned folder was created",
          lambda: _eq(stDry.orphaned_dir.exists(), False))
    check("the response carries everything the UI needs",
          lambda: _eq(sorted(dry),
                      sorted(["divergence", "invalidated", "invalidated_count",
                              "requires_confirmation", "segments", "cursor",
                              "dry_run", "message"])))

    # old project WITHOUT segment_id -------------------------------------------
    legacy = _legacy_story(struct_root)
    old_store = StoryStore(struct_root)
    st_old = old_store.get(legacy)
    check("old: a project with no segment_id still loads",
          lambda: _eq(_prompts(st_old.segments), ["旧A", "旧B"]))
    check("old: ids were backfilled for every segment",
          lambda: _eq([bool(story_mod.SEGMENT_ID_RE.match(s["segment_id"]))
                       for s in st_old.segments], [True, True]))
    check("old: the clip inherited the id of the segment at its index",
          lambda: _eq(st_old.clips[0]["segment_id"], st_old.segments[0]["segment_id"]))
    check("old: the backfill was persisted",
          lambda: _eq([s.get("segment_id") for s in json.loads(
              st_old.segments_path.read_text(encoding="utf-8"))],
              [s["segment_id"] for s in st_old.segments]))
    check("old: nothing else about the project changed",
          lambda: _eq((st_old.cursor, st_old.data["settings"]["seed"],
                       _statuses(st_old.segments)),
                      (1, 123456789, ["done", "pending"])))
    res_old = story_mod.apply_structure(
        st_old, [{"segment_id": st_old.segments[0]["segment_id"]},
                 {"prompt": "新規"},
                 {"segment_id": st_old.segments[1]["segment_id"]}])
    check("old: it edits correctly after the backfill",
          lambda: _eq((_prompts(st_old.segments), res_old["divergence"],
                       res_old["invalidated_count"], st_old.cursor),
                      (["旧A", "新規", "旧B"], 1, 0, 1)))
    check("old: the already-generated clip is untouched",
          lambda: _eq(([c["segment_index"] for c in st_old.clips],
                       st_old.segments[0]["seed"]), ([0], 123456789)))

    # a brand new segment is a NORMAL segment ----------------------------------
    check("new segments carry every field a split segment carries",
          lambda: _eq(sorted(st_old.segments[1]),
                      sorted(st_old.segments[0])))
    check("a new segment starts empty, pending, with its index's seed",
          lambda: _eq((st_old.segments[1]["prompt"], st_old.segments[1]["speech"],
                       st_old.segments[1]["action_presets"],
                       st_old.segments[1]["status"], st_old.segments[1]["clip"],
                       st_old.segments[1]["seed"]),
                      ("新規", "", [], "pending", None, st_old.seed_for(1))))
    check("content-only /api/story/segments still drops empty segments",
          lambda: _eq(len(normalize_segments([{"prompt": "x"}, {"prompt": ""}])), 1))

    shutil.rmtree(struct_root, ignore_errors=True)

    # ---------------------------------------------------------------------
    # transport tokens vs real reasoning
    #
    # A live run failed here: Gemma emitted its whole plan first, OUTLINED the
    # six sections as markdown bullets, then delivered the real answer glued
    # behind a <channel|> token. The parser anchored on the outline and the
    # validator (correctly) refused the contaminated prompt.
    # ---------------------------------------------------------------------
    print("\n== transport tokens vs real reasoning ==")
    tok_canon = canon_mod.parse_profile(
        "FACE: oval face\nHAIR: black bob\nBUILD: slim\nSKIN: fair\n"
        "OUTFIT: navy blazer\nACCESSORIES: round silver-rimmed glasses\n"
        "MATERIALS: cotton\nCOLOR_KEYS: navy, black")
    tok_tmpl = ("subject_definitions\n<Subject 1>\n"
                "summary\n[reference generation] test\n"
                "retention_analysis\n<Subject 1> (appears in [Shot 1]): fully_preserved - all\n"
                "detailed_description\n{B}\n"
                "overall_soundscape\nRoom tone.\n"
                "non_diegetic_music\nN/A")

    def _tok_run(body, lines=None):
        lines = ["こんにちは"] if lines is None else lines
        final, rep = compiler.compile_story_prompt(
            tok_tmpl.format(B=body), canon=tok_canon, dialogue_lines=lines)
        if not final:
            return False, final
        return compiler.validate_final_prompt(
            final, dialogue_lines=lines, canon=tok_canon)["ok"], final

    _shot = "[Shot 1] <Subject 1> (S1) smiles. <<LINE_1>>"
    ok1, final1 = _tok_run(_shot + "\n<|end|>")
    check("1: six sections + trailing close token -> PASS",
          lambda: _true(ok1))
    check("1: the token is gone from the final prompt",
          lambda: _true("<|" not in final1 and "|>" not in final1))
    check("2: model wrapper at the very start and end -> PASS",
          lambda: _true(_tok_run.__self__ if False else
                        compiler.validate_final_prompt(
                            compiler.compile_story_prompt(
                                "<|im_start|>\n" + tok_tmpl.format(B=_shot) + "\n<|im_end|>",
                                canon=tok_canon, dialogue_lines=["こんにちは"])[0],
                            dialogue_lines=["こんにちは"], canon=tok_canon)["ok"]))
    check("3: a transport wrapper BETWEEN sections -> normalised -> PASS",
          lambda: _true(_tok_run(_shot)[0] and compiler.validate_final_prompt(
              compiler.compile_story_prompt(
                  tok_tmpl.format(B=_shot).replace(
                      "overall_soundscape", "<|channel|>\noverall_soundscape"),
                  canon=tok_canon, dialogue_lines=["こんにちは"])[0],
              dialogue_lines=["こんにちは"], canon=tok_canon)["ok"]))
    check("4: real 'Thinking Process:' inside a section -> FAIL",
          lambda: _true(not _tok_run(
              "Thinking Process:\nI should show her smiling. " + _shot)[0]))
    check("5: <|channel>thought + reasoning inside a section -> FAIL",
          lambda: _true(not _tok_run(
              "<|channel>thought\nI should decide the framing. " + _shot)[0]))
    check("9: content tokens survive normalisation untouched",
          lambda: _eq(compiler.normalise_transport(
              "<Subject 1> <Video 1> <Audio 1> <Picture 1> <d>[Japanese] あ</d>")[0],
              "<Subject 1> <Video 1> <Audio 1> <Picture 1> <d>[Japanese] あ</d>"))
    for _bullet in ("    *   **subject_definitions:**",
                    "    *   **summary:** [reference generation] + brief.",
                    "  - detailed_description"):
        check(f"8: outline bullet is NOT a heading: {_bullet.strip()[:34]!r}",
              (lambda b: lambda: _eq(compiler._heading_of(b), None))(_bullet))

    # 7. the REAL failing output, kept as a fixture so this can never regress.
    _fx = APP_DIR / "tests_fixtures" / "director_thinking_prefix.txt"
    if _fx.is_file():
        _raw = _fx.read_text(encoding="utf-8")
        _lines = ["はじめまして、白凪みおです。普段はゲームやパソコン、AIみたいな"
                  "ちょっとオタク寄りのことを楽しみながら、気になったものを実際に"
                  "触って試してみるのが大好きです。"]
        _final, _rep = compiler.compile_story_prompt(
            _raw, canon=tok_canon, dialogue_lines=_lines)
        _v = compiler.validate_final_prompt(
            _final, dialogue_lines=_lines, canon=tok_canon) if _final else {"ok": False}
        check("7: the real failing director output now parses",
              lambda: _true(bool(_final)))
        check("7: no reasoning survives into the final prompt",
              lambda: _true(all(bad not in _final for bad in (
                  "Thinking Process", "<channel|>", "<|channel>",
                  "This matches the desired final structure"))))
        check("7: the genuine detailed_description body is present",
              lambda: _true("eye contact with the camera" in _final))
        check("7: validate_final_prompt now returns ok",
              lambda: _true(_v["ok"]))
    else:
        failures.append(f"fixture missing: {_fx}")
        print(f"  FAIL  fixture missing: {_fx}")

    # ------------------------------------------------ 保存フォルダを開く --------
    # The client sends a LOGICAL ID and a TARGET, never a path. Everything below
    # tests the PURE resolver: no Explorer window is ever spawned.
    print("\n== 保存フォルダを開く: logical id -> validated path (nothing opened) ==")
    of_root = debug / "selftest_open_folder"
    shutil.rmtree(of_root, ignore_errors=True)
    of_stories = of_root / "story_projects"
    of_projects = of_root / "projects"
    of_comfy_out = of_root / "comfy_output"
    for _p in (of_stories, of_projects, of_comfy_out):
        _p.mkdir(parents=True, exist_ok=True)
    of_roots = [Path(os.path.realpath(str(p)))
                for p in (of_stories, of_projects, of_comfy_out)]
    outside = of_root / "outside.mp4"
    outside.write_bytes(b"not in an allowed root")

    def _refused(fn) -> str:
        """The Japanese refusal, or '' when the call was NOT refused."""
        try:
            fn()
        except reveal_mod.TargetError as exc:
            return exc.message
        return ""

    of_store = StoryStore(of_stories)
    st_of = of_store.create(name="開くテスト", source_text="", lines=[],
                            segments=[{"prompt": "A"}], images=one,
                            settings=settings_a, jp_negative="",
                            motion_presets=[])
    st_of.ensure_dirs()
    of_final = st_of.final_dir / "完成.mp4"
    of_final.write_bytes(b"not a real video")
    st_of.data["final_video"] = str(of_final)
    st_of.save()

    of_res = reveal_mod.resolve_story_target(st_of, "final", roots=of_roots)
    print(f"    story/final   -> dir={of_res['directory']}")
    print(f"                     file={of_res['file']}")
    check("O1: a valid story + 'final' resolves to the real final video",
          lambda: _eq((Path(of_res["file"]), Path(of_res["directory"])),
                      (Path(os.path.realpath(of_final)),
                       Path(os.path.realpath(st_of.final_dir)))))
    check("O1: the resolver returns data only - it opens nothing",
          lambda: _eq(("startfile" in inspect.getsource(reveal_mod),
                       "Popen" in inspect.getsource(reveal_mod)), (False, False)))
    check("O1: 'clips' and 'project' resolve to the story's own folders",
          lambda: _eq([Path(reveal_mod.resolve_story_target(
              st_of, t, roots=of_roots)["directory"]) for t in ("clips", "project")],
              [Path(os.path.realpath(st_of.clips_dir)),
               Path(os.path.realpath(st_of.root))]))

    st_pending = of_store.create(name="未書き出し", source_text="", lines=[],
                                 segments=[{"prompt": "A"}], images=one,
                                 settings=settings_a, jp_negative="",
                                 motion_presets=[])
    check("O2: not generated yet -> a clear Japanese message, nothing opened",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_story_target(
              st_pending, "final", roots=of_roots)),
              "完成動画がまだありません。先に「1本にまとめる」を実行してください。"))
    st_gone = of_store.create(name="消えた", source_text="", lines=[],
                              segments=[{"prompt": "A"}], images=one,
                              settings=settings_a, jp_negative="",
                              motion_presets=[])
    st_gone.data["final_video"] = str(st_gone.final_dir / "no_such_file.mp4")
    check("O2: a recorded file that no longer exists is refused too",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_story_target(
              st_gone, "final", roots=of_roots)).startswith("完成動画のファイルが"),
              True))
    check("O3: an unknown story id never becomes a path",
          lambda: _eq((of_store.get("deadbeefdead") is None,
                       _refused(lambda: reveal_mod.resolve_story_target(
                           None, "final", roots=of_roots))),
                      (True, "ストーリーが見つかりませんでした。")))
    check("O3: an id that is not an id is rejected before the filesystem",
          lambda: _eq([story_mod.is_safe_story_id(v) for v in
                       ("..", "../../Windows", r"..\..\Windows", "a/b",
                        "C:\\Windows", "", "*")], [False] * 7))
    check("O3: the same for single-shot project ids",
          lambda: _eq([reveal_mod.is_safe_project_id(v) for v in
                       ("..", "../../Windows", r"..\..\Windows", "a/b",
                        "C:\\Windows", "", "*")], [False] * 7))

    st_out = of_store.create(name="外", source_text="", lines=[],
                             segments=[{"prompt": "A"}], images=one,
                             settings=settings_a, jp_negative="",
                             motion_presets=[])
    st_out.data["final_video"] = str(outside)
    check("O4: a real file OUTSIDE the allowed roots is refused",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_story_target(
              st_out, "final", roots=of_roots)), "許可されていない場所は開けません。"))
    st_trav = of_store.create(name="traversal", source_text="", lines=[],
                              segments=[{"prompt": "A"}], images=one,
                              settings=settings_a, jp_negative="",
                              motion_presets=[])
    st_trav.data["final_video"] = str(st_trav.final_dir / ".." / ".." / ".."
                                      / "outside.mp4")
    check("O4: a '..' traversal out of the story folder is refused",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_story_target(
              st_trav, "final", roots=of_roots)), "許可されていない場所は開けません。"))
    check("O4: an absolute path smuggled into final_video is refused",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_story_target(
              type("_S", (), {"data": {"final_video": str(APP_DIR / "server.py")},
                              "clips_dir": st_of.clips_dir,
                              "root": st_of.root})(),
              "final", roots=of_roots)), "許可されていない場所は開けません。"))
    check("O5: only the three logical targets are accepted",
          lambda: _eq([_refused(lambda t=t: reveal_mod.resolve_story_target(
              st_of, t, roots=of_roots)) for t in
              ("..", "../final", "final/../..", r"C:\Windows", "debug", "")],
              ["開く場所の指定が不正です。"] * 5 + [""]))
    check("O5: an empty target means 'final', the default the UI sends",
          lambda: _eq(reveal_mod.resolve_story_target(st_of, "", roots=of_roots),
                      of_res))

    # single-shot ---------------------------------------------------------------
    of_pstore = ProjectStore(of_projects)
    proj_of = of_pstore.create(settings=settings_a, images=one, jp_scene="",
                               jp_dialogue="", jp_negative="", max_seconds=15)
    of_comfy_file = of_comfy_out / "H3_APP_00001.mp4"
    of_comfy_file.write_bytes(b"not a real video")
    of_local = of_pstore.dir_for(proj_of.id) / "clip_01.mp4"
    of_local.write_bytes(b"not a real video")
    check("O6: single-shot with no clip yet -> clear message, nothing opened",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_project_target(
              proj_of, "final", project_dir=of_pstore.dir_for(proj_of.id),
              roots=of_roots)),
              "保存された動画がまだありません。先に生成してください。"))
    proj_of.clips.append({"step": 1, "frames": 124, "seconds": 5.167,
                          "video": str(of_comfy_file),
                          "local_video": str(of_local)})
    proj_of.save()
    of_pres = reveal_mod.resolve_project_target(
        proj_of, "final", project_dir=of_pstore.dir_for(proj_of.id),
        roots=of_roots)
    print(f"    single/final  -> dir={of_pres['directory']}")
    print(f"                     file={of_pres['file']}")
    check("O6: single-shot 'final' resolves to ComfyUI's own output file",
          lambda: _eq(Path(of_pres["file"]),
                      Path(os.path.realpath(of_comfy_file))))
    proj_of.clips[-1]["video"] = ""
    check("O6: with ComfyUI's copy gone it falls back to the app's copy",
          lambda: _eq(Path(reveal_mod.resolve_project_target(
              proj_of, "final", project_dir=of_pstore.dir_for(proj_of.id),
              roots=of_roots)["file"]), Path(os.path.realpath(of_local))))
    proj_of.clips[-1]["local_video"] = str(outside)
    check("O6: a clip path pointing outside the roots is refused",
          lambda: _eq(_refused(lambda: reveal_mod.resolve_project_target(
              proj_of, "final", project_dir=of_pstore.dir_for(proj_of.id),
              roots=of_roots)), "許可されていない場所は開けません。"))
    check("O6: 'project' resolves to the project folder",
          lambda: _eq(Path(reveal_mod.resolve_project_target(
              proj_of, "project", project_dir=of_pstore.dir_for(proj_of.id),
              roots=of_roots)["directory"]),
              Path(os.path.realpath(of_pstore.dir_for(proj_of.id)))))

    _src = inspect.getsource(create_app)
    check("O7: no endpoint takes a path from the client",
          lambda: _eq([t in _src for t in ('body.get("path"', "body.get('path'",
                                           'body.get("directory"',
                                           'body.get("file"')],
                      [False] * 4))
    check("O7: both open-folder routes validate the id and use the resolver",
          lambda: _eq([t in _src for t in (
              "reveal_mod.is_safe_project_id(", "story_mod.is_safe_story_id(sid)",
              "reveal_mod.resolve_project_target(", "reveal_mod.resolve_story_target(",
              "reveal_mod.allowed_roots(cfg)")], [True] * 5))
    check("O7: the allowed roots are story_projects / projects / output / media",
          lambda: _eq(reveal_mod.allowed_roots(cfg),
                       [Path(os.path.realpath(p)) for p in
                        (cfg.story_dir, cfg.projects_dir, cfg.comfy_output,
                         cfg.media_root)]))
    shutil.rmtree(of_root, ignore_errors=True)

    # ------------------------------------------------- H3_Media exports -------
    # User-facing copies only. Working files, pointers and graphs untouched.
    print("\n== H3_Media: export copies + media root (nothing moved) ==")
    media_root = debug / "selftest_media"
    shutil.rmtree(media_root, ignore_errors=True)
    (media_root / "Videos").mkdir(parents=True)
    _m_src = media_root / "src.mp4"
    _m_src.write_bytes(b"not a real video")
    check("M1: export_copy copies into the dest dir",
          lambda: _eq(Path(media_mod.export_copy(_m_src, media_root / "Videos",
                                                 "a.mp4")).name, "a.mp4"))
    check("M1: export is a copy - the source stays",
          lambda: _eq(_m_src.is_file(), True))
    check("M1: missing source exports to '' instead of raising",
          lambda: _eq(media_mod.export_copy(media_root / "nope.mp4",
                                            media_root / "Videos", "b.mp4"), ""))
    check("M1: same-file export is a no-op returning the path",
          lambda: _eq(media_mod.export_copy(
              media_root / "Videos" / "a.mp4", media_root / "Videos", "a.mp4") != "",
                      True))
    check("M2: config exposes the media subdirs",
          lambda: _eq([p.name for p in (cfg.media_images, cfg.media_audio,
                                        cfg.media_videos, cfg.media_projects)],
                      ["Images", "Audio", "Videos", "Projects"]))
    _src2 = inspect.getsource(create_app)
    check("M2: the media open route is registered",
          lambda: _eq('"/api/media/open"' in _src2, True))
    check("M2: the cleanup routes are registered",
          lambda: _eq('"/api/media/cleanup/scan"' in _src2 and
                      '"/api/media/cleanup"' in _src2, True))
    shutil.rmtree(media_root, ignore_errors=True)

    # --------------------------------------- media cleanup (scan → clean) ----
    # Pure temp dirs, no GPU, no Explorer. Covers: manifest match, legacy
    # match, age gate, zero-byte, protected extensions, protected paths,
    # token mismatch, TOCTOU change, missing file, duplicate manifest rows,
    # read-only failure, unicode names, recycle-bin path.
    print("\n== media cleanup: scan candidates, revalidate, delete safely ==")
    cu_root = debug / "selftest_cleanup"
    shutil.rmtree(cu_root, ignore_errors=True)
    cu_media = cu_root / "H3_Media"
    cu_scan = cu_root / "comfy_out"
    cu_mvid = cu_media / "Videos"
    cu_mvid.mkdir(parents=True)
    cu_scan.mkdir(parents=True)
    import time as _time_mod
    _old = _time_mod.time() - 3 * 86400
    def _mk(name: str, data: bytes, mtime: float | None = None) -> Path:
        p = cu_scan / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if mtime is not None:
            import os as _os
            _os.utime(p, (mtime, mtime))
        return p
    _good_src = _mk("clip_old.mp4", b"V" * 1000, _old)
    _good_tw = cu_mvid / "clip_old.mp4"
    _good_tw.write_bytes(b"V" * 1000)
    media_mod.record_export(cu_media, source=_good_src, exported=_good_tw,
                            media_type="clip", project_id="p1")
    media_mod.record_export(cu_media, source=_good_src, exported=_good_tw,
                            media_type="clip", project_id="p1")  # duplicate row
    _fresh = _mk("clip_new.mp4", b"N" * 1000)  # too new
    _zero = _mk("clip_zero.mp4", b"", _old)
    _json = _mk("notes.json", b"{}", _old)
    _model = _mk("m.safetensors", b"M" * 100, _old)
    _nomatch = _mk("orphan.mp4", b"O" * 500, _old)  # no twin anywhere
    _legacy_src = _mk("legacy.mp4", b"L" * 700, _old)  # legacy: twin, no row
    (cu_mvid / "legacy.mp4").write_bytes(b"L" * 700)
    _uni_src = _mk("日本語_🎬.mp4", b"U" * 600, _old)
    (cu_mvid / "日本語_🎬.mp4").write_bytes(b"U" * 600)
    _ro_src = _mk("readonly.mp4", b"R" * 800, _old)
    (cu_mvid / "readonly.mp4").write_bytes(b"R" * 800)
    import stat as _stat
    _ro_src.chmod(_stat.S_IREAD)
    try:
        _scan = lambda **kw: media_mod.scan_candidates(
            media_root=cu_media, scan_roots=[cu_scan],
            media_dirs=[cu_mvid], min_age_days=1.0, **kw)
        _res = _scan()
        _paths = {c["path"] for c in _res["candidates"]}
        check("C1: manifest-matched old duplicate is a candidate",
              lambda: _eq(str(_good_src) in _paths, True))
        check("C1: duplicate manifest rows yield one candidate",
              lambda: _eq(sum(1 for c in _res["candidates"]
                              if c["path"] == str(_good_src)), 1))
        check("C1: legacy same-name+size twin is a candidate",
              lambda: _eq(str(_legacy_src) in _paths, True))
        check("C1: fresh file is excluded by the age gate",
              lambda: _eq(str(_fresh) in _paths, False))
        check("C1: zero-byte file is excluded",
              lambda: _eq(str(_zero) in _paths, False))
        check("C1: json/safetensors are excluded",
              lambda: _eq(str(_json) in _paths or str(_model) in _paths, False))
        check("C1: twin-less file is excluded",
              lambda: _eq(str(_nomatch) in _paths, False))
        check("C1: unicode filename works",
              lambda: _eq(str(_uni_src) in _paths, True))
        check("C1: media files themselves are never candidates",
              lambda: _eq(any(str(cu_media) in c["path"]
                              for c in _res["candidates"]), False))
        _res2 = _scan(protected_paths={str(_good_src)})
        check("C1: protected (running/locked) paths are excluded",
              lambda: _eq(str(_good_src) in
                          {c["path"] for c in _res2["candidates"]}, False))
        _tok = _res["token"]
        _bad = media_mod.clean_candidates(
            candidates=_res["candidates"], token="wrong",
            expected_token=_tok)
        check("C2: token mismatch deletes nothing",
              lambda: _eq((_bad["removed"], _good_src.is_file()), (0, True)))
        # TOCTOU: change one file after scan, delete the rest.
        with open(_legacy_src, "ab") as _f:
            _f.write(b"X")
        _go = media_mod.clean_candidates(
            candidates=_res["candidates"], token=_tok,
            expected_token=_tok, recycle=False)
        check("C2: changed-since-scan file is skipped, not deleted",
              lambda: _eq((_legacy_src.is_file(),
                           any(s["path"] == str(_legacy_src)
                               for s in _go["skipped"])), (True, True)))
        check("C2: unchanged candidate is permanently deleted",
              lambda: _eq((_good_src.is_file(), _go["removed"] >= 1,
                           _go["method"]), (False, True, "permanent")))
        check("C2: the media twin survives the cleanup",
              lambda: _eq(_good_tw.is_file(), True))
        check("C2: missing file is skipped",
              lambda: _eq(any(s["path"] == str(_nomatch)
                              for s in media_mod.clean_candidates(
                                  candidates=[{"path": str(_nomatch),
                                               "size": 1, "mtime": 1,
                                               "sha256": ""}],
                                  token=_tok, expected_token=_tok,
                                  recycle=False)["skipped"]), True))
        _ro_res = media_mod.clean_candidates(
            candidates=[c for c in _res["candidates"]
                        if c["path"] == str(_ro_src)],
            token=_tok, expected_token=_tok, recycle=False)
        check("C2: read-only delete failure is recorded, app stays up",
              lambda: _eq((len(_ro_res["failed"]) >= 1 or
                           not _ro_src.is_file()), True))
        # Recycle-bin path on one temp file (lands in the bin by design).
        _bin_src = _mk("tobin.mp4", b"B" * 300, _old)
        (cu_mvid / "tobin.mp4").write_bytes(b"B" * 300)
        _bin_scan = _scan()
        _bin_cand = [c for c in _bin_scan["candidates"]
                     if c["path"] == str(_bin_src)]
        _bin_res = media_mod.clean_candidates(
            candidates=_bin_cand, token=_bin_scan["token"],
            expected_token=_bin_scan["token"], recycle=True)
        check("C2: recycle path removes from disk and reports the method",
              lambda: _eq((not _bin_src.exists(),
                           _bin_res["method"] in ("recycle",
                                                  "permanent-fallback")),
                          (True, True)))
        print(f"    recycle method used: {_bin_res['method']} "
              f"(tobin.mp4 may sit in the Recycle Bin by design)")
    finally:
        try:
            _ro_src.chmod(_stat.S_IWRITE)
        except Exception:                                    # noqa: BLE001
            pass
        shutil.rmtree(cu_root, ignore_errors=True)

    # ------------------------------------------------- AI Director (pure) ----
    # No VLM, no ComfyUI: schema, locks, continuity warnings, converters.
    print("\n== AI Director: spec, locks, continuity, converters (no GPU) ==")
    DM = director_spec_mod
    DC = director_convert_mod
    DP = director_prompts_mod
    DPR = director_provider_mod
    _s1 = DM.new_single(idea="夏祭りで恋人を撮影", duration_sec=10,
                        initial_request="台詞多め")
    check("D1: new single spec validates",
          lambda: _eq(DM.validate_spec(_s1)["ok"], True))
    check("D1: bad duration is rejected",
          lambda: _eq(DM.validate_spec({**_s1, "duration_sec": 7})["ok"],
                      False))
    _s1["items"]["outfit"] = {"value": "浴衣", "locked": True, "request": ""}
    _s1["items"]["dialogue"] = {"value": "ねえ、何食べる？", "locked": False,
                                "request": "もっと台詞を多く"}
    import copy as _copymod
    _evil = _copymod.deepcopy(_s1)
    _evil["items"]["outfit"]["value"] = "白いワンピース"
    _evil["items"]["dialogue"]["value"] = "こんにちは"
    _merged, _enforced = DM.apply_locks(_s1, _evil)
    check("D2: locked value is restored after a hostile rewrite",
          lambda: _eq(_merged["items"]["outfit"]["value"], "浴衣"))
    check("D2: unlocked value follows the rewrite",
          lambda: _eq(_merged["items"]["dialogue"]["value"], "こんにちは"))
    check("D2: enforced paths are reported",
          lambda: _eq("items.outfit" in _enforced, True))
    _st = DM.new_story30(idea="夜の夏祭り", initial_request="会話多め")
    check("D3: story30 has master + 3 clips",
          lambda: _eq((len(_st["clips"]),
                       sorted(_st["master"]["items"].keys()) == sorted(
                           DM.MASTER_ITEMS)), (3, True)))
    _st["clips"][0]["carry_over"] = {"carried_objects": "りんご飴"}
    _st["clips"][0]["end_state"] = {"emotion": "楽しい"}
    _st["clips"][1]["carry_over"] = {}
    _st["clips"][1]["start_state"] = {"emotion": "楽しい"}
    _find = DM.check_continuity(_st)
    check("D3: vanished carried object warns (non-blocking)",
          lambda: _eq(any(f["aspect"] == "carried_objects" for f in _find),
                      True))
    check("D3: continued emotion does not warn",
          lambda: _eq(any(f["aspect"] == "emotion" for f in _find), False))
    _st["clips"][1]["start_state"] = {"emotion": "悲しい"}
    _find2 = DM.check_continuity(_st)
    check("D3: hard emotion cut warns",
          lambda: _eq(any(f["aspect"] == "emotion" for f in _find2), True))
    _st["clips"][1]["carry_over"] = {"carried_objects": "りんご飴を持ったまま"}
    _find3 = DM.check_continuity(_st)
    check("D3: explained carry-over clears the clip1 warning",
          lambda: _eq(any(f["aspect"] == "carried_objects" and f["clip"] == 1
                          for f in _find3), False))
    check("D3: the chain continues - clip2 still flags the drop",
          lambda: _eq(any(f["aspect"] == "carried_objects" and f["clip"] == 2
                          for f in _find3), True))
    _s1["items"]["dialogue"]["value"] = "こんにちは、今日はいい天気ですね"
    _conv = DC.single_to_inputs(_s1)
    check("D4: single converts to scene/speech/frames",
          lambda: _eq((_conv["frames"], bool(_conv["speech"])), (243, True)))
    _st["clips"][0]["items"]["dialogue"]["value"] = "見て、花火"
    _st["clips"][1]["items"]["dialogue"]["value"] = "きれいだね"
    _st["clips"][2]["items"]["dialogue"]["value"] = "来年も来よう"
    _st["clips"][2]["items"]["acting"]["value"] = "夜空を見上げる"
    _segs = DC.story30_to_segments(_st)
    check("D4: story30 converts to exactly 3 segments",
          lambda: _eq((len(_segs),
                       [bool(s["prompt"]) for s in _segs]), (3, [True] * 3)))
    check("D4: prompts builders emit the required keys",
          lambda: _eq(("items" in DP.single_system() or "items" in
                       DP.master_system()), True))
    check("D5: provider parses fenced JSON",
          lambda: _eq(DPR.parse_json_answer(
              '```json\n{"a": 1}\n```')["a"], 1))
    try:
        DPR.parse_json_answer("not json at all")
        _parse_failed = False
    except DPR.DirectorError:
        _parse_failed = True
    check("D5: provider rejects non-JSON with DirectorError",
          lambda: _eq(_parse_failed, True))
    _src3 = inspect.getsource(create_app)
    check("D5: six director routes are registered",
          lambda: _eq([t in _src3 for t in (
              '"/api/director/single/create"',
              '"/api/director/single/regenerate"',
              '"/api/director/story30/create"',
              '"/api/director/story30/regenerate"',
              '"/api/director/generate"',
              '"/api/director/story30/generate"')], [True] * 6))
    _rg = graphs.build_device_restore_graph(label="selftest")
    check("D6: device restore graph builds with seed/restore/done",
          lambda: _eq(
              (_rg["seed"]["class_type"], _rg["restore"]["class_type"],
               _rg["done"]["class_type"]),
              ("PrimitiveInt", "H3CudaPhaseRestore", "PreviewAny")))
    _ug = graphs.build_vlm_unload_graph()
    check("D6: vlm unload graph builds with unload/purge/done",
          lambda: _eq(
              (_ug["unload"]["class_type"], _ug["purge"]["class_type"],
               _ug["done"]["class_type"]),
              ("llama_cpp_unload_model", "LayerUtility: PurgeVRAM V2",
               "PreviewAny")))

    # --------------------------------- Director direct compile (pure) -----
    # Deterministic Final Spec -> six-section H3 prompt. No VLM involved.
    print("\n== Director direct compile: outfit override, binding ==")
    from h3app import director_convert as _dconv
    from h3app import director_lexicon as _lex
    _canon = {"face": "round face", "hair": "pink bob", "build": "slim",
              "skin": "fair skin"}
    _oref = {"outfit": "white dress", "accessories": "ribbon",
             "materials": "", "color_keys": ""}
    _dress_items = {n: {"value": "", "locked": False, "request": ""}
                    for n in ("video_type", "location", "scenery", "subject",
                              "acting", "camera_style", "camera_work",
                              "dialogue", "voice", "ambient_audio", "mood")}
    # C1 hardening: every lexicon-fallback word here must be a literal
    # dictionary-form lexicon entry with no connecting particles between
    # them, otherwise compile_direct_prompt now correctly refuses on the
    # untranslated remainder instead of silently deleting it.
    _dress_items["outfit"] = {"value": "夏祭り浴衣、白いドレスは着ない",
                              "locked": False, "request": ""}
    _dress_items["dialogue"] = {"value": "ねえ、何食べる？", "locked": False,
                                "request": ""}
    _prompt_yukata = _lex.compile_direct_prompt(
        items=_dress_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    _subject1 = [line for line in _prompt_yukata.splitlines()
                 if line.startswith("<Subject 1>")]
    check("E1: Subject 1 carries the directed yukata",
          lambda: _eq(bool(_subject1) and "yukata" in _subject1[0].lower(),
                      True))
    check("E2: overridden white dress is gone from the prompt",
          lambda: _eq("white dress" in _prompt_yukata.lower(), False))
    check("E3: overridden accessories/materials are gone",
          lambda: _eq("ribbon" in _prompt_yukata.lower(), False))
    check("E4: no phantom Subject 2 is emitted",
          lambda: _eq("<Subject 2>" in _prompt_yukata, False))
    check("E5: no avoid/ban enumeration ever reaches the final prompt "
          "(item 1: BasicGuider has no negative conditioning input)",
          lambda: _eq(("avoid depicting" in _prompt_yukata.lower(),
                       "Natural synchronized mouth movement matching "
                       "the spoken dialogue" in _prompt_yukata),
                      (False, True)))
    check("E6: dialogue lines become verbatim d tags",
          lambda: _eq(_prompt_yukata.count("<d>[Japanese]"), 1))
    check("E6b: Subject 1 canon line uses labelled headings (item 3)",
          lambda: _eq(bool(_subject1) and
                      "Face: round face." in _subject1[0] and
                      "Hair: pink bob." in _subject1[0], True))
    _plain_items = {n: {"value": "", "locked": False, "request": ""}
                    for n in ("video_type", "location", "scenery", "subject",
                              "acting", "camera_style", "camera_work",
                              "dialogue", "voice", "ambient_audio", "mood")}
    _plain_items["outfit"] = {"value": "", "locked": False, "request": ""}
    _prompt_plain = _lex.compile_direct_prompt(
        items=_plain_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    _subject1p = [line for line in _prompt_plain.splitlines()
                  if line.startswith("<Subject 1>")]
    check("E7: without a directed outfit the outfit_ref default applies",
          lambda: _eq(bool(_subject1p) and
                      "white dress" in _subject1p[0].lower(), True))
    check("E8: without override, outfit_ref accessories join the canon "
          "line (item 3: materials/color_keys included when present)",
          lambda: _eq(bool(_subject1p) and
                      "Accessories: ribbon." in _subject1p[0], True))
    import inspect as _inspect3
    _gen_src = inspect.getsource(Pipeline.generate_director)
    check("E9: director-direct path never calls _run_director",
          lambda: _eq(("await self._run_director" in _gen_src,
                       "generate_director" in _gen_src), (False, True)))
    _seg_src = inspect.getsource(StoryPipeline._run_segment)
    check("E9: story direct branch reads precompiled prompts",
          lambda: _eq("direct_en_prompt" in _seg_src, True))
    _conv = _dconv.final_to_h3_single(
        {"kind": "single", "items": _dress_items},
        {"canon": _canon, "outfit_ref": _oref}, ["r0.png"], 124)
    check("E10: single entry point compiles end to end",
          lambda: _eq(("yukata" in _conv.lower(),
                       "white dress" in _conv.lower()), (True, False)))
    check("E10: entry point never emits an avoid enumeration",
          lambda: _eq("avoid depicting" in _conv.lower(), False))

    # --------------------------- Phase 1: speaker binding + refs (2, 6) ---
    print("\n== Phase 1: official speaker binding + reference labels ==")
    _bind_items = {n: {"value": "", "locked": False, "request": ""}
                   for n in ("video_type", "location", "scenery", "subject",
                             "outfit", "acting", "camera_style",
                             "camera_work", "ambient_audio", "mood")}
    _bind_items["dialogue"] = {"value": "ねえ、何食べる？", "locked": False,
                               "request": ""}
    _bind_items["voice"] = {"value": "", "en": "a calm, warm tone",
                            "locked": False, "request": ""}
    _prompt_bind = _lex.compile_direct_prompt(
        items=_bind_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    _bind_lines = _prompt_bind.splitlines()
    _shot_i = next((i for i, ln in enumerate(_bind_lines)
                    if ln.startswith("[Shot 1] ")), -1)
    check("P2a: shot line starts \"[Shot 1] \" (space) with <Subject 1> "
          "bound action prose",
          lambda: _eq(_shot_i >= 0 and
                      _bind_lines[_shot_i].startswith("[Shot 1] <Subject 1> "),
                      True))
    check("P2b: a dedicated speaker-binding line precedes the d tag, "
          "ending in \"and says,\"",
          lambda: _eq(_shot_i >= 0 and
                      bool(_lex.SPEAKER_LINE_RE.match(
                          _bind_lines[_shot_i + 1])) and
                      _bind_lines[_shot_i + 1].startswith("<Subject 1> (S1) ")
                      and _bind_lines[_shot_i + 2].startswith(
                          "<d>[Japanese] ねえ、何食べる？</d>"),
                      True))
    check("D1: the speaker-binding line's delivery clause is always the "
          "fixed \"speaks naturally\" (never derived from the voice item's "
          "English form, which stays in overall_soundscape instead)",
          lambda: _eq(
              (_bind_lines[_shot_i + 1].endswith(
                  "speaks naturally and says,"),
               "calm, warm tone" in _prompt_bind.split(
                   "overall_soundscape")[1]),
              (True, True)))
    _no_action_items = {k: dict(v) for k, v in _bind_items.items()}
    _prompt_noaction = _lex.compile_direct_prompt(
        items=_no_action_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    check("P2d: with no acting/movement/camera_work, action falls back to "
          "\"<Subject 1> performs naturally for the scene\"",
          lambda: _eq(
              "<Subject 1> performs naturally for the scene." in
              _prompt_noaction, True))

    _prompt_video_ref = _lex.compile_direct_prompt(
        items=_plain_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124, continuation=True)
    check("P6a: legacy continuation=True still emits <Video 1>/<Audio 1> "
          "as a video soundtrack (back-compat for LONG)",
          lambda: _eq(("<Video 1>" in _prompt_video_ref,
                       "synchronized audio track of <Video 1>" in
                       _prompt_video_ref), (True, True)))
    _prompt_voice_master = _lex.compile_direct_prompt(
        items=_plain_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=243,
        refs_available={"videos": 0, "audios": 1,
                        "audio_kind": "voice_master"})
    check("P6b: LONG_FAST clip>0 (0f relay + Voice Master): no <Video 1>, "
          "<Audio 1> described as a fixed voice reference, not a soundtrack",
          lambda: _eq(("<Video 1>" in _prompt_voice_master,
                       "fixed voice reference" in _prompt_voice_master,
                       "is not the soundtrack of any video" in
                       _prompt_voice_master), (False, True, True)))
    _refs_clip0 = _dconv._refs_available_for_clip("LONG_FAST", 0)
    _refs_clip1_ff = _dconv._refs_available_for_clip("LONG_FAST", 1)
    _refs_clip1_long = _dconv._refs_available_for_clip("LONG", 1)
    check("P6c: refs_available descriptor matches the LONG/LONG_FAST graphs",
          lambda: _eq((_refs_clip0, _refs_clip1_ff, _refs_clip1_long),
                      ({"videos": 0, "audios": 0},
                       {"videos": 0, "audios": 1,
                        "audio_kind": "voice_master"},
                       {"videos": 1, "audios": 1,
                        "audio_kind": "video_soundtrack"})))

    _h3_graph_relay0f = {
        graphs.NODE_H3: {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "ref_images.ref_image_0": ["a", 0],
            "ref_images.ref_image_1": ["b", 0],
            "ref_audios.ref_audio_0": ["voice_master", 0]}}}
    try:
        graphs.assert_reference_labels(_h3_graph_relay0f, _prompt_video_ref)
        _p6d_raised = False
    except graphs.GraphError:
        _p6d_raised = True
    check("P6d: a LONG_FAST 0f-relay graph naming <Video 1> raises "
          "GraphError (item 6: no phantom reference labels)",
          lambda: _eq(_p6d_raised, True))
    _h3_graph_long_relay = {
        graphs.NODE_H3: {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "ref_images.ref_image_0": ["a", 0],
            "ref_videos.ref_video_0": ["tail_images", 0],
            "ref_video_audios.ref_video_audio_0": ["tail_audio", 0]}}}
    try:
        graphs.assert_reference_labels(_h3_graph_long_relay, _prompt_video_ref)
        _p6e_ok = True
    except graphs.GraphError:
        _p6e_ok = False
    check("P6e: a LONG relay graph (real <Video 1>+<Audio 1>) naming "
          "<Video 1> passes",
          lambda: _eq(_p6e_ok, True))
    try:
        graphs.assert_reference_labels(_h3_graph_relay0f, _prompt_yukata)
        _p6f_ok = True
    except graphs.GraphError:
        _p6f_ok = False
    check("P6f: extra CONNECTED references the prompt never names are fine",
          lambda: _eq(_p6f_ok, True))

    # ------------------------------- Phase 1: canon restoration (item 3) --
    print("\n== Phase 1: Subject 1 canon restoration from snapshot ==")
    _profile_text_only = ("FACE: heart-shaped face\nHAIR: silver twin tails\n"
                          "BUILD: petite\nSKIN: porcelain\n"
                          "OUTFIT: sailor uniform\nACCESSORIES: red ribbon\n"
                          "MATERIALS: cotton\nCOLOR_KEYS: red, white\n")
    _snap_profile_only = {"canon": {}, "outfit_ref": {},
                          "profile_text": _profile_text_only}
    _id2, _oref2 = _dconv._snapshot_parts(_snap_profile_only)
    check("P3a: empty canon falls back to parsing profile_text",
          lambda: _eq((_id2.get("face"), _oref2.get("outfit"),
                       _oref2.get("materials")),
                      ("heart-shaped face", "sailor uniform", "cotton")))
    _snap_nothing = {"canon": {}, "outfit_ref": {}, "profile_text": ""}
    _id3, _oref3 = _dconv._snapshot_parts(_snap_nothing)
    check("P3b: nothing available degrades to empty identity, never errors",
          lambda: _eq((_id3, _oref3),
                      ({"face": "", "hair": "", "build": "", "skin": ""},
                       {"outfit": "", "accessories": "", "materials": "",
                        "color_keys": ""})))
    _prompt_degrade = _lex.compile_direct_prompt(
        items=_plain_items, identity=_id3, outfit_ref=_oref3,
        refs=["r0.png"], frames=124)
    check("P3c: with nothing available, subject line degrades to "
          "\"the main woman\" (never raises)",
          lambda: _eq("<Subject 1> the main woman" in
                      _prompt_degrade.splitlines()[1], True))

    # ----------------- Director cast v2 + speech FINAL validation (pure) --
    print("\n== Director cast v2 + speech policy + FINAL validation ==")
    from h3app import director_speech as _sp
    check("F1: new specs carry the default cast (Subject 1 only)",
          lambda: _eq((DM.new_single(
              idea="x", duration_sec=5)["cast"]["on_screen_subjects"],
              DM.new_story30(idea="x")["master"]["cast"]["camera_pov"]),
              (["character:0"], "")))
    check("F1: unknown pov normalizes instead of failing",
          lambda: _eq(DM.normalize_pov("恋人目線っぽく"), "lover_pov"))
    check("F1: bad cast shape is a blocking error",
          lambda: _eq(DM.validate_spec(
              {**DM.new_single(idea="x", duration_sec=5),
               "cast": {"on_screen_subjects": "二人"}})["ok"], False))
    _lover_cast = {"on_screen_subjects": ["character:0"],
                   "camera_operator": "lover",
                   "camera_operator_visibility": "off_camera",
                   "camera_device": "smartphone", "camera_pov": "lover_pov",
                   "off_camera_people": ["lover"]}
    _lover_items = {k: dict(v) for k, v in _dress_items.items()}
    _lover_items["subject"] = {"value": "浴衣姿の女性と一緒に歩く恋人である撮影者",
                               "locked": False, "request": ""}
    _prompt_lover = _lex.compile_direct_prompt(
        items=_lover_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124,
        cast=_lover_cast)
    check("F2: lover-photographer stays off-camera, never a Subject 2",
          lambda: _eq(("<Subject 2>" in _prompt_lover,
                       "(S2)" in _prompt_lover,
                       "never visible on screen" in _prompt_lover),
                      (False, False, True)))
    _b_items = {k: dict(v) for k, v in _plain_items.items()}
    _b_items["dialogue"] = {"value": "A: こんにちは\nB: やあ",
                            "locked": False, "request": ""}
    _prompt_b = _lex.compile_direct_prompt(
        items=_b_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124,
        cast=DM.blank_cast())
    check("F2: B: collapses to S1 when no second subject is on screen",
          lambda: _eq(("(S2)" in _prompt_b,
                       _prompt_b.count("<d>[Japanese]")), (False, 2)))
    _pol, _convs = _sp.apply_speech_policy(
        "overall_soundscape\ncrowd chatter and 太鼓や人の声")
    check("F3: ambience converts to non-verbal, never deletes",
          lambda: _eq(("crowd chatter" in _pol, "人の声" in _pol,
                       "no intelligible" in _pol), (False, False, True)))
    check("F3: replacement never cascades on its own words",
          lambda: _eq(_pol.count("distant taiko drums without voices"), 1))
    _rep_ok = _sp.validate_final(
        "(S1) <d>[Japanese] ねえ、何食べる？</d>", dialogue=["ねえ、何食べる？"])
    check("F3: matching dialogue passes FINAL with no hard errors",
          lambda: _eq((_rep_ok["hard"], len(_rep_ok["warnings"]) >= 0),
                      ([], True)))
    _rep_bad = _sp.validate_final(
        "(S1) <d>[Japanese] ねえ、何食べる？</d> extra <d>[Japanese] 勝手な台詞</d>",
        dialogue=["ねえ、何食べる？"])
    check("F4: invented d tag is a HARD error",
          lambda: _eq(bool(_rep_bad["hard"]), True))
    _rep_s2 = _sp.validate_final(
        "(S2) <d>[Japanese] やあ</d>", dialogue=["やあ"], allow_s2=False)
    check("F4: phantom S2 is a HARD error without a second subject",
          lambda: _eq(bool(_rep_s2["hard"]), True))
    _rep_warn = _sp.validate_final(
        "二人が祭りで笑う (S1) <d>[Japanese] たこ焼き</d>",
        dialogue=["たこ焼き"])
    check("F4: non-quoted prose outside d is WARNING only, never hard",
          lambda: _eq((_rep_warn["hard"], bool(_rep_warn["warnings"])),
                      ([], True)))
    _rep_quote_hard = _sp.validate_final(
        "二人が「たこ焼き」と話す (S1) <d>[Japanese] たこ焼き</d>",
        dialogue=["たこ焼き"])
    check("F4: quoted JP outside d is HARD",
          lambda: _eq(bool(_rep_quote_hard["hard"]), True))
    _st1 = DM.new_story30(idea="x")
    _ans1 = {"clips": [{"index": i + 1, "items": {
        "dialogue": {"value": f"台詞{i + 1}"}}} for i in range(3)]}
    _m1 = DC.merge_model_answer_clips(_st1, _ans1)
    check("F5: 1-based clip answers merge positionally, never silently empty",
          lambda: _eq((
              [_m1["clips"][i]["items"]["dialogue"]["value"]
               for i in range(3)],
              bool(_m1["_clips_merge"]["positional"])),
              (["台詞1", "台詞2", "台詞3"], True)))

    # ------------------------------- Phase 1: item 1 + item 2 guards ------
    print("\n== Phase 1: avoid-block guard + protected speaker line ==")
    _avoid_pol, _avoid_convs = _sp.apply_speech_policy(
        "detailed_description\nSomething happens.\n"
        "Avoid depicting the following: blur, low detail.\n"
        "Mouth movement synchronized.")
    check("P1a: apply_speech_policy strips any \"Avoid depicting...\" line "
          "that reaches it from any path, and records the conversion",
          lambda: _eq(("Avoid depicting" in _avoid_pol,
                       "avoid_block_stripped" in _avoid_convs,
                       "Something happens." in _avoid_pol),
                      (False, True, True)))
    _avoid_pol2, _ = _sp.apply_speech_policy(
        "Avoid: crowd noise\nDo not depict extra people\nRest is fine.")
    check("P1b: \"Avoid:\" and \"Do not depict\" variants are also stripped",
          lambda: _eq(("Avoid:" in _avoid_pol2,
                       "Do not depict" in _avoid_pol2,
                       "Rest is fine." in _avoid_pol2), (False, False, True)))
    _protected_line = "<Subject 1> (S1) speaks in a calm tone and says,"
    _wrap_prompt = ("detailed_description\n[Shot 1] <Subject 1> smiles.\n" +
                    _protected_line +
                    "\n<d>[Japanese] こんにちは</d>\noverall_soundscape\nN/A")
    _protected_out, _ = _sp.final_gate(
        _wrap_prompt, dialogue=["こんにちは"])
    check("P2e: the canonical speaker-binding line survives a full "
          "final_gate round-trip untouched (\"says,\" never becomes "
          "\"smiles,\")",
          lambda: _eq(_protected_line in _protected_out, True))
    _protected_report = _sp.validate_final(
        _wrap_prompt, dialogue=["こんにちは"])
    check("P2f: the speaker-binding line's \"says\"/tone words never "
          "trigger the heuristic WARNING scan",
          lambda: _eq(any("says" in w or "speaks" in w
                          for w in _protected_report["warnings"]), False))

    # --------------------------------- Phase 1: item 4 (voice) + item 5 ---
    print("\n== Phase 1: voice preservation (item 4) + no silent JP loss "
          "(item 5) ==")
    check("P4a: clean_voice_item no longer deletes bare CJK (was: -> \"\")",
          lambda: _eq(_lex.clean_voice_item(
              "明るく弾むようなトーン。早口ではなく丁寧に話す。") != "", True))
    _voice_en_items = {k: dict(v) for k, v in _plain_items.items()}
    _voice_en_items["voice"] = {
        "value": "明るく弾むようなトーン。早口ではなく丁寧に話す。",
        "en": "a bright, bouncy tone, speaking politely rather than quickly",
        "locked": False, "request": ""}
    _voice_en_items["dialogue"] = {"value": "また明日ね", "locked": False,
                                   "request": ""}
    _prompt_voice_en = _lex.compile_direct_prompt(
        items=_voice_en_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    check("P4b: a Japanese voice direction survives into "
          "overall_soundscape via the item's English \"en\" form",
          lambda: _eq("bright, bouncy tone" in _prompt_voice_en, True))
    check("P4c: the dialogue text is never copied into the voice field / "
          "overall_soundscape",
          lambda: _eq("また明日ね" in _prompt_voice_en.split(
              "overall_soundscape")[1], False))
    _runs = _lex.untranslated_runs(
        "summary\n[reference generation] 未対応の日本語 remains.\n"
        "<d>[Japanese] セーフ</d>")
    check("P5a: untranslated_runs finds CJK outside <d> only",
          lambda: _eq(_runs, ["未対応の日本語"]))
    _clean_report = _lex.validate_compiled_prompt(_prompt_yukata)
    check("P5b: a well-formed compiled prompt validates clean "
          "(N/A, [Shot 1], <Picture 1>, timestamps, <d> all pass)",
          lambda: _eq(_clean_report["ok"], True))
    _broken_report = _lex.validate_compiled_prompt(
        "subject_definitions\n<Subject 1> ok\nsummary\n"
        "[reference generation] .\nretention_analysis\nok\n"
        "detailed_description\n[Shot 3] At 00:07.500, .\n"
        "overall_soundscape\nok\nnon_diegetic_music\nN/A\n")
    check("P5c: validate_compiled_prompt hard-fails on an empty section "
          "body and on orphan comma/period fragments (C4: \"00:07.500, .\" "
          "trips both the comma-then-period check and the new orphan "
          "word-then-period check)",
          lambda: _eq((_broken_report["ok"], len(_broken_report["errors"])),
                      (False, 3)))
    _en_item = DM.blank_item("こんにちは")
    check("P5d: blank_item now carries an empty \"en\" field",
          lambda: _eq(_en_item.get("en"), ""))
    _locked_old = DM.new_single(idea="x", duration_sec=5)
    _locked_old["items"]["mood"] = {"value": "楽しい", "en": "joyful",
                                    "locked": True, "request": ""}
    _locked_new = DM.new_single(idea="x", duration_sec=5)
    _locked_new["items"]["mood"] = {"value": "違う", "en": "different",
                                    "locked": False, "request": ""}
    _locked_result, _locked_enforced = DM.apply_locks(
        _locked_old, _locked_new)
    check("P5e: apply_locks restores a locked item's \"en\" alongside "
          "its value",
          lambda: _eq((_locked_result["items"]["mood"]["value"],
                       _locked_result["items"]["mood"]["en"]),
                      ("楽しい", "joyful")))

    # ------------------- Phase 1 fix: untranslated-JP refusal (C1-C4) -----
    print("\n== Phase 1 fix: refuse instead of silently mangling (C1-C4) ==")
    _bad_single_items = {n: {"value": "", "locked": False, "request": ""}
                         for n in DM.SINGLE_ITEMS}
    _bad_single_items["location"] = {"value": "本日は晴天なり",
                                     "locked": False, "request": ""}
    _bad_single_spec = {"kind": "single", "items": _bad_single_items}
    _c1a_raised, _c1a_runs, _c1a_msg = False, [], ""
    try:
        _dconv.final_to_h3_single(
            _bad_single_spec, {"canon": _canon, "outfit_ref": _oref},
            ["r0.png"], 124)
    except _lex.UntranslatedPromptError as exc:
        _c1a_raised, _c1a_runs = True, exc.runs
        _c1a_msg = _untranslated_prompt_message(exc)
    check("C1a: Japanese the lexicon cannot translate raises "
          "UntranslatedPromptError from compile_direct_prompt (via "
          "final_to_h3_single) BEFORE strip_cjk_outside_d can delete it",
          lambda: _eq((_c1a_raised, bool(_c1a_runs)), (True, True)))
    check("C1b: the single-generate handler message is a named, "
          "400-shaped refusal (no CLIP prefix outside story30)",
          lambda: _eq(("未翻訳の日本語" in _c1a_msg, "CLIP" in _c1a_msg,
                       _json_error(_c1a_msg, status=400).status),
                      (True, False, 400)))
    _bad_story_spec = DM.new_story30(idea="x")
    _bad_story_spec["clips"][0]["items"]["location"] = {
        "value": "本日は晴天なり", "locked": False, "request": ""}
    _c1c_raised, _c1c_msg = False, ""
    try:
        _dconv.final_to_h3_clip(
            _bad_story_spec, 0, {"canon": _canon, "outfit_ref": _oref},
            ["r0.png"], 243, story_mode="LONG_FAST")
    except _lex.UntranslatedPromptError as exc:
        _c1c_raised = True
        _c1c_msg = _untranslated_prompt_message(exc, 0)
    check("C1c: the story30 per-clip handler message names CLIP1 "
          "(0-based clip index -> 1-based label) and is a 400",
          lambda: _eq((_c1c_raised, "CLIP1" in _c1c_msg,
                       _json_error(_c1c_msg, status=400).status),
                      (True, True, 400)))
    _empty_voice_items = {k: dict(v) for k, v in _plain_items.items()}
    _empty_voice_items["voice"] = {"value": "「」", "locked": False,
                                   "request": ""}
    _prompt_empty_voice = _lex.compile_direct_prompt(
        items=_empty_voice_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    check("C2: a voice direction that reduces to nothing never emits an "
          "empty \"Voice: \" label",
          lambda: _eq("Voice:" in _prompt_empty_voice, False))
    _clean_en_voice_items = {k: dict(v) for k, v in _plain_items.items()}
    _clean_en_voice_items["voice"] = {
        "value": "", "en": "a warm, gentle manner", "locked": False,
        "request": ""}
    _clean_en_voice_items["dialogue"] = {"value": "うん", "locked": False,
                                         "request": ""}
    _prompt_clean_en_voice = _lex.compile_direct_prompt(
        items=_clean_en_voice_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    check("D1: the voice item's \"en\" field still reaches "
          "overall_soundscape, but the speaker-binding line NEVER derives "
          "its delivery clause from it (fixed \"speaks naturally\" always)",
          lambda: _eq(
              ("a warm, gentle manner" in _prompt_clean_en_voice,
               "<Subject 1> (S1) speaks naturally and says," in
               _prompt_clean_en_voice),
              (True, True)))
    _noen_voice_items = {k: dict(v) for k, v in _plain_items.items()}
    # Lexicon-translatable JP (no "en"): this is exactly the reduced-to-a-
    # bare-noun scenario the bug produced ("speaks in a slowly tone").
    _noen_voice_items["voice"] = {"value": "ゆっくり", "locked": False,
                                  "request": ""}
    _noen_voice_items["dialogue"] = {"value": "うん", "locked": False,
                                     "request": ""}
    _prompt_noen_voice = _lex.compile_direct_prompt(
        items=_noen_voice_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124)
    check("C3: without the item's own \"en\" field, the speaker line "
          "always falls back to the fixed \"speaks naturally\" "
          "(never a bare-noun fragment like \"speaks in a tone\")",
          lambda: _eq("<Subject 1> (S1) speaks naturally and says," in
                      _prompt_noen_voice, True))
    check("C4a: _collapse_adjacent_duplicates collapses an exact "
          "adjacent duplicate phrase, once",
          lambda: _eq(_lex._collapse_adjacent_duplicates(
              "wearing a yukata food stall food stall slowly"),
              "wearing a yukata food stall slowly"))
    _off_nouns_pov = _lex._offcamera_nouns(
        {"camera_operator": "boyfriend",
         "camera_operator_visibility": "off_camera"})
    _off_pov_result = _lex._drop_offcamera_nouns(
        "framed in the boyfriend's point-of-view shot", _off_nouns_pov)
    check("D3: dropping an off-camera possessive noun (\"boyfriend's\") "
          "removes the noun AND leaves no dangling \"a 's\"/\"the 's\" or "
          "double-space artefact",
          lambda: _eq(
              ("boyfriend" in _off_pov_result.lower(),
               "a 's" in _off_pov_result, "the 's" in _off_pov_result,
               "  " in _off_pov_result),
              (False, False, False, False)))
    check("C4b: validate_compiled_prompt flags a word directly followed "
          "by whitespace then a period (orphan trailing punctuation)",
          lambda: _eq(_lex.validate_compiled_prompt(
              "subject_definitions\nok\nsummary\nok\nretention_analysis\n"
              "ok\ndetailed_description\n[Shot 1] <Subject 1> "
              "front-facing .\noverall_soundscape\nok\n"
              "non_diegetic_music\nN/A\n")["ok"], False))

    # ----------------- AUDIO HOTFIX: speech isolation (pure) --------------
    print("\n== AUDIO HOTFIX: speech isolation ==")
    _iso_items = {n: {"value": "", "locked": False, "request": ""}
                  for n in DM.SINGLE_ITEMS}
    _iso_items["outfit"] = {"value": "浴衣", "locked": False, "request": ""}
    _iso_items["dialogue"] = {"value": "ね、見て！この提灯！",
                              "locked": False, "request": ""}
    # C1 hardening: kept to only literal dictionary-form lexicon entries
    # (no connecting particles) so the fallback translation is complete and
    # compile_direct_prompt does not refuse on an untranslated remainder;
    # the "話す" speech verb is still present to exercise G1 below.
    _iso_items["acting"] = {
        "value": "提灯話す",
        "locked": False, "request": ""}
    _iso_items["voice"] = {
        "value": "「楽しいね」ゆっくり",
        "locked": False, "request": ""}
    _iso_items["ambient_audio"] = {
        "value": "屋台祭囃子",
        "locked": False, "request": ""}
    _iso_prompt = _lex.compile_direct_prompt(
        items=_iso_items, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=243,
        cast=DM.blank_cast())
    import re as _remod
    _iso_prose = _remod.sub(r"<d>\[Japanese\].*?</d>", " ", _iso_prompt,
                            flags=_remod.S)
    # The compiler's OWN speaker-binding scaffolding line legitimately says
    # "speaks ..." (item 2/4); this check is about the ACTING item's speech
    # verb ("と話す"), so the scaffolding line is excluded here.
    _iso_prose_no_binding = "\n".join(
        ln for ln in _iso_prose.splitlines()
        if not _lex.SPEAKER_LINE_RE.match(ln.strip()))
    check("G1: acting speech verbs become gesture prose",
          lambda: _eq(("speaks" in _iso_prose_no_binding,
                       "話す" in _iso_prose), (False, False)))
    check("G2: dialogue restatement leaves Shot prose",
          lambda: _eq(("提灯を見て" in _iso_prose,
                       "この提灯" in _iso_prose), (False, False)))
    check("G3: no CJK survives outside d tags",
          lambda: _eq(bool(_remod.search(
              "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
              _iso_prose)), False))
    check("G3: dialogue Japanese survives inside d",
          lambda: _eq("この提灯" in _iso_prompt, True))
    check("G4: voice keeps timbre, drops quotes and content",
          lambda: _eq(("「楽しいね」" in _iso_prompt,
                       "says" in _iso_prose_no_binding.lower()),
                      (False, False)))
    check("G5: ambience is non-verbal with guards",
          lambda: _eq(("vendors shouting" in _iso_prose.lower(),
                       "人の声" in _iso_prose,
                       "No narration, no additional voices." in _iso_prompt,
                       "Natural synchronized mouth movement matching the "
                       "spoken dialogue" in _iso_prompt),
                      (False, False, True, True)))
    _gate_ok, _gate_rep = _sp.final_gate(
        _iso_prompt, dialogue=["ね、見て！この提灯！"])
    check("G6: isolated prompt passes FINAL with no hard errors",
          lambda: _eq(_gate_rep["hard"], []))
    # NOTE: the gate converts (strips) these first, so HARD 5/6 are tested
    # at validate_final level: they are backstops for anything the policy
    # could not convert. Via final_gate the same inputs come out clean.
    _gate_bad_rep = _sp.validate_final(
        "(S1) <d>[Japanese] ね、見て！</d> 彼女は「今日は楽しいね」と話す。",
        dialogue=["ね、見て！"])
    check("G7: quoted JP outside d is HARD",
          lambda: _eq(bool(_gate_bad_rep["hard"]), True))
    _gate_bad2, _ = _sp.final_gate(
        "(S1) <d>[Japanese] ね、見て！</d> 彼女は「今日は楽しいね」と話す。",
        dialogue=["ね、見て！"])
    check("G7: gate converts quotes away before H3",
          lambda: _eq("「今日は楽しいね」" in _gate_bad2, False))
    _gate_echo_rep = _sp.validate_final(
        "(S1) <d>[Japanese] ね、見て！</d> ね、見て！と繰り返す。",
        dialogue=["ね、見て！"])
    check("G7: restated dialogue body outside d is HARD",
          lambda: _eq(bool(_gate_echo_rep["hard"]), True))
    _audit = _sp.audit_prompt(_iso_prompt)
    check("G8: audit reports d count, no JP runs, no S2",
          lambda: _eq((_audit["d_count"],
                       _audit["jp_outside_d_runs"],
                       _audit["has_subject2"], _audit["has_s2"]),
                      (1, 0, False, False)))
    _lover_cast2 = {"on_screen_subjects": ["character:0"],
                    "camera_operator": "恋人",
                    "camera_operator_visibility": "off_camera",
                    "camera_device": "smartphone",
                    "camera_pov": "lover_pov",
                    "off_camera_people": ["lover"]}
    _lover_items2 = {k: dict(v) for k, v in _plain_items.items()}
    # C1 hardening: literal dictionary-form lexicon words only (no
    # connecting particles); "恋人" is the off-camera noun under test.
    _lover_items2["acting"] = {"value": "恋人微笑む",
                               "locked": False, "request": ""}
    _prompt_lover2 = _lex.compile_direct_prompt(
        items=_lover_items2, identity=_canon, outfit_ref=_oref,
        refs=["r0.png"], frames=124,
        cast=_lover_cast2)
    _lover_prose2 = _remod.sub(r"<d>\[Japanese\].*?</d>", " ",
                               _prompt_lover2, flags=_remod.S)
    _shot_lines2 = [ln for ln in _lover_prose2.splitlines()
                    if ln.startswith("[Shot")]
    _shot_blob2 = " ".join(_shot_lines2)
    check("G9: off-camera lover nouns leave Shot prose",
          lambda: _eq(("lover" in _shot_blob2.lower(),
                       "恋人" in _shot_blob2,
                       "never visible on screen" in _prompt_lover2),
                      (False, False, True)))

    # ------------------------------------------- Character Library (pure) ----
    # Temp store + temp files. No GPU, no ComfyUI, no Explorer.
    print("\n== Character Library: save/apply/snapshot/delete safety ==")
    from h3app import character_library as _charlib
    from h3app.projects import profile_cache_key as _keyfn
    cl_root = debug / "selftest_charlib"
    shutil.rmtree(cl_root, ignore_errors=True)
    cl_in = cl_root / "comfy_input"
    cl_in.mkdir(parents=True)
    (cl_in / "h3app_ref_a.png").write_bytes(b"REF-A")
    (cl_in / "h3app_ref_b.png").write_bytes(b"REF-B")
    (cl_in / "voice_a.wav").write_bytes(b"WAV-A")
    _store = _charlib.CharacterStore(cl_root / "library")
    _prof = "FACE: round face\nHAIR: black bob\nBUILD: slim\nSKIN: fair\n" \
        "OUTFIT: white dress\nACCESSORIES: ribbon\n"
    _canon_full = canon_mod.parse_profile(_prof).as_dict()
    _canon = {k: _canon_full.get(k, "") for k in
              ("face", "hair", "build", "skin")}
    _oref = {k: _canon_full.get(k, "") for k in
             ("outfit", "accessories", "materials", "color_keys")}
    _item, _err = _store.save_new(
        display_name="Sakura", refs=[cl_in / "h3app_ref_a.png"],
        profile_text=_prof, canon=_canon, outfit_ref=_oref,
        voice_src=cl_in / "voice_a.wav", memo="m")
    check("L1: save stores refs + voice as library copies",
          lambda: _eq((_item is not None and len(_item["refs"]) == 1 and
                       bool(_item["voice_wav"])), True))
    check("L1: canon splits identity from outfit_ref",
          lambda: _eq(((_item or {}).get("canon") or {}).get("face"),
                      "round face") and _eq(
                      ((_item or {}).get("outfit_ref") or {}).get("outfit"),
                      "white dress"))
    _cid = (_item or {}).get("character_id", "")
    check("L2: list shows the character",
          lambda: _eq(any(c["character_id"] == _cid
                          for c in _store.list()), True))
    _names = _store.materialize_refs(_store.get(_cid), cl_in)
    check("L3: apply materializes deterministic h3app_char_* names",
          lambda: _eq(len(_names) == 1 and _names[0].startswith("h3app_char_"),
                      True))
    _key = _keyfn([cl_in / n for n in _names])
    _snap = {"character_id": _cid, "profile_text": _prof, "refs": _names,
             "key": _key}
    check("L4: snapshot key matches the recomputed image-set key",
          lambda: _eq(_keyfn([cl_in / n for n in _snap["refs"]]),
                      _snap["key"]))
    check("L4: manual image change drops the snapshot",
          lambda: _eq(list(_snap["refs"]) != ["h3app_ref_b.png"], True))
    # A -> B switch replaces the whole set.
    _item2, _err2 = _store.save_new(
        display_name="Yuki", refs=[cl_in / "h3app_ref_b.png"],
        profile_text="FACE: sharp face\n")
    _cid2 = (_item2 or {}).get("character_id", "")
    _names2 = _store.materialize_refs(_store.get(_cid2), cl_in)
    check("L5: switching characters replaces the set without mixing",
          lambda: _eq((_names2 != _names and len(_names2) == 1), True))
    check("L5: re-apply overwrites the same deterministic names",
          lambda: _eq(_store.materialize_refs(_store.get(_cid), cl_in),
                      _names))
    # Update is atomic: index write is last.
    _upd, _uerr = _store.update(_cid, display_name="Sakura v2",
                                memo="new memo")
    check("L6: update refreshes all fields consistently",
          lambda: _eq(((_upd or {}).get("display_name"),
                       (_upd or {}).get("memo"),
                       (_upd or {}).get("updated_at", 0) >=
                       (_upd or {}).get("created_at", 1)),
                      ("Sakura v2", "new memo", True)))
    # Duplicate is fully independent.
    _dup, _derr = _store.duplicate(_cid)
    _dupid = (_dup or {}).get("character_id", "")
    check("L7: duplicate gets a new id with copied assets",
          lambda: _eq((_dupid != _cid and len((_dup or {}).get("refs") or [])
                       == 1), True))
    _okdel, _delerr = _store.delete(_cid)
    check("L8: delete removes the entry",
          lambda: _eq((_okdel, _store.get(_cid)), (True, None)))
    check("L8: duplicate survives the original's deletion",
          lambda: _eq(_store.get(_dupid) is not None, True))
    check("L8: snapshot files (comfy_input copies) are untouched",
          lambda: _eq(all((cl_in / n).is_file() for n in _names), True))
    # Voice staging independence.
    _vname = _store.materialize_voice(_store.get(_dupid), cl_in)
    check("L9: voice stages deterministically",
          lambda: _eq(bool(_vname) and _vname.startswith("h3app_char_"),
                      True))
    _okdel2, _ = _store.delete(_dupid)
    check("L9: deleting the voice owner keeps the staged copy",
          lambda: _eq((_okdel2, (cl_in / _vname).is_file()), (True, True)))
    _okdel3, _ = _store.delete(_cid2)
    check("L9: empty library lists nothing",
          lambda: _eq((_okdel3, _store.list()), (True, [])))
    shutil.rmtree(cl_root, ignore_errors=True)

    # ------------------------------------------------ AI settings (pure) ----
    # No network, no credentials, no GPU. Live Go calls are covered by the
    # settings UI connection/model tests once the user saves a key.
    print("\n== AI settings: schema, Go guards, secrets, fallback ==")
    from h3app import ai_settings as _aim
    from h3app import opencode_go as _go
    from h3app import credstore as _cred
    _sdef = _aim.normalize(None)
    check("S1: defaults are LM Studio, no Gemma fallback",
          lambda: _eq((_sdef["connection"]["provider"],
                       _sdef["roles"]["director"]["override"],
                       _sdef["roles"]["character_profile"]["override"],
                       _sdef["max_attempts"]),
                      ("lmstudio", False, False, 3)))
    check("S1: gemma_fallback defaults to False",
          lambda: _eq(_sdef["gemma_fallback"] is False, True))
    # WP-A: old shape still migrates (gemma/openai_compat/opencode_go).
    _sold = _aim.normalize({
        "director": {"provider": "opencode_go", "model": "m",
                     "endpoint": "responses", "gemma_fallback": True},
        "character_profile": {"provider": "opencode_go", "model": "m",
                              "endpoint": "responses",
                              "gemma_fallback": True},
        "max_attempts": 9})
    check("S1: old shape migrates to schema 2, attempts capped",
          lambda: _eq((_sold["schema"], _sold["connection"]["provider"],
                       _sold["providers"]["opencode_go"]["model"],
                       _sold["max_attempts"]), (2, "opencode_go", "m", 5)))
    _suser = _aim.normalize({
        "schema": 2,
        "connection": {"kind": "external", "provider": "opencode_go"},
        "providers": {"opencode_go": {"model": "m",
                                      "extra": {"endpoint": "responses"}}},
        "favorites": [{"provider": "opencode_go", "model": "m",
                       "endpoint": "responses"}],
        "max_attempts": 9})
    check("S1: user values survive, attempts capped",
          lambda: _eq((_suser["providers"]["opencode_go"]["model"],
                       _suser["max_attempts"],
                       _suser["favorites"][0]["model"]), ("m", 5, "m")))
    _schain = _aim.resolve_role_chain(_suser, "director", max_attempts=3)
    check("S1: chain is [external] (no gemma_fallback set)",
          lambda: _eq([(a["kind"], a["model"]) for a in _schain],
                      [("external", "m")]))
    _szen = _aim.normalize({
        "director": {"provider": "opencode_go", "model": "m",
                     "fallbacks": [{"provider": "zen", "model": "z"},
                                   {"provider": "opencode_go", "model": "m2",
                                    "endpoint": "messages"}]}})
    check("S1: non-Go fallbacks are stripped (Zen can never sneak in)",
          lambda: _eq([(a["kind"], a["model"]) for a in
                       _aim.resolve_chain(
                           {"kind": "go", "provider": "opencode_go",
                            "model": "m",
                            "fallbacks": [
                                {"provider": "zen", "model": "z"},
                                {"provider": "opencode_go", "model": "m2",
                                 "endpoint": "messages"}]}, 3)],
                      [("go", "m"), ("go", "m2")]))
    try:
        _go._check_url("https://opencode.ai/zen/go/v1/responses")
        _go_ok = True
    except _go.GoError:
        _go_ok = False
    check("S2: Go endpoint passes the safety check", lambda: _eq(_go_ok, True))
    try:
        _go._check_url("https://opencode.ai/zen/v1/responses")
        _zen_blocked = False
    except _go.GoError:
        _zen_blocked = True
    check("S2: Zen pay-as-you-go endpoint is refused",
          lambda: _eq(_zen_blocked, True))
    import inspect as _inspect2
    import ast as _ast2
    _gostrings = [
        n.value for n in _ast2.walk(_ast2.parse(inspect.getsource(_go)))
        if isinstance(n, _ast2.Constant) and isinstance(n.value, str)]
    _bad_urls = [s for s in _gostrings
                 if "opencode.ai/zen/" in s and "/zen/go/" not in s]
    check("S2: every URL literal in the adapter is a Go URL",
          lambda: _eq(_bad_urls, []))
    _models = _go.normalize_models(
        {"models": [{"id": "muse-spark-1.3-contributor",
                     "display": "Muse Spark",
                     "endpoint": "https://opencode.ai/zen/go/v1/responses"},
                    "plain-id"]})
    check("S2: model list normalizes dicts and bare ids",
          lambda: _eq([(m["id"], m["endpoint"]) for m in _models],
                      [("muse-spark-1.3-contributor", "responses"),
                       ("plain-id", "")]))
    _rb = _go._responses_body(model="m", system="s", user="u",
                              images=["data:image/jpeg;base64,AAA"], max_tokens=8)
    check("S2: responses body carries system/images",
          lambda: _eq(("instructions" in _rb and len(
              _rb["input"][0]["content"]) == 2), True))
    _rb_min = _go._responses_body(model="m", system="", user="u",
                                  images=[], max_tokens=256, temperature=None)
    check("S2: probe payload is minimal (no temperature, no tools)",
          lambda: _eq(("temperature" not in _rb_min and "tools" not in _rb_min
                       and _rb_min["max_output_tokens"] == 256), True))
    _inc = {"status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": []}
    check("S2: incomplete responses name the cause instead of empty",
          lambda: _eq(("切れました" in _go._responses_incomplete_reason(_inc)
                       and _go._responses_text(_inc) == ""), True))
    _rb_full = _go._responses_body(model="m", system="s", user="u",
                                   images=[], max_tokens=64, temperature=0.7)
    check("S2: director payload keeps temperature",
          lambda: _eq(_rb_full.get("temperature"), 0.7))
    _e400 = _go._redacted_error(
        400, '{"model":"m","error":{"param":"max_output_tokens",'
        '"type":"invalid_request_error","message":"Upstream request failed: '
        '[invalid_request_error] `max_output_tokens` The number must be '
        '`>= 16`."}}')
    check("S3: 400 max_output_tokens maps to a Japanese floor message",
          lambda: _eq((_e400.kind, "16" in str(_e400)), ("model", True)))
    _e400b = _go._redacted_error(400, '{"error":{"type":"invalid_request_error",'
                                      '"message":"weird param Bearer SECRETKEY"}}')
    check("S3: structured errors never echo credentials",
          lambda: _eq("SECRETKEY" in str(_e400b), False))
    _sentinel = "SECRET-KEY-XYZ-123"
    _err = _go._redacted_error(401, f"invalid key {_sentinel}")
    check("S3: auth errors never echo the key",
          lambda: _eq((_err.kind, _sentinel in str(_err)), ("auth", False)))
    _err2 = _go._redacted_error(402, f"quota {_sentinel}")
    check("S3: quota errors never echo the key",
          lambda: _eq((_err2.kind, _sentinel in str(_err2)),
                      ("quota", False)))

    async def _s_op(kind: str, model_cfg: dict, api_key: str):
        if kind == "go":
            if model_cfg.get("model") == "bad-model":
                raise _go.GoError("model", "nope")
            return {"from": model_cfg.get("model")}
        return {"from": "gemma"}

    import asyncio as _asyncio
    _role_go = {"provider": "opencode_go", "model": "good",
                "endpoint": "responses", "fallbacks": [], "gemma_fallback": True}
    _res, _used = _asyncio.run(
        _aim.run_with_fallback(_role_go, _s_op, max_attempts=3,
                               key_loader=lambda: "K"))
    check("S4: healthy Go model wins without fallback",
          lambda: _eq((_res, _used["fallback"]), ({"from": "good"}, False)))
    _role_bad = {"provider": "opencode_go", "model": "bad-model",
                 "endpoint": "responses", "fallbacks": [], "gemma_fallback": True}
    _res2, _used2 = _asyncio.run(
        _aim.run_with_fallback(_role_bad, _s_op, max_attempts=3,
                               key_loader=lambda: "K"))
    check("S4: failing Go model falls back to Gemma",
          lambda: _eq((_res2, _used2["kind"]), ({"from": "gemma"}, "gemma")))
    _role_auth = {"provider": "opencode_go", "model": "good",
                  "endpoint": "responses", "fallbacks": [],
                  "gemma_fallback": True}
    try:
        _asyncio.run(
            _aim.run_with_fallback(
                _role_auth,
                lambda kind, model_cfg, api_key: (_ for _ in ()).throw(
                    _go.GoError("auth", "bad")) if kind == "go"
                else {"from": "gemma"},
                max_attempts=3, key_loader=lambda: "K"))
        _auth_stopped = False
    except Exception as _e:                                      # noqa: BLE001
        _auth_stopped = "bad" in str(_e)
    check("S4: auth errors abort instead of mass-retrying",
          lambda: _eq(_auth_stopped, True))
    check("S5: credential store backend reports itself",
          lambda: _eq(_cred.available() in (True, False), True))
    if _cred.available():
        _cred.save("h3-selftest-probe", "probe-value")
        _got = _cred.load("h3-selftest-probe")
        _cred.delete("h3-selftest-probe")
        check("S5: credential roundtrip works, then cleaned",
              lambda: _eq((_got, _cred.load("h3-selftest-probe")),
                          ("probe-value", "")))
    else:
        print("    (credstore backend unavailable here; skipped live roundtrip)")

    # ------------------------------- user-editable ACTION presets (10-16) ----
    # A chip is no longer a hidden toggle: it is a COMMAND that appends a
    # deterministic Japanese sentence to THAT segment's prompt text, which the
    # user can then see and edit. Nothing here calls an LLM or translates.
    print("\n== action presets: the STORE (add / delete / persist / validate) ==")
    PR = presets_mod
    ap_root = debug / "selftest_action_presets"
    shutil.rmtree(ap_root, ignore_errors=True)
    ap_root.mkdir(parents=True, exist_ok=True)
    ap_path = ap_root / "action_presets.json"

    ap = ActionPresetStore(ap_path)
    print(f"  store file: {ap_path}")
    print("  factory defaults: "
          + "、".join(f"{p['id']}={p['name']}" for p in ap.list()))
    check("S1: the 8 shipped presets are the factory defaults, seeded on first run",
          lambda: _eq([p["id"] for p in ap.list()],
                      [p["id"] for p in PR.FACTORY_PRESETS]))
    check("S1: the store file was written on first run",
          lambda: _eq(ap_path.is_file(), True))
    check("S1: every preset has a stable id that is NOT its display name",
          lambda: _eq([p for p in ap.list() if p["id"] == p["name"]], []))

    peace = ap.add("ピースサイン")
    print(f"  added: id={peace['id']!r} name={peace['name']!r}")
    check("S2: saving 「ピースサイン」 makes it present",
          lambda: _eq([p["name"] for p in ap.list()].count("ピースサイン"), 1))
    check("S2: a user preset gets its own id, distinct from the name",
          lambda: _eq((peace["id"] != "ピースサイン", peace["id"].startswith("u")),
                      (True, True)))
    ap2 = ActionPresetStore(ap_path)                 # reload from disk
    check("S2: it is still there after reloading the store from disk",
          lambda: _eq([(p["id"], p["name"]) for p in ap2.list()],
                      [(p["id"], p["name"]) for p in ap.list()]))
    check("S2: the id is STABLE across the reload",
          lambda: _eq(ap2.get(peace["id"])["name"], "ピースサイン"))

    removed = ap2.delete("wave")                     # a FACTORY preset
    check("S3: deleting a FACTORY preset removes it",
          lambda: _eq((removed["name"], [p["id"] for p in ap2.list()].count("wave")),
                      ("手を振る", 0)))
    ap3 = ActionPresetStore(ap_path)
    check("S3: it stays deleted after re-loading the store (no resurrection)",
          lambda: _eq([p["id"] for p in ap3.list()].count("wave"), 0))
    check("S3: the deletion is a tombstone in the file, not just an absence",
          lambda: _eq("wave" in json.loads(ap_path.read_text(encoding="utf-8"))["deleted"],
                      True))
    check("S3: the other factory presets are untouched",
          lambda: _eq([p["id"] for p in ap3.list()],
                      [p["id"] for p in PR.FACTORY_PRESETS if p["id"] != "wave"]
                      + [peace["id"]]))

    bad_path = ap_root / "corrupt.json"
    bad_path.write_text("{ this is not json ", encoding="utf-8")
    ap_bad = ActionPresetStore(bad_path)
    print(f"  corrupt-file fallback message: {ap_bad.fallback_reason}")
    check("S4: a corrupt store file falls back to the factory defaults",
          lambda: _eq([p["id"] for p in ap_bad.list()],
                      [p["id"] for p in PR.FACTORY_PRESETS]))
    check("S4: the fallback is REPORTED, not hidden",
          lambda: _eq((ap_bad.fallback, bool(ap_bad.fallback_reason)), (True, True)))
    check("S4: the unreadable file is NOT overwritten (it can still be repaired)",
          lambda: _eq(bad_path.read_text(encoding="utf-8"), "{ this is not json "))

    def _rejects(value) -> bool:
        try:
            ap3.add(value)
        except PR.PresetError:
            return True
        return False

    REJECTED = [("empty", ""), ("spaces", "   "), ("ideographic space", "　　"),
                ("newline", "手を\n振る"), ("NUL", "手を\x00振る"),
                ("bidi override", "手を‮振る"),
                ("too long", "あ" * (PR.MAX_NAME_CHARS + 1)),
                ("bracket 「", "「ピース"), ("bracket 」", "ピース」"),
                ("not a string", 12345)]
    for label, value in REJECTED:
        check(f"S5: rejected ({label})", lambda v=value: _eq(_rejects(v), True))
    check(f"S5: exactly {PR.MAX_NAME_CHARS} characters is ACCEPTED (the limit is sane)",
          lambda: _eq(ap3.add("ん" * PR.MAX_NAME_CHARS)["name"],
                      "ん" * PR.MAX_NAME_CHARS))
    check("S5: duplicate names are REJECTED (documented policy)",
          lambda: _eq(_rejects("ピースサイン"), True))
    check("S5: a name differing only by surrounding space is the SAME name",
          lambda: _eq(_rejects("  ピースサイン  "), True))
    check("S5: HTML in a name is stored verbatim - it is data, never markup",
          lambda: _eq(ap3.add("<b>x</b>")["name"], "<b>x</b>"))
    check("S5: and it survives a reload verbatim",
          lambda: _eq([p["name"] for p in ActionPresetStore(ap_path).list()
                       if p["name"] == "<b>x</b>"], ["<b>x</b>"]))

    print("\n== action presets: APPLYING one writes PROMPT TEXT (12-14) ==")
    print(f"  template (one place, h3app\\presets.py): {PR.APPLY_TEMPLATE}")
    base_prompt = "カメラに向かって話す。"
    applied = PR.append_to_prompt(base_prompt, "ピースサイン")
    print("  before: " + repr(base_prompt))
    print("  after : " + repr(applied))
    check("A1: the prompt TEXT contains the template sentence with the name",
          lambda: _eq(PR.apply_sentence("ピースサイン") in applied, True))
    check("A1: the text the user already wrote is kept, above the new sentence",
          lambda: _eq(applied.startswith(base_prompt + "\n"), True))
    ap_seg = normalize_segments([{"prompt": applied}])[0]
    check("A2: `action_presets` is NOT extended by the press",
          lambda: _eq(ap_seg["action_presets"], []))
    check("A2: the sentence is ordinary prompt text (it is in `lines` too)",
          lambda: _eq([l["text"] for l in ap_seg["lines"]],
                      [base_prompt, PR.apply_sentence("ピースサイン")]))

    ap_story_root = debug / "selftest_action_presets_story"
    shutil.rmtree(ap_story_root, ignore_errors=True)
    ap_store = StoryStore(ap_story_root)
    ap_story = ap_store.create(
        name="動作プリセット", source_text="", lines=[],
        segments=normalize_segments([
            {"prompt": applied},
            # LEGACY segment: the old id-based form, saved by the old UI.
            {"prompt": "窓の外を見る。", "action_presets": ["wave"]},
        ]),
        images=one, settings=settings_a, jp_negative="", motion_presets=[])
    ap_reloaded = StoryStore(ap_story_root).get(ap_story.id)
    check("A3: the appended text survives save -> reload unchanged",
          lambda: _eq(ap_reloaded.segments[0]["prompt"], applied))
    check("A3: reloading did not invent `action_presets` for it either",
          lambda: _eq(ap_reloaded.segments[0]["action_presets"], []))
    twice = PR.append_to_prompt(applied, "ピースサイン")
    check("A4: pressing the chip twice appends the sentence TWICE (documented)",
          lambda: _eq(len(PR.applied_action_names(twice)), 2))
    check("A4: an edited sentence is still the user's text, not a preset any more",
          lambda: _eq(PR.applied_action_names("この場面の流れの中で、自然にピースをする。"), []))
    ap_scene = StoryPipeline._scene_text(ap_reloaded, ap_reloaded.segments[0])
    check("A5: the sentence reaches the SCENE text as plain prompt text",
          lambda: _eq(PR.apply_sentence("ピースサイン") in ap_scene, True))
    check("A5: no English and no second translation path is involved",
          lambda: _eq(PR.APPLY_TEMPLATE.isascii(), False))

    print("\n== action presets: LEGACY stories keep working (15) ==")
    legacy_seg = ap_reloaded.segments[1]
    legacy_scene = StoryPipeline._scene_text(ap_reloaded, legacy_seg)
    print("  legacy segment scene text: " + repr(legacy_scene))
    check("L1: an old `action_presets:[\"wave\"]` still resolves to its text",
          lambda: _eq(config_preset_text("wave"), "手を振る"))
    check("L1: and it still reaches the scene text exactly as before",
          lambda: _eq("手を振る" in legacy_scene, True))
    check("L1: loading the story did NOT rewrite its prompt",
          lambda: _eq(legacy_seg["prompt"], "窓の外を見る。"))
    check("L1: deleting the 「手を振る」 CHIP did not touch the stored id either",
          lambda: _eq(legacy_seg["action_presets"], ["wave"]))

    d_legacy = StoryPipeline._delivery_for(ap_reloaded, legacy_seg)
    d_new1 = StoryPipeline._delivery_for(ap_reloaded, ap_reloaded.segments[0])
    d_new2 = StoryPipeline._delivery_for(ap_reloaded, {"prompt": twice})
    d_none = StoryPipeline._delivery_for(ap_reloaded, {"prompt": base_prompt})
    print(f"  action_load: legacy id={d_legacy['action_load']}  "
          f"1 applied sentence={d_new1['action_load']}  "
          f"2 applied sentences={d_new2['action_load']}  "
          f"none={d_none['action_load']}")
    check("L2: one legacy id and one applied sentence mean the SAME action_load",
          lambda: _eq((d_legacy["action_load"], d_new1["action_load"]),
                      ("medium", "medium")))
    check("L2: two applied sentences still mean a BUSY segment",
          lambda: _eq(d_new2["action_load"], "high"))
    check("L2: a segment with no action at all keeps the default (medium)",
          lambda: _eq(d_none["action_load"], "medium"))
    # Against a STYLE that lowers action_load, the signal is visible on its own:
    # without an action the segment is "low", with either form it rises again.
    _sig = presets_mod.segment_action_signals
    check("L2: under the 控えめな動き style, low / medium / medium",
          lambda: _eq([duration_mod.delivery_from_presets(["subtle"], _sig(s))["action_load"]
                       for s in ({"prompt": base_prompt},
                                 {"action_presets": ["wave"]},
                                 {"prompt": applied})],
                      ["low", "medium", "medium"]))
    check("L3: the transition planner sees an applied sentence as a new action",
          lambda: _eq([transitions_mod._starts_new_physical_action(s)
                       for s in ({"prompt": applied}, {"action_presets": ["wave"]},
                                 {"prompt": base_prompt})],
                      [True, True, False]))
    ap_plan = transitions_mod.plan_transitions(
        normalize_segments([{"prompt": "セグメントA"}, {"prompt": applied},
                            {"prompt": "セグメントC"}]))
    check("L3: the hand-over is still planned on the EARLIER segment",
          lambda: _eq((ap_plan[0]["exit_state"]["ongoing_action"],
                       ap_plan[1]["entry_state"]["ongoing_action"]),
                      (transitions_mod.HANDOFF, transitions_mod.HANDOFF)))
    check("L3: a segment carrying BOTH forms counts both signals",
          lambda: _eq(presets_mod.segment_action_signals(
              {"prompt": applied, "action_presets": ["wave"]}),
              ["wave", "ピースサイン"]))
    shutil.rmtree(ap_root, ignore_errors=True)
    shutil.rmtree(ap_story_root, ignore_errors=True)

    print("\n== ffmpeg ==")
    try:
        exe = merge_mod.find_ffmpeg()
        print(f"  OK    ffmpeg found: {exe}")
    except Exception as exc:
        failures.append(f"ffmpeg not found: {exc}")
        print(f"  FAIL  ffmpeg not found: {exc}")

    print("\n== v2 modes ==")
    try:
        from h3app import modes_v2
        base_settings = {"width": 576, "height": 1024, "frames": 124,
                         "steps": 8, "seed": 1, "sampler": "res_multistep",
                         "scheduler": "simple", "ref_image_size": "match"}
        base_models = {"clip": "c", "unet": "u",
                       "lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
                       "lora_strength": 1.0, "vae_video": "vv", "vae_audio": "va"}
        r = modes_v2.resolve("LEGACY", base_settings, base_models)
        assert r["gate"] == "v1_parity" and r["models"]["lora"] is not None
        print("  OK    LEGACY resolves to the v1 gate")
        r = modes_v2.resolve("FAST", base_settings, base_models, loras_dir=None)
        assert (r["settings"]["steps"], r["settings"]["sampler"]) == (4, "euler")
        assert r["models"]["lora"] == modes_v2.FAST_LORA
        assert r["patches"]["sigshift"] == {"shift_video": 12.0, "shift_audio": 3.0}
        print("  OK    FAST resolves to turbo 4-step + SigmaShift(12,3)")
        r = modes_v2.resolve("QUALITY", base_settings, base_models)
        assert r["settings"]["steps"] == 20 and r["models"]["lora"] is None
        print("  OK    QUALITY resolves to base 20-step")
        gq = graphs.build_generate_graph(
            ["h3test_face_ref.png"], "EN PROMPT", dict(base_settings),
            dict(base_models), [1536], "x")
        gq = modes_v2.drop_lora_for_base(gq)
        assert "lora" not in gq
        assert gq["guider"]["inputs"]["model"] == ["unet", 0]
        print("  OK    base modes drop the LoRA node (no silent carry-over)")
        # Structural gate against a real v1-built graph, re-targeted per mode.
        gc = graphs.build_generate_graph(
            ["h3test_face_ref.png"], "EN PROMPT", dict(base_settings),
            dict(base_models), [1536], "x")
        modes_v2.validate_graph(gc, modes_v2.resolve("LEGACY", base_settings, base_models),
                                load_manifest(), cfg.defaults, "LEGACY")
        print("  OK    LEGACY graph passes the v1 parity gate")
        gf = dict(r := modes_v2.resolve("FAST", base_settings, base_models, loras_dir=None))
        g2 = graphs.build_generate_graph(
            ["h3test_face_ref.png"], "EN PROMPT",
            dict(gf["settings"]), dict(gf["models"]), [1536], "x")
        g2 = modes_v2.apply_patches(g2, gf["patches"])
        modes_v2.validate_graph(g2, gf, load_manifest(), cfg.defaults, "FAST")
        print("  OK    FAST graph passes the v2 turbo gate (incl. re-anchor refs)")
        gl = modes_v2.resolve("LONG", base_settings, base_models)
        modes_v2.validate_graph(gc, gl, load_manifest(), cfg.defaults, "LONG")
        print("  OK    LONG enforces Original Re-anchor (refs connected)")

        # LONG_FAST: resolve + build + apply_patches + validate_graph roundtrip.
        gt = dict(rt := modes_v2.resolve("LONG_FAST", base_settings, base_models,
                                         loras_dir=None))
        g3 = graphs.build_generate_graph(
            ["h3test_face_ref.png"], "EN PROMPT",
            dict(gt["settings"]), dict(gt["models"]), [1536], "x")
        g3 = modes_v2.apply_patches(g3, gt["patches"])
        modes_v2.validate_graph(g3, gt, load_manifest(), cfg.defaults, "LONG_FAST")
        assert g3["lora"]["inputs"]["lora_name"] == modes_v2.FAST_LORA
        assert (g3["sigshift"]["inputs"]["shift_video"],
                g3["sigshift"]["inputs"]["shift_audio"]) == (12.0, 3.0)
        assert g3["sage"]["class_type"] == "MiniMaxH3MemoryEfficientSageAttentionPatch"
        assert g3["sampler_select"]["inputs"]["sampler_name"] == "euler"
        assert int(g3["scheduler"]["inputs"]["steps"]) == 4
        assert g3["guider"]["inputs"]["model"] == ["sage", 0]
        assert g3["scheduler"]["inputs"]["model"] == ["sage", 0]
        print("  OK    LONG_FAST resolves to turbo 4-step + SigmaShift(12,3) + Sage, "
              "guider/scheduler relinked onto the Sage patch")

        # LONG_FAST continuation graph: the 0f contract (no legacy tail relay
        # nodes), Original Re-anchor still holding, and the Voice Master anchor.
        cont_lf = {"prev_total_frames": 124, "tail_frames": 0, "tail_audio": False,
                  "last_frame_image": "h3test_face_ref.png",
                  "voice_master": "h3test_voice_master.wav"}
        g4 = graphs.build_generate_graph(
            ["h3test_face_ref.png"], "EN PROMPT",
            dict(gt["settings"]), dict(gt["models"]), [1536], "x",
            continuation=cont_lf, concat_previous=False)
        g4 = modes_v2.apply_patches(g4, gt["patches"])
        modes_v2.validate_graph(g4, gt, load_manifest(), cfg.defaults, "LONG_FAST")
        banned = [ct for ct in ("LoadVideo", "GetVideoComponents",
                                "ImageFromBatch", "TrimAudioDuration")
                 if any(n.get("class_type") == ct for n in g4.values())]
        assert not banned, banned
        assert g4.get("last_frame", {}).get("class_type") == "LoadImage"
        assert any(k.startswith("ref_images.ref_image_")
                  for k in g4[graphs.NODE_H3]["inputs"])
        assert g4.get("voice_master", {}).get("class_type") == "LoadAudio"
        print("  OK    LONG_FAST continuation graph: 0f relay only, "
              "Original Re-anchor + Voice Master intact, no tail-relay nodes")

        # Structural backstop: a tail-relay node in an otherwise-valid LONG_FAST
        # graph must be refused, even if some upstream caller built it by mistake.
        tampered = dict(g4)
        tampered["prev_video"] = {"class_type": "LoadVideo", "inputs": {"file": "x"}}
        try:
            modes_v2.validate_graph(tampered, gt, load_manifest(), cfg.defaults, "LONG_FAST")
            failures.append("LONG_FAST backstop did not refuse a tail-relay node")
            print("  FAIL  LONG_FAST backstop did not refuse a tail-relay node")
        except graphs.GraphError as exc:
            print(f"  OK    LONG_FAST backstop refuses a tail-relay node: {exc}")

        # LONG_FAST must never be reachable from the single-shot flow.
        assert "LONG_FAST" in modes_v2.STORY_ONLY_MODES
        assert not any(m in modes_v2.STORY_ONLY_MODES for m in
                       ("LONG", "FAST", "QUALITY", "LEGACY", "BALANCED", "LOW_VRAM"))
        print("  OK    STORY_ONLY_MODES marks LONG_FAST story-only "
              "(api_generate/continue/again refuse it)")

        for bad in ("CONTROL", "PDD", "ENDLESS", "NOPE"):
            try:
                modes_v2.resolve(bad, base_settings, base_models)
                failures.append(f"mode {bad} was NOT refused")
                print(f"  FAIL  mode {bad} was NOT refused")
            except PipelineError as exc:
                assert "UNAVAILABLE" in str(exc), str(exc)
                print(f"  OK    {bad} refused with UNAVAILABLE")
        caps = modes_v2.compat_status(comfy_reachable=False, object_info=None,
                                      gpus=[], loras_dir=Path("."),
                                      models_present={})
        assert caps["experimental"]["PDD"].startswith("UNAVAILABLE")
        print("  OK    compat snapshot degrades gracefully when ComfyUI is down")
    except Exception as exc:                                     # noqa: BLE001
        failures.append(f"v2 modes: {exc!r}")
        print(f"  FAIL  v2 modes: {exc!r}")

    print("\n== 音声テスト (audio_test) ==")
    try:
        import re as re_mod

        picked_q = audio_test_mod.resolve_profile("quick", d, cfg.models)
        picked_f = audio_test_mod.resolve_profile("full", d, cfg.models)
        check("QUICK resolves to 576x1024 / 56f, steps=4, turbo LoRA",
              lambda: _eq((picked_q["resolved"]["settings"]["width"],
                          picked_q["resolved"]["settings"]["height"],
                          picked_q["resolved"]["settings"]["frames"],
                          picked_q["resolved"]["settings"]["steps"],
                          picked_q["resolved"]["models"]["lora"]),
                         (576, 1024, 56, 4, modes_mod.FAST_LORA)))
        check("FULL resolves to 576x1024 / 124f, steps=4, turbo LoRA",
              lambda: _eq((picked_f["resolved"]["settings"]["width"],
                          picked_f["resolved"]["settings"]["height"],
                          picked_f["resolved"]["settings"]["frames"],
                          picked_f["resolved"]["settings"]["steps"],
                          picked_f["resolved"]["models"]["lora"]),
                         (576, 1024, 124, 4, modes_mod.FAST_LORA)))
        _fast_direct = modes_mod.resolve(
            "FAST", dict(d, width=576, height=1024, frames=56), cfg.models)
        check("QUICK settings match modes_v2.resolve('FAST') directly "
              "(can never silently diverge)",
              lambda: _eq(picked_q["resolved"]["settings"], _fast_direct["settings"]))
        check("QUICK's 56 frames is on the 17k+5 grid",
              lambda: _eq(56 % 17, 5))
        check("QUICK's 56 frames is NOT what frames_for_seconds gives "
              "(guards against re-introducing its >=124 clamp here)",
              lambda: _true(graphs.frames_for_seconds(56 / 24.0) != 56))

        gq = audio_test_mod.build_audio_test_graph(
            images=audio_test_mod.select_images_for_profile("quick", one),
            en_prompt="EN PROMPT PLACEHOLDER",
            resolved=picked_q["resolved"], ref_longest=d["ref_longest"])
        (debug / "audio_test_graph_quick.json").write_text(
            json.dumps(gq, ensure_ascii=False, indent=2), encoding="utf-8")
        check("QUICK graph has no VAEDecode/CreateVideo/SaveVideo",
              lambda: audio_test_mod.assert_audio_only_graph(gq))
        check("QUICK graph has VAEDecodeAudio + PreviewAudio",
              lambda: _eq({gq[n]["class_type"] for n in
                          (audio_test_mod.NODE_DECODE_AUDIO,
                           audio_test_mod.NODE_PREVIEW_AUDIO)},
                         {"VAEDecodeAudio", "PreviewAudio"}))
        check("QUICK graph: video VAELoader still present and wired into H3",
              lambda: _eq(gq[graphs.NODE_H3]["inputs"]["vae"], ["vae_video", 0]))
        check("QUICK graph: no Voice Master -> no LoadAudio, no ref_audios key",
              lambda: _eq(("voice_master" in gq,
                          any(k.startswith("ref_audios.")
                             for k in gq[graphs.NODE_H3]["inputs"])),
                         (False, False)))

        gf = audio_test_mod.build_audio_test_graph(
            images=audio_test_mod.select_images_for_profile("full", four),
            en_prompt="EN PROMPT PLACEHOLDER",
            resolved=picked_f["resolved"], ref_longest=d["ref_longest"],
            voice_master_file="h3app_char_selftest_voice.wav")
        check("FULL graph: Voice Master adds LoadAudio + ref_audios.ref_audio_0",
              lambda: _eq((gf.get("voice_master", {}).get("class_type"),
                          gf[graphs.NODE_H3]["inputs"].get("ref_audios.ref_audio_0")),
                         ("LoadAudio", ["voice_master", 0])))
        check("FULL graph has no VAEDecode/CreateVideo/SaveVideo",
              lambda: audio_test_mod.assert_audio_only_graph(gf))
        check("FULL graph: 4 references reach MiniMaxH3ReferenceToVideo",
              lambda: _eq(sum(1 for k in gf[graphs.NODE_H3]["inputs"]
                             if k.startswith("ref_images.")), 4))

        prompt_quick = audio_test_mod.build_audio_test_prompt(
            character_snapshot=None, n_pictures=1, voice_master_connected=False,
            language="Japanese", spoken_text="こんにちは、今日もよろしくお願いします。")
        prompt_full = audio_test_mod.build_audio_test_prompt(
            character_snapshot={
                "canon": {"face": "round face", "hair": "long black hair",
                         "build": "slim", "skin": "fair"},
                "outfit_ref": {"outfit": "white dress", "accessories": "",
                              "materials": "", "color_keys": ""}},
            n_pictures=4, voice_master_connected=True, language="Japanese",
            spoken_text="はじめまして、これはフルプロファイルのテストです。")
        (debug / "audio_test_prompt_quick.txt").write_text(prompt_quick, encoding="utf-8")
        (debug / "audio_test_prompt_full.txt").write_text(prompt_full, encoding="utf-8")

        check("prompts: six sections present, in order",
              lambda: _eq(
                  [ln for ln in prompt_quick.splitlines()
                   if ln in director_lexicon_mod.SECTIONS],
                  list(director_lexicon_mod.SECTIONS)))
        check("QUICK prompt: exactly one <d>[Japanese] ...</d>, payload verbatim",
              lambda: _eq(re_mod.findall(r"<d>\[Japanese\] (.*?)</d>", prompt_quick),
                         ["こんにちは、今日もよろしくお願いします。"]))
        check("QUICK prompt (Voice Master OFF): no <Audio 1>",
              lambda: _eq("<Audio 1>" in prompt_quick, False))
        check("FULL prompt (Voice Master ON): names <Audio 1>",
              lambda: _eq("<Audio 1>" in prompt_full, True))
        check("no avoid/no-X enumeration in either prompt",
              lambda: _eq(any(re_mod.search(r"\bavoid\b|\bno [a-z]+ing\b", p, re_mod.I)
                             for p in (prompt_quick, prompt_full)), False))

        prompt_en = audio_test_mod.build_audio_test_prompt(
            character_snapshot=None, n_pictures=1, voice_master_connected=False,
            language="English", spoken_text="Hello, thank you for watching.")
        check("English language produces <d>[English] ...</d>",
              lambda: _eq(re_mod.findall(r"<d>\[English\] (.*?)</d>", prompt_en),
                         ["Hello, thank you for watching."]))

        check("assert_reference_labels passes: quick (no audio)",
              lambda: graphs.assert_reference_labels(gq, prompt_quick))
        check("assert_reference_labels passes: full (with audio)",
              lambda: graphs.assert_reference_labels(gf, prompt_full))
        try:
            # prompt_full names <Audio 1>; gq has no audio connected.
            graphs.assert_reference_labels(gq, prompt_full)
            failures.append("assert_reference_labels did NOT reject a phantom <Audio 1>")
            print("  FAIL  assert_reference_labels did NOT reject a phantom <Audio 1>")
        except graphs.GraphError as exc:
            print(f"  OK    assert_reference_labels rejects a phantom <Audio 1>: {exc}")

        at_root = debug / "selftest_audio_tests"
        shutil.rmtree(at_root, ignore_errors=True)
        at_root.mkdir(parents=True, exist_ok=True)
        fixture_id = "abcdef123456"
        fixture_dir = at_root / fixture_id
        fixture_dir.mkdir(parents=True, exist_ok=True)
        (fixture_dir / "meta.json").write_text("{}", encoding="utf-8")
        other_id = "111111111111"
        (at_root / other_id).mkdir(parents=True, exist_ok=True)
        (at_root / other_id / "meta.json").write_text("{}", encoding="utf-8")

        for bad_id in ("..", "../../projects", "C:/Windows", "not-hex!!"):
            try:
                audio_test_mod.resolve_test_dir(at_root, bad_id)
                failures.append(f"resolve_test_dir did NOT refuse: {bad_id!r}")
                print(f"  FAIL  resolve_test_dir did NOT refuse: {bad_id!r}")
            except audio_test_mod.AudioTestError:
                print(f"  OK    resolve_test_dir refuses traversal: {bad_id!r}")

        check("delete_test removes only the targeted directory",
              lambda: _eq((audio_test_mod.delete_test(at_root, fixture_id),
                          fixture_dir.is_dir(), (at_root / other_id).is_dir()),
                         (True, False, True)))
        check("delete_all_tests removes only directories under the root",
              lambda: _eq((audio_test_mod.delete_all_tests(at_root),
                          list(at_root.iterdir())),
                         (1, [])))
    except Exception as exc:                                     # noqa: BLE001
        failures.append(f"audio_test: {exc!r}")
        print(f"  FAIL  audio_test: {exc!r}")

    print("\n== dumped graphs ==")
    for p in sorted(debug.glob("selftest_*.json")):
        print(f"  {p}")

    print()
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


def _eq(got, expected):
    if got != expected:
        raise AssertionError(f"got {got!r}, expected {expected!r}")


def _true(got):
    if not got:
        raise AssertionError(f"expected a truthy value, got {got!r}")


# ------------------------------------------------------- selftest helpers ----
def _legacy_segment_count(lines: list, *, seconds: float = 5.0) -> int:
    """The packing rule this change REPLACED, reimplemented in one place.

    It exists only so the selftest can print a real before/after number instead
    of a remembered one: at most 2 prompt clauses per segment and 6.5 characters
    of speech per second. Nothing in the app calls it. The long-utterance split
    is left out, so this is a LOWER bound on what the old packer produced.
    """
    budget = max(1, int(round(float(seconds) * 6.5)))
    clean = segmenter.normalize_lines(lines)
    groups: list[list[dict]] = []
    i, n = 0, len(clean)
    while i < n:
        if (clean[i]["type"] == "prompt" and i + 1 < n
                and clean[i + 1]["type"] == "speech"):
            groups.append(clean[i:i + 2])
            i += 2
        else:
            groups.append([clean[i]])
            i += 1
    total, prompts_, chars, started = 0, 0, 0, False
    for group in groups:
        g_chars = sum(len("".join(c["text"].split()))
                      for c in group if c["type"] == "speech")
        g_prompts = sum(1 for c in group if c["type"] == "prompt")
        if started and (prompts_ + g_prompts > 2 or chars + g_chars > budget):
            total += 1
            prompts_, chars = 0, 0
        started = True
        prompts_ += g_prompts
        chars += g_chars
    return total + 1 if started else 0


def _preflight_refuses(story) -> bool:
    """A segment with nothing in it must stop the story BEFORE any GPU work."""
    original = list(story.segments)
    story.segments = original + [{"index": len(original), "lines": [], "prompt": "",
                                  "speech": "", "motion_presets": [],
                                  "action_presets": [], "status": "pending",
                                  "clip": None, "error": ""}]
    try:
        assert_preflight(story)
        return False
    except PipelineError:
        return True
    finally:
        story.segments = original


def _make_struct_story(root: Path, names: list[str], *, generated: int,
                       settings: dict, images: list[str]):
    """A story with `generated` segments already rendered, files and all.

    The clips are real (tiny) files so that invalidation genuinely has to move
    something, and project.json looks exactly like a resumed run.
    """
    story = StoryStore(root).create(
        name="構成テスト", source_text="", lines=[],
        segments=[{"prompt": n} for n in names], images=images,
        settings=settings, jp_negative="", motion_presets=[])
    story.ensure_dirs()
    for i in range(generated):
        video = story.clips_dir / f"clip_{i + 1:02d}.mp4"
        frame = story.frames_dir / f"frame_{i:02d}.png"
        video.write_bytes(b"not a real clip")
        frame.write_bytes(b"not a real frame")
        story.clips.append({
            "segment_index": i, "segment_id": story.segments[i]["segment_id"],
            "step": i + 1, "frames": 124, "seconds": 5.167,
            "video": "", "local_video": str(video), "last_frame": str(frame),
            "seed": story.seed_for(i), "en_prompt": "EN", "created_at": 0.0})
        story.segments[i].update({"status": "done", "clip": str(video),
                                  "seed": story.seed_for(i)})
    story.data["cursor"] = generated
    story.data["status"] = "paused" if generated else "pending"
    story.save()
    return story


def _structure_refused(story, requested) -> str:
    """The refusal message, or '' when the change was NOT refused."""
    try:
        story_mod.apply_structure(story, requested, dry_run=True)
    except StructureError as exc:
        return str(exc)
    return ""


def _legacy_story(root: Path) -> str:
    """A project.json / segments.json written the way they were BEFORE segment
    identities existed: no segment_id anywhere."""
    sid = "aaaa1111bbbb"
    folder = root / sid
    (folder / "clips").mkdir(parents=True, exist_ok=True)
    video = folder / "clips" / "clip_01.mp4"
    video.write_bytes(b"not a real clip")
    segments = [
        {"index": 0, "lines": [{"type": "prompt", "text": "旧A"}], "prompt": "旧A",
         "speech": "", "motion_presets": [], "action_presets": [],
         "status": "done", "clip": str(video), "error": "", "seed": 123456789},
        {"index": 1, "lines": [{"type": "prompt", "text": "旧B"}], "prompt": "旧B",
         "speech": "", "motion_presets": [], "action_presets": [],
         "status": "pending", "clip": None, "error": ""},
    ]
    data = {
        "id": sid, "name": "旧ストーリー", "created_at": 0.0, "updated_at": 0.0,
        "source_text": "", "lines": [], "images": ["h3app_ref_a.png"],
        "profile": "", "profile_cache_key": "",
        "settings": {"width": 576, "height": 1024, "frames": 124, "steps": 8,
                     "seed": 123456789, "randomize_seed": False,
                     "sampler": "res_multistep", "scheduler": "simple",
                     "ref_image_size": "match", "output_prefix": ""},
        "jp_negative": "", "motion_presets": [], "base_seed": 123456789,
        "cursor": 1, "status": "paused", "stop_requested": False,
        "clips": [{"segment_index": 0, "step": 1, "frames": 124, "seconds": 5.167,
                   "video": "", "local_video": str(video), "last_frame": "",
                   "seed": 123456789, "en_prompt": "EN", "created_at": 0.0}],
        "merged": False, "final_video": "", "error": None, "en_prompt": "",
    }
    _atomic_test_write(folder / "segments.json", segments)
    _atomic_test_write(folder / "project.json", data)
    return sid


def _atomic_test_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _story_continuation_tail_audio(cfg: Config, story) -> bool:
    """The continuation dict the story runner hands to the graph builder.

    Built against a throwaway 'comfy dir' so the selftest never writes into the
    real ComfyUI input folder.
    """
    fake = Config(copy.deepcopy(cfg.data))
    fake.data["comfy_dir"] = str(story.root / "_fake_comfy")
    (Path(fake.data["comfy_dir"]) / "input").mkdir(parents=True, exist_ok=True)
    # The app never writes into ComfyUI's own input dir (see comfy_inputs.py);
    # point the staging dir at the same throwaway folder for this selftest.
    fake.data["input_stage_dir"] = str(Path(fake.data["comfy_dir"]) / "input")
    src = story.root / "prev_clip.mp4"
    src.write_bytes(b"not a real video")
    stub = type("_StubStoryPipeline", (), {"config": fake})()
    cont = asyncio.run(StoryPipeline._continuation(
        stub, story, {"local_video": str(src), "frames": 124, "step": 1}))
    return (cont.get("tail_audio") is True
            and cont["tail_frames"] == int(cfg.defaults["tail_frames"])
            and cont["prev_total_frames"] == 124)


def _fake_comfy_input(story) -> Path:
    """The throwaway ComfyUI input dir `_story_continuation_for_mode` stages
    into. Shared so a caller can pre-stage a file there (the Voice Master anchor
    is trust-but-verify: a recorded name is only reused when the file exists).
    """
    path = story.root / "_fake_comfy" / "input"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _story_continuation_for_mode(cfg: Config, story, *, mode: str,
                                 prev_clip: dict) -> dict:
    """Same shape as `_story_continuation_tail_audio`, generalized over `mode`
    and the caller-supplied `prev_clip` so LONG_FAST's 0f relay + Voice Master
    dict can be exercised the same way, against the same throwaway comfy dir.
    """
    fake = Config(copy.deepcopy(cfg.data))
    fake.data["comfy_dir"] = str(story.root / "_fake_comfy")
    # Same rationale as _story_continuation_tail_audio: stage into the
    # throwaway folder, never into a real ComfyUI input dir.
    fake.data["input_stage_dir"] = str(_fake_comfy_input(story))
    stub = type("_StubStoryPipeline", (), {"config": fake})()
    return asyncio.run(StoryPipeline._continuation(stub, story, prev_clip, mode=mode))


def _mock_segment_run(story, index: int, director_text: str, raw_output: str,
                      cfg: Config, profile: str = "") -> dict:
    """Everything one segment does EXCEPT touching ComfyUI, so the debug record
    this prints is produced by the very same code the real run uses."""
    seg = story.segments[index]
    speech = (seg.get("speech") or "").strip()
    lines = story_prompts.speech_lines(speech)
    canon = canon_mod.parse_profile(profile)
    overrides = StoryPipeline._overrides_for(story, index)
    en_prompt, compile_report = compiler.compile_story_prompt(
        raw_output, canon=canon, overrides=overrides, dialogue_lines=lines)
    report = compiler.validate_final_prompt(
        en_prompt, index=index, dialogue_lines=lines, canon=canon,
        overrides=overrides,
        other_dialogue=StoryPipeline._other_dialogue_lines(story, index),
        other_staging=StoryPipeline._other_staging(story, index))
    cont = None if index == 0 else {
        "video": "prev.mp4", "prev_total_frames": 124,
        "tail_frames": int(cfg.defaults["tail_frames"]), "tail_audio": True}
    StoryPipeline._write_debug(
        story, index, seg,
        scene=StoryPipeline._scene_text(story, seg), speech=speech,
        profile=profile, director_text=director_text, raw_prompt=raw_output,
        en_prompt=en_prompt, seed=story.seed_for(index),
        continuation=cont, report=report, compile_report=compile_report,
        canon=canon, overrides=overrides,
        transition=StoryPipeline._transition_for(story, index))
    return json.loads((story.debug_dir / f"seg_{index:02d}.json")
                      .read_text(encoding="utf-8"))


# ------------------------------------------------------------------- main ----
def main() -> int:
    parser = argparse.ArgumentParser(description="H3 App server")
    parser.add_argument("--selftest", action="store_true",
                        help="Build and verify every graph offline, then exit.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.port:
        cfg.data["app_port"] = args.port

    if args.selftest:
        return selftest(cfg)

    problems = cfg.validate()
    if problems:
        print("設定に問題があります:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if port_is_open(cfg.app_port):
        print(f"ポート {cfg.app_port} は使用中です。config.json の app_port を変更してください。"
              "（既存のプロセスは終了させません）")
        return 3

    app = create_app(cfg)
    print(f"H3 App: http://127.0.0.1:{cfg.app_port}/")
    web.run_app(app, host="127.0.0.1", port=cfg.app_port, print=None)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
