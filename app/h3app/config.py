# -*- coding: utf-8 -*-
"""Config load / merge / path validation.

Every default here is the H3_HETERO_V1 measured value. config.json may override
them, but the app tells the user (and assert_v1_parity refuses to pretend) when a
run is no longer the verified profile.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent                      # <H3_PROJECT_ROOT>
MANIFEST_PATH = PROJECT_ROOT / "H3_HETERO_V1_MANIFEST.json"
BUILD_DIR = PROJECT_ROOT / "build"
BARRIER_NODES_DIR = PROJECT_ROOT / "custom_nodes"

DEFAULT_CONFIG = {
    # Placeholder only: every installation sets comfy_dir in app/config.json
    # (config.validate() refuses to start until the folder exists).
    "comfy_dir": r"C:\ComfyUI",
    "comfy_python": "",
    # 8401/8402 are reserved by the verified runners; never reuse them here or a
    # measurement run and an app run could attach to each other's server.
    "comfy_port": 8411,
    "app_port": 8790,
    "auto_launch_comfy": True,
    # Legacy fallback only. Device flags here are stripped and replaced by the
    # saved GPU settings (h3app.gpu.build_plan -> --cuda-device <UUID,UUID>),
    # which keep both GPUs visible so SelectCLIPDevice(gpu:1) still works.
    "comfy_launch_args": ["--default-device", "0", "--reserve-vram", "1.0"],
    # Extra flags appended ONLY for the fast launch profile (FAST / LONG_FAST).
    # Overnight 2026-09-07 parity result: --fast changes LEGACY output bytes
    # (video+audio md5 differ), so it must never ride the base profile.
    # Empty list = base args + ["--fast"]. Non-empty replaces the addition.
    "comfy_launch_args_fast": [],
    # User-facing media root. Finals are exported here as copies; internal
    # working files (ComfyUI input/output, project working copies) stay put.
    # "" means PROJECT_ROOT / "H3_Media" (see the media_root property below).
    "media_root": "",
    "whitelist_custom_nodes": [],
    "output_prefix": "H3/APP",
    # Tab-closed watchdog. The browser pings /api/heartbeat; when it stops for
    # longer than this the app saves, stops after the CURRENT clip, releases the
    # models and (optionally) stops a ComfyUI it launched itself.
    "idle_timeout_seconds": 90,
    "stop_comfy_when_idle": True,
    "story": {
        # 30 SECOND default = 10 s x 3 clips (shootout 2026-09-07: 10x3 beats
        # 15x2 on background stability and total time).
        "seconds_per_segment": 10,
        "max_segments": 24,
    },
    "vlm": {
        "model": "gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf",
        "mmproj": "gemma-4-E4B-it-mmproj-BF16.gguf",
        "chat_handler": "Gemma4",
        "n_ctx": 16384,
    },
    "defaults": {
        "width": 576,
        "height": 1024,
        "frames": 124,
        "steps": 8,
        "seed": 123456789,
        "randomize_seed": False,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "ref_image_size": "match",
        "fps": 24,
        "seconds": 5,
        "max_seconds": 15,
        "tail_frames": 56,
        "ref_longest": [1536, 1152, 1152, 896],
    },
    "models": {
        "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        # v1 parity: the RAW turbo LoRA. All 518 keys fail to bind in this
        # environment, so it is effectively a no-op - and that is exactly the
        # state the 217.61 s measurement and the human quality check were made
        # in. Do NOT swap this for the *_pruned_comfyui build.
        "lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "lora_strength": 1.0,
        "vae_video": "minimax_h3_video_vae_fp16.safetensors",
        "vae_audio": "minimax_h3_audio_vae_fp32.safetensors",
    },
}

# 9:16 is the only measured framing. The other two are offered but flagged.
ASPECT_PRESETS = [
    {"id": "portrait", "label": "縦", "width": 576, "height": 1024, "measured": True},
    {"id": "landscape", "label": "横", "width": 1024, "height": 576, "measured": False},
    {"id": "square", "label": "スクエア", "width": 768, "height": 768, "measured": False},
]

SECONDS_PRESETS = [5, 10, 15]

# ストーリーモードの動きプリセット。text はそのままセグメントのプロンプト末尾に
# 日本語で連結され、ディレクターの SCENE 節に入る。"auto" は何も足さない。
#
# STYLE  = 全セグメントに足しても破綻しない「雰囲気・撮り方」。プロジェクト単位で
#          選べる（= 毎セグメントに連結される）。
# ACTION = 1回だけ起きる動作。プロジェクト単位に置くと毎クリップで手を振り続ける
#          ことになるので、必ず「そのセグメントだけ」に付く。
STYLE_PRESETS = [
    {"id": "auto", "label": "おまかせ", "text": "", "kind": "style"},
    {"id": "natural", "label": "自然な動き", "text": "自然な動き", "kind": "style"},
    {"id": "subtle", "label": "控えめな動き", "text": "控えめな動き", "kind": "style"},
    {"id": "static_camera", "label": "カメラ固定", "text": "カメラ固定", "kind": "style"},
    {"id": "calm", "label": "落ち着いた雰囲気", "text": "落ち着いた雰囲気", "kind": "style"},
]

ACTION_PRESETS = [
    {"id": "wave", "label": "手を振る", "text": "手を振る", "kind": "action"},
    {"id": "bow", "label": "お辞儀をする", "text": "お辞儀をする", "kind": "action"},
    {"id": "peace", "label": "ピースサインをする", "text": "ピースサインをする", "kind": "action"},
    {"id": "point", "label": "指を差す", "text": "指を差す", "kind": "action"},
    {"id": "sit", "label": "椅子に座る", "text": "椅子に座る", "kind": "action"},
    {"id": "tilt_head", "label": "首をかしげる", "text": "首をかしげる", "kind": "action"},
    {"id": "shy_smile", "label": "照れ笑いをする", "text": "照れ笑いをする", "kind": "action"},
    {"id": "kick_legs", "label": "足をばたつかせる", "text": "足をばたつかせる", "kind": "action"},
]

# 旧 UI / 旧 project.json にしか出てこない id。テキスト解決だけできればよいので
# 一覧には出さない（= 新規には選べない）が、スタイル扱いで残す。
LEGACY_STYLE_PRESETS = [
    {"id": "nod", "label": "軽くうなずく", "text": "軽くうなずく", "kind": "style"},
    {"id": "smile_talk", "label": "笑顔で話す", "text": "笑顔で話す", "kind": "style"},
]

# 旧名。プロジェクト単位のプリセット = STYLE のみ、という意味に変わった。
MOTION_PRESETS = STYLE_PRESETS

STYLE_PRESET_IDS = {p["id"] for p in STYLE_PRESETS} | {p["id"] for p in LEGACY_STYLE_PRESETS}
ACTION_PRESET_IDS = {p["id"] for p in ACTION_PRESETS}
MOTION_PRESET_TEXT = {p["id"]: p["text"]
                      for p in (STYLE_PRESETS + ACTION_PRESETS + LEGACY_STYLE_PRESETS)}


def is_action_preset(preset_id: str) -> bool:
    return str(preset_id) in ACTION_PRESET_IDS


def preset_text(preset_id: str) -> str:
    return (MOTION_PRESET_TEXT.get(str(preset_id)) or "").strip()


SAMPLERS = ["res_multistep", "euler", "euler_ancestral", "dpmpp_2m", "ddim", "uni_pc"]
SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta"]
REF_IMAGE_SIZES = ["match", "max"]

MAX_PIXEL_AREA = 1032192          # H3 canvas limit: 768 x 1344
FPS = 24                          # H3 is 24 fps. Not user-settable.


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(RuntimeError):
    pass


class Config:
    """Merged config plus the derived paths the rest of the app uses."""

    def __init__(self, data: dict, source: Path | None = None):
        self.data = data
        self.source = source

    # ------------------------------------------------------------ accessors --
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def comfy_dir(self) -> Path:
        return Path(self.data["comfy_dir"])

    @property
    def comfy_python(self) -> Path:
        raw = (self.data.get("comfy_python") or "").strip()
        if raw:
            return Path(raw)
        return self.comfy_dir / ".venv" / "Scripts" / "python.exe"

    @property
    def comfy_main(self) -> Path:
        return self.comfy_dir / "main.py"

    @property
    def comfy_input(self) -> Path:
        return self.comfy_dir / "input"

    @property
    def comfy_output(self) -> Path:
        return self.comfy_dir / "output"

    @property
    def comfy_temp(self) -> Path:
        """ComfyUI's temp dir (folder_paths default: <comfy_dir>/temp).

        PreviewAudio writes here (never to comfy_output); the audio test
        cleanup step needs this to resolve/verify the exact file it deletes.
        """
        return self.comfy_dir / "temp"

    @property
    def app_dir(self) -> Path:
        return APP_DIR

    @property
    def web_dir(self) -> Path:
        return APP_DIR / "web"

    @property
    def projects_dir(self) -> Path:
        return APP_DIR / "projects"

    @property
    def story_dir(self) -> Path:
        return APP_DIR / "story_projects"

    @property
    def debug_dir(self) -> Path:
        return APP_DIR / "_debug"

    # --------------------------------------------------- user media root ---
    @property
    def media_root(self) -> Path:
        return Path(self.data.get("media_root") or (PROJECT_ROOT / "H3_Media"))

    @property
    def media_images(self) -> Path:
        return self.media_root / "Images"

    @property
    def media_audio(self) -> Path:
        return self.media_root / "Audio"

    @property
    def media_videos(self) -> Path:
        return self.media_root / "Videos"

    @property
    def media_videos_final(self) -> Path:
        return self.media_videos / "完成動画"

    @property
    def media_videos_clips(self) -> Path:
        return self.media_videos / "クリップ"

    @property
    def media_projects(self) -> Path:
        return self.media_root / "Projects"

    @property
    def input_stage_dir(self) -> Path:
        """Where the app writes files it will hand to ComfyUI via /upload/image.

        The app never writes into ComfyUI's own input directory (see
        comfy_inputs.py); this is the app-owned staging area instead.
        """
        raw = (self.data.get("input_stage_dir") or "").strip()
        if raw:
            return Path(raw)
        return self.app_dir / "_comfy_input"

    @property
    def story(self) -> dict:
        return self.data.get("story") or {}

    @property
    def paths_yaml(self) -> Path:
        return APP_DIR / "comfy_paths.yaml"

    @property
    def comfy_port(self) -> int:
        return int(self.data["comfy_port"])

    @property
    def app_port(self) -> int:
        return int(self.data["app_port"])

    @property
    def defaults(self) -> dict:
        return self.data["defaults"]

    @property
    def models(self) -> dict:
        return self.data["models"]

    @property
    def vlm(self) -> dict:
        return self.data["vlm"]

    # ------------------------------------------------------------ behaviour --
    def ensure_dirs(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.story_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        for d in (self.media_images, self.media_audio, self.media_videos,
                  self.media_videos_final, self.media_videos_clips,
                  self.media_projects):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list = usable."""
        problems: list[str] = []
        if not self.comfy_dir.is_dir():
            problems.append(f"ComfyUI のフォルダが見つかりません: {self.comfy_dir}")
        if not self.comfy_main.is_file():
            problems.append(f"ComfyUI の main.py が見つかりません: {self.comfy_main}")
        if not self.comfy_python.is_file():
            problems.append(f"ComfyUI の Python が見つかりません: {self.comfy_python}")
        if not self.paths_yaml.is_file():
            problems.append(f"comfy_paths.yaml が見つかりません: {self.paths_yaml}")
        if not BARRIER_NODES_DIR.is_dir():
            problems.append(f"H3-Device-Barrier の親フォルダが見つかりません: {BARRIER_NODES_DIR}")
        if self.comfy_port in (8401, 8402):
            problems.append(
                "comfy_port に 8401/8402 は使えません（実測用ランナーの予約ポートです）。")
        if self.comfy_port == self.app_port:
            problems.append("comfy_port と app_port を同じ値にはできません。")
        return problems


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else (APP_DIR / "config.json")
    data = copy.deepcopy(DEFAULT_CONFIG)
    if p.is_file():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"config.json を読み込めません ({p}): {exc}") from exc
        if not isinstance(user, dict):
            raise ConfigError("config.json の中身は JSON オブジェクトである必要があります。")
        data = _deep_merge(DEFAULT_CONFIG, user)
    return Config(data, p if p.is_file() else None)


_CONFIG_WRITE_LOCK = threading.Lock()


def update_config_file(path: str | os.PathLike, updates: dict) -> dict:
    """Replace top-level keys in config.json, keeping every other key.

    Atomic (tmp + os.replace) and serialized, so a GPU save and an AI save
    arriving together cannot drop each other's section. Returns the written data.
    """
    if not isinstance(updates, dict):
        raise ConfigError("設定の更新内容が不正です。")
    p = Path(path)
    with _CONFIG_WRITE_LOCK:
        data: dict = {}
        if p.is_file():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ConfigError(f"config.json を読み込めません ({p}): {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError("config.json の中身は JSON オブジェクトである必要があります。")
            data = loaded
        for key, value in updates.items():
            data[key] = copy.deepcopy(value)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        return data


def load_manifest() -> dict:
    """H3_HETERO_V1_MANIFEST.json, read-only.

    assert_v1_parity compares against this file instead of hardcoded numbers so
    the app can never silently drift from the frozen v1 profile.
    """
    if not MANIFEST_PATH.is_file():
        raise ConfigError(f"v1 マニフェストが見つかりません: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def measured_seconds() -> float:
    try:
        return float(load_manifest()["measurements_seconds"]["v1_match"])
    except Exception:
        return 217.61
