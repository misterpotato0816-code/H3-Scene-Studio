# -*- coding: utf-8 -*-
"""ComfyUI input handoff: the app never writes into ComfyUI's own input dir.

Everything the app generates for a graph to load (LoadImage/LoadAudio/
LoadVideo, always named h3app_*) is written to `cfg.input_stage_dir` and
handed to ComfyUI right before submit through the official
`POST /upload/image` API (InputUploader). Legacy files that already exist
only in ComfyUI's `input` folder remain usable for reading (no upload
needed, and this module never writes there).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import aiohttp

from .errors import PipelineError

APP_PREFIX = "h3app_"


def safe_name(name) -> bool:
    """Plain basename, no separators/drive/.., must start with h3app_."""
    if not isinstance(name, str) or not name:
        return False
    if name != Path(name).name:
        return False
    if name in (".", "..") or ".." in name:
        return False
    if not name.startswith(APP_PREFIX):
        return False
    return True


def stage_path(cfg, name: str) -> Path:
    """Validated path under cfg.input_stage_dir; creates the stage dir."""
    if not safe_name(name):
        raise ValueError(f"invalid staged input name: {name!r}")
    stage_dir = Path(cfg.input_stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / name


def local_path(cfg, name: str) -> Path | None:
    """Where to READ `name` from: stage dir first, legacy comfy input else."""
    if not safe_name(name):
        return None
    staged = Path(cfg.input_stage_dir) / name
    if staged.is_file():
        return staged
    legacy = Path(cfg.comfy_input) / name
    if legacy.is_file():
        return legacy
    return None


def _unique_ordered(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def graph_input_names(graph: dict) -> list[str]:
    """Unique, ordered h3app_* names a graph loads via LoadImage/Audio/Video."""
    names: list[str] = []
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs") or {}
        if class_type == "LoadImage":
            value = inputs.get("image")
        elif class_type == "LoadAudio":
            value = inputs.get("audio")
        elif class_type == "LoadVideo":
            value = inputs.get("file")
        else:
            continue
        if isinstance(value, str) and safe_name(value):
            names.append(value)
    return _unique_ordered(names)


class InputUploader:
    """Delivers staged files to ComfyUI via POST {base}/upload/image."""

    def __init__(self, cfg, base_url: str):
        self.cfg = cfg
        self.base_url = base_url.rstrip("/")
        self._cache: dict[str, tuple[int, int]] = {}

    def reset(self) -> None:
        self._cache.clear()

    async def ensure(self, names: Iterable[str]) -> None:
        for name in names:
            await self._ensure_one(name)

    async def _ensure_one(self, name: str) -> None:
        staged = stage_path(self.cfg, name) if safe_name(name) else None
        if staged is not None and staged.is_file():
            stat = staged.stat()
            sig = (stat.st_size, stat.st_mtime_ns)
            if self._cache.get(name) == sig:
                return
            await self._upload(name, staged)
            self._cache[name] = sig
            return
        legacy = Path(self.cfg.comfy_input) / name if safe_name(name) else None
        if legacy is not None and legacy.is_file():
            return  # already sitting in ComfyUI's own input dir.
        raise PipelineError(
            f"参照ファイル「{name}」が見つかりません。画像を追加し直してください。",
            kind="input_missing")

    async def _upload(self, name: str, path: Path) -> None:
        data = path.read_bytes()
        form = aiohttp.FormData()
        form.add_field("image", io.BytesIO(data), filename=name,
                       content_type="application/octet-stream")
        form.add_field("type", "input")
        form.add_field("subfolder", "")
        form.add_field("overwrite", "true")
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.base_url}/upload/image",
                                        data=form) as resp:
                    if resp.status != 200:
                        raise PipelineError(
                            f"ComfyUIへ参照ファイルを渡せませんでした"
                            f"（HTTP {resp.status}）。ComfyUIが起動しているか確認し、"
                            "もう一度生成してください。",
                            kind="comfy_upload")
                    payload = await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise PipelineError(
                f"ComfyUIへ参照ファイルを渡せませんでした（{exc}）。"
                "ComfyUIが起動しているか確認し、もう一度生成してください。",
                kind="comfy_upload") from exc
        if (not isinstance(payload, dict) or payload.get("name") != name or
                payload.get("subfolder", "") != "" or
                payload.get("type") != "input"):
            raise PipelineError(
                f"ComfyUIへ参照ファイルを渡せませんでした（応答が一致しません: "
                f"{payload!r}）。ComfyUIが起動しているか確認し、もう一度生成してください。",
                kind="comfy_upload")
