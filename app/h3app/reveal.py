# -*- coding: utf-8 -*-
"""「保存フォルダを開く」: resolve a LOGICAL target to a real folder, safely.

THE RULE
--------
The browser never sends a path. It sends an id it already has (a story id or a
project id) and a TARGET from a fixed enum:

    final     the finished video (the merged story video / the last clip)
    clips     the folder the individual clips are kept in
    project   the project folder itself

The backend looks the path up from the project's own metadata and then proves,
before anything is opened, that the result is inside one of the allowed output
roots:

    app\\story_projects\\      story folders
    app\\projects\\            single-shot project folders
    <ComfyUI>\\output\\        where ComfyUI writes the original files

Comparison is done on the REALPATH, so `..`, a symlink, a junction or a drive
change cannot walk out of a root, and an id that does not look like an id this
app generated is refused before it ever reaches the filesystem.

Everything in here is PURE: it resolves and validates, it never opens anything.
That is what makes it testable from server.py --selftest without spawning an
Explorer window.

Public API
    TARGETS
    TargetError(message, kind, status)
    is_safe_project_id(pid)                          -> bool
    allowed_roots(cfg)                               -> [Path, ...]
    resolve_story_target(story, target, roots=)      -> dict
    resolve_project_target(project, target, root, roots=) -> dict
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# The complete set of things the UI is allowed to ask for.
TARGETS = ("final", "clips", "project")

# Single-shot project ids are uuid4().hex[:12]. Story ids are checked by
# story.is_safe_story_id, which uses the same shape.
_SAFE_PROJECT_ID = re.compile(r"^[0-9a-fA-F]{6,32}$")


class TargetError(RuntimeError):
    """A refused request. `message` is user-facing Japanese."""

    def __init__(self, message: str, *, kind: str = "not_found",
                 status: int = 404):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = int(status)


def is_safe_project_id(project_id) -> bool:
    return bool(project_id) and bool(_SAFE_PROJECT_ID.match(str(project_id)))


def _real(path) -> Path:
    """The path with `..`, symlinks and junctions resolved. Never raises.

    os.path.realpath is used rather than Path.resolve(strict=True) because the
    target may legitimately not exist yet - "not generated" has to be a clear
    Japanese message, not an exception from the path layer.
    """
    try:
        return Path(os.path.realpath(str(path)))
    except (OSError, ValueError):
        return Path(str(path))


def allowed_roots(cfg) -> list[Path]:
    """The only places the app will ever open."""
    roots = [cfg.story_dir, cfg.projects_dir, cfg.comfy_output]
    try:
        roots.append(cfg.media_root)
    except Exception:                                        # noqa: BLE001
        pass
    return [_real(r) for r in roots]


def _is_inside(path: Path, roots) -> bool:
    real = _real(path)
    for root in roots or ():
        try:
            if real == root or real.is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


def _check_target(target) -> str:
    value = str(target or "final")
    if value not in TARGETS:
        raise TargetError("開く場所の指定が不正です。", kind="prompt_rejected",
                          status=400)
    return value


def _guard(path, roots) -> Path:
    """Refuse anything that resolves outside the allowed roots."""
    if not str(path or "").strip():
        raise TargetError("開く場所が見つかりませんでした。")
    real = _real(path)
    if not _is_inside(real, roots):
        raise TargetError("許可されていない場所は開けません。", kind="prompt_rejected",
                          status=400)
    return real


def _folder_result(path, roots, *, target: str, kind: str, missing: str) -> dict:
    real = _guard(path, roots)
    if not real.is_dir():
        raise TargetError(missing)
    return {"kind": kind, "target": target, "directory": str(real), "file": ""}


def _file_result(path, roots, *, target: str, kind: str, missing: str) -> dict:
    real = _guard(path, roots)
    if not real.is_file():
        raise TargetError(missing)
    parent = _guard(real.parent, roots)
    return {"kind": kind, "target": target, "directory": str(parent),
            "file": str(real)}


# ------------------------------------------------------------------ story ----
def resolve_story_target(story, target, *, roots) -> dict:
    """Where 「保存フォルダを開く」 should take the user for THIS story."""
    if story is None:
        raise TargetError("ストーリーが見つかりませんでした。")
    what = _check_target(target)
    if what == "final":
        final = str(story.data.get("final_video") or "").strip()
        if not final:
            raise TargetError(
                "完成動画がまだありません。先に「1本にまとめる」を実行してください。")
        return _file_result(
            final, roots, target=what, kind="story",
            missing="完成動画のファイルが見つかりませんでした。"
                    "もう一度書き出してください。")
    if what == "clips":
        return _folder_result(
            story.clips_dir, roots, target=what, kind="story",
            missing="クリップのフォルダがまだありません。先に生成してください。")
    return _folder_result(
        story.root, roots, target=what, kind="story",
        missing="ストーリーのフォルダが見つかりませんでした。")


# ------------------------------------------------------------ single shot ----
def resolve_project_target(project, target, *, project_dir, roots) -> dict:
    """The same three targets for a single-shot project.

    `final` is the last generated clip: ComfyUI's own file when it is still
    there, otherwise the copy the app keeps in the project folder. Both live
    inside an allowed root; anything else is refused.
    """
    if project is None:
        raise TargetError("プロジェクトが見つかりませんでした。")
    what = _check_target(target)
    if what == "final":
        clips = list(project.clips or [])
        if not clips:
            raise TargetError("保存された動画がまだありません。先に生成してください。")
        last = clips[-1]
        export = last.get("export") or {}
        candidates = []
        if isinstance(export, dict) and export.get("ok") and export.get("path"):
            candidates.append(str(export["path"]).strip())
        candidates += [str(last.get("video") or "").strip(),
                      str(last.get("local_video") or "").strip()]
        for candidate in candidates:
            if not candidate:
                continue
            real = _real(candidate)
            if real.is_file() and _is_inside(real, roots):
                return _file_result(real, roots, target=what, kind="single",
                                    missing="保存された動画が見つかりませんでした。")
        # Nothing usable: say WHY, and open nothing.
        for candidate in candidates:
            if candidate and not _is_inside(candidate, roots):
                raise TargetError("許可されていない場所は開けません。",
                                  kind="prompt_rejected", status=400)
        raise TargetError("保存された動画が見つかりませんでした。")
    return _folder_result(
        project_dir, roots, target=what, kind="single",
        missing="プロジェクトのフォルダが見つかりませんでした。")
