# -*- coding: utf-8 -*-
"""Project store: one project = one video being built up clip by clip.

Everything lives on disk under app\\projects\\<id>\\ so a crash or a restart never
loses a 3.6 minute render, and so "続きを作る" works after closing the app.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from .canon import PROFILE_SCHEMA_VERSION

STAGES = [
    "参照画像を解析しています",
    "キャラクタープロファイルを作成しています",
    "プロンプトを生成しています",
    "H3 conditioning を実行しています",
    "動画を生成しています",
    "音声と動画を保存しています",
    "完了しました",
]


def profile_cache_key(paths: list[Path],
                      schema_version: int = PROFILE_SCHEMA_VERSION) -> str:
    """Hash of the image bytes, their slot order AND the profile schema version.

    Slot order matters: image 1 is the face reference the profile leans on, so
    the same files in a different order is a different profile.

    The schema version matters because a cached profile is only useful while the
    parser that turns it into a CharacterCanon understands it. Bumping
    canon.PROFILE_SCHEMA_VERSION changes every key, so entries written under the
    old schema simply stop matching in find_cached_profile - they are ignored,
    never re-read and never a crash.
    """
    h = hashlib.sha256()
    h.update(f"schema={int(schema_version)}|".encode("utf-8"))
    for i, p in enumerate(paths):
        h.update(f"|{i}|".encode("utf-8"))
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(p.name.encode("utf-8"))
    return h.hexdigest()


class Project:
    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = path

    # ------------------------------------------------------------ properties --
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def clips(self) -> list[dict]:
        return self.data.setdefault("clips", [])

    @property
    def total_frames(self) -> int:
        return sum(int(c["frames"]) for c in self.clips)

    @property
    def total_seconds(self) -> float:
        return round(sum(float(c["seconds"]) for c in self.clips), 2)

    @property
    def max_seconds(self) -> float:
        return float(self.data.get("max_seconds", 15))

    @property
    def can_continue(self) -> bool:
        return bool(self.clips) and self.total_seconds < self.max_seconds

    # ---------------------------------------------------------------- storage --
    def save(self) -> None:
        self.data["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def to_public(self) -> dict:
        d = dict(self.data)
        d["total_frames"] = self.total_frames
        d["total_seconds"] = self.total_seconds
        d["can_continue"] = self.can_continue
        return d


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, Project] = {}

    def dir_for(self, project_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", str(project_id)):
            raise ValueError("プロジェクトIDが不正です。")
        target = (self.root / project_id).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError("プロジェクトが保存領域の外を参照しています。")
        return target

    def create(self, *, settings: dict, images: list[str], jp_scene: str,
               jp_dialogue: str, jp_negative: str, max_seconds: float,
               profile: str = "", profile_cache_key_: str = "",
               en_prompt: str = "", parent_id: str = "") -> Project:
        pid = uuid.uuid4().hex[:12]
        data = {
            "id": pid,
            "created_at": time.time(),
            "updated_at": time.time(),
            "parent_id": parent_id,
            "settings": settings,
            "images": images,
            "jp_scene": jp_scene,
            "jp_dialogue": jp_dialogue,
            "jp_negative": jp_negative,
            "profile": profile,
            "profile_cache_key": profile_cache_key_,
            "en_prompt": en_prompt,
            "clips": [],
            "max_seconds": max_seconds,
            "status": "pending",
            # What the director has already been asked to stage. "続きを作る" reads
            # this so it never feeds the same scene/dialogue in a second time.
            "consumed": {"scene": [], "dialogue": []},
        }
        p = Project(data, self.dir_for(pid) / "project.json")
        p.save()
        with self._lock:
            self._cache[pid] = p
        return p

    def get(self, project_id: str) -> Project | None:
        try:
            directory = self.dir_for(project_id)
        except ValueError:
            return None
        with self._lock:
            if project_id in self._cache:
                return self._cache[project_id]
        path = directory / "project.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        p = Project(data, path)
        with self._lock:
            self._cache[project_id] = p
        return p

    def list_ids(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and (d / "project.json").is_file())

    def find_cached_profile(self, cache_key: str) -> str:
        """Reuse a profile whenever the exact same image set was analysed before.

        Saves a full VLM pass (and its model load) on "もう1本" and "続き".
        """
        if not cache_key:
            return ""
        for pid in self.list_ids():
            p = self.get(pid)
            if p and p.data.get("profile_cache_key") == cache_key and p.data.get("profile"):
                return p.data["profile"]
        return ""
