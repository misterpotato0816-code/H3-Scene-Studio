# -*- coding: utf-8 -*-
"""Character Library: reusable identity templates (H3-wide, project-independent).

Library = template source (owns its ref/voice copies under app/character_library/).
Project/Story = snapshot owner (copies materialized into the existing
comfy_input h3app_ref_* management + profile/cache fields at apply time).

Deleting or updating a Library character never touches snapshots, so past
projects keep generating. No new image loader: applied refs are plain
h3app_ref_* names the existing graphs already consume.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

LIB_DIRNAME = "character_library"
INDEX_NAME = "characters.json"
MAX_REFS = 4


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return "char_" + uuid.uuid4().hex[:8]


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


class CharacterStore:
    """JSON index + per-character asset dirs. All methods never raise."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / INDEX_NAME
        self._items: dict[str, dict] = {}
        self.load()

    # ------------------------------------------------------------- index ----
    def load(self) -> None:
        try:
            if self.index_path.is_file():
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._items = {str(k): v for k, v in data.items()
                                   if isinstance(v, dict)}
        except Exception:                                        # noqa: BLE001
            self._items = {}

    def _save(self) -> None:
        _atomic_write_json(self.index_path, self._items)

    def list(self) -> list[dict]:
        return [dict(v) for _, v in sorted(
            self._items.items(),
            key=lambda kv: kv[1].get("updated_at", 0), reverse=True)]

    def get(self, character_id: str) -> dict | None:
        item = self._items.get(str(character_id))
        return dict(item) if isinstance(item, dict) else None

    # -------------------------------------------------------------- CRUD ----
    def save_new(self, *, display_name: str, refs: list[Path],
                 profile_text: str = "", canon: dict | None = None,
                 outfit_ref: dict | None = None,
                 voice_src: Path | None = None, voice_note: str = "",
                 memo: str = "") -> tuple[dict | None, str]:
        """Stage assets first, write the index last (no half state)."""
        name = (display_name or "").strip() or "名前なし"
        character_id = _new_id()
        char_dir = self.root / character_id
        try:
            staged_refs = self._stage_files(char_dir, "ref", refs[:MAX_REFS])
            staged_voice = ""
            if voice_src is not None and Path(voice_src).is_file():
                staged = self._stage_files(char_dir, "voice", [voice_src])
                staged_voice = staged[0] if staged else ""
            now = _now()
            item = {
                "character_id": character_id,
                "display_name": name,
                "refs": staged_refs,
                "profile_text": profile_text or "",
                "canon": dict(canon or {}),
                "outfit_ref": dict(outfit_ref or {}),
                "voice_wav": staged_voice,
                "voice_note": voice_note or "",
                "memo": memo or "",
                "created_at": now,
                "updated_at": now,
            }
            self._items[character_id] = item
            self._save()
            return dict(item), ""
        except Exception as exc:                                 # noqa: BLE001
            shutil.rmtree(char_dir, ignore_errors=True)
            self._items.pop(character_id, None)
            return None, str(exc)[:300]

    def _stage_files(self, char_dir: Path, stem: str,
                     sources: list[Path | str]) -> list[str]:
        """Copy sources into the character dir. Returns relative names."""
        out: list[str] = []
        tmpdir = char_dir / "_tmp"
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, str]] = []
        for i, src in enumerate(sources):
            src_path = Path(str(src))
            if not src_path.is_file():
                continue
            suffix = src_path.suffix.lower() or ".bin"
            staged.append((src_path, f"{stem}_{i}{suffix}"))
        for src_path, name in staged:
            shutil.copy2(src_path, tmpdir / name)
            out.append(name)
        for _src, name in staged:
            (tmpdir / name).rename(char_dir / name)
        try:
            tmpdir.rmdir()
        except Exception:                                        # noqa: BLE001
            pass
        char_dir.mkdir(parents=True, exist_ok=True)
        return out

    def _asset(self, item: dict, name: str) -> Path | None:
        if not name:
            return None
        base = (self.root / item["character_id"] / Path(name).name)
        return base if base.is_file() else None

    def update(self, character_id: str, *, display_name: str | None = None,
               refs: list[Path] | None = None,
               profile_text: str | None = None,
               canon: dict | None = None, outfit_ref: dict | None = None,
               voice_src: Path | None = None, voice_note: str | None = None,
               voice_clear: bool = False,
               memo: str | None = None) -> tuple[dict | None, str]:
        """Atomic-ish: stage everything new first, single index write last."""
        item = self._items.get(str(character_id))
        if not isinstance(item, dict):
            return None, "キャラクターが見つかりません。"
        char_dir = self.root / item["character_id"]
        try:
            staged_refs = list(item.get("refs") or [])
            if refs is not None:
                # Remove replaced files only after the new set is staged.
                new_refs = self._stage_files(char_dir, "ref",
                                             refs[:MAX_REFS])
                for old in staged_refs:
                    if old not in new_refs:
                        try:
                            (char_dir / Path(old).name).unlink()
                        except Exception:                        # noqa: BLE001
                            pass
                staged_refs = new_refs
            staged_voice = str(item.get("voice_wav") or "")
            if voice_clear:
                if staged_voice:
                    try:
                        (char_dir / Path(staged_voice).name).unlink()
                    except Exception:                            # noqa: BLE001
                        pass
                staged_voice = ""
            elif voice_src is not None and Path(voice_src).is_file():
                new_voice = self._stage_files(char_dir, "voice", [voice_src])
                if new_voice:
                    if staged_voice and staged_voice not in new_voice:
                        try:
                            (char_dir / Path(staged_voice).name).unlink()
                        except Exception:                        # noqa: BLE001
                            pass
                    staged_voice = new_voice[0]
            if display_name is not None and str(display_name).strip():
                item["display_name"] = str(display_name).strip()
            if profile_text is not None:
                item["profile_text"] = profile_text
            if canon is not None:
                item["canon"] = dict(canon)
            if outfit_ref is not None:
                item["outfit_ref"] = dict(outfit_ref)
            if voice_note is not None:
                item["voice_note"] = voice_note
            if memo is not None:
                item["memo"] = memo
            item["refs"] = staged_refs
            item["voice_wav"] = staged_voice
            item["updated_at"] = _now()
            self._save()
            return dict(item), ""
        except Exception as exc:                                 # noqa: BLE001
            self.load()  # re-read last good index; staged files are inert
            return None, str(exc)[:300]

    def rename(self, character_id: str, display_name: str) -> tuple[bool, str]:
        item, err = self.update(character_id, display_name=display_name)
        return (item is not None), err

    def duplicate(self, character_id: str) -> tuple[dict | None, str]:
        src = self._items.get(str(character_id))
        if not isinstance(src, dict):
            return None, "キャラクターが見つかりません。"
        src_dir = self.root / src["character_id"]
        refs = [src_dir / n for n in (src.get("refs") or [])]
        voice = src_dir / str(src.get("voice_wav") or "") \
            if src.get("voice_wav") else None
        if voice is not None and not voice.is_file():
            voice = None
        return self.save_new(
            display_name=str(src.get("display_name") or "") + " のコピー",
            refs=[p for p in refs if p.is_file()],
            profile_text=str(src.get("profile_text") or ""),
            canon=dict(src.get("canon") or {}),
            outfit_ref=dict(src.get("outfit_ref") or {}),
            voice_src=voice,
            voice_note=str(src.get("voice_note") or ""),
            memo=str(src.get("memo") or ""))

    def delete(self, character_id: str) -> tuple[bool, str]:
        """Remove the index entry + Library-owned files ONLY. Snapshots live
        in project/story space and are never touched here."""
        item = self._items.pop(str(character_id), None)
        if not isinstance(item, dict):
            return False, "キャラクターが見つかりません。"
        try:
            self._save()
        except Exception as exc:                                 # noqa: BLE001
            return False, str(exc)[:300]
        shutil.rmtree(self.root / item["character_id"], ignore_errors=True)
        return True, ""

    # ---------------------------------------------------------- apply -------
    def materialize_refs(self, item: dict, comfy_input: str | Path) -> list[str]:
        """Copy Library refs into the existing h3app_ref_* management.

        Deterministic names per character+slot, so re-applying overwrites
        instead of accumulating orphans. Returns the staged file names.
        """
        names: list[str] = []
        char_dir = self.root / str(item.get("character_id", ""))
        short = "".join(c for c in str(item.get("character_id", ""))
                        if c.isalnum())[:16] or "char"
        for i, rel in enumerate(list(item.get("refs") or [])[:MAX_REFS]):
            src = char_dir / Path(str(rel)).name
            if not src.is_file():
                continue
            dest = Path(comfy_input) / f"h3app_char_{short}_{i}{src.suffix.lower()}"
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or \
                        dest.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dest)
                names.append(dest.name)
            except Exception:                                    # noqa: BLE001
                continue
        return names

    def materialize_voice(self, item: dict, comfy_input: str | Path) -> str:
        """Stage the Library voice wav the same deterministic way."""
        wav = str(item.get("voice_wav") or "")
        if not wav:
            return ""
        src = self.root / str(item.get("character_id", "")) / Path(wav).name
        if not src.is_file():
            return ""
        short = "".join(c for c in str(item.get("character_id", ""))
                        if c.isalnum())[:16] or "char"
        dest = Path(comfy_input) / f"h3app_char_{short}_voice.wav"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            return dest.name
        except Exception:                                        # noqa: BLE001
            return ""
