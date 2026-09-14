# -*- coding: utf-8 -*-
"""User-editable ACTION presets + the sentence a preset writes into a prompt.

WHAT CHANGED (requirements 10-16)
---------------------------------
An action preset used to be a hidden TOGGLE: pressing a chip stored
`action_presets: ["peace"]` on the segment and `story_runner._scene_text()`
appended the preset's Japanese text at generation time. The user could not see
it, edit it, add one or remove one.

Now a preset is a NAME the user owns («ピースサイン»), and pressing a chip is a
COMMAND: it appends `APPLY_TEMPLATE` to THAT segment's prompt text, where the
user can see and edit it. The prompt text is the source of truth from then on.
No translation, no morphological analysis, no LLM call happens here - turning
Japanese staging into English is still the Story Director + Compiler's job.

The legacy id-based path in config.py / story.py / story_runner.py is left
completely untouched, so a story saved with `action_presets: ["wave"]` still
generates exactly as it did.
"""
from __future__ import annotations

import json
import re
import unicodedata
import uuid
from pathlib import Path

from .config import ACTION_PRESETS, APP_DIR

# --------------------------------------------------------------- the store ---
# Lives NEXT TO config.json, never inside a story project: presets belong to the
# app, not to one story.
STORE_PATH = APP_DIR / "action_presets.json"
STORE_VERSION = 1

# The 8 shipped presets are FACTORY DEFAULTS, seeded on first run. Their ids are
# the SAME ids the legacy `action_presets` field uses, so nothing about an old
# story changes; the display name is the old label.
FACTORY_PRESETS = [{"id": p["id"], "name": p["label"]} for p in ACTION_PRESETS]
FACTORY_IDS = {p["id"] for p in FACTORY_PRESETS}

MAX_NAME_CHARS = 40
# 「」 delimit the name inside APPLY_TEMPLATE, so a name may not contain them:
# it would make the written sentence ambiguous to read back.
FORBIDDEN_NAME_CHARS = "「」"


# ------------------------------------------------------------- the template ---
# The ONE place the applied sentence is defined. Change it here and the chip,
# the API and the action-load/transition detection all follow.
APPLY_TEMPLATE = "この場面の流れの中で、自然に「{name}」の動作を加える。"

_PREFIX, _SUFFIX = APPLY_TEMPLATE.split("{name}")
# Reads back a sentence the apply step wrote, so a preset press keeps its effect
# on timing (duration.action_load) and on transition planning even though it no
# longer writes `action_presets`.
_APPLIED_RE = re.compile(
    re.escape(_PREFIX) + r"([^\n]{1,%d}?)" % MAX_NAME_CHARS + re.escape(_SUFFIX))


def apply_sentence(name: str) -> str:
    """The exact sentence a chip press appends. Deterministic, no I/O."""
    return APPLY_TEMPLATE.format(name=str(name).strip())


def append_to_prompt(prompt: str, name: str) -> str:
    """`prompt` with the preset sentence appended on its own line.

    Pressing the same chip twice appends the sentence twice - a chip is a
    command, and the user asked for the action twice.
    """
    body = (prompt or "").rstrip()
    sentence = apply_sentence(name)
    return f"{body}\n{sentence}" if body else sentence


def applied_action_names(text: str) -> list[str]:
    """Every preset sentence currently present in this prompt text, in order.

    Duplicates are KEPT: two sentences mean two actions, which is exactly the
    signal `duration.delivery_from_presets` used to get from two preset ids.
    """
    return [m.group(1).strip() for m in _APPLIED_RE.finditer(str(text or ""))
            if m.group(1).strip()]


def segment_action_signals(segment) -> list[str]:
    """The action signals of ONE segment: legacy preset ids + applied sentences.

    This is the bridge that stops requirement 15 from silently weakening the
    timing and transition model. Old segments contribute ids, new segments
    contribute the sentences their chip presses wrote into the prompt, and a
    segment holding both contributes both.
    """
    if not isinstance(segment, dict):
        return []
    out = [str(p) for p in (segment.get("action_presets") or []) if str(p).strip()]
    out += applied_action_names(segment.get("prompt") or "")
    return out


# -------------------------------------------------------------- validation ---
class PresetError(ValueError):
    """A refused preset operation. The message is user-facing Japanese."""


def _is_bad_char(ch: str) -> bool:
    return unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co")


def normalize_name(raw) -> str:
    """Validate a preset name and return the stored form. Raises PresetError."""
    if not isinstance(raw, str):
        raise PresetError("動作の名前を入力してください。")
    name = raw.strip()
    if not name:
        raise PresetError("動作の名前を入力してください（空白だけは登録できません）。")
    if any(_is_bad_char(ch) for ch in name):
        raise PresetError("動作の名前に使えない文字（制御文字）が含まれています。")
    if any(ch in name for ch in FORBIDDEN_NAME_CHARS):
        raise PresetError("動作の名前に「 」 は使えません。")
    if len(name) > MAX_NAME_CHARS:
        raise PresetError(f"動作の名前は {MAX_NAME_CHARS} 文字までです。")
    return name


def new_preset_id() -> str:
    return "u" + uuid.uuid4().hex[:12]


class ActionPresetStore:
    """The user's action preset list, persisted as JSON.

    File format (app\\action_presets.json)::

        {"version": 1,
         "presets": [{"id": "wave", "name": "手を振る"}, ...],
         "deleted": ["bow"]}

    `presets` is the FULL EFFECTIVE LIST (factory entries are seeded into it on
    first run), `deleted` is a tombstone list. The tombstones are what makes a
    deleted FACTORY preset stay deleted: seeding skips any factory id that was
    ever deleted, so it can never resurrect on the next start.

    Names are DATA. They are stored and returned verbatim; it is the UI's job to
    put them on screen with textContent, never innerHTML.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else STORE_PATH
        self._presets: list[dict] = []
        self._deleted: list[str] = []
        # Set when the file was unreadable/corrupt and factory defaults were
        # used instead. The API hands this to the user instead of hiding it.
        self.fallback = False
        self.fallback_reason = ""
        self.load()

    # ------------------------------------------------------------- loading ---
    def load(self) -> None:
        self.fallback = False
        self.fallback_reason = ""
        data = None
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON オブジェクトではありません")
            except Exception as exc:                       # noqa: BLE001
                data = None
                self.fallback = True
                self.fallback_reason = (
                    f"動作プリセットの保存ファイルを読み込めませんでした（{type(exc).__name__}）。"
                    "初期状態の動作プリセットで続けます。"
                    f"保存ファイル: {self.path}")

        self._deleted = []
        self._presets = []
        if isinstance(data, dict):
            for pid in data.get("deleted") or []:
                pid = str(pid).strip()
                if pid and pid not in self._deleted:
                    self._deleted.append(pid)
            for item in data.get("presets") or []:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("id") or "").strip()
                try:
                    name = normalize_name(item.get("name"))
                except PresetError:
                    continue                              # drop a broken entry
                if not pid or pid in self._deleted:
                    continue
                if any(p["id"] == pid for p in self._presets):
                    continue
                self._presets.append({"id": pid, "name": name})

        # Seed the factory defaults. A factory preset that was deleted stays
        # deleted; one that is simply new (added by a future version) appears.
        seeded = False
        known = {p["id"] for p in self._presets}
        for fp in FACTORY_PRESETS:
            if fp["id"] in known or fp["id"] in self._deleted:
                continue
            self._presets.append(dict(fp))
            seeded = True
        # Never write over a file we could not read: the user may still want to
        # repair it. A later add/delete saves the repaired list.
        if seeded and not self.fallback:
            self.save()

    # -------------------------------------------------------------- saving ---
    def save(self) -> None:
        payload = {
            "version": STORE_VERSION,
            "presets": [dict(p) for p in self._presets],
            "deleted": list(self._deleted),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------- operations ---
    def list(self) -> list[dict]:
        """The effective preset list, in display order."""
        return [{"id": p["id"], "name": p["name"],
                 # `label` keeps the shape the chip renderer already expects.
                 "label": p["name"],
                 "factory": p["id"] in FACTORY_IDS}
                for p in self._presets]

    def get(self, preset_id: str) -> dict | None:
        return next((dict(p) for p in self._presets
                     if p["id"] == str(preset_id)), None)

    def add(self, name) -> dict:
        """Add a preset. Duplicate names are REJECTED (case/space sensitive
        after stripping), so the chip row can never show two identical chips."""
        clean = normalize_name(name)
        if any(p["name"] == clean for p in self._presets):
            raise PresetError(f"「{clean}」はすでに登録されています。")
        pid = new_preset_id()
        while any(p["id"] == pid for p in self._presets) or pid in self._deleted:
            pid = new_preset_id()
        preset = {"id": pid, "name": clean}
        self._presets.append(preset)
        # An id may have been tombstoned before; adding it back clears that.
        self._deleted = [d for d in self._deleted if d != pid]
        self.save()
        self.fallback = False
        return dict(preset)

    def delete(self, preset_id: str) -> dict:
        """Delete a preset and remember the deletion across restarts.

        Prompt text a chip press already wrote is ORDINARY TEXT and is not
        touched: deleting the chip never edits a story.
        """
        pid = str(preset_id or "").strip()
        found = next((p for p in self._presets if p["id"] == pid), None)
        if found is None:
            raise PresetError("その動作プリセットは見つかりません。")
        self._presets = [p for p in self._presets if p["id"] != pid]
        if pid not in self._deleted:
            self._deleted.append(pid)
        self.save()
        self.fallback = False
        return dict(found)
