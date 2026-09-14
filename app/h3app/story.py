# -*- coding: utf-8 -*-
"""Story mode persistence.

One story = one long text cut into segments, each segment rendered as its own
clip and relayed into the next one by its tail. Everything is JSON on disk (no
pickle) under app\\story_projects\\<id>\\ :

    project.json     everything except the segment array
    segments.json    the segment array
    clips\\           a copy of every generated clip
    frames\\          the last-frame PNG of every clip
    final\\           the merged video
    orphaned\\        clips whose segment was edited away (never deleted)

A StoryProject is deliberately duck-compatible with projects.Project (.id,
.data, .clips, .save()) so pipeline.Pipeline._ensure_profile / _run_director /
_run_generate can be reused against it without forking the pipeline.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import duration
from . import transitions
from .canon import normalize_overrides
from .config import ACTION_PRESET_IDS

SAFE_ID = re.compile(r"^[0-9a-fA-F]{6,32}$")

STORY_STATUSES = ("pending", "running", "paused", "done", "error")

# Per-segment seed = base_seed + segment_index. Deterministic, so resuming a
# story reproduces exactly the same clip for the same segment.
SEED_MODULO = 2 ** 32

# A segment's PERMANENT identity, independent of its current position. `index`
# is where the segment sits right now and is renumbered on every structure
# change; `segment_id` never changes, which is what makes "the user moved B
# after C" distinguishable from "the user rewrote B and C".
SEGMENT_ID_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}$")

ORPHAN_DIRNAME = "orphaned"


def new_segment_id() -> str:
    return uuid.uuid4().hex


class StructureError(ValueError):
    """A refused structure change. The message is user-facing Japanese."""


def segment_seed(base_seed: int, index: int) -> int:
    return int(int(base_seed) + int(index)) % SEED_MODULO


def migrate_project_presets(data: dict) -> list[str]:
    """Project-level `motion_presets` now means STYLE only.

    An old project.json may carry action ids there (手を振る etc). Applying those
    to EVERY segment is exactly the "she waves in all six clips" bug, so they are
    dropped from the project level on load and recorded - never silently applied
    everywhere, and never silently pushed onto one arbitrary segment either.
    """
    ids = [str(p) for p in (data.get("motion_presets") or [])]
    kept = [p for p in ids if p not in ACTION_PRESET_IDS]
    dropped = [p for p in ids if p in ACTION_PRESET_IDS]
    if dropped:
        data["motion_presets"] = kept
        note = list(data.get("migrated_action_presets") or [])
        for p in dropped:
            if p not in note:
                note.append(p)
        data["migrated_action_presets"] = note
    return dropped


def is_safe_story_id(story_id: str) -> bool:
    """Story ids come off the URL, so they never get to touch the filesystem
    unless they look exactly like an id this module generated."""
    return bool(story_id) and bool(SAFE_ID.match(story_id))


def normalize_transition(raw) -> dict:
    """WP-C seam transition: how THIS segment joins to the PREVIOUS one.
    Ignored for segment 0 (there is no boundary before it). `mode` is
    validated against a fixed set - never sniffed out of prompt text."""
    if not isinstance(raw, dict):
        return {"mode": "natural", "keep_pause": False}
    mode = str(raw.get("mode") or "natural").strip().lower()
    if mode not in ("natural", "cut", "fade"):
        mode = "natural"
    return {"mode": mode, "keep_pause": bool(raw.get("keep_pause"))}


def normalize_segments(raw: list, *, seconds: float | None = None,
                       keep_empty: bool = False) -> list[dict]:
    """Accept whatever the UI sends and make it a well-formed segment array.

    `keep_empty` is used by the structure endpoint: a segment the user just
    ADDED is legitimately empty until they type into it, and dropping it there
    would silently shift every position the caller just asked for. Everywhere
    else an empty segment is still discarded, exactly as before.
    """
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        lines = []
        for ln in item.get("lines") or []:
            if not isinstance(ln, dict):
                continue
            text = (ln.get("text") or "").strip()
            if not text:
                continue
            ltype = "speech" if ln.get("type") == "speech" else "prompt"
            lines.append({"type": ltype, "text": text})

        prompt = item.get("prompt")
        speech = item.get("speech")
        if prompt is None:
            prompt = "\n".join(l["text"] for l in lines if l["type"] == "prompt")
        if speech is None:
            speech = "\n".join(l["text"] for l in lines if l["type"] == "speech")
        prompt = (prompt or "").strip()
        speech = (speech or "").strip()
        if not lines:
            lines = ([{"type": "prompt", "text": p} for p in prompt.split("\n") if p.strip()]
                     + [{"type": "speech", "text": s} for s in speech.split("\n") if s.strip()])
        raw_presets = [str(p) for p in (item.get("motion_presets") or []) if str(p).strip()]
        actions = [str(p) for p in (item.get("action_presets") or []) if str(p).strip()]
        # An action id sitting in a segment's motion_presets is fine (it belongs to
        # this ONE segment), it just belongs in the action list now.
        presets = [p for p in raw_presets if p not in ACTION_PRESET_IDS]
        for p in raw_presets:
            if p in ACTION_PRESET_IDS and p not in actions:
                actions.append(p)
        if not prompt and not speech and not presets and not actions \
                and not keep_empty:
            continue                              # never keep an empty segment
        status = item.get("status") if item.get("status") in (
            "pending", "running", "done", "error") else "pending"
        sid = str(item.get("segment_id") or "").strip()
        segment = {
            # Permanent identity. Assigned once, carried through every reorder.
            "segment_id": sid if SEGMENT_ID_RE.match(sid) else new_segment_id(),
            "index": len(out),
            "lines": lines,
            "prompt": prompt,
            "speech": speech,
            "motion_presets": presets,
            # One-off actions. NEVER broadcast to every segment.
            "action_presets": actions,
            # The ONLY way a persistent character trait may change, e.g.
            # {"outfit": "a black knee-length coat over the same dress"}.
            # A structured data field on purpose: never sniffed out of the scene
            # text, never a keyword list. Unknown keys are discarded.
            "attribute_overrides": normalize_overrides(
                item.get("attribute_overrides")),
            "status": status,
            "clip": item.get("clip"),
            "error": item.get("error") or "",
        }
        # HOW this segment is spoken. Optional and backward compatible: a
        # segment saved before deliveries existed simply has no field, and
        # duration.normalize_delivery() then applies the defaults - which is
        # exactly the behaviour it had before. Only a delivery the caller
        # actually sent is stored, so loading an old story changes nothing.
        if isinstance(item.get("delivery"), dict):
            segment["delivery"] = duration.normalize_delivery(item["delivery"])
        # An EXPLICIT continuity request for this segment (see transitions.py):
        # "from here on the framing is a wide shot", "she is seated from here".
        # Structured data with a fixed field set - never sniffed out of the scene
        # text. Absent, the segment simply continues whatever the previous one
        # ended in, which is what every existing story does.
        if isinstance(item.get("continuity"), dict):
            segment["continuity"] = transitions.normalize_state(item["continuity"])
        # つなぎ方 (WP-C): natural(既定)/cut/fade + 「間を残す」。Only meaningful
        # for index >= 1 (the boundary into this segment) but stored on every
        # segment the caller actually sent one for; the merge step ignores it
        # for index 0.
        if isinstance(item.get("transition"), dict):
            segment["transition"] = normalize_transition(item["transition"])
        # Debug metadata from the segmenter (estimated seconds, density, band,
        # why the segment ends where it does). Carried through so the story
        # runner can write it into the per-segment debug record. Never part of
        # any prompt text.
        if isinstance(item.get("timing"), dict):
            segment["timing"] = dict(item["timing"])
        # Precompiled H3 prompt from the AI Director path. When present, the
        # story runner uses it verbatim instead of calling the VLM director
        # again. Absent (all existing stories), behaviour is unchanged.
        if isinstance(item.get("direct_en_prompt"), str) and \
                item["direct_en_prompt"].strip():
            segment["direct_en_prompt"] = item["direct_en_prompt"]
        # The seed actually used, once the segment has been rendered.
        try:
            if item.get("seed") is not None:
                segment["seed"] = int(item["seed"])
        except (TypeError, ValueError):
            pass
        out.append(segment)
    return out


def content_edit(existing: list[dict], requested: list[dict]) -> list[dict]:
    """Content endpoint cannot erase locks, reorder IDs, or trust client state."""
    import copy
    if len(existing) != len(requested):
        raise ValueError("構成の変更には追加・削除・並べ替え操作を使ってください。")
    fields = ("prompt", "speech", "motion_presets", "action_presets",
              "attribute_overrides", "delivery", "continuity")
    # WP-C: the seam transition only affects HOW the final video is merged,
    # never the clip itself, so it may be changed freely - even on a
    # locked/approved/done segment - without the "edit invalidates the clip"
    # guards below applying to it.
    soft_fields = ("transition",)
    out = []
    for old, incoming in zip(existing, requested):
        if old["segment_id"] != incoming["segment_id"]:
            raise ValueError("セグメントの順序が変わっています。再読み込みしてください。")
        changed = any(old.get(k) != incoming.get(k) for k in fields)
        if changed and old.get("locked") in ("locked", "approved"):
            raise ValueError("ロック済みクリップは編集できません。先に解除してください。")
        if changed and old.get("status") == "done":
            raise ValueError("生成済みクリップは、先に再生成対象へ指定してから編集してください。")
        item = copy.deepcopy(old)
        for key in fields + ("lines",):
            if key in incoming:
                item[key] = copy.deepcopy(incoming[key])
        for key in soft_fields:
            if key in incoming:
                item[key] = copy.deepcopy(incoming[key])
        if changed:
            item.pop("direct_en_prompt", None)
            item.pop("timing", None)
        out.append(item)
    return out


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class StoryProject:
    """Duck-compatible with projects.Project for the parts the pipeline touches."""

    def __init__(self, data: dict, segments: list[dict], root: Path):
        self.data = data
        self._segments = segments
        self.root = root

    # ------------------------------------------------------------ properties --
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def segments(self) -> list[dict]:
        return self._segments

    @segments.setter
    def segments(self, value: list[dict]) -> None:
        self._segments = value

    @property
    def clips(self) -> list[dict]:
        return self.data.setdefault("clips", [])

    @property
    def cursor(self) -> int:
        return int(self.data.get("cursor", 0))

    @property
    def status(self) -> str:
        return self.data.get("status", "pending")

    @property
    def total_frames(self) -> int:
        return sum(int(c.get("frames", 0)) for c in self.clips)

    @property
    def total_seconds(self) -> float:
        return round(sum(float(c.get("seconds", 0)) for c in self.clips), 2)

    # ----------------------------------------------------------------- paths --
    @property
    def project_path(self) -> Path:
        return self.root / "project.json"

    @property
    def segments_path(self) -> Path:
        return self.root / "segments.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def orphaned_dir(self) -> Path:
        """Clips whose segment was edited away. Created on demand, never
        emptied by the app: invalidation must never destroy pixels."""
        return self.root / ORPHAN_DIRNAME

    @property
    def debug_dir(self) -> Path:
        """Per-segment debug records. Never surfaced in the normal UI."""
        return self.root / "debug"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.clips_dir, self.frames_dir, self.final_dir,
                  self.debug_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ seed --
    @property
    def base_seed(self) -> int:
        """The story's own base seed. Fixed for the life of the story."""
        value = self.data.get("base_seed")
        if value is None:
            value = (self.data.get("settings") or {}).get("seed", 0)
            self.data["base_seed"] = int(value)
        return int(self.data["base_seed"])

    def seed_for(self, index: int) -> int:
        return segment_seed(self.base_seed, index)

    # --------------------------------------------------------------- storage --
    def save(self) -> None:
        """segments.json FIRST, then project.json.

        project.json carries `cursor` and `clips`. Writing it last means a crash
        between the two writes can only ever leave a project whose cursor is
        BEHIND its segments - which re-renders one segment - instead of a project
        that claims clips whose segments were never persisted.
        """
        self.data["updated_at"] = time.time()
        self.ensure_dirs()
        _atomic_write_json(self.segments_path, self._segments)
        _atomic_write_json(self.project_path, self.data)

    # ---------------------------------------------------------------- public --
    def to_public(self) -> dict:
        d = dict(self.data)
        d["segments"] = [dict(s) for s in self._segments]
        d["segment_total"] = len(self._segments)
        d["total_frames"] = self.total_frames
        d["total_seconds"] = self.total_seconds
        d["can_merge"] = len(self.clips) >= 1
        return d

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.data.get("name", ""),
            "status": self.status,
            "cursor": self.cursor,
            "segment_total": len(self._segments),
            "clips_done": len(self.clips),
            "updated_at": self.data.get("updated_at", 0),
            "created_at": self.data.get("created_at", 0),
            "merged": bool(self.data.get("merged", False)),
        }

    def clip_for(self, segment_index: int) -> dict | None:
        for c in self.clips:
            try:
                idx = int(c.get("segment_index", -1))
            except (TypeError, ValueError):
                continue
            if idx == int(segment_index):
                return c
        return None

    def mergeable_clips(self) -> list[dict]:
        """The ONLY list the final merge is allowed to concatenate.

        Built from the SEGMENTS, not from clips[]: a clip is usable only if a
        segment still sits at its index, still carries its identity and still
        says `done` with that exact file. So even a project.json that (through
        a crash, a hand edit or a bug) still lists an invalidated clip can never
        put an orphaned clip into the final video.
        """
        by_index: dict[int, dict] = {}
        for clip in self.clips:
            try:
                idx = int(clip.get("segment_index", -1))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(self._segments)):
                continue
            seg = self._segments[idx]
            if seg.get("status") != "done":
                continue
            sid = str(clip.get("segment_id") or "").strip()
            seg_id = str(seg.get("segment_id") or "").strip()
            if sid and seg_id and sid != seg_id:
                continue
            seg_clip = str(seg.get("clip") or "").strip()
            if seg_clip and seg_clip not in (str(clip.get("local_video") or ""),
                                             str(clip.get("video") or "")):
                continue
            by_index[idx] = clip           # a later entry wins: it is the newer one
        return [by_index[i] for i in sorted(by_index)]


# ------------------------------------------------------------- identity ------
def ensure_segment_ids(story: StoryProject) -> bool:
    """Backfill `segment_id` on a project saved before identities existed.

    Segments get a fresh id; every clip inherits the id of the segment that
    currently sits at its `segment_index`, which is exactly what the old
    positional model meant. Returns True when something was added, so the
    caller can persist it once instead of on every load.
    """
    changed = False
    for seg in story.segments:
        if not isinstance(seg, dict):
            continue
        sid = str(seg.get("segment_id") or "").strip()
        if not SEGMENT_ID_RE.match(sid):
            seg["segment_id"] = new_segment_id()
            changed = True
    for clip in story.clips:
        if not isinstance(clip, dict):
            continue
        if str(clip.get("segment_id") or "").strip():
            continue
        try:
            idx = int(clip.get("segment_index", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(story.segments):
            clip["segment_id"] = story.segments[idx].get("segment_id", "")
            changed = True
    return changed


# ------------------------------------------------------------- structure -----
# The fields a caller may edit on an EXISTING segment. Everything else
# (status / clip / seed / index) is owned by the app.
_EDITABLE_FIELDS = ("prompt", "speech", "motion_presets", "action_presets",
                    "attribute_overrides", "delivery", "timing", "continuity")


def plan_structure(story: StoryProject, requested) -> dict:
    """Work out what "here is the new ordered segment list" means. PURE.

    Mutates nothing - not the story, not the request - so the same function
    serves the dry run and the commit.
    """
    if not isinstance(requested, list):
        raise StructureError("セグメントの指定が不正です。")

    old_segments = [dict(s) for s in story.segments if isinstance(s, dict)]
    for seg in old_segments:
        if not SEGMENT_ID_RE.match(str(seg.get("segment_id") or "").strip()):
            seg["segment_id"] = new_segment_id()
    old_ids = [str(s["segment_id"]) for s in old_segments]
    by_id = dict(zip(old_ids, old_segments))

    items: list[dict] = []
    seen: set[str] = set()
    for entry in requested:
        if not isinstance(entry, dict):
            raise StructureError("セグメントの指定が不正です。")
        sid = str(entry.get("segment_id") or "").strip()
        if not sid:
            # A NEW segment. Same path as an auto-split one, so it comes back
            # with every field a normal segment has.
            item = {k: v for k, v in entry.items()
                    if k in _EDITABLE_FIELDS or k == "lines"}
            items.append(item)
            continue
        if sid not in by_id:
            raise StructureError(
                "知らないセグメントが含まれています。"
                "画面を再読み込みしてから、もう一度お試しください。")
        if sid in seen:
            raise StructureError("同じセグメントが複数回指定されています。")
        seen.add(sid)
        base = dict(by_id[sid])
        # Content edits are allowed to travel with a structure change. `lines`
        # is regenerated when prompt/speech changed without it, so the two never
        # disagree.
        if "lines" in entry:
            base["lines"] = entry["lines"]
        elif "prompt" in entry or "speech" in entry:
            base.pop("lines", None)
        for key in _EDITABLE_FIELDS:
            if key in entry:
                base[key] = entry[key]
        items.append(base)

    segments = normalize_segments(items, keep_empty=True)
    if not segments:
        raise StructureError("セグメントを空にはできません。"
                             "少なくとも 1 つのセグメントが必要です。")

    new_ids = [str(s["segment_id"]) for s in segments]
    # Compare normalized content so legacy missing fields equal their defaults.
    old_content = {s["segment_id"]: s for s in normalize_segments(old_segments, keep_empty=True)}
    # DIVERGENCE: the first position at which the new order stops matching the
    # old one. Appending at the end therefore diverges at len(old) - i.e. it
    # invalidates nothing.
    common = min(len(old_ids), len(new_ids))
    divergence = common
    for i in range(common):
        if (new_ids[i] != old_ids[i] or any(
                segments[i].get(key) != old_content[old_ids[i]].get(key)
                for key in _EDITABLE_FIELDS)):
            divergence = i
            break

    # Normalization intentionally accepts content only. Preserve server-owned
    # locks on untouched segments; a structure edit cannot erase or invalidate
    # approved work, including downstream clips affected by a changed tail.
    for index, old in enumerate(old_segments):
        if old.get("locked") in ("locked", "approved") and index >= divergence:
            raise StructureError("ロック済みクリップに影響する変更です。先にロックを解除してください。")
    for segment in segments:
        old = by_id.get(segment["segment_id"])
        if old is None:
            continue
        if old.get("locked") in ("locked", "approved"):
            segment["locked"] = old["locked"]
        if any(segment.get(key) != old_content[segment["segment_id"]].get(key) for key in _EDITABLE_FIELDS):
            segment.pop("direct_en_prompt", None)
            segment.pop("timing", None)

    base_seed = story.base_seed
    for i, seg in enumerate(segments):
        seg["index"] = i
        if i >= divergence:
            # Moved, inserted or shifted: nothing generated for THIS position is
            # valid any more, and the seed follows the new index.
            seg["status"] = "pending"
            seg["clip"] = None
            seg["error"] = ""
            seg["seed"] = segment_seed(base_seed, i)
        # i < divergence: kept verbatim, including its stored seed and clip.

    kept: list[dict] = []
    invalid: list[tuple[int, dict]] = []
    for clip in story.clips:
        try:
            idx = int(clip.get("segment_index", -1))
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < divergence:
            kept.append(clip)                     # order preserved: clips[-1]
        else:                                     # stays the continuation source
            invalid.append((idx, clip))

    invalidated = [{
        "index": idx,
        "segment_id": str(clip.get("segment_id")
                          or (old_ids[idx] if 0 <= idx < len(old_ids) else "")),
        "clip": str(clip.get("local_video") or clip.get("video") or ""),
    } for idx, clip in invalid]

    # The cursor never jumps FORWARD over work that was never done, so it is the
    # divergence point clamped by wherever the story had actually got to.
    cursor = max(0, min(int(story.data.get("cursor", 0)), divergence, len(segments)))

    return {
        "divergence": divergence,
        "segments": segments,
        "cursor": cursor,
        "kept_clips": kept,
        "invalid_clips": invalid,
        "invalidated": invalidated,
    }


def _is_inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _orphan_path(dest_dir: Path, name: str) -> Path:
    """A never-colliding destination: the original name plus a suffix, so the
    same clip_03.mp4 can be invalidated any number of times."""
    src = Path(name)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    while True:
        candidate = dest_dir / f"{src.stem}__{stamp}_{uuid.uuid4().hex[:8]}{src.suffix}"
        if not candidate.exists():
            return candidate


def orphan_clip(story: StoryProject, clip: dict, *, old_index: int,
                reason: str) -> dict:
    """Move one invalidated clip's files into orphaned\\ and describe the move.

    Only files that live INSIDE the story folder are moved: ComfyUI's own output
    is never touched, and nothing is ever deleted.
    """
    dest_dir = story.orphaned_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved: list[dict] = []
    original_path = str(clip.get("local_video") or clip.get("video") or "")
    orphaned_path = ""
    for key in ("local_video", "video", "seg", "last_frame"):
        raw = str(clip.get(key) or "").strip()
        if not raw:
            continue
        src = Path(raw)
        if not src.is_file() or not _is_inside(src, story.root):
            continue
        if _is_inside(src, dest_dir):
            continue
        dest = _orphan_path(dest_dir, src.name)
        try:
            shutil.move(str(src), str(dest))
        except Exception as exc:                  # noqa: BLE001 - never fatal
            print(f"[story] could not orphan {src}: {exc}", flush=True)
            continue
        moved.append({"key": key, "from": str(src), "to": str(dest)})
        if key in ("local_video", "video") and not orphaned_path:
            original_path, orphaned_path = str(src), str(dest)
    return {
        "segment_id": str(clip.get("segment_id") or ""),
        "old_index": int(old_index),
        "original_path": original_path,
        "orphaned_path": orphaned_path,
        "invalidated_at": time.time(),
        "reason": reason,
        "step": int(clip.get("step", 0) or 0),
        "seed": clip.get("seed"),
        "files": moved,
    }


def apply_structure(story: StoryProject, requested, *, dry_run: bool = False,
                    reason: str = "structure_change") -> dict:
    """ADD / DELETE / REORDER, all of them: the caller sends the new order.

    `dry_run` computes the impact and writes NOTHING - no file move, no save and
    no in-memory change - so the UI can ask for confirmation first.
    """
    plan = plan_structure(story, requested)
    invalidated = plan["invalidated"]
    count = len(invalidated)

    if dry_run:
        message = (f"この変更で生成済みのクリップ {count} 本が無効になります。"
                   if count else "生成済みのクリップは影響を受けません。")
    else:
        records = [orphan_clip(story, clip, old_index=idx, reason=reason)
                   for idx, clip in plan["invalid_clips"]]
        story.segments = plan["segments"]
        story.data["clips"] = plan["kept_clips"]
        story.data["cursor"] = int(plan["cursor"])
        if count:
            # A merged video that contains a clip we just invalidated is a lie.
            story.data["merged"] = False
            story.data["final_video"] = ""
            history = story.data.setdefault("invalidated_clips", [])
            history.extend(records)
        story.save()
        message = (f"セグメントを保存しました（{len(plan['segments'])} 個）。"
                   + (f"生成済みのクリップ {count} 本を無効にしました。"
                      if count else ""))

    return {
        "divergence": int(plan["divergence"]),
        "invalidated": invalidated,
        "invalidated_count": count,
        "requires_confirmation": count > 0,
        "segments": [dict(s) for s in plan["segments"]],
        "cursor": int(plan["cursor"]),
        "dry_run": bool(dry_run),
        "message": message,
    }


class StoryStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, StoryProject] = {}

    def dir_for(self, story_id: str) -> Path:
        return self.root / story_id

    # ---------------------------------------------------------------- create --
    def create(self, *, name: str, source_text: str, lines: list[dict],
               segments: list[dict], images: list[str], settings: dict,
               jp_negative: str, motion_presets: list[str],
               profile: str = "", profile_cache_key_: str = "") -> StoryProject:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        data = {
            "id": sid,
            "name": name or "ストーリー",
            "created_at": now,
            "updated_at": now,
            "source_text": source_text or "",
            "lines": lines or [],
            "images": images,
            "profile": profile,
            "profile_cache_key": profile_cache_key_,
            "settings": settings,
            "jp_negative": jp_negative,
            # STYLE presets only. Action presets live on ONE segment.
            "motion_presets": list(motion_presets or []),
            "base_seed": int((settings or {}).get("seed", 0)),
            "cursor": 0,
            "status": "pending",
            "stop_requested": False,
            "clips": [],
            "merged": False,
            "final_video": "",
            "error": None,
            # Written by the pipeline; harmless duplicates of the last run.
            "en_prompt": "",
        }
        dropped = migrate_project_presets(data)
        if dropped:
            print(f"[story] project-level action presets dropped on create: {dropped}",
                  flush=True)
        story = StoryProject(data, normalize_segments(segments), self.dir_for(sid))
        ensure_segment_ids(story)
        story.save()
        with self._lock:
            self._cache[sid] = story
        return story

    # ------------------------------------------------------------------- get --
    def get(self, story_id: str) -> StoryProject | None:
        if not is_safe_story_id(story_id):
            return None
        with self._lock:
            cached = self._cache.get(story_id)
        if cached is not None:
            return cached
        root = self.dir_for(story_id)
        path = root / "project.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        segments: list[dict] = []
        seg_path = root / "segments.json"
        if seg_path.is_file():
            try:
                loaded = json.loads(seg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    segments = loaded
            except Exception:
                segments = []
        dropped = migrate_project_presets(data)
        if dropped:
            print(f"[story] {story_id}: project-level action presets migrated away "
                  f"(not applied to every segment): {dropped}", flush=True)
        story = StoryProject(data, segments, root)
        story.base_seed                     # backfill for stories saved before seeds
        # A project saved before segment identities existed keeps working: the
        # ids are assigned here, once, and persisted immediately so that every
        # later request talks about the same segments.
        if ensure_segment_ids(story):
            try:
                story.save()
            except Exception as exc:              # noqa: BLE001 - read-only disk
                print(f"[story] {story_id}: segment ids not persisted: {exc}",
                      flush=True)
        with self._lock:
            self._cache[story_id] = story
        return story

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir()
                      if d.is_dir() and (d / "project.json").is_file())

    def list_summaries(self) -> list[dict]:
        out = []
        for sid in self.list_ids():
            story = self.get(sid)
            if story is not None:
                out.append(story.summary())
        out.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return out

    def delete(self, story_id: str) -> bool:
        if not is_safe_story_id(story_id):
            return False
        root = self.dir_for(story_id)
        with self._lock:
            self._cache.pop(story_id, None)
        if not root.is_dir():
            return False
        shutil.rmtree(root, ignore_errors=True)
        return not root.exists()
