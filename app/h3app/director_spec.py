# -*- coding: utf-8 -*-
"""AI Director Spec: structured supervision data, independent of H3 Prompt.

A Director Spec is NEVER an H3 generation prompt. It is the user's
supervised design (idea -> items -> locks/requests -> regen) that only at
the very end converts into scene/speech/settings via director_convert.

Both kinds share the item shape {"value","locked","request"}:
  kind "single"  - 5/10/15s one-shot (12 direction items + timeline)
  kind "story30" - 30s (master{items, continuity_rules} + 3 clips)
"""
from __future__ import annotations

SPEC_VERSION = 2

# Cast / camera roles (Spec v2). The selected Character is ALWAYS the first
# entry of on_screen_subjects (= <Subject 1>). A second on-screen person
# exists ONLY when explicitly listed here; otherwise the compiler must never
# emit <Subject 2> / (S2). People behind the camera (camera_operator with
# visibility off_camera) are rendered as a fixed shooting-style phrase, never
# as a visible person. No vocabulary bans: the structure carries the roles.
CAST_POVS: tuple[str, ...] = (
    "", "lover_pov", "selfie", "observer", "handheld_third_person",
)

CAST_VISIBILITIES: tuple[str, ...] = ("", "off_camera", "on_screen")

_POV_ALIASES: tuple[tuple[str, str], ...] = (
    ("lover_pov", "lover_pov"),
    ("恋人視点", "lover_pov"),
    ("恋人", "lover_pov"),
    ("pov", "lover_pov"),
    ("selfie", "selfie"),
    ("自撮り", "selfie"),
    ("observer", "observer"),
    ("三人称", "observer"),
    ("客観", "observer"),
    ("handheld_third_person", "handheld_third_person"),
    ("手持ち", "handheld_third_person"),
)

SINGLE_ITEMS: tuple[str, ...] = (
    "video_type", "location", "scenery", "subject", "outfit", "acting",
    "camera_style", "camera_work", "dialogue", "voice", "ambient_audio",
    "mood",
)

MASTER_ITEMS: tuple[str, ...] = (
    "theme", "video_type", "protagonist", "world", "mood",
    "story_arc", "dialogue_policy", "shoot_policy",
)

CLIP_ITEMS: tuple[str, ...] = (
    "location", "acting", "movement", "camera_work", "dialogue",
    "voice", "ambient_audio", "mood", "handover",
)

# Aspects the continuity checker inspects (contradiction candidates only,
# never exact-match validation).
CONTINUITY_ASPECTS: tuple[str, ...] = (
    "identity", "outfit", "hairstyle", "carried_objects", "story_state",
    "emotion", "spatial",
)


def blank_item(value: str = "") -> dict:
    return {"value": value, "en": "", "locked": False, "request": ""}


def blank_cast() -> dict:
    return {"on_screen_subjects": ["character:0"],
            "camera_operator": "",
            "camera_operator_visibility": "",
            "camera_device": "",
            "camera_pov": "",
            "off_camera_people": []}


def normalize_pov(value: object) -> str:
    """Closest canonical camera_pov. Unknown expressions collapse, never fail."""
    text = str(value or "").strip()
    if not text:
        return ""
    low = text.lower()
    for alias, canonical in _POV_ALIASES:
        if alias in low or alias in text:
            return canonical
    if text in CAST_POVS:
        return text
    return ""


def normalize_cast(raw: object) -> dict:
    """Merge a cast dict over defaults; canonicalize pov/visibility."""
    cast = blank_cast()
    if not isinstance(raw, dict):
        return cast
    subjects = raw.get("on_screen_subjects")
    if isinstance(subjects, list) and subjects:
        cast["on_screen_subjects"] = [str(s) for s in subjects[:2]]
    if not cast["on_screen_subjects"]:
        cast["on_screen_subjects"] = ["character:0"]
    cast["camera_operator"] = str(raw.get("camera_operator") or "").strip()
    vis = str(raw.get("camera_operator_visibility") or "").strip().lower()
    cast["camera_operator_visibility"] = vis if vis in CAST_VISIBILITIES else ""
    cast["camera_device"] = str(raw.get("camera_device") or "").strip()
    cast["camera_pov"] = normalize_pov(raw.get("camera_pov"))
    others = raw.get("off_camera_people")
    if isinstance(others, list):
        cast["off_camera_people"] = [str(s).strip() for s in others[:4]
                                     if str(s).strip()]
    return cast


def allow_second_subject(cast: dict | None) -> bool:
    """True only when a second person is explicitly on screen."""
    subjects = (cast or {}).get("on_screen_subjects") or []
    return len([s for s in subjects if str(s).strip()]) > 1


def _items(names: tuple[str, ...], values: dict | None = None) -> dict:
    values = values or {}
    return {name: blank_item(str(values.get(name, ""))) for name in names}


def new_single(*, idea: str, duration_sec: int,
               initial_request: str = "") -> dict:
    if duration_sec not in (5, 10, 15):
        raise ValueError("single director duration must be 5, 10 or 15")
    return {
        "kind": "single",
        "version": SPEC_VERSION,
        "idea": idea.strip(),
        "duration_sec": duration_sec,
        "initial_request": (initial_request or "").strip(),
        "items": _items(SINGLE_ITEMS),
        "cast": blank_cast(),
        "global_request": "",
        "timeline": [],
    }


def new_story30(*, idea: str, initial_request: str = "") -> dict:
    clips = []
    for index in range(3):
        clips.append({
            "index": index,
            "duration_sec": 10,
            "items": _items(CLIP_ITEMS),
            "start_state": {},
            "end_state": {},
            "carry_over": {},
        })
    return {
        "kind": "story30",
        "version": SPEC_VERSION,
        "idea": idea.strip(),
        "duration_sec": 30,
        "initial_request": (initial_request or "").strip(),
        "master": {"items": _items(MASTER_ITEMS),
                   "continuity_rules": {"keep": [], "flexible": []},
                   "cast": blank_cast()},
        "clips": clips,
        "global_request": "",
    }


def _check_cast(cast: object, path: str, errors: list) -> None:
    """Blocking shape errors only; absent cast = defaults downstream."""
    if cast is None:
        return
    if not isinstance(cast, dict):
        errors.append(f"{path}: not an object")
        return
    subjects = cast.get("on_screen_subjects")
    if subjects is not None and not isinstance(subjects, list):
        errors.append(f"{path}.on_screen_subjects: must be a list")
    others = cast.get("off_camera_people")
    if others is not None and not isinstance(others, list):
        errors.append(f"{path}.off_camera_people: must be a list")
    pov = cast.get("camera_pov")
    if pov not in (None, "") and normalize_pov(pov) == "" and str(pov).strip():
        errors.append(f"{path}.camera_pov: unknown value {pov!r}")
    vis = cast.get("camera_operator_visibility")
    if vis not in (None, "") and str(vis).strip().lower() not in CAST_VISIBILITIES:
        errors.append(f"{path}.camera_operator_visibility: must be "
                      "off_camera or on_screen")


def _check_item(item, path: str, errors: list) -> None:
    if not isinstance(item, dict):
        errors.append(f"{path}: not an object")
        return
    if not isinstance(item.get("value"), str):
        errors.append(f"{path}.value: must be a string")
    # "en" (canonical English form) is optional for backward compatibility
    # with specs saved before Phase 1; a missing key stays valid.
    if "en" in item and not isinstance(item.get("en"), str):
        errors.append(f"{path}.en: must be a string")
    if not isinstance(item.get("locked"), bool):
        errors.append(f"{path}.locked: must be a boolean")
    if not isinstance(item.get("request"), str):
        errors.append(f"{path}.request: must be a string")


def validate_spec(spec: dict) -> dict:
    """Structural validation. Returns {"ok", "errors"} (blocking shape errors
    only; continuity is reported separately and is never blocking)."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return {"ok": False, "errors": ["spec: not an object"]}
    kind = spec.get("kind")
    if kind == "single":
        if spec.get("duration_sec") not in (5, 10, 15):
            errors.append("duration_sec: must be 5, 10 or 15")
        items = spec.get("items")
        if not isinstance(items, dict):
            errors.append("items: not an object")
        else:
            for name in SINGLE_ITEMS:
                if name in items:
                    _check_item(items[name], f"items.{name}", errors)
        _check_cast(spec.get("cast"), "cast", errors)
    elif kind == "story30":
        if spec.get("duration_sec") != 30:
            errors.append("duration_sec: story30 must be 30")
        master = spec.get("master")
        if not isinstance(master, dict) or not isinstance(
                master.get("items"), dict):
            errors.append("master.items: not an object")
        else:
            for name, item in master["items"].items():
                _check_item(item, f"master.items.{name}", errors)
            _check_cast(master.get("cast"), "master.cast", errors)
        clips = spec.get("clips")
        if not isinstance(clips, list) or len(clips) != 3:
            errors.append("clips: must be a list of 3")
        else:
            for clip in clips:
                if not isinstance(clip, dict):
                    errors.append("clips[]: not an object")
                    continue
                items = clip.get("items")
                if isinstance(items, dict):
                    for name, item in items.items():
                        _check_item(
                            item, f"clips[{clip.get('index')}].{name}", errors)
                for key in ("start_state", "end_state", "carry_over"):
                    if key in clip and not isinstance(clip[key], dict):
                        errors.append(f"clips[{clip.get('index')}].{key}: "
                                      "must be an object")
    else:
        errors.append(f"kind: unknown {kind!r}")
    return {"ok": not errors, "errors": errors}


def _iter_locked(spec: dict) -> list[tuple[str, str, str]]:
    """Yield (path, value, en) for every locked item in old and new specs."""
    out: list[tuple[str, str, str]] = []

    def _walk(items: dict, prefix: str) -> None:
        for name, item in (items or {}).items():
            if isinstance(item, dict) and item.get("locked"):
                en_value = item.get("en")
                out.append((f"{prefix}.{name}", str(item.get("value", "")),
                           str(en_value) if isinstance(en_value, str) else ""))

    kind = spec.get("kind")
    if kind == "single":
        _walk(spec.get("items"), "items")
    elif kind == "story30":
        _walk((spec.get("master") or {}).get("items"), "master.items")
        for clip in spec.get("clips") or []:
            if isinstance(clip, dict):
                _walk(clip.get("items"), f"clips[{clip.get('index')}].items")
    return out


def _set_by_path(spec: dict, path: str, value: str, en: str = "") -> bool:
    node: object = spec
    parts = path.split(".")
    for part in parts[:-1]:
        if part.startswith("clips[") and part.endswith("]"):
            try:
                index = int(part[len("clips["):-1])
            except ValueError:
                return False
            clips = node.get("clips") if isinstance(node, dict) else None
            node = None
            if isinstance(clips, list):
                for clip in clips:
                    if isinstance(clip, dict) and clip.get("index") == index:
                        node = clip
                        break
            if node is None:
                return False
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    last = parts[-1]
    if isinstance(node, dict) and isinstance(node.get(last), dict):
        node[last]["value"] = value
        node[last]["request"] = ""
        node[last]["en"] = en
        return True
    return False


def apply_locks(old: dict, new: dict) -> tuple[dict, list[str]]:
    """Restore every locked value from old into new (in place on new).

    Returns (new, enforced_paths). This is the app-side half of the lock
    guarantee: even if the model rewrote a locked item, the old value wins.
    A locked item's "en" (canonical English form) is restored alongside its
    value so a stale/mismatched translation can never survive a lock.
    """
    enforced: list[str] = []
    for path, value, en in _iter_locked(old):
        if _set_by_path(new, path, value, en):
            # Keep the lock flag itself on the restored item.
            enforced.append(path)
    # Locks the user set on the OLD spec survive onto the new one.
    for path, _value, _en in _iter_locked(old):
        _set_locked_flag(new, path, True)
    return new, enforced


def _set_locked_flag(spec: dict, path: str, locked: bool) -> bool:
    node: object = spec
    parts = path.split(".")
    for part in parts[:-1]:
        if part.startswith("clips[") and part.endswith("]"):
            try:
                index = int(part[len("clips["):-1])
            except ValueError:
                return False
            node = next((c for c in (node.get("clips") or [])
                         if isinstance(c, dict) and c.get("index") == index),
                        None)
            if node is None:
                return False
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    last = parts[-1]
    if isinstance(node, dict) and isinstance(node.get(last), dict):
        node[last]["locked"] = bool(locked)
        return True
    return False


def merge_cast_answer(current: dict | None, answer: object) -> dict:
    """Merge a model answer cast over the current one, then canonicalize.

    Non-empty answer fields win; unknown pov expressions collapse to the
    closest canonical value (never fail, never invent a second subject).
    """
    merged = normalize_cast(current)
    if not isinstance(answer, dict):
        return merged
    for key in ("camera_operator", "camera_operator_visibility",
                "camera_device", "camera_pov"):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    subjects = answer.get("on_screen_subjects")
    if isinstance(subjects, list) and subjects:
        cleaned = [str(s).strip() for s in subjects if str(s).strip()][:2]
        if cleaned:
            merged["on_screen_subjects"] = cleaned
    others = answer.get("off_camera_people")
    if isinstance(others, list):
        cleaned = [str(s).strip() for s in others if str(s).strip()][:4]
        merged["off_camera_people"] = cleaned
    return normalize_cast(merged)


def _clip_states(spec: dict) -> list[dict]:
    if spec.get("kind") != "story30":
        return []
    return [c for c in spec.get("clips", []) if isinstance(c, dict)]


def check_continuity(spec: dict) -> list[dict]:
    """Contradiction candidates between clips. WARNINGS ONLY - never blocks.

    Each finding: {"clip", "aspect", "message"}. A finding means "worth a
    look", not "invalid": intended moves, time passes and consumption are
    normal storytelling and must not be refused.
    """
    findings: list[dict] = []
    clips = _clip_states(spec)
    if len(clips) != 3:
        return findings
    master = spec.get("master") or {}
    rules = master.get("continuity_rules") or {}
    flexible = {str(x) for x in (rules.get("flexible") or [])}

    def _warn(prev: int, clip: int, aspect: str, message: str,
              field: str = "") -> None:
        if aspect in flexible:
            return
        findings.append({"clip": clip, "clip_display": clip + 1,
                         "prev_clip_display": prev + 1,
                         "aspect": aspect, "field": field,
                         "message": message})

    for prev, nxt in ((clips[0], clips[1]), (clips[1], clips[2])):
        prev_index = int(prev.get("index", 0))
        index = int(nxt.get("index", 0))
        end = prev.get("end_state") or {}
        start = nxt.get("start_state") or {}
        carry_prev = prev.get("carry_over") or {}
        carry_next = nxt.get("carry_over") or {}
        # Carried objects that silently vanish.
        for key in ("carried_objects", "所持品", "持ち物"):
            before = str((carry_prev.get(key) or end.get(key)) or "").strip()
            after = str((carry_next.get(key) or start.get(key)) or "").strip()
            if before and not after:
                _warn(prev_index, index, "carried_objects",
                      f"clip{index}開始時に「{before}」の継続が見えません。"
                      "使い切る・置く等の説明があれば問題ありません。",
                      field="carried_objects")
        # Day/night flip without a time-passage note.
        day_words = ("昼", "朝", "日中", "day", "morning", "noon")
        night_words = ("夜", "深夜", "夕", "night", "evening", "sunset")
        blob_before = " ".join(str(v) for v in
                               list(end.values()) + list(carry_prev.values()))
        blob_after = " ".join(str(v) for v in
                              list(start.values()) + list(carry_next.values()))
        flip = (any(w in blob_before for w in day_words) and
                any(w in blob_after for w in night_words)) or \
               (any(w in blob_before for w in night_words) and
                any(w in blob_after for w in day_words))
        if flip and "時間経過" not in blob_after and "time" not in blob_after.lower():
            _warn(prev_index, index, "spatial",
                  f"clip{index}で昼夜が切り替わっています。時間経過の説明が"
                  "あれば問題ありません。",
                  field="day_night")
        # Emotion hard cut without a bridge note. Paraphrases of the same
        # feeling (either string containing the other) are not a cut.
        emo_before = str(end.get("emotion") or end.get("感情") or "").strip()
        emo_after = str(start.get("emotion") or start.get("感情") or "").strip()
        if emo_before and emo_after and emo_before != emo_after and \
                emo_before not in emo_after and emo_after not in emo_before and \
                "きっかけ" not in blob_after and "reason" not in blob_after.lower():
            _warn(prev_index, index, "emotion",
                  f"clip{index}開始の感情（{emo_after}）が前clip終了"
                  f"（{emo_before}）と異なります。変化のきっかけがあれば"
                  "問題ありません。",
                  field="emotion")
    return findings
