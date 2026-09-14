# -*- coding: utf-8 -*-
"""Final Director Spec -> existing pipeline inputs.

Single: scene text + speech lines (+ settings patch for frames).
Story30: segments[] (prompt/speech/motion) for the existing story_create path.
Pure functions, no I/O. Continuity info is expanded explicitly into the
scene text (semantic continuity + existing H3 video-continuity mechanisms).
"""
from __future__ import annotations

from . import director_spec as spec_mod

_ITEM_TO_SCENE = (
    "video_type", "location", "scenery", "subject", "outfit", "acting",
    "camera_style", "camera_work", "mood",
)


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip())


def single_to_inputs(spec: dict) -> dict:
    """Return {"scene": str, "speech": str, "frames": int} for single-shot."""
    if spec.get("kind") != "single":
        raise ValueError("single_to_inputs needs a single spec")
    items = spec.get("items") or {}
    scene_parts = [str(items[n]["value"]).strip() for n in _ITEM_TO_SCENE
                   if isinstance(items.get(n), dict) and
                   _nonempty(items[n].get("value"))]
    voice = items.get("voice") if isinstance(items.get("voice"), dict) else {}
    ambient = items.get("ambient_audio") if isinstance(
        items.get("ambient_audio"), dict) else {}
    sounds: list[str] = []
    if _nonempty(voice.get("value")):
        sounds.append(str(voice["value"]).strip())
    if _nonempty(ambient.get("value")):
        sounds.append(str(ambient["value"]).strip())
    if sounds:
        scene_parts.append("音声: " + " / ".join(sounds))
    dialogue = items.get("dialogue") if isinstance(
        items.get("dialogue"), dict) else {}
    speech = str(dialogue.get("value") or "").strip()
    duration_sec = int(spec.get("duration_sec") or 5)
    frames = 124 if duration_sec <= 5 else (243 if duration_sec <= 10 else 362)
    return {"scene": "\n".join(scene_parts).strip(),
            "speech": speech, "frames": frames}


def _clip_scene(master: dict, clip: dict) -> str:
    items = clip.get("items") or {}
    parts: list[str] = []
    for name in ("location", "acting", "movement", "camera_work", "mood"):
        item = items.get(name)
        if isinstance(item, dict) and _nonempty(item.get("value")):
            parts.append(str(item["value"]).strip())
    carry = clip.get("carry_over") or {}
    carried = [f"{k}: {v}" for k, v in carry.items() if _nonempty(v)]
    if carried:
        parts.append("引き継ぎ: " + " / ".join(carried))
    start = clip.get("start_state") or {}
    starting = [f"{k}: {v}" for k, v in start.items() if _nonempty(v)]
    if starting:
        parts.append("開始状態: " + " / ".join(starting))
    voice = items.get("voice") if isinstance(items.get("voice"), dict) else {}
    ambient = items.get("ambient_audio") if isinstance(
        items.get("ambient_audio"), dict) else {}
    sounds = [str(x.get("value")).strip() for x in (voice, ambient)
              if _nonempty(x.get("value"))]
    if sounds:
        parts.append("音声: " + " / ".join(sounds))
    return "\n".join(parts).strip()


def story30_to_segments(spec: dict) -> list[dict]:
    """Return 3 segment dicts {prompt, speech} for story creation."""
    if spec.get("kind") != "story30":
        raise ValueError("story30_to_segments needs a story30 spec")
    master = spec.get("master") or {}
    out = []
    for clip in spec.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        items = clip.get("items") or {}
        dialogue = items.get("dialogue") if isinstance(
            items.get("dialogue"), dict) else {}
        out.append({"prompt": _clip_scene(master, clip),
                    "speech": str(dialogue.get("value") or "").strip()})
    if len(out) != 3:
        raise ValueError("story30 spec must convert to exactly 3 segments")
    return out


def dialogue_budget_warnings(spec: dict) -> list[str]:
    """Non-blocking: dialogue density vs duration (uses duration estimates)."""
    try:
        from . import duration as duration_mod
    except Exception:                                        # noqa: BLE001
        return []
    warnings: list[str] = []
    pairs: list[tuple[str, int]] = []
    if spec.get("kind") == "single":
        items = spec.get("items") or {}
        dialogue = items.get("dialogue") if isinstance(
            items.get("dialogue"), dict) else {}
        pairs.append((str(dialogue.get("value") or ""),
                      int(spec.get("duration_sec") or 5)))
    elif spec.get("kind") == "story30":
        for clip in spec.get("clips") or []:
            if not isinstance(clip, dict):
                continue
            items = clip.get("items") or {}
            dialogue = items.get("dialogue") if isinstance(
                items.get("dialogue"), dict) else {}
            pairs.append((str(dialogue.get("value") or ""),
                          int(clip.get("duration_sec") or 10)))
    for text, seconds in pairs:
        if not text.strip():
            continue
        try:
            need, _detail = duration_mod.estimate_speech_seconds(text)
        except Exception:                                    # noqa: BLE001
            continue
        if need > seconds * 0.85:
            warnings.append(
                f"台詞量（約{need:.1f}秒分）が尺{seconds}秒に対して多めです。"
                "早口にならないよう間・演技と配分してください。")
    return warnings


def merge_model_answer_single(spec: dict, answer: dict) -> dict:
    """Fill values from a model answer JSON; locks/requests on spec win."""
    items = spec.get("items") or {}
    ans = (answer.get("items") or {}) if isinstance(answer, dict) else {}
    for name in spec_mod.SINGLE_ITEMS:
        if isinstance(ans.get(name), dict) and _nonempty(ans[name].get("value")):
            items.setdefault(name, spec_mod.blank_item())
            if not items[name].get("locked"):
                items[name]["value"] = str(ans[name]["value"])
                items[name]["en"] = str(ans[name].get("en") or "")
    if isinstance(answer.get("timeline"), list):
        spec["timeline"] = answer["timeline"][:16]
    if isinstance(answer.get("cast"), dict):
        spec["cast"] = spec_mod.merge_cast_answer(
            spec.get("cast"), answer.get("cast"))
    return spec


def merge_model_answer_master(spec: dict, answer: dict) -> dict:
    master = spec.get("master") or {}
    items = master.get("items") or {}
    ans = (answer.get("items") or {}) if isinstance(answer, dict) else {}
    for name in spec_mod.MASTER_ITEMS:
        if isinstance(ans.get(name), dict) and _nonempty(ans[name].get("value")):
            items.setdefault(name, spec_mod.blank_item())
            if not items[name].get("locked"):
                items[name]["value"] = str(ans[name]["value"])
                items[name]["en"] = str(ans[name].get("en") or "")
    rules = (answer.get("continuity_rules") or {}) if isinstance(answer, dict) \
        else {}
    if isinstance(rules.get("keep"), list):
        master["continuity_rules"] = master.get("continuity_rules") or {}
        master["continuity_rules"]["keep"] = [str(x) for x in rules["keep"]][:12]
    if isinstance(rules.get("flexible"), list):
        master["continuity_rules"] = master.get("continuity_rules") or {}
        master["continuity_rules"]["flexible"] = [
            str(x) for x in rules["flexible"]][:12]
    if isinstance(answer.get("cast"), dict):
        master["cast"] = spec_mod.merge_cast_answer(
            master.get("cast"), answer.get("cast"))
    return spec


def merge_model_answer_clips(spec: dict, answer: dict,
                             only: tuple[int, ...] | None = None,
                             only_items: tuple[str, ...] | None = None) -> dict:
    """Merge a clips answer. `only` limits clip indices, `only_items` limits
    item names (single-item regen). Locked values always win."""
    ans_clips = (answer.get("clips") or []) if isinstance(answer, dict) else []
    by_index = {c.get("index"): c for c in ans_clips
                if isinstance(c, dict)}
    spec_clips = [c for c in (spec.get("clips") or [])
                  if isinstance(c, dict)]
    ordered = [c for c in ans_clips if isinstance(c, dict)]
    # Index mismatch fallback: some models number clips 1-based. Exact index
    # alignment wins when it covers every clip; otherwise zip by position so
    # a shifted numbering never merges half-empty (or fully empty).
    positional = False
    exact_hits = sum(
        1 for s in spec_clips if s.get("index") in by_index)
    if ordered and (exact_hits < len(spec_clips)
                    and len(ordered) == len(spec_clips)):
        by_index = {}
        for slot, ans in zip(spec_clips, ordered):
            by_index[slot.get("index")] = ans
        positional = True
    matched = 0
    for clip in spec.get("clips") or []:
        if not isinstance(clip, dict):
            continue
        index = clip.get("index")
        if only is not None and index not in only:
            continue
        ans = by_index.get(index, {})
        ans_items = ans.get("items") if isinstance(ans, dict) else {}
        if not isinstance(ans_items, dict):
            continue
        for name, ans_item in ans_items.items():
            if only_items is not None and name not in only_items:
                continue
            if not isinstance(ans_item, dict) or \
                    not _nonempty(ans_item.get("value")):
                continue
            slot = clip["items"].setdefault(name, spec_mod.blank_item())
            if not slot.get("locked"):
                slot["value"] = str(ans_item["value"])
                slot["en"] = str(ans_item.get("en") or "")
        for key in ("start_state", "end_state", "carry_over"):
            if isinstance(ans.get(key), dict) and ans[key]:
                clip[key] = {str(k): str(v) for k, v in
                             list(ans[key].items())[:12]}
                matched += 1
        if ans:
            matched += 1
    spec["_clips_merge"] = {"matched": matched,
                            "positional": positional,
                            "answer_clips": len(
                                [c for c in ans_clips
                                 if isinstance(c, dict)])}
    return spec


def _snapshot_parts(snapshot: dict | None) -> tuple[dict, dict]:
    """Identity traits + outfit reference for the deterministic compiler.

    Real snapshots on disk often carry an empty ``canon`` dict with all the
    detail sitting in ``profile_text`` instead (the profile VLM output before
    it is parsed). When ``canon`` is empty, fall back to parsing
    ``profile_text`` so the Subject 1 definition is not silently reduced to
    "the main woman".
    """
    from . import canon as canon_mod
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    canon = snapshot.get("canon") if isinstance(
        snapshot.get("canon"), dict) else {}
    canon = {k: str(v or "").strip() for k, v in canon.items()}
    outfit_ref = snapshot.get("outfit_ref") if isinstance(
        snapshot.get("outfit_ref"), dict) else {}
    outfit_ref = {k: str(outfit_ref.get(k, "")).strip() for k in
                 ("outfit", "accessories", "materials", "color_keys")}
    if not any(canon.values()) and not any(outfit_ref.values()):
        profile_text = str(snapshot.get("profile_text") or "").strip()
        if profile_text:
            full = canon_mod.parse_profile(profile_text).as_dict()
            canon = {k: full.get(k, "") for k in
                     ("face", "hair", "build", "skin")}
            outfit_ref = {k: full.get(k, "") for k in
                         ("outfit", "accessories", "materials", "color_keys")}
    identity = {k: str(canon.get(k, "")).strip()
                for k in ("face", "hair", "build", "skin")}
    return identity, outfit_ref


def _outfit_specified(items: dict) -> bool:
    slot = items.get("outfit")
    return bool(isinstance(slot, dict) and str(slot.get("value") or "").strip())


def _refs_available_for_clip(story_mode: str, clip_index: int) -> dict:
    """Reference availability descriptor matching the LONG / LONG_FAST graphs.

    clip 0 never continues from a previous clip: no Video, no Audio.
    LONG clips > 0 relay a real tail video + its own soundtrack (Video 1 +
    Audio 1 = the video's soundtrack). LONG_FAST clips > 0 use a 0f relay
    (no <Video 1>) with a standalone fixed Voice Master as <Audio 1>.
    """
    if clip_index <= 0:
        return {"videos": 0, "audios": 0}
    if str(story_mode or "LONG").upper() == "LONG_FAST":
        return {"videos": 0, "audios": 1, "audio_kind": "voice_master"}
    return {"videos": 1, "audios": 1, "audio_kind": "video_soundtrack"}


def final_to_h3_single(spec: dict, snapshot: dict | None, refs: list[str],
                       frames: int) -> str:
    """Final single Spec -> six-section H3 prompt. No AI involved."""
    from . import director_lexicon as lexicon_mod
    if not isinstance(spec, dict) or spec.get("kind") != "single":
        raise ValueError("final_to_h3_single needs a single spec")
    from .director_review import timeline_errors
    errors = timeline_errors(spec.get("timeline") or [], float(spec.get("duration_sec") or 5))
    if errors:
        raise ValueError(" / ".join(errors))
    items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
    identity, outfit_ref = _snapshot_parts(snapshot)
    cast = spec.get("cast") if isinstance(spec.get("cast"), dict) else None
    return lexicon_mod.compile_direct_prompt(
        items=items, identity=identity, outfit_ref=outfit_ref,
        refs=list(refs or []), frames=int(frames),
        timeline=spec.get("timeline") if isinstance(
            spec.get("timeline"), list) else [],
        cast=cast)


def final_to_h3_clip(spec: dict, clip_index: int, snapshot: dict | None,
                     refs: list[str], frames: int,
                     story_mode: str = "LONG") -> str:
    """Final story30 clip -> six-section H3 prompt. No AI involved."""
    from . import director_lexicon as lexicon_mod
    if not isinstance(spec, dict) or spec.get("kind") != "story30":
        raise ValueError("final_to_h3_clip needs a story30 spec")
    clip = next((c for c in (spec.get("clips") or [])
                 if isinstance(c, dict) and c.get("index") == clip_index), None)
    if clip is None:
        raise ValueError(f"clip {clip_index} not in spec")
    items = clip.get("items") if isinstance(clip.get("items"), dict) else {}
    identity, outfit_ref = _snapshot_parts(snapshot)
    master = spec.get("master") if isinstance(
        spec.get("master"), dict) else {}
    cast = master.get("cast") if isinstance(
        master.get("cast"), dict) else None
    return lexicon_mod.compile_direct_prompt(
        items=items, identity=identity, outfit_ref=outfit_ref,
        refs=list(refs or []), frames=int(frames),
        start_state=clip.get("start_state") if isinstance(
            clip.get("start_state"), dict) else {},
        end_state=clip.get("end_state") if isinstance(
            clip.get("end_state"), dict) else {},
        carry_over=clip.get("carry_over") if isinstance(
            clip.get("carry_over"), dict) else {},
        refs_available=_refs_available_for_clip(story_mode, clip_index),
        cast=cast)
