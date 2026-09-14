# -*- coding: utf-8 -*-
"""Story Runner v2 planning: cache key, lock/approve, selective regenerate.

Pure logic over the v1 story schema (segments with prompt/speech/status,
seed = base + index, cursor). v1 files are never migrated or rewritten;
v2 state lives in `<story_dir>/story_v2.json` next to them.
"""
from __future__ import annotations

import hashlib
import json

# Clip lifecycle. LOCKED/APPROVED clips are never rebuilt by a later run.
STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_LOCKED = "locked"
STATUS_APPROVED = "approved"
STATUS_ERROR = "error"

FROZEN_STATUSES = (STATUS_LOCKED, STATUS_APPROVED)

CACHE_KEY_FIELDS = (
    "model", "model_hash", "lora", "lora_hash", "preset",
    "prompt", "ref_hashes", "seed", "width", "height", "frames",
    "control_input", "sampler", "scheduler", "steps",
    "shift_video", "shift_audio", "ref_image_size", "tail_frames",
)


def cache_key(spec: dict) -> str:
    """Canonical SHA-256 over exactly the fields that change the pixels."""
    canon = {k: (spec or {}).get(k) for k in CACHE_KEY_FIELDS}
    blob = json.dumps(canon, ensure_ascii=False, sort_keys=True,
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def segment_seed(base_seed: int, index: int) -> int:
    return int(int(base_seed) + int(index)) % (2 ** 32)


def plan(segments: list[dict], v2state: dict, regenerate: set[int] | None = None,
         use_tail_relay: bool = True) -> list[dict]:
    """Per-segment action list. Never rebuilds a FROZEN clip.

    Actions: "locked" (frozen, keep file), "keep" (done + cache hit, keep file),
    "generate" (must build). A generated clip breaks the tail chain for every
    later non-frozen clip -> they become "generate" too, reported with
    relay_rebuilt=True. A frozen clip downstream of a regenerated one keeps its
    file but is flagged relay_stale=True (explicit, never silent).
    """
    regenerate = set(regenerate or [])
    clips = (v2state or {}).get("clips", {})
    out: list[dict] = []
    chain_broken = False
    for i, seg in enumerate(segments):
        st = clips.get(str(i), {})
        status = st.get("status", STATUS_PENDING)
        key = cache_key(st.get("spec", {})) if st.get("spec") else ""
        want_key = cache_key(st.get("want", {})) if st.get("want") else key
        rec = {"index": i, "action": "generate", "relay_rebuilt": False,
               "relay_stale": False, "cache_hit": False}
        if status in FROZEN_STATUSES and st.get("path"):
            rec.update(action="locked")
            if chain_broken:
                rec["relay_stale"] = True
        elif (i in regenerate) or status in (STATUS_PENDING, STATUS_ERROR) \
                or not st.get("path"):
            rec.update(action="generate")
            chain_broken = True
        elif key and key == want_key:
            rec.update(action="keep", cache_hit=True)
            # keep does not repair the chain: tail input changed upstream.
            if chain_broken and use_tail_relay:
                rec.update(action="generate", relay_rebuilt=True)
        else:
            rec.update(action="generate")
            chain_broken = True
        out.append(rec)
    return out


def resume_snapshot(cursor: int, clips: dict, base_seed: int,
                    ref_hashes: list[str], transitions: dict | None = None) -> dict:
    """Everything a stopped story needs to resume identically."""
    return {"version": 1, "cursor": int(cursor), "clips": clips,
            "base_seed": int(base_seed), "ref_hashes": list(ref_hashes),
            "transitions": transitions or {}}


def apply_snapshot(snapshot: dict) -> tuple[int, dict, int, list[str], dict]:
    s = snapshot or {}
    return (int(s.get("cursor", 0)), dict(s.get("clips", {})),
            int(s.get("base_seed", 0)), list(s.get("ref_hashes", [])),
            dict(s.get("transitions", {})))
