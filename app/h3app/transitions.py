# -*- coding: utf-8 -*-
"""The TRANSITION CONTRACT: how one clip is allowed to connect to the next.

WHY THIS MODULE EXISTS
----------------------
Every segment is directed in isolation: the director VLM is shown ONE segment's
scene text and writes a self-contained clip for it. The tail relay then splices
those clips together, and the seam shows - the camera distance jumps, the
subject is suddenly seated, the body teleports across the frame, a movement is
cut off half-way. Nothing in the prompt ever told the director where the clip
must START or where it must END.

This module adds that contract, as STRUCTURED DATA rather than prose:

    entry_state   how the clip must OPEN. Inherited from the previous clip's
                  exit_state, so a boundary is a hand-over, not a reset.
    main_action   whatever the segment's own SCENE REQUEST already says. Not
                  modelled here - this module never touches the segment text.
    exit_state    the state the clip must END in, chosen so that the NEXT
                  segment can start from it.

WHAT IS CARRIED (all generic, all expressible in an H3 prompt)
    body pose, facing direction, screen position, gaze, camera framing,
    camera angle, major props, interaction state, ongoing action

DEFAULT IS "UNSPECIFIED"
------------------------
Every field starts at UNSPECIFIED and stays there unless there is a REASON to
set it. UNSPECIFIED is not "anything goes": rendered into the director text it
reads "unchanged from the end of the previous clip", which is exactly the
continuity we want. The app never invents a pose it has no reason to believe -
it has no dictionary of poses, props or shot names, and it never reads the
Japanese scene text looking for one.

WHERE THE HAND-OFF IS PLANNED  (requirement 12)
-----------------------------------------------
A change of posture must be a transition performed ACROSS the boundary, not a
jump AT the boundary. So when segment N+1 introduces a new physical action, it
is segment N's EXIT state that is amended: its `ongoing_action` becomes
HANDOFF, meaning "already beginning the movement the next clip continues".
Segment N+1 then inherits that as its ENTRY state and is told to continue a
movement that is already under way. The later segment is never dropped into an
already-changed configuration.

"Introduces a new physical action" is decided from data the app already has -
the segment's ACTION presets (structured ids), its delivery `action_load`, and
an explicit per-segment `continuity` request. Never from prose.

CAMERA CONTINUITY  (requirement 13)
-----------------------------------
Framing and angle are ordinary continuity fields, so they persist across every
boundary by default. When a segment explicitly asks for a different framing or
angle, the constraint list says so and asks for the change to be reached as
continuous motion instead of a cut to a different shot size. A PREFERENCE, not
a lock: the director may still cut when the scene genuinely needs it.

Pure Python: no I/O, no clock, no randomness. The same segment list always
plans the same states, which is what makes it testable from server.py
--selftest and what makes a resumed story plan identically to a fresh one.

Public API
    CONTINUITY_FIELDS / UNSPECIFIED / HANDOFF
    normalize_state(raw)                     -> dict
    blank_state()                            -> dict
    plan_transitions(segments, deliveries)   -> [record, ...]
    record_for(segments, index, deliveries)  -> record
    transition_block(record)                 -> str   (director text)
"""
from __future__ import annotations

from . import presets as presets_mod
from .config import ACTION_PRESET_IDS

# The fields carried across a boundary. Ordered: this is also the order they are
# rendered in the director text.
CONTINUITY_FIELDS = (
    "body_pose",
    "facing",
    "screen_position",
    "gaze",
    "camera_framing",
    "camera_angle",
    "props",
    "interaction",
    "ongoing_action",
)

# The two fields that describe the SHOT rather than the subject. A segment that
# sets either of them is asking for a camera change.
CAMERA_FIELDS = ("camera_framing", "camera_angle")

# Fields whose value persists forward until something changes it. `ongoing_action`
# is deliberately NOT in here: an action that was under way at the end of one clip
# is not still under way at the end of the next one.
PERSISTENT_FIELDS = tuple(f for f in CONTINUITY_FIELDS if f != "ongoing_action")

# A field the app has no reason to constrain. Rendered as "unchanged", never as
# "free": not knowing what the pose is is exactly the reason to keep it.
UNSPECIFIED = "unspecified"

# The hand-off marker. Set on the EARLIER segment's exit_state when the NEXT
# segment introduces a new physical action.
HANDOFF = "beginning_the_next_action"

# Fields that, when a segment sets them explicitly, mean this segment changes the
# physical configuration (so the previous segment has to hand over into it).
_PHYSICAL_FIELDS = ("body_pose", "facing", "screen_position", "props", "interaction")

# How much physical business makes a segment an "action" segment on its own.
_BUSY_ACTION_LOAD = "high"

# Longest explicit value accepted from a caller. The value is pasted into the
# director prompt, so it is kept to one short clause.
MAX_VALUE_CHARS = 160

FIELD_LABELS = {
    "body_pose": "body pose",
    "facing": "facing direction",
    "screen_position": "screen position",
    "gaze": "gaze",
    "camera_framing": "camera framing",
    "camera_angle": "camera angle",
    "props": "major props",
    "interaction": "interaction state",
    "ongoing_action": "ongoing action",
}

# --------------------------------------------------------------- constraints --
# Generic, content-free rules. `id` is what the tests and the debug record talk
# about; `text` is what the director is shown.
CONSTRAINT_TEXT = {
    "first_clip":
        "This is the first clip: nothing is inherited. Whatever you establish here "
        "becomes the state the next clip has to continue from, so leave the "
        "character in a clear, stable configuration at the end.",
    "continue_from_previous":
        "The previous clip is still in motion at the moment this one starts. Open "
        "in the ENTRY STATE and carry that motion forward; do not re-establish the "
        "scene, do not restage the character and do not begin from a fresh setup.",
    "hold_unchanged":
        "A field marked \"unchanged\" is a CONSTRAINT, not a blank. Keep it exactly "
        "as the previous clip left it and invent no new value for it.",
    "exit_ready":
        "Finish in the EXIT STATE. The next clip is generated from the final frames "
        "of this one, so the last shot has to leave the character in that state.",
    "handoff_out":
        "The NEXT clip begins a new physical action. End this clip already moving "
        "into it - the weight shift, the turn or the reach starts here - so the "
        "change happens ACROSS the boundary instead of jumping at it.",
    "handoff_in":
        "The physical change this clip performs is already under way: the previous "
        "clip started the movement. Continue it from where it is. Do NOT open with "
        "the change already completed and do not perform it twice.",
    "camera_continuity":
        "CAMERA CONTINUITY: keep the shot size and the camera angle the previous "
        "clip ended on. Do not open on a different framing, and do not cut to a new "
        "shot size at the start of the clip.",
    # Real-machine finding (2026-09-13): a continuation whose director text only
    # said "consistent with outdoor natural light" came back on a plain grey
    # studio backdrop instead of the park the previous clip ended in. The
    # environment is part of the state to carry, and it has to be NAMED.
    "setting_continuity":
        "SETTING CONTINUITY: the location, background, set dressing and lighting "
        "are the same as at the end of the previous clip. Name that setting "
        "explicitly in [Shot 1] (the place, what is visible behind the character, "
        "the light). Do not move to a studio, a plain backdrop, or a different "
        "place unless the SCENE REQUEST asks for it.",
    "camera_change_as_motion":
        "This clip asks for a different framing or angle. Reach it as continuous "
        "motion - Push In, Pull Out, Pan, Truck, Tilt, Arc, a Tracking Shot, or the "
        "subject moving within the frame - rather than as a cut to a different shot "
        "size.",
}

# ------------------------------------------------------------- header text ----
TRANSITION_HEADER = (
    "\n\n# TRANSITION CONTRACT - WHERE THIS CLIP STARTS AND WHERE IT MUST END\n"
    "This clip is one shot of a longer video. The state below is BINDING: it is what\n"
    "makes consecutive clips look like one continuous take instead of separate takes.\n"
    "[Shot 1] must OPEN in the ENTRY STATE, and the final shot must FINISH in the\n"
    "EXIT STATE. Perform the SCENE REQUEST above between the two.")

ENTRY_HEADING = "--- ENTRY STATE: the clip must OPEN here ---"
EXIT_HEADING = "--- EXIT STATE: the clip must FINISH here ---"
CONSTRAINT_HEADING = "--- CONTINUITY CONSTRAINTS ---"
TRANSITION_FOOTER = (
    "--- TRANSITION CONTRACT END ---\n"
    "Reminder: opening in a configuration the ENTRY STATE does not describe is a\n"
    "visible jump at the cut, and so is ending anywhere other than the EXIT STATE.")


# ------------------------------------------------------------------ states ---
def blank_state() -> dict:
    """Every field unspecified: the app knows nothing and constrains nothing."""
    return {field: UNSPECIFIED for field in CONTINUITY_FIELDS}


def _clean_value(value) -> str:
    if not isinstance(value, str):
        return UNSPECIFIED
    text = " ".join(value.split())          # newlines and runs of space collapse
    if not text:
        return UNSPECIFIED
    return text[:MAX_VALUE_CHARS]


def normalize_state(raw) -> dict:
    """Accept whatever is stored on a segment and make it a well-formed state.

    Unknown keys are discarded, non-string values are discarded, and anything
    missing defaults to UNSPECIFIED. Never raises.
    """
    out = blank_state()
    if isinstance(raw, dict):
        for field in CONTINUITY_FIELDS:
            out[field] = _clean_value(raw.get(field))
    return out


def state_is_empty(state: dict) -> bool:
    return all((state or {}).get(f, UNSPECIFIED) == UNSPECIFIED
               for f in CONTINUITY_FIELDS)


# ------------------------------------------------------------- the planner ---
def _action_ids(segment: dict) -> list[str]:
    """The ACTION signals attached to THIS segment.

    Two sources, both STRUCTURED - the free Japanese scene text is still never
    inspected:
      * legacy `action_presets` / `motion_presets` ids (old stories), and
      * the deterministic sentences an action-preset chip press wrote into this
        segment's prompt (presets.APPLY_TEMPLATE). Without the second source a
        preset press would silently stop introducing a physical action once the
        chips stopped writing ids.
    """
    ids = [str(p) for p in (segment.get("action_presets") or []) if str(p).strip()]
    for p in (segment.get("motion_presets") or []):
        if str(p) in ACTION_PRESET_IDS and str(p) not in ids:
            ids.append(str(p))
    ids += presets_mod.applied_action_names(segment.get("prompt") or "")
    return ids


def _starts_new_physical_action(segment: dict, delivery=None) -> bool:
    """Does this segment introduce a new physical action?

    Decided from structured data only:
      * it carries ACTION presets (one-off physical business, by definition), or
      * its delivery says the segment is physically busy (action_load high), or
      * it explicitly requests a different body pose / position / props /
        interaction through its own `continuity` field.

    The Japanese scene text is never inspected.
    """
    if not isinstance(segment, dict):
        return False
    if _action_ids(segment):
        return True
    if isinstance(delivery, dict) and delivery.get("action_load") == _BUSY_ACTION_LOAD:
        return True
    requested = normalize_state(segment.get("continuity"))
    return any(requested[f] != UNSPECIFIED for f in _PHYSICAL_FIELDS)


def _requests_camera_change(segment: dict) -> bool:
    requested = normalize_state((segment or {}).get("continuity"))
    return any(requested[f] != UNSPECIFIED for f in CAMERA_FIELDS)


def plan_transitions(segments, *, deliveries=None) -> list[dict]:
    """Plan every boundary of a story at once. PURE and deterministic.

    Runs over the WHOLE segment list before anything is generated, because a
    boundary is a property of a PAIR of segments: `segment[N+1].entry_state` is
    `segment[N].exit_state`, and a change introduced by segment N+1 is handed
    over by amending segment N's exit_state.

    `deliveries` is the per-segment delivery descriptor (duration.py) when the
    caller has it; absent, only the presets and explicit requests are used.
    """
    segs = [s if isinstance(s, dict) else {} for s in (segments or [])]
    n = len(segs)
    deliveries = list(deliveries or [])
    deliveries += [None] * (n - len(deliveries))

    requested = [normalize_state(s.get("continuity")) for s in segs]
    new_action = [_starts_new_physical_action(segs[i], deliveries[i]) for i in range(n)]
    camera_change = [_requests_camera_change(s) for s in segs]

    records: list[dict] = []
    carried = blank_state()                      # the previous segment's exit state
    for i, seg in enumerate(segs):
        entry = dict(carried)

        # The exit state starts as the entry state: everything CONTINUES unless
        # there is a reason for it to change. `ongoing_action` is the exception -
        # a movement that was under way when the clip opened is not automatically
        # still under way when it ends.
        exit_state = dict(entry)
        exit_state["ongoing_action"] = UNSPECIFIED
        for field in CONTINUITY_FIELDS:
            if requested[i][field] != UNSPECIFIED:
                exit_state[field] = requested[i][field]

        # THE HAND-OFF, planned on the EARLIER segment (requirement 12).
        if i + 1 < n and new_action[i + 1] \
                and requested[i]["ongoing_action"] == UNSPECIFIED:
            exit_state["ongoing_action"] = HANDOFF

        constraints: list[dict] = []

        def _add(cid: str) -> None:
            constraints.append({"id": cid, "text": CONSTRAINT_TEXT[cid]})

        if i == 0:
            _add("first_clip")
        else:
            _add("continue_from_previous")
            _add("hold_unchanged")
            _add("setting_continuity")
            if new_action[i] and entry.get("ongoing_action") == HANDOFF:
                _add("handoff_in")
            if camera_change[i]:
                _add("camera_change_as_motion")
            else:
                _add("camera_continuity")
        if i + 1 < n:
            _add("exit_ready")
            if exit_state["ongoing_action"] == HANDOFF:
                _add("handoff_out")

        records.append({
            "segment_index": i,
            "segment_id": str(seg.get("segment_id") or ""),
            "previous_segment_id": (str(segs[i - 1].get("segment_id") or "")
                                    if i > 0 else ""),
            "is_first": i == 0,
            "is_last": i == n - 1,
            "starts_new_physical_action": bool(new_action[i]),
            "requests_camera_change": bool(camera_change[i]),
            "requested_state": dict(requested[i]),
            "entry_state": entry,
            "exit_state": exit_state,
            "continuity_constraints": constraints,
        })
        carried = dict(exit_state)
    return records


def record_for(segments, index: int, *, deliveries=None) -> dict:
    """One segment's transition record, planned in the context of all of them."""
    plan = plan_transitions(segments, deliveries=deliveries)
    if 0 <= index < len(plan):
        return plan[index]
    return {
        "segment_index": int(index), "segment_id": "", "previous_segment_id": "",
        "is_first": index == 0, "is_last": True,
        "starts_new_physical_action": False, "requests_camera_change": False,
        "requested_state": blank_state(),
        "entry_state": blank_state(), "exit_state": blank_state(),
        "continuity_constraints": [],
    }


# ------------------------------------------------------------- the renderer --
def _render_value(field: str, value: str, *, where: str, first: bool) -> str:
    """One state line, in English, for the director text."""
    if value == HANDOFF:
        if where == "entry":
            return ("already in motion from the previous clip - continue that "
                    "movement, do not restage it and do not start it again")
        return ("already beginning the movement the next clip continues - start "
                "it before this clip ends")
    if value == UNSPECIFIED:
        if where == "entry":
            if first:
                return "not inherited - you establish it in this clip"
            return "unchanged from the end of the previous clip"
        return "unchanged from how this clip opened"
    return value


def _state_lines(state: dict, *, where: str, first: bool) -> list[str]:
    return [f"  {FIELD_LABELS[field]}: "
            f"{_render_value(field, state.get(field, UNSPECIFIED), where=where, first=first)}"
            for field in CONTINUITY_FIELDS]


def transition_block(record: dict | None) -> str:
    """The TRANSITION CONTRACT section of the director user prompt.

    Empty string when there is no record, so a caller that does not plan
    transitions produces exactly the prompt it produced before.
    """
    if not isinstance(record, dict) or not record.get("entry_state"):
        return ""
    first = bool(record.get("is_first"))
    pieces = [TRANSITION_HEADER, ENTRY_HEADING]
    pieces += _state_lines(record["entry_state"], where="entry", first=first)
    pieces.append(EXIT_HEADING)
    pieces += _state_lines(record.get("exit_state") or blank_state(),
                           where="exit", first=first)
    constraints = record.get("continuity_constraints") or []
    if constraints:
        pieces.append(CONSTRAINT_HEADING)
        for c in constraints:
            text = c.get("text") if isinstance(c, dict) else str(c)
            if text:
                pieces.append(f"- {text}")
    pieces.append(TRANSITION_FOOTER)
    return "\n".join(pieces)
