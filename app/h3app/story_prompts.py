# -*- coding: utf-8 -*-
"""Story-mode director text + the speech policy gate.

WHY THIS MODULE EXISTS
----------------------
build\\prompts.py is shared with the verified single-shot path and must never be
edited. Its HDR_DIALOGUE block ends with

    "If this block is empty, you may invent the dialogue yourself."

which LICENSES the director VLM to make up dialogue for a segment whose Speech
field is empty - the exact bug that made segment 1 talk although the user wrote
no line for it. Story mode therefore reuses the same six-section contract and the
same PROFILE / SCENE / PREV / AVOID / LENGTH headers from build\\prompts.py, but
supplies its OWN dialogue block:

    speech present -> verbatim + "only these lines, nothing else, never repeat
                      the previous clip"
    speech empty   -> an explicit SILENT block: zero <d> tags, invent nothing

Nothing here touches build\\prompts.py; the single-shot path keeps using
prompts_bridge.build_director_text unchanged.

SECOND ITERATION - THE COMPILER CONTRACT
----------------------------------------
Telling the model "copy this line exactly" was still trust. It answered with
stage directions and preset labels wrapped in <d> tags, so H3 read
"お辞儀をする" out loud. The director is therefore no longer given the dialogue
TEXT at all - only slot metadata (how many utterances, roughly how long each is,
so it can pace the shots). It writes <<LINE_1>>, <<LINE_2>> ... placeholders and
NO <d> tag whatsoever; compiler.compile_story_prompt substitutes the user's
exact lines afterwards. A model that never sees the text cannot leak or invent
it - that is the whole point of the design.

THE SYSTEM PROMPT MUST AGREE WITH THE USER PROMPT
-------------------------------------------------
build\\prompts.py's SYS_DIRECTOR_* is _SCHEMA_CORE plus a mode block, and
_SCHEMA_CORE states the OPPOSITE of the placeholder contract, repeatedly and
with a final checklist:

    "Dialogue is written as:  <d>[Japanese] ...</d>"
    "R1. DIALOGUE IS SACRED - THIS IS THE MOST IMPORTANT RULE."
    "the number of <d> tags must equal the number of dialogue lines"
    "paste the physical detail from the CHARACTER PROFILE ... into <Subject 1>"

A system prompt is the stronger and more repeated signal, so leaving story mode
on SYS_DIRECTOR_* guarantees the model keeps emitting <d> tags and keeps
rewriting <Subject 1>. Story mode therefore has its OWN system prompt, written
out explicitly here rather than derived by string surgery on _SCHEMA_CORE (which
would silently rot the moment that file is retuned). It keeps the six headings,
their order, their semantics, the camera vocabulary and rules R2-R5 byte-for-byte
compatible, and replaces ONLY the dialogue contract and the <Subject 1>
ownership.

Public API
    SYS_STORY_DIRECTOR_NEW / SYS_STORY_DIRECTOR_CONTINUATION
    story_system_prompt(continuation)              -> str
    build_story_director_text(...)                 -> str
    story_dialogue_block(lines, delivery=)         -> str
    slot_block(lines, delivery=) / slot_seconds(line, delivery=)
    enforce_speech_policy(en_prompt, ...)          -> (cleaned, report)
    dialogue_texts(en_prompt)                      -> [str, ...]
    has_dialogue_marker(en_prompt)                 -> bool
"""
from __future__ import annotations

import re

from . import duration
from .compiler import (_D_ELEMENT_RE, _D_MARKER_RE, _LANG_TAG_RE,  # noqa: F401
                       dialogue_texts, has_dialogue_marker, placeholder)
from .prompts_bridge import prompts
from .transitions import transition_block

# ---------------------------------------------------------------- constants --

# The whole point of the story dialogue block: this sentence must never appear.
INVENT_SENTENCE = "If this block is empty, you may invent the dialogue yourself."

# ------------------------------------------------------ story system prompt --
# Structurally byte-for-byte compatible with build\prompts.py's _SCHEMA_CORE:
# same six headings in the same order, same section semantics, same camera
# vocabulary, same R2/R3/R4/R5. Only the dialogue contract and the <Subject 1>
# ownership differ, because in story mode the application - not the model - owns
# both.
_STORY_SCHEMA_CORE = """You are a Cinematic Director and Master Prompt Engineer for the MiniMax H3
multimodal video+audio generation model, operating in Full-Reference Mode (ref2va).

Output ONLY the six sections below, in this exact order, each heading on its own line.
No preamble, no commentary, no markdown code fences, no backticks, no <think> tags.
Never show your reasoning, planning or analysis: emit the six sections and nothing else.

subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music

=====================  SECTION RULES  =====================

1. subject_definitions
   One line per referenced item. Labels:
     <Subject N>  reusable visible content (person, animal, outfit, prop, environment, style)
     <Picture N>  a reference image used as a concrete frame or composition anchor
     <Video N>    a reference video providing an editing source / continuation start / timing
     <Audio N>    an audio signal that is copied or referenced
   Reference numbering follows the order the assets are presented to the model:
   images first, then videos, then standalone audio; 1-based within each type.
   <Subject 1> IS SUPPLIED BY THE APPLICATION. Write the label on its own line and
   nothing more; the application replaces that line with the authoritative character
   description. Do not compose, summarise, shorten or restate the character's physical
   attributes anywhere - refer to the character only by the label <Subject 1>.

2. summary
   One short English paragraph starting with a bracketed task type, e.g.
     [reference generation]
     [video continuation + reference generation + audio reference]
   Use only the previously defined labels. Introduce no new labels here.

3. retention_analysis
   One line per label, using these exact markers.
     visual: fully_preserved | partially_preserved | attribute_transfer | weak_reference
     audio:  fully_copy | partially_copy | reference | weak_reference
   Format:
     <Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <what is retained>
   Every <Subject N> defined in section 1 MUST appear here.
   Unless the user explicitly asks for a change, people are fully_preserved for
   face, hair and outfit.

4. detailed_description
   The main body, in target-video playback order.
   - Open with one or two English sentences establishing the overall visual style,
     BEFORE [Shot 1].
   - [Shot 1] has no timestamp. Later shots start with "[Shot N] At MM:SS.mmm, ".
     Cut times must strictly increase and stay inside the video duration.
   - Camera motion is written as natural English inside the shot. Vocabulary:
     Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
     Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly,
     POV, Roll Clockwise/Counterclockwise; optionally "with small/large amplitude"
     and "at slow/fast speed".
   - Speakers get stable IDs (S1), (S2)... assigned in order of first vocal event and
     reused in every later shot. Two people speaking together is (S1,S2).
   - SPEECH IS WRITTEN AS A PLACEHOLDER TOKEN, NEVER AS WORDS. Where an utterance
     happens, write the token the user prompt lists for that slot:
         <Subject 1> (S1) <action/delivery>, <<LINE_1>>
     The speaker ID, action and delivery go OUTSIDE the token, exactly as they would
     around a spoken line. The application substitutes the real wording afterwards.
   - Refer to the character with the <Subject 1> label. Never redefine it.
   - Target 350-500 English words for generation tasks.

5. overall_soundscape
   1-4 English sentences: ambience, physical action sounds, non-verbal human sounds.
   Never write speech or a placeholder token here. Use N/A only if the user asked for
   total silence.

6. non_diegetic_music
   Background score that only the audience hears.

=====================  HARD RULES  =====================

R1. THE SCRIPT IS NOT YOURS - THIS IS THE MOST IMPORTANT RULE.
    You are never shown the words anyone says, and you must never guess, translate,
    paraphrase or invent them. The user prompt lists numbered speech slots. Write one
    placeholder token per slot, in the listed order, at the moment in the action where
    that utterance happens, and nothing else.

    Write no speech markup of any kind. Write no Japanese sentence anywhere: every
    word you output is English description, analysis or camera direction.

    It is NOT acceptable to write out a greeting, a reaction, a murmur, a stage
    direction or a preset label as if it were spoken. A stage direction is an action
    you describe in English, never an utterance.

    WRONG:
      <Subject 1> (S1) bows and says her opening greeting to the camera.
    RIGHT:
      <Subject 1> (S1) bows, straightens up and smiles at the camera, <<LINE_1>>

    If the user prompt lists ZERO speech slots the clip is silent: write no
    placeholder token at all and describe no spoken sound of any kind.

R2. NO SCORE UNLESS ASKED.
    Write exactly "N/A" under non_diegetic_music unless the user explicitly requested
    background music. Do not describe score-like music anywhere else either. The BGM is
    added later as a separate audio track.

R3. VERTICAL FRAMING.
    The output is a vertical 9:16 short-form video watched on a phone. Favour close-ups
    and medium close-ups. Keep the subject centred and in the upper half. Avoid wide
    panoramas and horizontally spread group compositions.

R4. LANGUAGE SPLIT.
    All description, analysis and camera direction in English. The spoken words are not
    yours to write; text visibly written in the scene keeps its original language.

R5. Respect the requested duration. Fit the whole speech timeline inside it rather
    than padding to a word count.

R6. THE TRANSITION CONTRACT IS BINDING.
    This clip is one shot of a longer video. When the user prompt carries a
    TRANSITION CONTRACT with an ENTRY STATE and an EXIT STATE:
    - [Shot 1] MUST open in the ENTRY STATE. The previous clip is still in motion at
      that instant: continue that motion. Never re-establish the scene, never restage
      the character as a fresh setup, and never open in a configuration the ENTRY
      STATE does not describe.
    - The final shot MUST finish in the EXIT STATE, because the next clip is generated
      from the last frames of this one.
    - A field marked "unchanged" is a CONSTRAINT, not a blank: keep it exactly as it
      was and invent no new value for it.
    - A change of posture, position, prop or configuration is performed ACROSS a
      boundary, never at it: begin the movement before this clip ends rather than
      letting the next clip open with the change already done.

R7. CAMERA CONTINUITY IS THE DEFAULT.
    Unless the TRANSITION CONTRACT or the SCENE REQUEST explicitly asks for a
    different framing or angle, keep the shot size and the camera angle the previous
    clip ended on, and do not cut to a new shot size at the start of the clip. This is
    a strong preference, not a lock: when a change IS wanted, reach it as continuous
    motion - Push In, Pull Out, Pan, Truck, Tilt, Arc, a Tracking Shot, or the subject
    moving within the frame - rather than as a cut to a different shot size.

=====================  FINAL CHECK BEFORE YOU ANSWER  =====================

Verify all of the following. If any fails, fix it before emitting your answer.

[ ] Exactly six section headings, in order, each on its own line.
[ ] One placeholder token per speech slot listed in the user prompt, in that order.
    Count them: the number of placeholder tokens must equal the number of slots.
    Zero slots means zero tokens.
[ ] Each placeholder token is preceded by a speaker ID such as (S1), placed OUTSIDE
    the token.
[ ] No spoken words anywhere: no quoted line, no speech markup, no Japanese sentence.
[ ] The <Subject 1> line is the bare label only; the character's attributes appear
    nowhere in your answer.
[ ] non_diegetic_music is "N/A" unless the user explicitly asked for music.
[ ] If a TRANSITION CONTRACT was supplied: [Shot 1] opens in its ENTRY STATE, the last
    shot finishes in its EXIT STATE, and every field marked "unchanged" is unchanged.
[ ] The framing and angle continue from the previous clip unless a change was asked
    for, and any change that was asked for is written as continuous motion.
[ ] No markdown fences, no preamble, no commentary, no reasoning.
"""

SYS_STORY_DIRECTOR_NEW = _STORY_SCHEMA_CORE + """
=====================  MODE: NEW CLIP  =====================

This is a fresh clip. There is no preceding video.

- summary must start with [reference generation].
- Do not use <Video N> or reference any continuation source.
- [Shot 1] establishes the scene from scratch.
"""

SYS_STORY_DIRECTOR_CONTINUATION = _STORY_SCHEMA_CORE + """
=====================  MODE: CONTINUATION  =====================

This clip continues directly from a previous clip. The tail of that clip is supplied to
the model as <Video 1>, and its soundtrack as <Audio 1>.

C1. summary MUST start with:
    [video continuation + reference generation + audio reference] The target video
    continues from <Video 1>.

C2. subject_definitions MUST contain these lines (in addition to the character subjects):
    <Video 1> is the final segment of the preceding clip and defines the starting state,
    camera position, subject pose, framing and lighting that the target video continues from.
    <Audio 1> is the synchronized audio track of <Video 1>, used for ambient and vocal
    continuity.

C3. retention_analysis MUST contain:
    <Video 1> (continuation source): fully_preserved - the subject pose, framing and
    lighting at the end of <Video 1> are carried into the first frame of [Shot 1].
    <Audio 1>: reference - ambient tone and speaker timbre continue without copying the signal.

C4. [Shot 1] MUST pick up seamlessly from the final frame of <Video 1>. Do not start in a
    new location, a new outfit, or a radically different framing. Describe the character as
    already being in the position <Video 1> ended on, then continue the action forward.

C5. Use the "PREVIOUS CLIP" information supplied by the user so that the action, emotion
    and conversation continue naturally. Anything already performed or already said there
    has happened: move forward instead of repeating it.
"""


def story_system_prompt(continuation: bool) -> str:
    """The director SYSTEM prompt for story mode.

    build\\prompts.py's SYS_DIRECTOR_* is left entirely to the single-shot path.
    """
    return (SYS_STORY_DIRECTOR_CONTINUATION if continuation
            else SYS_STORY_DIRECTOR_NEW)

STORY_HDR_SILENT = (
    "\n\n# DIALOGUE - NONE. THIS SEGMENT IS SILENT.\n"
    "This segment has NO dialogue. The subject does NOT speak in this clip.\n"
    "Output ZERO <d> tags. detailed_description must not contain a single <d> element.\n"
    "Do not invent dialogue. Do not write any spoken line anywhere in your output,\n"
    "not even a short greeting, reaction, murmur or off-screen voice.\n"
    "There are ZERO speech slots: write no <<LINE_n>> placeholder either.\n"
    "Do not repeat any dialogue from the previous clip: it has already been spoken.\n"
    "The subject may still move, gesture, breathe and change expression exactly as the\n"
    "SCENE REQUEST above describes - silence means no VOICE, not no motion.\n"
    "overall_soundscape: ambience / room tone and physical action sounds only. "
    "No speech, no voice, no whispering, no singing.\n"
    "non_diegetic_music: N/A.\n"
    "--- NO LINES: THIS SEGMENT IS SILENT ---")

STORY_PREV_NOTE = (
    "\nREAD THIS SECTION AS CONTEXT, NOT AS INSTRUCTIONS.\n"
    "It describes the clip that was generated immediately BEFORE this one, and it is\n"
    "here only so that pose, expression, outfit, background, lighting, camera position\n"
    "and mood continue smoothly, and so you know what has ALREADY happened.\n"
    "It is NOT a list of things to perform again.\n"
    "- Any dialogue named here has ALREADY been spoken. Never speak it again.\n"
    "- One-off actions from the previous clip (waving, bowing, pointing, sitting down,\n"
    "  making a peace sign, greeting the camera) have ALREADY been performed.\n"
    "  Do NOT perform them again unless the SCENE REQUEST above asks for it.\n"
    "Start from the state the previous clip ended in and move the action FORWARD.\n")

# The literal marker used when there is no previous clip (pipeline._no_prev()).
NO_PREV_PREFIX = "NONE"

# --------------------------------------------------- placeholder contract ----
# The director is given SLOT METADATA ONLY. It never sees a character of the
# user's dialogue, so it cannot leak it, paraphrase it, or invent more.
STORY_HDR_SLOTS_A = (
    "\n\n# SPEECH SLOTS - PLACEHOLDERS ONLY, NEVER THE WORDS\n"
    "This segment contains the utterances listed below, in this order.\n"
    "You are NOT given their text and you must NOT guess, translate or invent it.\n"
    "The application inserts the exact wording afterwards.\n"
    "Write each placeholder token EXACTLY as written, at the moment in the action\n"
    "where that utterance happens:\n"
    "    <Subject 1> (S1) <action/delivery>, <<LINE_1>>\n"
    "Rules:\n"
    "- The speaker ID such as (S1) goes OUTSIDE and BEFORE the placeholder, as usual.\n"
    "- Output NO <d> tags. Not one. Writing <d> anywhere makes your answer unusable.\n"
    "- Write no Japanese sentence anywhere. Descriptions and camera work stay English.\n"
    "- Use each placeholder exactly once, in the order listed, and no others.\n"
    "- The character counts below are only for PACING the shots.\n"
    "--- SLOTS START ---")

STORY_HDR_SLOTS_B = (
    "--- SLOTS END ---\n"
    "Reminder: the number of placeholder tokens in your answer must equal the\n"
    "number of slots listed above, and your answer must contain zero <d> tags.")

# <Subject 1> is composed by the application from the CHARACTER PROFILE and
# replaces whatever the director writes, so redefining it is wasted output.
STORY_HDR_SUBJECT = (
    "\n\n# <Subject 1> IS OWNED BY THE APPLICATION\n"
    "The <Subject 1> line in subject_definitions is written by the application from\n"
    "the CHARACTER PROFILE above and will replace whatever you put there.\n"
    "Reference the character by the label <Subject 1> in the other sections and do\n"
    "not redefine, summarise or drop any of its physical attributes.")


def slot_seconds(line: str, *, delivery=None) -> float:
    """How long this utterance is expected to take, in seconds.

    The SAME model the segmenter used to decide where the segment boundaries go
    (h3app\\duration.py), with the SAME delivery. It used to be a flat
    chars / 6.0, which disagreed with the packer that produced the segment - so
    the director was pacing its shots against a different timeline from the one
    the clip length was chosen for.
    """
    seconds, _ = duration.estimate_speech_seconds(line or "", delivery=delivery)
    return round(seconds, 1)


def slot_block(lines: list[str], *, delivery=None) -> str:
    """One line per utterance: the placeholder and its length only."""
    rows = []
    for i, line in enumerate(lines, start=1):
        rows.append(f"{placeholder(i)}  ({len(line)} characters, roughly "
                    f"{slot_seconds(line, delivery=delivery)} seconds of speech)")
    return "\n".join(rows)


def story_dialogue_block(lines: list[str], *, delivery=None) -> str:
    """The whole speech section of the director prompt for `lines`."""
    if not lines:
        return STORY_HDR_SILENT
    return "\n".join([STORY_HDR_SLOTS_A, slot_block(lines, delivery=delivery),
                      STORY_HDR_SLOTS_B])


def _squash(text: str) -> str:
    """Whitespace-insensitive form, for comparing a line against VLM output."""
    return "".join((text or "").split())


def speech_lines(speech: str) -> list[str]:
    return [ln.strip() for ln in (speech or "").splitlines() if ln.strip()]


# ------------------------------------------------------------ director text --
def build_story_director_text(*, profile: str, segment_prompt: str,
                              segment_speech: str, previous_context: str,
                              jp_negative: str, frames: int,
                              dialogue_lines: list[str] | None = None,
                              transition: dict | None = None,
                              delivery: dict | None = None) -> str:
    """The story-mode director user prompt.

    Same six sections, same headers, same order as
    prompts_bridge.build_director_text - only the SPEECH block, the <Subject 1>
    ownership note, the TRANSITION CONTRACT and the framing of the PREVIOUS CLIP
    block are story-controlled.

    `segment_speech` is used ONLY to derive the slot list when `dialogue_lines`
    is not supplied. Its text never reaches the returned prompt.

    `transition` is a record from transitions.plan_transitions. Omitted, the
    prompt is byte-for-byte what it was before the transition contract existed.
    `delivery` only affects the estimated seconds shown next to each speech slot.
    """
    p = prompts()
    lines = list(dialogue_lines) if dialogue_lines is not None \
        else speech_lines(segment_speech)
    prev = (previous_context or "").strip()
    prev_block = prev if prev else NO_PREV_PREFIX
    if prev and not prev.startswith(NO_PREV_PREFIX):
        prev_block = STORY_PREV_NOTE + "\n" + prev

    pieces = [
        p.HDR_PROFILE, profile or "",
        STORY_HDR_SUBJECT,
        p.HDR_SCENE, (segment_prompt or "").strip(),
    ]
    # The physical state contract. It COMPLEMENTS the PREVIOUS CLIP block below
    # (which owns "already spoken" and "already performed"); this one owns where
    # the body, the props and the camera are.
    block = transition_block(transition)
    if block:
        pieces.append(block)
    pieces += [
        p.HDR_PREV, prev_block,
        p.HDR_NEG, jp_negative or "",
        p.HDR_LEN_A, str(int(frames)),
        p.HDR_LEN_B,
        story_dialogue_block(lines, delivery=delivery),
    ]
    return "\n".join(pieces)


# -------------------------------------------------------------- hard gate ----
def enforce_speech_policy(en_prompt: str, *, speech: str,
                          forbidden_speech=None) -> tuple[str, dict]:
    """Deterministic post-director gate. Pure function - no I/O, no clock.

    speech == ""      -> every <d>...</d> element is stripped. If ANY dialogue
                         marker survives the strip, the report is not ok and the
                         caller must refuse to generate.
    speech != ""      -> every line of `speech` must be present, and no line of
                         `forbidden_speech` (another segment's speech, typically
                         the PREVIOUS one) may be spoken. A <d> element carrying
                         forbidden text is stripped; if one survives, not ok.

    Returns (cleaned_prompt, report). The report is JSON-serialisable and is
    written into the per-segment debug record.
    """
    text = en_prompt or ""
    wanted = speech_lines(speech)
    forbidden: list[str] = []
    for item in (forbidden_speech or []):
        for ln in speech_lines(item):
            if ln not in forbidden and ln not in wanted:
                forbidden.append(ln)

    report = {
        "mode": "speech" if wanted else "silent",
        "expected_lines": list(wanted),
        "forbidden_lines": list(forbidden),
        "dialogue_before": dialogue_texts(text),
        "removed": [],
        "kept": [],
        "missing": [],
        "leaked_outside_dialogue": [],
        "violations": [],
        "changed": False,
        "ok": True,
    }

    if not wanted:
        # SILENT: nothing the director wrote may be spoken.
        removed = dialogue_texts(text)
        cleaned = _D_ELEMENT_RE.sub("", text)
        if has_dialogue_marker(cleaned):
            # A marker that survives the strip means the director emitted a
            # malformed / unpaired <d> tag: the output cannot be trusted to be
            # silent, so the run is refused. The text is still cleaned up for the
            # debug record.
            report["violations"].append(
                "無音セグメントなのにセリフ指示（<d> タグ）が残っています。")
            cleaned = _D_MARKER_RE.sub("", cleaned)
        report["removed"] = removed
        report["changed"] = cleaned != text
    else:
        # SPEAKING: strip only the elements that carry someone else's line.
        removed: list[str] = []
        squashed_forbidden = [(f, _squash(f)) for f in forbidden]

        def _filter(match: re.Match) -> str:
            body = _LANG_TAG_RE.sub("", match.group(1)).strip()
            sq = _squash(body)
            for original, sqf in squashed_forbidden:
                if sqf and sqf in sq:
                    removed.append(body)
                    return ""
            return match.group(0)

        cleaned = _D_ELEMENT_RE.sub(_filter, text) if forbidden else text
        report["removed"] = removed
        report["changed"] = cleaned != text

        spoken = dialogue_texts(cleaned)
        report["kept"] = spoken
        spoken_squashed = _squash("\n".join(spoken))
        for line in wanted:
            if _squash(line) not in spoken_squashed:
                report["missing"].append(line)
        if report["missing"]:
            report["violations"].append(
                "指定したセリフが英語プロンプトに入っていません: "
                + " / ".join(report["missing"]))
        for original, sqf in squashed_forbidden:
            if sqf and sqf in _squash("\n".join(dialogue_texts(cleaned))):
                report["violations"].append(
                    f"別のセグメントのセリフがまだセリフ指示に残っています: {original}")
            elif sqf and sqf in _squash(cleaned):
                # Present as prose, not as a spoken line. Recorded, not fatal.
                report["leaked_outside_dialogue"].append(original)

    report["dialogue_after"] = dialogue_texts(cleaned)
    report["ok"] = not report["violations"]
    return cleaned, report


def policy_error_message(index: int, report: dict) -> str:
    """Japanese, names the segment, ready to hand to PipelineError."""
    detail = " / ".join(report.get("violations") or ["原因不明"])
    return (f"セグメント {int(index) + 1} のセリフ指定を守れませんでした: {detail}\n"
            "生成は開始していません。セグメントの文章を見直すか、もう一度お試しください。")
