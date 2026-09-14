# -*- coding: utf-8 -*-
"""Deterministic prompt COMPILER + VALIDATOR for story mode.

WHY THIS MODULE EXISTS
----------------------
Story mode used to trust the director VLM's whole output. Three things went
wrong with that, every one of them unfixable by "prompt harder":

  1. The director invented dialogue. Stage directions and preset labels came
     back wrapped in <d> tags, so H3 read "お辞儀をする" out loud.
  2. Persistent character attributes vanished. A 4B model summarises, so an
     accessory in the CHARACTER PROFILE simply dropped out of <Subject 1>.
  3. Internal monologue leaked. `<|channel>thought`, "Thinking Process:" and
     code fences survived, because strip_tags only replaces three literals.

The design here inverts the trust relationship:

    the LLM supplies only STAGING; the APP compiles the final prompt.

  * Dialogue     the director never sees the text. It writes <<LINE_1>>,
                 <<LINE_2>> ... placeholders; substitute_lines() replaces them
                 with <d>[Japanese] {the user's exact line}</d>.
  * Character    the <Subject 1> line is BUILT from canon.CharacterCanon and
                 REPLACES whatever the director wrote.
  * Hygiene      the six sections are extracted by heading recognition; content
                 before the first heading, after the last, and under any unknown
                 heading is discarded. A surviving monologue marker inside a kept
                 section is a hard refusal, never a silent patch.

Everything here is a pure function: no I/O, no clock, no GPU, no vocabulary.

Public API
    SECTIONS / PLACEHOLDER_RE / PROMPT_SCHEMA_VERSION
    parse_sections(text)                      -> (dict|None, info)
    render_sections(sections)                 -> str
    lenient_clean(text)                        -> str      (single-shot safety)
    hygiene_violations(text)                   -> [name, ...]
    compile_story_prompt(...)                  -> (final, report)
    validate_final_prompt(...)                 -> report
    validation_error_message(index, report)    -> str (Japanese)
"""
from __future__ import annotations

import re

from .canon import CharacterCanon, PERSISTENT_FIELDS, normalize_overrides

PROMPT_SCHEMA_VERSION = 1

# The six sections build\prompts.py's _SCHEMA_CORE contracts for, in the only
# order they are ever emitted.
SECTIONS: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# ------------------------------------------------------------- <d> handling --
# <d ...>...</d>, tolerant of attributes, whitespace and casing.
_D_ELEMENT_RE = re.compile(r"<\s*d\b[^>]*>(.*?)<\s*/\s*d\s*>", re.IGNORECASE | re.DOTALL)
# Any surviving marker at all, including an unpaired opening or closing tag.
_D_MARKER_RE = re.compile(r"<\s*/?\s*d\b[^>]*>", re.IGNORECASE)
# "[Japanese]" style language tag inside a <d> element.
_LANG_TAG_RE = re.compile(r"^\s*\[[^\]]{0,32}\]\s*")

# (S1), (S2), (S1,S2) - the speaker ID, which lives OUTSIDE the <d> tag.
_SPEAKER_RE = re.compile(r"\(\s*S\d+(?:\s*[,、/]\s*S\d+)*\s*\)", re.IGNORECASE)

# How far back the speaker ID may sit before a <d> tag.
_SPEAKER_WINDOW = 400

DIALOGUE_LANG = "[Japanese]"


def dialogue_texts(text: str) -> list[str]:
    """Every <d>...</d> payload, language tag stripped, in document order."""
    return [_LANG_TAG_RE.sub("", m.group(1)).strip()
            for m in _D_ELEMENT_RE.finditer(text or "")]


def has_dialogue_marker(text: str) -> bool:
    return bool(_D_MARKER_RE.search(text or ""))


def dialogue_element(line: str) -> str:
    return f"<d>{DIALOGUE_LANG} {line}</d>"


# ----------------------------------------------------------- placeholders ----
# <<LINE_1>>, << line 2 >>, <<LINE-3>> - tolerant, but always explicit.
PLACEHOLDER_RE = re.compile(r"<<\s*LINE[\s_\-]*(\d+)\s*>>", re.IGNORECASE)


def placeholder(index_1based: int) -> str:
    return f"<<LINE_{int(index_1based)}>>"


# ---------------------------------------------------------------- hygiene ----
# Format-level markers of an LLM talking to itself. Deliberately NOT a list of
# banned words: these are structural artefacts, not vocabulary.
_MONOLOGUE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("special_token", re.compile(r"<\s*\|")),
    ("special_token_close", re.compile(r"\|\s*>")),
    ("think_tag", re.compile(r"<\s*/?\s*think", re.IGNORECASE)),
    ("channel_marker", re.compile(r"\bchannel\s*>?\s*(?:thought|analysis)", re.IGNORECASE)),
    ("thinking_process", re.compile(r"\bthinking\s+process\b", re.IGNORECASE)),
    ("thought_label", re.compile(r"(?:^|\n)\s*thoughts?\s*:", re.IGNORECASE)),
    # The CHANNEL NAME left behind once the transport token around it is
    # normalised away: "<|channel>thought" becomes a bare "thought" line. The
    # envelope is transport, but the channel it names is reasoning, so the
    # residue must still be caught - otherwise stripping the token would have
    # quietly weakened this gate.
    ("reasoning_channel",
     re.compile(r"(?:^|\n)[ \t]*(?:thought|analysis|reasoning)[ \t]*(?:\n|$)", re.IGNORECASE)),
    ("reasoning_label", re.compile(r"(?:^|\n)\s*(?:reasoning|analysis)\s*:", re.IGNORECASE)),
    ("code_fence", re.compile(r"```")),
)


def hygiene_violations(text: str) -> list[str]:
    """Names of the internal-monologue markers present in `text`."""
    body = text or ""
    return [name for name, pattern in _MONOLOGUE_PATTERNS if pattern.search(body)]


# ------------------------------------------- transport token normalisation ---
# Model/transport control tokens, matched by SHAPE rather than by a fixed
# vocabulary so a future model's tokens are handled too:
#     <|channel>  <channel|>  <|channel|>  <|end|>  <|im_end|>  <think> </think>
# The inner name may not contain <, > or | - that is what keeps this away from
# real content. `<Subject 1>`, `<d>`, `<Video 1>`, `<Picture 1>`, `<Audio 1>`
# carry no pipe, so they can never match and are passed through untouched.
_TRANSPORT_RE = re.compile(
    r"<\|[^<>|]{0,40}\|?>"      # <|channel>  <|end|>  <|im_start|>
    r"|<[^<>|]{0,40}\|>"        # <channel|>
    r"|<\s*/?\s*think\s*>",     # <think> </think>
    re.IGNORECASE)


def normalise_transport(text: str) -> tuple[str, dict]:
    """Strip TRANSPORT tokens before any semantic parsing. Stage 1 of the chain.

    Each token becomes a NEWLINE rather than "". That matters: the real answer
    can arrive glued to the end of a prose line, as in

        ... (This matches the desired final structure.)<channel|>subject_definitions

    where `subject_definitions` is the genuine first heading. Deleting the token
    would leave the heading buried mid-line and invisible to a line-based
    matcher; turning it into a line break exposes it.

    This removes the ENVELOPE only. Reasoning PROSE is deliberately left in
    place, for the parser to discard as preamble and - if any of it survives
    into a kept section - for the validator to reject.
    """
    body = str(text or "")
    found = [m.group(0) for m in _TRANSPORT_RE.finditer(body)]
    cleaned = _TRANSPORT_RE.sub("\n", body)
    return cleaned, {"removed_count": len(found),
                     "removed_kinds": sorted(set(t.strip() for t in found))}


# ------------------------------------------------------ section extraction ---
_ENUM_RE = re.compile(r"^\s*\d+\s*[.)\]]\s*")
_DECOR_RE = re.compile(r"[^0-9a-z]+")

# A list item: "* ", "- ", "+ " (marker then whitespace). A heading is never a
# bullet - a bulleted section name is an OUTLINE of the answer, not the answer.
# This is what let the model's own plan ("    *   **subject_definitions:**") be
# mistaken for the document body. Note "**bold**" is emphasis, not a bullet:
# the marker must be followed by whitespace to count.
_BULLET_RE = re.compile(r"^\s*[*+\-]\s+")

# heading id -> canonical section name.
_SECTION_MAP = {name: name for name in SECTIONS}


def _heading_of(line: str) -> str | None:
    """Recognise a section heading, tolerating case, decoration and spacing.

    "subject_definitions", "**Subject Definitions:**", "## SUBJECT DEFINITIONS",
    "3. retention_analysis" all resolve. Anything long enough to be prose does
    not, so a sentence that happens to mention a section name is not a heading.
    """
    raw = (line or "").strip()
    if not raw or len(raw) > 64:
        return None
    if _BULLET_RE.match(raw):
        return None
    raw = _ENUM_RE.sub("", raw)
    raw = raw.strip().strip("#*_-=>` \t").strip()
    raw = raw.rstrip(":：").strip().strip("#*_-=`").strip()
    ident = _DECOR_RE.sub("_", raw.lower()).strip("_")
    return _SECTION_MAP.get(ident)


def _select_block(lines: list[str]) -> tuple[int, int]:
    """Line range of the six-section answer, as (start, end).

    A model that reasons out loud emits its PLAN first and the answer last, so
    the six headings can legitimately appear more than once. The genuine answer
    is the LAST complete, in-order run of all six. Returns the whole document
    when no complete run exists, so a missing section is still reported as
    missing rather than silently reshaped.
    """
    heads = [(i, name) for i, line in enumerate(lines)
             if (name := _heading_of(line)) is not None]
    if not heads:
        return 0, len(lines)

    best: tuple[int, int] | None = None
    for pos, (line_no, name) in enumerate(heads):
        if name != SECTIONS[0]:
            continue
        need = 1
        last = line_no
        for later_line, later_name in heads[pos + 1:]:
            if later_name == SECTIONS[need]:
                need += 1
                last = later_line
                if need == len(SECTIONS):
                    break
            elif later_name == SECTIONS[0]:
                break          # a fresh run starts here; this one is incomplete
        if need == len(SECTIONS):
            end = len(lines)
            for later_line, later_name in heads:
                if later_line > last and later_name == SECTIONS[0]:
                    end = later_line
                    break
            best = (line_no, end)      # keep scanning: prefer the LAST run
    return best if best else (0, len(lines))


def parse_sections(text: str) -> tuple[dict | None, dict]:
    """Schema-bounded extraction of the six-section director output.

    Returns (sections, info). `sections` is None when a section is missing or
    empty; it is never a partially trusted dict.

    info carries what was thrown away, so a refusal can explain itself:
        found / missing / unknown_headings / discarded_prefix / discarded_unknown
    """
    body, transport = normalise_transport(text)
    lines = body.splitlines()
    start, end = _select_block(lines)
    if start > 0 or end < len(lines):
        # Everything outside the chosen block is preamble/epilogue prose.
        pre = [l for l in lines[:start] if l.strip()]
        lines = lines[start:end]
    else:
        pre = []

    sections: dict[str, list[str]] = {}
    order: list[str] = []
    unknown: list[str] = []
    discarded_prefix: list[str] = list(pre)
    discarded_unknown: list[str] = []
    current: str | None = None
    seen_any = False

    for line in lines:
        name = _heading_of(line)
        if name is not None:
            current = name
            seen_any = True
            if name not in sections:
                sections[name] = []
                order.append(name)
            continue
        stripped = line.strip()
        if stripped and _looks_like_unknown_heading(stripped):
            unknown.append(stripped)
            current = None
            continue
        if current is None:
            if stripped:
                (discarded_unknown if seen_any else discarded_prefix).append(stripped)
            continue
        sections[current].append(line)

    info = {
        "found": list(order),
        "missing": [s for s in SECTIONS if not (sections.get(s) and
                                                "".join(sections[s]).strip())],
        "unknown_headings": unknown,
        "discarded_prefix_lines": len(discarded_prefix),
        "discarded_unknown_lines": len(discarded_unknown),
        "discarded_text": "\n".join(discarded_prefix + discarded_unknown),
        "transport": transport,
        "block": {"start": start, "end": end},
    }
    if info["missing"]:
        return None, info
    return {s: "\n".join(sections[s]).strip() for s in SECTIONS}, info


# A short line ending in a colon that is NOT one of the six sections. Treated as
# an unknown heading so its block is discarded rather than folded into the
# previous section.
_UNKNOWN_HEADING_RE = re.compile(r"^[#*_\-=>`\s]*[A-Za-z][A-Za-z0-9 _\-/]{0,40}\s*[:：]\s*$")


def _looks_like_unknown_heading(line: str) -> bool:
    return bool(_UNKNOWN_HEADING_RE.match(line))


def render_sections(sections: dict) -> str:
    """The six sections, canonical order, heading on its own line."""
    blocks = []
    for name in SECTIONS:
        body = (sections.get(name) or "").strip()
        blocks.append(f"{name}\n{body}" if body else name)
    return "\n\n".join(blocks).strip()


def lenient_clean(text: str) -> str:
    """SINGLE-SHOT SAFETY.

    If the whole six-section contract parses cleanly AND no monologue marker
    survives inside the kept sections, return the re-emitted clean text.
    Otherwise return the input untouched, so the verified single-shot path can
    never be made worse by this parser.
    """
    body = str(text or "")
    sections, _info = parse_sections(body)
    if sections is None:
        return body
    rendered = render_sections(sections)
    if hygiene_violations(rendered):
        return body
    return rendered


# ---------------------------------------------------------------- compiler ---
def _subject_definitions_with_canon(block: str, subject_line: str,
                                    label: str = "<Subject 1>") -> tuple[str, dict]:
    """Replace whatever the director wrote for <Subject 1> with the canon line.

    Additional <Subject 1> definition lines are dropped: two contradictory
    definitions of the same label is exactly the drift this whole module exists
    to remove.
    """
    info = {"replaced": False, "inserted": False, "dropped_duplicates": 0,
            "director_subject_line": ""}
    if not subject_line:
        return block, info
    lines = block.splitlines()
    out: list[str] = []
    done = False
    prefix = label.lower()
    for line in lines:
        if line.strip().lower().startswith(prefix):
            if not done:
                info["replaced"] = True
                info["director_subject_line"] = line.strip()
                out.append(subject_line)
                done = True
            else:
                info["dropped_duplicates"] += 1
            continue
        out.append(line)
    if not done:
        info["inserted"] = True
        out.insert(0, subject_line)
    return "\n".join(out).strip(), info


_GAP_RE = re.compile(r"[ \t]{2,}")
_DANGLING_RE = re.compile(r"[ \t]*,[ \t]*(?=[,.;:])|[ \t]+([,.;:])")

# A placeholder token OR a whole <d>...</d> element, whichever comes first.
_SLOT_TOKEN_RE = re.compile(
    PLACEHOLDER_RE.pattern + r"|" + _D_ELEMENT_RE.pattern,
    re.IGNORECASE | re.DOTALL)


def substitute_lines(detailed: str, lines: list[str]) -> tuple[str, dict]:
    """Turn the director's staging into exactly the user's dialogue.

    RECOVER, THEN LET THE VALIDATOR DECIDE. A 4B model will sometimes write a
    <d> tag out of habit even when told not to, and refusing outright would
    block a run that costs minutes of GPU time per clip. So both kinds of token
    are consumed in a single ORDER-OF-APPEARANCE pass:

      <<LINE_k>>                     -> the next unused line
      <d> whose text IS the next
          expected line              -> recovered: it already is that slot
      <d> whose text is an expected
          line, but not the next one -> dropped; the line is still delivered in
                                        its correct position further down
      <d> whose text is NOT an
          expected line              -> INVENTED SPEECH. Dropped and reported.
                                        (For a silent segment every <d> lands
                                        here, which is exactly right.)

    Placeholder numbering is ignored on purpose: a model that emits <<LINE_2>>
    before <<LINE_1>> must not be able to reorder the user's script.

    fewer slots than lines -> the remaining lines are appended to the final shot
    more slots than lines  -> the extras are dropped
    Never invents, never reorders, never edits a character of the user's text.
    """
    body = str(detailed or "")
    out: list[str] = []
    pos = 0
    used = 0
    placeholders_found = 0
    placeholders_dropped: list[str] = []
    recovered: list[str] = []
    dropped_invented: list[str] = []
    dropped_out_of_order: list[str] = []
    speaker_added = 0
    expected = set(lines)

    def _emit_line(line: str) -> None:
        nonlocal speaker_added
        window = "".join(out)[-_SPEAKER_WINDOW:]
        cut = window.rfind("</d>")
        if cut >= 0:
            window = window[cut + 4:]
        if _SPEAKER_RE.search(window):
            prefix = ""
        else:
            prefix = "(S1) "
            speaker_added += 1
        out.append(prefix + dialogue_element(line))

    for m in _SLOT_TOKEN_RE.finditer(body):
        out.append(body[pos:m.start()])
        pos = m.end()
        is_placeholder = m.group(1) is not None
        if is_placeholder:
            placeholders_found += 1
            if used < len(lines):
                _emit_line(lines[used])
                used += 1
            else:
                placeholders_dropped.append(m.group(0))
            continue
        # A <d> element the director wrote itself.
        content = _LANG_TAG_RE.sub("", m.group(2) or "").strip()
        if used < len(lines) and content == lines[used]:
            _emit_line(lines[used])
            recovered.append(lines[used])
            used += 1
        elif content in expected:
            dropped_out_of_order.append(content)
        else:
            dropped_invented.append(content)
    out.append(body[pos:])
    text = "".join(out)

    appended = list(lines[used:])
    for line in appended:
        text = (text.rstrip()
                + f"\n<Subject 1> (S1) continues speaking to the camera, "
                  f"{dialogue_element(line)}")

    # An unpaired "<d" or "</d>" the element regex could not match: it carries no
    # recoverable text, so it is removed rather than left to break the output.
    cleaned = _strip_stray_markers(text)
    stray = len(_D_MARKER_RE.findall(text)) - len(_D_MARKER_RE.findall(cleaned))
    text = cleaned

    # Removing an element leaves "bows deeply,  then" - tidy the horizontal gap
    # it left behind without touching line structure.
    text = _GAP_RE.sub(" ", text)
    text = _DANGLING_RE.sub(r"\1", text)

    return text.strip(), {
        "placeholders_found": placeholders_found,
        "placeholders_used": used - len(recovered),
        "placeholders_dropped": placeholders_dropped,
        "recovered_dialogue": recovered,
        "dropped_invented_dialogue": dropped_invented,
        "dropped_out_of_order_dialogue": dropped_out_of_order,
        "stray_markers_removed": stray,
        "lines_expected": len(lines),
        "lines_appended": appended,
        "speaker_ids_inserted": speaker_added,
    }


def _strip_stray_markers(text: str) -> str:
    """Remove <d> markers that are not part of a well-formed element."""
    keep: list[str] = []
    pos = 0
    for m in _D_ELEMENT_RE.finditer(text):
        keep.append(_D_MARKER_RE.sub("", text[pos:m.start()]))
        keep.append(m.group(0))
        pos = m.end()
    keep.append(_D_MARKER_RE.sub("", text[pos:]))
    return "".join(keep)


def compile_story_prompt(raw_director_output: str, *,
                         canon: CharacterCanon | None = None,
                         overrides: dict | None = None,
                         dialogue_lines: list[str] | None = None) -> tuple[str, dict]:
    """Turn the director's STAGING into the final H3 prompt.

    Returns (final_prompt, report). `final_prompt` is "" when the output could
    not be parsed at all.

    RECOVER, THEN VALIDATE. Only the two things that cannot be recovered are
    compile-time refusals: an answer that is not the six-section contract, and
    internal monologue surviving inside a kept section. Everything else - a <d>
    the director wrote itself, invented speech, a missing or surplus placeholder,
    a redefined <Subject 1> - is repaired deterministically and recorded. The
    caller still refuses whenever report["ok"] is False, and
    validate_final_prompt remains the hard contract on the result.
    """
    lines = [str(x) for x in (dialogue_lines or [])]
    effective_overrides = normalize_overrides(overrides)
    base = canon or CharacterCanon()
    effective = base.with_overrides(effective_overrides)

    report: dict = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "expected_lines": list(lines),
        "silent": not lines,
        "overrides": dict(effective_overrides),
        "canon_fields": sorted(effective.filled()),
        "canon_used": not effective.is_empty,
        "sections": {},
        "subject": {},
        "reconciliation": {},
        "hygiene": [],
        "violations": [],
        "ok": False,
    }

    sections, info = parse_sections(raw_director_output)
    report["sections"] = {k: v for k, v in info.items() if k != "discarded_text"}
    report["discarded_text"] = info.get("discarded_text", "")
    if sections is None:
        report["violations"].append(
            "ディレクターの出力が6セクション形式ではありません（不足: "
            + ", ".join(info["missing"]) + "）。")
        return "", report

    kept = render_sections(sections)

    # Hygiene BEFORE anything else is trusted: a monologue marker inside a kept
    # section means the output is not a prompt, it is the model thinking aloud.
    report["hygiene"] = hygiene_violations(kept)
    if report["hygiene"]:
        report["violations"].append(
            "ディレクターの思考過程がプロンプトに混ざっています（"
            + ", ".join(report["hygiene"]) + "）。")

    report["director_dialogue"] = dialogue_texts(kept)

    # Speech belongs in detailed_description and nowhere else (section rule 5
    # forbids it in overall_soundscape). A <d> in any other section is removed
    # outright - it can never be one of this segment's slots.
    outside: list[str] = []
    for name in SECTIONS:
        if name == "detailed_description":
            continue
        if has_dialogue_marker(sections[name]):
            outside.extend(dialogue_texts(sections[name]))
            sections[name] = _strip_stray_markers(
                _D_ELEMENT_RE.sub("", sections[name])).strip()
    report["dropped_outside_detailed_description"] = outside

    subject_line = effective.subject_line()
    sections["subject_definitions"], subj_info = _subject_definitions_with_canon(
        sections["subject_definitions"], subject_line)
    report["subject"] = subj_info
    report["subject_line"] = subject_line

    sections["detailed_description"], recon = substitute_lines(
        sections["detailed_description"], lines)
    report["reconciliation"] = recon

    final = render_sections(sections)
    report["ok"] = not report["violations"]
    return final, report


# --------------------------------------------------------------- validator ---
def _speaker_id_before_every_tag(text: str) -> bool:
    body = text or ""
    last_end = 0
    for m in _D_ELEMENT_RE.finditer(body):
        window = body[max(last_end, m.start() - _SPEAKER_WINDOW):m.start()]
        if not _SPEAKER_RE.search(window):
            return False
        last_end = m.end()
    return True


def validate_final_prompt(final_prompt: str, *, index: int = 0,
                          dialogue_lines: list[str] | None = None,
                          canon: CharacterCanon | None = None,
                          overrides: dict | None = None,
                          other_dialogue: list[str] | None = None,
                          other_staging: list[str] | None = None) -> dict:
    """THE HARD GATE. Runs on the FINAL prompt, immediately before the H3 graph.

    Checks, in order:
      dialogue integrity   the <d> contents equal the segment's lines exactly,
                           same order, same count. Silent -> zero <d>.
                           Each <d> is preceded by a speaker ID outside the tag.
      character integrity  every persistent trait's canonical text is present;
                           a changed trait needs an explicit override.
      hygiene              no internal-monologue marker, nothing outside the six
                           sections.
      segment integrity    no other segment's dialogue or staging leaked in.
    """
    text = str(final_prompt or "")
    expected = [str(x) for x in (dialogue_lines or [])]
    effective_overrides = normalize_overrides(overrides)
    base = canon or CharacterCanon()
    effective = base.with_overrides(effective_overrides)

    report: dict = {
        "segment_index": int(index),
        "schema_version": PROMPT_SCHEMA_VERSION,
        "mode": "silent" if not expected else "speech",
        "expected_lines": list(expected),
        "actual_lines": dialogue_texts(text),
        "overrides": dict(effective_overrides),
        "persistent_missing": [],
        "persistent_checked": sorted(effective.persistent_traits()),
        "hygiene": [],
        "outside_sections": 0,
        "leaked_dialogue": [],
        "leaked_staging": [],
        "violations": [],
        "ok": False,
    }

    # ---------------------------------------------------- dialogue integrity --
    actual = report["actual_lines"]
    if not expected:
        if has_dialogue_marker(text):
            report["violations"].append(
                "無音セグメントなのにセリフ指示（<d> タグ）が入っています: "
                + (" / ".join(actual) if actual else "壊れた <d> タグ"))
    else:
        if actual != expected:
            report["violations"].append(
                "セリフが指定どおりではありません。指定: ["
                + " / ".join(expected) + "] / 実際: [" + " / ".join(actual) + "]")
        stray = _D_MARKER_RE.findall(text)
        if len(stray) != 2 * len(actual):
            report["violations"].append("壊れた <d> タグが残っています。")
        if not _speaker_id_before_every_tag(text):
            report["violations"].append(
                "セリフの直前に話者ID（(S1) など）がありません。")

    # --------------------------------------------------- character integrity --
    for field, value in effective.persistent_traits().items():
        if value and value not in text:
            report["persistent_missing"].append(field)
    if report["persistent_missing"]:
        report["violations"].append(
            "キャラクターの固定設定がプロンプトから消えています: "
            + ", ".join(report["persistent_missing"]))
    # Which persistent traits were allowed to differ from the profile, and why.
    # The presence check above is what enforces it: for a field WITHOUT an
    # override the value looked for is the profile's own text, for a field WITH
    # one it is the override's text. A trait can therefore only change where an
    # explicit structured override exists.
    report["persistent_overridden"] = sorted(
        f for f in PERSISTENT_FIELDS
        if f in effective_overrides and getattr(base, f, "") != effective_overrides[f])
    report["persistent_unchanged"] = sorted(
        f for f in effective.persistent_traits()
        if f not in report["persistent_overridden"])

    # ------------------------------------------------------------- hygiene ---
    report["hygiene"] = hygiene_violations(text)
    if report["hygiene"]:
        report["violations"].append(
            "プロンプトに思考過程の痕跡が残っています（"
            + ", ".join(report["hygiene"]) + "）。")
    sections, info = parse_sections(text)
    report["outside_sections"] = (int(info.get("discarded_prefix_lines", 0))
                                  + int(info.get("discarded_unknown_lines", 0)))
    if sections is None:
        report["violations"].append(
            "最終プロンプトが6セクション形式ではありません（不足: "
            + ", ".join(info.get("missing", [])) + "）。")
    elif report["outside_sections"] or info.get("unknown_headings"):
        report["violations"].append("6セクション以外の内容が混ざっています。")

    # ------------------------------------------------------ segment integrity -
    own = set(expected)
    for line in (other_dialogue or []):
        line = str(line).strip()
        if line and line not in own and line in text:
            report["leaked_dialogue"].append(line)
    if report["leaked_dialogue"]:
        report["violations"].append(
            "他のセグメントのセリフが混ざっています: "
            + " / ".join(report["leaked_dialogue"]))
    for staging in (other_staging or []):
        staging = str(staging).strip()
        if len(staging) >= 6 and staging in text:
            report["leaked_staging"].append(staging)
    if report["leaked_staging"]:
        report["violations"].append(
            "他のセグメントの指示文が混ざっています: "
            + " / ".join(report["leaked_staging"]))

    report["ok"] = not report["violations"]
    return report


def validation_error_message(index: int, *reports: dict) -> str:
    """Japanese, names the segment, ready to hand to PipelineError."""
    detail: list[str] = []
    for rep in reports:
        detail.extend(rep.get("violations") or [])
    if not detail:
        detail = ["原因不明"]
    return (f"セグメント {int(index) + 1} のプロンプトを検証できませんでした:\n"
            + "\n".join(f"- {d}" for d in detail)
            + "\n生成は開始していません。もう一度お試しください。")
