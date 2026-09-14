# -*- coding: utf-8 -*-
"""Character Canon: the app's own, schema-bounded model of the reference sheet.

WHY THIS MODULE EXISTS
----------------------
The CHARACTER PROFILE produced by the profile VLM is a *text blob*. Story mode
used to paste that blob into the director prompt and then trust whatever the
director wrote into ``<Subject 1>``. A 4B-class model summarises: an accessory
that is in the profile silently disappears from segment 1 and only reappears in
segment 2 because the scene text happened to mention it.

The fix is to stop treating the profile as prose. This module parses it into a
bounded dataclass whose fields are exactly the headings build\\prompts.py's
SYS_CHARACTER_PROFILE asks the VLM to emit:

    FACE HAIR BUILD SKIN OUTFIT ACCESSORIES MATERIALS COLOR_KEYS

Anything else in the blob - preamble, commentary, an invented heading, internal
monologue - is DISCARDED, because it is not in the schema. The compiler then
BUILDS the ``<Subject 1>`` line from these fields and the validator asserts the
persistent ones survived into the final prompt.

Nothing here knows a single character-specific word. "glasses", "red scarf",
"a scar over the left eyebrow" are all just the value of a field.

Public API
    PROFILE_SCHEMA_VERSION
    CANON_FIELDS / PERSISTENT_FIELDS / MUTABLE_FIELDS
    parse_profile(text)            -> CharacterCanon
    normalize_overrides(raw)       -> dict[field, str]
    CharacterCanon.with_overrides / .persistent_traits / .subject_line
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields as dataclass_fields, replace

# Bumped whenever the parser or the field set changes. Folded into the profile
# cache key (projects.profile_cache_key) so a schema change automatically
# invalidates every profile that was cached under the old rules.
PROFILE_SCHEMA_VERSION = 2

# The schema. Order is the order used when the <Subject 1> line is composed.
CANON_FIELDS: tuple[str, ...] = (
    "face", "hair", "build", "skin", "outfit", "accessories",
    "materials", "color_keys",
)

# Carried into EVERY segment. A persistent trait may only change through an
# explicit structured per-segment override.
PERSISTENT_FIELDS: tuple[str, ...] = (
    "face", "hair", "build", "skin", "outfit", "accessories",
)

MUTABLE_FIELDS: tuple[str, ...] = tuple(f for f in CANON_FIELDS
                                        if f not in PERSISTENT_FIELDS)

# Human-facing heading for each field, used when the canonical subject line is
# composed. Deliberately the profile's own heading, so the text the validator
# looks for is the text the VLM was asked to produce.
_HEADINGS = {f: f.upper() for f in CANON_FIELDS}

# Title-case heading ("Color Keys", not "COLOR_KEYS"), used by prose-style
# consumers (e.g. the deterministic H3 compiler's <Subject 1> line). Derived
# from CANON_FIELDS so no second heading map is hand-maintained.
_TITLE_HEADINGS = {f: f.replace("_", " ").title() for f in CANON_FIELDS}


def heading(field: str) -> str:
    """Title-case human heading for a canon field ("Color Keys")."""
    return _TITLE_HEADINGS.get(field, str(field or "").replace("_", " ").title())


def _key_id(name: str) -> str:
    """Case / punctuation / spacing insensitive identity of a heading."""
    return "".join(ch for ch in str(name or "").upper() if ch.isalnum())


# heading id -> canon field. Built from the schema itself plus a couple of
# spelling tolerances, so a future VLM writing "Colour keys:" still lands.
_KEY_MAP: dict[str, str] = {_key_id(f): f for f in CANON_FIELDS}
_KEY_MAP.update({
    "COLOURKEYS": "color_keys",
    "COLORKEY": "color_keys",
    "COLOUR": "color_keys",
    "COLOR": "color_keys",
    "BODY": "build",
    "PHYSIQUE": "build",
    "CLOTHING": "outfit",
    "CLOTHES": "outfit",
    "ACCESSORY": "accessories",
    "MATERIAL": "materials",
})

# "FACE: ...", "**Face:** ...", "- HAIR : ...", "# OUTFIT: ..." - a key line is
# a SHORT leading word group followed by a colon. Anything longer than that is
# prose and cannot start a field.
_KEY_LINE_RE = re.compile(
    r"^\s*(?:[-*•#>\s]*)\**\s*([A-Za-z][A-Za-z0-9 _/-]{1,28}?)\s*\**\s*[:：]\s*(.*)$")

_WS_RE = re.compile(r"\s+")


def _clean(value: str) -> str:
    """One line, no decoration, no angle brackets (they are label syntax)."""
    text = _WS_RE.sub(" ", str(value or "")).strip()
    text = text.strip("*_ \t")
    # A VLM that copies the template literally emits "<face shape, ...>".
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    return text.strip()


@dataclass(frozen=True)
class CharacterCanon:
    face: str = ""
    hair: str = ""
    build: str = ""
    skin: str = ""
    outfit: str = ""
    accessories: str = ""
    materials: str = ""
    color_keys: str = ""

    # ------------------------------------------------------------- accessors --
    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclass_fields(self)}

    def filled(self) -> dict:
        return {k: v for k, v in self.as_dict().items() if v}

    @property
    def is_empty(self) -> bool:
        return not self.filled()

    def persistent_traits(self) -> dict:
        """Field -> canonical text, for the traits that must survive every cut."""
        return {f: getattr(self, f) for f in PERSISTENT_FIELDS if getattr(self, f)}

    def mutable_traits(self) -> dict:
        return {f: getattr(self, f) for f in MUTABLE_FIELDS if getattr(self, f)}

    # -------------------------------------------------------------- mutation --
    def with_overrides(self, overrides: dict | None) -> "CharacterCanon":
        """Apply an explicit, structured per-segment override.

        This is the ONLY way a persistent trait is allowed to change. It is a
        data field, never a keyword sniffed out of the scene text.
        """
        clean = normalize_overrides(overrides)
        if not clean:
            return self
        return replace(self, **clean)

    # ----------------------------------------------------------- composition --
    def profile_text(self) -> str:
        """The canon re-emitted as the profile's own ``KEY: value`` block.

        Fed to the director instead of the raw blob, so preamble, commentary and
        internal monologue in the profile can never reach the director prompt.
        """
        return "\n".join(f"{_HEADINGS[f]}: {getattr(self, f)}"
                         for f in CANON_FIELDS if getattr(self, f))

    def subject_line(self, label: str = "<Subject 1>") -> str:
        """The canonical subject_definitions line, composed by the app.

        Returns "" when nothing parsed, so an unparseable profile degrades to
        "leave whatever the director wrote alone" instead of erasing it.
        """
        parts = [f"{_HEADINGS[f]}: {getattr(self, f)}"
                 for f in CANON_FIELDS if getattr(self, f)]
        if not parts:
            return ""
        return (f"{label} is the main character, identical in every shot. "
                + " ".join(parts)
                + " These attributes are fixed: keep them exactly as written in "
                  "every shot of this clip.")


def parse_profile(text: str) -> CharacterCanon:
    """Schema-bounded parse of the CHARACTER PROFILE block.

    - Content before the first recognised ``KEY:`` line is discarded.
    - Unknown ``KEY:`` lines and everything under them are discarded.
    - A line that is not a ``KEY:`` line continues the current field.
    """
    values: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in str(text or "").splitlines():
        m = _KEY_LINE_RE.match(raw_line)
        if m:
            field = _KEY_MAP.get(_key_id(m.group(1)))
            if field is None:
                current = None                 # unknown heading: discard it
                continue
            current = field
            values.setdefault(field, [])
            rest = _clean(m.group(2))
            if rest:
                values[field].append(rest)
            continue
        if current is None:
            continue                            # preamble / junk / unknown block
        piece = _clean(raw_line)
        if piece:
            values[current].append(piece)
    return CharacterCanon(**{k: _clean(" ".join(v)) for k, v in values.items()})


def normalize_overrides(raw) -> dict:
    """Keep only known canon fields with a non-empty string value."""
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        field = _KEY_MAP.get(_key_id(key))
        if field is None or not isinstance(value, str):
            continue
        cleaned = _clean(value)
        if cleaned:
            out[field] = cleaned
    return out


def merge_overrides(items) -> dict:
    """Fold a sequence of per-segment override dicts, later wins.

    Persistence semantics: once a segment explicitly changes the outfit, the
    NEXT segment keeps the new outfit (that is what "persistent" means). A
    segment that carries no override changes nothing.
    """
    merged: dict[str, str] = {}
    for item in items or []:
        merged.update(normalize_overrides(item))
    return merged
