# -*- coding: utf-8 -*-
"""How long does this text take to SAY?

Pure Python: no I/O, no clock, no LLM, no randomness. The same text with the
same delivery always produces the same number, which is what makes the
segmenter testable from server.py --selftest.

WHY NOT len(text)
-----------------
A character count is not a speech unit. 「2026年」 is 5 characters but is read
"にせんにじゅうろくねん" (10 mora); 「きゃっきゃ」 is 5 characters but only 3
mora. Cutting a script into clips by character count therefore cuts it in the
wrong places, and the error is worst exactly where a self-introduction lives:
numbers, product names and small kana.

WHAT IS ESTIMATED
-----------------
    seconds = mora / MORA_PER_SECOND * speech_multiplier
              + pause_seconds * pause_multiplier

MORA is the Japanese timing unit: a regular kana is one mora, a small kana
(ゃゅょぁぃぅぇぉ) belongs to the kana before it and adds nothing, and ー / っ /
ん each occupy one mora of their own even though they carry no vowel.

HONEST APPROXIMATIONS (there is no dictionary in this app)
    * A kanji cannot be read without a dictionary. KANJI_MORA is a documented
      AVERAGE, not a fact: running Japanese text averages roughly 1.5-2 mora per
      kanji (one-kanji verbs stems are short, two-kanji nouns are usually 4 mora
      for 2 kanji). 1.8 is the starting point.
    * A latin run is read either as an English-ish word transliterated into
      katakana, or - when it is a short all-caps token such as AI / RTX / GPU -
      letter by letter (エー・アール・ティー・エックス). The two cases are told
      apart STRUCTURALLY, by the shape of the character run, never by a word
      list.
    * A digit run is read as a numeral, which is far longer than one mora per
      character (5090 -> ごせんきゅうじゅう).

EVERY NUMBER IN THIS MODULE IS A PROVISIONAL STARTING POINT to be calibrated
against real H3 output. They are all collected here, at the top, and every
estimate returns a `detail` dict carrying the intermediate values, so a
measured clip can be compared against the estimate that produced it.

Public API
    estimate_speech_seconds(text, *, delivery)  -> (seconds, detail)
    estimate_action_seconds(text, *, delivery)  -> (seconds, detail)
    count_moras(text)                           -> float
    normalize_delivery(delivery)                -> dict
    delivery_from_presets(style_presets, action_presets, explicit) -> dict
    classify_band(density)                      -> "too_short" | ...
    PACK_HEADROOM                               -> how far past the nominal clip
                                                   length the packer may fill
"""
from __future__ import annotations

# --------------------------------------------------------------- base rate ---
# THE one base rate, and the ONLY knob that decides how much text fits a clip.
#
# CALIBRATION EVIDENCE (measured, not assumed)
# --------------------------------------------
#   * REAL H3 OUTPUT: a 271-visible-character Japanese line came back as a
#     10.33 s clip -> 26.2 characters/second. That is what H3 actually does, and
#     it is audibly too fast: the whole passage landed in 2 clips.
#   * NATURAL TARGET: the same 271 characters delivered at a pace that sounds
#     unhurried wants 3-4 clips of 5 s, i.e. 15-20 s, i.e. 13.6-18.1
#     characters/second. ~15.8 characters/second sits in the middle of that.
#   * CONVERSION: this module does not count characters, it counts MORA, and
#     ordinary Japanese narration comes out of count_moras() at roughly 1.16
#     mora per visible non-punctuation character (kanji are charged KANJI_MORA,
#     kana one each). An estimate also charges PAUSE_* seconds for punctuation,
#     which is about a third of the total on normal prose.
#     Solving "mora / MORA_PER_SECOND + pauses = characters / 15.8" on ordinary
#     narration gives ~27.9. Rounded to 28.0.
#
# THIS IS AN EFFECTIVE RATE, NOT A HUMAN SPEAKING RATE. A person reads Japanese
# at 7-8 mora/second. H3 does not: it compresses the line it is given into the
# clip it is given, and it is measurably ~1.7x faster even than the "natural"
# target above. The number below therefore absorbs three things at once - H3's
# compression, the KANJI_MORA / DIGIT_MORA approximations, and the pause model -
# and its only job is to answer "how much text is a comfortable 5 seconds".
#
# TO RE-TUNE: measure one real clip, divide its visible non-punctuation
# character count by its duration in seconds to get chars/second, decide the
# chars/second you actually want, and scale this constant by the ratio. Nothing
# else in the app has to change.
MORA_PER_SECOND = 28.0

# --------------------------------------------------- mora of a written run ---
# Provisional averages. See the module docstring: these are approximations,
# not linguistic facts.
KANJI_MORA = 1.8                # average reading length of one kanji
DIGIT_MORA = 2.0                # one digit inside a spoken numeral
ACRONYM_LETTER_MORA = 2.6       # a latin letter spelled out (エー / アール / エックス)
LATIN_WORD_MORA_PER_CHAR = 1.6  # a latin word transliterated into katakana

# ------------------------------------------------------------------ pauses ---
# Punctuation is silence, and silence costs clip time just like speech does.
PAUSE_COMMA = 0.20              # 、 ， ,
PAUSE_SENTENCE = 0.35           # 。 ！ ？ ! ? .
PAUSE_UTTERANCE = 0.50          # a line / utterance boundary: longer still
# One trailing utterance pause is charged for every utterance: a spoken line is
# not immediately followed by the next sound, and a clip that ends on the last
# syllable sounds clipped.

# ---------------------------------------------------------- stage direction ---
# A stage direction is not spoken, but it still occupies the clip. These are the
# seconds ONE stage-direction clause is worth at each action load. They are
# small on purpose: physical business is normally performed WHILE the character
# speaks (see the docstring of segmenter._segment_seconds), so this cost only
# lengthens a clip when there is not enough speech to cover it.
ACTION_SECONDS = {"low": 0.30, "medium": 0.50, "high": 0.90}

# ---------------------------------------------------------------- delivery ---
# A delivery descriptor is DATA, not branching code: four named axes, each with
# named levels and one documented multiplier per level. All provisional.
#
#   pace         how fast the words themselves come out
#   emotion      excitement speeds a reading up, sadness slows it down
#   pause_level  how much air is left around the punctuation
#   action_load  how much physical business the segment carries
DELIVERY_AXES = ("pace", "emotion", "pause_level", "action_load")

DEFAULT_DELIVERY = {
    "pace": "normal",
    "emotion": "neutral",
    "pause_level": "normal",
    "action_load": "medium",
}

# Multiplier applied to the MORA-BASED part of the estimate.
PACE_MULTIPLIER = {"slow": 1.25, "normal": 1.00, "fast": 0.85}
EMOTION_MULTIPLIER = {"neutral": 1.00, "calm": 1.05, "excited": 0.92, "sad": 1.12}
# Multiplier applied to the PAUSE part only. The pace multiplier is applied to
# the pauses as well (someone who talks fast also breathes less), so the two
# compose: pause_multiplier = pace * pause_level.
PAUSE_MULTIPLIER = {"low": 0.60, "normal": 1.00, "high": 1.50}

DELIVERY_LEVELS = {
    "pace": tuple(PACE_MULTIPLIER),
    "emotion": tuple(EMOTION_MULTIPLIER),
    "pause_level": tuple(PAUSE_MULTIPLIER),
    "action_load": tuple(ACTION_SECONDS),
}

# How much physical business a segment carries, derived from how many ACTION
# presets it was given. Ordered weakest-first so two hints can be combined by
# taking the stronger one.
_ACTION_LOAD_ORDER = ("low", "medium", "high")

# Preset id -> delivery hint. The ids are the ones in config.STYLE_PRESETS /
# config.ACTION_PRESETS; an id that is not listed changes nothing, so a new
# preset never silently acquires a delivery. Nothing here is character-specific
# or script-specific: it only says what the STYLE the user picked means for
# speed and breathing.
STYLE_DELIVERY_HINTS = {
    "calm":          {"pace": "slow", "emotion": "calm", "pause_level": "high"},
    "subtle":        {"pause_level": "high", "action_load": "low"},
    "static_camera": {"action_load": "low"},
}


def normalize_delivery(delivery=None) -> dict:
    """Fill in the defaults and drop anything unknown. Never raises.

    A segment saved before deliveries existed has no `delivery` field at all;
    it gets DEFAULT_DELIVERY, which is exactly the behaviour it had before.
    """
    out = dict(DEFAULT_DELIVERY)
    if isinstance(delivery, dict):
        for axis in DELIVERY_AXES:
            value = delivery.get(axis)
            if isinstance(value, str) and value in DELIVERY_LEVELS[axis]:
                out[axis] = value
    return out


def delivery_multipliers(delivery=None) -> dict:
    d = normalize_delivery(delivery)
    pace = PACE_MULTIPLIER[d["pace"]]
    return {
        "speech": pace * EMOTION_MULTIPLIER[d["emotion"]],
        "pause": pace * PAUSE_MULTIPLIER[d["pause_level"]],
        "action_seconds": ACTION_SECONDS[d["action_load"]],
    }


def delivery_from_presets(style_presets=(), action_presets=(),
                          explicit=None) -> dict:
    """The delivery of one segment, from what the app already knows.

    STYLE presets (project level) say how the whole story is delivered, the
    segment's own ACTION presets say how much physical business THIS segment
    carries, and an explicit per-segment `delivery` field beats both. Nothing is
    guessed from the text.
    """
    out = dict(DEFAULT_DELIVERY)
    for pid in style_presets or ():
        for axis, level in (STYLE_DELIVERY_HINTS.get(str(pid)) or {}).items():
            if axis in DELIVERY_AXES and level in DELIVERY_LEVELS[axis]:
                out[axis] = level
    actions = [str(p) for p in (action_presets or []) if str(p).strip()]
    if actions:
        # One extra action is normal business, two or more is a busy segment.
        wanted = "high" if len(actions) >= 2 else "medium"
        if _ACTION_LOAD_ORDER.index(wanted) > _ACTION_LOAD_ORDER.index(out["action_load"]):
            out["action_load"] = wanted
    if isinstance(explicit, dict):
        for axis in DELIVERY_AXES:
            value = explicit.get(axis)
            if isinstance(value, str) and value in DELIVERY_LEVELS[axis]:
                out[axis] = value
    return out


# ------------------------------------------------------- character classes ---
_SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")
_MORA_SYMBOLS = set("ーっッんン")      # counted explicitly: they ARE one mora
_COMMA_CHARS = set("、，,")
_SENTENCE_CHARS = set("。！？!?．.")
_DIGITS = set("0123456789０１２３４５６７８９")


def _char_class(ch: str) -> str:
    o = ord(ch)
    if ch in _COMMA_CHARS:
        return "comma"
    if ch in _SENTENCE_CHARS:
        return "sentence"
    if ch in _DIGITS:
        return "digit"
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "latin"
    if 0x3041 <= o <= 0x309F or 0x30A1 <= o <= 0x30FF or o == 0x30FC:
        return "kana"
    if 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or o == 0x3005:
        return "kanji"
    if ch in "\r\n":
        return "newline"
    if ch.isspace():
        return "space"
    return "other"


def _runs(text: str):
    """Group the text into runs of one character class. The run - not the
    character - is the unit that can be turned into a reading."""
    kind = ""
    buf: list[str] = []
    for ch in text or "":
        k = _char_class(ch)
        if k != kind:
            if buf:
                yield kind, "".join(buf)
            kind, buf = k, [ch]
        else:
            buf.append(ch)
    if buf:
        yield kind, "".join(buf)


def _kana_moras(run: str) -> float:
    """Small kana ride on the kana before them; ー / っ / ん stand alone."""
    total = 0.0
    for ch in run:
        if ch in _MORA_SYMBOLS:
            total += 1.0
        elif ch in _SMALL_KANA:
            total += 0.0
        else:
            total += 1.0
    return total


def _latin_moras(run: str) -> tuple[float, str]:
    """(mora, how it was read). Structural, never a word list.

    A short ALL-CAPS run is an initialism and is spelled out letter by letter;
    anything else is read as a word transliterated into katakana.
    """
    if run.isupper() and len(run) <= 5:
        return len(run) * ACRONYM_LETTER_MORA, "acronym"
    return len(run) * LATIN_WORD_MORA_PER_CHAR, "word"


def _run_moras(kind: str, run: str) -> tuple[float, str]:
    if kind == "kana":
        return _kana_moras(run), "kana"
    if kind == "kanji":
        return len(run) * KANJI_MORA, "kanji"
    if kind == "digit":
        return len(run) * DIGIT_MORA, "numeral"
    if kind == "latin":
        return _latin_moras(run)
    return 0.0, kind


def count_moras(text: str) -> float:
    """Speech units in `text`. Punctuation and whitespace are not spoken."""
    return round(sum(_run_moras(k, r)[0] for k, r in _runs(text or "")), 3)


# ---------------------------------------------------------------- estimate ---
def estimate_speech_seconds(text: str, *, delivery=None) -> tuple[float, dict]:
    """How long the character needs to SAY `text`, in seconds.

    Returns (seconds, detail). `detail` carries every intermediate value - the
    mora count, the pause seconds, the multipliers and a per-run breakdown - so
    a measured clip can be diffed against the estimate that produced it.
    """
    body = text or ""
    mult = delivery_multipliers(delivery)
    moras = 0.0
    pause = 0.0
    breakdown: list[dict] = []
    for kind, run in _runs(body):
        if kind == "comma":
            pause += PAUSE_COMMA * len(run)
            breakdown.append({"kind": "comma", "text": run, "mora": 0.0,
                              "pause": round(PAUSE_COMMA * len(run), 3)})
            continue
        if kind == "sentence":
            pause += PAUSE_SENTENCE * len(run)
            breakdown.append({"kind": "sentence", "text": run, "mora": 0.0,
                              "pause": round(PAUSE_SENTENCE * len(run), 3)})
            continue
        if kind == "newline":
            n = len([c for c in run if c == "\n"]) or 1
            pause += PAUSE_UTTERANCE * n
            breakdown.append({"kind": "utterance_boundary", "text": "", "mora": 0.0,
                              "pause": round(PAUSE_UTTERANCE * n, 3)})
            continue
        m, read_as = _run_moras(kind, run)
        moras += m
        if m:
            breakdown.append({"kind": read_as, "text": run,
                              "mora": round(m, 3), "pause": 0.0})
    if body.strip():
        # The breath after the last word. Charged once per utterance.
        pause += PAUSE_UTTERANCE
        breakdown.append({"kind": "utterance_end", "text": "", "mora": 0.0,
                          "pause": PAUSE_UTTERANCE})

    spoken = moras / MORA_PER_SECOND * mult["speech"]
    silent = pause * mult["pause"]
    seconds = round(spoken + silent, 3)
    detail = {
        "kind": "speech",
        "text": body,
        "mora": round(moras, 3),
        "mora_per_second": MORA_PER_SECOND,
        "spoken_seconds": round(spoken, 3),
        "pause_seconds_raw": round(pause, 3),
        "pause_seconds": round(silent, 3),
        "seconds": seconds,
        "delivery": normalize_delivery(delivery),
        "multipliers": {k: round(v, 3) for k, v in mult.items()},
        "runs": breakdown,
    }
    return seconds, detail


def estimate_action_seconds(text: str, *, delivery=None) -> tuple[float, dict]:
    """Clip time taken by ONE stage-direction clause.

    Cost per clause, from `action_load` - NOT a per-character cost and NOT a
    per-clause CAP. A longer instruction does not automatically take longer to
    perform; how much physical business the segment carries is what does.
    """
    mult = delivery_multipliers(delivery)
    seconds = round(mult["action_seconds"], 3) if (text or "").strip() else 0.0
    return seconds, {
        "kind": "action",
        "text": text or "",
        "mora": 0.0,
        "seconds": seconds,
        "delivery": normalize_delivery(delivery),
        "multipliers": {k: round(v, 3) for k, v in mult.items()},
    }


# ------------------------------------------------------------------- bands ---
# HOW FULL IS THE PACKER ALLOWED TO MAKE A CLIP
# ---------------------------------------------
# The estimate above is a NATURAL-DELIVERY estimate, and H3 measurably delivers
# faster than that (26.2 chars/s measured vs the ~15.8 chars/s the estimate is
# calibrated to - a factor of ~1.7). Refusing to put one more sentence in a clip
# because the estimate says 5.2 s instead of 5.0 s therefore throws away a
# boundary for a margin that is inside the noise of the model itself.
#
# So the packer is allowed to fill to PACK_HEADROOM x the clip's nominal
# seconds. 1.15 is deliberately far below the 1.7 that the measurement would
# permit: the goal is "a density H3 can speak naturally", not "every syllable
# provably fits".
PACK_HEADROOM = 1.15

# A segment is classified by DENSITY = estimated seconds / NOMINAL segment
# seconds (the headroom is not divided out, so the number stays readable: 1.0
# means "estimated to be exactly one clip long").
#
#   too_short   < 0.55  Under ~2.75 s of content in a 5 s clip. That is one
#                       short utterance, or a couple of stage directions with
#                       nothing being said over them - a leftover, not a shot.
#                       The packer tries to merge it into a neighbour. Raised
#                       from 0.45: at 0.45 the pieces left behind by splitting a
#                       long utterance (~2.2-2.7 s each) were classified
#                       `comfortable` and so were never merged, which is one of
#                       the three reasons a 4-scene script came out as 19 clips.
#   comfortable        The normal result of greedy packing.
#   dense       > 0.95  Under ~0.25 s free at a 5 s budget, which is less than
#                       the smallest thing the packer could ever add. The
#                       segment is effectively full - not a problem, just full.
#   overloaded  > 1.15  Past what the packer will accept (== PACK_HEADROOM), so
#                       it is only reachable by a single unit that cannot be cut
#                       any further: one sentence with no comma in it that is
#                       longer than a clip. Flagged, never silently truncated.
#
# NOTHING here forces a minimum amount of silence.
BAND_TOO_SHORT = 0.55
BAND_DENSE = 0.95
BAND_OVERLOADED = PACK_HEADROOM


def classify_band(density: float) -> str:
    d = float(density)
    if d > BAND_OVERLOADED:
        return "overloaded"
    if d > BAND_DENSE:
        return "dense"
    if d < BAND_TOO_SHORT:
        return "too_short"
    return "comfortable"
