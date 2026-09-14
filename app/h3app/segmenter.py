# -*- coding: utf-8 -*-
"""Rule-based Japanese story splitter.

Pure Python: no I/O, no clock, no LLM, no randomness. The same text always
produces the same segments, which is what makes the UI's "分割をやり直す" button
predictable and what makes it testable from server.py --selftest.

The output is a STARTING POINT. Every segment is meant to be edited by hand in
the UI afterwards, so the rules here aim for "obviously reasonable" rather than
"linguistically correct".

Public API
    split_lines(text)                       -> [{"type","text"}]   (INFERS types)
    segment_lines(lines, seconds=5.0, delivery=None)       -> [segment, ...]
    segments_from_text(text, seconds=5.0, delivery=None)   -> [segment, ...]
    normalize_lines(lines)                  -> [{"type","text"}]   (no inference)
    segments_from_lines(lines, seconds=5.0, delivery=None) -> [segment, ...]
                                                             (no inference)
    segment_timing(segment, seconds=5.0, delivery=None)    -> {...}

STRUCTURE CONTRACT
    The quotative-と heuristic and the quote detection live in split_lines and
    are only ever applied when converting RAW PLAIN TEXT. Once the user has
    marked a line in the editor, that mark is authoritative: segments_from_lines
    never re-classifies a line, never moves text between prompt and speech, and
    never rewrites a line's characters. It only decides WHERE the segment
    boundaries go - preferring boundaries between lines, and only falling back to
    sentence ends (then commas) inside a single utterance that is longer than one
    segment can speak.

SPLIT PRIORITY - THE ONE RULE THAT DECIDES EVERY BOUNDARY
    paragraph / scene block  >  sentence end (。！？)  >  clause  >  、
    It is enforced STRUCTURALLY, not by scoring:
      * paragraph  A typed line, and a scene line together with the utterance it
                   introduces, are ONE BEAT. The packer moves whole beats and can
                   never cut inside one, so a line boundary is the only boundary
                   that exists until a beat alone does not fit a clip.
      * sentence   Only a beat whose utterance ALONE busts the clip is opened up,
                   and it is opened up into SENTENCES. Those sentences are handed
                   back to the SAME packer as ordinary beats, so they get packed,
                   candidate-preferred and merged exactly like typed lines do.
      * clause     An action marker (その後 / 次に ...) is a preferred cut point.
      * 、         LAST RESORT. Only when ONE sentence, at this segment's delivery,
                   still does not fit a clip. Every piece produced that way is
                   flagged `fallback_comma_split` so it is visible in the trace.
    There is exactly ONE code path: nothing bypasses the packer.

SCENE INHERITANCE
    When one scene's dialogue spans several segments, the follow-on segments get
    a short, GENERIC continuation note as their prompt (SCENE_CONTINUATION_NOTE)
    plus a `scene` trace pointing back at the scene line they came from. The
    scene text itself is NOT copied forward: repeating "walk to the chair and sit
    down" in three consecutive clips makes the character do it three times. What
    actually carries the pose across the boundary is the Transition Planner
    (h3app\\transitions.py), whose entry state for a follow-on segment is the
    previous segment's exit state - i.e. exactly the configuration the scene
    established - and whose default for every field is "unchanged".

HOW THE BOUNDARIES ARE CHOSEN - ESTIMATED SPEECH DURATION
    A clip is a fixed number of SECONDS, so the only thing that can decide how
    much text fits in one is how long that text takes to SAY. duration.py turns
    text into seconds (mora-based, see that module); this module packs beats
    into the time budget.

    Nothing breaks a segment by COUNTING. There is no maximum number of stage
    directions per segment and no punctuation mark that forces a cut. An action
    marker (その後 / 次に ...) and a sentence end are SPLIT CANDIDATES: the
    packer prefers to cut there when it has to cut, and ignores them when it
    does not. That is the whole difference from the previous version, which
    broke a segment after two prompt clauses or at any action marker and so
    produced silent one-instruction clips out of a script that only had a few
    seconds of dialogue in it.

A segment is
    {"index": int,
     "lines": [{"type","text"}, ...],
     "prompt": "<prompt lines joined by \\n>",
     "speech": "<speech lines joined by \\n>",
     "delivery": {"pace","emotion","pause_level","action_load"},
     "timing": {"speech_units", "estimated_speech_seconds", "estimated_seconds",
                "segment_duration", "pack_capacity", "density", "band",
                "split_reason", "boundary_type", "entry_boundary_type",
                "fallback_comma_split", "source_units", "scene_index",
                "scene_indices", "scene_continuation", "delivery", ...},
     "status": "pending", "clip": None, "error": ""}

`timing` is debug metadata: it explains why the segment ends where it ends and
carries the numbers a real H3 clip can be measured against. It is never part of
any prompt text.
"""
from __future__ import annotations

from . import duration

LineType = str          # "prompt" | "speech"

TYPE_PROMPT = "prompt"
TYPE_SPEECH = "speech"

# 開き括弧 -> 閉じ括弧。左右が違う記号だけをここに置く。
_OPEN_TO_CLOSE = {
    "「": "」",   # 「 」
    "『": "』",   # 『 』
    "“": "”",   # “ ”
    "〈": "〉",   # 〈 〉
    "《": "》",   # 《 》
}
# 左右が同じ記号は数えずに「次に出てきた同じ記号」で閉じる。
_SYMMETRIC = {'"', "″"}

# 行動転換マーカー。強い区切りとして扱い、マーカーは「後ろの節」に付ける。
ACTION_MARKERS = (
    "その後",        # その後
    "そのあと",  # そのあと
    "次に",              # 次に
    "つぎに",        # つぎに
    "それから",  # それから
    "そして",        # そして
    "まず",              # まず
    "最後に",        # 最後に
    "続いて",        # 続いて
)

# 文末記号（節の区切り、記号は前の節に残す）
_SENTENCE_END = "。！？!?."
# 読点（もっとも弱い区切り）
_COMMA = "、，,"
# マーカーの直前に来てよい文字（語中の偶然一致を強い区切りにしない）
_MARKER_PREV_OK = set(_SENTENCE_END + _COMMA + "\n」』”〉》\"")

# 1 節がこれより長ければ読点でさらに割る。
_LONG_CLAUSE = 40

# 境界の種類（強い順）。trace にそのまま入る。
BOUNDARY_PARAGRAPH = "paragraph"   # 行 / 場面ブロックの切れ目
BOUNDARY_SENTENCE = "sentence"     # 。！？ の直後
BOUNDARY_CLAUSE = "clause"         # 行動転換マーカー（その後 / 次に ...）
BOUNDARY_COMMA = "comma"           # 読点。最後の手段だけに使う
BOUNDARY_END = "end"               # 物語の終わり（切れ目ではない）

# 同じ場面が複数クリップにまたがるときに、2 本目以降へ付ける継続メモ。
# 場面文そのものはコピーしない（動作を毎クリップやり直してしまう）。姿勢・位置・
# カメラの引き継ぎは Transition Planner の entry/exit state が担当する。
SCENE_CONTINUATION_NOTE = (
    "（同じ場面の続き）直前のクリップの姿勢・立ち位置・カメラをそのまま引き継ぎ、"
    "新しい動作は始めずに、同じ流れのまま話し続ける。"
)

# セグメントの既定の長さ（秒）。実際の値は呼び出し側 (config.story) から来る。
DEFAULT_SECONDS = 5.0

# 引用の「と」に続く発話動詞。語幹の前方一致で見る（活用は追わない）:
#   話す / 話しながら / 話します / 紹介する / 紹介を始める ... はすべて拾える。
SPEECH_VERBS = (
    "話", "言", "語", "答え", "叫", "つぶや", "呟", "しゃべ", "喋",
    "紹介", "挨拶", "あいさつ", "説明", "告げ", "呼びかけ", "述べ", "尋ね",
    "問いかけ", "口にす", "つげ",
)
# 節末が動詞らしいかの判定に使う末尾文字。「とピースサイン」のような体言止めの
# 断片だけを見分けられればよいので、これで十分。
_VERB_TAIL = set("うくぐすずつぬぶむるたらだてでろよん")


def _at_clause_boundary(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in _MARKER_PREV_OK


def _speech_verb_follows(text: str, index: int) -> bool:
    """Is there a speech verb between `index` and the end of this sentence?"""
    end = len(text)
    for i in range(index, len(text)):
        if text[i] in _SENTENCE_END:
            end = i
            break
    window = text[index:end]
    return any(v in window for v in SPEECH_VERBS)


def _looks_verbless(text: str) -> bool:
    body = text.strip().rstrip(_SENTENCE_END + _COMMA)
    if not body:
        return True
    if any(v in body for v in SPEECH_VERBS):
        return False
    return body[-1] not in _VERB_TAIL


def _split_quotative(text: str) -> list[tuple[str, str]]:
    """Recover unquoted speech written with the quotative pattern <発話>と<発話動詞>.

    「はじめまして…よろしくお願いします。と笑顔で自己紹介する」 - the part before
    the と is what the character SAYS; the と and the verb phrase describe HOW it
    is said and stay in the prompt. Without this the greeting would be handed to
    the director as a camera instruction and the clip would have no dialogue.

    Nothing is changed when no quotative と is found, so plain stage directions
    come through exactly as before.
    """
    quots = [i for i, ch in enumerate(text)
             if ch == "と" and _at_clause_boundary(text, i)
             and _speech_verb_follows(text, i + 1)]
    if not quots:
        return [(TYPE_PROMPT, text)]

    # An action-transition marker is a hard boundary: the utterance can only
    # reach back as far as the marker that starts its own beat, and the marker
    # itself stays on the prompt side.
    marks: list[int] = []
    for i in range(1, len(text)):
        m = _marker_at(text, i)
        if m and text[i - 1] in _MARKER_PREV_OK:
            end = i + len(m)
            while end < len(text) and text[end] in _COMMA:
                end += 1
            marks.append(end)

    out: list[tuple[str, str]] = []
    pos = 0
    for q in quots:
        region = max([pos] + [m for m in marks if pos <= m <= q])
        before = text[region:q].strip().rstrip(_COMMA).strip()
        if not before:
            continue
        if region > pos:
            out.append((TYPE_PROMPT, text[pos:region]))
        out.append((TYPE_SPEECH, before))
        pos = q
    out.append((TYPE_PROMPT, text[pos:]))
    return [(t, s) for t, s in out if s.strip()]


# ------------------------------------------------------------------ quotes ---
def _quote_spans(text: str) -> list[tuple[int, int]]:
    """Top-level quoted spans as (start, end_exclusive), quote marks included."""
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _OPEN_TO_CLOSE:
            close = _OPEN_TO_CLOSE[ch]
            depth, j = 1, i + 1
            while j < n:
                if text[j] == ch:
                    depth += 1
                elif text[j] == close:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n:
                spans.append((i, j + 1))
                i = j + 1
                continue
        elif ch in _SYMMETRIC:
            j = text.find(ch, i + 1)
            if j != -1:
                spans.append((i, j + 1))
                i = j + 1
                continue
        i += 1
    return spans


def _walk(text: str, out: list[tuple[str, str]]) -> None:
    """Flatten text into (type, chunk) runs.

    Only a LEAF quote (one that contains no further quote) becomes speech. A
    quote that wraps other quotes is a container - typically the user pasting a
    whole script inside 「...」 - so its marks are dropped and the inside is
    walked again. Without this rule the entire input would land in one speech
    line and every action instruction would be read as dialogue.
    """
    pos = 0
    for start, end in _quote_spans(text):
        if start > pos:
            out.append((TYPE_PROMPT, text[pos:start]))
        inner = text[start + 1:end - 1]
        if _quote_spans(inner):
            _walk(inner, out)
        else:
            out.append((TYPE_SPEECH, inner))
        pos = end
    if pos < len(text):
        out.append((TYPE_PROMPT, text[pos:]))


def split_lines(text: str) -> list[dict]:
    """Text -> typed editor lines. Unquoted text is a prompt line by default."""
    runs: list[tuple[str, str]] = []
    _walk(text or "", runs)

    lines: list[dict] = []
    for kind, chunk in runs:
        if kind == TYPE_SPEECH:
            body = chunk.strip()
            if body:
                lines.append({"type": TYPE_SPEECH, "text": body})
            continue
        # A real newline is a hard break the user typed: keep it as a line break.
        for piece in chunk.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            body = piece.strip()
            if not body:
                continue
            for kind, part in _split_quotative(body):
                part = part.strip()
                if part:
                    lines.append({"type": kind, "text": part})
    return lines


# ----------------------------------------------------------------- clauses ---
def _marker_at(text: str, index: int) -> str:
    for m in ACTION_MARKERS:
        if text.startswith(m, index):
            return m
    return ""


def _split_on_markers(text: str) -> list[tuple[str, bool]]:
    """(clause, starts_with_action_marker). The marker stays with what follows."""
    cuts = [0]
    for i in range(1, len(text)):
        if _marker_at(text, i) and text[i - 1] in _MARKER_PREV_OK:
            cuts.append(i)
    cuts.append(len(text))
    out: list[tuple[str, bool]] = []
    for a, b in zip(cuts, cuts[1:]):
        body = text[a:b].strip()
        if body:
            out.append((body, bool(_marker_at(body, 0))))
    return out


def _split_sentences(text: str) -> list[str]:
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_END:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _split_commas(text: str) -> list[str]:
    """Weakest split. Only used to break up a clause that is already too long."""
    if len(text) <= _LONG_CLAUSE:
        return [text]
    pieces, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _COMMA and len("".join(buf).strip()) >= _LONG_CLAUSE // 2:
            piece = "".join(buf).strip()
            if piece:
                pieces.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        if pieces and len(tail) < 4:
            pieces[-1] += tail
        else:
            pieces.append(tail)
    return pieces or [text]


def _prompt_clauses(text: str) -> list[tuple[str, bool, str]]:
    """Prompt line -> [(clause, strong_boundary, boundary_type), ...].

    `boundary_type` records HOW this clause was separated from the one before it,
    so the trace can answer "why was it cut here" without guessing afterwards.
    """
    out: list[tuple[str, bool, str]] = []
    for chunk, strong in _split_on_markers(text):
        first = True
        for sentence in _split_sentences(chunk) or [chunk]:
            pieces = _split_commas(sentence)
            for k, piece in enumerate(pieces):
                if strong and first:
                    kind = BOUNDARY_CLAUSE
                elif k:
                    kind = BOUNDARY_COMMA
                else:
                    kind = BOUNDARY_SENTENCE
                out.append((piece, strong and first, kind))
                first = False
    return out


def _speech_seconds(text: str, delivery: dict) -> float:
    return duration.estimate_speech_seconds(text, delivery=delivery)[0]


def _comma_fallback(sentence: str, budget: float,
                    delivery: dict) -> list[str]:
    """LAST RESORT. Cut ONE sentence at its 、 because it alone busts a clip.

    Only ever reached from _speech_units, and only for a sentence that does not
    fit even on its own. The pieces are grown greedily so a comma is used as
    rarely as the budget allows, and every piece it produces is flagged
    `fallback_comma_split` by the caller. When there is no comma either, the
    sentence comes back whole and the segment is flagged `overloaded` rather than
    being cut in the middle of a word.
    """
    parts = list(_iter_comma_parts(sentence))
    if len(parts) < 2:
        return [sentence]
    out: list[str] = []
    buf = ""
    for part in parts:
        if buf and _speech_seconds(buf + part, delivery) > budget:
            out.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        out.append(buf)
    return [p for p in out if p.strip()] or [sentence]


def _speech_units(text: str, budget: float,
                  delivery: dict) -> list[tuple[str, str, bool]]:
    """An over-long utterance -> [(text, boundary_type, fallback_comma), ...].

    SENTENCES, not clip-sized chunks. The pieces are not packed here on purpose:
    they go straight back into the ordinary packer as beats, so a short sentence
    can be joined to its neighbour, a cut can still be moved onto a stronger
    candidate, and a `too_short` piece can still be merged away. Splitting AND
    packing in this function is what made the internal splitter a second,
    unmanaged code path that produced comma-terminated fragments the packer never
    got a chance to reassemble.

    A 、 is used only for a sentence that does not fit a clip on its own.
    """
    sentences = _split_sentences(text) or [text]
    out: list[tuple[str, str, bool]] = []
    for sentence in sentences:
        if _speech_seconds(sentence, delivery) <= budget:
            out.append((sentence, BOUNDARY_SENTENCE, False))
            continue
        pieces = _comma_fallback(sentence, budget, delivery)
        if len(pieces) < 2:
            # Nothing legal to cut at. Keep the sentence whole and let the band
            # say `overloaded`; never cut mid-word.
            out.append((sentence, BOUNDARY_SENTENCE, False))
            continue
        for k, piece in enumerate(pieces):
            out.append((piece, BOUNDARY_COMMA if k else BOUNDARY_SENTENCE, True))
    return out or [(text, BOUNDARY_SENTENCE, False)]


def _iter_comma_parts(text: str):
    buf = ""
    for ch in text:
        buf += ch
        if ch in _COMMA:
            yield buf
            buf = ""
    if buf:
        yield buf


# ------------------------------------------------------------------- beats ---
def _make_beat(clauses: list[dict], delivery: dict, *, strong: bool = False,
               force_break: bool = False, reason: str = "",
               boundary_type: str = BOUNDARY_PARAGRAPH,
               fallback_comma: bool = False, source_units=(),
               scene_index: int = -1, continuation: bool = False) -> dict:
    """A beat is the smallest thing the packer moves: it is never split.

    Its cost is measured in SECONDS, twice: the time its utterances take to say
    and the time its stage directions take to perform. The two are kept apart
    because they do not add up (see _group_seconds).

    TRACE FIELDS (requirement 7) travel on the beat and end up in the segment's
    `timing` block:
      boundary_type   how this beat was separated from the one before it
      fallback_comma  this beat only exists because one sentence had to be cut
                      at a 、
      source_units    the indices of the user's own lines this beat came from
      scene_index     the index of the scene line that introduced it
      continuation    this beat is a follow-on piece of a split utterance
    """
    items = [{"type": TYPE_SPEECH if c.get("type") == TYPE_SPEECH else TYPE_PROMPT,
              "text": (c.get("text") or "").strip()} for c in clauses]
    items = [c for c in items if c["text"]]
    speech = [c["text"] for c in items if c["type"] == TYPE_SPEECH]
    prompts_ = [c["text"] for c in items if c["type"] == TYPE_PROMPT]
    speech_seconds = sum(_speech_seconds(t, delivery) for t in speech)
    action_seconds = sum(duration.estimate_action_seconds(t, delivery=delivery)[0]
                         for t in prompts_)
    speech_index = next((k for k, c in enumerate(items)
                         if c["type"] == TYPE_SPEECH), None)
    return {
        "clauses": items,
        # A SPLIT CANDIDATE, not a break: the packer prefers to cut here.
        "strong": bool(strong),
        # An unconditional cut in front of this beat. NOTHING sets it any more:
        # the pieces of a cut-up utterance used to force one, which is exactly
        # what stopped them from ever being packed or merged. Kept as the one
        # escape hatch for a future beat that genuinely cannot share a clip.
        "force_break": bool(force_break),
        "break_reason": reason,
        "speech_index": speech_index,
        "speech_seconds": round(speech_seconds, 3),
        "action_seconds": round(action_seconds, 3),
        "speech_units": duration.count_moras("".join(speech)),
        "boundary_type": str(boundary_type or BOUNDARY_PARAGRAPH),
        "fallback_comma": bool(fallback_comma),
        "source_units": [int(i) for i in (source_units or ())],
        "scene_index": int(scene_index),
        "continuation": bool(continuation),
    }


def _group_seconds(beats: list[dict]) -> tuple[float, float, float, float]:
    """(estimated, speech, action, speech_units) for a list of beats.

    ESTIMATED IS NOT speech + action. A stage direction is normally performed
    WHILE the character speaks - she can smile, bow and sit down in the middle
    of a sentence - so physical business only lengthens a clip when there is not
    enough speech to cover it. Treating the two as additive is what made a
    scripted moment with three stage directions look like it needed its own
    silent clip.
    """
    speech = sum(b["speech_seconds"] for b in beats)
    action = sum(b["action_seconds"] for b in beats)
    units = sum(b["speech_units"] for b in beats)
    return round(max(speech, action), 3), round(speech, 3), round(action, 3), units


def _cap(budget: float) -> float:
    """How full the packer may make a clip. See duration.PACK_HEADROOM.

    The nominal clip length is NOT the cap: the estimate is a natural-delivery
    estimate and H3 delivers measurably faster than that, so refusing one more
    sentence over a 0.2 s overshoot throws away a real boundary for a margin
    inside the model's own noise.
    """
    return budget * duration.PACK_HEADROOM


def _fits(beats: list[dict], budget: float) -> bool:
    return _group_seconds(beats)[0] <= _cap(budget) + 1e-9


def _clauses_for(lines: list[dict]) -> list[dict]:
    """Typed lines -> flat clause list.

    A clause that starts with the quotative と keeps it when it carries a verb
    ("と話しながら椅子に座る" describes HOW the line is delivered). A verbless
    fragment ("とピースサイン") would read as a dangling connective on its own, so
    the と is dropped and it becomes a plain stage direction.
    """
    clauses: list[dict] = []
    prev_type = ""
    for index, line in enumerate(lines or []):
        ltype = TYPE_SPEECH if line.get("type") == TYPE_SPEECH else TYPE_PROMPT
        body = (line.get("text") or "").strip()
        if not body:
            continue
        # Two prompt lines in a row can only come from a newline the user typed.
        line_strong = bool(clauses) and ltype == TYPE_PROMPT and prev_type == TYPE_PROMPT
        if ltype == TYPE_SPEECH:
            clauses.append({"type": TYPE_SPEECH, "text": body,
                            "strong": False, "to_lead": False,
                            "boundary_type": BOUNDARY_PARAGRAPH,
                            "source_index": index})
        else:
            fallback = [(body, False, BOUNDARY_PARAGRAPH)]
            for k, (text, strong, kind) in enumerate(_prompt_clauses(body) or fallback):
                stripped = text.lstrip(_COMMA).strip()
                to_lead = stripped.startswith("と")
                if to_lead and _looks_verbless(stripped[1:]):
                    text = stripped[1:].strip() or stripped
                clauses.append({
                    "type": TYPE_PROMPT, "text": text,
                    "strong": strong or (k == 0 and line_strong),
                    "to_lead": to_lead,
                    # The first clause of a typed line IS a line boundary; an
                    # action marker still outranks that as a named cut point.
                    "boundary_type": (kind if (k or kind == BOUNDARY_CLAUSE)
                                      else BOUNDARY_PARAGRAPH),
                    "source_index": index,
                })
        prev_type = ltype
    return clauses


def _beats(clauses: list[dict], delivery: dict) -> list[dict]:
    """Group clauses that must never be separated.

    A beat is [introducing prompt?] + [speech] + [trailing と-action prompts]:
    a quote and the clause that introduces it are one moment on screen, so they
    always end up in the same clip.
    """
    beats: list[dict] = []
    i, n = 0, len(clauses)
    scene = -1
    while i < n:
        if clauses[i]["type"] == TYPE_SPEECH:
            group = [clauses[i]]
            j = i + 1
        elif (i + 1 < n and clauses[i + 1]["type"] == TYPE_SPEECH):
            group = [clauses[i], clauses[i + 1]]
            j = i + 2
        else:
            group = [clauses[i]]
            j = i + 1
        if any(c["type"] == TYPE_SPEECH for c in group):
            while (j < n and clauses[j]["type"] == TYPE_PROMPT
                   and clauses[j]["to_lead"] and not clauses[j]["strong"]):
                group.append(clauses[j])
                j += 1
        if group[0]["type"] == TYPE_PROMPT:
            scene = group[0].get("source_index", -1)
        beats.append(_make_beat(
            group, delivery, strong=bool(group[0]["strong"]),
            boundary_type=group[0].get("boundary_type", BOUNDARY_PARAGRAPH),
            source_units=sorted({c.get("source_index", -1) for c in group}),
            scene_index=scene))
        i = j
    return beats


def _split_long_beat(beat: dict, budget: float, delivery: dict) -> list[dict]:
    """One beat whose utterance alone busts the budget -> SENTENCE beats.

    The introducing clause rides on the first piece and the trailing と-action on
    the last. What comes out is ORDINARY BEATS - not finished segments: they are
    returned to the same packer everything else goes through, so several short
    sentences of one utterance can share a clip and a stranded piece can still be
    merged. Nothing here decides a segment boundary.
    """
    group = beat["clauses"]
    idx = beat["speech_index"]
    intro, speech, trailing = group[:idx], group[idx], group[idx + 1:]
    units = _speech_units(speech["text"], budget, delivery)
    out: list[dict] = []
    for k, (piece, kind, fallback) in enumerate(units):
        clauses: list[dict] = []
        if k == 0:
            clauses.extend(intro)
        clauses.append({"type": TYPE_SPEECH, "text": piece.strip()})
        if k == len(units) - 1:
            clauses.extend(trailing)
        out.append(_make_beat(
            clauses, delivery,
            strong=bool(beat["strong"]) if k == 0 else False,
            # A cut in front of a follow-on sentence is still an arithmetic cut,
            # but naming it tells the trace WHICH utterance was opened up.
            reason="long_utterance" if k else beat.get("break_reason", ""),
            boundary_type=(beat["boundary_type"] if k == 0 else kind),
            fallback_comma=fallback,
            source_units=beat["source_units"],
            scene_index=beat["scene_index"],
            continuation=bool(k)))
    return out


def _expand_long(beats: list[dict], budget: float, delivery: dict) -> list[dict]:
    """Open up only the beats that cannot fit a clip at all. One code path."""
    cap = _cap(budget)
    out: list[dict] = []
    for beat in beats:
        if beat["speech_index"] is not None and beat["speech_seconds"] > cap:
            out.extend(_split_long_beat(beat, cap, delivery))
        else:
            out.append(beat)
    return out


# ----------------------------------------------------------------- packing ---
def _pack(beats: list[dict], budget: float) -> list[dict]:
    """Greedy pack into the time budget. The ONLY reason to start a new segment
    is that the next beat would not fit (or is the continuation of an utterance
    that had to be cut). No count of anything ends a segment."""
    groups: list[dict] = []
    cur: list[dict] = []
    for beat in beats:
        if cur and (beat["force_break"] or not _fits(cur + [beat], budget)):
            groups.append({"beats": cur,
                           "reason": beat["break_reason"] or "budget"})
            cur = []
        cur.append(beat)
    if cur:
        groups.append({"beats": cur, "reason": "end"})
    return groups


# A cut the packer made because the budget ran out, whatever it is called in the
# trace. `long_utterance` is one of these now: a follow-on sentence no longer
# FORCES a break, it just did not fit, so the cut in front of it is as movable as
# any other arithmetic cut.
_ARITHMETIC_REASONS = ("budget", "long_utterance")


def _prefer_candidates(groups: list[dict], budget: float) -> list[dict]:
    """Move a purely arithmetic cut onto the nearest SPLIT CANDIDATE.

    The greedy pass cuts wherever the budget runs out, which can land in the
    middle of a described action. If a stronger candidate boundary (an action
    marker, or a hard line break the user typed) sits a beat or two earlier, and
    cutting there still leaves a usable segment on both sides, the cut is moved
    there. It is never moved when that would push a segment over the budget or
    leave a stub behind - a candidate is a preference, not a rule.
    """
    for i in range(len(groups) - 1):
        left, right = groups[i], groups[i + 1]
        if left["reason"] not in _ARITHMETIC_REASONS or not right["beats"]:
            continue
        if right["beats"][0]["strong"]:
            continue                       # already cutting on a candidate
        for j in range(len(left["beats"]) - 1, 0, -1):
            if not left["beats"][j]["strong"]:
                continue
            moved = left["beats"][j:]
            kept = left["beats"][:j]
            if not _fits(moved + right["beats"], budget):
                continue
            if _group_seconds(kept)[0] / budget < duration.BAND_TOO_SHORT:
                continue                   # would leave a stub behind
            left["beats"], right["beats"] = kept, moved + right["beats"]
            left["reason"] = "action_marker"
            break
    return [g for g in groups if g["beats"]]


def _can_absorb(short: list[dict], other: list[dict], budget: float) -> bool:
    """May a too_short group be merged into `other`?

    Normally: only if the result still fits the clip.

    THE SILENT-LEFTOVER EXCEPTION. A too_short group with NO SPEECH in it is a
    clip in which nobody says anything - a couple of stage directions the packer
    could not place. That is a defect; a clip a few tenths over its estimate is
    not. And absorbing it usually costs NOTHING: stage directions are performed
    WHILE the character speaks (see _group_seconds), so as long as the neighbour
    has more speech seconds than the pair has action seconds, the merged estimate
    is the neighbour's own estimate. So a silent leftover may be merged whenever
    doing so does not make the neighbour any longer than it already was.
    """
    combined = short + other
    if _fits(combined, budget):
        return True
    if any(b["speech_index"] is not None for b in short):
        return False
    return _group_seconds(combined)[0] <= _group_seconds(other)[0] + 1e-9


def _merge_short(groups: list[dict], budget: float) -> list[dict]:
    """Merge a `too_short` segment into a neighbour.

    A segment that carries barely any content is not a shot, it is a leftover.
    The previous neighbour is tried first so that a stage direction stays
    attached to what it introduces. An utterance that genuinely needed its own
    clip keeps it: only a group the packer left behind is ever moved.

    NOTE ON REACHABILITY. The greedy pass only cuts when the next beat does not
    fit, and _group_seconds grows monotonically, so a pair of ordinary groups can
    never be re-joined inside the budget - this pass is not what stops the
    over-splitting any more, the unified beat/packer path is. What is left for it
    is the SILENT leftover, which greedy packing genuinely can strand.
    """
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            density = _group_seconds(g["beats"])[0] / budget
            if duration.classify_band(density) != "too_short":
                continue
            for j in (i - 1, i + 1):
                if not 0 <= j < len(groups):
                    continue
                if not _can_absorb(g["beats"], groups[j]["beats"], budget):
                    continue
                a, b = (j, i) if j < i else (i, j)
                combined = groups[a]["beats"] + groups[b]["beats"]
                groups[a] = {"beats": combined, "reason": groups[b]["reason"],
                             "merged": True}
                del groups[b]
                changed = True
                break
            if changed:
                break
    return groups


def _timing(beats: list[dict], budget: float, delivery: dict, *, reason: str,
            merged: bool = False, boundary_type: str = BOUNDARY_END,
            scene_continuation: bool = False) -> dict:
    """The segment's debug block. Requirement 7: it has to answer BOTH
    "why was it cut here" and "why wasn't the next sentence included"."""
    estimated, speech, action, units = _group_seconds(beats)
    density = round(estimated / budget, 3) if budget else 0.0
    source_units: list[int] = []
    for b in beats:
        for i in b.get("source_units") or ():
            if i >= 0 and i not in source_units:
                source_units.append(i)
    scenes = [b.get("scene_index", -1) for b in beats if b.get("scene_index", -1) >= 0]
    return {
        "speech_units": round(units, 2),
        "estimated_speech_seconds": speech,
        "estimated_action_seconds": action,
        "estimated_seconds": estimated,
        "segment_duration": round(float(budget), 3),
        # What the packer was actually allowed to fill (nominal x PACK_HEADROOM).
        "pack_capacity": round(_cap(float(budget)), 3),
        "density": density,
        "band": duration.classify_band(density),
        "split_reason": ("merged" if merged else reason),
        "merged": bool(merged),
        # WHERE the cut that ends this segment falls, in the split priority.
        "boundary_type": str(boundary_type),
        # HOW this segment was entered - the boundary in front of its first beat.
        "entry_boundary_type": (beats[0].get("boundary_type", BOUNDARY_PARAGRAPH)
                                if beats else BOUNDARY_PARAGRAPH),
        # True only when a 、 had to be used because ONE sentence did not fit.
        "fallback_comma_split": any(b.get("fallback_comma") for b in beats),
        # Which of the user's own lines this segment came from, and which scene
        # line introduced it - so every segment traces back to its scene.
        "source_units": source_units,
        "scene_index": (scenes[0] if scenes else -1),
        "scene_indices": sorted(set(scenes)),
        "scene_continuation": bool(scene_continuation),
        "delivery": dict(delivery),
        "mora_per_second": duration.MORA_PER_SECOND,
    }


def _finalise(groups: list[dict], budget: float, delivery: dict) -> list[dict]:
    segments: list[dict] = []
    for i, group in enumerate(groups):
        beats = group["beats"]
        lines = [dict(c) for b in beats for c in b["clauses"]]
        if not lines:
            continue                        # never emit an empty segment
        # SCENE INHERITANCE. A follow-on piece of a split utterance has no scene
        # line of its own, and a Speech-only clip gives the director nothing to
        # stage. It gets a short GENERIC continuation note - never a copy of the
        # scene text, which would make the character redo the whole action - and
        # the Transition Planner supplies the actual pose through its entry
        # state. See the module docstring.
        continuation = bool(beats and beats[0].get("continuation"))
        if continuation and lines[0]["type"] != TYPE_PROMPT:
            lines.insert(0, {"type": TYPE_PROMPT, "text": SCENE_CONTINUATION_NOTE})
        else:
            continuation = False
        # The boundary that ENDS this segment is the boundary in front of what
        # comes next; the last segment does not end on a cut at all.
        nxt = groups[i + 1]["beats"] if i + 1 < len(groups) else []
        boundary = (nxt[0].get("boundary_type", BOUNDARY_PARAGRAPH) if nxt
                    else BOUNDARY_END)
        segments.append({
            "index": len(segments),
            "lines": lines,
            "prompt": "\n".join(ln["text"] for ln in lines if ln["type"] == TYPE_PROMPT),
            "speech": "\n".join(ln["text"] for ln in lines if ln["type"] == TYPE_SPEECH),
            "delivery": dict(delivery),
            "timing": _timing(beats, budget, delivery,
                              reason=group.get("reason", "end"),
                              merged=bool(group.get("merged")),
                              boundary_type=boundary,
                              scene_continuation=continuation),
            "status": "pending",
            "clip": None,
            "error": "",
        })
    return segments


def _budget(seconds) -> float:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = DEFAULT_SECONDS
    return max(1.0, value)


def _segments_from_beats(beats: list[dict], budget: float,
                         delivery: dict) -> list[dict]:
    beats = _expand_long(beats, budget, delivery)
    groups = _merge_short(_prefer_candidates(_pack(beats, budget), budget), budget)
    return _finalise(groups, budget, delivery)


def segment_lines(lines: list[dict], *, seconds: float = DEFAULT_SECONDS,
                  delivery=None) -> list[dict]:
    """Typed lines -> segments that each take about `seconds` to perform."""
    budget = _budget(seconds)
    d = duration.normalize_delivery(delivery)
    return _segments_from_beats(_beats(_clauses_for(lines), d), budget, d)


def segments_from_text(text: str, *, seconds: float = DEFAULT_SECONDS,
                       delivery=None) -> list[dict]:
    return segment_lines(split_lines(text), seconds=seconds, delivery=delivery)


# ------------------------------------------------- explicit structure entry ---
def normalize_lines(lines: list[dict]) -> list[dict]:
    """Clean a structured line list WITHOUT inferring anything.

    A line marked speech stays speech, a line marked prompt stays prompt, and the
    text is passed through verbatim apart from surrounding whitespace.
    """
    out: list[dict] = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        text = (line.get("text") or "").strip()
        if not text:
            continue
        ltype = TYPE_SPEECH if line.get("type") == TYPE_SPEECH else TYPE_PROMPT
        out.append({"type": ltype, "text": text})
    return out


def _explicit_beats(lines: list[dict]) -> list[list[tuple[int, dict]]]:
    """Group explicit lines that must stay in the same clip.

    Exactly one rule: a prompt line immediately followed by a speech line is the
    clause that introduces that utterance, so the two are one moment on screen.
    Nothing else is regrouped - the user's line order is the structure.

    Each line is returned with its own index, so every segment can be traced back
    to the lines - and the scene - it was built from.
    """
    beats: list[list[tuple[int, dict]]] = []
    i, n = 0, len(lines)
    while i < n:
        if (lines[i]["type"] == TYPE_PROMPT and i + 1 < n
                and lines[i + 1]["type"] == TYPE_SPEECH):
            beats.append([(i, lines[i]), (i + 1, lines[i + 1])])
            i += 2
        else:
            beats.append([(i, lines[i])])
            i += 1
    return beats


def segments_from_lines(lines: list[dict], *, seconds: float = DEFAULT_SECONDS,
                        delivery=None) -> list[dict]:
    """Structured lines -> segments. No type inference, ever.

    Used whenever the UI supplies an explicit prompt/speech structure (the editor,
    /api/story/preview with `lines`, and story creation). Raw plain text still
    goes through split_lines + segment_lines.

    The line TEXT and the line TYPE are passed through verbatim: the only thing
    decided here is where the segment boundaries go. An action marker at the
    start of a prompt line is read as a split CANDIDATE - it never re-types a
    line, never moves text between prompt and speech and never forces a cut.
    """
    budget = _budget(seconds)
    d = duration.normalize_delivery(delivery)
    clean = normalize_lines(lines)
    beats: list[dict] = []
    scene = -1
    for group in _explicit_beats(clean):
        head_index, head = group[0]
        marker = bool(head["type"] == TYPE_PROMPT and _marker_at(head["text"], 0))
        if head["type"] == TYPE_PROMPT:
            # A prompt line is a SCENE: it owns itself and every following
            # utterance until the next prompt line introduces a new one.
            scene = head_index
        beats.append(_make_beat(
            [ln for _, ln in group], d, strong=marker,
            # A typed line is a paragraph boundary; an action marker at the head
            # of one is a named candidate and outranks it in the trace.
            boundary_type=BOUNDARY_CLAUSE if marker else BOUNDARY_PARAGRAPH,
            source_units=[i for i, _ in group],
            scene_index=scene))
    return _segments_from_beats(beats, budget, d)


# --------------------------------------------------------- debug metadata ----
def segment_timing(segment: dict, *, seconds: float = DEFAULT_SECONDS,
                   delivery=None) -> dict:
    """The duration reasoning for ONE segment, recomputed from its content.

    Used by the story runner so the per-segment debug record carries the same
    numbers the packer used, even for a segment the user edited or built by
    hand in the Segment Editor.
    """
    budget = _budget(seconds)
    d = duration.normalize_delivery(delivery)
    lines = [ln for ln in normalize_lines(segment.get("lines") or [])]
    if not lines:
        lines = ([{"type": TYPE_PROMPT, "text": p}
                  for p in (segment.get("prompt") or "").split("\n") if p.strip()]
                 + [{"type": TYPE_SPEECH, "text": s}
                    for s in (segment.get("speech") or "").split("\n") if s.strip()])
    beat = _make_beat(lines, d)
    timing = _timing([beat], budget, d, reason="stored")
    stored = segment.get("timing")
    if isinstance(stored, dict):
        # The numbers are recomputed from what is really about to be generated;
        # the TRACE (why the boundary is here) can only come from the packer run
        # that created the segment, so it is carried over when it is there.
        if stored.get("split_reason"):
            timing["split_reason"] = str(stored["split_reason"])
        for key in ("boundary_type", "entry_boundary_type"):
            if isinstance(stored.get(key), str) and stored[key]:
                timing[key] = stored[key]
        for key in ("fallback_comma_split", "scene_continuation", "merged"):
            if key in stored:
                timing[key] = bool(stored[key])
        if isinstance(stored.get("source_units"), list):
            timing["source_units"] = [int(i) for i in stored["source_units"]
                                      if isinstance(i, int)]
        for key in ("scene_index",):
            if isinstance(stored.get(key), int):
                timing[key] = stored[key]
        if isinstance(stored.get("scene_indices"), list):
            timing["scene_indices"] = [int(i) for i in stored["scene_indices"]
                                       if isinstance(i, int)]
    return timing
