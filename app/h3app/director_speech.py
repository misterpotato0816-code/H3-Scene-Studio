# -*- coding: utf-8 -*-
"""Director speech policy + FINAL validation (Spec v2).

Pipeline order (strict): Spec -> compiler -> speech policy -> FINAL
validation -> H3. This module owns the middle two steps for the AI Director
deterministic path only; the normal story path keeps its own gate
(story_prompts.enforce_speech_policy).

HARD errors stop generation before any GPU work (fail-fast):
  - <d> count != dialogue line count
  - <d> text != the specified dialogue (order-sensitive)
  - a <d> with no dialogue origin
  - <Subject 2> / (S2) without an explicit second on-screen subject

WARNINGs (heuristics, never block): quoted speech-looking text outside <d>,
speaking verbs in prose, human-voice ambience leftovers.

Ambience handling converts instead of deleting: crowd/human sounds become
explicitly non-verbal ambience phrases so the scene keeps its atmosphere
without giving H3 anything speakable.
"""
from __future__ import annotations

import re as _re

# Deterministic ambience conversion (applied in order, longest first).
# Crowd/human sounds -> non-verbal ambience. Never deleted outright.
_AMBIENCE_MAP: tuple[tuple[str, str], ...] = (
    ("indistinct crowd ambience, no intelligible background speech",
     ("crowd chatter", "crowd murmur", "cheering crowds", "cheering",
      "crowd speech", "people talking", "human voices", "calling voices",
      "conversations", "conversation voices",
      "歓声", "ざわめき", "人混みのにぎわい", "にぎわい")),
    ("distant vendor calls without intelligible words",
     ("vendors shouting", "vendor calls", "stall calls",
      "屋台の呼び込み", "呼び込みの声", "呼び込み")),
    ("distant taiko drums without voices",
     ("太鼓", "taiko")),
    ("indistinct human presence without intelligible speech",
     ("人の声", "人々の声", "話し声", "crowd voices")),
    ("distant festival music without lyrics or voices",
     ("祭囃子",)),
)

# Heuristic WARNING patterns (outside <d> only). Japanese quotes, speaking
# verbs, residual voice nouns. Voice-quality descriptions ("calm female
# voice") intentionally NOT matched: they describe timbre, not content.
_WARN_QUOTES = ("「", "」", '"', '"', '"')
_WARN_SPEECH_VERBS = (
    "話す", "語る", "ささやく", "歌う", "つぶやく", "叫ぶ",
    "says", "speaks", "speaking", "whispers", "whispering", "sings",
    "singing", "mutters", "muttering", "shouts", "talks",
)
_WARN_VOICE_NOUNS = (
    "会話の声", "話し声", "語りかけ",
)

_D_TAG = _re.compile(r"<d>\[Japanese\](.*?)</d>", _re.S)

# Phase 1 item 1: the graph has no negative conditioning (BasicGuider,
# positive only). Any "avoid X" enumeration is noise the model cannot act
# on, and must never re-enter the final prompt via any path (deterministic
# OR VLM/story). This guard strips it even if something upstream re-adds it.
_AVOID_LINE_RE = _re.compile(
    r"^\s*(avoid depicting the following:|avoid:|do not depict)",
    _re.IGNORECASE)

# Phase 1 item 2: the canonical speaker-binding line
# ("<Subject N> (Sn[, Sm...]) ... and says,") is compiler scaffolding, not
# Director prose. It must survive speech-verb neutralization untouched (the
# literal word "says," must never become "smiles,") and must not trigger the
# heuristic speech-verb / quote WARNING scan below.
_SPEAKER_LINE_RE = _re.compile(
    r"^<Subject \d+> \(S\d+(?:\s*,\s*S\d+)*\)[^\n]*\bsays,\s*$")


def _is_protected_speaker_line(line: str) -> bool:
    return bool(_SPEAKER_LINE_RE.match(line.strip()))


def _strip_protected_lines(text: str) -> str:
    return "\n".join(line for line in str(text or "").split("\n")
                     if not _is_protected_speaker_line(line))


def dialogue_lines(dialogue_value: object) -> list[str]:
    """Ordered dialogue lines, speaker prefixes stripped (mirrors _shots)."""
    from . import director_lexicon as lexicon_mod
    pairs = lexicon_mod._dialogue_with_speakers(
        str(dialogue_value or ""), allow_s2=True)
    return [line for _speaker, line in pairs if line]


def apply_speech_policy(prompt: str) -> tuple[str, list[str]]:
    """Sanitize prose only. Dialogue bytes are never rewritten."""
    from . import director_lexicon as lexicon_mod
    conversions: list[str] = []
    parts = _re.split(r"(<d>\[Japanese\].*?</d>)", str(prompt or ""), flags=_re.S)
    for index in range(0, len(parts), 2):
        text = parts[index]
        kept = []
        for line in text.split("\n"):
            if _AVOID_LINE_RE.match(line.strip()):
                conversions.append("avoid_block_stripped")
                continue
            kept.append(line)
        text = "\n".join(kept)
        for replacement, sources in _AMBIENCE_MAP:
            spans = text.split(replacement)
            for pos in range(len(spans)):
                for source in sources:
                    if source and source in spans[pos]:
                        spans[pos] = spans[pos].replace(source, replacement)
                        conversions.append(source)
            text = replacement.join(spans)
        lines = []
        for line in text.split("\n"):
            after = line if _is_protected_speaker_line(line) else lexicon_mod.neutralize_speech_verbs_en(line)
            if after != line:
                conversions.append("verb_neutralized")
            lines.append(after)
        text, removed = lexicon_mod.strip_cjk_outside_d("\n".join(lines))
        if removed:
            conversions.append(f"cjk_stripped:{removed}")
        parts[index] = text
    return "".join(parts), sorted(set(conversions))


def _strip_d_tags(prompt: str) -> str:
    return _D_TAG.sub(" ", str(prompt or ""))


def validate_final(prompt: str, *, dialogue: list[str],
                   allow_s2: bool = False) -> dict:
    """FINAL validation. Returns {"hard": [...], "warnings": [...]}.

    hard non-empty = refuse generation. warnings = heuristic observations.
    """
    hard: list[str] = []
    warnings: list[str] = []
    text = str(prompt or "")
    d_contents = [m.group(1).strip() for m in _D_TAG.finditer(text)]
    markers = _re.findall(r"<\s*/?\s*d\b[^>]*(?:>|$)", text, flags=_re.I)
    if len(markers) != 2 * len(d_contents) or any(
            mark not in ("<d>", "</d>") for mark in markers):
        hard.append("壊れた発話タグ、または日本語以外の発話タグがあります。")
    expected = [str(line).strip() for line in (dialogue or [])
                if str(line).strip()]
    if any("<" in line or ">" in line for line in expected):
        hard.append("台詞に制御タグを含めることはできません。")
    if len(d_contents) != len(expected):
        hard.append(
            f"台詞行数と発話タグ数が不一致（台詞{len(expected)}行／"
            f"タグ{len(d_contents)}個）。生成を停止しました。")
    else:
        for pos, (got, want) in enumerate(zip(d_contents, expected)):
            if got != want:
                hard.append(
                    f"発話タグ{pos + 1}が指定台詞と不一致。生成を停止しました。")
                break
    if "<Subject 2>" in text and not allow_s2:
        hard.append("二人目が明示されていないのに<Subject 2>が存在。"
                    "生成を停止しました。")
    if "(S2)" in text and not allow_s2:
        hard.append("二人目が明示されていないのに(S2)の台詞が存在。"
                    "生成を停止しました。")
    prose = _strip_d_tags(text)
    # HARD: quoted Japanese outside <d> is speakable text, full stop.
    for mark_open, mark_close in (("「", "」"), ('"', '"'), ("'", "'")):
        for quoted in _re.findall(
                _re.escape(mark_open) + r"(.*?)" + _re.escape(mark_close),
                prose, flags=_re.S):
            if _re.search(
                    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
                    quoted):
                hard.append("発話タグ外に日本語の引用文が存在。"
                            "生成を停止しました。")
                break
        if hard and hard[-1].startswith("発話タグ外に日本語の引用文"):
            break
    # HARD: the dialogue body restated outside <d>.
    for line in expected:
        if line and line in prose:
            hard.append("指定台詞の本文が発話タグ外に再掲されている。"
                        "生成を停止しました。")
            break
    # Heuristic scans skip the canonical speaker-binding line: "says,"
    # (and any manner word in the delivery phrase) there is compiler
    # scaffolding, never an unspecified/hallucinated utterance.
    warn_prose = _strip_protected_lines(prose)
    if _re.search(
            "[぀-ヿ㐀-䶿一-鿿豈-﫿]",
            warn_prose):
        warnings.append(
            "発話タグ外に日本語proseが残存しています。幻覚発話の種に"
            "なる可能性があります。")
    low = warn_prose.lower()
    for verb in _WARN_SPEECH_VERBS:
        if verb in warn_prose or verb in low:
            warnings.append(
                f"発話タグ外に話す描写らしき表現（{verb}）があります。"
                "指定外の発話になる可能性があります。")
            break
    for noun in _WARN_VOICE_NOUNS:
        if noun in warn_prose:
            warnings.append(
                f"環境音に人声らしき表現（{noun}）が残っています。"
                "非言語化の確認を推奨します。")
            break
    return {"hard": hard, "warnings": warnings}


def final_gate(prompt: str, *, dialogue: list[str],
               allow_s2: bool = False) -> tuple[str, dict]:
    """Policy -> validate -> re-validate. Returns (prompt, report).

    report = {"hard", "warnings", "conversions"}. hard non-empty means the
    caller must refuse generation before any GPU work.
    """
    cleaned, conversions = apply_speech_policy(prompt)
    report = validate_final(cleaned, dialogue=dialogue, allow_s2=allow_s2)
    rechecked = validate_final(cleaned, dialogue=dialogue, allow_s2=allow_s2)
    report = {"hard": rechecked["hard"],
              "warnings": sorted(set(report["warnings"]) |
                                 set(rechecked["warnings"])),
              "conversions": conversions}
    return cleaned, report


def audit_prompt(prompt: str) -> dict:
    """Evidence summary for the Final Prompt Dump (no secrets involved)."""
    import re as _audit_re
    text = str(prompt or "")
    d_contents = [m.group(1).strip()
                  for m in _D_TAG.finditer(text)]
    prose = _strip_d_tags(text)
    cjk = _audit_re.findall(
        "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", prose)
    low = prose.lower()
    verbs = sorted({v for v in
                    ("says", "speaks", "speaking", "talks", "talking",
                     "whispers", "whispering", "asks", "replies", "tells",
                     "shouts", "sings")
                    if v in low})
    quotes = sorted({q for q in ("「", "」", '"', '"')
                     if q in prose})
    bg_voice = sorted({b for b in
                       ("people talking", "vendors shouting",
                        "human voices", "calling voices", "conversations",
                        "crowd speech", "人の声", "呼び込み")
                       if b in prose or b in low})
    return {"d_count": len(d_contents), "d_contents": d_contents,
            "jp_outside_d_runs": len(cjk),
            "jp_outside_d_sample": cjk[:5],
            "speech_verbs_outside_d": verbs,
            "quotes_outside_d": quotes,
            "background_voice_hits": bg_voice,
            "has_subject2": "<Subject 2>" in text,
            "has_s2": "(S2)" in text}
