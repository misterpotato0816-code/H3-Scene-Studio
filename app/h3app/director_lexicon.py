# -*- coding: utf-8 -*-
"""Deterministic Director Spec -> six-section H3 prompt compiler.

No AI/VLM re-interpretation: the Final Spec plus the Character snapshot are
rendered into the exact six-section format the MiniMax H3 node consumes.
Structure is fixed; only listed JP terms are translated via _LEXICON, and
anything unlisted passes through inline (Qwen reads Japanese fragments far
better than dropped content).

Absolute rules encoded here (see OUTFIT/PROMPT PATH AUDIT):
- Subject 1 = the selected Character. Never invent Subject 2.
- Director outfit beats Character outfit_ref, always. When it does, the old
  outfit/accessories/materials/color_keys never reach the prompt.
- The "changing outfit" ban is dropped exactly in that case, nothing else.
"""
from __future__ import annotations

import re

SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_LEXICON: tuple[tuple[str, str], ...] = (
    # outfits (longest phrases first at lookup time via sorted())
    ("白いドレス", "white dress"),
    ("浴衣姿", "wearing a yukata"),
    ("浴衣", "yukata"),
    ("着物", "kimono"),
    ("制服", "school uniform"),
    ("水着", "swimsuit"),
    ("コート", "coat"),
    ("シャツ", "shirt"),
    ("スカート", "skirt"),
    ("ズボン", "trousers"),
    ("ドレス", "dress"),
    ("甚平", "jinbei"),
    ("法被", "happi coat"),
    ("下駄", "geta sandals"),
    # places / setting
    ("夏祭り", "summer festival"),
    ("屋台", "food stall"),
    ("花火大会", "fireworks festival"),
    ("花火", "fireworks"),
    ("神社", "shrine"),
    ("海辺", "seaside"),
    ("ビーチ", "beach"),
    ("カフェ", "cafe"),
    ("プールサイド", "poolside"),
    ("部屋", "room"),
    ("街", "city street"),
    ("夜景", "night city view"),
    ("夜", "night"),
    ("夕方", "evening"),
    ("朝", "morning"),
    ("昼", "daytime"),
    # actions
    ("振り返る", "looks back over her shoulder"),
    ("振り向く", "turns around"),
    ("歩く", "walks"),
    ("走る", "runs"),
    ("笑う", "smiles"),
    ("微笑む", "smiles softly"),
    ("話す", "speaks"),
    ("語りかける", "talks"),
    ("見つめる", "gazes at"),
    ("眺める", "gazes at"),
    ("見上げる", "looks up at"),
    ("座る", "sits"),
    ("立つ", "stands"),
    ("手を振る", "waves her hand"),
    ("手を伸ばす", "reaches out her hand"),
    ("近づく", "moves closer"),
    ("寄る", "moves closer"),
    ("離れる", "steps back"),
    ("踊る", "dances"),
    ("食べる", "eats"),
    ("飲む", "drinks"),
    ("買う", "buys"),
    ("覗く", "peers at"),
    ("指さす", "points at"),
    ("うなずく", "nods"),
    ("首をかしげる", "tilts her head"),
    ("お辞儀", "bows"),
    # camera
    ("手持ち", "handheld"),
    ("撮影者", "camera operator"),
    ("カメラマン", "camera operator"),
    ("スマートフォン", "smartphone"),
    ("スマホ", "smartphone"),
    ("自撮り", "selfie-style"),
    ("固定カメラ", "static camera"),
    ("固定", "static"),
    ("追従", "tracking"),
    ("追いかける", "following"),
    ("パン", "pan"),
    ("ズームイン", "zoom in"),
    ("ズームアウト", "zoom out"),
    ("ズーム", "zoom"),
    ("アップ", "close-up"),
    ("寄り", "push in"),
    ("引き", "pull back"),
    ("俯瞰", "overhead shot"),
    ("ローアングル", "low angle"),
    ("ゆっくり", "slowly"),
    ("自然に", "naturally"),
    ("激しく", "strongly"),
    ("ゆるやか", "gently"),
    # mood / emotion / relations
    ("楽しい", "joyful"),
    ("嬉しい", "happy"),
    ("照れた", "shy"),
    ("照れ", "shy"),
    ("切ない", "wistful"),
    ("寂しい", "lonely"),
    ("驚いた", "surprised"),
    ("自然な", "natural"),
    ("恋人", "lover"),
    ("ロマンチック", "romantic"),
    ("シネマティック", "cinematic"),
    ("映画風", "cinematic film-like"),
    ("日常感", "everyday casualness"),
    ("Vlog風", "vlog-style"),
    ("Vlog", "vlog"),
    # sound
    ("祭囃子", "festival music"),
    ("歓声", "cheering crowds"),
    ("拍手", "applause"),
    ("波の音", "sound of waves"),
    ("セミ", "cicadas"),
    ("雨音", "sound of rain"),
    ("ざわめき", "crowd murmur"),
    ("静寂", "silence"),
    # misc direction words
    ("正面", "front-facing"),
    ("カメラ目線", "looking into the camera"),
    ("横顔", "profile view"),
    ("全身", "full body"),
    ("上半身", "upper body"),
    ("逆光", "backlit"),
    ("提灯", "paper lanterns"),
    ("りんご飴", "candy apple"),
    ("たこ焼き", "takoyaki"),
    ("かき氷", "shaved ice"),
    # C1: the compiler's own JP speech-verb-neutralization substitutes
    # (_VERB_JP targets below) must always be fully translatable - they are
    # deterministic compiler output, never user text, so leaving any of
    # them untranslated would make UntranslatedPromptError unreachable on
    # the passing path for any spec that uses a speech verb at all.
    ("手を振りながら微笑む", "waves her hand while smiling"),
    ("優しく見つめて微笑む", "gazes gently and smiles"),
    ("笑顔でうなずき合う", "smiles and nods together"),
    ("少し身を寄せて柔らかく微笑む", "leans in slightly and smiles softly"),
    ("笑顔で向き合う", "turns to face them with a smile"),
    ("表情豊かにうなずく", "nods with an expressive face"),
    ("穏やかに微笑む", "smiles calmly"),
    ("楽しそうに微笑む", "smiles happily"),
    ("笑顔のやり取り", "an exchange of smiles"),
)

_SORTED_LEXICON = tuple(sorted(_LEXICON, key=lambda kv: -len(kv[0])))


# ---- speech isolation (AUDIO HOTFIX) ------------------------------------
# Only <d>[Japanese] ...</d> may carry speakable language. Everything below
# converts or removes speech vectors OUTSIDE <d>; dialogue itself is never
# summarized or rewritten by the compiler.

# JP speech verbs -> visual acting (longest first at lookup time).
_VERB_JP: tuple[tuple[str, str], ...] = (
    ("呼びかける", "手を振りながら微笑む"),
    ("語りかける", "優しく見つめて微笑む"),
    ("会話する", "笑顔でうなずき合う"),
    ("ささやく", "少し身を寄せて柔らかく微笑む"),
    ("囁く", "少し身を寄せて柔らかく微笑む"),
    ("話しかける", "笑顔で向き合う"),
    ("話す", "表情豊かにうなずく"),
    ("語る", "穏やかに微笑む"),
    ("歌う", "楽しそうに微笑む"),
    # NOTE: no bare「言う」: という (explanatory "called") is everywhere and
    # must not become a smile.
    ("会話", "笑顔のやり取り"),
)

# EN speech verbs -> visual acting, specific patterns before generics.
_VERB_EN: tuple[tuple[str, str], ...] = (
    ("speaks excitedly to the camera",
     "faces the camera with an excited smile and expressive gestures"),
    ("speaks to the camera", "faces the camera with a natural smile"),
    ("talks to the camera", "faces the camera with a natural smile"),
    ("says to the camera", "smiles toward the camera"),
    ("whispers softly",
     "leans slightly closer with a gentle expression"),
    ("whispers", "leans slightly closer with a gentle expression"),
    ("calls out", "waves with a bright smile"),
    ("shouts", "opens her arms with a lively smile"),
    ("sings", "smiles with a joyful expression"),
    ("replies", "nods with a warm smile"),
    ("asks", "tilts her head with a curious smile"),
    ("tells", "gestures expressively with a smile"),
    ("speaks", "shows natural facial reactions"),
    ("talks", "shows natural facial reactions"),
    ("says", "smiles"),
)


def neutralize_speech_verbs_jp(text: str) -> str:
    out = str(text or "")
    for jp, acting in sorted(_VERB_JP, key=lambda kv: -len(kv[0])):
        if jp in out:
            out = out.replace(jp, acting)
    return out


def neutralize_speech_verbs_en(text: str) -> str:
    import re as _re
    out = str(text or "")
    for pattern, acting in _VERB_EN:
        out = _re.sub(pattern, acting, out, flags=_re.IGNORECASE)
    return out


_CJK_RUN = ("["
            "\u3000-\u303f"      # CJK punctuation (。、 etc. - not speakable)
            "\u3040-\u30ff"      # hiragana + katakana
            "\u3400-\u4dbf"      # CJK ext A
            "\u4e00-\u9fff"      # CJK unified
            "\uf900-\ufaff"      # CJK compat
            "]+"
            )


def strip_cjk_outside_d(prompt: str) -> tuple[str, int]:
    """Remove CJK runs everywhere except inside <d>...</d>. Idempotent.

    Returns (cleaned, removed_count). Dialogue Japanese is protected; scene
    prose must be English (proper-noun exceptions are accepted as lost).
    """
    import re as _re
    parts = _re.split(r"(<d>\[Japanese\].*?</d>)", str(prompt or ""),
                      flags=_re.S)
    removed = 0
    for index in range(0, len(parts), 2):
        cleaned, count = _re.subn(_CJK_RUN, " ", parts[index])
        removed += count
        parts[index] = cleaned
    out = "".join(parts)
    out = _re.sub(r"[ \t]{2,}", " ", out)
    # C4: removing a CJK run can leave an orphan space directly before
    # trailing punctuation ("front-facing ."). Collapse it here, the same
    # last-mile spot that already normalizes whitespace.
    out = _re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    out = _re.sub(r"\n{3,}", "\n\n", out)
    return out, removed


def _collapse_adjacent_duplicates(text: str) -> str:
    """Collapse an EXACT adjacent duplicate word-run, once.

    "food stall food stall" -> "food stall". Only immediate, verbatim
    repeats are touched (no semantic matching); runs of up to 6 words are
    checked, longest first, so a duplicated multi-word phrase collapses as
    a whole rather than leaving a partial echo behind.
    """
    words = str(text or "").split(" ")
    out: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        matched = False
        max_len = min(6, (n - i) // 2)
        for length in range(max_len, 0, -1):
            if words[i:i + length] == words[i + length:i + 2 * length]:
                out.extend(words[i:i + length])
                i += 2 * length
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _join_sentences(fragments: list[str]) -> str:
    """Join independent item texts as separate terminated sentences.

    D2: concatenating acting/movement/camera_work (or ambient/voice/"No
    narration") with a bare space produces a run-on ("... approaches the
    food stall, then turns back ... her films handheld ... shot." with no
    boundary between independent clauses). Each fragment has its trailing
    '.'/','/whitespace stripped, fragments are joined with '. ', and the
    whole run is terminated with a single '.'. Returns "" for no fragments.
    """
    parts: list[str] = []
    for frag in fragments:
        frag = str(frag or "").strip()
        frag = frag.rstrip(" .,")
        if frag:
            parts.append(frag)
    if not parts:
        return ""
    return ". ".join(parts) + "."


def _sentences(text: str) -> list[str]:
    import re as _re
    return [s.strip() for s in _re.split(r"(?<=[。．.\n])", str(text or ""))
            if s.strip()]


def remove_dialogue_echoes(prose: str, dialogue_lines: list[str]) -> str:
    """Drop prose sentences restating the dialogue (meaning -> gesture only).

    A sentence is dropped when it shares a >=6-char verbatim run (JP) or a
    >=3-word run (EN) with any dialogue line. Quoted spans are removed first.
    """
    import re as _re
    lines = [str(line).strip() for line in (dialogue_lines or [])
             if str(line).strip()]
    if not lines:
        return str(prose or "")
    jp_keys: list[str] = []
    en_keys: list[str] = []
    for line in lines:
        bare = _re.sub(r"[「」\"\"''\(\)（）]", " ", line)
        for chunk in _re.split(r"\s+", bare):
            chunk = chunk.strip(" 、。，．・!！?？")
            if len(chunk) >= 6:
                jp_keys.append(chunk)
        words = _re.findall(r"[A-Za-z']+", bare.lower())
        for start in range(len(words) - 2):
            en_keys.append(" ".join(words[start:start + 3]))
    kept: list[str] = []
    for sentence in _sentences(prose):
        bare = _re.sub(r"[「」\"\"''\(\)（）]", " ", sentence)
        if any(key in bare for key in jp_keys):
            continue
        low = bare.lower()
        if any(key in low for key in en_keys):
            continue
        kept.append(sentence)
    return " ".join(kept)


def clean_voice_item(text: str) -> str:
    """Voice = timbre/delivery only. Strip quotes and says-X content clauses.

    Bare manner verbs collapse to delivery nouns (slow speech -> slow
    delivery); content clauses are removed, never kept. Unlike earlier
    versions, this NEVER deletes CJK: a Japanese voice direction with no
    quoted dialogue and no "says/speaks ..." clause must survive so it can
    still be translated (to_english / item "en") instead of vanishing.
    """
    import re as _re
    out = str(text or "")
    out = _re.sub(r"[「」].*?[「」]", " ", out)
    out = _re.sub(r"\".*?\"", " ", out)
    out = _re.sub(r"(says|speaks|talks|tells)[^,.;。]*", " ", out,
                  flags=_re.IGNORECASE)
    out = _re.sub(r"\b(speaks?|talking|talks?)\b", "delivery", out,
                  flags=_re.IGNORECASE)
    return " ".join(out.split())


def to_english(text: str) -> str:
    """Deterministic JP->EN term replacement. Unknown spans pass through.

    Replacements are padded with spaces before collapsing whitespace, so two
    adjacent JP terms translate to two separate English words instead of a
    single glued token (was: "たこ焼き屋台" -> "takoyakifood stall").
    """
    out = str(text or "")
    for jp, en in _SORTED_LEXICON:
        if jp in out:
            out = out.replace(jp, " " + en + " ")
    return " ".join(out.split())


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines()
            if line.strip()]


def _drop_negated_clauses(text: str) -> str:
    """Remove "don't wear X" style clauses before translation.

    Without this, "白いドレスは着ない" would translate into a positive
    "white dress" token. Negation markers are clothing-specific so normal
    prose ("問題なし" etc.) is untouched.
    """
    import re as _re
    parts = _re.split(r"[。、\n]", str(text or ""))
    keep = [p for p in parts
            if p.strip() and not _re.search(
                r"(着ない|着ません|脱いで|脱ぐ|やめて|禁止|着用しない)", p)]
    return "。".join(keep)


def _dialogue_with_speakers(dialogue: str,
                            allow_s2: bool = True) -> list[tuple[str, str]]:
    """Split dialogue lines; A:/B: (half/full width) prefixes become S1/S2.

    When no second on-screen subject exists, B: collapses to S1: inventing a
    second speaker voice is exactly how a phantom person leaks into the video.
    """
    import re as _re
    out: list[tuple[str, str]] = []
    for line in _split_lines(dialogue):
        match = _re.match(r"\s*([ABAB])\s*[:：]\s*(.*)$", line)
        if match and match.group(2).strip():
            letter = match.group(1)
            speaker = "S1" if letter == "A" else ("S2" if allow_s2 else "S1")
            out.append((speaker, match.group(2).strip()))
        else:
            out.append(("S1", line))
    return out


_POV_PHRASES = {
    "lover_pov": "filmed from the lover's point of view on a handheld "
                 "smartphone, the operator never visible on screen",
    "selfie": "selfie-style framing held at arm's length",
    "observer": "third-person observer framing",
    "handheld_third_person": "handheld third-person framing",
}


def _offcamera_nouns(cast: dict | None) -> list[str]:
    """Person nouns that must not appear in Shot/action prose.

    When the operator stays off-camera, their name (and the generic lover
    word it maps to) is dropped from action/label prose so H3 cannot cast
    them as a visible second person. The Camera phrase keeps its own fixed
    wording; only action prose is filtered.
    """
    if not isinstance(cast, dict):
        return []
    if str(cast.get("camera_operator_visibility") or "").strip() != \
            "off_camera":
        return []
    nouns: list[str] = []
    operator = str(cast.get("camera_operator") or "").strip()
    if operator:
        nouns.append(operator)
        for token in operator.replace("である", " ").split():
            token = token.strip(" 、。，．・")
            if len(token) >= 2:
                nouns.append(token)
    nouns.extend(["lover", "lovers", "恋人",
                    "camera operator", "撮影者", "カメラマン"])
    return sorted(set(nouns), key=len, reverse=True)


def _clean_after_noun_drop(text: str) -> str:
    """D3: deterministic wreckage cleanup after _drop_offcamera_nouns.

    Removing a person noun mid-phrase (e.g. "lover" out of "lover's") can
    leave a dangling possessive ("a 's point-of-view shot"), a stranded
    article immediately before punctuation/another article, or doubled
    spaces and orphaned commas. This only tidies grammar; it never restores
    the dropped noun.
    """
    import re as _re
    out = str(text or "")
    # Dangling possessive: "a 's" / "the 's" -> drop the whole fragment.
    out = _re.sub(r"\b(a|an|the)\s+'s\b", " ", out, flags=_re.IGNORECASE)
    # A bare leading/orphaned "'s" with no owner left before it.
    out = _re.sub(r"(^|\s)'s\b", r"\1", out)
    # An article immediately followed by punctuation (nothing left to
    # modify) or by another article.
    out = _re.sub(r"\b(a|an|the)\s+(?=[.,;:!?])", " ", out, flags=_re.IGNORECASE)
    out = _re.sub(r"\b(a|an|the)\s+(a|an|the)\b", r"\2", out, flags=_re.IGNORECASE)
    # Doubled spaces and orphaned commas.
    out = _re.sub(r"\s{2,}", " ", out)
    out = _re.sub(r"\s*,\s*,\s*", ", ", out)
    out = _re.sub(r"\s+,", ",", out)
    out = _re.sub(r"^\s*,\s*", "", out)
    return out.strip()


def _drop_offcamera_nouns(text: str, nouns: list[str]) -> str:
    import re as _re
    out = str(text or "")
    for noun in nouns:
        if not noun:
            continue
        if _re.search(r"[A-Za-z]", noun):
            out = _re.sub(r"\b" + _re.escape(noun) + r"\b", " ", out,
                          flags=_re.IGNORECASE)
        elif noun in out:
            out = out.replace(noun, " ")
    out = " ".join(out.split())
    return _clean_after_noun_drop(out)


def _camera_phrase(cast: dict | None) -> str:
    """Off-camera people become a shooting-style phrase, never a person."""
    if not isinstance(cast, dict):
        return ""
    bits: list[str] = []
    pov = str(cast.get("camera_pov") or "").strip()
    if pov in _POV_PHRASES:
        bits.append(_POV_PHRASES[pov])
    elif str(cast.get("camera_device") or "").strip():
        bits.append("shot on a %s" % to_english(
            str(cast.get("camera_device") or "").strip()))
    operator = str(cast.get("camera_operator") or "").strip()
    vis = str(cast.get("camera_operator_visibility") or "").strip()
    if operator and vis == "off_camera":
        bits.append("the %s stays off-camera and is never shown" %
                    to_english(operator))
    return ". ".join(b for b in bits if b)


def _mmss(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes, rem = divmod(total_ms, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def _has_cjk(text: str) -> bool:
    return bool(re.search(_CJK_RUN, str(text or "")))


class UntranslatedPromptError(ValueError):
    """Raised when Japanese the 111-entry lexicon could not translate would
    otherwise be silently deleted by strip_cjk_outside_d.

    Carries the offending CJK runs (``.runs``) so callers can refuse
    generation and name the untranslated fragments instead of shipping a
    prompt with them quietly stripped out.
    """

    def __init__(self, runs: list[str]):
        self.runs = list(runs or [])
        super().__init__(
            "untranslated Japanese outside <d>: " +
            ", ".join(sorted(set(self.runs))[:5]))


def _item_manual_en(slot: dict | None) -> str:
    """The item's own canonical English form, when usable (Option A).

    Usable = present, non-empty, and containing no CJK (a stale/garbled
    translation left in Japanese is treated as absent, never surfaced).
    """
    if not isinstance(slot, dict):
        return ""
    candidate = slot.get("en")
    if isinstance(candidate, str) and candidate.strip() and \
            not _has_cjk(candidate):
        return candidate.strip()
    return ""


def _identity_phrase(identity: dict) -> str:
    """Legacy comma-joined identity phrase (kept for callers that want it)."""
    parts = [str(identity.get(k, "")).strip()
             for k in ("face", "hair", "build", "skin")]
    return ", ".join(p for p in parts if p)


def _subject_definitions_line(identity: dict, outfit_en: str,
                              outfit_changed: bool,
                              outfit_ref: dict | None) -> str:
    """Labelled <Subject 1> canon line (restores the Character canon).

    Reuses canon.py's own field list/heading formatter instead of a second
    hand-maintained heading map. Degrades to the historical "the main
    woman" phrasing when nothing at all is available.
    """
    from . import canon as canon_mod
    identity = identity or {}
    outfit_ref = outfit_ref or {}
    fields: list[tuple[str, str]] = []
    for key in ("face", "hair", "build", "skin"):
        value = str(identity.get(key, "") or "").strip()
        if value:
            fields.append((key, value))
    if outfit_en:
        fields.append(("outfit", outfit_en))
    if not outfit_changed:
        for key in ("accessories", "materials", "color_keys"):
            value = str(outfit_ref.get(key, "") or "").strip()
            if value:
                fields.append((key, value))
    if not fields:
        return "<Subject 1> the main woman" + \
            ((", " + outfit_en) if outfit_en else "")
    body = " ".join(f"{canon_mod.heading(key)}: {value}."
                    for key, value in fields)
    return ("<Subject 1> is the main character, identical in every shot. " +
            body)


def compile_direct_prompt(*, items: dict, identity: dict,
                          outfit_ref: dict, refs: list[str],
                          frames: int, fps: int = 24,
                          timeline: list[dict] | None = None,
                          start_state: dict | None = None,
                          end_state: dict | None = None,
                          carry_over: dict | None = None,
                          continuation: bool = False,
                          refs_available: dict | None = None,
                          cast: dict | None = None) -> str:
    """Render the six sections. Raises ValueError when self-check fails.

    `refs_available` (preferred over the legacy `continuation` bool) is
    {"pictures": int (optional, defaults to len(refs)), "videos": int,
     "audios": int, "audio_kind": "video_soundtrack"|"voice_master"}. It
     drives which <Picture N>/<Video N>/<Audio N> reference labels are
     emitted; a label is never emitted for a reference that will not
     actually be connected in the graph (see graphs.assert_reference_labels).
    `continuation` (legacy): True == today's LONG behaviour (videos=1,
     audios=1, audio_kind="video_soundtrack"); ignored when refs_available
     is given.
    """
    duration = max(1.0, frames / float(fps or 24))
    from . import director_spec as _spec_mod
    cast = _spec_mod.normalize_cast(cast)
    allow_s2 = _spec_mod.allow_second_subject(cast)

    if refs_available is not None:
        n_videos = int(refs_available.get("videos", 0) or 0)
        n_audios = int(refs_available.get("audios", 0) or 0)
        audio_kind = str(refs_available.get("audio_kind") or
                         ("video_soundtrack" if n_videos else "voice_master"))
    else:
        n_videos = 1 if continuation else 0
        n_audios = 1 if continuation else 0
        audio_kind = "video_soundtrack"
    n_pictures = len(refs or [])

    def _item(name: str) -> str:
        slot = items.get(name)
        return str(slot.get("value", "")).strip() \
            if isinstance(slot, dict) else ""

    def _item_en(name: str) -> str:
        """Item text normalized for H3: speech verbs -> acting (never <d>).

        Prefers the item's own canonical English ("en") when usable (Phase
        1, item 5); otherwise falls back to the deterministic lexicon.
        Dialogue is never routed through this function.
        """
        raw = _item(name)
        manual_en = _item_manual_en(items.get(name))
        if name == "voice":
            base = manual_en if manual_en else to_english(raw)
            return clean_voice_item(base)
        if manual_en:
            return neutralize_speech_verbs_en(manual_en)
        return neutralize_speech_verbs_en(to_english(
            neutralize_speech_verbs_jp(raw)))

    # Dialogue lines (verbatim, never rewritten) double as the echo filter:
    # prose restating them is reduced to gesture elsewhere.
    _echo_lines = [line for _sp, line in
                   _dialogue_with_speakers(_item("dialogue"),
                                           allow_s2=True) if line]
    _drop_nouns = _offcamera_nouns(cast)

    _outfit_raw = _item("outfit")
    _outfit_manual_en = _item_manual_en(items.get("outfit"))
    if _outfit_raw:
        if _outfit_manual_en:
            outfit_en = neutralize_speech_verbs_en(_outfit_manual_en)
        else:
            director_outfit = neutralize_speech_verbs_jp(
                _drop_negated_clauses(_outfit_raw))
            outfit_en = neutralize_speech_verbs_en(to_english(director_outfit))
        outfit_changed = True
    else:
        outfit_en = str((outfit_ref or {}).get("outfit") or "").strip()
        outfit_changed = False
    identity_en = {k: str((identity or {}).get(k, "")).strip()
                   for k in ("face", "hair", "build", "skin")}
    subject_line = _subject_definitions_line(
        identity_en, outfit_en, outfit_changed, outfit_ref)

    sections: dict[str, list[str]] = {name: [] for name in SECTIONS}
    sections["subject_definitions"].append(subject_line)
    for index in range(n_pictures):
        sections["subject_definitions"].append(
            f"<Picture {index + 1}> reference image {index + 1} provides "
            f"the character appearance anchor.")
    for index in range(n_videos):
        sections["subject_definitions"].append(
            f"<Video {index + 1}> is the final segment of the preceding "
            "clip and defines the starting state, camera position, "
            "subject pose, framing and lighting that the target video "
            "continues from.")
    for index in range(n_audios):
        if audio_kind == "voice_master":
            if _echo_lines:
                sections["subject_definitions"].append(
                    f"<Audio {index + 1}> is a fixed voice reference "
                    "defining the speaker's timbre for this clip; it is not "
                    "the soundtrack of any video.")
            else:
                sections["subject_definitions"].append(
                    f"<Audio {index + 1}> is a fixed voice reference retained "
                    "only for the required reference binding; it does not "
                    "define or request timbre, speech, or vocal output for "
                    "this silent clip and is not the soundtrack of any video.")
        else:
            sections["subject_definitions"].append(
                f"<Audio {index + 1}> is the synchronized audio track of "
                f"<Video {index + 1}>, used for ambient and vocal "
                "continuity.")

    _summary_core = (_item_en("mood") or _item_en("video_type") or
                     "a short vertical video")
    _summary_core = _drop_offcamera_nouns(_summary_core, _drop_nouns)
    if n_videos:
        sections["summary"].append(
            "[video continuation + reference generation + audio reference] "
            "The target video continues from <Video 1>. " +
            _summary_core + ".")
    else:
        sections["summary"].append(
            "[reference generation] " + _summary_core + ".")

    if outfit_changed:
        sections["retention_analysis"].append(
            "<Subject 1> (appears in [Shot 1]): attribute_transfer - outfit "
            f"changed to {outfit_en} as directed; face, hair and build "
            "fully_preserved.")
    else:
        sections["retention_analysis"].append(
            "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, "
            "build and outfit.")
    if n_videos:
        sections["retention_analysis"].append(
            "<Video 1> (continuation source): fully_preserved - the subject "
            "pose, framing and lighting at the end of <Video 1> are carried "
            "into the first frame of [Shot 1].")
    if n_audios:
        if audio_kind == "voice_master":
            if _echo_lines:
                sections["retention_analysis"].append(
                    "<Audio 1>: reference - the speaker's timbre is carried "
                    "forward without copying the signal.")
            else:
                sections["retention_analysis"].append(
                    "<Audio 1>: reference binding only for this silent clip "
                    "- no speech or vocal output is requested, and the "
                    "signal is not copied.")
        else:
            sections["retention_analysis"].append(
                "<Audio 1>: reference - ambient tone and speaker timbre "
                "continue without copying the signal.")

    style_bits = [_item_en("mood"), _item_en("video_type"),
                  _item_en("scenery") or _item_en("location")]
    style_bits = [_drop_offcamera_nouns(b, _drop_nouns) for b in style_bits]
    style_bits = [b for b in style_bits if b]
    opener = ("A vertical 9:16 short-form video" +
              (", " + ", ".join(style_bits) if style_bits else "") + ".")
    body: list[str] = [opener]
    camera_phrase = _camera_phrase(cast)
    if camera_phrase:
        body.append("Camera: " + camera_phrase + ".")
    if start_state:
        started = [f"{k}: {v}" for k, v in start_state.items()
                   if str(v or "").strip()]
        if started:
            body.append("<Subject 1> starts: " + "; ".join(started) + ".")
    if carry_over:
        carried = [str(v).strip() for v in carry_over.values()
                   if str(v or "").strip()]
        if carried:
            body.append("Ongoing: " + " / ".join(carried) + ".")
    voice = _item_en("voice")
    # D1: the speaker-binding line's delivery clause is NEVER derived from
    # the voice description. A free-form voice sentence ("a bright, playful
    # voice from a young woman, cheerful and lightly bubbly") cannot be
    # reduced to a grammatical clause deterministically (the old
    # voice_delivery_phrase() produced "speaks in a bright, playful from a
    # young woman, ... tone"). The verified gold prompt uses a fixed acting
    # clause here instead; the voice description's correct home is
    # overall_soundscape, where it still appears in full via `voice` above.
    delivery_phrase = "speaks naturally"
    shots = _shots(items, timeline, duration, allow_s2=allow_s2,
                   echo_lines=_echo_lines, drop_nouns=_drop_nouns,
                   delivery_phrase=delivery_phrase)
    body.extend(shots)
    if end_state:
        ended = [f"{k}: {v}" for k, v in end_state.items()
                 if str(v or "").strip()]
        if ended:
            body.append("Ending state: " + "; ".join(ended) + ".")
    # Positive performance instruction keyed to the actual speech contract.
    # An unconditional lip-sync sentence makes a zero-dialogue shot ask H3 to
    # invent "spoken dialogue" and was reproduced as audible gibberish with
    # continuous mouth movement.  Keep both branches affirmative because the
    # graph uses BasicGuider with positive conditioning only.
    if _echo_lines:
        body.append("Natural synchronized mouth movement matching the spoken "
                    "dialogue, with accurate lip timing and clear articulation.")
    else:
        silent_subjects = "<Subject 1> and <Subject 2>" if allow_s2 else \
            "<Subject 1>"
        silent_verb = "remain" if allow_s2 else "remains"
        if n_videos:
            body.append(
                f"{silent_subjects} {silent_verb} silent; after the "
                "preceding video's ending pose is carried into the first "
                "frame of [Shot 1], their lips settle gently closed and stay "
                "relaxed without speech-like movement.")
        else:
            body.append(
                f"{silent_subjects} {silent_verb} silent with relaxed, gently "
                "closed lips throughout, without speech-like mouth movement.")
    sections["detailed_description"] = body

    ambient = _item_en("ambient_audio")
    sounds: list[str] = []
    # C2: never emit an empty "Voice: " label. A voice item that reduces to
    # nothing (or to bare punctuation) after cleaning must be omitted
    # entirely, not rendered as a hollow "Voice: ." fragment.
    _voice_label = voice.strip(" .")
    # A timbre/delivery description has no role when there are no dialogue
    # tags. Keeping "Voice:" in a silent scene would reintroduce an implicit
    # request for vocal output even though the visual prompt forbids speech.
    if _echo_lines and _voice_label and re.search(r"\w", _voice_label):
        sounds.append(f"Voice: {_voice_label}.")
    if ambient:
        # D2: ambient text must be its own terminated sentence, otherwise it
        # runs straight into "No narration..." with no separator.
        _ambient_sentence = _join_sentences([ambient])
        if _ambient_sentence:
            sounds.append(_ambient_sentence)
    if not _echo_lines:
        sounds.append("No dialogue or speech-like vocalization.")
    sounds.append("No narration, no additional voices.")
    sections["overall_soundscape"].append(
        " ".join(sounds) if sounds else
        "Natural location ambience matching the scene. "
        "No narration, no additional voices.")
    sections["non_diegetic_music"].append("N/A")

    rendered: list[str] = []
    for name in SECTIONS:
        rendered.append(name)
        rendered.extend(sections[name])
        rendered.append("")
    prompt = "\n".join(rendered).strip() + "\n"
    # C1: detect untranslated Japanese OUTSIDE <d> BEFORE it is silently
    # deleted by strip_cjk_outside_d below. Refusing here (loudly) beats
    # shipping mangled English the lexicon could not translate.
    _pre_strip_runs = untranslated_runs(prompt)
    if _pre_strip_runs:
        raise UntranslatedPromptError(_pre_strip_runs)
    # Speech isolation last mile: no CJK prose may survive outside <d>.
    # (strip_cjk_outside_d stays as a defensive no-op safety net on this,
    # the passing, path - see its docstring.)
    prompt, _stripped = strip_cjk_outside_d(prompt)
    _self_check(prompt, items=items,
                banned_outfit_ref=(outfit_ref if outfit_changed else None),
                allow_s2=allow_s2)
    return prompt


# A canonical speaker-binding line ("<Subject 1> (S1) ... and says,"). This
# exact shape is protected from speech-verb neutralization and heuristic
# WARNING scans in director_speech.py: it is scaffolding, not Director prose,
# and "says," in it must never be rewritten into "smiles,".
SPEAKER_LINE_RE = re.compile(
    r"^<Subject \d+> \(S\d+(?:\s*,\s*S\d+)*\)[^\n]*\bsays,\s*$")


def _shots(items: dict, timeline: list[dict] | None,
           duration: float, allow_s2: bool = True,
           echo_lines: list[str] | None = None,
           drop_nouns: list[str] | None = None,
           delivery_phrase: str = "speaks naturally") -> list[str]:
    """Shot list with dialogue distributed in order across shots.

    Restores the official speaker binding: every shot line is bound to
    <Subject 1>, and every dialogue line is preceded by its own
    "<Subject N> (Sn) {delivery} and says," line before the <d> element,
    on its own line.

    Acting/movement/camera/label prose is de-echoed (dialogue restatements
    become gesture-only), speech verbs are neutralized, and off-camera
    person nouns are dropped before use.
    """
    dialogue_lines = _dialogue_with_speakers(
        str((items.get("dialogue") or {}).get("value", ""))
        if isinstance(items.get("dialogue"), dict) else "",
        allow_s2=allow_s2)
    echo_lines = list(echo_lines or [])
    drop_nouns = list(drop_nouns or [])

    def _action(name: str) -> str:
        slot = items.get(name)
        raw = str(slot.get("value", "")).strip() \
            if isinstance(slot, dict) else ""
        if not raw:
            return ""
        raw = _drop_offcamera_nouns(raw, drop_nouns)
        raw = remove_dialogue_echoes(raw, echo_lines)
        manual_en = _item_manual_en(slot)
        if manual_en:
            text = neutralize_speech_verbs_en(manual_en)
        else:
            text = to_english(neutralize_speech_verbs_jp(raw))
            text = neutralize_speech_verbs_en(text)
        text = _drop_offcamera_nouns(text, drop_nouns)
        return remove_dialogue_echoes(text, echo_lines)

    acting = _action("acting")
    movement = _action("movement")
    camera = _action("camera_work")
    action_bits = [b for b in [acting, movement, camera] if b]
    # D2: acting/movement/camera_work are independent item texts; joined
    # with a bare space they read as one ungrammatical run-on sentence.
    # Join them as separate, terminated sentences instead.
    action = _join_sentences(action_bits) if action_bits else \
        "performs naturally for the scene."
    # C4: acting/movement/camera_work often restate the same setting noun
    # (e.g. two items both mention the food stall); collapse an exact
    # adjacent duplicate before it reaches the shot line.
    action = _collapse_adjacent_duplicates(action)

    slots: list[tuple[float, float, str]] = []
    clean_timeline = [entry for entry in (timeline or [])
                      if isinstance(entry, dict)]
    if clean_timeline:
        for entry in clean_timeline[:8]:
            try:
                start = max(0.0, min(float(entry.get("t0", 0.0)), duration))
                end = max(start, min(float(entry.get("t1", duration)),
                                     duration))
            except (TypeError, ValueError):
                continue
            label = str(entry.get("en") or entry.get("label") or entry.get("action") or "")
            label = _drop_offcamera_nouns(label, drop_nouns)
            label = remove_dialogue_echoes(label, echo_lines)
            label = neutralize_speech_verbs_en(to_english(
                neutralize_speech_verbs_jp(label)))
            label = remove_dialogue_echoes(label, echo_lines)
            slots.append((start, end, label))
    if not slots:
        slots = [(0.0, duration, "")]
    shots: list[str] = []
    per_shot = max(1, (len(dialogue_lines) + len(slots) - 1) // len(slots))
    for index, (start, end, label) in enumerate(slots):
        head = "[Shot 1] " if index == 0 else \
            f"[Shot {index + 1}] At {_mmss(start)}, "
        act_bits = [b for b in [action, label] if b]
        act_text = _join_sentences(act_bits) if act_bits else \
            "performs naturally for the scene."
        act_text = _collapse_adjacent_duplicates(act_text)
        # D4 defence in depth: the Director's own `en` text is instructed
        # never to restate <Subject 1>, but if it slips through anyway, do
        # not double it up with the app's own prefix.
        if "<Subject 1>" in act_text:
            block_lines = [f"{head}{act_text}"]
        else:
            block_lines = [f"{head}<Subject 1> {act_text}"]
        chunk = dialogue_lines[index * per_shot:(index + 1) * per_shot]
        for speaker, line in chunk:
            subject_label = "<Subject 1>" if speaker == "S1" else \
                "<Subject 2>"
            block_lines.append(
                f"{subject_label} ({speaker}) {delivery_phrase} and says,")
            block_lines.append(f"<d>[Japanese] {line}</d>")
        shots.append("\n".join(block_lines))
    return shots


def _self_check(prompt: str, *, items: dict,
                banned_outfit_ref: dict | None,
                allow_s2: bool = True) -> None:
    """Structural guarantees. Raises ValueError (caller bug, never user)."""
    errors: list[str] = []
    positions = []
    for name in SECTIONS:
        if name == SECTIONS[0]:
            found = 0 if prompt.startswith(name + "\n") else -1
        else:
            marker = f"\n{name}\n"
            found = prompt.find(marker, positions[-1] if positions else 0)
        if found < 0:
            errors.append(f"missing section: {name}")
        else:
            positions.append(found)
    dialogue = items.get("dialogue") if isinstance(
        items.get("dialogue"), dict) else {}
    expected_d = len(_split_lines(str(dialogue.get("value", ""))))
    actual_d = prompt.count("<d>[Japanese]")
    if expected_d != actual_d:
        errors.append(f"d-tag count {actual_d} != dialogue lines {expected_d}")
    if "<Subject 2>" in prompt and not allow_s2:
        errors.append("phantom Subject 2 emitted")
    if "(S2)" in prompt and not allow_s2:
        errors.append("phantom S2 speaker emitted")
    lowered = prompt.lower()
    if banned_outfit_ref:
        for key in ("outfit", "accessories", "materials", "color_keys"):
            value = str(banned_outfit_ref.get(key) or "").strip().lower()
            if value and value in lowered:
                errors.append(f"overridden outfit_ref leaked: {key}")
    if errors:
        raise ValueError("direct prompt self-check failed: " +
                         "; ".join(errors))


# ---- pre-generation prompt validation (item 5: no silent deletion) -------

def untranslated_runs(prompt: str) -> list[str]:
    """CJK runs remaining OUTSIDE <d>...</d>. Detection, not deletion.

    `strip_cjk_outside_d` stays the final safety net, but this must be able
    to see what would be silently removed BEFORE that net runs, so callers
    can refuse instead of shipping a mangled prompt.
    """
    parts = re.split(r"(<d>\[Japanese\].*?</d>)", str(prompt or ""),
                     flags=re.S)
    runs: list[str] = []
    for index in range(0, len(parts), 2):
        runs.extend(re.findall(_CJK_RUN, parts[index]))
    return runs


def _section_bodies(prompt: str) -> dict[str, str]:
    """Raw text of every section, keyed by section name (order preserved)."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in str(prompt or "").split("\n"):
        if line in SECTIONS:
            current = line
            bodies[current] = []
            continue
        if current is not None:
            bodies[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in bodies.items()}


def validate_compiled_prompt(prompt: str) -> dict:
    """Hard pre-generation validation. Returns {"ok": bool, "errors": [...]}.

    Fails on: (1) untranslated CJK left outside <d> - defensive here: on the
    normal compile_direct_prompt path this can no longer fire because
    compile_direct_prompt already raises UntranslatedPromptError earlier
    (before strip_cjk_outside_d runs); this branch stays as an assertion for
    any prompt string that reaches this function by another route, (2) a
    section whose body has no alphanumeric/word content once
    labels/tags/punctuation are stripped (catches
    "summary: [reference generation] ."), and (3) orphan punctuation
    artefacts (", ,", ", .", a line that is only ".", or a word directly
    followed by whitespace then a period, e.g. "front-facing ."). Legitimate
    content ("N/A", "[Shot 1]", "<Picture 1>", "00:03.500," and <d> payloads)
    must validate clean.
    """
    text = str(prompt or "")
    errors: list[str] = []

    runs = untranslated_runs(text)
    if runs:
        sample = ", ".join(sorted(set(runs))[:5])
        errors.append(f"未翻訳の日本語が残っています: {sample}")

    outside = re.sub(r"<d>\[Japanese\].*?</d>", " ", text, flags=re.S)
    if re.search(r",\s*,", outside):
        errors.append("不正な区切り記号（連続するカンマ）があります。")
    if re.search(r",\s*\.", outside):
        errors.append("不正な区切り記号（カンマの直後にピリオド）があります。")
    if re.search(r"(^|\n)[ \t]*\.[ \t]*(\n|$)", outside):
        errors.append("空文（ピリオドのみの行）があります。")
    if re.search(r"\S\s+\.", outside):
        errors.append(
            "孤立した句読点（単語の直後に空白を挟んでピリオド）があります。")

    for name, body in _section_bodies(text).items():
        stripped = re.sub(r"<[^>]*>", " ", body)
        stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
        if not re.search(r"[^\W\d_]", stripped, re.UNICODE):
            errors.append(f"セクション「{name}」の内容が空、または断片的です。")

    return {"ok": not errors, "errors": errors}
