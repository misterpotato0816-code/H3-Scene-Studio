# -*- coding: utf-8 -*-
"""AI Director prompt texts (pure builders, no I/O).

The VLM answers in JSON only. Every builder states the exact keys it wants
so the provider layer can parse without guessing. Japanese values, JSON keys
in English (matches director_spec field names).
"""
from __future__ import annotations

import json as _json

from . import director_spec as spec_mod

_JSON_RULE = (
    "考え過程・前置き・後書き・コードフェンスを一切出力しない。"
    "最初の一文字から最後の一文字までJSONオブジェクトのみを出力する。"
    "値はすべて日本語の自然文（空欄可）。不明な項目は空文字にし、捏造しない。"
)

_REPAIR_NUDGE = (
    "前回の回答は形式違反である。推論過程・説明文は一切書かず、"
    "JSONオブジェクトのみを出力し直すこと。"
)

_SINGLE_KEYS = list(spec_mod.SINGLE_ITEMS)
_MASTER_KEYS = list(spec_mod.MASTER_ITEMS)
_CLIP_KEYS = list(spec_mod.CLIP_ITEMS)


def _items_schema(names: list[str]) -> dict:
    return {name: {"value": "", "en": "", "locked": False, "request": ""}
            for name in names}


_EN_RULE = (
    "セリフは話し言葉の日本語にし、書き言葉や説明調の長文を避ける。"
    "感情・間・抑揚・動作の指示をdialogueの本文へ混ぜない。"
    "疑問・断定・驚きに合う句読点を使い、自然な息継ぎの余白を残す。"
    "各項目のvalueは日本語、enには同じ内容を表す簡潔な英語"
    "（H3プロンプト用の映像・音声描写としての英語）を書く。"
    "enは<Subject 1>を主語として書かない（主語はアプリ側が自動で付けるため）。"
    "動詞句・名詞句で短く書く。1項目には1つの短い記述のみを書き、複数の文を"
    "つなげない。人物を「a young woman」のような言い換えで再登場させない。"
    "ただしdialogue項目のenは台詞を翻訳しないため常に空文字にする。"
)


_CAST_RULE = (
    "登場人物の役割もcastに書く。on_screen_subjectsの先頭は必ず"
    "\"character:0\"（選択Character＝Subject 1）。画面内に二人目がいる場合のみ"
    "二人目を追加し、いない場合は絶対に追加しない。撮影者・カメラマンは"
    "on_screen_subjectsに入れずcamera_operatorに書き、画面に映らないなら"
    "camera_operator_visibilityを\"off_camera\"にする。camera_povは\"\"・"
    "\"lover_pov\"・\"selfie\"・\"observer\"・\"handheld_third_person\"のいずれか。"
)

_CAST_SCHEMA = {"on_screen_subjects": ["character:0"],
                "camera_operator": "", "camera_operator_visibility": "",
                "camera_device": "", "camera_pov": "",
                "off_camera_people": []}


def single_system() -> str:
    return (
        "あなたはショート動画の監督（AI Director）である。"
        "ユーザーのアイデア・動画尺・要望から、動画一本分の監督案を作る。"
        f"{_JSON_RULE}"
        f"{_EN_RULE}"
        f"{_CAST_RULE}"
        "出力形式: " + _json.dumps(
            {"items": _items_schema(_SINGLE_KEYS),
              "cast": _CAST_SCHEMA,
              "timeline": [{"t0": 0.0, "t1": 2.0, "label": "", "en": ""}]},
            ensure_ascii=False)
    )


def _dialogue_budget_line(duration_sec: int) -> str:
    """Duration-aware speech budget (prompt-level, never compiler rewrites).

    The compiler must not summarize or alter dialogue; length is controlled
    here, at generation time, using real speaking-time sense.
    """
    if duration_sec <= 5:
        return "台詞は短い一言のみ（1文）。尺に収まらない量は書かない。"
    if duration_sec <= 10:
        return "台詞は自然に話し切れる1〜2文まで。長い台詞は書かない。"
    return "台詞は自然に話し切れる2〜3文まで。詰め込みは禁止。"


def single_user(*, idea: str, duration_sec: int, initial_request: str,
                ref_hint: str = "") -> str:
    parts = [f"アイデア: {idea}", f"動画尺: {duration_sec}秒"]
    if initial_request.strip():
        parts.append(f"監督への要望: {initial_request.strip()}")
    if ref_hint.strip():
        parts.append(f"参照画像の手がかり: {ref_hint.strip()}")
    parts.append(_dialogue_budget_line(duration_sec))
    parts.append("タイムラインは0秒から開始し、時刻の重複や逆転を禁止する。"
                 "最終t1は動画尺以内にする。labelは日本語、enは同内容の英語。"
                 "各ショットは構図・視線・主動作・カメラ速度・終わりの姿勢を明確にする。")
    parts.append(
        f"{duration_sec}秒に収まる台詞量・演技量にすること。"
        "早口を前提にしない。間・表情・カメラの時間も残す。"
        "服装・カメラ・場所等を勝手に固定しない（アイデアから自然なものを選ぶ）。")
    return "\n".join(parts)


def master_system() -> str:
    return (
        "あなたは30秒ショート動画の総監督である。まず全体を統括する"
        "MASTER DIRECTIONを作る。CLIP固有の詳細（場所・演技・台詞・構図）は"
        "ここでは決めない。維持すべき条件と変化してよい条件を区別する。"
        f"{_JSON_RULE}"
        f"{_EN_RULE}"
        f"{_CAST_RULE}"
        "出力形式: " + _json.dumps(
            {"items": _items_schema(_MASTER_KEYS),
              "cast": _CAST_SCHEMA,
              "continuity_rules": {"keep": [], "flexible": []}},
            ensure_ascii=False)
    )


def master_user(*, idea: str, initial_request: str) -> str:
    parts = [f"アイデア: {idea}", "動画尺: 30秒（10秒×3クリップ構成の全体設計）"]
    if initial_request.strip():
        parts.append(f"監督への要望: {initial_request.strip()}")
    parts.append("continuity_rules.keepには全CLIP共通で維持する条件を、"
                 "flexibleにはStory展開として変化してよい観点を列挙する。")
    return "\n".join(parts)


def clips_system() -> str:
    return (
        "あなたは30秒ショート動画の監督である。与えられたMASTER DIRECTIONの"
        "もとで、CLIP 1〜3（各10秒）を一括設計する。3本を独立に作らず、"
        "一本の動画として会話・展開・所持品・感情・場所移動を通して設計する。"
        "各CLIPにstart_state/end_state/carry_overを持たせ、"
        "CLIP Nの終了がN+1の開始へ自然につながるようにする。"
        f"{_JSON_RULE}"
        f"{_EN_RULE}"
        "出力形式: " + _json.dumps(
            {"clips": [{"index": 0, "duration_sec": 10,
                        "items": _items_schema(_CLIP_KEYS),
                        "start_state": {}, "end_state": {}, "carry_over": {}}]},
            ensure_ascii=False)
    )


def clips_user(*, master: dict, idea: str = "",
               locked_note: str = "") -> str:
    parts = ["MASTER DIRECTION:",
             _json.dumps(master, ensure_ascii=False)]
    if idea.strip():
        parts.append(f"元アイデア: {idea.strip()}")
    if locked_note.strip():
        parts.append(f"固定条件（変更禁止）: {locked_note.strip()}")
    parts.append("MASTERのcast（登場人物・撮影者の役割）は変更禁止。"
                 "二人目を画面に足さない。台詞は各CLIPに自然に話し切れる"
                 "短い1文ずつ配分し、一つの長い台詞を話し続ける設計に"
                 "しない。各10秒に動き・間・表情の時間を残す。")
    return "\n".join(parts)


def regen_system(*, scope: str, shape: dict | None = None) -> str:
    base = (
        "あなたはショート動画の監督である。現在の監督案の一部だけを作り直す。"
        f"対象範囲: {scope}。"
        "回答のJSONには変更対象の項目のみを含め、対象外の項目は含めない。"
        "変更箇所は尺・演技・前後文脈・Story全体との整合性を保つ。"
        f"{_JSON_RULE}"
        f"{_EN_RULE}"
    )
    if shape is not None:
        base += "\n出力形式: " + _json.dumps(shape, ensure_ascii=False)
    return base


def single_item_shape(name: str) -> dict:
    return {"items": {name: {"value": "", "en": "", "locked": False,
                             "request": ""}}}


def single_full_shape() -> dict:
    return {"items": _items_schema(_SINGLE_KEYS), "timeline": []}


def story_shape(*, with_master: bool = True) -> dict:
    doc: dict = {"clips": [{"index": 0,
                            "items": _items_schema(_CLIP_KEYS),
                            "start_state": {}, "end_state": {},
                            "carry_over": {}}]}
    if with_master:
        doc["master"] = {"items": _items_schema(_MASTER_KEYS)}
    return doc


def regen_user(*, current: dict, scope: str, item_request: str = "",
               global_request: str = "", boundary: str = "") -> str:
    parts = [f"対象範囲: {scope}",
             "現在の監督案:", _json.dumps(current, ensure_ascii=False)]
    if item_request.strip():
        parts.append(f"この範囲への要望（最優先）: {item_request.strip()}")
    if global_request.strip():
        parts.append(f"全体要望: {global_request.strip()}")
    if boundary.strip():
        parts.append(f"境界条件（前後と矛盾させない）: {boundary.strip()}")
    return "\n".join(parts)
