"""Deterministic shot timing review; no generation, files, or network."""
import math

from . import director_convert, director_speech


def timeline_errors(timeline, seconds: float) -> list[str]:
    errors = []
    previous_end = 0.0
    if not isinstance(timeline, list):
        return ["タイムラインは配列で指定してください。"]
    if len(timeline) > 8:
        errors.append("1クリップのショットは最大8個です。")
    for index, shot in enumerate(timeline):
        try:
            start, end = float(shot["t0"]), float(shot["t1"])
            if not all(math.isfinite(t) for t in (start, end)):
                raise ValueError
            if start < previous_end or end <= start or end > seconds:
                raise ValueError
            if index == 0 and start != 0:
                raise ValueError
            previous_end = end
        except (ValueError, TypeError, KeyError):
            errors.append(f"ショット{index + 1}: 0秒開始・重複なし・尺内の開始/終了時刻にしてください。")
    return errors


def review(spec: dict) -> dict:
    errors = []
    if spec.get("kind") == "single":
        errors += timeline_errors(spec.get("timeline") or [], float(spec.get("duration_sec") or 5))
        parts = [("単発", spec.get("items") or {})]
    else:
        parts = [(f"CLIP {c.get('index', 0) + 1}", c.get("items") or {})
                 for c in spec.get("clips") or [] if isinstance(c, dict)]
    lines = []
    for label, items in parts:
        spoken = director_speech.dialogue_lines((items.get("dialogue") or {}).get("value"))
        for text in spoken:
            if "<" in text or ">" in text:
                errors.append(f"{label}: 台詞内に制御タグは使えません。")
        lines.append({"clip": label, "dialogue": spoken, "silent": not spoken})
    return {"ok": not errors, "errors": errors, "lines": lines,
            "warnings": director_convert.dialogue_budget_warnings(spec),
            "timing_source": "estimate", "audio_verified": False}
