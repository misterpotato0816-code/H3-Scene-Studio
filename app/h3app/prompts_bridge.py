# -*- coding: utf-8 -*-
"""Read-only bridge to E:\\AI-Projects\\H3\\build\\prompts.py.

The system/user prompt wording is part of the finished pipeline and was tuned
against real VLM output. Copying it here would create a second source of truth
that silently rots, so the app imports the original module instead and never
writes to it.
"""
from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from .config import BUILD_DIR

_PROMPTS: ModuleType | None = None

# The exact names graphs.py depends on. Checked at import so a wording refactor
# upstream fails here with a clear message instead of producing an empty prompt.
REQUIRED = (
    "SYS_CHARACTER_PROFILE", "USER_CHARACTER_PROFILE",
    "SYS_DIRECTOR_NEW", "SYS_DIRECTOR_CONTINUATION",
    "HDR_PROFILE", "HDR_SCENE", "HDR_PREV", "HDR_NEG",
    "HDR_LEN_A", "HDR_LEN_B", "HDR_DIALOGUE", "HDR_DIALOGUE_END",
    "NEGATIVE_DEFAULT", "JP_SCENE_DEFAULT", "JP_DIALOGUE_DEFAULT",
)


class PromptsUnavailable(RuntimeError):
    pass


def prompts() -> ModuleType:
    global _PROMPTS
    if _PROMPTS is not None:
        return _PROMPTS

    src = BUILD_DIR / "prompts.py"
    if not src.is_file():
        raise PromptsUnavailable(
            f"プロンプト定義が見つかりません: {src}\n"
            "H3 本体の build フォルダが移動・削除されていないか確認してください。")

    # Loaded under a private name so it cannot collide with anything ComfyUI or
    # a node pack might have already put in sys.modules as "prompts".
    spec = importlib.util.spec_from_file_location("h3_build_prompts", src)
    if spec is None or spec.loader is None:
        raise PromptsUnavailable(f"プロンプト定義を読み込めません: {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_build_prompts"] = mod
    # "never writes to it" includes bytecode: the default import machinery would
    # drop a __pycache__ into the frozen build/ directory, which is read-only for
    # this app by policy.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise PromptsUnavailable(f"プロンプト定義の読み込みに失敗しました ({src}): {exc}") from exc
    finally:
        sys.dont_write_bytecode = previous

    missing = [n for n in REQUIRED if not hasattr(mod, n)]
    if missing:
        raise PromptsUnavailable(
            f"プロンプト定義に必要な項目がありません ({src}): {', '.join(missing)}")

    _PROMPTS = mod
    return mod


def build_director_text(profile: str, jp_scene: str, jp_prev: str, jp_negative: str,
                        jp_dialogue: str, frames: int) -> str:
    """The joined director user prompt.

    Order and delimiter are copied from build_phase1.py's `pieces` chain of
    2-input JoinStringMulti nodes. Dialogue goes LAST on purpose: a 4B-class VLM
    follows the most recent instruction most reliably (see prompts.HDR_DIALOGUE).
    """
    p = prompts()
    pieces = [
        p.HDR_PROFILE, profile,
        p.HDR_SCENE, jp_scene,
        p.HDR_PREV, jp_prev,
        p.HDR_NEG, jp_negative,
        p.HDR_LEN_A, str(frames),
        p.HDR_LEN_B,
        p.HDR_DIALOGUE, jp_dialogue, p.HDR_DIALOGUE_END,
    ]
    return "\n".join(pieces)


def strip_tags(text: str) -> str:
    """Mirror of the "CR Text Replace" node in group 40.

    Done in Python rather than as a node so the app keeps one round-trip fewer
    and can log the raw model output alongside the cleaned one.
    """
    for token in ("```", "<think>", "</think>"):
        text = text.replace(token, "")
    return text.strip()
