# -*- coding: utf-8 -*-
"""Failure -> Japanese user-facing message.

Every message answers three things, because those are the only three questions a
user actually has when a 3.6 minute job dies:
  where  : どこで止まったか (stage + node)
  retry  : もう一度やって直るのか
  config : 設定の問題なのか、直すなら何を
"""
from __future__ import annotations

import re


class PipelineError(RuntimeError):
    """Carries the stage/node context the classifier needs."""

    def __init__(self, message: str, *, kind: str = "unknown", stage: str = "",
                 node: str = "", detail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.stage = stage
        self.node = node
        self.detail = detail or message


class UserCancelled(PipelineError):
    def __init__(self, stage: str = "", node: str = ""):
        super().__init__("ユーザーが中止しました", kind="cancelled", stage=stage, node=node)


# kind -> (title, retry, config problem, what to do)
_TABLE = {
    "comfy_launch": (
        "生成エンジンの起動に失敗しました",
        True,
        True,
        "config.json の comfy_dir / comfy_python が正しいか確認してください。"
        "_debug\\comfy.err に起動時のログが残っています。",
    ),
    "port_in_use": (
        "ポートが使用中です",
        False,
        True,
        "config.json の comfy_port（既定 8411）または app_port（既定 8790）を"
        "空いている番号に変更してください。既存のプロセスは終了させません。",
    ),
    "missing_nodes": (
        "必要なノードが導入されていません",
        False,
        True,
        "不足しているノードパックを ComfyUI の custom_nodes に導入してから再起動してください。",
    ),
    "no_gpu1": (
        "必要なGPUが見えていません",
        False,
        True,
        "設定 > GPU で動画生成側と別処理側のGPUを確認し、H3が起動したComfyUIで生成してください"
        "（別の方法で起動したComfyUIはH3の保存済みGPU設定を使いません）。",
    ),
    "missing_model": (
        "モデルファイルが見つかりません",
        False,
        True,
        "モデルの実体（comfy_paths.yaml の base_path 配下）と "
        "comfy_paths.yaml の設定を確認してください。",
    ),
    "oom": (
        "GPU のメモリが不足しました",
        True,
        True,
        "参照画像の枚数を減らす、出力解像度を 576x1024 に戻す、"
        "詳細設定の参照画像サイズを下げる、のいずれかを試してください。",
    ),
    "vlm_abort": (
        "文章生成モデルが異常終了しました",
        True,
        True,
        "生成エンジンごと落ちています。もう一度実行してください。"
        "繰り返す場合は config.json の vlm 設定（n_ctx）を小さくしてください。",
    ),
    "timeout": (
        "応答が止まったため中断しました",
        True,
        False,
        "一時的な停止の可能性があります。もう一度実行してください。",
    ),
    "cancelled": (
        "中止しました",
        True,
        False,
        "設定は保持しています。そのまま再実行できます。",
    ),
    "parity": (
        "検証済みの生成設定から外れています",
        False,
        True,
        "詳細設定の「既定値に戻す」を押してください。"
        "モデル名・GPU の割り当てを変更した場合は config.json を元に戻してください。",
    ),
    "turbo_bind": (
        "Turbo LoRAがモデルに結合できません（TURBO_BIND_FAILED）",
        False,
        True,
        "LEGACYへの自動切替はしません。Turbo LoRAファイルを確認するか、"
        "LEGACYモードを選んで再実行してください。",
    ),
    "mode_unavailable": (
        "そのモードは利用できません",
        False,
        True,
        "Experimental機能は対応ノードの導入後に有効化されます。"
        "FAST / QUALITY / LEGACY / LONG を使ってください。",
    ),
    "prompt_rejected": (
        "生成エンジンがリクエストを受け付けませんでした",
        False,
        True,
        "入力値のいずれかが範囲外です。詳細設定を既定値に戻して試してください。",
    ),
    "comfy_not_running": (
        "ComfyUI が起動していません",
        False,
        False,
        "まだ一度も生成していないか、すでに終了しています。"
        "GPU 上にモデルは残っていないため、解放の必要はありません。",
    ),
    "busy": (
        "いま生成中です",
        True,
        False,
        "生成が終わるか中止したあとで、もう一度実行してください。",
    ),
    "gpu_config": (
        "GPU設定に問題があります",
        False,
        True,
        "設定 > GPU を開き、GPUの割り当て（運用モード・動画生成側・別処理側・予約VRAM）を"
        "見直してから、もう一度実行してください。",
    ),
    "gpu_mismatch": (
        "起動中のComfyUIのGPU割り当てを確認できません",
        False,
        True,
        "設定 > GPU の「現在適用中」を確認してください。"
        "H3が起動したものではないComfyUIは自動では再起動しません。"
        "そのComfyUIを終了してからもう一度実行すると、H3が保存済み設定で起動します。",
    ),
    "network": (
        "ComfyUI との通信に失敗しました",
        True,
        False,
        "もう一度実行してください。繰り返す場合は _debug\\comfy.err を確認してください。",
    ),
    "ai_config": (
        "AI設定に問題があるため処理を続けられません",
        False,
        True,
        "設定 > LLM接続 で、接続先（LM Studio など）とモデルを指定し、"
        "URL / モデル / 画像入力対応を確認してください。",
    ),
    "input_missing": (
        "参照ファイルが見つかりません",
        True,
        False,
        "画像を追加し直してください。",
    ),
    "comfy_upload": (
        "ComfyUIへ参照ファイルを渡せませんでした",
        True,
        True,
        "ComfyUIが起動しているか確認し、もう一度生成してください。",
    ),
    "unknown": (
        "原因不明のエラーで停止しました",
        True,
        False,
        "もう一度実行してください。繰り返す場合は _debug\\comfy.err を確認してください。",
    ),
}

_PATTERNS = (
    ("oom", re.compile(r"cuda out of memory|outofmemoryerror|allocation on device", re.I)),
    ("vlm_abort", re.compile(r"ggml_assert|llama\.cpp|windows fatal exception|abort\(\)", re.I)),
    ("no_gpu1", re.compile(r"gpu:1|only \d+ cuda device", re.I)),
    ("missing_model", re.compile(r"value not in list|not in \(?list of|no such file.*safetensors", re.I)),
    ("missing_nodes", re.compile(r"required nodes missing|node type not found|does not exist", re.I)),
    ("timeout", re.compile(r"no workflow progress|did not become ready|timeout", re.I)),
)

# class_type prefix -> where it comes from. Used to name the missing pack.
NODE_PACKS = (
    ("llama_cpp_", "ComfyUI-llama-cpp（任意。接続先「ComfyUI内ローカルGemma」を"
     "選んだ場合のみ必要。配布元にライセンス表記なし）"),
    ("LayerUtility: ", "comfyui_layerstyle"),
    ("H3Cuda", "custom_nodes/H3-Device-Barrier（このリポジトリ内。comfy_paths.yaml の custom_nodes に含める）"),
    ("H3Gated", "custom_nodes/H3-Device-Barrier（このリポジトリ内。comfy_paths.yaml の custom_nodes に含める）"),
    ("H3Clip", "custom_nodes/H3-Device-Barrier（このリポジトリ内。comfy_paths.yaml の custom_nodes に含める）"),
)


def pack_for(class_type: str) -> str:
    for prefix, pack in NODE_PACKS:
        if class_type.startswith(prefix):
            return pack
    return "ComfyUI 本体"


def classify(exc: BaseException, stage: str = "", node: str = "") -> dict:
    """Build the 3-part message dict the UI renders."""
    kind = getattr(exc, "kind", None)
    stage = getattr(exc, "stage", "") or stage
    node = getattr(exc, "node", "") or node
    text = str(exc)

    if not kind or kind == "unknown":
        for candidate, pattern in _PATTERNS:
            if pattern.search(text):
                kind = candidate
                break
    if not kind:
        kind = "unknown"

    title, retry, is_config, advice = _TABLE.get(kind, _TABLE["unknown"])
    where = stage or "生成処理"
    if node:
        where += f"（処理: {node}）"

    return {
        "kind": kind,
        "title": title,
        "where": f"止まった場所: {where}",
        "retry": "再試行: はい、もう一度実行すれば直る可能性があります"
                 if retry else "再試行: いいえ、同じ設定では同じ結果になります",
        "config": ("設定の問題: はい — " + advice) if is_config
                  else ("設定の問題: いいえ — " + advice),
        # The same advice without the error-card prefix, for callers that show a
        # plain one-line reason instead of the three-part card.
        "advice": advice,
        "retryable": bool(retry),
        "detail": getattr(exc, "detail", "") or text,
    }
