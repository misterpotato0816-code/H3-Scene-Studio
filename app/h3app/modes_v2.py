# -*- coding: utf-8 -*-
"""v2 Generation Modes for the H3 app. Additive layer over the v1 chain.

UI modes (FAST default) resolve to concrete settings/models/patches and a
validation gate. LEGACY keeps the byte-exact v1 path + assert_v1_parity.
No silent substitution: unavailable modes and bind failures raise
PipelineError with explicit kinds the UI renders verbatim.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from .errors import PipelineError

FAST_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
FAST_MIN_BINDABLE_KEYS = 600
# Overnight 2026-09-07 Phase 3: int8_convrot VAE, measured -28% (124f, no-fast)
# and -6% stacked on --fast (362f), MAE 2.2 vs fp16, no color shift.
# FAST-family only: LEGACY keeps the fp16 file for byte-parity.
FAST_VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"

# UI order. advanced/experimental only change presentation, never behaviour.
UI_MODES = [
    {"id": "FAST", "label": "FAST（高速）", "default": True,
     "desc": "Ref2VA Turbo 4step。実測135.6秒。"},
    {"id": "QUALITY", "label": "QUALITY（高品質）", "default": False,
     "desc": "Base 20step。実測403.5秒。"},
    {"id": "LEGACY", "label": "LEGACY（v1完全fallback）", "default": False,
     "desc": "実測217.61秒のv1構成そのまま。"},
    {"id": "LONG", "label": "LONG（Story連鎖）", "default": False,
     "desc": "Story Runner v2。Original Re-anchor有効。"},
]
UI_MODES_ADVANCED = [
    {"id": "BALANCED", "label": "BALANCED", "desc": "Base 10step。実測260.7秒。"},
    {"id": "LOW_VRAM", "label": "LOW VRAM", "desc": "省VRAMパッチ。576x1024では削減効果なし、画質同等。"},
    {"id": "LONG_FAST", "label": "LONG_FAST（Story高速）",
     "desc": "Story連鎖 + Turbo 4step + Sage。30秒実測469.3秒（480x864）。"
             "声は1本目のクリップで固定されます。"},
]
UI_MODES_EXPERIMENTAL = [
    {"id": "CONTROL", "label": "CONTROL", "reason": "H3 FunControlノードが未導入（Wan用のみ存在）"},
    {"id": "PDD", "label": "PDD", "reason": "PDD対応ノードが未導入"},
    {"id": "ENDLESS", "label": "ENDLESS", "reason": "未設計"},
]

RUNNABLE_MODES = ("FAST", "QUALITY", "LEGACY", "LONG", "LONG_FAST", "BALANCED", "LOW_VRAM")
STORY_MODES = ("LONG", "LONG_FAST")
# Modes whose whole job (director + generate graphs) runs on the fast launch
# profile (base comfy_launch_args + comfy_launch_args_fast/--fast).
# Overnight 2026-09-07 parity result: --fast changes LEGACY output bytes, so
# every other mode stays on the base profile. The profile is job-level, never
# per-graph: switching between director and generate graphs would restart the
# server twice per segment.
FAST_LAUNCH_MODES = ("FAST", "LONG_FAST")


def launch_profile(mode: str) -> str:
    """Return "fast" for FAST-family modes, "base" for everything else."""
    return "fast" if (mode or "LEGACY").upper() in FAST_LAUNCH_MODES else "base"
# Modes that only make sense as a multi-clip Story chain. Accepting them on the
# single-shot path would pair the turbo recipe with the legacy tail relay, a
# combination that was never measured.
STORY_ONLY_MODES = ("LONG_FAST",)


def mode_error(mode: str, code: str, message: str) -> PipelineError:
    kind = {"UNAVAILABLE": "mode_unavailable",
            "TURBO_BIND_FAILED": "turbo_bind"}.get(code, "parity")
    return PipelineError(f"{code}: {message}", kind=kind,
                         stage="生成設定", detail=f"mode={mode} {code}: {message}")


def experimental_reason(mode: str) -> str | None:
    for m in UI_MODES_EXPERIMENTAL:
        if m["id"] == mode:
            return m["reason"]
    return None


def lora_bindable_keys(loras_dir: Path, lora_name: str) -> int:
    """Count header keys carrying the ComfyUI prefix. Header only, no weights."""
    path = Path(loras_dir) / lora_name
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return sum(1 for k in header if k != "__metadata__"
               and str(k).startswith("diffusion_model."))


def _check_turbo_lora_bind(mode: str, loras_dir: Path | None) -> None:
    """Shared FAST/LONG_FAST gate: refuse (never silently swap) on bad Turbo LoRA."""
    if loras_dir is None:
        return
    try:
        bound = lora_bindable_keys(Path(loras_dir), FAST_LORA)
    except Exception as exc:                                   # noqa: BLE001
        raise mode_error(mode, "TURBO_BIND_FAILED",
                         f"Turbo LoRAを読めません: {exc}") from exc
    if bound < FAST_MIN_BINDABLE_KEYS:
        raise mode_error(
            mode, "TURBO_BIND_FAILED",
            f"Turbo LoRAのbind数が不足です（{bound} < {FAST_MIN_BINDABLE_KEYS}）。"
            "LEGACYへ自動切替はしません。LEGACYを選んで再実行してください。")


def resolve(mode: str, settings: dict, models: dict, *,
            loras_dir: Path | None = None) -> dict:
    """Return {"settings","models","gate","patches"}. Never substitutes silently."""
    mode = (mode or "LEGACY").upper()
    reason = experimental_reason(mode)
    if reason is not None:
        raise mode_error(mode, "UNAVAILABLE", reason)
    if mode not in RUNNABLE_MODES:
        raise mode_error(mode, "UNAVAILABLE", f"不明なモードです: {mode}")

    settings = dict(settings)
    models = dict(models)
    patches: dict = {}
    if mode == "LEGACY":
        return {"settings": settings, "models": models,
                "gate": "v1_parity", "patches": patches}
    if mode == "FAST":
        settings.update({"steps": 4, "sampler": "euler", "scheduler": "simple"})
        models.update({"lora": FAST_LORA, "lora_strength": 1.0,
                       "vae_video": FAST_VAE_VIDEO})
        patches["sigshift"] = {"shift_video": 12.0, "shift_audio": 3.0}
        _check_turbo_lora_bind(mode, loras_dir)
        return {"settings": settings, "models": models,
                "gate": "v2_turbo", "patches": patches}
    if mode == "QUALITY":
        settings.update({"steps": 20, "sampler": "res_multistep",
                         "scheduler": "simple"})
        models.update({"lora": None})
        return {"settings": settings, "models": models,
                "gate": "v2_base", "patches": patches}
    if mode == "BALANCED":
        settings.update({"steps": 10, "sampler": "res_multistep",
                         "scheduler": "simple"})
        models.update({"lora": None})
        return {"settings": settings, "models": models,
                "gate": "v2_base", "patches": patches}
    if mode == "LOW_VRAM":
        models.update({"lora": None})
        patches["lowvram"] = {"head_chunks": 4}
        patches["chunkff"] = {"chunks": 2, "seq_threshold": 4096}
        return {"settings": settings, "models": models,
                "gate": "v2_lowvram", "patches": patches}
    if mode == "LONG_FAST":
        # Story mode running the measured V3 recipe: FAST's turbo settings plus
        # Sage attention. ORIGINAL RE-ANCHOR is enforced at validate time, same
        # as LONG (refs must stay connected; tail relay is motion aid only).
        settings.update({"steps": 4, "sampler": "euler", "scheduler": "simple"})
        models.update({"lora": FAST_LORA, "lora_strength": 1.0,
                       "vae_video": FAST_VAE_VIDEO})
        patches["sigshift"] = {"shift_video": 12.0, "shift_audio": 3.0}
        patches["sage"] = {}
        _check_turbo_lora_bind(mode, loras_dir)
        return {"settings": settings, "models": models,
                "gate": "v2_turbo_long", "patches": patches}
    # LONG: same sampling chain as LEGACY (story runner adds tail relay per clip).
    # ORIGINAL RE-ANCHOR is enforced at validate time (refs must stay connected).
    return {"settings": settings, "models": models,
            "gate": "v1_parity", "patches": patches}


def _require_original_reanchor(graph: dict) -> None:
    """ORIGINAL RE-ANCHOR: tail relay is motion/pose/context aid only.
    Identity truth stays the ORIGINAL references, every clip."""
    from . import graphs
    h3 = [n for n in graph.values()
          if n.get("class_type") == "MiniMaxH3ReferenceToVideo"]
    refs = [k for k in h3[0]["inputs"]
            if k.startswith("ref_images.ref_image_")]
    if not refs:
        raise graphs.GraphError(
            "LONG要否: Original Referenceが接続されていません。"
            "tailだけの連鎖はidentity driftを起こすため禁止です。")


def _require_no_tail_relay(graph: dict) -> None:
    """LONG_FAST structural backstop: the 0f contract (last-frame relay only,
    never the legacy tail-video/tail-audio nodes) must hold even if a caller
    upstream builds a continuation dict that was never meant for this mode."""
    from . import graphs
    banned = {"LoadVideo", "GetVideoComponents", "ImageFromBatch",
              "TrimAudioDuration"}
    present = sorted({n.get("class_type") for n in graph.values()
                      if n.get("class_type") in banned})
    if present:
        raise graphs.GraphError(
            "LONG_FAST: 0フレームtail relayの契約に反しています。"
            f"禁止ノードが含まれています: {', '.join(present)}")


def validate_graph(graph: dict, resolved: dict, manifest: dict,
                   defaults: dict, mode: str, *, clip_device: str = "gpu:1") -> None:
    """Mode-aware submit gate. LEGACY delegates to the frozen v1 parity check."""
    from . import graphs
    gate = resolved["gate"]
    settings = resolved["settings"]
    models = resolved["models"]
    if gate == "v1_parity":
        graphs.assert_v1_parity(graph, settings, models, manifest,
                                defaults=defaults, clip_device=clip_device)
        if mode == "LONG":
            _require_original_reanchor(graph)
        return
    if gate == "v2_turbo_long":
        _require(graph, settings, models, mode, resolved["patches"], clip_device=clip_device)
        _require_original_reanchor(graph)
        _require_no_tail_relay(graph)
        return
    # v2 gates: same structural invariants as v1, with the mode's own values.
    _require(graph, settings, models, mode, resolved["patches"], clip_device=clip_device)


def _require(graph: dict, settings: dict, models: dict, mode: str,
             patches: dict, *, clip_device: str = "gpu:1") -> None:
    from . import graphs

    def _find(class_type: str):
        return [(k, n) for k, n in graph.items()
                if n.get("class_type") == class_type]

    h3 = _find("MiniMaxH3ReferenceToVideo")
    if len(h3) != 1:
        raise graphs.GraphError(f"{mode}: H3ノードが1個ではありません。")
    if h3[0][1]["inputs"].get("ref_image_size") != settings["ref_image_size"]:
        raise graphs.GraphError(f"{mode}: ref_image_size不一致。")
    sel = _find("SelectCLIPDevice")
    if len(sel) != 1 or sel[0][1]["inputs"].get("device") != clip_device:
        raise graphs.GraphError(f"{mode}: SelectCLIPDevice({clip_device})がありません。")
    gated = _find("H3GatedCLIPLoader")
    if len(gated) != 1 or gated[0][1]["inputs"].get("clip_name") != models["clip"]:
        raise graphs.GraphError(f"{mode}: text encoder不一致。")
    unet = _find("UNETLoader")
    if len(unet) != 1 or unet[0][1]["inputs"].get("unet_name") != models["unet"]:
        raise graphs.GraphError(f"{mode}: UNET不一致。")
    lora = _find("LoraLoaderModelOnly")
    if models.get("lora"):
        if len(lora) != 1 or lora[0][1]["inputs"].get("lora_name") != models["lora"]:
            raise graphs.GraphError(f"{mode}: Turbo LoRA不一致（黙って差替え禁止）。")
    elif lora:
        raise graphs.GraphError(f"{mode}: Base構成にLoRAノードが残っています。")
    ks = _find("KSamplerSelect")
    if len(ks) != 1 or ks[0][1]["inputs"].get("sampler_name") != settings["sampler"]:
        raise graphs.GraphError(f"{mode}: sampler不一致。")
    sch = _find("BasicScheduler")
    if len(sch) != 1 or int(sch[0][1]["inputs"].get("steps", -1)) != int(settings["steps"]):
        raise graphs.GraphError(f"{mode}: steps不一致。")
    if "sigshift" in patches:
        ss = _find("MiniMaxH3SigmaShift")
        exp = patches["sigshift"]
        if len(ss) != 1 or float(ss[0][1]["inputs"].get("shift_video", -1)) != float(exp["shift_video"]) \
                or float(ss[0][1]["inputs"].get("shift_audio", -1)) != float(exp["shift_audio"]):
            raise graphs.GraphError(f"{mode}: SigmaShift(12,3)がありません。")
    if "lowvram" in patches and len(_find("MiniMaxLowVRAMAttention")) != 1:
        raise graphs.GraphError(f"{mode}: LowVRAM Attentionがありません。")
    if "chunkff" in patches and len(_find("MiniMaxChunkFeedForward")) != 1:
        raise graphs.GraphError(f"{mode}: ChunkFeedForwardがありません。")
    if "sage" in patches and len(_find("MiniMaxH3MemoryEfficientSageAttentionPatch")) != 1:
        raise graphs.GraphError(f"{mode}: Sage Attentionがありません。")


def compat_status(*, comfy_reachable: bool, object_info: dict | None,
                  gpus: list[str], loras_dir: Path,
                  models_present: dict[str, bool]) -> dict:
    """Startup checklist rows for the UI. Missing optionals never fail boot."""
    oi = object_info or {}
    turbo_ok = False
    try:
        turbo_ok = lora_bindable_keys(Path(loras_dir), FAST_LORA) >= FAST_MIN_BINDABLE_KEYS
    except Exception:                                        # noqa: BLE001
        turbo_ok = False
    ref2va = "MiniMaxH3ReferenceToVideo" in oi if oi else None
    modes = {"FAST": bool(turbo_ok and (ref2va is not False)),
             "QUALITY": ref2va is not False,
             "LEGACY": ref2va is not False,
             "LONG": ref2va is not False,
             # LONG_FAST reuses FAST's Turbo LoRA on top of LONG's tail relay,
             # so it needs both conditions.
             "LONG_FAST": bool(turbo_ok and (ref2va is not False))}
    return {
        "comfyui": "OK" if comfy_reachable else ("UNKNOWN" if oi is None else "NG"),
        "gpu0": gpus[0] if len(gpus) > 0 else "NOT FOUND",
        "gpu1": gpus[1] if len(gpus) > 1 else "NOT FOUND",
        "h3_ref2va": "OK" if ref2va else ("UNKNOWN" if ref2va is None else "NG"),
        "turbo_lora": "OK" if turbo_ok else "NG",
        "models": models_present,
        "modes": modes,
        "experimental": {m["id"]: f"UNAVAILABLE: {m['reason']}"
                         for m in UI_MODES_EXPERIMENTAL},
    }


def drop_lora_for_base(graph: dict) -> dict:
    """Base Ref2VA: remove the LoRA node, rewire model consumers to UNET."""
    graph.pop("lora", None)
    for node_id in ("guider", "scheduler"):
        node = graph.get(node_id) or {}
        inputs = node.get("inputs") or {}
        if isinstance(inputs.get("model"), list) and inputs["model"][1] == 0:
            inputs["model"] = ["unet", 0]
    return graph


def apply_patches(graph: dict, patches: dict) -> dict:
    def _relink(node_id: str, new_src: str) -> None:
        inputs = graph[node_id].get("inputs") or {}
        if isinstance(inputs.get("model"), list) and inputs["model"][1] == 0:
            inputs["model"] = [new_src, 0]

    has_lora = "lora" in graph
    src = "lora" if has_lora else "unet"
    if "lowvram" in patches:
        graph["lowvram"] = {
            "class_type": "MiniMaxLowVRAMAttention",
            "inputs": {"model": [src, 0],
                       "head_chunks": int(patches["lowvram"]["head_chunks"])},
        }
        src = "lowvram"
    if "chunkff" in patches:
        graph["chunkff"] = {
            "class_type": "MiniMaxChunkFeedForward",
            "inputs": {"model": [src, 0],
                       "chunks": int(patches["chunkff"]["chunks"]),
                       "seq_threshold": int(patches["chunkff"]["seq_threshold"])},
        }
        src = "chunkff"
    if "sigshift" in patches:
        graph["sigshift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": [src, 0],
                       "shift_video": float(patches["sigshift"]["shift_video"]),
                       "shift_audio": float(patches["sigshift"]["shift_audio"])},
        }
        src = "sigshift"
    if "sage" in patches:
        graph["sage"] = {
            "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
            "inputs": {"model": [src, 0]},
        }
        src = "sage"
    if src != ("lora" if has_lora else "unet"):
        for node_id in ("guider", "scheduler"):
            _relink(node_id, src)
    return graph
