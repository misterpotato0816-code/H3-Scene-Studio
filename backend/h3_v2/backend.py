# -*- coding: utf-8 -*-
"""v2 backend selector + graph builder. Thin layer over the v1 chain.

LEGACY_V1 builds byte-identical graphs via app.h3app.graphs (proven by
parity test). Other presets reuse the same builder with explicit overrides
and post-processed model-patch chains. No duplicated graph generation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from .feature_flags import FeatureFlags
from .presets import PRESETS, Preset, PresetError, get_preset

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent  # E:\AI-Projects\H3
APP_ROOT = PROJECT_ROOT / "app"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from h3app import graphs as v1_graphs  # noqa: E402  (reuse, never mutate v1)


class BackendError(RuntimeError):
    pass


def select_backend(overlay: dict | None) -> str:
    """'legacy_v1' (default) or 'h3_v2'. Unknown names fail loudly."""
    name = ((overlay or {}).get("backend") or "legacy_v1")
    if name not in ("legacy_v1", "h3_v2"):
        raise BackendError(f"BACKEND_UNKNOWN: {name!r}")
    return name


def preset_settings(preset: Preset, frames: int | None = None) -> dict:
    return {
        "width": preset.width, "height": preset.height,
        "frames": int(frames) if frames else preset.frames,
        "seed": 123456789, "sampler": preset.sampler, "steps": preset.steps,
        "scheduler": preset.scheduler, "ref_image_size": preset.ref_image_size,
    }


def preset_models(preset: Preset) -> dict:
    return {
        "clip": preset.clip, "unet": preset.model, "lora": preset.lora,
        "lora_strength": preset.lora_strength,
        "vae_video": "minimax_h3_video_vae_fp16.safetensors",
        "vae_audio": "minimax_h3_audio_vae_fp32.safetensors",
    }


def _relink_model_input(node: dict, new_src: str) -> None:
    inputs = node.get("inputs") or {}
    if isinstance(inputs.get("model"), list) and inputs["model"][1] == 0:
        inputs["model"] = [new_src, 0]


def build(preset_id: str, images: list[str], en_prompt: str,
          ref_longest: list[int], output_prefix: str,
          continuation: dict | None = None, concat_previous: bool = True,
          flags: FeatureFlags | None = None,
          allow_unverified: bool = False,
          frames: int | None = None,
          preset_obj: Preset | None = None,
          voice_master: str | None = None) -> dict:
    """Build the API graph for a preset. Raises PresetError, never degrades.

    allow_unverified=True is the TEST HARNESS path only (benchmark runs):
    it still refuses unavailable/experimental-disabled presets, but builds
    unverified ones so they can be measured. Every harness run must log that
    the graph was unverified at build time.
    preset_obj: harness-only field overrides (sage/clip/steps probes). Only
    honored with allow_unverified=True, and the base id must exist and not
    be unavailable. Production callers always pass None.
    """
    flags = flags or FeatureFlags()
    if preset_obj is not None and not allow_unverified:
        raise PresetError("PRESET_OVERRIDE_FORBIDDEN: overrides need the harness path")
    if allow_unverified:
        preset = preset_obj if preset_obj is not None else PRESETS.get(str(preset_id))
        if preset is None or preset.id not in PRESETS:
            raise PresetError(f"PRESET_UNKNOWN: {preset_id!r}")
        if preset.status == "unavailable":
            code = "PDD_UNAVAILABLE" if preset.id == "PDD" else f"{preset.id}_UNAVAILABLE"
            raise PresetError(f"{code}: {preset.reason}")
    else:
        preset = get_preset(preset_id, flags)
    if len(images) > 9:
        raise BackendError("TOO_MANY_REFS: H3 allows at most 9 ref_images")

    graph = v1_graphs.build_generate_graph(
        images, en_prompt, preset_settings(preset, frames), preset_models(preset),
        ref_longest, output_prefix, continuation,
        concat_previous=concat_previous, voice_master=voice_master)

    src = "lora"
    if preset.lora is None:
        # Base Ref2VA: drop the LoRA node, rewire model consumers to UNET.
        graph.pop("lora", None)
        for node_id in ("guider", "scheduler"):
            _relink_model_input(graph[node_id], "unet")
        src = "unet"

    if preset.lowvram_heads > 0:
        if not flags.low_vram_attention:
            raise BackendError("LOWVRAM_DISABLED: feature_flags.low_vram_attention=false")
        graph["lowvram"] = {
            "class_type": "MiniMaxLowVRAMAttention",
            "inputs": {"model": [src, 0], "head_chunks": int(preset.lowvram_heads)},
        }
        src = "lowvram"
    if preset.chunkff_chunks > 0:
        if not flags.chunk_ffn:
            raise BackendError("CHUNKFFN_DISABLED: feature_flags.chunk_ffn=false")
        graph["chunkff"] = {
            "class_type": "MiniMaxChunkFeedForward",
            "inputs": {"model": [src, 0], "chunks": int(preset.chunkff_chunks),
                       "seq_threshold": int(preset.chunkff_threshold)},
        }
        src = "chunkff"
    if preset.shift_video is not None or preset.shift_audio is not None:
        graph["sigshift"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": [src, 0],
                       "shift_video": float(preset.shift_video or 12.0),
                       "shift_audio": float(preset.shift_audio or 3.0)},
        }
        src = "sigshift"
    if preset.sage:
        # H3-dedicated Sage attention, wired into the execution graph.
        # May raise at runtime if sageattention is too old: surfaced, never
        # silently skipped (the harness records the failure loudly).
        graph["sage"] = {
            "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
            "inputs": {"model": [src, 0]},
        }
        src = "sage"
    if src not in ("lora", "unet"):
        for node_id in ("guider", "scheduler"):
            _relink_model_input(graph[node_id], src)
    return graph
