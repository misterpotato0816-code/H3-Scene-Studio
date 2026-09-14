# -*- coding: utf-8 -*-
"""Preset engine. Pure data + gating; no I/O, no GPU.

Only LEGACY_V1 is verified (byte-parity with H3_HETERO_V1_MANIFEST.json).
Every other preset carries status + reason and is BLOCKED until a generation
test flips it to verified with evidence. Guessing values from memory is not
how a preset becomes verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    id: str
    backend: str          # "ref2va_base" | "ref2va_turbo" | "legacy_v1_chain"
    experimental: bool
    status: str           # "verified" | "unverified" | "unavailable"
    reason: str           # human-readable: why blocked, or verification source
    model: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    clip: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    lora: str | None = "minimax_h3_turbo_v4_step600_ema.safetensors"
    lora_strength: float = 1.0
    min_bindable_keys: int = 0  # Turbo proof: generation must log >= this
    width: int = 576
    height: int = 1024
    frames: int = 124
    steps: int = 8
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    ref_image_size: str = "match"
    shift_video: float | None = None  # None = node default (12.0)
    shift_audio: float | None = None  # None = node default (3.0)
    lowvram_heads: int = 0      # 0 = off; >0 = MiniMaxLowVRAMAttention head_chunks
    chunkff_chunks: int = 0     # 0 = off; >0 = MiniMaxChunkFeedForward chunks
    chunkff_threshold: int = 4096
    sage: bool = False          # H3-dedicated Sage attention patch (KJNodes)


PRESETS: dict[str, Preset] = {
    # Verified 2026-08-14: 217.61 s. The RAW turbo LoRA binds 0 keys here, so
    # this is effectively Base Ref2VA 8-step - and that no-op state is FROZEN.
    "LEGACY_V1": Preset(
        id="LEGACY_V1", backend="legacy_v1_chain", experimental=False,
        status="verified",
        reason="H3_HETERO_V1_MANIFEST.json v1_match=217.61s; RAW turbo LoRA 518-key no-op is the measured state",
    ),
    # Candidate only. ckpt850 binds (diffusion_model. prefix) but its metadata
    # warns AdaLN adapters were removed -> 4-step distillation may be degraded.
    # Verified recipe (NOT yet verified on THIS machine):
    # upstream ModelTC/Minimax-H3-Turbo example_workflows/
    # video_minimax_h3_ref2v_lightx2v_turbo.json =
    #   LoraLoaderModelOnly(ref2v_turbo_4step_v0.1_comfyui_bf16, 1.0)
    #   -> MiniMaxH3SigmaShift(12, 3) -> euler + simple 4 steps.
    # Header proof: 624/624 tensors carry the diffusion_model. prefix
    # (vs 0/518 for the RAW v4 file), so binding is expected.
    "FAST": Preset(
        id="FAST", backend="ref2va_turbo", experimental=False,
        status="verified",
        reason=("A/B 2026-09-06: 135.6s vs LEGACY 230.3s (1.70x), 624/624 keys "
                "bound, 0 runtime warnings, AV streams valid, frame compare "
                "benchmarks/compare_FAST_vs_LEGACY.jpg coherent "
                "(identity kept, no smear/breakage)"),
        lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        min_bindable_keys=600,
        steps=4, sampler="euler", scheduler="simple",
        shift_video=12.0, shift_audio=3.0,
    ),
    "BALANCED": Preset(
        id="BALANCED", backend="ref2va_base", experimental=False,
        status="verified",
        reason=("A/B 2026-09-06: 260.7s total / 209.1s sampling, AV valid, "
                "frame compare benchmarks/compare_PHASE3_mid.jpg coherent"),
        lora=None, steps=10,
    ),
    "QUALITY": Preset(
        id="QUALITY", backend="ref2va_base", experimental=False,
        status="verified",
        reason=("A/B 2026-09-06: 403.5s total / 351.5s sampling, AV valid, "
                "frame compare benchmarks/compare_PHASE3_mid.jpg coherent"),
        lora=None, steps=20, sampler="res_multistep", scheduler="simple",
    ),
    "LOW_VRAM": Preset(
        id="LOW_VRAM", backend="ref2va_base", experimental=False,
        status="verified",
        reason=("A/B 2026-09-06: 222.4s total / 169.5s sampling, AV valid, "
                "frame compare benchmarks/compare_LOWVRAM_vs_LEGACY.jpg "
                "identical quality. VRAM: no reduction at 576x1024/124f "
                "(11377 vs 10578 MB peak, run variance); benefit case is "
                "larger frame counts. Never auto-forced."),
        lora=None, steps=8, lowvram_heads=4, chunkff_chunks=2,
    ),
    "CONTROL": Preset(
        id="CONTROL", backend="ref2va_base", experimental=False,
        status="unavailable",
        reason="no H3 FunControl node in ComfyUI 0.34.5 / KJNodes (Wan nodes are not H3-compatible)",
    ),
    "PDD": Preset(
        id="PDD", backend="ref2va_base", experimental=True,
        status="unavailable", reason="no PDD node/scheduler in this ComfyUI generation",
    ),
    "LONG": Preset(
        id="LONG", backend="ref2va_base", experimental=True,
        status="verified",
        reason=("2-clip tail-relay chain 2026-09-06: A 226.7s + B 417.3s, "
                "10.33s concat output, joint compare "
                "benchmarks/compare_LONG_joint.jpg coherent, no drift break. "
                "Opt-in via feature_flags.long_video."),
        lora=None, steps=8,
    ),
    "ENDLESS": Preset(
        id="ENDLESS", backend="ref2va_base", experimental=True,
        status="unavailable", reason="Phase 8: not designed yet",
        lora=None, steps=8,
    ),
    # V3 LONG_FAST: Turbo4 + Euler/simple + SigmaShift 12/3 + H3 Sage, chained
    # by tail relay with Voice Master audio. Starts UNVERIFIED; the night loop
    # flips it only with 2x wall-clock evidence + quality proof (see
    # docs/V3_LONG_FAST_SPEC.md). Never auto-substituted for anything.
    "LONG_FAST": Preset(
        id="LONG_FAST", backend="ref2va_turbo", experimental=True,
        status="unverified",
        reason=("V3 speed loop: production 2x362f @480x864, 0f+lastframe+VM, "
                "serial wall ~469s (pipe362 rejected: GPU1 decode 3x slower, "
                "net loss). Stays unverified (time target unmet + "
                "human_review_required on motion/audio-content). "
                "See docs/V3_LONG_FAST.md."),
        lora="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        min_bindable_keys=600,
        steps=4, sampler="euler", scheduler="simple",
        shift_video=12.0, shift_audio=3.0, sage=True,
        width=480, height=864,
    ),
}

UI_MODES = ("FAST", "BALANCED", "QUALITY", "CONTROL", "LOW_VRAM", "EXPERIMENTAL")


class PresetError(RuntimeError):
    pass


def get_preset(preset_id: str, flags=None) -> Preset:
    """Return a runnable preset or raise PresetError with an explicit code.

    Never silently substitutes: TURBO_UNAVAILABLE / PRESET_UNVERIFIED /
    PRESET_UNAVAILABLE / EXPERIMENTAL_DISABLED tell the caller exactly why.
    """
    try:
        preset = PRESETS[str(preset_id)]
    except KeyError:
        raise PresetError(f"PRESET_UNKNOWN: {preset_id!r}") from None
    if preset.status == "unavailable":
        code = "PDD_UNAVAILABLE" if preset.id == "PDD" else f"{preset.id}_UNAVAILABLE"
        raise PresetError(f"{code}: {preset.reason}")
    if preset.experimental and flags is not None:
        from .feature_flags import EXPERIMENTAL_FLAGS  # local import: no cycle
        gate = {"PDD": "pdd", "LONG": "long_video", "LONG_FAST": "long_fast",
                "ENDLESS": "endless"}.get(preset.id)
        if gate and not getattr(flags, gate, False):
            raise PresetError(f"EXPERIMENTAL_DISABLED ({gate}=false): {preset.id}")
    if preset.status == "unverified":
        raise PresetError(f"PRESET_UNVERIFIED: {preset.id}: {preset.reason}")
    return preset


def verified_presets() -> list[str]:
    return [pid for pid, p in PRESETS.items() if p.status == "verified"]
