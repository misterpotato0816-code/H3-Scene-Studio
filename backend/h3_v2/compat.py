# -*- coding: utf-8 -*-
"""ComfyUI compatibility layer. Pure probe + thin fetch helper.

Startup rule: read the LIVE /object_info, never assume node existence from
memory. A missing optional feature disables its UI entry; the app itself
must stay bootable.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field


H3_LIMITS_FALLBACK = {"ref_images": 4, "ref_videos": 1, "ref_audios": 1}
# v1 app caps references at 4 images; the node allows up to 9/3/3/3.
V2_REF_IMAGE_MAX = 9


@dataclass(frozen=True)
class Capabilities:
    comfyui_version: str = "unknown"
    ref2va: bool = False
    turbo_loader: bool = False       # H3-specific loader node (LoRA path used instead)
    lowvram_attention: bool = False
    chunk_ffn: bool = False
    funcontrol_h3: bool = False
    pdd: bool = False
    select_clip_options: tuple = ()
    h3_limits: dict = field(default_factory=lambda: dict(H3_LIMITS_FALLBACK))

    def disabled_ui_modes(self) -> list[str]:
        """UI modes that must render disabled (with reasons logged by caller)."""
        out = []
        if not self.funcontrol_h3:
            out.append("CONTROL")
        if not self.pdd:
            out.append("PDD")
        return out

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in
             ("comfyui_version", "ref2va", "turbo_loader", "lowvram_attention",
              "chunk_ffn", "funcontrol_h3", "pdd", "h3_limits")}
        d["select_clip_options"] = list(self.select_clip_options)
        d["disabled_ui_modes"] = self.disabled_ui_modes()
        return d


def _has(object_info: dict, name: str) -> bool:
    return isinstance(object_info, dict) and name in object_info


def probe_capabilities(object_info: dict) -> Capabilities:
    """Pure: object_info dict -> Capabilities. Testable without ComfyUI."""
    object_info = object_info or {}
    node = object_info.get("MiniMaxH3ReferenceToVideo") or {}
    inputs = (node.get("input") or {})
    optional = inputs.get("optional") or {}
    limits = dict(H3_LIMITS_FALLBACK)
    for key, field_name in (("ref_images", "ref_images"),
                            ("ref_videos", "ref_videos"),
                            ("ref_video_audios", "ref_audios")):
        spec = optional.get(key) or {}
        maxn = (spec.get("max") if isinstance(spec, dict) else None)
        if isinstance(maxn, int) and maxn > 0:
            limits[field_name] = maxn
    select = object_info.get("SelectCLIPDevice") or {}
    select_inputs = (select.get("input") or {})
    options: tuple = ()
    for section in ("required", "optional"):
        device = ((select_inputs.get(section) or {}).get("device") or [])
        if isinstance(device, list) and len(device) > 1 and isinstance(device[0], list):
            options = tuple(device[0])
    return Capabilities(
        ref2va=_has(object_info, "MiniMaxH3ReferenceToVideo"),
        turbo_loader=any(_has(object_info, n) for n in
                         ("MiniMaxH3TurboLoader", "H3TurboLoader")),
        lowvram_attention=_has(object_info, "MiniMaxLowVRAMAttention"),
        chunk_ffn=_has(object_info, "MiniMaxChunkFeedForward"),
        funcontrol_h3=any(_has(object_info, n) for n in
                          ("MiniMaxH3FunControl", "H3FunControl",
                           "MiniMaxFunControlToVideo")),
        pdd=any("pdd" in str(k).lower() and "h3" in str(k).lower()
                for k in object_info),
        select_clip_options=options,
        h3_limits=limits,
    )


def fetch_object_info(base_url: str, timeout: int = 60) -> dict:
    """Read-only GET. Raises on failure so the caller can degrade gracefully."""
    url = base_url.rstrip("/") + "/object_info"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
