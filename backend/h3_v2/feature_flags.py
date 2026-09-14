# -*- coding: utf-8 -*-
"""Feature flags. Pure: no I/O, no clock.

Experimental flags default OFF. Stable generation must never die because an
experimental module raised: callers gate on these flags BEFORE touching
experimental code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureFlags:
    turbo: bool = True              # FAST preset family (needs bound LoRA proof)
    funcontrol: bool = False        # no H3 FunControl node in ComfyUI 0.34.5
    low_vram_attention: bool = True  # node exists; opt-in per preset, never forced
    chunk_ffn: bool = True          # node exists; opt-in per preset, never forced
    pdd: bool = False               # experimental, default off
    long_video: bool = False        # experimental, default off
    long_fast: bool = False         # V3 LONG_FAST night loop, default off
    endless: bool = False           # experimental, default off
    benchmark: bool = True
    ab_compare: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "FeatureFlags":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: bool(v) for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


EXPERIMENTAL_FLAGS = ("pdd", "long_video", "long_fast", "endless")


def experimental_enabled(flags: FeatureFlags) -> list[str]:
    """Names of experimental flags currently on (for logging/gating)."""
    return [f for f in EXPERIMENTAL_FLAGS if getattr(flags, f)]
