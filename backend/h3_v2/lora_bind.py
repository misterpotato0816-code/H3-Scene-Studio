# -*- coding: utf-8 -*-
"""LoRA bind verification. Pure: safetensors key lists -> bind verdict.

v1 lesson: a LoRA filename in the graph proves nothing. The RAW turbo LoRA
(518 keys, no 'diffusion_model.' prefix) binds 0 keys on the ComfyUI pruned
model. FAST must prove binding with counts, or report TURBO_UNAVAILABLE.
"""
from __future__ import annotations

PRUNED_PREFIX = "diffusion_model."


def count_bindable_keys(tensor_names: list[str], prefix: str = PRUNED_PREFIX) -> int:
    return sum(1 for k in tensor_names if str(k).startswith(prefix))


def classify_lora(tensor_names: list[str], min_bindable_keys: int = 1) -> dict:
    """Return a verdict dict. `bound=False` means TURBO_UNAVAILABLE upstream."""
    total = len(tensor_names)
    bindable = count_bindable_keys(tensor_names)
    bound = bindable >= max(1, int(min_bindable_keys))
    return {
        "total_keys": total,
        "bindable_keys": bindable,
        "unbindable_keys": total - bindable,
        "bound": bound,
        "code": "TURBO_OK" if bound else "TURBO_UNAVAILABLE",
    }


def read_tensor_names(path: str) -> list[str]:
    """Read ONLY the safetensors header (no weight data loaded)."""
    import json
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]
