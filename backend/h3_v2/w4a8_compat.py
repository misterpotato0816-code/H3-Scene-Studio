# -*- coding: utf-8 -*-
"""Static LoRA↔checkpoint key-overlap check (header-only, no weights, no GPU).

Usage: <venv python> backend/h3_v2/w4a8_compat.py --ckpt FILE --lora FILE
Exit 0 + JSON. bindable = LoRA keys whose base weight exists in checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys


def header_keys(path: str) -> set[str]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    return {k for k in hdr if k != "__metadata__"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()
    ckpt = header_keys(args.ckpt)
    lora = header_keys(args.lora)
    lora_dm = sorted(k for k in lora if k.startswith("diffusion_model."))
    base = {re.sub(r"\.lora_(A|B)\.weight$", "", k) for k in lora_dm}
    # Checkpoints may store weights without the diffusion_model. prefix.
    base_np = {re.sub(r"^diffusion_model\.", "", b) for b in base}
    hit = sorted(b for b in base if b in ckpt or b + ".weight" in ckpt)
    hit_np = sorted(b for b in base_np if b in ckpt or b + ".weight" in ckpt)
    # W4A8 quantized names sometimes carry suffixes; try stem match too.
    stems = {b.rsplit(".", 1)[0] for b in base}
    ckpt_stems = {c.rsplit(".", 1)[0] for c in ckpt}
    stem_hit = sorted(s for s in stems if s in ckpt_stems)
    print(json.dumps({
        "ckpt_keys": len(ckpt),
        "lora_keys": len(lora),
        "lora_dm_keys": len(lora_dm),
        "lora_base": len(base),
        "exact_hit": len(hit),
        "noprefix_hit": len(hit_np),
        "stem_hit": len(stem_hit),
        "sample_ckpt": sorted(ckpt)[:5],
        "sample_lora_base": sorted(base)[:5],
        "sample_miss": sorted(set(base) - set(hit))[:10],
        "qkv_variants": sorted(c for c in ckpt if "qkv" in c)[:6],
        "mlp_variants": sorted(c for c in ckpt if "mlp.fc1" in c)[:6],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
