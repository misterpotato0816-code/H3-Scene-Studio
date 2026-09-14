# -*- coding: utf-8 -*-
"""Coarse identity-drift record (STABILITY phase: measure only, never gate).

face-embedding models are a new heavy dependency, so this phase records a
lightweight perceptual-hash distance (PIL only, already a project dependency)
between the first reference image and each generated clip's first frame.
Interpretation: 0 = near-identical, larger = more drift. This is NOT a face
identity metric; the automatic face metric waits for the QUALITY phase.
Reference conditioning itself is untouched.
"""
from __future__ import annotations


def dhash_bits(path_a: str, path_b: str, *, size: int = 16) -> int | None:
    """Hamming distance between dHash values. None when not computable."""
    try:
        from PIL import Image as _Image
    except Exception:                                        # noqa: BLE001
        return None
    try:
        hashes = []
        for path in (path_a, path_b):
            with _Image.open(path) as im:
                im = im.convert("L").resize((size + 1, size),
                                            _Image.BILINEAR)
                pixels = list(im.getdata())
                bits = 0
                for row in range(size):
                    for col in range(size):
                        left = pixels[row * (size + 1) + col]
                        right = pixels[row * (size + 1) + col + 1]
                        bits = (bits << 1) | (1 if left > right else 0)
                hashes.append(bits)
        return bin(hashes[0] ^ hashes[1]).count("1")
    except Exception:                                        # noqa: BLE001
        return None
