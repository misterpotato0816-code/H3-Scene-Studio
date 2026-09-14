# -*- coding: utf-8 -*-
"""Custom-node loader for the H3V2 pipeline R&D package."""
from .h3_pipeline_nodes import (H3LatentProbe, H3LatentTailSlice,
                                 H3LoadLatentRaw, H3SaveLatentRaw)

NODE_CLASS_MAPPINGS = {
    "H3LatentProbe": H3LatentProbe,
    "H3SaveLatentRaw": H3SaveLatentRaw,
    "H3LoadLatentRaw": H3LoadLatentRaw,
    "H3LatentTailSlice": H3LatentTailSlice,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
