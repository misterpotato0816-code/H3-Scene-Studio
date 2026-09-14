# -*- coding: utf-8 -*-
"""H3 LONG_FAST pipeline nodes (isolated test copy).

Loaded ONLY by the Phase-2 server-B / probe servers via an extra
custom_nodes path. Never installed into production ComfyUI, never
whitelisted for the verified runners.
"""
from __future__ import annotations

import hashlib
import logging
import os

import folder_paths
import torch

from comfy_api.latest import io


def _describe_latent(samples: dict) -> str:
    out = []
    try:
        for key, val in samples.items():
            if isinstance(val, torch.Tensor):
                out.append(f"{key}: Tensor{tuple(val.shape)} {val.dtype} {val.device}")
            else:
                subs = []
                try:
                    for t in val:
                        subs.append(f"Tensor{tuple(t.shape)} {t.dtype}")
                except Exception as exc:                          # noqa: BLE001
                    subs.append(f"<uniterable {type(val).__name__}: {exc}>")
                out.append(f"{key}: {type(val).__name__}[{len(subs)}] "
                           + " | ".join(subs[:8]))
    except Exception as exc:                                      # noqa: BLE001
        out.append(f"<describe failed: {exc!r}>")
    return "; ".join(out)


class H3LatentProbe(io.ComfyNode):
    """Log the H3 AV latent structure. No outputs, no side effects."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentProbe",
            display_name="H3 Latent Probe (log structure)",
            category="H3V2/pipeline",
            description="Logs samples structure (shapes/dtypes) to the console. "
                        "Pass-through of the model input is NOT needed; "
                        "wire samples only.",
            is_experimental=True,
            is_output_node=True,
            inputs=[io.Latent.Input("samples")],
            outputs=[io.Latent.Output(display_name="samples")],
        )

    @classmethod
    def execute(cls, samples) -> io.NodeOutput:
        logging.info("[H3LatentProbe] %s", _describe_latent(samples))
        return io.NodeOutput({"samples": samples["samples"]})


class H3SaveLatentRaw(io.ComfyNode):
    """Save H3 AV latents (stock SaveLatent crashes on NestedTensor).
    Format: torch.save({"samples": ...}) under output/latents.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SaveLatentRaw",
            display_name="H3 Save Latent (raw)",
            category="H3V2/pipeline",
            description="Saves the raw H3 AV latent dict for the GPU1 VAE "
                        "pipeline. No tensor conversion is attempted.",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input("filename_prefix",
                                default="H3/latent", multiline=False),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, samples, filename_prefix) -> io.NodeOutput:
        out_dir = os.path.join(folder_paths.get_output_directory(), "latents")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(str(filename_prefix).strip() or "H3_latent")
        digest = hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        path = os.path.join(out_dir, f"{base}_{digest}.h3latent.pt")
        torch.save({"samples": samples["samples"]}, path)
        logging.info("[H3SaveLatentRaw] saved %s", path)
        return io.NodeOutput()


class H3LoadLatentRaw(io.ComfyNode):
    """Load a raw H3 AV latent dict saved by H3SaveLatentRaw."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadLatentRaw",
            display_name="H3 Load Latent (raw)",
            category="H3V2/pipeline",
            description="Loads a .h3latent.pt file back to LATENT.",
            is_experimental=True,
            inputs=[
                io.String.Input("filepath", multiline=False,
                                tooltip="Absolute path to a .h3latent.pt file."),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
            ],
        )

    @classmethod
    def execute(cls, filepath) -> io.NodeOutput:
        path = str(filepath).strip()
        if not os.path.isfile(path):
            raise RuntimeError(f"H3LoadLatentRaw: file not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        samples = data["samples"] if isinstance(data, dict) else data
        logging.info("[H3LoadLatentRaw] loaded %s", path)
        return io.NodeOutput({"samples": samples})


class H3LatentTailSlice(io.ComfyNode):
    """Slice the video tail (+ leading context) out of an H3 AV latent pack.

    Video layout is [B, C=24, T, H, W]: time is dim 2 (dim 1 is channels).
    Part 1 (audio) passes through untouched. A 1-frame time slice hits the
    VAE's single-frame (_adaptive_decode) fast path.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LatentTailSlice",
            display_name="H3 Latent Tail Slice",
            category="H3V2/pipeline",
            description="Keeps the last `tail_latent_frames` video latent "
                        "frames plus `context_latent_frames` leading frames "
                        "for the temporal VAE. Audio passes through.",
            is_experimental=True,
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("tail_latent_frames", default=1, min=1, max=512,
                             step=1),
                io.Int.Input("context_latent_frames", default=0, min=0,
                             max=512, step=1),
            ],
            outputs=[
                io.Latent.Output(display_name="samples"),
            ],
        )

    @classmethod
    def execute(cls, samples, tail_latent_frames,
                context_latent_frames) -> io.NodeOutput:
        from comfy.nested_tensor import NestedTensor
        parts = list(samples["samples"].unbind())
        if len(parts) < 1:
            raise RuntimeError("H3LatentTailSlice: empty latent pack")
        video = parts[0]
        total = int(video.shape[2])
        want = int(tail_latent_frames) + int(context_latent_frames)
        start = max(0, total - want)
        video = video.narrow(2, start, total - start)
        logging.info("[H3LatentTailSlice] video time %d -> %d (audio passthrough)",
                     int(parts[0].shape[2]), int(video.shape[2]))
        return io.NodeOutput({"samples": NestedTensor([video] + parts[1:])})
