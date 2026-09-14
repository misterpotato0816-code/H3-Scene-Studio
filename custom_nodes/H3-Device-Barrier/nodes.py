# -*- coding: utf-8 -*-
"""H3 CUDA phase barrier.

Why this exists
---------------
The H3 workflow runs a llama.cpp VLM (Gemma-4) before the H3 phase. llama.cpp
selects its own CUDA device, and the process-wide *current device* can be left
pointing at that GPU after the model is unloaded.

ComfyUI decides a text encoder's base load/offload device at CLIPLoader time
from ``torch.cuda.current_device()`` (``model_management.text_encoder_device()``).
``SelectCLIPDevice`` then retargets that patcher relative to the remembered base
devices. If the base was captured while the current device was already GPU1,
the retarget is not the same operation the measured H3_HETERO_V1 build performed.

These nodes make the boundary explicit:

* ``H3CudaProbe``          - logs ``torch.cuda.current_device()`` at a point in the graph.
* ``H3CudaPhaseRestore``   - restores the current device to a chosen GPU and verifies it.
* ``H3GatedCLIPLoader``    - a CLIPLoader that cannot execute before a barrier input
                             is ready, so the base device capture happens after the restore.
* ``H3ClipDeviceReport``   - logs a CLIP patcher's load/offload device.

Nothing here changes ComfyUI or any third-party node. It only observes, sets the
current device, and delegates the actual loading to ComfyUI's own CLIPLoader.
"""
from __future__ import annotations

import logging

import torch

LOG = logging.getLogger(__name__)
TAG = "[H3Barrier]"


class _Any(str):
    """Type sentinel that matches any ComfyUI socket type."""

    def __ne__(self, other):
        return False

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash("*")


ANY = _Any("*")


def _cuda_state() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    idx = torch.cuda.current_device()
    try:
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = "?"
    return f"cuda:{idx} ({name})"


def _log(label: str, extra: str = "") -> str:
    line = f"{TAG} {label}: current_device={_cuda_state()}"
    if extra:
        line += f" | {extra}"
    LOG.info(line)
    print(line, flush=True)
    return line


def _patcher_devices(clip) -> str:
    p = getattr(clip, "patcher", None)
    if p is None:
        return "no patcher"
    return (f"load_device={getattr(p, 'load_device', None)} "
            f"offload_device={getattr(p, 'offload_device', None)}")


class H3CudaProbe:
    """Log torch.cuda.current_device() at this point in the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY, {}),
                             "label": ("STRING", {"default": "probe"})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "run"
    CATEGORY = "H3/device"
    DESCRIPTION = "Passthrough. Logs torch.cuda.current_device() with a label."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def run(self, anything, label):
        _log(label)
        return (anything,)


class H3CudaPhaseRestore:
    """Restore the process-wide CUDA current device before the H3 phase.

    Fails loudly rather than continuing on the wrong device.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY, {}),
                             "target_gpu": ("INT", {"default": 0, "min": 0, "max": 15}),
                             "label": ("STRING", {"default": "restore"}),
                             "synchronize": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "run"
    CATEGORY = "H3/device"
    DESCRIPTION = ("Sets torch.cuda.current_device() back to target_gpu and verifies it. "
                   "Raises if the device cannot be restored.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def run(self, anything, target_gpu, label, synchronize):
        if not torch.cuda.is_available():
            _log(f"{label}/pre", "cuda unavailable - nothing to restore")
            return (anything,)

        count = torch.cuda.device_count()
        if target_gpu >= count:
            raise RuntimeError(
                f"{TAG} target_gpu={target_gpu} but only {count} CUDA device(s) are visible. "
                f"Check the H3 GPU settings (設定 > GPU): the launch args must keep every required GPU visible.")

        before = torch.cuda.current_device()
        _log(f"{label}/pre")

        if before != target_gpu:
            if synchronize:
                # Let whatever the previous phase queued finish on its own device
                # before the current device moves. No cache is freed here.
                torch.cuda.synchronize(torch.device(f"cuda:{before}"))
            torch.cuda.set_device(target_gpu)

        after = torch.cuda.current_device()
        _log(f"{label}/post", f"changed={before != after} before=cuda:{before}")

        if after != target_gpu:
            raise RuntimeError(
                f"{TAG} failed to restore CUDA current device: wanted cuda:{target_gpu}, "
                f"still on cuda:{after}. Stopping before the H3 phase.")
        return (anything,)


class H3GatedCLIPLoader:
    """CLIPLoader that waits for a barrier input before it executes.

    ComfyUI captures a text encoder's base load/offload device when the loader
    runs. A plain CLIPLoader has no inputs, so nothing orders it against the VLM
    phase. The ``barrier`` input creates that dependency explicitly.

    Loading itself is delegated to ComfyUI's own CLIPLoader - no logic is copied.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        from nodes import CLIPLoader
        types = CLIPLoader.INPUT_TYPES()
        return {"required": {"barrier": (ANY, {}),
                             "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                             "type": types["required"]["type"],
                             "device": (["default", "cpu"], {"advanced": True})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "run"
    CATEGORY = "H3/device"
    DESCRIPTION = ("ComfyUI's CLIPLoader with an extra barrier input so the base device "
                   "capture is ordered after the CUDA phase restore.")

    def run(self, barrier, clip_name, type, device="default"):
        from nodes import CLIPLoader
        _log("cliploader/pre", f"clip_name={clip_name} device={device}")
        clip = CLIPLoader().load_clip(clip_name=clip_name, type=type, device=device)[0]
        _log("cliploader/post", _patcher_devices(clip))
        return (clip,)


class H3ClipDeviceReport:
    """Log a CLIP patcher's load/offload device (passthrough)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",),
                             "label": ("STRING", {"default": "clip"})}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "run"
    CATEGORY = "H3/device"
    DESCRIPTION = "Passthrough. Logs the CLIP patcher's load/offload device."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def run(self, clip, label):
        _log(label, _patcher_devices(clip))
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "H3CudaProbe": H3CudaProbe,
    "H3CudaPhaseRestore": H3CudaPhaseRestore,
    "H3GatedCLIPLoader": H3GatedCLIPLoader,
    "H3ClipDeviceReport": H3ClipDeviceReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CudaProbe": "H3 CUDA Probe",
    "H3CudaPhaseRestore": "H3 CUDA Phase Restore",
    "H3GatedCLIPLoader": "H3 Gated CLIP Loader",
    "H3ClipDeviceReport": "H3 CLIP Device Report",
}
