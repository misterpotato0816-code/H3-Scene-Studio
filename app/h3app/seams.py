# -*- coding: utf-8 -*-
"""Story clip boundary analysis and natural-seam correction (WP-C).

The H3 model conditions clip N+1 on clip N's LAST `tail_frames` frames (tail
relay - see story_runner._continuation) so it does NOT emit an exact temporal
continuation of clip N. The measured defect (see
docs/DESIGN_2026-09-13_ADDITIONS.md, WP-C) is a luma/motion jump at the
boundary plus an almost-frozen head on the new clip and dead-air audio from
naive concatenation.

This module implements the analysis half of the fix: decode just the frames
around a boundary, score candidate cut points inside the new clip's head, and
compute the (optional, bounded) colour ramp / interpolation / audio-trim
corrections. `merge.merge_story` uses these to assemble the final video.

Dependencies are the already-required ffmpeg/ffprobe binaries plus numpy and
cv2 (both already vendored in the ComfyUI venv) - no new dependency per the
common rules in the design doc.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import merge as merge_mod
from .errors import PipelineError

try:
    import cv2                                      # noqa: PLC0415 - optional
except Exception:                                    # pragma: no cover
    cv2 = None


class SeamError(PipelineError):
    """A seam-improvement step could not be performed honestly (e.g. RIFE was
    requested but is not installed). Never caught to silently fall back to a
    different method - the caller must see the Japanese reason."""

    def __init__(self, message: str, **kw):
        kw.setdefault("kind", "missing_model")
        kw.setdefault("stage", "つなぎ目の調整")
        super().__init__(message, **kw)


# Scoring weights for choose_cut(). Sum to 1.0; SSIM dominates because a
# structural mismatch (a different pose/frame entirely) is the worst kind of
# seam, luma is second because it is what a viewer notices first (the
# measured defect: 113.1 -> 104.1 across the cut).
W_SSIM = 0.40
W_LUMA = 0.25
W_COLOR = 0.15
W_FACE = 0.10
W_MOTION = 0.10
# Penalty per remaining near-frozen head frame left AFTER the cut (see
# choose_cut `static_frames`): cutting into the frozen run keeps the "stops
# for a moment" feel, but the LAST frozen frame is usually the closest match
# to the previous clip (the model holds the relayed pose there), so one
# leftover frame is free and the rest cost 0.30 / max_frames each.
W_STATIC = 0.30

# color_match_params() thresholds: below these the difference is within what
# two independently-generated clips normally show and correcting it would
# just add its own visible ramp.
COLOR_LUMA_THRESHOLD = 6.0
COLOR_DIFF_THRESHOLD = 4.0

# Colour ramp length (frames). A clip that comes out globally darker (the
# measured case: ~10 luma lower for its WHOLE length) must not be pulled
# back to the previous clip's level and then released within a few frames -
# that reads as a brightness wobble right after the seam. Instead the
# correction decays over 1-3 s so the eye never sees a step in either
# direction (0.1-0.4 luma per frame).
RAMP_MIN_FRAMES = 24
RAMP_MAX_FRAMES = 72
RAMP_FRAMES_PER_UNIT = 6.0

# Upper bound on how much of the new clip's head may be dropped in total
# (static frames + cut offset). 18 frames = 0.75 s at 24 fps.
HEAD_TRIM_MAX_FRAMES = 18

# needs_review() thresholds (post-correction).
REVIEW_LUMA_THRESHOLD = 25.0
REVIEW_SSIM_THRESHOLD = 0.5
REVIEW_STATIC_THRESHOLD = 24

_FFPROBE: str | None = None


def find_ffprobe() -> str:
    global _FFPROBE
    if _FFPROBE:
        return _FFPROBE
    found = shutil.which("ffprobe")
    if found:
        _FFPROBE = found
        return _FFPROBE
    # ffmpeg and ffprobe ship side by side everywhere H3 supports (PATH or
    # imageio_ffmpeg's own bin dir uses the same naming convention).
    exe = merge_mod.find_ffmpeg()
    candidate = Path(exe).with_name(Path(exe).name.replace("ffmpeg", "ffprobe"))
    if candidate.is_file():
        _FFPROBE = str(candidate)
        return _FFPROBE
    raise PipelineError(
        "つなぎ目の解析に使う ffprobe が見つかりませんでした。"
        "ffmpeg と同じ場所に ffprobe を導入してください。",
        kind="missing_model", stage="つなぎ目の調整")


async def _run_capture(args: list[str], *, timeout: float = 120) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        raise PipelineError("つなぎ目の解析処理が時間内に終わりませんでした。",
                            kind="timeout", stage="つなぎ目の調整")
    return out or b"", err or b""


async def probe_video(path) -> dict:
    """{"width","height","fps","frames"} via ffprobe. Best effort: missing
    fields default to 0."""
    exe = find_ffprobe()
    args = [exe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
            "-count_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
    out, _ = await _run_capture(args, timeout=60)
    lines = out.decode("utf-8", errors="replace").strip().splitlines()
    result = {"width": 0, "height": 0, "fps": 0.0, "frames": 0}
    keys = ("width", "height", "r_frame_rate", "frames")
    for key, value in zip(keys, lines):
        if key == "r_frame_rate":
            num, _, den = value.partition("/")
            try:
                result["fps"] = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                result["fps"] = 0.0
        else:
            try:
                result[key] = int(value)
            except ValueError:
                pass
    return result


async def probe_size(path) -> tuple[int, int]:
    info = await probe_video(path)
    return int(info["width"]), int(info["height"])


# --------------------------------------------------------------- decoding ---
async def decode_frames(path, start: int, count: int, *,
                        width: int | None = None, height: int | None = None) -> np.ndarray:
    """Decode `count` RGB24 frames starting at frame `start`.

    Uses an ffmpeg rawvideo pipe with a `select` filter so only the requested
    range is ever decoded (no full-clip decode for a boundary check)."""
    if count <= 0:
        return np.zeros((0, height or 0, width or 0, 3), dtype=np.uint8)
    exe = merge_mod.find_ffmpeg()
    if width is None or height is None:
        width, height = await probe_size(path)
    if not width or not height:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    end = start + count - 1
    vf = f"select='between(n\\,{start}\\,{end})'"
    args = [exe, "-hide_banner", "-nostdin", "-v", "error",
            "-i", str(path), "-vf", vf, "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out, _ = await _run_capture(args, timeout=120)
    frame_bytes = width * height * 3
    n = len(out) // frame_bytes if frame_bytes else 0
    if n <= 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    arr = np.frombuffer(out[: n * frame_bytes], dtype=np.uint8)
    return arr.reshape((n, height, width, 3)).copy()


async def find_speech_onset_frame(path, *, fps: float = 24.0,
                                  max_seconds: float = 1.0) -> int | None:
    """Frame index (relative to clip start) where dialogue audio begins, or
    None if no clear onset is found in the first `max_seconds`. Used to
    exclude cut candidates AFTER the new clip has already started talking."""
    exe = merge_mod.find_ffmpeg()
    sr = 16000
    args = [exe, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
            "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-"]
    out, _ = await _run_capture(args, timeout=60)
    samples = np.frombuffer(out, dtype=np.float32)
    if samples.size == 0:
        return None
    win = int(sr * 0.02)                            # 20 ms RMS windows
    n_windows = len(samples) // win
    if n_windows < 2:
        return None
    baseline = None
    for i in range(n_windows):
        seg = samples[i * win:(i + 1) * win]
        rms = float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0
        if baseline is None:
            baseline = max(rms, 1e-4)
            continue
        if rms > max(baseline * 3.0, 0.02):
            onset_seconds = (i * win) / sr
            return max(0, int(round(onset_seconds * fps)))
    return None


# --------------------------------------------------------------- metrics ---
def _luma(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _motion_magnitude(frames: np.ndarray) -> float:
    if frames.shape[0] < 2:
        return 0.0
    diffs = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return float(diffs.mean())


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Wang et al. SSIM on two single-channel uint8 images."""
    if cv2 is None:
        return 1.0 if np.array_equal(a, b) else 0.0
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = a.astype(np.float64)
    img2 = b.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.where(den == 0, 1e-8, den)
    return float(ssim_map.mean())


_FACE_CASCADE = None
_FACE_CASCADE_LOADED = False


def _face_cascade():
    global _FACE_CASCADE, _FACE_CASCADE_LOADED
    if _FACE_CASCADE_LOADED:
        return _FACE_CASCADE
    _FACE_CASCADE_LOADED = True
    if cv2 is None:
        return None
    try:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.is_file():
            # Some cv2 wheels ship without the data dir; OpenCV would print a
            # persistence error to stderr on every call, so check first.
            return None
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            return None
        _FACE_CASCADE = cascade
    except Exception:                                # noqa: BLE001
        _FACE_CASCADE = None
    return _FACE_CASCADE


def _face_position_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    cascade = _face_cascade()
    if cascade is None:
        return None
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    fa = cascade.detectMultiScale(ga, 1.1, 4)
    fb = cascade.detectMultiScale(gb, 1.1, 4)
    if len(fa) == 0 or len(fb) == 0:
        return None
    def center(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x + w / 2.0, y + h / 2.0)
    ca, cb = center(fa), center(fb)
    diag = float((ga.shape[0] ** 2 + ga.shape[1] ** 2) ** 0.5) or 1.0
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 / diag)


def _global_motion_vector(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float]:
    if cv2 is None:
        return (0.0, 0.0)
    try:
        (dx, dy), _response = cv2.phaseCorrelate(
            gray_a.astype(np.float32), gray_b.astype(np.float32))
        return (float(dx), float(dy))
    except Exception:                                # noqa: BLE001
        return (0.0, 0.0)


def boundary_metrics(a_tail: np.ndarray, b_head: np.ndarray) -> dict:
    """Compare the LAST frame of `a_tail` against the FIRST frame of `b_head`,
    plus the short-window motion on each side."""
    empty = {"luma_diff": 0.0, "abs_luma_diff": 0.0, "color_diff": 0.0,
             "ssim": None, "motion_a": 0.0, "motion_b": 0.0,
             "motion_mismatch": 0.0, "face_diff": None,
             "global_motion": (0.0, 0.0)}
    if a_tail.shape[0] == 0 or b_head.shape[0] == 0:
        return empty
    last_a = a_tail[-1]
    first_b = b_head[0]
    y_a = float(_luma(last_a).mean())
    y_b = float(_luma(first_b).mean())
    luma_diff = y_b - y_a

    color_diff = 0.0
    ssim = None
    face_diff = None
    global_motion = (0.0, 0.0)
    if cv2 is not None:
        lab_a = cv2.cvtColor(last_a, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(first_b, cv2.COLOR_RGB2LAB).astype(np.float32)
        color_diff = float(np.abs(
            lab_a[..., 1:].mean(axis=(0, 1)) - lab_b[..., 1:].mean(axis=(0, 1))).sum())
        gray_a = cv2.cvtColor(last_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(first_b, cv2.COLOR_RGB2GRAY)
        ssim = _ssim_gray(gray_a, gray_b)
        face_diff = _face_position_diff(last_a, first_b)
        global_motion = _global_motion_vector(gray_a, gray_b)

    motion_a = _motion_magnitude(a_tail[-6:])
    motion_b = _motion_magnitude(b_head[:6])
    return {
        "luma_diff": luma_diff, "abs_luma_diff": abs(luma_diff),
        "color_diff": color_diff, "ssim": ssim,
        "motion_a": motion_a, "motion_b": motion_b,
        "motion_mismatch": abs(motion_a - motion_b),
        "face_diff": face_diff, "global_motion": global_motion,
    }


def frame_diffs(frames: np.ndarray) -> list[float]:
    """Mean absolute difference between each consecutive frame pair."""
    if frames.shape[0] < 2:
        return []
    return [float(np.abs(frames[i + 1].astype(np.float32)
                         - frames[i].astype(np.float32)).mean())
            for i in range(frames.shape[0] - 1)]


def detect_static_head(b_frames: np.ndarray, thresh: float = 0.6,
                       max_frames: int = 12, *,
                       reference_motion: float | None = None) -> int:
    """How many leading frames of `b_frames` are near-frozen (mean absolute
    difference to the next frame below the threshold), i.e. the "almost
    frozen" head measured on the real defect clip.

    Frame differences are median-filtered (width 3) before the comparison so
    a single encoder/keyframe spike at frame 0 does not hide a frozen head
    behind it. When `reference_motion` (the clip's own typical frame
    difference) is given, the threshold is raised to 40 % of it so a clip
    that is simply calm overall is not mistaken for frozen - low frame
    difference alone is never treated as an unwanted freeze."""
    diffs = frame_diffs(b_frames)
    if not diffs:
        return 0
    if reference_motion:
        thresh = max(thresh, 0.4 * float(reference_motion))
    smoothed = [float(np.median(diffs[max(0, i - 1):max(0, i - 1) + 3]))
                for i in range(len(diffs))]
    count = 0
    limit = min(max_frames, len(smoothed))
    for i in range(limit):
        if smoothed[i] < thresh:
            count += 1
        else:
            break
    return count


def choose_cut(a_tail: np.ndarray, b_head: np.ndarray, max_offset: int = 12,
               speech_onset_frame: int | None = None,
               static_frames: int = 0) -> dict:
    """Weighted search over offsets 0..max_offset into `b_head` for the best
    cut point against `a_tail`'s last frame. Offsets at/after
    `speech_onset_frame` are excluded (the new clip has already started
    talking by then, so an earlier frame is always the better splice).
    `static_frames` is the length of the near-frozen run at the head of
    `b_head` (detect_static_head); every frozen frame that would remain
    after the cut, beyond one, adds W_STATIC / max_offset to the score."""
    n = b_head.shape[0]
    if n == 0 or a_tail.shape[0] == 0:
        return {"offset": 0, "score": None, "candidates": []}
    limit = min(max_offset, n - 1)
    if speech_onset_frame is not None:
        limit = min(limit, max(0, speech_onset_frame - 1))
    candidates = []
    denom = float(max(1, max_offset))
    for offset in range(0, limit + 1):
        head_at = b_head[offset:offset + 1]
        metrics = boundary_metrics(a_tail, head_at)
        ssim = metrics["ssim"] if metrics["ssim"] is not None else 0.5
        remaining_static = max(0, int(static_frames) - offset - 1)
        score = (
            W_SSIM * (1.0 - ssim)
            + W_LUMA * (metrics["abs_luma_diff"] / 255.0)
            + W_COLOR * (metrics["color_diff"] / 255.0)
            + W_FACE * (metrics["face_diff"] or 0.0)
            + W_MOTION * (metrics["motion_mismatch"] / 255.0)
            + W_STATIC * (remaining_static / denom)
        )
        candidates.append({"offset": offset, "score": score, "metrics": metrics,
                           "remaining_static": remaining_static})
    best = min(candidates, key=lambda c: c["score"])
    return {"offset": best["offset"], "score": best["score"], "candidates": candidates}


# -------------------------------------------------------------- colour ---
def detect_intended_lighting_change(b_head: np.ndarray, min_frames: int = 4,
                                    threshold: float = 10.0) -> bool:
    """True when `b_head` itself shows a continuous, monotonic luminance
    change over its first frames - the signature of an intended scene/lighting
    transition rather than a generation artifact. Colour correction must not
    fight an intended change."""
    if b_head.shape[0] < min_frames:
        return False
    ys = [float(_luma(f).mean()) for f in b_head[:min_frames]]
    diffs = np.diff(ys)
    if diffs.size == 0:
        return False
    monotonic = bool(np.all(diffs >= -0.5)) or bool(np.all(diffs <= 0.5))
    return monotonic and abs(ys[-1] - ys[0]) > threshold


def color_match_params(a_tail: np.ndarray, b_head: np.ndarray, *,
                       mode: str = "natural", intended_change: bool = False) -> dict:
    """Gain/offset/ramp for nudging `b_head`'s colour back toward `a_tail`'s
    last frame. Only ever proposed for `mode == "natural"` and never when
    `intended_change` (a real scene/lighting change) is true."""
    off_result = {"apply": False, "gain": [1.0, 1.0, 1.0],
                  "offset": [0.0, 0.0, 0.0], "ramp_frames": 0}
    if mode != "natural" or intended_change:
        return off_result
    if a_tail.shape[0] == 0 or b_head.shape[0] == 0:
        return off_result
    last_a = a_tail[-1]
    first_b = b_head[0]
    y_a = float(_luma(last_a).mean())
    y_b = float(_luma(first_b).mean())
    delta_y = abs(y_a - y_b)
    color_diff = 0.0
    if cv2 is not None:
        lab_a = cv2.cvtColor(last_a, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(first_b, cv2.COLOR_RGB2LAB).astype(np.float32)
        color_diff = float(np.abs(
            lab_a[..., 1:].mean(axis=(0, 1)) - lab_b[..., 1:].mean(axis=(0, 1))).sum())
    if delta_y <= COLOR_LUMA_THRESHOLD and color_diff <= COLOR_DIFF_THRESHOLD:
        return off_result
    mean_a = last_a.reshape(-1, 3).astype(np.float32).mean(axis=0)
    mean_b = first_b.reshape(-1, 3).astype(np.float32).mean(axis=0)
    offset = (mean_a - mean_b).tolist()
    magnitude = max(delta_y, color_diff)
    ramp_frames = int(min(RAMP_MAX_FRAMES,
                          max(RAMP_MIN_FRAMES, round(magnitude * RAMP_FRAMES_PER_UNIT))))
    return {"apply": True, "gain": [1.0, 1.0, 1.0], "offset": offset,
            "ramp_frames": ramp_frames}


def apply_color_ramp(frames: np.ndarray, params: dict, *, start_index: int = 0) -> np.ndarray:
    """Reference (numpy) form of the ramp, used by the unit tests to pin the
    decay shape. The production merge applies the LUMA part of `params` in
    the ffmpeg graph (merge._luma_ramp_filter); the per-channel chroma part
    is computed for the report but not applied.

    Linearly-decaying gain/offset correction over the first `ramp_frames`
    frames only (weight 1.0 at frame 0 down toward 0.0 by `ramp_frames`).
    `start_index` is the absolute ramp position of `frames[0]` so a long ramp
    can be processed in chunks. Works one frame at a time in float32 so a
    60-frame 576x1024 ramp never needs a full float copy of the stack."""
    if not params.get("apply") or frames.shape[0] == 0:
        return frames
    ramp = int(params.get("ramp_frames", 0))
    if ramp <= 0:
        return frames
    out = frames.copy()
    gain = np.array(params.get("gain", [1.0, 1.0, 1.0]), dtype=np.float32)
    offset = np.array(params.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
    for i in range(out.shape[0]):
        pos = start_index + i
        if pos >= ramp:
            break
        weight = 1.0 - (pos / float(ramp))
        frame = out[i].astype(np.float32)
        frame = frame * (1.0 + (gain - 1.0) * weight) + offset * weight
        out[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------- interpolation ---
def rife_available(comfy_object_info: dict | None) -> bool:
    """True only if ComfyUI's object_info advertises FrameInterpolationModelLoader
    WITH at least one installed checkpoint. Never assumed true."""
    if not comfy_object_info:
        return False
    node = comfy_object_info.get("FrameInterpolationModelLoader")
    if not node:
        return False
    try:
        options = node["input"]["required"]["ckpt_name"][0]
    except Exception:                                # noqa: BLE001
        return False
    return bool(options)


async def interpolate_pair(a_last: np.ndarray, b_first: np.ndarray, n: int = 2,
                           method: str = "ffmpeg", *,
                           comfy_object_info: dict | None = None) -> list[np.ndarray]:
    """Produce `n` (2..4) intermediate frames between `a_last` and `b_first`.

    `method="ffmpeg"` (the default, explicitly labelled) uses ffmpeg's
    `minterpolate` filter. `method="rife"` is only honoured when ComfyUI's
    FrameInterpolationModelLoader has an installed model; otherwise this
    raises SeamError with a clear Japanese reason instead of silently falling
    back to ffmpeg."""
    n = max(2, min(4, int(n)))
    if method == "rife":
        if not rife_available(comfy_object_info):
            raise SeamError(
                "補間モデル(RIFE)が導入されていません。"
                "ComfyUIのFrameInterpolationModelLoaderで使用できるモデルが"
                "見つかりませんでした。既定のffmpeg(minterpolate)方式を"
                "使用するか、RIFEモデルを導入してください。")
        # RIFE execution goes through the ComfyUI graph, not this module -
        # callers with a verified model must render it themselves and pass
        # the frames back in; there is no local RIFE runtime here.
        raise SeamError(
            "RIFE補間はComfyUIグラフ経由でのみ実行できます。"
            "この関数はffmpeg方式のみ実行します。")
    if method != "ffmpeg":
        raise SeamError(f"未知の補間方式です: {method}")
    if a_last.shape != b_first.shape:
        raise SeamError("補間する2フレームのサイズが一致しません。")
    exe = merge_mod.find_ffmpeg()
    h, w = a_last.shape[:2]
    with tempfile.TemporaryDirectory(prefix="h3-interp-") as td:
        raw_in = Path(td) / "in.rgb"
        with open(raw_in, "wb") as f:
            f.write(np.ascontiguousarray(a_last).tobytes())
            f.write(np.ascontiguousarray(b_first).tobytes())
        args = [exe, "-hide_banner", "-nostdin", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "1",
                "-i", str(raw_in),
                "-vf", f"minterpolate=fps={n + 1}:mi_mode=blend",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        out, _ = await _run_capture(args, timeout=60)
    frame_bytes = w * h * 3
    total = len(out) // frame_bytes if frame_bytes else 0
    frames = [np.frombuffer(out[i * frame_bytes:(i + 1) * frame_bytes], dtype=np.uint8)
              .reshape(h, w, 3).copy() for i in range(total)]
    # Drop the (near-)copies of the two anchor frames, keep only interior ones.
    if len(frames) > 2:
        return frames[1:-1]
    return frames


# -------------------------------------------------------------------- audio ---
def audio_adjust(offset_seconds: float, *, keep_pause: bool = False,
                 silent_segment: bool = False) -> dict:
    """How much of the NEXT clip's audio head to trim to stay in sync with a
    video-side cut of `offset_seconds`. Scripted pauses (keep_pause, or a
    segment with no dialogue) are never trimmed - silence is never removed
    wholesale, only kept exactly in sync with the video cut when a cut is
    actually being made for a non-silent segment."""
    if keep_pause or silent_segment:
        return {"trim_seconds": 0.0, "trim_ms": 0, "reason": "scripted_pause"}
    trim = max(0.0, float(offset_seconds))
    return {"trim_seconds": trim, "trim_ms": int(round(trim * 1000)),
            "reason": "video_offset_sync"}
