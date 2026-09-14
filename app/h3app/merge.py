# -*- coding: utf-8 -*-
"""ffmpeg helpers: concatenate a story's clips, and grab a clip's last frame.

Everything runs through asyncio.create_subprocess_exec so a 20-clip merge never
blocks the event loop (the SSE stream and the heartbeat keep flowing).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from .errors import PipelineError

_FFMPEG: str | None = None

# Voice Master extraction shape. MUST match the recipe the "voice stays fixed"
# claim was measured against - backend/h3_v2/ab_run.py::_extract_voice_master
# ("first 5 s of clip audio", pcm_s16le, 32000 Hz, stereo). Any deviation here
# is an unverified change to the anchor's audio characteristics.
VOICE_MASTER_SECONDS = 5
VOICE_MASTER_SAMPLE_RATE = 32000
VOICE_MASTER_CHANNELS = 2


def find_ffmpeg() -> str:
    """PATH first, then the copy imageio_ffmpeg ships inside the venv."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG

    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return _FFMPEG
    try:
        import imageio_ffmpeg                       # noqa: PLC0415 - optional dep
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            _FFMPEG = str(exe)
            return _FFMPEG
    except Exception:
        pass
    raise PipelineError(
        "動画を結合するための ffmpeg が見つかりませんでした。"
        "ffmpeg をインストールして PATH に追加するか、"
        "ComfyUI の Python 環境に imageio-ffmpeg を導入してください。",
        kind="missing_model", stage="動画の書き出し")


async def _run(args: list[str], *, timeout: float = 1800) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        raise PipelineError("ffmpeg の処理が時間内に終わりませんでした。",
                            kind="timeout", stage="動画の書き出し")
    return proc.returncode, (out or b"").decode("utf-8", errors="replace")


def _concat_list(paths: list[Path], list_path: Path) -> None:
    lines = []
    for p in paths:
        # The concat demuxer wants forward slashes and single-quote escaping.
        text = str(Path(p).resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{text}'")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def merge_clips(paths: list[Path], out_path: Path, *,
                      list_path: Path | None = None) -> Path:
    """Concatenate clips into out_path. Stream copy first, re-encode on failure."""
    usable = [Path(p) for p in paths if Path(p).is_file()]
    if not usable:
        raise PipelineError("結合できる動画がありません。", kind="unknown",
                            stage="動画の書き出し")
    exe = find_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = Path(list_path or out_path.with_suffix(".concat.txt"))
    _concat_list(usable, list_path)

    base = [exe, "-hide_banner", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path)]

    code, log = await _run(base + ["-c", "copy", str(out_path)])
    if code == 0 and out_path.is_file() and out_path.stat().st_size > 0:
        _cleanup(list_path)
        return out_path

    # Different clips can carry slightly different stream parameters; re-encoding
    # always works, it just costs time.
    code2, log2 = await _run(base + [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out_path)])
    _cleanup(list_path)
    if code2 == 0 and out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    tail = "\n".join((log2 or log).splitlines()[-25:])
    raise PipelineError("動画の結合に失敗しました。", kind="unknown",
                        stage="動画の書き出し", detail=tail)


def _cleanup(path: Path) -> None:
    try:
        path.unlink()
    except Exception:
        pass


async def extract_last_frame(video_path, out_png) -> Path | None:
    """Best effort: the last frame of a clip as a PNG thumbnail.

    Returns None instead of raising - a missing thumbnail must never fail a clip
    that already cost minutes of GPU time.
    """
    src = Path(video_path)
    dest = Path(out_png)
    if not src.is_file():
        return None
    try:
        exe = find_ffmpeg()
    except PipelineError:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    for seek in ("-0.15", "-0.5", "-1"):
        args = [exe, "-hide_banner", "-nostdin", "-y", "-sseof", seek,
                "-i", str(src), "-update", "1", "-frames:v", "1", str(dest)]
        try:
            code, _ = await _run(args, timeout=120)
        except PipelineError:
            return None
        if code == 0 and dest.is_file() and dest.stat().st_size > 0:
            return dest
    return None


async def extract_audio(video_path, out_wav) -> Path | None:
    """Best effort: a clip's Voice Master material as a PCM WAV suitable for
    ComfyUI's LoadAudio node - the first VOICE_MASTER_SECONDS seconds at
    VOICE_MASTER_SAMPLE_RATE Hz / VOICE_MASTER_CHANNELS channels, matching the
    recipe the measured benchmark used (see the module-level comment above).

    Returns None instead of raising - same convention as extract_last_frame,
    so a missing/silent source clip never fails the clip that already cost
    minutes of GPU time. Callers that need the anchor to exist (Voice Master)
    decide for themselves whether None is fatal.
    """
    src = Path(video_path)
    dest = Path(out_wav)
    if not src.is_file():
        return None
    try:
        exe = find_ffmpeg()
    except PipelineError:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [exe, "-hide_banner", "-nostdin", "-y", "-i", str(src),
            "-ss", "0", "-t", str(VOICE_MASTER_SECONDS),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(VOICE_MASTER_SAMPLE_RATE), "-ac", str(VOICE_MASTER_CHANNELS),
            str(dest)]
    try:
        code, _ = await _run(args, timeout=120)
    except PipelineError:
        return None
    if code == 0 and dest.is_file() and dest.stat().st_size > 0:
        return dest
    return None


# ============================================================== WP-C: seams ==
# Natural story clip seams (docs/DESIGN_2026-09-13_ADDITIONS.md, WP-C). The
# H3 model conditions clip N+1 on clip N's LAST tail_frames frames, so it does
# NOT emit an exact continuation - the seam is instead realised by choosing a
# cut point inside clip N+1's head, optionally colour-ramping it back toward
# clip N and inserting a couple of interpolated frames, then trimming clip
# N+1's audio by the exact same amount that was trimmed from its video so the
# whole thing stays in A/V sync. `merge_clips` above is untouched and remains
# the implementation for the plain `cut` transition.

async def _probe_duration(path) -> float:
    from . import seams as seams_mod                # local: avoids a cycle
    exe = seams_mod.find_ffprobe()
    args = [exe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
    _code, out = await _run(args, timeout=60)
    try:
        return float((out or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def _seam_entry(video, start_frame: int = 0, end_frame: int | None = None, *,
                audio=None, audio_start: float | None = None,
                audio_end: float | None = None, fps: float = 24.0) -> dict:
    """One piece of the final timeline. Video is addressed in FRAMES of
    `video` (end exclusive, None = to the end); audio is addressed in seconds
    of `audio` (None = silence for the piece's duration). Keeping the two
    explicit is what makes "the audio was trimmed by exactly the video cut"
    a fact rather than a hope."""
    entry = {"video": Path(video), "start_frame": int(start_frame),
             "end_frame": (int(end_frame) if end_frame is not None else None),
             "audio": (Path(audio) if audio is not None else None),
             "vf": "", "fps": float(fps)}
    if audio is not None:
        entry["audio_start"] = (float(audio_start) if audio_start is not None
                                else start_frame / fps)
        entry["audio_end"] = (float(audio_end) if audio_end is not None
                              else (end_frame / fps if end_frame is not None else None))
    return entry


def _build_concat_filter(entries: list[dict], fps: float, *,
                         sample_rate: int = 48000) -> tuple[list[str], str]:
    """Input arguments + filter_complex string that concatenates `entries`
    with ffmpeg's concat FILTER (each input decoded on its own, exact
    frame/second trims). The concat DEMUXER was tried first and rejected: it
    hands the staging snippets and the H3 clips to one decoder as a single
    stream, which mis-decoded the snippet frames (measured Lab a/b swung from
    126/142 to 163/120) and drifted the audio by ~0.4 s at the seam."""
    inputs: list[str] = []
    parts: list[str] = []
    labels: list[str] = []
    n_inputs = 0
    for i, entry in enumerate(entries):
        v_in = n_inputs
        inputs += ["-i", str(entry["video"])]
        n_inputs += 1
        a_idx = n_inputs
        if entry.get("audio") is not None:
            inputs += ["-i", str(entry["audio"])]
        else:
            # Silent piece (e.g. interpolated frames): generate exactly its
            # duration of silence so nothing downstream shifts.
            n_frames = entry["end_frame"] - entry["start_frame"]
            # The piece's own fps (the snippet was encoded at the boundary's
            # clip fps), not the output fps, so its silence is exactly as
            # long as its frames.
            dur = max(n_frames / float(entry.get("fps") or fps), 0.001)
            inputs += ["-f", "lavfi", "-t", f"{dur:.6f}",
                       "-i", f"anullsrc=r={sample_rate}:cl=stereo"]
        n_inputs += 1
        v_trim = f"trim=start_frame={entry['start_frame']}"
        if entry["end_frame"] is not None:
            v_trim += f":end_frame={entry['end_frame']}"
        extra = f",{entry['vf']}" if entry.get("vf") else ""
        parts.append(f"[{v_in}:v]{v_trim},setpts=PTS-STARTPTS{extra},fps={fps:g},"
                     f"format=yuv420p,setsar=1[v{i}]")
        if entry.get("audio") is not None:
            a_trim = f"atrim=start={entry['audio_start']:.6f}"
            if entry.get("audio_end") is not None:
                a_trim += f":end={entry['audio_end']:.6f}"
            parts.append(f"[{a_idx}:a]{a_trim},asetpts=PTS-STARTPTS,"
                         f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:"
                         f"channel_layouts=stereo[a{i}]")
        else:
            parts.append(f"[{a_idx}:a]aformat=sample_fmts=fltp:sample_rates={sample_rate}:"
                         f"channel_layouts=stereo[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    parts.append("".join(labels) + f"concat=n={len(entries)}:v=1:a=1[v][a]")
    return inputs, ";".join(parts)


def _luma_ramp_filter(luma_offset: float, ramp_frames: int) -> str:
    """ffmpeg `hue` brightness expression that adds `luma_offset` (RGB luma
    units) at the first frame of the stream it is applied to and decays
    linearly to 0 by `ramp_frames`. `hue` rebuilds its luma LUT every frame
    when the expression uses `n`, so the ramp is as smooth as 8-bit allows
    (one code value per step); `eq` was measured to update only every ~12
    frames and is not used. hue's `b` unit is 25.5 Y code values, and Y code
    values are 219/255 of RGB units (tv range).

    Applied INSIDE the concat graph to clip B's own bt709 yuv420p frames, so
    there is no RGB round trip and no colour-matrix mismatch (an earlier RGB
    staging snippet lost ~4 luma and shifted chroma through the auto-inserted
    bt601->bt709 conversion)."""
    if ramp_frames <= 0 or abs(luma_offset) < 1e-6:
        return ""
    b = float(luma_offset) * (219.0 / 255.0) / 25.5
    # The escaped comma keeps max() intact inside the filter graph string.
    return f"hue=b='{b:.6f}*max(0\\,1-n/{int(ramp_frames)})'"


async def _write_frames_clip(frames: np.ndarray, out_path: Path, fps: float) -> Path:
    """Encode a small in-memory frame stack as a near-lossless, video-only
    staging clip (crf 0) - an implementation necessity for the ramp and
    interpolation snippets. Audio for these pieces is taken straight from the
    source clip (or generated silence) by _build_concat_filter, and the
    video the user sees is re-encoded exactly once by merge_story()."""
    exe = find_ffmpeg()
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [exe, "-hide_banner", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "pipe:0", "-an",
            # Same matrix/range/tags as the H3 clips so the concat graph
            # never auto-converts colour between pieces.
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "iec61966-2-1", "-color_range", "tv",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(out_path)]
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(
            proc.communicate(input=np.ascontiguousarray(frames).tobytes()), timeout=120)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        raise PipelineError("つなぎ目の一時クリップ書き出しが時間内に終わりませんでした。",
                            kind="timeout", stage="動画の書き出し")
    if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        tail = "\n".join((out or b"").decode("utf-8", errors="replace").splitlines()[-25:])
        raise PipelineError("つなぎ目の一時クリップ書き出しに失敗しました。",
                            kind="unknown", stage="動画の書き出し", detail=tail)
    return out_path


async def _build_xfade_snippet(clip_a: Path, clip_b: Path, fade_dur: float,
                               out_path: Path) -> Path:
    """A short (fade_dur-long) 0.3s crossfade splice: only the tail of A and
    the head of B, so it slots into the concat list between "A minus its
    tail" and "B minus its head" without duplicating either clip."""
    exe = find_ffmpeg()
    dur_a = await _probe_duration(clip_a)
    tail_start = max(0.0, dur_a - fade_dur)
    filt = (
        f"[0:v]trim=start={tail_start:.6f},setpts=PTS-STARTPTS[va];"
        f"[1:v]trim=end={fade_dur:.6f},setpts=PTS-STARTPTS[vb];"
        f"[va][vb]xfade=transition=fade:duration={fade_dur:.6f}:offset=0[v];"
        f"[0:a]atrim=start={tail_start:.6f},asetpts=PTS-STARTPTS[aa];"
        f"[1:a]atrim=end={fade_dur:.6f},asetpts=PTS-STARTPTS[ab];"
        f"[aa][ab]acrossfade=d={fade_dur:.6f}[a]"
    )
    args = [exe, "-hide_banner", "-nostdin", "-y",
            "-i", str(clip_a), "-i", str(clip_b),
            "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(out_path)]
    code, log = await _run(args, timeout=120)
    if code != 0 or not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
        tail = "\n".join(log.splitlines()[-25:])
        raise PipelineError("フェード結合に失敗しました。", kind="unknown",
                            stage="動画の書き出し", detail=tail)
    return Path(out_path)


def _write_seams_report(report: list[dict], needs_review: bool, *, story_id: str = "",
                        report_dir: Path | None = None,
                        app_debug_dir: Path | None = None) -> None:
    """story_projects/<id>/seams_<ts>.json and app/_debug/ - never fatal."""
    payload = {"story_id": story_id, "written_at": time.time(),
              "needs_review": needs_review, "boundaries": report}
    ts = int(time.time())
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    for target_dir, name in (
        (report_dir, f"seams_{ts}.json"),
        (app_debug_dir, f"seams_{story_id or 'story'}_{ts}.json"),
    ):
        if not target_dir:
            continue
        try:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            (Path(target_dir) / name).write_text(text, encoding="utf-8")
        except Exception as exc:                      # noqa: BLE001 - never fatal
            print(f"[seams] report not written to {target_dir}: {exc}", flush=True)


async def merge_story(clips: list, out_path, boundaries: list[dict] | None = None,
                      debug_dir=None, *, story_id: str = "", fps: float = 24.0,
                      app_debug_dir=None, comfy_object_info: dict | None = None,
                      max_offset: int = 12, tail_frames: int = 32,
                      head_frames: int = 40) -> dict:
    """Concatenate `clips` into `out_path`, applying the per-boundary
    transition plan.

    `boundaries[i]` describes the join between clips[i] and clips[i+1]:
    `{"mode": "natural"|"cut"|"fade", "keep_pause": bool, "has_speech": bool}`.
    Missing entries default to natural/no-pause/has-speech. With 0 or 1 usable
    clips this degenerates to the plain `merge_clips` (nothing to seam).

    Returns `{"path", "report": [...], "needs_review": bool}`. `report[i]`
    documents boundary i: frame numbers/times before and after, luma
    diff/SSIM before and after, removed static frames, the cut offset, the
    colour-correction params actually used (or `{"apply": False}`), the
    interpolated-frame count, the audio trim in ms (always equal to the video
    trim, or 0 when nothing was trimmed), the chosen mode, and the full
    before/after metrics dict."""
    from . import seams as seams_mod                # local: avoids a cycle

    usable = [Path(p) for p in clips if Path(p).is_file()]
    out_path = Path(out_path)
    if len(usable) <= 1:
        result = await merge_clips(usable, out_path)
        return {"path": result, "report": [], "needs_review": False}

    boundaries = [dict(b) for b in (boundaries or [])]
    while len(boundaries) < len(usable) - 1:
        boundaries.append({"mode": "natural", "keep_pause": False, "has_speech": True})

    report: list[dict] = []
    needs_review = False
    info_first = await seams_mod.probe_video(usable[0])
    out_fps = float(info_first.get("fps") or fps or 24.0)

    with tempfile.TemporaryDirectory(prefix="h3-seam-") as td:
        snippet_dir = Path(td)
        # Timeline pieces; the last piece is always "the rest of the current
        # clip", which the next boundary may shorten (fade) before appending.
        entries: list[dict] = [_seam_entry(usable[0], 0, None, audio=usable[0], fps=out_fps)]

        for i, boundary in enumerate(boundaries):
            clip_a, clip_b = usable[i], usable[i + 1]
            mode = str(boundary.get("mode") or "natural").lower()
            if mode not in ("natural", "cut", "fade"):
                mode = "natural"
            keep_pause = bool(boundary.get("keep_pause"))
            has_speech = bool(boundary.get("has_speech", True))
            info_a = await seams_mod.probe_video(clip_a)
            info_b = await seams_mod.probe_video(clip_b)
            clip_fps = float(info_b.get("fps") or out_fps)
            frames_a = int(info_a.get("frames", 0))
            frames_b = int(info_b.get("frames", 0))

            if mode == "cut":
                entries.append(_seam_entry(clip_b, 0, None, audio=clip_b, fps=clip_fps))
                report.append({
                    "index": i, "mode": "cut",
                    "frame_before": frames_a,
                    "time_before": (frames_a / clip_fps) if clip_fps else None,
                    "frame_after": 0, "time_after": 0.0,
                    "luma_diff_before": None, "luma_diff_after": None,
                    "ssim_before": None, "ssim_after": None,
                    "static_removed": 0, "cut_offset": 0,
                    "color_correction": {"apply": False}, "interpolated_frames": 0,
                    "audio_trim_ms": 0, "trim_reason": "cut",
                    "needs_review": False,
                    "metrics_before": None, "metrics_after": None,
                })
                continue

            if mode == "fade":
                fade_dur = 0.3
                fade_frames = max(1, int(round(fade_dur * clip_fps)))
                dur_a = await _probe_duration(clip_a)
                # Shorten "the rest of A" so the crossfade snippet replaces
                # A's tail and B's head instead of duplicating them.
                a_end_frame = max(entries[-1]["start_frame"], frames_a - fade_frames)
                entries[-1]["end_frame"] = a_end_frame
                entries[-1]["audio_end"] = a_end_frame / clip_fps if clip_fps else None
                snippet = snippet_dir / f"fade_{i:02d}.mp4"
                await _build_xfade_snippet(clip_a, clip_b, fade_dur, snippet)
                entries.append(_seam_entry(snippet, 0, None, audio=snippet, fps=clip_fps))
                entries.append(_seam_entry(clip_b, fade_frames, None, audio=clip_b,
                                           fps=clip_fps))
                report.append({
                    "index": i, "mode": "fade",
                    "frame_before": frames_a, "time_before": dur_a,
                    "frame_after": fade_frames, "time_after": fade_dur,
                    "luma_diff_before": None, "luma_diff_after": None,
                    "ssim_before": None, "ssim_after": None,
                    "static_removed": 0, "cut_offset": 0,
                    "color_correction": {"apply": False}, "interpolated_frames": 0,
                    "audio_trim_ms": int(fade_dur * 1000), "trim_reason": "fade",
                    "needs_review": False,
                    "metrics_before": None, "metrics_after": None,
                })
                continue

            # ---------------------------------------------------- natural --
            frames_a_eff = frames_a or tail_frames
            tail_n = min(tail_frames, frames_a_eff)
            a_tail = await seams_mod.decode_frames(
                clip_a, max(0, frames_a_eff - tail_n), tail_n,
                width=int(info_a.get("width") or 0), height=int(info_a.get("height") or 0))
            frames_b_eff = frames_b or head_frames
            head_n = min(head_frames, frames_b_eff)
            b_head = await seams_mod.decode_frames(
                clip_b, 0, head_n,
                width=int(info_b.get("width") or 0), height=int(info_b.get("height") or 0))

            metrics_before = seams_mod.boundary_metrics(a_tail, b_head)

            # Head trimming (frozen frames + cut-point offset) only when the
            # segment has dialogue and the user did not ask to keep the pause:
            # a scripted pause or a silent segment is left exactly as
            # generated, video AND audio, so nothing intended is removed.
            allow_trim = has_speech and not keep_pause
            trim_reason = ("video_offset_sync" if allow_trim
                           else ("scripted_pause" if keep_pause else "silent_segment"))

            onset = None
            if allow_trim:
                try:
                    onset = await seams_mod.find_speech_onset_frame(clip_b, fps=clip_fps)
                except Exception:                      # noqa: BLE001
                    onset = None

            static_n = 0
            offset = 0
            head_diffs = seams_mod.frame_diffs(b_head)
            reference_motion = float(np.median(head_diffs)) if head_diffs else 0.0
            if allow_trim:
                static_n = seams_mod.detect_static_head(
                    b_head, reference_motion=reference_motion)
                if onset is not None:
                    # Never eat into the frames where the line starts.
                    static_n = min(static_n, max(0, onset))

            # One search over the whole allowed window: the frozen run is a
            # penalty inside the score (not a hard pre-cut), so the chosen
            # frame can be the LAST frozen one when that is what matches the
            # previous clip best (measured: the model holds the relayed pose
            # there, then starts a new movement such as a blink).
            search = 0
            if allow_trim:
                search = min(seams_mod.HEAD_TRIM_MAX_FRAMES, max(max_offset, static_n + 4))
            cut = seams_mod.choose_cut(a_tail, b_head, max_offset=search,
                                       speech_onset_frame=onset,
                                       static_frames=static_n)
            total_removed = int(cut.get("offset", 0)) if allow_trim else 0
            static_removed = min(static_n, total_removed)
            offset = total_removed - static_removed
            b_after_static = b_head

            b_after_cut = (b_head[total_removed:] if total_removed < b_head.shape[0]
                          else b_head[-1:])
            intended = seams_mod.detect_intended_lighting_change(b_after_cut)
            metrics_after = seams_mod.boundary_metrics(a_tail, b_after_cut)
            # A boundary that is really a different shot (structure or
            # exposure far beyond what a colour nudge can bridge) gets no
            # correction: pulling a new scene toward the old one for three
            # seconds only adds a second visible change. It is flagged for
            # review instead (needs_review below).
            scene_change = (
                (metrics_after["ssim"] is not None
                 and metrics_after["ssim"] < seams_mod.REVIEW_SSIM_THRESHOLD)
                or metrics_after["abs_luma_diff"] > seams_mod.REVIEW_LUMA_THRESHOLD)
            color_params = seams_mod.color_match_params(
                a_tail, b_after_cut, mode="natural",
                intended_change=intended or scene_change)
            if scene_change:
                color_params = dict(color_params, skipped="scene_change")

            interpolated = 0
            extra_frames: list = []
            # Interpolated frames only where the head may be edited at all:
            # a scripted pause / silent segment is left exactly as generated.
            if allow_trim and metrics_after["motion_mismatch"] > 40.0 \
                    and b_after_cut.shape[0] > 0:
                try:
                    extra_frames = await seams_mod.interpolate_pair(
                        a_tail[-1], b_after_cut[0], n=2, method="ffmpeg")
                    interpolated = len(extra_frames)
                except seams_mod.SeamError:
                    extra_frames, interpolated = [], 0

            audio = seams_mod.audio_adjust(
                total_removed / clip_fps if clip_fps else 0.0,
                keep_pause=keep_pause, silent_segment=not has_speech)

            ramp_frames = (int(color_params.get("ramp_frames", 0))
                          if color_params.get("apply") else 0)
            head_start = total_removed
            remaining_b = max(0, frames_b - head_start) if frames_b else ramp_frames
            ramp_frames = min(ramp_frames, remaining_b)
            luma_offset = 0.0
            if color_params.get("apply") and ramp_frames > 0:
                off = color_params.get("offset", [0.0, 0.0, 0.0])
                luma_offset = float(0.299 * off[0] + 0.587 * off[1] + 0.114 * off[2])
            # What is actually applied is a LUMA ramp (hue=b=): the report
            # says so, and a boundary whose only difference is chroma gets
            # no correction rather than a pointless sub-1-unit ramp.
            if color_params.get("apply") and ramp_frames > 0 and abs(luma_offset) >= 1.0:
                color_params = dict(color_params, ramp_frames=ramp_frames,
                                    luma_offset=luma_offset, method="hue_luma_ramp",
                                    chroma_corrected=False)
            else:
                skipped = ("chroma_only_not_supported"
                           if color_params.get("apply") else color_params.get("skipped"))
                luma_offset = 0.0
                color_params = dict(color_params, apply=False, ramp_frames=0,
                                    luma_offset=0.0, chroma_corrected=False)
                if skipped:
                    color_params["skipped"] = skipped

            if extra_frames:
                interp_path = snippet_dir / f"interp_{i:02d}.mp4"
                stack = np.stack(extra_frames)
                await _write_frames_clip(stack, interp_path, clip_fps)
                entries.append(_seam_entry(interp_path, 0, int(stack.shape[0]),
                                           audio=None, fps=clip_fps))

            # Clip B from its new head, with the (optional) luma ramp applied
            # to its own frames inside the concat graph; audio starts at the
            # exact same frame so the trim is identical on both tracks.
            b_entry = _seam_entry(clip_b, head_start, None, audio=clip_b, fps=clip_fps)
            b_entry["vf"] = _luma_ramp_filter(luma_offset, ramp_frames)
            entries.append(b_entry)

            boundary_review = (
                (metrics_after["ssim"] is not None
                 and metrics_after["ssim"] < seams_mod.REVIEW_SSIM_THRESHOLD)
                or metrics_after["abs_luma_diff"] > seams_mod.REVIEW_LUMA_THRESHOLD
                or static_n > seams_mod.REVIEW_STATIC_THRESHOLD
            )
            needs_review = needs_review or boundary_review

            report.append({
                "index": i, "mode": "natural",
                "frame_before": frames_a,
                "time_before": (frames_a / clip_fps) if clip_fps else None,
                "frame_after": total_removed,
                "time_after": (total_removed / clip_fps) if clip_fps else None,
                "luma_diff_before": metrics_before["luma_diff"],
                "luma_diff_after": metrics_after["luma_diff"],
                "ssim_before": metrics_before["ssim"],
                "ssim_after": metrics_after["ssim"],
                "static_removed": static_removed,
                "static_detected": static_n,
                "cut_offset": offset,
                "speech_onset_frame": onset,
                "reference_motion": reference_motion,
                "color_correction": color_params,
                "interpolated_frames": interpolated,
                "audio_trim_ms": audio["trim_ms"],
                "trim_reason": trim_reason,
                "needs_review": boundary_review,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
            })

        inputs, filter_graph = _build_concat_filter(entries, out_fps)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        args = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y"] + inputs + [
            "-filter_complex", filter_graph, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path)]
        code, log = await _run(args, timeout=3600)
        if code != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
            tail = "\n".join(log.splitlines()[-25:])
            raise PipelineError("つなぎ目補正つきの結合に失敗しました。", kind="unknown",
                                stage="動画の書き出し", detail=tail)

    _write_seams_report(report, needs_review, story_id=story_id,
                        report_dir=debug_dir, app_debug_dir=app_debug_dir)
    return {"path": out_path, "report": report, "needs_review": needs_review}
