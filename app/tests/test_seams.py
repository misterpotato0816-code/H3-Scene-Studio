"""WP-C: natural story clip seams.

CPU-only. Synthetic numpy frames exercise the pure analysis functions with no
I/O; a tiny real clip pair (generated with ffmpeg's lavfi testsrc) exercises
the full merge_story() ffmpeg pipeline when ffmpeg is on PATH, and is skipped
otherwise. No network, no GPU, no ComfyUI.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3app import seams
from h3app.errors import PipelineError

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _solid(h, w, r, g, b) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = r
    frame[..., 1] = g
    frame[..., 2] = b
    return frame


def _stack(frames) -> np.ndarray:
    return np.stack(frames, axis=0)


# --------------------------------------------------------------- metrics ---
class BoundaryMetricsTests(unittest.TestCase):
    def test_luma_diff_detected(self):
        a_tail = _stack([_solid(32, 32, 120, 120, 120)] * 3)
        b_head = _stack([_solid(32, 32, 40, 40, 40)] * 3)
        metrics = seams.boundary_metrics(a_tail, b_head)
        self.assertGreater(metrics["abs_luma_diff"], 50.0)
        self.assertLess(metrics["luma_diff"], 0)          # got darker

    def test_matching_frames_have_small_diff(self):
        frame = _solid(32, 32, 90, 90, 90)
        a_tail = _stack([frame, frame])
        b_head = _stack([frame, frame])
        metrics = seams.boundary_metrics(a_tail, b_head)
        self.assertAlmostEqual(metrics["abs_luma_diff"], 0.0, places=3)
        if metrics["ssim"] is not None:
            self.assertGreater(metrics["ssim"], 0.99)

    def test_empty_input_never_raises(self):
        empty = np.zeros((0, 4, 4, 3), dtype=np.uint8)
        metrics = seams.boundary_metrics(empty, empty)
        self.assertEqual(metrics["abs_luma_diff"], 0.0)


class StaticHeadTests(unittest.TestCase):
    def test_frozen_head_is_counted_and_removed(self):
        frozen = _solid(16, 16, 200, 100, 50)
        moving = [_solid(16, 16, 200 - i * 10, 100, 50) for i in range(6)]
        frames = _stack([frozen] * 4 + moving)
        n = seams.detect_static_head(frames, thresh=0.6, max_frames=12)
        self.assertGreaterEqual(n, 3)
        remaining = frames[n:]
        # Once the static head is dropped, consecutive frames actually differ.
        diff = float(np.abs(remaining[1].astype(np.float32)
                            - remaining[0].astype(np.float32)).mean())
        self.assertGreater(diff, 0.6)

    def test_no_static_head_returns_zero(self):
        moving = _stack([_solid(16, 16, 10 * i, 0, 0) for i in range(6)])
        self.assertEqual(seams.detect_static_head(moving), 0)

    def test_single_spike_at_frame_zero_does_not_hide_a_frozen_head(self):
        # Measured on the real defect clip: frame 0->1 differs by ~1.6 (first
        # decoded frame), then frames 1..8 are near-frozen (0.2-0.5).
        first = _solid(16, 16, 100, 100, 100)
        frozen = _solid(16, 16, 102, 102, 102)
        moving = [_solid(16, 16, 102 + 10 * i, 102, 102) for i in range(1, 6)]
        frames = _stack([first] + [frozen] * 6 + moving)
        n = seams.detect_static_head(frames, thresh=0.6, max_frames=12)
        self.assertGreaterEqual(n, 5)
        self.assertLessEqual(n, 7)

    def test_calm_clip_is_not_treated_as_frozen(self):
        # Every frame moves by ~1.0; the clip's own typical motion is 1.0, so
        # a threshold relative to it must NOT call the head frozen even though
        # the absolute threshold alone (0.6 < 1.0) would not either - and a
        # higher reference must not turn ordinary calm motion into "static".
        frames = _stack([_solid(16, 16, 100 + i, 100 + i, 100 + i) for i in range(12)])
        self.assertEqual(seams.detect_static_head(frames, reference_motion=1.0), 0)
        # Raising the reference to 5.0 makes 40 % of it = 2.0 > 1.0: now these
        # frames ARE below the clip's own motion and count as frozen.
        self.assertGreater(seams.detect_static_head(frames, reference_motion=5.0), 0)

    def test_frame_diffs_helper(self):
        frames = _stack([_solid(4, 4, 0, 0, 0), _solid(4, 4, 3, 3, 3), _solid(4, 4, 3, 3, 3)])
        self.assertEqual([round(d, 3) for d in seams.frame_diffs(frames)], [3.0, 0.0])
        self.assertEqual(seams.frame_diffs(frames[:1]), [])


class ChooseCutTests(unittest.TestCase):
    def test_best_offset_matches_previous_tail(self):
        target = _solid(24, 24, 128, 64, 32)
        a_tail = _stack([target, target])
        # offset 3 is an exact match to a_tail's last frame; the rest are far off.
        head = [_solid(24, 24, 5, 5, 5) for _ in range(3)] + [target] \
            + [_solid(24, 24, 250, 250, 250) for _ in range(4)]
        b_head = _stack(head)
        result = seams.choose_cut(a_tail, b_head, max_offset=7)
        self.assertEqual(result["offset"], 3)
        self.assertEqual(len(result["candidates"]), 8)

    def test_static_run_penalty_prefers_its_last_frame(self):
        # Four frozen head frames that all match A's tail equally well, then
        # frames that drift away. Without the penalty offset 0 wins (equal
        # similarity, first found); with static_frames=4 the LAST frozen
        # frame (offset 3) wins so at most one frozen frame remains.
        target = _solid(24, 24, 128, 64, 32)
        a_tail = _stack([target, target])
        head = [target] * 4 + [_solid(24, 24, 128 + 20 * i, 64, 32) for i in range(1, 6)]
        b_head = _stack(head)
        plain = seams.choose_cut(a_tail, b_head, max_offset=8)
        self.assertEqual(plain["offset"], 0)
        penalised = seams.choose_cut(a_tail, b_head, max_offset=8, static_frames=4)
        self.assertEqual(penalised["offset"], 3)
        self.assertEqual(penalised["candidates"][3]["remaining_static"], 0)
        # Speech onset still caps the search even inside the frozen run.
        capped = seams.choose_cut(a_tail, b_head, max_offset=8, static_frames=4,
                                  speech_onset_frame=2)
        self.assertLessEqual(capped["offset"], 1)

    def test_speech_onset_excludes_later_offsets(self):
        target = _solid(24, 24, 128, 64, 32)
        a_tail = _stack([target, target])
        head = [_solid(24, 24, 5, 5, 5) for _ in range(3)] + [target] \
            + [_solid(24, 24, 250, 250, 250) for _ in range(4)]
        b_head = _stack(head)
        # onset at frame 2: only offsets 0..1 may be chosen even though 3 is
        # the true best match.
        result = seams.choose_cut(a_tail, b_head, max_offset=7, speech_onset_frame=2)
        self.assertLessEqual(result["offset"], 1)
        self.assertEqual(len(result["candidates"]), 2)


# ------------------------------------------------------------------ colour --
class ColorRampTests(unittest.TestCase):
    def test_ramp_decays_monotonically(self):
        a_tail = _stack([_solid(16, 16, 200, 200, 200)])
        b_head = _stack([_solid(16, 16, 40, 40, 40)] * 8)
        params = seams.color_match_params(a_tail, b_head, mode="natural")
        self.assertTrue(params["apply"])
        self.assertGreaterEqual(params["ramp_frames"], seams.RAMP_MIN_FRAMES)
        self.assertLessEqual(params["ramp_frames"], seams.RAMP_MAX_FRAMES)
        corrected = seams.apply_color_ramp(b_head, params)
        n_ramp = min(params["ramp_frames"], b_head.shape[0])
        diffs = [float(np.abs(corrected[i].astype(np.float32)
                              - b_head[i].astype(np.float32)).mean())
                for i in range(n_ramp)]
        # Strictly decaying correction magnitude, and nothing left past the ramp.
        for i in range(len(diffs) - 1):
            self.assertGreaterEqual(diffs[i], diffs[i + 1])
        self.assertGreater(diffs[0], 0.0)
        if params["ramp_frames"] < corrected.shape[0]:
            untouched = float(np.abs(
                corrected[params["ramp_frames"]].astype(np.float32)
                - b_head[params["ramp_frames"]].astype(np.float32)).mean())
            self.assertAlmostEqual(untouched, 0.0, places=3)

    def test_no_correction_for_cut_or_fade_mode(self):
        a_tail = _stack([_solid(16, 16, 200, 200, 200)])
        b_head = _stack([_solid(16, 16, 40, 40, 40)] * 4)
        for mode in ("cut", "fade"):
            params = seams.color_match_params(a_tail, b_head, mode=mode)
            self.assertFalse(params["apply"], mode)

    def test_no_correction_on_intended_lighting_change(self):
        # Monotonic, large luminance ramp WITHIN b_head itself = a real
        # scene/lighting transition, not a generation seam artifact.
        b_head = _stack([_solid(16, 16, v, v, v) for v in (20, 60, 100, 140, 180)])
        self.assertTrue(seams.detect_intended_lighting_change(b_head))
        a_tail = _stack([_solid(16, 16, 200, 200, 200)])
        params = seams.color_match_params(
            a_tail, b_head, mode="natural",
            intended_change=seams.detect_intended_lighting_change(b_head))
        self.assertFalse(params["apply"])

    def test_small_difference_is_not_corrected(self):
        a_tail = _stack([_solid(16, 16, 100, 100, 100)])
        b_head = _stack([_solid(16, 16, 102, 100, 100)] * 4)
        params = seams.color_match_params(a_tail, b_head, mode="natural")
        self.assertFalse(params["apply"])


# ------------------------------------------------------------------- audio --
class AudioAdjustTests(unittest.TestCase):
    def test_keep_pause_trims_nothing(self):
        result = seams.audio_adjust(0.4, keep_pause=True)
        self.assertEqual(result["trim_seconds"], 0.0)
        self.assertEqual(result["trim_ms"], 0)
        self.assertEqual(result["reason"], "scripted_pause")

    def test_silent_segment_trims_nothing(self):
        result = seams.audio_adjust(0.4, silent_segment=True)
        self.assertEqual(result["trim_ms"], 0)

    def test_normal_cut_trims_exactly_the_video_offset(self):
        result = seams.audio_adjust(0.25)
        self.assertEqual(result["trim_ms"], 250)
        self.assertEqual(result["reason"], "video_offset_sync")


# ---------------------------------------------------------- interpolation --
class InterpolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_rife_without_model_raises_explicit_error(self):
        a = _solid(16, 16, 10, 10, 10)
        b = _solid(16, 16, 200, 200, 200)
        with self.assertRaises(seams.SeamError) as ctx:
            await seams.interpolate_pair(a, b, n=2, method="rife",
                                         comfy_object_info=None)
        self.assertIn("RIFE", str(ctx.exception))
        self.assertIsInstance(ctx.exception, PipelineError)

    async def test_rife_available_check_requires_installed_checkpoint(self):
        self.assertFalse(seams.rife_available(None))
        self.assertFalse(seams.rife_available({"FrameInterpolationModelLoader": {}}))
        ok_info = {"FrameInterpolationModelLoader":
                  {"input": {"required": {"ckpt_name": [["rife47.pth"]]}}}}
        self.assertTrue(seams.rife_available(ok_info))

    async def test_unknown_method_raises(self):
        a = _solid(16, 16, 10, 10, 10)
        b = _solid(16, 16, 200, 200, 200)
        with self.assertRaises(seams.SeamError):
            await seams.interpolate_pair(a, b, method="something-else")


# ------------------------------------------------------- merge_story (real) --
@unittest.skipUnless(HAS_FFMPEG, "ffmpeg not on PATH")
class MergeStoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from h3app import merge as merge_mod
        self.merge_mod = merge_mod
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-seam-test-")
        self.root = Path(self.tmp.name)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def _make_clip(self, name, color, duration=1.0, size="64x64", fps=24):
        exe = self.merge_mod.find_ffmpeg()
        out = self.root / name
        args = [exe, "-hide_banner", "-nostdin", "-y",
                "-f", "lavfi", "-i",
                f"color=c={color}:s={size}:r={fps}:d={duration}",
                "-f", "lavfi", "-i", f"anullsrc=r=32000:cl=stereo",
                "-shortest", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                "-c:a", "aac", str(out)]
        code, log = await self.merge_mod._run(args, timeout=60)
        self.assertEqual(code, 0, log)
        return out

    async def test_report_keys_and_needs_review_flag(self):
        clip_a = await self._make_clip("a.mp4", "white")
        clip_b = await self._make_clip("b.mp4", "black")
        out = self.root / "final.mp4"
        result = await self.merge_mod.merge_story(
            [clip_a, clip_b], out,
            boundaries=[{"mode": "natural", "keep_pause": False, "has_speech": False}],
            debug_dir=self.root, story_id="seam-test")
        self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(len(result["report"]), 1)
        entry = result["report"][0]
        for key in ("index", "mode", "frame_before", "time_before", "frame_after",
                    "time_after", "luma_diff_before", "luma_diff_after",
                    "ssim_before", "ssim_after", "static_removed", "cut_offset",
                    "color_correction", "interpolated_frames", "audio_trim_ms",
                    "needs_review", "metrics_before", "metrics_after"):
            self.assertIn(key, entry)
        self.assertEqual(entry["mode"], "natural")
        # White -> black is an extreme jump: flags for review, and such a
        # different shot is NOT colour-corrected toward the previous one.
        self.assertTrue(entry["needs_review"])
        self.assertTrue(result["needs_review"])
        self.assertFalse(entry["color_correction"]["apply"])
        self.assertEqual(entry["color_correction"].get("skipped"), "scene_change")
        # The report was also written to story_projects/<id>/seams_<ts>.json.
        self.assertTrue(any(self.root.glob("seams_*.json")))

    async def _make_beep_clip(self, name, color, *, beep_at, duration=2.0,
                              size="64x64", fps=24):
        """A clip whose audio is silent except for a 1 kHz tone starting at
        `beep_at` seconds - lets a test locate the audio after a merge."""
        exe = self.merge_mod.find_ffmpeg()
        out = self.root / name
        args = [exe, "-hide_banner", "-nostdin", "-y",
                "-f", "lavfi", "-i",
                f"color=c={color}:s={size}:r={fps}:d={duration}",
                "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=32000",
                "-filter_complex",
                f"[1:a]volume='if(gte(t,{beep_at}),1,0)':eval=frame,"
                f"atrim=end={duration},aformat=channel_layouts=stereo[a]",
                "-map", "0:v", "-map", "[a]", "-shortest",
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", str(out)]
        code, log = await self.merge_mod._run(args, timeout=60)
        self.assertEqual(code, 0, log)
        return out

    async def _first_loud_second(self, path) -> float:
        exe = self.merge_mod.find_ffmpeg()
        proc = await __import__("asyncio").create_subprocess_exec(
            exe, "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
            stdout=__import__("asyncio").subprocess.PIPE,
            stderr=__import__("asyncio").subprocess.PIPE)
        out, _ = await proc.communicate()
        samples = np.frombuffer(out, dtype=np.float32)
        win = 160                                     # 10 ms
        for i in range(len(samples) // win):
            seg = samples[i * win:(i + 1) * win]
            if float(np.sqrt(np.mean(np.square(seg)))) > 0.05:
                return i * 0.01
        return -1.0

    async def test_keep_pause_and_silent_segments_trim_nothing(self):
        # Frozen head on B (identical frames) that "natural" WOULD trim.
        clip_a = await self._make_beep_clip("kp_a.mp4", "0x808080", beep_at=5.0, duration=1.0)
        clip_b = await self._make_beep_clip("kp_b.mp4", "0x999999", beep_at=1.5, duration=2.0)
        for boundary in ({"mode": "natural", "keep_pause": True, "has_speech": True},
                         {"mode": "natural", "keep_pause": False, "has_speech": False}):
            out = self.root / f"kp_{int(boundary['keep_pause'])}_{int(boundary['has_speech'])}.mp4"
            result = await self.merge_mod.merge_story(
                [clip_a, clip_b], out, boundaries=[boundary],
                debug_dir=self.root, story_id="seam-kp")
            entry = result["report"][0]
            self.assertEqual(entry["static_removed"], 0)
            self.assertEqual(entry["cut_offset"], 0)
            self.assertEqual(entry["frame_after"], 0)
            self.assertEqual(entry["audio_trim_ms"], 0)
            self.assertIn(entry["trim_reason"], ("scripted_pause", "silent_segment"))
            # B's beep lands exactly where naive concatenation would put it.
            loud = await self._first_loud_second(out)
            self.assertAlmostEqual(loud, 1.0 + 1.5, delta=0.06)

    async def test_natural_trim_moves_audio_by_exactly_the_video_trim(self):
        # B has a 6-frame frozen head (identical frames) and no dialogue in
        # its first second, so natural mode trims the head; the beep must
        # arrive earlier by exactly the trimmed frames' duration.
        clip_a = await self._make_beep_clip("sync_a.mp4", "0x808080", beep_at=5.0, duration=1.0)
        clip_b = await self._make_beep_clip("sync_b.mp4", "0x999999", beep_at=1.5, duration=2.0)
        out = self.root / "sync.mp4"
        result = await self.merge_mod.merge_story(
            [clip_a, clip_b], out,
            boundaries=[{"mode": "natural", "keep_pause": False, "has_speech": True}],
            debug_dir=self.root, story_id="seam-sync")
        entry = result["report"][0]
        removed = entry["frame_after"]
        self.assertEqual(entry["audio_trim_ms"], int(round(removed / 24.0 * 1000)))
        self.assertLessEqual(removed, seams.HEAD_TRIM_MAX_FRAMES)
        loud = await self._first_loud_second(out)
        self.assertAlmostEqual(loud, 1.0 + 1.5 - removed / 24.0, delta=0.06)

    def test_luma_ramp_filter_shape(self):
        filt = self.merge_mod._luma_ramp_filter(9.5, 57)
        self.assertTrue(filt.startswith("hue=b='"))
        self.assertIn("1-n/57", filt)
        self.assertEqual(self.merge_mod._luma_ramp_filter(0.0, 57), "")
        self.assertEqual(self.merge_mod._luma_ramp_filter(9.5, 0), "")

    def test_concat_filter_indexes_inputs_in_order(self):
        e1 = self.merge_mod._seam_entry("a.mp4", 0, None, audio="a.mp4", fps=24)
        e2 = self.merge_mod._seam_entry("i.mp4", 0, 2, audio=None, fps=24)
        e3 = self.merge_mod._seam_entry("b.mp4", 8, None, audio="b.mp4", fps=24)
        e3["vf"] = "hue=b='0.1'"
        inputs, graph = self.merge_mod._build_concat_filter([e1, e2, e3], 24.0)
        # a, a, i, <silence>, b, b -> six inputs, silence is the lavfi one.
        self.assertEqual(inputs.count("-i"), 6)
        self.assertIn("anullsrc", " ".join(inputs))
        self.assertIn("[0:v]trim=start_frame=0,", graph)
        self.assertIn("[1:a]atrim=start=0.000000,", graph)
        self.assertIn("[2:v]trim=start_frame=0:end_frame=2,", graph)
        self.assertIn("[3:a]aformat=", graph)
        self.assertIn("[4:v]trim=start_frame=8,setpts=PTS-STARTPTS,hue=b='0.1',", graph)
        self.assertIn("[5:a]atrim=start=0.333333,", graph)
        self.assertTrue(graph.endswith("concat=n=3:v=1:a=1[v][a]"))

    async def test_cut_mode_applies_no_correction(self):
        clip_a = await self._make_clip("a2.mp4", "white")
        clip_b = await self._make_clip("b2.mp4", "black")
        out = self.root / "final_cut.mp4"
        result = await self.merge_mod.merge_story(
            [clip_a, clip_b], out, boundaries=[{"mode": "cut"}],
            debug_dir=self.root, story_id="seam-cut-test")
        self.assertTrue(Path(result["path"]).is_file())
        entry = result["report"][0]
        self.assertEqual(entry["mode"], "cut")
        self.assertFalse(entry["color_correction"]["apply"])
        self.assertEqual(entry["static_removed"], 0)


if __name__ == "__main__":
    unittest.main()
