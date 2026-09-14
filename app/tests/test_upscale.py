# -*- coding: utf-8 -*-
"""WP-B tests: h3app/upscale.py + the /api/upscale/* block in server.py.

CPU-only, no ComfyUI, no GPU, no network: ffmpeg/ffprobe and ComfyClient are
always mocked. aiohttp TestServer covers the HTTP layer only for the two
contracts B4 asks for (busy 409, path-traversal rejection); the planning/run
logic is exercised directly against h3app.upscale.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server                                                      # noqa: E402
from aiohttp.test_utils import TestClient, TestServer               # noqa: E402

from h3app import config                                            # noqa: E402
from h3app import upscale as upscale_mod                            # noqa: E402
from h3app.comfy import ComfyClient                                 # noqa: E402
from h3app.errors import PipelineError                              # noqa: E402
from h3app.pipeline import Pipeline                                  # noqa: E402
from h3app.projects import ProjectStore                             # noqa: E402
from h3app.story import StoryStore                                  # noqa: E402


# --------------------------------------------------------------- geometry --
class TargetDimsTests(unittest.TestCase):
    def test_portrait_short_edge_preset_keeps_ratio(self):
        w, h = upscale_mod.target_dims(576, 1024, "1080x1920")
        self.assertEqual((w, h), (1080, 1920))

    def test_landscape_short_edge_preset_keeps_ratio(self):
        w, h = upscale_mod.target_dims(1024, 576, "1080x1920")
        self.assertEqual((w, h), (1920, 1080))

    def test_x2_preset_doubles_both_axes(self):
        w, h = upscale_mod.target_dims(576, 1024, "x2")
        self.assertEqual((w, h), (1152, 2048))

    def test_even_rounding_on_odd_ratio(self):
        # 1080 * 1023 / 577 = 1913.86... -> nearest even.
        w, h = upscale_mod.target_dims(577, 1023, "1080x1920")
        self.assertEqual(w, 1080)
        self.assertEqual(h % 2, 0)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            upscale_mod.target_dims(576, 1024, "nope")

    def test_invalid_dims_raise(self):
        with self.assertRaises(ValueError):
            upscale_mod.target_dims(0, 1024, "x2")


# ------------------------------------------------------------------- plan --
class PlanTests(unittest.TestCase):
    SRC = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
          "duration": 5.0, "has_audio": True}

    ESRGAN_OBJECT_INFO = {
        "LoadVideo": {}, "GetVideoComponents": {}, "UpscaleModelLoader": {
            "input": {"required": {"model_name": [["RealESRGAN_x4plus.pth"]]}}},
        # The core tiled node is what the plan requires; the kjnodes batched
        # node may be present but is never chosen (it OOMs on whole frames).
        "ImageUpscaleWithModel": {}, "ImageUpscaleWithModelBatched": {},
        "ImageScale": {}, "CreateVideo": {}, "SaveVideo": {},
    }
    SEEDVR2_OBJECT_INFO = {
        "SeedVR2LoadDiTModel": {}, "SeedVR2LoadVAEModel": {},
        "SeedVR2VideoUpscaler": {}, "LoadVideo": {}, "GetVideoComponents": {},
        "CreateVideo": {}, "SaveVideo": {},
    }

    def test_esrgan_ok_when_node_and_model_present(self):
        result = upscale_mod.plan(
            self.SRC, {"preset": "1080x1920", "method": "esrgan"},
            {"effective_mode": "single"}, self.ESRGAN_OBJECT_INFO,
            {"esrgan": None})
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["target_w"], result["target_h"]), (1080, 1920))
        self.assertEqual(result["graph_inputs"]["device"], "cuda:0")
        self.assertEqual(result["graph_inputs"]["upscale_node"], "ImageUpscaleWithModel")

    def test_esrgan_missing_node_reported(self):
        info = dict(self.ESRGAN_OBJECT_INFO)
        del info["ImageScale"]
        result = upscale_mod.plan(
            self.SRC, {"preset": "1080x1920", "method": "esrgan"},
            {"effective_mode": "single"}, info, {"esrgan": None})
        self.assertFalse(result["ok"])
        names = [m["name"] for m in result["missing"]]
        self.assertIn("ImageScale", names)

    def test_esrgan_missing_model_when_not_on_disk_or_registry(self):
        info = dict(self.ESRGAN_OBJECT_INFO)
        info["UpscaleModelLoader"] = {
            "input": {"required": {"model_name": [[]]}}}
        result = upscale_mod.plan(
            self.SRC, {"preset": "1080x1920", "method": "esrgan"},
            {"effective_mode": "single"}, info, {"esrgan": None})
        self.assertFalse(result["ok"])
        models = [m for m in result["missing"] if m["kind"] == "model"]
        self.assertEqual(models[0]["name"], upscale_mod.ESRGAN_MODEL)
        self.assertEqual(models[0]["license"], "BSD-3-Clause")

    def test_seedvr2_missing_when_files_absent(self):
        result = upscale_mod.plan(
            self.SRC, {"preset": "x2", "method": "seedvr2"},
            {"effective_mode": "dual"}, self.SEEDVR2_OBJECT_INFO,
            {"seedvr2_dit": None, "seedvr2_vae": None})
        self.assertFalse(result["ok"])
        missing_names = {m["name"] for m in result["missing"]}
        self.assertIn(upscale_mod.SEEDVR2_DIT_FILE, missing_names)
        self.assertIn(upscale_mod.SEEDVR2_VAE_FILE, missing_names)
        for m in result["missing"]:
            self.assertEqual(m["source_url"], upscale_mod.SEEDVR2_SOURCE_URL)
            self.assertEqual(m["license"], "Apache-2.0")

    def test_seedvr2_ok_when_files_present(self):
        result = upscale_mod.plan(
            self.SRC, {"preset": "x2", "method": "seedvr2"},
            {"effective_mode": "dual"}, self.SEEDVR2_OBJECT_INFO,
            {"seedvr2_dit": Path("dit.safetensors"), "seedvr2_vae": Path("vae.safetensors")})
        self.assertTrue(result["ok"], result)

    def test_seedvr2_gpu_assignment_follows_plan_not_uuids(self):
        dual = upscale_mod.plan(
            self.SRC, {"preset": "x2", "method": "seedvr2"},
            {"effective_mode": "dual", "video": {"uuid": "GPU-fake-uuid-1"},
             "aux": {"kind": "gpu", "uuid": "GPU-fake-uuid-2"}},
            self.SEEDVR2_OBJECT_INFO,
            {"seedvr2_dit": Path("d"), "seedvr2_vae": Path("v")})
        gi = dual["graph_inputs"]
        self.assertEqual(gi["dit_device"], "cuda:0")
        self.assertEqual(gi["vae_device"], "cuda:1")
        self.assertEqual(gi["offload_device"], "cpu")
        for value in gi.values():
            self.assertNotIn("fake-uuid", str(value))

        single = upscale_mod.plan(
            self.SRC, {"preset": "x2", "method": "seedvr2"},
            {"effective_mode": "single"}, self.SEEDVR2_OBJECT_INFO,
            {"seedvr2_dit": Path("d"), "seedvr2_vae": Path("v")})
        self.assertEqual(single["graph_inputs"]["vae_device"], "cuda:0")

    def test_lanczos_ok_with_no_missing_and_no_node_requirements(self):
        result = upscale_mod.plan(
            self.SRC, {"preset": "1080x1920", "method": "lanczos"},
            {"effective_mode": "single"}, None, None)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["missing"], [])
        self.assertEqual((result["target_w"], result["target_h"]), (1080, 1920))
        self.assertEqual(result["graph_inputs"],
                         {"target_w": 1080, "target_h": 1920, "filter": "lanczos"})

    def test_lanczos_landscape_never_stretched_to_portrait(self):
        result = upscale_mod.plan(
            {"width": 1024, "height": 576, "fps": 24.0, "frames": 120,
             "duration": 5.0, "has_audio": True},
            {"preset": "1080x1920", "method": "lanczos"},
            {"effective_mode": "single"}, None, None)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["target_w"], result["target_h"]), (1920, 1080))

    def test_seedvr2_defaults_match_design(self):
        result = upscale_mod.plan(
            self.SRC, {"preset": "1080x1920", "method": "seedvr2"},
            {"effective_mode": "dual"}, self.SEEDVR2_OBJECT_INFO,
            {"seedvr2_dit": Path("d"), "seedvr2_vae": Path("v")})
        gi = result["graph_inputs"]
        # Measured on the real machine: only the fully-offloaded, tiled
        # configuration fits a 12 GB card at 1080x1920 (see upscale.py).
        self.assertEqual(gi["blocks_to_swap"], upscale_mod.SEEDVR2_BLOCKS_TO_SWAP)
        self.assertTrue(gi["swap_io_components"])
        self.assertTrue(gi["encode_tiled"])
        self.assertTrue(gi["decode_tiled"])
        self.assertEqual(gi["tile_size"], upscale_mod.SEEDVR2_TILE)
        self.assertEqual(gi["batch_size"], 5)
        self.assertEqual(gi["temporal_overlap"], 2)
        self.assertEqual(gi["color_correction"], "lab")
        self.assertEqual(gi["resolution"], 1080)
        self.assertEqual(gi["max_resolution"], 1920)


# ------------------------------------------------------------- build_graph --
class BuildGraphTests(unittest.TestCase):
    def test_esrgan_graph_wires_expected_nodes(self):
        plan_result = {
            "method": "esrgan", "preset": "1080x1920",
            "graph_inputs": {"upscale_node": "ImageUpscaleWithModelBatched",
                             "upscale_model": "RealESRGAN_x4plus.pth",
                             "per_batch": 5, "target_w": 1080, "target_h": 1920},
        }
        graph = upscale_mod.build_graph(plan_result, "h3app_upscale_x.mp4", "pfx")
        self.assertEqual(graph["load_video"]["inputs"]["file"], "h3app_upscale_x.mp4")
        self.assertEqual(graph["save_video"]["inputs"]["filename_prefix"], "pfx")
        # SaveVideo.execute() requires format/codec (validator does not).
        self.assertEqual(graph["save_video"]["inputs"]["format"], "auto")
        self.assertEqual(graph["save_video"]["inputs"]["codec"], "auto")
        self.assertEqual(graph["scale"]["inputs"]["width"], 1080)
        self.assertEqual(graph["upscale"]["inputs"]["per_batch"], 5)
        # kjnodes' batched node takes "images" (real ComfyUI rejected "image").
        self.assertEqual(graph["upscale"]["inputs"]["images"], ["components", 0])
        self.assertNotIn("image", graph["upscale"]["inputs"])

    def test_esrgan_core_node_uses_singular_image_input(self):
        plan_result = {
            "method": "esrgan", "preset": "1080x1920",
            "graph_inputs": {"upscale_node": "ImageUpscaleWithModel",
                             "upscale_model": "RealESRGAN_x4plus.pth",
                             "per_batch": 5, "target_w": 1080, "target_h": 1920},
        }
        graph = upscale_mod.build_graph(plan_result, "h3app_upscale_x.mp4", "pfx")
        self.assertEqual(graph["upscale"]["inputs"]["image"], ["components", 0])
        self.assertNotIn("per_batch", graph["upscale"]["inputs"])

    def test_seedvr2_graph_wires_expected_nodes(self):
        plan_result = {
            "method": "seedvr2", "preset": "x2",
            "graph_inputs": {"dit_model": "d.safetensors", "vae_model": "v.safetensors",
                             "dit_device": "cuda:0", "vae_device": "cuda:1",
                             "offload_device": "cpu", "blocks_to_swap": 20,
                             "decode_tiled": True, "batch_size": 5,
                             "temporal_overlap": 2, "color_correction": "lab",
                             "resolution": 1080, "max_resolution": 1920},
        }
        graph = upscale_mod.build_graph(plan_result, "h3app_upscale_y.mp4", "pfx2")
        self.assertEqual(graph["dit"]["inputs"]["device"], "cuda:0")
        self.assertEqual(graph["vae"]["inputs"]["device"], "cuda:1")
        self.assertEqual(graph["dit"]["inputs"]["blocks_to_swap"], 20)
        # VAE tiling is wired for BOTH encode and decode (encode was the
        # stage that OOMed on the real machine).
        self.assertTrue(graph["vae"]["inputs"]["encode_tiled"])
        self.assertEqual(graph["vae"]["inputs"]["encode_tile_size"],
                         upscale_mod.SEEDVR2_TILE)
        self.assertTrue(graph["dit"]["inputs"]["swap_io_components"])


# ------------------------------------------------------------------- probe --
class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_parses_ffprobe_json(self):
        payload = (
            b'{"streams":[{"codec_type":"video","width":576,"height":1024,'
            b'"r_frame_rate":"24/1","nb_frames":"120"},'
            b'{"codec_type":"audio"}],"format":{"duration":"5.0"}}'
        )

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return payload, b""

        with patch.object(upscale_mod, "find_ffprobe", return_value="ffprobe"), \
                patch("asyncio.create_subprocess_exec", AsyncMock(return_value=FakeProc())):
            result = await upscale_mod.probe("dummy.mp4")
        self.assertEqual(result["width"], 576)
        self.assertEqual(result["height"], 1024)
        self.assertEqual(result["fps"], 24.0)
        self.assertEqual(result["frames"], 120)
        self.assertTrue(result["has_audio"])


# --------------------------------------------------------------------- run --
class TestConfig(config.Config):
    def __init__(self, data, root):
        super().__init__(data, None)
        self._root = root

    @property
    def debug_dir(self):
        return self._root / "debug"


def _make_pipeline(root: Path) -> Pipeline:
    data = copy.deepcopy(config.DEFAULT_CONFIG)
    data.update(comfy_dir=str(root / "comfy"), media_root=str(root / "media"),
               input_stage_dir=str(root / "stage"))
    cfg = TestConfig(data, root)
    cfg.ensure_dirs()
    store = ProjectStore(cfg.projects_dir)
    client = ComfyClient(cfg)
    return Pipeline(cfg, store, client)


def _seed_history(cfg, filename: str, payload: bytes = b"UPSCALED") -> dict:
    cfg.comfy_output.mkdir(parents=True, exist_ok=True)
    (cfg.comfy_output / filename).write_bytes(payload)
    return {"outputs": {"save_video": {"video": [
        {"filename": filename, "subfolder": "", "type": "output"}]}}}


class RunTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-upscale-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _src(self, name="src.mp4", payload=b"SOURCE-BYTES") -> Path:
        p = self.root / name
        p.write_bytes(payload)
        return p

    def _plan(self, method="esrgan", target_w=1080, target_h=1920):
        gi = ({"upscale_node": "ImageUpscaleWithModelBatched",
              "upscale_model": "RealESRGAN_x4plus.pth", "per_batch": 5,
              "device": "cuda:0", "target_w": target_w, "target_h": target_h}
             if method == "esrgan" else
             {"dit_model": upscale_mod.SEEDVR2_DIT_FILE,
              "vae_model": upscale_mod.SEEDVR2_VAE_FILE,
              "dit_device": "cuda:0", "vae_device": "cuda:0",
              "offload_device": "cpu", "blocks_to_swap": 20,
              "decode_tiled": True, "batch_size": 5, "temporal_overlap": 2,
              "color_correction": "lab", "resolution": min(target_w, target_h),
              "max_resolution": max(target_w, target_h),
              "target_w": target_w, "target_h": target_h})
        return {"ok": True, "method": method, "preset": "1080x1920",
               "target_w": target_w, "target_h": target_h, "scale": 1.875,
               "graph_inputs": gi, "missing": [], "warnings": []}

    async def test_success_exports_and_leaves_source_untouched(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        original = src.read_bytes()
        history = _seed_history(pipeline.config, "up_0001.mp4")
        job = upscale_mod.UpscaleJob("up_test1")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}
        out_probe = {"width": 1080, "height": 1920, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}

        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[src_probe, out_probe])), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph",
                            AsyncMock(return_value=history)):
            result = await upscale_mod.run(pipeline, src, self._plan(), runner)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["export"]["ok"], result["export"])
        self.assertTrue(Path(result["export"]["path"]).is_file())
        self.assertEqual(src.read_bytes(), original)
        self.assertTrue(runner.state["finished"])
        self.assertEqual(runner.state["result"]["export"]["name"],
                         result["export"]["name"])

    async def test_numbering_never_overwrites(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        job1 = upscale_mod.UpscaleJob("up_n1")
        job2 = upscale_mod.UpscaleJob("up_n2")
        runner1 = upscale_mod.UpscaleRunner(pipeline, job1)
        runner2 = upscale_mod.UpscaleRunner(pipeline, job2)
        probes = {"width": 1080, "height": 1920, "fps": 24.0, "frames": 120,
                 "duration": 5.0, "has_audio": False}

        async def run_once(runner, filename):
            history = _seed_history(pipeline.config, filename)
            with patch.object(upscale_mod, "probe",
                              AsyncMock(side_effect=[probes, probes])), \
                    patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                    patch.object(pipeline.client, "run_graph",
                                AsyncMock(return_value=history)):
                return await upscale_mod.run(pipeline, src, self._plan(), runner)

        r1 = await run_once(runner1, "up_a.mp4")
        r2 = await run_once(runner2, "up_b.mp4")
        self.assertTrue(r1["ok"] and r2["ok"])
        self.assertNotEqual(Path(r1["export"]["path"]).name,
                            Path(r2["export"]["path"]).name)
        self.assertTrue(Path(r1["export"]["path"]).is_file())
        self.assertTrue(Path(r2["export"]["path"]).is_file())

    async def test_verification_mismatch_returns_ok_false(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        history = _seed_history(pipeline.config, "up_bad.mp4")
        job = upscale_mod.UpscaleJob("up_mismatch")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}
        # Wrong aspect ratio vs the plan's 1080x1920 target.
        bad_out_probe = {"width": 1000, "height": 1000, "fps": 24.0,
                        "frames": 120, "duration": 5.0, "has_audio": False}

        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[src_probe, bad_out_probe])), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph",
                            AsyncMock(return_value=history)):
            result = await upscale_mod.run(pipeline, src, self._plan(), runner)

        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])
        self.assertTrue(runner.state["finished"])

    async def test_failure_never_exports(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        job = upscale_mod.UpscaleJob("up_fail")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}

        with patch.object(upscale_mod, "probe", AsyncMock(return_value=src_probe)), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph",
                            AsyncMock(side_effect=PipelineError(
                                "生成エンジンでエラーが発生しました", kind="unknown"))):
            with self.assertRaises(PipelineError):
                await upscale_mod.run(pipeline, src, self._plan(), runner)

        self.assertFalse(pipeline.config.media_videos_final.is_dir() and
                         any(pipeline.config.media_videos_final.iterdir()))

    async def test_oom_retry_shrinks_vae_tile_and_batch(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        history = _seed_history(pipeline.config, "up_oom.mp4")
        job = upscale_mod.UpscaleJob("up_oom")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        probes = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                 "duration": 5.0, "has_audio": False}
        seen_blocks: list[int] = []
        seen_tiles: list[tuple] = []

        async def fake_run_graph(graph, **kwargs):
            seen_blocks.append(graph["dit"]["inputs"]["blocks_to_swap"])
            seen_tiles.append((graph["vae"]["inputs"]["encode_tile_size"],
                               graph["upscale"]["inputs"]["batch_size"]))
            if len(seen_blocks) < 2:
                raise PipelineError("CUDA out of memory", kind="oom")
            return history

        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[probes, probes])), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph", fake_run_graph):
            result = await upscale_mod.run(
                pipeline, src, self._plan(method="seedvr2", target_w=1080, target_h=1920),
                runner)

        self.assertTrue(result["ok"], result)
        # The retry keeps every block swapped and attacks the stage that
        # actually OOMs: VAE tile halved, batch dropped to 1.
        self.assertEqual(seen_blocks, [20, upscale_mod.MAX_BLOCKS_TO_SWAP])
        self.assertEqual(seen_tiles, [(upscale_mod.SEEDVR2_TILE, 5),
                                      (upscale_mod.SEEDVR2_TILE // 2, 1)])
        self.assertEqual(len(result["debug"]["retries"]), 1)
        retry_gi = result["debug"]["retries"][0]["graph_inputs"]
        self.assertEqual(retry_gi["blocks_to_swap"], upscale_mod.MAX_BLOCKS_TO_SWAP)
        self.assertEqual(retry_gi["tile_size"], upscale_mod.SEEDVR2_TILE // 2)
        self.assertEqual(retry_gi["batch_size"], 1)

    async def test_oom_retry_halves_per_batch_for_esrgan(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        history = _seed_history(pipeline.config, "up_oom_esrgan.mp4")
        job = upscale_mod.UpscaleJob("up_oom_e")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        probes = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                 "duration": 5.0, "has_audio": False}
        seen_batches: list[int] = []

        async def fake_run_graph(graph, **kwargs):
            seen_batches.append(graph["upscale"]["inputs"]["per_batch"])
            if len(seen_batches) < 2:
                raise PipelineError("CUDA out of memory", kind="oom")
            return history

        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[probes, probes])), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph", fake_run_graph):
            result = await upscale_mod.run(pipeline, src, self._plan(method="esrgan"), runner)

        self.assertTrue(result["ok"], result)
        self.assertEqual(seen_batches, [5, 2])

    def _lanczos_plan(self, target_w=1080, target_h=1920):
        return {"ok": True, "method": "lanczos", "preset": "1080x1920",
               "target_w": target_w, "target_h": target_h, "scale": 1.875,
               "graph_inputs": {"target_w": target_w, "target_h": target_h,
                                "filter": "lanczos"},
               "missing": [], "warnings": []}

    async def test_lanczos_run_never_touches_comfy(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        original = src.read_bytes()
        job = upscale_mod.UpscaleJob("up_lanczos1")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}
        out_probe = {"width": 1080, "height": 1920, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}

        async def fake_ffmpeg(args, cancel, **kwargs):
            out_path = Path(args[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"UPSCALED")
            return 0, ""

        must_not_be_called = AsyncMock(
            side_effect=AssertionError("lanczos must not reach ComfyUI"))
        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[src_probe, out_probe])), \
                patch.object(upscale_mod, "_run_ffmpeg_cancellable", fake_ffmpeg), \
                patch.object(pipeline, "_release_local_llm", must_not_be_called), \
                patch.object(pipeline.client, "ensure_ready", must_not_be_called), \
                patch.object(pipeline.client, "run_graph", must_not_be_called), \
                patch.object(pipeline.client, "free_models", must_not_be_called):
            result = await upscale_mod.run(pipeline, src, self._lanczos_plan(), runner)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["method"], "lanczos")
        self.assertEqual(result["peak_vram_mib"], 0)
        self.assertTrue(result["export"]["ok"], result["export"])
        self.assertTrue(Path(result["export"]["path"]).is_file())
        self.assertEqual(src.read_bytes(), original)
        self.assertTrue(runner.state["finished"])

    async def test_lanczos_verification_mismatch_returns_ok_false(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        job = upscale_mod.UpscaleJob("up_lanczos_bad")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}
        bad_out_probe = {"width": 1000, "height": 1000, "fps": 24.0,
                         "frames": 120, "duration": 5.0, "has_audio": False}

        async def fake_ffmpeg(args, cancel, **kwargs):
            out_path = Path(args[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"UPSCALED")
            return 0, ""

        with patch.object(upscale_mod, "probe",
                          AsyncMock(side_effect=[src_probe, bad_out_probe])), \
                patch.object(upscale_mod, "_run_ffmpeg_cancellable", fake_ffmpeg):
            result = await upscale_mod.run(pipeline, src, self._lanczos_plan(), runner)

        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])
        self.assertTrue(runner.state["finished"])

    async def test_oom_gives_up_after_max_retries(self):
        pipeline = _make_pipeline(self.root)
        src = self._src()
        job = upscale_mod.UpscaleJob("up_oom_giveup")
        runner = upscale_mod.UpscaleRunner(pipeline, job)
        probes = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                 "duration": 5.0, "has_audio": False}
        calls = {"n": 0}

        async def fake_run_graph(graph, **kwargs):
            calls["n"] += 1
            raise PipelineError("CUDA out of memory", kind="oom")

        with patch.object(upscale_mod, "probe", AsyncMock(return_value=probes)), \
                patch.object(pipeline.client, "free_models", AsyncMock(return_value={})), \
                patch.object(pipeline.client, "run_graph", fake_run_graph):
            with self.assertRaises(PipelineError):
                await upscale_mod.run(
                    pipeline, src, self._plan(method="seedvr2"), runner)
        self.assertEqual(calls["n"], upscale_mod.MAX_OOM_RETRIES + 1)


# ------------------------------------------------------------ resolve_source --
class ResolveSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-upscale-src-")
        self.root = Path(self.tmp.name)
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(media_root=str(self.root / "media"))
        self.cfg = config.Config(data, None)
        self.cfg.media_videos_final.mkdir(parents=True, exist_ok=True)
        (self.cfg.media_videos_final / "final.mp4").write_bytes(b"X")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_file_under_final_resolves(self):
        path, err = upscale_mod.resolve_source(
            self.cfg, None, None, {"kind": "file", "name": "final.mp4"})
        self.assertEqual(err, "")
        self.assertEqual(path, self.cfg.media_videos_final / "final.mp4")

    def test_path_traversal_rejected(self):
        for name in ("../final.mp4", "..\\final.mp4", "C:\\Windows\\win.ini",
                    "sub/final.mp4"):
            path, err = upscale_mod.resolve_source(
                self.cfg, None, None, {"kind": "file", "name": name})
            self.assertIsNone(path, name)
            self.assertTrue(err, name)

    def test_missing_file_rejected(self):
        path, err = upscale_mod.resolve_source(
            self.cfg, None, None, {"kind": "file", "name": "nope.mp4"})
        self.assertIsNone(path)
        self.assertTrue(err)

    def test_unknown_kind_rejected(self):
        path, err = upscale_mod.resolve_source(self.cfg, None, None, {"kind": "weird"})
        self.assertIsNone(path)
        self.assertTrue(err)


# ------------------------------------------------------------------- HTTP --
class UpscaleTestConfig(config.Config):
    def __init__(self, data, root):
        super().__init__(data, None)
        self._root = root

    @property
    def projects_dir(self):
        return self._root / "projects"

    @property
    def story_dir(self):
        return self._root / "stories"

    @property
    def debug_dir(self):
        return self._root / "debug"

    @property
    def app_dir(self):
        return self._root


class UpscaleHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-upscale-http-")
        self.root = Path(self.tmp.name)
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(self.root / "comfy"),
                   media_root=str(self.root / "media"),
                   input_stage_dir=str(self.root / "stage"),
                   auto_launch_comfy=False)
        with patch.object(server.presets_mod, "STORE_PATH",
                          self.root / "actions.json"):
            self.app = server.create_app(UpscaleTestConfig(data, self.root))
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.cfg = self.app["cfg"]
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_options_rejects_path_traversal(self):
        res = await self.client.get(
            "/api/upscale/options",
            params={"kind": "file", "name": "../secret.mp4"})
        self.assertEqual(res.status, 404)
        body = await res.json()
        self.assertFalse(body["ok"])

    async def test_start_returns_409_when_pipeline_busy(self):
        self.cfg.media_videos_final.mkdir(parents=True, exist_ok=True)
        (self.cfg.media_videos_final / "a.mp4").write_bytes(b"X")
        pipeline = self.app["pipeline"]
        pipeline.active = SimpleNamespace(state={"finished": False})
        res = await self.client.post("/api/upscale/start", json={
            "source": {"kind": "file", "name": "a.mp4"},
            "preset": "1080x1920", "method": "esrgan"})
        self.assertEqual(res.status, 409)
        body = await res.json()
        self.assertFalse(body["ok"])

    async def test_options_lanczos_skips_ensure_ready_and_returns_methods_source(self):
        self.cfg.media_videos_final.mkdir(parents=True, exist_ok=True)
        (self.cfg.media_videos_final / "a.mp4").write_bytes(b"X")
        src_probe = {"width": 576, "height": 1024, "fps": 24.0, "frames": 120,
                    "duration": 5.0, "has_audio": False}
        client = self.app["client"]
        must_not_be_called = AsyncMock(
            side_effect=AssertionError("lanczos options must not launch ComfyUI"))
        with patch.object(server.upscale_mod, "probe",
                          AsyncMock(return_value=src_probe)), \
                patch.object(client, "ensure_ready", must_not_be_called):
            res = await self.client.get(
                "/api/upscale/options",
                params={"kind": "file", "name": "a.mp4",
                       "preset": "1080x1920", "method": "lanczos"})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"], body)
        self.assertTrue(body["plan"]["ok"])
        self.assertEqual(body["plan"]["missing"], [])
        method_ids = {m["id"] for m in body["methods"]}
        self.assertEqual(method_ids, {"lanczos", "esrgan", "seedvr2"})
        self.assertEqual(body["source"], src_probe)

    async def test_start_rejects_traversal_before_busy_check(self):
        res = await self.client.post("/api/upscale/start", json={
            "source": {"kind": "file", "name": "../secret.mp4"},
            "preset": "1080x1920", "method": "esrgan"})
        self.assertEqual(res.status, 404)


if __name__ == "__main__":
    unittest.main()
