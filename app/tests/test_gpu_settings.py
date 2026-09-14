"""Task B: GPU detection, planning, launch args and HTTP settings surface.

No GPU, no ComfyUI, no network: nvidia-smi and ComfyUI /system_stats are both
mocked/injected. Real hardware is 4070 Ti (12282 MiB) + 3060 (12288 MiB).
"""
import asyncio
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server                                                       # noqa: E402
from aiohttp.test_utils import TestClient, TestServer                # noqa: E402
from h3app import audio_test as audio_test_mod                       # noqa: E402
from h3app import comfy as comfy_mod                                 # noqa: E402
from h3app import config, gpu, graphs, modes_v2                      # noqa: E402
from h3app.errors import PipelineError                               # noqa: E402


# --------------------------------------------------------------- fixtures ----
RTX_4070TI = {"index": 0, "uuid": "GPU-4070", "name": "NVIDIA GeForce RTX 4070 Ti",
             "pci_bus_id": "00000000:01:00.0", "total_mib": 12282, "free_mib": 10262,
             "used_mib": 1733, "compute_cap": "8.9"}
RTX_3060 = {"index": 1, "uuid": "GPU-3060", "name": "NVIDIA GeForce RTX 3060",
           "pci_bus_id": "00000000:03:00.0", "total_mib": 12288, "free_mib": 11374,
           "used_mib": 740, "compute_cap": "8.6"}


def _detected(gpus, ram_total_mib=64 * 1024, ok=True, error=""):
    return {"ok": ok, "error": error, "source": "nvidia-smi", "gpus": gpus,
            "ram_total_mib": ram_total_mib, "detected_at": 0.0}


def _csv_row(g, with_cc=True):
    row = f"{g['index']}, {g['uuid']}, {g['name']}, {g['pci_bus_id']}, " \
          f"{g['total_mib']}, {g['free_mib']}, {g['used_mib']}"
    if with_cc:
        row += f", {g['compute_cap']}"
    return row


# ------------------------------------------------------------- detect() ------
class DetectTests(unittest.TestCase):
    def test_parses_with_compute_cap(self):
        out = _csv_row(RTX_4070TI) + "\n" + _csv_row(RTX_3060) + "\n"
        d = gpu.detect(refresh=True, runner=lambda fields: (0, out, ""), now=0.0)
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["gpus"]), 2)
        self.assertEqual(d["gpus"][0]["compute_cap"], "8.9")
        self.assertEqual(d["gpus"][0]["total_mib"], 12282)

    def test_retries_without_compute_cap(self):
        calls = []

        def runner(fields):
            calls.append(list(fields))
            if "compute_cap" in fields:
                return 1, "", "field not supported"
            return 0, _csv_row(RTX_4070TI, with_cc=False), ""

        d = gpu.detect(refresh=True, runner=runner, now=1.0)
        self.assertTrue(d["ok"])
        self.assertEqual(d["gpus"][0]["compute_cap"], "")
        self.assertEqual(len(calls), 2)

    def test_garbage_rows_ignored(self):
        out = "not,a,valid,row\n" + _csv_row(RTX_4070TI) + "\n\n"
        d = gpu.detect(refresh=True, runner=lambda f: (0, out, ""), now=2.0)
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["gpus"]), 1)

    def test_no_gpu_detected(self):
        d = gpu.detect(refresh=True, runner=lambda f: (1, "", "no devices"), now=3.0)
        self.assertFalse(d["ok"])
        self.assertEqual(d["gpus"], [])

    def test_cache_reused_within_ttl(self):
        calls = {"n": 0}

        def runner(fields):
            calls["n"] += 1
            return 0, _csv_row(RTX_4070TI), ""

        gpu.detect(refresh=True, runner=runner, now=10.0)
        gpu.detect(refresh=False, runner=runner, now=10.1)
        self.assertEqual(calls["n"], 1)
        gpu.detect(refresh=False, runner=runner, now=20.0)
        self.assertEqual(calls["n"], 2)


# ---------------------------------------------------------------- ranking ----
class RankingTests(unittest.TestCase):
    def test_4070ti_ranks_first_despite_smaller_total(self):
        # 3060 reports 6 MiB MORE total; round(/1024) ties both at 12, so
        # compute_cap breaks the tie in favour of the 4070 Ti (8.9 > 8.6).
        ranked = gpu._ranked([RTX_3060, RTX_4070TI])
        self.assertEqual(ranked[0]["uuid"], RTX_4070TI["uuid"])
        self.assertEqual(ranked[1]["uuid"], RTX_3060["uuid"])


# ------------------------------------------------------------- build_plan ----
class BuildPlanTests(unittest.TestCase):
    def test_auto_dual_matches_measured_profile(self):
        d = _detected([RTX_4070TI, RTX_3060])
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        self.assertTrue(p["ok"], p["errors"])
        self.assertEqual(p["effective_mode"], "dual")
        self.assertEqual(p["video"]["uuid"], RTX_4070TI["uuid"])
        self.assertEqual(p["aux"]["uuid"], RTX_3060["uuid"])
        self.assertEqual(p["clip_device"], "gpu:1")
        self.assertTrue(p["measured"])
        self.assertEqual(p["warnings"], [])
        self.assertEqual(p["launch_args"], [
            "--cuda-device", f"{RTX_4070TI['uuid']},{RTX_3060['uuid']}",
            "--reserve-vram", "1.0"])

    def test_auto_single_gpu_uses_cpu_aux(self):
        d = _detected([RTX_4070TI])
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        self.assertTrue(p["ok"])
        self.assertEqual(p["effective_mode"], "single")
        self.assertEqual(p["aux"], {"kind": "cpu"})
        self.assertEqual(p["clip_device"], "cpu")
        self.assertTrue(any("CPU" in w for w in p["warnings"]))

    def test_no_gpu_detected_error(self):
        d = _detected([], ok=False, error="boom")
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        self.assertFalse(p["ok"])
        self.assertIn("NVIDIAのGPUを検出できません。", p["errors"])

    def test_dual_needs_two_gpus(self):
        d = _detected([RTX_4070TI])
        s = {"mode": "dual", "video_gpu": "", "aux_device": "", "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])
        self.assertIn("デュアルGPUには2枚以上のGPUが必要です（検出: 1枚）。", p["errors"])

    def test_dual_unknown_uuid_rejected(self):
        d = _detected([RTX_4070TI, RTX_3060])
        s = {"mode": "dual", "video_gpu": "GPU-missing", "aux_device": "", "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])
        self.assertTrue(any("見つかりません" in e for e in p["errors"]))

    def test_dual_same_gpu_rejected(self):
        d = _detected([RTX_4070TI, RTX_3060])
        s = {"mode": "dual", "video_gpu": RTX_4070TI["uuid"],
            "aux_device": RTX_4070TI["uuid"], "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])
        self.assertTrue(any("24GB" in e for e in p["errors"]))

    def test_dual_cpu_aux_rejected(self):
        d = _detected([RTX_4070TI, RTX_3060])
        s = {"mode": "dual", "video_gpu": "", "aux_device": "cpu", "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])
        self.assertTrue(any("シングルGPU" in e for e in p["errors"]))

    def test_single_aux_gpu_rejected(self):
        d = _detected([RTX_4070TI, RTX_3060])
        s = {"mode": "single", "video_gpu": "", "aux_device": RTX_3060["uuid"],
            "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])
        self.assertTrue(any("別処理側にGPUを指定できません" in e for e in p["errors"]))

    def test_single_ram_shortfall_rejected(self):
        d = _detected([RTX_4070TI], ram_total_mib=4 * 1024)
        s = {"mode": "single", "video_gpu": "", "aux_device": "", "reserve_vram_gb": 1.0}
        p = gpu.build_plan(s, d, text_encoder_bytes=25 * gpu.GiB)
        self.assertFalse(p["ok"])
        self.assertTrue(any("RAM" in e for e in p["errors"]))

    def test_reserve_out_of_range_rejected(self):
        d = _detected([RTX_4070TI, RTX_3060])
        for bad in (-0.1, 4.1):
            s = {"mode": "auto", "video_gpu": "", "aux_device": "", "reserve_vram_gb": bad}
            p = gpu.build_plan(s, d)
            self.assertFalse(p["ok"], bad)

    def test_reserve_too_large_for_gpu_rejected(self):
        d = _detected([{**RTX_4070TI, "total_mib": 3000}])
        s = {"mode": "auto", "video_gpu": "", "aux_device": "", "reserve_vram_gb": 2.0}
        p = gpu.build_plan(s, d)
        self.assertFalse(p["ok"])

    def test_untested_small_gpu_warning(self):
        small = {**RTX_4070TI, "total_mib": 8000}
        d = _detected([small])
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        self.assertTrue(p["ok"])
        self.assertTrue(any("未検証" in w for w in p["warnings"]))

    def test_low_free_vram_warning(self):
        low_free = {**RTX_4070TI, "free_mib": 100}
        d = _detected([low_free])
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        self.assertTrue(p["ok"])
        self.assertTrue(any("空きVRAMが少なくなっています" in w for w in p["warnings"]))

    def test_reserve_formatting_keeps_decimal(self):
        d = _detected([RTX_4070TI, RTX_3060])
        s = {"mode": "auto", "video_gpu": "", "aux_device": "", "reserve_vram_gb": 1.5}
        p = gpu.build_plan(s, d)
        self.assertIn("1.5", p["launch_args"])
        s2 = {"mode": "auto", "video_gpu": "", "aux_device": "", "reserve_vram_gb": 2}
        p2 = gpu.build_plan(s2, d)
        self.assertIn("2.0", p2["launch_args"])

    def test_never_sums_vram_across_gpus(self):
        d = _detected([RTX_4070TI, RTX_3060])
        p = gpu.build_plan(gpu.DEFAULT_GPU_SETTINGS, d)
        total_each = [g["total_mib"] for g in [p["video"], p["aux"]]]
        self.assertNotIn(24564, total_each)  # never a summed 24GB-ish figure


class StripDeviceFlagsTests(unittest.TestCase):
    def test_strips_all_forms(self):
        args = ["--default-device", "0", "--reserve-vram=1.0", "--fast",
                "--cuda-device", "GPU-a,GPU-b", "--other", "x"]
        out = gpu.strip_device_flags(args)
        self.assertEqual(out, ["--fast", "--other", "x"])

    def test_empty_and_none(self):
        self.assertEqual(gpu.strip_device_flags([]), [])
        self.assertEqual(gpu.strip_device_flags(None), [])


class NormalizeSettingsTests(unittest.TestCase):
    def test_default_when_absent(self):
        self.assertEqual(gpu.normalize_settings(None), gpu.DEFAULT_GPU_SETTINGS)

    def test_migrates_reserve_from_legacy_launch_args(self):
        out = gpu.normalize_settings(None, legacy_launch_args=["--default-device", "0",
                                                                "--reserve-vram", "1.5"])
        self.assertEqual(out["reserve_vram_gb"], 1.5)
        self.assertEqual(out["mode"], "auto")

    def test_unknown_mode_falls_back_to_auto(self):
        out = gpu.normalize_settings({"mode": "bogus"})
        self.assertEqual(out["mode"], "auto")


# ------------------------------------------------------------ applied_from_stats
class AppliedFromStatsTests(unittest.TestCase):
    def _stats(self, argv, device_names):
        return {"system": {"argv": argv},
                "devices": [{"type": "cuda", "index": i, "name": n,
                            "vram_total": 12 * gpu.GiB, "vram_free": 10 * gpu.GiB}
                           for i, n in enumerate(device_names)]}

    def test_uuid_cuda_device(self):
        argv = ["main.py", "--cuda-device",
               f"{RTX_4070TI['uuid']},{RTX_3060['uuid']}", "--reserve-vram", "1.0"]
        stats = self._stats(argv, ["cuda:0 NVIDIA GeForce RTX 4070 Ti : native",
                                   "cuda:1 NVIDIA GeForce RTX 3060 : native"])
        d = _detected([RTX_4070TI, RTX_3060])
        applied = gpu.applied_from_stats(stats, d)
        self.assertTrue(applied["known"])
        self.assertEqual(applied["video_uuid"], RTX_4070TI["uuid"])
        self.assertEqual(applied["aux_uuid"], RTX_3060["uuid"])
        self.assertEqual(applied["reserve_vram_gb"], 1.0)

    def test_default_device_0_maps_by_name(self):
        argv = ["main.py", "--default-device", "0"]
        stats = self._stats(argv, ["cuda:0 NVIDIA GeForce RTX 4070 Ti : native",
                                   "cuda:1 NVIDIA GeForce RTX 3060 : native"])
        d = _detected([RTX_4070TI, RTX_3060])
        applied = gpu.applied_from_stats(stats, d)
        self.assertTrue(applied["known"])
        self.assertEqual(applied["visible_uuids"],
                        [RTX_4070TI["uuid"], RTX_3060["uuid"]])

    def test_duplicate_names_unknown(self):
        argv = ["main.py"]
        stats = self._stats(argv, ["cuda:0 NVIDIA GeForce RTX 4070 Ti : native",
                                   "cuda:1 NVIDIA GeForce RTX 4070 Ti : native"])
        dup = {**RTX_3060, "name": RTX_4070TI["name"]}
        d = _detected([RTX_4070TI, dup])
        applied = gpu.applied_from_stats(stats, d)
        self.assertFalse(applied["known"])
        self.assertTrue(applied["note"])

    def test_unmatched_name_unknown(self):
        argv = ["main.py"]
        stats = self._stats(argv, ["cuda:0 NVIDIA GeForce RTX 9999 : native"])
        d = _detected([RTX_4070TI])
        applied = gpu.applied_from_stats(stats, d)
        self.assertFalse(applied["known"])


# ------------------------------------------------------ graphs / gates -------
class GraphClipDeviceTests(unittest.TestCase):
    def _images(self):
        return ["h3app_ref_a.png"]

    def _settings(self):
        return {"width": 576, "height": 1024, "frames": 124, "ref_image_size": "match",
               "seed": 123456789, "sampler": "res_multistep", "scheduler": "simple",
               "steps": 8}

    def _models(self):
        return {"clip": "clip.safetensors", "unet": "unet.safetensors",
               "vae_video": "v.safetensors", "vae_audio": "a.safetensors",
               "lora": "lora.safetensors", "lora_strength": 1.0}

    def test_default_clip_device_is_gpu1(self):
        g = graphs.build_generate_graph(
            self._images(), "hello", self._settings(), self._models(),
            [1536, 1152, 1152, 896], "H3/APP/x")
        sel = [n for n in g.values() if n.get("class_type") == "SelectCLIPDevice"]
        self.assertEqual(sel[0]["inputs"]["device"], "gpu:1")

    def test_clip_device_override_cpu(self):
        g = graphs.build_generate_graph(
            self._images(), "hello", self._settings(), self._models(),
            [1536, 1152, 1152, 896], "H3/APP/x", clip_device="cpu")
        sel = [n for n in g.values() if n.get("class_type") == "SelectCLIPDevice"]
        self.assertEqual(sel[0]["inputs"]["device"], "cpu")

    def test_parity_gate_respects_clip_device(self):
        g = graphs.build_generate_graph(
            self._images(), "hello", self._settings(), self._models(),
            [1536, 1152, 1152, 896], "H3/APP/x", clip_device="cpu")
        manifest = {}
        with self.assertRaises(graphs.GraphError):
            graphs.assert_v1_parity(g, self._settings(), self._models(), manifest,
                                    clip_device="gpu:1")
        # Passes with the matching clip_device.
        graphs.assert_v1_parity(g, self._settings(), self._models(), manifest,
                                clip_device="cpu")

    def test_audio_test_graph_clip_device(self):
        resolved = {"settings": self._settings(), "models": self._models()}
        g = audio_test_mod.build_audio_test_graph(
            images=self._images(), en_prompt="hi", resolved=resolved,
            ref_longest=[1536, 1152, 1152, 896], clip_device="cpu")
        sel = [n for n in g.values() if n.get("class_type") == "SelectCLIPDevice"]
        self.assertEqual(sel[0]["inputs"]["device"], "cpu")


class ComfyGateTests(unittest.TestCase):
    def _client(self):
        root = Path(tempfile.mkdtemp(prefix="h3-gate-"))
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(root / "comfy"))
        cfg = config.Config(data, None)
        cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        client = comfy_mod.ComfyClient(cfg)
        return client

    def test_gate_accepts_known_gpu_option(self):
        client = self._client()
        client.object_info = {
            "SelectCLIPDevice": {"input": {"required": {
                "device": [["default", "cpu", "gpu:0", "gpu:1"]]}}},
            "Foo": {},
        }
        graph = {"a": {"class_type": "Foo", "inputs": {}},
                "b": {"class_type": "SelectCLIPDevice", "inputs": {"device": "gpu:1"}}}
        client.gate(graph)  # must not raise

    def test_gate_rejects_unknown_gpu_option(self):
        client = self._client()
        client.object_info = {
            "SelectCLIPDevice": {"input": {"required": {
                "device": [["default", "cpu"]]}}},
            "Foo": {},
        }
        graph = {"a": {"class_type": "Foo", "inputs": {}},
                "b": {"class_type": "SelectCLIPDevice", "inputs": {"device": "gpu:1"}}}
        with self.assertRaises(PipelineError) as ctx:
            client.gate(graph)
        self.assertEqual(ctx.exception.kind, "no_gpu1")

    def test_gate_allows_cpu_always(self):
        client = self._client()
        client.object_info = {
            "SelectCLIPDevice": {"input": {"required": {
                "device": [["default", "cpu"]]}}},
            "Foo": {},
        }
        graph = {"a": {"class_type": "Foo", "inputs": {}},
                "b": {"class_type": "SelectCLIPDevice", "inputs": {"device": "cpu"}}}
        client.gate(graph)  # must not raise


class EnsureReadyGpuTests(unittest.IsolatedAsyncioTestCase):
    def _client(self):
        root = Path(tempfile.mkdtemp(prefix="h3-ensure-"))
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(root / "comfy"))
        cfg = config.Config(data, None)
        cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        client = comfy_mod.ComfyClient(cfg)
        return client

    async def test_own_server_restarts_on_signature_change(self):
        client = self._client()
        client.object_info = {"ok": True}
        client.process = SimpleNamespace(poll=lambda: None)
        client.attached = False
        client._profile = "base"
        client.launched_gpu_signature = ("--cuda-device", "OLD", "--reserve-vram", "1.0")

        new_plan = {"ok": True, "launch_args": ["--cuda-device", "NEW", "--reserve-vram", "1.0"],
                   "effective_mode": "single", "video": {"uuid": "NEW"}, "aux": {"kind": "cpu"}}
        restarted = {"n": 0}

        def fake_launch(profile="base"):
            restarted["n"] += 1
            client.launched_gpu_signature = tuple(new_plan["launch_args"])

        with patch("h3app.comfy.port_is_open", return_value=True), \
             patch.object(comfy_mod.gpu_mod, "current_plan", return_value=new_plan), \
             patch.object(client, "shutdown") as mock_shutdown, \
             patch.object(client, "_wait_port_closed", new=AsyncMock()), \
             patch.object(client, "_launch", side_effect=fake_launch) as mock_launch, \
             patch.object(client, "_wait_ready", new=AsyncMock()):
            await client.ensure_ready("base")

        mock_shutdown.assert_called_once()
        mock_launch.assert_called_once()

    async def test_attached_mismatch_refused(self):
        client = self._client()
        client.object_info = {"ok": True}
        client.process = None
        client.attached = True
        client._profile = "base"

        plan = {"ok": True, "launch_args": ["--cuda-device", "A,B", "--reserve-vram", "1.0"],
               "effective_mode": "dual", "video": {"uuid": "A"}, "aux": {"kind": "gpu", "uuid": "B"}}
        bad_applied = {"known": True, "visible_uuids": ["OTHER", "B"], "note": ""}

        with patch("h3app.comfy.port_is_open", return_value=True), \
             patch.object(comfy_mod.gpu_mod, "current_plan", return_value=plan), \
             patch.object(client, "system_stats", new=AsyncMock(return_value={})), \
             patch.object(comfy_mod.gpu_mod, "applied_from_stats", return_value=bad_applied), \
             patch.object(comfy_mod.gpu_mod, "detect", return_value={}), \
             patch.object(client, "_wait_ready", new=AsyncMock()):
            with self.assertRaises(PipelineError) as ctx:
                await client.ensure_ready("base")
        self.assertEqual(ctx.exception.kind, "gpu_mismatch")

    async def test_attached_match_accepted_and_cached(self):
        client = self._client()
        client.object_info = {"ok": True}
        client.process = None
        client.attached = True
        client._profile = "base"

        plan = {"ok": True, "launch_args": ["--cuda-device", "A,B", "--reserve-vram", "1.0"],
               "effective_mode": "dual", "video": {"uuid": "A"}, "aux": {"kind": "gpu", "uuid": "B"}}
        good_applied = {"known": True, "visible_uuids": ["A", "B"], "note": ""}
        calls = {"n": 0}

        async def fake_stats():
            calls["n"] += 1
            return {}

        with patch("h3app.comfy.port_is_open", return_value=True), \
             patch.object(comfy_mod.gpu_mod, "current_plan", return_value=plan), \
             patch.object(client, "system_stats", new=fake_stats), \
             patch.object(comfy_mod.gpu_mod, "applied_from_stats", return_value=good_applied), \
             patch.object(comfy_mod.gpu_mod, "detect", return_value={}), \
             patch.object(client, "_wait_ready", new=AsyncMock()):
            await client.ensure_ready("base")
            await client.ensure_ready("base")
        self.assertEqual(calls["n"], 1)  # second call served from the verified cache


# --------------------------------------------------------------- HTTP API ----
class GpuHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-gpu-http-")
        root = Path(self.tmp.name)

        class TestConfig(config.Config):
            @property
            def projects_dir(self): return root / "projects"
            @property
            def story_dir(self): return root / "stories"
            @property
            def debug_dir(self): return root / "debug"
            @property
            def app_dir(self): return root

        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(root / "comfy"), media_root=str(root / "media"),
                   auto_launch_comfy=False)
        self.config_path = root / "config.json"
        import json
        self.config_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        cfg = TestConfig(data, self.config_path)
        with patch.object(server.presets_mod, "STORE_PATH", root / "actions.json"):
            self.app = server.create_app(cfg)
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

        self.detected = _detected([RTX_4070TI, RTX_3060])
        self._detect_patch = patch.object(server.gpu_mod, "detect", return_value=self.detected)
        self._detect_patch.start()

    async def asyncTearDown(self):
        self._detect_patch.stop()
        await self.client.close()
        self.tmp.cleanup()

    async def test_get_settings_gpu_defaults(self):
        res = await self.client.get("/api/settings/gpu")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["plan"]["ok"])
        self.assertEqual(body["plan"]["effective_mode"], "dual")
        self.assertFalse(body["applied"]["running"])
        self.assertFalse(body["busy"])

    async def test_validate_endpoint(self):
        res = await self.client.post("/api/settings/gpu/validate",
                                     json={"settings": {"mode": "single"}})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["plan"]["ok"])
        self.assertEqual(body["plan"]["effective_mode"], "single")

    async def test_validate_endpoint_rejects_invalid(self):
        res = await self.client.post("/api/settings/gpu/validate",
                                     json={"settings": {"mode": "dual", "aux_device": "cpu"}})
        body = await res.json()
        self.assertFalse(body["plan"]["ok"])

    async def test_save_round_trip_preserves_unrelated_keys_and_strips_flags(self):
        # Seed a legacy device flag alongside an unrelated key.
        self.app["cfg"].data["comfy_launch_args"] = ["--default-device", "0",
                                                      "--reserve-vram", "1.0"]
        self.app["cfg"].data["idle_timeout_seconds"] = 42
        import json
        self.config_path.write_text(
            json.dumps(self.app["cfg"].data, ensure_ascii=False), encoding="utf-8")

        res = await self.client.post("/api/settings/gpu",
                                     json={"settings": {"mode": "single",
                                                        "reserve_vram_gb": 1.5}})
        self.assertEqual(res.status, 200, await res.text())
        on_disk = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["idle_timeout_seconds"], 42)
        self.assertEqual(on_disk["gpu"]["mode"], "single")
        self.assertEqual(on_disk["gpu"]["reserve_vram_gb"], 1.5)
        self.assertEqual(on_disk["comfy_launch_args"], [])

    async def test_save_rejects_invalid_plan_400(self):
        res = await self.client.post("/api/settings/gpu",
                                     json={"settings": {"mode": "dual", "aux_device": "cpu"}})
        self.assertEqual(res.status, 400)
        body = await res.json()
        self.assertFalse(body["ok"])

    async def test_save_refused_while_busy_409(self):
        self.app["pipeline"].active = SimpleNamespace(state={"finished": False})
        res = await self.client.post("/api/settings/gpu",
                                     json={"settings": {"mode": "single"}})
        self.assertEqual(res.status, 409)

    async def test_save_refused_while_story_running_409(self):
        self.app["story_pipeline"].runners["s1"] = SimpleNamespace(state={"finished": False})
        res = await self.client.post("/api/settings/gpu",
                                     json={"settings": {"mode": "single"}})
        self.assertEqual(res.status, 409)

    async def test_apply_reports_stopped_when_not_running(self):
        res = await self.client.post("/api/settings/gpu/apply", json={})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertIn("停止中", body["message"])

    async def test_apply_refuses_foreign_attached_server(self):
        self.app["client"].is_reachable = AsyncMock(return_value=True)
        self.app["client"].attached = True
        self.app["client"].process = None
        res = await self.client.post("/api/settings/gpu/apply", json={})
        self.assertEqual(res.status, 409)

    async def test_restart_required_when_reserve_vram_mismatches(self):
        # Devices/UUIDs match the saved plan exactly; only reserve_vram_gb differs.
        self.app["client"].applied_state = AsyncMock(return_value={
            "known": True, "running": True, "owned": True,
            "launch_args": ["--cuda-device", "GPU-4070,GPU-3060",
                            "--reserve-vram", "1.5"],
            "video_uuid": "GPU-4070", "aux_uuid": "GPU-3060",
            "visible_uuids": ["GPU-4070", "GPU-3060"],
            "reserve_vram_gb": 1.5, "devices": [], "note": "",
        })
        res = await self.client.get("/api/settings/gpu")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["restart_required"])
        self.assertIn("予約VRAM", body["restart_reason"])
        self.assertIn("1.5GB", body["restart_reason"])
        self.assertIn("1.0GB", body["restart_reason"])

    async def test_restart_required_when_reserve_vram_unknown(self):
        # Devices match, but the running ComfyUI has no --reserve-vram flag.
        self.app["client"].applied_state = AsyncMock(return_value={
            "known": True, "running": True, "owned": True,
            "launch_args": ["--cuda-device", "GPU-4070,GPU-3060"],
            "video_uuid": "GPU-4070", "aux_uuid": "GPU-3060",
            "visible_uuids": ["GPU-4070", "GPU-3060"],
            "reserve_vram_gb": None, "devices": [], "note": "",
        })
        res = await self.client.get("/api/settings/gpu")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["restart_required"])
        self.assertIn("予約VRAMを確認できません", body["restart_reason"])
        self.assertIn("1.0GB", body["restart_reason"])

    async def test_restart_not_required_when_everything_matches(self):
        self.app["client"].applied_state = AsyncMock(return_value={
            "known": True, "running": True, "owned": True,
            "launch_args": ["--cuda-device", "GPU-4070,GPU-3060",
                            "--reserve-vram", "1.0"],
            "video_uuid": "GPU-4070", "aux_uuid": "GPU-3060",
            "visible_uuids": ["GPU-4070", "GPU-3060"],
            "reserve_vram_gb": 1.0, "devices": [], "note": "",
        })
        res = await self.client.get("/api/settings/gpu")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertFalse(body["restart_required"])
        self.assertEqual(body["restart_reason"], "")

    async def test_restart_not_required_when_comfyui_stopped(self):
        self.app["client"].applied_state = AsyncMock(return_value={
            "known": False, "running": False, "owned": False,
            "launch_args": [], "video_uuid": "", "aux_uuid": "",
            "visible_uuids": [], "reserve_vram_gb": None,
            "devices": [], "note": "ComfyUIは停止しています。",
        })
        res = await self.client.get("/api/settings/gpu")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertFalse(body["restart_required"])
        self.assertEqual(body["restart_reason"], "")


if __name__ == "__main__":
    unittest.main()
