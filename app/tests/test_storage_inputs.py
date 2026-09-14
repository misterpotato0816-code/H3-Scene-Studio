"""Task A: storage/generated-video export + ComfyUI input handoff.

CPU-only, no ComfyUI, no GPU, no network to external hosts. Fake HTTP
servers use aiohttp TestServer on ephemeral ports.
"""
from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from h3app import comfy_inputs
from h3app import config
from h3app import media as media_mod
from h3app import system_info as system_info_mod
from h3app.comfy import ComfyClient
from h3app.errors import PipelineError
from h3app.story_runner import StoryPipeline
from h3app.story import StoryStore


def _tiny_png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(200, 50, 50)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- media.py --
class ExportVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-export-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _src(self, name="src.mp4", payload=b"FAKE-VIDEO-BYTES"):
        p = self.root / name
        p.write_bytes(payload)
        return p

    def test_success_and_unique_naming_on_collision(self):
        dest_dir = self.root / "out"
        src1 = self._src("a.mp4", b"AAAA")
        result1 = media_mod.export_video(
            src1, dest_dir=dest_dir, filename="clip", kind="final")
        self.assertTrue(result1["ok"], result1)
        self.assertEqual(Path(result1["path"]).name, "clip.mp4")

        src2 = self._src("b.mp4", b"BBBBBBBB")
        result2 = media_mod.export_video(
            src2, dest_dir=dest_dir, filename="clip", kind="final")
        self.assertTrue(result2["ok"], result2)
        self.assertEqual(Path(result2["path"]).name, "clip_2.mp4")
        # Never overwrites: the first file's bytes are untouched.
        self.assertEqual((dest_dir / "clip.mp4").read_bytes(), b"AAAA")
        self.assertEqual((dest_dir / "clip_2.mp4").read_bytes(), b"BBBBBBBB")

    def test_missing_source_never_raises(self):
        result = media_mod.export_video(
            self.root / "nope.mp4", dest_dir=self.root / "out",
            filename="clip", kind="final")
        self.assertFalse(result["ok"])
        self.assertIn("見つかりません", result["error"])

    def test_partial_failure_cleans_up_only_its_own_file(self):
        dest_dir = self.root / "out"
        dest_dir.mkdir(parents=True, exist_ok=True)
        pre_existing = dest_dir / "other.mp4"
        pre_existing.write_bytes(b"UNRELATED")
        src = self._src()
        with patch("os.fsync", side_effect=OSError(28, "No space left on device")):
            result = media_mod.export_video(
                src, dest_dir=dest_dir, filename="crash", kind="clip")
        self.assertFalse(result["ok"])
        self.assertIn("ディスク容量", result["error"])
        # The file THIS call created must be gone; nothing else touched.
        self.assertFalse((dest_dir / "crash.mp4").exists())
        self.assertTrue(pre_existing.is_file())
        self.assertEqual(pre_existing.read_bytes(), b"UNRELATED")

    def test_records_manifest_on_success(self):
        media_root = self.root / "media"
        src = self._src()
        result = media_mod.export_video(
            src, dest_dir=media_root / "Videos" / "完成動画", filename="clip",
            media_root=media_root, kind="final", project_id="proj1")
        self.assertTrue(result["ok"], result)
        rows = media_mod.read_manifest(media_root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["media_type"], "final")
        self.assertEqual(rows[0]["project_id"], "proj1")

    def test_name_relative_to_videos_folder(self):
        media_root = self.root / "media"
        src = self._src()
        cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        cfg.data["media_root"] = str(media_root)
        result = media_mod.export_video(
            src, dest_dir=cfg.media_videos_final, filename="abc_s01",
            media_root=media_root, kind="final")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["name"], str(Path("完成動画") / "abc_s01.mp4"))


# --------------------------------------------------------------- config.py --
class ConfigStorageTests(unittest.TestCase):
    def test_default_media_root_is_project_relative(self):
        cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        self.assertEqual(cfg.media_root, config.PROJECT_ROOT / "H3_Media")

    def test_videos_final_and_clips_subfolders(self):
        cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        cfg.data["media_root"] = r"C:\fake\media"
        self.assertEqual(cfg.media_videos_final, cfg.media_videos / "完成動画")
        self.assertEqual(cfg.media_videos_clips, cfg.media_videos / "クリップ")

    def test_input_stage_dir_default_uses_app_dir_property(self):
        class Sub(config.Config):
            @property
            def app_dir(self):
                return Path("Z:/somewhere")
        cfg = Sub(copy.deepcopy(config.DEFAULT_CONFIG))
        self.assertEqual(cfg.input_stage_dir, Path("Z:/somewhere") / "_comfy_input")

    def test_input_stage_dir_explicit_override(self):
        cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        cfg.data["input_stage_dir"] = r"C:\fake\stage"
        self.assertEqual(cfg.input_stage_dir, Path(r"C:\fake\stage"))


# ------------------------------------------------------------ comfy_inputs --
class ComfyInputsPureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-inputs-")
        self.root = Path(self.tmp.name)
        self.cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        self.cfg.data["input_stage_dir"] = str(self.root / "stage")
        self.cfg.data["comfy_dir"] = str(self.root / "comfy")

    def tearDown(self):
        self.tmp.cleanup()

    def test_safe_name(self):
        self.assertTrue(comfy_inputs.safe_name("h3app_ref_abc.png"))
        self.assertFalse(comfy_inputs.safe_name("../h3app_ref_abc.png"))
        self.assertFalse(comfy_inputs.safe_name("evil.png"))
        self.assertFalse(comfy_inputs.safe_name(""))
        self.assertFalse(comfy_inputs.safe_name(None))

    def test_local_path_prefers_stage_then_legacy(self):
        name = "h3app_ref_x.png"
        self.assertIsNone(comfy_inputs.local_path(self.cfg, name))
        legacy = self.cfg.comfy_input
        legacy.mkdir(parents=True, exist_ok=True)
        (legacy / name).write_bytes(b"LEGACY")
        self.assertEqual(comfy_inputs.local_path(self.cfg, name), legacy / name)
        staged = comfy_inputs.stage_path(self.cfg, name)
        staged.write_bytes(b"STAGED")
        self.assertEqual(comfy_inputs.local_path(self.cfg, name), staged)

    def test_graph_input_names_unique_ordered(self):
        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "h3app_ref_a.png"}},
            "2": {"class_type": "LoadAudio", "inputs": {"audio": "h3app_ref_a.png"}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": "h3app_ref_b.mp4"}},
            "4": {"class_type": "LoadImage", "inputs": {"image": "not_h3app.png"}},
            "5": {"class_type": "Other", "inputs": {"image": "h3app_ref_c.png"}},
        }
        self.assertEqual(comfy_inputs.graph_input_names(graph),
                         ["h3app_ref_a.png", "h3app_ref_b.mp4"])


class FakeUploadServer:
    """Minimal stand-in for ComfyUI's POST /upload/image."""

    def __init__(self, *, mismatch=False):
        self.calls = []
        self.mismatch = mismatch

    async def handle(self, request):
        reader = await request.multipart()
        fields = {}
        image_bytes = b""
        image_name = ""
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "image":
                image_name = part.filename
                image_bytes = await part.read()
            else:
                fields[part.name] = (await part.read()).decode("utf-8")
        self.calls.append({"fields": fields, "image_name": image_name,
                           "image_bytes": image_bytes})
        name = "wrong_name.png" if self.mismatch else image_name
        return web.json_response({"name": name, "subfolder": "", "type": "input"})


class InputUploaderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-uploader-")
        self.root = Path(self.tmp.name)
        self.cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        self.cfg.data["input_stage_dir"] = str(self.root / "stage")
        self.cfg.data["comfy_dir"] = str(self.root / "comfy")

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def _fake_server(self, **kwargs):
        fake = FakeUploadServer(**kwargs)
        app = web.Application()
        app.router.add_post("/upload/image", fake.handle)
        server_ = TestServer(app)
        await server_.start_server()
        return fake, server_

    async def test_uploads_staged_file_with_expected_fields(self):
        fake, srv = await self._fake_server()
        try:
            uploader = comfy_inputs.InputUploader(self.cfg, str(srv.make_url("")))
            staged = comfy_inputs.stage_path(self.cfg, "h3app_ref_a.png")
            staged.write_bytes(b"PNGDATA")
            await uploader.ensure(["h3app_ref_a.png"])
            self.assertEqual(len(fake.calls), 1)
            call = fake.calls[0]
            self.assertEqual(call["fields"]["type"], "input")
            self.assertEqual(call["fields"]["subfolder"], "")
            self.assertEqual(call["fields"]["overwrite"], "true")
            self.assertEqual(call["image_name"], "h3app_ref_a.png")
            self.assertEqual(call["image_bytes"], b"PNGDATA")
        finally:
            await srv.close()

    async def test_cache_avoids_reupload(self):
        fake, srv = await self._fake_server()
        try:
            uploader = comfy_inputs.InputUploader(self.cfg, str(srv.make_url("")))
            staged = comfy_inputs.stage_path(self.cfg, "h3app_ref_a.png")
            staged.write_bytes(b"PNGDATA")
            await uploader.ensure(["h3app_ref_a.png"])
            await uploader.ensure(["h3app_ref_a.png"])
            self.assertEqual(len(fake.calls), 1)
            uploader.reset()
            await uploader.ensure(["h3app_ref_a.png"])
            self.assertEqual(len(fake.calls), 2)
        finally:
            await srv.close()

    async def test_name_mismatch_raises_comfy_upload_error(self):
        fake, srv = await self._fake_server(mismatch=True)
        try:
            uploader = comfy_inputs.InputUploader(self.cfg, str(srv.make_url("")))
            staged = comfy_inputs.stage_path(self.cfg, "h3app_ref_a.png")
            staged.write_bytes(b"PNGDATA")
            with self.assertRaises(PipelineError) as ctx:
                await uploader.ensure(["h3app_ref_a.png"])
            self.assertEqual(ctx.exception.kind, "comfy_upload")
        finally:
            await srv.close()

    async def test_legacy_comfy_input_file_needs_no_upload(self):
        fake, srv = await self._fake_server()
        try:
            uploader = comfy_inputs.InputUploader(self.cfg, str(srv.make_url("")))
            legacy = self.cfg.comfy_input
            legacy.mkdir(parents=True, exist_ok=True)
            (legacy / "h3app_ref_legacy.png").write_bytes(b"OLD")
            await uploader.ensure(["h3app_ref_legacy.png"])
            self.assertEqual(len(fake.calls), 0)
        finally:
            await srv.close()

    async def test_missing_file_raises_input_missing(self):
        fake, srv = await self._fake_server()
        try:
            uploader = comfy_inputs.InputUploader(self.cfg, str(srv.make_url("")))
            with self.assertRaises(PipelineError) as ctx:
                await uploader.ensure(["h3app_ref_missing.png"])
            self.assertEqual(ctx.exception.kind, "input_missing")
        finally:
            await srv.close()


class RunGraphHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_graph_ensures_inputs_before_gate(self):
        cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory(prefix="h3-rg-") as tmp:
            root = Path(tmp)
            cfg.data["comfy_dir"] = str(root / "comfy")
            cfg.data["input_stage_dir"] = str(root / "stage")
            client = ComfyClient(cfg)
            client.ensure_ready = AsyncMock()
            calls = []

            async def fake_ensure(names):
                calls.append(("ensure", list(names)))

            client.inputs.ensure = fake_ensure

            def fake_gate(graph):
                calls.append(("gate", None))
                raise RuntimeError("stop-before-network")

            client.gate = fake_gate
            graph = {
                "1": {"class_type": "LoadImage",
                      "inputs": {"image": "h3app_ref_a.png"}},
                "2": {"class_type": "LoadAudio",
                      "inputs": {"audio": "h3app_ref_b.wav"}},
                "3": {"class_type": "LoadVideo",
                      "inputs": {"file": "h3app_ref_c.mp4"}},
            }
            with self.assertRaises(RuntimeError):
                await client.run_graph(graph, on_event=lambda e: None)
            self.assertEqual(calls[0],
                             ("ensure", ["h3app_ref_a.png", "h3app_ref_b.wav",
                                        "h3app_ref_c.mp4"]))
            self.assertEqual(calls[1], ("gate", None))


# ------------------------------------------------------------------- HTTP --
class TestConfig(config.Config):
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


class StorageAndImagesHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-http-")
        self.root = Path(self.tmp.name)
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(self.root / "comfy"),
                   media_root=str(self.root / "media"),
                   input_stage_dir=str(self.root / "stage"),
                   auto_launch_comfy=False)
        with patch.object(server.presets_mod, "STORE_PATH",
                          self.root / "actions.json"):
            self.app = server.create_app(TestConfig(data, self.root))
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.cfg = self.app["cfg"]
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def _upload(self, files):
        """files: list of (fieldname, filename, bytes)."""
        from aiohttp import FormData
        form = FormData()
        for field, filename, payload in files:
            form.add_field(field, payload, filename=filename,
                           content_type="application/octet-stream")
        return await self.client.post("/api/images", data=form)

    async def test_upload_success_into_stage_dir(self):
        res = await self._upload([("files", "a.png", _tiny_png_bytes())])
        self.assertEqual(res.status, 200, await res.text())
        body = await res.json()
        self.assertTrue(body["ok"])
        name = body["images"][0]["name"]
        self.assertTrue(name.startswith("h3app_ref_"))
        self.assertTrue((self.cfg.input_stage_dir / name).is_file())
        # No leftover .part temp files.
        self.assertEqual(
            [p for p in self.cfg.input_stage_dir.glob(".upload_*.part")], [])

    async def test_atomic_rollback_when_second_file_corrupt(self):
        res = await self._upload([
            ("files", "good.png", _tiny_png_bytes()),
            ("files", "bad.png", b"NOT-AN-IMAGE"),
        ])
        self.assertEqual(res.status, 400)
        body = await res.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body.get("failed_file"), "bad.png")
        # Nothing from this request was kept.
        remaining = list(self.cfg.input_stage_dir.glob("h3app_ref_*"))
        self.assertEqual(remaining, [])
        remaining_parts = list(self.cfg.input_stage_dir.glob(".upload_*.part"))
        self.assertEqual(remaining_parts, [])

    async def test_size_limit_rejected(self):
        big = b"\x00" * (32 * 1024 * 1024 + 1)
        res = await self._upload([("files", "huge.png", big)])
        self.assertEqual(res.status, 400)
        body = await res.json()
        self.assertIn("32MB", body["message"])

    async def test_more_than_four_files_rejected(self):
        files = [("files", f"{i}.png", _tiny_png_bytes()) for i in range(5)]
        res = await self._upload(files)
        self.assertEqual(res.status, 400)
        remaining = list(self.cfg.input_stage_dir.glob("h3app_ref_*"))
        self.assertEqual(remaining, [])

    async def test_thumb_served_from_stage_dir(self):
        res = await self._upload([("files", "a.png", _tiny_png_bytes())])
        body = await res.json()
        name = body["images"][0]["name"]
        thumb_res = await self.client.get(f"/api/thumb/{name}")
        self.assertEqual(thumb_res.status, 200)
        self.assertEqual(thumb_res.content_type, "image/jpeg")

    async def test_generate_refuses_missing_image(self):
        res = await self.client.post("/api/generate", json={
            "images": ["h3app_ref_doesnotexist.png"],
            "jp_scene": "テスト", "jp_dialogue": ""})
        self.assertEqual(res.status, 400)
        body = await res.json()
        self.assertFalse(body["ok"])
        self.assertIn("見つかりません", body["message"])

    # --------------------------------------------------------- media/open --
    async def test_media_open_default_target_videos(self):
        with patch("server.os.startfile", create=True) as start:
            res = await self.client.post("/api/media/open", json={})
        self.assertEqual(res.status, 200, await res.text())
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["opened"])
        self.assertEqual(body["target"], "videos")
        self.assertEqual(Path(body["directory"]), self.cfg.media_videos)
        start.assert_called_once()

    async def test_media_open_validate_only_does_not_open(self):
        with patch("server.os.startfile", create=True) as start:
            res = await self.client.post(
                "/api/media/open", json={"target": "finals", "validate_only": True})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["opened"])
        start.assert_not_called()

    async def test_media_open_unknown_target_400(self):
        res = await self.client.post("/api/media/open", json={"target": "bogus"})
        self.assertEqual(res.status, 400)

    async def test_media_open_access_denied_mapping(self):
        exc = OSError("denied")
        exc.winerror = 5
        with patch("server.os.startfile", create=True, side_effect=exc):
            res = await self.client.post("/api/media/open", json={"target": "clips"})
        self.assertEqual(res.status, 500)
        body = await res.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["reason_code"], "access_denied")
        self.assertEqual(Path(body["directory"]), self.cfg.media_videos_clips)
        self.assertIn("アクセス拒否", body["message"])

    # ------------------------------------------------------------ storage --
    async def test_settings_storage_endpoint(self):
        res = await self.client.get("/api/settings/storage")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        for key in ("media_root", "videos_dir", "finals_dir", "clips_dir",
                   "images_dir", "input_stage_dir", "comfy_input_dir",
                   "output_prefix", "exists", "writable", "process"):
            self.assertIn(key, body)
        self.assertIn("videos", body["writable"])
        self.assertIn("input_stage", body["writable"])


class SystemInfoTests(unittest.TestCase):
    def test_never_raises_and_has_expected_keys(self):
        info = system_info_mod.process_identity()
        for key in ("user", "session_id", "console_session_id",
                   "interactive", "note"):
            self.assertIn(key, info)


# ------------------------------------------------------ story/single export --
class ExportRecordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_story_merge_records_final_export(self):
        with tempfile.TemporaryDirectory(prefix="h3-merge-") as tmp:
            root = Path(tmp)
            cfg = config.Config(copy.deepcopy(config.DEFAULT_CONFIG))
            cfg.data["comfy_dir"] = str(root / "comfy")
            cfg.data["media_root"] = str(root / "media")
            cfg.data["input_stage_dir"] = str(root / "stage")

            story_store = StoryStore(root / "stories")
            story = story_store.create(
                name="テスト", source_text="", lines=[],
                segments=[{"lines": [{"type": "prompt", "text": "s"}]}],
                images=[], settings={}, jp_negative="", motion_presets=[])
            clip_video = story.clips_dir / "clip_01.mp4"
            clip_video.parent.mkdir(parents=True, exist_ok=True)
            clip_video.write_bytes(b"FAKE-MERGED-INPUT")
            story.segments[0]["status"] = "done"
            story.segments[0]["clip"] = str(clip_video)
            story.save()

            fake_pipeline = SimpleNamespace(config=cfg)
            story_pipeline = StoryPipeline(fake_pipeline, story_store)

            async def fake_merge_clips(paths, out_path, **kwargs):
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"MERGED-BYTES")
                return out_path

            with patch("h3app.story_runner.merge_mod.merge_clips",
                      fake_merge_clips):
                result_path = await story_pipeline.merge(story)

            self.assertTrue(Path(result_path).is_file())
            export = story.data.get("final_export")
            self.assertIsNotNone(export)
            self.assertTrue(export["ok"], export)
            self.assertTrue(Path(export["path"]).is_file())
            self.assertTrue(str(Path(export["path"])).startswith(
                str(cfg.media_videos_final)))


# --------------------------------------------------- /api/story/merge HTTP --
class StoryMergeHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-merge-http-")
        self.root = Path(self.tmp.name)
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(self.root / "comfy"),
                   media_root=str(self.root / "media"),
                   input_stage_dir=str(self.root / "stage"),
                   auto_launch_comfy=False)
        with patch.object(server.presets_mod, "STORE_PATH",
                          self.root / "actions.json"):
            self.app = server.create_app(TestConfig(data, self.root))
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.cfg = self.app["cfg"]
        self.story_store = self.app["story_store"]
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

        self.story = self.story_store.create(
            name="テスト結合", source_text="", lines=[],
            segments=[{"lines": [{"type": "prompt", "text": "s"}]}],
            images=[], settings={}, jp_negative="", motion_presets=[])
        clip_video = self.story.clips_dir / "clip_01.mp4"
        clip_video.parent.mkdir(parents=True, exist_ok=True)
        clip_video.write_bytes(b"FAKE-CLIP-BYTES")
        seg_id = self.story.segments[0].get("segment_id") or ""
        self.story.segments[0]["status"] = "done"
        self.story.segments[0]["clip"] = str(clip_video)
        self.story.clips.append({
            "segment_index": 0, "segment_id": seg_id,
            "local_video": str(clip_video), "video": str(clip_video),
        })
        self.story.save()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def _merged_bytes(self, paths, out_path, **kwargs):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"MERGED-BYTES")
        return out_path

    async def test_merge_response_includes_successful_export(self):
        with patch("h3app.story_runner.merge_mod.merge_clips",
                  self._merged_bytes):
            res = await self.client.post(
                "/api/story/merge", json={"story_id": self.story.id})
        self.assertEqual(res.status, 200, await res.text())
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertIn("export", body)
        self.assertTrue(body["export"]["ok"], body["export"])
        self.assertTrue(Path(body["export"]["path"]).is_file())

    async def test_merge_response_includes_failed_export(self):
        failed = {"ok": False, "kind": "final", "path": "", "name": "",
                  "error": "書き込み権限がありません"}
        with patch("h3app.story_runner.merge_mod.merge_clips",
                  self._merged_bytes), \
             patch("h3app.media.export_video", return_value=failed):
            res = await self.client.post(
                "/api/story/merge", json={"story_id": self.story.id})
        self.assertEqual(res.status, 200, await res.text())
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertIn("export", body)
        self.assertFalse(body["export"]["ok"])
        self.assertIn("書き込み権限がありません", body["export"]["error"])

    async def test_merge_response_includes_seams_summary(self):
        """WP-C: a second clip makes merge_story() take the seam path (not the
        single-clip merge_clips shortcut); the response must carry the
        per-boundary report and needs_review alongside `export`."""
        clip_video2 = self.story.clips_dir / "clip_02.mp4"
        clip_video2.write_bytes(b"FAKE-CLIP-BYTES-2")
        self.story.segments.append({
            "segment_id": "seg-0002", "index": 1, "lines": [],
            "prompt": "s2", "speech": "", "status": "done",
            "clip": str(clip_video2),
            "transition": {"mode": "cut", "keep_pause": False},
        })
        self.story.clips.append({
            "segment_index": 1, "segment_id": "seg-0002",
            "local_video": str(clip_video2), "video": str(clip_video2),
        })
        self.story.save()

        fake_report = [{"index": 0, "mode": "cut", "needs_review": False}]

        async def _fake_merge_story(paths, out_path, **kwargs):
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"MERGED-WITH-SEAMS")
            self.assertEqual(kwargs.get("boundaries"), [
                {"mode": "cut", "keep_pause": False, "has_speech": False}])
            return {"path": out_path, "report": fake_report, "needs_review": False}

        with patch("h3app.story_runner.merge_mod.merge_story", _fake_merge_story):
            res = await self.client.post(
                "/api/story/merge", json={"story_id": self.story.id})
        self.assertEqual(res.status, 200, await res.text())
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertIn("export", body)
        self.assertIn("seams", body)
        self.assertEqual(body["seams"]["report"], fake_report)
        self.assertFalse(body["seams"]["needs_review"])


if __name__ == "__main__":
    unittest.main()
