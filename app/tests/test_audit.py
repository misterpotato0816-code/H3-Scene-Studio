"""CPU-only regressions for the September audit; no models or user media."""
import asyncio
import copy
import inspect
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from h3app import (config, director_lexicon, director_provider,
                   director_review, director_spec)
from h3app import director_speech, local_security, opencode_go, speech_guard, story
from h3app import voice_rehearsal as voice
from h3app.errors import PipelineError
from h3app.pipeline import Pipeline
from h3app.projects import ProjectStore


def prompt(line="今日は、いい天気だね。"):
    return "video_description\nA quiet scene.\ndetailed_description\n" + (
        f"<Subject 1> (S1) speaks naturally and says, <d>[Japanese]{line}</d>\n"
        if line else "A person watches the sky.\n") + "soundscape\nSoft wind."


def query():
    return {"accent_phrases": [{"moras": [{"text": "ア"}, {"text": "メ"}], "accent": 1}],
            "kana": "ア'メ", "speedScale": 1, "intonationScale": 1}


class Contracts(unittest.TestCase):
    def test_ambience_words_inside_exact_dialogue_survive(self):
        line = "あの人の声、太鼓で聞こえないね。"
        cleaned, report = speech_guard.prepare(prompt(line) + "\n人の声", line)
        self.assertIn(f"<d>[Japanese]{line}</d>", cleaned)
        self.assertNotIn("人の声", director_speech._strip_d_tags(cleaned))
        self.assertFalse(report["hard"])

    def test_speech_policy_is_idempotent(self):
        first, _ = speech_guard.prepare(prompt(), "今日は、いい天気だね。", delivery="warm")
        second, _ = speech_guard.prepare(first, "今日は、いい天気だね。", delivery="warm")
        self.assertEqual(first, second)
        self.assertEqual(first.count("Speech performance:"), 1)

    def test_wrong_or_unscripted_dialogue_stops(self):
        for spoken in ("", "明日は雨だね。"):
            with self.subTest(spoken=spoken), self.assertRaises(PipelineError):
                speech_guard.prepare(prompt(), spoken)

    def test_malformed_and_foreign_tags_stop(self):
        for tag in ("<d", "<d>", "</d>", "<D>[Japanese]はい</D>",
                    "<d>[English]hello</d>", "<d>[Japanese]はい<d>ね</d>"):
            with self.subTest(tag=tag), self.assertRaises(PipelineError):
                speech_guard.prepare(prompt("") + tag, "")

    def test_quoted_unscripted_english_stops(self):
        with self.assertRaises(PipelineError):
            speech_guard.prepare(prompt("") + '\nShe says "hello".', "")

    def test_second_speaker_requires_explicit_cast(self):
        with self.assertRaises(PipelineError):
            speech_guard.prepare(prompt().replace("<Subject 1>", "<Subject 2>"), "今日は、いい天気だね。")

    def test_silent_scene_has_no_performance_instruction(self):
        result, _ = speech_guard.prepare(prompt(""), "")
        self.assertNotIn("Speech performance:", result)

    def test_silent_director_prompt_does_not_request_invented_dialogue(self):
        spec = director_spec.new_single(idea="無言", duration_sec=5)
        spec["items"]["acting"].update(
            value="静かに微笑む", en="smiles quietly", locked=True)
        spec["items"]["voice"].update(
            value="明るい声", en="a bright voice", locked=True)
        result = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"])
        self.assertNotIn("spoken dialogue", result)
        self.assertNotIn("<d>", result)
        self.assertIn("remains silent with relaxed, gently closed lips", result)
        self.assertNotIn("Voice:", result)
        self.assertNotIn("a bright voice", result)
        self.assertIn("No dialogue or speech-like vocalization.", result)

    def test_director_speech_and_silence_edges(self):
        spec = director_spec.new_single(idea="会話", duration_sec=5)
        spec["items"]["dialogue"].update(
            value="うん", en="", locked=True)
        spoken = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"])
        lip_sync = ("Natural synchronized mouth movement matching the spoken "
                    "dialogue, with accurate lip timing and clear articulation.")
        self.assertEqual(spoken.count(lip_sync), 1)
        self.assertEqual(spoken.count("<d>[Japanese] うん</d>"), 1)

        spec["items"]["dialogue"].update(value="  \n\t")
        whitespace = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"])
        self.assertNotIn("<d>", whitespace)
        self.assertNotIn("spoken dialogue", whitespace)
        self.assertIn("remains silent", whitespace)

    def test_two_subject_and_continuation_silence_contracts(self):
        spec = director_spec.new_single(idea="二人で無言", duration_sec=5)
        spec["cast"]["on_screen_subjects"] = ["character:0", "character:1"]
        two_subjects = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"])
        self.assertIn("<Subject 1> and <Subject 2> remain silent", two_subjects)
        self.assertIn("closed lips throughout", two_subjects)

        continuation = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"],
            refs_available={"videos": 1, "audios": 1,
                            "audio_kind": "video_soundtrack"})
        self.assertIn(
            "preceding video's ending pose is carried into the first frame "
            "of [Shot 1]", continuation)
        self.assertNotIn("continuation source's first-frame pose", continuation)
        self.assertIn("lips settle gently closed", continuation)
        self.assertNotIn("closed lips throughout", continuation)

    def test_silent_voice_master_is_binding_only(self):
        spec = director_spec.new_single(idea="Voice Master付き無言", duration_sec=5)
        refs_available = {"videos": 0, "audios": 1,
                          "audio_kind": "voice_master"}
        silent = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"], refs_available=refs_available)
        self.assertIn("<Audio 1>", silent)
        self.assertIn("fixed voice reference", silent)
        self.assertIn("does not define or request timbre, speech, or vocal output", silent)
        self.assertIn("reference binding only for this silent clip", silent)
        self.assertNotIn("defining the speaker's timbre", silent)
        self.assertNotIn("speaker's timbre is carried forward", silent)

        spec["items"]["dialogue"].update(value="うん", en="", locked=True)
        spoken = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"], refs_available=refs_available)
        self.assertIn("defining the speaker's timbre for this clip", spoken)
        self.assertIn("speaker's timbre is carried forward", spoken)
        self.assertIn("<d>[Japanese] うん</d>", spoken)

    def test_photographer_role_translates_and_never_read_aloud(self):
        # Regression: bare 撮影者 survived to_english, failed the English
        # conversion check, and risked leaking into speech. It must behave
        # exactly like the analogous 恋人 precedent.
        self.assertEqual(
            director_lexicon.to_english("撮影者"), "camera operator")
        self.assertEqual(
            director_lexicon.to_english("カメラマン"), "camera operator")
        translated = director_lexicon.to_english("撮影者恋人")
        self.assertNotIn("撮影者", translated)
        self.assertEqual(director_lexicon.untranslated_runs(translated), [])

        spec = director_spec.new_single(idea="撮影", duration_sec=5)
        spec["cast"]["camera_operator"] = "撮影者"
        spec["cast"]["camera_operator_visibility"] = "off_camera"
        spec["items"]["acting"].update(
            value="静かに微笑む", en="smiles quietly", locked=True)
        compiled = director_lexicon.compile_direct_prompt(
            items=spec["items"], identity={}, outfit_ref={}, refs=["ref.png"],
            frames=124, cast=spec["cast"])
        self.assertNotIn("撮影者", compiled)
        self.assertNotIn("カメラマン", compiled)
        # Off-camera operator becomes a style phrase, never a visible
        # second person or a spoken line.
        self.assertIn("camera operator stays off-camera", compiled)
        self.assertNotIn("<Subject 2>", compiled)
        for spoken in re.findall(
                r"<d>\[Japanese\](.*?)</d>", compiled, flags=re.S):
            self.assertNotIn("撮影者", spoken)
        self.assertTrue(
            director_lexicon.validate_compiled_prompt(compiled)["ok"])

    def test_parser_uses_final_not_largest_or_first_fenced_example(self):
        for text in ('{"example":{"long":"ignored"}}\n{"final":true}',
                     '```json\n{"example":true}\n```\n{"final":true}',
                     'reasoning {broken\n{"final":true}'):
            with self.subTest(text=text):
                self.assertEqual(director_provider.parse_json_answer(text), {"final": True})

    def test_timeline_rejects_overlap_and_nonfinite(self):
        for timeline in ([{"t0": 1, "t1": 3}], [{"t0": 0, "t1": 6}],
                         [{"t0": 0, "t1": 3}, {"t0": 2, "t1": 4}],
                         [{"t0": 0, "t1": float("nan")}], [{"t0": None, "t1": 4}]):
            self.assertTrue(director_review.timeline_errors(timeline, 5))
        self.assertFalse(director_review.timeline_errors([{"t0": 0, "t1": 2}, {"t0": 2, "t1": 5}], 5))

    def test_review_never_claims_audio_was_verified(self):
        result = director_review.review(director_spec.new_single(idea="海", duration_sec=5))
        self.assertTrue(result["ok"])
        self.assertFalse(result["audio_verified"])
        self.assertTrue(result["lines"][0]["silent"])

    def test_content_edit_preserves_locks_and_server_state(self):
        old = story.normalize_segments([{"prompt": "A scene", "speech": "はい"}])
        old[0].update(locked="approved", status="done", clip={"video": "kept.mp4"}, direct_en_prompt="cached")
        incoming = story.normalize_segments(old)
        incoming[0].update(status="pending", clip=None, direct_en_prompt="injected")
        result = story.content_edit(old, incoming)
        self.assertEqual(result, old)
        incoming[0]["speech"] = "変更"
        with self.assertRaises(ValueError):
            story.content_edit(old, incoming)

    def test_content_edit_invalidates_stale_compiled_dialogue(self):
        old = story.normalize_segments([{"prompt": "A scene", "speech": "はい"}])
        old[0]["direct_en_prompt"] = "cached"
        incoming = copy.deepcopy(old)
        incoming[0]["speech"] = "いいえ"
        self.assertNotIn("direct_en_prompt", story.content_edit(old, incoming)[0])
        incoming[0]["segment_id"] = "different"
        with self.assertRaises(ValueError):
            story.content_edit(old, incoming)

    def test_structure_edit_cannot_erase_or_invalidate_approved_clip(self):
        segments = story.normalize_segments([{"prompt": "first"}, {"prompt": "second"}])
        segments[1]["locked"] = "approved"
        obj = story.StoryProject({"id": "abc123", "base_seed": 1, "clips": [], "cursor": 0}, segments, Path("unused"))
        plan = story.plan_structure(obj, [{"segment_id": s["segment_id"]} for s in segments])
        self.assertEqual(plan["segments"][1]["locked"], "approved")
        for requested in ([{"segment_id": segments[0]["segment_id"]}],
                          [{"segment_id": segments[0]["segment_id"], "prompt": "changed"},
                           {"segment_id": segments[1]["segment_id"]}]):
            with self.assertRaises(story.StructureError):
                story.plan_structure(obj, requested)

    def test_structure_content_change_invalidates_matching_clip(self):
        segments = story.normalize_segments([{"prompt": "first", "direct_en_prompt": "stale"}])
        obj = story.StoryProject({"id": "abc123", "base_seed": 1, "clips": [{"segment_index": 0}], "cursor": 1}, segments, Path("unused"))
        plan = story.plan_structure(obj, [{"segment_id": segments[0]["segment_id"], "prompt": "changed"}])
        self.assertEqual(plan["divergence"], 0)
        self.assertEqual(len(plan["invalid_clips"]), 1)
        self.assertNotIn("direct_en_prompt", plan["segments"][0])

    def test_windows_and_posix_output_escape_rejected(self):
        for value in ("../out", "H3/../../out", "C:\\private", "\\\\server\\share", "/tmp/out", "H3//out"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                local_security.output_prefix(value)
        self.assertEqual(local_security.output_prefix("H3\\APP"), "H3/APP")

    def test_static_prefix_sibling_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "web"
            root.mkdir()
            sibling = Path(tmp) / "web-private"
            sibling.mkdir()
            secret = sibling / "secret.txt"
            secret.write_text("test fixture")
            self.assertIsNone(local_security.contained_file(root, "../web-private/secret.txt"))
            linked = root / "linked"
            if os.name == "nt":
                # File symlinks require Developer Mode or SeCreateSymbolicLinkPrivilege
                # on Windows. A directory junction exercises the same resolved-path
                # escape without requiring either, so this security test remains live
                # on a normal production machine instead of failing during setup.
                subprocess.run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(linked), str(sibling)],
                    check=True, capture_output=True,
                )
                self.assertEqual(linked.resolve(), sibling.resolve())
                escaped = "linked/secret.txt"
            else:
                linked.symlink_to(secret)
                self.assertEqual(linked.resolve(), secret.resolve())
                escaped = "linked"
            self.assertIsNone(local_security.contained_file(root, escaped))

    def test_project_id_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            for value in ("../other", "/absolute", "C:\\other"):
                self.assertIsNone(store.get(value))
                with self.assertRaises(ValueError):
                    store.dir_for(value)

    def test_key_endpoint_exact_host_and_path(self):
        for url in ("https://evil.test/zen/go/v1/models", "http://opencode.ai/zen/go/v1/models",
                    "https://opencode.ai.evil.test/zen/go/v1/models", "https://opencode.ai/zen/v1/models"):
            with self.assertRaises(opencode_go.GoError):
                opencode_go._check_url(url)
        opencode_go._check_url("https://opencode.ai/zen/go/v1/models")

    def test_config_defaults_do_not_leak_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent.json"
            a, b = config.load_config(path), config.load_config(path)
            a.defaults["steps"] = 1
            self.assertEqual(b.defaults["steps"], 8)

    def test_voice_query_bounds_and_copy(self):
        original = query()
        normalized = voice.normalize_query(original)
        self.assertIn("kana", original)
        self.assertNotIn("kana", normalized)
        self.assertEqual(normalized["outputSamplingRate"], 24000)
        for key, value in (("speedScale", 0), ("pitchScale", float("nan")), ("intonationScale", 3)):
            with self.assertRaises(voice.RehearsalError):
                voice.normalize_query({**query(), key: value})


class BoundaryHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = web.Application(middlewares=[local_security.local_boundary])
        app.router.add_route("*", "/api/check", lambda r: web.json_response({"ok": True}))
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_same_origin_json_allowed(self):
        origin = str(self.client.make_url("/")).rstrip("/")
        res = await self.client.post("/api/check", json={}, headers={"Origin": origin})
        self.assertEqual(res.status, 200)
        self.assertEqual(res.headers["X-Content-Type-Options"], "nosniff")

    async def test_cross_origin_and_rebinding_rejected(self):
        for headers in ({"Origin": "https://evil.test"}, {"Host": "evil.test"},
                        {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}):
            res = await self.client.post("/api/check", json={}, headers=headers)
            self.assertEqual(res.status, 403)

    async def test_simple_form_post_rejected(self):
        res = await self.client.post("/api/check", data={"operation": "write"})
        self.assertEqual(res.status, 415)


class AppHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-test-")
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
        data.update(comfy_dir=str(root / "comfy"), media_root=str(root / "media"), auto_launch_comfy=False)
        with patch.object(server.presets_mod, "STORE_PATH", root / "actions.json"):
            self.app = server.create_app(TestConfig(data))
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_real_heartbeat_and_static_assets(self):
        self.assertEqual((await self.client.post("/api/heartbeat", json={})).status, 200)
        for url in ("/", "/static/app.js", "/static/studio.js", "/static/studio.css", "/static/settings.js"):
            res = await self.client.get(url)
            self.assertEqual(res.status, 200)
            self.assertIn("no-store", res.headers["Cache-Control"])

    async def test_content_save_keeps_omitted_delivery_and_approved_lock(self):
        segments = story.normalize_segments([{"prompt": "海を見る", "speech": "いいね。",
                                               "delivery": {"pace": "slow"}}])
        obj = self.app["story_store"].create(name="fixture", source_text="", lines=[],
            segments=segments, images=[], settings={}, jp_negative="", motion_presets=[])
        obj.segments[0]["locked"] = "approved"
        obj.save()
        body = {"story_id": obj.id, "segments": [{"segment_id": obj.segments[0]["segment_id"],
                                                  "prompt": "海を見る", "speech": "いいね。"}]}
        res = await self.client.post("/api/story/segments", json=body)
        self.assertEqual(res.status, 200, await res.text())
        saved = self.app["story_store"].get(obj.id).segments[0]
        self.assertEqual(saved["locked"], "approved")
        self.assertEqual(saved["delivery"], segments[0]["delivery"])
        body["segments"][0]["speech"] = "別の台詞"
        self.assertEqual((await self.client.post("/api/story/segments", json=body)).status, 409)

    async def test_direction_review_returns_actionable_error(self):
        spec = director_spec.new_single(idea="海", duration_sec=5)
        spec["timeline"] = [{"t0": 0, "t1": 9}]
        res = await self.client.post("/api/director/review", json={"spec": spec})
        self.assertEqual(res.status, 400)
        self.assertTrue((await res.json())["errors"])

    async def test_voice_routes_binary_and_engine_unavailable(self):
        wav = b"RIFF" + bytes(4) + b"WAVEfixture"
        with patch.object(voice.VoiceRehearsal, "synthesize", AsyncMock(return_value=wav)):
            res = await self.client.post("/api/voice-rehearsal/preview", json={"query": query(), "speaker": 0})
            self.assertEqual(res.status, 200)
            self.assertEqual(res.content_type, "audio/wav")
            self.assertEqual(await res.read(), wav)
        with patch.object(voice.VoiceRehearsal, "speakers", AsyncMock(side_effect=voice.RehearsalError("起動してください"))):
            res = await self.client.get("/api/voice-rehearsal/speakers")
            self.assertEqual(res.status, 503)
            self.assertIn("起動", (await res.json())["message"])


class AsyncContracts(unittest.IsolatedAsyncioTestCase):
    async def test_precompiled_director_does_not_require_local_vlm(self):
        pipeline = object.__new__(Pipeline)
        pipeline._run = AsyncMock(side_effect=AssertionError("unexpected VLM graph"))
        pipeline._run_generate = AsyncMock()
        pipeline._record_consumed = Mock()
        project = SimpleNamespace(data={"en_prompt": prompt()}, save=Mock())
        await pipeline.generate_director(SimpleNamespace(project=project, set_stage=Mock()))
        pipeline._run.assert_not_awaited()
        pipeline._run_generate.assert_awaited_once()

    async def test_busy_gpu_does_not_start_or_leak_coroutine(self):
        pipeline = object.__new__(Pipeline)
        pipeline._lock = asyncio.Lock()
        pipeline.active = SimpleNamespace(state={"finished": False})
        job = AsyncMock()
        coro = job()
        with self.assertRaises(PipelineError):
            await pipeline.run_audio_test(coro)
        job.assert_not_awaited()
        self.assertEqual(inspect.getcoroutinestate(coro), inspect.CORO_CLOSED)

    async def test_video_guard_rejects_before_gpu_access(self):
        pipeline = object.__new__(Pipeline)
        project = SimpleNamespace(data={"jp_dialogue": "いいえ", "settings": {}})
        with self.assertRaises(PipelineError):
            await pipeline._run_generate(SimpleNamespace(state={}), project, prompt(), step=1, continuation=None)

    async def test_voice_accent_edit_recalculates_before_synthesis(self):
        service = voice.VoiceRehearsal()
        wav = b"RIFF" + bytes(4) + b"WAVEfixture"
        service._request = AsyncMock(side_effect=[[{"recalculated": True}], wav])
        result = await service.synthesize(query(), 0)
        self.assertEqual(result, wav)
        calls = service._request.call_args_list
        self.assertEqual([c.args[1] for c in calls], ["/mora_data", "/synthesis"])
        self.assertEqual(calls[1].kwargs["data"]["accent_phrases"], [{"recalculated": True}])

    async def test_voice_http_rejects_redirect_and_invalid_wav(self):
        app = web.Application()
        app.router.add_get("/redirect", lambda r: web.HTTPFound("/unexpected"))
        app.router.add_get("/bad", lambda r: web.Response(body=b"not audio"))
        async with TestServer(app) as fixture:
            with patch.object(voice, "BASE", str(fixture.make_url("/")).rstrip("/")):
                for path in ("/redirect", "/bad"):
                    with self.assertRaises(voice.RehearsalError):
                        await voice.VoiceRehearsal()._request("GET", path, binary=True)


if __name__ == "__main__":
    unittest.main()
