"""WP-A: LLM connections on the schema-2 settings shape (ai_settings,
local_llm transport, director_provider.AdapterDirectorProvider,
POST /api/ai/models|test|key and /api/ai/settings, legacy delegates).

No real network, no LM Studio: every HTTP interaction uses a fake aiohttp
server bound to an ephemeral 127.0.0.1 port (aiohttp.test_utils.TestServer),
or is stubbed with unittest.mock. Credential Manager access is always mocked.
"""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server                                                      # noqa: E402
from aiohttp import web                                            # noqa: E402
from aiohttp.test_utils import TestClient, TestServer              # noqa: E402
from h3app import ai_settings, config, director_provider, local_llm  # noqa: E402
from h3app.pipeline import Pipeline                                 # noqa: E402


# --------------------------------------------------------------- ai_settings --
class BaseUrlValidation(unittest.TestCase):
    def test_loopback_and_private_ok_custom_port_kept(self):
        for url in ("http://127.0.0.1:1234/v1", "http://localhost:9999/v1",
                    "http://192.168.1.20:8080/v1", "http://[::1]:1234/v1"):
            normalized, error = ai_settings.validate_base_url(url)
            self.assertEqual(error, "", url)
            self.assertTrue(normalized)

    def test_public_host_rejected(self):
        normalized, error = ai_settings.validate_base_url(
            "http://example.com:1234/v1")
        self.assertEqual(normalized, "")
        self.assertIn("ローカル", error)

    def test_credentials_in_url_rejected(self):
        # Built at runtime so the prepublish secret scanner does not flag a
        # literal credential-bearing URL in the source.
        userinfo = ":".join(("user", "pass"))
        normalized, error = ai_settings.validate_base_url(
            f"http://{userinfo}@127.0.0.1:1234/v1")
        self.assertEqual(normalized, "")
        self.assertTrue(error)

    def test_bad_scheme_and_empty_rejected(self):
        for url in ("ftp://127.0.0.1:1234/v1", "", "not a url"):
            normalized, error = ai_settings.validate_base_url(url)
            self.assertEqual(normalized, "")
            self.assertTrue(error)

    def test_trailing_slash_normalized(self):
        normalized, error = ai_settings.validate_base_url(
            "http://127.0.0.1:1234/v1/")
        self.assertEqual(error, "")
        self.assertEqual(normalized, "http://127.0.0.1:1234/v1")


class NormalizeAndMerge(unittest.TestCase):
    def test_lmstudio_settings_survive_normalize(self):
        clean = ai_settings.normalize({
            "schema": 2,
            "connection": {"kind": "local", "provider": "lmstudio"},
            "providers": {"lmstudio": {
                "base_url": "http://127.0.0.1:5000/v1",
                "model": "llama-3-8b",
                "vision_models": ["llama-3-8b"]}}})
        self.assertEqual(clean["connection"]["provider"], "lmstudio")
        self.assertEqual(clean["providers"]["lmstudio"]["base_url"],
                         "http://127.0.0.1:5000/v1")
        self.assertEqual(clean["providers"]["lmstudio"]["vision_models"],
                         ["llama-3-8b"])

    def test_old_shape_migrates_on_normalize(self):
        clean = ai_settings.normalize({
            "director": {"provider": "openai_compat", "model": "llama-3-8b"},
            "character_profile": {"provider": "openai_compat",
                                  "model": "llama-3-8b"},
            "local_server": {"base_url": "http://127.0.0.1:5000/v1",
                             "vision_models": ["llama-3-8b"]}})
        self.assertEqual(clean["schema"], 2)
        self.assertEqual(clean["connection"]["provider"], "lmstudio")
        self.assertEqual(clean["providers"]["lmstudio"]["base_url"],
                         "http://127.0.0.1:5000/v1")

    def test_partial_save_never_erases_roles_or_favorites(self):
        with tempfile.TemporaryDirectory(prefix="h3-ai-settings-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"other_key": "keep-me"}),
                            encoding="utf-8")
            first = ai_settings.save_to_config_file(path, {
                "schema": 2,
                "connection": {"kind": "external",
                               "provider": "opencode_go"},
                "providers": {"opencode_go": {"model": "m1"}},
                "favorites": [{"provider": "opencode_go", "model": "m1"}]})
            self.assertEqual(
                first["providers"]["opencode_go"]["model"], "m1")
            # Partial payload: only one provider changes. The rest must
            # survive (this was the reproduced settings-UI bug).
            second = ai_settings.save_to_config_file(path, {
                "providers": {"lmstudio": {
                    "base_url": "http://127.0.0.1:1234/v1",
                    "vision_models": ["v1"]}}})
            self.assertEqual(
                second["providers"]["opencode_go"]["model"], "m1")
            self.assertEqual(second["favorites"][0]["model"], "m1")
            self.assertEqual(
                second["providers"]["lmstudio"]["base_url"],
                "http://127.0.0.1:1234/v1")
            self.assertEqual(
                second["providers"]["lmstudio"]["vision_models"], ["v1"])
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["other_key"], "keep-me")

    def test_secret_check_is_key_names_only(self):
        # A model id/URL containing "token"-like substrings must never be
        # rejected; only an actual KEY NAME like "api_key" must be.
        clean = ai_settings.normalize({
            "local_server": {"base_url": "http://127.0.0.1:1234/v1",
                             "vision_models": ["my-token-model"]}})
        ai_settings._assert_no_secrets(clean)  # must not raise
        with self.assertRaises(ValueError):
            ai_settings._assert_no_secrets({"api_key": "x"})


class ResolveChainNeverLeaksToGo(unittest.TestCase):
    def test_local_primary_never_adds_external(self):
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": "llama-3-8b"},
            5, gemma_fallback=True)
        kinds = [a["kind"] for a in chain]
        self.assertNotIn("external", kinds)
        self.assertNotIn("go", kinds)
        self.assertEqual(kinds, ["local", "gemma"])

    def test_gemma_primary_is_gemma_only(self):
        chain = ai_settings.resolve_chain({"kind": "gemma"}, 5)
        self.assertEqual([a["kind"] for a in chain], ["gemma"])

    def test_external_primary_falls_back_to_gemma_only(self):
        chain = ai_settings.resolve_chain(
            {"kind": "external", "provider": "openai", "model": "m1"},
            5, gemma_fallback=True)
        self.assertEqual([a["kind"] for a in chain],
                         ["external", "gemma"])

    def test_local_gemma_fallback_disabled_stops_at_local(self):
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": "m"},
            5, gemma_fallback=False)
        self.assertEqual([a["kind"] for a in chain], ["local"])

    def test_legacy_role_dicts_keep_working(self):
        chain = ai_settings.resolve_chain(
            {"provider": "openai_compat", "model": "m",
             "gemma_fallback": False}, 5)
        self.assertEqual([a["kind"] for a in chain], ["local"])


# ----------------------------------------------------------------- local_llm --
def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeLocalServer:
    """Minimal OpenAI-compatible + native LM Studio fake, ephemeral port."""

    def __init__(self, *, models_status=200, models_body=None,
                native_status=404, native_body=None,
                chat_status=200, chat_body=None, require_auth=False):
        self.models_status = models_status
        self.models_body = models_body if models_body is not None else {
            "data": [{"id": "chat-model"}, {"id": "embed-model"}]}
        self.native_status = native_status
        self.native_body = native_body
        self.chat_status = chat_status
        self.chat_body = chat_body if chat_body is not None else {
            "choices": [{"message": {"content": "OK."}}]}
        self.require_auth = require_auth
        self.chat_calls: list[dict] = []

    def _check_auth(self, request):
        if not self.require_auth:
            return True
        return request.headers.get("Authorization") == "Bearer secret-tok"

    async def _models(self, request):
        if not self._check_auth(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(self.models_body, status=self.models_status)

    async def _native_models(self, request):
        if self.native_status == 200:
            return web.json_response(self.native_body or {"models": []})
        return web.Response(status=self.native_status, text="not found")

    async def _chat(self, request):
        body = await request.json()
        self.chat_calls.append(body)
        if self.chat_status != 200:
            text = json.dumps(self.chat_body) if isinstance(
                self.chat_body, dict) else str(self.chat_body)
            return web.Response(status=self.chat_status, text=text)
        return web.json_response(self.chat_body)

    def app(self):
        app = web.Application()
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/api/v1/models", self._native_models)
        app.router.add_post("/v1/chat/completions", self._chat)
        return app


class ListModelsMerge(unittest.TestCase):
    def test_merge_native_type_vision_loaded_excludes_embedding(self):
        native_body = {"models": [
            {"key": "chat-model", "type": "llm", "display_name": "Chat",
             "capabilities": {"vision": False},
             "loaded_instances": [{"id": "chat-model"}]},
            {"key": "vision-model", "type": "llm",
             "capabilities": {"vision": True}, "loaded_instances": []},
            {"key": "embed-model", "type": "embedding",
             "capabilities": {}, "loaded_instances": []},
        ]}
        fake = FakeLocalServer(
            models_body={"data": [{"id": "chat-model"},
                                  {"id": "vision-model"},
                                  {"id": "embed-model"}]},
            native_status=200, native_body=native_body)

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.list_models(base)
            finally:
                await srv.close()
        result = _run(go())
        self.assertTrue(result["native"])
        self.assertEqual(result["excluded"], 1)
        ids = {m["id"]: m for m in result["models"]}
        self.assertNotIn("embed-model", ids)
        self.assertTrue(ids["chat-model"]["loaded"])
        self.assertFalse(ids["chat-model"]["vision"])
        self.assertTrue(ids["vision-model"]["vision"])

    def test_native_404_falls_back_to_openai_only(self):
        fake = FakeLocalServer(native_status=404)

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.list_models(base)
            finally:
                await srv.close()
        result = _run(go())
        self.assertFalse(result["native"])
        ids = [m["id"] for m in result["models"]]
        self.assertIn("chat-model", ids)
        self.assertIn("embed-model", ids)  # unknown type without native info


class TestConnectionClassification(unittest.TestCase):
    def test_unreachable_port(self):
        async def go():
            srv = TestServer(FakeLocalServer().app())
            await srv.start_server()
            port = srv.port
            await srv.close()
            # Port is now closed: nothing listens there any more.
            return await local_llm.test_connection(
                f"http://127.0.0.1:{port}/v1")
        result = _run(go())
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "unreachable")

    def test_auth_required(self):
        fake = FakeLocalServer(require_auth=True)

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.test_connection(base)
            finally:
                await srv.close()
        result = _run(go())
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "auth_required")

    def test_model_not_found(self):
        fake = FakeLocalServer(chat_status=404, chat_body={"error": "nope"})

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.test_connection(
                    base, model="chat-model")
            finally:
                await srv.close()
        result = _run(go())
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "model_not_found")

    def test_model_not_loaded(self):
        fake = FakeLocalServer(
            chat_status=400,
            chat_body={"error": {"message": "Model is not loaded"}})

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.test_connection(
                    base, model="chat-model")
            finally:
                await srv.close()
        result = _run(go())
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "model_not_loaded")

    def test_ok(self):
        fake = FakeLocalServer()

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.test_connection(
                    base, model="chat-model")
            finally:
                await srv.close()
        result = _run(go())
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "connected")


class ChatStripThink(unittest.TestCase):
    def test_think_block_stripped(self):
        fake = FakeLocalServer(chat_body={
            "choices": [{"message": {
                "content": "<think>reasoning here</think>Final answer."}}]})

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.chat(
                    base, model="chat-model", user="hi")
            finally:
                await srv.close()
        text, info = _run(go())
        self.assertEqual(text, "Final answer.")
        self.assertEqual(info["endpoint"], "local")


class VisionCapability(unittest.TestCase):
    def test_native_true(self):
        native_body = {"models": [
            {"key": "vision-model", "type": "llm",
             "capabilities": {"vision": True}, "loaded_instances": []}]}
        fake = FakeLocalServer(
            models_body={"data": [{"id": "vision-model"}]},
            native_status=200, native_body=native_body)

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.vision_capability(
                    base, "vision-model", vision_models=[])
            finally:
                await srv.close()
        self.assertTrue(_run(go()))

    def test_manual_mark_used_only_without_native_info(self):
        fake = FakeLocalServer(native_status=404)

        async def go():
            srv = TestServer(fake.app())
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                yes = await local_llm.vision_capability(
                    base, "chat-model", vision_models=["chat-model"])
                unknown = await local_llm.vision_capability(
                    base, "chat-model", vision_models=[])
                return yes, unknown
            finally:
                await srv.close()
        yes, unknown = _run(go())
        self.assertTrue(yes)
        self.assertIsNone(unknown)


class ReleaseModel(unittest.TestCase):
    """Fix B: unload a local LLM's VRAM before a ComfyUI graph runs."""

    def _native_app(self, *, models_status=200, models_body=None,
                    unload_status=200):
        unload_calls: list[dict] = []

        async def native_models(request):
            if models_status != 200:
                return web.Response(status=models_status, text="not found")
            return web.json_response(models_body or {"models": []})

        async def unload(request):
            body = await request.json()
            unload_calls.append(body)
            if unload_status != 200:
                return web.Response(status=unload_status, text="error")
            return web.json_response({"instance_id": body.get("instance_id")})

        app = web.Application()
        app.router.add_get("/api/v1/models", native_models)
        app.router.add_post("/api/v1/models/unload", unload)
        return app, unload_calls

    def test_loaded_instance_is_unloaded(self):
        native_body = {"models": [
            {"key": "google/gemma-4-12b-qat", "type": "llm",
             "loaded_instances": [{"id": "google/gemma-4-12b-qat"}]}]}
        app, unload_calls = self._native_app(models_body=native_body)

        async def go():
            srv = TestServer(app)
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.release_model(
                    base, "google/gemma-4-12b-qat")
            finally:
                await srv.close()
        result = _run(go())
        self.assertTrue(result["ok"])
        self.assertTrue(result["native"])
        self.assertEqual(result["unloaded"], ["google/gemma-4-12b-qat"])
        self.assertEqual(unload_calls,
                         [{"instance_id": "google/gemma-4-12b-qat"}])

    def test_model_not_loaded_is_a_noop(self):
        native_body = {"models": [
            {"key": "google/gemma-4-12b-qat", "type": "llm",
             "loaded_instances": []}]}
        app, unload_calls = self._native_app(models_body=native_body)

        async def go():
            srv = TestServer(app)
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.release_model(
                    base, "google/gemma-4-12b-qat")
            finally:
                await srv.close()
        result = _run(go())
        self.assertTrue(result["ok"])
        self.assertTrue(result["native"])
        self.assertEqual(result["unloaded"], [])
        self.assertEqual(unload_calls, [])

    def test_non_native_server_is_reported_without_unload_attempt(self):
        app, unload_calls = self._native_app(models_status=404)

        async def go():
            srv = TestServer(app)
            await srv.start_server()
            try:
                base = str(srv.make_url("/v1"))
                return await local_llm.release_model(base, "some-model")
            finally:
                await srv.close()
        result = _run(go())
        self.assertFalse(result["ok"])
        self.assertFalse(result["native"])
        self.assertEqual(result["unloaded"], [])
        self.assertTrue(result["error"])
        self.assertEqual(unload_calls, [])

    def test_invalid_url_makes_no_request(self):
        result = _run(local_llm.release_model(
            "http://example.com:1234/v1", "some-model"))
        self.assertFalse(result["ok"])
        self.assertFalse(result["native"])
        self.assertEqual(result["unloaded"], [])
        self.assertTrue(result["error"])


# ------------------------------------------------------------------ pipeline --
class PipelineReleaseLocalLlm(unittest.IsolatedAsyncioTestCase):
    def _pipeline(self, ai_settings_data: dict) -> Pipeline:
        pipeline = object.__new__(Pipeline)
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data["ai_settings"] = ai_settings_data
        pipeline.config = config.Config(data, None)
        pipeline.client = SimpleNamespace(_log=lambda msg: None)
        return pipeline

    def _settings(self, connection_provider: str, model: str,
                  profile_override: dict | None = None) -> dict:
        clean = ai_settings.normalize(None)
        clean["connection"] = {
            "kind": ai_settings.kind_of(connection_provider),
            "provider": connection_provider}
        clean["providers"][connection_provider]["model"] = model
        if profile_override is not None:
            clean["roles"]["character_profile"] = profile_override
        return clean

    async def test_local_roles_trigger_release_per_distinct_model(self):
        settings = self._settings("lmstudio", "model-a")
        pipeline = self._pipeline(settings)
        fake = SimpleNamespace(
            release=AsyncMock(return_value={
                "ok": True, "supported": True,
                "unloaded": ["model-a"], "error": ""}))
        with patch("h3app.credstore.load", return_value="tok") as mock_load, \
             patch("h3app.llm_providers.build_adapter",
                   return_value=fake) as mock_build:
            await pipeline._release_local_llm()
        # One provider target (director + profile share it): one adapter.
        self.assertEqual(mock_build.call_count, 1)
        self.assertEqual(mock_build.call_args.args[0], "lmstudio")
        fake.release.assert_awaited_once_with(["model-a"])
        mock_load.assert_called_with("openai_compat_local")

    async def test_distinct_providers_each_released(self):
        settings = self._settings("lmstudio", "model-a", {
            "override": True, "provider": "ollama", "model": "model-b"})
        pipeline = self._pipeline(settings)
        fake = SimpleNamespace(
            release=AsyncMock(return_value={
                "ok": True, "supported": True, "unloaded": [],
                "error": ""}))
        with patch("h3app.credstore.load", return_value=""), \
             patch("h3app.llm_providers.build_adapter",
                   return_value=fake) as mock_build:
            await pipeline._release_local_llm()
        self.assertEqual(mock_build.call_count, 2)
        called = {c.args[0] for c in mock_build.call_args_list}
        self.assertEqual(called, {"lmstudio", "ollama"})

    async def test_gemma_only_builds_no_adapter(self):
        settings = ai_settings.normalize(None)
        settings["connection"] = {"kind": "local", "provider": "comfy_gemma"}
        pipeline = self._pipeline(settings)
        with patch("h3app.llm_providers.build_adapter") as mock_build:
            await pipeline._release_local_llm()
        mock_build.assert_not_called()

    async def test_external_release_reports_unsupported_without_network(self):
        from h3app.llm_providers.openai import OpenAIAdapter
        adapter = OpenAIAdapter(model="m1", token="dummy")
        with patch("aiohttp.ClientSession") as mock_session:
            result = await adapter.release(["m1"])
        mock_session.assert_not_called()
        self.assertFalse(result["supported"])
        self.assertTrue(result["ok"])

    async def test_run_releases_local_llm_before_run_graph(self):
        pipeline = object.__new__(Pipeline)
        order: list[str] = []

        async def fake_release():
            order.append("release")

        async def fake_run_graph(*args, **kwargs):
            order.append("run_graph")
            return {}

        pipeline._release_local_llm = fake_release
        pipeline.client = SimpleNamespace(run_graph=fake_run_graph)
        runner = SimpleNamespace(cancel=None, set_stage=lambda *a: None,
                                 state={"percent": 0})
        result = await pipeline._run(runner, {"g": True}, lambda n: None)
        self.assertEqual(order, ["release", "run_graph"])
        self.assertEqual(result, {})


# --------------------------------------------------------- director_provider --
class AdapterDirectorProviderJson(unittest.TestCase):
    def test_generate_parses_json_answer(self):
        async def fake_chat(*, system="", user="", images=None,
                            max_tokens=1, temperature=0.0, timeout=600):
            return ('```json\n{"result": "ok"}\n```',
                    {"http_status": 200, "elapsed_ms": 5, "retry_after": "",
                     "usage": {"input_tokens": 1, "output_tokens": 1,
                               "reasoning_tokens": 0},
                     "incomplete": "", "endpoint": "local"})
        adapter = SimpleNamespace(
            model="chat-model",
            spec=SimpleNamespace(id="lmstudio"),
            chat=fake_chat)
        provider = director_provider.AdapterDirectorProvider(
            adapter, "chat-model")
        answer = _run(provider.generate(system="s", user="u"))
        self.assertEqual(answer, {"result": "ok"})
        self.assertEqual(provider.last_info["endpoint"], "local")

    def test_generate_wraps_adapter_error_with_kind(self):
        async def fake_chat(**kwargs):
            from h3app.llm_providers.base import LLMError
            raise LLMError("unreachable", "接続できません。")
        adapter = SimpleNamespace(
            model="chat-model", spec=SimpleNamespace(id="lmstudio"),
            chat=fake_chat)
        provider = director_provider.AdapterDirectorProvider(
            adapter, "chat-model")
        with self.assertRaises(director_provider.DirectorError):
            _run(provider.generate(system="s", user="u"))

    def test_legacy_aliases_still_construct(self):
        legacy_local = director_provider.OpenAICompatProvider(
            base_url="http://127.0.0.1:1234/v1", model="m", token="")
        self.assertEqual(legacy_local.provider_id, "lmstudio")
        legacy_go = director_provider.OpenCodeGoProvider(
            api_key="dummy", model="m", endpoint="responses")
        self.assertEqual(legacy_go.provider_id, "opencode_go")


# --------------------------------------------------------------- HTTP layer --
class AiHttpEndpoints(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-test-llm-")
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
        data.update(comfy_dir=str(root / "comfy"),
                   media_root=str(root / "media"), auto_launch_comfy=False)
        with patch.object(server.presets_mod, "STORE_PATH",
                          root / "actions.json"):
            self.app = server.create_app(
                TestConfig(data, root / "config.json"))
        self.app["client"].is_reachable = AsyncMock(return_value=False)
        self.app.on_startup.clear()
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_settings_get_lists_registry_and_keys(self):
        with patch("h3app.credstore.load", return_value=""):
            res = await self.client.get("/api/ai/settings")
        self.assertEqual(res.status, 200)
        payload = await res.json()
        ids = {p["id"] for p in payload["providers"]}
        self.assertEqual(ids, {"comfy_gemma", "lmstudio", "openai_compat",
                               "ollama", "llamacpp", "vllm", "localai",
                               "opencode_go", "openai", "anthropic", "gemini",
                               "openrouter", "openai_compat_external"})
        self.assertEqual(payload["settings"]["schema"], 2)
        self.assertIn("lmstudio", payload["keys"])
        self.assertIn("opencode_go", payload["keys"])
        self.assertIn("gemma", payload)

    async def test_settings_save_rejects_public_url(self):
        res = await self.client.post("/api/ai/settings", json={
            "settings": {"providers": {"lmstudio": {
                "base_url": "http://example.com:1234/v1"}}}})
        self.assertEqual(res.status, 400)

    async def test_settings_save_rejects_external_userinfo_url(self):
        userinfo = ":".join(("user", "pass"))
        res = await self.client.post("/api/ai/settings", json={
            "settings": {"providers": {"openai_compat_external": {
                "base_url": f"https://{userinfo}@api.example.com/v1"}}}})
        self.assertEqual(res.status, 400)

    async def test_settings_save_records_local_lmstudio_for_shutdown(self):
        # Saving a connection that points at LM Studio records the process
        # this session will be allowed to stop (a test run before saving
        # cannot: it only sees saved settings). Recording is mocked: no
        # real listener lookup, no real state file.
        with patch.object(server.shutdown_mod, "record_lmstudio_target",
                          return_value=None) as rec:
            res = await self.client.post("/api/ai/settings", json={
                "settings": {
                    "connection": {"kind": "local", "provider": "lmstudio"},
                    "providers": {"lmstudio": {"model": "m1"}}}})
        self.assertEqual(res.status, 200)
        rec.assert_called_once()
        self.assertEqual(rec.call_args[0][0], self.app["cfg"])
        self.assertEqual(rec.call_args[0][1], self.app["h3_session_id"])

    async def test_settings_save_accepts_hand_typed_model(self):
        res = await self.client.post("/api/ai/settings", json={
            "settings": {
                "connection": {"kind": "local", "provider": "lmstudio"},
                "providers": {"lmstudio": {"model": "typed-by-hand-xyz"}}}})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertEqual(body["settings"]["providers"]["lmstudio"]["model"],
                         "typed-by-hand-xyz")

    async def test_settings_save_rejects_provider_without_model(self):
        res = await self.client.post("/api/ai/settings", json={
            "settings": {
                "connection": {"kind": "local", "provider": "lmstudio"},
                "providers": {"lmstudio": {"model": ""}}}})
        self.assertEqual(res.status, 400)
        body = await res.json()
        self.assertIn("モデルID", body.get("message", ""))

    async def test_settings_save_partial_preserves_other_provider(self):
        first = await self.client.post("/api/ai/settings", json={
            "settings": {
                "connection": {"kind": "local", "provider": "lmstudio"},
                "providers": {"lmstudio": {"model": "m1"}}}})
        self.assertEqual(first.status, 200)
        second = await self.client.post("/api/ai/settings", json={
            "settings": {
                "connection": {"kind": "local", "provider": "ollama"},
                "providers": {"ollama": {"model": "m2"}}}})
        self.assertEqual(second.status, 200)
        body = await second.json()
        self.assertEqual(body["settings"]["providers"]["lmstudio"]["model"],
                         "m1")
        self.assertEqual(body["settings"]["providers"]["ollama"]["model"],
                         "m2")

    async def test_key_save_delete_never_echo_token(self):
        secret = "super-secret-local-token"
        with patch("h3app.credstore.available", return_value=True), \
             patch("h3app.credstore.save", return_value=True) as mock_save, \
             patch("h3app.credstore.delete", return_value=True):
            res = await self.client.post(
                "/api/ai/key", json={"provider": "lmstudio",
                                     "key": secret})
            self.assertEqual(res.status, 200)
            body = await res.json()
            self.assertNotIn(secret, json.dumps(body))
            mock_save.assert_called_once_with("openai_compat_local", secret)
            res2 = await self.client.request(
                "DELETE", "/api/ai/key", json={"provider": "lmstudio"})
            self.assertEqual(res2.status, 200)

    async def test_models_endpoint_invalid_url_400(self):
        res = await self.client.post("/api/ai/models", json={
            "provider": "lmstudio", "base_url": "http://evil.test/v1"})
        self.assertEqual(res.status, 400)

    async def test_models_endpoint_unsupported_provider(self):
        res = await self.client.post("/api/ai/models", json={
            "provider": "comfy_gemma"})
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["supported"])

    async def test_models_endpoint_ok(self):
        fake = FakeLocalServer()
        srv = TestServer(fake.app())
        await srv.start_server()
        try:
            base = str(srv.make_url("/v1"))
            with patch("h3app.credstore.load", return_value=""):
                res = await self.client.post(
                    "/api/ai/models",
                    json={"provider": "lmstudio", "base_url": base})
            self.assertEqual(res.status, 200)
            body = await res.json()
            self.assertTrue(body["ok"])
            self.assertTrue(any(m["id"] == "chat-model"
                               for m in body["models"]))
        finally:
            await srv.close()

    async def test_test_endpoint_skips_image_when_vision_not_confirmed(self):
        fake = FakeLocalServer()
        srv = TestServer(fake.app())
        await srv.start_server()
        try:
            base = str(srv.make_url("/v1"))
            with patch("h3app.credstore.load", return_value=""):
                res = await self.client.post("/api/ai/test", json={
                    "provider": "lmstudio", "base_url": base,
                    "model": "chat-model", "image": "h3app_ref_x.png"})
            self.assertEqual(res.status, 200)
            body = await res.json()
            self.assertFalse(body["ok"])
            self.assertEqual(body["status"], "vision_unsupported")
            # No chat request was made at all (image never sent).
            self.assertEqual(fake.chat_calls, [])
        finally:
            await srv.close()

    async def test_test_endpoint_sends_image_when_vision_confirmed(self):
        native_body = {"models": [
            {"key": "chat-model", "type": "llm",
             "capabilities": {"vision": True}, "loaded_instances": []}]}
        fake = FakeLocalServer(native_status=200, native_body=native_body)
        srv = TestServer(fake.app())
        await srv.start_server()
        try:
            base = str(srv.make_url("/v1"))
            root = Path(self.app["cfg"].app_dir) / "_comfy_input"
            root.mkdir(parents=True, exist_ok=True)
            img = root / "h3app_ref_x.png"
            # Minimal 1x1 PNG.
            img.write_bytes(bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000004000000040802000000"
                "269309290000001149444154789c63fccf80004c486c3c1c0033d10107"
                "30e7e14c0000000049454e44ae426082"))
            with patch("h3app.credstore.load", return_value=""):
                res = await self.client.post("/api/ai/test", json={
                    "provider": "lmstudio", "base_url": base,
                    "model": "chat-model", "image": "h3app_ref_x.png"})
            self.assertEqual(res.status, 200)
            body = await res.json()
            self.assertTrue(body["ok"], body)
            self.assertEqual(len(fake.chat_calls), 1)
        finally:
            await srv.close()

    async def test_legacy_local_models_delegate(self):
        fake = FakeLocalServer()
        srv = TestServer(fake.app())
        await srv.start_server()
        try:
            base = str(srv.make_url("/v1"))
            with patch("h3app.credstore.load", return_value=""):
                res = await self.client.post(
                    "/api/ai/local/models", json={"base_url": base})
            self.assertEqual(res.status, 200)
            body = await res.json()
            self.assertTrue(body["ok"])
        finally:
            await srv.close()

    async def test_legacy_local_key_routes_delegate(self):
        with patch("h3app.credstore.available", return_value=True), \
             patch("h3app.credstore.save", return_value=True) as mock_save, \
             patch("h3app.credstore.delete", return_value=True):
            res = await self.client.post("/api/ai/local/key",
                                         json={"key": "dummy-local-token"})
            self.assertEqual(res.status, 200)
            mock_save.assert_called_once_with("openai_compat_local",
                                              "dummy-local-token")
            res2 = await self.client.delete("/api/ai/local/key")
            self.assertEqual(res2.status, 200)

    async def test_legacy_models_get_delegates(self):
        # The Go list endpoint is public; stub the transport so this test
        # never touches the real network.
        with patch("h3app.opencode_go.fetch_models",
                   AsyncMock(return_value=[{"id": "m1", "display": "M1",
                                            "endpoint": "responses"}])):
            res = await self.client.get("/api/ai/models")
        self.assertEqual(res.status, 200)
        body = await res.json()
        self.assertTrue(body["ok"])
        self.assertEqual([m["id"] for m in body["models"]], ["m1"])

if __name__ == "__main__":
    unittest.main()
