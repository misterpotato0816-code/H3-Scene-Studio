# -*- coding: utf-8 -*-
"""WP-A: user-selectable LLM providers (registry, schema-2 settings, adapters,
fallback discipline, pipeline wiring).

No real network, GPU or processes: HTTP is an ephemeral aiohttp TestServer,
credentials are dummies, credstore is mocked.
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
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from h3app import ai_settings  # noqa: E402
from h3app import config as config_mod  # noqa: E402
from h3app import director_provider  # noqa: E402
from h3app.llm_providers import (REGISTRY, build_adapter, for_kind, get,  # noqa: E402
                                 key_name)
from h3app.llm_providers.base import LLMError  # noqa: E402
from h3app.pipeline import Pipeline  # noqa: E402

DUMMY = "dummy-secret-for-tests"

SCHEMA2_GO = {
    "schema": 2,
    "connection": {"kind": "external", "provider": "opencode_go"},
    "roles": {
        "director": {"override": False, "provider": "", "model": ""},
        "character_profile": {"override": False, "provider": "",
                              "model": ""},
    },
    "providers": {
        "opencode_go": {"base_url": "", "model": "m1",
                        "vision_models": [],
                        "extra": {"endpoint": "responses"}},
    },
    "gemma_fallback": True,
    "max_attempts": 3,
    "favorites": [],
}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------- registry --
class OpenCodeGoErrorMappingTest(unittest.TestCase):
    def test_5xx_is_reported_as_provider_side_not_connectivity(self):
        from h3app import opencode_go as go
        err = go._redacted_error(
            500, '{"type":"error","error":{"type":"error","message":"Internal server error"}}')
        self.assertEqual(err.kind, "provider")
        self.assertIn("HTTP 500", str(err))
        self.assertIn("Internal server error", str(err))
        self.assertIn("API Key と接続は有効", str(err))
        self.assertNotIn("接続できませんでした", str(err))

    def test_other_status_keeps_network_wording(self):
        from h3app import opencode_go as go
        err = go._redacted_error(418, "teapot")
        self.assertEqual(err.kind, "network")


class OpenCodeGoProbeTest(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_answer_counts_as_connected(self):
        from h3app.llm_providers import opencode_go as go_adapter
        adapter = build_adapter("opencode_go", base_url="", model="m1", token="k")

        async def fake_chat(**kw):
            raise LLMError("model", "回答が途中で切れました（出力トークン不足）。")
        with patch.object(adapter, "chat", new=fake_chat):
            result = await adapter.test("m1")
        self.assertTrue(result["ok"])
        self.assertTrue(result["text_ok"])
        self.assertIn("note", result)

    async def test_provider_error_fails_the_probe_with_its_message(self):
        adapter = build_adapter("opencode_go", base_url="", model="m1", token="k")

        async def fake_chat(**kw):
            raise LLMError("provider", "OpenCode Go 側でサーバーエラーが発生しました（HTTP 500）。")
        with patch.object(adapter, "chat", new=fake_chat):
            result = await adapter.test("m1")
        self.assertFalse(result["ok"])
        self.assertIn("HTTP 500", result["error"])


class RegistryTest(unittest.TestCase):
    def test_all_providers_registered(self):
        self.assertEqual(len(REGISTRY), 13)
        for pid in ("comfy_gemma", "lmstudio", "openai_compat", "ollama",
                    "llamacpp", "vllm", "localai", "opencode_go", "openai",
                    "anthropic", "gemini", "openrouter",
                    "openai_compat_external"):
            self.assertIn(pid, REGISTRY, pid)

    def test_spec_json_carries_key_name_for_the_ui(self):
        # Regression: to_json() omitted key_name, so settings.js hid the
        # API key / token row for EVERY provider (real-machine finding).
        for pid, spec in REGISTRY.items():
            data = spec.to_json()
            self.assertIn("key_name", data, pid)
            self.assertEqual(data["key_name"], spec.key_name, pid)
        # Every external provider that needs a key exposes a name for it;
        # the ComfyUI Gemma provider has none (no row to show).
        for spec in for_kind("external"):
            if spec.needs_key:
                self.assertTrue(spec.to_json()["key_name"], spec.id)
        self.assertEqual(get("comfy_gemma").to_json()["key_name"], "")

    def test_for_kind_split(self):
        local = {s.id for s in for_kind("local")}
        external = {s.id for s in for_kind("external")}
        self.assertIn("lmstudio", local)
        self.assertIn("comfy_gemma", local)
        self.assertIn("opencode_go", external)
        self.assertIn("openai_compat_external", external)
        self.assertFalse(local & external)

    def test_key_name_compat(self):
        self.assertEqual(key_name("lmstudio"), "openai_compat_local")
        self.assertEqual(key_name("opencode_go"), "opencode_go")
        self.assertEqual(key_name("openai"), "llm:openai")
        self.assertEqual(key_name("comfy_gemma"), "")

    def test_key_names_are_separated(self):
        names = [s.key_name for s in REGISTRY.values() if s.key_name]
        self.assertEqual(len(names), len(set(names)))

    def test_external_providers_never_unload(self):
        for pid in ("opencode_go", "openai", "anthropic", "gemini",
                    "openrouter", "openai_compat_external"):
            adapter = build_adapter(pid, model="m", token=DUMMY)
            result = _run(adapter.release(["m"]))
            self.assertFalse(result["supported"], pid)
            self.assertTrue(result["ok"], pid)

    def test_unknown_provider_rejected(self):
        self.assertIsNone(get("nope"))
        with self.assertRaises(ValueError):
            build_adapter("nope")


# ------------------------------------------------------- settings migration --
class MigrationTest(unittest.TestCase):
    def test_old_gemma_roles_become_comfy_gemma_connection(self):
        clean = ai_settings.normalize({
            "director": {"provider": "gemma", "model": ""},
            "character_profile": {"provider": "gemma", "model": ""}})
        self.assertEqual(clean["schema"], 2)
        self.assertEqual(clean["connection"]["provider"], "comfy_gemma")
        self.assertFalse(clean["roles"]["director"]["override"])
        self.assertFalse(clean["roles"]["character_profile"]["override"])

    def test_old_local_role_migrates_base_url_and_model(self):
        clean = ai_settings.normalize({
            "director": {"provider": "openai_compat", "model": "m1"},
            "character_profile": {"provider": "openai_compat",
                                  "model": "m1"},
            "local_server": {"base_url": "http://127.0.0.1:5000/v1",
                             "vision_models": ["m1"]}})
        self.assertEqual(clean["connection"]["provider"], "lmstudio")
        self.assertEqual(clean["providers"]["lmstudio"]["base_url"],
                         "http://127.0.0.1:5000/v1")
        self.assertEqual(clean["providers"]["lmstudio"]["model"], "m1")
        self.assertEqual(clean["providers"]["lmstudio"]["vision_models"],
                         ["m1"])

    def test_different_roles_director_becomes_connection(self):
        clean = ai_settings.normalize({
            "director": {"provider": "opencode_go", "model": "m1",
                         "endpoint": "responses"},
            "character_profile": {"provider": "openai_compat",
                                  "model": "m2"},
            "local_server": {"base_url": "http://127.0.0.1:1234/v1"}})
        self.assertEqual(clean["connection"]["provider"], "opencode_go")
        role = clean["roles"]["character_profile"]
        self.assertTrue(role["override"])
        self.assertEqual(role["provider"], "lmstudio")
        self.assertEqual(role["model"], "m2")
        self.assertEqual(clean["providers"]["opencode_go"]["extra"],
                         {"endpoint": "responses"})

    def test_effective_selection_override_and_connection(self):
        clean = ai_settings.normalize(copy.deepcopy(SCHEMA2_GO))
        director = ai_settings.effective_selection(clean, "director")
        self.assertEqual((director["provider"], director["model"],
                          director["kind"]), ("opencode_go", "m1", "external"))
        clean["roles"]["character_profile"] = {
            "override": True, "provider": "lmstudio", "model": "m2"}
        profile = ai_settings.effective_selection(clean, "character_profile")
        self.assertEqual((profile["provider"], profile["model"],
                          profile["kind"]), ("lmstudio", "m2", "local"))

    def test_save_restore_roundtrip_keeps_schema2(self):
        with tempfile.TemporaryDirectory(prefix="h3-wpa-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            written = ai_settings.save_to_config_file(
                path, copy.deepcopy(SCHEMA2_GO))
            self.assertEqual(written["schema"], 2)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["ai_settings"]["connection"]
                             ["provider"], "opencode_go")
            self.assertNotIn(DUMMY, json.dumps(loaded))

    def test_provider_switch_keeps_other_providers(self):
        with tempfile.TemporaryDirectory(prefix="h3-wpa-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            ai_settings.save_to_config_file(path, {
                "schema": 2,
                "connection": {"kind": "local", "provider": "lmstudio"},
                "providers": {"lmstudio": {"model": "m-local"}}})
            second = ai_settings.save_to_config_file(path, {
                "connection": {"kind": "local", "provider": "ollama"},
                "providers": {"ollama": {"model": "m-ollama"}}})
            self.assertEqual(second["providers"]["lmstudio"]["model"],
                             "m-local")
            self.assertEqual(second["providers"]["ollama"]["model"],
                             "m-ollama")

    def test_secrets_rejected_by_key_name_only(self):
        clean = ai_settings.normalize({
            "schema": 2,
            "providers": {"lmstudio": {"model": "my-token-model"}}})
        ai_settings._assert_no_secrets(clean)  # must not raise
        with self.assertRaises(ValueError):
            ai_settings._assert_no_secrets({"api_key": "x"})


# ------------------------------------------------------- fallback discipline --
class FallbackDisciplineTest(unittest.TestCase):
    def test_local_failure_never_calls_external(self):
        seen: list[str] = []

        async def op(kind, model_cfg, token):
            seen.append(kind)
            if kind == "local":
                raise LLMError("network", "down")
            return "gemma-text"

        result, used = _run(ai_settings.run_with_fallback(
            {"kind": "local", "provider": "lmstudio", "model": "m1"},
            op, gemma_fallback=True))
        self.assertEqual(result, "gemma-text")
        self.assertNotIn("external", seen)
        self.assertNotIn("go", seen)
        self.assertEqual(used["kind"], "gemma")

    def test_external_auth_stops_immediately(self):
        seen: list[str] = []

        async def op(kind, model_cfg, token):
            seen.append(kind)
            if kind == "external":
                raise LLMError("auth", "bad key")
            return "gemma-text"

        with self.assertRaises(director_provider.DirectorError):
            _run(ai_settings.run_with_fallback(
                {"kind": "external", "provider": "openai", "model": "m1"},
                op, gemma_fallback=True))
        self.assertNotIn("gemma", seen)

    def test_external_non_auth_falls_back_to_gemma(self):
        async def op(kind, model_cfg, token):
            if kind == "external":
                raise LLMError("network", "blip")
            return "gemma-text"

        result, used = _run(ai_settings.run_with_fallback(
            {"kind": "external", "provider": "openai", "model": "m1"},
            op, gemma_fallback=True))
        self.assertEqual((result, used["kind"]), ("gemma-text", "gemma"))

    def test_resolve_role_chain_shapes(self):
        clean = ai_settings.normalize(copy.deepcopy(SCHEMA2_GO))
        chain = ai_settings.resolve_role_chain(clean, "director")
        self.assertEqual([(a["kind"], a["model"]) for a in chain],
                         [("external", "m1"), ("gemma", "")])
        plain = ai_settings.normalize(None)
        self.assertEqual(
            [(a["kind"], a["model"]) for a in
             ai_settings.resolve_role_chain(plain, "director")],
            [("gemma", "")])


# ------------------------------------------------------------- fake servers --
class CompatFake:
    """OpenAI-compatible fake (also stands in for fixed-URL providers when
    their BASE constant is patched to the ephemeral server)."""

    def __init__(self):
        self.calls: list[dict] = []

    async def _models(self, request):
        self.calls.append({"path": "/models",
                           "auth": request.headers.get("Authorization")})
        return web.json_response({"data": [{"id": "m1"}]})

    async def _chat(self, request):
        body = await request.json()
        self.calls.append({"path": "/chat", "auth": request.headers.get(
            "Authorization"), "body": body})
        return web.json_response({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    def app(self):
        app = web.Application()
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/chat/completions", self._chat)
        return app


class TestServerMixin:
    def _with_server(self, app, fn):
        """Run fn(srv) on one loop: start, exercise, close (same loop)."""

        async def go():
            srv = TestServer(app)
            await srv.start_server()
            try:
                return await fn(srv)
            finally:
                await srv.close()

        return _run(go())


class OpenAICompatAdapterTest(TestServerMixin, unittest.TestCase):
    def test_list_chat_test_shapes_and_headers(self):
        from h3app.llm_providers.openai_compat import OpenAICompatAdapter
        fake = CompatFake()

        async def go(srv):
            base = str(srv.make_url("/v1"))
            adapter = OpenAICompatAdapter(base_url=base, model="m1",
                                          token=DUMMY)
            listing = await adapter.list_models()
            text, info = await adapter.chat(system="s", user="u")
            probed = await adapter.test("m1")
            return listing, text, info, probed

        listing, text, info, probed = self._with_server(fake.app(), go)
        self.assertEqual([m["id"] for m in listing["models"]], ["m1"])
        self.assertTrue(listing["supported"])
        self.assertEqual(text, "hi")
        self.assertEqual(info["usage"]["input_tokens"], 1)
        self.assertTrue(probed["ok"])
        self.assertTrue(probed["text_ok"])
        self.assertEqual(probed["model"], "m1")
        auths = [c["auth"] for c in fake.calls]
        self.assertTrue(all(a == f"Bearer {DUMMY}" for a in auths), auths)
        chat_body = next(c["body"] for c in fake.calls if c["path"] == "/chat")
        self.assertEqual(chat_body["model"], "m1")
        self.assertNotIn(DUMMY, json.dumps(
            [listing, info, probed], ensure_ascii=False))

    def test_external_compat_adapter_same_surface(self):
        from h3app.llm_providers.openai_compat_external import (
            OpenAICompatExternalAdapter, validate_external_user_url)
        ok, err = validate_external_user_url("https://api.example.com/v1")
        self.assertEqual((ok, err),
                         ("https://api.example.com/v1", ""))
        # userinfo fixture built at runtime so the prepublish secret
        # scan never sees a literal credential-in-URL string on disk.
        userinfo = ":".join(("user", "pass"))
        _ok2, err2 = validate_external_user_url(
            f"https://{userinfo}@api.example.com/v1")
        self.assertTrue(err2)
        fake = CompatFake()

        async def go(srv):
            adapter = OpenAICompatExternalAdapter(
                base_url=str(srv.make_url("/v1")), model="m1", token=DUMMY)
            probed = await adapter.test("m1")
            released = await adapter.release(["m1"])
            return probed, released

        probed, released = self._with_server(fake.app(), go)
        self.assertTrue(probed["ok"])
        self.assertFalse(released["supported"])

    def test_llamacpp_subclass_reaches_chat(self):
        from h3app.llm_providers.llamacpp import LlamaCppAdapter
        from h3app.llm_providers.localai import LocalAIAdapter
        from h3app.llm_providers.vllm import VLLMAdapter
        fake = CompatFake()

        async def go(srv):
            out = []
            for cls in (LlamaCppAdapter, VLLMAdapter, LocalAIAdapter):
                adapter = cls(base_url=str(srv.make_url("/v1")), model="m1")
                text, _info = await adapter.chat(user="u")
                released = await adapter.release(["m1"])
                out.append((cls.spec.id, text, released["supported"]))
            return out

        results = self._with_server(fake.app(), go)
        self.assertEqual(
            results,
            [("llamacpp", "hi", False), ("vllm", "hi", False),
             ("localai", "hi", False)])
        self.assertTrue(any(c["path"] == "/chat" for c in fake.calls))


class FixedExternalAdapterTest(TestServerMixin, unittest.TestCase):
    def test_openai_headers_and_body(self):
        from h3app.llm_providers import openai as openai_mod
        fake = CompatFake()

        async def go(srv):
            with patch.object(openai_mod, "BASE", str(srv.make_url("/v1"))):
                adapter = openai_mod.OpenAIAdapter(model="m1", token=DUMMY)
                listing = await adapter.list_models()
                text, _info = await adapter.chat(user="u")
                return listing, text

        listing, text = self._with_server(fake.app(), go)
        self.assertEqual([m["id"] for m in listing["models"]], ["m1"])
        self.assertEqual(text, "hi")
        chat = next(c for c in fake.calls if c["path"] == "/chat")
        self.assertEqual(chat["auth"], f"Bearer {DUMMY}")
        self.assertNotIn(DUMMY, json.dumps(listing))

    def test_openrouter_vision_from_modalities(self):
        from h3app.llm_providers import openrouter as or_mod

        async def _models(request):
            return web.json_response({"data": [
                {"id": "or-m", "name": "OR M",
                 "architecture": {"input_modalities": ["text", "image"]}}]})

        async def _chat(request):
            body = await request.json()
            assert body["model"] == "or-m"
            return web.json_response({
                "choices": [{"message": {"content": "hi"}}], "usage": {}})

        app = web.Application()
        app.router.add_get("/v1/models", _models)
        app.router.add_post("/v1/chat/completions", _chat)

        async def go(srv):
            with patch.object(or_mod, "BASE", str(srv.make_url("/v1"))):
                adapter = or_mod.OpenRouterAdapter(model="or-m", token=DUMMY)
                listing = await adapter.list_models()
                vision = await adapter.vision_capability("or-m")
                probed = await adapter.test("or-m")
                return listing, vision, probed

        listing, vision, probed = self._with_server(app, go)
        self.assertTrue(listing["models"][0]["vision"])
        self.assertTrue(vision)
        self.assertTrue(probed["ok"])
        self.assertTrue(probed["image_ok"])

    def test_anthropic_headers_and_blocks(self):
        from h3app.llm_providers import anthropic as an_mod
        seen: list[dict] = []

        async def _models(request):
            seen.append(dict(request.headers))
            return web.json_response({"data": [
                {"id": "claude-x", "display_name": "Claude X"}]})

        async def _messages(request):
            body = await request.json()
            seen.append({"body": body, "key": request.headers.get("x-api-key"),
                         "ver": request.headers.get("anthropic-version")})
            return web.json_response({
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 4, "output_tokens": 6}})

        app = web.Application()
        app.router.add_get("/v1/models", _models)
        app.router.add_post("/v1/messages", _messages)

        async def go(srv):
            with patch.object(an_mod, "BASE", str(srv.make_url("/v1"))):
                adapter = an_mod.AnthropicAdapter(model="claude-x",
                                                  token=DUMMY)
                listing = await adapter.list_models()
                text, info = await adapter.chat(user="u")
                return listing, text, info

        listing, text, info = self._with_server(app, go)
        self.assertEqual(listing["models"][0]["id"], "claude-x")
        self.assertEqual(text, "hi")
        self.assertEqual(info["usage"]["input_tokens"], 4)
        self.assertEqual(seen[1]["ver"], "2023-06-01")
        self.assertEqual(seen[1]["key"], DUMMY)
        images = seen[1]["body"]["messages"][0]["content"]
        self.assertIsInstance(images, list)
        self.assertNotIn(DUMMY, json.dumps(listing))

    def test_gemini_headers_and_inline_data(self):
        from h3app.llm_providers import gemini as ge_mod
        seen: list[dict] = []

        async def _models(request):
            seen.append({"key": request.headers.get("x-goog-api-key")})
            return web.json_response({"models": [
                {"name": "models/gemini-x", "displayName": "Gemini X"}]})

        async def _gen(request):
            body = await request.json()
            seen.append({"body": body})
            return web.json_response({
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {"promptTokenCount": 2,
                                  "candidatesTokenCount": 3}})

        app = web.Application()
        app.router.add_get("/v1beta/models", _models)
        app.router.add_post("/v1beta/models/{tail:.*}", _gen)

        async def go(srv):
            with patch.object(ge_mod, "BASE", str(srv.make_url("/v1beta"))):
                adapter = ge_mod.GeminiAdapter(model="gemini-x", token=DUMMY)
                listing = await adapter.list_models()
                text, info = await adapter.chat(user="u")
                return listing, text, info

        listing, text, info = self._with_server(app, go)
        self.assertEqual(listing["models"][0]["id"], "gemini-x")
        self.assertEqual(text, "hi")
        self.assertEqual(info["usage"]["input_tokens"], 2)
        self.assertEqual(seen[0]["key"], DUMMY)


class OllamaAdapterTest(TestServerMixin, unittest.TestCase):
    def _app(self, calls: list):
        async def tags(request):
            return web.json_response({"models": [{"name": "llama3"}]})

        async def show(request):
            return web.json_response({"capabilities": ["completion",
                                                       "vision"]})

        async def chat(request):
            body = await request.json()
            calls.append(body)
            return web.json_response({
                "message": {"content": "hi"},
                "prompt_eval_count": 3, "eval_count": 5})

        async def generate(request):
            body = await request.json()
            calls.append({"generate": body})
            return web.json_response({})

        app = web.Application()
        app.router.add_get("/api/tags", tags)
        app.router.add_post("/api/show", show)
        app.router.add_post("/api/chat", chat)
        app.router.add_post("/api/generate", generate)
        return app

    def test_chat_images_and_release(self):
        from h3app.llm_providers.ollama import OllamaAdapter
        calls: list = []

        async def go(srv):
            adapter = OllamaAdapter(base_url=str(srv.make_url("")),
                                    model="llama3")
            listing = await adapter.list_models()
            vision = await adapter.vision_capability("llama3")
            text, _info = await adapter.chat(
                user="u", images=["data:image/jpeg;base64,QUJD"])
            released = await adapter.release(["llama3"])
            return listing, vision, text, released

        listing, vision, text, released = self._with_server(
            self._app(calls), go)
        self.assertEqual([m["id"] for m in listing["models"]], ["llama3"])
        self.assertTrue(vision)
        self.assertEqual(text, "hi")
        chat_body = next(c for c in calls if "messages" in c)
        images = chat_body["messages"][-1]["images"]
        self.assertEqual(images, ["QUJD"])
        gen = next(c["generate"] for c in calls if "generate" in c)
        self.assertEqual(gen, {"model": "llama3", "keep_alive": 0})
        self.assertTrue(released["supported"])
        self.assertEqual(released["unloaded"], ["llama3"])


class LMStudioAdapterTest(TestServerMixin, unittest.TestCase):
    def _app(self, unload_calls: list):
        async def models(request):
            return web.json_response({"data": [{"id": "m1"}]})

        async def native(request):
            return web.json_response({"models": [{
                "key": "m1", "type": "llm", "display_name": "M1",
                "capabilities": {"vision": True},
                "loaded_instances": [{"id": "m1"}]}]})

        async def unload(request):
            body = await request.json()
            unload_calls.append(body)
            return web.json_response({})

        async def chat(request):
            return web.json_response({
                "choices": [{"message": {"content": "hi"}}], "usage": {}})

        app = web.Application()
        app.router.add_get("/v1/models", models)
        app.router.add_get("/api/v1/models", native)
        app.router.add_post("/api/v1/models/unload", unload)
        app.router.add_post("/v1/chat/completions", chat)
        return app

    def test_list_test_release_roundtrip(self):
        from h3app.llm_providers.lmstudio import LMStudioAdapter
        unload_calls: list = []

        async def go(srv):
            adapter = LMStudioAdapter(base_url=str(srv.make_url("/v1")),
                                      model="m1", token=DUMMY)
            listing = await adapter.list_models()
            vision = await adapter.vision_capability("m1")
            probed = await adapter.test("m1")
            released = await adapter.release(["m1"])
            return listing, vision, probed, released

        listing, vision, probed, released = self._with_server(
            self._app(unload_calls), go)
        self.assertTrue(listing["models"][0]["vision"])
        self.assertTrue(vision)
        self.assertTrue(probed["ok"])
        self.assertTrue(probed["image_ok"])
        self.assertEqual(unload_calls, [{"instance_id": "m1"}])
        self.assertTrue(released["supported"])
        self.assertEqual(released["unloaded"], ["m1"])
        self.assertNotIn(DUMMY, json.dumps(
            [listing, probed, released], ensure_ascii=False))


class ComfyGemmaAndGoTest(unittest.TestCase):
    def test_comfy_gemma_placeholders(self):
        from h3app.llm_providers.comfy_gemma import ComfyGemmaAdapter
        adapter = ComfyGemmaAdapter()
        listing = _run(adapter.list_models())
        self.assertFalse(listing["supported"])
        self.assertTrue(_run(adapter.vision_capability("anything")))
        with self.assertRaises(LLMError):
            _run(adapter.chat(user="u"))
        released = _run(adapter.release(["m"]))
        self.assertFalse(released["supported"])

    def test_opencode_go_wiring_uses_transport(self):
        from h3app.llm_providers import opencode_go as go_adapter_mod
        adapter = go_adapter_mod.OpenCodeGoAdapter(model="m1", token=DUMMY,
                                                  extra={"endpoint":
                                                         "responses"})
        with patch("h3app.opencode_go.fetch_models",
                   AsyncMock(return_value=[
                       {"id": "m1", "display": "M1"}])) as mock_list, \
             patch("h3app.opencode_go.generate_text_full",
                   AsyncMock(return_value=("hi", {"endpoint":
                                                  "responses"}))) as mock_gen:
            listing = _run(adapter.list_models())
            text, _info = _run(adapter.chat(user="u"))
            probed = _run(adapter.test("m1"))
        self.assertEqual(listing["models"][0]["id"], "m1")
        mock_list.assert_awaited_once()
        self.assertEqual(mock_list.await_args.kwargs.get("api_key"), DUMMY)
        self.assertEqual(text, "hi")
        mock_gen.assert_awaited()
        kwargs = mock_gen.await_args.kwargs
        self.assertEqual((kwargs.get("api_key"), kwargs.get("model")),
                         (DUMMY, "m1"))
        self.assertTrue(probed["ok"])
        self.assertIsNone(_run(adapter.vision_capability("m1")))
        self.assertFalse(_run(adapter.release(["m1"]))["supported"])
        self.assertNotIn(DUMMY, json.dumps(
            [listing, probed], ensure_ascii=False))


# ------------------------------------------------------- director + pipeline --
class AdapterDirectorProviderTest(unittest.TestCase):
    def test_generate_parses_and_reports(self):
        adapter = SimpleNamespace(
            model="",
            spec=SimpleNamespace(id="lmstudio"),
            chat=AsyncMock(return_value=(
                '{"result": "ok"}',
                {"http_status": 200, "elapsed_ms": 5, "retry_after": "",
                 "usage": {"input_tokens": 1, "output_tokens": 1,
                           "reasoning_tokens": 0},
                 "incomplete": "", "endpoint": "local"})))
        provider = director_provider.AdapterDirectorProvider(adapter, "m1")
        answer = _run(provider.generate(system="s", user="u"))
        self.assertEqual(answer, {"result": "ok"})
        adapter.chat.assert_awaited_once()
        self.assertEqual(provider.last_info["endpoint"], "local")

    def test_generate_wraps_adapter_error_with_kind(self):
        async def boom(**kwargs):
            raise LLMError("auth", "bad key")

        adapter = SimpleNamespace(
            model="", spec=SimpleNamespace(id="openai"), chat=boom)
        provider = director_provider.AdapterDirectorProvider(adapter, "m1")
        with self.assertRaises(director_provider.DirectorError) as ctx:
            _run(provider.generate(system="s", user="u"))
        self.assertEqual(ctx.exception.kind, "auth")


class PipelineProviderProfileTest(unittest.TestCase):
    def _pipeline(self, settings: dict) -> Pipeline:
        pipeline = object.__new__(Pipeline)
        data = copy.deepcopy(config_mod.DEFAULT_CONFIG)
        data["ai_settings"] = settings
        pipeline.config = config_mod.Config(data, None)
        pipeline.client = SimpleNamespace(_log=lambda msg: None)
        return pipeline

    def _profile_settings(self, provider: str, model: str,
                          fallback: bool = True) -> dict:
        clean = ai_settings.normalize(None)
        clean["connection"] = {"kind": ai_settings.kind_of(provider),
                               "provider": provider}
        clean["providers"][provider]["model"] = model
        clean["gemma_fallback"] = fallback
        return clean

    def _run_profile(self, pipeline, fake):
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter",
                   return_value=fake), \
             patch("h3app.credstore.load", return_value=""), \
             patch("h3app.opencode_go.image_to_data_url",
                   return_value="data:image/jpeg;base64,QUJD"):
            pipeline._image_paths = lambda images: [Path("x.png")]
            return _run(pipeline._try_provider_profile(runner, ["a.png"])), \
                runner

    def test_vision_false_sends_no_images(self):
        fake = SimpleNamespace(
            vision_capability=AsyncMock(return_value=False),
            chat=AsyncMock(return_value=("text", {})))
        pipeline = self._pipeline(self._profile_settings("lmstudio", "m1"))
        text, runner = self._run_profile(pipeline, fake)
        self.assertEqual(text, "")
        fake.chat.assert_not_awaited()
        self.assertIn("画像を送信しませんでした", runner.state["warning"])

    def test_vision_none_sends_no_images(self):
        fake = SimpleNamespace(
            vision_capability=AsyncMock(return_value=None),
            chat=AsyncMock(return_value=("text", {})))
        pipeline = self._pipeline(self._profile_settings("ollama", "m1"))
        text, runner = self._run_profile(pipeline, fake)
        self.assertEqual(text, "")
        fake.chat.assert_not_awaited()
        self.assertIn("画像を送信しませんでした", runner.state["warning"])

    def test_vision_true_sends_images(self):
        fake = SimpleNamespace(
            vision_capability=AsyncMock(return_value=True),
            chat=AsyncMock(return_value=(" profile-text ", {})))
        pipeline = self._pipeline(self._profile_settings("lmstudio", "m1"))
        text, _runner = self._run_profile(pipeline, fake)
        self.assertEqual(text, "profile-text")
        kwargs = fake.chat.await_args.kwargs
        self.assertTrue(kwargs.get("images"))

    def test_gemma_connection_never_calls_provider(self):
        pipeline = self._pipeline(ai_settings.normalize(None))
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter") as mock_build:
            result = _run(pipeline._try_provider_profile(runner, ["a.png"]))
        self.assertEqual(result, "")
        mock_build.assert_not_called()

    def test_no_fallback_raises_config_error(self):
        from h3app.errors import PipelineError
        fake = SimpleNamespace(
            vision_capability=AsyncMock(return_value=False),
            chat=AsyncMock(return_value=("text", {})))
        pipeline = self._pipeline(
            self._profile_settings("lmstudio", "m1", fallback=False))
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter",
                   return_value=fake), \
             patch("h3app.credstore.load", return_value=""):
            with self.assertRaises(PipelineError):
                _run(pipeline._try_provider_profile(runner, ["a.png"]))

    def test_release_uses_each_effective_adapter_once(self):
        released: list = []

        class FakeAdapter:
            def __init__(self, *args, **kwargs):
                pass

            async def release(self, models):
                released.append(sorted(models))
                return {"ok": True, "supported": True,
                        "unloaded": list(models), "error": ""}

        clean = ai_settings.normalize(None)
        clean["connection"] = {"kind": "local", "provider": "lmstudio"}
        clean["providers"]["lmstudio"]["model"] = "m1"
        clean["roles"]["character_profile"] = {
            "override": True, "provider": "ollama", "model": "m2"}
        pipeline = self._pipeline(clean)
        with patch("h3app.llm_providers.build_adapter",
                   side_effect=lambda pid, **kw: FakeAdapter()) \
                as mock_build, \
             patch("h3app.credstore.load", return_value=""):
            _run(pipeline._release_local_llm())
        self.assertEqual(mock_build.call_count, 2)
        self.assertEqual(sorted(released), [["m1"], ["m2"]])

    def test_release_skips_gemma_without_adapter(self):
        pipeline = self._pipeline(ai_settings.normalize(None))
        with patch("h3app.llm_providers.build_adapter") as mock_build:
            _run(pipeline._release_local_llm())
        mock_build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
