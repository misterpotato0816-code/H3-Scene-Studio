# -*- coding: utf-8 -*-
"""LM Studio is the default LLM connection; Gemma fallback is opt-in
(OFF by default) since H3 Scene Studio must work without the
ComfyUI-llama-cpp node pack. See ai_settings.py module docstring.

No real network, GPU or processes: credstore/build_adapter are mocked.
"""
import asyncio
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3app import ai_settings                                      # noqa: E402
from h3app import config as config_mod                             # noqa: E402
from h3app import director_provider                                # noqa: E402
from h3app.errors import PipelineError                             # noqa: E402
from h3app.pipeline import Pipeline                                # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --------------------------------------------------------------- (a) defaults --
class DefaultConnectionTest(unittest.TestCase):
    def test_normalize_none_defaults_to_lmstudio_no_gemma_fallback(self):
        clean = ai_settings.normalize(None)
        self.assertEqual(clean["connection"]["provider"], "lmstudio")
        self.assertIs(clean["gemma_fallback"], False)

    def test_normalize_empty_schema_defaults_to_lmstudio(self):
        clean = ai_settings.normalize({"schema": 2})
        self.assertEqual(clean["connection"]["provider"], "lmstudio")
        self.assertIs(clean["gemma_fallback"], False)

    def test_normalize_unknown_provider_defaults_to_lmstudio(self):
        clean = ai_settings.normalize(
            {"schema": 2, "connection": {"provider": "nonsense"}})
        self.assertEqual(clean["connection"]["provider"], "lmstudio")


# ------------------------------------------------------------- (b) resolve_chain --
class ResolveChainUnconfiguredTest(unittest.TestCase):
    def test_local_no_model_is_empty_chain(self):
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": ""}, 5)
        self.assertEqual(chain, [])

    def test_local_no_model_with_gemma_fallback_still_empty(self):
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": ""}, 5,
            gemma_fallback=True)
        self.assertEqual(chain, [])

    def test_external_no_model_is_empty_chain(self):
        chain = ai_settings.resolve_chain(
            {"kind": "external", "provider": "openai", "model": ""}, 5)
        self.assertEqual(chain, [])

    def test_local_with_model_no_gemma_fallback_key_is_local_only(self):
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": "m"}, 5)
        self.assertEqual([(a["kind"], a["model"]) for a in chain],
                         [("local", "m")])

    def test_lmstudio_no_model_stays_empty_unchanged(self):
        # Regression guard: the comfy_gemma fix must not affect lmstudio.
        chain = ai_settings.resolve_chain(
            {"kind": "local", "provider": "lmstudio", "model": ""}, 5)
        self.assertEqual(chain, [])


# ------------------------------------------------------------ (c) run_with_fallback --
class RunWithFallbackUnconfiguredTest(unittest.TestCase):
    def test_unconfigured_primary_raises_before_calling_op(self):
        op = AsyncMock()
        with self.assertRaises(director_provider.DirectorError) as ctx:
            _run(ai_settings.run_with_fallback(
                {"kind": "local", "provider": "lmstudio", "model": ""},
                op, max_attempts=3))
        self.assertIn("設定 > LLM接続", str(ctx.exception))
        self.assertEqual(ctx.exception.kind, "ai_config")
        op.assert_not_called()


# ------------------------------------------------- explicit comfy_gemma selection --
class ComfyGemmaExplicitSelectionTest(unittest.TestCase):
    """Regression: selecting comfy_gemma explicitly (kind pre-set to "local"
    by effective_selection/kind_of, which never report kind "gemma") must
    still resolve to a Gemma-only attempt, not an empty/unconfigured chain.
    """

    def test_resolve_role_chain_is_gemma_only(self):
        clean = ai_settings.normalize({
            "schema": 2,
            "connection": {"kind": "local", "provider": "comfy_gemma"}})
        chain = ai_settings.resolve_role_chain(clean, "director")
        self.assertEqual(
            chain, [{"kind": "gemma", "provider": "comfy_gemma",
                    "model": ""}])

    def test_run_with_fallback_calls_op_once_as_gemma(self):
        op = AsyncMock(return_value="ok")
        result, used = _run(ai_settings.run_with_fallback(
            {"kind": "local", "provider": "comfy_gemma", "model": ""}, op))
        self.assertEqual(result, "ok")
        op.assert_awaited_once_with("gemma", {}, "")
        self.assertEqual(used["kind"], "gemma")


# --------------------------------------------------------- (d) _try_provider_director --
class PipelineProviderDirectorTest(unittest.TestCase):
    def _pipeline(self, settings: dict) -> Pipeline:
        pipeline = object.__new__(Pipeline)
        data = copy.deepcopy(config_mod.DEFAULT_CONFIG)
        data["ai_settings"] = settings
        pipeline.config = config_mod.Config(data, None)
        pipeline.client = SimpleNamespace(_log=lambda msg: None)
        return pipeline

    def _director_settings(self, provider: str, model: str,
                           fallback: bool = False) -> dict:
        clean = ai_settings.normalize(None)
        clean["connection"] = {"kind": ai_settings.kind_of(provider),
                               "provider": provider}
        clean["providers"][provider]["model"] = model
        clean["gemma_fallback"] = fallback
        return clean

    def test_lmstudio_returns_adapter_text(self):
        fake = SimpleNamespace(
            chat=AsyncMock(return_value=(" hello director ", {})))
        pipeline = self._pipeline(self._director_settings("lmstudio", "m1"))
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter",
                   return_value=fake), \
             patch("h3app.credstore.load", return_value=""):
            text = _run(pipeline._try_provider_director(
                runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(text, "hello director")
        fake.chat.assert_awaited_once()
        self.assertIsNone(fake.chat.await_args.kwargs.get("images"))

    def test_gemma_connection_returns_empty_no_adapter(self):
        clean = ai_settings.normalize(None)
        clean["connection"] = {"kind": "local", "provider": "comfy_gemma"}
        pipeline = self._pipeline(clean)
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter") as mock_build:
            result = _run(pipeline._try_provider_director(
                runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(result, "")
        mock_build.assert_not_called()

    def test_no_model_no_fallback_raises_config_error(self):
        pipeline = self._pipeline(
            self._director_settings("lmstudio", "", fallback=False))
        runner = SimpleNamespace(state={})
        with self.assertRaises(PipelineError) as ctx:
            _run(pipeline._try_provider_director(
                runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(ctx.exception.kind, "ai_config")
        self.assertIn("設定 > LLM接続", str(ctx.exception))

    def test_no_model_with_fallback_returns_empty_and_warns(self):
        pipeline = self._pipeline(
            self._director_settings("lmstudio", "", fallback=True))
        runner = SimpleNamespace(state={})
        result = _run(pipeline._try_provider_director(
            runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(result, "")
        self.assertTrue(runner.state.get("warning"))

    def test_llm_error_no_fallback_raises_config_error(self):
        from h3app.llm_providers.base import LLMError
        fake = SimpleNamespace(
            chat=AsyncMock(side_effect=LLMError("network", "down")))
        pipeline = self._pipeline(
            self._director_settings("lmstudio", "m1", fallback=False))
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter",
                   return_value=fake), \
             patch("h3app.credstore.load", return_value=""):
            with self.assertRaises(PipelineError) as ctx:
                _run(pipeline._try_provider_director(
                    runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(ctx.exception.kind, "ai_config")

    def test_empty_answer_no_fallback_raises_instead_of_gemma_handoff(self):
        fake = SimpleNamespace(chat=AsyncMock(return_value=("   ", {})))
        pipeline = self._pipeline(
            self._director_settings("lmstudio", "m1", fallback=False))
        runner = SimpleNamespace(state={})
        with patch("h3app.llm_providers.build_adapter",
                   return_value=fake),              patch("h3app.credstore.load", return_value=""):
            with self.assertRaises(PipelineError) as ctx:
                _run(pipeline._try_provider_director(
                    runner, system_prompt="sys", user_prompt="user"))
        self.assertEqual(ctx.exception.kind, "ai_config")
        self.assertIn("空", str(ctx.exception))


# ---------------------------------------------------------- (e) _run_director wiring --
class RunDirectorUsesProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_provider_result_skips_graph_and_run(self):
        from h3app import pipeline as pipeline_mod

        pipeline = object.__new__(pipeline_mod.Pipeline)

        project = SimpleNamespace(data={
            "settings": {"frames": 24},
            "images": [],
            "jp_scene": "",
            "jp_dialogue": "",
            "jp_negative": "",
            "profile": "",
        })
        runner = SimpleNamespace(
            set_stage=lambda *a, **kw: None, state={})

        with patch.object(pipeline_mod.Pipeline, "_try_provider_director",
                          new=AsyncMock(return_value="hello")), \
             patch.object(pipeline_mod.graphs, "build_director_graph") \
                as mock_graph, \
             patch.object(pipeline_mod.Pipeline, "_run",
                          new=AsyncMock()) as mock_run:
            result = await pipeline._run_director(
                runner, project, continuation=False,
                jp_prev="", clean_sections=False)

        self.assertEqual(result, "hello")
        mock_graph.assert_not_called()
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
