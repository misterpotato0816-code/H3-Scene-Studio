# -*- coding: utf-8 -*-
"""Fix A: a stalled ComfyUI job must be interrupted before the app gives up.

run_graph()'s stall branch is extracted into ComfyClient._on_stall() so this
can be tested directly, without faking a websocket connection: see comfy.py
run_graph (around the STALL_SECONDS check).
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3app import comfy as comfy_mod                                # noqa: E402
from h3app import config                                            # noqa: E402
from h3app.errors import PipelineError                              # noqa: E402


class OnStallInterruptsBeforeRaising(unittest.IsolatedAsyncioTestCase):
    def _client(self) -> comfy_mod.ComfyClient:
        root = Path(tempfile.mkdtemp(prefix="h3-stall-"))
        data = copy.deepcopy(config.DEFAULT_CONFIG)
        data.update(comfy_dir=str(root / "comfy"))
        cfg = config.Config(data, None)
        return comfy_mod.ComfyClient(cfg)

    async def test_interrupt_awaited_then_timeout_raised(self):
        client = self._client()
        client.interrupt = AsyncMock()
        with self.assertRaises(PipelineError) as ctx:
            await client._on_stall(301.0, "KSampler")
        client.interrupt.assert_awaited_once()
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertEqual(ctx.exception.node, "KSampler")

    async def test_interrupt_runs_even_when_current_node_unknown(self):
        client = self._client()
        client.interrupt = AsyncMock()
        with self.assertRaises(PipelineError) as ctx:
            await client._on_stall(305.0, None)
        client.interrupt.assert_awaited_once()
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertIn("不明", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
