# -*- coding: utf-8 -*-
"""WP-D D4: shutdown-sequence tests. No real processes, no network.

psutil, subprocess, shutil.which, credstore, the LLM unload call, the state
file helpers and the event loop are all mocked. Dummy stand-ins only: no
real token or key appears anywhere in this file.
"""
import json
import sys
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3app import config as config_mod                              # noqa: E402
from h3app import procstate as procstate_mod                        # noqa: E402
from h3app import shutdown as shutdown_mod                          # noqa: E402

APP_PORT = 8790
COMFY_PORT = 8411
LM_PORT = 1234
LM_EXE = r"C:\Program Files\LM Studio\LM Studio.exe"

DUMMY_TOKEN = "DUMMY-TOKEN-ABC-123"


class FakeCfg:
    """Minimal stand-in for config.Config (attribute + .get access)."""

    def __init__(self, data):
        self.data = dict(data)
        self.app_port = data.get("app_port", APP_PORT)
        self.comfy_port = data.get("comfy_port", COMFY_PORT)

    def get(self, key, default=None):
        return self.data.get(key, default)


def _local_ai_settings(model="test-model"):
    return {
        "director": {"provider": "openai_compat", "model": model},
        "character_profile": {"provider": "gemma", "model": ""},
        "local_server": {"base_url": f"http://127.0.0.1:{LM_PORT}/v1"},
    }


def _seq(cfg_data=None, *, busy=False, owned=True, story=None):
    cfg = FakeCfg({
        "app_port": APP_PORT,
        "comfy_port": COMFY_PORT,
        "shutdown": {"stop_lm_studio": True},
        **(cfg_data or {}),
    })
    pipeline = SimpleNamespace(
        is_busy=busy,
        _release_local_llm=mock.AsyncMock())
    client = SimpleNamespace(
        interrupt=mock.AsyncMock(),
        is_reachable=mock.AsyncMock(return_value=True),
        free_models=mock.AsyncMock(return_value={"freed_mib": 10}),
        owned=owned,
        shutdown=mock.Mock())
    logs: list = []
    seq = shutdown_mod.ShutdownSequence(
        cfg, pipeline, client, story, session_id="session-1",
        log=logs.append)
    return seq, cfg, pipeline, client, logs


def _lm_listener(name="LM Studio", exe=LM_EXE, pid=7777):
    return {"pid": pid, "name": name, "exe": exe}


LM_CMDLINE = [LM_EXE, "--run-as-service"]


def _lm_entry(session="session-1", pid=7777, port=LM_PORT, create_time=1000.0,
              exe=LM_EXE, cmdline=None):
    """What record_lmstudio_target() stores for the process H3 connected to."""
    return {"pid": pid, "create_time": create_time, "exe": exe,
            "cmdline": list(LM_CMDLINE if cmdline is None else cmdline),
            "session_id": session, "port": port, "owned_by_h3": False,
            "role": "lmstudio", "loaded_models": []}


def _lm_live(pid=7777, create_time=1000.0, exe=LM_EXE, cmdline=None):
    """What procstate.identify(pid) returns for the live process."""
    return {"pid": pid, "create_time": create_time, "exe": exe,
            "cmdline": list(LM_CMDLINE if cmdline is None else cmdline)}


def _identify_for(live_by_pid: dict):
    """identify() stub: known pids return their live identity, others None."""
    return lambda pid: live_by_pid.get(int(pid)) if pid else None


class FreeModelsNotRunningTest(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_comfy_is_a_skip_not_a_failure(self):
        seq, _cfg, _pipeline, client, _logs = _seq(owned=False)
        client.is_reachable = mock.AsyncMock(return_value=False)
        with mock.patch.object(shutdown_mod, "lmstudio_target", return_value=None),              mock.patch.object(procstate_mod, "find_listener", return_value=None),              mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        step = next(s for s in result["steps"] if s["name"] == "comfy_free_models")
        self.assertTrue(step["ok"])
        self.assertTrue(step["skipped"])
        client.free_models.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertEqual(result["failed"], [])


class StateFileSessionScopeTest(unittest.IsolatedAsyncioTestCase):
    """h3_state.json went missing on the real machine: an older app instance
    finishing its cleanup after a newer instance had started deleted the
    whole file. clear_state(session_id) must drop only that session's rows."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h3-state-")
        root = Path(self.tmp.name)
        self._patches = [
            mock.patch.object(procstate_mod, "RUNTIME_DIR", root),
            mock.patch.object(procstate_mod, "STATE_PATH", root / "h3_state.json"),
        ]
        for pt in self._patches:
            pt.start()

    def tearDown(self):
        for pt in self._patches:
            pt.stop()
        self.tmp.cleanup()

    def test_clear_state_keeps_other_sessions_entries(self):
        procstate_mod.write_state({
            "app": {"pid": 1, "session_id": "old", "role": "app"},
            "comfyui": {"pid": 2, "session_id": "old", "role": "comfyui"},
        })
        # The newer instance records itself while the old one is still up.
        procstate_mod.write_state({**procstate_mod.read_state(),
                                   "app": {"pid": 3, "session_id": "new", "role": "app"}})
        procstate_mod.clear_state("old")
        state = procstate_mod.read_state()
        self.assertEqual(list(state.keys()), ["app"])
        self.assertEqual(state["app"]["session_id"], "new")
        self.assertTrue(procstate_mod.STATE_PATH.is_file())
        # Dropping the last session removes the file entirely.
        procstate_mod.clear_state("new")
        self.assertFalse(procstate_mod.STATE_PATH.is_file())

    def test_remove_entry_is_session_scoped(self):
        procstate_mod.write_state({
            "lmstudio": {"pid": 5, "session_id": "new", "role": "lmstudio"}})
        procstate_mod.remove_entry("lmstudio", "old")
        self.assertIn("lmstudio", procstate_mod.read_state())
        procstate_mod.remove_entry("lmstudio", "new")
        self.assertEqual(procstate_mod.read_state(), {})

    def test_sequence_clear_passes_its_session(self):
        seq, _cfg, _pipeline, _client, _logs = _seq(owned=False)
        with mock.patch.object(shutdown_mod, "lmstudio_target", return_value=None), \
             mock.patch.object(procstate_mod, "find_listener", return_value=None), \
             mock.patch.object(procstate_mod, "clear_state") as mock_clear:
            asyncio.get_event_loop_policy()
            result = asyncio.run(seq.run(interrupt=False, self_exit=False))
        self.assertTrue(result["ok"])
        mock_clear.assert_called_once_with("session-1")

    def test_prune_dead_entries_keeps_live_and_current(self):
        procstate_mod.write_state({
            "app": {"pid": 11, "session_id": "cur", "role": "app"},
            "comfyui": {"pid": 12, "session_id": "old", "role": "comfyui",
                        "create_time": 1.0, "exe": "x", "cmdline": ["x"]},
            "lmstudio": {"pid": 13, "session_id": "old", "role": "lmstudio",
                         "create_time": 1.0, "exe": "y", "cmdline": ["y"]},
        })
        alive = {"pid": 13, "create_time": 1.0, "exe": "y", "cmdline": ["y"]}
        with mock.patch.object(procstate_mod, "identify",
                               side_effect=lambda pid: alive if pid == 13 else None):
            pruned = procstate_mod.prune_dead_entries("cur")
        self.assertEqual(pruned, ["comfyui"])
        self.assertEqual(sorted(procstate_mod.read_state()), ["app", "lmstudio"])

    def _unlocked(self):
        """A _state_lock() replacement that never acquires the lock."""
        import contextlib

        @contextlib.contextmanager
        def _never():
            yield False
        return mock.patch.object(procstate_mod, "_state_lock", _never)

    def test_lock_failure_leaves_file_bytes_untouched(self):
        self.assertTrue(procstate_mod.write_state({"app": {"pid": 1, "session_id": "s1"}}))
        before = procstate_mod.STATE_PATH.read_bytes()
        with self._unlocked():
            self.assertFalse(procstate_mod.write_state({"app": {"pid": 2}}))
            self.assertFalse(procstate_mod.remove_entry("app"))
            self.assertFalse(procstate_mod.clear_state("s1"))
            self.assertFalse(procstate_mod.clear_state(None))
            self.assertEqual(procstate_mod.prune_dead_entries("other"), [])
            self.assertIsNone(procstate_mod._update_state(lambda st: st.clear()))
            with mock.patch.object(procstate_mod, "identify",
                                   return_value={"pid": 9, "create_time": 1.0,
                                                 "exe": "x", "cmdline": ["x"]}):
                self.assertIsNone(procstate_mod.record_entry("comfyui", 9, session_id="s1"))
        self.assertEqual(procstate_mod.STATE_PATH.read_bytes(), before)
        leftovers = [p.name for p in procstate_mod.RUNTIME_DIR.iterdir()
                     if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_replace_failure_is_not_reported_as_success(self):
        # Two entries so every failing update below goes through the
        # temp-file + os.replace path (removing the LAST entry deletes the
        # file instead, which is a different, legitimate outcome).
        self.assertTrue(procstate_mod.write_state({
            "app": {"pid": 1, "session_id": "s1"},
            "comfyui": {"pid": 2, "session_id": "s1"}}))
        before = procstate_mod.STATE_PATH.read_bytes()
        with mock.patch.object(procstate_mod.os, "replace",
                               side_effect=OSError("disk full")):
            self.assertFalse(procstate_mod.write_state({"app": {"pid": 2}}))
            self.assertFalse(procstate_mod.remove_entry("app"))
            with mock.patch.object(procstate_mod, "identify",
                                   return_value={"pid": 9, "create_time": 1.0,
                                                 "exe": "x", "cmdline": ["x"]}):
                self.assertIsNone(procstate_mod.record_entry("comfyui", 9, session_id="s1"))
        self.assertEqual(procstate_mod.STATE_PATH.read_bytes(), before)
        leftovers = [p.name for p in procstate_mod.RUNTIME_DIR.iterdir()
                     if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [], "temp files must be cleaned up")

    def test_temp_file_names_are_unique_per_writer(self):
        seen = []
        real_replace = procstate_mod.os.replace

        def spy(src, dst):
            seen.append(Path(src).name)
            return real_replace(src, dst)
        with mock.patch.object(procstate_mod.os, "replace", side_effect=spy):
            procstate_mod.write_state({"a": {"session_id": "x"}})
            procstate_mod.write_state({"b": {"session_id": "x"}})
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        self.assertTrue(all(n.startswith("h3_state.json.") and n.endswith(".tmp") for n in seen))

    def test_sequence_reports_clear_state_failure_honestly(self):
        seq, _cfg, _pipeline, _client, _logs = _seq(owned=False)
        with mock.patch.object(shutdown_mod, "lmstudio_target", return_value=None), \
             mock.patch.object(procstate_mod, "find_listener", return_value=None), \
             mock.patch.object(procstate_mod, "clear_state", return_value=False):
            result = asyncio.run(seq.run(interrupt=False, self_exit=False))
        step = next(s for s in result["steps"] if s["name"] == "clear_state")
        self.assertFalse(step["ok"])
        self.assertIn("clear_state", result["failed"])

    def test_concurrent_updates_do_not_lose_entries(self):
        import threading
        errors = []

        def worker(role, n):
            try:
                for i in range(n):
                    procstate_mod._update_state(
                        lambda st, r=role, k=i: st.__setitem__(r, {"i": k}))
            except Exception as exc:                            # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"r{j}", 30)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        state = procstate_mod.read_state()
        self.assertEqual(sorted(state), ["r0", "r1", "r2", "r3"])
        self.assertTrue(all(state[r]["i"] == 29 for r in state))


class OrderTest(unittest.IsolatedAsyncioTestCase):
    async def test_unload_free_comfy_lmstudio_app_state_order(self):
        seq, _cfg, pipeline, client, _logs = _seq()
        events: list = []
        pipeline._release_local_llm.side_effect = (
            lambda: events.append("release"))
        client.free_models.side_effect = (
            lambda: events.append("free") or {"freed_mib": 10})
        client.shutdown.side_effect = lambda: events.append("comfy")

        fake_proc = mock.Mock()
        fake_proc.children.return_value = []
        fake_psutil = SimpleNamespace(
            Process=mock.Mock(return_value=fake_proc),
            wait_procs=mock.Mock(return_value=([fake_proc], [])))

        def find_listener(port):
            if int(port) == LM_PORT:
                return _lm_listener()
            return None

        async def fake_release(base_url, model, token=None):
            events.append("unload")
            return {"ok": True, "native": True, "unloaded": ["i1"],
                    "error": ""}

        fake_proc.terminate.side_effect = lambda: events.append("lm_term")
        fake_loop = mock.Mock()

        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={
                                   "base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                   "port": LM_PORT, "models": ["m1"]}), \
             mock.patch.object(procstate_mod, "find_listener",
                               side_effect=find_listener), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": _lm_entry()}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=_identify_for({7777: _lm_live()})), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state",
                               side_effect=lambda sid=None: events.append("clear") or True), \
             mock.patch.object(shutdown_mod.credstore_mod, "load",
                               return_value=""), \
             mock.patch.object(shutdown_mod.local_llm_mod, "release_model",
                               side_effect=fake_release), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch("asyncio.get_running_loop",
                        return_value=fake_loop):
            result = await seq.run(interrupt=False, self_exit=True)

        self.assertEqual(
            events,
            ["release", "free", "comfy", "unload", "lm_term", "clear"])
        self.assertEqual(
            [s["name"] for s in result["steps"]],
            ["stop_accepting", "interrupt", "release_llm",
             "comfy_free_models", "stop_comfyui", "stop_lmstudio",
             "schedule_exit", "clear_state"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["failed"], [])
        fake_loop.call_later.assert_called_once()
        self.assertEqual(fake_loop.call_later.call_args[0][0], 0.5)
        # The deferred callback really exits the app process.
        with mock.patch("os._exit") as mock_exit:
            fake_loop.call_later.call_args[0][1]()
        mock_exit.assert_called_once_with(0)

    async def test_idle_run_skips_interrupt_and_exit(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        by_name = {s["name"]: s for s in result["steps"]}
        self.assertTrue(by_name["interrupt"]["skipped"])
        self.assertTrue(by_name["stop_lmstudio"]["skipped"])
        self.assertTrue(by_name["schedule_exit"]["skipped"])
        self.assertTrue(result["ok"])

    async def test_repeat_run_stays_ok(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        ctx = [
            mock.patch.object(shutdown_mod, "lmstudio_target",
                              return_value=None),
            mock.patch.object(procstate_mod, "find_listener",
                              return_value=None),
            mock.patch.object(procstate_mod, "read_state", return_value={}),
            mock.patch.object(procstate_mod, "remove_entry"),
            mock.patch.object(procstate_mod, "clear_state"),
        ]
        for c in ctx:
            c.start()
            self.addCleanup(c.stop)
        first = await seq.run(interrupt=False, self_exit=False)
        second = await seq.run(interrupt=False, self_exit=False)
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])


class BusyTest(unittest.IsolatedAsyncioTestCase):
    async def test_busy_without_interrupt_returns_409_shape(self):
        seq, _cfg, pipeline, client, _logs = _seq(busy=True)
        result = await seq.run(interrupt=False, self_exit=False)
        self.assertFalse(result["ok"])
        self.assertTrue(result["busy"])
        pipeline._release_local_llm.assert_not_awaited()
        client.free_models.assert_not_awaited()
        client.shutdown.assert_not_called()

    async def test_busy_with_interrupt_stops_work_first(self):
        story = SimpleNamespace(
            runners=["s1"],
            is_running=mock.Mock(return_value=True),
            store=SimpleNamespace(get=mock.Mock(return_value=object())),
            request_stop=mock.AsyncMock())
        seq, _cfg, _pipeline, client, _logs = _seq(busy=True, story=story)
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=True, self_exit=False)
        client.interrupt.assert_awaited_once()
        story.request_stop.assert_awaited_once()
        by_name = {s["name"]: s for s in result["steps"]}
        self.assertFalse(by_name["interrupt"]["skipped"])
        self.assertTrue(result["ok"])


class ComfyOwnershipTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, owned):
        seq, _cfg, _pipeline, client, _logs = _seq(owned=owned)
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry") as mock_rm, \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        return result, client, mock_rm

    async def test_owned_comfyui_is_stopped(self):
        result, client, mock_rm = await self._run(owned=True)
        client.shutdown.assert_called_once()
        # Session-scoped: only this run's record may be dropped.
        mock_rm.assert_called_once_with(procstate_mod.ROLE_COMFYUI, "session-1")
        step = next(s for s in result["steps"]
                    if s["name"] == "stop_comfyui")
        self.assertTrue(step["ok"])
        self.assertFalse(step["skipped"])

    async def test_attached_comfyui_is_left_alone(self):
        result, client, mock_rm = await self._run(owned=False)
        client.shutdown.assert_not_called()
        mock_rm.assert_not_called()
        step = next(s for s in result["steps"]
                    if s["name"] == "stop_comfyui")
        self.assertTrue(step["skipped"])


class LmStudioTest(unittest.IsolatedAsyncioTestCase):
    async def test_setting_disabled_skips_without_lookup(self):
        seq, _cfg, _pipeline, _client, _logs = _seq(
            cfg_data={"shutdown": {"stop_lm_studio": False}})
        with mock.patch.object(procstate_mod, "find_listener") as mock_find, \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        # _ports_report still probes app/comfy ports, but the LM Studio port
        # must never be looked up when the setting disables the step.
        looked_up = [c.args[0] for c in mock_find.call_args_list]
        self.assertNotIn(LM_PORT, looked_up)
        step = next(s for s in result["steps"]
                    if s["name"] == "stop_lmstudio")
        self.assertTrue(step["skipped"])

    async def test_setting_defaults_to_true(self):
        seq, _cfg, _pipeline, _client, _logs = _seq(cfg_data={})
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        step = next(s for s in result["steps"]
                    if s["name"] == "stop_lmstudio")
        # No local-LLM role in use: skipped for absence, not for the flag.
        self.assertTrue(step["skipped"])

    async def test_non_lmstudio_server_is_never_stopped(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        fake_psutil = mock.Mock()
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={
                                   "base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                   "port": LM_PORT, "models": ["m1"]}), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener(
                                   name="python.exe",
                                   exe=r"C:\venv\Scripts\python.exe")), \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch("subprocess.run") as mock_run:
            step = await seq._step_stop_lmstudio()
        self.assertTrue(step["skipped"])
        fake_psutil.Process.assert_not_called()
        mock_run.assert_not_called()

    async def test_lms_server_stop_and_tree_terminate(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        child = mock.Mock()
        fake_proc = mock.Mock()
        fake_proc.children.return_value = [child]
        fake_psutil = SimpleNamespace(
            Process=mock.Mock(return_value=fake_proc),
            wait_procs=mock.Mock(return_value=([fake_proc, child], [])))
        unload_calls: list = []

        async def fake_release(base_url, model, token=None):
            unload_calls.append((base_url, model, token))
            return {"ok": True, "native": True, "unloaded": ["i1"],
                    "error": ""}

        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={
                                   "base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                   "port": LM_PORT, "models": ["m1"]}), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener()), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": _lm_entry()}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=_identify_for({7777: _lm_live()})), \
             mock.patch.object(shutdown_mod.credstore_mod, "load",
                               return_value=""), \
             mock.patch.object(shutdown_mod.local_llm_mod, "release_model",
                               side_effect=fake_release), \
             mock.patch("shutil.which",
                        return_value=r"C:\cli\lms.exe"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry",
                               return_value=True) as mock_rm:
            step = await seq._step_stop_lmstudio()
        self.assertTrue(step["ok"], step)
        self.assertFalse(step["skipped"])
        self.assertEqual(len(unload_calls), 1)  # H3-used model only
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0][-2:], ["server", "stop"])
        fake_psutil.Process.assert_called_once_with(7777)  # the RECORDED pid
        child.terminate.assert_called_once()
        fake_proc.terminate.assert_called_once()
        mock_rm.assert_called_once_with(procstate_mod.ROLE_LMSTUDIO, "session-1")

    async def test_kill_fallback_when_process_lingers(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        fake_proc = mock.Mock()
        fake_proc.children.return_value = []
        fake_psutil = SimpleNamespace(
            Process=mock.Mock(return_value=fake_proc),
            wait_procs=mock.Mock(return_value=([], [fake_proc])))
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={
                                   "base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                   "port": LM_PORT, "models": []}), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener()), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": _lm_entry()}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=_identify_for({7777: _lm_live()})), \
             mock.patch.object(shutdown_mod.credstore_mod, "load",
                               return_value=""), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("subprocess.run"), \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry", return_value=True):
            step = await seq._step_stop_lmstudio()
        self.assertTrue(step["ok"])
        fake_proc.kill.assert_called_once()

    # ---- PR #1 review: identity-limited termination -------------------
    async def _no_local_action(self, *, target, entry, live, which=r"C:\cli\lms.exe",
                               cfg_data=None):
        """Run the step and assert NO local CLI stop / terminate happened."""
        seq, _cfg, _pipeline, _client, _logs = _seq(cfg_data=cfg_data)
        fake_psutil = mock.Mock()
        with mock.patch.object(shutdown_mod, "lmstudio_target", return_value=target), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener()), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": entry} if entry else {}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=_identify_for(live)), \
             mock.patch.object(shutdown_mod.credstore_mod, "load", return_value=""), \
             mock.patch.object(shutdown_mod.local_llm_mod, "release_model",
                               new=mock.AsyncMock(return_value={"ok": True})) as rel, \
             mock.patch("shutil.which", return_value=which), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry", return_value=True):
            step = await seq._step_stop_lmstudio()
        mock_run.assert_not_called()
        fake_psutil.Process.assert_not_called()
        return step, rel

    async def test_lan_target_never_touches_local_lmstudio_on_same_port(self):
        # Configured server is another PC (192.168.1.50:1234) while an
        # unrelated LM Studio listens on THIS PC's 1234: nothing local may
        # be stopped, even with a matching record present.
        step, rel = await self._no_local_action(
            target={"base_url": f"http://192.168.1.50:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(), live={7777: _lm_live()})
        self.assertTrue(step["skipped"])
        self.assertIn("このPCではない", step["message"])
        rel.assert_not_called()

    async def test_no_record_for_this_session_skips(self):
        step, _rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=None, live={7777: _lm_live()})
        self.assertTrue(step["skipped"])
        self.assertIn("記録がない", step["message"])

    async def test_record_of_another_session_skips(self):
        step, _rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(session="someone-else"), live={7777: _lm_live()})
        self.assertTrue(step["skipped"])
        self.assertIn("別のH3セッション", step["message"])

    async def test_pid_reuse_or_mismatch_skips(self):
        # Same pid, different create_time (recycled PID) ...
        step, _rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(), live={7777: _lm_live(create_time=5000.0)})
        self.assertTrue(step["skipped"])
        self.assertIn("一致しません", step["message"])
        # ... different executable at that pid ...
        step, _rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(), live={7777: _lm_live(exe=r"C:\other\app.exe",
                                                   cmdline=[r"C:\other\app.exe"])})
        self.assertTrue(step["skipped"])
        # ... or the process is simply gone.
        step, _rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(), live={})
        self.assertTrue(step["skipped"])

    async def test_recorded_port_must_match_target(self):
        step, _rel = await self._no_local_action(
            target={"base_url": "http://127.0.0.1:2345/v1",
                    "port": 2345, "models": ["m1"]},
            entry=_lm_entry(port=LM_PORT), live={7777: _lm_live()})
        self.assertTrue(step["skipped"])
        self.assertIn("ポート", step["message"])

    async def test_setting_off_keeps_a_matching_target(self):
        step, rel = await self._no_local_action(
            target={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                    "port": LM_PORT, "models": ["m1"]},
            entry=_lm_entry(), live={7777: _lm_live()},
            cfg_data={"shutdown": {"stop_lm_studio": False}})
        self.assertTrue(step["skipped"])
        rel.assert_not_called()

    async def test_recheck_before_terminate_aborts_when_identity_changed(self):
        # Matches before the unload/CLI work, but by the time terminate()
        # would run the pid belongs to something else: no terminate.
        seq, _cfg, _pipeline, _client, _logs = _seq()
        fake_psutil = mock.Mock()
        answers = [_lm_live(), _lm_live(create_time=9999.0)]
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={"base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                             "port": LM_PORT, "models": ["m1"]}), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": _lm_entry()}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=lambda pid: answers.pop(0)), \
             mock.patch.object(shutdown_mod.credstore_mod, "load", return_value=""), \
             mock.patch.object(shutdown_mod.local_llm_mod, "release_model",
                               new=mock.AsyncMock(return_value={"ok": True})), \
             mock.patch("shutil.which", return_value=r"C:\cli\lms.exe"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry", return_value=True):
            step = await seq._step_stop_lmstudio()
        self.assertTrue(step["skipped"])
        mock_run.assert_called_once()          # CLI stop ran (identity held then)
        fake_psutil.Process.assert_not_called()  # but nothing was terminated

    def test_target_is_local_only_for_loopback(self):
        for url, expected in (("http://127.0.0.1:1234/v1", True),
                              ("http://localhost:1234/v1", True),
                              ("http://[::1]:1234/v1", True),
                              ("http://192.168.1.50:1234/v1", False),
                              ("http://lmstudio.lan:1234/v1", False),
                              ("", False)):
            self.assertEqual(shutdown_mod.target_is_local({"base_url": url}), expected, url)
        self.assertFalse(shutdown_mod.target_is_local(None))

    def test_record_lmstudio_target_records_only_local_lmstudio(self):
        cfg = FakeCfg({"ai_settings": _local_ai_settings()})
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener()), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "record_entry",
                               return_value=_lm_entry()) as rec:
            self.assertIsNotNone(shutdown_mod.record_lmstudio_target(cfg, "session-1"))
        rec.assert_called_once()
        self.assertEqual(rec.call_args[0][:2], (procstate_mod.ROLE_LMSTUDIO, 7777))
        self.assertFalse(rec.call_args[1]["owned_by_h3"])
        # Remote target: nothing recorded.
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={"base_url": "http://192.168.1.50:1234/v1",
                                             "port": 1234, "models": []}), \
             mock.patch.object(procstate_mod, "record_entry") as rec2:
            self.assertIsNone(shutdown_mod.record_lmstudio_target(cfg, "session-1"))
        rec2.assert_not_called()
        # Local port held by something that is not LM Studio: nothing recorded.
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=_lm_listener(name="python.exe",
                                                         exe=r"C:\venv\python.exe")), \
             mock.patch.object(procstate_mod, "record_entry") as rec3:
            self.assertIsNone(shutdown_mod.record_lmstudio_target(cfg, "session-1"))
        rec3.assert_not_called()

    def test_lmstudio_target_only_for_local_roles(self):
        cfg = FakeCfg({"ai_settings": _local_ai_settings()})
        target = shutdown_mod.lmstudio_target(cfg)
        self.assertIsNotNone(target)
        self.assertEqual(target["port"], LM_PORT)
        self.assertEqual(target["models"], ["test-model"])

        gemma_cfg = FakeCfg({"ai_settings": {
            "director": {"provider": "gemma", "model": ""},
            "character_profile": {"provider": "gemma", "model": ""},
            "local_server": {"base_url": ""},
        }})
        self.assertIsNone(shutdown_mod.lmstudio_target(gemma_cfg))

    def test_shutdown_setting_save_restore_round_trip(self):
        tmp = Path(tempfile.mkdtemp(prefix="h3-cfg-")) / "config.json"
        tmp.write_text(json.dumps({"app_port": APP_PORT}), encoding="utf-8")
        data = config_mod.update_config_file(
            tmp, {"shutdown": {"stop_lm_studio": False}})
        self.assertFalse(data["shutdown"]["stop_lm_studio"])
        reloaded = json.loads(tmp.read_text(encoding="utf-8"))
        self.assertFalse(reloaded["shutdown"]["stop_lm_studio"])
        self.assertEqual(reloaded["app_port"], APP_PORT)  # others kept


class FailureContinuesTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_step_failure_still_runs_the_rest(self):
        seq, _cfg, pipeline, client, _logs = _seq()
        pipeline._release_local_llm.side_effect = RuntimeError("unload down")
        client.free_models.side_effect = RuntimeError("free down")
        client.shutdown.side_effect = RuntimeError("comfy down")
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state"):
            result = await seq.run(interrupt=False, self_exit=False)
        client.free_models.assert_awaited_once()
        client.shutdown.assert_called_once()
        self.assertFalse(result["ok"])
        self.assertEqual(result["failed"],
                         ["release_llm", "comfy_free_models",
                          "stop_comfyui"])


class PortsReportTest(unittest.TestCase):
    def _seq(self):
        seq, _cfg, _pipeline, _client, _logs = _seq()
        return seq

    def test_free_port_reported_free(self):
        seq = self._seq()
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}):
            ports = seq._ports_report()
        by_port = {p["port"]: p for p in ports}
        self.assertTrue(by_port[APP_PORT]["free"])
        self.assertTrue(by_port[COMFY_PORT]["free"])

    def test_managed_port_uses_live_identity(self):
        seq = self._seq()
        recorded = {
            "pid": 4242, "create_time": 1_700_000_000.0,
            "exe": r"C:\venv\python.exe", "cmdline": ["python.exe"],
            "session_id": "session-1", "port": APP_PORT,
            "owned_by_h3": True, "role": "app", "loaded_models": [],
        }
        live = {"pid": 4242, "create_time": 1_700_000_000.5,
                "exe": r"C:\venv\python.exe", "cmdline": ["python.exe"]}
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 4242, "name": "python",
                                             "exe": r"C:\venv\python.exe"}), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"app": recorded}), \
             mock.patch.object(procstate_mod, "identify", return_value=live):
            ports = seq._ports_report()
        entry = next(p for p in ports if p["port"] == APP_PORT)
        self.assertFalse(entry["free"])
        self.assertTrue(entry["managed"])

    def test_recycled_pid_is_not_managed(self):
        seq = self._seq()
        recorded = {
            "pid": 4242, "create_time": 1_700_000_000.0,
            "exe": r"C:\venv\python.exe", "cmdline": ["python.exe"],
            "session_id": "session-1", "port": APP_PORT,
            "owned_by_h3": True, "role": "app", "loaded_models": [],
        }
        reused = {"pid": 4242, "create_time": 1_700_100_000.0,
                  "exe": r"C:\venv\python.exe", "cmdline": ["python.exe"]}
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value=None), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 4242, "name": "python",
                                             "exe": r"C:\venv\python.exe"}), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"app": recorded}), \
             mock.patch.object(procstate_mod, "identify", return_value=reused):
            ports = seq._ports_report()
        entry = next(p for p in ports if p["port"] == APP_PORT)
        self.assertFalse(entry["free"])
        self.assertFalse(entry["managed"])


class NoSecretsTest(unittest.IsolatedAsyncioTestCase):
    async def test_token_never_reaches_logs_steps_or_ports(self):
        seq, _cfg, pipeline, client, logs = _seq()

        def find_listener(port):
            if int(port) == LM_PORT:
                return _lm_listener()
            return None

        fake_proc = mock.Mock()
        fake_proc.children.return_value = []
        fake_psutil = SimpleNamespace(
            Process=mock.Mock(return_value=fake_proc),
            wait_procs=mock.Mock(return_value=([fake_proc], [])))
        with mock.patch.object(shutdown_mod, "lmstudio_target",
                               return_value={
                                   "base_url": f"http://127.0.0.1:{LM_PORT}/v1",
                                   "port": LM_PORT, "models": ["m1"]}), \
             mock.patch.object(procstate_mod, "find_listener",
                               side_effect=find_listener), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value={"lmstudio": _lm_entry()}), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=_identify_for({7777: _lm_live()})), \
             mock.patch.object(procstate_mod, "remove_entry"), \
             mock.patch.object(procstate_mod, "clear_state", return_value=True), \
             mock.patch.object(shutdown_mod.credstore_mod, "load",
                               return_value=DUMMY_TOKEN), \
             mock.patch.object(shutdown_mod.local_llm_mod, "release_model",
                               return_value={"ok": False, "native": False,
                                             "unloaded": [],
                                             "error": "down"}) as mock_rel, \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("subprocess.run"), \
             mock.patch.object(shutdown_mod, "psutil", fake_psutil):
            result = await seq.run(interrupt=False, self_exit=False)
        # The token IS passed to the unload call (functional), ...
        mock_rel.assert_awaited_once()
        self.assertEqual(mock_rel.call_args[1].get("token"), DUMMY_TOKEN)
        # ... but must never surface in logs, steps, or the ports report.
        blob = json.dumps(result, ensure_ascii=False) + "\n".join(logs)
        self.assertNotIn(DUMMY_TOKEN, blob)


class TestBuiltAppNeverRunsShutdownTest(unittest.IsolatedAsyncioTestCase):
    """Regression (2026-09-13): the unit-test suites build the real aiohttp
    app with on_startup cleared; their teardown then ran the real shutdown
    sequence, which sent /interrupt and /free to whatever ComfyUI was on the
    default port (a running story generation was cancelled by a test run)
    and deleted the real h3_state.json. on_cleanup must do nothing of the
    sort unless on_startup actually ran."""

    async def test_cleanup_skips_sequence_when_startup_was_cleared(self):
        import copy
        import tempfile
        from pathlib import Path
        from aiohttp.test_utils import TestClient, TestServer
        import server

        with tempfile.TemporaryDirectory(prefix="h3-cleanup-") as td:
            root = Path(td)

            class _Cfg(config_mod.Config):
                def __init__(self, data):
                    super().__init__(data, None)

                @property
                def projects_dir(self): return root / "projects"

                @property
                def story_dir(self): return root / "stories"

                @property
                def debug_dir(self): return root / "debug"

                @property
                def app_dir(self): return root

            data = copy.deepcopy(config_mod.DEFAULT_CONFIG)
            data.update(comfy_dir=str(root / "comfy"), media_root=str(root / "media"),
                        input_stage_dir=str(root / "stage"), auto_launch_comfy=False)
            with mock.patch.object(server.presets_mod, "STORE_PATH", root / "actions.json"):
                app = server.create_app(_Cfg(data))
            app.on_startup.clear()
            seq = app["shutdown_seq"]
            with mock.patch.object(seq, "run", new=mock.AsyncMock()) as run_mock, \
                 mock.patch.object(app["client"], "interrupt", new=mock.AsyncMock()) as intr, \
                 mock.patch.object(procstate_mod, "clear_state") as clear_mock:
                client = TestClient(TestServer(app))
                await client.start_server()
                await client.close()
            run_mock.assert_not_called()
            intr.assert_not_called()
            clear_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
