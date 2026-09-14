# -*- coding: utf-8 -*-
"""WP-D D4: launcher + procstate tests. No real processes, no network.

Everything external (psutil, subprocess.Popen, urllib, webbrowser, input,
time.sleep, the state file location) is mocked or redirected to tmp dirs.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3app import launcher as launcher_mod                          # noqa: E402
from h3app import procstate as procstate_mod                        # noqa: E402

APP_PORT = 8790
COMFY_PORT = 8411

# State entries must never grow secret-shaped keys; this is the whole schema.
ALLOWED_STATE_KEYS = {
    "pid", "create_time", "exe", "cmdline", "session_id", "port",
    "owned_by_h3", "role", "loaded_models",
}

FAKE_IDENT = {
    "pid": 4242,
    "create_time": 1_700_000_000.0,
    "exe": r"C:\venv\Scripts\python.exe",
    "cmdline": ["python.exe", "-X", "utf8", "server.py"],
}


def _fake_cfg(**over):
    cfg = SimpleNamespace(
        app_port=APP_PORT,
        comfy_port=COMFY_PORT,
        comfy_python=Path(r"C:\comfy\.venv\Scripts\python.exe"),
        comfy_dir=Path(r"C:\comfy"),
        paths_yaml=Path(r"C:\h3\app\comfy_paths.yaml"),
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _recorded(pid=4242, **over):
    entry = dict(FAKE_IDENT)
    entry.update({
        "pid": pid,
        "session_id": "session-1",
        "port": APP_PORT,
        "owned_by_h3": True,
        "role": procstate_mod.ROLE_APP,
        "loaded_models": [],
    })
    entry.update(over)
    return entry


class MatchesTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(procstate_mod.matches(
            _recorded(), dict(FAKE_IDENT)))

    def test_create_time_within_tolerance(self):
        live = dict(FAKE_IDENT, create_time=FAKE_IDENT["create_time"] + 1.9)
        self.assertTrue(procstate_mod.matches(_recorded(), live))

    def test_create_time_outside_tolerance_is_recycled_pid(self):
        live = dict(FAKE_IDENT, create_time=FAKE_IDENT["create_time"] + 2.1)
        self.assertFalse(procstate_mod.matches(_recorded(), live))

    def test_pid_mismatch(self):
        self.assertFalse(procstate_mod.matches(
            _recorded(), dict(FAKE_IDENT, pid=9999)))

    def test_exe_mismatch(self):
        self.assertFalse(procstate_mod.matches(
            _recorded(), dict(FAKE_IDENT, exe=r"C:\other\app.exe")))

    def test_cmdline_mismatch(self):
        self.assertFalse(procstate_mod.matches(
            _recorded(), dict(FAKE_IDENT, cmdline=["other.exe"])))

    def test_none_inputs(self):
        self.assertFalse(procstate_mod.matches(None, dict(FAKE_IDENT)))
        self.assertFalse(procstate_mod.matches(_recorded(), None))


class StateFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3-state-"))
        self.state_path = self.tmp / "h3_state.json"
        self.patch_state = mock.patch.object(
            procstate_mod, "STATE_PATH", self.state_path)
        self.patch_runtime = mock.patch.object(
            procstate_mod, "RUNTIME_DIR", self.tmp)
        self.patch_state.start()
        self.patch_runtime.start()
        self.addCleanup(self.patch_state.stop)
        self.addCleanup(self.patch_runtime.stop)

    def test_write_read_round_trip(self):
        procstate_mod.write_state({"app": _recorded()})
        self.assertEqual(procstate_mod.read_state()["app"]["pid"], 4242)

    def test_record_entry_stores_no_secret_shaped_keys(self):
        with mock.patch.object(procstate_mod, "identify",
                               return_value=dict(FAKE_IDENT)):
            entry = procstate_mod.record_entry(
                procstate_mod.ROLE_APP, 4242, session_id="session-1",
                port=APP_PORT, owned_by_h3=True)
        self.assertIsNotNone(entry)
        self.assertLessEqual(set(entry.keys()), ALLOWED_STATE_KEYS)
        raw = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn("token", raw.lower())
        self.assertNotIn("key", raw.lower().replace("turkey", ""))
        data = json.loads(raw)
        self.assertLessEqual(set(data["app"].keys()), ALLOWED_STATE_KEYS)

    def test_record_entry_writes_nothing_when_unidentifiable(self):
        with mock.patch.object(procstate_mod, "identify", return_value=None):
            self.assertIsNone(procstate_mod.record_entry(
                procstate_mod.ROLE_APP, 4242, session_id="session-1"))
        self.assertFalse(self.state_path.is_file())

    def test_stale_entries(self):
        procstate_mod.write_state({
            "app": _recorded(session_id="old"),
            "comfyui": _recorded(pid=1111, session_id="new"),
        })
        stale = procstate_mod.stale_entries("new")
        self.assertIn("app", stale)
        self.assertNotIn("comfyui", stale)

    def test_find_listener_matches_listen_port_only(self):
        listen = SimpleNamespace(
            status="LISTEN",
            laddr=SimpleNamespace(port=COMFY_PORT), pid=7777)
        other_port = SimpleNamespace(
            status="LISTEN",
            laddr=SimpleNamespace(port=9999), pid=8888)
        established = SimpleNamespace(
            status="ESTABLISHED",
            laddr=SimpleNamespace(port=COMFY_PORT), pid=9999)

        class FakeProc:
            def name(self):
                return "python.exe"

            def exe(self):
                return r"C:\venv\python.exe"

        fake_psutil = SimpleNamespace(
            CONN_LISTEN="LISTEN",
            net_connections=mock.Mock(
                return_value=[other_port, established, listen]),
            Process=mock.Mock(return_value=FakeProc()))
        with mock.patch.object(procstate_mod, "psutil", fake_psutil):
            found = procstate_mod.find_listener(COMFY_PORT)
        self.assertEqual(found, {"pid": 7777, "name": "python.exe",
                                 "exe": r"C:\venv\python.exe"})

    def test_find_listener_none_when_psutil_missing(self):
        with mock.patch.object(procstate_mod, "psutil", None):
            self.assertIsNone(procstate_mod.find_listener(COMFY_PORT))


class StartCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3-launch-"))
        self.patchers = [
            mock.patch.object(launcher_mod, "_load_config"),
            mock.patch.object(launcher_mod, "preflight", return_value=[]),
            mock.patch.object(procstate_mod, "RUNTIME_DIR", self.tmp),
            mock.patch.object(procstate_mod, "STATE_PATH",
                               self.tmp / "h3_state.json"),
            mock.patch.object(launcher_mod, "_http_get_ok", return_value=True),
            mock.patch.object(launcher_mod.webbrowser, "open"),
            mock.patch("subprocess.Popen"),
            mock.patch("time.sleep"),
        ]
        self.mocks = [p.start() for p in self.patchers]
        for p in self.patchers:
            self.addCleanup(p.stop)
        (self.mock_load, self.mock_preflight, _, _,
         self.mock_http, self.mock_browser, self.mock_popen,
         self.mock_sleep) = self.mocks
        self.cfg = _fake_cfg()
        self.mock_load.return_value = self.cfg

    def _own_state(self):
        return {"app": _recorded()}

    def test_double_start_opens_browser_without_relaunch(self):
        with mock.patch.object(procstate_mod, "read_state",
                               return_value=self._own_state()), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 4242, "name": "python",
                                             "exe": FAKE_IDENT["exe"]}), \
             mock.patch.object(procstate_mod, "identify",
                               return_value=dict(FAKE_IDENT)):
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 0)
        self.mock_popen.assert_not_called()
        self.mock_browser.assert_called_once_with(
            f"http://127.0.0.1:{APP_PORT}/")

    def test_foreign_port_occupant_is_never_killed(self):
        fake_psutil = mock.Mock()
        with mock.patch.object(procstate_mod, "read_state",
                               return_value=self._own_state()), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 9999, "name": "other",
                                             "exe": r"C:\other\app.exe"}), \
             mock.patch.object(launcher_mod, "psutil", fake_psutil):
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 3)
        self.mock_popen.assert_not_called()
        fake_psutil.Process.assert_not_called()

    def test_recycled_pid_does_not_count_as_own(self):
        reused = dict(FAKE_IDENT,
                      create_time=FAKE_IDENT["create_time"] + 500.0)
        with mock.patch.object(procstate_mod, "read_state",
                               return_value=self._own_state()), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 4242, "name": "python",
                                             "exe": FAKE_IDENT["exe"]}), \
             mock.patch.object(procstate_mod, "identify",
                               return_value=reused), \
             mock.patch.object(launcher_mod, "psutil", mock.Mock()):
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 3)
        self.mock_popen.assert_not_called()

    def test_fresh_start_launches_detached_with_log(self):
        proc = mock.Mock()
        proc.pid = 5555
        proc.poll.return_value = None
        self.mock_popen.return_value = proc
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "record_entry",
                               return_value=_recorded(pid=5555)) as mock_rec:
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 0)
        self.mock_popen.assert_called_once()
        _args, kwargs = self.mock_popen.call_args
        argv = _args[0]
        self.assertIsInstance(argv, list)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(argv[1:3], ["-X", "utf8"])
        self.assertTrue(argv[3].endswith("server.py"))
        self.assertIn("creationflags", kwargs)
        self.assertIsNotNone(kwargs.get("stdout"))  # app.out.log capture
        self.assertEqual(kwargs.get("cwd"), str(launcher_mod.APP_DIR))
        mock_rec.assert_called_once()
        self.mock_browser.assert_called_once()

    def test_command_generation_with_space_and_japanese_path(self):
        app_dir = self.tmp / "空白 日本語 dir" / "app"
        app_dir.mkdir(parents=True)
        proc = mock.Mock()
        proc.pid = 5555
        proc.poll.return_value = None
        self.mock_popen.return_value = proc
        with mock.patch.object(launcher_mod, "APP_DIR", app_dir), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "record_entry",
                               return_value=_recorded(pid=5555)):
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 0)
        argv = self.mock_popen.call_args[0][0]
        self.assertIsInstance(argv, list)  # list form: no quoting bugs
        self.assertIn(str(app_dir / "server.py"), argv)
        self.assertEqual(self.mock_popen.call_args[1].get("cwd"), str(app_dir))

    def test_preflight_failure_blocks_launch(self):
        self.mock_preflight.return_value = ["ComfyUIのPythonが見つかりません"]
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=None):
            rc = launcher_mod.cmd_start([])
        self.assertEqual(rc, 2)
        self.mock_popen.assert_not_called()

    def test_stale_h3_comfyui_is_reclaimed(self):
        stale = {"comfyui": _recorded(pid=1111, port=COMFY_PORT,
                                      role=procstate_mod.ROLE_COMFYUI)}
        fake_proc = mock.Mock()
        fake_psutil = SimpleNamespace(Process=mock.Mock(
            return_value=fake_proc))
        with mock.patch.object(procstate_mod, "read_state",
                               return_value=stale), \
             mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 1111, "name": "python",
                                             "exe": FAKE_IDENT["exe"]}), \
             mock.patch.object(procstate_mod, "identify",
                               return_value=dict(FAKE_IDENT, pid=1111)), \
             mock.patch.object(launcher_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry") as mock_rm:
            launcher_mod._check_comfy_port(self.cfg)
        fake_psutil.Process.assert_called_once_with(1111)
        fake_proc.terminate.assert_called_once()
        mock_rm.assert_called_once_with(procstate_mod.ROLE_COMFYUI)

    def test_reclaim_falls_back_to_kill(self):
        fake_proc = mock.Mock()
        fake_proc.wait.side_effect = Exception("timeout")
        fake_psutil = SimpleNamespace(Process=mock.Mock(
            return_value=fake_proc))
        with mock.patch.object(launcher_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry"):
            launcher_mod._reclaim_process(_recorded())
        fake_proc.terminate.assert_called_once()
        fake_proc.kill.assert_called_once()


class StopCommandTest(unittest.TestCase):
    def setUp(self):
        patchers = [
            mock.patch.object(launcher_mod, "_load_config"),
            mock.patch("time.sleep"),
        ]
        self.mock_load, self.mock_sleep = [p.start() for p in patchers]
        for p in patchers:
            self.addCleanup(p.stop)
        self.cfg = _fake_cfg()
        self.mock_load.return_value = self.cfg

    def test_stop_is_idempotent_when_nothing_listening(self):
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=None), \
             mock.patch.object(launcher_mod, "_post_shutdown") as mock_post:
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 0)
        mock_post.assert_not_called()

    def test_stop_posts_graceful_shutdown(self):
        body = {"ok": True, "steps": [{"name": "stop_comfyui", "ok": True,
                                       "skipped": False, "message": "done"}],
                "failed": [], "ports": []}
        listener = {"pid": 4242, "name": "python", "exe": "python.exe"}
        # Listening before the POST, then (the app exits ~0.5 s after
        # answering) free on the next look.
        with mock.patch.object(procstate_mod, "find_listener",
                               side_effect=[listener, None, None]), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               return_value={"status": 200,
                                             "body": body}) as mock_post:
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 0)
        mock_post.assert_called_once_with(self.cfg, interrupt=False)

    def test_stop_waits_for_port_release_and_reports_if_still_busy(self):
        body = {"ok": True, "steps": [], "failed": [], "ports": []}
        listener = {"pid": 4242, "name": "python", "exe": "python.exe"}
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value=listener), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               return_value={"status": 200, "body": body}), \
             mock.patch.object(launcher_mod, "STOP_WAIT_S", 0.2), \
             mock.patch.object(launcher_mod.time, "sleep"):
            rc = launcher_mod.cmd_stop([])
        # The sequence ran, but the port never freed: say so, do not claim
        # "stopped", and never kill the listener here.
        self.assertEqual(rc, 6)

    def test_stop_409_confirms_then_resends_with_interrupt(self):
        busy = {"status": 409, "body": {"ok": False, "busy": True}}
        done = {"status": 200, "body": {"ok": True, "steps": [],
                                        "failed": [], "ports": []}}
        listener = {"pid": 4242, "name": "python", "exe": "python.exe"}
        with mock.patch.object(procstate_mod, "find_listener",
                               side_effect=[listener, None, None]), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               side_effect=[busy, done]) as mock_post, \
             mock.patch("builtins.input", return_value="y"):
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_post.call_count, 2)
        mock_post.assert_called_with(self.cfg, interrupt=True)

    def test_stop_409_cancelled_by_user(self):
        busy = {"status": 409, "body": {"ok": False, "busy": True}}
        with mock.patch.object(procstate_mod, "find_listener",
                               return_value={"pid": 4242, "name": "python",
                                             "exe": "python.exe"}), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               return_value=busy) as mock_post, \
             mock.patch("builtins.input", return_value="n"):
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 1)
        mock_post.assert_called_once_with(self.cfg, interrupt=False)

    def test_fallback_stops_only_identity_matched_owned_entries(self):
        state = {
            "app": _recorded(),
            "comfyui": _recorded(pid=1111, port=COMFY_PORT,
                                 role=procstate_mod.ROLE_COMFYUI),
            "lmstudio": _recorded(pid=2222, port=1234, owned_by_h3=False,
                                  role=procstate_mod.ROLE_LMSTUDIO),
        }
        fake_proc = mock.Mock()
        fake_psutil = SimpleNamespace(Process=mock.Mock(
            return_value=fake_proc))
        with mock.patch.object(procstate_mod, "find_listener",
                               side_effect=[{"pid": 4242}, None]), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state",
                               return_value=state), \
             mock.patch.object(procstate_mod, "identify",
                               side_effect=lambda pid: dict(FAKE_IDENT, pid=pid)
                               if pid in (4242, 1111) else None), \
             mock.patch.object(launcher_mod, "psutil", fake_psutil), \
             mock.patch.object(procstate_mod, "remove_entry") as mock_rm, \
             mock.patch.object(procstate_mod, "clear_state") as mock_clear:
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 0)
        # app + H3-owned comfyui only; the foreign lmstudio entry is skipped
        # (and lmstudio is never in the fallback role list at all).
        touched = {c.args[0] for c in
                   fake_psutil.Process.call_args_list}
        self.assertLessEqual(touched, {4242, 1111})
        # Only the records it actually stopped are dropped; the file is never
        # deleted wholesale (the lmstudio record must survive for next time).
        removed = {c.args[0] for c in mock_rm.call_args_list}
        self.assertLessEqual(removed, {procstate_mod.ROLE_APP, procstate_mod.ROLE_COMFYUI})
        mock_clear.assert_not_called()

    def test_fallback_reports_still_listening_port(self):
        with mock.patch.object(procstate_mod, "find_listener",
                               side_effect=[{"pid": 4242, "name": "other",
                                             "exe": r"C:\other\app.exe"},
                                            {"pid": 9999, "name": "other",
                                             "exe": r"C:\other\app.exe"}]), \
             mock.patch.object(launcher_mod, "_post_shutdown",
                               return_value=None), \
             mock.patch.object(procstate_mod, "read_state", return_value={}), \
             mock.patch.object(procstate_mod, "clear_state"):
            rc = launcher_mod.cmd_stop([])
        self.assertEqual(rc, 6)


class BatWrapperTest(unittest.TestCase):
    def _read_bat(self, name):
        path = launcher_mod.REPO_ROOT / name
        raw = path.read_bytes()
        return raw.decode("ascii"), raw  # strict: ASCII only by design

    def test_run_bat_is_thin_ascii_wrapper(self):
        text, _raw = self._read_bat("RUN_H3.bat")
        self.assertIn("-X utf8 -m h3app.launcher start", text)
        self.assertIn('"%~dp0', text)  # every path is quoted
        self.assertNotIn("chcp", text.lower())
        self.assertNotIn("runas", text.lower())

    def test_stop_bat_is_thin_ascii_wrapper(self):
        text, _raw = self._read_bat("STOP_H3.bat")
        self.assertIn("-X utf8 -m h3app.launcher stop", text)
        self.assertIn('"%~dp0', text)
        self.assertNotIn("chcp", text.lower())
        self.assertNotIn("runas", text.lower())


if __name__ == "__main__":
    unittest.main()
