# -*- coding: utf-8 -*-
"""Process-ownership state for the launcher and shutdown sequence.

State file: app/_runtime/h3_state.json. Records ONLY what is needed to tell
"a process H3 started/knows about" apart from "some other process that
happens to be using the same port": pid, create_time, exe path, cmdline,
port, an owning H3 session id (a fresh uuid4 per app start, NOT a Windows
logon session id and NOT a secret), whether H3 launched it, and its role.

Never write API keys, tokens, or any credstore value here. Every read/write
is best-effort: a missing/corrupt state file just means "nothing recorded",
never a hard failure for the caller (launcher/shutdown must keep working
even if psutil is unavailable or the file is unreadable).
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except Exception:                                                # noqa: BLE001
    psutil = None  # type: ignore[assignment]

APP_DIR = Path(__file__).resolve().parent.parent          # .../app
RUNTIME_DIR = APP_DIR / "_runtime"
STATE_PATH = RUNTIME_DIR / "h3_state.json"

# Inter-process lock around every read-modify-write of the state file. The
# launcher (its own process) and the app both record entries at start-up,
# and an old app instance's cleanup can overlap a new instance's start; a
# plain read/replace pair loses one side's entries. The lock is a sibling
# file held for the duration of one update only (a few ms).
LOCK_TIMEOUT_S = 3.0

# pid + create_time (within this many seconds) + exe + cmdline must all match
# for `matches()` to treat a live process as "the one we recorded". This is
# what prevents a *different* process that happens to reuse a recycled PID
# from being mistaken for the one H3 started/attached to.
CREATE_TIME_TOLERANCE_S = 2.0

ROLE_APP = "app"
ROLE_COMFYUI = "comfyui"
ROLE_LMSTUDIO = "lmstudio"


def new_session_id() -> str:
    """A fresh id for one H3 launch (start->stop). Not a secret, not a login id."""
    return str(uuid.uuid4())


def _safe(fn, default: Any = ""):
    try:
        return fn()
    except Exception:                                            # noqa: BLE001
        return default


def identify(pid: int) -> Optional[dict]:
    """psutil snapshot of a live process, or None if it cannot be read."""
    if psutil is None or not pid:
        return None
    try:
        proc = psutil.Process(int(pid))
        with proc.oneshot():
            return {
                "pid": int(proc.pid),
                "create_time": float(_safe(proc.create_time, 0.0)),
                "exe": str(_safe(proc.exe, "")),
                "cmdline": list(_safe(proc.cmdline, [])),
            }
    except Exception:                                            # noqa: BLE001
        return None


def matches(recorded: Optional[dict], live: Optional[dict]) -> bool:
    """True only when pid + create_time (+/-2s) + exe + cmdline all agree."""
    if not recorded or not live:
        return False
    try:
        if int(recorded.get("pid", -1)) != int(live.get("pid", -2)):
            return False
    except Exception:                                            # noqa: BLE001
        return False
    try:
        rt = float(recorded.get("create_time", 0.0))
        lt = float(live.get("create_time", -1.0))
        if abs(rt - lt) > CREATE_TIME_TOLERANCE_S:
            return False
    except Exception:                                            # noqa: BLE001
        return False
    if str(recorded.get("exe") or "") != str(live.get("exe") or ""):
        return False
    if list(recorded.get("cmdline") or []) != list(live.get("cmdline") or []):
        return False
    return True


def is_alive_and_matching(recorded: dict) -> bool:
    """Convenience: identify() the recorded pid right now and compare."""
    pid = recorded.get("pid")
    if not pid:
        return False
    return matches(recorded, identify(int(pid)))


def _lock_path() -> Path:
    return STATE_PATH.with_suffix(STATE_PATH.suffix + ".lock")


@contextlib.contextmanager
def _state_lock():
    """Best-effort exclusive lock on the state file (msvcrt on Windows,
    fcntl elsewhere). Times out after LOCK_TIMEOUT_S and proceeds unlocked
    rather than blocking a shutdown forever; the caller's write is still
    atomic (tmp + os.replace), the lock only serialises the read->write."""
    handle = None
    locked = False
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(_lock_path(), "a+b")
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                if os.name == "nt":
                    import msvcrt                          # noqa: PLC0415
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl                           # noqa: PLC0415
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
    except Exception:                                            # noqa: BLE001
        handle = None
        locked = False
    try:
        yield locked
    finally:
        if handle is not None:
            try:
                if locked:
                    if os.name == "nt":
                        import msvcrt                      # noqa: PLC0415
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl                       # noqa: PLC0415
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:                                    # noqa: BLE001
                pass
            try:
                handle.close()
            except Exception:                                    # noqa: BLE001
                pass


def _read_unlocked() -> dict:
    try:
        if STATE_PATH.is_file():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:                                            # noqa: BLE001
        pass
    return {}


class StateWriteError(RuntimeError):
    """The state file could not be updated (lock not acquired within
    LOCK_TIMEOUT_S, or the atomic replace failed). The file on disk is left
    exactly as it was; callers must not report the update as done."""


def _write_unlocked(state: dict) -> None:
    """Write `state` atomically (unique temp file + os.replace) or delete the
    file when `state` is empty. Raises StateWriteError on any failure; the
    existing file is never left half-written and the temp file is removed."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not state:
        # An empty state means "nothing recorded": no file at all.
        try:
            if STATE_PATH.is_file():
                STATE_PATH.unlink()
        except Exception as exc:                                 # noqa: BLE001
            raise StateWriteError(f"状態ファイルを削除できません: {exc}") from exc
        return
    # Per-writer temp name: two processes sharing one ".tmp" would clobber
    # each other's bytes before either replace landed.
    tmp = STATE_PATH.with_name(
        f"{STATE_PATH.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except Exception as exc:                                     # noqa: BLE001
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:                                        # noqa: BLE001
            pass
        raise StateWriteError(f"状態ファイルを書き込めません: {exc}") from exc


def read_state() -> dict:
    return _read_unlocked()


def write_state(state: dict) -> bool:
    """Replace the whole state. False when the lock could not be taken or
    the write failed - the file is untouched in that case."""
    try:
        with _state_lock() as locked:
            if not locked:
                return False
            _write_unlocked(state)
            return True
    except StateWriteError:
        return False


def _update_state(mutate) -> Optional[dict]:
    """Locked read -> mutate(state) -> write. `mutate` edits the dict in
    place; the resulting dict is written (or the file removed if empty).
    Returns the written dict, or None when the lock was not acquired or the
    write failed (nothing on disk changed in either case)."""
    try:
        with _state_lock() as locked:
            if not locked:
                return None
            state = _read_unlocked()
            mutate(state)
            _write_unlocked(state)
            return state
    except StateWriteError:
        return None


def clear_state(session_id: Optional[str] = None) -> bool:
    """Forget recorded entries. With `session_id`, ONLY the entries that
    this session recorded are dropped, so an old app instance finishing its
    cleanup after a new instance has already started can never delete the
    new instance's records (the way the whole-file delete used to). Without
    a session id (launcher fallback after it terminated everything it could
    match) the file is removed as before. Returns False (and changes
    nothing) when the lock could not be taken."""
    if session_id is None:
        try:
            with _state_lock() as locked:
                if not locked:
                    return False
                _write_unlocked({})
                return True
        except StateWriteError:
            return False

    def _drop_session(state: dict) -> None:
        for role in list(state.keys()):
            entry = state.get(role)
            if isinstance(entry, dict) and entry.get("session_id") == session_id:
                state.pop(role, None)

    return _update_state(_drop_session) is not None


def record_entry(role: str, pid: int, *, session_id: str,
                 port: Optional[int] = None, owned_by_h3: bool = True,
                 loaded_models: Optional[list] = None) -> Optional[dict]:
    """Identify `pid` and record/replace the entry for `role`. Returns the
    stored entry, or None if the process could not be identified (nothing
    is written in that case: a bad entry is worse than no entry)."""
    ident = identify(pid)
    if ident is None:
        return None
    entry = dict(ident)
    entry.update({
        "session_id": session_id,
        "port": port,
        "owned_by_h3": bool(owned_by_h3),
        "role": role,
        "loaded_models": list(loaded_models or []),
    })
    if _update_state(lambda state: state.__setitem__(role, entry)) is None:
        # Lock/write failure: the record did NOT land, say so.
        return None
    return entry


def remove_entry(role: str, session_id: Optional[str] = None) -> bool:
    """Drop one role's entry. With `session_id`, only if that session
    recorded it (see clear_state). False when nothing could be written."""
    def _drop(state: dict) -> None:
        entry = state.get(role)
        if entry is None:
            return
        if session_id is not None and isinstance(entry, dict) and \
                entry.get("session_id") != session_id:
            return
        state.pop(role, None)

    return _update_state(_drop) is not None


def prune_dead_entries(current_session_id: Optional[str] = None) -> list[str]:
    """Remove entries whose recorded process is no longer alive/matching
    (a previous run that ended without cleanup). Entries of
    `current_session_id` and entries whose process still matches are kept.
    Returns the roles that were pruned."""
    pruned: list[str] = []

    def _prune(state: dict) -> None:
        for role in list(state.keys()):
            entry = state.get(role)
            if not isinstance(entry, dict):
                state.pop(role, None)
                pruned.append(role)
                continue
            if current_session_id is not None and \
                    entry.get("session_id") == current_session_id:
                continue
            if is_alive_and_matching(entry):
                continue
            state.pop(role, None)
            pruned.append(role)

    if _update_state(_prune) is None:
        return []
    return pruned


def stale_entries(current_session_id: Optional[str] = None) -> dict:
    """Entries left over from a previous H3 run (different/absent session id)."""
    state = read_state()
    if current_session_id is None:
        return state
    return {role: entry for role, entry in state.items()
            if entry.get("session_id") != current_session_id}


def find_listener(port: int) -> Optional[dict]:
    """{"pid","name","exe"} of whatever is LISTENing on `port`, or None.

    Best-effort: any psutil failure (missing permission, no psutil at all)
    is reported as "unknown" (None) rather than raised, since this only ever
    feeds a diagnostic message.
    """
    if psutil is None:
        return None
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception:                                            # noqa: BLE001
        return None
    for c in conns:
        try:
            if c.status != psutil.CONN_LISTEN:
                continue
            laddr = c.laddr
            if not laddr or int(getattr(laddr, "port", -1)) != int(port):
                continue
        except Exception:                                        # noqa: BLE001
            continue
        pid = c.pid
        if not pid:
            return {"pid": None, "name": "", "exe": ""}
        try:
            proc = psutil.Process(pid)
            return {"pid": int(pid), "name": str(_safe(proc.name, "")),
                    "exe": str(_safe(proc.exe, ""))}
        except Exception:                                        # noqa: BLE001
            return {"pid": int(pid), "name": "", "exe": ""}
    return None
