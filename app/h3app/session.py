# -*- coding: utf-8 -*-
"""Tab-closed watchdog.

The browser pings POST /api/heartbeat every 10 s and fires POST /api/detach from
navigator.sendBeacon when the tab closes. When the client has been gone longer
than `idle_timeout_seconds`, this:

  1. asks a running story to stop - AFTER the current clip, never mid-clip
  2. waits until nothing is running (a single-shot clip is left to finish too)
  3. releases the models
  4. optionally stops ComfyUI, but ONLY a ComfyUI this app launched itself

It never kills a ComfyUI the app merely attached to, and it never touches this
server's own process.
"""
from __future__ import annotations

import asyncio
import time

CHECK_INTERVAL_SECONDS = 15


class SessionWatchdog:
    def __init__(self, cfg, pipeline, story_pipeline, client, log=None):
        self.cfg = cfg
        self.pipeline = pipeline
        self.story_pipeline = story_pipeline
        self.client = client
        self._log = log or (lambda m: print(m, flush=True))
        self._task: asyncio.Task | None = None
        self._last_seen = time.monotonic()
        self._seen_any = False        # never act before a client has ever connected
        self._detached = False
        self._acted = False
        self._stop_asked = False

    # ------------------------------------------------------------- client io --
    def heartbeat(self) -> dict:
        self._last_seen = time.monotonic()
        self._seen_any = True
        self._detached = False
        self._acted = False
        self._stop_asked = False
        return self.status()

    def detach(self) -> dict:
        self._detached = True
        self._seen_any = True
        self._last_seen = time.monotonic() - self.timeout - 1
        return self.status()

    @property
    def timeout(self) -> float:
        try:
            return max(10.0, float(self.cfg.get("idle_timeout_seconds", 90)))
        except Exception:
            return 90.0

    def status(self) -> dict:
        return {
            "connected": self._seen_any and not self._detached,
            "idle_timeout_seconds": self.timeout,
            "stop_comfy_when_idle": bool(self.cfg.get("stop_comfy_when_idle", True)),
            "busy": bool(self.pipeline.is_busy),
        }

    # ------------------------------------------------------------- lifecycle --
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):   # noqa: BLE001
            pass

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(CHECK_INTERVAL_SECONDS)
                try:
                    await self._tick()
                except Exception as exc:              # noqa: BLE001
                    self._log(f"[watchdog] tick failed: {type(exc).__name__}: {exc}")
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------ tick --
    async def _tick(self) -> None:
        if not self._seen_any:
            return                                    # nobody has ever connected
        gone_for = time.monotonic() - self._last_seen
        if not self._detached and gone_for <= self.timeout:
            return

        if not self._stop_asked:
            self._stop_asked = True
            await self._ask_stories_to_stop()

        if self.pipeline.is_busy:
            # A clip is in flight. Let it finish; we come back in 15 s.
            return
        if self._acted:
            return
        self._acted = True
        self._log(f"[watchdog] client gone for {gone_for:.0f}s -> releasing models")
        await self._release_and_maybe_stop_comfy()

    async def _ask_stories_to_stop(self) -> None:
        for runner in list(getattr(self.story_pipeline, "runners", {}).values()):
            if runner.state.get("finished"):
                continue
            try:
                story = runner.story
                story.data["stop_requested"] = True
                story.save()
                runner.stop.set()
                self._log(f"[watchdog] stop requested for story {story.id}")
            except Exception as exc:                  # noqa: BLE001
                self._log(f"[watchdog] could not request stop: {exc}")

    async def _release_and_maybe_stop_comfy(self) -> None:
        try:
            await self.pipeline.release_models()
        except Exception as exc:                      # noqa: BLE001
            self._log(f"[watchdog] release skipped: {exc}")

        if not self.cfg.get("stop_comfy_when_idle", True):
            return
        # Only ever a ComfyUI this process launched. `attached` means someone
        # else's server is on that port and it is not ours to stop.
        if self.client.attached or self.client.process is None:
            return
        if self.client.process.poll() is not None:
            return
        self._log("[watchdog] stopping the ComfyUI this app launched")
        try:
            self.client.shutdown()
        except Exception as exc:                      # noqa: BLE001
            self._log(f"[watchdog] ComfyUI stop failed: {exc}")
