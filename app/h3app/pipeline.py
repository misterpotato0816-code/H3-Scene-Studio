# -*- coding: utf-8 -*-
"""Orchestration of the three graphs, stage events, and continuation logic."""
from __future__ import annotations

import asyncio
import json
import random
import re
import shutil
import time
from pathlib import Path

from . import comfy as comfy_mod
from . import compiler as compiler_mod
from . import graphs
from . import speech_guard, local_security
from .comfy import ComfyClient
from .config import Config, load_manifest, measured_seconds
from .errors import PipelineError, classify
from .projects import STAGES, Project, ProjectStore, profile_cache_key
from .prompts_bridge import build_director_text, prompts, strip_tags

# node id -> stage index, per graph. Graph ids are assigned in graphs.py.
STAGE_ANALYSE, STAGE_PROFILE, STAGE_PROMPT, STAGE_COND, STAGE_VIDEO, STAGE_SAVE, STAGE_DONE = range(7)

_GRAPH_C_STAGES = {
    "restore": STAGE_COND, "clip": STAGE_COND, "clip_report_base": STAGE_COND,
    "clip_select": STAGE_COND, "clip_report_target": STAGE_COND,
    "vae_video": STAGE_COND, "vae_audio": STAGE_COND, "unet": STAGE_COND,
    "lora": STAGE_COND, "prev_video": STAGE_COND, "prev_components": STAGE_COND,
    "tail_images": STAGE_COND, "tail_audio": STAGE_COND, "h3": STAGE_COND,
    "noise": STAGE_VIDEO, "guider": STAGE_VIDEO, "sampler_select": STAGE_VIDEO,
    "scheduler": STAGE_VIDEO, "sampler": STAGE_VIDEO,
    "decode_video": STAGE_SAVE, "decode_audio": STAGE_SAVE,
    "concat_images": STAGE_SAVE, "concat_audio": STAGE_SAVE,
    "create_video": STAGE_SAVE, "save_video": STAGE_SAVE,
    "create_video_seg": STAGE_SAVE, "save_video_seg": STAGE_SAVE,
}


def _graph_a_stage(node_id: str) -> int:
    if node_id.startswith(("load_", "scale_", "sheet")):
        return STAGE_ANALYSE
    return STAGE_PROFILE


class Runner:
    """One generation in flight. Owns its event stream and cancel flag."""

    def __init__(self, pipeline: "Pipeline", project: Project):
        self.pipeline = pipeline
        self.project = project
        self.cancel = asyncio.Event()
        self.started = time.monotonic()
        self.state = {
            "project_id": project.id,
            "stages": [{"name": n, "status": "待機"} for n in STAGES],
            "percent": 0,
            "elapsed": 0,
            "eta": None,
            "error": None,
            "finished": False,
        }
        self.subscribers: list[asyncio.Queue] = []

    # -------------------------------------------------------------- events ---
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(q)
        q.put_nowait(self.snapshot())
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    def snapshot(self) -> dict:
        self.state["elapsed"] = round(time.monotonic() - self.started, 1)
        return json.loads(json.dumps(self.state, ensure_ascii=False))

    def emit(self) -> None:
        snap = self.snapshot()
        for q in list(self.subscribers):
            try:
                q.put_nowait(snap)
            except Exception:
                pass

    def set_stage(self, index: int, status: str = "実行中") -> None:
        for i, st in enumerate(self.state["stages"]):
            if i < index and st["status"] != "失敗":
                st["status"] = "完了"
            elif i == index:
                st["status"] = status
        self.emit()

    def fail(self, exc: BaseException, stage_index: int) -> None:
        stage_name = STAGES[stage_index] if 0 <= stage_index < len(STAGES) else ""
        self.state["stages"][stage_index]["status"] = "失敗"
        self.state["error"] = classify(exc, stage=stage_name)
        self.state["finished"] = True
        self.emit()

    def finish(self) -> None:
        for st in self.state["stages"]:
            if st["status"] in ("待機", "実行中"):
                st["status"] = "完了"
        self.state["percent"] = 100
        self.state["finished"] = True
        self.emit()


class _BusyMarker:
    """Occupies Pipeline.active for a non-Runner job (currently: audio test).

    Only needs to look like a finished-or-not Runner to Pipeline.is_busy.
    """

    def __init__(self):
        self.state = {"finished": False}


class Pipeline:
    def __init__(self, config: Config, store: ProjectStore, client: ComfyClient):
        self.config = config
        self.store = store
        self.client = client
        self.runners: dict[str, Runner] = {}
        self.active: Runner | None = None
        self._lock = asyncio.Lock()
        self._manifest = None

    def _loras_dir(self) -> Path:
        """Shared loras dir, read once from comfy_paths.yaml (single place).

        Falls back to the install-local models/loras. Used for the FAST
        header-level TURBO bind pre-check before anything is submitted.
        """
        try:
            text = Path(self.config.paths_yaml).read_text(encoding="utf-8")
            m = re.search(r"base_path:\s*['\"]([^'\"]+)['\"]", text)
            if m:
                cand = Path(m.group(1)) / "loras"
                if cand.is_dir():
                    return cand
        except Exception:                                    # noqa: BLE001
            pass
        return self.config.comfy_dir / "models" / "loras"

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = load_manifest()
        return self._manifest

    # ------------------------------------------------------------ scheduling --
    async def start(self, project: Project, coro_factory) -> Runner:
        return await self.start_runner(Runner(self, project), coro_factory)

    async def start_runner(self, runner: Runner, coro_factory) -> Runner:
        """Take the single job slot for an already-built runner.

        Story mode goes through here with its own StoryRunner subclass, so a
        story occupies exactly the same slot as a single-shot generation and
        /api/generate, /api/again, /api/continue and /api/release-models are all
        refused for its whole duration.
        """
        async with self._lock:
            if self.active is not None and not self.active.state["finished"]:
                raise PipelineError(
                    "いま別の動画を生成中です。終わるか中止してから実行してください。",
                    kind="unknown")
            self.runners[runner.project.id] = runner
            self._prune_runners(keep=runner.project.id)
            self.active = runner
        asyncio.get_running_loop().create_task(self._guard(runner, coro_factory(runner)))
        return runner

    @property
    def is_busy(self) -> bool:
        return self.active is not None and not self.active.state["finished"]

    async def run_audio_test(self, coro):
        """Take the SAME single job slot as video generation, run `coro`, release.

        音声テスト (pronunciation test) is an independent entry point with its
        own graph builder (h3app.audio_test); it must never run concurrently
        with a video generation, so it shares self._lock/self.active with
        start_runner() above rather than a second slot. `coro` is an
        already-constructed coroutine (the audio-test submit + save flow);
        this wrapper owns nothing about its internals.
        """
        async with self._lock:
            if self.is_busy:
                coro.close()
                raise PipelineError(
                    "いま別の動画を生成中です。終わるか中止してから実行してください。",
                    kind="unknown")
            marker = _BusyMarker()
            self.active = marker
        try:
            return await coro
        finally:
            if self.active is marker:
                self.active = None

    async def release_models(self) -> dict:
        """Drop loaded models from VRAM. ComfyUI itself keeps running.

        Guarded by the same lock that serialises generation, so it cannot slip
        between a job's stages and pull the weights out from under a run.
        """
        async with self._lock:
            if self.is_busy:
                raise PipelineError(
                    "生成中のためモデルを解放できません。"
                    "生成が終わるか中止してから実行してください。",
                    kind="busy")
            return await self.client.free_models()

    # The UI only ever polls the runner it just started, but a long-lived server
    # would otherwise accumulate one finished Runner per generation forever.
    _RUNNER_HISTORY = 8

    def _prune_runners(self, keep: str) -> None:
        # dict preserves insertion order, so this drops the oldest first.
        finished = [pid for pid, r in self.runners.items()
                    if pid != keep and r.state.get("finished")]
        for pid in finished[:-self._RUNNER_HISTORY]:
            self.runners.pop(pid, None)

    async def _guard(self, runner: Runner, coro) -> None:
        try:
            await coro
        except BaseException as exc:      # noqa: BLE001 - surfaced to the UI
            index = self._current_stage_index(runner)
            runner.fail(exc, index)
            runner.project.data["status"] = "error"
            runner.project.save()
        finally:
            if self.active is runner:
                self.active = None

    @staticmethod
    def _current_stage_index(runner: Runner) -> int:
        for i, st in enumerate(runner.state["stages"]):
            if st["status"] == "実行中":
                return i
        for i, st in enumerate(runner.state["stages"]):
            if st["status"] == "待機":
                return i
        return len(STAGES) - 1

    def cancel(self, project_id: str) -> bool:
        runner = self.runners.get(project_id)
        if runner is None or runner.state["finished"]:
            return False
        runner.cancel.set()
        asyncio.get_running_loop().create_task(self.client.interrupt())
        return True

    # ------------------------------------------------------------- graph run --
    @staticmethod
    def _launch_profile(project: Project) -> str:
        """Job-level ComfyUI launch profile from the project mode.

        The whole job (profile/director/generate graphs) shares one profile so
        the server never restarts mid-job. FAST-family → "fast" (--fast),
        everything else → "base" (LEGACY byte-parity untouched).
        """
        from . import modes_v2
        return modes_v2.launch_profile(str(project.data.get("mode") or "LEGACY"))

    async def _release_local_llm(self) -> None:
        """Unload provider models (LM Studio etc.) before a ComfyUI graph runs.

        Real-machine finding (2026-09-13): with director/character_profile
        pointed at a local OpenAI-compatible server, its model (~7 GB, split
        across both GPUs) stays resident after the profile call and starves
        the following video job (6:55 "Model Initializing", ~415 s/step,
        stall timeout hit while ComfyUI kept the GPU at 100%). Best-effort
        only: never raises, never blocks generation on a slow/unreachable
        server, and does nothing when neither role uses an unloadable
        provider. Each effective provider's own adapter.release() is used,
        so LM Studio native calls are never sent to other providers
        (non-unloadable and external providers report supported=False
        without any network). ComfyUI-internal Gemma is NOT handled here
        (its unload travels with the director call itself).
        """
        try:
            from . import ai_settings as ai_mod
            from . import credstore as cred_mod
            from .llm_providers import build_adapter, key_name
        except Exception:                                        # noqa: BLE001
            return
        settings = ai_mod.normalize(self.config.get("ai_settings"))
        targets: dict[str, dict] = {}
        for role_name in ("director", "character_profile"):
            try:
                selection = ai_mod.effective_selection(settings, role_name)
            except Exception:                                    # noqa: BLE001
                continue
            if selection.get("kind") == "gemma":
                continue
            model = str(selection.get("model") or "").strip()
            if not model:
                continue
            pid = str(selection.get("provider") or "")
            slot = targets.setdefault(pid, {
                "base_url": selection.get("base_url") or "",
                "extra": selection.get("extra") or {},
                "vision_models": selection.get("vision_models") or [],
                "models": []})
            if model not in slot["models"]:
                slot["models"].append(model)
        for pid, target in targets.items():
            if pid == "lmstudio":
                # The generation path used LM Studio: record the process
                # identity this session connected to (shutdown will only
                # terminate a process that still matches this record).
                try:
                    from . import shutdown as shutdown_mod
                    shutdown_mod.record_lmstudio_target(
                        self.config, str(getattr(self.client, "session_id", "") or ""))
                except Exception:                                # noqa: BLE001
                    pass
            name = key_name(pid)
            token = cred_mod.load(name) if name else ""
            try:
                adapter = build_adapter(
                    pid, base_url=target["base_url"], token=token,
                    extra=target["extra"],
                    vision_models=target["vision_models"])
                result = await adapter.release(target["models"])
            except Exception as exc:                             # noqa: BLE001
                self.client._log(
                    f"[local_llm] release failed for {pid}: {exc}")
                continue
            if result.get("ok") and result.get("supported"):
                self.client._log(
                    f"[local_llm] released {pid} {target['models']}: "
                    f"{result.get('unloaded')}")
            else:
                self.client._log(
                    f"[local_llm] release skipped for {pid} "
                    f"{target['models']}: {result.get('error')}")

    async def _run(self, runner: Runner, graph: dict, stage_of, *,
                   progress_stage: int | None = None,
                   salvage_nodes: list[str] | None = None,
                   profile: str = "base") -> dict:
        await self._release_local_llm()

        def on_event(ev: dict) -> None:
            node = ev.get("node") or ""
            if ev["type"] == "executing" and node:
                idx = stage_of(node)
                if idx is not None:
                    runner.set_stage(idx)
            elif ev["type"] == "progress":
                idx = stage_of(node) if node else progress_stage
                if idx == progress_stage and ev.get("max"):
                    runner.state["percent"] = int(100 * ev["value"] / max(1, ev["max"]))
                    runner.emit()

        return await self.client.run_graph(graph, on_event=on_event, cancel=runner.cancel,
                                           salvage_nodes=salvage_nodes,
                                           profile=profile)

    # ------------------------------------------------------------- resources --
    def _image_paths(self, images: list[str]) -> list[Path]:
        from . import comfy_inputs
        return [comfy_inputs.local_path(self.config, n) or
                (self.config.comfy_input / n) for n in images]

    def _eta(self, settings: dict) -> float | None:
        """Measured 217.61 s applies at the default profile only."""
        d = self.config.defaults
        if all(str(settings.get(k)) == str(d.get(k)) for k in
               ("width", "height", "frames", "steps", "sampler", "scheduler", "ref_image_size")):
            return measured_seconds()
        base = measured_seconds()
        scale = (int(settings["frames"]) / float(d["frames"])) * \
                (int(settings["width"]) * int(settings["height"])) / \
                (int(d["width"]) * int(d["height"]))
        return round(base * max(0.5, scale), 1)

    # ------------------------------------------------------------ VLM stages --
    async def _try_provider_profile(self, runner: Runner,
                                      images: list[str]) -> str:
        """Character profile via the role's effective provider (WP-A).

        Resolves character_profile to its effective provider/model, sends
        images ONLY when vision_capability(model) is True, otherwise sends
        nothing and records the standing Japanese warning. Returns "" so
        the caller falls through to the local Gemma graph (standing
        fallback), UNLESS gemma_fallback is False, in which case this
        raises PipelineError (kind="ai_config"). Never calls another
        provider from this path (a local failure never reaches external).
        """
        try:
            from . import ai_settings as ai_mod
            from . import credstore as cred_mod
            from .llm_providers import LLMError as _LLMError
            from .llm_providers import build_adapter, key_name
            from .prompts_bridge import prompts
        except Exception:                                        # noqa: BLE001
            return ""
        settings = ai_mod.normalize(self.config.get("ai_settings"))
        try:
            selection = ai_mod.effective_selection(
                settings, "character_profile")
        except Exception:                                        # noqa: BLE001
            return ""
        if selection.get("kind") == "gemma":
            return ""
        gemma_fallback = bool(settings.get("gemma_fallback", True))
        pid = str(selection.get("provider") or "")
        model = str(selection.get("model") or "").strip()

        def _fail(message: str) -> str:
            if gemma_fallback:
                warn = runner.state.get("warning")
                runner.state["warning"] = (str(warn) + " / " + message) \
                    if warn else message
                return ""
            raise PipelineError(message, kind="ai_config")

        if not model:
            return _fail("ローカルLLMの設定が不足しています。"
                        "設定 > LLM接続 でプロバイダーとモデルを指定してください。")
        name = key_name(pid)
        token = cred_mod.load(name) if name else ""
        try:
            adapter = build_adapter(
                pid, base_url=selection.get("base_url") or "",
                model=model, token=token, extra=selection.get("extra") or {},
                vision_models=selection.get("vision_models") or [])
        except Exception as exc:                                 # noqa: BLE001
            return _fail(f"ローカルLLMでの解析に失敗しました（{exc}）。")
        try:
            vision_ok = await adapter.vision_capability(model)
        except Exception:                                        # noqa: BLE001
            vision_ok = None
        if vision_ok is not True:
            return _fail(
                f"キャラクター解析のモデル「{model}」は画像入力に対応していることを"
                "確認できないため、画像を送信しませんでした。")
        try:
            from . import opencode_go as go_mod
            urls = [go_mod.image_to_data_url(p, max_px=768)
                    for p in self._image_paths(images)[:4]]
        except Exception as exc:                                 # noqa: BLE001
            return _fail(f"参照画像を読み込めませんでした（{exc}）。")
        prompts_obj = prompts()
        try:
            text, _info = await adapter.chat(
                system=str(prompts_obj.SYS_CHARACTER_PROFILE),
                user=str(prompts_obj.USER_CHARACTER_PROFILE), images=urls,
                max_tokens=1024, temperature=0.35)
            return str(text or "").strip()
        except _LLMError as exc:
            return _fail(str(exc))
        except Exception as exc:                                 # noqa: BLE001
            return _fail(f"ローカルLLMでの解析に失敗しました（{exc}）。")

    async def _ensure_profile(self, runner: Runner, project: Project) -> str:
        images = project.data["images"]
        key = profile_cache_key(self._image_paths(images))
        project.data["profile_cache_key"] = key

        cached = project.data.get("profile") or self.store.find_cached_profile(key)
        if cached:
            runner.set_stage(STAGE_ANALYSE, "完了")
            runner.set_stage(STAGE_PROFILE, "完了")
            project.data["profile"] = cached
            project.save()
            return cached

        provider_profile = await self._try_provider_profile(runner, images)
        if provider_profile:
            runner.set_stage(STAGE_ANALYSE, "完了")
            runner.set_stage(STAGE_PROFILE, "完了")
            project.data["profile"] = provider_profile
            project.save()
            return provider_profile

        P = prompts()
        runner.set_stage(STAGE_ANALYSE)
        graph = graphs.build_profile_graph(
            images=images, vlm=self.config.vlm,
            ref_longest=self.config.defaults["ref_longest"],
            system_prompt=P.SYS_CHARACTER_PROFILE,
            user_prompt=P.USER_CHARACTER_PROFILE)
        self._dump(graph, f"{project.id}_profile")
        history = await self._run(runner, graph, _graph_a_stage,
                                  profile=self._launch_profile(project))
        profile = strip_tags(comfy_mod.text_output(history, "preview"))
        project.data["profile"] = profile
        project.save()
        runner.set_stage(STAGE_PROFILE, "完了")
        return profile

    async def _run_director(self, runner: Runner, project: Project, *,
                            continuation: bool, jp_prev: str,
                            jp_scene: str | None = None,
                            jp_dialogue: str | None = None,
                            user_prompt: str | None = None,
                            system_prompt: str | None = None,
                            clean_sections: bool = True) -> str:
        """jp_scene / jp_dialogue default to the project's stored values.

        `user_prompt` overrides the assembled director text entirely. Story mode
        passes its own (story_prompts.build_story_director_text) for its
        placeholder contract. Empty speech means silence in both paths.

        They are overridable because feeding the SAME scene and the SAME dialogue
        to the director again is exactly what made "続きを作る" repeat the previous
        beat: story mode passes the current segment's own text, and single-shot
        continuation passes a carry-forward instruction with the already-spoken
        dialogue removed.

        `system_prompt` overrides the director SYSTEM prompt the same way. Story
        mode MUST pass its own: build\\prompts.py's SYS_DIRECTOR_* is
        _SCHEMA_CORE, which mandates <d>[Japanese] ...</d> dialogue ("R1.
        DIALOGUE IS SACRED") and tells the model to compose <Subject 1> from the
        profile. Both are the exact opposite of the story placeholder contract,
        and a system prompt outweighs a user prompt - leaving story mode on
        SYS_DIRECTOR_* makes the model emit <d> tags on every segment. The
        single-shot path never passes it, so its system prompt is untouched.

        `clean_sections` runs compiler.lenient_clean on the model's answer: if
        the whole six-section contract parses AND no internal-monologue marker
        survives, the cleaned re-emission is used; otherwise the text is left
        exactly as strip_tags produced it. That is why the single-shot path can
        take it safely. Story mode passes False because it needs the RAW output:
        its compiler refuses on a monologue marker instead of quietly removing
        it, and a silent clean-up would hide the refusal.
        """
        P = prompts()
        runner.set_stage(STAGE_PROMPT)
        settings = project.data["settings"]
        scene = project.data.get("jp_scene", "") if jp_scene is None else jp_scene
        dialogue = project.data.get("jp_dialogue", "") if jp_dialogue is None else jp_dialogue
        if user_prompt is None:
            user_prompt = build_director_text(
                profile=project.data.get("profile", ""),
                jp_scene=scene,
                jp_prev=jp_prev,
                jp_negative=project.data.get("jp_negative", "") or P.NEGATIVE_DEFAULT,
                jp_dialogue=dialogue,
                frames=int(settings["frames"]))
        if system_prompt is None:
            system_prompt = (P.SYS_DIRECTOR_CONTINUATION if continuation
                             else P.SYS_DIRECTOR_NEW)
        graph = graphs.build_director_graph(
            images=project.data["images"], vlm=self.config.vlm,
            ref_longest=self.config.defaults["ref_longest"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=random.randint(1, 2 ** 31 - 1))
        self._dump(graph, f"{project.id}_director")
        history = await self._run(runner, graph, lambda n: STAGE_PROMPT,
                                  profile=self._launch_profile(project))
        en_prompt = strip_tags(comfy_mod.text_output(history, "preview"))
        if not en_prompt:
            raise PipelineError("プロンプトを生成できませんでした（出力が空です）。",
                                stage=STAGES[STAGE_PROMPT])
        if clean_sections:
            en_prompt = compiler_mod.lenient_clean(en_prompt)
        return en_prompt

    # ----------------------------------------------------------- generation ---
    async def _run_generate(self, runner: Runner, project: Project, en_prompt: str,
                            *, step: int, continuation: dict | None,
                            local_dir: Path | None = None,
                            finish: bool = True,
                            concat_previous: bool = True,
                            expected_speech: str | None = None,
                            export_kind: str = "final",
                            export_stem: str | None = None) -> dict:
        """Build + submit graph C and record the clip.

        `local_dir`        where to keep the app-side copy (default: the normal
                           project folder). Story mode points this at its clips\\.
        `finish`           False keeps the runner (and therefore the single job
                           slot) open, so a story can run segment after segment.
        `concat_previous`  False = tail relay without in-graph concatenation.
        """
        from . import director_spec
        spec = project.data.get("director") or {}
        cast = (spec.get("master") or {}).get("cast") if spec.get("kind") == "story30" else spec.get("cast")
        en_prompt, speech_report = speech_guard.prepare(
            en_prompt,
            project.data.get("jp_dialogue", "") if expected_speech is None else expected_speech,
            allow_s2=director_spec.allow_second_subject(director_spec.normalize_cast(cast)),
            delivery=project.data["settings"].get("speech_delivery", "natural"))
        runner.state["speech_report"] = speech_report
        settings = project.data["settings"]
        runner.state["eta"] = self._eta(settings)
        runner.set_stage(STAGE_COND)

        # v2 Generation Mode (default LEGACY = byte-exact v1 behaviour).
        # resolve() never substitutes silently: unavailable modes and
        # TURBO_BIND_FAILED raise before anything is submitted.
        from . import modes_v2
        mode = str(project.data.get("mode") or "LEGACY").upper()
        loras_dir = self._loras_dir()
        resolved = modes_v2.resolve(mode, settings, self.config.models,
                                    loras_dir=loras_dir)
        settings = resolved["settings"]
        models = resolved["models"]

        # 詳細設定の「保存先」があればそれを使う。無ければ config.json の既定。
        prefix_base = (settings.get("output_prefix") or "").strip() or self.config["output_prefix"]
        try:
            prefix_base = local_security.output_prefix(prefix_base)
        except ValueError as exc:
            raise PipelineError(str(exc), kind="prompt_rejected") from exc
        prefix = f"{prefix_base}/{project.id}_s{step}"

        # GPU plan: resolved once per generate, before any graph is submitted.
        # An invalid saved GPU configuration must never reach ComfyUI.
        from . import gpu as gpu_mod
        plan = gpu_mod.current_plan(self.config)
        if not plan.get("ok"):
            raise PipelineError(
                "GPU設定に問題があるため生成できません: " + "; ".join(plan.get("errors") or []),
                kind="gpu_config")
        clip_device = plan.get("clip_device") or "gpu:1"

        graph = graphs.build_generate_graph(
            images=project.data["images"], en_prompt=en_prompt, settings=settings,
            models=models,
            ref_longest=self.config.defaults["ref_longest"],
            output_prefix=prefix, continuation=continuation,
            concat_previous=concat_previous, clip_device=clip_device)
        if resolved["patches"]:
            graph = modes_v2.apply_patches(graph, resolved["patches"])
        if not models.get("lora"):
            # Base modes: the v1 builder always emits a LoRA node; drop it so
            # the graph cannot silently carry the config's LoRA file.
            graph = modes_v2.drop_lora_for_base(graph)

        # Submit gate: LEGACY = frozen v1 parity, v2 modes = mode-aware checks.
        modes_v2.validate_graph(graph, resolved, self.manifest,
                                self.config.defaults, mode, clip_device=clip_device)
        # Phase 1 item 6: no <Picture N>/<Video N>/<Audio N> may name a
        # reference that is not actually connected in this graph (covers
        # single, continue, story LONG, story LONG_FAST, VLM and direct
        # paths alike - every one of them reaches here).
        try:
            graphs.assert_reference_labels(graph, en_prompt)
        except graphs.GraphError as exc:
            raise PipelineError(
                f"プロンプトが未接続の参照を名指ししています: {exc}",
                stage=STAGES[STAGE_COND])
        self._dump(graph, f"{project.id}_generate_s{step}")

        # W1 device hygiene (overnight 2026-09-07): the director's VLM cycle
        # leaves torch current_device=cuda:1, and the VAELoader nodes below
        # execute before the in-graph H3CudaPhaseRestore, capturing the wrong
        # device (video VAE decodes on GPU1: 124f vae 61s -> 159s). Reset first.
        # Placement only - no pixels, weights, or sampler state are touched.
        # Always runs (even without a director this job): ~seconds of insurance.
        restore_graph = graphs.build_device_restore_graph(
            label=f"app-h3-restore-s{step}")
        self._dump(restore_graph, f"{project.id}_restore_s{step}")
        t_restore_start = time.monotonic()
        await self._run(runner, restore_graph, lambda n: STAGE_COND,
                        progress_stage=STAGE_COND,
                        profile=self._launch_profile(project))
        restore_s = round(time.monotonic() - t_restore_start, 1)

        # If a later node dies after a video is already on disk, keep the video:
        # this run cost minutes of GPU time and the file is the whole point.
        t_gen_start = time.monotonic()
        history = await self._run(runner, graph, _GRAPH_C_STAGES.get,
                                  progress_stage=STAGE_VIDEO,
                                  salvage_nodes=[graphs.NODE_SAVE, graphs.NODE_SAVE_SEG],
                                  profile=self._launch_profile(project))
        gen_seconds = round(time.monotonic() - t_gen_start, 1)

        partial = history.get("_partial_error")
        saved = comfy_mod.saved_video(history, graphs.NODE_SAVE)
        if saved is None:
            if partial:
                # Nothing usable under the main save. Re-raise as a normal failure
                # so the UI still explains what broke.
                raise PipelineError(partial["message"], node=partial.get("node", ""),
                                    stage=STAGES[STAGE_SAVE])
            raise PipelineError("動画は生成されましたが保存結果を取得できませんでした。",
                                stage=STAGES[STAGE_SAVE])
        full_path = comfy_mod.output_path(self.config.comfy_output, saved)
        seg_path = None
        seg = comfy_mod.saved_video(history, graphs.NODE_SAVE_SEG)
        if seg is not None:
            seg_path = comfy_mod.output_path(self.config.comfy_output, seg)

        # Keep a copy next to the project state: the ComfyUI output folder is
        # shared with every other workflow on this machine.
        local_dir = Path(local_dir) if local_dir is not None else self.store.dir_for(project.id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"clip_{step:02d}.mp4"
        try:
            shutil.copy2(full_path, local_path)
        except Exception:
            local_path = full_path

        # Generated-video export: MP4 copy into the user-facing 生成動画
        # folder (完成動画 for finals, クリップ for story segments). This
        # never touches the ComfyUI output or the project's own working
        # copy - it only adds a user-visible export, and export_video()
        # itself never raises or overwrites another file.
        from . import media as media_mod
        stem = export_stem or f"{project.id}_s{step:02d}"
        export_dir = (self.config.media_videos_clips if export_kind == "clip"
                     else self.config.media_videos_final)
        export = media_mod.export_video(
            local_path, dest_dir=export_dir, filename=stem,
            media_root=self.config.media_root, kind=export_kind,
            project_id=project.id)
        export_warning = ""
        if not export["ok"]:
            export_warning = (
                f"動画は生成されましたが、生成動画フォルダへの保存に失敗しました"
                f"（{export['error']}）。プロジェクト内の動画は残っています。")

        export_seg = None
        if seg_path is not None:
            seg_export = media_mod.export_video(
                seg_path, dest_dir=self.config.media_videos_clips,
                filename=f"{project.id}_s{step:02d}_seg",
                media_root=self.config.media_root, kind="clip",
                project_id=project.id)
            export_seg = seg_export
            if not seg_export["ok"] and not export_warning:
                export_warning = (
                    f"動画は生成されましたが、生成動画フォルダへの保存に失敗しました"
                    f"（{seg_export['error']}）。プロジェクト内の動画は残っています。")

        # STABILITY phase: coarse drift record only (never gates).
        # Best-effort frame grab; a failure records None, never an error.
        _identity_frame = None
        try:
            from . import merge as merge_mod
            _identity_frame = await merge_mod.extract_last_frame(
                str(local_path), local_dir / "identity_frame.png")
        except Exception:                                    # noqa: BLE001
            _identity_frame = None
        clip = {
            "step": step,
            "mode": mode,
            "frames": int(settings["frames"]),
            "seconds": graphs.seconds_for_frames(int(settings["frames"])),
            "video": str(full_path),
            "video_file": Path(str(full_path)).name,
            "video_dir": str(Path(str(full_path)).parent),
            "gen_seconds": gen_seconds,
            "local_video": str(local_path),
            "seg": str(seg_path) if seg_path else "",
            "seed": int(settings["seed"]),
            "en_prompt": en_prompt,
            "created_at": time.time(),
            "restore_s": restore_s,
            # STABILITY phase: coarse drift record only (never gates).
            "identity": _single_identity(
                self.config, project.data.get("images") or [],
                _identity_frame),
            # Set when the run failed partway but the video itself survived.
            "warning": (f"動画は保存できましたが、途中で問題が起きました"
                        f"（処理: {partial.get('node') or '不明'}）。"
                        f"再生して問題なければそのまま使えます。") if partial else "",
            "export": export,
        }
        if export_seg is not None:
            clip["export_seg"] = export_seg
        if export_warning:
            clip["warning"] = (f"{clip['warning']} {export_warning}".strip()
                              if clip["warning"] else export_warning)
        project.clips.append(clip)
        if partial:
            runner.state["warning"] = clip["warning"]
        if export_warning:
            existing_warning = runner.state.get("warning") or ""
            runner.state["warning"] = (f"{existing_warning} {export_warning}".strip()
                                       if existing_warning else export_warning)
        project.data["en_prompt"] = en_prompt
        project.data["en_prompt_dialogue"] = (project.data.get("jp_dialogue", "")
                                             if expected_speech is None else expected_speech)
        if finish:
            project.data["status"] = "done"
        project.save()
        if finish:
            runner.finish()
        return clip

    # ------------------------------------------------------------- entry pts --
    @staticmethod
    def _record_consumed(project: Project, *, scene: str, dialogue: str) -> None:
        """Remember what the director was already asked to stage.

        Without this the app has no way of knowing that the dialogue lines have
        already been spoken, which is precisely how "続きを作る" ended up
        re-staging the opening beat every time.
        """
        consumed = project.data.setdefault("consumed", {"scene": [], "dialogue": []})
        consumed.setdefault("scene", [])
        consumed.setdefault("dialogue", [])
        scene = (scene or "").strip()
        dialogue = (dialogue or "").strip()
        if scene:
            consumed["scene"].append(scene)
        if dialogue:
            consumed["dialogue"].append(dialogue)

    async def generate_new(self, runner: Runner) -> None:
        project = runner.project
        project.data["status"] = "running"
        project.save()
        await self._ensure_profile(runner, project)
        en_prompt = await self._run_director(runner, project, continuation=False, jp_prev=_no_prev())
        await self._run_generate(runner, project, en_prompt, step=1, continuation=None)
        self._record_consumed(project, scene=project.data.get("jp_scene", ""),
                              dialogue=project.data.get("jp_dialogue", ""))
        project.save()

    async def generate_director(self, runner: Runner) -> None:
        """AI Director direct path: project.data["en_prompt"] is precompiled.

        No _ensure_profile, no _run_director: the Final Spec was already
        compiled deterministically at request time. Stages flip to done
        instantly (no fake generating duration).
        """
        project = runner.project
        project.data["status"] = "running"
        project.save()
        en_prompt = str(project.data.get("en_prompt") or "").strip()
        if not en_prompt:
            raise PipelineError("H3用プロンプトがありません。",
                                stage=STAGES[STAGE_PROMPT], kind="prompt_rejected")
        runner.set_stage(STAGE_ANALYSE, "完了")
        runner.set_stage(STAGE_PROFILE, "完了")
        runner.set_stage(STAGE_PROMPT, "完了")
        # Local Director calls unload within their own shared job slot. A
        # precompiled/cloud Director prompt must not require llama.cpp nodes.
        await self._run_generate(runner, project, en_prompt, step=1,
                                 continuation=None)
        self._record_consumed(project, scene=project.data.get("jp_scene", ""),
                              dialogue=project.data.get("jp_dialogue", ""))
        project.save()

    async def generate_again(self, runner: Runner) -> None:
        """No VLM at all: cached profile + cached English prompt, new seed."""
        project = runner.project
        project.data["status"] = "running"
        project.save()
        runner.set_stage(STAGE_ANALYSE, "完了")
        runner.set_stage(STAGE_PROFILE, "完了")
        runner.set_stage(STAGE_PROMPT, "完了")
        await self._run_generate(runner, project, project.data["en_prompt"], step=1,
                                 continuation=None,
                                 expected_speech=project.data.get("en_prompt_dialogue"))
        self._record_consumed(project, scene=project.data.get("jp_scene", ""),
                              dialogue=project.data.get("jp_dialogue", ""))
        project.save()

    async def generate_continue(self, runner: Runner, next_instruction: str = "") -> None:
        project = runner.project
        project.data["status"] = "running"
        project.save()
        prev = project.clips[-1]
        step = int(prev["step"]) + 1

        await self._ensure_profile(runner, project)
        jp_prev = _previous_summary(project, prev)
        # THE continuation fix. Re-feeding project.data["jp_scene"] and
        # ["jp_dialogue"] verbatim is what made the director write the same beat
        # again: from the model's point of view nothing about the request had
        # changed. The already-spoken lines are dropped (they are in PREVIOUS
        # CLIP now, not in DIALOGUE), and SCENE becomes either what the user just
        # typed or a plain "carry it forward" instruction.
        next_scene = (next_instruction or "").strip() or CONTINUE_SCENE_DEFAULT
        en_prompt = await self._run_director(runner, project, continuation=True,
                                             jp_prev=jp_prev, jp_scene=next_scene,
                                             jp_dialogue="")

        # LoadVideo can only see files sitting in ComfyUI's input root.
        src = Path(prev["video"])
        if not src.is_file():
            src = Path(prev.get("local_video", ""))
        if not src.is_file():
            raise PipelineError("前の動画ファイルが見つかりませんでした。",
                                stage=STAGES[STAGE_COND], kind="missing_model")
        from . import comfy_inputs
        dest_name = f"h3app_{project.id}_s{prev['step']}.mp4"
        dest = comfy_inputs.stage_path(self.config, dest_name)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)

        continuation = {
            "video": dest_name,
            "prev_total_frames": project.total_frames,
            "tail_frames": int(self.config.defaults["tail_frames"]),
            # Single-shot continuation always relays the tail audio, unchanged.
            "tail_audio": True,
        }
        await self._run_generate(runner, project, en_prompt, step=step,
                                 continuation=continuation, expected_speech="")
        self._record_consumed(project, scene=next_scene, dialogue="")
        project.save()

    # ------------------------------------------------------------------ misc --
    def _dump(self, graph: dict, name: str) -> None:
        try:
            self.config.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.config.debug_dir / f"{name}.json").write_text(
                json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# SCENE text used for "続きを作る" when the user typed no new instruction.
# Deliberately an instruction about MOVEMENT, never a restatement of the original
# request, so the director cannot simply stage the opening beat again.
CONTINUE_SCENE_DEFAULT = (
    "前のクリップの直後から、同じ場所・同じ人物のまま自然に続ける。"
    "すでに行った動作やすでに話した内容は繰り返さず、動きを次の段階へ進める。"
)


def _single_identity(config, images: list, frame: object) -> dict:
    """Coarse drift record for single-shot clips (measure only, never gate).

    Frame vs the first reference image, dHash distance via PIL only.
    """
    try:
        from . import comfy_inputs
        from . import identity_check as identity_mod
        first_ref = (comfy_inputs.local_path(config, images[0])
                    if images else None)
        return {"dhash_bits": identity_mod.dhash_bits(
            str(first_ref) if first_ref else "",
            str(frame) if frame else "")}
    except Exception:                                        # noqa: BLE001
        return {"dhash_bits": None}


def _no_prev() -> str:
    """The literal marker, not an empty string.

    An empty PREVIOUS CLIP block reads to the VLM as "there was a previous clip
    you were not told about"; the marker tells it there was none.
    """
    return "NONE - this is a new clip, ignore this section."


_SUMMARY_RE = re.compile(r"^\s*summary\s*$(.*?)^\s*retention_analysis\s*$",
                         re.MULTILINE | re.DOTALL)


def _previous_summary(project: Project, prev_clip: dict) -> str:
    """Auto-fill PREVIOUS CLIP: the JP scene text plus the previous summary section.

    The English `summary` section is the model's own compressed statement of what
    the last clip was, which continues far more reliably than the Japanese request
    alone. If it cannot be parsed, the Japanese scene text is used by itself.
    """
    jp_scene = project.data.get("jp_scene", "").strip()
    m = _SUMMARY_RE.search(prev_clip.get("en_prompt", "") or "")
    if m:
        summary = m.group(1).strip()
        if summary:
            return f"{jp_scene}\n\n[previous summary] {summary}" if jp_scene else summary
    return jp_scene or _no_prev()
