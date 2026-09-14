# -*- coding: utf-8 -*-
"""Story mode orchestration.

An UPPER-LEVEL controller over the existing pipeline, not a second pipeline:
every segment goes through Pipeline._ensure_profile / _run_director /
_run_generate, so the graph that reaches ComfyUI is byte-for-byte the verified
H3_HETERO_V1 chain.

Per segment:
  1. profile (analysed ONCE per story, cached on the project afterwards)
  2. director, with the SEGMENT's own jp_scene / jp_dialogue
  3. generate, tail-relayed from the previous clip with concat_previous=False
  4. copy the clip, extract its last frame, cursor += 1, SAVE   <- autosave
  5. honour stop_requested between segments (never mid-clip)
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

from . import canon as canon_mod
from . import comfy_inputs
from . import compiler as compiler_mod
from . import duration as duration_mod
from . import merge as merge_mod
from . import pipeline as pipeline_mod
from . import segmenter as segmenter_mod
from . import presets as presets_mod
from . import story_prompts
from . import transitions as transitions_mod
from .config import ACTION_PRESET_IDS, FPS, preset_text
from .errors import PipelineError, classify
from .pipeline import STAGE_COND, STAGE_PROMPT, Pipeline, Runner
from .projects import STAGES
from .prompts_bridge import prompts
from .story import StoryProject, StoryStore


# --------------------------------------------------------------- pre-flight --
def preflight_records(story: StoryProject) -> list[dict]:
    """One check record per segment. Pure: no I/O, no GPU, no clock.

    Produced BEFORE anything is submitted so the UI can show exactly what is
    about to be generated - in particular which segments are silent and which
    seed each one will use.
    """
    cursor = max(0, int(story.data.get("cursor", 0)))
    records: list[dict] = []
    for i, seg in enumerate(story.segments):
        prompt = (seg.get("prompt") or "").strip()
        speech = (seg.get("speech") or "").strip()
        styles = [str(p) for p in (seg.get("motion_presets") or [])
                  if str(p) not in ACTION_PRESET_IDS]
        actions = [str(p) for p in (seg.get("action_presets") or [])]
        actions += [str(p) for p in (seg.get("motion_presets") or [])
                    if str(p) in ACTION_PRESET_IDS and str(p) not in actions]
        done = seg.get("status") == "done" and story.clip_for(i) is not None
        pending = (i >= cursor) and not done

        problems: list[str] = []
        warnings: list[str] = []
        if pending and not prompt and not speech and not styles and not actions:
            problems.append(f"セグメント {i + 1}: 指示もセリフもありません。"
                            "文章を入力するか、このセグメントを削除してください。")
        if pending and speech and i > 0:
            prev_speech = (story.segments[i - 1].get("speech") or "").strip()
            if prev_speech and prev_speech == speech:
                warnings.append(f"セグメント {i + 1}: 直前のセグメントと同じセリフです。")
        for ln in (seg.get("lines") or []):
            if isinstance(ln, dict) and ln.get("type") not in ("prompt", "speech"):
                problems.append(f"セグメント {i + 1}: 行の種類が不正です "
                                f"（{ln.get('type')!r}）。")
                break

        records.append({
            "index": i,
            "prompt": prompt,
            "speech": speech,
            "silent": not speech,
            "seed": story.seed_for(i),
            "previous_context": i > 0,
            "style_presets": styles,
            "action_presets": actions,
            "pending": pending,
            "problems": problems,
            "warnings": warnings,
        })
    return records


def assert_preflight(story: StoryProject) -> list[dict]:
    """Refuse to start a story whose segments are inconsistent."""
    records = preflight_records(story)
    problems = [p for r in records for p in r["problems"]]
    if not any(r["pending"] for r in records):
        problems.append("生成するセグメントがありません。"
                        "すべて生成済みか、カーソルが最後まで進んでいます。")
    if problems:
        raise PipelineError(
            "セグメントの内容を確認してください。生成は開始していません。\n"
            + "\n".join(problems), kind="prompt_rejected")
    return records


class StoryRunner(Runner):
    """The existing Runner's event/stage shape plus the story-level fields."""

    def __init__(self, pipeline: Pipeline, story: StoryProject):
        super().__init__(pipeline, story)
        self.story = story
        self.stop = asyncio.Event()
        self.state.update({
            "story_id": story.id,
            "story_status": story.data.get("status", "pending"),
            "segment_index": int(story.data.get("cursor", 0)),
            "segment_total": len(story.segments),
            "segment_prompt": "",
            "segment_speech": "",
            "clips_done": len(story.clips),
        })

    def begin_segment(self, index: int, segment: dict) -> None:
        self.state["segment_index"] = index
        self.state["segment_total"] = len(self.story.segments)
        self.state["segment_prompt"] = segment.get("prompt", "")
        self.state["segment_speech"] = segment.get("speech", "")
        self.state["percent"] = 0
        self.state["warning"] = ""
        for st in self.state["stages"]:
            st["status"] = "待機"
        self.emit()

    def end_segment(self) -> None:
        for st in self.state["stages"]:
            if st["status"] in ("待機", "実行中"):
                st["status"] = "完了"
        self.state["percent"] = 100
        self.state["clips_done"] = len(self.story.clips)
        self.emit()

    def set_story_status(self, status: str) -> None:
        self.state["story_status"] = status
        self.state["clips_done"] = len(self.story.clips)
        self.emit()


class StoryPipeline:
    """Holds a reference to the existing Pipeline; owns no lock of its own."""

    def __init__(self, pipeline: Pipeline, store: StoryStore):
        self.pipeline = pipeline
        self.store = store
        self.config = pipeline.config
        self.runners: dict[str, StoryRunner] = {}

    # ------------------------------------------------------------ scheduling --
    def runner_for(self, story_id: str) -> StoryRunner | None:
        return self.runners.get(story_id)

    def is_running(self, story_id: str) -> bool:
        runner = self.runners.get(story_id)
        return runner is not None and not runner.state["finished"]

    async def start(self, story: StoryProject) -> StoryRunner:
        if self.is_running(story.id):
            raise PipelineError("このストーリーはすでに生成中です。", kind="busy")
        if not story.segments:
            raise PipelineError("セグメントがありません。先に文章を分割してください。",
                                kind="prompt_rejected")
        # Pre-flight gate: runs BEFORE the job slot is taken and before any GPU
        # work, so an inconsistent story never reaches ComfyUI at all.
        records = assert_preflight(story)
        story.data["preflight"] = records
        story.save()
        runner = StoryRunner(self.pipeline, story)
        previous = self.runners.get(story.id)
        # Registered BEFORE the task can start, so the watchdog can never see a
        # running story it does not know how to stop.
        self.runners[story.id] = runner
        try:
            # The SAME job slot as single-shot generation.
            await self.pipeline.start_runner(runner, lambda r: self._run(r))
        except Exception:
            if previous is not None:
                self.runners[story.id] = previous
            else:
                self.runners.pop(story.id, None)
            raise
        return runner

    async def request_stop(self, story: StoryProject) -> bool:
        """一時保存して停止. The clip in flight is allowed to finish."""
        story.data["stop_requested"] = True
        story.save()
        runner = self.runners.get(story.id)
        if runner is not None and not runner.state["finished"]:
            runner.stop.set()
            runner.set_story_status("stopping")
            return True
        # Not running: nothing to wait for, just leave it resumable.
        if story.data.get("status") == "running":
            story.data["status"] = "paused"
        story.data["stop_requested"] = False
        story.save()
        return False

    # ----------------------------------------------------------------- loop ---
    async def _run(self, runner: StoryRunner) -> None:
        story = runner.story
        story.data["status"] = "running"
        story.data["stop_requested"] = False
        story.data["error"] = None
        story.save()
        runner.set_story_status("running")

        final = "done"
        try:
            while True:
                total = len(story.segments)
                cursor = int(story.data.get("cursor", 0))
                if cursor >= total:
                    final = "done"
                    break
                if runner.stop.is_set() or story.data.get("stop_requested"):
                    final = "paused"
                    break
                segment = story.segments[cursor]
                if segment.get("status") == "done" and story.clip_for(cursor):
                    # Never regenerate a segment that already produced a clip.
                    # LOCKED/APPROVED is the explicit form of the same promise.
                    story.data["cursor"] = cursor + 1
                    story.save()
                    continue
                await self._run_segment(runner, story, cursor, segment)
        except Exception as exc:                      # noqa: BLE001 - shown in UI
            final = "error"
            info = classify(exc, stage=STAGES[self.pipeline._current_stage_index(runner)])
            story.data["error"] = info
            cursor = int(story.data.get("cursor", 0))
            if 0 <= cursor < len(story.segments):
                story.segments[cursor]["status"] = "error"
                story.segments[cursor]["error"] = info.get("title", str(exc))
            runner.fail(exc, self.pipeline._current_stage_index(runner))

        story.data["status"] = final
        story.data["stop_requested"] = False
        story.save()
        runner.set_story_status(final)
        if not runner.state["finished"]:
            runner.finish()

        if final == "paused":
            # Requirement 6: once stopped, let go of the models. Safe here because
            # the runner is already marked finished, so release_models() no longer
            # sees a busy pipeline and cannot deadlock on the job lock.
            await self._release_quietly()

    async def _release_quietly(self) -> None:
        try:
            await self.pipeline.release_models()
        except Exception as exc:                      # noqa: BLE001
            print(f"[story] release_models skipped: {exc}", flush=True)

    # -------------------------------------------------------------- segment ---
    async def _run_segment(self, runner: StoryRunner, story: StoryProject,
                           index: int, segment: dict) -> None:
        runner.begin_segment(index, segment)
        segment["status"] = "running"
        segment["error"] = ""
        story.save()

        profile = await self.pipeline._ensure_profile(runner, story)

        # CHARACTER CANON. Parsed by the app, never rewritten by the LLM. The
        # director is shown the canon's own KEY: block rather than the raw blob,
        # so preamble or internal monologue in the profile cannot travel on.
        canon = canon_mod.parse_profile(profile)
        overrides = self._overrides_for(story, index)
        profile_for_director = canon.profile_text() or profile

        # Tail source = the previous SEGMENT's own clip, never clips[-1]:
        # when clip i is regenerated, clip i+1.. clips are stale and the only
        # valid motion/pose aid is the kept predecessor. Identity truth stays
        # the ORIGINAL references (re-anchored every clip by the graph builder).
        prev_clip = story.clip_for(index - 1) if index > 0 else None
        jp_scene = self._scene_text(story, segment)
        jp_dialogue = (segment.get("speech") or "").strip()
        dialogue_lines = story_prompts.speech_lines(jp_dialogue)
        jp_prev = self._prev_text(story, index, prev_clip)

        # TRANSITION CONTRACT. Planned over the WHOLE segment list, because a
        # boundary belongs to a PAIR of segments: the hand-off into a new
        # physical action is written onto the EARLIER segment's exit state, so
        # the change is performed across the cut instead of at it. Pure and
        # deterministic, so a resumed story plans exactly the same states.
        delivery = self._delivery_for(story, segment)
        transition = self._transition_for(story, index)
        segment["entry_state"] = dict(transition["entry_state"])
        segment["exit_state"] = dict(transition["exit_state"])

        # Deterministic per-segment seed. Written into settings BEFORE the graph is
        # built, so the clip the pipeline records already carries the right value.
        seed = story.seed_for(index)
        story.data["settings"]["seed"] = int(seed)
        segment["seed"] = int(seed)

        # AI Director direct path: precompiled en_prompt skips the second
        # VLM director AND the story compiler entirely. Nothing downstream
        # changes (continuation, debug, generate, merge all run as usual).
        direct_precompiled = str(segment.get("direct_en_prompt") or "").strip()
        if direct_precompiled:
            runner.set_stage(STAGE_PROMPT, "完了")
            t_director_start = t_director_end = time.monotonic()
            raw_prompt = ""
            compile_report = {"ok": True}
            report = {"ok": True}
            director_text = ""
            en_prompt = direct_precompiled
        else:
            # Story-controlled director text: SLOT METADATA ONLY. The dialogue text
            # itself is never handed to the model - it writes <<LINE_n>> placeholders
            # and the compiler below substitutes the user's exact lines.
            director_text = story_prompts.build_story_director_text(
                profile=profile_for_director,
                segment_prompt=jp_scene,
                segment_speech=jp_dialogue,
                dialogue_lines=dialogue_lines,
                previous_context=jp_prev,
                jp_negative=story.data.get("jp_negative", "") or prompts().NEGATIVE_DEFAULT,
                frames=int(story.data["settings"]["frames"]),
                transition=transition, delivery=delivery)

            # The SYSTEM prompt is story-owned too. build\prompts.py's SYS_DIRECTOR_*
            # mandates <d>[Japanese] dialogue and tells the model to compose
            # <Subject 1> itself - both the opposite of the compiler contract, and a
            # system prompt beats a user prompt every time.
            t_director_start = time.monotonic()
            raw_prompt = await self.pipeline._run_director(
                runner, story, continuation=prev_clip is not None, jp_prev=jp_prev,
                jp_scene=jp_scene, jp_dialogue=jp_dialogue, user_prompt=director_text,
                system_prompt=story_prompts.story_system_prompt(
                    continuation=prev_clip is not None),
                clean_sections=False)
            t_director_end = time.monotonic()

            # COMPILE: schema-bounded extraction, canonical <Subject 1>, placeholder
            # substitution. Then VALIDATE the final text. Both are pure and both run
            # BEFORE a single frame is generated.
            en_prompt, compile_report = compiler_mod.compile_story_prompt(
                raw_prompt, canon=canon, overrides=overrides,
                dialogue_lines=dialogue_lines)
            report = compiler_mod.validate_final_prompt(
                en_prompt, index=index, dialogue_lines=dialogue_lines,
                canon=canon, overrides=overrides,
                other_dialogue=self._other_dialogue_lines(story, index),
                other_staging=self._other_staging(story, index))
            if not compile_report["ok"] or not report["ok"]:
                raise PipelineError(
                    compiler_mod.validation_error_message(index, compile_report, report),
                    stage=STAGES[STAGE_PROMPT], kind="prompt_rejected")

        mode = str(story.data.get("mode") or "LONG").upper()
        continuation = await self._continuation(story, prev_clip, mode=mode)
        self._write_debug(story, index, segment,
                          scene=jp_scene, speech=jp_dialogue, profile=profile,
                          director_text=director_text, raw_prompt=raw_prompt,
                          en_prompt=en_prompt, seed=seed,
                          continuation=continuation, report=report,
                          compile_report=compile_report,
                          canon=canon, overrides=overrides,
                          transition=transition)

        t_generate_start = time.monotonic()
        export_stem = f"{_safe_name(story.data.get('name') or story.id)}_{story.id}_clip{index + 1:02d}"
        clip = await self.pipeline._run_generate(
            runner, story, en_prompt, step=index + 1, continuation=continuation,
            local_dir=story.clips_dir, finish=False, concat_previous=False,
            expected_speech=jp_dialogue,
            export_kind="clip", export_stem=export_stem)
        t_generate_end = time.monotonic()

        clip["segment_index"] = index
        clip["jp_scene"] = jp_scene
        clip["jp_dialogue"] = jp_dialogue
        clip["seed"] = int(seed)
        clip.setdefault("last_frame", "")

        frame = await merge_mod.extract_last_frame(
            clip.get("local_video") or clip.get("video"),
            story.frames_dir / f"frame_{index:02d}.png")
        clip["last_frame"] = str(frame) if frame else ""
        # STABILITY phase: coarse drift record only (never gates generation).
        try:
            from . import identity_check as identity_mod
            refs = story.data.get("images") or []
            first_ref = comfy_inputs.local_path(self.config, refs[0]) if refs else None
            clip["identity"] = {
                "dhash_bits": identity_mod.dhash_bits(
                    str(first_ref) if first_ref else "",
                    str(frame) if frame else "")}
        except Exception:                                    # noqa: BLE001
            clip["identity"] = {"dhash_bits": None}

        if mode == "LONG_FAST" and index == 0:
            # Voice anchor: clip 0's own audio, always re-derived whenever clip 0
            # is (re)generated - never bookkept as write-once. LONG_FAST drops
            # tail_audio entirely (0f relay), so without a fixed Voice Master the
            # identity's voice is unconditioned after clip 0 - never a later
            # clip's audio, or the anchor could drift like the tail relay it
            # replaces. Regenerating clip 0 (api_story_regenerate) and a
            # structure change that makes segment 0 pending both route through
            # here, so re-deriving unconditionally covers both without needing
            # separate invalidation calls scattered elsewhere.
            await self._stage_voice_master(
                story, clip.get("local_video") or clip.get("video"))

        segment["status"] = "done"
        segment["error"] = ""
        segment["clip"] = clip.get("local_video") or clip.get("video", "")
        # Overnight 2026-09-07 thrash instrumentation (additive, behaviour-free):
        # per-segment director/generate walls + the director→generate gap, which
        # contains the PurgeVRAM + H3 re-stage cost. Persisted by story.save().
        segment["timings"] = {
            "director_s": round(t_director_end - t_director_start, 1),
            "gap_director_to_generate_s": round(t_generate_start - t_director_end, 1),
            "generate_s": round(t_generate_end - t_generate_start, 1),
            "gen_sampling_s": clip.get("gen_seconds", 0),
            "restore_s": clip.get("restore_s", 0),
            # Direct path: the Director LLM ran in the create phase, not per
            # clip; totals live in story.data["director_llm"]. VLM path: the
            # per-clip director call above.
            "director_llm_s": 0.0 if direct_precompiled else round(
                t_director_end - t_director_start, 1),
        }
        # v2 clip cache identity: same key + existing file = never rebuild.
        segment["v2cache"] = {
            "key": _clip_cache_key(
                story, index, en_prompt, seed,
                self.pipeline.config.models),
            "clip": segment["clip"],
        }
        story.data["cursor"] = index + 1
        story.data["merged"] = False
        story.data["status"] = "running"
        story.save()                                   # per-clip autosave
        runner.end_segment()

    # ----------------------------------------------------------------- text ---
    @staticmethod
    def _scene_text(story: StoryProject, segment: dict) -> str:
        """SCENE text for ONE segment.

        Project-level presets are STYLE only ("落ち着いた雰囲気"), so appending them
        to every segment is harmless. ACTION presets ("手を振る") are one-off and
        come exclusively from THIS segment - broadcasting them is what made the
        character wave in every single clip.
        """
        parts: list[str] = []
        body = (segment.get("prompt") or "").strip()
        if body:
            parts.append(body)
        style_ids = [str(p) for p in (story.data.get("motion_presets") or [])
                     if str(p) not in ACTION_PRESET_IDS]
        style_ids += [str(p) for p in (segment.get("motion_presets") or [])
                      if str(p) not in ACTION_PRESET_IDS]
        action_ids = [str(p) for p in (segment.get("action_presets") or [])]
        action_ids += [str(p) for p in (segment.get("motion_presets") or [])
                       if str(p) in ACTION_PRESET_IDS]
        seen: set[str] = set()
        for pid in style_ids + action_ids:
            text = preset_text(pid)
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
        if parts:
            return "\n".join(parts)
        # Only "おまかせ" and no free text: give the director something concrete
        # rather than an empty SCENE block.
        try:
            return prompts().JP_SCENE_DEFAULT
        except Exception:
            return "自然な動きでカメラに向かって話す。"

    @staticmethod
    def _delivery_for(story: StoryProject, segment: dict) -> dict:
        """How this segment is spoken, from what the app already knows.

        Project + segment STYLE presets, the segment's own ACTION signals, and
        the segment's explicit `delivery` field if it has one. Nothing is
        guessed from free text.

        An ACTION signal is either a legacy `action_presets` id (old stories) or
        one deterministic sentence that an action-preset chip press wrote into
        this segment's prompt (presets.APPLY_TEMPLATE). Counting both is what
        keeps `action_load` meaningful now that a chip press writes prompt text
        instead of an id.
        """
        styles = [str(p) for p in (story.data.get("motion_presets") or [])
                  if str(p) not in ACTION_PRESET_IDS]
        styles += [str(p) for p in (segment.get("motion_presets") or [])
                   if str(p) not in ACTION_PRESET_IDS]
        actions = [str(p) for p in (segment.get("action_presets") or [])]
        actions += [str(p) for p in (segment.get("motion_presets") or [])
                    if str(p) in ACTION_PRESET_IDS and str(p) not in actions]
        actions += presets_mod.applied_action_names(segment.get("prompt") or "")
        return duration_mod.delivery_from_presets(
            styles, actions, explicit=segment.get("delivery"))

    @staticmethod
    def _transitions(story: StoryProject) -> list[dict]:
        """The transition plan for the WHOLE story. Pure: same in, same out."""
        return transitions_mod.plan_transitions(
            story.segments,
            deliveries=[StoryPipeline._delivery_for(story, seg)
                        for seg in story.segments])

    @staticmethod
    def _transition_for(story: StoryProject, index: int) -> dict:
        """One segment's entry/exit contract, planned in context."""
        return transitions_mod.record_for(
            story.segments, index,
            deliveries=[StoryPipeline._delivery_for(story, seg)
                        for seg in story.segments])

    @staticmethod
    def _segment_seconds(story: StoryProject) -> float:
        """The clip length this story renders at, in seconds. 24 fps is fixed."""
        try:
            return float(int(story.data["settings"]["frames"])) / 24.0
        except (KeyError, TypeError, ValueError):
            return float(segmenter_mod.DEFAULT_SECONDS)

    @staticmethod
    def _prev_text(story: StoryProject, index: int, prev_clip: dict | None) -> str:
        """PREVIOUS CLIP block. Never the CURRENT segment's own text.

        No dialogue TEXT at all, from any segment. The director is told only how
        many lines were already spoken, so it knows the conversation moved on
        without ever holding a character of the script - a model that cannot see
        the words cannot repeat or invent them.
        """
        if prev_clip is None or index <= 0:
            return pipeline_mod._no_prev()
        pieces: list[str] = []
        previous = story.segments[index - 1] if index - 1 < len(story.segments) else {}
        prev_prompt = (previous.get("prompt") or "").strip()
        prev_lines = story_prompts.speech_lines(previous.get("speech") or "")
        if prev_prompt:
            pieces.append(f"[what already happened] {prev_prompt}")
        if prev_lines:
            pieces.append(
                f"[the previous clip already contained {len(prev_lines)} spoken "
                "line(s). They have been said. Do not speak or re-stage them.]")
        m = pipeline_mod._SUMMARY_RE.search(prev_clip.get("en_prompt", "") or "")
        if m and m.group(1).strip():
            pieces.append(f"[previous summary] {m.group(1).strip()}")
        return "\n\n".join(pieces) or pipeline_mod._no_prev()

    @staticmethod
    def _other_speech(story: StoryProject, index: int) -> list[str]:
        """Every OTHER segment's speech: none of it may be spoken in this clip."""
        return [(seg.get("speech") or "").strip()
                for i, seg in enumerate(story.segments)
                if i != index and (seg.get("speech") or "").strip()]

    @staticmethod
    def _other_dialogue_lines(story: StoryProject, index: int) -> list[str]:
        """Individual lines of every OTHER segment, minus this segment's own."""
        own = set(story_prompts.speech_lines(
            (story.segments[index].get("speech") or "")
            if 0 <= index < len(story.segments) else ""))
        out: list[str] = []
        for i, seg in enumerate(story.segments):
            if i == index:
                continue
            for line in story_prompts.speech_lines(seg.get("speech") or ""):
                if line not in own and line not in out:
                    out.append(line)
        return out

    @staticmethod
    def _other_staging(story: StoryProject, index: int) -> list[str]:
        """Other segments' SCENE text, minus anything this segment also asks for.

        These are Japanese instruction lines. The final prompt is English apart
        from the <d> payloads, so one of them appearing verbatim means another
        segment's staging leaked in.
        """
        own_scene = StoryPipeline._scene_text(
            story, story.segments[index]) if 0 <= index < len(story.segments) else ""
        own = {ln.strip() for ln in own_scene.splitlines() if ln.strip()}
        out: list[str] = []
        for i, seg in enumerate(story.segments):
            if i == index:
                continue
            for line in (seg.get("prompt") or "").splitlines():
                line = line.strip()
                if line and line not in own and line not in out:
                    out.append(line)
        return out

    @staticmethod
    def _overrides_for(story: StoryProject, index: int) -> dict:
        """Explicit structured attribute overrides in force for this segment.

        Folded from segment 0 up to `index`, later wins: once a segment says the
        outfit changed, the change PERSISTS - that is what makes the trait
        persistent rather than a one-shot decoration. A segment that carries no
        `attribute_overrides` changes nothing.
        """
        return canon_mod.merge_overrides(
            seg.get("attribute_overrides") for seg in story.segments[:index + 1])

    # ---------------------------------------------------------------- debug ---
    @staticmethod
    def _write_debug(story: StoryProject, index: int, segment: dict, *,
                     scene: str, speech: str, profile: str, director_text: str,
                     raw_prompt: str, en_prompt: str, seed: int,
                     continuation: dict | None, report: dict,
                     compile_report: dict | None = None,
                     canon=None, overrides: dict | None = None,
                     transition: dict | None = None) -> None:
        """app\\story_projects\\<id>\\debug\\seg_NN.json - everything needed to
        explain one segment afterwards. Never surfaced in the normal UI."""
        # The duration reasoning behind this segment, recomputed from what is
        # actually about to be generated. Written next to the real clip so the
        # estimate can be calibrated against the measured result later.
        delivery = StoryPipeline._delivery_for(story, segment)
        timing = segmenter_mod.segment_timing(
            segment, seconds=StoryPipeline._segment_seconds(story),
            delivery=delivery)
        # Requirement 15: the transition contract this segment was actually
        # given. A bad boundary can then be attributed - "the director was told
        # the wrong thing" is a different bug from "H3 ignored what it was told".
        tr = transition if isinstance(transition, dict) else {}
        record = {
            "segment_index": index,
            "written_at": time.time(),
            "source_lines": [dict(ln) for ln in (segment.get("lines") or [])],
            "segment_prompt": (segment.get("prompt") or "").strip(),
            "segment_speech": speech,
            "silent": not speech,
            "style_presets": [str(p) for p in (story.data.get("motion_presets") or [])]
                             + [str(p) for p in (segment.get("motion_presets") or [])
                                if str(p) not in ACTION_PRESET_IDS],
            "action_presets": [str(p) for p in (segment.get("action_presets") or [])]
                              + [str(p) for p in (segment.get("motion_presets") or [])
                                 if str(p) in ACTION_PRESET_IDS],
            "effective_scene_text": scene,
            # Duration model (see h3app\\duration.py). Estimates, not measurements.
            "delivery": delivery,
            "timing": timing,
            # Transition contract (see h3app\\transitions.py).
            "segment_id": str(segment.get("segment_id") or ""),
            "previous_segment_id": str(tr.get("previous_segment_id") or ""),
            "entry_state": dict(tr.get("entry_state") or {}),
            "exit_state": dict(tr.get("exit_state") or {}),
            "continuity_constraints": [dict(c) for c in
                                       (tr.get("continuity_constraints") or [])],
            "transition_block": transitions_mod.transition_block(tr),
            "profile_used": bool(profile),
            "director_user_prompt": director_text,
            "director_raw_output": raw_prompt,
            "final_en_prompt": en_prompt,
            "seed": int(seed),
            "previous_clip_reference": continuation is not None,
            "tail_frames": int((continuation or {}).get("tail_frames", 0)),
            "tail_audio": bool((continuation or {}).get("tail_audio", True))
                          if continuation else False,
            # The validator's structured report: the hard gate that decided
            # whether this segment was allowed to reach the H3 graph.
            "speech_policy": report,
            "validation": report,
            "compile_report": compile_report or {},
            "character_canon": (canon.as_dict()
                                if canon is not None and hasattr(canon, "as_dict")
                                else {}),
            "attribute_overrides": dict(overrides or {}),
            "profile_schema_version": canon_mod.PROFILE_SCHEMA_VERSION,
        }
        try:
            story.debug_dir.mkdir(parents=True, exist_ok=True)
            (story.debug_dir / f"seg_{index:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:                      # noqa: BLE001 - never fatal
            print(f"[story] debug record not written for segment {index}: {exc}",
                  flush=True)

    # ---------------------------------------------------------- voice anchor --
    async def _stage_voice_master(self, story: StoryProject, clip_video) -> str:
        """(Re-)derive the Voice Master anchor from `clip_video` - whichever
        clip is CURRENTLY segment 0 - and stage it as
        `h3app_story_{id}_voice_master.wav`. Always overwrites the staged file:
        unlike last-frame staging, a size comparison is not enough here (two
        different 5s/32kHz/stereo clips can coincide in byte size), and a stale
        anchor left behind after clip 0 changed is exactly the bug this fixes.
        Raises the typed PipelineError if clip 0's audio cannot be extracted.
        """
        wav = await merge_mod.extract_audio(
            clip_video, story.frames_dir / "voice_master_00.wav")
        if wav is None:
            raise PipelineError(
                "1本目のクリップから音声を抽出できませんでした。"
                "LONG_FASTは声を固定するためVoice Masterが必須です。",
                stage=STAGES[STAGE_COND], kind="missing_model")
        voice_dest_name = f"h3app_story_{story.id}_voice_master.wav"
        voice_dest = comfy_inputs.stage_path(self.config, voice_dest_name)
        # copy2 is not atomic, but that is safe here: Pipeline has a single
        # global job slot so segments run strictly sequentially (no reader can
        # observe a partially written file), story.data["voice_master"] is only
        # assigned AFTER copy2 returns, story.save() only happens after the
        # whole segment completes, and the destination filename is
        # deterministic so any later re-derivation fully overwrites it. A
        # future change to the concurrency model should revisit this.
        shutil.copy2(wav, voice_dest)
        story.data["voice_master"] = voice_dest_name
        # H3_Media export (best-effort copy; staging logic untouched).
        try:
            from . import media as media_mod
            media_mod.export_copy(wav, self.pipeline.config.media_audio,
                                  f"{story.id}_voice_master.wav",
                                  media_root=self.pipeline.config.media_root,
                                  media_type="voice", project_id=story.id)
        except Exception:                                    # noqa: BLE001
            pass
        return voice_dest_name

    # --------------------------------------------------------- continuation ---
    async def _continuation(self, story: StoryProject, prev_clip: dict | None, *,
                      mode: str = "LONG", tail_audio: bool = True) -> dict | None:
        """`tail_audio` is threaded through to the graph builder and defaults to
        today's behaviour (the tail's audio conditions the new clip). It exists so
        that "a silent segment must not inherit the previous voice" is one flag.

        `mode` selects the relay shape. Every mode except LONG_FAST keeps today's
        byte-identical tail-video relay (tail_frames from config.defaults,
        tail_audio=True). LONG_FAST uses the measured V3 recipe's 0-frame tail
        relay instead: no tail video/audio conditioning at all, only the previous
        clip's last extracted frame passed as one extra reference image, plus a
        fixed Voice Master that always tracks whatever clip is CURRENTLY
        segment 0 (see `_stage_voice_master`), for every later clip.
        """
        if prev_clip is None:
            return None
        if mode == "LONG_FAST":
            frame_src = Path(prev_clip.get("last_frame") or "")
            if not frame_src.is_file():
                raise PipelineError(
                    "前のクリップの最終フレームが見つかりませんでした。"
                    "LONG_FASTは0フレームtail relayのため必須です。",
                    stage=STAGES[STAGE_COND], kind="missing_model")
            dest_name = (f"h3app_story_{story.id}_s{int(prev_clip.get('step', 0))}"
                        "_lastframe.png")
            dest = comfy_inputs.stage_path(self.config, dest_name)
            if not dest.exists() or dest.stat().st_size != frame_src.stat().st_size:
                shutil.copy2(frame_src, dest)
            # Voice anchor: clip 0's own audio, never a later clip's - normally
            # already staged in _run_segment right after clip 0 finishes. But a
            # story resumed from a snapshot, or one whose clip 0 is
            # LOCKED/APPROVED (see backend/h3_v2/story_v2.py plan()'s FROZEN
            # statuses) never runs that step again even though clip 0's video
            # is right there on disk - so derive the anchor from it now instead
            # of refusing. Only clip 0's video being genuinely unavailable is a
            # real refusal: an unanchored LONG_FAST run must stay impossible.
            voice_master = story.data.get("voice_master")
            if voice_master and comfy_inputs.local_path(self.config, voice_master) is None:
                # Trust but verify: a previously staged anchor may reference a
                # file that no longer exists on disk. Fall through to the same
                # re-derivation path used when no anchor was recorded at all.
                voice_master = None
            if not voice_master:
                clip0 = story.segments[0] if story.segments else {}
                clip0_video = str(clip0.get("clip") or "")
                if not clip0_video or not Path(clip0_video).is_file():
                    raise PipelineError(
                        "Voice Masterが見つかりませんでした。"
                        "1本目のクリップの生成が完了しているか確認してください。",
                        stage=STAGES[STAGE_COND], kind="missing_model")
                voice_master = await self._stage_voice_master(story, clip0_video)
            return {
                "prev_total_frames": int(prev_clip.get("frames", 0)),
                "tail_frames": 0,
                "tail_audio": False,
                "last_frame_image": dest_name,
                "voice_master": voice_master,
            }
        src = Path(prev_clip.get("local_video") or "")
        if not src.is_file():
            src = Path(prev_clip.get("video") or "")
        if not src.is_file():
            raise PipelineError("前のクリップの動画ファイルが見つかりませんでした。",
                                stage=STAGES[STAGE_COND], kind="missing_model")
        dest_name = f"h3app_story_{story.id}_s{int(prev_clip.get('step', 0))}.mp4"
        dest = comfy_inputs.stage_path(self.config, dest_name)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        return {
            "video": dest_name,
            # The PREVIOUS SEGMENT's own frame count, not the cumulative total:
            # nothing is concatenated in story mode, so the file LoadVideo reads
            # is exactly one clip long.
            "prev_total_frames": int(prev_clip.get("frames", 0)),
            "tail_frames": int(self.config.defaults["tail_frames"]),
            "tail_audio": bool(tail_audio),
        }

    # ---------------------------------------------------------------- merge ---
    async def merge(self, story: StoryProject) -> Path:
        if self.is_running(story.id):
            raise PipelineError("生成中は書き出しできません。停止してから実行してください。",
                                kind="busy")
        # NOT story.clips: the merge input is derived from the SEGMENTS, so a
        # clip that a structure change invalidated (or one left behind in
        # project.json by a crash) can never reach the final video.
        clips = story.mergeable_clips()
        paths = [Path(c.get("local_video") or c.get("video", "")) for c in clips]
        out = story.final_dir / f"{_safe_name(story.data.get('name') or story.id)}.mp4"
        # WP-C: per-boundary seam plan, one entry per adjacent clip pair.
        boundaries = _boundaries_for(story, clips)
        seam_result = await merge_mod.merge_story(
            paths, out, boundaries=boundaries, debug_dir=story.root,
            story_id=story.id, fps=float(FPS),
            app_debug_dir=self.config.debug_dir)
        result = Path(seam_result["path"])
        story.data["final_video"] = str(result)
        story.data["merged"] = True
        story.data["seams"] = {
            "report": seam_result.get("report", []),
            "needs_review": bool(seam_result.get("needs_review", False)),
        }
        # Generated-video export: merged story video into 完成動画. Never
        # raises; failure is recorded but does not undo the merge itself.
        from . import media as media_mod
        stem = f"{_safe_name(story.data.get('name') or story.id)}_{story.id}"
        export = media_mod.export_video(
            result, dest_dir=self.pipeline.config.media_videos_final,
            filename=stem, media_root=self.pipeline.config.media_root,
            kind="final", project_id=story.id)
        story.data["final_export"] = export
        story.save()
        return result


def _clip_cache_key(story, index: int, en_prompt: str, seed: int,
                    models: dict) -> str:
    """Cache identity for one finished clip (mirrors backend/h3_v2/story_v2).

    Same key + file on disk means the clip is reused, never regenerated.
    """
    import hashlib as _hashlib
    spec = {
        "model": (models or {}).get("unet", ""),
        "model_hash": "manifest-frozen",
        "lora": (models or {}).get("lora"),
        "lora_hash": "",
        "preset": str((story.data or {}).get("mode") or "LONG"),
        "prompt": en_prompt,
        "ref_hashes": [story.data.get("profile_cache_key", "")],
        "seed": int(seed),
        "width": int((story.data.get("settings") or {}).get("width", 0)),
        "height": int((story.data.get("settings") or {}).get("height", 0)),
        "frames": int((story.data.get("settings") or {}).get("frames", 0)),
        "control_input": "",
        "sampler": str((story.data.get("settings") or {}).get("sampler", "")),
        "scheduler": str((story.data.get("settings") or {}).get("scheduler", "")),
        "steps": int((story.data.get("settings") or {}).get("steps", 0)),
        "shift_video": None,
        "shift_audio": None,
        "ref_image_size": str((story.data.get("settings") or {}).get("ref_image_size", "")),
        "tail_frames": 56,
    }
    blob = json.dumps(spec, ensure_ascii=False, sort_keys=True,
                      default=str).encode("utf-8")
    return _hashlib.sha256(blob).hexdigest()


def _safe_name(name: str) -> str:
    cleaned = "".join(ch for ch in str(name) if ch not in '\\/:*?"<>|\r\n\t').strip()
    cleaned = cleaned.rstrip(". ")
    return cleaned[:60] or f"story_{int(time.time())}"


def _boundaries_for(story: StoryProject, clips: list[dict]) -> list[dict]:
    """WP-C: one seam plan entry per adjacent pair in `clips` (the SAME
    filtered/ordered list merge() is about to concatenate), read from the
    SEGMENT the later clip belongs to - never from clip position, so a
    segment invalidated out of the merge never shifts another's transition."""
    boundaries = []
    for clip in clips[1:]:
        try:
            idx = int(clip.get("segment_index", -1))
        except (TypeError, ValueError):
            idx = -1
        seg = story.segments[idx] if 0 <= idx < len(story.segments) else {}
        transition = seg.get("transition") or {"mode": "natural", "keep_pause": False}
        boundaries.append({
            "mode": transition.get("mode", "natural"),
            "keep_pause": bool(transition.get("keep_pause")),
            "has_speech": bool((seg.get("speech") or "").strip()),
        })
    return boundaries
