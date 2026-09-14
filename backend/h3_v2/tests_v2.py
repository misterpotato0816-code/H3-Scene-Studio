# -*- coding: utf-8 -*-
"""Phase 1 tests. Stdlib only: <comfy venv python> backend/h3_v2/tests_v2.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
APP_ROOT = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(ROOT.parent))

from h3_v2.backend import build, preset_models, preset_settings, select_backend
from h3_v2.benchmark import BenchmarkRecord, ab_table, append_jsonl, load_jsonl
from h3_v2.compat import Capabilities, probe_capabilities
from h3_v2.feature_flags import FeatureFlags, experimental_enabled
from h3_v2.lora_bind import classify_lora, count_bindable_keys
from h3_v2.presets import PRESETS, PresetError, get_preset, verified_presets
from h3_v2.story_v2 import (STATUS_APPROVED, STATUS_DONE, STATUS_LOCKED,
                             apply_snapshot, cache_key, plan, resume_snapshot,
                             segment_seed)
from h3app import graphs as v1_graphs

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK    " if cond else "  FAIL  ") + name + (f" -- {detail}" if detail and not cond else ""))


def _expect_code(pid, flags):
    try:
        get_preset(pid, flags)
        return "RUNNABLE"
    except PresetError as e:
        return str(e).split(":")[0]


def main() -> int:
    print("== flags ==")
    f = FeatureFlags()
    check("experimental off by default", (f.pdd, f.long_video, f.endless) == (False, False, False))
    check("experimental_enabled() empty", experimental_enabled(f) == [])
    check("from_dict ignores unknown keys", FeatureFlags.from_dict({"pdd": True, "nope": 1}).pdd is True)

    print("== presets ==")
    check("5 stable + LONG verified",
          verified_presets() == ["LEGACY_V1", "FAST", "BALANCED", "QUALITY",
                                 "LOW_VRAM", "LONG"])
    check("FAST runnable (verified)",
          _expect_code("FAST", f) == "RUNNABLE")
    check("FAST graph carries turbo chain",
          (lambda g: g["lora"]["inputs"]["lora_name"].endswith(".safetensors")
           and g["scheduler"]["inputs"]["steps"] == 4
           and g["sampler_select"]["inputs"] == {"sampler_name": "euler"}
           and g["sigshift"]["inputs"]["shift_video"] == 12.0)(
              build("FAST", ["h3test_face_ref.png"], "prompt", [1536],
                    "H3/T", flags=f)))
    leg = get_preset("LEGACY_V1", f)
    check("LEGACY_V1 values = manifest",
          (leg.steps, leg.sampler, leg.scheduler, leg.ref_image_size, leg.lora_strength)
          == (8, "res_multistep", "simple", "match", 1.0))
    for pid, code in (("CONTROL", "CONTROL_UNAVAILABLE"),
                      ("PDD", "PDD_UNAVAILABLE"), ("NOPE", "PRESET_UNKNOWN")):
        try:
            get_preset(pid, f)
            check(f"{pid} blocked", False, "no error raised")
        except PresetError as e:
            check(f"{pid} blocked with {code}", code in str(e), str(e))
    try:
        get_preset("LONG", f)
        check("LONG gated by flag", False)
    except PresetError as e:
        check("LONG gated by flag", "EXPERIMENTAL_DISABLED" in str(e), str(e))
    check("LONG runnable with flag (verified experimental)",
          _expect_code("LONG", FeatureFlags(long_video=True)) == "RUNNABLE")

    print("== backend selection ==")
    check("default backend legacy_v1", select_backend(None) == "legacy_v1")
    check("h3_v2 selectable", select_backend({"backend": "h3_v2"}) == "h3_v2")
    try:
        select_backend({"backend": "wat"})
        check("unknown backend rejected", False)
    except Exception as e:
        check("unknown backend rejected", "BACKEND_UNKNOWN" in str(e))

    print("== graph parity ==")
    images, prompt, longest = ["h3test_face_ref.png"], "a test prompt", [1536, 1152, 1152, 896]
    v1g = v1_graphs.build_generate_graph(
        images, prompt, preset_settings(leg), preset_models(leg), longest, "H3/V2T")
    v2g = build("LEGACY_V1", images, prompt, longest, "H3/V2T")
    check("LEGACY_V1 graph byte-identical to v1 builder",
          json.dumps(v1g, sort_keys=True) == json.dumps(v2g, sort_keys=True))
    check("v1 chain keeps hetero routing",
          v2g["clip_select"]["inputs"].get("device") == "gpu:1")

    print("== lora bind ==")
    raw = [f"blocks.{i}.attn.qkv_proj.lora_A.weight" for i in range(518)]
    check("raw turbo keys bind 0", count_bindable_keys(raw) == 0)
    check("raw turbo -> TURBO_UNAVAILABLE",
          classify_lora(raw, 400)["code"] == "TURBO_UNAVAILABLE")
    pruned = [f"diffusion_model.blocks.{i}.attn.qkv_proj.lora_A.weight" for i in range(416)]
    check("pruned keys bind 416", count_bindable_keys(pruned) == 416)
    check("pruned passes min 400", classify_lora(pruned, 400)["code"] == "TURBO_OK")

    print("== compat probe ==")
    fake = {"MiniMaxH3ReferenceToVideo": {"input": {"optional": {
                "ref_images": {"max": 9}, "ref_videos": {"max": 3},
                "ref_video_audios": {"max": 3}}}},
            "MiniMaxLowVRAMAttention": {}, "MiniMaxChunkFeedForward": {},
            "SelectCLIPDevice": {"input": {"required": {"device": [[
                "default", "cpu"]]}}}}
    caps = probe_capabilities(fake)
    check("probe finds nodes", caps.ref2va and caps.lowvram_attention and caps.chunk_ffn)
    check("probe finds no funcontrol/pdd", (not caps.funcontrol_h3) and (not caps.pdd))
    check("CONTROL disabled in UI", caps.disabled_ui_modes() == ["CONTROL", "PDD"])
    check("limits parsed", caps.h3_limits["ref_images"] == 9)
    check("empty info degrades", probe_capabilities({}).ref2va is False)

    print("== benchmark store ==")
    with tempfile.TemporaryDirectory() as td:
        store = str(Path(td) / "bench.jsonl")
        rec = BenchmarkRecord(preset="LEGACY_V1", steps=8, t_total=217.6,
                              gpu0_peak_vram_mb=11200, success=True,
                              output_path="out/a.mp4")
        append_jsonl(store, rec)
        rows = load_jsonl(store)
        check("jsonl roundtrip", len(rows) == 1 and rows[0].t_total == 217.6)
        table = ab_table(rows)
        check("ab row shape", table[0]["preset"] == "LEGACY_V1" and table[0]["output"] == "out/a.mp4")

    print("== long_fast graphs ==")
    from h3app.graphs import GraphError
    lf = PRESETS["LONG_FAST"]
    check("LONG_FAST starts unverified", lf.status == "unverified")
    try:
        get_preset("LONG_FAST", f)
        check("LONG_FAST gated without flag", False)
    except PresetError as e:
        check("LONG_FAST gated without flag",
              "EXPERIMENTAL_DISABLED" in str(e), str(e)[:80])
    check("LONG_FAST 480x864", (lf.width, lf.height) == (480, 864))
    check("LONG_FAST sage on", lf.sage is True)
    # preset_obj overrides must LAND in the graph (regression: silently
    # ignored overrides once invalidated three probe runs).
    import dataclasses as _dc
    g_ov = build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
                 "H3/T", flags=FeatureFlags(long_fast=True),
                 allow_unverified=True,
                 preset_obj=_dc.replace(lf, sage=False, steps=3,
                                        clip="qwen3vl_4b_fp8_scaled.safetensors"))
    check("override sage off lands", "sage" not in g_ov)
    check("override steps land", g_ov["scheduler"]["inputs"]["steps"] == 3)
    check("override clip lands",
          g_ov["clip"]["inputs"]["clip_name"] == "qwen3vl_4b_fp8_scaled.safetensors")
    try:
        build("LONG_FAST", ["x.png"], "p", [1536], "H3/T",
              preset_obj=_dc.replace(lf, steps=3))
        check("override without harness refused", False)
    except PresetError as e:
        check("override without harness refused",
              "OVERRIDE_FORBIDDEN" in str(e))
    # Fresh-clip external Voice Master: <Audio 1> with no continuation.
    g_vm = build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
                 "H3/T", flags=FeatureFlags(long_fast=True),
                 allow_unverified=True, voice_master="V2_VM_EXT.wav")
    check("fresh clip carries external VM",
          g_vm["h3"]["inputs"].get("ref_audios.ref_audio_0") == ["voice_master", 0]
          and g_vm["voice_master"]["inputs"] == {"audio": "V2_VM_EXT.wav"}
          and "ref_video_audios.ref_video_audio_0" not in g_vm["h3"]["inputs"])
    g_lf = build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
                 "H3/T", flags=FeatureFlags(long_fast=True),
                 allow_unverified=True)
    check("sage wired after sigshift",
          g_lf["sage"]["inputs"] == {"model": ["sigshift", 0]}
          and g_lf["guider"]["inputs"]["model"] == ["sage", 0])
    check("turbo chain intact",
          g_lf["scheduler"]["inputs"]["steps"] == 4
          and g_lf["sampler_select"]["inputs"] == {"sampler_name": "euler"})
    cont5 = {"video": "prev.mp4", "prev_total_frames": 124, "tail_frames": 5,
             "tail_audio": False, "voice_master": "vm.wav"}
    g5 = build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
               "H3/T", cont5, concat_previous=False,
               flags=FeatureFlags(long_fast=True), allow_unverified=True)
    h5 = g5["h3"]["inputs"]
    check("5f: tail video connected, tail audio NOT",
          h5.get("ref_videos.ref_video_0") == ["tail_images", 0]
          and "ref_video_audios.ref_video_audio_0" not in h5
          and h5.get("ref_audios.ref_audio_0") == ["voice_master", 0])
    cont0 = {"video": "prev.mp4", "prev_total_frames": 124, "tail_frames": 0,
             "tail_audio": False, "last_frame_image": "lf.png"}
    g0 = build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
               "H3/T", cont0, concat_previous=False,
               flags=FeatureFlags(long_fast=True), allow_unverified=True)
    h0 = g0["h3"]["inputs"]
    check("0f: no tail nodes, last frame as ref image",
          "tail_images" not in g0 and "prev_video" not in g0
          and h0.get("ref_images.ref_image_1") == ["last_frame", 0])
    bad = {"video": "prev.mp4", "prev_total_frames": 124, "tail_frames": 5,
           "tail_audio": True, "voice_master": "vm.wav"}
    try:
        build("LONG_FAST", ["h3test_face_ref.png"], "prompt", [1536],
              "H3/T", bad, concat_previous=False,
              flags=FeatureFlags(long_fast=True), allow_unverified=True)
        check("voice-master conflict raises GraphError", False)
    except GraphError as e:
        check("voice-master conflict raises GraphError",
              "VOICE_MASTER_CONFLICT" in str(e), str(e)[:80])

    print("== story v2 ==")
    spec = {"model": "m", "preset": "FAST", "prompt": "hi", "seed": 7,
            "width": 576, "height": 1024, "frames": 124}
    check("cache key stable", cache_key(spec) == cache_key(dict(spec)))
    check("cache key sensitive",
          cache_key(spec) != cache_key({**spec, "seed": 8}))
    check("cache key ignores extras",
          cache_key(spec) == cache_key({**spec, "ui_note": "x"}))
    check("segment seed deterministic",
          segment_seed(100, 3) == segment_seed(100, 3) == 103)
    segs = [{}, {}, {}, {}]
    st = {"clips": {
        "0": {"status": STATUS_LOCKED, "path": "c0.mp4",
              "spec": spec, "want": spec},
        "1": {"status": STATUS_LOCKED, "path": "c1.mp4",
              "spec": spec, "want": spec},
        "2": {"status": STATUS_DONE, "path": "c2.mp4",
              "spec": spec, "want": spec},
        "3": {"status": STATUS_DONE, "path": "c3.mp4",
              "spec": spec, "want": spec}}}
    actions = [r["action"] for r in plan(segs, st, regenerate={2})]
    check("lock/regen plan", actions == ["locked", "locked", "generate", "generate"],
          str(actions))
    st2 = {"clips": {
        "0": {"status": STATUS_DONE, "path": "c0.mp4", "spec": spec, "want": spec},
        "1": {"status": STATUS_APPROVED, "path": "c1.mp4", "spec": spec, "want": spec}}}
    p2 = plan([{}, {}], st2, regenerate={0})
    check("approved clip survives upstream regen",
          p2[1]["action"] == "locked" and p2[1]["relay_stale"] is True)
    snap = resume_snapshot(2, st["clips"], 123, ["rh1"], {"t": 1})
    cur, clips, seed, rhs, tr = apply_snapshot(snap)
    check("resume roundtrip",
          (cur, seed, rhs, tr) == (2, 123, ["rh1"], {"t": 1})
          and len(clips) == 4)

    print(f"\nV2 SELFTEST: {len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
