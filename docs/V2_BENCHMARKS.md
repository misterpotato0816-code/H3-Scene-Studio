# H3 Shorts Studio v2 — Benchmarks

Store: `benchmarks/v2_bench.jsonl` (BenchmarkRecord rows). No human quality verdicts —
humans compare the saved MP4s side by side.

## Official baselines (frozen)

- v1 historical: **217.61 s** (H3_HETERO_V1_MANIFEST.json, 8-step res_multistep,
  RAW turbo = 518-key no-op).
- v1 latest A/B re-measurement: **230.3 s** (same chain, current machine state).
- **FAST verified: 135.6 s** (Ref2VA Turbo 4-step euler/simple + SigmaShift 12/3,
  624/624 keys bound, 0 runtime warnings).

## Performance regression rule

If a future FAST run on the same profile (576x1024 / 124f / seed 123456789 /
gold prompt+ref) exceeds **176 s** (135.6 x 1.3), treat it as a performance
regression: investigate before flipping any preset to verified.

## Attempt 1 (2026-09-06, BLOCKED by environment)

- Row 1: harness misconfiguration (wrong paths yaml → H3CudaPhaseRestore missing).
  Fixed: harness now uses `app/comfy_paths.yaml` + whitelist
  (`H3-Device-Barrier`, `comfyui_layerstyle`).
- Row 2: graph accepted, executed to sampler, then ~10x slow motion:
  **Dead by Daylight was running on GPU0** (124 s/step vs ~20 s/step in v1).
  Harness also mis-declared stall (progress heartbeats ignored). Fixed heartbeat.
- Bind telemetry works: server log shows `lora key not loaded: blocks.*`
  for the RAW turbo file = runtime TURBO_UNAVAILABLE signature, matching the
  header proof (0/518 bindable).

## Baseline reference (v1 manifest, for A/B comparison)

- LEGACY_V1: 217.61 s total (8-step res_multistep/simple/match, turbo no-op).
- Target: FAST (Ref2VA Turbo 4-step euler/simple + SigmaShift 12/3) clearly faster
  with no broken quality. Independent community data point: 20-step base 6.9 min
  → 4-step turbo 2.6 min (2.68x) at 0.31 MP.

## Attempt 2 (2026-09-06, SUCCESS — both arms)

Same prompt (gold_final_prompt.txt) / seed 123456789 / ref h3test_face_ref.png /
576x1024 / 124f / match. Isolated ComfyUI :8402, no GPU contention.

| | LEGACY_V1 | FAST |
|---|---|---|
| total | 230.3 s | **135.6 s (1.70x)** |
| conditioning | 15.9 s | 14.9 s |
| sampling | 177.5 s | **81.9 s** |
| VAE decode | 30.0 s | 29.5 s |
| audio decode | 1.9 s | 1.5 s |
| GPU0 peak | 10578 MB | 10368 MB |
| GPU1 peak | 1731 MB | 1731 MB |
| LoRA bind | 0/518 (expected no-op, 518 warnings = parity) | **624/624, 0 warnings** |
| output | V2_AB_legacy_00002_.mp4 (1.47 MB) | V2_AB_fast_00001_.mp4 (1.38 MB) |
| streams | 5.17 s, 576x1024@24, h264 + AAC stereo | same, audio NOT broken |

Compare sheet: `benchmarks/compare_FAST_vs_LEGACY.jpg` (t=1/2.5/4.2 s, left=legacy).
FAST speed+bind proof complete. Final quality sign-off is human (see HUMAN_GATE
pattern: eye-check the sheet, then FAST may flip to verified).

## Attempt 3 — Phase 3 + 4 arms (2026-09-06, SUCCESS)

| | BALANCED (base 10-step) | QUALITY (base 20-step) | LOW_VRAM (base 8-step + LoVRAM4 + ChunkFF2) |
|---|---|---|---|
| total | 260.7 s | 403.5 s | 222.4 s |
| sampling | 209.1 s | 351.5 s | 169.5 s |
| GPU0 peak | 10637 MB | 10779 MB | 11377 MB |
| GPU1 peak | 1731 MB | 1731 MB | 1731 MB |
| output | V2_AB_balanced_00001_.mp4 | V2_AB_quality_00001_.mp4 | V2_AB_lowvram_00001_.mp4 |

Sheets: `compare_PHASE3_mid.jpg` (balanced/quality/fast @2.5s, all coherent),
`compare_LOWVRAM_vs_LEGACY.jpg` (identical quality confirmed).
LOW_VRAM honest note: no VRAM reduction at 576x1024/124f (run variance dominates);
functional parity verified, never auto-forced. Benefit case = larger frames.

## Attempt 4 — Phase 8 LONG chain (2026-09-06, SUCCESS)

LONG preset (base 8-step), tail_frames=56, concat_previous=True on clip B.

| | A: 10s x 3 (243f) | B: 15s x 2 (362f) |
|---|---|---|
| clip times | ~13 + ~13 + 12.3 min | ~25 + 24.1 min |
| total | ~40 min | ~50 min |
| output | V2_AB_long30a_full.mp4 (**30.41 s**) | V2_AB_long30b_full.mp4 (30.20 s) |
| identity | stable all joints | face OK |
| background | same stage all clips | **drift in clip 2** (different stage + garbled baked text) |

Sheets: `compare_30A_joints.jpg`, `compare_30B_joint.jpg`.
**Decision: 30 SECOND default = 10s x 3.** Fewer relays than 5s x 6, no
background drift seen in B, faster total. Story `seconds_per_segment` default
is now 10.

LONG preset (base 8-step), tail_frames=56, concat_previous=True on clip B.

| | LONG clip A | LONG clip B (relay + concat) |
|---|---|---|
| total | 226.7 s | 417.3 s |
| sampling | 174.5 s | 280.6 s |
| GPU0 / GPU1 peak | 10619 / 1731 MB | 11613 / 3075 MB |
| output | V2_AB_long_a_00001_.mp4 (5.17 s) | V2_AB_long_b_00001_.mp4 (**10.33 s** concat) |

Drift evidence: `compare_LONG_joint.jpg` (t=4.5 vs 5.8 s across the joint —
identity, outfit, stage continuous, no cut glitch).

## Attempt 5 — 30 SECOND shootout (2026-09-06/07)

Same prompt/seed/ref, Original Re-anchor every clip, tail_frames=56.
