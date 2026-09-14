# H3 Shorts Studio v2 — Presets (Phase 1)

UI shows modes only: FAST / BALANCED / QUALITY / CONTROL / LOW VRAM / EXPERIMENTAL.
Sampler/scheduler/steps/shifts stay inside the preset engine.

| Preset | Backend | Status | FROZEN / candidate values |
|---|---|---|---|
| LEGACY_V1 | legacy_v1_chain | ✅ verified | 8-step res_multistep/simple/match, RAW turbo LoRA = 518-key no-op. Source: H3_HETERO_V1_MANIFEST.json, 217.61 s |
| FAST | ref2va_turbo | ✅ verified | ref2v_turbo_4step_v0.1_comfyui_bf16, 4-step euler/simple, SigmaShift 12/3. Source: upstream example graph + A/B 2026-09-06 (135.6s vs 230.3s, 1.70x, 624/624 bound) |
| BALANCED | ref2va_base | ✅ verified | no LoRA, 10 steps. A/B 2026-09-06: 260.7s |
| QUALITY | ref2va_base | ✅ verified | no LoRA, 20 steps res_multistep/simple. A/B: 403.5s |
| LOW_VRAM | ref2va_base | ✅ verified | no LoRA, 8 steps + LoVRAM heads=4 + ChunkFF=2. A/B: 222.4s, frames identical, no VRAM win at 576x1024/124f — opt-in only |
| CONTROL | ref2va_base | 🚫 unavailable | no H3 FunControl node in ComfyUI 0.34.5 / KJNodes. Phase 5, flag-gated |
| PDD | ref2va_base | 🚫 unavailable (experimental) | no PDD node/scheduler. Phase 7, `experimental.pdd=false` default |
| LONG / ENDLESS | ref2va_base | ✅ verified / 🚫 unavailable (experimental) | LONG: 2-clip tail-relay chain proven (A 226.7s + B 417.3s = 10.33s concat, joint coherent). Opt-in via `long_video`. ENDLESS: not designed |

Sigma defaults (MiniMaxH3SigmaShift): shift_video=12.0, shift_audio=3.0 (node defaults,
verified 2026-09-06 via /object_info). FAST shifts stay `None` (= node default) until
upstream/measured evidence says otherwise — no guessing.
