# H3 Shorts Studio v2 — Known Issues (Phase 5/7 status included)

## V2-1 — H3 FunControl node does not exist (Phase 5 outcome)

- Status: `CONTROL` preset = unavailable, `feature_flags.funcontrol = false` default.
- Environment: ComfyUI 0.34.5 + KJNodes, 2134 nodes probed — only **Wan**
  FunControl nodes exist, which are not H3-compatible.
- Requirements to enable: an H3-targeted control node (pose/depth/canny/…).
- Enable method (when it appears): flip flag → add control-input wiring behind
  the flag → generation test (identity from Reference + structure from Control).
- Stable tracks are isolated: missing FunControl disables only CONTROL.

## V2-2 — PDD unavailable (Phase 7 outcome)

- Status: `PDD` preset = unavailable, `experimental.pdd = false` default.
- No PDD LoRA/scheduler/head node in this ComfyUI generation. No patching
  attempted — API-mismatch patching could break stable H3 generation.
- Enable method: re-probe `/object_info`; enable only with compat proof.

## V2-3 — LOW_VRAM shows no reduction at the v1 profile (Phase 4 finding)

- Functional parity verified (identical frames), but peak VRAM 11377 vs 10578 MB
  (run variance dominates at 576x1024/124f). Benefit case = larger frame counts.
- LOW_VRAM is opt-in per preset, never auto-forced.

## Carried from v1 (docs/08_known_issues.md, still valid)

- torch.compile unusable with H3 (DLPack conflict + SageAttention Triton fault).
- llama.cpp `vram_limit` must stay `-1`.
- Hetero `gpu:1` routing needs the isolated runner env (both GPUs visible).
