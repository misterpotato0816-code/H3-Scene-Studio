# H3 Shorts Studio v2 — Compatibility (Phase 1)

Probed 2026-09-06 against production ComfyUI **0.34.5** (frontend 1.49.6,
torch 2.10+cu130, 2134 nodes, Desktop :8188 single-GPU view).

| Capability | Result |
|---|---|
| MiniMaxH3ReferenceToVideo | ✅ (ref_images≤9, ref_videos≤3, ref_video_audios≤3, ref_audios≤3) |
| MiniMaxH3SigmaShift (12.0 / 3.0) | ✅ |
| MiniMax Low VRAM Attention / Chunk FeedForward | ✅ |
| H3-specific Turbo loader | ❌ (Turbo = LoraLoaderModelOnly + preset; bind must be proven) |
| H3 FunControl | ❌ (only Wan nodes — not H3-compatible) → CONTROL disabled |
| PDD | ❌ → PDD disabled, stable untouched |
| SelectCLIPDevice | ✅ but options are view-dependent: `gpu:1` only with both GPUs visible |

## Known issues carried from v1 (docs/08_known_issues.md)

- torch.compile unusable with H3 (dynamic-VRAM DLPack conflict + SageAttention Triton fault).
- llama.cpp `vram_limit` must stay `-1`; device isolation via launch flags only.
- Dual-GPU hetero routing requires the isolated runner env (ports 8401/8402,
  `--default-device 0`, no `--cuda-device`); Desktop single-GPU view cannot do it.
