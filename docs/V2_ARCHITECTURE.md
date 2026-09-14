# H3 Shorts Studio v2 — Architecture (Phase 1)

v1 (`app/h3app/*`, `_hetero_test/*`, manifest, GOLD) is read-only.
v2 lives beside it and reuses the pure v1 graph builder — no duplicated graph code.

```text
H3/
├─ RUN_H3_HETERO_V1.ps1 / RUN_OPT_REF_MATCH.ps1 / RUN_FINAL_GPU1_E2E.ps1  (v1 FROZEN)
├─ H3_HETERO_V1_MANIFEST.json                                            (v1 FROZEN)
├─ app/h3app/            legacy_v1 backend (untouched)
├─ backend/h3_v2/        v2 backend (NEW)
│  ├─ backend.py         select_backend() + build() over v1 builder
│  ├─ presets.py         preset engine (only LEGACY_V1 verified)
│  ├─ feature_flags.py   stable/experimental flags, experimental off by default
│  ├─ compat.py          /object_info probe -> Capabilities
│  ├─ lora_bind.py       safetensors-header bind proof (TURBO_OK / TURBO_UNAVAILABLE)
│  ├─ benchmark.py       BenchmarkRecord + JSONL store + A/B table
│  └─ tests_v2.py        stdlib selftest (27 checks, Phase 1)
├─ workflows/v2/         (Phase 2+: exported verified graphs)
└─ docs/V2_*.md/json     audit, baseline, presets, compatibility, benchmarks
```

## Backend selection

- Overlay `backend/h3_v2/config.json` (absent → `legacy_v1`). Default never changes v1 behaviour.
- `build()` for LEGACY_V1 is byte-identical to `app.h3app.graphs.build_generate_graph`
  (parity test in `tests_v2.py`). Base presets drop the `lora` node and rewire
  `guider`/`scheduler` to `unet`. Low-VRAM/sigma options append a model-patch chain
  (`lowvram → chunkff → sigshift`) — never a second graph implementation.
- Unverified/unavailable presets raise explicit codes
  (`PRESET_UNVERIFIED`, `*_UNAVAILABLE`, `EXPERIMENTAL_DISABLED`, `TURBO_UNAVAILABLE`).
  Silent fallback to Base is forbidden.

## Compatibility layer

- `probe_capabilities(object_info)` is pure and tested with synthetic input.
- Live rule: `SelectCLIPDevice` options depend on visible GPUs
  (`gpu:1` exists only when both GPUs are visible, i.e. isolated runner 8401/8402;
  Desktop :8188 shows `default/cpu` only). The launcher must pick routing from live
  options, never from memory.
