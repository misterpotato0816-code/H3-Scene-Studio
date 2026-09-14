# H3 GOLD Execution Cache v2 Static / Mock Test Report

Date: 2026-08-13

Scope: implementation diff plus static and CPU-only mock validation. This
verification did not start the ComfyUI server or GPU and did not retry real
cache capture. It follows the single prior capture whose serializer boundary
rejected `comfy.nested_tensor.NestedTensor`.

Python runtime used for compile/import/mock only:

`E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe`

Observed libraries: torch `2.10.0+cu130`, safetensors `0.8.0`.

## Result

Existing mock assertions: **36 total / 36 PASS / 0 FAIL**.

New real-ComfyUI NestedTensor assertions: **14 total / 14 PASS / 0 FAIL**.

- format v2 execution key canonicalization: PASS
- v2 key separation from legacy v1: PASS
- reference content hash change miss: PASS
- reference slot order change miss: PASS
- video VAE hash change miss: PASS
- audio VAE hash change miss: PASS
- width/height/length/ref_image_size change miss: PASS
- CONDITIONING + LATENT simultaneous round-trip: PASS
- LATENT nested dict/list/tuple/tensor/scalar/bytes metadata: PASS
- `samples` exact equality/dtype/shape/requires_grad/CPU: PASS
- `noise_mask` exact equality/dtype/shape/CPU: PASS
- `batch_index` and nested metadata preservation: PASS
- conditioning-root tensor corruption rejection: PASS
- latent-root tensor corruption rejection: PASS
- manifest modification rejection: PASS
- missing latent root rejection: PASS
- format version 1 rejection: PASS
- discovered legacy conditioning-only cache rejection: PASS
- package import and five node mappings: PASS
- Execution loader has no CLIP/VLM/VAE/reference inputs: PASS
- Execution saver inputs are CONDITIONING + LATENT: PASS
- Execution loader outputs are CONDITIONING + LATENT: PASS
- fixed UTF-8 GOLD prompt SHA guard: PASS
- static CAPTURE/HIT graph equivalence: PASS
- `EmptyMiniMaxH3LatentAV` absent from official HIT graph: PASS
- compileall: PASS

NestedTensor coverage:

- imported the installed `comfy.nested_tensor.NestedTensor`: PASS
- `samples` and `noise_mask` exact-type round-trip: PASS
- video/audio list order, dtype, shape, requires-grad, values, and CPU storage: PASS
- inner tensor path and value SHA256 metadata: PASS
- inner `tensors[0]` and `tensors[1]` value corruption rejection: PASS
- ordered-list manifest modification rejected by bundle integrity: PASS

For the two tensor corruption tests, the modified safetensors file's outer
full-file hash and manifest bundle hash were recalculated deliberately. The
loader therefore reached and rejected the altered tensor through its original
root-specific value SHA256, rather than stopping only at the outer file hash.

## Format and identity

- schema: `h3_gold_execution_cache`
- format version: `2`
- roots: `conditioning`, `latent`
- fixed GOLD prompt SHA256:
  `62CCB107A37FA1D6EED26E5892D6BDDE2A3EBB0446CB294145997029EAD059B3`
- text encoder SHA256:
  `BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923`
- video VAE SHA256:
  `7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`
- audio VAE SHA256:
  `8E505D95DD1561D47ABD43D4238FD40D9BB1AE9E147ED0A4CBA778D76AE4DB48`
- `ref_image_0` content SHA256:
  `885D9BE9D35B6041959A2DA1D544F83F95BFA1BC32BA57413532A687473E4562`
- ComfyUI commit: `dec5d9450a5290bcf63430409ea41018e67f41c3`
- `nodes_minimax_h3.py` SHA256:
  `5C3F78EFEF772850F90660C5FF82E31078B5BAEF10F427053ED82FB8B4A7ECD1`
- current prepared v2 key:
  `A9C9CB503121AABC3DFBF94472FEB32808A3F63984F69581F676148744C6850E`

No real cache directory or persistent entry was created by these tests. Mock
cache files existed only under an automatically deleted OS temporary directory.

The prepared real identity was recalculated after the codec change. Format
version remains `2` and the key remains
`A9C9CB503121AABC3DFBF94472FEB32808A3F63984F69581F676148744C6850E`.

## Static graph result

CAPTURE:

```text
MiniMaxH3ReferenceToVideo.positive -> H3GoldExecutionCacheSave.conditioning -> BasicGuider.conditioning
MiniMaxH3ReferenceToVideo.LATENT   -> H3GoldExecutionCacheSave.latent       -> Sampler.latent_image
```

HIT:

```text
H3GoldExecutionCacheLoad.conditioning -> BasicGuider.conditioning
H3GoldExecutionCacheLoad.latent       -> Sampler.latent_image
```

The current core schema declares the source node outputs as CONDITIONING then
LATENT and returns `NodeOutput(cond, latent)`. Execution Save pass-through and
Execution Load expose the same two types and consumer meanings.
