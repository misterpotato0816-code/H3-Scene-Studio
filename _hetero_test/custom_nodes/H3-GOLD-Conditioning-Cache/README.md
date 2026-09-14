# H3 GOLD Execution Persistent Cache

Isolated ComfyUI custom nodes for storing the exact two outputs of
`MiniMaxH3ReferenceToVideo`—`CONDITIONING` and `LATENT`—in one persistent cache
entry. Cache HIT runs need no VLM, CLIP/text encoder, VAE, reference image, or
replacement latent node.

`H3 GOLD Fixed Prompt` reads the prepared UTF-8 text file, removes its single
storage-only terminal line ending, and refuses to output the string unless its
SHA256 is `62CCB107A37FA1D6EED26E5892D6BDDE2A3EBB0446CB294145997029EAD059B3`.
Internal whitespace and line endings are not normalized.

## Official nodes and format

- `H3GoldExecutionCacheSave`: accepts `CONDITIONING` and `LATENT` together.
- `H3GoldExecutionCacheLoad`: returns `CONDITIONING` and `LATENT` together.
- schema: `h3_gold_execution_cache`
- format version: `2`

The older `H3GoldConditioningCacheSave` and `H3GoldConditioningCacheLoad` nodes
remain only for compatibility. They are legacy, conditioning-only format v1 and
must not be used for GOLD speed or quality comparisons. The v2 key includes its
schema, format version, and exact root list, so it cannot collide with v1. The
Execution loader also explicitly rejects a discovered v1-only cache.

## Files on disk

For cache key `<key>` the saver atomically publishes:

- `<key>.safetensors`: every tensor under both roots, detached, cloned,
  contiguous, original dtype, and CPU resident.
- `<key>.manifest.json`: exact recursive structures and integrity metadata.

The v2 manifest includes:

- `format_version`, `cache_key`, `roots.conditioning`, `roots.latent`;
- prompt and text-encoder SHA256;
- video/audio VAE SHA256, ordered reference slot/hash list, geometry and
  `ref_image_size` when supplied by identity settings;
- `tensor_count`, `total_tensor_bytes`;
- each tensor's root, structure path, dtype, shape, `requires_grad`, and value
  SHA256;
- safetensors full-file SHA256 and `bundle_integrity_sha256`.

The recursive serializer preserves dict/list/tuple/tensor/scalar/string/bool/
None/bytes values and `comfy.nested_tensor.NestedTensor` for both roots. A
NestedTensor is represented by a `comfy_nested_tensor` structure node whose
ordered `.tensors` list is recursively serialized; each inner tensor keeps its
own CPU safetensors payload and dtype/shape/requires-grad/value-SHA metadata.
Load reconstructs the exact ComfyUI type with `NestedTensor(restored_list)`.
No LATENT fields are assumed: `samples`, `noise_mask`, `batch_index`, and any
supported nested metadata are retained as received. Pickle and `torch.save`
are not used.

## Cache identity

The v2 key is SHA256 over canonical UTF-8 JSON containing:

- execution schema, format version 2, and ordered roots;
- exact GOLD prompt SHA256;
- text encoder model name and full-file SHA256;
- parsed/canonicalized tokenize and encoder settings.

Prepared GOLD settings additionally pin:

- ordered reference entries (`slot`, type, filename, content SHA256);
- reference preprocessing and `ref_image_size`;
- width, height, and length;
- video/audio VAE model names and full-file SHA256 values;
- ComfyUI commit;
- `MiniMaxH3ReferenceToVideo` node ID and source-file SHA256.

Identity file:
`E:/AI-Projects/H3/_hetero_test/gold/gold_cache_identity.json`.

## One-time CAPTURE graph

```text
H3GoldFixedPrompt.gold_prompt
  -> MiniMaxH3ReferenceToVideo.prompt
H3GoldFixedPrompt.gold_prompt
  -> H3GoldExecutionCacheSave.gold_prompt

MiniMaxH3ReferenceToVideo.positive (CONDITIONING)
  -> H3GoldExecutionCacheSave.conditioning
  -> BasicGuider.conditioning

MiniMaxH3ReferenceToVideo.LATENT
  -> H3GoldExecutionCacheSave.latent
  -> Sampler.latent_image
```

Both pass-through outputs from the saver feed the same consumers as the
original node. The one-time real capture is outside the static implementation
task and has not been run.

## Official cache-HIT graph

```text
H3GoldFixedPrompt.gold_prompt
  -> H3GoldExecutionCacheLoad.gold_prompt

H3GoldExecutionCacheLoad.conditioning (CONDITIONING)
  -> BasicGuider.conditioning

H3GoldExecutionCacheLoad.latent (LATENT)
  -> Sampler.latent_image
```

`EmptyMiniMaxH3LatentAV` is forbidden and absent from this HIT graph. The exact
LATENT emitted during capture is restored from disk. The loader has no VLM,
CLIP, VAE, reference-image, or latent input.
