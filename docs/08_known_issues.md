# 08. 既知の問題（環境固有・上流由来）

本ワークフローの設計上の問題ではなく、**依存ソフトウェア側の問題**として分離記録します。
将来のバージョンで解消されうるため、再検証の起点として残します。

---

## ISSUE-1 — Dynamic VRAM (DLPack) と torch.compile / Dynamo の競合

| | |
|---|---|
| 分類 | 機能非互換 |
| 影響 | torch.compile が使用不可（生成は失敗、プロセスは生存） |
| 発生条件 | ComfyUI の dynamic VRAM loading が有効 かつ transformer blocks を torch.compile |
| 確認環境 | ComfyUI v0.30.2 / torch 2.10.0+cu130 / KJNodes v1.5.0 / RTX 4070 Ti 12GB |

### 症状

```
torch._dynamo.exc.TorchRuntimeError:
Dynamo failed to run FX node with fake tensors:
  call_method __dlpack__(*(FakeTensor(..., device='cuda:0',
                            size=(26312, 28672), dtype=torch.bfloat16),), **{'stream': -1})
RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor).
  If you're using torch.compile/export/fx, it is likely that we are erroneously
  tracing into a custom kernel. To fix this, please wrap the custom kernel into
  an opaque custom op.
```

### 原因

ComfyUI の動的 VRAM ローディングは `__dlpack__` 経由で重みを転送します。
Dynamo のトレース中はテンソルが `FakeTensor`（データポインタなし）に置き換わるため、
DLPack 変換が失敗します。

### 回避策の状況

| 回避策 | 結果 |
|---|---|
| KJNodes ノードの `disable_dynamic_vram=True` | **効果なし。**同じエラー。ログは引き続き `prepared for dynamic VRAM loading` |
| ComfyUI CLI `--disable-dynamic-vram` | **検証中（Phase C'）** — ノード側とは別経路 |
| `torch.compiler.allow_in_graph` | **使用しない**（ご指示）。安易な抑制は不整合を招く |
| `torch.library.custom_op` で opaque 化 | 上流（ComfyUI / KJNodes / SageAttention）へのパッチが必要。**現段階では実装しない** |

エラーメッセージ自体が opaque custom op 化を推奨しており、上流対応で解消しうる問題です。

---

## ISSUE-2 — SageAttention の vendored Triton カーネルを Inductor が compile して Windows ハードフォールト

| | |
|---|---|
| 分類 | **クラッシュ（プロセス即死）** |
| 影響 | ComfyUI プロセスが強制終了。実行中のキューは失われる |
| 発生条件 | KJNodes の H3 SageAttention パッチ適用下で transformer blocks を torch.compile |
| 確認環境 | 同上 |

### 症状

まず Triton のコンパイルエラー：

```
triton.compiler.errors.CompilationError: at 1:0:
def _quant_key_per_thread_int8_i64_kernel(Input, Output, Scale, L,
^
IndexError('Function argument index out of range')
```

続いて生成された inductor コード内でハードフォールト：

```
Windows fatal exception: code 0x80000003
  File "...\torchinductor_<user>\dx\cdxjmjirbj...py", line 495 in call
  File "...torch\_inductor\output_code.py", line 638 in __call__
```

### 原因

`_quant_key_per_thread_int8_i64_kernel` は KJNodes が
**SageAttention から vendored した Triton カーネル**です
（`ltxv_nodes.py` のコメント: "Vendored from sageattention's quant_per_thread.py
with the row offsets promoted to int64 to avoid overflow on large sequences"）。

Inductor がこのサードパーティ Triton カーネルを再コンパイルしようとして破綻します。
Python 例外ではなく**ネイティブの abort/breakpoint** なので try/except では捕捉できません。

### 重要性

Phase A で確定したとおり、H3 の self-attention は
`MiniMaxH3MemoryEfficientSageAttentionPatch` によってこのカスタムカーネルに置換されています。
**つまり本ワークフローの標準構成では、torch.compile を有効にすると
ISSUE-2 を踏む可能性が常にあります。**

→ 現構成では torch.compile を有効化しないでください（速度ではなく**安定性**の理由）。

### 未調査（将来候補）

以下は未確認のため、記録のみとします。

- KJNodes / SageAttention 側に `torch.library.custom_op` 対応が既にあるか
- 上流 Issue/PR に torch.compile 対応があるか
- SageAttention 本家に対応実装があるか

---

## ISSUE-3 — llama.cpp の `vram_limit` 部分指定でプロセス即死

| | |
|---|---|
| 分類 | **クラッシュ（プロセス即死）** |
| 影響 | ComfyUI プロセスが強制終了 |
| 発生条件 | `llama_cpp_model_loader.vram_limit` を中途半端な値（例 6）にし、可視 GPU が1枚 |
| 回避 | **`vram_limit = -1`（自動）にする。**デバイス隔離は `--cuda-device 0` で行う |

```
ggml-backend.cpp:1367: GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
Fatal Python error: Aborted
  → llama_cpp/llama.py:622 __init__   (モデルロード中)
```

CPU/GPU のレイヤー分割が細かくなりすぎ、llama.cpp の分割入力数上限を超えて `abort()` します。

---

## ISSUE-4 — LiteGraph のグループ矩形重なりでノードが巻き添えバイパス

| | |
|---|---|
| 分類 | 設計上の落とし穴（自作ワークフロー側で対処済み） |
| 影響 | 「グループをバイパス」した際に隣接グループのノードも無効化される |
| 対処 | カラム間隔を広げ、**ビルド時アサーション**で重なりを検出（現在 overlapping = 0） |

LiteGraph はグループ所属を矩形包含だけで判定するため、
**ComfyUI 上で右クリック →「Bypass Group Nodes」でも同じ事故が起きます。**
自作ワークフローを作る際は、グループ矩形を重ねないよう注意してください。
