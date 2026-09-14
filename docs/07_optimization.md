# 07. 高速化・品質向上の検証

**正式ベースライン = Test 3B**（[06_test_log.md](06_test_log.md)）
576×1024 / 124f / 8 steps / 13.93 sec/step / 総 230.90 秒

ベースラインのワークフローは `workflows/baseline/H3_baseline_test3b.json` に凍結保存済み。
実験はすべて別ファイルで行い、ベースラインは変更しません。

---

## Phase A — Attention 経路の確定 ✅ 完了

### 問い

「ノードがある＝使われている」とは限らない。実際に H3 の self-attention を実行しているのは何か。

### 実行時ログ（Test 3B）

```
[INFO] Using pytorch attention                                          ← ComfyUI 全体の既定
[INFO] Using sage attention mode: auto                                  ← PathchSageAttentionKJ
[INFO] Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
```

3つとも出ています。ログだけでは決着しないのでコードを読みました。

### コードによる確定

`comfyui-kjnodes/nodes/ltxv_nodes.py` L2181-2196:

```python
for idx, block in enumerate(diffusion_model.blocks):
    model_clone.add_object_patch(
        f"diffusion_model.blocks.{idx}.attn.forward",
        minimax_sageattn_forward.__get__(block.attn, block.attn.__class__))
```

**`MiniMaxH3MemoryEfficientSageAttentionPatch` は各ブロックの `attn.forward` を丸ごと差し替えます。**
ノードの description にも "overrides the attention mode" と明記されています。

したがって：

| 経路 | H3 self-attention での実効 |
|---|---|
| ComfyUI `pytorch attention` | **不使用**（forward を置換済み） |
| `PathchSageAttentionKJ` (mode=auto) | **H3 ブロックには効かない**（同上）。他の attention 箇所には影響しうる |
| **`MiniMaxH3MemoryEfficientSageAttentionPatch`** | **これが実行されている** |

失敗時は `RuntimeError` を送出する実装で、Test 3B は成功しているため適用は確実です。

### 実際に呼ばれるカーネル（sm89 / Ada）

`_resolve_qattn("sm89")` を同一ロジックで再現した結果：

```
resolved sm89 obj      : torch.ops.sageattention_qattn_sm89
  f32_fuse             : True
  f16_fuse (より高速)   : True
sageplus_sm89_available: True
=> pv_accum_dtype      : fp32+fp16   (quant_v_scale_max = 2.25)
=> kernel              : qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
```

> 注：素朴に `import sageattention._qattn_sm89` すると失敗します。
> 実体は **torch カスタムオペ名前空間** `torch.ops.sageattention_qattn_sm89` です。

### 結論

**Ada で利用可能な最速の attention 経路を、すでに使っています。**

- QK は int8、PV は fp8、累算 fp16（SageAttention+ 相当）
- Blackwell 専用（sm120/121）分岐は対象外
- sm80/sm86 分岐（3060 側）は使われていない

→ **Phase A の「4経路を比較」は実質的に意味がありません。**
現構成が最速で、切り替える先がありません。

### 唯一の A/B 候補

`PathchSageAttentionKJ` は H3 ブロックには効いていないため、**外しても速度・品質とも不変**の可能性が高いです。
ノードを1つ減らせるだけなので優先度は低いですが、確認は容易です。

### 副産物：VRAM 調整ノブを発見

`minimax_sageattn_forward` は `transformer_options["minimax_head_chunks"]` を読みます（既定 1）。

```python
n = min(transformer_options.get("minimax_head_chunks", 1), self.heads)
```

2 以上にすると head をグループ分割して attention を実行し、**int8/fp8 の作業セットがグループ数分の1になります**。
GPU0 peak が 11,333 / 12,282 MiB とほぼ上限なので、
**解像度や尺を上げて OOM したときの逃げ道**として有効です（速度は落ちる可能性あり）。

---

## Phase B — ComfyUI 公式高速化オプション ⚠️ 大半が本構成では無効

### この ComfyUI に存在するフラグ

`comfy/cli_args.py` L190-194:

```python
class PerformanceFeature(enum.Enum):
    Fp16Accumulation        = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps               = "cublas_ops"
    AutoTune                = "autotune"
```

### 適用条件をコードで確認

`comfy/ops.py` の分岐順序が決定的でした。

```python
    logging.info("Native ops: ...")
    return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled)   # ← ここで return

if (fp8_compute and (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) ...):
    return fp8_ops
if (PerformanceFeature.CublasOps in args.fast and CUBLAS_IS_AVAILABLE
        and weight_dtype == torch.float16 and ...):
    return cublas_ops
```

**量子化メタデータを持つモデルは `mixed_precision_ops` で早期 return し、
fp8_ops / cublas_ops の分岐に到達しません。**

Test 3B のログがそれを裏付けています。

```
[INFO] Detected mixed precision quantization
[INFO] Found quantization metadata version 1
[INFO] Using mixed precision operations
[INFO] Using MixedPrecisionOps for text encoder
```

### 判定

| フラグ | 本構成での効果 | 根拠 |
|---|---|---|
| `fp8_matrix_mult` | **無効** | quant_config 有りで早期 return |
| `cublas_ops` | **無効** | 同上。加えて `weight_dtype == float16` 条件も満たさない（int8/nvfp4） |
| `fp16_accumulation` | 効く可能性あり（要 A/B） | `torch.backends.cuda.matmul.allow_fp16_accumulation = True` はグローバル設定。ただし `PRIORITIZE_FP16 = True` も同時に立つため**品質差が出うる** |
| `autotune` | 効果は小さいと予想（要 A/B） | 実体は `torch.backends.cudnn.benchmark`。H3 DiT は transformer で conv がほぼ無い。VAE デコードには効きうる |

→ **A/B する価値があるのは `fp16_accumulation` と `autotune` の2つだけ**です。

---

## ★重要な発見：NVFP4 は Ada で「エミュレーション」されている

Test 3B のログに決定的な行がありました。

```
[INFO] Native ops: int8_tensorwise, float8_e4m3fn, convrot_w4a4, float8_e5m2
       , emulated ops: nvfp4, mxfp8
```

**`nvfp4` は emulated（エミュレーション）側です。**

| 量子化形式 | Ada (sm89) での扱い |
|---|---|
| `int8_tensorwise` | **native** |
| `convrot_w4a4` | **native** ← DiT の int8_convrot はこちら |
| `float8_e4m3fn` / `float8_e5m2` | **native** |
| **`nvfp4`** | **emulated** ← 採用中のテキストエンコーダ |
| `mxfp8` | emulated |

### 意味すること

1. **DiT（int8_convrot）はネイティブ実行**されています。ここは最適です
2. **テキストエンコーダ（nvfp4）はソフトウェア逆量子化**されています
3. それでも int8 の CPU 実行より **48倍速** なので、実用上は圧倒的に有利
4. ただし **Test 3B で観測した「衣装が参照から離れる」現象の説明候補**になります

### 現状の選択肢（Comfy-Org/MiniMax-H3 の全ラインナップ）

| ファイル | サイズ | Ada での扱い | 12GB GPU |
|---|---:|---|---|
| `qwen3vl_32b_minimax_h3_bf16` | 51.5 GB | native | ✗ |
| `qwen3vl_32b_minimax_h3_int8_convrot` | 27.1 GB | **native** | ✗（CPU なら可・約37分） |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` | 15.7 GB | **emulated** | △（11.4 GB peak で通る） |

**fp8 版のテキストエンコーダは公開されていません。**
native かつ 12GB に載る選択肢が存在しない、というのが現状です。

未検証の候補として GGUF `qwen3vl-32B-MiniMax-H3-Q4_K_M`（約19GB）があります。
ComfyUI-GGUF は別の実装経路なので、emulated 判定に該当しない可能性があります。

---

## Phase C — torch.compile ❌ 不採用（構造的に適用不可）

### 事前確認

| 項目 | 結果 |
|---|---|
| torch | 2.10.0+cu130 |
| 利用可能 backend | `cudagraphs`, `inductor`, `openxla`, `tvm` |
| inductor は Windows で動くか | **動く**（トイモデルで初回 compile 19.4 秒）。MSVC 不要、Triton が GPU 側を処理 |
| `cl.exe` / `CUDA_HOME` | いずれも未設定だが inductor の GPU 経路には不要だった |
| 対象 module | `MiniMaxH3Model.blocks`（L448）→ `compile_transformer_blocks_only` が正しく検出 |

`TorchCompileModelAdvanced` は `diffusion_model.blocks.{i}` を compile keys に選びます。
ログでも確認：

```
[INFO] TorchCompileModelAdvanced: Compile key list:
[INFO]  - diffusion_model.blocks.0
```

### Run 1（最も安全な構成）

`backend=inductor / mode=default / fullgraph=False / dynamic=false /
compile_transformer_blocks_only=True / dynamo_cache_size_limit=64`

**結果：失敗。**

```
torch._dynamo.exc.TorchRuntimeError:
Dynamo failed to run FX node with fake tensors:
  call_method __dlpack__(*(FakeTensor(..., device='cuda:0',
                            size=(26312, 28672), dtype=torch.bfloat16),), **{'stream': -1})
RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor).
```

### 原因（構造的）

`__dlpack__` は **ComfyUI の動的 VRAM ローディング**が重みを転送する経路です。
`FakeTensor` にはデータポインタが無いため、Dynamo はここをトレースできません。

`TorchCompileModelAdvanced` 自身のツールチップにも
「Disable dynamic VRAM feature as it can cause issues with compile」とあり、既知の衝突です。

### Run 2（`disable_dynamic_vram = True`）

**同じ `__dlpack__` エラーで失敗。**
`do not support disabling dynamic VRAM` の警告は出ていない（＝オプションは受理された）にもかかわらず、
ログには引き続き `Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged.` が出ました。

そして決定的な行：

```
loaded partially; 5213.46 MB usable, 4118.63 MB loaded,
                  15877.52 MB offloaded, 1176.11 MB buffer reserved, lowvram patches: 182
```

**12GB カードで H3 DiT に使えるのは約 5.2 GB、20 GB のうち 15.9 GB はオフロード必須です。**

### 結論

| | |
|---|---|
| 判定 | **現環境では不採用** |
| 理由 | この組み合わせ（ComfyUI の dynamic VRAM loading × MiniMax H3 × KJNodes `TorchCompileModelAdvanced`）では DLPack/FakeTensor 競合を回避できず実用不可だった |
| sec/step | 測定に到達せず（サンプリングが1 step も進まない） |
| 深追いしない理由 | ご指示の「compile crash → 深追いせず不採用」に該当。2構成とも同一の根本原因で失敗 |

> **この結論の範囲について**
>
> 否定しているのは「**現在の**組み合わせ」だけです。
> 具体的には ComfyUI v0.30.2 の dynamic VRAM loading（DLPack 経由の重みストリーミング）と
> KJNodes `TorchCompileModelAdvanced` を、12GB VRAM の本機で組み合わせた場合です。
>
> 以下は**否定していません**：
> - 将来の ComfyUI 実装（DLPack 経路を opaque custom op でラップするなど）
> - 別の compile 経路（ComfyUI 本体の `TorchCompileModel`、独自ラッパ、AOT 系など）
> - オフロードが不要になる構成（モデルが丸ごと載る VRAM 容量、より小さい量子化など）
>
> エラーメッセージ自体が
> "please wrap the custom kernel into an opaque custom op" と回避策を示しており、
> 上流側で対応されれば状況は変わりえます。

### ⚠️ 追加所見：単に失敗するのではなく、プロセスごとクラッシュする

`disable_dynamic_vram=True` の試行では compile がもう少し先まで進み、
**KJNodes が vendored している SageAttention の Triton カーネル**を
inductor がコンパイルしようとして失敗しました。

```
triton.compiler.errors.CompilationError: at 1:0:
def _quant_key_per_thread_int8_i64_kernel(Input, Output, Scale, L,
^
IndexError('Function argument index out of range')
```

その後、生成された inductor コード内で **Windows のハードフォールト**が発生：

```
Windows fatal exception: code 0x80000003
  File "...\torchinductor_<user>\dx\cdxjmjirbj...py", line 495 in call
  File "...torch\_inductor\output_code.py", line 638 in __call__
```

**ComfyUI プロセスが即死します（Python 例外ではないため捕捉不可）。**

これは Phase A の結論と直結しています。
H3 の attention は KJNodes の Triton/CUDA カスタムカーネルに置き換えられているため、
inductor がそれを再コンパイルしようとして破綻します。

→ **速度が出ないというだけでなく、安定性を損なう**ため、
現構成では torch.compile を有効化しないでください。

### 副産物：TF32 が無効

compile 試行中に inductor が警告を出しました。

```
TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
Consider setting `torch.set_float32_matmul_precision('high')`
```

`--fast` フラグ群には無い **torch レベルの設定**です。
ただし DiT は int8_convrot、TE は nvfp4 なので、fp32 matmul は累算経路に限られる見込みです。
**Phase B の A/B 候補に追加**（`fp16_accumulation` / `autotune` と並ぶ3つ目）。

---

## 実験ステータス

| Phase | 状態 |
|---|---|
| A Attention 経路 | ✅ 完了（現構成が Ada 最速。切替先なし） |
| B 公式高速化オプション | ⚠️ 調査完了。A/B 候補は `fp16_accumulation` / `autotune` / **TF32** の3つ |
| C torch.compile | ❌ **不採用**（DLPack×Dynamo の構造的衝突） |
| D 3060 を VLM 専用化 | 未着手 |
| E 3060 を後処理専用化 | 未着手 |
| F Character Master 品質 | 未着手 |
| G 品質プリセット | 未着手 |
| H DisTorch2 / MultiGPU | 未着手 |
| I Continuation (Test 4/5) | 未着手 |

---

## Phase C' — process-level `--disable-dynamic-vram`（完了・不採用）

`--disable-dynamic-vram` は **KJNodes ノードの `disable_dynamic_vram` とは別経路**の
正式 CLI オプションであることを確認（`comfy/cli_args.py` L179、`enables_dynamic_vram()`）。
隔離インスタンス（port 8400）で検証。本番 Desktop は未変更。

### 実測（同一 seed / prompt / 参照画像）

| 構成 | 定常 sec/step | 総時間 | RAM peak | 判定 |
|---|---:|---:|---:|---|
| **Test 3B baseline**（sage + dynamic VRAM） | **13.93** | **230.90 s** | 61.21 GiB | **採用中** |
| C'0（sage + dyn VRAM OFF） | 19.18 | 372.57 s | 63.49 GiB | 不採用 |
| C'1（sage 除去 + dyn OFF + compile） | 約 26〜28 | 513.93 s | — | 不採用 |

C'1 のステップ推移（compile 償却）: 156s → 77s → 27s → 26s → 28s → 27s

### 結論

1. `--disable-dynamic-vram` は**動作するが遅い**（+38% / step）。DiT のオフロード量は
   dynamic VRAM 時と同一（15,877.52 MB）で、違いは「流し方」だけ。legacy lowvram のほうが遅い
2. torch.compile は**クラッシュせず完走したが、baseline の約2倍遅い**
3. **SageAttention の寄与 > torch.compile の寄与。** sage を外すと step 時間がほぼ倍増し、
   compile では回収できない
4. C'1 がクラッシュしなかったことで、**ISSUE-2 の原因が SageAttention の
   vendored Triton カーネルであることが対照実験で確定**

### 上流調査（実装せず・記録のみ）

**thu-ml/SageAttention**

| # | 状態 | 内容 |
|---|---|---|
| #218 | Closed/Merged | "Add `torch.compile` support to SageAttention"（2025-07-25） |
| #337 | Open | Dynamo が fused quant kernel をトレースできず graph break |
| #162 | Open | SageAttention 2 + torch.compile で出力破損 |
| #74 | Open | full CUDA graph 化には custom operator 対応が必要 |
| #371 | Open PR | カーネルが current stream を使わず CUDA graph から漏れ → **silent corruption** |

**kijai/ComfyUI-KJNodes**

| # | 状態 | 内容 |
|---|---|---|
| #670 | Open | **Patch Sage Attention KJ で `Fatal Python error: Aborted`**（Qwen / Flux 2）← ISSUE-2 と同種が既報 |
| #686 | Open | torch.compile キャッシュの pickle 失敗で毎回約70秒の再コンパイル |

一部対応（#218）はあるが graph break・CUDA graph 漏れ・silent corruption が未解決。
**自前で custom_op 化する価値は現時点で低い。** 将来候補として記録。
特に #371 は「速度は出るが品質が静かに劣化する」危険があるため要警戒。

---

## 次回の再開手順

1. 実験用インスタンス起動（本番 Desktop とは別プロセス）:

```
.venv\Scripts\python.exe main.py --port 8399 --cuda-device 0 --reserve-vram 1.0 \
    --extra-model-paths-config <shared_model_paths.yaml のコピー>
```

`--cuda-device 0` は必須（llama.cpp による current device 汚染を防ぐ）。

2. ワークフロー: `workflows/H3_phase1_core.json`（= Test 3B baseline）
3. テスト参照画像: `h3test_face_ref.png`（ComfyUI ローカル input に配置済み）

### 未実施

| Phase | 内容 |
|---|---|
| **B** | `fp16_accumulation` / `autotune` / TF32 を各1回 A/B（数%未満なら終了） |
| **D** | RTX 3060 を Gemma VLM 専用 worker に分離（LM Studio / llama-server / OpenAI互換API の順で検討） |
| E | RTX 3060 を SeedVR2 / Frame Interpolation worker に |
| F | Character Master 品質（1枚→2枚→3枚、`match` vs `max`） |
| G | FAST / BALANCED / QUALITY プリセット |
| H | DisTorch2 / MultiGPU v2（隔離環境のみ。libcudart.so バグ要再確認） |
| I | Test 4 Tail Relay / Test 5 ref_video_audio |

**Single-clip Latency の限界はまだ結論を出していません。**
未検証: Phase B 軽量最適化 / AIMDO multi-device / DisTorch2 / model offload。
