# 03. 不足項目一覧

実機を走査した結果に基づきます（推測ではありません）。

- モデル：`<COMFYUI_SHARED_MODELS>` / `...\ComfyUI\ComfyUI\models` / `E:\AI\Models` を実列挙
- ノード：動作中 ComfyUI の `/object_info`（2039 ノード）と突合
- ライブラリ：`ComfyUI\ComfyUI\.venv\Scripts\python.exe` で import 検証

---

## A. 未導入モデル（これが無いと H3 生成ができません）

| # | ファイル | 想定サイズ | 配置先 | 用途 |
|---:|---|---:|---|---|
| 1 | **H3 Ref2VA 本体**<br>`minimax_h3_ref2va_pruned_fp8_scaled.safetensors`<br>または `MiniMax-H3-Ref2VA-Q4_K_S.gguf` | 約19GB<br>約11〜12GB | `models\diffusion_models\` | **必須。** 参照画像による人物保持の中核 |
| 2 | **H3 用テキストエンコーダ**<br>`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`<br>または `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 約32GB<br>約19GB | `models\text_encoders\` | **必須。**（下の B-1 も参照） |
| 3 | `minimax_h3_turbo_v4_step600_ema.safetensors` | 数百MB〜数GB | `models\loras\` | 高速化。無くても Steps を上げれば動く |
| 4 | `minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors` | 数百MB〜数GB | `models\loras\` | 同上 |

サーバ側 `POST /api/prompt` の検証でも、**エラーはこの4件だけ**でした。
他 65 ノードはすべて通過しています。

### 12GB VRAM 向けの推奨

| | 推奨 | 理由 |
|---|---|---|
| 本体 | **GGUF Q4_K_S** | 約11〜12GB。4070 Ti に載る |
| TextEnc | **GGUF Q4_K_M** | 64GB RAM への部分オフロードが効く |

GGUF を使う場合は `50 H3 Models` で
`UnetLoaderGGUF` / `CLIPLoaderGGUF` を Ctrl+B で有効化し、
`UNETLoader` / `CLIPLoader` をバイパスしてください。

ディスクは E: に 2,850GB 空きがあるので、両方落として比べるのも現実的です。

---

## B. 要確認事項（判断が必要）

### B-1. テキストエンコーダは INT8 ConvRot を基準にする（確認済み）

**結論：4070 Ti では `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` を基準にします。**
Phase 1 のワークフローは既にこれを既定にしてあります。

#### 記事ワークフローでの実際の選択値（再確認した結果）

当初「記事は NVFP4 を使っているかもしれない」と書きましたが、**誤りでした。**
記事 JSON を精査した結果は次のとおりです。

| 記事 | ノード | 実際の widget 値 |
|---|---|---|
| turbolora 版 | `CLIPLoader` #128 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| turbolora 版 | `ClipLoaderGGUF` #43157 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| R2V 版 | `CLIPLoader` #128 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |

`nvfp4` という文字列が JSON に出てくるのは、**ワークフロー添付のモデル DL メタデータ内の URL だけ**でした。

```
https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

つまり **NVFP4 が実際に選択・実動していた証拠は資料から確認できません。**
おそらくこのメタデータの URL 経由で NVFP4 が落ちてきたものと思われます。

#### 現在入っているもの

```
models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   14.61 GB
```

NVFP4 は NVIDIA Blackwell 世代（RTX 50 シリーズ）向けの 4bit 形式です。
RTX 4070 Ti は Ada（sm_89）、RTX 3060 は Ampere（sm_86）でネイティブ対応がありません。
**動作するという確証も、しないという確証もありません（本機で未検証）。**

#### 方針

| 優先 | ファイル | 状態 |
|---|---|---|
| **1（基準）** | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | **要DL。** 記事の実選択値。ワークフローの既定 |
| 2（12GB 向け） | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | 要DL。RAM オフロードが効く |
| 3（最後の手段） | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 導入済みだが**実動の裏付けなし**。ワークフローにバイパス状態で併置 |

### B-2. `ComfyUI-MultiGPU` は現時点で**有効化しないでください**（調査済み・未解決）

```
custom_nodes\ComfyUI-MultiGPU.disabled   ← 無効化されたままにしてください
```

無効化された理由は **Windows で `libcudart.so` を読みに行って落ちる既知バグ**でほぼ確定です。
調査結果は以下のとおりで、**現在版でも解消していません。**

#### 該当コード（導入済み v2.6.4）

`p2p_registry.py` L16-21:

```python
def _get_libcudart():
    """Load libcudart.so once and cache the handle."""
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so")   # ← Linux 固有名がハードコード
    return _libcudart
```

- プラットフォーム判定なし
- Windows 用の `cudart64_*.dll` へのフォールバックなし
- **try/except なし** → `FileNotFoundError` がそのまま呼び出し元へ伝播

#### 本機での再現（実測）

```
platform: win32 | os.name: nt
CDLL(libcudart.so): FileNotFoundError: Could not find module 'libcudart.so'
comfy_kitchen importable: True
torch lib dir: ...\.venv\Lib\site-packages\torch\lib
  found: cudart64_13.dll          ← 本来読むべきファイル（torch 2.10.0+cu130 同梱）
```

#### 上流の状況（未修正）

| | 内容 |
|---|---|
| 導入済み | v2.6.4 / commit `b51c99a` (2026-05-08) / branch `main` / 作業ツリーはクリーン |
| ファイル導入元 | PR #180「Fix DLPack P2P Cross-Device Transfer (cu130)」2026-03-17 merge |
| `p2p_registry.py` の変更履歴 | commit `b526e11`(2026-03-17) **の1件のみ**。以後修正されていない |
| Issue #187 | 「Could not find module 'libcudart.so'」2026-03-31 起票・**Open**（コメント3件） |
| Issue #210 | 「p2p_registry.py hardcodes libcudart.so - always crashes on Windows (fix included)」2026-08-04 起票・**Open** |
| 上流 `main` の現在の中身 | `ctypes.CDLL("libcudart.so")` のまま。Windows 対応は入っていない |

#### 危険な点：起動時には落ちない

呼び出し元 `__init__.py` L556-557 は短絡評価です。

```python
if _valid_cuda(tensor_device) and _valid_cuda(exec_device):
    if tensor_device.index != exec_device.index and not p2p_registry.can_access_peer(...):
```

`tensor_device.index != exec_device.index` が真になったときだけ `can_access_peer` が呼ばれます。
つまり **ノードのロードや単一 GPU の生成では何も起きず、
「実際に GPU をまたいだ瞬間」＝ MultiGPU を使いたい場面でだけ生成が落ちます。**
「有効化したら普通に動いた」と誤認しやすい、たちの悪い壊れ方です。

なお本機は `comfy_kitchen` が導入済み（0.2.8）のため、この DLPack ガードパッチは**適用される側**です。

#### 結論と選択肢

**Phase 1 は MultiGPU を一切使っていないため、無効のままで支障ありません。**

将来デュアル GPU を使いたい場合の選択肢：

| 案 | 内容 | リスク |
|---|---|---|
| **A. 触らない（推奨）** | 無効のまま。単一 GPU + CPU オフロードで運用 | なし |
| B. ローカルパッチ | `_get_libcudart()` を Windows 対応に書き換える（Issue #210 に修正案あり）。`torch/lib/cudart64_13.dll` を絶対パスで読む + try/except で P2P 不可扱いにフォールバック | 上流更新で上書きされる。要再適用 |
| C. 上流の修正待ち | Issue #210 のクローズを待つ | 時期不明 |

> B を実施する場合は、**別ディレクトリに複製した検証用インスタンス**で先に確認してください。
> 本番インスタンスで直接有効化するのは避けるべきです。
> なお 4070 Ti と 3060 は NVLink がないため P2P はどのみち不可で、
> 正しく修正しても動作は「CPU 経由でステージング」になります。速度面の期待値は高くありません。

### B-3. `MiniMaxH3MemoryEfficientSageAttentionPatch` — 原因は KJNodes のバージョン差（確認済み）

**結論：条件付き登録ではなく、単純なバージョン差です。導入済み KJNodes v1.4.8 がこのノードより古い。**

#### 提供元（記事 JSON の properties で確認）

```json
"cnr_id": "comfyui-kjnodes",
"ver": "1289b52fbb6d64a339a4047b9ea74cf7758ccf1e",
"Node name for S&R": "MiniMaxH3MemoryEfficientSageAttentionPatch"
```

→ `kijai/ComfyUI-KJNodes`。

#### 追加された commit

| | |
|---|---|
| commit | `1289b52fbb6d64a339a4047b9ea74cf7758ccf1e` |
| 日付 | **2026-08-03** |
| メッセージ | "Add MiniMaxH3MemoryEfficientSageAttentionPatch reduces peak VRAM use when using sageattention slightly" |
| 変更ファイル | `__init__.py` (+2) / **`nodes/ltxv_nodes.py` (+67)** |

#### バージョンの時系列

| バージョン | 日付 | |
|---|---|---|
| v1.4.8 | 2026-07-27 | **← 導入済み（本機）** |
| commit `1289b52` | **2026-08-03** | ← ノード追加 |
| v1.4.9 | 2026-08-05 | ← このノードを含む最初のリリース |
| v1.5.0 | 2026-08-07 | ← 上流の現在 |

**v1.4.8 はノード追加の 7 日前です。** 記事 WF のファイル名は `2026-08-09-...` で、
`ver` が commit ハッシュであることから、記事側は git チェックアウト版を使っていたと分かります。

#### 「条件付き登録では？」の検証結果 → 否定

KJNodes の `__init__.py` には確かに条件付き登録があり、
`#ltxv` ブロック（L343-367）の `try/except Exception` が失敗すると
そこに含まれるノード群が **warning だけ出して丸ごと消えます**。
このノードは `nodes/ltxv_nodes.py` にあるので、まさにこのブロックの対象です。

そこで実際に稼働中の ComfyUI（`/object_info` 2039ノード）で、
同ブロックのノードが登録されているか確認しました。

```
LTXVAddGuideMulti                          present
LTXVAddGuidesFromBatch                     present
LTXVAudioVideoMask                         present
LTX2_NAG                                   present
LTXVChunkFeedForward                       present
LTX2SamplingPreviewOverride                present
LTX2LoraLoaderAdvanced                     present
WanVideoMemoryEfficientSageAttentionPatch  present   ← 同じ命名ファミリー
PatchTritonVAE                             present
```

**9個すべて present。try/except は成功しています。** よって条件付き登録は原因ではありません。

さらに、導入済み KJNodes ツリー全体を `minimax` で大文字小文字無視の全文検索した結果は
**0 件**でした（`/object_info` 上の `Minimax*` は `comfy_api_nodes.nodes_minimax` の
Hailuo クラウド API ノードで無関係）。
同ファミリーの `LTX2MemoryEfficientSageAttentionPatch` と
`WanVideoMemoryEfficientSageAttentionPatch` は存在しており、**H3 版だけが後発**という状況です。

#### 現状の扱い

Phase 1 では `PathchSageAttentionKJ`（v1.4.8 に存在・導入済み）だけを使っています。
このノードは無くても動きますが、名前のとおり
「sageattention 使用時のピーク VRAM をわずかに削減」するものなので、
**12GB 環境では入れる価値があります。**

**ご指示どおり KJNodes の更新は行っていません。**
更新する場合は v1.4.9 以上（現行 v1.5.0）が必要です。
ただし ComfyUI Manager 経由の cnr インストール（`.tracking` あり・git 管理外）なので、
更新は Manager から行い、事前にスナップショットを取ることを推奨します。
記事側でも BYPASS されている `SpectrumApplyMiniMaxH3` / `SolAttnPatch` は引き続き不採用です。

---

## C. 未導入ノード

**Phase 1 のワークフローで必要なノードは、すべて導入済みです。追加インストールは不要。**

参考までに、記事版が使っていて未導入のもの（本設計では代替済み）：

| ノード | 記事での用途 | Phase 1 での扱い |
|---|---|---|
| `ImpactMakeImageList` | 参照画像のリスト化 | `BatchImagesNode`（本体）で代替 |
| `ImageListToBatch+` | リスト→バッチ | 同上。不要 |
| `RTXVideoSuperResolution` | 2倍アップスケール | `RealESRGAN`（導入済）で代替。入れれば使用可 |
| `LTXVLoopingSampler` / `LTXVSpatioTemporalTiledVAEDecode` | LTX 2パス | LTX パスごと不採用 |
| `SolAttnPatch` / `SpectrumApplyMiniMaxH3` | 各種パッチ | 記事側でも BYPASS。不採用 |
| `MiniMaxH3MemoryEfficientSageAttentionPatch` | H3 省メモリ Attention | **代替なし**（B-3 参照） |
| `mape Variable` | 変数管理 | 直接結線で代替 |

---

## D. 動作未確認の部分（重要）

**正直に書きます。以下はまだ検証できていません。**

| 項目 | 状態 | 理由 |
|---|---|---|
| **H3 での実際の動画生成** | ❌ **未実施** | Ref2VA 本体が未導入で物理的に不可能 |
| 生成品質・人物の一貫性 | ❌ 未検証 | 上記のため |
| 12GB VRAM での実行可否・速度 | ❌ 未検証 | 上記のため |
| Turbo LoRA 2段（1.0 + 0.5）の妥当性 | ⚠️ 記事の値を踏襲しただけ | 実測していない |
| `MiniMaxH3SigmaShift (12, 6)` | ⚠️ 記事の値を踏襲 | 本体既定は (12, 3) |
| `res_multistep` + `simple` / 8 steps | ⚠️ 記事の値を踏襲 | 実測していない |
| **Tail Relay が実際につながるか** | ⚠️ **結線のみ確認** | `graphToPrompt()` で `ref_video_0` への配線は確認済み。実生成での連続性は未検証 |
| Character Profile の抽出品質 | ⚠️ 未検証 | Gemma-4 E4B は導入済みなので**すぐ試せます** |
| 日本語セリフの一字一句保持 | ⚠️ 未検証 | 同上。**すぐ試せます** |
| Whisper `large-v3-turbo` の初回DL | ⚠️ 未実施 | 選択肢としては存在を確認済み |
| BGM 合成チェーンの音質 | ⚠️ 未検証 | ノードの仕様はソースで確認済み |
| アップスケール後の 9:16 整形 | ⚠️ 未検証 | — |

### 検証できたこと

| 検証 | 結果 |
|---|---|
| 全ノード型が実在する | ✅ |
| 静的バリデーション | ✅ 0 error |
| ComfyUI 1.47.12 でロードできる | ✅ core 94/94 ノード・102/102 リンク、full 110/110・118/118、console error 0 |
| 実行グラフへの変換 | ✅ 正常 |
| バイパスしたノードが正しく外れる | ✅ `BatchImagesNode` は有効な2枚だけを受け取る |
| Tail Relay 有効時の結線 | ✅ `ref_video_0` ← ImageFromBatch、`ref_video_audio_0` ← TrimAudioDuration |
| サーバ側 prompt 検証 | ✅ 未導入モデル4件以外はすべて通過 |

---

## E. すでに揃っていて追加不要なもの

調査で判明した「実は入っていた」もの。**当初の想定より不足は少ないです。**

| | 実体 | 備考 |
|---|---|---|
| **Gemma-4 VLM** | `models\LLM\gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf` (7.48GB) | ComfyUI インストール側の `models` にありました |
| **その mmproj** | `models\LLM\gemma-4-E4B-it-mmproj-BF16.gguf` (0.92GB) | 画像を見るのに必須。あります |
| chat_handler `Gemma4` | ComfyUI-llama-cpp に存在 | |
| **H3 映像 VAE** | `minimax_h3_video_vae_fp16.safetensors` (4.85GB) | |
| **H3 音声 VAE** | `minimax_h3_audio_vae_fp32.safetensors` (0.56GB) | |
| **SeedVR2 モデル** | `models\SEEDVR2\seedvr2_ema_7b_fp16.safetensors` (15.35GB) + VAE | Phase 2 の高品質アップスケール候補 |
| アップスケーラ | `RealESRGAN_x4plus.pth` | Phase 1 で使用 |
| Whisper `large-v3-turbo` | 選択肢に存在（初回実行時に自動DL） | |
| sageattention / triton 3.7.1 | 導入済み | `PathchSageAttentionKJ` が機能する |
| llama_cpp 0.3.40 / whisper / gguf / deep_translator / accelerate | 導入済み | |
| ComfyUI 本体の音声編集ノード群 | `AudioMerge` `AudioConcat` `AudioEqualizer3Band` `TrimAudioDuration` ほか | 外部 DAW なしで音声編集が完結 |
| ACE-Step ノード | 本体に同梱（モデルは未導入） | Phase 2 の BGM 生成候補 |

> **モデルの置き場所が3系統に分かれています。**
> `ComfyUI-Shared\models`（既定）/ `ComfyUI\ComfyUI\models`（インストール同梱）/
> `E:\AI\Models`（`extra_model_paths.yaml`、ほぼ空）。
> 新しく落とすモデルは `ComfyUI-Shared\models` に置くのが分かりやすいです。

---

## F. ご用意いただきたい素材

| 素材 | 必須度 | 備考 |
|---|---|---|
| **キャラクター参照画像**（顔がはっきり分かるもの） | **必須** | 1枚でも可。長辺 1536px 以上あると有利 |
| 上半身・全身・衣装の参照 | 任意 | あるほど一貫性が上がる |
| 声質リファレンス音声（日本語 5〜15秒） | 任意 | 複数クリップで声を揃えたい場合は推奨 |
| BGM 音源 | 任意 | YouTube 公開ならライセンス要確認 |

配置先：`<COMFYUI_SHARED_ROOT>\input\`

---

## G. 最短で動かすためのチェックリスト

```
□ 1. H3 Ref2VA 本体            → models\diffusion_models\
□ 2. H3 用テキストエンコーダ    → models\text_encoders\
□ 3. Turbo LoRA ×2（任意）      → models\loras\
□ 4. キャラクター参照画像       → ComfyUI-Shared\input\

B-1 / B-2 / B-3 はいずれも調査完了。未回答の確認事項はありません。

- B-1 → INT8 ConvRot を基準に（ワークフロー既定済み）
- B-2 → MultiGPU は**無効のまま**
- B-3 → KJNodes v1.4.9+ が必要（更新は未実施）
```

1〜2 が揃えば H3 生成が動きます。
**4 だけでも、今日から 20〜40（Profile 抽出とプロンプト生成）は試せます。**
