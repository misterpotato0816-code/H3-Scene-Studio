# 01. 構造説明

`H3_phase1_core.json` / `H3_phase1_full.json` のグループ構成とデータフロー。

---

## データフロー概観

```
[10 Character Master]  参照画像1〜4枚 → 役割別リサイズ → キャラシート化
        │                                   │
        │                                   ├─────────────► [20] VLM が見る
        │                                   │
        └───────────────────────────────────┼─────────────► [60] H3 ref_image_0..3
                                            ▼
[20 Character Profile]  VLM が 顔/髪/肌/体格/衣装/素材 を抽出 → 編集可能なテキスト
        │
        ▼
[30 日本語入力]  ①シーン ②セリフ確定 ③前クリップ ④避けたい要素 + 尺(秒)→フレーム数
        │
        ▼
[40 H3 Director Prompt]  Profile + 日本語 + 尺 を結合 → VLM → 6セクション英語プロンプト
        │                 SYS-A(新規) / SYS-B(続き) を Any Switch で切替
        │                 → 編集可能 → VLM アンロード → VRAM 解放
        ▼
[50 H3 Models (Stable=v1)]  Ref2VA + VAE×2 + TurboLoRA(raw)
        │              TextEnc int8_convrot → SelectCLIPDevice(gpu:1) ★GPU1で conditioning
        │
[55 Experimental]  pruned TurboLoRA×2 → SageAttn → H3Mem → SigmaShift（既定バイパス）
        │
        ▼
[60 H3 Reference to Video] ◄── [45 Tail Relay] 前クリップ末尾 → ref_video_0 / ref_video_audio_0
        │                  ◄── 声質リファレンス → ref_audio_0
        │  SamplerCustomAdvanced (8 step) → VAEDecode(映像) / VAEDecodeAudio(音声)
        ▼
[70 Audio]  H3音声(セリフ+環境音) + BGM(別トラック) → 音量 → EQ → AudioMerge
        │
        ▼
[80 Output]  CreateVideo(fps=24) → SaveVideo   ← これが次クリップの入力になる
        │
        ├─► [85 Upscale]  RealESRGAN → 1080x1920 (full のみ)
        └─► [90 検証]     Whisper で生成音声を文字起こし (full のみ)
```

---

## グループ別

### 10 Character Master（人物固定の入力）

| ノード | 役割 |
|---|---|
| `LoadImage` ×4 | ①顔寄り ②上半身 ③全身 ④衣装・小物 |
| `LayerUtility: ImageScaleByAspectRatio V2` ×4 | 長辺 1536 / 1152 / 1152 / 896 px、32の倍数、crop、lanczos |
| `BatchImagesNode` | 4枚を1バッチに |
| `CR Image Grid Panel` | 2列のキャラシート化（VLM に1枚で見せる） |
| `PreviewImage` | シート確認 |

**初期状態：** ①② が有効、③④ はバイパス。
③④ を使うときは `LoadImage` と直下の resize の**両方**を選んで Ctrl+B。

**なぜ役割ごとにサイズを変えるか：**
参照画像は latent 化されて**毎サンプリングステップに乗る**ため、
全部を最大解像度にすると 12GB VRAM では速度が厳しくなります。
主役の顔だけ厚く、脇を薄くする配分にしています。

**バイパス時の挙動（実測）：** バイパスした参照は実行グラフから完全に外れ、
`BatchImagesNode` と H3 の両方で「そのスロットが存在しない」状態になります。
None が流れて落ちることはありません（`graphToPrompt()` で確認済み）。

### 20 Character Profile（VLM 抽出）

`llama_cpp_model_loader`（Gemma-4 E4B + mmproj, chat_handler=Gemma4, n_ctx=16384）
→ `llama_cpp_instruct_adv`（inference_mode=`images`, temperature 0.35）
→ `CR Text Replace` なしで直接 `ShowText|pysssss`

出力は固定フォーマットです。

```
FACE: ...
HAIR: ...
BUILD: ...
SKIN: ...
OUTFIT: ...
ACCESSORIES: ...
MATERIALS: ...
COLOR_KEYS: ...
```

黄色の `ShowText` は **その場で編集できます**。
`Any Switch (rgthree)` で「手動上書き」に切り替えれば、保存しておいた Profile を貼り付けて
VLM の再実行を飛ばせます（`any_01` が優先、バイパスすると `any_02` にフォールバック）。

### 30 日本語入力

| ノード | 内容 |
|---|---|
| ① シーン指示 | 日本語で自由記述 |
| ② セリフ確定 | 1行1発話。**空なら AI が創作** |
| ③ 前クリップ情報 | 既定は `NONE - this is a new clip`。続き生成時に書き換える |
| ④ 避けたい要素 | リップシンク崩れ・同一性ドリフト系のネガティブ |
| `PrimitiveFloat` 秒 → `MathExpression` | `min(362, max(124, round(a*b) + (5 - round(a*b) % 17) % 17))` |
| `Display Any` | 実際のフレーム数を表示 |

③ を**バイパスにしていない**のは意図的です。バイパスすると下流の結合チェーンに
None が流れるため、テキストで `NONE` を明示する方式にしています。

### 40 H3 Director Prompt

固定ヘッダ（`hdr:*`、折りたたみ表示）とユーザー入力を **2入力の JoinStringMulti を12段** 重ねて結合します。

> KJNodes の `JoinStringMulti` は読み込み時に動的入力 `string_3` 以降を作り直すため、
> 12入力版を手書きすると**リンクが静かに消えます**（実測で5本消失）。
> スキーマ宣言済みの `string_1` / `string_2` だけを使う2入力チェーンにしてあります。

結合順：

```
CHARACTER PROFILE ヘッダ → Profile本文
SCENE ヘッダ            → ①シーン指示
DIALOGUE ヘッダ         → ②セリフ確定
PREVIOUS CLIP ヘッダ    → ③前クリップ情報
AVOID ヘッダ            → ④避けたい要素
TARGET LENGTH ヘッダ    → フレーム数 → CANVAS 説明
```

システムプロンプトは2種類を `Any Switch` で切替：

- **SYS-A（新規）** … `[reference generation]`。`<Video N>` を使わない
- **SYS-B（続き）** … `[video continuation + reference generation + audio reference]`。
  `<Video 1>` / `<Audio 1>` の定義・保持宣言・[Shot 1] の書き出しを強制

両方に共通のハードルール：

| | 内容 |
|---|---|
| R1 | **セリフは一字一句コピー。**②が空のときだけ AI が創作してよい |
| R2 | `non_diegetic_music` は原則 `N/A`（BGM は後段で別トラック合成するため） |
| R3 | 縦型 9:16。クローズアップ主体、横широ構図を避ける |
| R4 | 説明・カメラ指示は英語、セリフ・歌詞・画面内文字は原文のまま |
| R5 | 指定尺にセリフを収める |

生成後は `CR Text Replace` で ``` と `<think>` を除去 → `ShowText` で編集可能
→ `llama_cpp_unload_model` → `LayerUtility: PurgeVRAM V2` で VLM を VRAM から追い出します。

### 45 Tail Relay（続き生成）

```
LoadVideo(前クリップ) → GetVideoComponents ─┬─ images → GetImageSizeAndCount ─ count
                                             │                    │
                                             │   MathExpression: max(0, count - tail)
                                             │                    ▼
                                             │            ImageFromBatch(batch_index, length=tail)
                                             │                    └─► h3.ref_videos.ref_video_0
                                             │
                                             └─ audio → TrimAudioDuration
                                                          start_index = -(tail/24)   ← 末尾から
                                                          duration    =  (tail/24)
                                                          └─► h3.ref_video_audios.ref_video_audio_0
```

`tail フレーム数` は `PrimitiveInt`（既定 56）で、そこから秒数を自動計算します。
**固定仕様ではなく可変**です。39 / 56 / 73 / 90 が `n % 17 == 5` を満たすので切り捨てロスが出ません。

**初期状態は全ノードがバイパス。** 続き生成をするときにグループごと Ctrl+B で有効化します。

続き生成中も **Character Master は同時に接続されたまま**です。
`<Video 1>` だけに頼ると数クリップで人物が崩れるため、参照画像による同一性アンカーを併用します。

### 50 H3 Models (Stable=v1)

**H3_HETERO_V1（実測 217.61 秒）の構成を厳密に再現するグループです。**
仕様は `H3_HETERO_V1_MANIFEST.json` / `docs/H3_HETERO_V1_FINAL.md` が正。

```
UNETLoader (Ref2VA pruned int8_convrot)
  → LoraLoaderModelOnly (turbo_v4_step600_ema  ※raw / 1.0)
  → （55 は既定バイパス）→ BasicGuider / BasicScheduler

CLIPLoader (qwen3vl_32b int8_convrot, device=default)
  → SelectCLIPDevice (device=gpu:1)      ★異種GPU高速化の本体
  → MiniMaxH3ReferenceToVideo.clip
```

#### ★ CUDA フェーズ境界（40 → 50 の間）— 必須

**llama.cpp（Gemma-4 VLM）はプロセス全体の CUDA current device を GPU1 へ動かし、
`llama_cpp_unload_model` を通しても戻しません。**実測ログ：

```
1-vlm-pre       current_device=cuda:1     ← Profile VLM の時点で既に移動済み
2-vlm-post      current_device=cuda:1
3-unload-post   current_device=cuda:1     ← unload では戻らない
4-restore/post  current_device=cuda:0     ← H3CudaPhaseRestore が復元
```

ComfyUI はテキストエンコーダの **base load/offload device を CLIPLoader 実行時の
`torch.cuda.current_device()` から確定**し、`SelectCLIPDevice` はその base を基準に
retarget します。汚染されたまま H3 へ入ると base が cuda:1 になり、
GPU1 側の TE ストリーミングが約 11 GiB まで膨らんで **conditioning 中に OOM** します。

#### 誤診しないための注記（重要）

原因調査の過程で「参照画像の枚数が多いことが OOM の原因」という仮説を立てましたが、
**この仮説は実測で否定済みです。**

| 参照画像 | GPU1 peak VRAM | 結果 |
|---:|---:|---|
| 2枚 | **10,959 MiB** | 同一箇所で OOM |
| 1枚 | **10,991 MiB** | 同一箇所で OOM |

枚数を半分にしても GPU1 の使用量も OOM 発生位置も変化しませんでした。
（この 2 つの実測値は証拠として残します。）

真の原因は **VLM / llama.cpp 実行後に CUDA current device が cuda:1 へ残留し、
H3 CLIPLoader の base device が誤って cuda:1 になっていたこと**です。
`H3-Device-Barrier` の `H3CudaPhaseRestore` で cuda:0 へ Restore して解決済みで、
修正後の GPU1 peak は **7,249 MiB**、conditioning は 24.98 秒で完走しています。

したがって、**参照画像を減らしても同種の OOM は直りません。**
同じ症状が再発した場合は、まず `[H3Barrier]` の `current_device` ログを確認してください。

そのため 40 グループの末尾に境界を置いています。

```
Any Switch → 2-vlm-post probe → llama_cpp_unload_model
  → 3-unload-post probe → PurgeVRAM V2
  → ★ H3CudaPhaseRestore (target_gpu=0)
       ├→ H3GatedCLIPLoader.barrier      （CLIP のロードを必ずこの後にする）
       └→ MiniMaxH3ReferenceToVideo.prompt
```

**単に Reset ノードを置くだけでは不十分**です。ComfyUI は依存関係で実行順を決めるため、
入力を持たない素の `CLIPLoader` は VLM より先に走りうる。
`H3GatedCLIPLoader` の `barrier` 入力がこの順序を保証します。

復元に失敗した場合は黙って続行せず、H3 生成に入る前に例外で停止します。

実装は `custom_nodes/H3-Device-Barrier/`（H3プロジェクト側の最小追加）。
ComfyUI 本体と第三者 custom node は改変していません。

#### GPU 割当

| GPU | 実機 | 役割 | 決定箇所 |
|---|---|---|---|
| GPU1 | RTX 3060 | text / reference conditioning | `SelectCLIPDevice(device=gpu:1)` |
| GPU0 | RTX 4070 Ti | DiT sampling / Video VAE / Audio VAE | 起動引数 `--default-device 0` |

v1 実測の段別時間：conditioning 16.42 s（GPU1）／sampling 164.65 s（GPU0, 8 steps）／
video decode 27.98 s／audio decode 1.52 s＝合計 217.61 s。

#### Turbo LoRA が raw のままな理由

v1 が固定している `minimax_h3_turbo_v4_step600_ema.safetensors` は
本機で **518/518 キーが bind に失敗**します（実測）。つまり 217.61 秒版は
**実質 Turbo LoRA 未適用**で走っており、目視の品質確認もその状態で行われました。
Stable 側は v1 の実効状態を変えないため、意図的にこのまま維持します。
正しく bind する `_pruned_comfyui` 版は 55 Experimental にあります。

代替ローダーをバイパス状態で併置してあります。

| | 有効 | バイパス（代替） |
|---|---|---|
| 本体 | `UNETLoader` int8_convrot | `UnetLoaderGGUF` Q4_K_S（未導入） |
| TextEnc | `CLIPLoader` int8_convrot | `CLIPLoader` nvfp4（Test 3B 基準）/ `CLIPLoaderGGUF` Q4_K_M（未導入） |

### 55 Experimental（未測定・既定バイパス）

Stable の LoRA と guider/scheduler の**間**に挟まっています。
グループごとバイパスしている既定状態が v1 厳密一致で、
グループを有効化すると以下の経路になります。

```
… → LoraLoaderModelOnly (turbo_v4_step600_ema_pruned_comfyui, 1.0)   ※正しく bind
    → LoraLoaderModelOnly (turbo_4step_ckpt850_pruned_comfyui, 0.5)
    → PathchSageAttentionKJ (auto)
    → MiniMaxH3MemoryEfficientSageAttentionPatch
    → MiniMaxH3SigmaShift (video 12.0 / audio 6.0)
```

Test 3B（単GPU・Sage 有効）の sampler は 13.93 s/step、
v1（GPU1 conditioning・Sage なし）は 20.58 s/step でした。生成条件は同一です。
両者の併用が速いかどうかは**未測定**であり、断定できません。
また Turbo LoRA が実際に効くため、同一 seed でも絵が変わります。

### 60 H3 Reference to Video

`MiniMaxH3ReferenceToVideo` の入力：

| 入力 | 接続元 |
|---|---|
| `ref_image_0..3` | Character Master（①②は既定有効、③④は既定バイパス） |
| `ref_image_4` | 未接続（予備） |
| `ref_video_0` | Tail Relay（既定バイパス） |
| `ref_video_audio_0` | Tail Relay（既定バイパス） |
| `ref_audio_0` | 声質リファレンス `LoadAudio`（既定バイパス） |
| `prompt` | 40 の Any Switch |
| `length` | 30 の MathExpression |
| `width`/`height` | 既定 768 × 1344（ウィジェット直接指定） |
| `ref_image_size` | `max` |

サンプリングは
`RandomNoise` + `BasicGuider` + `KSamplerSelect(res_multistep)` + `BasicScheduler(simple, 8)`
→ `SamplerCustomAdvanced` → `VAEDecode` / `VAEDecodeAudio`（どちらも `output` スロット）。

### 70 Audio

H3 は映像と音声を**1本の混ざったステレオ**で出すため、後から BGM だけ差し替えることはできません。
そこで R2 ルールで H3 側に BGM を作らせず、BGM を別トラックで足す構成にしています。

```
H3音声 ─────────────────────────────► AudioMerge.audio1
BGM → TrimAudioDuration → AudioAdjustVolume(-14dB) → AudioEqualizer3Band(中域-4dB) → .audio2
```

`AudioMerge` は **audio1 の長さに audio2 を合わせ**、合成後にクリップ防止の正規化を行います。
必ず audio1 = H3 音声にしてください（本体ソースで確認済み）。

BGM を変えたいときは `LoadAudio` を差し替えて **70 以降だけ再実行**すれば済みます。

### 80 Output / 85 Upscale / 90 検証

- `CreateVideo(fps=24)` → `SaveVideo`（`H3/日付/h3_時刻`）
- full のみ：`UpscaleModelLoader(RealESRGAN_x4plus)` → `ImageUpscaleWithModel` →
  9:16 の 1080×1920 に整形 → 別途保存
- full のみ：`Apply Whisper`(large-v3-turbo, Japanese) → `ShowText` で
  **生成音声のセリフを文字起こし**。30 の「②セリフ確定」と見比べる

---

## ノード依存（使用パック）

| パック | 使っているノード |
|---|---|
| **ComfyUI 本体 v0.30.2** | `MiniMaxH3ReferenceToVideo` `MiniMaxH3SigmaShift` `UNETLoader` `CLIPLoader` `VAELoader` `LoraLoaderModelOnly` `RandomNoise` `BasicGuider` `KSamplerSelect` `BasicScheduler` `SamplerCustomAdvanced` `VAEDecode` `VAEDecodeAudio` `LoadImage` `PreviewImage` `BatchImagesNode` `ImageFromBatch` `ImageUpscaleWithModel` `UpscaleModelLoader` `LoadAudio` `PreviewAudio` `SaveAudioMP3` `AudioMerge` `AudioAdjustVolume` `AudioEqualizer3Band` `TrimAudioDuration` `LoadVideo` `GetVideoComponents` `CreateVideo` `SaveVideo` `PrimitiveInt` `PrimitiveFloat` `MarkdownNote` |
| **comfyui-kjnodes** | `PathchSageAttentionKJ` `JoinStringMulti` `GetImageSizeAndCount` |
| **rgthree-comfy** | `Any Switch (rgthree)` `Display Any (rgthree)` |
| **ComfyUI-GGUF** | `UnetLoaderGGUF` `CLIPLoaderGGUF`（代替ローダー、既定バイパス） |
| **ComfyUI-llama-cpp** | `llama_cpp_model_loader` `llama_cpp_parameters` `llama_cpp_instruct_adv` `llama_cpp_unload_model` |
| **comfyui_layerstyle** | `LayerUtility: ImageScaleByAspectRatio V2` `LayerUtility: PurgeVRAM V2` |
| **ComfyUI_Comfyroll_CustomNodes** | `CR Image Grid Panel` `CR Text Replace` `CR Integer To String` |
| **comfyui-custom-scripts** | `ShowText\|pysssss` `MathExpression\|pysssss` |
| **comfyui_tinyterranodes** | `ttN text` |
| **ComfyUI-Whisper** | `Apply Whisper`（full のみ） |

**すべて導入済みです。追加インストールは不要です。**
