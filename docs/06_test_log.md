# 06. Phase 1 実生成テストログ

**このファイルは実測値のみを記録します。未実施の欄は「未実施」と書き、推測値は入れません。**

---

## 測定環境ベースライン

| 項目 | 値 | 取得方法 |
|---|---|---|
| GPU0 | RTX 4070 Ti / 12,282 MiB | `nvidia-smi` |
| GPU1 | RTX 3060 / 12,288 MiB | `nvidia-smi` |
| Driver | 610.88 | `nvidia-smi` |
| **GPU0 アイドル時使用量** | **1,425 MiB** | 測定時点（デスクトップ描画分） |
| GPU1 アイドル時使用量 | 0 MiB | |
| RAM | 64 GB | |
| torch | 2.10.0+cu130 / CUDA デバイス 2 | `.venv` |
| sageattention | 導入済み | import 成功 |
| ComfyUI | v0.30.2 | |
| KJNodes | **v1.5.0**（本作業で 1.4.8 から更新） | |
| ComfyUI-MultiGPU | `.disabled`（有効化しない） | |

> **VRAM 実測値は GPU0 のアイドル 1,425 MiB を含む総使用量**で記録します。
> 生成そのものの使用量を知りたい場合はこの値を差し引いてください。

---

## テスト用参照画像

ご提供のキャラクター画像がまだ無いため、**既存の input フォルダにあった画像を代用**しました。

| | |
|---|---|
| 元画像 | `kantan_20260812_112915_299d28.png`（802 × 1317） |
| テスト用 | `h3test_face_ref.png`（544 × 487、上半身クロップを新規作成） |
| 特徴 | ピンクのボブ＋ヘアクリップ、白／ラベンダーの衣装、リボンチョーカー、パフスリーブ |

同一性の検証に使える程度に特徴がはっきりしているため選びました。
**本番ではご自身のキャラクターシートに差し替えてください。**
元画像は変更していません（クロップは別ファイルとして新規保存）。

---

## 導入モデル（案3）

| ファイル | サイズ | 保存先 | 状態 |
|---|---:|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.53 GiB | `models\diffusion_models\` | ダウンロード中 |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.28 GiB | `models\text_encoders\` | ダウンロード中 |
| `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.73 GiB | `models\loras\` | ダウンロード中 |
| `minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors` | 0.58 GiB | `models\loras\` | ダウンロード中 |

---

## Test 1 — Character Profile 抽出（VLM のみ）✅ 成功

| 項目 | 実測値 |
|---|---|
| 実行グラフ | 19ノード（10/20/30 のみ、H3 系は全バイパス） |
| 使用モデル | `gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf` + `mmproj-BF16` |
| 参照画像 | `h3test_face_ref.png` 1枚（544×487） |
| **生成時間** | **24.1 秒** |
| **GPU0 peak** | **6,295 MiB**（アイドル 1,425 込み） |
| **GPU1 peak** | **3,545 MiB** ← llama.cpp が2枚目にも展開 |
| GPU0 平均使用率 | 13.8 % |
| **RAM peak** | **26.65 GiB** |
| エラー | なし |

### 出力（実際の抽出結果）

```
FACE: Round face, almond-shaped eyes, light purple irises, thin eyebrows, small nose,
      small mouth, fair skin tone
HAIR: Pastel pink, medium length, layered cut with blunt bangs covering the forehead,
      straight texture, worn down
BUILD: Slender build, average proportions, narrow shoulders, upright posture
SKIN: Fair tone, matte finish, smooth texture
OUTFIT: White corset-style bodice with a plunging neckline, light purple ribbon detailing
        around the neck and chest, white ruffled sleeves extending past the elbow,
        matching light purple cuffs on the wrists, white layered skirt visible at the bottom
ACCESSORIES: Small silver hair clips near the temples
MATERIALS: Smooth matte fabric for the main outfit, sheer ruffled material for the
           sleeves/cuffs, satin-like finish on the ribbons
COLOR_KEYS: Pastel pink, white, light purple, silver
```

**評価：良好。** 指定した8項目フォーマットを完全に遵守。
髪型（レイヤードボブ＋ぱっつん前髪）、シルバーのヘアクリップ、白コルセット＋
ラベンダーのリボンなど、参照画像の特徴を正確に拾えています。

### 途中で発生した問題

`LoadImage` が `Invalid image file` で validation エラー。
**原因：** ヘッドレス起動した ComfyUI の input ディレクトリは
`ComfyUI\ComfyUI\input`（7ファイル）で、Desktop 版が使う
`ComfyUI-Shared\input`（161ファイル）とは**別**でした。
**対処：** テスト画像をローカル input にコピー。既存ファイルは変更していません。

---

## Test 2 — 日本語 → H3 6セクション英語プロンプト ✅ 成功（1回目は失敗、修正後に成功）

| 項目 | 実測値 |
|---|---|
| 実行グラフ | 50ノード（10/20/30/40、H3 系はバイパス） |
| **生成時間** | **21.3 秒** |
| 入力（日本語） | シーン指示＋セリフ2行、尺 5.5秒 → **141 フレーム**に自動整列 |
| エラー | なし |

### 1回目の失敗 — R1（セリフ逐語コピー）不履行

6セクション構造・`[reference generation]`・`fully_preserved`・`non_diegetic_music: N/A`
はすべて正しく出ましたが、**`<d>[Japanese] …</d>` タグが1つも出ず**、
"delivering the first line" のように**言い換えられて**しまいました。

**切り分け：** 組み立て後のユーザープロンプトを一時 `ShowText` で実測したところ、
セリフは正しく VLM に渡っていました（結合チェーンは正常）。
つまり配線の問題ではなく、**純粋に指示追従の失敗**。

**修正（3点）：**
1. `R1` に「WRONG / RIGHT」の具体例を追加し、最重要ルールと明記
2. システムプロンプト末尾に **FINAL CHECK チェックリスト**を追加
   （`<d>` タグ数 = 与えたセリフ行数、を数えて検証させる）
3. **セリフブロックをユーザープロンプトの最後尾へ移動**
   （4B級モデルは直近の指示ほど追従しやすいため）＋
   `--- LINES START --- / --- LINES END ---` で明示的に囲む

### 2回目（修正後）の出力（抜粋）

```
detailed_description
The visual style is vibrant and theatrical, ...
[Shot 1] Static Shot. <Subject 1> stands centre stage, facing the camera with an
energetic and welcoming smile. The background is a richly textured wall composed of
stacked, retro brown CRT television sets, softly glowing. ...
<Subject 1> (S1) smiles brightly at the camera and says,
<d>[Japanese] みんな、来てくれてありがとう！</d>
[Shot 2] At 00:03.500, Push In slowly on <Subject 1>'s face, maintaining a medium
close-up composition. ...
<Subject 1> (S1) maintains her smile and says,
<d>[Japanese] 今日は最後まで楽しんでいってね。</d>

non_diegetic_music
N/A
```

**評価：合格。** チェック項目すべて達成。

| 検査項目 | 結果 |
|---|---|
| 6セクションが正しい順序 | ✅ |
| `<Subject 1>` に Character Profile の身体的特徴が入る | ✅ |
| `summary` が `[reference generation]` で開始 | ✅ |
| `retention_analysis` が `fully_preserved` | ✅ |
| **セリフが一字一句 `<d>[Japanese]` 内に入る** | ✅ 2行とも完全一致 |
| `<d>` タグ数 = セリフ行数 | ✅ 2 = 2 |
| 話者ID `(S1)` がタグの外 | ✅ |
| `non_diegetic_music: N/A` | ✅ |
| 縦型・クローズアップ主体（R3） | ✅ |
| カット時刻が昇順（`[Shot 2] At 00:03.500`） | ✅ |

## Test 3 — Ref2VA 新規生成（短尺）

### 試行1 ❌ CUDA out of memory

| 項目 | 実測値 |
|---|---|
| 設定 | 576×1024 / 124フレーム / 8 steps / seed 123456789 / `ref_image_size=max` |
| 結果 | **OOM**（`torch.AcceleratorError: CUDA error: out of memory`） |
| 落ちた箇所 | `Requested to load MiniMaxH3TEModel_` の直後 |
| **GPU0 peak** | 6,576 MiB |
| **GPU1 peak** | **11,247 MiB** ← 原因 |
| **RAM peak** | **46.9 GiB** |
| 経過 | 約 655 秒でエラー |

#### 原因（特定済み）

**llama.cpp（Gemma VLM）が GPU1 に 11.2 GiB を確保したまま解放されず、
その状態で H3 の 25.28 GiB テキストエンコーダを読もうとして落ちていました。**

構造的な問題が2つありました。

1. **実行順序が保証されていなかった。**
   `llama_cpp_unload_model` → `PurgeVRAM V2` は繋いであったものの、
   H3 側は `Any Switch`（プロンプト）に依存するだけで
   **アンロード完了を待つデータ依存が無い**。
   そのため ComfyUI は VLM がまだ VRAM を掴んだままでも H3 のロードを開始できてしまう。

2. **VLM の VRAM 上限が無制限だった。**
   `llama_cpp_model_loader.vram_limit = -1` のため、
   llama.cpp が GPU0/GPU1 の両方にレイヤーを展開していました。

補足：`HostBuffer.read_file_slice failed` も併発。RAM 46.9 GiB とかなり逼迫していました。

#### 修正（3点）

| # | 修正 | 意図 |
|---|---|---|
| 1 | **H3 の `prompt` を `Any Switch → llama_cpp_unload_model → PurgeVRAM V2 → H3` と経由させる** | プロンプトをアンロード経路に通すことで、**VLM 解放が H3 ロードの前提条件になる**（データ依存で順序を強制）。記事WFも同じ構造でした |
| 2 | `vram_limit` を `-1` → **`6`** | llama.cpp が2枚目の GPU まで占有するのを防ぐ |
| 3 | 両 VLM ノードの `force_offload` を `True` | 実行後に確実に退避させる |

修正後は VLM 実行中の GPU1 使用量が **11,247 → 約 2,000 MiB** に低下しました。

### 試行2（順序修正後）❌ 再び OOM、ただし別原因

順序の問題は解消しましたが、今度は**テキストエンコーダ自身が乗り切らず**に落ちました。

| 項目 | 実測値 |
|---|---|
| GPU0 peak | 4,123 MiB（ほぼ空いたまま） |
| **GPU1 peak** | **10,955 MiB** ← ここで OOM |
| 落ちた箇所 | `nodes_minimax_h3.py:277 clip.encode_from_tokens_scheduled(tokens)`<br>→ `comfy/ops.py cast_bias_weight` → `model_management.cast_to_gathered`<br>→ `memory_management.read_tensor_file_slice_into` |

#### ComfyUI のログが示した配置

```
CLIP/text encoder model load device: cuda:1, offload device: cpu, current: cpu
Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged.
Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading.  4965MB Staged.
```

**ComfyUI は自前の動的 VRAM ローディングで、25.9 GB のテキストエンコーダを
cuda:1（3060 / 12GB）に割り当てていました。**
順序修正は効いており VLM は解放済み（GPU1 は VLM 実行中 2,000 MiB まで低下）でしたが、
エンコードの forward 中に重みを gather する作業セットが 12GB を超えて落ちました。

> ⚠️ **当初ここに「ComfyUI 本体が自動で2枚目の GPU に分散している」と書きましたが誤りでした。**
> 実際は `get_torch_device()` が `torch.cuda.current_device()` を返す**単一デバイス方式**で、
> **llama.cpp が現在デバイスを 1 に変更したのを ComfyUI が継承していた**だけです。
> 対照実験：同じ起動引数で VLM を含まないグラフを流すと `VAE load device: cuda:0` になりました。
> つまり **Test 3 の DiT とサンプリングは RTX 3060 側で走っていた**ことになります。
> 詳細は [04_roadmap.md](04_roadmap.md) の §7 を参照。

#### 修正

`CLIPLoader.device` を **`default` → `cpu`** に変更。
同じモデルのまま、実行デバイスだけを CPU に固定します。

- 25.9 GB は 64 GB RAM に収まる
- 4070 Ti を丸ごと DiT 用に空けられる
- エンコードは1プロンプトにつき1回なので、遅くても全体への影響は限定的
- 大容量 GPU に移す場合は `default` に戻す

### 試行3（CPU テキストエンコーダ）

`CLIPLoader.device = "cpu"` により **OOM は解消し、サンプリングまで到達**しました。

| 段階 | 実測 |
|---|---|
| テキストエンコード（CPU） | **約 37 分**（`loaded completely; 25883.83 MB, full load: True`） |
| DiT ロード | `Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged.` |
| サンプリング | **約 55〜66 秒/step**（8 steps ≒ 7.5 分） |
| GPU0 | 1,666 MiB（DiT は cuda:1 側に載った） |
| GPU1 | 最大 9,961 MiB |
| RAM | 38.4 GiB → 増加中 |

> ⚠️ **テキストエンコードが 37 分**は反復作業に耐えません。
> ComfyUI はノード出力をキャッシュするため「同じプロンプトで seed だけ変更」なら
> 再エンコードは走りませんが、**プロンプトを変えるたびに約37分**かかります。
> 12GB VRAM 環境では `int8_convrot`（25.9GB）は実用的でない、というのが実測の結論です。
> 対策候補は [04_roadmap.md](04_roadmap.md) の代替表を参照。

---

## ★重要な発見：Turbo LoRA の片方が「読み込まれていない」

C-3（LoRA と本体の組み合わせ）が**実測で決着**しました。

### 実測ログ

```
total "lora key not loaded" warnings: 518
  500  blocks.*
   16  token_refiner.*
    2  final_layer.*
```

**518 は `minimax_h3_turbo_v4_step600_ema.safetensors` のテンソル総数と完全に一致します。**
つまり **このLoRAは1キーも適用されていません（strength 1.0 が完全に無効）。**
`minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors` は警告ゼロ＝正常に適用。

### 原因（事前の静的解析と一致）

| LoRA | キー命名 | ComfyUI での結果 |
|---|---|---|
| `minimax_h3_turbo_v4_step600_ema`（larryvrh 生） | `blocks.0.attn.qkv_proj.lora_A.weight`（**接頭辞なし**） | ❌ 1つも一致せず |
| `minimax_h3_turbo_4step_ckpt850_pruned_comfyui`（drbaph 変換版） | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight` | ✅ 208モジュール全一致 |

ComfyUI の LoRA キーマッパは **`diffusion_model.` 接頭辞**を前提としています。
drbaph 版のメタデータにある `conversion_type: prefix_conversion_with_adaln_pairs_removed` /
`converted_for: ComfyUI` はまさにこの変換のことでした。

### 意味すること

**記事WFの「Turbo LoRA 2段（1.0 + 0.5）」は、実際には
`ckpt850_pruned_comfyui @ 0.5` の1枚だけで動いていた**ことになります。

### 対処案（未実施・ファイル名がご指定と異なるため）

`drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` に、同じ LoRA の ComfyUI 変換版があります。

```
minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors   620,285,592 B
```

これに差し替えれば v4_step600_ema が実際に効くようになります。
ただし**ご指定のファイル名とは別物**なので、独断では入れ替えていません。

---

### 試行3 の最終結果 ✅ 成功（動画・音声とも生成）

| 項目 | 実測値 |
|---|---|
| **出力ファイル** | `output\H3\20260812\h3_221119_00001_.mp4`（1.05 MB） |
| **解像度** | 576 × 1024（厳密 9:16） |
| **frames / 秒数** | **124 フレーム / 5.17 秒 @ 24fps**（指定どおり） |
| **音声** | **AAC / 32,000 Hz / ステレオ2ch**（H3 の音声VAE仕様と一致） |
| 使用モデル | `minimax_h3_ref2va_pruned_int8_convrot`（DiT 19,995 MB Staged）<br>`qwen3vl_32b_minimax_h3_int8_convrot`（CPU、25,884 MB）<br>video VAE 4,965 MB / audio VAE 576 MB |
| steps / sampler / scheduler | 8 / `res_multistep` / `simple` |
| Turbo LoRA | ⚠️ **下の訂正を参照** |
| SigmaShift | ⚠️ **下の訂正を参照（実際には未適用）** |

> ### ⚠️ 訂正：Test 3 の実際の構成は上表と違いました
>
> 後述の**グループ重なりバグ**（私が作り込んだもの）により、
> 「45 Tail Relay」をバイパスした際に **50 H3 Models の4ノードも巻き込まれていました**。
>
> ```
> #80 LoraLoaderModelOnly                        (2枚目の Turbo LoRA)
> #81 PathchSageAttentionKJ
> #82 MiniMaxH3MemoryEfficientSageAttentionPatch
> #83 MiniMaxH3SigmaShift
> ```
>
> したがって **Test 3 の実際の構成は次のとおり**でした。
>
> | 項目 | 報告した値 | **実際** |
> |---|---|---|
> | Turbo LoRA 1 | @1.0 適用 | raw版のため **518/518 キー bind 失敗＝無効** |
> | Turbo LoRA 2 | @0.5 適用 | **バイパスされ未実行** |
> | SageAttention | 有効 | **バイパス** |
> | H3 省メモリ Attention | 有効 | **バイパス** |
> | SigmaShift | v12 / a6 | **バイパス（モデル既定のまま）** |
>
> つまり **Test 3 は「Turbo LoRA なし・SigmaShift なし・素の DiT・8 steps」で
> あの品質を出していた**ことになります。
> LoRA/Shift の効果は Test 3 の数値からは評価できません。
| seed | 123456789（fixed） |
| **サンプリング時間** | **7分23秒（55.50 秒/step）** |
| テキストエンコード時間 | 約37分（CPU、初回のみ・同一プロンプトならキャッシュ） |
| **GPU0 peak** | **4,107 MiB** / 平均使用率 26.4 % |
| **GPU1 peak** | **10,729 MiB** / 平均使用率 48.7 % |
| **RAM peak** | **62.22 GiB**（64 GB 中）← ほぼ上限 |
| エラー | なし |

#### 人物一致：良好

生成 124 フレームから 6 枚を抽出して確認。

| 参照画像の特徴 | 生成結果 |
|---|---|
| ピンクのレイヤードボブ＋ぱっつん前髪 | ✅ 全フレームで保持 |
| 星型／シルバーのヘアクリップ | ✅ 保持 |
| 白コルセット＋ラベンダーのリボン | ✅ 保持 |
| 白パフスリーブ＋フリルカフス | ✅ 保持 |
| リボンチョーカー | ✅ 保持 |

**6フレーム通して顔の同一性が崩れていません。**
背景（レトロCRTテレビの壁・ステージ照明）もプロンプト指示どおりに再現され、
`[Shot 2] Push In` の指示どおり後半2フレームで寄りになっています。口も動いています。

#### 音声継続性・セリフ一致：完全一致

生成音声を Whisper（medium, ja）で文字起こし：

```
みんな来てくれてありがとう 今日は最後まで楽しんでいってね
```

入力した「②セリフ確定」：

```
みんな、来てくれてありがとう！
今日は最後まで楽しんでいってね。
```

**句読点（、！。）を除いて一字一句一致。2行とも、順序どおり。**

→ **日本語セリフ入力 → VLM → `<d>[Japanese]` → H3 → 実際の日本語音声**
という設計の中核チェーンが、実測でエンドツーエンドに機能することを確認しました。



---

## ★自作バグ：グループの bounding box が重なっていた

### 症状

「45 Tail Relay」グループをバイパスすると、
**隣の「50 H3 Models」に属する4ノードも一緒にバイパス**されていました。

### 原因

グループ 45 と 50 を**同じ x 列**に配置していたため、両者の矩形が y 方向で重なっていました。

```
45 Tail Relay 続き生成   x[2660..3170] y[1490..3380]
50 H3 Models            x[2660..3170] y[-110..2150]
                                        ↑ y 1490..2150 が重複
```

**LiteGraph はグループの所属ノードを「矩形に含まれるか」だけで判定します。**
重なった領域のノードは両方のグループに属し、片方をバイパスすると他方も巻き添えになります。

これはテストハーネス固有の問題ではなく、
**ComfyUI 上で右クリック →「Bypass Group Nodes」でも同じことが起きます。**

### 修正

1. Tail Relay を独立した列（x=5750）へ移動
2. 全カラム間隔を広げ、pad 込みでも接触しないようにした
3. **ビルド時アサーションを追加** — グループ矩形が1つでも重なればビルド失敗

```python
# LiteGraph assigns a node to a group by containment, so two overlapping
# groups silently share nodes and "bypass this group" hits the neighbour.
if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
    raise AssertionError("group bounding boxes overlap: ...")
```

このアサーションは追加直後に**別の重なりを2件検出**しました
（`30 日本語入力`×`40 H3 Director Prompt`、`80 Output`×`85 Upscale`）。
いずれも修正済みで、現在は **overlapping nodes = 0**。

---

## ★llama.cpp のクラッシュ（vram_limit の副作用）

`--cuda-device 0` と `vram_limit: 6` の併用で、ComfyUI プロセスごと**強制終了**しました。

```
ggml-backend.cpp:1367: GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS) failed
Fatal Python error: Aborted
  → llama_cpp/llama.py:622 __init__   (モデルロード中)
```

`vram_limit` を中途半端に絞ると CPU/GPU のレイヤー分割が細かくなり、
llama.cpp の分割入力数上限を超えて `abort()` します。
Python 例外ではないため **try/except では捕捉できず、プロセスが落ちます。**

**対処：`vram_limit` を `-1`（自動）へ戻しました。**
絞った目的は「llama.cpp が GPU1 を掴むのを防ぐ」ことでしたが、
`--cuda-device 0` で GPU1 自体が見えなくなるため不要になりました。

---

## Test 3B — 修正後の比較実行 ✅

Test 3 と**同じプロンプト・seed・解像度・尺・steps**で、次の3点だけ変えて再実行。

| 変更点 | Test 3 | Test 3B |
|---|---|---|
| GPU 配置 | cuda:1（RTX 3060）※llama.cpp が現在デバイスを移動 | **cuda:0（RTX 4070 Ti）** `--cuda-device 0` |
| Turbo LoRA | 実質なし（1枚は bind 失敗、1枚はバイパス） | **2枚とも適用**（`208 patches attached`） |
| テキストエンコーダ | int8_convrot / CPU | **nvfp4_awq / cuda:0** |

### 実測比較

| 指標 | Test 3 | **Test 3B** | 差 |
|---|---:|---:|---|
| **sec / step** | 55.50 | **13.93** | **約 4.0 倍速** |
| **サンプリング合計** | 7分23秒 | **1分51秒** | |
| **テキストエンコード** | 約37分 | **46秒**（別計測） | **約48倍速** |
| **総実行時間** | 約45分 | **230.90 秒（3分51秒）** | **約11倍速** |
| GPU0 peak | 4,107 MiB | **11,333 MiB** | 4070Ti を使い切る |
| GPU1 peak | 10,729 MiB | **0 MiB** | 3060 は未使用 |
| GPU0 平均使用率 | 26.4 % | **43.7 %** | |
| RAM peak | 62.22 GiB | 61.21 GiB | ほぼ同じ |
| lora key not loaded | **518** | **0** | |
| DiT patches attached | **0** | **208** | LoRA が実際に効いている証拠 |

### 出力

`h3_225706_00001_.mp4` — 576×1024 / 124f / 5.17s / AAC 32kHz ステレオ（Test 3 と同仕様）

**音声：** Whisper 文字起こしで**2行とも完全一致**。
今回はセグメントも正しく2分割されました。

```
[ 0.00- 2.50] みんな来てくれてありがとう!
[ 2.50- 5.00] 今日は最後まで楽しんでいってね!
```

### 映像品質の所見（主観・要注意）

| 観点 | 所見 |
|---|---|
| 動きと表情 | **Test 3B のほうが明確に良い。** ウインク、ピースサイン、口の開閉など変化が豊か |
| Push In | 3B のほうが強く効いており、後半は明確な寄り |
| 質感 | 3B のほうが肌・髪のディテールが多い |
| **衣装の忠実度** | ⚠️ **3B のほうが参照画像から離れている**（星モチーフが増え、ボディスの形状が変化） |
| 顔の同一性 | どちらも保持。崩れなし |

> ⚠️ **この品質差の原因は特定できません。**
> GPU・LoRA・エンコーダの**3変数を同時に変更**しているため、
> 衣装の変化が Turbo LoRA によるものか、NVFP4 エンコーダの数値差か、
> あるいは単なるサンプリングの揺らぎかは、この比較からは切り分けられません。
> 切り分けるには1変数ずつの追加実験が必要です。

### NVFP4 単体の検証結果（Ada で動作するか）

| 項目 | int8_convrot (CPU) | **nvfp4_awq (cuda:0)** |
|---|---:|---:|
| Staged サイズ | 25,882 MB | **14,956 MB** |
| エンコード時間 | 約 37 分 | **46 秒** |
| GPU0 peak | 4,107 MiB | 11,351 MiB |
| RAM peak | 62.22 GiB | **33.11 GiB** |
| conditioning 生成 | 成功 | **成功**（`{'samples': <NestedTensor>}`） |
| エラー | なし | **なし** |

**NVFP4 は RTX 4070 Ti（Ada）でロード・完走しました。**
ログ上も `Detected mixed precision quantization` /
`Using MixedPrecisionOps for text encoder` と正しく認識されています。

> ただし **「動く」ことと「数値的に等価」であることは別**です。
> Ada にネイティブの NVFP4 演算器は無いため、
> ソフトウェアでの逆量子化が行われている可能性があります。
> 上記の衣装忠実度の差がこれに起因する可能性は否定できません。
> **正式採用の可否は、1変数ずつの比較を経てからご判断ください。**

## Test 4 — Continuation（Tail Relay）

| 項目 | 結果 |
|---|---|
| 状態 | 未実施 |

## Test 5 — ref_video_audio 併用

| 項目 | 結果 |
|---|---|
| 状態 | 未実施 |

---

## 記録項目テンプレート（各生成で埋める）

```
使用モデル      :
解像度          :
frames / 秒数   :
steps / sampler / scheduler :
Turbo LoRA と strength      :
SigmaShift      :
seed            :
生成時間        :
GPU0 peak VRAM  :
GPU1 peak VRAM  :
RAM peak        :
エラー          :
人物一致        :
前クリップ接続感:
音声継続性      :
```
