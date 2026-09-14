# H3 ハード制約リファレンス

本設計の数値はすべて、インストール済み ComfyUI 本体のソース
`<COMFYUI_DIR>\comfy_extras\nodes_minimax_h3.py`
を直接読んで確認したものです（推測ではありません）。実装時はこの表を仕様の根拠として使ってください。

---

## 1. 定数

```python
CANVAS_MULTIPLE      = 32
BASE_SHORT_EDGE      = 768
MAX_PIXELS           = 768 * 1344   # = 1,032,192
REF_IMAGE_SHORT_EDGE = 2048
FPS                  = 24           # 固定。変更不可
AUDIO_LATENT_FPS     = 40
```

- **fps は 24 固定。** `CreateVideo` の fps も必ず 24 にすること。30fps / 60fps は存在しない。
- 出力解像度は幅・高さとも **32 の倍数**。
- 実用上の面積上限は **1,032,192 px**。

---

## 2. フレーム数のグリッド（最重要）

```python
def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n
```

**フレーム数は必ず `n ≡ 5 (mod 17)`** に切り上げられる。
`MiniMaxH3ReferenceToVideo` の `length` は `min=5, max=3600, step=17`、**default=124**。
tooltip 記載の学習済みレンジは **124〜362 フレーム**。

### 使用可能な尺の一覧（24fps）

| frames | 秒 | 備考 |
|---:|---:|---|
| 124 | 5.17 | 学習レンジ下限 / ノード既定値 |
| 141 | 5.88 | |
| 158 | 6.58 | |
| 175 | 7.29 | |
| 192 | 8.00 | ちょうど 8 秒 |
| 209 | 8.71 | |
| 226 | 9.42 | |
| 243 | 10.13 | |
| 260 | 10.83 | |
| 277 | 11.54 | |
| 294 | 12.25 | 記事ワークフローの採用値 |
| 311 | 12.96 | |
| 328 | 13.67 | |
| 345 | 14.38 | |
| **362** | **15.08** | tooltip 記載レンジの上端 |

**→ 1クリップは 5〜15 秒程度を基本に考えるのが無難です。**

ノードの `length` は `min=5, max=3600` なので、**362 より大きい値も入力自体は可能**です。
tooltip に "trained range is ~124-362" とあることから、その外は未検証領域と考えて
ワークフローの既定ではこの範囲にクランプしていますが、**絶対的な上限ではありません。**
長尺を試す場合は `フレーム数` ノードの式の `min(362, max(124, ...))` を書き換えてください。

30〜60秒のショートを作る場合、現実的には複数クリップの連結になります。

### 尺計算式（ワークフローに実装する式）

```
frames = max(124, min(362, ceil(fps_target_sec * 24)))
while frames % 17 != 5: frames += 1
```

記事版の `MathExpression` 式もこれと等価：
`max(5, round(b*a)) + (5 - (max(5, round(b*a)) % 17)) % 17`

---

## 3. Latent 形状

```python
def video_latent_t(fc):  return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2
video latent : [B, 24, latent_t, height//16, width//16]
audio latent : [B, 32, 2, round(duration * 40)]
```

362 フレーム / 768×1344 の場合 → video `[1, 24, 107, 48, 84]` / audio `[1, 32, 2, 603]`

---

## 4. 縦型（9:16）キャンバスの選び方

`MAX_PIXELS = 1,032,192` と「32の倍数」の制約から、9:16 系で実用的なのは次の通り。

| 解像度 | 面積 | 実アスペクト | 用途 |
|---|---:|---|---|
| **768 × 1344** | 1,032,192（上限ちょうど） | 4:7 = 0.5714 | **本番**。最大の素解像度。9:16(0.5625) とは 1.6% ずれるので仕上げで微クロップ |
| 736 × 1312 | 965,632 | 0.5610 | 本番の軽量版。9:16 に近い |
| **576 × 1024** | 589,824 | **0.5625（厳密に 9:16）** | **下書き**。約 43% 軽く、アスペクト誤差ゼロ |
| 288 × 512 | 147,456 | 0.5625 | 構図・セリフ確認用の超高速プレビュー |

> 参考：`adapt_canvas()`（参照動画のリサイズ用）に 9:16 を通すと 768×1344 が返る。
> これが H3 にとっての「縦型の素の最大値」。

---

## 5. 参照入力の仕様（`MiniMaxH3ReferenceToVideo`）

| 入力 | 型 | 最大数 | 内容 |
|---|---|---:|---|
| `ref_images.ref_image_N` | IMAGE | **9** | `<Picture 1..N>` / `<Subject>` の視覚ソース |
| `ref_videos.ref_video_N` | IMAGE（フレーム束） | **3** | `<Video 1..N>`。24fps、2〜15秒 |
| `ref_video_audios.ref_video_audio_N` | AUDIO | **3** | 同番号の ref_video のサウンドトラック |
| `ref_audios.ref_audio_N` | AUDIO | **3** | 単独の `<Audio N>`（声質参照・BGM参照など） |
| `ref_image_size` | Combo | — | `match` / `max` |

> **記事ワークフローは `ref_video_*` を一切使っていません。** ここが「続きの動画」の実装ポイント。

### 5.1 `ref_image_size` の実際の挙動（重要な落とし穴）

```python
if ref_image_size == "match":
    scale = min(1.0, sqrt((width*height) / (w*h)))    # 生成面積に合わせて「縮小のみ」
else:  # "max"
    scale = min(1.0, 2048 / min(w, h))                # 短辺 2048 まで「縮小のみ」
```

両方とも `min(1.0, …)` なので **拡大は絶対に起きない**。

**したがって、参照画像が 1024px 程度の場合 `max` と `match` はほぼ同じ結果になります。**
記事ワークフローは参照画像を 1024 kilopixel に縮めてから `max` を指定しているため、
この設定は実質的に効いていないと考えられます。

`max` の恩恵（同一性の忠実度）を引き出したいなら、**参照画像は高解像度なほど有利**です。
ただし「短辺 2048px 必須」という意味ではありません。下の 5.2 の通りコストとのトレードオフなので、
主役だけ厚く配分するのが現実的です。低解像度の参照でも生成自体は問題なく動きます。

### 5.2 参照トークンのコスト

参照画像は latent 化され `(h/16) × (w/16)` トークンとして**毎サンプリングステップに乗る**
（ノードの tooltip にも "can be several times slower" と明記）。

| 参照画像サイズ | latent トークン数 | 相対コスト |
|---|---:|---:|
| 768 × 768 | 2,304 | ×1 |
| 1024 × 1024 | 4,096 | ×1.8 |
| 1536 × 1536 | 9,216 | ×4 |
| 2048 × 2048 | 16,384 | ×7.1 |

**Phase 1 の既定配分：** 顔寄り 1536 / 上半身 1152 / 全身 1152 / 衣装 896（いずれも長辺 px）、
`ref_image_size = max`。合計 18k トークン程度。

これは目安であり実測値ではありません。12GB VRAM で重い場合は下げ、
同一性が足りない場合は顔寄りだけ上げる、という調整を想定しています。

### 5.3 参照動画の仕様（続き生成の要）

```python
frames = resize(video_frames, cw, ch)         # adapt_canvas でキャンバス決定
if frames.shape[0] > frame_count: 切り詰め
if n < 5: エラー ("need at least 5 frames")
while n % 17 != 5: n -= 1                     # ← 切り下げ
# Qwen には 2fps にダウンサンプルして timestamps 付きで見せる
```

**参照動画のフレーム数も `n ≡ 5 (mod 17)` に切り下げられる。**
そのため、あらかじめグリッドに乗る長さで切り出すのが安全。

| テール長 | 秒数 | 用途 |
|---:|---:|---|
| 39 | 1.63 | 最小限のつなぎ。ドリフトが最も少ない |
| **56** | **2.33** | **推奨**。動きの流れを保ちつつ軽い |
| 73 | 3.04 | 会話の途中でつなぐ場合 |
| 90 | 3.75 | 長回し感を維持したい場合 |

### 5.4 参照の提示順序（プロンプトの番号と一致させる）

ソースのコメントより：

> References enter the presentation in fixed order: **images, then videos**
> (each soundtrack's `<Audio j>` label right before its `<Video k>`), **then standalone audio**.
> Ordinals are 1-based per type.

つまり `<Picture i>` / `<Video k>` / `<Audio j>` の番号は **入力スロットの順番**で決まる。
プロンプト内のラベル番号は必ずこれに一致させること。

> ⚠️ `ref_video_audio_0` を接続すると、その `<Audio j>` ラベルは **`<Video k>` の直前**に発行される。
> 単独 `ref_audio_*` を併用する場合、`<Audio>` の通し番号がずれるので注意。

---

## 6. その他

- **音声 VAE のサンプルレート：** 32000 Hz（`getattr(audio_vae, "audio_sample_rate", 32000)`）。
  異なる SR の音声は自動リサンプルされる。
- **`MiniMaxH3SigmaShift` の既定値：** `shift_video=12.0`, `shift_audio=3.0`。
  記事ワークフローは `(12, 6)` を採用（音声側を強めに振っている）。
- 本体に同梱される H3 系ノードは 4 つのみ：
  `EmptyMiniMaxH3LatentAV` / `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` / `MiniMaxH3SigmaShift`。
  記事で使われている `MiniMaxH3MemoryEfficientSageAttentionPatch` と `SpectrumApplyMiniMaxH3` は
  **本体には存在せず、出所不明の外部パック**（[06_missing_data.md](06_missing_data.md) 参照）。
- `MiniMaxH3ImageToVideo` は t2va / fl2va 用（first_frame / last_frame）。
  インストール済みの `minimax_h3_fl2va_*` はこちら向けで、**Ref2VA とは別モデル**。
