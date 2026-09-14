# H3 高速化検証 2026-09-07

対象: LONG_FAST / Ref2VA Turbo4 / 480x864 / seed 123456789 / 同一 prompt・reference
ハード: RTX 4070 Ti 12GB (GPU0, sm89) + RTX 3060 12GB (GPU1) / Windows / CUDA 13.0 / torch 2.10.0+cu130
ComfyUI: v0.34.5 (7fd919f0) / comfy-aimdo 0.4.15 / comfy-kitchen 0.2.31

証拠一式: `docs/optimization_2026-09-07/`

---

## 1. 今日試した候補

| # | 候補 | 手段 | 結果 |
|---|---|---|---|
| A/C/D | EasyCache / LazyCache（コア実装） | 124f 実測 | NO_GAIN |
| A' | ComfyUI_H3FBC | 静的 | 棄却 |
| A'' | BMB12d3/ComfyUI-H3-Ref2VA-Accelerator v0.4.2 | 導入 + 124f 実測 | FAIL |
| B | SLA / block-sparse (NABLA) | 静的 | 棄却 |
| C' | TeaCache (WanVideoTeaCacheKJ) | 静的 | 棄却 |
| E | FastH3 / VSA | 未着手 | 次回 |
| F | upstream ComfyUI master core | 隔離 worktree + 124f 実測 | NO_GAIN |
| G | 最新 Ref2VA Turbo / LoRA | 未着手 | 次回 |
| H | Comfy Kitchen INT8 attention | 124f 実測 | NO_GAIN |
| H' | Flash Attention | 静的 | 棄却 |
| I | TorchCompileVAE | 静的 | 棄却 |
| I' | 非compile系 VAE 高速化 | 未着手 | 次回 |
| J | `--fast` (fp16_accumulation/fp8_matrix_mult/cublas_ops/autotune) | 124f + 362f 実測 | **採用** |
| J' | `--disable-pinned-memory` | 124f + 362f 実測 | LOW_MEMORY 専用 |
| J'' | `--async-offload 4` / `--disable-smart-memory` / `--fast-disk` / `cublas_ops` 単独 | 124f 実測 | NO_GAIN |
| — | `v = v.clone()` 除去 | model.py 改変 + 124f | NO_GAIN |

---

## 2. 採用

### `--fast` （LONG_FAST / FAST 本番）

362f x2、pinned ON 固定、A→B→A→B 交互ペア n=2:

```
                    wall     sampling   cond    VAE
A1 fast OFF        475.9     258.4     47.8   121.8
B1 fast ON         457.4     258.2     52.8    95.0
A2 fast OFF        475.9     255.8     50.0   121.6
B2 fast ON         450.8     252.5     52.5    94.6

mean  A: wall 475.9  samp 257.1  vae 121.7
      B: wall 454.1  samp 255.4  vae  94.8
```

- wall: **475.9 -> 454.1s (-4.6%)**
- VAE: **121.7 -> 94.8s (-22.1%)**、変動も +-39% -> +-0.4% に安定化
- sampling: 257.1 -> 255.4s（実質同等、寄与なし）
- conditioning: 47.8/50.0 -> 52.8/52.5（+2.8s のコスト）

VAE 短縮は `--fast` 由来と確定（pinned ON のままでも 94.6-95.0 に収束）。

---

## 3. 棄却（再試験不要）

| 候補 | 判定 | 根拠 |
|---|---|---|
| `v = v.clone()` 除去 | NO_GAIN | samp 77.4 -> 80.8、VRAM 10,979 -> 10,983MB（+4MB） |
| Comfy Kitchen INT8 attention | NO_GAIN | Sage 77.4 vs CK 79.0。`Using Comfy Kitchen attention` でログ確認済み |
| EasyCache / LazyCache | NO_GAIN | cache hit 0。`cumulative_change_rate 0.789` vs `reuse_threshold 0.2`。4-step 蒸留では機構的に成立しない |
| Ref2VA Accelerator v0.4.2 | FAIL | Balanced / Observe とも `finish_full_step: full-step state is incomplete`。README 推奨接続順で配置済み。検証済 envelope は 20-step RES Multistep |
| ComfyUI_H3FBC | 静的棄却 | 4-step Turbo では warmup_steps=3 + dense_last_step=True で全step が埋まる（README 記載） |
| SLA / NABLA | 静的棄却 | コアに SLA 実装なし。NABLA は "currently only works with Kadinsky5" |
| Flash Attention | 静的棄却 | `FLASH_ATTENTION_IS_AVAILABLE = False` |
| TorchCompile 系 | 静的棄却 | dynamic VRAM の DLPack 変換と競合（`08_known_issues.md` ISSUE-1） |
| `--async-offload 4` | NO_GAIN | samp 77.5 / total 128.8 |
| `--disable-smart-memory` | NO_GAIN | samp 74.1 |
| `cublas_ops` 単独 | NO_GAIN | samp 80.0 |
| upstream master core | NO_GAIN | 交互ペア n=4: 61.1 -> 60.1（-1.6%、paired sd 1.0） |

---

## 4. 実測値

### 124f / 480x864（短尺）

交互ペア n=4、両群 NOPIN + `--fast`:

```
0.34.5 : samp mean 61.1 sd 1.1  [59.9, 60.1, 61.9, 62.5]
master : samp mean 60.1 sd 0.4  [59.4, 60.3, 60.4, 60.5]
paired diff -1.0s sd 1.0
```

production(pin ON/fast OFF) vs stack(NOPIN/fast ON)、交互ペア n=4:

```
samp  75.2 -> 71.5  (-5.0%)   paired -3.8s sd 2.1
cond  15.8 -> 17.1  (+8.5%)   paired +1.3s sd 0.2
vae   29.7 -> 23.6 (-20.4%)   paired -6.1s sd 1.8
total 126.9 -> 119.5 (-5.8%)  paired -7.4s sd 4.1
```

### 362f x2 / 480x864（長尺 = 30秒相当）

```
                 wall           sampling        VAE
prod (pin ON)    441.7 / 514.2  250.2 / 262.8  120.5 / 167.5
stack (NOPIN)    480.3 / 785.6  306.3 / 533.0   92.8 /  95.4
```

FAST-only 切り分けは §2 参照。

---

## 5. 品質結果

`--fast` 362f 出力 vs `--fast` なし 362f 出力（同一 seed / prompt / reference）:

- identity: 同等
- 顔・肌ディテール: 同等
- 手: 指の破綻なし、両者同等
- 微細テクスチャ（コルセット十字・リボン・銀バー・髪クリップ）: 同等
- 背景: 同等
- streams: 両者 h264 480x864 / 362f / 15.083s + AAC
- 音声: -19.5dB vs -20.0dB（差 0.5dB）
- NaN / 黒フレーム / アーティファクト: なし

**判定: PASS**

**未確認**: リップシンクは静止画では判定不能。動画試聴による人間確認が必要（human_review_required）。

証拠: `optimization_2026-09-07/compare_FASTONLY.jpg`, `compare_FASTONLY_face.jpg`

---

## 6. 測定ミスと原因

今日の検証では、測定手法に起因する誤判定を複数回出した。記録として残す。

### ミス1: 冷えた1本目と温まった後続の比較

sweep 形式で構成を1本ずつ変えて連続実行したため、各 sweep の1本目（冷）と後続（温）を横断比較していた。

- `master core` を **-53.7%** と報告 -> 再測で -19% -> 交互ペアで **-1.6%（NO_GAIN）**
- 同一構成の 0.34.5 baseline が 62.3 / 74.1 / 76.2 と ±11秒 振れていた
- 交互・連続実行にすると sd 1.1秒 まで収束した

### ミス2: run 間の状態汚染（長尺）

362f では前の run が残した状態が次の run を汚染する。

- VAE 167.5秒 を記録した `A_prod_2` は **NOPIN run の直後**
- クリーンな状態から始めた A1/A2 では 121.8 / 121.6 に収まった
- A1/A2 の wall は 475.9 / 475.9 と完全一致

### ミス3: n=1 での判定

序盤、単一 run の結果で判定・報告した。この環境のばらつきを ±2秒 と誤って見積もったことが原因。実際は条件次第で ±11秒 以上。

### ミス4: sweep driver のポート解放待ち欠落

前 run の ComfyUI がポートを掴んだまま次 run が起動して失敗し、集計が前の record を読み直していた（`cublas_ops` / `fast_disk` の初回）。ガード追加後に再測。

### ミス5: メモリ圧の誤読

`--disable-pinned-memory` を「メモリ圧を下げる = 良い」と解釈した。実測では pinned ON のほうが shared 39.9GB / RAM空き 3.1GB とメモリ圧が高いにもかかわらず速い。**圧の低さと速度は無関係**。

---

## 7. 新しいベンチ測定ルール

ベンチマーク時のみ必須。通常の本番生成では不要。

1. 各 run 開始前に PRERUN snapshot を取る
   - shared GPU memory / dedicated VRAM / system RAM free / commit / pagefile
   - GPU temperature / clock / power
2. 冷えた1本目と温まった後続を直接比較しない
3. 可能な限り **A -> B -> A -> B の交互ペア**で測定する
4. n=1 で判定しない。差がノイズ幅（短尺 sd ~1、長尺は状態依存）を超えることを確認する
5. sweep driver は次 run 前にポート解放を待ち、record が増えない場合は FAILED と明記する
6. 長尺判定では「構成差より実行時メモリ状態差の方が大きくないか」を必ず確認する

---

## 8. production 最終フラグ

```
LONG_FAST / FAST 本番:
    --reserve-vram 1.0
    --fast
    （pinned memory は既定のまま ON）

LOW_MEMORY / OOM 回避時のみ:
    --disable-pinned-memory
```

`--disable-pinned-memory` の分類:
**Performance optimization ではなく Memory-pressure fallback**。
LONG_FAST では sampling が 250-263s -> 306-533s に悪化するため速度目的では使用しない。

---

## 9. rollback 状態

| 対象 | 状態 |
|---|---|
| ComfyUI production tree | v0.34.5 (7fd919f0)、git status クリーン |
| `comfy/ldm/minimax/model.py` | `v = v.clone()` 復元済み、SHA256 一致 |
| comfy-aimdo | 0.4.15（変更なし） |
| comfy-kitchen | 0.2.31（変更なし） |
| `backend/h3_v2/ab_run.py` | 実験フック撤去、SHA256 `a2739d70...` 一致 |
| ComfyUI-H3-Ref2VA-Accelerator | 削除済み |
| 隔離 master worktree | 削除（diff は `optimization_2026-09-07/upstream_master_h3_diff.patch` に保存） |
| 隔離 pylibs (197MB) | 削除 |

### MP4 整理

```
BEFORE  : 220 files / 0.305 GB
DELETED :  54 files / 63.9 MB
AFTER   : 167 files / 0.245 GB
```

削除対象: `ComfyUI/input/` の relay 重複30件（元 output の存在を1件ずつ照合）+ scratchpad 退避24件。
BROKEN_REFERENCES: なし。
NO_GAIN / FAIL / rejected experiment の output MP4 は
`REJECTED_EXPERIMENTS（証拠つき、losing run も保存）` の方針に従い保持。

---

## 10. 次回候補

長尺から始めない。**静的調査 -> 短尺 124f** の順で進める。

優先順位:

1. **FastH3 / VSA の Ref2VA 対応状況**
   upstream master の `Attention.__init__` に `gate_compress` と
   `to_gate_compress` Linear が追加されており（"VSA gate, unused by the dense
   forward; consumed by sparse attention patches"）、`DiTBlock.forward` にも
   `attention=None` の差し替えフックが入っている。VSA 受け入れの下地は
   upstream 側に存在する。
2. **最新 Ref2VA Turbo / LoRA / distilled checkpoint**
   現在ディスク上は 4本（ref2v_turbo_4step_v0.1 / turbo_4step_ckpt850 /
   turbo_v4_step600_ema / 同 pruned）。
3. **非compile 系 VAE 高速化**
   `--fast` 適用後も VAE は 94.8s（362f x2）で総時間の約21%。
   `VAEDecodeTiled` 等が候補。
4. **H3 専用 kernel / attention の新実装**

各候補は生成前に以下を静的確認する:

- Ref2VA 対応
- Turbo4 対応
- Character Master 維持
- Voice Master 維持
- Sage 併用可否
- RTX 4070 Ti / Ada sm89 / Windows / CUDA13 対応

---

## 参考: 未解決の観測

- upstream master には PDD head bank（`_pdd_head`）、FunControl ノード
  （`MiniMaxH3FunControlNetApply`）が追加されている。速度ではなく機能。
  ただし master の H3 経路は `comfy_aimdo.malloc_graph` を要求し、
  0.4.15 には存在しない（0.5.2 で追加）。production venv 共有のため
  アップグレードは production に影響する。
- conditioning は 15.8-21.5s の範囲で振れる。32B text encoder
  （25,882MB staged）の再ステージが疑わしいが未特定。
