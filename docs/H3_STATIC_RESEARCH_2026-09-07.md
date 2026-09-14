# H3 静的調査レポート 2026-09-07（GPU不使用）

対象: MiniMax H3 / Ref2VA / LONG_FAST（480x864, 362f x2 = 30秒）
現行基準: `minimax_h3_ref2va_pruned_int8_convrot.safetensors` + Ref2V Turbo4 v0.1
環境: RTX 4070 Ti 12GB (sm89) + RTX 3060 12GB / Windows 11 / CUDA 13.0
      torch 2.10.0+cu130 / **triton 3.7.1（TMA descriptor 利用可）** / Python 3.13.12
      ComfyUI v0.34.5 (7fd919f0) / comfy-aimdo 0.4.15 / comfy-kitchen 0.2.31
本日の生成・推論・モデルDL: **なし**（静的調査のみ）

---

## STATIC_RESEARCH_SUMMARY

現行の 362f x2（`--fast` 適用時 wall 454.1s / bench）の内訳は
sampling 255.4 / VAE 94.8 / conditioning 52.5 / その他 51.4 秒。
本日の調査で、**sampling 以外に削れる wall time が実測ベースで 130〜240 秒分ある**
ことが確認できた。最大の発見は 2 点。

1. **int8_convrot VAE が現行 ComfyUI v0.34.5 でそのまま動く**（PR #15334 が
   2026-08-06 に merge 済み、`comfy/sd.py:995-999` に検出コードを実地確認）。
   VAE ファイルを差し替えるだけで、コード変更もノード変更も不要。
2. **アプリ本体に構造的な model thrash がある**。director graph が
   毎セグメント `PurgeVRAM V2(purge_models=True)` を実行し、その直後に
   H3 stack（DiT 20.0GB + TE 25.9GB）を再ステージしている。
   bench の warm クリップと比べて全クリップが cold 相当になる。

FastH3 / VSA は **Ref2VA 非対応で確定**。STATIC_REJECT。

---

## TOP_CANDIDATES（期待値順）

### 1. int8_convrot Video VAE への差し替え  ★最優先

| 項目 | 内容 |
|---|---|
| 入手元 | `Kijai/MiniMax-H3-experimental` / `minimax_h3_video_vae_int8_convrot.safetensors` |
| サイズ | 3.17 GB（現行 fp16 は 4.9GB → **VRAM 約 1.9GB 削減**） |
| 最終更新 | ComfyUI 側 PR #15334 merge 2026-08-06 |
| Ref2VA | 影響なし（VAE は DiT と独立） |
| Turbo4 | 影響なし |
| Character/Voice Master | 影響なし（identity は DiT・音声は Audio VAE 側） |
| Sage 併用 | 可（別経路） |
| Windows / Ada sm89 | 可。ただし後述リスクあり |
| 12GB 現実性 | 現行より **有利**（VRAM が減る） |
| 追加 Python 依存 | **なし** |
| production 侵襲性 | **最小**。`app/config.json` の `vae_video` 1行 |
| 公開ベンチ | 作者 kijai「約1.5倍」/ 第三者 rzgar「19.5s → 10.0s = 1.95倍」 |
| ベンチ条件 | rzgar: RTX 5090 / torch 2.12.0+cu130 / 5秒 1344x768 |
| 現構成との差 | fp16 VAE → int8+ConvRot 量子化 decoder |
| **期待高速化** | VAE 94.8s → **55〜63s**（**−32〜40s / 全体 −7〜9%**） |
| 品質リスク | 低。実測 rel RMS 0.97%（報告者いわく不可視） |

**リスク（必ず記録）**: 同 discussion で「Windows では INT8 が fp16 と同等〜遅い」
という報告があり、Linux へ移行して解決している。当環境は Windows なので
**効果ゼロまたは逆効果の可能性が実在する**。これが 124f で最初に測るべき理由。

---

### 2. アプリの director / generate 間 model thrash 除去  ★コード変更のみ

`app/h3app/graphs.py:236-243` の director graph が毎セグメント:

```
llama_cpp_unload_model  →  LayerUtility: PurgeVRAM V2 (purge_models=True)
```

を実行し、直後に generate graph が H3 stack を再ロードする。

bench 実測に現れる cold / warm 差（`benchmarks/v2_bench.jsonl`）:

```
lf124x6 (512x896):  clip a samp 82.8  →  clip b-f samp 37.1〜38.7   （cold ペナルティ 約45s）
lf243x3 (512x896):  clip a samp 117.8 →  clip b,c  95.1〜96.0        （約22s）
lf362   (480x864):  clip a samp 135〜141 → clip b 117〜122           （約20s）
```

アプリでは PurgeVRAM のため **全クリップが cold 側**になる。

さらに app 実測（`app/story_projects/bdc87288ec92`, 480x864/362f x2, LONG_FAST）:

```
gen_seconds  clip0 332.6  +  clip1 341.5  =  674.1s
同条件 bench（--fast なし）                =  475.9s
差                                          = 198.2s（99.1s/clip）
```

うち約 20s/clip は上記 cold ペナルティで説明がつくが、**残り約 79s/clip は未計測**。
再ステージ実測が次回 GPU テストの最優先項目。

**改修案**（今日は実装しない）:
- director を全セグメント分まとめて先に実行 → VLM 常駐 1 回、以後 H3 stack 常駐。
  director の依存は前セグメントの **テキスト**（`en_prompt` の summary）だけで、
  生成済みピクセルには依存しない（`story_runner.py:500-513` で確認）。
  ピクセル依存があるのは generate 側の `last_frame` のみ。
- 期待効果: N=2 で **−40〜198s**、N=6 なら更に大きい。

---

### 3. ComfyUI-sol-attn（Sol-Attn / H3 専用ノード）

| 項目 | 内容 |
|---|---|
| repo | `Saganaki22/ComfyUI-sol-attn` |
| 最終更新 | **2026-08-13**（v0.6.2「add SM86 support」） |
| Ref2VA | **対応と判断（静的確認済み）** — 下記根拠 |
| Turbo4 | 影響なし（attention 経路のみ） |
| Character/Voice Master | `sink_conditioning="exact_kv"` が text/**reference**/**audio** の KV を exact 保持 |
| Sage 併用 | **併用が公式推奨**。KJ Sage → H3 Sage → H3 Sol の順 |
| Windows | 可（作者環境が Windows 11） |
| Ada sm89 | pointer forward kernel 対応。**community-validated / 未ベンチ** |
| 12GB 現実性 | 有利（strided view で q/k/v コピー削減。32K tokens で 1,344MiB 削減） |
| 必要モデル | **なし** |
| 追加 Python 依存 | **なし**（triton 3.6+ が要件、当環境は **3.7.1** で充足） |
| production 侵襲性 | 中（custom_nodes への git clone、graph に 1〜2 ノード追加） |
| 公開ベンチ | attention 単体で vs Sage **1.38〜1.65x**(bf16) / full-model **9.91 → 8.92 s/it (−10%)** |
| ベンチ条件 | RTX 5090 (SM120) / B=1 H=56 D=128 bf16 / 合成テンソル。**full-model は 5090** |
| **期待高速化** | sampling 255.4s → **230〜255s（0〜−10%）** |
| 品質リスク | 中。近似 attention。bf16 exact モードで SDPA との rel L2 0.00097 |

**Ref2VA 対応の静的根拠**（実地確認）:
- `minimax.py:_make_span_injector` が
  `PackedLayout(*signature, keyframes=..., refs=payload.get("refs"), frame_count=...)`
  を呼ぶ。当環境の `comfy/ldm/minimax/model.py:321` の
  `PackedLayout.__init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None)`
  と**シグネチャが一致**し、`refs`（= Ref2VA の参照トークン）を明示的に扱う。
- patch 対象は `diffusion_model.blocks.{i}.attn.forward` で、判定は
  `diffusion_model.__class__.__name__ != "MiniMaxH3Model"`。Ref2VA も同一クラス。
- 既存の Sage patch を fallback として**引き継ぐ**実装になっている
  （`prior = object_patches.get(...attn.forward)`）。

**最大の懸念（要検証）**: token 数。

```
480x864 / 362f → latent 107 x (30x54) = 173,340 tokens
480x864 / 124f → latent  37 x (30x54) =  59,940 tokens
```

Sol-Attn の検証済み上限は **103,237 tokens**。しかも BENCHMARKS.md は
「PyTorch SDPA 自体が H3 の strided qkv view で 100K tokens 超に int32 stride
オーバーフローを起こす」と明記している。
→ **124f（59,940 tokens）で先に試すべき**。362f をいきなり試さない。

---

### 4. W4A8 Ref2VA チェックポイント

| 項目 | 内容 |
|---|---|
| ファイル | `Kijai/MiniMax-H3-experimental` / `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` |
| サイズ | **11.8 GB**（現行 int8_convrot 21GB の **約56%**） |
| ComfyUI 対応 | **有**。`comfy/quant_ops.py:276` に `QUANT_ALGOS["asym_w4a8_int8"]` が無条件登録、`AsymW4A8Int8Layout` も `__all__` に存在 |
| Ref2VA | ファイル名どおり Ref2VA 専用ビルド |
| 12GB 現実性 | **大幅に有利**。DiT が 12GB カードにほぼ収まる |
| production 侵襲性 | 小（config の `model` 1行）だが**モデル本体の変更** |
| 期待効果 | staging / offload トラフィック削減。app 側の再ロードコストに直撃 |
| **未確認（必ず静的確認）** | **Turbo4 LoRA が bind するか**。現行は 624 keys bind。W4A8 量子化重みへの LoRA merge が同じキー名で通るか未検証 |
| 品質リスク | **高**。4bit 重み。identity 保持は要目視確認 |

---

### 5. GPU1 パイプライン decode（既存コード・未配線）

`backend/h3_v2/ab_run.py:452-520` に **実装済みかつ実測済み**。

- Server A (GPU0): conditioning + sampling + anchor + raw latent 保存（full decode なし）
- Server B (GPU1/3060): raw latent → full VAE decode → mp4
- anchor = `H3LatentTailSlice(tail=1, context=0)` の部分 decode（約1秒）で
  次クリップの `last_frame` を即座に供給

実測（`benchmarks/v2_bench.jsonl`, backend `h3_v2-pipe-B`, 362f/480x864）:

```
server B decode  clip a 188.2s（model load 込み） / clip b 147.4s
```

GPU0 側の同条件 VAE は 47.4s/clip（`--fast` 時）。3060 は約3倍遅い。

**現状の判定**: 362f では sampling 1クリップ ≈ 127.7s < decode_B 147.4s のため
decode が新たな律速になり、ほぼ**収支ゼロ**。
**int8 VAE を server B に載せれば** decode_B ≈ 87s < 127.7s となり、
**VAE がまるごと wall から消える**（−47s 以上）。
→ 候補1 とセットで初めて意味を持つ。単独では推さない。

**必要作業**: `backend/h3_v2/h3v2pipe/h3_pipeline_nodes.py` を
`app/comfy_paths.yaml` に登録（`H3-Device-Barrier` と同じ方式）。外部依存なし。

---

## STATIC_REJECTED

| 候補 | 理由 |
|---|---|
| **FastH3 / VSA（全形態）** | **Ref2VA 非対応**。FastVideo 公式が「Ref2VA は独立の reference transformer を使い専用の蒸留 checkpoint が必要、FL2VA/Ref2VA は開発中」と明記。Preview v1 は全て **T2VA のみ** |
| `barelymining/ComfyUI-MiniMax-H3-FastVideo` | README に「**FL2VA base only** — T2VA, Ref2VA, L2VA variants would need matching FastVideo LoRAs (not yet released)」。要求 checkpoint も `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `minimax_h3_fastvideo_vsa_datafree_..._int8_convrot.safetensors` (22.9GB) | 上記 VSA の T2VA checkpoint。Ref2VA 入力を持たない |
| **Ref2V Turbo 8step v1.0 768p**（2026-09-04 リリース） | **steps が 4 → 8 で倍増**。速度候補ではなく品質候補。Ref2VA には**新しい 4-step は存在しない**（repo 内 Ref2V は 4step v0.1 544p と 8step v1.0 768p の2種のみ。4step の 768p 更新は FL2V 側だけ v1.0/v1.1/v1.2） |
| `PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot` | Turbo を重みへ merge 済みだが**推奨 8 steps**。かつ input-major variant は「original output-major Comfy layout を前提とする runtime とは非互換」と明記。主対象は Apple silicon / H3ddle |
| VAE 空間タイリングの調整 | 480x864 → latent 30x54。`tile_size=256` が両辺を上回るため `split_tiles` が単一タイルを返す（`vae.py:426`）。**当解像度ではそもそもタイリングしていない** |
| VAE 時間チャンクの調整 | `vae_ratio_t=4, clip_length=17 → tokens_chunk_size=5, token_drop=3 → token_overlap=2`、stride=3。`clip_length` を増やしても `overlap=(-3)%chunk` が連動し **stride は 3 のまま**、チャンク当たり仕事量だけ増える。現行が既に最効率 |
| TAESD H3 | preview 用軽量 decoder。最終出力品質に使えない。当アプリの last_frame は完成 mp4 から ffmpeg 抽出なので anchor 用途にも不要 |
| クリップ分割数の変更（243fx3 / 124fx6） | 30秒あたり実測（512x896）: **362f x2 = 485.1s** / 243f x3 = 539.3s / 124f x6 = 597.9s。**現行の 362f x2 が最速**。分割を増やすと固定費（cond 約22s + VAE + load）が効いて遅くなる |
| ffmpeg 再エンコード | `merge.py:78` は stream copy 優先、失敗時のみ libx264。無駄な再エンコードなし |
| クリップ間の in-graph 連結 | story mode は `concat_previous=False`（`story_runner.py:362`）。前クリップの再 decode は発生していない |

---

## MODEL_CANDIDATES

| 名前 | サイズ | URL | Ref2VA 互換 | 期待効果 |
|---|---|---|---|---|
| `minimax_h3_video_vae_int8_convrot.safetensors` | **3.17 GB** | `Kijai/MiniMax-H3-experimental` | 無関係（VAE） | VAE **−32〜40s**、VRAM −1.9GB |
| `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | **11.8 GB** | `Kijai/MiniMax-H3-experimental` | Ref2VA 専用ビルド | staging 大幅減。LoRA bind 要確認 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 1.96 GB | `lightx2v/Minimax-h3-Turbo` | Ref2V 用 | **速度は悪化**（8 steps）。768p 品質枠として記録のみ |

現行維持: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（21GB） +
`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`（1.96GB、bind 624 keys）。
**Ref2VA 向けにこれより速い 4-step は 2026-09-07 時点で存在しない。**

---

## FASTH3_VSA_STATUS

```
現在 Ref2VA に対応しているか          : いいえ
FL2VA 限定か                          : はい（ComfyUI ノードは FL2VA base only。
                                        FastVideo 本体の Preview v1 は T2VA only）
VSA 用 checkpoint / LoRA が必要か     : 必要。VSA gate tensor を含む専用 adapter で、
                                        汎用 PEFT loader では読めない
upstream の gate_compress との接続    : upstream master の Attention に
                                        gate_compress / to_gate_compress、
                                        DiTBlock.forward に attention=None override
                                        という受け口は存在する。ただし master の H3 経路は
                                        comfy_aimdo.malloc_graph を import しており、
                                        comfy-aimdo 0.5.2 が必要（現行 0.4.15、venv は
                                        production と共有）
Sage と排他か併用か                    : VSA が attention コストを完全に支配するため
                                        併用しても差が出ない（作者計測）
4-step でも効果があるか                : ある（VSA 90% sparsity 前提で蒸留されている）
公開速度比較の条件                     : 1344x768 / 24fps / audio 付き 15秒 /
                                        B200 1枚 で 47.2秒 = dense 比 14.38x。
                                        コンシューマ GPU の数字ではない

判定: STATIC_REJECT（Ref2VA 非対応のため）
再確認トリガ: FastVideo が Ref2VA 蒸留 checkpoint を公開した時点
```

---

## VAE_CANDIDATES

現行: fp16 VAE、`--fast` 適用で 362f x2 = 94.8s（47.4s/clip）。全体の約 21%。

| 候補 | 判定 | 期待効果 | リスク |
|---|---|---|---|
| **int8_convrot VAE** | **採用候補（最優先）** | −32〜40s | Windows で効果が出ない報告あり。色・temporal は rel RMS 0.97% |
| **GPU1 パイプライン decode** | 条件付き | int8 と併用で −47s 以上 | 3060 単独では decode が律速。実装済みだが app 未配線 |
| VAEDecodeTiled / tile 調整 | 棄却 | — | 480x864 では未タイリング |
| temporal chunk 調整 | 棄却 | — | stride 3 が構造的下限 |
| fp16/bf16 decode 切替 | 棄却 | — | `working_dtypes = [float16, float32]` で既に fp16 |
| TorchCompileVAE | 棄却（既定） | — | dynamic VRAM の DLPack 変換と競合（ISSUE-1） |
| latent upscaler（`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`） | 保留 | 低解像度生成＋latent 拡大で decode 回避 | 480x864 は既に低解像度。品質戦略が変わる。今回は評価対象外 |

---

## WORKFLOW_WALLTIME_OPPORTUNITIES

GPU sampling 以外で削れる wall time。実測または実コード確認に基づく。

| # | 箇所 | 現状 | 機会 | 根拠 |
|---|---|---|---|---|
| **W1** | director → generate の model thrash | 毎セグメント `PurgeVRAM V2(purge_models=True)` 後に H3 stack 再ロード | director をまとめて先行実行し、generate を連続化 | `graphs.py:236-243`、bench cold/warm 差 20〜45s/clip、app-bench 差 99.1s/clip |
| **W2** | production に `--fast` が未適用 | `comfy_launch_args = ['--default-device','0','--reserve-vram','1.0']` | 昨日採用決定した `--fast` が**入っていない** | `app/config.json` を実地確認 |
| **W3** | VAE が critical path 上 | full decode 完了 → mp4 → ffmpeg で last_frame 抽出 → 次クリップ | anchor 部分 decode（約1s）で次クリップを解放し、full decode を後段/GPU1 へ | `ab_run.py:452-480` に実装済み |
| **W4** | pipeline node が app に未登録 | `h3v2pipe` は bench の whitelist 経由のみ | `app/comfy_paths.yaml` に追加すれば app からも使える | `comfy_paths.yaml` の `h3_device_barrier` と同方式 |
| — | 前クリップの再 decode | **無し**（`concat_previous=False`） | — | `story_runner.py:362` |
| — | ffmpeg 再エンコード | **無し**（stream copy 優先） | — | `merge.py:78` |
| — | Voice Master 再処理 | index==0 のみ。5秒 wav 抽出、1秒未満 | — | `story_runner.py:374`, `merge.py:145` |
| — | profile（VLM 人物解析） | **story 全体で1回**、キャッシュ済み | — | `story_runner.py:274`, `pipeline.py:263-289` |
| — | reference 画像の再処理 | クリップごとに LoadImage + scale。数百ms | — | `graphs.py:64-97` |
| — | クリップ分割数 | 362f x2 が3案中最速 | — | 上記 STATIC_REJECTED 参照 |

**注**: W1 の「99.1s/clip」のうち説明できているのは約 20s/clip のみ。
残り約 79s/clip は**未計測**であり、再ステージが原因という推測にとどまる。
次回 GPU テストの1本目でここを実測する。

---

## DOWNLOAD_LATER

| 優先 | ファイル | サイズ | 用途 |
|---|---|---|---|
| 1 | `minimax_h3_video_vae_int8_convrot.safetensors` | 3.17 GB | VAE 差し替え |
| 2 | `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | 11.8 GB | DiT 差し替え（LoRA bind 要確認） |
| 3 | `ComfyUI-sol-attn`（git clone） | 数 MB | attention ノード。モデル不要 |
| — | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 1.96 GB | 768p 品質枠。**速度目的では不要** |

合計（1〜3）: **約 15 GB**

---

## NEXT_GPU_TESTS（次回 GPU 使用時の順序・すべて 124f / 480x864 / same seed から）

| 順 | 内容 | 判定基準 | 中止条件 |
|---|---|---|---|
| **0** | **計測のみ**: app の 362f x2 を 1 本、stage 別タイムスタンプ付きで実行し、purge 後の H3 再ロードに何秒かかっているか実測 | W1 の 79s/clip を数値で確定 | — |
| 1 | int8_convrot VAE を 124f で A/B（交互ペア n≥2） | VAE 時間 −20% 以上 | 色ずれ・temporal 破綻・decode 失敗 |
| 2 | 1 が通れば 362f で A/B | VAE −20% 維持 | 同上 |
| 3 | sol-attn を clone、**124f のみ**で Sage 単体 vs Sage+Sol を A/B（tau=1.3, sink_conditioning=exact_kv, dense_blocks="0-2,-1"） | sampling −5% 以上 かつ identity 保持 | kernel error / NaN / 黒フレーム / 音ズレ → 即座に Sage 単体へ戻す |
| 4 | 3 が通れば 362f（**173,340 tokens で検証済み上限 103,237 を超える**ため慎重に） | sampling −5% 維持 | int32 stride overflow / 出力破綻 |
| 5 | W4A8 Ref2VA を DL、**まず LoRA bind key 数だけ静的確認**（600 以上か） | bind ≥ 600 | bind 失敗なら生成せず終了 |
| 6 | 5 が通れば 124f で identity 比較 | identity 保持 | 顔の同一性が崩れたら即棄却 |
| 7 | 勝者の組み合わせ（int8 VAE + Sol + W4A8） | — | — |
| 8 | W1 改修後に app 側 362f x2 を実測 | app wall −100s 以上 | — |

**原則**（前回と同じ）: 1候補ずつ / まず1本 / 交互ペアで測る /
production 変更前に hash 取得 / 破綻したら周辺コードを直さず元へ戻す。

---

## EXPECTED_BEST_CASE

### bench レベル（362f x2 / 480x864 / `--fast` / 現在 454.1s）

| 項目 | 現在 | 適用後 | 根拠 |
|---|---|---|---|
| sampling | 255.4 | 230〜255 | Sol-Attn 公開値 −10%（5090 full-model）。sm89 未ベンチのため 0 も想定 |
| VAE | 94.8 | 55〜63 | int8 VAE 1.5〜1.7x（公開値 1.5〜1.95x の下限側を採用） |
| conditioning | 52.5 | 52.5 | 改善候補なし |
| その他 | 51.4 | 35〜50 | W4A8 で staging 減。不確実 |
| **wall** | **454.1** | **373〜421** | **−7〜18%** |

GPU1 パイプライン decode を併用し VAE を wall から外せた場合の下限: **約 350s**。

### app レベル 30秒（現在 約 810s = 13分30秒）

| 施策 | 削減 | 確度 |
|---|---|---|
| W2: production に `--fast` 適用 | −22s | 高（bench 実測済み） |
| W1: director/generate の thrash 除去 | −40 〜 −198s | 中（下限のみ bench 実測、上限は推定） |
| int8 VAE | −32 〜 −40s | 中（Windows リスクあり） |
| Sol-Attn | 0 〜 −25s | 低（sm89 未ベンチ、173K tokens 未検証） |
| **合計** | **−94 〜 −285s** | |
| **予想 wall** | **525 〜 716s ＝ 約 8.7 〜 12 分** | |

### 5分（300秒）目標について

**現行の Ref2VA 構成では 480x864 / 30秒で 5 分に到達しない。**

sampling 単体が bench で 255.4s あり、上記候補をすべて成功させても
sampling は 230s 前後にしか下がらない。conditioning 52.5s と
クリップ固定費を足すと、**理論下限が約 350s（bench）/ 約 500s（app）** になる。

300秒を狙うには sampling 自体を半減させる必要があり、それが可能なのは:

1. **Ref2VA 対応の sparse-attention 蒸留**（= FastH3 の Ref2VA 版）。
   現在 FastVideo が開発中と公表しているが**未公開**。公開されれば
   T2VA での 14x（B200）に相当する削減が期待でき、5分は現実的になる。
2. **解像度を下げて外部アップスケール**。参考事例（RTX 4070 12GB）では
   720p 生成 + Real-ESRGAN 2x で FHD 直接生成 3分55秒 → **2分30秒**。
   ただし品質戦略の変更であり、人物同一性への影響は別途検証が必要。
3. **2-step 蒸留**。Ref2VA 向けには存在しない。

**結論**: 品質と人物同一性を維持したまま今日の候補だけで狙えるのは
**約 8.7 〜 12 分**（現在 13分30秒）。5 分到達には上記 1 または 2 が必要で、
1 は外部リリース待ち、2 はユーザーの品質方針の判断事項。

---

## STATE

```
STATE:  RESEARCH_ONLY
GPU使用: なし
生成:    なし
モデルDL: なし
production変更: なし
```
