# H3 静的調査レポート 2026-09-07 PART2（GPU不使用・追加分）

前回: `docs/H3_STATIC_RESEARCH_2026-09-07.md`（以下「前回報告」）。
本報告はその差分・追加のみを記録する。現行基準・環境・測定値は前回報告と同一。

本日の生成・推論・モデルDL・production変更: **なし**（静的調査のみ）。

---

## STATIC_RESEARCH_SUMMARY

前回報告の結論（int8 VAE最優先 / W1 thrash / Sol-Attn / W4A8 / GPU1パイプライン）は維持。
今回の新規発見は1件: **公式 PDD Acc LoRA の Ref2VA trunk**（Alibaba PAI, 8/26公開）。
ただし nfe=4 が下限のため **速度候補ではなく品質候補**。速度の期待値は前回報告のまま。

Ref2VA向け 2-step 蒸留は存在しないことを確認（larryvrh v4の2-stepはFL2VA trunk）。

---

## TOP_CANDIDATES（期待値順・最大5件）

### 1. int8_convrot Video VAEへの差し替え（前回報告 #1、維持・補強）

追加情報:

- 要件 ComfyUI 0.31.0+（production v0.34.5で充足、black-output問題なし）
- Windows リスクの裏付け追加: discussion #21 で5090複数名が「Windowsでは同等〜遅い」、
  Linux移行で解決。「INT8 VAE時にGPU使用率が低い」との報告あり
- 第三者実測（rzgar）: 5s 1344x768 decode 19.5s → 10.0s（1.95x）、rel RMS 0.97%
- 期待効果・品質リスクは前回報告どおり（VAE 94.8s → 55〜63s）

### 2. director/generate間 model thrash除去（前回報告 #2、維持・補強）

追加確認（`graphs.py:148-242`、`pipeline.py:291-360`、`comfy.py:351-441`）:

- director graph Bはセグメント毎に VLM loader（`_vlm_loader`）→ 推論 → `llama_cpp_unload_model`
  → `PurgeVRAM V2(purge_models=True)` を実行。VLM（約11GB）の load/unload も毎回発生
- generate graph も `_ref_chain`（LoadImage+scale）を毎回再実行（CPU数百ms、軽微）
- W1改修（director全先行）で VLM load も1回化できる。効果見積もりは前回報告どおり

### 3. Sol-Attn（前回報告 #3、維持・実装選択肢を追加）

追加情報:

- **kijai本人の `kijai/ComfyUI-SolAttn_triton`** を確認。4090/5090 + MiniMax H3でテスト済み。
  Saganaki22版（4ノード構成）より侵襲性が小さい代替実装。8/3公開、H3パスを主張
- Saganaki22版 v0.6.2（8/13、SM86追加）が最新。warmed-cache再測定（8/9）で
  bf16 1.38〜1.65x Sage（8K〜65K tokens）、residual int8_qk 1.73〜1.97x。
  full-model実測は依然 −10%（5090）のみ
- Windows注意: Tritonカーネルは初回compile遅延＋ `triton-windows` が必要。
  当環境 triton 3.7.1 の充足は前回確認済みだが、Windows上での実動作は未検証
- 期待効果は前回報告どおり（sampling −0〜10%）。124f先行の原則も維持

### 4. W4A8 Ref2VA checkpoint（前回報告 #4、維持・互換性情報の明確化）

追加情報:

- ComfyUI mainlineに `asym_w4a8_int8` 対応がmerge済み（#15308、8/7）。
  Kijai版・Winnougan版（`Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI`）は
  **標準loaderで読める**。custom branch不要（production v0.34.5で充足）
- `starsfriday/MiniMax-H3-w4a8` 系は独自loader＋Linux＋nvccビルド必須 → 採用対象外
- 未確認事項は前回報告どおり（Turbo4 LoRA bind、4bit品質）。12.5GB

### 5. GPU1パイプラインdecode（前回報告 #5、維持）

変更なし。int8 VAEとセットでのみ有効。単独では推さない。

---

## STATIC_REJECTED（今回追加分。前回分は前回報告参照）

| 候補 | 理由 |
|---|---|
| **PDD Acc LoRA Ref2VA**（`alibaba-pai/MiniMax-H3-Acc-LoRAs`、公式8/26） | **速度候補として棄却**。nfe下限=4で現行Turbo4と同forward数。品質候補としてMODEL_CANDIDATESに記録。別途custom node（`Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc`）＋euler＋学習済みsigma grid＋CFG1.0が必須で、現行graph（BasicScheduler/simple）とは非互換のgraph改変が必要 |
| **larryvrh Turbo v4**（`minimax_h3_turbo_v4_step600_ema`等） | **FL2VA trunk**。discussion #10で作者「ref2vは未対応・planned」。Ref2VA適用でidentity破壊の報告あり。現行configの既定loraがこれ（518-key no-op）であることと整合。2-step動作報告もFL2VA側 |
| **Turbo-SLA**（lightx2v、8/20） | **FL2V限定**。Ref2VA trunkなし |
| **Ref2V Turbo 8-step v1.0 768p**（9/4） | 前回報告どおり速度候補として棄却。8stepで遅い。品質枠記録のみ |
| **PulpCut merged Turbo INT8 ConvRot** | 前回報告どおり棄却（8-step推奨＋input-major非互換） |
| **FunControl**（ControlNet Union、PR #15860未merge） | 機能追加（制御性）であり速度寄与なし。モデル+2.3〜4.2GBでVRAM悪化。速度調査の対象外 |
| **H3 SGLang Pack** | worker/server型infraが必要。production構成と非互換 |
| **starsfriday W4A8系** | 独自loader＋Linux＋nvccビルド必須。Windows production対象外 |
| **Sage FP8系backend切替**（`sage_fp8_cuda_fast`等） | H3経路はKJNodes memory-efficient Sage patch固定相当。単体attentionベンチの数字でありH3 Ref2VA実測なし。Sol-Attn検証後に残件として再評価 |

FastH3/VSA: **STATIC_REJECT維持**。Preview v1は9/2更新分もT2VA only。
Ref2VA蒸留は「next few weeks」（8/27 blog）で未公開。

---

## MODEL_CANDIDATES（今回の差分）

| 名前 | サイズ | URL | Ref2VA互換 | 期待効果 |
|---|---|---|---|---|
| `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`（公式PDD） | **1.4 GB** | `alibaba-pai/MiniMax-H3-Acc-LoRAs` | Ref2VA専用trunk | 速度±0（nfe4）。公式蒸留の品質向上候補。要custom node |
| `minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors`（ 걔ComfyUI-key再配布） | 1.7 GB | `aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI` | 同上 | 同上。in-memory変換不要版 |
| （前回3件は維持: int8 VAE 3.17GB / W4A8 Ref2VA 11.8-12.5GB / Ref2V Turbo 8-step 1.96GB品質枠） | — | — | — | — |

PDD適用条件（静的確認）: `MiniMaxH3PDDAccApply` node、sampler euler、
Apply nodeのsigmas出力→SamplerCustomAdvanced、BasicGuider CFG 1.0、
SigmaShift 12.0/3.0 exact、他distill LoRAと非併用、step-cache系と非併用。
現行Turbo4（624 keys bind）との品質A/Bは将来のGPU枠で。

---

## FASTH3_VSA_STATUS

前回報告から変更なし: **STATIC_REJECT（Ref2VA非対応）**。
再確認トリガ: FastVideoのRef2VA蒸留checkpoint公開。

---

## VAE_CANDIDATES

前回報告から変更なし: int8_convrot最優先、GPU1パイプラインは条件付き。
追加の裏付けはTOP #1参照。

---

## WORKFLOW_WALLTIME_OPPORTUNITIES（追加分。W1〜W4は前回報告参照）

| # | 箇所 | 現状 | 機会 | 根拠 |
|---|---|---|---|---|
| **W2再確認** | production `app/config.json:7` | `comfy_launch_args` に `--fast` **依然なし** | 採用済み flags の適用漏れ（−22s、高確度） | 実地確認 |
| W5 | ComfyUI idle-stop | `stop_comfy_when_idle` 既定True、idle 90sで停止 | story毎にcold start（node pack loadで数十秒）。常駐化で1回化 | `session.py:64`、`comfy.py:33` |
| W6 | VLM load/unload毎セグメント | director毎にVLM load→unload | W1改修に包含（director先行で1回化） | `graphs.py:198-242` |
| W7 | 参照画像の二重load | director graph＋generate graphで各々LoadImage+scale | 共通化でCPU数百ms×2N。効果微小 | `graphs.py:153-154,196-197,296` |
| W8 | merge fallback | stream-copy失敗時のみlibx264 medium/CRF18 | 失敗時のみ遅い。preset veryfast化は品質方針次第。稀路 | `merge.py:99-102` |
| — | `extract_last_frame` | `-sseof`単発抽出×最大3回 | 無駄なし | `merge.py:118-142` |
| — | Voice Master抽出 | index==0のみ、先頭5s PCM | 無駄なし（1秒未満） | `merge.py:145-176`、`story_runner.py:375-386` |
| — | conditioning cache | プロンプトがclip毎に異なるためTE再実行は不可避 | 同一prompt連投時のみGOLD cache可（V3報告の将来案） | — |

---

## DOWNLOAD_LATER（更新）

| 優先 | ファイル | サイズ | 用途 |
|---|---|---|---|
| 1 | `minimax_h3_video_vae_int8_convrot.safetensors` | 3.17 GB | VAE差し替え（維持） |
| 2 | `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`（Kijai版、標準loader） | 12.5 GB | DiT差し替え（維持） |
| 3 | `ComfyUI-sol-attn` **または** `kijai/ComfyUI-SolAttn_triton`（git clone） | 数 MB | attention（kijai版を侵襲性で優先検討） |
| 4（品質枠・新規） | `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`＋PDD-Acc node pack | 1.4 GB＋数MB | PDD品質A/B用。速度目的では不要 |
| — | 8-step Turbo 768p | 1.96 GB | 品質枠（維持） |

合計（1〜3）: **約15.7GB**。4を足すと **約17.1GB**。

---

## NEXT_GPU_TESTS（前回順序を維持し、PDDを品質枝として追加）

0〜8は前回報告どおり。追加:

| 順 | 内容 | 判定基準 |
|---|---|---|
| 9 | PDD-Acc Ref2VA（nfe=4）を隔離worktree＋124fでTurbo4と品質A/B（速度比較ではない） | identity・顔・手・audioがTurbo4以上なら品質候補に格上げ |

原則維持: 1候補ずつ / 124f先行 / 交互ペア / production変更前にhash / 破綻即 revert。

---

## EXPECTED_BEST_CASE

前回報告の推定を維持:

- bench（362f x2 / --fast / 現454.1s）→ **373〜421s（−7〜18%）**、GPU1併用下限約350s
- app 30秒（現約810s）→ **525〜716s（約8.7〜12分）**
- **5分到達は現行Ref2VA構成では不可**。PDDは同step数のためこの壁を動かさない。
  壁を破るのはFastH3 Ref2VA版（未公開）か解像度戦略変更のみ。

---

## STATE

```
STATE:  RESEARCH_ONLY
GPU使用: なし
生成:    なし
モデルDL: なし
production変更: なし
```
