# H3 異種GPU高速化 v1 完成版

## 完成状態

RTX 4070 Ti と RTX 3060 を併用するH3動画生成の実機確認済み構成を、`H3_HETERO_V1` として固定する。追加の高速化研究は行わない。

- 正式入口: `<H3_PROJECT_ROOT>\RUN_H3_HETERO_V1.ps1`
- 実測済み実装入口: `<H3_PROJECT_ROOT>\RUN_OPT_REF_MATCH.ps1`
- v1固定マニフェスト: `<H3_PROJECT_ROOT>\H3_HETERO_V1_MANIFEST.json`
- 308.37秒版フォールバック: `<H3_PROJECT_ROOT>\RUN_FINAL_GPU1_E2E.ps1`

正式入口は、実測済みの `RUN_OPT_REF_MATCH.ps1` を変更せず呼び出す薄いラッパーである。旧実験名を無理にrenameせず、既存参照とSHAゲートを維持した。

## GPU役割

| GPU | 実機 | 役割 | 固定根拠 |
|---|---|---|---|
| GPU1 | RTX 3060 | H3 text/reference conditioning | graph node 9 `SelectCLIPDevice(device=gpu:1)` |
| GPU0 | RTX 4070 Ti | H3 DiT sampling、Video VAE、Audio VAE | ComfyUI起動引数 `--default-device 0` |

GPU振り分けはComfyUI coreの `comfy_extras/nodes_multigpu.py` にある `SelectCLIPDevice` と、起動引数で行う。研究用custom node `ComfyUI-HeteroVRAM` は最終グラフから直接参照されていない。

## 固定生成設定

以下はv1で変更しない。

| 項目 | 固定値 |
|---|---|
| model | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| text encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors`、strength `1.0` |
| 解像度 | 576 x 1024 |
| frames | 124 |
| steps | 8 |
| seed | 123456789 |
| sampler | `res_multistep` |
| scheduler | `simple` |
| ref_image_size | `match` |
| reserve VRAM | 1.0 GiB |
| reference image | `h3test_face_ref.png` |
| reference前処理 | longest 1536、lanczos、32の倍数 |
| video/audio VAE | `minimax_h3_video_vae_fp16.safetensors` / `minimax_h3_audio_vae_fp32.safetensors` |

モデル実体は `<COMFYUI_SHARED_MODELS>`、参照画像はComfyUIの `input` にあり、2026-08-14の整理時点で存在を確認済み。

## 最終実測と採用判断

| 構成 | 時間 | 位置付け |
|---|---:|---|
| v1 `match` | 217.61秒（約3分38秒） | 正式採用 |
| 異種GPU `max` | 308.37秒 | 安全な比較・フォールバック |
| 旧CPU中心 | 約555秒相当 | 比較元 |

`match` は `max` から90.75秒、29.43%短縮した。旧構成比では約60.8%短縮、約2.55倍速相当。

主な追加高速化は `MiniMaxH3ReferenceToVideo.ref_image_size=max` から `match` への変更である。実測用Pythonは308.37秒版グラフのファイルSHAとcanonical SHAを確認し、生成入力の差を `ref_image_size`、保存先の差をfilename prefixだけに制限している。

実機目視で `max` と `match` の顔、identity、画質、動き、口の動きに明確な品質差を感じないことを確認済みとして、`match` を採用する。比較画像のpixel metricsは採否基準ではなく、人間の目視判定を優先した。

音声の若干のノイズ感は両版に存在するため、本高速化とは別課題とする。v1完成作業では音声処理を変更していない。

## 必要ファイルと依存関係

実運用の呼び出し順は次のとおり。

1. `RUN_H3_HETERO_V1.ps1`
2. `RUN_OPT_REF_MATCH.ps1`
3. `_hetero_test\opt_ref_match\run_opt_ref_match_inner.ps1`
4. `_hetero_test\opt_ref_match\run_opt_ref_match.py`
5. `_hetero_test\final_gpu1_e2e\final_prompt.json` を308秒版固定ベースとして読み込む
6. `ref_image_size=match` の候補グラフを構築し、ComfyUI APIへ送る

必要な設定・nodeは以下。

- `_hetero_test\hetero_paths.yaml`
- `_hetero_test\custom_nodes\H3-GOLD-Conditioning-Cache`
- ComfyUI core `SelectCLIPDevice`
- ComfyUIインストール側 `custom_nodes\comfyui_layerstyle`
- `_hetero_test\custom_nodes\comfyui-kjnodes-hetero-test`（現行whitelist互換性のため保持）
- `_hetero_test\gold\gold_final_prompt.txt`
- ComfyUI Python環境の `aiohttp`、`psutil`
- 比較シート作成用の `av`（PyAV）、`numpy`、`Pillow`

`H3-GOLD-Conditioning-Cache` は最終グラフでは固定プロンプトの読み込みとSHA検証に使う。CPU GOLD cache自体のロード・保存は最終生成では行わないが、検証基準として必ず保全する。

## 起動方法

PowerShellで次を実行する。

```powershell
Set-Location -LiteralPath '<H3_PROJECT_ROOT>'
.\RUN_H3_HETERO_V1.ps1
```

これは重い動画生成を1回開始する。port 8402が使用中の場合は停止し、既存プロセスをkillしない。出力はComfyUIの `output\H3\20260814_OPT` 配下へ連番保存される。

実測済みrunnerは実行時に作業用 `candidate_prompt.json`、`opt_result.json`、log、DBを更新する。v1の固定値と整理時点のハッシュは `H3_HETERO_V1_MANIFEST.json` を正とし、比較証跡を保存したい場合は実行前に別名コピーする。

## 元に戻す場合の安全な方法

`match` に問題が見つかった場合、ファイルを編集・renameせず、308.37秒の `max` 版入口を使う。

```powershell
Set-Location -LiteralPath '<H3_PROJECT_ROOT>'
.\RUN_FINAL_GPU1_E2E.ps1
```

このフォールバックは `_hetero_test\final_gpu1_e2e` 内で独立しており、`ref_image_size=max` の固定グラフを使う。CPU GOLD cacheやmanifestを再生成・rename・上書きして戻してはいけない。

## ファイル分類（削除前に確定）

### KEEP

- `RUN_H3_HETERO_V1.ps1`、`RUN_OPT_REF_MATCH.ps1`、`_hetero_test\opt_ref_match` のv1実装・結果・比較資料
- `RUN_FINAL_GPU1_E2E.ps1`、`_hetero_test\final_gpu1_e2e` の308.37秒フォールバック一式
- `_hetero_test\gold` 全体。CPU GOLD cache/manifest、GPU1 TE検証データ、固定promptを含む
- `_hetero_test\custom_nodes\H3-GOLD-Conditioning-Cache`
- `_hetero_test\custom_nodes\comfyui-kjnodes-hetero-test`
- `_hetero_test\hetero_paths.yaml`
- `H3_HETERO_V1_MANIFEST.json` と本書
- 実測動画 `gpu1_live_e2e_00001_.mp4` と `ref_match_00001_.mp4`（ComfyUI出力側）

### DELETE

最終実装から参照されず、port 8401/8402にlistenerがいないことを確認したうえで、次の0 byte一時lockだけを実際に削除した。

- `_hetero_test\final_gpu1_e2e\final.db.lock`
- `_hetero_test\opt_ref_match\opt.db.lock`

### REVIEW

- `_hetero_v010_archive`、`_hetero_v011_archive`、`_hetero_v011_src`、`_hetero_v013_src`: HeteroVRAM研究履歴
- `_hetero_test\custom_nodes\ComfyUI-HeteroVRAM`: 最終グラフ未参照だが研究・再現資料
- `_hetero_test\custom_nodes\H3-GPU1-TE-Eager`、`H3-AIMDO-Diagnostics`、`comfyui_tinyterranodes-hetero-test`: 過去検証用
- `_overnight`: 自動研究の履歴、失敗ログ、バックアップを含む
- `_backup`: 外部custom nodeの保全コピー
- `_results`: 過去測定結果
- rootの `PATCH_MANIFEST.json`、`OPT_PATCH_MANIFEST.json`、`README_HOTFIX*.txt`、`APPLY_ASSISTANT_PATCH.md`: パッチ適用履歴
- 各実測ディレクトリのDB、log、events: 再現・監査資料として今回は残す

迷いのあるファイルは削除していない。特にGOLD配下は一時ファイルに見えるものも含め、検証証跡の可能性があるため丸ごとKEEPとした。

## 完成時確認

- 217.61秒版の入口からPython、固定ベースグラフ、設定、custom node、モデル、参照画像まで存在確認済み
- 308.37秒版フォールバックを別入口・別ディレクトリで保全
- CPU GOLD cache/manifestのSHA-256を整理前に確認
- GPU1 TE capture cache/manifestのSHA-256を整理前に確認
- 最終結果JSONと両実測動画の存在を確認
- 整理中にComfyUI起動、API送信、動画生成、cache再生成は行っていない
