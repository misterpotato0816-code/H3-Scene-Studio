# H3 Shorts Studio v2 — PHASE 0 AUDIT / FREEZE

Date (UTC): 2026-09-06. Machine evidence only; no guessing.
Frozen hashes + versions: `docs/V2_BASELINE.json`. v1 files are READ-ONLY from here on.

## 1. v1 入口と凍結対象

- 正式入口 `RUN_H3_HETERO_V1.ps1`（薄いラッパ）→ 実測実装 `RUN_OPT_REF_MATCH.ps1`
  → `_hetero_test/opt_ref_match/run_opt_ref_match_inner.ps1`（isolated ComfyUI `:8402` 起動）
  → `run_opt_ref_match.py`（`../final_gpu1_e2e/final_prompt.json` を base に読替え、
  `ref_image_size=max→match` のみ変更＋SHA/canonical-SHA検証→POST /api/prompt）
- フォールバック `RUN_FINAL_GPU1_E2E.ps1`（308.37s, `ref_image_size=max`, `:8401`）
- 固定プロファイル: 576x1024 / 124f / 8step / res_multistep / simple / match / seed 123456789
- 基準: v1 match **217.61s** / fallback max **308.37s** / 旧CPU約555s
- `app/server.py --selftest`: **PASSED**（ComfyUI/GPUなし全ビルダ検証）
- git リポジトリなし。CHECKPOINT = 本書＋V2_BASELINE.json のみ。

## 2. ハードウェアとデュアルGPUの実態

- GPU0 RTX 4070 Ti 12GB（DiT sampling＋Video/Audio VAE, `--default-device 0`）
- GPU1 RTX 3060 12GB（text/reference conditioning, `SelectCLIPDevice gpu:1`）
- **本番 Desktop（:8188）は cuda:0 のみ公開**。`SelectCLIPDevice` の選択肢も `default/cpu` のみ。
  v1 の `gpu:1` ルーティングは isolated runner（8401/8402, 両GPU可視, `--cuda-device`不使用）でのみ成立。
  → v2 compat layer は起動時 `/object_info` の実選択肢を読んで適応し、欠落機能は disabled 化。
  アプリ全体を起動不能にしない。

## 3. ComfyUI / nodes 互換性（core 0.34.5, frontend 1.49.6, torch 2.10+cu130, 2134 nodes）

| 機能 | 状態 |
|---|---|
| MiniMaxH3ReferenceToVideo | ✅ AVAILABLE（ref_images≤9 / ref_videos≤3 / ref_video_audios≤3 / ref_audios≤3） |
| MiniMaxH3SigmaShift（video 12.0 / audio 3.0） | ✅ |
| MiniMax Low VRAM Attention / Chunk FeedForward | ✅（KJNodes, 出力一致設計とコードコメントで確認） |
| LoraLoaderModelOnly / SelectCLIPDevice / SageAttentionPatch / H3AddGuide | ✅ |
| H3 FunControl node | ❌ 存在しない（Wan用のみ。H3非対応）→ `funcontrol=false` 既定、CONTROL disabled |
| PDD node / H3 Turbo loader node | ❌ 存在しない → `experimental.pdd=false` 既定、PDD単独 unavailable |
| FL2VA モデル（first-last-frame） | ✅ ファイル有り → LONG 連鎖の候補（要検証） |

## 4. Turbo LoRA 再検証（ファイル名仮定の排除）

- `minimax_h3_turbo_v4_step600_ema.safetensors`（v1指定, 518 tensors）:
  キーに `diffusion_model.` prefix が無く、ComfyUI prune モデルに **0 key bind＝実質 no-op**。
  app 側コメント（config.py:68-71, README:268）と一致。**v1=217.61s は実質 Base Ref2VA 8-step の記録。**
- `*_pruned_comfyui` 2種（416 tensors, prefix有り）: bind はするが metadata が
  「AdaLN adapter 全除去・4-step distillation は劣化/破損の可能性」と明記。生成テスト必須。
- 結論: v2 FAST は bind 実証（bound-key 数ログ＋`TURBO_UNAVAILABLE`明示、Base への黙従 fallback 禁止）なしに名乗らせない。

## 5. 依存関係図（要旨）

- `app/h3app/*`: pure builders（graphs/compiler/duration/segmenter/transitions/canon）
  ＋ comfy client ＋ pipeline（A profile→B director→C generate の3グラフ編成）
  ＋ story_runner（同一グラフを使う上位 controller、tail relay＝前clip末尾tail_frames＋audio継承）
- 既知の制約: torch.compile 不可（ISSUE-1/2）、llama.cpp vram_limit 部分指定で即死（ISSUE-3, -1=自動のみ）

## 6. v2 実装方針（不変条件）

- 新規コードは `backend/h3_v2/`＋`workflows/v2/`＋`docs/V2_*` のみ。v1 パスへの書込み禁止。
- backend 選択制（既定 `legacy_v1`）。experimental は flag 既定 off＋物理/論理分離。
- benchmark は JSONL（preset/model/LoRA/shift/seed/timings/VRAM/outcome）必須。体感判断禁止。
- 次: PHASE 1（backend separation＋compat layer＋feature flags＋preset engine骨格＋benchmark schema）。
