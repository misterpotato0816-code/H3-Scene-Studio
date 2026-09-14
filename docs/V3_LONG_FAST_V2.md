# V3 LONG_FAST — 速度追求 第二夜（ directive 追補実行記録）

前夜 PARTIAL（2x362f @512x896 ≈ 500s）からの継続。結論は本文末尾。

## SELECTED_SEGMENTATION: 2x362f（production baseline 固定）

SEGMENTATION_SWEEP（同一 LONG_FAST 条件、0f+lastframe+VM+sage、真の wall）：

| 構成 | wall | 内訳 | joints |
|---|---|---|---|
| A: 2x362f @512x896 | **502.7 / 498.8s**（2回） | A~244 + B~240 | 1、coherent |
| B: 3x243f @512x896 | **557.9s**（189.7+174.1+175.5） | 30.41s 出力 | 2、coherent |
| C: 6x124f @512x896 | **617.1s** | 31.03s 出力 | 5、全て coherent |

証拠: `compare_SW243_joints.jpg`、`compare_SW124_joints123.jpg`、
`compare_SW124_joints45.jpg`（identity/pose/bg/mouth 全 joint 目視合格、
崩壊なし）。joint 数を理由に短尺構成を棄却しない方針のもと、
速度で A が最速のため **2x362f を production baseline に固定**。

## PIPELINE（ANCHOR-FIRST、2-server 実装→棄却）

実装内容:
- `H3SaveLatentRaw` / `H3LoadLatentRaw`（stock SaveLatent は NestedTensor で
  crash するため custom。latent 5MB、handoff は安価）
- `H3LatentTailSlice`（video [B,C,T,H,W] の dim2=time を narrow。初版は
  dim1 誤りで VAE エラー→修正済み）
- `H3LatentProbe`（構造特定: video(1,24,37,54,30)+audio(1,32,2,207)）
- Server B（:8403、`--default-device 1`、VAE のみ、GPU1 12GB に十分収まる）
- Overlap: A-cond+samp と B-decode を `asyncio.gather` で並列、
  anchor PNG ポーリングで次 clip 開始

実測（pipe362 @480x864）: **WALL 587.8s**（serial 469s より悪い）。
- B-decode on GPU1: 155〜188s（GPU0 の 61s の約3倍。3060 の conv が遅い）
- A-submit が競合で低速化（238s vs 220s、PCIe/メモリ帯域競合）
- overlap_saved 見積 199.7s あったが、露出 tail（156s）＋低速化で相殺
- よって **GPU1 VAE pipeline は棄却（rollback、serial 維持）**

ANCHOR_DECODE（Phase 2A、採用）:
- K=1/C=0 partial decode **0.9s**、GT 比較 MSE 119（自然変動 71 に対しやや高めだが
  目視で face/framing/color 同等 → `compare_ANCHOR_gt_c0_c8.jpg`）
- C=4（14.2s）/C=8（5.0s）より C=0 を採用。最小 context = 0 で確定。

Phase 2B（latent-direct）: H3 ノードは IMAGE 型 reference のみ受け付け、
latent 受け口が存在しないため**API 根拠で即棄却**（実装なし）。

CACHE（§7-8、不採用・根拠あり）:
- Character Master encode: ref 1536→1024 縮小で cond 不変 → 画像部 ≈0s。
  cache しても削減なし。
- Voice Master encode: LoadAudio 0.0s 計測。削減なし。
- 残 cond（約20s）は TE 由来とみられるが、互換 TE が存在しない
  （4b は matmul crash で hard reject 済み）。cache 対象なし。

FFMPEG（§9）: 既に準拠（clip 毎 CreateVideo+SaveVideo のみ、最終結合のみ
`-c copy`）。中間 remux なし。変更なし。

RESOLUTION: **480x864 を production に固定**（-7%、joint 目視で劣化なし →
`compare_480_joint.jpg`）。480 未満には落とさない。

## FULL_30S_RUNS（最終・serial production 構成）

- lf362 @512x896: 502.7 / 498.8s（30.20s 出力、2回確認）
- lf362 @480x864: 469.3s（A 231.4 + B 220.4 + orch 17.5）
- pipe362 @480x864: 587.8s（棄却、証拠として保持）
- 平均 production wall: **約469s**（TARGET 300 / ACCEPT 360 未達）

## TIMINGS（362f relay steady、480x864）

load 1.4 / conditioning ~20 / sampling ~125 / VAE ~61 / audio ~1.7 /
relay ~5.5 / merge ~12 / other 0.0 / orchestration ~17.5（2clip分）

## GPU

GPU0 peak ≈ 8142〜10988 MB / GPU1 ≈ 1699〜2563 MB（2s poll 参考値）。

## QUALITY_EVIDENCE

- joints: compare_480_joint.jpg（採用構成）、compare_SW*（sweep 全構成）
- anchor: compare_ANCHOR_gt_c0_c8.jpg（C=0 が GT 同等）
- audio: audio_480.json（-19.0/-18.1dB）、audio_lf362.json、pipeA -19.4dB
- human_review_required: 継ぎ目モーション連続性（動画試聴）、音声内容
  （セリフ/口パク）。静止画 gate は全 PASS。

## TESTS

- tests_v2: 48/48（次項で再確認）
- v1 --selftest: PASS（次項で再確認）
- frozen SHA: runners/manifest/evidence 一致（amendment 5件は承認済みのみ）

## REJECTED_EXPERIMENTS（今夜分、証拠保持）

- 3x243 / 6x124（速度で 2x362 に敗北、品質は合格）
- GPU1 VAE pipeline（net loss、rollback）
- latent-direct（API 根拠で棄却）
- Character/Voice encode cache（削減対象 ≈0s のため不採用）
