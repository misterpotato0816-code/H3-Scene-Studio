# V3 LONG_FAST — 夜間ループ結果（PARTIAL）

Spec: `docs/V3_LONG_FAST_SPEC.md`（文面再構成版）。
結論: **TARGET(<=300s)/ACCEPT(<=360s) は安全な手段では到達不能**。
最良安全構成の実測 floor = **約500s（2回確認: 502.7 / 498.8s）**。
LONG_FAST は unverified のまま。verified 経路は無傷。

## 採用候補構成（best safe, 要human review）

- Base ref2va pruned int8 + Ref2VA Turbo 4-step LoRA（624/624 bind、警告0）
- Euler / simple / 4 steps / strength 1.0 / SigmaShift 12/3
- H3 専用 Sage attention（実行グラフ接続済み、かつ高速化に寄与）
- 512x896、0f relay + last-frame anchor + Voice Master（tail audio 未接続）
- 2 x 362f = 724f = 30.2s

FULL_30S_RUNS（真の wall clock、orch 分離済み）:
- run1: WALL 502.7s（A 244.5 + B 240.6 + orch 17.6）
- run2: WALL 498.8s（A 243.1 + B 238.1 + orch 17.6）
- 出力: `V2_AB_lf362_full.mp4`（30.20s、h264 + AAC stereo）

## TIMINGS（代表、B-clip steady-state、362f）

load 1.4 / conditioning 23.6 / sampling 135.6 / VAE 60.8 / audio 1.7 /
relay 5.5 / merge 11.9 / other 0.0 / orchestration ~17.6（2clip分）

## GPU（nvidia-smi 2s poll、参考値）

GPU0 peak ≈ 10617〜11451 MB / GPU1 peak ≈ 1699〜3075 MB。

## QUALITY_EVIDENCE

- `benchmarks/compare_LF362_joint.jpg`（継ぎ目前後＋終端、identity 連続、崩壊なし）
- `benchmarks/audio_lf362.json`（A -20.9dB / B -18.9dB、dropout なし）
- `benchmarks/compare_LF5F.jpg`、`audio_lf5f.json`（5f 連鎖も coherent）
- human_review_required: 継ぎ目の動き連続性（動画試聴）、音声内容（セリフ/口パク）、
  15s超 clip の後半集中力。静止画では合格、動画確認は人間。

## 計測器修正（先行実施）

- `prev_video/prev_components/tail_images/tail_audio/voice_master` → relay、
  `load_*/scale_*` → conditioning、`last_frame` → conditioning に分類。
- `t_other`（未分類ノード、検出時 застройка警告＋記録）、`t_relay`、`t_merge`、
  `t_orchestration`（orch_*.json）を追加。未分類のまま性能判断しない。

## REJECTED_EXPERIMENTS（証拠つき、losing run も保存）

- 5f tail: 動作するが 0f より速くない → 22f は spec に従い未実施。
- ref_longest 1536→1024: cond 不変 → 棄却。
- prompt 短縮: cond -2s のみ → 情報のみ。
- sage-off（有効測定）: むしろ低速（B 53.1 vs 37.6）→ sage 維持。
- TE-4b: sampler で matmul hard crash（dim 2560 vs 5120）→ アーキ非互換で棄却。
- 3-step（有効測定）: -7s のみ、off-recipe リスク → 棄却。
- no-lastframe: -11s だが継ぎ目 reframe → 棄却。
- 480x864: -7% のみ → 512x896 維持。
- 無効測定の反省: harness の preset 置換が build に届かず sage-off/3-step/te4b
  各1回が無効化。`preset_obj` 経路＋graph 内容検証テストで修正・再測定済み。
  無効ファイルは `benchmarks/invalid_te4b_ran_with_32B/` に隔離（削除せず）。

## なぜ 360s に届かないか（構造的 floor）

per-submit 固定費（362f）: cond ≈24 + VAE ≈61 + relay/merge/load/aud ≈19 ≈
104s。sampling ≈136s。2 clip で 485s。安全な手段で固定費も sampling も
削れない（VAE/TE/DiT は ComfyUI API から手が出ない動的 offload 構造、
51GB staged > 12GB VRAM）。将来案: VAE decode と次 clip sampling の
2-GPU パイプライン化、GOLD conditioning cache（同一 prompt のみ）。

## TESTS

- tests_v2: 48/48（LONG_FAST graph/GraphError/preset_obj 回帰含む）
- v1 --selftest: PASSED（graphs.py 加筆後）
- frozen SHA: runners/manifest/evidence 一致、4+1 amendment は承認済み変更のみ
- verified 経路: LEGACY parity test green（byte-identical 維持）
