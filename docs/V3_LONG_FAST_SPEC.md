# V3 LONG_FAST — 確定仕様・成功条件・夜間ループ手順

> NOTE: `V3_LONG_FAST_SPEC.md` は未配布だったため、本書は user directive
> （2026-09-07 実施指示）の文面を確定仕様として再構成したものである。
> `[解釈]` は実施側の具体化であり、測定で覆ったら更新する。

## 1. 目的

30秒完成物を wall clock TARGET <= 300s / ACCEPT <= 360s で安定生成する
LONG_FAST 構成を確定させる。画質・identity・継ぎ目・音声を壊さない。

## 2. 固定要素（本命）

- Base: Ref2VA pruned int8（v1 と同一）+ Ref2VA Turbo 4-step LoRA
  （`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`）
- Sampler Euler / scheduler simple / 4 steps / strength 1.0
- MiniMaxH3SigmaShift video 12 / audio 3（4-step の必須条件）
- MiniMaxH3MemoryEfficientSageAttentionPatch（H3専用Sage、実行グラフに接続）
- continuation 本命 = Character Master（Original画像再注入）
  + previous last frame + silent Nf video + fixed Voice Master
- Voice Master 使用時、continuation の `ref_video_audios` は必ず未接続。
  違反は GraphError（提出前に落とす）。
- 外部 Voice Master 使用時は Clip 1 から `<Audio 1>` として適用する。
  - `[解釈]` Voice Master = Clip 1 生成音声から抽出した固定音声セグメント
    （新規DLなしで調達）。Clip 2 以降の `ref_audios.ref_audio_0` に接続。
    Clip 1 自身は Voice Master なし（抽出元のため）。

## 3. 探索順序（夜間ループ）

1. 計測器修正を先に（ab_run stage 分類 + `t_other` + orchestration 記録）。
2. 単 clip 計測（512x896、turbo4）。
3. tail 比較: 0f vs 5f。5f に改善傾向がある場合だけ 22f を試す。
   - `[解釈]` 0f = tail video なし（last frame 画像のみ + refs）。
     5f/22f = previous clip 末尾 Nf の silent video を `ref_video_0` に接続
     （audio は付けない）。
4. 解像度: 512x896 から開始。必要なら 480x864。480x864 未満に勝手に落とさない。
5. 30秒構成選択 → wall clock 計測 → 採用構成で最低2回確認。
6. 各ループ: 最大の実測ボトルネック → 最小安全修正 → テスト → 最小A/B →
   勝てば維持、負ければ戻す。losing run の証拠も消さない。

## 4. 成功条件（全て満たして DONE）

- [ ] 30秒完成物の真の wall clock が TARGET <= 300s（最低2回確認）。
      ACCEPT <= 360s は暫定採用の下限。
- [ ] Turbo4 + Euler/simple + SigmaShift 12/3 + H3 Sage が実行グラフに接続
      （提出グラフ + 実行ログで確認）。
- [ ] Voice Master 制約の GraphError テストが通る。
- [ ] 顔・継ぎ目・音声の証拠（joint 比較、identity 比較、audio 測定）を保存。
- [ ] LEGACY / FAST / LONG 等の verified 経路が壊れていない
      （tests_v2 + v1 selftest + frozen SHA）。
- [ ] 人間の目視/試聴が必要な項目は `human_review_required` と明記し、
      verified を偽装しない。

## 5. 停止条件

- DONE（上記全て）→ 最終報告して終了。
- GPU/ComfyUI クラッシュ → 原因調査・復旧を試みる。黙って機能 OFF にして
  成功扱いにしない。復旧不能なら BLOCKED + 原因 + 再開手順を報告。
- 朝までに TARGET 未達でも、測定済みの最良構成 + human_review_required で
  PARTIAL 報告（証拠は全残し）。

## 6. 禁止

- 一発実装で終了しない。5秒動画・30秒1回では終了しない。
- `prev_video / prev_components / tail_images` 等を未分類のまま性能判断しない。
- 新モデルDL（初期フェーズ）。
- verified への早期昇格。LONG_FAST は証拠完備まで unverified。
