# H3 App UI_AUDIT.md — 現UI監査（2026-09-08、コード読解ベース）

対象: `app/web/index.html`（299行）、`app/web/app.js`（2436行）、
`app/web/style.css`（434行）。既にdark基調（--bg #14171c）。

## 現在の画面構造

1. header.topbar: タイトル＋説明文＋engineStatus＋compatStatus（長文1行）＋sysbar（モデル解放）
2. .modeswitch: 単発生成／ストーリーモード切替
3. cardImages（1参照画像）→ cardSingleText（2指示）→ cardSettings（3設定＋生成ボタン）
4. cardStoryScript（S1台本）→ cardStoryOptions（S2）→ cardStorySegments（S3）→ cardStoryRun（S4）
5. cardAdvanced（4詳細）→ cardStatus（実行状態）→ cardResult（5完成）／cardStoryFinal（S5）→ cardStoryList（S6）

単一縦積み＋mode-single/mode-storyのhidden切替。routingなし。

## 情報過多な場所

- **compatStatus**: ComfyUI/GPU0/GPU1/H3/Turbo/全mode可否を1行テキストで羅列。常時全文表示で可読性が低い → System chip＋detailsへ
- **engineStatus**: 2行目の冗長文 → chip化
- **resultMeta**: `モード ／ ファイル ／ 生成秒 ／ 場所（フルパス）` を1行連結。フルパス常時表示 → rows＋detailsへ
- **cardSettings**: mode pills＋seconds＋aspect＋Advanced＋生成ボタンが1カードに平置き

## 分かりにくい操作

- 生成ボタンが設定カード内に埋没（最重要CTAなのにstickyでない）
- 生成不可理由が単発側にない（story側のstoryStartWhyのみ存在）。btnGenerateはdisabledになるが理由表示なし
- mode pillsが通常/Advanced/Experimentalの3箇所に分散。Experimentalは理由がnoteに集約
- story開始までの導線がS1→S4に分散し、現在位置が分からない（navなし）

## 重複UI

- 保存フォルダ導線が3箇所（btnOpenFolder / btnStoryOpenFolder / btnStoryRevealFile）＋今回追加の全体ボタンで4箇所。役割分担を明示する必要あり（個別revealは維持）
- 進捗表示が2系統（cardStatusのstageList＋storyのstatuspanel）。統合はせず、sticky barから両方へ誘導する

## redesign対象

- topbar → sticky＋chips＋H3フォルダボタン
- 左nav追加（Source/Character・Voice/Generation/Story/Outputへのanchor）
- mode pills → mode cards（FAST cyan / QUALITY purple / LONG_FAST cyan / LEGACY gray、選択時glow）
- 生成CTA → sticky bottom action bar（状態文＋実行/中止ミラー）
- compat → chip＋details（Diagnostics）
- resultMeta → rows＋details（フルパス格納）
- accentをcyan系へ微調整（現行#4f9cf7は維持範囲でglow等を追加）

## 維持対象（動作・ID不変）

- 全既存ID・API・イベント購読・台本エディタ・segment操作・lock/approve/regen
- stageList/progress/errorCard構造、story statuspanel、Advanced内容
- generation graph・config既定値・LEGACY parity（UI層のみ、生成層不変）
- per-result/per-storyの保存先ボタン（reveal.py経路を維持）

## メディア保存経路（現状）

- upload → ComfyUI input/h3app_ref_*（作業用）
- 単発local → app/projects/<id>/clip_XX.mp4。ComfyUI output原本あり
- story clips/frames → app/story_projects/<sid>/clips|frames。final → final_dir
- voice master → frames/voice_master_00.wav＋ComfyUI input stage
- benchmarks/・_debug・fixturesは証拠系（H3_Media対象外）
- reveal.py allowed_roots: story_dir・projects_dir・comfy_outputの3箇所
