# H3-review 追加修正 設計書（2026-09-13）

利用者指示（LLM接続先の汎用化 / FAST生成後アップスケール / ストーリー接続改善 / 起動・終了ランチャー）を
4つの作業パッケージ（WP-A〜D）に分割する。各WPは独立した worktree（HEAD `d848309` ベース）で実装し、
メインセッションが統合・レビュー・実機確認を行う。

共通ルール:
- 変更禁止: `app/config.json`, `app/comfy_paths.yaml`, `H3_Media/`, `app/_debug/`, `app/projects/`, `app/story_projects/`, `<H3_PROJECT_ROOT>`。
- 秘密情報（APIキー、トークン）は `h3app.credstore`（Windows資格情報ストア）のみ。config.json・状態ファイル・ログ・テスト・レポートに平文を書かない。
- ネットワーク・実プロセス・実GPUに触るテストは禁止（aiohttp TestServer / unittest.mock で代替）。
- UI文言は日本語、コード内コメントは英語。既存の関数・ヘルパー（`_json_error`, `postJson`, `toast`, `media.export_video`, `config.update_config_file`）を再利用。
- 特定GPUのUUID・番号・機種名・個人パスをコードに埋め込まない（`gpu.build_plan` の結果を使う）。
- テスト実行: `<COMFYUI_DIR>/.venv/Scripts/python.exe`（psutil 7.2 / numpy 2.4 / cv2 5.0 / PIL / aiohttp あり）。
- 各WP完了時: selftest PASSED / tests_v2 49 / unittest OK / `node --check app/web/*.js` / prepublish PASS / `git diff --check`。

---

## WP-A: LLM接続先のユーザー選択式化

### A1. 設定データ（`h3app/ai_settings.py`）
新しい `ai_settings` 形（config.json の `ai_settings` セクション、`config.update_config_file` で保存）:

```json
{
  "schema": 2,
  "connection": {"kind": "local" | "external", "provider": "<provider_id>"},
  "roles": {
    "director":          {"override": false, "provider": "", "model": ""},
    "character_profile": {"override": false, "provider": "", "model": ""}
  },
  "providers": {
    "<provider_id>": {"base_url": "", "model": "", "vision_models": [], "extra": {}}
  },
  "gemma_fallback": true,
  "max_attempts": 3,
  "favorites": []
}
```
- `connection` が既定の接続先。各 role は `override=false` なら connection を使い、`override=true` なら role 固有の provider/model を使う（UI では「人物解析だけ別のモデルを使う」チェック）。
- `providers` はプロバイダーごとの保存済み設定。プロバイダーを切り替えても他の保存内容は保持する。
- 旧形（schema なし、`director.provider in {gemma, openai_compat, opencode_go}`, `local_server.base_url`）は `normalize()` で自動移行:
  `gemma`→`comfy_gemma`, `openai_compat`→`lmstudio`（base_url は `local_server.base_url`）, `opencode_go`→`opencode_go`。
  両 role が同じなら `connection` に、異なれば director を connection、character_profile を override にする。
- `_assert_no_secrets` はそのまま維持。

### A2. プロバイダー定義（新規 `h3app/llm_providers/`）
```
h3app/llm_providers/__init__.py   REGISTRY: {id: ProviderSpec}, get(id), for_kind(kind)
h3app/llm_providers/base.py       ProviderSpec(dataclass), BaseAdapter(ABC), LLMError(kind, message)
h3app/llm_providers/openai_compat.py   汎用 OpenAI 互換 (chat.completions, GET /models, Bearer)
h3app/llm_providers/lmstudio.py   openai_compat + ネイティブ /api/v1/models（vision/loaded）+ unload
h3app/llm_providers/ollama.py     ネイティブ /api/tags, /api/show(capabilities), /api/chat(images=base64), unload=/api/generate keep_alive=0（H3が使ったモデルのみ）
h3app/llm_providers/llamacpp.py   openai_compat（/v1）。/props で確認。unload 非対応（capabilities.unload=False）
h3app/llm_providers/vllm.py       openai_compat。unload 非対応
h3app/llm_providers/localai.py    openai_compat。unload 非対応
h3app/llm_providers/comfy_gemma.py  既存 GemmaDirectorProvider をラップ（ComfyUI内蔵）
h3app/llm_providers/opencode_go.py  既存 opencode_go.py を移動せずラップ（fetch_models / generate_text_full）
h3app/llm_providers/openai.py     https://api.openai.com/v1 chat.completions（画像は image_url data URL）, GET /models
h3app/llm_providers/anthropic.py  https://api.anthropic.com/v1/messages（x-api-key, anthropic-version: 2023-06-01, 画像は base64 source）, GET /v1/models
h3app/llm_providers/gemini.py     https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent（x-goog-api-key, inline_data）, GET /v1beta/models
h3app/llm_providers/openrouter.py https://openrouter.ai/api/v1 chat.completions（Bearer, HTTP-Referer/X-Title 任意）, GET /models（architecture.input_modalities で vision 判定）
h3app/llm_providers/openai_compat_external.py 任意URLの外部 OpenAI 互換（Bearer）
```
`ProviderSpec` フィールド: `id, label, kind("local"|"external"), default_base_url, url_editable(bool), needs_key(bool), key_name("llm:<id>"; 互換のため lmstudio→"openai_compat_local", opencode_go→"opencode_go"), supports_model_list, supports_unload, vision_detection("native"|"list"|"heuristic"|"manual"), fields([{name,label,type,placeholder}])`。

`BaseAdapter` 共通 API（すべて非同期、例外は `LLMError(kind in auth|network|model|parse|unsupported, message)`、秘密は絶対にメッセージへ含めない）:
- `list_models() -> {"models":[{"id","display","vision":bool|None,"loaded":bool|None}], "supported":bool, "error":str}`
- `test(model) -> {"ok","target":base_url_or_host,"model","text_ok","image_ok"|None,"latency_ms","error"}`
  接続テストは軽量テキスト1回（max_tokens 8）＋ vision 判定（native/list なら参照、manual/heuristic なら「未確認」）。
- `chat(system, user, images=[], max_tokens, temperature) -> (text, info)` info は既存 `local_llm.chat` と同じキー（http_status, elapsed_ms, usage...）。
- `vision_capability(model) -> True|False|None`
- `release(models: list[str]) -> {"ok","supported","unloaded":[...],"error"}` unload 非対応プロバイダーは `supported=False` を返しネットワークに触らない。**外部プロバイダーは常に supported=False**。
- ローカル系 URL は `ai_settings.validate_base_url`（ループバック/プライベートIPのみ）。外部系は https 必須、`openai_compat_external` はユーザーURL（http/https、userinfo 禁止）。
- 既存 `local_llm.py` / `opencode_go.py` の実装は内部で再利用してよい（挙動を変えない）。

### A3. 呼び出し側の差し替え
- `director_provider.py`: `OpenAICompatProvider` / `OpenCodeGoProvider` を汎用 `AdapterDirectorProvider(adapter, model)` に統一（Gemma は現行のまま）。
- `ai_settings.resolve_chain` / `run_with_fallback`: `kind` を `"gemma"|"local"|"external"` に整理。**ローカル失敗時に外部へは絶対に進まない**（gemma_fallback のみ）。外部の auth エラーは即停止。
- `pipeline._try_local_profile` / `_try_go_profile` を `_try_provider_profile` に統合: role の実効プロバイダーを解決し、`vision_capability` が True のときだけ画像を送る。False/None のときは送らず警告（既存文言を踏襲）。
- `pipeline._release_local_llm`: 実効プロバイダーの adapter.release() を使う（LM Studio 固有 API を他プロバイダーへ送らない）。
- `server.py` の `/api/ai/*` を整理:
  - `GET /api/ai/settings` → `{"ok", "settings", "providers": [ProviderSpec を JSON 化], "keys": {provider_id: bool}}`
  - `POST /api/ai/settings` 保存（従来どおり秘密不可）
  - `POST /api/ai/key` `{provider, key}` / `DELETE`（body `{provider}`）→ credstore の key_name へ
  - `POST /api/ai/models` `{provider, base_url?}` → adapter.list_models
  - `POST /api/ai/test` `{provider, base_url?, model}` → adapter.test
  - 旧 `/api/ai/local/*`, `/api/ai/test-connection`, `/api/ai/test-model`, `/api/ai/models`(GET) は新ルートへ委譲するか削除（フロントを合わせて更新）。
  - `api/director/*` の temp_model は `{provider, model}` を受ける。

### A4. 設定画面（`app/web/settings.js`, `index.html` の `#tabLlm`）
階層: **接続種別**（ローカル / 外部API のセグメントボタン）→ **プロバイダー**（種別で絞ったボタン列）→ **使用モデル**（一覧取得できれば select、できなければ/失敗時は手入力 input。両方併存: select + 「手入力」トグル）→ **プロバイダー固有の接続情報**（サーバーURL、APIキー/トークンの保存・削除、ComfyUI内蔵は「モデル: …」表示のみ）。
- 接続テストボタン → 結果行に「接続先 / モデル / テキスト対応 / 画像対応」を表示。
- 「人物解析だけ別のモデルを使う」チェック → 展開時のみ role 用の provider/model 選択。
- 見出し直下に現在の選択サマリー（例: `ローカル › LM Studio › google/gemma-4-12b-qat`）。
- 既存の下書き（draft）/保存済み比較・未保存バッジ・破棄確認の仕組みを維持し、再描画は draft から行う（保存済み値で作り直さない）。
- プロバイダー切替時は `providers[id]` の保存済み値を draft に読み出す（他の id の値は消さない）。
- 「LM Studio等」のような曖昧表記を使わない。選択肢は REGISTRY にある実装済みプロバイダーだけ。

### A5. テスト（`app/tests/test_llm_providers.py` 新規、既存 `test_llm_connections.py` は新形に更新）
- 旧→新の設定移行、各プロバイダーの保存・復元、providers 切替で他プロバイダーの値が保持される。
- キー名分離（provider ごとに credstore の name が異なる）。
- ローカル失敗時に外部 adapter が呼ばれない（run_with_fallback）。
- list_models 非対応/失敗時の手入力（settings 保存で `model` が任意文字列でも通る）。
- vision False/None のモデルに画像を渡さない（pipeline）。
- 各 adapter のリクエスト整形（URL・ヘッダー名・本文の形）を aiohttp TestServer で検証。レスポンスに秘密が出ない。
- release: LM Studio のみ unload POST、Ollama は keep_alive=0、他は supported=False でリクエストなし、外部は常に supported=False。
- フロント: `app/tests/ui_smoke.cjs` があれば、未保存の選択が再描画で戻らないケースを追加（なければ settings.js の純関数を切り出して node で検証）。

---

## WP-B: FAST 生成後の動画アップスケール

### B1. バックエンド（新規 `h3app/upscale.py`）
- `probe(path) -> {"width","height","fps","frames","duration","has_audio"}`（ffprobe）。
- `plan(source_probe, target: {"preset": "1080x1920"|"x2", "method": "esrgan"|"seedvr2"}, gpu_plan, object_info, model_files) -> {"ok","target_w","target_h","scale","method","graph_inputs","missing":[{"kind":"node"|"model","name","source_url","license","size_bytes"}],"warnings":[]}`
  - アスペクト比維持: 1080x1920 は短辺1080に合わせ、元比率と一致しなければ長辺は比率から計算（例 576x1024 → 1080x1920 は比率一致）。x2 は幅高さ2倍（偶数丸め）。
  - `esrgan`: 必要ノード `LoadVideo, GetVideoComponents, UpscaleModelLoader, ImageUpscaleWithModelBatched(kjnodes; 無ければ ImageUpscaleWithModel), ImageScale, CreateVideo, SaveVideo`、モデル `RealESRGAN_x4plus.pth`（object_info の UpscaleModelLoader options で確認）。x4 → ImageScale(lanczos) で目標へ縮小。
  - `seedvr2`: 必要ノード `SeedVR2LoadDiTModel, SeedVR2LoadVAEModel, SeedVR2VideoUpscaler`、モデルは **object_info の options ではなく実ファイル**（`<comfy_dir>/models/SEEDVR2/` と comfy_paths.yaml の SEEDVR2 ディレクトリ）で確認。本機に存在するのは `seedvr2_ema_7b_fp16.safetensors`（16.5GB）と `ema_vae_fp16.safetensors`。無い場合は missing に `source_url="https://huggingface.co/numz/SeedVR2_comfyUI"`, `license="Apache-2.0"`, サイズを載せて **ダウンロードしない**。
  - GPU: `gpu_plan.video.uuid` に対応する ComfyUI 側の `cuda:0` を DiT device、dual なら VAE を `cuda:1`、single なら VAE も `cuda:0`。`blocks_to_swap` は VRAM 12GB 級では 7B fp16 に対し 20（初期値。OOM 時は +8 して再試行、最大36）、`offload_device="cpu"`、VAE `decode_tiled=True`。`batch_size=5, temporal_overlap=2, color_correction="lab"`。
  - `resolution` は短辺（1080）、`max_resolution` は長辺（1920）。
- `build_graph(plan, source_video_name, prefix) -> graph dict`。入力動画は `comfy_inputs` の既存経路（`POST /upload/image` 相当の動画アップロード: 既存 story の `h3app_story_*.mp4` 受け渡しと同じ関数）で ComfyUI へ渡す。
- `run(pipeline, source_path, plan, runner) -> result`:
  1. `pipeline._release_local_llm()` と `client.free_models()`（生成モデル解放）。
  2. グラフ実行（既存 `client.run_graph` + Runner の進捗イベント。ステージ: 入力を準備 / モデルを読み込み / アップスケール / 動画を書き出し / 完了）。
  3. 出力動画（ComfyUI output）に **元動画の音声をストリームコピーで結合**（`ffmpeg -i up.mp4 -i src.mp4 -map 0:v -map 1:a -c copy -shortest`）。音声なし元動画は映像のみ。
  4. `media.export_video(..., filename=f"{src_stem}_upscaled_{tag}")`（tag: `1080p` / `x2`）で 完成動画 へ。同名は既存 `_2`,`_3` 採番。元動画は触らない。
  5. 検証: 出力の fps・フレーム数・尺（±1フレーム）・アスペクト比・音声有無を probe で照合し、不一致は `ok=False` と理由。
  - OOM（`kind == "oom"` の PipelineError）: seedvr2 は blocks_to_swap を増やして最大2回再試行、esrgan は per_batch を半減。再試行内容を result.debug に記録。
  - 中止: Runner.cancel → `client.interrupt()`。
- 結果: `{"ok","output","export":{...},"method","elapsed_s","peak_vram_mib"(system_stats差分の最大値をポーリング),"debug":{...},"error"}`。失敗時は成功表示しない。

### B2. API（`server.py`）
- `GET /api/upscale/options?path=<完成動画内の相対名 or project clip id>` → plan の事前判定（missing・警告・想定容量）。
- `POST /api/upscale/start {"source": {"kind":"project"|"story"|"file","id"/"name"}, "preset","method"}` → 409 if busy、`{"ok","job_id"}`。進捗は既存 `/api/events/{job_id}`（Runner を再利用、`job_type:"upscale"`）。
- `POST /api/upscale/cancel {"job_id"}`。
- source は `H3_Media/Videos/完成動画` または `クリップ` 配下、もしくは project/story のクリップだけ許可（パス検証）。

### B3. UI（`app.js` / `index.html`）
- 完成動画カード（`#cardResult`）とストーリーの完成動画欄に「高画質化」ボタン → ポップオーバー: 解像度（1080×1920 / 2倍）、方式（標準アップスケール(ESRGAN) / 高品質AI動画アップスケール(SeedVR2)）、開始。
- options API の missing があればボタンを無効化せず、押下時に不足ノード・モデル・配布元・ライセンス・容量を表示して確認を求める（導入は利用者作業、アプリはダウンロードしない）。
- 進捗バー・中止・失敗理由の表示、完了時に保存先パスと `パスをコピー`。

### B4. テスト（`app/tests/test_upscale.py`）
- plan: 比率維持、偶数丸め、missing の生成、GPU割り当ての反映、UUID 非依存。
- run: ffmpeg を subprocess モックで、元動画が上書き・削除されないこと、同名採番、音声/fps/尺/比率検証で不一致時 ok=False、OOM 再試行の blocks_to_swap 増分、失敗時に export が行われないこと。
- API: busy 409、パス検証で外部パス拒否。

---

## WP-C: ストーリー動画のつなぎ目

### C1. 解析（新規 `h3app/seams.py`、依存は ffmpeg/ffprobe + numpy + cv2 のみ）
- `decode_frames(path, start, count) -> np.ndarray[N,H,W,3]`（ffmpeg rawvideo パイプ、必要範囲のみ）。
- `boundary_metrics(a_tail: frames, b_head: frames) -> dict`: 平均輝度差（Y）、色差（Lab 平均 a/b 差）、SSIM（cv2 で実装、グレースケール）、フレーム間動き量（連続フレームの平均絶対差）、顔位置差（cv2 Haar frontalface。検出できなければ None）、カメラ移動方向（グローバル位相相関 `cv2.phaseCorrelate` のベクトル）。
- `detect_static_head(b_frames, thresh=0.6, max_frames=12) -> int` 冒頭のほぼ同一フレーム数（平均絶対差 < thresh が連続）。
- `choose_cut(a_tail, b_head, max_offset=12) -> {"offset", "score", "candidates":[...]}` 候補 offset（0..max）ごとに `a_last` と `b_head[offset]` のスコア = w1·(1-SSIM) + w2·|ΔY|/255 + w3·色差 + w4·顔位置差 + w5·動き量不整合。**台詞開始より後ろは候補から除外**（音声RMS onset を `extract_audio` + numpy で求める）。
- `color_match_params(a_tail, b_head) -> {"apply": bool, "gain": [r,g,b], "offset": [..], "ramp_frames": 6..12}` ΔY > 6 または色差 > 4 で適用。ramp は差の大きさで 6〜12。**意図された照明変化**（transition mode が cut/fade、または b_head 内で輝度が連続的に変化している=場面転換の兆候）には適用しない。
- `apply_color_ramp(frames, params)` numpy で先頭 ramp_frames に線形減衰の補正。
- `interpolate_pair(a_last, b_first, n=2..4, method="ffmpeg")` 2枚から `minterpolate` で中間フレームを得る。`method="rife"` は ComfyUI の `FrameInterpolationModelLoader` にモデルがある場合のみ。無ければ `LLMError` ではなく `SeamError("補間モデル(RIFE)が導入されていません…")` を返し、**黙って別方式にしない**（既定は ffmpeg minterpolate と明示）。
- `audio_adjust(b_audio_offset_s, scripted_pause: bool) -> trim` 映像の offset と同じ秒数だけ B の音声先頭を切る（同期維持）。脚本上の間（segment の speech が空 or 台詞開始前の間が意図的＝dialogue 先頭に `…`/`（間）` など既存の間表現、または利用者が「間を残す」指定）は削らない。無音の一律削除はしない。

### C2. 結合パイプライン（`merge.py` / `story_runner.merge`）
- 新 `merge_story(clips, out_path, boundaries: list[{"mode":"natural"|"cut"|"fade", "keep_pause": bool}], debug_dir) -> {"path","report":[...],"needs_review": bool}`。
  - `natural`（既定）: C1 の手順 = 静止フレーム除去 → cut offset 選択 → 色補正ランプ → 必要時のみ補間（cut 後の動き量が閾値超なら 2〜4フレーム）→ 音声を同じ量トリム。
  - `cut`: 何も補正せず連結。
  - `fade`: 0.3秒クロスフェード（xfade）+ 音声 acrossfade。
  - 映像は最終的に 1 回だけ再エンコード（libx264 crf 18, yuv420p, fps 維持）。音声は必要な区間のみトリムし concat（AAC 192k）。
- 各接続点の report に: 前後フレーム番号と時刻、平均輝度差、SSIM/差分、除去静止フレーム数、色補正有無と gain/offset/ramp、補間有無とフレーム数、音声調整量(ms)、選択方式、before/after の指標。`story_projects/<id>/seams_<ts>.json` と `app/_debug/` に保存。
- 極端な差（補正後も ΔY > 25 または SSIM < 0.5、または静止 > 24 フレーム）は `needs_review=True` とし、UI に「再生成または確認が必要」を表示（成功トーストだけで終わらせない）。
- 継続コンテキスト: 現行 LONG は `tail_frames=56` の tail relay 済み。`config.defaults["tail_frames"]` はそのまま。次クリップ冒頭の重複区間は「先頭 max_offset フレームからの接続位置選択」で実現する（モデルが正確な時間的継続を出力しない前提）。LONG_FAST（最終フレーム1枚）でも同じ結合処理を適用。
- 既存の `merge_clips` は残す（cut 用の内部実装）。`api_story_merge` の応答に `export` に加えて `seams`（report 要約, needs_review）を返す。

### C3. UI
- ストーリーの各セグメント（2つ目以降）に「つなぎ方: 自然につなぐ（既定） / カット / フェード」の select と「間を残す」チェック。`/api/story/segments` で保存（story.data.segments[i].transition）。
- 結合結果に接続点ごとの要約（輝度差 before→after、除去フレーム、補正有無）と `needs_review` 警告。

### C4. テスト（`app/tests/test_seams.py`）
- 合成フレーム（numpy 生成）で: 輝度差検出、静止フレーム検出と除去数、cut offset 選択、色補正ランプの単調減衰、cut/fade モードでは補正しない、意図的場面転換（輝度が連続変化）では補正しない、無音の一律削除をしない（keep_pause で trim=0）、RIFE 未導入時の明示エラー、report のキー、`api_story_merge` が `export` と `seams` を返す。ffmpeg 呼び出しは subprocess モックまたは小さな実ファイルを生成（ffmpeg が PATH に無い環境では skip）。

---

## WP-D: 起動・終了ランチャーと安全な終了処理

### D1. 実行状態（新規 `h3app/procstate.py`）
- 状態ファイル: `app/_runtime/h3_state.json`（`.gitignore` に `/app/_runtime/` を追加）。**PID・パス・ポート以外の秘密は書かない**。
- `ProcessIdentity = {pid, create_time, exe, cmdline, session_id(=H3 起動セッションの uuid), port, owned_by_h3: bool, role: "app"|"comfyui"|"lmstudio", loaded_models: []}`。
- `identify(pid) -> ProcessIdentity|None`（psutil: create_time/exe/cmdline）。`matches(recorded, live)` は pid + create_time（±2秒）+ exe + cmdline 一致を要求。
- `read_state()/write_state()/clear_state()`、`stale_entries()` 次回起動時の古い状態検出。
- `find_listener(port) -> {pid, name, exe}`（psutil.net_connections）。

### D2. ランチャー（新規 `h3app/launcher.py`、`python -m h3app.launcher start|stop|status`、将来 EXE 化可能な単一エントリ）
- `start`: `%~dp0` 相当の基準（`Path(__file__).resolve().parents[2]`）。検査: Python/venv（config の comfy_python か `<comfy_dir>/.venv/Scripts/python.exe`）、`app/server.py`, `app/config.json`, `app/comfy_paths.yaml`, `custom_nodes/H3-Device-Barrier`。ポート: app_port が LISTEN → その PID を identify、状態ファイルの app と一致すれば「起動済み」としてブラウザーだけ開く（二重起動しない）。一致しない別プロセスなら PID・実行ファイル名・ポートを表示して停止（自動終了しない）。comfy_port も同様に表示のみ。古い状態ファイルの H3 起動 ComfyUI が identity 一致で生存していれば表示して回収（terminate → 5秒 → kill、対象は一致した PID のみ）。app を `subprocess.Popen(..., creationflags=CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW)` で起動しログを `app/_runtime/app.out.log` へ。`GET /api/config` が 200 になるまで最大60秒待機 → `webbrowser.open`。失敗時は日本語で原因と対処を表示し、BAT 側で `pause`。
- `stop`: app が応答すれば `POST /api/shutdown {"mode":"graceful"}`（実行中なら `{"force": false}` で 409 → 利用者に確認して `interrupt: true` で再送）。応答が無い場合は状態ファイルから identity 一致するプロセスだけを順に終了。最後にポート解放を確認して残っていればポート・PID・プロセス名・H3管理対象か・推奨対処を表示。冪等（何も残っていなければ「終了済み」で 0 終了）。
- `status`: 状態ファイルと実プロセスの照合結果を表示。
- `RUN_H3.bat` / `STOP_H3.bat`（リポジトリ直下、ASCII のみ。`chcp 65001` は使わず、Python 側で日本語出力）: `%~dp0` から python を解決して `-X utf8 -m h3app.launcher start|stop` を呼ぶ薄い呼び出し口。失敗時 `pause`。管理者権限を要求しない。空白・日本語パスに対応（すべて引用）。

### D3. サーバー側終了処理（`server.py` + 新規 `h3app/shutdown.py`）
- `POST /api/shutdown {"interrupt": bool}` → `ShutdownSequence.run()`。実行中（pipeline.is_busy or story running）で `interrupt=false` → 409 `{"busy": true, "running": ...}`。
- 順序（各ステップは try/except で継続、結果を配列で返す）: ①受付停止フラグ（generate/story start は 503）②実行中なら interrupt ③実効プロバイダーの adapter.release（H3 が使ったモデルのみ; 非対応は "unsupported" として表示; 外部は呼ばない）④ ComfyUI 内蔵 LLM 解放（既存 llama unload）と `client.free_models()`（POST /free）⑤ H3 起動の ComfyUI を `client.shutdown()`（identity 一致確認後 terminate→kill）; 起動前から存在した ComfyUI は identity を照合し H3 管理外なら触らない ⑥ LM Studio: 設定 `shutdown.stop_lm_studio`（既定 true）が有効なら、接続先が lmstudio のときポートの LISTEN PID を identify し exe 名に `LM Studio` を含むことを確認 → モデル unload → `lms server stop`（PATH にあれば）→ プロセスツリー terminate（psutil children 含む）→ 5秒待って kill。`taskkill /IM` 禁止。他の任意 OpenAI 互換サーバーは終了しない。⑦ app 自身: 応答を返した後 `loop.call_later(0.5, raise GracefulExit)`。⑧ 状態ファイル片付け。
- 結果: `{"ok", "steps":[{"name","ok","skipped","message"}], "failed":[...], "ports":[{port, free, pid, name, managed}]}`。
- `Ctrl+C`/`SIGBREAK`/aiohttp cleanup でも同じ Sequence（app 自身の停止を除く）を実行（`signal.signal` + `on_cleanup`）。
- UI: 右上に「H3を終了」ボタン（設定ダイアログにも同じ）。押下 → 確認ダイアログ → 実行中なら「中止して終了 / キャンセル」→ 結果（失敗項目だけ強調）を表示 → ページは「H3は終了しました。再起動は RUN_H3.bat」を表示。**タブを閉じただけでは終了しない**（既存 watchdog の挙動は変更しない）。
- 設定: `config["shutdown"] = {"stop_lm_studio": true}`。設定画面（LLM接続タブの LM Studio 欄、または新「起動/終了」小節）に「H3終了時にLM Studioを終了」チェック、保存・復元。LM Studio が H3 起動前から動いていた（状態ファイルに owned_by_h3=false で記録、または起動時に検出）場合は「既に起動していたLM Studioも終了対象になります」を表示。

### D4. テスト（`app/tests/test_launcher.py`, `test_shutdown.py`）
psutil・subprocess・ネットワークはすべてモック。二重起動防止、H3 自身のポート再利用、無関係プロセスのポート競合で自動終了しない、PID 再利用（create_time 不一致）で終了しない、H3 管理下の ComfyUI だけ終了、unload → free → comfy stop → lmstudio stop → app の順序、LM Studio 設定の保存・復元、非 LM Studio 互換サーバーを終了しない、途中失敗でも後続継続、STOP 2回目が冪等、終了後ポート解放判定、状態ファイル/ログに秘密が出ない（credstore の値を含めない）、空白・日本語パスでのコマンド生成。

---

## 統合順序とファイル所有
| ファイル | A | B | C | D |
|---|---|---|---|---|
| `h3app/ai_settings.py`, `llm_providers/*`, `director_provider.py`, `local_llm.py`, `opencode_go.py` | 主 | | | release 呼び出しのみ利用 |
| `h3app/pipeline.py` | profile/release 部分 | `upscale.py` から `_release_local_llm`/`free_models` を呼ぶだけ | | |
| `h3app/upscale.py` | | 主 | | |
| `h3app/seams.py`, `merge.py`, `story_runner.py`, `story.py` | | | 主 | |
| `h3app/procstate.py`, `launcher.py`, `shutdown.py`, `RUN_H3.bat`, `STOP_H3.bat`, `.gitignore` | | | | 主 |
| `server.py` | `/api/ai/*` 区画 | `/api/upscale/*` 区画（新関数群を1箇所に追加） | `api_story_merge`/segments | `/api/shutdown`、signal、startup 状態記録 |
| `web/settings.js`, `index.html#tabLlm` | 主 | | | LM Studio 終了設定の小節のみ（末尾に追加） |
| `web/app.js`, `index.html` | | 完成動画カード | ストーリー欄 | ヘッダー「H3を終了」 |
| `web/style.css` | 追加のみ | 追加のみ | 追加のみ | 追加のみ |

競合を減らすため、各WPは server.py / app.js / index.html への追加を「自WP名のコメントブロック」でまとめ、既存行の書き換えは必要最小限にする。
