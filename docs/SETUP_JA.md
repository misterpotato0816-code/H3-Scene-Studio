# H3 セットアップ・起動・日本語音声の既知事項

このリポジトリにはソース、ワークフロー、設定例、静的テストを収録します。モデル、生成動画、参照画像、音声、キャッシュ、データベース、ログ、個人用設定は収録しません。

## 必要な環境

| 項目 | 要件 | 備考 |
|---|---|---|
| OS | Windows 11（64 bit） | 動作確認は Windows 11 のみ。`RUN_H3.bat` / `STOP_H3.bat` は Windows 用です |
| GPU | NVIDIA GPU（CUDA）。検証機は RTX 4070 Ti 12 GB + RTX 3060 12 GB | MiniMax H3 の int8 版で 12 GB 級 2 枚（動画 GPU + text encoder GPU）を使いました。1 枚構成は設定画面の「シングル」（text encoder を CPU）で動かす設計ですが、検証は限定的です |
| ComfyUI | 0.34 以降（`MiniMaxH3ReferenceToVideo`、`SelectCLIPDevice`、`LoadVideo` / `SaveVideo` を含む版） | ComfyUI Desktop でも手動インストール版でも可。H3 は ComfyUI の `main.py` を子プロセスとして起動します |
| Python | ComfyUI が使う Python 環境（3.12/3.13 で確認） | 別の Python を用意する必要はありません。`requirements.txt` は ComfyUI の環境へ追加インストールします |
| 追加ソフト | FFmpeg / ffprobe（PATH に通す） | 動画結合・つなぎ目補正・標準拡大に必須。WinGet: `winget install Gyan.FFmpeg` |
| 任意 | LM Studio、VOICEVOX ENGINE | LLM 接続先 / 台詞の試聴 |

## 導入の流れ

1. ComfyUI を導入し、下記のカスタムノードとモデルを配置する
2. このリポジトリを取得し、ComfyUI の Python 環境へ `requirements.txt` を導入する
3. `app/config.json` と `app/comfy_paths.yaml` を設定例から作る
4. `RUN_H3.bat` で起動し、設定画面で GPU と LLM 接続先を選ぶ（初回生成の前に必ず設定してください。推奨は LM Studio: `http://127.0.0.1:1234/v1` を起動し、モデルを 1 つロードしてから、設定 > LLM接続 でプロバイダーとモデルを選びます。未設定のまま生成すると、設定 > LLM接続 を案内するメッセージで停止します）

## モデルと外部依存

モデルはリポジトリ外の ComfyUI モデルストアに配置します。既定プロファイルが参照する主なファイルは次のとおりです。

- `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
- `loras/minimax_h3_turbo_v4_step600_ema.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`

> **任意**: 接続先に「ComfyUI内ローカルGemma」プロバイダーを選ぶ場合のみ、次の GGUF も配置します（既定の LLM 接続は LM Studio 等の外部サーバーで、これらは不要です）。
> - `LLM/gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf`
> - `LLM/gemma-4-E4B-it-mmproj-BF16.gguf`

H3 固有の `custom_nodes/H3-Device-Barrier` はリポジトリに含まれます。その他の ComfyUI 本体・カスタムノード・モデルは、それぞれの配布元から導入してください。モデル本体や ComfyUI の複製はこのリポジトリにコミットしません。利用条件は [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を参照してください。

### モデルの入手先

| ファイル | 入手先 | 配置先（`<COMFYUI_SHARED_MODELS>` 配下） |
|---|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（約 19.5 GiB） | https://huggingface.co/Comfy-Org/MiniMax-H3 （MiniMax H3 Community License） | `diffusion_models/` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（約 25.3 GiB） | 同上 | `text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors`、`minimax_h3_audio_vae_fp32.safetensors` | 同上 | `vae/` |
| `minimax_h3_turbo_v4_step600_ema.safetensors` | https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora （Apache-2.0） | `loras/` |
| `gemma-4-E4B-it-*.gguf`（本体 Q8_0 と mmproj、**任意**: 「ComfyUI内ローカルGemma」プロバイダーを選ぶ場合のみ） | Gemma 4 E4B の GGUF 配布元（Gemma Terms of Use）。`app/config.json` の `vlm.model` / `vlm.mmproj` にファイル名を書きます | `LLM/` |

`build/download_models.py <models_dir>` で MiniMax H3 本体と LoRA を Hugging Face から取得し、サイズを照合できます（`huggingface-hub` が必要。GGUF は対象外）。

### 必要なカスタムノード（ComfyUI の `custom_nodes/` に導入）

| ノードパック | 使う理由 | 入手先 |
|---|---|---|
| `custom_nodes/H3-Device-Barrier`（このリポジトリ） | GPU 割り当ての検証と text encoder の配置 | `comfy_paths.yaml` の `h3_device_barrier.custom_nodes` で指す |
| ComfyUI_LayerStyle（`LayerUtility: PurgeVRAM V2`、`ImageScaleByAspectRatio V2`） | VRAM 解放と参照画像のリサイズ | https://github.com/chflame163/ComfyUI_LayerStyle |

不足しているノードは、生成開始時に「不足ノード」としてパック名付きで表示されます。

### 任意のカスタムノード

既定の設定（LLM 接続 = LM Studio、Gemma フォールバック OFF）ではどちらも使いません。

| ノードパック | 使う理由 | 入手先 |
|---|---|---|
| ComfyUI-llama-cpp（`llama_cpp_*`） | ComfyUI 内蔵のローカル LLM（接続先「ComfyUI内ローカルGemma」を選んだ場合、または設定の「失敗した場合はComfyUI内Gemmaで続行する」を自分で ON にした場合のみ）。**配布元にライセンス表記がありません**（[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)） | https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm |
| ComfyUI-SeedVR2_VideoUpscaler | AI 高画質化（SeedVR2） | https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler |

### AI高画質化に使う追加モデル（任意）

「高画質化」の **標準拡大（Lanczos）** は ffmpeg だけで動作し、追加モデルは不要です。**AI高画質化** を使う場合だけ次を導入します。アプリは自動ダウンロードしません（不足していれば画面に配布元・ライセンス・容量を表示します）。

| 方式 | 必要なもの | 配置先 |
|---|---|---|
| RealESRGAN | `RealESRGAN_x4plus.pth`（[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)、BSD-3-Clause、約 64 MB）。ComfyUI 標準ノード `UpscaleModelLoader` / `ImageUpscaleWithModel` を使用 | `<COMFYUI_SHARED_MODELS>/upscale_models/` |
| SeedVR2 | カスタムノード [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) と、`seedvr2_ema_7b_fp16.safetensors`（約 16.5 GB）+ `ema_vae_fp16.safetensors`（約 0.5 GB）（[numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI)、Apache-2.0） | `<COMFYUI_SHARED_MODELS>/SEEDVR2/`（`comfy_paths.yaml` の `SEEDVR2:` 行） |

12 GB クラスの GPU では、SeedVR2 7B fp16 を「全ブロック CPU オフロード + 入出力層オフロード + VAE の 512 px タイル化」で実行します（アプリが自動設定）。576×1024・248 フレームを 1080×1920 にした実測は、Lanczos 約 2 秒、RealESRGAN 約 13 分、SeedVR2 約 53 分でした。

## 初回セットアップ

### Python パッケージ

ComfyUI の Python 環境（例: `<COMFYUI_DIR>\.venv\Scripts\python.exe`、ComfyUI Desktop なら同梱の Python）へ導入します。別の Python 環境を作らないでください（`torch` と `comfy` モジュールは ComfyUI 側のものを使います）。

```powershell
& '<COMFYUI_PYTHON>' -m pip install -r .\requirements.txt
```

### 設定ファイル

リポジトリのルートで、個人用設定を設定例から作ります。

```powershell
Copy-Item .\app\config.example.json .\app\config.json
Copy-Item .\app\comfy_paths.example.yaml .\app\comfy_paths.yaml
```

`app/config.json` で最低限次を自分の環境に合わせます。

| キー | 内容 |
|---|---|
| `comfy_dir` | ComfyUI 本体（`main.py` がある）フォルダの絶対パス |
| `comfy_python` | ComfyUI の Python（空なら `<comfy_dir>\.venv\Scripts\python.exe` を使用） |
| `media_root` | 生成動画の保存先ルート（`Videos\完成動画` / `Videos\クリップ` が作られます） |
| `comfy_port` / `app_port` | ComfyUI（既定 8411）と H3 画面（既定 8790）のポート。他のソフトと重なる場合は変更 |
| `vlm.model` / `vlm.mmproj`（**任意**） | 接続先「ComfyUI内ローカルGemma」プロバイダーだけが使う GGUF のファイル名（`LLM/` 配下）。既定の LM Studio 接続では使いません |

`app/comfy_paths.yaml` の `<COMFYUI_SHARED_MODELS>`（モデルの置き場）と `<H3_PROJECT_ROOT>`（このリポジトリの絶対パス）を実在するパスへ置き換えます。この YAML は H3 が ComfyUI を起動するときに `--extra-model-paths-config` として渡され、モデルと `custom_nodes/H3-Device-Barrier` の場所を ComfyUI に教えます。

`ai_settings`（LLM 接続先）、`gpu`（GPU 割り当て）、`shutdown`（終了時に LM Studio を止めるか）は設定画面から保存できるので、ファイルを直接編集する必要はありません。

R&D 用の `backend/h3_v2/ab_run.py` を使う場合だけ、次も作成します。

```powershell
Copy-Item .\backend\h3_v2\paths_rnd.example.yaml .\backend\h3_v2\paths_rnd.yaml
```

これらの実定ファイルと、ユーザーが編集する `app/action_presets.json` は `.gitignore` の対象です。API キーは設定ファイルへ書かず、アプリが利用する Windows Credential Manager に保存してください。

## 起動と終了

通常はリポジトリ直下の **`RUN_H3.bat`** をダブルクリックします。`app/config.json` の `comfy_python`（無ければ `<comfy_dir>\.venv\Scripts\python.exe`）で `python -X utf8 -m h3app.launcher start` を実行し、`http://127.0.0.1:8790/` をブラウザーで開きます。

- 既に H3 が起動していれば（状態ファイル `app/_runtime/h3_state.json` と実プロセスの PID・起動時刻・実行ファイルが一致）ブラウザーを開くだけで二重起動しません。ポートを別のプロセスが使っていれば PID と実行ファイルを表示して停止し、そのプロセスは終了しません。
- 前回クラッシュして残った H3 起動の ComfyUI は、記録と一致する場合だけ回収します。既存の ComfyUI（H3 が起動していないもの）は終了せず接続を試みます。
- ログは `app/_runtime/app.out.log`、ComfyUI のログは `app/_debug/comfy.log` / `comfy.err` です。

終了は画面右上の **「H3を終了」**、または **`STOP_H3.bat`** です。どちらも同じ順で処理します: 新規ジョブ受付停止 → 実行中なら中止確認 → ローカル LLM のモデル解放（LM Studio の場合）→ ComfyUI 内モデル解放 → H3 が起動した ComfyUI の終了 → 設定が有効なら LM Studio の終了 → H3 自身の終了 → 状態ファイル整理。終了後にポート（8790 / ComfyUI / LM Studio）の解放を確認し、残っていれば PID と対処を表示します。名前による一括終了（`taskkill /IM`）は行いません。タブを閉じただけでは終了しません。

「H3終了時にLM Studioを終了する」は設定 › LLM接続 の「起動と終了」で切り替えます（既定 ON）。H3 の起動前から動いていた LM Studio も対象になるため、共有利用中なら OFF にしてください。

旧ランチャー `RUN_H3_V2.bat` / `app/RUN_H3_APP.ps1` は互換のため残していますが、状態ファイルや安全な終了処理は `RUN_H3.bat` / `STOP_H3.bat` 側だけが行います。単発 FAST ベンチマーク `RUN_V2_FAST.ps1` は実際の動画生成を開始します。

## 設定画面（GPU・LLM接続・保存先）

右上「設定」に 3 つのタブがあります。未保存の変更はバッジで示し、閉じる時に破棄確認をします。

- **保存先**: 生成動画は `media_root` 配下の `Videos\完成動画`（単発の完成動画・ストーリー結合・高画質化の出力）と `Videos\クリップ`（ストーリーの各クリップ）へ、同名なら `_2`, `_3` を付けて上書きせずに書き出します。右上「生成動画フォルダ」でエクスプローラーが開きます。参照画像は `app/_comfy_input` に取り込み、生成直前に ComfyUI の `/upload/image` で渡します（ComfyUI の `input` へ直接書きません）。
- **GPU**: `nvidia-smi` で検出した GPU から、自動 / シングル / デュアル（動画生成 GPU と text encoder 用 GPU）を選びます。保存すると ComfyUI の起動引数 `--cuda-device <UUID,...> --reserve-vram <GB>` に反映され、次回 ComfyUI 起動時から有効です。H3 が起動していない ComfyUI は停止せず、割り当てが一致しない場合は生成を拒否します。
- **LLM接続**: 接続種別（ローカル / 外部API）→ プロバイダー → 使用モデル（一覧取得または手入力）→ 接続情報の順に選びます。
  - ローカル: LM Studio（既定の接続先）、OpenAI 互換ローカルサーバー、Ollama、llama.cpp server、vLLM、LocalAI（URL 編集可、ループバック / プライベート IP のみ）、ComfyUI内ローカルGemma（**任意**: 要 ComfyUI-llama-cpp、ライセンス表記なし）
  - 外部 API: OpenCode Go、OpenAI、Anthropic、Google Gemini、OpenRouter、外部 OpenAI 互換（固定エンドポイントは「接続先（固定）」として表示）
  - API キー / トークンは Windows 資格情報ストアにプロバイダーごとに保存します（設定ファイルへは書きません）。接続テストは**保存済み**のキーを使うので、入力後に「API Keyを保存」を押してからテストしてください。
  - 画像入力に対応していると確認できないモデルには画像を送りません（LM Studio は一覧から判定、他は手動確認チェック）。ローカル接続が失敗しても外部 API へは送信しません。
  - 「人物解析だけ別のモデルを使う」で人物解析のプロバイダー / モデルを分けられます。

## ストーリーのつなぎ目

ストーリーモードでは、2 本目以降の各セグメントに「つなぎ方」を指定できます。

- **自然につなぐ（既定）**: 前クリップ末尾と次クリップ先頭を解析し、先頭の準静止フレームと接続位置を選んで最大 18 フレーム（0.75 s）だけ削り、音声も同じフレーム位置で削って同期を保ちます。輝度差が 6 以上なら次クリップに 24〜72 フレームで減衰する輝度ランプをかけます（クロスフェードや補間は行いません）。台詞開始より後ろは削りません。
- **カット**: 補正なしで連結します。
- **フェード**: 0.3 秒のクロスフェード（映像・音声）。
- **間を残す**（チェック）または無言のセグメント: 映像も音声も一切削りません。

結合結果には接続点ごとの要約（輝度差の前後、除去フレーム数、色補正、音声調整）が表示されます。構図や露出が大きく違う場合（別ショットになった等）は色補正を行わず「要確認」と表示します。その場合は該当クリップを再生成してください。レポートは `app/story_projects/<id>/seams_*.json` にも保存されます。

## 高画質化（標準拡大 / AI高画質化）

完成動画カードとストーリーの完成動画カードの「高画質化」で、1080×1920 または 2 倍へ拡大します（アスペクト比は維持し、横動画を縦に引き伸ばしません）。

- **標準拡大（Lanczos, FFmpeg）**: 解像度変換のみ。追加モデル不要で数秒。
- **AI高画質化（RealESRGAN / SeedVR2）**: モデルでディテールを補完します。必要なノード・モデルが無い場合は画面に表示し、自動ダウンロードはしません。AI方式が失敗しても標準拡大へは切り替えません（失敗として表示）。

出力は `Videos\完成動画\<元の名前>_upscaled_1080p.mp4`（または `_x2`）で、元動画の音声をそのまま結合し、fps・フレーム数・尺・比率・音声の有無を元動画と照合します。

## 静的テスト

`<COMFYUI_PYTHON>` は使用中の ComfyUI 仮想環境の Python に置き換えます。次のテストは GPU 生成を行いません。

```powershell
& '<COMFYUI_PYTHON>' -X utf8 .\app\server.py --selftest
& '<COMFYUI_PYTHON>' -X utf8 .\backend\h3_v2\tests_v2.py
& '<COMFYUI_PYTHON>' -X utf8 -m unittest discover -s .\app\tests
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepublish-check.ps1
```

`app/tests` は実プロセス・GPU・ネットワークに触れません（実アプリを組み立てるテストも、起動処理を経ていないため終了シーケンスを実行しません）。

## 日本語音声で確認済みのこと

2026-09-12のアプリ監査・改善と、その検証範囲は [監査結果](AUDIT_2026-09-12.md) に記載しています。以下の生成音声に関する実測は既存の記録です。今回の変更後に試聴した結果ではありません。

- 5.17 秒の記録済みテストでは、日本語の2行のセリフが句読点を除いて文字起こしと一致しました。
- 記録済みの LONG_FAST 出力には AAC ステレオ音声があり、レベル測定ではドロップアウトは検出されていません。
- API キーそのものは JSON、ログ、プロジェクト書き出しに保存しない実装です。

## 日本語音声の未解決事項

- 日本語音声では、短い呼びかけや感嘆の抑揚・リズムが不自然になる場合があります。例：『見て見て！』が平坦に発音される場合があります。台詞の文字起こし一致は、イントネーションの自然さを保証しません。（外部音声エンジンへの切り替えは未実装です。）

- 口語の語尾・間・イントネーション、およびH3が生成する音声そのものの指定外発声は実機での再評価が必要です。アプリは全動画生成経路で指定台詞とプロンプトを照合し、混入を検出した場合は停止します。

- H3 生成音声には、わずかなノイズ感が残る場合があります。アプリ側での解消は未実装です。
- LONG_FAST の自動測定は音声の存在、レベル、ドロップアウトを確認するだけです。日本語の発音、セリフ内容、口パク、声質の一貫性は人が動画を再生して確認する必要があります。
- 30秒構成では継ぎ目の音声と口パクを含む最終品質確認が残っています。静止画比較や音量値だけで合格とは扱いません。
- 横・スクエア画面、および既定時間以外の音声品質は未測定です。

## 読み・アクセントのリハーサル（任意）

1. [VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine) を別途起動し、ローカルの `127.0.0.1:50021` で利用できる状態にします。
2. H3の「音声テスト」→「読み・アクセントのリハーサル」で接続し、話者を選択します。
3. 台詞を入力して読みを解析し、アクセント位置・話速・抑揚を調整して試聴します。
4. 必要なら「読みをH3音声テストへ」で台詞の文字列を取り込み、H3本来の声も試聴します。

VOICEVOXの調整値はH3動画には適用されません。実際の声・口パクはH3で生成して確認します。エンジン未導入でも既存のH3生成は使えます。アプリはエンジンやモデルを自動ダウンロードしません。

追加の回帰テストは、リポジトリ直下で次を実行できます（GPU不要）。

```powershell
& '<COMFYUI_PYTHON>' -X utf8 -m unittest discover -s .\app\tests -v
```

## よくある導入エラー

| 症状 | 原因と対処 |
|---|---|
| `RUN_H3.bat` が「Python not found」で止まる | `app/config.json` の `comfy_python` が空で、`<comfy_dir>\.venv\Scripts\python.exe` も無い。ComfyUI の Python の絶対パスを `comfy_python` に書く |
| 起動直後に「設定に問題があります」 | `comfy_dir` / `comfy_paths.yaml` / `custom_nodes/H3-Device-Barrier` のいずれかが見つからない。表示されたパスを確認 |
| 「ポート 8790 は使用中です」 | 別の H3 か他のソフトが使用中。既に H3 が起動していればブラウザーを開くだけ。他のソフトなら `app_port` を変更 |
| 「ポート 8411 は既にプロセスが使用しています」 | H3 以外が起動した ComfyUI。H3 はそれに接続を試み、GPU 設定が一致しない場合は生成を拒否する。止めるか `comfy_port` を変更 |
| 生成開始で「不足ノード」「モデルファイルが見つかりません」 | 上記のカスタムノード / モデルが未導入か、`comfy_paths.yaml` の場所が違う。ComfyUI 単体で `custom_nodes` が読み込まれているかも確認 |
| 「GPU のメモリが不足しました」 | 参照画像の枚数を減らす、参照画像サイズを下げる、設定画面の GPU タブで予約 VRAM を見直す。他のアプリ（LM Studio の常駐モデル等）が VRAM を使っていないか確認。H3 は生成前にローカル LLM のモデル解放を試みます |
| 「ローカルLLMサーバーに接続できません」 | LM Studio 等のサーバーが起動していない、または URL/ポートが違う。接続テストは**保存済み**のキーと URL を使うので、変更後は保存してからテスト |
| 「LLM接続が未設定です。設定 > LLM接続 で接続先とモデルを指定してください。」で生成が止まる | LLM 接続が未設定（初回起動直後など）。設定 > LLM接続 で接続先（LM Studio 推奨）とモデルを選んで保存する |
| 監督案の作成が長い / 止まる | 「ComfyUI内ローカルGemma」を選んでいる場合、GGUF が大きすぎるか VRAM 競合。LM Studio 等（既定）に切り替えるか、小さい GGUF を使う |
| ffmpeg / ffprobe が見つからない | PATH に追加する（新しいターミナルで反映）。`ffmpeg -version` で確認 |
| STOP 後もポートが残る | 表示された PID と実行ファイルを確認して手動で終了。H3 は自分が記録したプロセス以外を終了しません |

## Git に含めないもの

`.gitignore` は、個人用設定、認証情報候補、ユーザーの参照画像・音声、プロジェクト状態、MP4、モデル、キャッシュ、ログ、DB、バックアップ、ビルド済み EXE を除外します。ローカルの実データは削除しません。

コミット前は次を実行し、実際に Git が追跡するファイルだけを再検査します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepublish-check.ps1
```
