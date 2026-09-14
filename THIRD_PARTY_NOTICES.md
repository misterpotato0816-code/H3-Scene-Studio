# 第三者ソフトウェア・モデルの利用条件（THIRD PARTY NOTICES）

## 0. ライセンスの適用範囲

- **H3 自身のコードと文書**（`app/`、`backend/h3_v2/`、`build/`、`custom_nodes/H3-Device-Barrier/`、`_hetero_test/custom_nodes/H3-GOLD-Conditioning-Cache/` と `_hetero_test/` の実験スクリプト、`workflows/`、`scripts/`、`docs/`、ルートのランチャー）: **MIT**（`LICENSE`）。
- 上記の MIT は、実行時に利用する外部ソフトウェア・カスタムノード・モデルには**及びません**。それぞれの利用条件は下記のとおりで、各配布元の最新版が優先します（確認日: 2026-09-14）。
- **公開版に含めない第三者コード**: 開発用の非公開リポジトリには、異種 GPU 実験のために [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)（GPL-3.0）を改変したコピー（`_hetero_test/custom_nodes/comfyui-kjnodes-hetero-test/`）が同梱の `LICENSE` 付きで保存されています。H3 の実行・テスト・導入手順はこれを参照しません。公開版のツリーには含めず、MIT の対象にもしません（GPL-3.0 のまま非公開側で保持）。

## 1. 公開版に同梱している第三者コード

なし（`_hetero_test/custom_nodes/comfyui-kjnodes-hetero-test/` は上記のとおり公開版から除外）。

## 2. 実行時に必要・任意の外部ソフトウェア（同梱していません）

| ソフトウェア | 用途 | ライセンス | 入手先 |
|---|---|---|---|
| ComfyUI | 生成エンジン本体（H3 は HTTP/WebSocket で接続。`custom_nodes/H3-Device-Barrier` は ComfyUI 内で動くカスタムノード） | GPL-3.0 | https://github.com/Comfy-Org/ComfyUI |
| ComfyUI-llama-cpp（`llama_cpp_model_loader` / `llama_cpp_parameters` / `llama_cpp_instruct_adv` / `llama_cpp_unload_model`） | ComfyUI 内蔵ローカル LLM（接続先「ComfyUI内ローカルGemma」と、他プロバイダー失敗時の代替）。LM Studio 等を接続先にし、「失敗した場合はComfyUI内Gemmaで続行する」を OFF にすれば不要 | **ライセンス表記なし**（配布元リポジトリの GitHub 表示・Comfy Registry・pyproject・README のいずれにもライセンス種別が無く、pyproject が参照する `LICENSE` ファイルも未収録。2026-09-14 再確認） | https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm （Comfy Registry 名: comfyui-llama-cpp） |
| ComfyUI_LayerStyle（`LayerUtility: PurgeVRAM V2` / `ImageScaleByAspectRatio V2`） | VRAM 解放・参照画像のリサイズ | MIT | https://github.com/chflame163/ComfyUI_LayerStyle |
| ComfyUI-SeedVR2_VideoUpscaler（任意、AI 高画質化） | SeedVR2 による動画アップスケール | Apache-2.0 | https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler |
| FFmpeg / ffprobe | 動画結合・つなぎ目補正・標準拡大 | LGPL-2.1+ / GPL（ビルド構成による） | https://ffmpeg.org/ |
| LM Studio（任意） | ローカル LLM サーバー | 配布元の利用規約 | https://lmstudio.ai/ |
| Ollama / llama.cpp server / vLLM / LocalAI（任意） | ローカル LLM サーバー（接続のみ、モックテスト済み） | 各配布元 | — |
| VOICEVOX ENGINE（任意） | 台詞の読み・アクセント試聴 | 配布元の利用規約（合成音声の利用条件は話者ごと） | https://github.com/VOICEVOX/voicevox_engine |
| Python パッケージ（aiohttp, huggingface-hub, pywin32, Pillow, psutil, numpy） | `requirements.txt` | 各パッケージのライセンス（Apache-2.0 / BSD / PSF 等） | PyPI |

## 3. モデル（同梱していません。利用者が各配布元の条件に同意して入手します）

| モデル | 用途 | ライセンス | 入手先 / 参照 |
|---|---|---|---|
| MiniMax H3（`minimax_h3_ref2va_pruned_int8_convrot.safetensors`、`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`、`minimax_h3_video_vae_fp16.safetensors`、`minimax_h3_audio_vae_fp32.safetensors`） | 動画・音声生成の本体 | **MiniMax H3 Community License Agreement**（年間売上 2,000 万米ドル超の商用利用は別途許諾、Acceptable Use Policy と適用地域の制限あり） | https://huggingface.co/Comfy-Org/MiniMax-H3 、ライセンス本文 https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE |
| MiniMax H3 Turbo LoRA（`minimax_h3_turbo_v4_step600_ema.safetensors`） | FAST/LONG_FAST の高速化 | Apache-2.0（モデルカード記載） | https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora |
| Gemma 4 系 GGUF（既定設定の `gemma-4-E4B-it-*.gguf`、LM Studio 経由の `google/gemma-4-12b-qat` など） | ローカル LLM（監督案・人物解析） | **Gemma Terms of Use**（https://ai.google.dev/gemma/terms ）と Prohibited Use Policy。第三者の派生（量子化・微調整）版はその配布元の表記も確認 | Google / 各配布元 |
| RealESRGAN（`RealESRGAN_x4plus.pth`） | AI 高画質化（任意） | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| SeedVR2（`seedvr2_ema_7b_fp16.safetensors`、`ema_vae_fp16.safetensors`） | AI 高画質化（任意） | Apache-2.0（モデルカード記載） | https://huggingface.co/numz/SeedVR2_comfyUI |

### MiniMax H3 に関する表示

MiniMax H3 Community License Agreement は、再配布時のライセンス提供と NOTICE の同梱、商用製品 UI での「MiniMax H3」表示を求めています。本アプリは MiniMax H3 専用であり、名称と画面に「MiniMax H3」を明示しています。

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

### Gemma に関する表示

> Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

### ライセンス表記のない外部ノードについて

- **ComfyUI_Comfyroll_CustomNodes への依存は解消済み**: 参照画像のコンタクトシート作成（`CR Image Grid Panel`）を ComfyUI 標準の `ImageStitch` ノードに置き換えました（2026-09-14、実 ComfyUI 0.34.5 で 1〜4 枚の合成を確認）。導入不要です。
- **ComfyUI-llama-cpp は未解決**: 配布元にライセンス表記がありません（上表）。H3 はこのパックを同梱・再配布せず、ComfyUI 上のノード名で呼び出すだけですが、「同梱しないこと」と「利用条件が明確であること」は別です。利用は各配布元の条件（表記が無い場合は著作権者の許諾範囲）に従って各自で判断してください。ライセンス表記を依頼する問い合わせ文案は `docs/LICENSE_INQUIRY_DRAFT.md` にあります（未送信）。このパックを使わない構成（LM Studio / Ollama / llama.cpp server / vLLM / LocalAI / 外部 API を接続先にし、Gemma フォールバックを OFF）でも H3 は動作します。

## 4. 出力物について

生成した動画・音声の利用条件は、使用したモデル（特に MiniMax H3 の Acceptable Use Policy と適用地域）と、参照画像・台詞の権利に従います。H3 はこれらの権利を付与しません。
