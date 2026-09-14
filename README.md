# H3 Scene Studio — MiniMax H3 動画制作アプリ

ローカル PC の ComfyUI と MiniMax H3 モデルを使って、参照画像と日本語の指示・台詞から短い縦動画（音声付き）を作るためのアプリです。
人物の一貫性保持、続き生成（ストーリー）、日本語 → 英語プロンプトの自動変換、生成後の高画質化までをブラウザーの画面から操作します。
ComfyUI 本体・カスタムノード・モデルは同梱しません。導入手順は **[docs/SETUP_JA.md](docs/SETUP_JA.md)** を参照してください。

> H3 自身のコードと文書は **MIT License**（[LICENSE](LICENSE)）です。外部ソフトウェアとモデルの条件は
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています（ComfyUI GPL-3.0、MiniMax H3 Community License、Gemma Terms of Use など。MIT はこれらを上書きしません）。

## できること（初期公開版）

- **単発生成**: 参照画像（最大 4 枚）+ 日本語の演出指示 + 台詞から、5〜15 秒の縦動画（音声付き）を生成。ローカル LLM が日本語を H3 用の英語プロンプトへ変換し、指定した台詞以外が混入しないよう検証します。
- **ストーリーモード**: 複数セグメントを連続生成し、前クリップの末尾を引き継いで人物・カメラ・場所を継続。結合時に「自然につなぐ」（接続位置の選択と輝度ランプ、音声の同期トリム）/ カット / フェードを選べます。
- **高画質化**: 標準拡大（FFmpeg Lanczos、追加モデル不要）と AI 高画質化（RealESRGAN / SeedVR2、要モデル）を区別して実行。横動画を縦に引き伸ばしません。
- **設定画面**: GPU の割り当て（自動 / シングル / デュアル）、LLM 接続先（ローカル 7 種 / 外部 API 6 種、モデル選択、API キーは OS の資格情報ストア）、生成動画の保存先。
- **起動と終了**: `RUN_H3.bat` で起動、画面の「H3を終了」または `STOP_H3.bat` で終了（モデル解放 → H3 が起動した ComfyUI の終了 → 設定により LM Studio の終了 → ポート解放）。名前による一括終了は行いません。
- **音声テスト**: 任意の VOICEVOX ENGINE で台詞の読み・アクセントを試聴（H3 の音声には適用されません）。

## 必要な環境（概要）

- Windows 11、NVIDIA GPU（動作確認は 12 GB 級 GPU 2 枚構成。1 枚構成は「シングル」設定で対応しますが検証は限定的です）
- ComfyUI 0.34 以降（MiniMax H3 ノードを含む版）と、その Python 環境
- FFmpeg / ffprobe、`requirements.txt` の Python パッケージ
- モデル: MiniMax H3 本体・VAE・Turbo LoRA、ローカル LLM 用の GGUF（Gemma 4 系）。任意: RealESRGAN、SeedVR2、LM Studio
- 詳細と入手先: [docs/SETUP_JA.md](docs/SETUP_JA.md)、条件: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## 既知の制約（正直に）

- **日本語音声**: 短い呼びかけや感嘆の抑揚・リズムが不自然になる場合があります（例：『見て見て！』が平坦に発音される）。台詞の文字起こしが一致しても、イントネーションの自然さは保証しません。外部音声エンジンへの切り替えは未実装です。
- **ストーリーの連続性**: 人物・背景・つなぎ目は生成結果によりばらつきます。別ショットになった場合は結合結果に「要確認」と表示するので、該当クリップを再生成してください。「自然につなぐ」は輝度段差と冒頭の停止感を軽減しますが、構図の差までは埋めません。
- **高画質化の速度とメモリ**: 方式と環境で大きく異なります。検証機（RTX 4070 Ti 12 GB + RTX 3060 12 GB、576×1024 → 1080×1920、248 フレーム）では Lanczos 約 2 秒、RealESRGAN 約 13 分、SeedVR2 7B fp16 約 53 分（ピーク VRAM 約 5 GB、全ブロック CPU オフロード）でした。他の環境での所要時間・成否は未検証です。
- **LLM 接続先**: 実接続を確認したのは **OpenCode Go** と **LM Studio** です。OpenAI / Anthropic / Google Gemini / OpenRouter / 外部 OpenAI 互換、Ollama / llama.cpp server / vLLM / LocalAI は**モックテストのみ**（実サーバー・実キーでは未確認）。
- **環境**: 検証は開発機 1 台（Windows 11、上記 GPU）で行いました。他の GPU 構成・OS では未検証で、動作を保証しません。新規環境での確認は、依存関係の導入・設定読み込み・オフラインの自己診断までです（動画生成は未確認）。
- **外部ノードのライセンス**: ComfyUI 内蔵 LLM に使う ComfyUI-llama-cpp は配布元にライセンス表記が見当たりません。導入は各自の判断でお願いします。使わない構成（LM Studio 等を接続先にし、Gemma フォールバックを OFF）でも動作します（[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `app/` | アプリ本体（aiohttp サーバー、Web UI、`h3app` パッケージ、静的テスト） |
| `custom_nodes/H3-Device-Barrier/` | ComfyUI 側に置く H3 用カスタムノード（GPU 割り当ての検証） |
| `backend/h3_v2/` | v2 生成バックエンドと R&D 用スクリプト（`dbg_*.py` / `ab_run.py` は開発機の環境に依存） |
| `workflows/`, `build/` | Phase 1 の ComfyUI ワークフローとビルダー（開発記録） |
| `_hetero_test/` | 異種 GPU 実験の記録とスクリプト（公開版には KJNodes の改変コピーを含めません） |
| `docs/` | 導入手順（`SETUP_JA.md`）、設計・監査・実機検証の記録 |
| `RUN_H3.bat` / `STOP_H3.bat` | 起動・終了 |
| `RUN_H3_V2.bat`, `RUN_*.ps1` | 旧ランチャーと計測用ランナー（開発機のパスが埋め込まれています。使う場合は編集が必要） |

## 開発記録

以下は Phase 1（ワークフロー構築時点）の記録です。当時の検証機のパスは `<H3_PROJECT_ROOT>` / `<COMFYUI_DIR>` に置き換えています。
その後の監査と実機検証は [docs/AUDIT_2026-09-12.md](docs/AUDIT_2026-09-12.md) と [docs/VERIFICATION_2026-09-14.md](docs/VERIFICATION_2026-09-14.md) にあります（実機の計測値・未確認事項を含む）。

---

ComfyUI 単体で使う、MiniMax H3 専用ワークフローです（HACO 非依存）。
中核は **① 人物の一貫性保持 / ② 動画の続き生成 / ③ 日本語 → 英語 H3 プロンプト自動変換** の3つ。

> **品質目標：** YouTube ショート等で見かける完成度の高い短尺動画と同程度の
> 見た目・人物一貫性・動き・音声・編集品質を、ローカル H3 で達成すること。
> **投稿・公開・配信地域は要件に含みません**（縦 9:16 はその品質水準に合わせた画面仕様です）。

- **Phase 1 実装済み**（このリポジトリの `workflows/`）
- Phase 2 は [docs/04_roadmap.md](docs/04_roadmap.md)

---

## 納品物

| ファイル | 内容 |
|---|---|
| `workflows/H3_phase1_core.json` | 94ノード / 102リンク。中核3機能 + 続き生成 + 音声分離の骨格 |
| `workflows/H3_phase1_full.json` | 110ノード / 118リンク。core + BGM合成 + アップスケール + Whisper検証 |
| `build/` | 上記 JSON の生成スクリプト（再生成・改造用） |
| [docs/01_structure.md](docs/01_structure.md) | 構造説明・グループ構成・データフロー |
| [docs/02_usage.md](docs/02_usage.md) | 実行手順 / Character Master の使い方 / continuation の使い方 |
| [docs/03_missing.md](docs/03_missing.md) | **不足モデル・不足ノード・動作未確認部分** |
| [docs/04_roadmap.md](docs/04_roadmap.md) | Phase 2 以降の改善候補 |
| [docs/99_h3_model_facts.md](docs/99_h3_model_facts.md) | ComfyUI 本体ソースから確認した H3 のハード仕様 |

ワークフローは ComfyUI の workflows フォルダにもコピー済みです（下記「変更したファイル」参照）。

---

## 検証状況（実際に確認できたことだけ）

| 検証 | 方法 | 結果 |
|---|---|---|
| ノード型がすべて実在する | 実機 ComfyUI の `/object_info`（2039ノード）と突合 | ✅ 全ノード実在 |
| リンク・スロット整合 | 静的バリデータ `build/validate.py` | ✅ **0 error** |
| widgets_values の個数・順序 | スキーマから自動生成 + 実ワークフローの実例と照合 | ✅ 0 error |
| **フロントエンドで開ける** | ComfyUI 1.47.12 に実際にロード | ✅ **94/94 ノード, 102/102 リンク, console error 0**（full も 110/110, 118/118） |
| 実行グラフへの変換 | `app.graphToPrompt()` | ✅ バイパス解決も含めて正常（core 69ノード / full+全有効 104ノード） |
| 続き生成の結線 | 45グループを有効化して再変換 | ✅ `ref_video_0` ← ImageFromBatch、`ref_video_audio_0` ← TrimAudioDuration |
| サーバ側 prompt 検証 | `POST /api/prompt` | ⚠️ **未導入モデル4件のみエラー**。他 65 ノードは全て通過 |
| **実際に動画を生成** | — | ❌ **未実施**（H3 Ref2VA 本体が未導入のため物理的に不可） |

> 最後の行が重要です。**まだ1本も生成していません。**
> グラフとして正しいことは確認済みですが、生成品質・速度・VRAM 収支は未検証です。

---

## いま何が動いて、何が動かないか

| 機能 | 現状 |
|---|---|
| Character Master（参照画像の取り込み・整形・シート化） | ✅ 今すぐ動く |
| Character Profile 抽出（VLM） | ✅ 今すぐ動く（Gemma-4 E4B 導入済み） |
| 日本語 → H3 英語プロンプト生成 | ✅ 今すぐ動く |
| H3 での動画生成 | ❌ **モデル2点の DL 待ち**（[docs/03_missing.md](docs/03_missing.md)） |
| 続き生成 / Tail Relay | ⚠️ 結線は完了。H3 が動けば使える |
| 音声分離・BGM 合成 | ⚠️ 同上（ノードはすべて導入済み） |

つまり **20〜40 グループ（VLM部分）は今日から試せます。**
先にプロンプト生成の品質を詰めておくと、モデルが揃ったときすぐ回せます。

---

## 変更したファイル / 作成したファイル

### 既存 ComfyUI 環境への変更（追加のみ・上書きなし）

```
<COMFYUI_DIR>\user\default\workflows\H3\
  ├── H3_phase1_core.json   (新規)
  └── H3_phase1_full.json   (新規)
```

- **既存ファイルの上書き・削除は一切していません。** バックアップ不要です。
- 調査のため CPU モードの ComfyUI を **一時的に** ポート 8399 で起動し、検証後に停止しました。
  通常使う Desktop 版の設定・モデル・カスタムノードには触れていません。
- 検証中に一度だけ `POST /api/prompt` を送りました。未導入モデルで即座に失敗する部分実行で、
  出力ファイルは生成されていません（確認済み）。プロセスは停止済みです。

### 新規作成（`<H3_PROJECT_ROOT>`）

```
<H3_PROJECT_ROOT>\
├── README.md
├── workflows\
│   ├── H3_phase1_core.json
│   └── H3_phase1_full.json
├── build\
│   ├── wfbuild.py        スキーマ駆動のワークフロービルダ
│   ├── build_phase1.py   グラフ定義（ここを編集して再生成）
│   ├── prompts.py        システムプロンプト・既定テキスト
│   └── validate.py       静的バリデータ
└── docs\
    ├── 01_structure.md
    ├── 02_usage.md
    ├── 03_missing.md
    ├── 04_roadmap.md
    └── 99_h3_model_facts.md
```

---

## 再生成のしかた

グラフを直したいときは JSON を手で触らず、`build/build_phase1.py` を編集して再生成してください。
ノードのスロット番号や widget 順はスキーマから自動導出されるため、手書きよりずれません。

```bash
python <H3_PROJECT_ROOT>/build/build_phase1.py <object_info.json> <H3_PROJECT_ROOT>/workflows
```

`object_info.json` は動作中の ComfyUI から取得します。

```bash
curl -s http://127.0.0.1:8188/object_info -o object_info.json
```

生成後は必ずバリデータを通してください。

```bash
python <H3_PROJECT_ROOT>/build/validate.py object_info.json <H3_PROJECT_ROOT>/workflows/H3_phase1_core.json
```

---

## 設計方針（なぜこうしたか）

1. **H3 本命。Krea2 / LTX は考え方だけ借りた。**
   Krea2 の「人物参照と背景参照を分ける」発想は Character Master の4スロット構成に、
   LTX の FaceID 的な同一性維持は Character Profile のテキスト固定に、
   LTX のリップシンク検証の発想は Whisper による出力検証に置き換えています。
   モデルそのものは一切持ち込んでいません（追加 30GB の DL を避けるため）。

2. **追加カスタムノードは 0。**
   記事版が使っていた `ImpactMakeImageList` / `ImageListToBatch+` /
   `RTXVideoSuperResolution` / `LTXVLoopingSampler` などは、
   導入済みパックと ComfyUI 本体 v0.30.2 の標準ノードで置き換えました。
   ただし「増やさない」ことを目的化はしていません。Phase 2 で有用なら足します
   （候補は [docs/04_roadmap.md](docs/04_roadmap.md)）。

3. **SetNode / GetNode を使っていない。**
   フロントエンド専用の仮想ノードで、生成物として正しさを検証しにくいためです。
   配線は長くなりますが、全リンクが実際にロードされることを確認できています。

4. **1クリップの尺は 5〜15秒を基本に、可変。**
   H3 のノード tooltip 記載の学習済みレンジが 124〜362 フレーム（24fps）です。
   その外も入力自体は可能なので、上限は固定せず「秒数」ノードで調整できるようにしました。

5. **参照画像は高解像度ほど有利、ただし必須ではない。**
   `ref_image_size = max` は短辺 2048 まで使いますが**拡大はしません**。
   参照が小さいと `max` にしても効果が出にくい、という性質があります
   （[docs/99_h3_model_facts.md](docs/99_h3_model_facts.md) §5.1）。
   既定は 顔1536 / 上半身1152 / 全身1152 / 衣装896 px（長辺）にしています。
