# 05. モデル導入前検証（ダウンロード実行前）

ご指定の各モデルについて、ファイル名・入手元・容量・記事WFとの一致を機械的に確認した結果です。
**この時点ではまだ1件もダウンロードしていません。**

---

## A. ライセンス確認（先に実施）

### 記事添付 `QA-about-License.md` と実ライセンス本文が食い違っていました

添付資料の見出しは
「Why is MiniMax-H3's open-weight license currently **limited to** the EU, UK, South Korea, and US?」
と読め、**その4地域だけが許可**されているように見えます。

しかし HuggingFace の `MiniMaxAI/MiniMax-H3` の LICENSE 本文（MiniMax H3 Community License Agreement）は逆でした。

```
"Applicable Territory" means worldwide, excluding the Excluded Territories.

"Excluded Territories" means the European Union, the United Kingdom,
the Republic of Korea and the United States of America.
```

```
V.4  You may not use, reproduce, modify, distribute, or display the
     MiniMax H3 Works or any of their Outputs or results outside the
     Applicable Territory.
```

モデルカードにも `Application form (only for USA/EU/UK/South Korea)` とあり、
**EU / UK / 韓国 / 米国が「除外地域」で、そこでは個別申請が必要**という趣旨で一貫しています。
添付資料の見出しが誤解を招く表現だったと判断しました（本文の趣旨は LICENSE と一致）。

### 本機の所在地

```
TimeZone     : Tokyo Standard Time (UTC+09:00)
HomeLocation : Japan (GeoId 122)
SystemLocale : ja-JP
```

**日本は Excluded Territories に含まれない = Applicable Territory 内**。
オープンウェイトの取得・利用について、ライセンス上の障害は確認されませんでした。

### 参考記録（今回の設計条件ではありません）

V.4 は Outputs（生成物）にも及ぶ書き方になっており、
Applicable Territory の外での display を制限していると読めます。

**本プロジェクトの目的は「YouTube ショートで見かける完成度の短尺動画と同等の品質を
ローカル H3 で達成すること」であり、投稿・公開・配信地域は要件に含みません。**
したがって公開を前提としたライセンス運用判断は設計条件から除外します。

上記は調査記録としてのみ残します（将来公開を検討する場合の出発点として）。
なお私は法律の専門家ではないため、この解釈は判断材料であって結論ではありません。

> Turbo LoRA（larryvrh 版）は別ライセンスで **apache-2.0** でした。

---

## B. ファイル検証結果

すべて HuggingFace API で実ファイルの存在とバイト数を確認済み。**推測・類似名の代用はしていません。**

| # | ファイル名（確認済み・完全一致） | 入手元 | 実サイズ | 保存先 |
|---:|---|---|---:|---|
| 1 | `MiniMax-H3-Ref2VA-Q4_K_S.gguf` | `Abiray/MiniMax-H3-GGUF` → `unet/` | 19,853,333,792 B<br>**18.49 GiB** | `models\unet\` |
| 2 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3` → `diffusion_models/` | 20,970,379,616 B<br>**19.53 GiB** | `models\diffusion_models\` |
| 3 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `Comfy-Org/MiniMax-H3` → `text_encoders/` | 27,141,342,152 B<br>**25.28 GiB** | `models\text_encoders\` |
| 4 | `minimax_h3_turbo_v4_step600_ema.safetensors` | `larryvrh/MiniMax-H3-Turbo-Lora` | 779,849,816 B<br>**0.73 GiB** | `models\loras\` |
| 5 | `minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors` | `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI` | 620,285,592 B<br>**0.58 GiB** | `models\loras\` |
| | | | **合計 64.61 GiB** | |

- ディスク空き E: 2,850 GB → **容量問題なし**
- 二重ダウンロードの確認：3系統（`ComfyUI-Shared\models` / `ComfyUI\ComfyUI\models` / `E:\AI\Models`）を
  全走査済み。上記5点はいずれも**未導入**。既存ファイルの削除・上書きは発生しません
- 保存先は `ComfyUI-Shared\models`（`shared_model_paths.yaml` の既定）配下

### 記事WFとの一致状況

| # | 記事WFでの扱い |
|---:|---|
| 1 | ❌ 記事は GGUF を使っていない（ご指定による追加） |
| 2 | ✅ 記事 `UNETLoader` の**DLメタデータURL**と完全一致 |
| 3 | ✅ 記事 `CLIPLoader` #128 / `ClipLoaderGGUF` #43157 の**実選択値**と完全一致 |
| 4 | ✅ 記事 `LoraLoaderModelOnly` #5956 の実選択値と完全一致（strength 1.0） |
| 5 | ✅ 記事 `LoraLoaderModelOnly` #43139 の実選択値と完全一致（strength 0.5） |

> #5 について：同リポジトリには `minimax_h3_turbo_4step_**ema_**ckpt850_pruned_comfyui.safetensors`
> という**別ファイル**（620,285,592 B、同容量）も存在します。
> 記事の値は `_ema_` **なし**なので、そちらを使います。代用はしません。

---

## C. 実行前に判断が必要な点（サイズ前提の訂正）

### C-1. `Q4_K_S` は 12GB VRAM に載りません

ご指定時の想定（12GB VRAM 環境の本命）と実サイズが食い違います。
**私が前回「約11〜12GB」と書いたのが誤りでした。訂正します。**

`Abiray/MiniMax-H3-GGUF` の `unet/` は **pruned でない本体（bf16 66.28 GiB）** の量子化です。

| quant | サイズ |
|---|---:|
| Q3_K_S / Q3_K_M | 14.50 GiB |
| Q4_0 | 17.36 GiB |
| **Q4_K_S / Q4_K_M** | **18.49 GiB** |
| Q5_K_S / Q5_K_M | 22.25 GiB |
| Q6_K | 26.28 GiB |

18.49 GiB は 12GB VRAM に収まらず、**CPU オフロード前提**になります。

別に `Abiray/MiniMax-H3-Pruned-GGUF`（pruned bf16 40.23 GiB 由来）があり、
そちらの Q4 系なら 11〜12 GiB 前後になると見込まれます（未検証）。
**ただし別ファイル名なので、ご指示の「類似名で代用しない」に従い勝手には切り替えません。**

### C-2. テキストエンコーダ 25.28 GiB

`int8_convrot` は 25.28 GiB。これも 12GB には載らず、ほぼ全量 CPU オフロードになります。

DiT 18.49 + TE 25.28 = **43.8 GiB** の重みを 64 GB RAM で回すことになります。
Phase 1 はフェーズ分離（VLM→解放→TE→解放→サンプリング）を実装済みなので
動作の見込みはありますが、**かなり遅くなる**と予想されます。

参考：同リポジトリの `nvfp4_awq` は 15.69 GiB（＝導入済みの 14.61 GB と一致）。
R2V 記事が挙げていた GGUF `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf`（約19 GiB）も選択肢です。

### C-3. Turbo LoRA と本体の組み合わせが未確認

`drbaph` の README によれば `_pruned_comfyui` 系は
「**pruned / curve-form モデル**使用時に ComfyUI 内蔵の H3 LoRA ローダーで読めるように変換したもの」です。

記事は次の組み合わせでした。

| LoRA | 変換 | strength |
|---|---|---:|
| `minimax_h3_turbo_v4_step600_ema`（larryvrh 生） | **未変換** | 1.0 |
| `minimax_h3_turbo_4step_ckpt850_pruned_comfyui`（drbaph） | **pruned用に変換済** | 0.5 |

- #1（**非** pruned の GGUF）と組む場合、`_pruned_comfyui` 側が合わない可能性
- #2（pruned int8_convrot）と組む場合、larryvrh の生 LoRA 側が合わない可能性

どちらが正しい組み合わせかは、実際にロードしてみないと判断できません。

---

## D. 判断をお願いしたい点

C-1〜C-3 を踏まえ、どれで進めるかご指示ください。**ご返答があれば即座に着手します。**

| 案 | 内容 | DL量 |
|---|---|---:|
| **案1（指示どおり）** | 表Bの5点をそのまま取得。DiT は CPU オフロードで回す | 64.6 GiB |
| **案2（12GB実効重視）** | 表Bのうち #1 を `Abiray/MiniMax-H3-Pruned-GGUF` の Q4 系に差し替え（要ファイル名確認）。#2 は pruned なので LoRA #5 と整合しやすい | 約 57 GiB |
| **案3（まず最小で通す）** | #2 pruned int8_convrot + #3 TE + LoRA #4#5 のみ先に取得し、Test 1〜3 まで通してから GGUF を判断 | 46.1 GiB |

私としては、**案3 → 動いたら案1の GGUF を追加**が、無駄なダウンロードが少なく実測も早く出ると考えます。
記事WFと同じ pruned int8_convrot 構成なので、LoRA の組み合わせも記事準拠になり C-3 の不確実性が減ります。

「案1で」等ひと言いただければ、そのまま取得 → 再ビルド → 検証 → Test 1〜5 まで進めます。
