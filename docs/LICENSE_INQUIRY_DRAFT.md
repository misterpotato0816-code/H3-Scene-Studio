# ライセンス表記の確認依頼（文案・未送信）

対象: ComfyUI-llama-cpp（https://github.com/lihaoyun6/ComfyUI-llama-cpp_vlm 、Comfy Registry 名 `comfyui-llama-cpp`）

H3 は `llama_cpp_model_loader` / `llama_cpp_parameters` / `llama_cpp_instruct_adv` / `llama_cpp_unload_model` の 4 ノードを
ComfyUI 上で呼び出しています（コードの同梱・改変・再配布はしていません）。2026-09-14 時点で、リポジトリの GitHub 表示・
Comfy Registry・`pyproject.toml`（`license = {file = "LICENSE"}` と記載）・README のいずれにもライセンス種別が示されておらず、
参照されている `LICENSE` ファイルも収録されていません。

送信するかどうか、送信先（GitHub Issue か）は利用者の判断です。以下は Issue 用の文案です。

---

**Title:** License file referenced in pyproject.toml is missing — could you clarify the license?

Hello, and thank you for ComfyUI-llama-cpp.

I maintain H3 Scene Studio, an open-source (MIT) ComfyUI front end for MiniMax H3 that uses your
`llama_cpp_model_loader`, `llama_cpp_parameters`, `llama_cpp_instruct_adv` and `llama_cpp_unload_model`
nodes to run a local VLM inside ComfyUI. We do not bundle or modify your code; users install your node pack
themselves and our documentation points them to this repository.

While preparing the public release I noticed that `pyproject.toml` declares `license = {file = "LICENSE"}`,
but the repository does not contain a `LICENSE` file, and no license is shown on GitHub or the Comfy Registry.
Could you add the license file (or state the license in the README)? Knowing the terms would let us tell our
users clearly under which conditions they may install and use your nodes, and let us credit you correctly in
our third-party notices.

Thank you for your time.

---

## 送信しない場合の扱い

- H3 の公開版では、このパックが「ライセンス表記なし」であることを README / SETUP / THIRD_PARTY_NOTICES に明記し、導入は各自の判断とする。
- このパックを使わない構成（LM Studio / Ollama / llama.cpp server / vLLM / LocalAI / 外部 API を接続先にし、「失敗した場合はComfyUI内Gemmaで続行する」を OFF）で H3 は動作する。
