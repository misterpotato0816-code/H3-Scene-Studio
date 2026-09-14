// WP-A UI regression: pure helpers from app/web/settings.js.
// Verifies the hierarchy logic the LLM tab renders from:
// kind -> provider -> model, override resolution, summary text,
// test-result row, and unsaved-change (draft badge) detection.
// Run: node app/tests/llm_settings_ui.test.cjs
"use strict";
const assert = require("assert/strict");
const path = require("path");
const settings = require(path.join(__dirname, "..", "web", "settings.js"));

const SPECS = [
  { id: "comfy_gemma", label: "ComfyUI内ローカルGemma", kind: "local" },
  { id: "lmstudio", label: "LM Studio", kind: "local" },
  { id: "ollama", label: "Ollama", kind: "local" },
  { id: "opencode_go", label: "OpenCode Go", kind: "external" },
  { id: "openai", label: "OpenAI", kind: "external" },
];

function draft() {
  return {
    schema: 2,
    connection: { kind: "local", provider: "lmstudio" },
    roles: {
      director: { override: false, provider: "", model: "" },
      character_profile: { override: false, provider: "", model: "" },
    },
    providers: {
      lmstudio: { base_url: "http://127.0.0.1:1234/v1", model: "m-local", vision_models: [], extra: {} },
      ollama: { base_url: "http://127.0.0.1:11434", model: "m-ollama", vision_models: [], extra: {} },
      opencode_go: { base_url: "", model: "m-go", vision_models: [], extra: {} },
    },
    gemma_fallback: true,
    max_attempts: 3,
    favorites: [],
  };
}

// Kind labels are Japanese, providers filter by kind.
assert.equal(settings.llmKindLabel("local"), "ローカル");
assert.equal(settings.llmKindLabel("external"), "外部API");
assert.deepEqual(
  settings.llmProvidersForKind(SPECS, "local").map((p) => p.id),
  ["comfy_gemma", "lmstudio", "ollama"]);
assert.equal(settings.llmProviderSpec(SPECS, "openai").label, "OpenAI");

// Connection is the default; override wins for the profile role.
let d = draft();
assert.deepEqual(settings.llmEffective(d, "director"), { provider: "lmstudio", model: "m-local" });
assert.deepEqual(settings.llmEffective(d, "character_profile"), { provider: "lmstudio", model: "m-local" });
d.roles.character_profile = { override: true, provider: "opencode_go", model: "m-go" };
assert.deepEqual(settings.llmEffective(d, "character_profile"), { provider: "opencode_go", model: "m-go" });
// Director still follows the connection after the override changes.
assert.deepEqual(settings.llmEffective(d, "director"), { provider: "lmstudio", model: "m-local" });

// Summary line shows the current selection (no vague wording).
d = draft();
assert.equal(
  settings.llmSummaryText(d, SPECS),
  "ローカル › LM Studio › m-local");
d.roles.character_profile = { override: true, provider: "opencode_go", model: "m-go" };
assert.equal(
  settings.llmSummaryText(d, SPECS),
  "ローカル › LM Studio › m-local ／ 人物解析: OpenCode Go › m-go");

// Switching providers keeps the other providers' saved values.
d = draft();
d.connection = { kind: "local", provider: "ollama" };
assert.equal(d.providers.lmstudio.model, "m-local");
assert.equal(settings.llmEffective(d, "director").model, "m-ollama");

// Unsaved selection is detected (dirty badge) instead of silently reverting.
const saved = draft();
const edited = draft();
edited.connection = { kind: "external", provider: "opencode_go" };
assert.equal(settings.isDraftDirty(saved, edited), true);
assert.equal(settings.isDraftDirty(saved, draft()), false);

// Test-result row shows target / model / text / image in one line.
assert.equal(
  settings.llmTestResultText({ ok: true, target: "http://127.0.0.1:1234/v1", model: "m-local", text_ok: true, image_ok: true, error: "" }),
  "接続先: http://127.0.0.1:1234/v1 ／ モデル: m-local ／ テキスト: OK ／ 画像: 対応");
assert.equal(
  settings.llmTestResultText({ ok: true, target: "host", model: "m", text_ok: true, image_ok: null, error: "" }),
  "接続先: host ／ モデル: m ／ テキスト: OK ／ 画像: 未確認");
assert.equal(
  settings.llmTestResultText({ ok: false, message: "先にモデルを選んでください。" }),
  "先にモデルを選んでください。");
assert.equal(settings.llmImageSupportLabel(false), "非対応");

console.log("PASS: llm_settings_ui (10 assertions)");
