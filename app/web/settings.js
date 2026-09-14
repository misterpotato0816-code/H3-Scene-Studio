// H3 Settings dialog: 保存先 / GPU / LLM接続 tabs.
// Plain ES2020, no build step, no CDN. Shares $ / api / postJson / toast /
// state from app.js (classic scripts share the top-level lexical scope).
"use strict";

// ------------------------------------------------------------- pure utils --
// Kept dependency-free (no DOM) so they can be exercised from plain Node
// scripts via require() in addition to `node --check`.

function stableStringify(value) {
  const seen = (k, v) => {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const out = {};
      Object.keys(v).sort().forEach((key) => { out[key] = v[key]; });
      return out;
    }
    return v;
  };
  return JSON.stringify(value, seen);
}

function isDraftDirty(saved, draft) {
  if (!saved || !draft) return false;
  return stableStringify(saved) !== stableStringify(draft);
}

const LLM_STATUS_CHIPS = {
  connected: { label: "接続OK", cls: "ok" },
  model_not_loaded: { label: "モデル未ロード", cls: "ng" },
  model_not_found: { label: "モデル未ロード", cls: "ng" },
  invalid_url: { label: "設定不備", cls: "ng" },
  auth_required: { label: "認証が必要", cls: "ng" },
  vision_unsupported: { label: "画像非対応", cls: "ng" },
  unreachable: { label: "未接続", cls: "" },
  timeout: { label: "未接続", cls: "" },
  error: { label: "未接続", cls: "" },
};

function llmStatusChipInfo(status) {
  return LLM_STATUS_CHIPS[status] || { label: "未接続", cls: "" };
}

const GPU_MODE_LABELS = { auto: "自動", single: "シングルGPU", dual: "デュアルGPU" };
function gpuModeLabel(mode) { return GPU_MODE_LABELS[mode] || mode || "-"; }

function shortUuid(uuid) {
  if (!uuid) return "-";
  const parts = String(uuid).split("-");
  if (parts.length >= 2 && parts[0].toUpperCase() === "GPU") {
    return `${parts[0]}-${parts[1]}…`;
  }
  return uuid.length > 13 ? `${uuid.slice(0, 13)}…` : uuid;
}

function fmtMiBValue(mib) {
  if (mib === null || mib === undefined) return "-";
  const n = Number(mib);
  if (!Number.isFinite(n)) return "-";
  return `${n.toLocaleString("ja-JP")} MiB`;
}

function fmtHHMM(date) {
  const d = date || new Date();
  try {
    return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return "";
  }
}

// ------------------------------------------------------- WP-A pure utils --
// Schema-2 draft helpers. Dependency-free (no DOM): exercised from plain
// Node in app/tests/llm_settings_ui.test.cjs in addition to `node --check`.
const LLM_KIND_LABELS = { local: "ローカル", external: "外部API" };

function llmKindLabel(kind) {
  return LLM_KIND_LABELS[kind] || kind || "-";
}

function llmProviderSpec(providers, id) {
  return (providers || []).find((p) => p && p.id === id) || null;
}

function llmProvidersForKind(providers, kind) {
  return (providers || []).filter((p) => p && p.kind === kind);
}

function llmEffective(draft, role) {
  // Effective {provider, model} for one role of a schema-2 draft.
  // override=false (or an empty override provider) follows the connection.
  if (!draft) return { provider: "", model: "" };
  const conn = draft.connection || {};
  let provider = conn.provider || "";
  let model = (((draft.providers || {})[provider]) || {}).model || "";
  const roleCfg = ((draft.roles || {})[role]) || {};
  if (roleCfg.override && roleCfg.provider) {
    provider = roleCfg.provider;
    model = roleCfg.model || "";
  }
  return { provider, model };
}

function llmSummaryText(draft, providers) {
  const label = (id) => {
    const spec = llmProviderSpec(providers, id);
    return (spec && spec.label) || id || "-";
  };
  if (!draft) return "-";
  const conn = draft.connection || {};
  const entry = ((draft.providers || {})[conn.provider]) || {};
  const main = `${llmKindLabel(conn.kind)} › ${label(conn.provider)} › ${entry.model || "（モデル未選択）"}`;
  const prof = ((draft.roles || {}).character_profile) || {};
  if (prof.override && prof.provider) {
    return `${main} ／ 人物解析: ${label(prof.provider)} › ${prof.model || "（モデル未選択）"}`;
  }
  return main;
}

function llmImageSupportLabel(v) {
  return v === true ? "対応" : v === false ? "非対応" : "未確認";
}

function llmTestResultText(result) {
  // "接続先 / モデル / テキスト対応 / 画像対応" in one summary line.
  if (!result) return "";
  const bits = [];
  if (result.target) bits.push(`接続先: ${result.target}`);
  if (result.model) bits.push(`モデル: ${result.model}`);
  if (result.text_ok !== undefined) bits.push(`テキスト: ${result.text_ok ? "OK" : "NG"}`);
  if (result.image_ok !== undefined || result.ok) {
    bits.push(`画像: ${llmImageSupportLabel(result.image_ok)}`);
  }
  let text = bits.join(" ／ ");
  if (!result.ok && (result.error || result.message)) {
    text += (text ? " ― " : "") + (result.error || result.message);
  }
  return text;
}

// ---------------------------------------------------------------- browser --
if (typeof window !== "undefined") {
  (function () {
    // ===================================================== shared dialog ==
    let activeTab = "storage";
    const TAB_IDS = ["storage", "gpu", "llm"];

    function switchTab(name) {
      if (!TAB_IDS.includes(name)) return;
      activeTab = name;
      TAB_IDS.forEach((id) => {
        const btn = $(`tabBtn${id[0].toUpperCase()}${id.slice(1)}`);
        const panel = $(`tab${id[0].toUpperCase()}${id.slice(1)}`);
        const on = id === name;
        if (btn) {
          btn.classList.toggle("on", on);
          btn.setAttribute("aria-selected", on ? "true" : "false");
        }
        if (panel) panel.hidden = !on;
      });
    }

    function wireTabs() {
      const tablist = document.querySelector("#settingsDialog .tabs");
      TAB_IDS.forEach((id) => {
        const btn = $(`tabBtn${id[0].toUpperCase()}${id.slice(1)}`);
        if (btn) btn.onclick = () => switchTab(id);
      });
      if (tablist) {
        tablist.addEventListener("keydown", (e) => {
          if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
          const ix = TAB_IDS.indexOf(activeTab);
          const next = e.key === "ArrowRight"
            ? TAB_IDS[(ix + 1) % TAB_IDS.length]
            : TAB_IDS[(ix - 1 + TAB_IDS.length) % TAB_IDS.length];
          switchTab(next);
          const nb = $(`tabBtn${next[0].toUpperCase()}${next.slice(1)}`);
          if (nb) nb.focus();
        });
      }
    }

    function anyDirty() {
      return isAiDirty() || isGpuDirty();
    }

    function attemptClose() {
      if (anyDirty()) {
        const ok = window.confirm("保存していない変更があります。破棄して閉じますか？");
        if (!ok) return;
        discardAiDraft();
        discardGpuDraft();
      }
      const dlg = $("settingsDialog");
      if (dlg && dlg.open) dlg.close();
    }

    function openDialog() {
      const dlg = $("settingsDialog");
      if (!dlg) return;
      switchTab(activeTab);
      dlg.showModal();
      loadStorage().catch(() => {});
      loadAiSettings().catch(() => {});
      loadGpu(false).catch(() => {});
    }

    // ==================================================== 保存先タブ =======
    async function loadStorage() {
      try {
        const res = await api("/api/settings/storage");
        renderStorage(res);
      } catch (e) {
        const msg = $("storageMsg");
        if (msg) msg.textContent = (e && e.body && e.body.message) || e.message;
      }
    }

    function renderStorage(res) {
      const set = (id, text) => { const el = $(id); if (el) el.textContent = text; };
      set("storageVideosPath", res.videos_dir || "-");
      set("storageFinalsPath", res.finals_dir ? `完成動画: ${res.finals_dir}` : "");
      set("storageClipsPath", res.clips_dir ? `クリップ: ${res.clips_dir}` : "");
      set("storageRootPath", res.media_root || "-");
      set("storageStagePath", res.input_stage_dir || "-");
      const note = $("storageProcessNote");
      const procNote = res.process && res.process.note;
      if (note) {
        note.textContent = procNote || "";
        note.hidden = !procNote;
      }
    }

    async function openMediaTarget(target) {
      try {
        const res = await postJson("/api/media/open", { target });
        if (res && res.directory && typeof showMediaFolderNotice === "function") {
          showMediaFolderNotice(res.directory);
        }
      } catch (e) {
        toast((e && e.body && e.body.message) || e.message);
        const directory = e && e.body && e.body.directory;
        if (directory && typeof showMediaFolderNotice === "function") showMediaFolderNotice(directory);
      }
    }

    function wireStorage() {
      document.querySelectorAll("#tabStorage [data-open-target]").forEach((btn) => {
        btn.onclick = () => openMediaTarget(btn.dataset.openTarget);
      });
    }

    // ========================================================== GPUタブ ====
    let gpuData = null;   // last GET /api/settings/gpu response
    let gpuDraft = null;  // {mode, video_gpu, aux_device, reserve_vram_gb}
    let gpuValidatePlan = null;
    let gpuValidateTimer = null;

    function normalizeGpuSettings(raw) {
      const s = raw || {};
      return {
        mode: s.mode || "auto",
        video_gpu: s.video_gpu || "",
        aux_device: s.aux_device || "",
        reserve_vram_gb: (typeof s.reserve_vram_gb === "number") ? s.reserve_vram_gb : 1.0,
      };
    }

    function isGpuDirty() {
      if (!gpuData || !gpuDraft) return false;
      return isDraftDirty(normalizeGpuSettings(gpuData.saved), gpuDraft);
    }

    function discardGpuDraft() {
      if (gpuData) gpuDraft = normalizeGpuSettings(gpuData.saved);
    }

    async function loadGpu(refresh) {
      try {
        const res = await api(`/api/settings/gpu${refresh ? "?refresh=1" : ""}`);
        gpuData = res;
        if (!gpuDraft) gpuDraft = normalizeGpuSettings(res.saved);
        renderGpu();
        await validateGpuDraft();
      } catch (e) {
        const msg = $("gpuSaveMsg");
        if (msg) msg.textContent = (e && e.body && e.body.message) || e.message;
      }
    }

    function gpuOptionsHtml(gpus, extra) {
      let html = "";
      (gpus || []).forEach((g) => {
        html += `<option value="${g.uuid}">${g.name} (${shortUuid(g.uuid)})</option>`;
      });
      (extra || []).forEach(([v, t]) => { html += `<option value="${v}">${t}</option>`; });
      return html;
    }

    function renderGpu() {
      if (!gpuData) return;
      const detected = gpuData.detected || { gpus: [] };
      const tbody = $("gpuTableBody");
      if (tbody) {
        tbody.innerHTML = "";
        (detected.gpus || []).forEach((g) => {
          const tr = document.createElement("tr");
          const uuidTd = document.createElement("td");
          uuidTd.textContent = shortUuid(g.uuid);
          uuidTd.title = g.uuid;
          tr.innerHTML =
            `<td>${g.name || "-"}</td><td></td><td>${g.pci_bus_id || "-"}</td>` +
            `<td>${fmtMiBValue(g.total_mib)}</td><td>${fmtMiBValue(g.free_mib)}</td>`;
          tr.children[1].replaceWith(uuidTd);
          tbody.appendChild(tr);
        });
      }
      const ram = $("gpuRamTotal");
      if (ram) ram.textContent = detected.ram_total_mib
        ? `メモリ合計: ${fmtMiBValue(detected.ram_total_mib)}` : "";

      // Mode choices reflect the DRAFT, never the saved settings.
      document.querySelectorAll("#gpuModeChoices .choice").forEach((b) => {
        b.classList.toggle("on", b.dataset.mode === gpuDraft.mode);
      });

      const videoSel = $("gpuVideoSelect");
      const auxSel = $("gpuAuxSelect");
      if (videoSel) {
        videoSel.innerHTML = `<option value="">（自動選択）</option>` + gpuOptionsHtml(detected.gpus);
        videoSel.value = gpuDraft.video_gpu || "";
        videoSel.disabled = gpuDraft.mode === "auto" || !!(gpuData.busy);
      }
      if (auxSel) {
        const auxExtra = [["cpu", "CPU（メインメモリ）"]];
        auxSel.innerHTML = `<option value="">（自動選択）</option>` + gpuOptionsHtml(detected.gpus, auxExtra);
        auxSel.value = gpuDraft.aux_device || "";
        auxSel.disabled = gpuDraft.mode === "auto" || !!(gpuData.busy);
        if (gpuDraft.mode === "single") {
          // Single GPU: aux is always CPU; explain instead of offering a GPU pick.
          auxSel.value = "cpu";
          auxSel.disabled = true;
        }
      }
      const reserve = $("gpuReserve");
      if (reserve) {
        reserve.value = gpuDraft.reserve_vram_gb;
        reserve.disabled = !!(gpuData.busy);
      }

      renderGpuStatus();
      renderGpuBusy();
    }

    function renderGpuStatus() {
      const savedChip = $("gpuStatusSaved");
      const appliedChip = $("gpuStatusApplied");
      const dirtyChip = $("gpuStatusDirty");
      if (savedChip && gpuData) {
        const s = normalizeGpuSettings(gpuData.saved);
        savedChip.textContent = `保存済み: ${gpuModeLabel(s.mode)}`;
        savedChip.className = "status-chip";
      }
      if (appliedChip && gpuData) {
        const ap = gpuData.applied || {};
        appliedChip.textContent = ap.running
          ? `現在適用中: ${ap.owned ? "H3が起動" : "外部プロセス"}`
          : "現在適用中: 停止中";
        appliedChip.className = "status-chip" + (ap.running ? " ok" : "");
      }
      const dirty = isGpuDirty();
      if (dirtyChip) dirtyChip.hidden = !dirty;
      const saveBtn = $("btnGpuSave");
      if (saveBtn) {
        const planOk = !gpuValidatePlan || gpuValidatePlan.ok;
        saveBtn.disabled = !dirty || !planOk || !!(gpuData && gpuData.busy);
      }
      const applyBtn = $("btnGpuApply");
      if (applyBtn && gpuData) {
        applyBtn.disabled = !gpuData.can_apply_now || !!gpuData.busy;
        applyBtn.textContent = "ComfyUIを再起動して適用";
        applyBtn.title = gpuData.can_apply_now ? "" : (gpuData.apply_block_reason || "");
      }
      const appliedInfo = $("gpuAppliedInfo");
      if (appliedInfo && gpuData && gpuData.applied) {
        const ap = gpuData.applied;
        const bits = [];
        if (ap.launch_args && ap.launch_args.length) bits.push(`起動引数: ${ap.launch_args.join(" ")}`);
        if (ap.devices && ap.devices.length) {
          bits.push(`実際のデバイス順: ${ap.devices.map((d) => d.name).join(" / ")}`);
        }
        appliedInfo.textContent = bits.join(" ／ ");
      }
      const restart = $("gpuRestartNotice");
      if (restart) {
        restart.textContent = gpuData && gpuData.restart_reason || "";
        restart.hidden = !(gpuData && gpuData.restart_required);
      }
    }

    function renderGpuBusy() {
      const busyNotice = $("gpuBusyNotice");
      const busy = !!(gpuData && gpuData.busy);
      if (busyNotice) busyNotice.hidden = !busy;
      document.querySelectorAll("#gpuModeChoices .choice").forEach((b) => { b.disabled = busy; });
    }

    async function validateGpuDraft() {
      if (!gpuDraft) return;
      try {
        const res = await postJson("/api/settings/gpu/validate", { settings: gpuDraft });
        gpuValidatePlan = res.plan;
        renderGpuPlan(res.plan);
      } catch (e) {
        const body = (e && e.body) || {};
        gpuValidatePlan = body.plan || { ok: false, errors: [e.message] };
        renderGpuPlan(gpuValidatePlan);
      }
      renderGpuStatus();
    }

    function renderGpuPlan(plan) {
      const roles = $("gpuPlanRoles");
      const errs = $("gpuPlanErrors");
      const warns = $("gpuPlanWarnings");
      if (roles) {
        roles.textContent = plan && plan.roles
          ? plan.roles.map((r) => `${r.role}: ${r.device}`).join(" ／ ") : "";
      }
      if (errs) errs.textContent = (plan && plan.errors || []).join(" ／ ");
      if (warns) warns.textContent = (plan && plan.warnings || []).join(" ／ ");
    }

    function scheduleValidateGpu() {
      clearTimeout(gpuValidateTimer);
      gpuValidateTimer = setTimeout(() => { validateGpuDraft().catch(() => {}); }, 300);
    }

    async function saveGpu() {
      const msg = $("gpuSaveMsg");
      try {
        const res = await postJson("/api/settings/gpu", { settings: gpuDraft });
        gpuData = res;
        gpuDraft = normalizeGpuSettings(res.saved);
        renderGpu();
        if (msg) msg.textContent = `保存しました（${fmtHHMM()}）`;
      } catch (e) {
        const body = (e && e.body) || {};
        if (msg) msg.textContent = body.message || e.message;
        if (body.plan) { gpuValidatePlan = body.plan; renderGpuPlan(body.plan); }
      }
    }

    async function applyGpu() {
      const msg = $("gpuSaveMsg");
      try {
        const res = await postJson("/api/settings/gpu/apply", {});
        if (msg) msg.textContent = res.message || "";
      } catch (e) {
        const body = (e && e.body) || {};
        if (msg) msg.textContent = body.message || e.message;
      }
      loadGpu(false).catch(() => {});
    }

    function wireGpu() {
      $("btnGpuRefresh").onclick = () => loadGpu(true).catch(() => {});
      $("btnGpuSave").onclick = () => saveGpu();
      $("btnGpuApply").onclick = () => applyGpu();
      document.querySelectorAll("#gpuModeChoices .choice").forEach((b) => {
        b.onclick = () => {
          gpuDraft.mode = b.dataset.mode;
          renderGpu();
          scheduleValidateGpu();
        };
      });
      const videoSel = $("gpuVideoSelect");
      if (videoSel) videoSel.onchange = () => { gpuDraft.video_gpu = videoSel.value; renderGpuStatus(); scheduleValidateGpu(); };
      const auxSel = $("gpuAuxSelect");
      if (auxSel) auxSel.onchange = () => { gpuDraft.aux_device = auxSel.value; renderGpuStatus(); scheduleValidateGpu(); };
      const reserve = $("gpuReserve");
      if (reserve) reserve.oninput = () => {
        gpuDraft.reserve_vram_gb = parseFloat(reserve.value) || 0;
        renderGpuStatus();
        scheduleValidateGpu();
      };
    }

    // ========================================================= LLM接続タブ =
    // WP-A: 接続種別→プロバイダー→使用モデル→接続情報の階層UI。
    // 再描画は draftAi から行い、保存済み値で作り直さない。未保存の選択は
    // draft へ即時反映されるため再描画で戻らない。手入力トグル等のUI状態は
    // aiUi（draft 外）に保持する。
    let savedAi = null;   // last known-saved /api/ai/settings payload
    let draftAi = null;   // user-editable draft; re-renders never overwrite it
    const modelsCache = {};  // provider_id -> [{id, display, vision}]
    let aiExtra = { providers: [], keys: {}, gemma: {} };
    const aiUi = { manualMain: false, manualProfile: false,
                   lastProviderByKind: {} };
    const llmTestCache = { main: null, profile: null };

    function ensureDraftAi() {
      if (!draftAi && savedAi) draftAi = JSON.parse(JSON.stringify(savedAi));
    }

    function isAiDirty() {
      if (!savedAi || !draftAi) return false;
      return isDraftDirty(savedAi, draftAi);
    }

    function discardAiDraft() {
      if (savedAi) draftAi = JSON.parse(JSON.stringify(savedAi));
      llmTestCache.main = null;
      llmTestCache.profile = null;
    }

    function specOf(id) {
      return llmProviderSpec(aiExtra.providers, id);
    }

    function currentProviderId() {
      return (draftAi && draftAi.connection && draftAi.connection.provider) || "";
    }

    function ensureProviderEntry(pid) {
      draftAi.providers = draftAi.providers || {};
      if (!draftAi.providers[pid]) {
        const spec = specOf(pid) || {};
        draftAi.providers[pid] = {
          base_url: spec.default_base_url || "",
          model: "",
          vision_models: [],
          extra: {},
        };
      }
      return draftAi.providers[pid];
    }

    function modelsFor(pid) {
      return modelsCache[pid] || [];
    }

    function isManualMode(which, pid) {
      const flag = which === "profile" ? aiUi.manualProfile : aiUi.manualMain;
      return flag || modelsFor(pid).length === 0;
    }

    async function loadAiSettings() {
      try {
        const res = await api("/api/ai/settings");
        savedAi = res.settings;
        aiExtra = {
          providers: res.providers || [],
          keys: res.keys || {},
          gemma: res.gemma || {},
        };
        ensureDraftAi();
      } catch (e) {
        // Keep any existing draft; only show an error if we never loaded anything.
        if (!savedAi) savedAi = null;
      }
      renderLlmTab();
      renderDirectorTemp();
      // Prefetch model lists for the connection + override providers so the
      // selects are populated without an extra click.
      const pids = new Set();
      if (draftAi) {
        if (currentProviderId()) pids.add(currentProviderId());
        const prof = (draftAi.roles || {}).character_profile || {};
        if (prof.override && prof.provider) pids.add(prof.provider);
      }
      await Promise.allSettled([...pids].map((pid) => refreshModels(pid, true)));
      renderLlmTab();
      renderDirectorTemp();
    }

    async function refreshModels(pid, quiet) {
      if (!pid) return;
      ensureDraftAi();
      const spec = specOf(pid) || {};
      const entry = ensureProviderEntry(pid);
      const payload = { provider: pid };
      if (spec.url_editable && entry.base_url) payload.base_url = entry.base_url;
      const msg = $("llmConnMsg");
      try {
        const res = await postJson("/api/ai/models", payload);
        modelsCache[pid] = res.models || [];
        if (!quiet && msg) {
          // A connection failure comes back as ok:true + models:[] + error;
          // show the real reason instead of "the list was empty".
          msg.textContent = modelsCache[pid].length
            ? `${modelsCache[pid].length}件のモデルを取得しました。`
            : (res.error
              ? `モデル一覧を取得できませんでした: ${res.error} 手入力で指定できます。`
              : "モデル一覧が空でした。手入力で指定できます。");
        }
      } catch (e) {
        modelsCache[pid] = [];
        const body = (e && e.body) || {};
        if (!quiet && msg) msg.textContent = body.message || e.message;
      }
    }

    function selectKind(kind) {
      ensureDraftAi();
      if (draftAi.connection.kind === kind) { renderLlmTab(); return; }
      draftAi.connection.kind = kind;
      const remembered = aiUi.lastProviderByKind[kind];
      const list = llmProvidersForKind(aiExtra.providers, kind);
      const pid = (remembered && specOf(remembered) && specOf(remembered).kind === kind)
        ? remembered
        : (list[0] && list[0].id) || "";
      draftAi.connection.provider = pid;
      if (pid) {
        aiUi.lastProviderByKind[kind] = pid;
        ensureProviderEntry(pid);
      }
      aiUi.manualMain = false;
      llmTestCache.main = null;
      renderLlmTab();
      if (pid) refreshModels(pid, true).then(renderLlmTab);
    }

    function selectProvider(pid) {
      ensureDraftAi();
      const spec = specOf(pid);
      if (!spec) return;
      aiUi.lastProviderByKind[spec.kind] = pid;
      draftAi.connection.provider = pid;
      draftAi.connection.kind = spec.kind;
      ensureProviderEntry(pid);
      aiUi.manualMain = false;
      llmTestCache.main = null;
      renderLlmTab();
      refreshModels(pid, true).then(renderLlmTab);
    }

    function selectProfileProvider(pid) {
      ensureDraftAi();
      if (!specOf(pid)) return;
      draftAi.roles.character_profile.provider = pid;
      aiUi.manualProfile = false;
      llmTestCache.profile = null;
      renderLlmTab();
      refreshModels(pid, true).then(renderLlmTab);
    }

    function renderChoices(boxId, items, current, onPick) {
      const box = $(boxId);
      if (!box) return;
      box.innerHTML = "";
      items.forEach((item) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "choice" + (item.value === current ? " on" : "");
        b.textContent = item.label;
        b.onclick = () => onPick(item.value);
        box.appendChild(b);
      });
    }

    function renderModelPicker(selId, manualId, toggleId, pid, value, which) {
      const sel = $(selId);
      const manual = $(manualId);
      const toggle = $(toggleId);
      if (!sel) return;
      const list = modelsFor(pid);
      const manualMode = isManualMode(which, pid);
      sel.hidden = manualMode;
      if (manual) {
        manual.hidden = !manualMode;
        if (manualMode && document.activeElement !== manual) manual.value = value || "";
      }
      if (toggle) toggle.textContent = manualMode ? "一覧から選択" : "手入力";
      if (manualMode) return;
      sel.innerHTML = "";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "（選択）";
      sel.appendChild(empty);
      list.forEach((m) => {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = m.display || m.id;
        sel.appendChild(o);
      });
      sel.value = list.some((m) => m.id === value) ? value : "";
    }

    function selectedModelOf(selId, manualId, pid, which) {
      const sel = $(selId);
      const manual = $(manualId);
      if (isManualMode(which, pid)) {
        return manual ? manual.value.trim() : "";
      }
      return sel && !sel.hidden ? (sel.value || "") : "";
    }

    function renderVision(stateId, rowId, cbId, pid, model) {
      const state = $(stateId);
      const row = $(rowId);
      const cb = $(cbId);
      if (!state || !draftAi) return;
      if (!pid || !model) {
        state.textContent = "未確認";
        if (row) row.hidden = true;
        return;
      }
      const entry = modelsFor(pid).find((m) => m.id === model);
      const visionModels = (((draftAi.providers || {})[pid]) || {}).vision_models || [];
      if (entry && entry.vision === true) {
        state.textContent = "対応";
        if (row) row.hidden = true;
      } else if (entry && entry.vision === false) {
        state.textContent = "非対応";
        if (row) row.hidden = true;
      } else {
        const marked = visionModels.includes(model);
        state.textContent = marked ? "対応（手動確認）" : "未確認";
        if (row) { row.hidden = false; if (cb) cb.checked = marked; }
      }
    }

    function renderLlmTab() {
      if (!draftAi) return;
      const pid = currentProviderId();
      const spec = specOf(pid) || {};
      if (pid) ensureProviderEntry(pid);
      const entry = ((draftAi.providers || {})[pid]) || {};

      const summary = $("llmSummary");
      if (summary) summary.textContent = llmSummaryText(draftAi, aiExtra.providers);

      renderChoices("llmKindChoices", [
        { value: "local", label: "ローカル" },
        { value: "external", label: "外部API" },
      ], draftAi.connection.kind, selectKind);
      renderChoices("llmProviderChoices",
        llmProvidersForKind(aiExtra.providers, draftAi.connection.kind)
          .map((p) => ({ value: p.id, label: p.label })),
        pid, selectProvider);

      renderModelPicker("llmModelSelect", "llmModelManual", "btnLlmManualToggle",
        pid, entry.model || "", "main");

      const gInfo = $("llmGemmaInfo");
      const urlRow = $("llmUrlRow");
      const keyRow = $("llmKeyRow");
      const isGemma = pid === "comfy_gemma";
      if (gInfo) {
        gInfo.hidden = !isGemma;
        if (isGemma) {
          const gemma = aiExtra.gemma || {};
          const avail = gemma.available === true ? "利用可能"
            : gemma.available === false ? "利用不可" : "未確認";
          gInfo.textContent = `モデル: ${gemma.model || "-"} ／ mmproj: ${gemma.mmproj || "-"} ／ 状態: ${avail}`;
        }
      }
      // Fixed-endpoint providers (OpenCode Go, OpenAI, ...) show their URL
      // read-only so the user can see WHERE the key is sent; editable ones
      // get the input as before.
      if (urlRow) urlRow.hidden = isGemma;
      const urlInput = $("llmBaseUrl");
      const urlLabel = document.querySelector("#llmUrlRow .lbl");
      if (urlInput && urlRow && !urlRow.hidden) {
        if (spec.url_editable) {
          urlInput.readOnly = false;
          if (document.activeElement !== urlInput) urlInput.value = entry.base_url || "";
          urlInput.placeholder = spec.default_base_url || "";
          if (urlLabel) urlLabel.textContent = "サーバーURL";
        } else {
          urlInput.readOnly = true;
          urlInput.value = spec.default_base_url || "";
          urlInput.placeholder = "";
          if (urlLabel) urlLabel.textContent = "接続先（固定）";
        }
      }
      if (keyRow) keyRow.hidden = isGemma || !spec.key_name;
      const keyInput = $("llmApiKey");
      if (keyInput && keyRow && !keyRow.hidden) {
        const keyLabel = document.querySelector("#llmKeyRow .lbl");
        if (keyLabel) keyLabel.textContent = spec.kind === "local" ? "トークン" : "API Key";
        keyInput.placeholder = aiExtra.keys[pid] ? "設定済み（●●●●●●●●）" : "未設定";
        const saveBtn = $("btnLlmKeySave");
        if (saveBtn) saveBtn.textContent = spec.kind === "local" ? "トークンを保存" : "API Keyを保存";
        const delBtn = $("btnLlmKeyDelete");
        if (delBtn) delBtn.textContent = spec.kind === "local" ? "トークンを削除" : "API Keyを削除";
      }

      renderVision("llmMainVisionState", "llmMainVisionManualRow", "llmMainVisionManual",
        pid, entry.model || "");
      const testRow = $("llmTestResult");
      if (testRow) testRow.textContent = llmTestResultText(llmTestCache.main);

      const role = (draftAi.roles || {}).character_profile || {};
      const ovr = $("aiProfileOverride");
      if (ovr) ovr.checked = !!role.override;
      const ovrBlock = $("llmProfileOverrideBlock");
      if (ovrBlock) ovrBlock.hidden = !role.override;
      if (role.override) {
        renderChoices("llmProfileProvider",
          (aiExtra.providers || []).map((p) => ({ value: p.id, label: p.label })),
          role.provider || "", selectProfileProvider);
        renderModelPicker("llmProfileModelSelect", "llmProfileModelManual",
          "btnLlmProfileManualToggle", role.provider || "", role.model || "", "profile");
        renderVision("llmProfileVisionState", "llmProfileVisionManualRow",
          "llmProfileVisionManual", role.provider || "", role.model || "");
        const pMsg = $("aiProfileTestMsg");
        if (pMsg) pMsg.textContent = llmTestResultText(llmTestCache.profile);
      }

      const fav = $("aiGemmaFallback");
      if (fav) fav.checked = !!draftAi.gemma_fallback;

      const savedChip = $("llmStatusSaved");
      if (savedChip) savedChip.textContent = savedAi ? "保存済み" : "";
      const dirtyChip = $("llmStatusDirty");
      if (dirtyChip) dirtyChip.hidden = !isAiDirty();
    }

    function renderDirectorTemp() {
      const sel = $("directorTempModel");
      if (!sel || !savedAi) return;
      const eff = llmEffective(savedAi, "director");
      const list = modelsFor(eff.provider);
      const prev = sel.value;
      sel.innerHTML = "";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "（設定画面の既定モデルを使用）";
      sel.appendChild(empty);
      list.forEach((m) => {
        const o = document.createElement("option");
        o.value = JSON.stringify({ provider: eff.provider, model: m.id, endpoint: "" });
        o.textContent = m.display || m.id;
        sel.appendChild(o);
      });
      sel.value = prev;
      sel.onchange = () => {
        if (!sel.value) { if (typeof setDirectorTempModel === "function") setDirectorTempModel(null); return; }
        try {
          const m = JSON.parse(sel.value);
          if (typeof setDirectorTempModel === "function") setDirectorTempModel(m);
        } catch (e) { /* ignore */ }
      };
    }

    function collectAiFromDraft() {
      // Pull the latest values the user typed/selected back into draftAi
      // before it is sent to the server.
      ensureDraftAi();
      const pid = currentProviderId();
      const spec = specOf(pid) || {};
      if (pid) {
        draftAi.connection.kind = spec.kind || draftAi.connection.kind;
        const entry = ensureProviderEntry(pid);
        entry.model = selectedModelOf("llmModelSelect", "llmModelManual", pid, "main");
        const urlInput = $("llmBaseUrl");
        if (urlInput && spec.url_editable) entry.base_url = urlInput.value.trim();
        const visionCb = $("llmMainVisionManual");
        if (visionCb && entry.model) {
          const marked = new Set(entry.vision_models || []);
          if (visionCb.checked) marked.add(entry.model);
          else marked.delete(entry.model);
          entry.vision_models = [...marked];
        }
      }
      const ovr = $("aiProfileOverride");
      const role = draftAi.roles.character_profile;
      if (ovr) role.override = ovr.checked;
      if (role.override) {
        role.model = selectedModelOf("llmProfileModelSelect", "llmProfileModelManual",
          role.provider || "", "profile");
        const visionCb = $("llmProfileVisionManual");
        if (visionCb && role.provider && role.model) {
          const pentry = ensureProviderEntry(role.provider);
          const marked = new Set(pentry.vision_models || []);
          if (visionCb.checked) marked.add(role.model);
          else marked.delete(role.model);
          pentry.vision_models = [...marked];
        }
      }
      const fav = $("aiGemmaFallback");
      if (fav) draftAi.gemma_fallback = fav.checked;
      return draftAi;
    }

    async function saveAiSettings() {
      const msg = $("aiSettingsMsg");
      const settings = collectAiFromDraft();
      try {
        const res = await postJson("/api/ai/settings", { settings });
        savedAi = res.settings;
        draftAi = JSON.parse(JSON.stringify(savedAi));
        renderLlmTab();
        renderDirectorTemp();
        if (msg) msg.textContent = `保存しました（${fmtHHMM()}）`;
      } catch (e) {
        if (msg) msg.textContent = (e && e.body && e.body.message) || e.message;
      }
    }

    async function testProviderModel(which) {
      collectAiFromDraft();
      const isProfile = which === "profile";
      let pid;
      let model;
      if (isProfile) {
        const eff = llmEffective(draftAi, "character_profile");
        pid = eff.provider;
        model = eff.model;
      } else {
        pid = currentProviderId();
        model = (((draftAi.providers || {})[pid]) || {}).model || "";
      }
      const msgId = isProfile ? "aiProfileTestMsg" : "llmTestResult";
      const msg = $(msgId);
      if (!msg) return;
      if (!pid || pid === "comfy_gemma") {
        const text = "ComfyUI内ローカルGemmaはこの画面ではテストできません。上の接続情報をご確認ください。";
        if (isProfile) llmTestCache.profile = { ok: false, message: text };
        else llmTestCache.main = { ok: false, message: text };
        msg.textContent = text;
        return;
      }
      if (!model) { msg.textContent = "先にモデルを選んでください。"; return; }
      msg.textContent = "テスト中…";
      const entry = ((draftAi.providers || {})[pid]) || {};
      const payload = { provider: pid, model };
      const spec = specOf(pid) || {};
      if (spec.url_editable && entry.base_url) payload.base_url = entry.base_url;
      if (isProfile && typeof state !== "undefined" && state.images && state.images.length) {
        payload.image = state.images[0].name;
      }
      try {
        const res = await postJson("/api/ai/test", payload);
        if (isProfile) llmTestCache.profile = res;
        else llmTestCache.main = res;
        msg.textContent = llmTestResultText(res) || res.message || "";
      } catch (e) {
        const body = (e && e.body) || {};
        const failed = { ok: false, message: body.message || e.message };
        if (isProfile) llmTestCache.profile = failed;
        else llmTestCache.main = failed;
        msg.textContent = body.message || e.message;
      }
    }

    // ==================================================== WP-D: shutdown ==
    // "H3終了時にLM Studioを終了する" checkbox: restored from
    // GET /api/settings/shutdown, saved via POST /api/settings/shutdown.
    async function loadShutdownSetting() {
      const box = $("shutdownStopLmStudio");
      const pre = $("shutdownPreMsg");
      try {
        const res = await api("/api/settings/shutdown");
        if (box) box.checked = res.stop_lm_studio !== false;
        if (pre) pre.hidden = !res.lmstudio_pre_existing;
      } catch (e) {
        if (box) box.checked = true;
        if (pre) pre.hidden = true;
      }
    }

    async function saveShutdownSetting() {
      const msg = $("shutdownSettingMsg");
      const box = $("shutdownStopLmStudio");
      try {
        const res = await postJson("/api/settings/shutdown", {
          stop_lm_studio: box ? !!box.checked : true,
        });
        if (box) box.checked = !!res.stop_lm_studio;
        if (msg) msg.textContent = `保存しました（${fmtHHMM()}）`;
      } catch (e) {
        if (msg) msg.textContent = (e && e.body && e.body.message) || e.message;
      }
    }

    function wireLlm() {
      const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
      on("llmModelSelect", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("llmModelManual", "input", () => { collectAiFromDraft(); renderLlmTab(); });
      on("btnLlmManualToggle", "click", () => {
        collectAiFromDraft();
        aiUi.manualMain = !aiUi.manualMain;
        renderLlmTab();
      });
      on("llmBaseUrl", "input", () => { collectAiFromDraft(); renderLlmTab(); });
      on("llmMainVisionManual", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("btnLlmModels", "click", () => refreshModels(currentProviderId(), false).then(renderLlmTab));
      on("btnLlmTest", "click", () => testProviderModel("main"));
      on("aiProfileOverride", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("llmProfileModelSelect", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("llmProfileModelManual", "input", () => { collectAiFromDraft(); renderLlmTab(); });
      on("btnLlmProfileManualToggle", "click", () => {
        collectAiFromDraft();
        aiUi.manualProfile = !aiUi.manualProfile;
        renderLlmTab();
      });
      on("llmProfileVisionManual", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("btnLlmProfileModels", "click", () => {
        collectAiFromDraft();
        const eff = llmEffective(draftAi, "character_profile");
        refreshModels(eff.provider, false).then(renderLlmTab);
      });
      on("btnAiProfileTest", "click", () => testProviderModel("profile"));
      on("aiGemmaFallback", "change", () => { collectAiFromDraft(); renderLlmTab(); });
      on("btnAiSettingsSave", "click", () => saveAiSettings());
      on("btnLlmKeyShow", "click", () => {
        const k = $("llmApiKey");
        if (k) k.type = k.type === "password" ? "text" : "password";
      });
      on("btnLlmKeySave", "click", async () => {
        collectAiFromDraft();
        const msg = $("llmConnMsg");
        const pid = currentProviderId();
        try {
          const res = await postJson("/api/ai/key", { provider: pid, key: $("llmApiKey").value });
          if ($("llmApiKey")) $("llmApiKey").value = "";
          aiExtra.keys = res.keys || aiExtra.keys;
          renderLlmTab();
          if (msg) msg.textContent = "保存しました。";
        } catch (e) { if (msg) msg.textContent = (e && e.body && e.body.message) || e.message; }
      });
      on("btnLlmKeyDelete", "click", async () => {
        const msg = $("llmConnMsg");
        const pid = currentProviderId();
        try {
          const res = await api("/api/ai/key", {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: pid }),
          });
          aiExtra.keys = (res && res.keys) || aiExtra.keys;
          renderLlmTab();
          if (msg) msg.textContent = "削除しました。";
        } catch (e) { if (msg) msg.textContent = (e && e.body && e.body.message) || e.message; }
      });
      // WP-D: shutdown setting lives at the end of tabLlm.
      const btnShutdownSave = $("btnShutdownSettingSave");
      if (btnShutdownSave) btnShutdownSave.onclick = () => saveShutdownSetting();
    }

    // ============================================================= boot ===
    async function boot() {
      // Loaded once at page startup so 今回だけ使用するモデル is populated
      // even before the user ever opens the settings dialog.
      await loadAiSettings();
      await loadShutdownSetting(); // WP-D: restore stop_lm_studio default.
    }

    function wire() {
      wireTabs();
      wireStorage();
      wireGpu();
      wireLlm();
      $("btnAiSettings").onclick = () => openDialog();
      $("btnAiSettingsClose").onclick = () => attemptClose();
      const dlg = $("settingsDialog");
      if (dlg) {
        dlg.addEventListener("cancel", (e) => {
          e.preventDefault();
          attemptClose();
        });
      }
    }

    window.H3Settings = { wire, boot, renderDirectorTemp };
  })();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    stableStringify, isDraftDirty, llmStatusChipInfo, gpuModeLabel, shortUuid,
    fmtMiBValue, fmtHHMM,
    llmKindLabel, llmProviderSpec, llmProvidersForKind, llmEffective,
    llmSummaryText, llmImageSupportLabel, llmTestResultText,
  };
}
