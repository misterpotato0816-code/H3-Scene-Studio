// H3 App front end. Plain ES2020, no build step, no CDN.
// The UI never mentions ComfyUI, node names or "Tail Relay": those are
// implementation details the user has no way to act on.
"use strict";

const $ = (id) => document.getElementById(id);

const ROLES = [
  "画像1 顔メイン(必須)",
  "画像2 上半身",
  "画像3 補助",
  "画像4 補助",
];

const STORY_STATUS_JP = {
  pending: "待機",
  running: "生成中",
  stopping: "停止の準備中（今のクリップが終わるまで）",
  paused: "一時停止",
  done: "完了",
  error: "エラー",
};

const state = {
  cfg: null,
  mode: "single",      // "single" | "story"
  // ユーザーが編集できる「動作」プリセット（app\action_presets.json）。
  // ここに入るのは {id, name, label, factory} だけで、選択状態は持たない。
  // チップは切り替えスイッチではなく「押した瞬間に文章を書き足す命令」。
  actionPresets: [],
  actionTemplate: "この場面の流れの中で、自然に「{name}」の動作を加える。",
  actionMaxName: 40,
  actionManage: false, // プリセット管理モード（削除の×を出す）
  images: [],          // [{name, thumb}]
  seconds: 5,
  aspect: "portrait",
  genMode: "FAST",     // v2 Generation Mode: FAST default (single-shot)
  projectId: null,
  project: null,
  events: null,
  lastAction: null,    // for 再試行
  busy: false,         // a generation is running: block anything that touches models
  story: {
    id: null,
    name: "",
    lines: [],         // [{type:"prompt"|"speech", text}]
    lineMode: "prompt",
    baseSeed: null,
    // [{index, prompt, speech, status, motion_presets, action_presets, seed, clip, error}]
    segments: [],
    presets: [],       // selected STYLE preset ids (apply to every segment)
    status: "pending",
    cursor: 0,
    clipsDone: 0,
    merged: false,
    running: false,
    events: null,
    summaries: [],
  },
  audioTest: { list: [], busy: false, lang: "Japanese" },
};

// Single place that decides which buttons are live. Releasing models while a
// generation is running would pull the weights out from under it, so the whole
// system row is gated on the same flag. Story mode shares the SAME flag, so a
// running story blocks single-shot generation and vice versa.
function refreshButtons() {
  const busy = state.busy;
  const st = state.story;
  const storyRunning = st.running;

  $("btnGenerate").disabled = busy || state.images.length === 0;
  $("btnRelease").disabled = busy;
  $("btnAgain").disabled = busy;
  const p = state.project;
  $("btnContinue").disabled = busy || !(p && p.can_continue);

  // WP-B: upscaling reads the model/pipeline slot too, so it shares the
  // same busy gate as generation and story runs.
  if (upscalePanels.result) {
    upscalePanels.result.setDisabled(busy || storyRunning || !state.projectId);
  }
  if (upscalePanels.story) {
    upscalePanels.story.setDisabled(busy || storyRunning || !state.story.id);
  }

  const startBtn = $("btnStoryStart");
  if (startBtn) {
    startBtn.disabled = busy || state.images.length === 0 || st.segments.length === 0;
    startBtn.textContent = storyIsResumable() ? "再開" : "ストーリー開始";

    // A disabled button with no stated reason is a dead end for the user.
    const why = $("storyStartWhy");
    let reason = "";
    if (busy) reason = "いま別の生成が動いています。終わるまで待つか、中止してください。";
    else if (state.images.length === 0) reason = "上の「参照画像」に画像を1枚以上追加すると開始できます。";
    else if (st.segments.length === 0) reason = "先に台本を書いて「分割し直す」か「長文から自動で作る」を押してください。";
    why.textContent = reason;
    why.hidden = !reason;

    $("btnStoryStop").disabled = !storyRunning;
    $("btnStoryDiscard").disabled = !st.id || storyRunning;
    $("btnStoryMerge").disabled = busy || !st.id || st.clipsDone === 0;
    $("btnStorySaveSegments").disabled = storyRunning || !st.id || st.segments.length === 0;
    $("btnStoryResplit").disabled = storyRunning;
    $("btnStoryFromText").disabled = storyRunning;
    $("btnAddLine").disabled = storyRunning;
    $("btnClearLines").disabled = storyRunning;
    $("btnStoryPreflight").disabled = storyRunning || st.segments.length === 0;
  }

  // 構成の変更（追加 / 削除 / 並べ替え）。サーバーも断るが、UI からも誘わない。
  const structLocked = busy || storyRunning;
  const addSeg = $("btnSegmentAdd");
  if (addSeg) addSeg.disabled = structLocked;
  document.querySelectorAll("#segmentList [data-struct]").forEach((b) => {
    b.disabled = structLocked || b.dataset.structOff === "1";
  });

  updateActionbar();
  window.H3Studio?.sync();
  renderIdentity();
  refreshAudioTestButtons();
}

function currentGenBlockReason() {
  if (state.busy) return "生成中です。終わるまでお待ちください。";
  if (state.images.length === 0) return "参照画像を1枚以上追加してください。";
  return "";
}

// Sticky bottom action bar: mirrors the real buttons, never bypasses them.
function updateActionbar() {
  const bar = $("actionbar");
  if (!bar) return;
  // Audio has its own test/preview controls; the video shortcut is inapplicable.
  bar.hidden = state.mode === "audio";
  const storyMode = state.mode === "story";
  const directorMode = state.mode === "director";
  const goBtn = $("btnNowGo");
  const cancelBtn = $("btnNowCancel");
  const st = $("actionState");
  if (directorMode) {
    const has = !!(state.director && state.director.spec);
    goBtn.textContent = "この監督案で生成";
    const blocked = state.busy || !has || state.images.length === 0;
    goBtn.disabled = blocked;
    st.textContent = state.busy ? "生成中…"
      : !has ? "監督案を作ると生成できます。"
      : state.images.length === 0 ? "参照画像を1枚以上追加してください。"
      : "監督案OK — 生成できます";
    cancelBtn.hidden = !state.busy;
  } else if (storyMode) {
    const s = state.story;
    goBtn.textContent = storyIsResumable() ? "ストーリーを再開" : "ストーリーを生成";
    const blocked = state.busy || state.images.length === 0 || s.segments.length === 0;
    goBtn.disabled = blocked;
    st.textContent = s.running ? `Clip ${s.cursor + 1} / ${s.segments.length} 生成中`
      : (blocked ? currentGenBlockReason() || "台本の準備をしてください。"
        : `LONG STORY 準備OK（${s.segments.length} segments）`);
    cancelBtn.hidden = !s.running;
  } else {
    goBtn.textContent = "動画を生成";
    const reason = currentGenBlockReason();
    goBtn.disabled = !!reason;
    st.textContent = state.busy ? "生成中…" : (reason || `${state.genMode} 準備OK`);
    cancelBtn.hidden = !state.busy;
  }
  // Progress mirror (single-shot bar; story has its own panel).
  const src = $("progressBar");
  const dst = $("actionBar");
  if (src && dst) dst.style.width = src.style.width || "0";
}

function hideMediaFolderNotice() {
  const box = $("mediaFolderNotice");
  if (box) box.hidden = true;
}

function showMediaFolderNotice(directory) {
  const box = $("mediaFolderNotice");
  if (!box || !directory) return;
  $("mediaFolderNoticeText").textContent = directory;
  box.dataset.path = directory;
  box.hidden = false;
}

async function openMediaFolder() {
  try {
    const res = await postJson("/api/media/open", { target: "videos" });
    if (res && res.directory) showMediaFolderNotice(res.directory);
  } catch (e) {
    toast((e && e.body && e.body.message) || e.message);
    const directory = e && e.body && e.body.directory;
    if (directory) showMediaFolderNotice(directory);
  }
}

// ------------------------------------------------------- AI Director ----
state.director = { duration: 10, spec: null, enforced: [] };

const DIRECTOR_LABELS = {
  video_type: "動画タイプ", location: "場所", scenery: "風景/背景",
  subject: "人物", outfit: "服装", acting: "演技/動作",
  camera_style: "カメラ形式", camera_work: "カメラワーク",
  dialogue: "台詞", voice: "音声/話し方", ambient_audio: "環境音",
  mood: "雰囲気", movement: "動き", handover: "引き継ぎ",
  theme: "全体テーマ", protagonist: "主役", world: "世界観",
  story_arc: "ストーリー展開", dialogue_policy: "会話方針",
  shoot_policy: "撮影方針",
};
const directorLabel = (name) => DIRECTOR_LABELS[name] || name;

function directorDurations() {
  const box = $("directorDurations");
  if (!box || box.children.length) return;
  [[5, "5秒"], [10, "10秒"], [15, "15秒"], [30, "30秒"]].forEach(([v, t]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "choice" + (v === state.director.duration ? " on" : "");
    b.textContent = t;
    b.onclick = () => {
      state.director.duration = v;
      [...box.children].forEach((c) => c.classList.remove("on"));
      b.classList.add("on");
    };
    box.appendChild(b);
  });
}

async function directorCreate() {
  const idea = $("directorIdea").value.trim();
  if (!idea) { toast("アイデアを入力してください。"); return; }
  const btn = $("btnDirectorCreate");
  btn.disabled = true;
  try {
    const dur = state.director.duration;
    const req = $("directorRequest").value;
    const path = dur === 30 ? "/api/director/story30/create" : "/api/director/single/create";
    const payload = dur === 30
      ? { idea, request: req }
      : { idea, duration_sec: dur, request: req };
    payload.temp_model = directorTempModelPayload();
    payload.session_id = directorSessionId(true);
    const res = await postJson(path, payload);
    state.director.spec = res.spec;
    state.director.enforced = [];
    state.director.used = res.used || null;
    state.director.masterUsed = res.master_used || null;
    renderDirectorResult(res);
    $("cardDirectorResult").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
  }
}

function continuityText(c) {
  // Natural phrasing with explicit 1-based pair: which clip starts, which
  // previous clip it is compared against, and what changed.
  const nxt = c.clip_display || (c.clip || 0) + 1;
  const prev = c.prev_clip_display || nxt - 1;
  const raw = c.message || "";
  if (c.aspect === "emotion") {
    const m = raw.match(/clip\d+開始の感情（(.*?)）が前clip終了\s*（(.*?)）と異なります/);
    if (m) {
      return `CLIP${nxt}開始時の感情「${m[1]}」が、CLIP${prev}終了時「${m[2]}」から変化しています。変化のきっかけがあれば問題ありません。`;
    }
  }
  if (c.aspect === "carried_objects") {
    const m = raw.match(/clip\d+開始時に「(.*?)」の継続が見えません/);
    if (m) {
      return `CLIP${nxt}開始時に「${m[1]}」の継続が見えません（CLIP${prev}からの引き継ぎ）。使い切る・置く等の説明があれば問題ありません。`;
    }
  }
  if (c.aspect === "spatial") {
    return `CLIP${nxt}で昼夜が切り替わっています（CLIP${prev}との比較）。時間経過の説明があれば問題ありません。`;
  }
  return `CLIP${nxt} 確認（CLIP${prev}との比較）: ${raw}`;
}

function directorItemCard(container, opts) {
  // opts: {title, item, onRegen} — item may be undefined (state dicts render raw).
  const card = document.createElement("div");
  card.className = "dircard";
  const head = document.createElement("div");
  head.className = "dirhead";
  const title = document.createElement("span");
  title.className = "dirtitle";
  title.textContent = opts.title;
  head.appendChild(title);
  if (opts.item) {
    const lock = document.createElement("label");
    lock.className = "dirlock";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!opts.item.locked;
    cb.onchange = () => { opts.item.locked = cb.checked; };
    lock.append(cb, document.createTextNode("ロック"));
    head.appendChild(lock);
  }
  card.appendChild(head);
  const isDialogue = opts.item && opts.title === directorLabel("dialogue");
  const val = document.createElement(isDialogue ? "textarea" : "p");
  val.className = "dirval";
  val.textContent = opts.text || "";
  if (isDialogue) {
    val.value = opts.text || "";
    val.classList.add("dialogue-edit");
    val.setAttribute("aria-label", "話す台詞を直接編集");
    val.disabled = !!opts.item.locked;
    val.oninput = () => { opts.item.value = val.value; opts.item.en = ""; };
    const lock = head.querySelector("input[type=checkbox]");
    lock?.addEventListener("change", () => { val.disabled = lock.checked; });
  }
  card.appendChild(val);
  if (opts.item) {
    const adjust = document.createElement("details");
    adjust.className = "dir-adjust"; adjust.open = !!opts.item.request;
    const summary = document.createElement("summary"); summary.textContent = "変更をAIに依頼";
    adjust.appendChild(summary);
    const req = document.createElement("textarea");
    req.rows = 2;
    req.placeholder = "この項目への注文（例：もっと台詞を多く）";
    req.value = opts.item.request || "";
    req.oninput = () => { opts.item.request = req.value; };
    adjust.appendChild(req);
    if (opts.onRegen) {
      const row = document.createElement("div");
      row.className = "actions";
      const rb = document.createElement("button");
      rb.type = "button";
      rb.className = "btn";
      rb.textContent = "この項目だけ再生成";
      rb.onclick = opts.onRegen;
      row.appendChild(rb);
      adjust.appendChild(row);
    }
    card.appendChild(adjust);
  }
  container.appendChild(card);
}

function directorUsedLine(used, masterUsed) {
  // Secrets never appear here: `used` carries kind/model/timing only.
  if (!used) return "";
  const kindLabel = used.kind === "go" ? "OpenCode Go"
    : used.kind === "gemma" ? "ローカルGemma"
    : used.kind === "local" ? "ローカルLLM" : (used.kind || "不明");
  const model = used.model || (used.kind === "gemma" ? "既定VLM" : "不明");
  const secs = used.elapsed_ms ? `${(used.elapsed_ms / 1000).toFixed(1)}秒` : "時間不明";
  const bits = [`使用: ${kindLabel}／${model}／${secs}`];
  if (used.fallback) {
    const errs = (used.errors || []).filter((e) => e && e.error);
    const reason = errs.length ? `（${errs[errs.length - 1].error}）` : "";
    bits.push(`代替で生成（試行${used.attempts || "?"}回${used.repairs ? `・再送${used.repairs}回` : ""}）${reason}`);
  }
  const usage = used.usage || {};
  if (usage.output_tokens || usage.reasoning_tokens || usage.input_tokens) {
    bits.push(`token 入力${usage.input_tokens || 0}／出力${usage.output_tokens || 0}` +
      (usage.reasoning_tokens ? `／推論${usage.reasoning_tokens}` : ""));
  }
  if (used.incomplete) bits.push("出力が途中で切れました");
  if (used.retry_after) bits.push(`再試行待機: ${used.retry_after}秒`);
  if (masterUsed && masterUsed.model) {
    const msecs = masterUsed.elapsed_ms ? `${(masterUsed.elapsed_ms / 1000).toFixed(1)}秒` : "";
    bits.push(`MASTER: ${masterUsed.model}${msecs ? `／${msecs}` : ""}`);
  }
  return bits.join("　");
}

function renderDirectorUsed(res) {
  const el = $("directorUsed");
  if (!el) return;
  el.textContent = directorUsedLine(
    (res && res.used) || state.director.used,
    (res && res.master_used) || state.director.masterUsed);
}

function directorShowWarnings(res) {
  renderDirectorUsed(res);
  const box = $("directorWarnings");
  box.innerHTML = "";
  (res.warnings || []).forEach((w) => {
    const p = document.createElement("p");
    p.className = "warn";
    p.textContent = w;
    box.appendChild(p);
  });
  (res.continuity || []).forEach((c) => {
    const p = document.createElement("p");
    p.className = "warn";
    p.textContent = continuityText(c);
    box.appendChild(p);
  });
  const enf = $("directorEnforced");
  const paths = res.enforced || state.director.enforced || [];
  enf.textContent = paths.length ? `ロック維持: ${paths.join("、")}` : "";
}

function renderDirectorResult(res) {
  const spec = state.director.spec;
  if (!spec) return;
  if (res) directorShowWarnings(res);
  const box = $("directorItems");
  box.innerHTML = "";
  if (spec.kind === "single") {
    Object.keys(spec.items || {}).forEach((name) => {
      directorItemCard(box, {
        title: directorLabel(name),
        item: spec.items[name],
        text: spec.items[name].value,
        onRegen: () => directorRegen({ item: name }),
      });
    });
    if ((spec.timeline || []).length) {
      const t = document.createElement("div");
      t.className = "dircard";
      t.innerHTML = "<div class='dirhead'><span class='dirtitle'>タイムライン</span></div>";
      spec.timeline.forEach((s) => {
        const row = document.createElement("div"); row.className = "timeline-row";
        for (const [key, title] of [["t0", "開始秒"], ["t1", "終了秒"]]) {
          const label = document.createElement("label"); label.textContent = title;
          const input = document.createElement("input"); input.type = "number";
          input.min = 0; input.max = spec.duration_sec; input.step = 0.1;
          input.value = s[key]; input.disabled = state.busy;
          input.oninput = () => { s[key] = input.value === "" ? null : Number(input.value); };
          label.appendChild(input); row.appendChild(label);
        }
        const p = document.createElement("p");
        p.className = "dirval dim";
        p.textContent = s.label || s.action || "";
        row.appendChild(p); t.appendChild(row);
      });
      box.appendChild(t);
    }
  } else {
    const msec = document.createElement("div");
    msec.className = "dirgroup";
    msec.innerHTML = "<h3>MASTER</h3>";
    Object.keys((spec.master.items || {})).forEach((name) => {
      directorItemCard(msec, {
        title: directorLabel(name),
        item: spec.master.items[name],
        text: spec.master.items[name].value,
        onRegen: () => directorRegen({ masterItem: name }),
      });
    });
    box.appendChild(msec);
    (spec.clips || []).forEach((clip) => {
      const csec = document.createElement("div");
      csec.className = "dirgroup";
      csec.innerHTML = `<h3>CLIP ${clip.index + 1}</h3>`;
      Object.keys(clip.items || {}).forEach((name) => {
        directorItemCard(csec, {
          title: directorLabel(name),
          item: clip.items[name],
          text: clip.items[name].value,
          onRegen: () => directorRegen({ clip: clip.index, item: name }),
        });
      });
      ["start_state", "end_state", "carry_over"].forEach((k) => {
        const st = clip[k] || {};
        const keys = Object.keys(st);
        if (keys.length) {
          directorItemCard(csec, {
            title: { start_state: "開始状態", end_state: "終了状態", carry_over: "引き継ぎ" }[k],
            text: keys.map((kk) => `${kk}: ${st[kk]}`).join(" ／ "),
          });
        }
      });
      const row = document.createElement("div");
      row.className = "actions";
      const rb = document.createElement("button");
      rb.type = "button";
      rb.className = "btn";
      rb.textContent = `CLIP ${clip.index + 1}だけ再生成`;
      rb.onclick = () => directorRegen({ clip: clip.index });
      row.appendChild(rb);
      csec.appendChild(row);
      box.appendChild(csec);
    });
  }
  setMode("director");
  refreshButtons();
}

async function directorRegen(scope) {
  const spec = state.director.spec;
  if (!spec) return;
  spec.global_request = $("directorGlobal").value;
  const isStory = spec.kind === "story30";
  let path, payload;
  if (!isStory) {
    path = "/api/director/single/regenerate";
    payload = { spec, scope_item: (scope && scope.item) || "" };
  } else {
    path = "/api/director/story30/regenerate";
    const bodyScope = {};
    if (scope && scope.clip !== undefined && scope.clip !== null) {
      bodyScope.clips = [scope.clip];
    }
    const items = [];
    if (scope && scope.item) items.push(scope.item);
    if (scope && scope.masterItem) items.push(scope.masterItem);
    if (items.length) bodyScope.items = items;
    payload = { spec, scope: bodyScope };
  }
  payload.temp_model = directorTempModelPayload();
  payload.session_id = directorSessionId(false);
  try {
    const res = await postJson(path, payload);
    state.director.spec = res.spec;
    state.director.enforced = res.enforced || [];
    state.director.used = res.used || state.director.used || null;
    if (res.master_used) state.director.masterUsed = res.master_used;
    renderDirectorResult(res);
    toast("再ディレクションしました。");
  } catch (e) {
    toast(e.message);
  }
}

async function directorGenerate() {
  const spec = state.director.spec;
  if (!spec) { toast("先に監督案を作ってください。"); return; }
  if (!state.images.length) { toast("参照画像を1枚以上追加してください。"); return; }
  spec.global_request = ($("directorGlobal") || {}).value || spec.global_request || "";
  setBusy(true);
  try {
    const review = await postJson("/api/director/review", { spec });
    if (!review.ok) throw new Error((review.errors || []).join(" ／ "));
    if (spec.kind === "story30") {
      const res = await postJson("/api/director/story30/generate", {
        spec,
        images: state.images.map((i) => i.name),
        name: ($("storyName") || {}).value || "",
        aspect: state.aspect,
        advanced: advancedPayload(),
        character_snapshot: state.characterSnapshot,
        temp_model: directorTempModelPayload(),
        session_id: directorSessionId(false),
        llm: { master_used: state.director.masterUsed,
               clips_used: state.director.used },
      });
      toast("30秒ストーリーを開始しました。ストーリーモードで進捗を確認できます。");
      await refreshStory(res.story_id);
      setMode("story");
    } else {
      const res = await postJson("/api/director/generate", {
        spec,
        images: state.images.map((i) => i.name),
        mode: state.genMode,
        aspect: state.aspect,
        advanced: advancedPayload(),
        character_snapshot: state.characterSnapshot,
        temp_model: directorTempModelPayload(),
        session_id: directorSessionId(false),
        llm: { used: state.director.used },
      });
      state.lastAction = "director";
      state.projectId = res.project_id;
      $("cardStatus").hidden = false;
      $("cardResult").hidden = true;
      listen(res.project_id);
      setMode("single");
      $("cardStatus").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (e) {
    toast(e.message);
    setBusy(false);
  }
}

// ------------------------------------------------- Character Library ----
state.character = { list: [], selectedId: "" };
state.characterSnapshot = null;

function characterById(id) {
  return (state.character.list || []).find((c) => c.character_id === id) || null;
}

async function loadCharacters() {
  try {
    const res = await api("/api/character/list");
    state.character.list = res.characters || [];
  } catch (e) {
    state.character.list = [];
  }
  renderCharacterSelects();
  renderCharacterList();
  renderIdentity();
}

function renderCharacterSelects() {
  ["characterSelect", "directorCharacter"].forEach((sid) => {
    const sel = $(sid);
    if (!sel) return;
    const cur = state.character.selectedId;
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "（未選択）";
    sel.appendChild(none);
    state.character.list.forEach((c) => {
      const o = document.createElement("option");
      o.value = c.character_id;
      o.textContent = c.display_name || c.character_id;
      sel.appendChild(o);
    });
    sel.value = cur;
  });
}

function syncCharacterSelects(fromId) {
  ["characterSelect", "directorCharacter"].forEach((sid) => {
    const sel = $(sid);
    if (sel && sid !== fromId) sel.value = state.character.selectedId;
  });
  renderIdentity();
}

function characterImagesChanged() {
  // A manual image edit after apply drops the snapshot so the normal
  // profile-cache path takes over (server revalidates anyway).
  const snap = state.characterSnapshot;
  if (snap && JSON.stringify(snap.refs || []) !==
      JSON.stringify(state.images.map((i) => i.name))) {
    state.characterSnapshot = null;
  }
  renderIdentity();
}

async function applySelectedCharacter() {
  const id = state.character.selectedId;
  if (!id) { toast("キャラクターを選んでください。"); return; }
  try {
    const res = await postJson("/api/character/apply", { character_id: id });
    // Replace the whole reference set as one unit (never merges with old).
    state.images = (res.images || []).map((name) => ({
      name, thumb: `/api/thumb/${name}`,
    }));
    state.characterSnapshot = res.snapshot || null;
    renderSlots();
    renderIdentity();
    refreshButtons();
    toast("キャラクターを適用しました。");
  } catch (e) {
    toast(e.message);
  }
}

function renderCharacterList() {
  const box = $("characterList");
  if (!box) return;
  box.innerHTML = "";
  state.character.list.forEach((c) => {
    const el = document.createElement("div");
    el.className = "charitem";
    if (c.thumb) {
      const im = document.createElement("img");
      im.src = c.thumb;
      im.alt = c.display_name || "";
      el.appendChild(im);
    }
    const tx = document.createElement("div");
    const nm = document.createElement("div");
    nm.className = "nm";
    nm.textContent = c.display_name || c.character_id;
    const mt = document.createElement("div");
    mt.className = "meta2";
    mt.textContent = `参照 ${(c.refs || []).length}枚` +
      (c.voice_wav ? "・声あり" : "");
    tx.append(nm, mt);
    el.appendChild(tx);
    const sp = document.createElement("div");
    sp.className = "sp";
    el.appendChild(sp);
    const use = document.createElement("button");
    use.type = "button";
    use.className = "btn";
    use.textContent = "使用";
    use.onclick = async () => {
      state.character.selectedId = c.character_id;
      renderCharacterSelects();
      await applySelectedCharacter();
    };
    const det = document.createElement("button");
    det.type = "button";
    det.className = "btn";
    det.textContent = "詳細";
    det.onclick = () => showCharacterDetail(c.character_id);
    el.append(use, det);
    box.appendChild(el);
  });
  if (!state.character.list.length) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "保存済みキャラクターはまだありません。「現在の人物を新規保存」から追加できます。";
    box.appendChild(p);
  }
}

function showCharacterDetail(id) {
  const c = characterById(id);
  const box = $("characterDetail");
  if (!c || !box) return;
  box.hidden = false;
  box.innerHTML = "";
  const rows = [
    ["名前", c.display_name || ""],
    ["参照画像", `${(c.refs || []).length}枚`],
    ["プロフィール", (c.profile_text || "").slice(0, 400)],
    ["声", c.voice_wav ? `あり${c.voice_note ? `（${c.voice_note}）` : ""}` : "なし"],
    ["メモ", c.memo || ""],
  ];
  rows.forEach(([k, v]) => {
    const r = document.createElement("div");
    r.className = "metarow";
    const kk = document.createElement("span");
    kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("span");
    vv.className = "v"; vv.textContent = v || "-";
    r.append(kk, vv);
    box.appendChild(r);
  });
  const acts = document.createElement("div");
  acts.className = "actions";
  const mk = (label, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn";
    b.textContent = label;
    b.onclick = fn;
    acts.appendChild(b);
    return b;
  };
  mk("このキャラクターを使用", async () => {
    state.character.selectedId = id;
    renderCharacterSelects();
    await applySelectedCharacter();
  });
  mk("名前変更", async () => {
    const name = window.prompt("新しい名前", c.display_name || "");
    if (name === null) return;
    try {
      await postJson("/api/character/rename",
                     { character_id: id, name });
      await loadCharacters();
    } catch (e) { toast(e.message); }
  });
  mk("更新（現在の画像・設定で上書き）", async () => {
    if (!window.confirm("現在の参照画像で上書きしますか？")) return;
    try {
      const res = await postJson("/api/character/update", {
        character_id: id,
        images: state.images.map((i) => i.name),
      });
      state.character.selectedId = id;
      await loadCharacters();
      showCharacterDetail(id);
      toast("更新しました。" + (res.character ? "" : ""));
    } catch (e) { toast(e.message); }
  });
  mk("複製", async () => {
    try {
      const res = await postJson("/api/character/duplicate",
                                 { character_id: id });
      await loadCharacters();
      showCharacterDetail(res.character.character_id);
    } catch (e) { toast(e.message); }
  });
  const del = mk("削除", async () => {
    if (!window.confirm(`「${c.display_name}」を削除します。過去の作品には影響しません。よろしいですか？`)) return;
    try {
      await postJson("/api/character/delete", { character_id: id });
      if (state.character.selectedId === id) state.character.selectedId = "";
      await loadCharacters();
      box.hidden = true;
    } catch (e) { toast(e.message); }
  });
  del.classList.add("danger");
  box.appendChild(acts);
}

function openCharacterEditor() {
  const ed = $("characterEditor");
  if (!ed) return;
  ed.hidden = false;
  $("charEditTitle").textContent = "現在の人物を新規保存";
  $("charName").value = "";
  $("charMemo").value = "";
  const prof = (state.project && state.project.profile) || "";
  $("charProfile").value = prof;
  $("characterDetail").hidden = true;
  ed.dataset.editing = "";
}

async function saveCharacterEditor() {
  const editing = $("characterEditor").dataset.editing || "";
  const payload = {
    name: $("charName").value.trim(),
    memo: $("charMemo").value,
    profile_text: $("charProfile").value,
    images: state.images.map((i) => i.name),
  };
  try {
    let res;
    if (editing) {
      payload.character_id = editing;
      res = await postJson("/api/character/update", payload);
    } else {
      res = await postJson("/api/character/save", payload);
    }
    await loadCharacters();
    $("characterEditor").hidden = true;
    if (res.character) showCharacterDetail(res.character.character_id);
    toast("保存しました。");
  } catch (e) {
    toast(e.message);
  }
}

// ------------------------------------------------------- AI settings ----
// The settings dialog (保存先 / GPU / LLM接続) lives entirely in settings.js.
// app.js only keeps the minimal glue the rest of the app depends on: the
// director session id, and a read-only accessor for "今回だけ使用するモデル"
// (temp_model), which settings.js maintains via window.H3Settings.
state.ai = { tempModel: null };

function directorSessionId(fresh) {
  if (fresh || !state.director.sessionId) {
    const parts = new Uint32Array(4);
    (window.crypto || {}).getRandomValues
      ? window.crypto.getRandomValues(parts)
      : parts.map(() => Math.floor(Math.random() * 4294967296));
    state.director.sessionId =
      [...parts].map((p) => p.toString(16).padStart(8, "0")).join("");
  }
  return state.director.sessionId;
}

// Called by settings.js whenever the director temp-model select changes.
function setDirectorTempModel(m) {
  state.ai.tempModel = (m && m.model) ? m : null;
}

function directorTempModelPayload() {
  return state.ai.tempModel || null;
}

function wireAiSettings() {
  // Dialog open/close, tabs, storage/GPU/LLM tab logic: settings.js.
  if (window.H3Settings && typeof window.H3Settings.wire === "function") {
    window.H3Settings.wire();
  }
}

// ============================================================== WP-D ====
// Safe shutdown UI: header button (+ settings dialog copy) -> confirm dialog
// -> POST /api/shutdown -> per-step result. Closing the tab never stops H3.
function wireShutdown() {
  const dlg = $("shutdownDialog");
  const result = $("shutdownResult");
  const stepsBox = $("shutdownSteps");
  const btnDo = $("btnShutdownDo");
  const btnForce = $("btnShutdownForce");
  const btnCancel = $("btnShutdownCancel");
  if (!dlg || !btnDo) return;

  function openShutdownDialog() {
    if (result) { result.hidden = true; result.textContent = ""; result.className = "sysmsg"; }
    if (stepsBox) { stepsBox.hidden = true; stepsBox.textContent = ""; }
    if (btnForce) btnForce.hidden = true;
    btnDo.disabled = false;
    btnDo.textContent = "終了する";
    if (typeof dlg.showModal === "function" && !dlg.open) dlg.showModal();
  }

  function renderShutdownResult(data) {
    const steps = (data && data.steps) || [];
    if (result) {
      const failed = (data && data.failed) || [];
      if (data && data.ok) {
        result.textContent = "H3は終了しました。再起動は RUN_H3.bat です。";
        result.className = "sysmsg ok";
      } else {
        result.textContent = failed.length
          ? `終了処理に失敗した項目があります: ${failed.join("、")}`
          : "終了処理が完了しませんでした。STOP_H3.bat で再度お試しください。";
        result.className = "sysmsg ng";
      }
      result.hidden = false;
    }
    if (stepsBox) {
      stepsBox.textContent = "";
      steps.forEach((s) => {
        const p = document.createElement("p");
        const mark = s.skipped ? "--" : (s.ok ? "OK" : "NG");
        p.textContent = `[${mark}] ${s.name}: ${s.message || ""}`;
        stepsBox.appendChild(p);
      });
      stepsBox.hidden = steps.length === 0;
    }
    btnDo.disabled = !!(data && data.ok);
  }

  async function requestShutdown(interrupt) {
    btnDo.disabled = true;
    if (btnForce) btnForce.hidden = true;
    try {
      const data = await postJson("/api/shutdown", { interrupt: !!interrupt });
      renderShutdownResult(data);
    } catch (e) {
      if (e && e.status === 409) {
        // Busy: offer force-stop instead of failing silently.
        if (result) {
          result.textContent = (e.body && e.body.message) || "生成中です。中止して終了しますか？";
          result.className = "sysmsg ng";
          result.hidden = false;
        }
        if (btnForce) btnForce.hidden = false;
        btnDo.disabled = false;
      } else {
        if (result) {
          result.textContent = (e && e.message) || "終了要求に失敗しました。";
          result.className = "sysmsg ng";
          result.hidden = false;
        }
        btnDo.disabled = false;
      }
    }
  }

  const btn1 = $("btnShutdown");
  if (btn1) btn1.onclick = openShutdownDialog;
  const btn2 = $("btnShutdown2");
  if (btn2) btn2.onclick = openShutdownDialog;
  btnDo.onclick = () => requestShutdown(false);
  if (btnForce) btnForce.onclick = () => requestShutdown(true);
  if (btnCancel) btnCancel.onclick = () => dlg.close();
}

// ------------------------------------------------------- media cleanup ----
const cleanupState = { token: "", candidates: [], age: "1" };

function fmtMB(bytes) {
  const mb = bytes / 1048576;
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${mb.toFixed(1)} MB`;
}

async function cleanupScan() {
  const btn = $("btnCleanupScan");
  const res = $("cleanupResult");
  const sum = $("cleanupSummary");
  btn.disabled = true;
  try {
    cleanupState.age = $("cleanupAge").value || "1";
    const data = await api(`/api/media/cleanup/scan?max_age_days=${encodeURIComponent(cleanupState.age)}`);
    cleanupState.token = data.token || "";
    cleanupState.candidates = data.candidates || [];
    res.hidden = false;
    const n = data.candidate_count || 0;
    sum.textContent = n === 0
      ? `削除候補はありません（スキャン ${data.scanned_count || 0}件、除外 ${data.excluded_count || 0}件）。`
      : `削除候補 ${n}件 ／ 回収見込み ${fmtMB(data.candidate_bytes || 0)}（スキャン ${data.scanned_count || 0}件、除外 ${data.excluded_count || 0}件）。` +
        (data.folders || []).map((f) => `${f.folder.split(/[/\\]/).pop()}: ${f.files}件`).join(" ／ ");
    const wrap = $("cleanupListWrap");
    const list = $("cleanupList");
    if (wrap && list) {
      wrap.hidden = n === 0;
      list.innerHTML = "";
      cleanupState.candidates.slice(0, 200).forEach((c) => {
        const row = document.createElement("div");
        row.className = "crow";
        const sz = document.createElement("span");
        sz.className = "csize";
        sz.textContent = fmtMB(c.size);
        const nm = document.createElement("span");
        nm.textContent = String(c.path).split(/[/\\]/).pop();
        nm.title = c.path;
        row.append(sz, nm);
        list.appendChild(row);
      });
      if (n > 200) {
        const more = document.createElement("div");
        more.textContent = `…ほか ${n - 200}件`;
        list.appendChild(more);
      }
    }
    $("btnCleanupDo").disabled = n === 0;
    $("cleanupDone").hidden = true;
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
  }
}

async function cleanupDo() {
  const n = cleanupState.candidates.length;
  if (!n || !cleanupState.token) return;
  if (!window.confirm(`内部コピーを ${n}件 削除します。H3_Media本体は削除されません。よろしいですか？`)) return;
  const btn = $("btnCleanupDo");
  btn.disabled = true;
  try {
    const data = await postJson("/api/media/cleanup", {
      token: cleanupState.token,
      ids: cleanupState.candidates.map((c) => c.path),
      max_age_days: cleanupState.age,
    });
    const el = $("cleanupDone");
    const method = data.method === "permanent" ? "（完全削除）" : data.method === "recycle" ? "（ごみ箱へ移動）" : "";
    el.hidden = false;
    el.textContent = `${data.removed || 0} files removed${method} ／ ${fmtMB(data.recovered_bytes || 0)} recovered ／ ` +
      `failed: ${(data.failed || []).length} ／ skipped: ${(data.skipped || []).length}`;
    if (!data.ok && data.error) toast(data.error);
    cleanupState.token = "";
    cleanupState.candidates = [];
    $("btnCleanupDo").disabled = true;
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
  }
}

function cleanupCancel() {
  cleanupState.token = "";
  cleanupState.candidates = [];
  $("cleanupResult").hidden = true;
}
function renderIdentity() {
  const box = $("identityRows");
  if (!box) return;
  const storedSnap = (state.project && state.project.character_snapshot) ||
    (state.story && state.story.characterSnapshot) || null;
  if (!state.character.selectedId && storedSnap && storedSnap.character_id &&
      characterById(storedSnap.character_id)) {
    state.character.selectedId = storedSnap.character_id;
    ["characterSelect", "directorCharacter"].forEach((sid) => {
      const selEl = $(sid);
      if (selEl) selEl.value = state.character.selectedId;
    });
  }
  const sel = characterById(state.character.selectedId);
  const snap = state.characterSnapshot;
  const charState = !sel ? "未選択"
    : snap && JSON.stringify(snap.refs || []) ===
      JSON.stringify(state.images.map((i) => i.name))
      ? `${sel.display_name}（適用中）` : `${sel.display_name}（画像が変更されました）`;
  const missingNote = (state.project && state.project.character_missing) ||
    (state.story && state.story.characterMissing);
  const rows = [
    ["キャラクター", charState],
    ["参照画像", state.images.length ? `${state.images.length} 枚設定済み` : "未設定"],
    ["Character", "生成時にプロフィールを自動抽出"],
    ["Voice", state.story && state.story.id ? "ストーリー進行に合わせて自動管理" : "LONG_FAST 選択時に自動固定"],
  ];
  if (missingNote) {
    rows.push(["注意", "元のLibrary Characterは削除済み。保存済みsnapshotを使用しています（生成可）"]);
  }
  box.innerHTML = "";
  rows.forEach(([k, v]) => {
    const r = document.createElement("div");
    r.className = "metarow";
    const kk = document.createElement("span");
    kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("span");
    vv.className = "v"; vv.textContent = v;
    r.append(kk, vv);
    box.appendChild(r);
  });
  refreshAudioTestButtons();
}

function storyIsResumable() {
  const st = state.story;
  if (!st.id || st.running) return false;
  if (st.status === "paused" || st.status === "error") return true;
  return st.cursor > 0 && st.cursor < st.segments.length;
}

function setBusy(v) {
  state.busy = v;
  refreshButtons();
}

// ------------------------------------------------------------------ utils --
async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { body = { ok: false, message: "サーバー応答を解釈できませんでした。" }; }
  if (!res.ok || body.ok === false) {
    // The whole body travels with the error: a refusal can carry the pre-flight
    // list, and showing that list is far more useful than a bare toast.
    const err = new Error(body.message || (body.errors || []).join(" ／ ") || `HTTP ${res.status}`);
    err.body = body;
    err.status = res.status;
    throw err;
  }
  return body;
}

function postJson(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4200);
}

function fmtSeconds(s) {
  return `${Math.round(s)}秒`;
}

function fmtDate(ts) {
  if (!ts) return "-";
  try {
    return new Date(ts * 1000).toLocaleString("ja-JP");
  } catch (e) {
    return "-";
  }
}

let releaseTimer = null;
function showRelease(message, isError) {
  const el = $("releaseStatus");
  el.textContent = message;
  el.className = "sysmsg " + (isError ? "ng" : "ok");
  el.hidden = false;
  clearTimeout(releaseTimer);
  releaseTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

// ------------------------------------------------------- v2 modes/compat ---
const MODE_SUB = {
  FAST: "短時間生成",
  QUALITY: "品質優先",
  LONG_FAST: "長尺向け高速生成",
  LEGACY: "旧互換モード",
  LONG: "ストーリー連鎖",
  BALANCED: "中間設定",
  LOW_VRAM: "省VRAM",
};
function modeTier(id) {
  if (id === "QUALITY") return "quality";
  if (id === "LEGACY") return "legacy";
  if (id === "FAST" || id === "LONG_FAST" || id === "LONG") return "fast";
  return "";
}
function renderModes(cfg) {
  const box = $("genModes");
  if (!box) return;
  box.innerHTML = "";
  box.className = "modecards";
  (cfg.modes || [{ id: "LEGACY", label: "LEGACY", desc: "" }]).forEach((m) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "modecard" + (m.id === state.genMode ? " on" : "");
    b.dataset.tier = modeTier(m.id);
    const t = document.createElement("div");
    t.className = "t " + modeTier(m.id);
    t.textContent = m.label || m.id;
    const d = document.createElement("div");
    d.className = "d";
    d.textContent = MODE_SUB[m.id] || m.desc || "";
    b.title = m.desc || "";
    b.append(t, d);
    b.onclick = () => {
      state.genMode = m.id;
      [...box.children].forEach((c) => c.classList.remove("on"));
      b.classList.add("on");
      $("modeDesc").textContent = m.desc || "";
      refreshButtons();
    };
    box.appendChild(b);
    if (m.id === state.genMode) $("modeDesc").textContent = m.desc || "";
  });
  const adv = $("genModesAdv");
  if (adv) {
    adv.innerHTML = "";
    (cfg.modes_advanced || []).forEach((m) => {
      // Story専用モード（例: LONG_FAST）は単発生成モードのAdvancedには出さない。
      if (STORY_MODE_IDS.includes(m.id) && state.mode !== "story") return;
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice";
      b.textContent = (m.label || m.id) + "（Advanced）";
      b.title = m.desc || "";
      b.onclick = () => { state.genMode = m.id; $("modeDesc").textContent = (m.desc || "") + " — Advanced"; };
      adv.appendChild(b);
    });
  }
  const exp = $("genModesExp");
  if (exp) {
    exp.innerHTML = "";
    (cfg.modes_experimental || []).forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice off";
      b.textContent = (m.label || m.id) + "（利用不可）";
      b.title = m.reason || "";
      b.disabled = true;
      exp.appendChild(b);
    });
    const note = $("expNote");
    if (note) {
      note.textContent = (cfg.modes_experimental || [])
        .map((m) => `${m.label || m.id}: ${m.reason || ""}`).join(" ／ ");
    }
  }
}

function renderCompat(compat) {
  const el = $("compatStatus");
  const parts = compat ? [
    `ComfyUI ${compat.comfyui || "?"}`,
    `GPU0 ${compat.gpu0 || "?"}`,
    `GPU1 ${compat.gpu1 || "?"}`,
    `H3 ${compat.h3_ref2va || "?"}`,
    `Turbo ${compat.turbo_lora || "?"}`,
  ] : ["互換性: 不明"];
  const modes = compat ? Object.entries(compat.modes || {})
    .map(([k, v]) => `${k} ${v ? "READY" : "不可"}`).join(" / ") : "";
  if (el) {
    el.textContent = "互換性: " + parts.join(" / ") + (modes ? " — " + modes : "");
    el.className = "engine sr-only " + (compat && compat.comfyui === "NG" ? "ng" : "ok");
  }
  // Topbar chips: compact always-visible state.
  const chips = $("sysChips");
  if (chips) {
    chips.innerHTML = "";
    const dot = (ok) => (ok === "OK" ? "●" : ok === "NG" ? "●" : "○");
    const chip = (text, cls) => {
      const s = document.createElement("span");
      s.className = "chip " + (cls || "");
      s.textContent = text;
      chips.appendChild(s);
    };
    if (!compat) { chip("System ?", ""); return; }
    const ready = compat.comfyui !== "NG";
    chip(`ComfyUI ${dot(compat.comfyui)} ${compat.comfyui === "NG" ? "OFFLINE" : "READY"}`,
      compat.comfyui === "NG" ? "ng" : "ok");
    const shortGpu = (n) => (String(n || "?").match(/RTX\s?\S+(?:\s?Ti)?/) || [n || "?"])[0];
    chip(`GPU0 ${shortGpu(compat.gpu0)}`, "");
    chip(`GPU1 ${shortGpu(compat.gpu1)}`, "");
    chip(state.genMode || "FAST", "mode");
    chip(state.busy ? "WORKING" : "IDLE", state.busy ? "busy" : "");
    const sum = $("diagSummary");
    if (sum) sum.textContent = ready ? "System ● READY" : "System ● ISSUE — 詳細を開く";
  }
  // Diagnostics details: full list, out of the way until opened.
  const body = $("diagBody");
  if (body) {
    body.innerHTML = "";
    if (!compat) { body.textContent = "互換性: 不明"; return; }
    parts.concat(modes ? [modes] : []).forEach((p) => {
      const s = document.createElement("span");
      const bad = /NG|不可|\?/.test(p);
      s.className = bad ? "d-ng" : "d-ok";
      s.textContent = p;
      body.appendChild(s);
    });
  }
}

async function clipLock(index, action) {
  try {
    const res = await postJson("/api/story/clip-lock", {
      story_id: state.story.id, index, action,
    });
    const seg = state.story.segments[res.index];
    if (seg) seg.locked = res.locked;
    renderSegments();
    toast(res.locked ? `クリップ ${res.index + 1} を${res.locked === "approved" ? "承認" : "ロック"}しました。`
      : `クリップ ${res.index + 1} のロックを解除しました。`);
  } catch (e) {
    toast(e.message);
  }
}

async function regenClip(index) {
  try {
    const res = await postJson("/api/story/regenerate", {
      story_id: state.story.id, indices: [index],
    });
    toast(`クリップ ${res.rebuilt.map((i) => i + 1).join(", ")} を再生成します。` +
      (res.kept.length ? `（ロック済み ${res.kept.map((i) => i + 1).join(", ")} は保持）` : "") +
      "ストーリー開始で再開してください。");
    await refreshStory(state.story.id);
  } catch (e) {
    toast(e.message);
  }
}

// ------------------------------------------------------------------ boot ---
async function boot() {
  // The whole UI is driven by /api/config: modes, presets, defaults, compat.
  // Nothing below can run before it arrives.
  const cfg = await api("/api/config");
  state.cfg = cfg;
  state.seconds = cfg.defaults.seconds;
  // 動作プリセットはユーザーのデータ。config と一緒に届くが、
  // 追加・削除のたびに /api/action-presets で取り直す。
  applyPresetsPayload({
    presets: cfg.action_presets || (cfg.presets && cfg.presets.actions) || [],
    template: cfg.action_preset_template,
    max_name: cfg.action_preset_max_name,
  });
  wireActionPresetDialog();

  $("engineStatus").textContent = cfg.comfy_reachable
    ? "エンジン: 準備できています"
    : "エンジン: 生成をはじめたときに自動で起動します（最初の1回は数分かかります）";
  $("engineStatus").className = "engine " + (cfg.comfy_reachable ? "ok" : "ng");

  renderModes(cfg);
  renderCompat(cfg.compat);

  // seconds / aspect choices
  const secBox = $("secondsChoices");
  cfg.presets.seconds.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "choice" + (s === state.seconds ? " on" : "");
    b.textContent = s === cfg.defaults.seconds ? `${s}秒（推奨）` : `${s}秒`;
    b.onclick = () => {
      state.seconds = s;
      [...secBox.children].forEach((c) => c.classList.remove("on"));
      b.classList.add("on");
    };
    secBox.appendChild(b);
  });

  const aspBox = $("aspectChoices");
  cfg.presets.aspects.forEach((a) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "choice" + (a.id === state.aspect ? " on" : "");
    b.textContent = a.measured ? `${a.label}（推奨）` : `${a.label}（未測定）`;
    b.onclick = () => {
      state.aspect = a.id;
      [...aspBox.children].forEach((c) => c.classList.remove("on"));
      b.classList.add("on");
      $("advWidth").value = a.width;
      $("advHeight").value = a.height;
      window.H3Studio?.sync();
    };
    aspBox.appendChild(b);
  });

  fillSelect($("advSampler"), cfg.presets.samplers, cfg.defaults.sampler);
  fillSelect($("advScheduler"), cfg.presets.schedulers, cfg.defaults.scheduler);
  fillSelect($("advRefSize"), cfg.presets.ref_image_sizes, cfg.defaults.ref_image_size);
  $("jpNegative").placeholder = cfg.negative_default
    ? "空欄なら推奨の設定を使います"
    : "空欄可";
  $("storyNegative").placeholder = $("jpNegative").placeholder;
  resetAdvanced();
  renderSlots();
  renderStages(null);
  renderMotionChips();
  renderSegments();
  renderStoryPanel(null);

  $("genHint").textContent =
    `既定の設定で 1本あたり およそ ${Math.round(cfg.measured_seconds)}秒 かかります。`;

  refreshStoryList().catch(() => {});
  // Saved AI settings first, model list after: the saved provider/model must
  // be visible even if the list fetch is slow, offline, 429 or errors.
  // List failure never clears the saved selection (temporary option stays).
  if (window.H3Settings && typeof window.H3Settings.boot === "function") {
    window.H3Settings.boot().catch(() => {});
  }
}

function fillSelect(sel, values, current) {
  sel.innerHTML = "";
  values.forEach((v) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  });
}

function resetAdvanced() {
  const d = state.cfg.defaults;
  $("advSeed").value = d.seed;
  $("advRandomSeed").checked = !!d.randomize_seed;
  $("advSteps").value = d.steps;
  $("advSampler").value = d.sampler;
  $("advScheduler").value = d.scheduler;
  $("advRefSize").value = d.ref_image_size;
  const preset = state.cfg.presets.aspects.find((a) => a.id === state.aspect)
    || state.cfg.presets.aspects[0];
  $("advWidth").value = preset.width;
  $("advHeight").value = preset.height;
  $("advFramesMode").value = "auto";
  $("advFrames").value = d.frames;
  $("advPrefix").value = "";
}

function advancedPayload() {
  return {
    speech_delivery: $("speechDelivery").value,
    seed: parseInt($("advSeed").value, 10),
    randomize_seed: $("advRandomSeed").checked,
    steps: parseInt($("advSteps").value, 10),
    sampler: $("advSampler").value,
    scheduler: $("advScheduler").value,
    ref_image_size: $("advRefSize").value,
    width: parseInt($("advWidth").value, 10),
    height: parseInt($("advHeight").value, 10),
    frames_mode: $("advFramesMode").value,
    frames: parseInt($("advFrames").value, 10),
    output_prefix: $("advPrefix").value,
  };
}

// ------------------------------------------------------------------ mode ---
function setMode(mode) {
  state.mode = mode === "story" ? "story" : mode === "director" ? "director"
    : mode === "audio" ? "audio" : "single";
  const isStory = state.mode === "story";
  const isDirector = state.mode === "director";
  const isAudio = state.mode === "audio";
  if (!isStory && STORY_MODE_IDS.includes(state.genMode)) {
    // Story専用モードのまま単発生成モードへ切り替えると、STORY_ONLY_MODESを
    // 検証しないまま/api/generateへ送ってしまう。既定の単発モードへ戻す。
    state.genMode = "FAST";
    $("modeDesc").textContent = "";
  }
  if (state.cfg) renderModes(state.cfg);
  $("btnModeSingle").classList.toggle("on", !isStory && !isDirector && !isAudio);
  $("btnModeStory").classList.toggle("on", isStory);
  const btnDir = $("btnModeDirector");
  if (btnDir) btnDir.classList.toggle("on", isDirector);
  const btnAudio = $("btnModeAudio");
  if (btnAudio) btnAudio.classList.toggle("on", isAudio);
  $("modeHint").textContent = isAudio
    ? "台詞の発音だけを短時間で確認します。動画は作りません。"
    : isDirector
    ? "アイデアを書いて監督案を作り、気になる部分だけ注文して動画を作ります。"
    : isStory
    ? "長い文章を場面ごとに分けて、続き物の動画を作ります。"
    : "1本だけ作るモードです。";

  document.querySelectorAll(".mode-single").forEach((el) => {
    if (el.id === "cardResult") {
      el.hidden = isStory || isDirector || isAudio ||
        !(state.project && state.project.clips && state.project.clips.length);
    } else {
      el.hidden = isStory || isDirector || isAudio;
    }
  });
  document.querySelectorAll(".mode-story").forEach((el) => {
    if (el.id === "cardStoryFinal") {
      el.hidden = !isStory || !state.story.merged;
    } else {
      el.hidden = !isStory;
    }
  });
  document.querySelectorAll(".mode-director").forEach((el) => {
    if (el.id === "cardDirectorResult") {
      el.hidden = !isDirector || !state.director.spec;
    } else {
      el.hidden = !isDirector;
    }
  });
  document.querySelectorAll(".mode-audio").forEach((el) => {
    el.hidden = !isAudio;
  });

  // 単発生成の開始ボタンと中止ボタンは、単発生成モードのときだけ意味がある。
  $("btnGenerate").hidden = isStory || isAudio;
  $("genHint").hidden = isStory || isAudio;
  $("btnCancel").hidden = isStory || isAudio;
  // Directorモードでは生成設定の出力時間は使わない（Director尺に一本化）。
  // 矛盾状態（生成設定5秒＋Director30秒）を作らせないためのhide。
  const secRow = $("secondsRow");
  if (secRow) secRow.hidden = isDirector;
  // 音声テストモードでは cardSettings（生成モード/秒数/画面比）と
  // cardAdvanced はどの値も使わない。他モードでの挙動は変えない。
  const settingsCard = $("cardSettings");
  if (settingsCard) settingsCard.hidden = isAudio;
  const advancedCard = $("cardAdvanced");
  if (advancedCard) advancedCard.hidden = isAudio;
  // Side nav follows the same mode split.
  document.querySelectorAll("#sidenav [data-nav]").forEach((a) => {
    const modes = ["single", "story", "director", "audio"].filter((m) => m in a.dataset);
    a.classList.toggle("navhide", modes.length > 0 && !modes.includes(state.mode));
  });
  if (isAudio) {
    refreshAudioTestList().catch(() => {});
  }
  refreshButtons();
}

// ---------------------------------------------------------------- images ---
function renderSlots() {
  const box = $("slots");
  box.innerHTML = "";
  for (let i = 0; i < 4; i++) {
    const img = state.images[i];
    const el = document.createElement("div");
    el.className = "slot" + (img ? " filled" : "");
    if (img) {
      const im = document.createElement("img");
      im.src = img.thumb;
      im.alt = ROLES[i];
      const role = document.createElement("div");
      role.className = "role";
      role.textContent = ROLES[i];
      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn";
      del.textContent = "削除";
      del.onclick = () => {
        state.images.splice(i, 1);
        renderSlots();
        characterImagesChanged();
      };
      el.append(im, role, del);
    } else {
      const role = document.createElement("div");
      role.className = "role";
      role.textContent = ROLES[i];
      const p = document.createElement("div");
      p.className = "role";
      p.textContent = "（空き）";
      el.append(role, p);
    }
    box.appendChild(el);
  }
  const ready = state.images.length > 0;
  refreshButtons();
  $("imageWarning").hidden = ready;
}

async function uploadFiles(files) {
  const room = 4 - state.images.length;
  if (room <= 0) { toast("参照画像は最大4枚までです。"); return; }

  // Non-image files are rejected client-side (named) before anything is sent.
  const all = [...files];
  const nonImages = all.filter((f) => f.type && !f.type.startsWith("image/"));
  const images = all.filter((f) => !f.type || f.type.startsWith("image/"));
  if (nonImages.length) {
    toast(`画像以外のファイルは追加しませんでした: ${nonImages.map((f) => f.name).join(" ／ ")}`);
  }
  if (!images.length) return;

  const overflow = images.length - room;
  const picked = images.slice(0, room);
  if (overflow > 0) {
    toast(`参照画像は最大4枚までです（${overflow}枚は追加しませんでした）`);
  }
  if (!picked.length) return;

  const fd = new FormData();
  picked.forEach((f) => fd.append("files", f, f.name));

  const btn = $("btnAddImage");
  const label = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "画像を追加しています…"; }
  try {
    const res = await api("/api/images", { method: "POST", body: fd });
    state.images.push(...res.images);
    renderSlots();
    characterImagesChanged();
  } catch (e) {
    const body = (e && e.body) || {};
    toast(body.message || e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = label; }
  }
}

// ---------------------------------------------------------------- stages ---
function renderStages(progress) {
  const list = $("stageList");
  const names = state.cfg ? state.cfg.stages : [];
  const stages = progress ? progress.stages : names.map((n) => ({ name: n, status: "待機" }));
  list.innerHTML = "";
  stages.forEach((s) => {
    const li = document.createElement("li");
    li.className = s.status === "実行中" ? "run"
      : s.status === "完了" ? "done"
      : s.status === "失敗" ? "fail" : "";
    const a = document.createElement("span");
    a.textContent = s.name;
    const b = document.createElement("span");
    b.className = "st";
    b.textContent = s.status;
    li.append(a, b);
    list.appendChild(li);
  });

  if (progress) {
    $("progressBar").style.width = (progress.percent || 0) + "%";
    $("elapsed").textContent = "経過 " + fmtSeconds(progress.elapsed || 0);
    $("etaText").textContent = progress.eta ? "目安 およそ " + fmtSeconds(progress.eta) : "";
  }
}

function showError(err) {
  const box = $("errorCard");
  if (!err) {
    box.hidden = true;
    $("btnStoryResume").hidden = true;
    return;
  }
  $("errTitle").textContent = err.title || "エラーが起きました";
  $("errWhere").textContent = err.where || "";
  $("errRetry").textContent = err.retry || "";
  $("errConfig").textContent = err.config || "";
  $("errDetail").textContent = err.detail || "";
  $("btnRetry").hidden = state.mode === "story" || !err.retryable || !state.lastAction;
  $("btnStoryResume").hidden = state.mode !== "story" || !state.story.id;
  box.hidden = false;
}

// ------------------------------------------------------------- generation --
function listen(projectId) {
  if (state.events) state.events.close();
  const es = new EventSource(`/api/events/${projectId}`);
  state.events = es;
  es.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    renderStages(data);
    showError(data.error);
    if (data.finished) {
      es.close();
      state.events = null;
      $("btnCancel").disabled = true;
      setBusy(false);
      if (!data.error) loadProject(projectId);
    }
  };
  es.addEventListener("end", () => { es.close(); state.events = null; });
  es.onerror = () => { /* the poll below still refreshes the final state */ };
}

async function startGeneration(kind) {
  $("cardStatus").hidden = false;
  $("cardResult").hidden = true;
  $("btnCancel").disabled = false;
  setBusy(true);
  showError(null);
  renderStages(null);
  $("progressBar").style.width = "0%";

  try {
    let res;
    if (kind === "new") {
      res = await api("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          images: state.images.map((i) => i.name),
          jp_scene: $("jpScene").value,
          jp_dialogue: $("jpDialogue").value,
          jp_negative: $("jpNegative").value,
          seconds: state.seconds,
          aspect: state.aspect,
          mode: state.genMode,
          advanced: advancedPayload(),
          character_snapshot: state.characterSnapshot,
        }),
      });
    } else {
      const path = kind === "again" ? "/api/again" : "/api/continue";
      res = await api(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: state.projectId }),
      });
    }
    state.lastAction = kind;
    state.projectId = res.project_id;
    listen(res.project_id);
    $("cardStatus").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast(e.message);
    showError({
      title: "生成をはじめられませんでした",
      where: "止まった場所: 開始前",
      retry: "再試行: はい",
      config: "設定の問題: " + e.message,
      retryable: true,
      detail: e.message,
    });
    // The job never started, so nothing is holding the models.
    setBusy(false);
  }
}

async function loadProject(projectId) {
  try {
    const res = await api(`/api/project/${projectId}`);
    state.project = res.project;
    renderResult();
  } catch (e) {
    toast(e.message);
  }
}

function renderResult() {
  const p = state.project;
  if (!p || !p.clips.length) return;
  const last = p.clips[p.clips.length - 1];
  $("cardResult").hidden = state.mode !== "single";
  const player = $("player");
  player.src = `/api/video/${p.id}/${last.step}?t=${Date.now()}`;
  player.load();

  const total = Math.round(p.total_seconds);
  const max = Math.round(p.max_seconds);
  $("lengthCounter").textContent = `現在 ${total} / ${max}秒`;

  // v2: where the file is, how long it took, which mode made it.
  // Structured rows for glanceability; full path lives in Details.
  const rows = $("resultRows");
  if (rows) {
    rows.innerHTML = "";
    const items = [];
    if (last.mode) items.push(["モード", last.mode]);
    if (last.video_file) items.push(["ファイル", last.video_file]);
    if (last.frames) items.push(["フレーム", `${last.frames}f`]);
    if (last.seconds) items.push(["動画の長さ", `${Math.round(last.seconds * 10) / 10}秒`]);
    if (last.gen_seconds) items.push(["生成時間", `${Math.round(last.gen_seconds)}秒`]);
    items.forEach(([k, v]) => {
      const r = document.createElement("div");
      r.className = "metarow";
      const kk = document.createElement("span");
      kk.className = "k"; kk.textContent = k;
      const vv = document.createElement("span");
      vv.className = "v"; vv.textContent = v;
      r.append(kk, vv);
      rows.appendChild(r);
    });
  }
  const rpath = $("resultPath");
  if (rpath) rpath.textContent = last.video_dir ? `保存先: ${last.video_dir}` : "";
  const meta = $("resultMeta");
  if (meta) {
    const bits = [];
    if (last.mode) bits.push(`モード ${last.mode}`);
    if (last.video_file) bits.push(`ファイル ${last.video_file}`);
    if (last.gen_seconds) bits.push(`生成 ${Math.round(last.gen_seconds)}秒`);
    meta.textContent = bits.join(" ／ ");
  }

  // Where the finished MP4 actually landed (or why it didn't).
  const exportEl = $("resultExportStatus");
  if (exportEl) {
    const exp = last.export;
    if (exp && exp.ok) {
      exportEl.textContent = `生成動画フォルダに保存: ${exp.name}`;
      exportEl.hidden = false;
    } else if (exp && exp.ok === false) {
      exportEl.textContent =
        `⚠ 生成動画フォルダへの保存に失敗しました（${exp.error || "不明なエラー"}）。動画はプロジェクト内に残っています。`;
      exportEl.hidden = false;
    } else {
      exportEl.textContent = "";
      exportEl.hidden = true;
    }
  }

  // Saved, but something downstream of the save went wrong.
  const warn = $("resultWarning");
  warn.textContent = last.warning || "";
  warn.hidden = !last.warning;

  const cont = $("btnContinue");
  if (!p.can_continue) {
    cont.disabled = true;
    cont.textContent = "最大の長さです";
  } else {
    cont.disabled = false;
    cont.textContent = "この動画の続きを作る";
  }
  refreshButtons();
  if (state.mode === "single") {
    $("cardResult").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ======================================================== ストーリーモード ==
// ======================================================== 台本エディタ ======
// The script is ONE contenteditable field. Colour is presentation only; the
// truth is a run array [{type:"prompt"|"speech", text}] where a run's text may
// contain "\n" for the line breaks inside that run. Type is NEVER inferred from
// colour, and never from the text itself.
//
// DOM shape produced by renderEditor (the canonical form):
//   <div class="ln"><span class="run prompt" data-type="prompt">…</span>…</div>
// An empty line is <div class="ln"><br></div>.
const ZWSP = "\u200B";
const BLOCK_TAGS = /^(DIV|P|LI|UL|OL|SECTION|ARTICLE|BLOCKQUOTE|PRE|H[1-6]|TABLE|TR|TD)$/;

const editorState = {
  mode: "prompt",     // the type new text gets
  composing: false,   // a Japanese IME conversion is in flight: hands off the DOM
  restoring: false,   // we are moving the caret ourselves
  runs: [],           // last known structure
};

const scriptEditor = () => $("scriptEditor");

function isBlockEl(el) {
  return el && el.nodeType === 1 && BLOCK_TAGS.test(el.nodeName);
}

// A <br> that only exists to give an empty (or last) line its height carries no
// line break of its own: the block already provides one. It counts as filler
// when nothing visible follows it inside the same line - an invisible anchor
// span sitting after it does not make it a real line break.
function isTrailingBr(br, root) {
  let n = br;
  while (n && n !== root) {
    for (let s = n.nextSibling; s; s = s.nextSibling) {
      if (s.nodeName === "BR") return false;
      if (visLen(s.textContent || "")) return false;
    }
    const p = n.parentNode;
    if (!p || p === root || isBlockEl(p)) return true;
    n = p;
  }
  return true;
}

function visLen(s) { return s.split(ZWSP).join("").length; }

// visible index -> raw index inside a text node that may hold ZWSP anchors
function rawIndex(raw, visible) {
  let seen = 0;
  for (let i = 0; i < raw.length; i++) {
    if (seen === visible) return i;
    if (raw[i] !== ZWSP) seen++;
  }
  return raw.length;
}

function nodeType(node) {
  let n = node;
  while (n && n !== scriptEditor()) {
    if (n.nodeType === 1 && n.dataset && n.dataset.type) {
      return n.dataset.type === "speech" ? "speech" : "prompt";
    }
    n = n.parentNode;
  }
  return null;
}

// ------------------------------------------------------------ DOM -> array --
function readEditor() {
  const root = scriptEditor();
  const out = [];
  if (!root) return out;
  let started = false;

  const push = (text, type) => {
    if (!text) return;
    const t = type || (out.length ? out[out.length - 1].type : editorState.mode);
    if (out.length && out[out.length - 1].type === t) out[out.length - 1].text += text;
    else out.push({ type: t, text });
    started = true;
  };

  const walk = (node, type) => {
    const kids = node.childNodes;
    for (let i = 0; i < kids.length; i++) {
      const k = kids[i];
      if (k.nodeType === 3) {
        push(k.data.split(ZWSP).join(""), type);
      } else if (k.nodeType !== 1) {
        continue;
      } else if (k.nodeName === "BR") {
        if (!isTrailingBr(k, root)) push("\n", type);
      } else if (isBlockEl(k)) {
        if (started) push("\n", type);
        started = true;
        walk(k, type);
      } else if (k.dataset && k.dataset.type) {
        walk(k, k.dataset.type === "speech" ? "speech" : "prompt");
      } else {
        walk(k, type);
      }
    }
  };
  walk(root, null);
  return out.filter((r) => r.text !== "");
}

// ------------------------------------------------------------ array -> DOM --
function buildScriptDom(runs) {
  const frag = document.createDocumentFragment();
  let blk = document.createElement("div");
  blk.className = "ln";
  frag.appendChild(blk);

  (runs || []).forEach((r) => {
    const type = r.type === "speech" ? "speech" : "prompt";
    const parts = String(r.text == null ? "" : r.text).split("\n");
    parts.forEach((part, i) => {
      if (i > 0) {
        if (!blk.firstChild) blk.appendChild(document.createElement("br"));
        blk = document.createElement("div");
        blk.className = "ln";
        frag.appendChild(blk);
      }
      if (part) {
        const sp = document.createElement("span");
        sp.className = "run " + type;
        sp.dataset.type = type;
        sp.textContent = part;
        blk.appendChild(sp);
      }
    });
  });
  if (!blk.firstChild) blk.appendChild(document.createElement("br"));
  return frag;
}

function canonicalHtml(runs) {
  const box = document.createElement("div");
  box.appendChild(buildScriptDom(runs));
  return box.innerHTML;
}

function renderEditor(runs) {
  const root = scriptEditor();
  if (!root) return;
  const clean = (runs || [])
    .map((r) => ({ type: r.type === "speech" ? "speech" : "prompt",
                   text: String(r.text == null ? "" : r.text) }))
    .filter((r) => r.text !== "");
  root.innerHTML = "";
  root.appendChild(buildScriptDom(clean));
  editorState.runs = clean;
  updateEditorMeta();
}

function updateEditorMeta() {
  const root = scriptEditor();
  if (!root) return;
  const lines = runsToLines(editorState.runs);
  const p = lines.filter((l) => l.type === "prompt").length;
  const s = lines.filter((l) => l.type === "speech").length;
  root.dataset.empty = lines.length ? "0" : "1";
  const stats = $("scriptStats");
  if (stats) stats.textContent = `Prompt ${p}行 ／ Speech ${s}行`;
}

// ------------------------------------------------------------- caret maths --
// Offsets are counted over the visible text, with one character per line break
// and ZWSP anchors ignored - exactly the same numbering the run array uses.
function offsetAt(target, targetOff) {
  const root = scriptEditor();
  let count = 0;
  let started = false;
  let found = null;

  const walk = (node) => {
    if (found !== null) return;
    const kids = node.childNodes;
    for (let i = 0; i < kids.length; i++) {
      if (found !== null) return;
      if (node === target && i === targetOff) { found = count; return; }
      const k = kids[i];
      if (k.nodeType === 3) {
        if (k === target) {
          found = count + visLen(k.data.slice(0, targetOff));
          return;
        }
        const v = visLen(k.data);
        if (v) { count += v; started = true; }
      } else if (k.nodeType === 1) {
        if (k.nodeName === "BR") {
          if (!isTrailingBr(k, root)) { count += 1; started = true; }
        } else if (isBlockEl(k)) {
          if (started) count += 1;
          started = true;
          walk(k);
        } else {
          walk(k);
        }
      }
    }
    if (found === null && node === target && targetOff >= kids.length) found = count;
  };

  walk(root);
  return found === null ? count : found;
}

// The inverse, against the canonical DOM (always used right after a render).
function locateOffset(target) {
  const root = scriptEditor();
  const blocks = [...root.children];
  let count = 0;
  let last = { node: root, off: 0 };
  for (let b = 0; b < blocks.length; b++) {
    if (b > 0) count += 1;
    const blk = blocks[b];
    const spans = [...blk.querySelectorAll("span")];
    if (!spans.length) {
      if (target <= count) return { node: blk, off: 0 };
      last = { node: blk, off: 0 };
      continue;
    }
    for (const sp of spans) {
      const t = sp.firstChild;
      const len = t && t.nodeType === 3 ? t.data.length : 0;
      if (target <= count + len) {
        return t ? { node: t, off: Math.max(0, target - count) } : { node: sp, off: 0 };
      }
      count += len;
      last = t ? { node: t, off: len } : { node: sp, off: 0 };
    }
  }
  return last;
}

function selectionOffsets() {
  const root = scriptEditor();
  const sel = window.getSelection();
  if (!root || !sel || !sel.rangeCount) return null;
  const r = sel.getRangeAt(0);
  if (!root.contains(r.startContainer) || !root.contains(r.endContainer)) return null;
  const a = offsetAt(r.startContainer, r.startOffset);
  const b = offsetAt(r.endContainer, r.endOffset);
  return { start: Math.min(a, b), end: Math.max(a, b) };
}

function setSelectionOffsets(start, end) {
  const root = scriptEditor();
  if (!root) return;
  const sel = window.getSelection();
  if (!sel) return;
  editorState.restoring = true;
  try {
    const a = locateOffset(start);
    const b = locateOffset(typeof end === "number" ? end : start);
    const r = document.createRange();
    r.setStart(a.node, a.off);
    r.setEnd(b.node, b.off);
    sel.removeAllRanges();
    sel.addRange(r);
  } catch (e) {
    /* the caret is a nicety; never let it break editing */
  } finally {
    editorState.restoring = false;
  }
}

// Rebuild the DOM into canonical form, keeping the caret where it was. Never
// called while an IME composition is running.
function normaliseEditor() {
  const root = scriptEditor();
  if (!root || editorState.composing) return;
  const runs = readEditor();
  if (root.innerHTML === canonicalHtml(runs)) {
    editorState.runs = runs;
    updateEditorMeta();
    return;
  }
  const sel = selectionOffsets();
  renderEditor(runs);
  if (sel) setSelectionOffsets(sel.start, sel.end);
}

// ------------------------------------------------- structural edit helpers --
function runsToChars(runs) {
  const chars = [];
  (runs || []).forEach((r) => {
    const t = r.type === "speech" ? "speech" : "prompt";
    for (let i = 0; i < r.text.length; i++) chars.push([r.text[i], t]);
  });
  return chars;
}

function charsToRuns(chars) {
  const out = [];
  chars.forEach(([ch, t]) => {
    if (out.length && out[out.length - 1].type === t) out[out.length - 1].text += ch;
    else out.push({ type: t, text: ch });
  });
  return out;
}

// Convert exactly [start,end) to `type`. Splits the runs it lands inside and
// merges neighbours of the same type, so the structure stays minimal.
function applyTypeToOffsets(start, end, type) {
  const chars = runsToChars(readEditor());
  const a = Math.max(0, Math.min(chars.length, start));
  const b = Math.max(0, Math.min(chars.length, end));
  for (let i = a; i < b; i++) chars[i][1] = type;
  renderEditor(charsToRuns(chars));
  setSelectionOffsets(a, b);
  afterEditorChange();
}

function insertTextAtOffsets(start, end, text, type) {
  const chars = runsToChars(readEditor());
  const a = Math.max(0, Math.min(chars.length, start));
  const b = Math.max(a, Math.min(chars.length, end));
  const added = String(text).replace(/\r\n|\r/g, "\n").split("")
    .map((ch) => [ch, type === "speech" ? "speech" : "prompt"]);
  chars.splice(a, b - a, ...added);
  renderEditor(charsToRuns(chars));
  setSelectionOffsets(a + added.length);
  afterEditorChange();
}

// A collapsed caret cannot carry a type, so a mode switch drops an anchor span
// of the new type at the caret. The ZWSP keeps the span alive until the first
// character is typed; every offset calculation ignores it.
function blockOf(node) {
  const root = scriptEditor();
  let n = node && node.nodeType === 1 ? node : (node && node.parentNode);
  while (n && n !== root && !isBlockEl(n)) n = n.parentNode;
  return n && n !== root ? n : null;
}

function insertAnchor(type) {
  const root = scriptEditor();
  const sel = window.getSelection();
  if (!root || !sel || !sel.rangeCount) return;
  const r = sel.getRangeAt(0);
  if (!root.contains(r.startContainer)) return;
  // An empty line holds a filler <br>. Dropping the anchor after it would turn
  // that filler into a real line break, so the filler goes first.
  const blk = blockOf(r.startContainer);
  if (blk && visLen(blk.textContent || "") === 0 && blk.querySelector("br")) {
    [...blk.querySelectorAll("br")].forEach((br) => br.remove());
    r.setStart(blk, 0);
    r.collapse(true);
  }
  r.deleteContents();
  const sp = document.createElement("span");
  sp.className = "run " + type;
  sp.dataset.type = type;
  sp.textContent = ZWSP;
  r.insertNode(sp);
  const nr = document.createRange();
  nr.setStart(sp.firstChild, 1);
  nr.collapse(true);
  editorState.restoring = true;
  sel.removeAllRanges();
  sel.addRange(nr);
  editorState.restoring = false;
}

function focusEditorEnd() {
  const root = scriptEditor();
  if (!root) return;
  root.focus();
  const last = root.lastElementChild;
  if (!last) return;
  const r = document.createRange();
  if (visLen(last.textContent || "") === 0) {
    // an empty last line: before its filler <br>, never after it
    r.setStart(last, 0);
    r.collapse(true);
  } else {
    r.selectNodeContents(last);
    r.collapse(false);
  }
  const sel = window.getSelection();
  editorState.restoring = true;
  sel.removeAllRanges();
  sel.addRange(r);
  editorState.restoring = false;
}

// ------------------------------------------------------------- mode + wire --
function setLineMode(mode) {
  const m = mode === "speech" ? "speech" : "prompt";
  editorState.mode = m;
  state.story.lineMode = m;
  const bp = $("btnLineModePrompt");
  const bs = $("btnLineModeSpeech");
  if (bp) bp.classList.toggle("on", m === "prompt");
  if (bs) bs.classList.toggle("on", m === "speech");
  const lab = $("lineModeLabel");
  if (lab) {
    lab.textContent = "現在：" + (m === "speech" ? "Speech" : "Prompt");
    lab.classList.toggle("speech", m === "speech");
  }
}

// (a) nothing selected -> the mode for what is typed next
// (b) something selected -> convert exactly that selection
function onModeButton(mode) {
  const m = mode === "speech" ? "speech" : "prompt";
  const root = scriptEditor();
  const sel = window.getSelection();
  const inEditor = !!(root && sel && sel.rangeCount
    && root.contains(sel.getRangeAt(0).commonAncestorContainer));
  setLineMode(m);
  if (inEditor && !sel.isCollapsed) {
    const off = selectionOffsets();
    if (off && off.end > off.start) { applyTypeToOffsets(off.start, off.end, m); return; }
  }
  if (!inEditor) focusEditorEnd();
  else root.focus();
  insertAnchor(m);
}

function afterEditorChange() {
  updateEditorMeta();
  saveScriptDraft();
}

let draftTimer = null;
function saveScriptDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    try {
      localStorage.setItem("h3.story.script", JSON.stringify(editorState.runs));
      localStorage.setItem("h3.story.name", $("storyName") ? $("storyName").value : "");
    } catch (e) { /* private mode / quota: the editor still works */ }
  }, 300);
}

function loadScriptDraft() {
  try {
    const raw = localStorage.getItem("h3.story.script");
    if (!raw) return false;
    const runs = JSON.parse(raw);
    if (!Array.isArray(runs) || !runs.length) return false;
    renderEditor(runs);
    const nm = localStorage.getItem("h3.story.name");
    if (nm && $("storyName") && !$("storyName").value) $("storyName").value = nm;
    return true;
  } catch (e) {
    return false;
  }
}

function wireEditor() {
  const root = scriptEditor();
  if (!root) return;

  // Japanese input is the primary case: NOTHING may touch the DOM between
  // compositionstart and compositionend, or the conversion commits early / the
  // caret jumps / characters double.
  root.addEventListener("compositionstart", () => { editorState.composing = true; });
  root.addEventListener("compositionend", () => {
    editorState.composing = false;
    // Let the browser finish inserting the committed text first.
    setTimeout(() => { normaliseEditor(); afterEditorChange(); }, 0);
  });

  root.addEventListener("input", () => {
    if (editorState.composing) return;
    normaliseEditor();
    afterEditorChange();
  });

  // Plain text only, in the current mode, line breaks kept. No foreign HTML.
  root.addEventListener("paste", (ev) => {
    const cd = ev.clipboardData || window.clipboardData;
    if (!cd) return;
    const text = cd.getData("text/plain") || "";
    ev.preventDefault();
    if (!text) return;
    const off = selectionOffsets() || { start: 0, end: 0 };
    insertTextAtOffsets(off.start, off.end, text, editorState.mode);
  });

  // The mode follows the caret, the same way a bold button follows the caret.
  document.addEventListener("selectionchange", () => {
    if (editorState.composing || editorState.restoring) return;
    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return;
    const r = sel.getRangeAt(0);
    if (!root.contains(r.startContainer)) return;
    const t = nodeType(r.startContainer);
    if (t && t !== editorState.mode) setLineMode(t);
  });

  // contextmenu is deliberately NOT bound: copy / paste / spellcheck must be
  // the browser's own.
  renderEditor(editorState.runs);
}

function runsToLines(runs) {
  const out = [];
  (runs || []).forEach((r) => {
    const type = r.type === "speech" ? "speech" : "prompt";
    String(r.text).split("\n").forEach((piece) => {
      const t = piece.trim();
      if (t) out.push({ type, text: t });
    });
  });
  return out;
}

function linesToRuns(lines) {
  const runs = [];
  (lines || []).forEach((l, i) => {
    const type = l.type === "speech" ? "speech" : "prompt";
    const text = (i ? "\n" : "") + (l.text || "");
    if (runs.length && runs[runs.length - 1].type === type) runs[runs.length - 1].text += text;
    else runs.push({ type, text });
  });
  return runs;
}

function currentLines() {
  const runs = readEditor();
  editorState.runs = runs;
  return runsToLines(runs);
}

// ------------------------------------------------------- style presets -----
function renderMotionChips() {
  const box = $("motionChips");
  if (!box) return;
  const presets = (state.cfg && (state.cfg.style_presets
    || (state.cfg.presets && state.cfg.presets.styles)
    || state.cfg.motion_presets
    || (state.cfg.presets && state.cfg.presets.motions))) || [];
  box.innerHTML = "";
  presets.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "choice" + (state.story.presets.includes(p.id) ? " on" : "");
    b.textContent = p.label;
    b.onclick = () => {
      const at = state.story.presets.indexOf(p.id);
      if (at >= 0) state.story.presets.splice(at, 1);
      else state.story.presets.push(p.id);
      b.classList.toggle("on", state.story.presets.includes(p.id));
    };
    box.appendChild(b);
  });
}

// ------------------------------------------------------------- segments ----
function actionPresets() {
  return state.actionPresets || [];
}

// サーバーが決めた文型を1か所で使う（アプリ側で作文しない）。
function actionSentence(name) {
  const tpl = state.actionTemplate || "この場面の流れの中で、自然に「{name}」の動作を加える。";
  return tpl.replace("{name}", String(name || "").trim());
}

function applyPresetsPayload(res) {
  if (!res) return;
  if (Array.isArray(res.presets)) state.actionPresets = res.presets;
  if (res.template) state.actionTemplate = res.template;
  if (res.max_name) state.actionMaxName = res.max_name;
  if (res.fallback && res.message) toast(res.message);
}

async function refreshActionPresets() {
  try {
    applyPresetsPayload(await api("/api/action-presets"));
  } catch (e) {
    // 一覧が取れなくても、既に画面にあるチップとプロンプト文章は無事。
    toast("動作プリセットを読み込めませんでした: " + e.message);
  }
  renderSegments();
}

// ------------------------------------------- 動作プリセットの追加 / 削除 ----
function actionPresetError(msg) {
  const el = $("actionPresetError");
  if (!el) return;
  el.textContent = msg || "";
  el.hidden = !msg;
}

function openActionPresetDialog() {
  const dlg = $("actionPresetDialog");
  if (!dlg) return;
  const input = $("actionPresetName");
  if (input) {
    input.value = "";
    input.maxLength = state.actionMaxName || 40;
  }
  actionPresetError("");
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "open");
  if (input) input.focus();
}

function closeActionPresetDialog() {
  const dlg = $("actionPresetDialog");
  if (!dlg) return;
  if (typeof dlg.close === "function" && dlg.open) dlg.close();
  else dlg.removeAttribute("open");
}

async function saveActionPreset() {
  const input = $("actionPresetName");
  const name = input ? input.value : "";
  try {
    const res = await postJson("/api/action-presets", { name });
    applyPresetsPayload(res);
    closeActionPresetDialog();
    renderSegments();
    toast(`動作「${(res.preset && res.preset.name) || name}」を追加しました。`);
  } catch (e) {
    // サーバーの日本語の断り文をモーダルの中に出す（閉じない）。
    actionPresetError(e.message);
  }
}

async function deleteActionPreset(p) {
  const label = p.label || p.name || "";
  const ok = confirm(`動作プリセット「${label}」を一覧から削除します。\n`
    + "すでにセグメントの指示に書き足した文章は消えません。よろしいですか？");
  if (!ok) return;
  try {
    applyPresetsPayload(await postJson("/api/action-presets/delete", { id: p.id }));
    renderSegments();
    toast(`動作「${label}」を削除しました。`);
  } catch (e) {
    toast("削除できませんでした: " + e.message);
  }
}

function wireActionPresetDialog() {
  const save = $("btnActionPresetSave");
  const cancel = $("btnActionPresetCancel");
  const input = $("actionPresetName");
  if (save) save.onclick = () => { saveActionPreset(); };
  if (cancel) cancel.onclick = () => closeActionPresetDialog();
  if (input) {
    input.oninput = () => actionPresetError("");
    input.onkeydown = (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); saveActionPreset(); }
    };
  }
}

// 各セグメントの seed は base_seed + index（サーバーと同じ規則）。
function segmentSeed(seg, index) {
  if (seg && seg.seed !== undefined && seg.seed !== null && seg.seed !== "") {
    return Number(seg.seed);
  }
  const base = state.story.baseSeed;
  if (base === null || base === undefined) return null;
  return (Number(base) + index) % 4294967296;
}

function segStatusJp(s) {
  return s === "running" ? "生成中" : s === "done" ? "完了" : s === "error" ? "エラー" : "待機";
}

// ------------------------------------------- 構成の変更（追加/削除/並べ替え）--
// 追加・削除・並べ替えはどれも「新しい並び順のセグメント一覧を送る」1つの操作。
// 送る前に必ず dry_run で影響を聞き、生成済みクリップが失われるときだけ確認する。
function structureLocked() {
  return state.busy || state.story.running;
}

function blankSegment() {
  // 新規セグメントは segment_id を持たない。サーバーが正式な形に組み立てる。
  return {
    index: 0, segment_id: null, prompt: "", speech: "", lines: [],
    motion_presets: [], action_presets: [], attribute_overrides: null,
    seed: null, status: "pending", clip: null, error: "",
  };
}

// サーバーに送る entry。既存セグメントは segment_id を必ず載せる。
function structureEntries(segments) {
  return (segments || []).map((s) => {
    const e = {
      prompt: (s.prompt || "").trim(),
      speech: (s.speech || "").trim(),
      motion_presets: s.motion_presets || [],
      action_presets: s.action_presets || [],
    };
    if (s.segment_id) e.segment_id = s.segment_id;
    if (s.attribute_overrides) e.attribute_overrides = s.attribute_overrides;
    return e;
  });
}

function showStructureNote(message, isWarn) {
  [$("storySegmentNote"), $("storyInvalidNote")].forEach((el) => {
    if (!el) return;
    el.textContent = message || "";
    el.className = isWarn ? "warn" : "hint";
    el.hidden = !message;
  });
}

function applyStructureResult(res, label) {
  const st = state.story;
  // 画面はサーバーが返した並びで作り直す（手元で作り直さない）。
  applySegments(res.segments || []);
  if (typeof res.cursor === "number") st.cursor = res.cursor;
  const count = res.invalidated_count || 0;
  if (count) {
    st.merged = false;
    st.clipsDone = Math.max(0, st.clipsDone - count);
    $("cardStoryFinal").hidden = true;
    showStructureNote(
      `${label}：セグメント ${(res.divergence || 0) + 1} 以降の生成済みクリップ ${count} 本を`
      + "無効にしました。完成動画は最新ではありません。もう一度生成してください。", true);
  } else {
    showStructureNote(`${label}：生成済みのクリップは影響を受けていません。`, false);
  }
  renderStoryPanel(null);
  refreshButtons();
  toast(res.message || label);
  // クリップ本数・結合状態など、応答に含まれない部分だけを取り直す。
  if (st.id) refreshStory(st.id).catch(() => {});
}

async function applyStructure(nextSegments, label) {
  if (structureLocked()) {
    toast("生成中はセグメントの構成を変更できません。先に停止してください。");
    return false;
  }
  if (!nextSegments.length) {
    toast("セグメントを空にはできません。少なくとも 1 つのセグメントが必要です。");
    return false;
  }
  // まだサーバーにプロジェクトが無い＝生成済みのものが何も無いので、
  // その場で並べ替えるだけ。確認は不要（失うものが無い）。
  if (!state.story.id) {
    state.story.segments = nextSegments.map((s, i) => Object.assign(s, { index: i }));
    renderSegments();
    showStructureNote(`${label}：まだ生成していないので、失われるクリップはありません。`, false);
    toast(label + "。");
    return true;
  }

  const payload = {
    story_id: state.story.id,
    segments: structureEntries(nextSegments),
  };
  try {
    const dry = await postJson("/api/story/structure",
                               Object.assign({ dry_run: true }, payload));
    if (dry.requires_confirmation) {
      const at = (dry.divergence || 0) + 1;
      const ok = confirm(
        `この変更を行うと、セグメント ${at} 以降の生成済みクリップ `
        + `${dry.invalidated_count} 本が無効になります。続行しますか？`);
      if (!ok) { toast("変更をやめました。何も変わっていません。"); return false; }
    }
    const res = await postJson("/api/story/structure",
                               Object.assign({ dry_run: false }, payload));
    applyStructureResult(res, label);
    return true;
  } catch (e) {
    // サーバーの日本語の断り文をそのまま見せる。
    toast(e.message);
    showStructureNote(e.message, true);
    return false;
  }
}

function segMove(index, delta) {
  const segs = state.story.segments.slice();
  const to = index + delta;
  if (to < 0 || to >= segs.length) return;
  const moved = segs.splice(index, 1)[0];
  segs.splice(to, 0, moved);
  applyStructure(segs, `セグメント ${index + 1} を${delta < 0 ? "上" : "下"}に移動しました`);
}

function segDelete(index) {
  const segs = state.story.segments.slice();
  if (segs.length <= 1) {
    toast("セグメントを空にはできません。少なくとも 1 つのセグメントが必要です。");
    return;
  }
  segs.splice(index, 1);
  applyStructure(segs, `セグメント ${index + 1} を削除しました`);
}

function segInsert(at) {
  const segs = state.story.segments.slice();
  const pos = Math.max(0, Math.min(segs.length, at));
  segs.splice(pos, 0, blankSegment());
  applyStructure(segs, pos >= state.story.segments.length
    ? "セグメントを最後に追加しました"
    : `${pos + 1} 番目にセグメントを追加しました`);
}

// ------------------------------------------------------------- segments ----
function renderSegments() {
  const box = $("segmentList");
  if (!box) return;
  const segs = state.story.segments;
  const total = segs.length;
  $("storySegmentSummary").textContent = total
    ? `全 ${total} セグメント（完成したクリップ: ${state.story.clipsDone} 本）`
    : "まだセグメントがありません。長文から自動で作るか、行を書いて「分割し直す」を押してください。";

  box.innerHTML = "";

  const locked = structureLocked();
  // カードとカードのあいだ（先頭の前を含む）に置く挿入ボタン。
  const insertRow = (at) => {
    const row = document.createElement("div");
    row.className = "seginsert";
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn ins";
    b.dataset.struct = "insert";
    b.textContent = "＋";
    b.title = `ここ（${at + 1} 番目）にセグメントを追加`;
    b.setAttribute("aria-label", `${at + 1} 番目にセグメントを追加`);
    b.disabled = locked;
    b.onclick = () => segInsert(at);
    row.appendChild(b);
    box.appendChild(row);
  };

  segs.forEach((seg, i) => {
    insertRow(i);
    const card = document.createElement("div");
    const cls = seg.status === "running" ? " run"
      : seg.status === "done" ? " done"
      : seg.status === "error" ? " err" : "";
    card.className = "segcard" + cls;

    const head = document.createElement("div");
    head.className = "seghead";
    const title = document.createElement("span");
    title.textContent = `セグメント ${i + 1} / 合計${total}`;
    const badge = document.createElement("span");
    badge.className = "badge " + (seg.status === "running" ? "run"
      : seg.status === "done" ? "done" : seg.status === "error" ? "err" : "");
    badge.textContent = segStatusJp(seg.status);

    // 並べ替えと削除。矢印だけでは意味が分からないので、title と aria-label を付ける。
    const tools = document.createElement("div");
    tools.className = "segtools";
    const mkTool = (label, aria, off, cls, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn segtool" + (cls ? " " + cls : "");
      b.textContent = label;
      b.title = aria;
      b.setAttribute("aria-label", aria);
      b.dataset.struct = "1";
      if (off) b.dataset.structOff = "1";
      b.disabled = locked || off;
      b.onclick = fn;
      tools.appendChild(b);
      return b;
    };
    mkTool("↑", `セグメント ${i + 1} を上に移動`, i === 0, "", () => segMove(i, -1));
    mkTool("↓", `セグメント ${i + 1} を下に移動`, i === total - 1, "", () => segMove(i, 1));
    mkTool("削除", `セグメント ${i + 1} を削除`, total <= 1, "danger", () => segDelete(i));

    head.append(title, badge, tools);

    const lp = document.createElement("label");
    lp.className = "lbl";
    lp.textContent = "動きや場面の指示（Prompt）";
    const tp = document.createElement("textarea");
    tp.rows = 2;
    tp.value = seg.prompt || "";
    tp.placeholder = "この場面で何をするか";
    tp.oninput = () => { seg.prompt = tp.value; };

    // 空のセリフ =「しゃべらない」。placeholder だけでは伝わらないので、
    // 明示のバッジを出す。
    const srow = document.createElement("div");
    srow.className = "seglbl";
    const ls = document.createElement("span");
    ls.className = "lbl";
    ls.textContent = "セリフ（Speech）";
    const silent = document.createElement("span");
    silent.className = "badge silent";
    silent.textContent = "発話なし";
    silent.title = "このセグメントではしゃべりません（AIが勝手にセリフを作ることはありません）。";
    srow.append(ls, silent);

    const ts = document.createElement("textarea");
    ts.rows = 2;
    ts.className = "speech";
    ts.value = seg.speech || "";
    ts.placeholder = "空欄なら、このセグメントではしゃべりません";
    const syncSilent = () => { silent.hidden = !!(seg.speech || "").trim(); };
    ts.oninput = () => { seg.speech = ts.value; syncSilent(); };
    syncSilent();

    card.append(head, lp, tp, srow, ts);

    // WP-C: seam transition INTO this segment (there is no boundary before
    // segment 1). Saved via /api/story/segments as segment.transition, and
    // may be changed at any time (even on a locked/done clip) since it only
    // affects how the FINAL video is assembled, never the clip itself.
    if (i > 0) {
      const tr = seg.transition || { mode: "natural", keep_pause: false };
      const trow = document.createElement("div");
      trow.className = "seamrow";
      const tl = document.createElement("label");
      tl.className = "lbl";
      tl.textContent = "つなぎ方";
      const sel = document.createElement("select");
      sel.className = "seammode";
      [["natural", "自然につなぐ"], ["cut", "カット"], ["fade", "フェード"]]
        .forEach(([value, label]) => {
          const opt = document.createElement("option");
          opt.value = value;
          opt.textContent = label;
          if ((tr.mode || "natural") === value) opt.selected = true;
          sel.appendChild(opt);
        });
      sel.onchange = () => {
        seg.transition = { ...(seg.transition || {}), mode: sel.value };
      };
      const pauseLabel = document.createElement("label");
      pauseLabel.className = "seampause";
      const pauseBox = document.createElement("input");
      pauseBox.type = "checkbox";
      pauseBox.checked = !!tr.keep_pause;
      pauseBox.onchange = () => {
        seg.transition = { ...(seg.transition || {}), keep_pause: pauseBox.checked };
      };
      pauseLabel.append(pauseBox, document.createTextNode(" 間を残す"));
      pauseLabel.title = "台詞前の間や無音を、つなぎ目補正で削らずそのまま残します。";
      trow.append(tl, sel, pauseLabel);
      card.appendChild(trow);
    }

    // v2 clip control: lock / approve / regenerate one clip.
    // textContent only, never innerHTML (names are user data).
    const ctl = document.createElement("div");
    ctl.className = "choices segclips";
    const lockState = seg.locked || "";
    const mkCtl = (label, title, fn, disabled) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice";
      b.textContent = label;
      b.title = title;
      b.disabled = !!disabled;
      b.onclick = fn;
      ctl.appendChild(b);
    };
    if (seg.status === "done") {
      const tag = document.createElement("span");
      tag.className = "badge" + (lockState ? " done" : "");
      tag.textContent = lockState === "approved" ? "承認済み"
        : lockState === "locked" ? "ロック中" : "生成済み";
      ctl.appendChild(tag);
      if (!lockState) {
        mkCtl("ロック", "このクリップを固定。以後の再生成で作り直しません。", () => clipLock(i, "lock"), locked);
        mkCtl("承認", "このクリップを承認済みにします。", () => clipLock(i, "approve"), locked);
      } else {
        mkCtl("解除", "ロック/承認を外します。", () => clipLock(i, "unlock"), locked);
      }
      mkCtl("このクリップだけ再生成", "このクリップ以降を作り直します（ロック済みは保持）。", () => regenClip(i), locked);
    }
    card.appendChild(ctl);

    // このセグメントだけの動作（ACTION）。押した瞬間に上のプロンプトへ
    // 文章を書き足す「命令ボタン」で、選択状態は持たない。
    const la = document.createElement("span");
    la.className = "lbl";
    la.textContent = "このセグメントだけの動作";
    const chips = document.createElement("div");
    chips.className = "choices segactions";
    actionPresets().forEach((p) => {
      const wrap = document.createElement("span");
      wrap.className = "actionchip";
      const b = document.createElement("button");
      b.type = "button";
      b.className = "choice";
      // 名前は「データ」。innerHTML では絶対に入れない（<b>x</b> は文字のまま）。
      b.textContent = p.label || p.name || "";
      b.title = actionSentence(p.label || p.name || "") + "\nを、上の指示に書き足します。";
      b.onclick = () => {
        const body = (tp.value || "").replace(/\s+$/, "");
        const sentence = actionSentence(p.label || p.name || "");
        tp.value = body ? body + "\n" + sentence : sentence;
        seg.prompt = tp.value;
        // 何が起きたか見えるように、書き足した行までスクロールして知らせる。
        tp.focus();
        tp.setSelectionRange(tp.value.length, tp.value.length);
        tp.scrollTop = tp.scrollHeight;
        toast("「" + (p.label || p.name || "") + "」を指示に書き足しました。文章は自由に直せます。");
      };
      wrap.appendChild(b);
      if (state.actionManage) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "choice chipdel";
        del.textContent = "×";
        del.title = `「${p.label || p.name}」を一覧から削除`;
        del.setAttribute("aria-label", `動作プリセット ${p.label || p.name} を削除`);
        del.onclick = () => deleteActionPreset(p);
        wrap.appendChild(del);
      }
      chips.appendChild(wrap);
    });
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "btn addaction";
    addBtn.textContent = "＋ 動作を追加";
    addBtn.onclick = () => openActionPresetDialog();
    const manageBtn = document.createElement("button");
    manageBtn.type = "button";
    manageBtn.className = "btn manageaction" + (state.actionManage ? " on" : "");
    manageBtn.textContent = state.actionManage ? "管理を終了" : "プリセット管理";
    manageBtn.onclick = () => { state.actionManage = !state.actionManage; renderSegments(); };
    chips.append(addBtn, manageBtn);
    const ah = document.createElement("p");
    ah.className = "hint";
    ah.textContent = state.actionManage
      ? "「×」を押すと、その動作プリセットを一覧から削除します（確認が出ます）。"
        + "すでに指示に書き足した文章は消えません。"
      : "動作を押すと、上の「動きや場面の指示」に文章が書き足されます。"
        + "書き足したあとは自由に直せます。";
    card.append(la, chips, ah);

    const seedLine = document.createElement("p");
    seedLine.className = "seedline";
    const seed = segmentSeed(seg, i);
    seedLine.textContent = "seed: " + (seed === null ? "開始時に決まります" : String(seed));
    card.appendChild(seedLine);

    if (seg.error) {
      const e = document.createElement("p");
      e.className = "segerr";
      e.textContent = seg.error;
      card.appendChild(e);
    }

    if (seg.status === "done" && state.story.id) {
      const foot = document.createElement("div");
      foot.className = "segfoot";
      const img = document.createElement("img");
      img.src = `/api/story/${state.story.id}/frame/${i}?t=${Date.now()}`;
      img.alt = `セグメント ${i + 1} の最後の場面`;
      img.onerror = () => { img.hidden = true; };
      const vid = document.createElement("video");
      vid.controls = true;
      vid.preload = "none";
      vid.playsInline = true;
      vid.hidden = true;
      const play = document.createElement("button");
      play.type = "button";
      play.className = "btn";
      play.textContent = "再生";
      play.onclick = () => {
        if (vid.hidden) {
          vid.hidden = false;
          vid.src = `/api/story/${state.story.id}/video/${i}?t=${Date.now()}`;
          vid.load();
        }
        if (vid.paused) vid.play(); else vid.pause();
      };
      foot.append(img, play);
      card.append(foot, vid);
    }

    box.appendChild(card);
  });
  refreshButtons();
}

function segmentsPayload() {
  // `lines` is left out on purpose: the backend rebuilds it from the edited
  // prompt / speech text, so an edit in the textarea can never disagree with it.
  return state.story.segments.map((s) => ({
    // 既存セグメントの身元。これを落とすと、サーバーは「文章を直しただけ」の
    // セグメントを新しいセグメントと見なし、生成済みクリップを無駄に捨てる。
    segment_id: s.segment_id || undefined,
    prompt: (s.prompt || "").trim(),
    speech: (s.speech || "").trim(),
    motion_presets: s.motion_presets || [],
    // 1回だけの動作。そのセグメントにだけ保存される。
    action_presets: s.action_presets || [],
    status: s.status || "pending",
    clip: s.clip || null,
    error: s.error || "",
  })).filter((s) => s.prompt || s.speech
    || (s.motion_presets || []).length || (s.action_presets || []).length);
}

function applySegments(segments) {
  state.story.segments = (segments || []).map((s, i) => ({
    index: i,
    // サーバーが付けた身元。並べ替え・追加・削除・保存のすべてで持ち回る。
    segment_id: s.segment_id || null,
    prompt: s.prompt || "",
    speech: s.speech || "",
    lines: s.lines || [],
    motion_presets: s.motion_presets || [],
    action_presets: s.action_presets || [],
    attribute_overrides: s.attribute_overrides || null,
    seed: s.seed !== undefined ? s.seed : null,
    status: s.status || "pending",
    clip: s.clip || null,
    error: s.error || "",
  }));
  renderSegments();
}

function applyLines(lines) {
  state.story.lines = (lines || [])
    .map((l) => ({ type: l.type === "speech" ? "speech" : "prompt", text: l.text || "" }))
    .filter((l) => l.text);
  renderEditor(linesToRuns(state.story.lines));
  saveScriptDraft();
}

// ------------------------------------------------------------ story API ----
async function storyPreviewFromText() {
  const text = $("storyLongText").value.trim();
  if (!text) { toast("長文を入力してください。"); return; }
  try {
    const res = await postJson("/api/story/preview", { text, seconds: state.seconds });
    applyLines(res.lines);
    applySegments(res.segments);
    if (res.over_limit) {
      toast(`セグメントが多すぎます（最大 ${res.max_segments} 個）。文章を短くしてください。`);
    } else {
      toast(`${res.segments.length} 個のセグメントに分けました。`);
    }
  } catch (e) {
    toast("分割できませんでした: " + e.message);
  }
}

async function storyResplit() {
  const lines = currentLines();
  if (!lines.length) { toast("台本が空です。先に台本を書いてください。"); return; }
  try {
    // 構造つきで送るので、サーバー側では種類の推測は一切行われない。
    const res = await postJson("/api/story/preview", { lines, seconds: state.seconds });
    applySegments(res.segments);
    if (res.over_limit) {
      toast(`セグメントが多すぎます（最大 ${res.max_segments} 個）。文章を短くしてください。`);
    } else {
      toast(`${res.segments.length} 個のセグメントに分け直しました。`);
    }
  } catch (e) {
    toast("分割できませんでした: " + e.message);
  }
}

function segmentIsEmpty(s) {
  return !(s.prompt || "").trim() && !(s.speech || "").trim()
    && !(s.motion_presets || []).length && !(s.action_presets || []).length;
}

async function storySaveSegments(quiet) {
  if (!state.story.id) {
    if (!quiet) toast("先に「ストーリー開始」でプロジェクトを作ってください。");
    return false;
  }
  const segs = state.story.segments;
  // 追加したばかりの空のセグメントは、内容だけの保存では捨てられてしまう。
  // 並び順は変わらないので、構成の保存に回しても失われるクリップは出ない。
  if (segs.length && segs.some(segmentIsEmpty) && segs.every((s) => s.segment_id)) {
    try {
      const res = await postJson("/api/story/structure", {
        story_id: state.story.id,
        segments: structureEntries(segs),
        dry_run: false,
      });
      applySegments(res.segments || []);
      if (typeof res.cursor === "number") state.story.cursor = res.cursor;
      if (!quiet) toast("セグメントを保存しました。");
      return true;
    } catch (e) {
      if (!quiet) toast("保存できませんでした: " + e.message);
      return false;
    }
  }
  try {
    const res = await postJson("/api/story/segments", {
      story_id: state.story.id,
      segments: segmentsPayload(),
    });
    applyStory(res.story);
    if (!quiet) toast("セグメントを保存しました。");
    return true;
  } catch (e) {
    if (!quiet) toast("保存できませんでした: " + e.message);
    return false;
  }
}

// ------------------------------------------------------------- pre-flight --
function clearPreflight() {
  const panel = $("preflightPanel");
  if (!panel) return;
  panel.hidden = true;
  $("preflightMessage").textContent = "";
  $("preflightList").innerHTML = "";
}

function renderPreflight(records, message, isError) {
  const panel = $("preflightPanel");
  if (!panel) return;
  panel.hidden = false;
  const msg = $("preflightMessage");
  msg.textContent = message || "";
  msg.className = "preflightmsg" + (isError ? " ng" : "");
  msg.hidden = !message;

  const list = $("preflightList");
  list.innerHTML = "";
  (records || []).forEach((r) => {
    const row = document.createElement("div");
    row.className = "pfrow";

    const head = document.createElement("div");
    head.className = "pfhead";
    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = `セグメント ${(r.index || 0) + 1}`;
    head.appendChild(nm);
    if (r.silent) {
      const b = document.createElement("span");
      b.className = "badge silent";
      b.textContent = "発話なし";
      head.appendChild(b);
    }
    const seed = document.createElement("span");
    seed.className = "k";
    seed.textContent = `seed: ${r.seed}`;
    const ctx = document.createElement("span");
    ctx.className = "k";
    ctx.textContent = "前クリップ参照: " + (r.previous_context ? "あり" : "なし");
    head.append(seed, ctx);
    row.appendChild(head);

    const p = document.createElement("p");
    p.className = "pfline";
    p.textContent = "Prompt: " + (r.prompt || "（なし）");
    row.appendChild(p);

    const s = document.createElement("p");
    s.className = "pfline speech";
    s.textContent = "Speech: " + (r.silent ? "発話なし（しゃべりません）" : r.speech);
    row.appendChild(s);

    if ((r.style_presets || []).length || (r.action_presets || []).length) {
      const pr = document.createElement("p");
      pr.className = "pfline";
      pr.textContent = "雰囲気: " + ((r.style_presets || []).join("、") || "なし")
        + " ／ 動作: " + ((r.action_presets || []).join("、") || "なし");
      row.appendChild(pr);
    }

    (r.problems || []).forEach((t) => {
      const el = document.createElement("p");
      el.className = "pfprob";
      el.textContent = "要修正: " + t;
      row.appendChild(el);
    });
    (r.warnings || []).forEach((t) => {
      const el = document.createElement("p");
      el.className = "pfwarn";
      el.textContent = "注意: " + t;
      row.appendChild(el);
    });

    list.appendChild(row);
  });
}

async function storyPreflight() {
  if (!state.story.segments.length) {
    toast("セグメントがありません。先に「分割し直す」を押してください。");
    return;
  }
  const btn = $("btnStoryPreflight");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "確認中...";
  try {
    if (!state.story.id) await storyCreate();
    else await storySaveSegments(true);
    const res = await api(`/api/story/${state.story.id}/preflight`);
    const bad = (res.preflight || []).some((r) => (r.problems || []).length);
    renderPreflight(res.preflight || [],
      bad ? "このままでは開始できません。下の「要修正」を直してください。"
          : "この内容で生成します。", bad);
  } catch (e) {
    if (e.body && e.body.preflight) renderPreflight(e.body.preflight, e.message, true);
    else { clearPreflight(); toast("確認できませんでした: " + e.message); }
  } finally {
    btn.textContent = label;
    refreshButtons();
  }
}

function applyStory(story, opts) {
  if (!story) return;
  const st = state.story;
  const replaceScript = !!(opts && opts.replaceScript);
  st.id = story.id || st.id;
  st.name = story.name || "";
  st.status = story.status || "pending";
  st.cursor = typeof story.cursor === "number" ? story.cursor : 0;
  st.clipsDone = (story.clips || []).length;
  st.merged = !!story.merged;
  st.finalPath = story.final_video || st.finalPath || "";
  st.seams = story.seams || st.seams || null;
  st.characterSnapshot = story.character_snapshot || st.characterSnapshot || null;
  st.characterMissing = !!story.character_missing;
  st.presets = story.motion_presets || st.presets;
  if (story.base_seed !== undefined && story.base_seed !== null) {
    st.baseSeed = Number(story.base_seed);
  }
  if (story.name) $("storyName").value = story.name;
  if (story.jp_negative && !$("storyNegative").value) {
    $("storyNegative").value = story.jp_negative;
  }
  if (story.source_text && !$("storyLongText").value) {
    $("storyLongText").value = story.source_text;
  }
  // 保存済みの lines で台本を上書きするのは「プロジェクトを開いた」ときだけ。
  // 保存のたびに書き戻すと、保存後に書き足した内容が消えてしまう。
  if (story.lines && story.lines.length
      && (replaceScript || !readEditor().length)) {
    applyLines(story.lines);
  }
  applySegments(story.segments || []);
  renderMotionChips();
  renderStoryPanel(null);
  if (st.merged) showFinalVideo();
  refreshButtons();
}

// Story-only mode ids (see h3app.modes_v2.STORY_MODES). Single-shot modes
// (FAST/QUALITY/LEGACY/BALANCED/LOW_VRAM) must never be sent to story/create;
// the server keeps its LONG default when this key is omitted.
const STORY_MODE_IDS = ["LONG", "LONG_FAST"];

async function storyCreate() {
  const body = {
    name: $("storyName").value.trim(),
    images: state.images.map((i) => i.name),
    source_text: $("storyLongText").value,
    lines: currentLines(),
    segments: segmentsPayload(),
    jp_negative: $("storyNegative").value,
    motion_presets: state.story.presets,
    seconds: state.seconds,
    aspect: state.aspect,
    advanced: advancedPayload(),
    character_snapshot: state.characterSnapshot,
  };
  if (STORY_MODE_IDS.includes(state.genMode)) {
    body.mode = state.genMode;
  }
  const res = await postJson("/api/story/create", body);
  state.story.id = res.story_id;
  applyStory(res.story);
  return res.story_id;
}

async function storyStart() {
  if (state.busy) { toast("ほかの生成が動いています。終わってから開始してください。"); return; }
  if (!state.images.length) { toast("参照画像を1枚以上追加してください。"); return; }
  if (!state.story.segments.length) { toast("セグメントがありません。先に分割してください。"); return; }

  $("cardStatus").hidden = false;
  showError(null);
  renderStages(null);
  $("progressBar").style.width = "0%";
  setBusy(true);
  try {
    if (!state.story.id) {
      await storyCreate();
    } else {
      await storySaveSegments(true);
    }
    const started = await postJson("/api/story/start", { story_id: state.story.id });
    if (started.preflight && started.preflight.length) {
      renderPreflight(started.preflight, "この内容で生成しています。", false);
    }
    state.story.running = true;
    state.story.status = "running";
    listenStory(state.story.id);
    refreshButtons();
    renderStoryPanel(null);
    $("cardStoryRun").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    state.story.running = false;
    setBusy(false);
    // 断られたときは、サーバーの日本語メッセージと問題のあるセグメントを出す。
    if (e.body && e.body.preflight) {
      renderPreflight(e.body.preflight, e.message, true);
      $("cardStoryRun").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    toast(e.message);
    showError({
      title: "ストーリーをはじめられませんでした",
      where: "止まった場所: 開始前",
      retry: "再試行: はい",
      config: "設定の問題: " + e.message,
      detail: e.message,
    });
  }
}

async function storyStop() {
  if (!state.story.id) return;
  try {
    const res = await postJson("/api/story/stop", { story_id: state.story.id });
    toast(res.message || "停止を要求しました。");
  } catch (e) {
    toast("停止できませんでした: " + e.message);
  }
}

async function storyDiscard(storyId, name) {
  const sid = storyId || state.story.id;
  if (!sid) return;
  const label = name || state.story.name || "このストーリー";
  if (!confirm(`「${label}」を破棄します。生成済みのクリップもすべて削除され、元に戻せません。よろしいですか？`)) return;
  try {
    await postJson("/api/story/discard", { story_id: sid });
    toast("破棄しました。");
    if (sid === state.story.id) resetStoryState();
    await refreshStoryList();
  } catch (e) {
    toast("破棄できませんでした: " + e.message);
  }
}

function resetStoryState() {
  const st = state.story;
  if (st.events) { st.events.close(); st.events = null; }
  st.id = null;
  st.status = "pending";
  st.cursor = 0;
  st.clipsDone = 0;
  st.merged = false;
  st.running = false;
  st.segments.forEach((s) => { s.status = "pending"; s.clip = null; s.error = ""; });
  $("cardStoryFinal").hidden = true;
  renderSegments();
  renderStoryPanel(null);
  refreshButtons();
}

async function storyMerge() {
  if (!state.story.id) return;
  const btn = $("btnStoryMerge");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "書き出し中...";
  try {
    const res = await postJson("/api/story/merge", { story_id: state.story.id });
    state.story.merged = true;
    state.story.finalPath = res.final_video || "";
    state.story.seams = res.seams || null;
    showFinalVideo();
    if (res.seams && res.seams.needs_review) {
      toast("⚠ つなぎ目に大きな差があります。完成動画を確認し、必要なら該当クリップを再生成してください。");
    }
    const exp = res.export;
    if (exp && exp.ok === false) {
      toast(`⚠ 生成動画フォルダへの保存に失敗しました（${exp.error || "不明なエラー"}）。動画はプロジェクト内に残っています。`);
    } else if (exp && exp.ok) {
      toast(`完成動画を書き出しました。生成動画フォルダに保存: ${exp.name}`);
    } else {
      toast("完成動画を書き出しました。");
    }
  } catch (e) {
    toast("書き出せませんでした: " + e.message);
  } finally {
    btn.textContent = label;
    refreshButtons();
  }
}

function showFinalVideo() {
  if (!state.story.id) return;
  const card = $("cardStoryFinal");
  card.hidden = state.mode !== "story";
  const v = $("storyPlayer");
  v.src = `/api/story/${state.story.id}/final?t=${Date.now()}`;
  v.load();
  $("storyFinalPath").textContent = state.story.finalPath
    ? `保存先: ${state.story.finalPath}` : "";
  renderSeamSummary(state.story.seams);
  if (state.mode === "story") card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// WP-C: one line per boundary - what the merge actually did at each seam
// (before/after luma difference, dropped head frames, colour ramp, audio
// trim) plus the review flag, so a merge never ends in a bare success toast.
const SEAM_MODE_JP = { natural: "自然", cut: "カット", fade: "フェード" };
function renderSeamSummary(seams) {
  const box = $("storySeams");
  if (!box) return;
  const report = (seams && Array.isArray(seams.report)) ? seams.report : [];
  if (!report.length) { box.hidden = true; box.textContent = ""; return; }
  box.textContent = "";
  const fmt = (v) => (typeof v === "number" ? v.toFixed(1) : "-");
  report.forEach((r, i) => {
    const line = document.createElement("div");
    const mode = SEAM_MODE_JP[r.mode] || r.mode || "-";
    const parts = [`つなぎ目 ${i + 1}（${mode}）`];
    if (r.mode === "natural") {
      parts.push(`輝度差 ${fmt(r.luma_diff_before)} → ${fmt(r.luma_diff_after)}`);
      const removed = (r.static_removed || 0) + (r.cut_offset || 0);
      parts.push(removed ? `先頭 ${removed}フレーム除去（静止 ${r.static_removed || 0}）` : "先頭の除去なし");
      const cc = r.color_correction || {};
      parts.push(cc.apply ? `色補正 ${cc.ramp_frames}フレームで解除` : "色補正なし");
      parts.push(`音声 ${r.audio_trim_ms || 0}ms 調整`);
      if (r.trim_reason === "scripted_pause") parts.push("間を保持");
    }
    if (r.needs_review) parts.push("⚠ 要確認");
    line.textContent = parts.join(" / ");
    if (r.needs_review) line.classList.add("warn");
    box.appendChild(line);
  });
  box.hidden = false;
}

// ---------------------------------------------------------- story status ---
function renderStoryPanel(data) {
  const st = state.story;
  const total = (data && data.segment_total) || st.segments.length;
  const done = data && typeof data.clips_done === "number" ? data.clips_done : st.clipsDone;
  const statusKey = (data && data.story_status) || st.status || "pending";

  $("storyPanelName").textContent = ($("storyName").value.trim() || st.name || "（未設定）");
  $("storyPanelStatus").textContent = STORY_STATUS_JP[statusKey] || "待機";
  $("storyPanelProgress").textContent = `${done} / ${total}`;
  $("storyProgressBar").style.width = total ? `${Math.round((done / total) * 100)}%` : "0%";

  const idx = data && typeof data.segment_index === "number" ? data.segment_index : null;
  if (idx !== null && st.running) {
    $("storyPanelCurrent").textContent = `セグメント ${idx + 1} / 合計${total}`;
    $("storyPanelPrompt").textContent = (data && data.segment_prompt) || "";
    $("storyPanelSpeech").textContent = (data && data.segment_speech) || "";
  } else {
    $("storyPanelCurrent").textContent = st.running ? "準備中" : "-";
    $("storyPanelPrompt").textContent = "";
    $("storyPanelSpeech").textContent = "";
  }
}

function listenStory(storyId) {
  const st = state.story;
  if (st.events) st.events.close();
  const es = new EventSource(`/api/story/${storyId}/events`);
  st.events = es;
  es.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch (e) { return; }
    renderStages(data);
    renderStoryPanel(data);
    showError(data.error);

    if (typeof data.segment_index === "number" && data.story_status === "running") {
      st.segments.forEach((s, i) => {
        if (i < data.segment_index && s.status !== "error") s.status = "done";
        else if (i === data.segment_index && s.status !== "done") s.status = "running";
      });
      st.clipsDone = typeof data.clips_done === "number" ? data.clips_done : st.clipsDone;
      renderSegments();
    }

    if (data.finished) {
      es.close();
      st.events = null;
      st.running = false;
      setBusy(false);
      refreshStory(storyId).catch(() => {});
      refreshStoryList().catch(() => {});
    }
  };
  es.addEventListener("end", () => {
    if (st.events === es) st.events = null;
    es.close();
    if (st.running) {
      st.running = false;
      setBusy(false);
      refreshStory(storyId).catch(() => {});
    }
  });
  es.onerror = () => { /* the reload below still refreshes the final state */ };
}

async function refreshStory(storyId, opts) {
  const res = await api(`/api/story/${storyId}`);
  const story = res.story;
  state.story.running = !!story.running;
  applyStory(story, opts);
  if (story.error) showError(story.error);
  if (state.story.running) {
    setBusy(true);
    listenStory(storyId);
  }
  renderStoryPanel(story.progress || null);
  refreshButtons();
}

async function storyOpen(storyId) {
  try {
    if (state.story.events) { state.story.events.close(); state.story.events = null; }
    state.story.id = storyId;
    state.story.finalPath = "";
    await refreshStory(storyId, { replaceScript: true });
    toast("プロジェクトを開きました。");
    $("cardStoryScript").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    toast("開けませんでした: " + e.message);
  }
}

// ------------------------------------------------------------ story list ---
async function refreshStoryList() {
  const box = $("storyList");
  if (!box) return;
  let stories = [];
  try {
    const res = await api("/api/story/list");
    stories = res.stories || [];
  } catch (e) {
    box.innerHTML = "";
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "一覧を取得できませんでした: " + e.message;
    box.appendChild(p);
    return;
  }
  state.story.summaries = stories;
  box.innerHTML = "";
  if (!stories.length) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "保存済みのプロジェクトはまだありません。";
    box.appendChild(p);
    return;
  }
  stories.forEach((s) => {
    const row = document.createElement("div");
    row.className = "storyitem";

    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = s.name || "（名前なし）";

    const meta = document.createElement("span");
    meta.className = "meta2";
    meta.textContent = `状態: ${STORY_STATUS_JP[s.status] || s.status}　`
      + `進捗: ${s.cursor} / ${s.segment_total}　`
      + `更新: ${fmtDate(s.updated_at)}　`
      + `結合済み: ${s.merged ? "はい" : "いいえ"}`;

    const sp = document.createElement("span");
    sp.className = "sp";

    const open = document.createElement("button");
    open.type = "button";
    open.className = "btn";
    open.textContent = "開く";
    open.onclick = () => storyOpen(s.id);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "btn danger";
    del.textContent = "破棄";
    del.onclick = () => storyDiscard(s.id, s.name);

    row.append(nm, meta, sp, open, del);
    box.appendChild(row);
  });
}

// ---------------------------------------------- heartbeat / tab closing ----
function startHeartbeat() {
  const beat = () => {
    postJson("/api/heartbeat", {}).catch(() => {});
  };
  beat();
  setInterval(beat, 10000);
}

function detachBeacon() {
  try {
    if (navigator.sendBeacon) {
      if (state.story.running && state.story.id) {
        navigator.sendBeacon("/api/story/stop", new Blob(
          [JSON.stringify({ story_id: state.story.id })], { type: "application/json" }));
      }
      navigator.sendBeacon("/api/detach", new Blob(["{}"], { type: "application/json" }));
    }
  } catch (e) { /* the tab is going away anyway */ }
}

// ------------------------------------------------------------- 音声テスト ---
function audioTestVoiceMasterAvailable() {
  const snap = state.characterSnapshot;
  return !!(snap && snap.voice_file);
}

function audioTestBlockReason() {
  if (state.busy) return "生成中です。終わるまでお待ちください。";
  if (state.images.length === 0) return "参照画像を1枚以上追加してください。";
  return "";
}

function refreshAudioTestButtons() {
  const reason = audioTestBlockReason();
  const blocked = !!reason;
  const q = $("btnAudioTestQuick");
  const f = $("btnAudioTestFull");
  if (q) q.disabled = blocked;
  if (f) f.disabled = blocked;
  const why = $("audioTestWhy");
  if (why) { why.textContent = reason; why.hidden = !reason; }
  const vm = $("audioTestVoiceMaster");
  const vmHint = $("audioTestVoiceMasterHint");
  if (vm) {
    const available = audioTestVoiceMasterAvailable();
    vm.disabled = !available;
    if (!available) vm.checked = false;
    if (vmHint) vmHint.hidden = available;
  }
  const delAll = $("btnAudioTestDeleteAll");
  if (delAll) delAll.disabled = state.audioTest.list.length === 0;
}

function fmtAudioTestRow(t) {
  const row = document.createElement("div");
  row.className = "audiotestrow";

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = `/api/audio-test/audio/${encodeURIComponent(t.test_id)}`;
  row.appendChild(audio);

  const meta = document.createElement("div");
  meta.className = "audiotestmeta";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = String(t.profile || "").toUpperCase();
  meta.appendChild(badge);
  const when = document.createElement("span");
  when.textContent = fmtDate(t.created_at);
  meta.appendChild(when);
  row.appendChild(meta);

  const text = document.createElement("p");
  text.className = "audiotesttext";
  text.textContent = `台詞: ${t.text || ""}`;
  row.appendChild(text);
  if (t.spoken_text && t.spoken_text !== t.text) {
    const spoken = document.createElement("p");
    spoken.className = "audiotesttext";
    spoken.textContent = `発音用テキスト: ${t.spoken_text}`;
    row.appendChild(spoken);
  }

  const info = document.createElement("p");
  info.className = "hint";
  info.textContent = `Voice Master: ${t.voice_master_enabled ? "ON" : "OFF"} ／ 所要時間 ${t.generation_seconds}秒`;
  row.appendChild(info);

  const actions = document.createElement("div");
  actions.className = "actions";
  const del = document.createElement("button");
  del.type = "button";
  del.className = "btn danger";
  del.textContent = "削除";
  del.onclick = () => deleteAudioTest(t.test_id, t.text || "");
  actions.appendChild(del);
  row.appendChild(actions);

  return row;
}

function renderAudioTestList() {
  const box = $("audioTestList");
  if (!box) return;
  box.innerHTML = "";
  const list = state.audioTest.list;
  const empty = $("audioTestListEmpty");
  if (empty) empty.hidden = list.length > 0;
  list.forEach((t) => box.appendChild(fmtAudioTestRow(t)));
  refreshAudioTestButtons();
}

async function refreshAudioTestList() {
  const res = await api("/api/audio-test/list");
  state.audioTest.list = res.tests || [];
  renderAudioTestList();
}

async function deleteAudioTest(testId, label) {
  if (!confirm(`「${label}」を削除します。よろしいですか？`)) return;
  try {
    await postJson("/api/audio-test/delete", { test_id: testId });
    await refreshAudioTestList();
    toast("削除しました。");
  } catch (e) {
    toast(e.message);
  }
}

async function deleteAllAudioTests() {
  const n = state.audioTest.list.length;
  if (n === 0) return;
  if (!confirm(`音声テスト ${n}件を削除します。本番動画・Character・Projectは削除されません。`)) return;
  try {
    const res = await postJson("/api/audio-test/delete-all", {});
    await refreshAudioTestList();
    toast(`${res.count != null ? res.count : n}件を削除しました。`);
  } catch (e) {
    toast(e.message);
  }
}

async function runAudioTest(profile) {
  const reason = audioTestBlockReason();
  if (reason) { toast(reason); return; }
  const status = $("audioTestStatus");
  const text = $("audioTestText").value;
  const spoken = $("audioTestSpoken").value;
  const language = state.audioTest.lang || "Japanese";
  const voiceMaster = $("audioTestVoiceMaster").checked && audioTestVoiceMasterAvailable();
  const notes = $("audioTestNotes").value;

  setBusy(true);
  if (status) status.textContent = "生成中…（最大2分ほどかかります）";
  try {
    await postJson("/api/audio-test/run", {
      text,
      spoken_text: spoken,
      language,
      profile,
      voice_master: voiceMaster,
      images: state.images.map((i) => i.name),
      character_snapshot: state.characterSnapshot,
      notes,
      want_mp3: false,
    });
    if (status) status.textContent = "";
    toast("音声テストが完了しました。");
    await refreshAudioTestList();
  } catch (e) {
    if (status) status.textContent = e.message;
    toast(e.message);
  } finally {
    setBusy(false);
  }
}

// ======================================================= WP-B: 高画質化 UI ==
// Inline panel (not a modal) attached under the result card's actions row.
// Talks to /api/upscale/* and follows /api/events/{job_id} exactly like a
// normal generation (same SSE shape, state.job_type === "upscale").
const upscalePanels = {};

function upscaleFmtSize(bytes) {
  const n = Number(bytes) || 0;
  if (n <= 0) return "";
  const gb = n / (1024 ** 3);
  if (gb >= 1) return `約${gb.toFixed(1)}GB`;
  return `約${Math.max(1, Math.round(n / (1024 ** 2)))}MB`;
}

function upscaleFmtSource(source) {
  if (!source || !source.width || !source.height) return "元動画: -";
  const bits = [`${source.width}×${source.height}`];
  if (source.fps) bits.push(`${Math.round(source.fps * 10) / 10}fps`);
  if (source.duration) bits.push(`${Math.round(source.duration * 10) / 10}秒`);
  bits.push(source.has_audio ? "音声あり" : "音声なし");
  return `元動画: ${bits.join(" / ")}`;
}

function createUpscalePanel({ btnId, panelId, getSource }) {
  const btn = $(btnId);
  const panel = $(panelId);
  if (!btn || !panel) return null;

  const st = { jobId: null, events: null };

  panel.innerHTML = `
    <h3>高画質化</h3>
    <p class="counter dim" data-role="source"></p>
    <div class="row">
      <span class="lbl">出力解像度</span>
      <label><input type="radio" name="${panelId}_preset" value="1080x1920" checked> 1080×1920</label>
      <label><input type="radio" name="${panelId}_preset" value="x2"> 2倍</label>
    </div>
    <div class="row">
      <span class="lbl">方式</span>
      <div class="upscale-method-group">
        <p class="upscale-method-heading">標準拡大（解像度変換のみ、追加モデル不要）</p>
        <label><input type="radio" name="${panelId}_method" value="lanczos" checked> 標準拡大（Lanczos, FFmpeg）</label>
        <p class="upscale-method-heading">AI高画質化（モデルでディテール補完）</p>
        <label><input type="radio" name="${panelId}_method" value="esrgan"> AI高画質化（RealESRGAN）</label>
        <label><input type="radio" name="${panelId}_method" value="seedvr2"> AI高画質化（SeedVR2 7B）</label>
      </div>
    </div>
    <p class="counter dim" data-role="target"></p>
    <div class="upscale-missing" data-role="missing" hidden></div>
    <div class="actions">
      <button type="button" class="btn primary" data-role="start">開始</button>
      <button type="button" class="btn" data-role="cancel" hidden>中止</button>
    </div>
    <div class="upscale-progress" data-role="progress" hidden>
      <div class="progress"><div class="bar" data-role="fill"></div></div>
      <p class="counter" data-role="stage"></p>
    </div>
    <div class="upscale-result" data-role="result" hidden></div>
  `;

  const q = (role) => panel.querySelector(`[data-role="${role}"]`);
  const presetRadios = () => Array.from(panel.querySelectorAll(`input[name="${panelId}_preset"]`));
  const methodRadios = () => Array.from(panel.querySelectorAll(`input[name="${panelId}_method"]`));
  const currentPreset = () => (presetRadios().find((r) => r.checked) || {}).value || "1080x1920";
  const currentMethod = () => (methodRadios().find((r) => r.checked) || {}).value || "lanczos";

  function renderMissing(plan) {
    const box = q("missing");
    box.innerHTML = "";
    const missing = (plan && plan.missing) || [];
    if (!missing.length) { box.hidden = true; return; }
    const title = document.createElement("p");
    title.className = "warn";
    title.textContent = "不足しているノード・モデルがあります。H3は自動でダウンロードしません。導入は利用者の作業です。";
    box.appendChild(title);
    missing.forEach((m) => {
      const line = document.createElement("div");
      line.className = "upscale-missing-item";
      const bits = [`［${m.kind}］${m.name}`];
      if (m.source_url) bits.push(`配布元: ${m.source_url}`);
      if (m.license) bits.push(`ライセンス: ${m.license}`);
      const size = upscaleFmtSize(m.size_bytes);
      if (size) bits.push(`容量: ${size}`);
      line.textContent = bits.join(" / ");
      box.appendChild(line);
    });
    box.hidden = false;
  }

  async function refreshOptions() {
    const source = getSource();
    if (!source) return;
    q("target").textContent = "取得中…";
    try {
      const params = new URLSearchParams({
        kind: source.kind, preset: currentPreset(), method: currentMethod(),
      });
      if (source.id) params.set("id", source.id);
      if (source.name) params.set("name", source.name);
      if (source.step !== undefined && source.step !== null) params.set("step", source.step);
      if (source.final) params.set("final", "1");
      const res = await api(`/api/upscale/options?${params.toString()}`);
      q("source").textContent = upscaleFmtSource(res.source);
      const p = res.plan;
      q("target").textContent = (p && p.target_w)
        ? `出力サイズ: ${p.target_w}×${p.target_h}` : "";
      renderMissing(p);
    } catch (e) {
      q("target").textContent = "";
      renderMissing(null);
      toast("高画質化の設定を取得できませんでした: " + e.message);
    }
  }

  [...presetRadios(), ...methodRadios()].forEach((r) => {
    r.addEventListener("change", refreshOptions);
  });

  function setRunningUI(running) {
    q("start").hidden = running;
    q("cancel").hidden = !running;
    q("progress").hidden = !running;
    [...presetRadios(), ...methodRadios()].forEach((r) => { r.disabled = running; });
  }

  function renderProgress(data) {
    q("fill").style.width = (data.percent || 0) + "%";
    const stages = data.stages || [];
    const current = stages.find((s) => s.status === "実行中")
      || [...stages].reverse().find((s) => s.status === "完了" || s.status === "失敗");
    q("stage").textContent = current ? `${current.name}（${current.status}）` : "";
  }

  function renderDone(data) {
    setRunningUI(false);
    const box = q("result");
    box.innerHTML = "";
    box.hidden = false;
    if (data.error) {
      const p = document.createElement("p");
      p.className = "warn";
      p.textContent = (data.error.title || "失敗しました")
        + (data.error.config ? `: ${data.error.config}` : "");
      box.appendChild(p);
      return;
    }
    const result = data.result;
    if (!result || !result.ok) {
      const p = document.createElement("p");
      p.className = "warn";
      p.textContent = (result && result.error) || "高画質化に失敗しました。";
      box.appendChild(p);
      return;
    }
    const exp = result.export || {};
    const p = document.createElement("p");
    p.className = "counter";
    p.textContent = `完成: ${exp.name || exp.path || ""}`;
    box.appendChild(p);
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn";
    copyBtn.textContent = "パスをコピー";
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(exp.path || "");
        toast("パスをコピーしました。");
      } catch (e) { toast("コピーできませんでした。"); }
    };
    box.appendChild(copyBtn);
  }

  function renderCancelled() {
    setRunningUI(false);
    const box = q("result");
    box.innerHTML = "";
    box.hidden = false;
    const p = document.createElement("p");
    p.className = "counter";
    p.textContent = "中止しました。";
    box.appendChild(p);
  }

  function listenJob(jobId) {
    if (st.events) st.events.close();
    const es = new EventSource(`/api/events/${jobId}`);
    st.events = es;
    es.onmessage = (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch (e) { return; }
      renderProgress(data);
      if (data.finished) {
        es.close();
        st.events = null;
        st.jobId = null;
        if (data.error && data.error.kind === "cancelled") renderCancelled();
        else renderDone(data);
      }
    };
    es.addEventListener("end", () => { es.close(); st.events = null; });
    es.onerror = () => { /* the panel just stays as last rendered */ };
  }

  async function start(confirmMissing) {
    const source = getSource();
    if (!source) return;
    q("result").hidden = true;
    try {
      const res = await postJson("/api/upscale/start", {
        source, preset: currentPreset(), method: currentMethod(),
        confirm_missing: !!confirmMissing,
      });
      st.jobId = res.job_id;
      setRunningUI(true);
      renderProgress({ percent: 0, stages: [] });
      listenJob(res.job_id);
    } catch (e) {
      if (e.status === 409 && e.body && e.body.plan && !confirmMissing) {
        const missing = (e.body.plan.missing || [])
          .map((m) => `［${m.kind}］${m.name}`).join("\n");
        if (window.confirm(
            "必要なノードまたはモデルが不足しています。H3は自動でダウンロードしません。\n"
            + missing
            + "\n\n導入済みであることを確認したうえで、このまま開始しますか？")) {
          await start(true);
        }
        return;
      }
      toast("高画質化を開始できませんでした: " + e.message);
    }
  }

  q("start").onclick = () => start(false);
  q("cancel").onclick = async () => {
    if (!st.jobId) return;
    try {
      await postJson("/api/upscale/cancel", { job_id: st.jobId });
    } catch (e) { toast("中止できませんでした: " + e.message); }
  };

  btn.onclick = () => {
    const opening = panel.hidden;
    panel.hidden = !opening;
    if (opening) refreshOptions();
  };

  return {
    setDisabled(disabled) {
      btn.disabled = disabled;
      if (disabled) {
        const why = disabled === true ? "" : disabled;
        btn.title = why ? String(why) : "生成中は使用できません。";
      } else {
        btn.title = "";
      }
    },
  };
}

// ------------------------------------------------------------------ wire ---
window.addEventListener("DOMContentLoaded", () => {
  $("btnAddImage").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { uploadFiles(e.target.files); e.target.value = ""; };

  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((k) => dz.addEventListener(k, (e) => {
    e.preventDefault(); dz.classList.add("hover");
  }));
  ["dragleave", "drop"].forEach((k) => dz.addEventListener(k, (e) => {
    e.preventDefault(); dz.classList.remove("hover");
  }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
  });

  // A file dropped anywhere outside the dropzone must never navigate the
  // page away from the app; only intercept drags that carry files.
  const isFileDrag = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes("Files");
  window.addEventListener("dragover", (e) => { if (isFileDrag(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => { if (isFileDrag(e)) e.preventDefault(); });

  const mediaNotice = $("mediaFolderNotice");
  if (mediaNotice) {
    $("btnCopyMediaPath").onclick = async () => {
      const path = mediaNotice.dataset.path || "";
      if (!path) return;
      try {
        await navigator.clipboard.writeText(path);
        toast("パスをコピーしました。");
      } catch (e) { toast("コピーできませんでした。"); }
    };
    $("btnCloseMediaNotice").onclick = () => hideMediaFolderNotice();
  }

  $("btnGenerate").onclick = () => startGeneration("new");
  $("btnAgain").onclick = () => startGeneration("again");
  $("btnContinue").onclick = () => startGeneration("continue");

  // WP-B: 高画質化 panels (single-result card + story final card).
  upscalePanels.result = createUpscalePanel({
    btnId: "btnUpscaleResult", panelId: "upscalePanelResult",
    getSource: () => (state.projectId ? { kind: "project", id: state.projectId } : null),
  });
  upscalePanels.story = createUpscalePanel({
    btnId: "btnUpscaleStory", panelId: "upscalePanelStory",
    getSource: () => (state.story.id ? { kind: "story", id: state.story.id, final: true } : null),
  });
  refreshButtons();
  // Sticky action bar mirrors the real buttons (never bypasses them).
  $("btnNowGo").onclick = () => {
    if (state.mode === "story") { storyStart(); }
    else if (state.mode === "director") { directorGenerate(); }
    else if (state.mode === "audio") { return; }
    else { $("btnGenerate").click(); }
  };
  $("btnNowCancel").onclick = () => {
    if (state.mode === "story") { storyStop(); }
    else { $("btnCancel").click(); }
  };
  document.querySelectorAll("#sidenav [data-nav]").forEach((a) => {
    a.addEventListener("click", () => {
      document.querySelectorAll("#sidenav [data-nav]").forEach((x) => x.classList.remove("on"));
      a.classList.add("on");
    });
  });
  $("btnMediaFolder").onclick = () => openMediaFolder();
  $("btnCharacterApplyView").onclick = () => applySelectedCharacter();
  $("btnCharacterSave").onclick = () => openCharacterEditor();
  $("btnCharacterRefresh").onclick = () => loadCharacters();
  $("btnCharacterDoSave").onclick = () => saveCharacterEditor();
  $("btnCharacterCancel").onclick = () => { $("characterEditor").hidden = true; };
  $("characterSelect").onchange = (e) => {
    state.character.selectedId = e.target.value;
    syncCharacterSelects("characterSelect");
  };
  $("directorCharacter").onchange = (e) => {
    state.character.selectedId = e.target.value;
    syncCharacterSelects("directorCharacter");
  };
  $("btnCleanupScan").onclick = () => cleanupScan();
  $("btnCleanupDo").onclick = () => cleanupDo();
  $("btnCleanupCancel").onclick = () => cleanupCancel();
  $("btnResetAdvanced").onclick = () => { resetAdvanced(); toast("既定値に戻しました。"); };

  $("btnRelease").onclick = async () => {
    if (state.busy) { toast("生成中のためモデルを解放できません。"); return; }
    const btn = $("btnRelease");
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "解放中...";
    try {
      const res = await postJson("/api/release-models", {});
      showRelease(res.message, false);
    } catch (e) {
      showRelease(e.message, true);
    } finally {
      btn.textContent = label;
      refreshButtons();
    }
  };
  $("btnRetry").onclick = () => { if (state.lastAction) startGeneration(state.lastAction); };

  $("btnCancel").onclick = async () => {
    if (!state.projectId) return;
    try {
      const res = await fetch("/api/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: state.projectId }),
      });
      const body = await res.json();
      toast(body.message || "中止を要求しました。");
    } catch (e) { toast("中止できませんでした: " + e.message); }
  };

  $("btnPlay").onclick = () => {
    const v = $("player");
    if (v.paused) { v.play(); } else { v.pause(); }
  };

  // 保存フォルダを開く。送るのは「どのプロジェクトか」と「どこを開きたいか」だけで、
  // パスは一切送らない（サーバー側が project.json から解決して検証する）。
  $("btnOpenFolder").onclick = async () => {
    if (!state.projectId) return;
    try {
      await postJson("/api/open-folder", {
        project_id: state.projectId, target: "final", reveal: true,
      });
    } catch (e) { toast(e.message); }
  };

  // ------------------------------------------------------------- story ----
  $("btnModeSingle").onclick = () => setMode("single");
  $("btnModeStory").onclick = () => { setMode("story"); refreshStoryList().catch(() => {}); };
  $("btnModeDirector").onclick = () => { setMode("director"); directorDurations(); };
  const btnModeAudio = $("btnModeAudio");
  if (btnModeAudio) btnModeAudio.onclick = () => setMode("audio");

  // ------------------------------------------------------------ 音声テスト --
  const audioLangBox = $("audioTestLang");
  if (audioLangBox) {
    audioLangBox.querySelectorAll(".choice").forEach((b) => {
      b.onclick = () => {
        state.audioTest.lang = b.dataset.lang;
        [...audioLangBox.children].forEach((c) => c.classList.remove("on"));
        b.classList.add("on");
      };
    });
  }
  const btnAudioQuick = $("btnAudioTestQuick");
  if (btnAudioQuick) btnAudioQuick.onclick = () => runAudioTest("quick");
  const btnAudioFull = $("btnAudioTestFull");
  if (btnAudioFull) btnAudioFull.onclick = () => runAudioTest("full");
  const btnAudioDeleteAll = $("btnAudioTestDeleteAll");
  if (btnAudioDeleteAll) btnAudioDeleteAll.onclick = () => deleteAllAudioTests();

  // mousedown の既定動作を止めないと、ボタンを押した瞬間に台本の選択範囲が
  // 消えてしまい「選んでから種類を変える」が使えない。
  [$("btnLineModePrompt"), $("btnLineModeSpeech")].forEach((b) => {
    b.addEventListener("mousedown", (ev) => ev.preventDefault());
  });
  $("btnLineModePrompt").onclick = () => onModeButton("prompt");
  $("btnLineModeSpeech").onclick = () => onModeButton("speech");
  $("btnAddLine").onclick = () => {
    const runs = readEditor();
    const len = runs.reduce((n, r) => n + r.text.length, 0);
    insertTextAtOffsets(len, len, "\n", editorState.mode);
    scriptEditor().focus();
    insertAnchor(editorState.mode);
  };
  $("btnClearLines").onclick = () => {
    renderEditor([]);
    saveScriptDraft();
    focusEditorEnd();
  };
  $("btnStoryPreflight").onclick = () => storyPreflight();
  $("btnStoryFromText").onclick = () => storyPreviewFromText();
  $("btnStoryResplit").onclick = () => storyResplit();
  $("btnStorySaveSegments").onclick = () => storySaveSegments(false);
  $("btnSegmentAdd").onclick = () => segInsert(state.story.segments.length);

  $("btnStoryStart").onclick = () => storyStart();
  $("btnStoryStop").onclick = () => storyStop();
  $("btnStoryDiscard").onclick = () => storyDiscard(null, null);
  $("btnStoryMerge").onclick = () => storyMerge();
  $("btnStoryResume").onclick = () => storyStart();
  $("btnStoryRefreshList").onclick = () => refreshStoryList();
  $("storyName").oninput = () => renderStoryPanel(null);

  $("btnStoryPlay").onclick = () => {
    const v = $("storyPlayer");
    if (v.paused) { v.play(); } else { v.pause(); }
  };
  // ストーリーの完成動画。story_id と target だけを送る。開けなかったときだけ
  // 保存先の文字列を出す（クリップのフォルダは主要な操作としては出さない）。
  async function storyOpenFolder(reveal) {
    if (!state.story.id) return;
    try {
      await postJson("/api/story/open-folder", {
        story_id: state.story.id, target: "final", reveal: !!reveal,
      });
    } catch (e) {
      const path = state.story.finalPath;
      if (path) {
        toast(`${e.message}\n保存先: ${path}`);
        try { await navigator.clipboard.writeText(path); } catch (e2) { /* ignore */ }
      } else {
        toast(e.message);
      }
    }
  }
  $("btnStoryOpenFolder").onclick = () => storyOpenFolder(false);
  $("btnStoryRevealFile").onclick = () => storyOpenFolder(true);

  // ------------------------------------------------------- AI Director ----
  $("btnDirectorCreate").onclick = () => directorCreate();
  $("btnDirectorRegen").onclick = () => directorRegen(null);
  $("btnDirectorGenerate").onclick = () => directorGenerate();
  directorDurations();

  wireAiSettings();
  wireShutdown(); // WP-D: header/settings "H3を終了" button + result dialog.

  wireEditor();
  setLineMode("prompt");
  loadScriptDraft();
  $("storyName").addEventListener("input", saveScriptDraft);

  setMode("single");
  loadCharacters().catch(() => {});
  startHeartbeat();
  window.addEventListener("pagehide", detachBeacon);
  window.addEventListener("beforeunload", detachBeacon);

  boot().catch((e) => toast("起動に失敗しました: " + e.message));
});
