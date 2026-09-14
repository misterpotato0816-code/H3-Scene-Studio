"use strict";

(() => {
  const el = (id) => document.getElementById(id);
  let query = null, analyzedText = "", analyzedSpeaker = "", previewUrl = "", busy = false;
  const status = (text) => { el("voiceRehearsalStatus").textContent = text; };

  function sync() {
    document.body.dataset.studioMode = state.mode;
    document.body.classList.toggle("studio-action-visible", !el("actionbar").hidden);
    const source = state.images[0];
    el("studioSourceCount").textContent = source ? `${state.images.length} SOURCE${state.images.length > 1 ? "S" : ""}` : "NO SOURCE";
    const image = el("studioReference");
    image.hidden = !source;
    el("studioEmpty").hidden = !!source;
    if (source && image.getAttribute("src") !== source.thumb) image.src = source.thumb;
    el("studioFormat").textContent = `${state.aspect.toUpperCase()} / ${state.mode === "audio" ? "VOICE" : state.genMode}`;
    for (const [id, mode] of [["btnModeSingle", "single"], ["btnModeStory", "story"], ["btnModeDirector", "director"], ["btnModeAudio", "audio"]]) {
      el(id).setAttribute("aria-pressed", String(state.mode === mode));
    }
  }
  window.H3Studio = { sync };
  el("studioGoDirector").onclick = () => { setMode("director"); el("cardDirector").scrollIntoView({ behavior: "smooth" }); };
  el("studioGoVoice").onclick = () => { setMode("audio"); el("cardAudioTest").scrollIntoView({ behavior: "smooth" }); };

  const nav = document.createElement("a");
  nav.href = "#cardRehearsal"; nav.dataset.nav = ""; nav.dataset.audio = "";
  nav.textContent = "読み・アクセント";
  nav.classList.add("navhide"); el("sidenav").appendChild(nav);
  const presets = document.createElement("div"); presets.className = "studio-presetbar";
  const ideas = [
    ["ワンカット", "カットを割らず、ひとつの動作を始まりから終わりまで見せる。カメラは小さくゆっくり動かす。"],
    ["会話の余白", "普段の会話のような口語にする。台詞の前後に視線と息継ぎの間を残し、語尾を急がない。演技指示は台詞に含めない。"],
    ["静かな映画", "感情は視線と小さな表情で伝える。落ち着いた構図と柔らかな光。過剰なカメラ移動を避ける。"],
  ];
  for (const [label, instruction] of ideas) {
    const button = document.createElement("button"); button.type = "button"; button.className = "btn"; button.textContent = label;
    button.onclick = () => { const field = el("directorRequest"); if (!field.value.includes(instruction)) field.value += (field.value ? "\n" : "") + instruction; field.focus(); };
    presets.appendChild(button);
  }
  el("directorRequest").after(presets);

  el("btnDirectionReview").onclick = async () => {
    const box = el("directionReview"); box.textContent = "確認中…";
    try {
      const response = await fetch("/api/director/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ spec: state.director.spec }) });
      const report = await response.json(); box.replaceChildren();
      for (const text of [...(report.errors || []), ...(report.warnings || []), ...(report.lines || []).map((c) => `${c.clip}：${c.silent ? "無言" : c.dialogue.join(" ／ ")}`), "音声・口パクは生成後に試聴して確認してください。"]) {
        const p = document.createElement("p"); p.textContent = text; box.appendChild(p);
      }
      if (report.message) box.textContent = report.message;
    } catch (error) { box.textContent = error.message; }
  };

  async function withBusy(action) {
    if (busy) return;
    busy = true;
    const buttons = ["btnVoiceConnect", "btnVoiceAnalyze", "btnVoicePlay"];
    buttons.forEach((id) => { el(id).disabled = true; });
    try { await action(); } catch (error) { status(error.message); }
    finally { busy = false; buttons.forEach((id) => { el(id).disabled = false; }); el("btnVoicePlay").disabled = !query; }
  }
  el("btnVoiceConnect").onclick = () => withBusy(async () => {
    status("VOICEVOXに接続しています…");
    const result = await api("/api/voice-rehearsal/speakers");
    const select = el("voiceSpeaker"); select.replaceChildren();
    for (const speaker of result.speakers || []) for (const style of speaker.styles || []) {
      const option = document.createElement("option"); option.value = style.id; option.textContent = `${speaker.name} / ${style.name}`; select.appendChild(option);
    }
    query = null; status("接続しました。台詞の読みを解析できます。");
  });
  function invalidate() {
    query = null; el("btnVoicePlay").disabled = true;
    el("voiceAccents").replaceChildren();
    el("voicePreviewPlayer").pause(); el("voicePreviewPlayer").hidden = true;
    status("台詞または話者を変更しました。もう一度、読みを解析してください。");
  }
  el("rehearsalText").oninput = invalidate; el("voiceSpeaker").onchange = invalidate;
  el("btnVoiceCopy").onclick = () => { el("rehearsalText").value = el("audioTestSpoken").value || el("audioTestText").value; invalidate(); };
  el("btnVoiceAnalyze").onclick = () => withBusy(async () => {
    const text = el("rehearsalText").value, speaker = el("voiceSpeaker").value;
    if (!speaker) throw new Error("先にVOICEVOXへ接続し、話者を選んでください。");
    query = null; status("読みを解析しています…");
    const result = await postJson("/api/voice-rehearsal/query", { text, speaker });
    if (text !== el("rehearsalText").value || speaker !== el("voiceSpeaker").value) { status("入力が変わりました。もう一度解析してください。"); return; }
    query = result.query; analyzedText = text; analyzedSpeaker = speaker;
    const box = el("voiceAccents"); box.replaceChildren();
    (query.accent_phrases || []).forEach((phrase, index) => {
      const card = document.createElement("div"); card.className = "accent-phrase";
      const label = document.createElement("label"); label.textContent = phrase.moras.map((m) => m.text).join("");
      const select = document.createElement("select"); select.setAttribute("aria-label", `アクセント句${index + 1}の位置`);
      phrase.moras.forEach((mora, i) => { const option = document.createElement("option"); option.value = i + 1; option.textContent = `アクセント ${i + 1}：${mora.text}`; select.appendChild(option); });
      select.value = phrase.accent; select.onchange = () => { phrase.accent = Number(select.value); };
      label.appendChild(select); card.appendChild(label); box.appendChild(card);
    });
    status("アクセント位置・話速・抑揚を調整し、試聴してください。");
  });
  for (const [range, output] of [["voiceSpeed", "voiceSpeedValue"], ["voiceIntonation", "voiceIntonationValue"]]) {
    el(range).oninput = () => { el(output).value = Number(el(range).value).toFixed(2); };
  }
  el("btnVoicePlay").onclick = () => withBusy(async () => {
    if (!query || analyzedText !== el("rehearsalText").value || analyzedSpeaker !== el("voiceSpeaker").value) throw new Error("もう一度、読みを解析してください。");
    query.speedScale = Number(el("voiceSpeed").value); query.intonationScale = Number(el("voiceIntonation").value);
    status("試聴音声を作っています…");
    const response = await fetch("/api/voice-rehearsal/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, speaker: analyzedSpeaker }) });
    if (!response.ok) throw new Error((await response.json()).message || "試聴に失敗しました。");
    const blob = await response.blob();
    if (!query) { status("入力が変わったため試聴を取り消しました。"); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    const player = el("voicePreviewPlayer"); player.src = previewUrl; player.hidden = false;
    status("VOICEVOXでの試聴です。H3本番の声・口パクは別途確認してください。");
    await player.play().catch(() => {});
  });
  el("btnVoiceUseReading").onclick = () => {
    el("audioTestSpoken").value = el("rehearsalText").value;
    el("cardAudioTest").scrollIntoView({ behavior: "smooth" });
    toast("読みの文字列を取り込みました。アクセント調整値はH3に適用されません。");
  };
  window.addEventListener("pagehide", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
  new MutationObserver(sync).observe(el("actionbar"), { attributes: true, attributeFilter: ["hidden"] });
  new MutationObserver(sync).observe(el("slots"), { childList: true });
  sync();
})();
