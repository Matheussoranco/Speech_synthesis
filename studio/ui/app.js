"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (s) => { s = Math.max(0, s || 0); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };

let state = { status: null, voices: [] };

// ── API ───────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = body?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || res.statusText);
  }
  return body;
}

function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

/** Submit a job, show its progress in `box`, resolve with its result. */
async function runJob(start, box, button) {
  box.hidden = false;
  box.className = "job";
  box.innerHTML = `<div class="line"><span class="msg">Enviando…</span><button type="button">Cancelar</button></div><div class="bar"><div></div></div>`;
  if (button) button.disabled = true;
  let job;
  try {
    job = await start();
    $("button", box).onclick = () => api(`/api/jobs/${job.id}/cancel`, { method: "POST" });
    while (job.status === "queued" || job.status === "running") {
      $(".msg", box).textContent = job.status === "queued" ? "Na fila (outra tarefa em andamento)…" : job.message;
      $(".bar > div", box).style.width = `${Math.round(job.progress * 100)}%`;
      await new Promise((r) => setTimeout(r, 700));
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "cancelled") { box.hidden = true; return null; }
    if (job.status === "error") throw new Error(job.error);
    box.hidden = true;
    refreshStatus();
    return job.result;
  } catch (e) {
    box.className = "job error";
    box.textContent = e.message;
    return null;
  } finally {
    if (button) button.disabled = false;
  }
}

// ── tabs ──────────────────────────────────────────────────────────────

function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "history") loadHistory();
  if (name === "voices") loadVoices();
  try { localStorage.setItem("tab", name); } catch {}
}
$$(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));

// ── status ────────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    const dev = s.gpu ? `GPU ${s.gpu.name.replace(/NVIDIA |GeForce /g, "")} · ${s.gpu.vram_free_mb} MB livres` : "CPU (lento)";
    const models = [s.tts_loaded && "voz", s.asr_loaded && "transcrição"].filter(Boolean);
    $("#status-text").textContent = `${dev}${models.length ? " · carregado: " + models.join(", ") : ""}`;
    return s;
  } catch {
    $("#status-text").textContent = "Servidor desconectado";
  }
}
$("#unload").onclick = async () => { await api("/api/unload", { method: "POST" }); refreshStatus(); toast("Memória liberada"); };

function fillSelects(s) {
  for (const sel of $$(".lang-select")) {
    const opts = sel.classList.contains("no-auto") ? [] : [`<option value="">Detectar automaticamente</option>`];
    for (const l of s.languages) opts.push(`<option value="${esc(sel.id === "dub-target" ? l.name : l.code)}">${esc(l.label)}</option>`);
    sel.innerHTML = opts.join("");
  }
  $("#dub-target").value = "Portuguese";
  $("#speak-format").innerHTML = s.formats.map((f) => `<option value="${f}">${f.toUpperCase()}</option>`).join("");
  $("#speak-format").value = "mp3";
  const labels = {
    male: "Masculino", female: "Feminino", child: "Criança", teenager: "Adolescente", "young adult": "Jovem adulto",
    "middle-aged": "Meia-idade", elderly: "Idoso", "very low pitch": "Muito grave", "low pitch": "Grave",
    "moderate pitch": "Médio", "high pitch": "Agudo", "very high pitch": "Muito agudo",
  };
  const fill = (id, key) => {
    $(id).innerHTML = `<option value="">Qualquer</option>` +
      s.design_options[key].map((o) => `<option value="${esc(o)}">${esc(labels[o] || o.replace(" accent", ""))}</option>`).join("");
  };
  fill("#d-gender", "gender"); fill("#d-age", "age"); fill("#d-pitch", "pitch"); fill("#d-accent", "accent");
  $("#dub-hint").textContent = s.translation_models.length
    ? `Tradução feita localmente pelo Ollama (${s.translation_models[0]}).`
    : "Sem Ollama: só é possível dublar para inglês ou para o mesmo idioma.";
}

// ── recording ─────────────────────────────────────────────────────────

/** Toggle microphone recording on `button`; calls onDone(file) when stopped. */
function recorder(button, onDone) {
  let rec = null;
  button.onclick = async () => {
    if (rec) { rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { toast("Não foi possível acessar o microfone"); return; }
    const chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      button.textContent = "● Gravar"; button.classList.remove("recording");
      const blob = new Blob(chunks, { type: rec.mimeType });
      rec = null;
      onDone(new File([blob], "gravacao.webm", { type: blob.type }));
    };
    rec.start();
    button.textContent = "■ Parar"; button.classList.add("recording");
  };
}

function previewFile(audioEl, file) {
  if (audioEl.src) URL.revokeObjectURL(audioEl.src);
  audioEl.src = URL.createObjectURL(file);
  audioEl.hidden = false;
}

// ── Falar ─────────────────────────────────────────────────────────────

let speakMode = "auto";
let cloneRecording = null;

$$("#voice-mode button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
function setMode(mode) {
  speakMode = mode;
  $$("#voice-mode button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === mode));
  $$("#tab-speak .mode-box").forEach((m) => (m.hidden = m.dataset.for !== mode));
}

$("#speak-text").oninput = () => ($("#speak-count").textContent = `${$("#speak-text").value.length.toLocaleString("pt-BR")} caracteres`);
$("#speak-file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("#speak-text").value = await f.text();
  $("#speak-text").oninput();
  e.target.value = "";
};
$("#speak-speed").oninput = (e) => ($("#speed-val").textContent = `${Number(e.target.value).toFixed(2).replace(/0$/, "")}×`);
$("#clone-file").onchange = (e) => { cloneRecording = null; if (e.target.files[0]) previewFile($("#clone-preview"), e.target.files[0]); };
recorder($("#clone-rec"), (file) => { cloneRecording = file; $("#clone-file").value = ""; previewFile($("#clone-preview"), file); });

$("#speak-go").onclick = async () => {
  const fd = new FormData();
  fd.append("text", $("#speak-text").value);
  fd.append("mode", speakMode);
  fd.append("language", $("#speak-lang").value);
  fd.append("speed", $("#speak-speed").value);
  fd.append("format", $("#speak-format").value);
  fd.append("num_step", $("#speak-steps").value);
  fd.append("seed", $("#speak-seed").value);
  if (speakMode === "voice") {
    if (!$("#speak-voice").value) return toast("Escolha uma voz salva");
    fd.append("voice_id", $("#speak-voice").value);
  } else if (speakMode === "design") {
    const parts = ["#d-gender", "#d-age", "#d-pitch", "#d-accent"].map((id) => $(id).value).filter(Boolean);
    if ($("#d-whisper").checked) parts.push("whisper");
    if (!parts.length) return toast("Escolha pelo menos uma característica da voz");
    fd.append("instruct", parts.join(", "));
  } else if (speakMode === "clone") {
    const file = cloneRecording || $("#clone-file").files[0];
    if (!file) return toast("Envie ou grave um áudio de referência");
    fd.append("ref_audio", file);
    fd.append("ref_text", $("#clone-text").value);
  }
  const result = await runJob(() => api("/api/generate", { method: "POST", body: fd }), $("#speak-job"), $("#speak-go"));
  if (!result) return;
  const box = $("#speak-result");
  box.hidden = false;
  box.innerHTML = `
    <audio controls autoplay src="${esc(result.audio_url)}"></audio>
    <div class="row gap wrap between">
      <span class="muted small">${fmtTime(result.duration)} de áudio</span>
      <span class="row gap-s">
        <a class="file-btn small" href="${esc(result.audio_url)}" download>Baixar</a>
        <button class="ghost small" id="save-as-voice">Salvar como voz</button>
      </span>
    </div>`;
  $("#save-as-voice").onclick = () => saveHistoryAsVoice(result.history_id);
};

async function saveHistoryAsVoice(historyId, box = $("#speak-job")) {
  const name = prompt("Nome da nova voz:");
  if (!name) return;
  const fd = new FormData();
  fd.append("history_id", historyId);
  fd.append("name", name);
  const voice = await runJob(() => api("/api/voices/from-history", { method: "POST", body: fd }), box);
  if (voice) { toast(`Voz "${voice.name}" salva`); loadVoices(); }
}

// ── Vozes ─────────────────────────────────────────────────────────────

let voiceRecording = null;
$("#v-file").onchange = (e) => { voiceRecording = null; if (e.target.files[0]) previewFile($("#v-preview"), e.target.files[0]); };
recorder($("#v-rec"), (file) => { voiceRecording = file; $("#v-file").value = ""; previewFile($("#v-preview"), file); });

$("#voice-form").onsubmit = async (e) => {
  e.preventDefault();
  const file = voiceRecording || $("#v-file").files[0];
  if (!file) return toast("Envie ou grave um áudio");
  const fd = new FormData();
  fd.append("name", $("#v-name").value);
  fd.append("audio_file", file);
  fd.append("ref_text", $("#v-text").value);
  fd.append("language", $("#v-lang").value);
  const voice = await runJob(() => api("/api/voices", { method: "POST", body: fd }), $("#voice-job"), $("#voice-form button.primary"));
  if (!voice) return;
  toast(`Voz "${voice.name}" salva`);
  e.target.reset(); voiceRecording = null; $("#v-preview").hidden = true;
  loadVoices();
};

async function loadVoices() {
  state.voices = await api("/api/voices");
  const opts = state.voices.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}</option>`).join("");
  const current = $("#speak-voice").value;
  $("#speak-voice").innerHTML = opts;
  if (current) $("#speak-voice").value = current;
  $("#speak-voice").hidden = !state.voices.length;
  $("#speak-voice-empty").hidden = !!state.voices.length;
  $("#dub-voice").innerHTML = `<option value="">Voz original (clonada do vídeo)</option>` + opts;

  const list = $("#voice-list");
  if (!state.voices.length) { list.innerHTML = `<p class="empty">Nenhuma voz salva.</p>`; return; }
  list.innerHTML = state.voices.map((v) => `
    <div class="item" data-id="${esc(v.id)}">
      <div class="head">
        <span class="title">${esc(v.name)}</span>
        <span class="badge">${{ upload: "gravada", design: "desenhada", clone: "clonada", auto: "gerada" }[v.source] || esc(v.source)}</span>
      </div>
      <audio controls preload="none" src="${esc(v.audio_url)}"></audio>
      <input class="ref-text" value="${esc(v.ref_text)}" title="Transcrição da referência">
      <div class="actions">
        <button class="small use">Usar</button>
        <button class="small ghost save-text">Salvar transcrição</button>
        <button class="small ghost rename">Renomear</button>
        <button class="small ghost danger delete">Excluir</button>
      </div>
    </div>`).join("");
  $$(".item", list).forEach((el) => {
    const id = el.dataset.id;
    const patch = (body) => api(`/api/voices/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    $(".use", el).onclick = () => { showTab("speak"); setMode("voice"); $("#speak-voice").value = id; };
    $(".save-text", el).onclick = async () => { await patch({ ref_text: $(".ref-text", el).value }); toast("Transcrição salva"); };
    $(".rename", el).onclick = async () => {
      const name = prompt("Novo nome:", $(".title", el).textContent);
      if (name) { await patch({ name }); loadVoices(); }
    };
    $(".delete", el).onclick = async () => {
      if (!confirm("Excluir esta voz?")) return;
      await api(`/api/voices/${id}`, { method: "DELETE" });
      loadVoices();
    };
  });
}

// ── Transcrever ───────────────────────────────────────────────────────

$("#tr-form").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("file", $("#tr-file").files[0]);
  fd.append("language", $("#tr-lang").value);
  fd.append("task", $("#tr-task").value);
  const r = await runJob(() => api("/api/transcribe", { method: "POST", body: fd }), $("#tr-job"), $("#tr-form button.primary"));
  if (!r) return;
  const box = $("#tr-result");
  box.hidden = false;
  box.innerHTML = `
    <div class="row between wrap gap">
      <span class="muted small">Idioma detectado: ${esc(r.language)} · ${r.segments.length} trechos</span>
      <span class="links">${Object.entries(r.files).map(([k, u]) => `<a href="${esc(u)}" download>.${k}</a>`).join("")}
        <a href="#" id="tr-copy">Copiar texto</a></span>
    </div>
    <textarea rows="12" readonly>${esc(r.text)}</textarea>`;
  $("#tr-copy").onclick = (ev) => { ev.preventDefault(); navigator.clipboard.writeText(r.text); toast("Copiado"); };
};

// ── Dublar ────────────────────────────────────────────────────────────

$("#dub-vol").oninput = (e) => ($("#dub-vol-val").textContent = `${Math.round(e.target.value * 100)}%`);
$("#dub-form").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("file", $("#dub-file").files[0]);
  fd.append("target_language", $("#dub-target").value);
  fd.append("source_language", $("#dub-src").value);
  fd.append("voice_id", $("#dub-voice").value);
  fd.append("original_volume", $("#dub-vol").value);
  const r = await runJob(() => api("/api/dub", { method: "POST", body: fd }), $("#dub-job"), $("#dub-form button.primary"));
  if (!r) return;
  const box = $("#dub-result");
  box.hidden = false;
  const media = r.files.video ? `<video controls src="${esc(r.files.video)}"></video>` : `<audio controls src="${esc(r.files.audio)}"></audio>`;
  box.innerHTML = `${media}
    <div class="links">${Object.entries(r.files).map(([k, u]) => `<a href="${esc(u)}" download>Baixar ${k === "srt" ? "legenda" : k === "video" ? "vídeo" : "áudio"}</a>`).join("")}</div>
    <details><summary class="muted small">Falas traduzidas (${r.lines.length})</summary>
      <table class="lines">${r.lines.map((l) => `<tr><td>${fmtTime(l.start)}</td><td>${esc(l.text)}</td></tr>`).join("")}</table>
    </details>`;
};

// ── Histórico ─────────────────────────────────────────────────────────

const KIND = { auto: "Fala", design: "Voz desenhada", voice: "Voz salva", clone: "Clonagem", transcribe: "Transcrição", dub: "Dublagem" };

async function loadHistory() {
  const items = await api("/api/history");
  const list = $("#history-list");
  if (!items.length) { list.innerHTML = `<p class="empty">Nada gerado ainda.</p>`; return; }
  list.innerHTML = items.map((it) => {
    const f = it.files;
    const media = f.video ? `<video controls preload="none" src="${esc(f.video)}" style="max-height:260px"></video>`
      : f.audio ? `<audio controls preload="none" src="${esc(f.audio)}"></audio>` : "";
    const links = Object.entries(f).map(([k, u]) => `<a href="${esc(u)}" download>.${esc(u.split(".").pop())}</a>`);
    const canSave = ["auto", "design", "voice", "clone"].includes(it.kind);
    return `
      <div class="item" data-id="${esc(it.id)}">
        <div class="head">
          <span class="badge">${esc(KIND[it.kind] || it.kind)}</span>
          <span class="title" title="${esc(it.title)}">${esc(it.title)}</span>
          <span class="muted small">${new Date(it.created_at * 1000).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" })}</span>
        </div>
        ${media}
        <div class="actions">
          <span class="links small">${links.join("")}</span>
          ${canSave ? `<button class="small ghost save">Salvar como voz</button>` : ""}
          <button class="small ghost danger delete">Excluir</button>
        </div>
      </div>`;
  }).join("");
  $$(".item", list).forEach((el) => {
    $(".save", el)?.addEventListener("click", () => {
      const box = document.createElement("div");
      el.append(box);
      saveHistoryAsVoice(el.dataset.id, box);
    });
    $(".delete", el).onclick = async () => {
      if (!confirm("Excluir este item e seus arquivos?")) return;
      await api(`/api/history/${el.dataset.id}`, { method: "DELETE" });
      el.remove();
    };
  });
}

// ── boot ──────────────────────────────────────────────────────────────

(async () => {
  const s = await refreshStatus();
  if (s) fillSelects(s);
  await loadVoices().catch(() => {});
  let tab = "speak";
  try { tab = localStorage.getItem("tab") || tab; } catch {}
  showTab(tab);
  setInterval(refreshStatus, 15000);
})();
