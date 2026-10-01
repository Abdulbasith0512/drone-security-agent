"use strict";
/* Drone Security Analyst console. Presentation only: it fetches what the backend computed, lays it
   out and cross-links it. Every number, time, description, alert and answer on the page comes from
   the API; nothing here decides anything. */

const $ = (sel, root = document) => root.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";
/* One picture size for everything. The server decodes a video frame by reading the video up to it,
   and caches the result per (frame, size), so a second size would decode every frame twice. */
const FRAME_W = 640;

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "style") el.style.cssText = v;   // CSSOM: allowed under the strict CSP
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}

/* icons are drawn with the DOM: the CSP allows no inline images and no inline styles */
const ICONS = {
  video: ["M3 6h12v12H3z", "M15 10l6-3v10l-6-3"],
  observations: ["M4 8V5a1 1 0 0 1 1-1h3", "M16 4h3a1 1 0 0 1 1 1v3", "M20 16v3a1 1 0 0 1-1 1h-3", "M8 20H5a1 1 0 0 1-1-1v-3", "M8 12h8"],
  events: ["o:12,12,9", "M12 7v5l3 2"],
  entities: ["o:12,8,4", "M4 21c0-4 3.6-7 8-7s8 3 8 7"],
  rules: ["M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z", "M8.5 12l2.5 2.5 4.5-5"],
  alerts: ["M6 16v-5a6 6 0 1 1 12 0v5l2 2H4z", "M10 21h4"],
  analyst: ["M4 5h16v11H9l-5 4z", "M8 9h8", "M8 12h5"],
  check: ["o:12,12,10", "M8 12l3 3 5-6"],
  generic: ["o:12,12,9"],
};
function icon(name) {
  const root = svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true", focusable: "false" });
  for (const part of ICONS[name] || ICONS.generic) {
    if (part.startsWith("o:")) {
      const [cx, cy, r] = part.slice(2).split(",");
      root.append(svg("circle", { cx, cy, r }));
    } else root.append(svg("path", { d: part }));
  }
  return root;
}

/* Pictures are fetched through a short queue: the viewer's frame first, thumbnails only when they
   scroll into view, two at a time. A burst of requests would make the server decode them all at
   once and starve the one the user is looking at. */
const loader = { queue: [], active: 0, max: 2, observer: null };
function resetLoader() {
  loader.queue = [];
  if (loader.observer) loader.observer.disconnect();
  loader.observer = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        loader.observer.unobserve(entry.target);
        loadFrame(entry.target, Number(entry.target.dataset.frame), false);
      }
    }, { rootMargin: "0px 300px" })
    : null;
}
function loadFrame(img, frameId, urgent) {
  const job = { img, src: frameUrl(frameId, FRAME_W) };
  if (urgent) loader.queue.unshift(job); else loader.queue.push(job);
  setTimeout(pump, 0);   // after the element has been put into the page
}
function pump() {
  while (loader.active < loader.max && loader.queue.length) {
    const job = loader.queue.shift();
    if (!job.img.isConnected) continue;   // replaced meanwhile: never requested
    loader.active++;
    const done = () => { loader.active = Math.max(0, loader.active - 1); pump(); };
    job.img.addEventListener("load", done, { once: true });
    job.img.addEventListener("error", done, { once: true });
    job.img.src = job.src;
  }
}

const state = {
  token: 0, online: null, runs: [], run: null,
  overview: null, events: null, alerts: null, entities: null, observations: null, rules: null,
  errors: {}, sessionId: null, idx: null,
  sel: { obs: null, event: null, alert: null, origin: "obs" },
  hlEvents: new Set(), hlObs: new Set(),
  eventDetail: new Map(), alertDetail: new Map(), expanded: null, aboutOpen: false,
  zoneFilter: "", typeFilter: "",
};

/* ---------- formatting helpers ---------- */
const pretty = (s) => String(s).replace(/_/g, " ");
const titleCase = (s) => pretty(s).replace(/\b\w/g, (c) => c.toUpperCase());
const fmtTime = (iso) => (iso ? iso.slice(0, 19).replace("T", " ") : "");
const fmtClock = (iso) => (iso ? iso.slice(11, 19) : "");
const refId = (ref) => ref.slice(ref.indexOf(":") + 1);
const refKind = (ref) => ref.slice(0, ref.indexOf(":"));
const subject = (attrs) => Object.entries(attrs || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
function fmtDur(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}
function fmtVideoTime(seconds) {   // m:ss.t
  const tenths = Math.max(0, Math.round(seconds * 10));
  const m = Math.floor(tenths / 600);
  return `${m}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}
function localMs(iso) {   // epoch shifted by the timestamp's own UTC offset, for axis labels
  const m = /([+-])(\d\d):(\d\d)$/.exec(iso);
  const off = m ? (m[1] === "-" ? -1 : 1) * (Number(m[2]) * 60 + Number(m[3])) : 0;
  return Date.parse(iso) + off * 60000;
}
const clockLabel = (ms, withSeconds) => new Date(ms).toISOString().slice(11, withSeconds ? 19 : 16);
function objectSummary(objects) {
  const counts = {};
  for (const o of objects) counts[o.type] = (counts[o.type] || 0) + 1;
  const parts = Object.entries(counts).map(([t, n]) => `${n} ${t}`);
  return parts.length ? parts.join(" · ") : "no objects";
}

/* ---------- network ---------- */
async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}
const runUrl = (name, suffix) => `/api/runs/${encodeURIComponent(name)}${suffix}`;
const runApi = (suffix) => api(runUrl(state.run.name, suffix));
const frameUrl = (frameId, w) => `${runUrl(state.run.name, `/frames/${frameId}.jpg`)}?w=${w}`;

function showError(err) {
  const box = $("#error");
  box.hidden = false;
  box.textContent = `Error: ${err.message || err}`;
}
function clearError() { $("#error").hidden = true; }

/* ---------- derived data, all from the API payloads ---------- */
function buildIndex() {
  const obs = state.observations ? state.observations.observations : [];
  const events = state.events ? state.events.events : [];
  const entities = state.entities ? state.entities.entities : [];
  const alerts = state.alerts ? state.alerts.alerts : [];
  const entityType = new Map();   // entity ref -> the type of the events linked to it
  for (const e of events) if (e.entity_ref && !entityType.has(e.entity_ref)) entityType.set(e.entity_ref, e.type);
  state.idx = {
    obs, events, entities, alerts, entityType,
    obsById: new Map(obs.map((o) => [o.observation_id, o])),
    obsPos: new Map(obs.map((o, i) => [o.observation_id, i])),
    eventById: new Map(events.map((e) => [e.event_id, e])),
    entityByRef: new Map(entities.map((e) => [e.ref, e])),
    alertByRef: new Map(alerts.map((a) => [a.ref, a])),
  };
}
const provenances = () => [...new Set(state.idx.obs.map((o) => o.provenance).filter(Boolean))];
const PROVENANCE_LABEL = { replayed: "VLM replay", canned: "Canned replay", scripted: "Scripted text" };
const storedOutputs = () => { const p = provenances(); return p.length > 0 && p.every((x) => x === "replayed" || x === "canned"); };
const vlmRun = () => { const p = provenances(); return p.length > 0 && p.every((x) => x === "replayed"); };
const simulatedClock = () => !!state.overview && state.overview.limitations.some((t) => /simulated/i.test(t));
const curObs = () => (state.idx && state.sel.obs ? state.idx.obsById.get(state.sel.obs) : null);
const curEvent = () => (state.idx && state.sel.event ? state.idx.eventById.get(state.sel.event) : null);
function ensureVisible(el) {
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const bar = $(".top");   // only a sticky header covers the page: on tablets and phones it scrolls away
  const top = (bar && getComputedStyle(bar).position === "sticky" ? bar.offsetHeight : 0) + 8;
  if (rect.top < top || rect.bottom > window.innerHeight) el.scrollIntoView({ behavior: "smooth", block: "start" });
}
function goPanel(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

/* ---------- small components ---------- */
const sevBadge = (sev) => h("span", { class: `badge sev sev-${sev}`, text: sev });
const statusBadge = (status) => h("span", { class: `badge status status-${status}`, text: status });
const typeLabel = (t) => h("span", { class: "typelabel" }, h("i", { class: `shape ${t}`, "aria-hidden": "true" }), t);
function chip(ref, label) {
  return h("button", { class: "chip", type: "button", title: `Open ${ref}`, onclick: () => openRef(ref) }, label || ref);
}
function sourceBadges(o, compact) {
  const origin = state.run.description_origin;
  return [
    o.carried_forward
      ? h("span", { class: "badge carried", title: "A repeat of the last accepted description for a similar frame. It is NOT a new inference; no description was produced for this frame." }, compact ? "CARRIED" : "CARRIED FORWARD")
      : h("span", { class: "badge fresh", title: `New description for this frame: ${origin}` }, compact ? "NEW" : "NEW OBSERVATION"),
    compact ? null : " ",
    compact ? null : h("span", { class: "badge src", title: "The source label stored with the observation" }, o.source),
  ];
}
function panelError(target, what) {
  target.className = "";
  target.replaceChildren(h("div", { class: "panel-error", role: "alert" },
    h("p", { text: `Could not load ${what}: ${state.errors[what] || "unknown error"}` }),
    h("button", { class: "btn small", type: "button", text: "Retry", onclick: () => loadRun(state.run.name) })));
}
function empty(target, text) {
  target.className = "";
  target.replaceChildren(h("div", { class: "empty", text }));
}
function thumb(o, cls) {
  const video = state.run.video;
  const img = h("img", { alt: "", decoding: "async", dataset: { frame: o.frame_id },
    width: FRAME_W, height: video ? Math.round((FRAME_W * video.height) / video.width) : null });
  img.addEventListener("error", () => img.replaceWith(h("div", { class: "nopic small muted", text: "frame unavailable" })));
  if (loader.observer) loader.observer.observe(img); else loadFrame(img, o.frame_id, false);
  return cls ? h("div", { class: cls }, img) : img;
}

/* ---------- boot ---------- */
async function init() {
  try {
    state.runs = (await api("/api/runs")).runs;
    state.online = true;
    const select = $("#run-select");
    const option = (r) => h("option", { value: r.name, text: `${titleCase(r.name)} · ${r.label}` });
    const [primary, ...others] = state.runs;
    select.replaceChildren(
      h("optgroup", { label: "Primary demo" }, option(primary)),
      others.length ? h("optgroup", { label: "Secondary runs" }, ...others.map(option)) : null);
    select.addEventListener("change", () => loadRun(select.value));
    const wanted = new URLSearchParams(location.hash.slice(1)).get("run");
    const start = state.runs.find((r) => r.name === wanted) || primary;
    select.value = start.name;
    window.addEventListener("hashchange", () => {   // back/forward or an edited address opens that run
      const name = new URLSearchParams(location.hash.slice(1)).get("run");
      if (name && state.run && name !== state.run.name && state.runs.some((r) => r.name === name)) loadRun(name);
    });
    await loadRun(start.name);
  } catch (err) { state.online = false; renderStatusChips(); showError(err); }
}

function setResult(key, result) {
  if (result.status === "fulfilled") state[key] = result.value;
  else state.errors[key] = (result.reason && result.reason.message) || String(result.reason);
}

async function loadRun(name) {
  const token = ++state.token;
  clearError();
  Object.assign(state, {
    run: state.runs.find((r) => r.name === name),
    overview: null, events: null, alerts: null, entities: null, observations: null, rules: null,
    errors: {}, sessionId: null, idx: null, expanded: null, aboutOpen: false, zoneFilter: "", typeFilter: "",
    sel: { obs: null, event: null, alert: null, origin: "obs" }, hlEvents: new Set(), hlObs: new Set(),
    eventDetail: new Map(), alertDetail: new Map(),
  });
  history.replaceState(null, "", `#run=${encodeURIComponent(name)}`);
  $("#run-select").value = name;
  resetLoader();
  renderSkeletons();
  renderStatusChips();
  const jobs = [["overview", "/overview"], ["events", "/events"], ["alerts", "/alerts"], ["entities", "/entities"],
    ["observations", "/observations?limit=200"]];
  const settled = await Promise.allSettled([...jobs.map(([, suffix]) => api(runUrl(name, suffix))), api("/api/rules")]);
  if (token !== state.token) return;   // the user switched runs while this one was loading
  jobs.forEach(([key], i) => setResult(key, settled[i]));
  setResult("rules", settled[jobs.length]);
  state.online = settled.some((r) => r.status === "fulfilled");
  if (state.overview) state.run = state.overview.run;
  buildIndex();
  renderAll();
  if (state.idx.obs.length) selectObs(state.idx.obs[0].observation_id);
}

function renderSkeletons() {
  const skel = (...heights) => heights.map((px) => h("div", { class: "skel", style: `height:${px}px;margin-top:10px` }));
  const fill = (sel, ...heights) => { const el = $(sel); el.className = ""; el.setAttribute("aria-busy", "true"); el.replaceChildren(...skel(...heights)); };
  fill("#video-body", 320, 34);
  fill("#investigation-body", 90, 120, 90);
  fill("#timeline-body", 120, 160);
  fill("#alerts-body", 90);
  fill("#overview-body", 90, 90);
  fill("#analyst-body", 180);
  $("#kpi-strip").replaceChildren(...[0, 1, 2, 3, 4].map(() => h("div", { class: "skel", style: "height:74px" })));
  $("#strip").replaceChildren(...[0, 1, 2, 3].map(() => h("div", { class: "skel", style: "flex:0 0 210px;height:150px" })));
  $("#pipeline").replaceChildren(...[0, 1, 2, 3, 4, 5, 6].map(() => h("div", { class: "skel", style: "height:110px" })));
  $("#kind-banner").replaceChildren();
}

function renderAll() {
  ["#video-body", "#investigation-body", "#timeline-body", "#alerts-body", "#overview-body", "#analyst-body"]
    .forEach((sel) => $(sel).removeAttribute("aria-busy"));
  renderStatusChips(); renderBanner(); renderVideo(); renderKpis(); renderStrip(); renderInvestigation();
  renderTimeline(); renderAlerts(); renderEntities(); renderAnalyst(); renderPipeline();
}

/* ---------- header chips and the data-kind banner ---------- */
function renderStatusChips() {
  const chipEl = (cls, text, title) => h("li", { class: `chip-status ${cls}`, title }, cls === "ok" || cls === "err" ? h("i", { class: "dot", "aria-hidden": "true" }) : null, text);
  const chips = [state.online === false
    ? chipEl("err", "API offline", "The dashboard could not reach its API")
    : chipEl("ok", "System online", "The dashboard is reading its run databases through the API")];
  if (state.run) {
    chips.push(chipEl("info", state.run.label, state.run.detail));
    if (state.idx) {
      for (const p of provenances()) chips.push(chipEl("ai", PROVENANCE_LABEL[p] || p, state.run.description_origin));
      if (storedOutputs()) chips.push(chipEl("info", "Reproducible", "Read back from stored output: no model is called now, so the same run gives the same data"));
    }
  }
  $("#status-chips").replaceChildren(...chips);
}

function renderBanner() {
  const r = state.run, box = $("#kind-banner");
  box.className = `banner k-${r.kind}`;
  const span = state.overview && state.overview.observation_span;
  const context = [titleCase(r.name),
    span ? `${fmtTime(span.first)} → ${fmtClock(span.last)} (${span.first.slice(19)}${simulatedClock() ? ", simulated clock" : ""})` : null,
    r.video ? `${r.video.name} · ${r.video.width}×${r.video.height} · ${r.video.fps} fps · ${r.video.duration_seconds}s` : "no video: text frames"]
    .filter(Boolean).join("  ·  ");
  const parts = [h("span", { class: "kind", text: r.label }),
    h("div", { class: "text" }, h("strong", { text: r.detail }), h("span", { class: "context mono", text: context })),
    h("button", { class: "btn small", type: "button", "aria-expanded": String(state.aboutOpen), "aria-controls": "about-more",
      text: state.aboutOpen ? "Hide details" : "About this data", onclick: () => { state.aboutOpen = !state.aboutOpen; renderBanner(); } })];
  if (state.aboutOpen && state.overview) {
    const o = state.overview;
    parts.push(h("div", { class: "more", id: "about-more" },
      h("div", { class: "facts" },
        h("span", { text: `Zones: ${o.zones.map((z) => `${pretty(z.zone)} ${z.observations}`).join(", ") || "none"}` }),
        h("span", { text: `Objects: ${Object.entries(o.object_counts).map(([k, n]) => `${k} ${n}`).join(", ") || "none"}` }),
        h("span", { text: `Sources: ${o.sources.map((s) => `${s.source} ${s.count} (${s.carried_forward ? "carried forward" : "new"})`).join(", ")}` })),
      h("ul", {}, ...o.limitations.map((t) => h("li", { text: t })))));
  }
  box.replaceChildren(...parts);
}

/* ---------- video viewer ---------- */
function renderVideo() {
  const body = $("#video-body");
  if (state.errors.observations) { panelError(body, "observations"); return; }
  const list = state.idx.obs, r = state.run;
  if (!list.length) { empty(body, "No observations are stored for this run."); return; }
  body.className = "";
  const viewer = h("div", { class: "viewer", id: "viewer" });
  if (r.video) viewer.style.aspectRatio = `${r.video.width} / ${r.video.height}`;
  const position = (o, i) => (r.video
    ? (o.frame_id / Math.max(1, r.video.frame_count - 1)) * 100
    : (list.length > 1 ? (i / (list.length - 1)) * 100 : 50));
  const marks = list.map((o, i) => h("button", {
    class: `mark${o.carried_forward ? " carried" : ""}`, type: "button", "data-obs": o.observation_id,
    style: `left:${position(o, i).toFixed(2)}%`, onclick: () => selectObs(o.observation_id),
    title: `${fmtTime(o.time)} · frame ${o.frame_id}${o.carried_forward ? " · carried forward" : ""}`,
    "aria-label": `Observation ${i + 1} of ${list.length}, frame ${o.frame_id}${r.video ? `, ${fmtVideoTime(o.frame_id / r.video.fps)}` : ""}${o.carried_forward ? ", carried forward" : ""}`,
  }));
  const scrub = h("div", { class: "scrub", role: "group", "aria-label": "Observation timeline", onkeydown: scrubKey },
    h("div", { class: "track" }), ...marks);
  const ends = h("div", { class: "scrub-ends", "aria-hidden": "true" },
    h("span", { text: r.video ? fmtVideoTime(0) : fmtClock(list[0].time) }),
    h("span", { text: r.video ? fmtVideoTime(r.video.duration_seconds) : fmtClock(list[list.length - 1].time) }));
  const transport = h("div", { class: "transport" },
    h("button", { class: "btn", type: "button", text: "◀ Previous", "aria-label": "Previous observation", onclick: () => stepObs(-1) }),
    h("span", { class: "where", id: "where", "aria-live": "polite" }),
    h("button", { class: "btn", type: "button", text: "Next ▶", "aria-label": "Next observation", onclick: () => stepObs(1) }));
  body.replaceChildren(viewer, transport, scrub, ends);
}

function scrubKey(e) {
  const list = state.idx.obs;
  if (!list.length) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") stepObs(e.key === "ArrowLeft" ? -1 : 1);
  else if (e.key === "Home") selectObs(list[0].observation_id);
  else if (e.key === "End") selectObs(list[list.length - 1].observation_id);
  else return;
  e.preventDefault();
  const mark = $(`.mark[data-obs="${state.sel.obs}"]`);
  if (mark) mark.focus();
}

function stepObs(delta) {
  const list = state.idx.obs, i = state.idx.obsPos.get(state.sel.obs) ?? 0;
  const next = list[Math.max(0, Math.min(list.length - 1, i + delta))];
  if (next) selectObs(next.observation_id);
}

function renderViewer() {
  const viewer = $("#viewer"), o = curObs(), r = state.run;
  if (!viewer || !o) return;
  const ev = curEvent();
  const pill = (cls, text) => h("span", { class: `ovpill ${cls}`, text });
  const stage = [];
  if (o.image) {
    viewer.classList.add("loading-frame");
    const img = h("img", { alt: `Video frame ${o.frame_id}: ${o.description}`, decoding: "async" });
    img.addEventListener("load", () => { img.classList.add("ready"); viewer.classList.remove("loading-frame"); });
    img.addEventListener("error", () => {
      viewer.classList.remove("loading-frame");
      img.replaceWith(h("div", { class: "nopic", text: "This frame could not be loaded." }));
    });
    loadFrame(img, o.frame_id, true);
    stage.push(img);
  } else {
    viewer.classList.remove("loading-frame");
    stage.push(h("div", { class: "textframe" },
      h("p", { class: "small muted", text: "Text frame: this observation has a description but no pixels." }),
      h("p", { class: "desc", text: o.description }),
      h("p", { class: "small muted", text: objectSummary(o.objects) })));
  }
  const when = r.video ? `${fmtVideoTime(o.frame_id / r.video.fps)} / ${fmtVideoTime(r.video.duration_seconds)}` : fmtTime(o.time);
  const eventShown = ev && state.sel.origin !== "obs";
  stage.push(
    h("div", { class: "ov ov-tl" }, pill("", pretty(o.zone))),
    h("div", { class: "ov ov-tr" }, pill("", r.label),
      ...provenances().map((p) => pill("", PROVENANCE_LABEL[p] || p))),
    h("div", { class: "ov ov-bottom" },
      h("div", { class: "ovrow" }, pill("time", when), pill("time", `frame ${o.frame_id}`), o.carried_forward ? pill("carried", "Carried forward") : null),
      h("div", { class: "ovrow" }, eventShown ? pill("event", `Event detected · ${ev.event_id}`) : pill("", "Observation"))));
  viewer.replaceChildren(...stage);
  const i = state.idx.obsPos.get(o.observation_id);
  $("#where").textContent = `Observation ${i + 1} of ${state.idx.obs.length} · ${fmtTime(o.time)}`;
  viewer.setAttribute("aria-label", `Video frame ${o.frame_id}`);
}

/* ---------- KPI strip and observation strip ---------- */
function renderKpis() {
  const box = $("#kpi-strip");
  if (!state.overview) { box.replaceChildren(h("div", { class: "panel-error", text: `Totals unavailable: ${state.errors.overview || ""}` })); return; }
  const c = state.overview.counts, src = state.overview.sources;
  const fresh = src.filter((s) => !s.carried_forward).reduce((a, s) => a + s.count, 0);
  const carried = src.filter((s) => s.carried_forward).reduce((a, s) => a + s.count, 0);
  const vlm = state.idx && state.idx.obs.length && vlmRun();
  const items = [
    ["Observations", c.observations, `${c.detected_objects} objects`],
    ["Events", c.events, ""], ["Entities", c.entities, ""], ["Alerts", c.alerts, ""],
    ["VLM frames", vlm ? fresh : "n/a", vlm ? `${carried} carried forward` : "no vision model in this run"],
  ];
  box.replaceChildren(...items.map(([label, n, caption]) => h("div", { class: "kpi", "data-kpi": label.toLowerCase().replace(" ", "-") },
    h("div", { class: "l", text: label }), h("div", { class: "n", text: String(n) }), h("div", { class: "c", text: caption }))));
}

function renderStrip() {
  const strip = $("#strip");
  if (state.errors.observations) { strip.replaceChildren(h("div", { class: "panel-error", text: "Observations unavailable." })); return; }
  const list = state.idx.obs, o = state.observations;
  $("#strip-note").textContent = o ? `${plural(o.total, "observation")}${o.truncated ? ` (showing ${o.returned})` : ""}` : "";
  if (!list.length) { strip.replaceChildren(h("div", { class: "empty", text: "No observations." })); return; }
  strip.replaceChildren(...list.map((x, i) => h("button", {
    class: `scard${x.carried_forward ? " carried" : ""}`, type: "button", "data-obs": x.observation_id, onclick: () => selectObs(x.observation_id),
    "aria-label": `Observation ${i + 1}, frame ${x.frame_id}, ${x.carried_forward ? "carried forward" : "new"}: ${objectSummary(x.objects)}`,
  },
  x.image ? thumb(x, "pic") : null,
  h("div", { class: "meta" }, h("span", { class: "mono", text: fmtClock(x.time) }), sourceBadges(x, true)),
  h("div", { class: "objs", text: `${pretty(x.zone)} · ${objectSummary(x.objects)}` }),
  h("div", { class: "desc", text: x.description }))));
}

/* ---------- investigation ---------- */
function renderInvestigation() {
  const box = $("#investigation-body");
  if (state.errors.observations) { panelError(box, "observations"); return; }
  const o = curObs();
  if (!o) { empty(box, "Nothing to investigate: no observations are stored for this run."); return; }
  box.className = "";
  box.replaceChildren(observationSection(o), eventSection(o, curEvent()), evidenceSection(o, curEvent()));
}

function observationSection(o) {
  const video = state.run.video;
  return h("section", { class: "sec" }, h("h3", { text: "Current observation" }),
    h("dl", { class: "kv" },
      h("dt", { text: "Timestamp" }), h("dd", {}, h("span", { class: "mono", text: fmtTime(o.time) }), simulatedClock() ? [" ", h("span", { class: "badge sim", text: "SIMULATED" })] : null),
      h("dt", { text: "Zone" }), h("dd", { text: pretty(o.zone) }),
      video ? [h("dt", { text: "Video time" }), h("dd", { class: "mono", text: fmtVideoTime(o.frame_id / video.fps) })] : null),
    o.objects.length
      ? h("ul", { class: "objlist", "aria-label": "Objects" }, ...o.objects.map((x) => h("li", {}, typeLabel(x.type),
        h("span", { text: [x.activity, subject(x.attributes)].filter(Boolean).join(" · ") }))))
      : h("p", { class: "small muted", text: "No objects detected in this observation." }),
    h("p", { class: "desc-text", text: o.description }),
    o.telemetry_simulated ? h("p", { class: "small muted" }, h("span", { class: "badge sim", text: "SIMULATED telemetry" }),
      ` battery ${o.telemetry_simulated.battery_percent}%, altitude ${o.telemetry_simulated.altitude_m} m, sample age ${o.telemetry_simulated.age_seconds}s${o.telemetry_simulated.stale ? " (STALE)" : ""}`) : null);
}

function eventButtons(o, selectedId) {
  return h("div", { class: "eventpick" }, ...o.event_refs.map((ref) => {
    const ev = state.idx.eventById.get(refId(ref));
    return h("button", { type: "button", class: ev && ev.event_id === selectedId ? "sel" : "", "data-event": refId(ref),
      onclick: () => selectEvent(refId(ref), { scroll: false }) },
    ev ? [typeLabel(ev.type), h("span", { text: subject(ev.attributes) || "-" }), h("span", { class: "mono muted", text: ev.event_id })] : ref);
  }));
}

function eventSection(o, ev) {
  const sec = h("section", { class: "sec" }, h("h3", { text: "Current event" }));
  if (!ev) {
    sec.append(o.event_refs.length
      ? h("div", {}, h("p", { class: "small muted", text: `${plural(o.event_refs.length, "event")} in this observation. Select one to see its details.` }), eventButtons(o, null))
      : h("p", { class: "small muted", text: "No event is linked to this observation: nothing was detected in it." }));
    return sec;
  }
  const entity = ev.entity_ref ? state.idx.entityByRef.get(ev.entity_ref) : null;
  sec.append(
    h("p", {}, typeLabel(ev.type), " ", chip(ev.ref)),
    h("dl", { class: "kv" },
      h("dt", { text: "Start" }), h("dd", { class: "mono", text: fmtTime(ev.first_seen) }),
      h("dt", { text: "End" }), h("dd", { class: "mono", text: fmtTime(ev.last_seen) }),
      h("dt", { text: "Duration" }), h("dd", { text: ev.observation_count === 1 ? "single observation" : `${fmtDur(ev.duration_seconds)} · ${plural(ev.observation_count, "observation")}` }),
      h("dt", { text: "Subject" }), h("dd", { text: subject(ev.attributes) || "-" }),
      h("dt", { text: "Activity" }), h("dd", { text: ev.activities.join(", ") || "-" }),
      h("dt", { text: "Entity" }), h("dd", {}, entity ? chip(entity.ref, entity.label) : "none (people are never linked)"),
      h("dt", { text: "Closed because" }), h("dd", { text: pretty(ev.close_reason) })),
    ev.alert_refs.length
      ? h("div", { class: "sec" }, h("h3", { text: "Alerts" }), ...ev.alert_refs.map((ref) => {
        const a = state.idx.alertByRef.get(ref);
        return h("p", {}, a ? [sevBadge(a.severity), " "] : null, chip(ref, a ? a.title : ref));
      }))
      : h("p", { class: "neutral" }, icon("check"), "No security rule triggered for this event."));
  if (o.event_refs.length > 1) sec.append(h("p", { class: "small muted", text: "Other events in this observation:" }), eventButtons(o, ev.event_id));
  return sec;
}

function evidenceSection(o, ev) {
  const sec = h("section", { class: "sec" }, h("h3", { text: "Evidence" }),
    h("dl", { class: "kv" },
      h("dt", { text: "Frame" }), h("dd", {}, chip(o.frame_ref, `frame ${o.frame_id}`)),
      h("dt", { text: "Timestamp" }), h("dd", { class: "mono", text: fmtTime(o.time) }),
      h("dt", { text: "Source" }), h("dd", {}, sourceBadges(o))));
  if (o.carried_forward) sec.append(h("p", { class: "small muted", text: "Carried forward: this repeats the last accepted description. No new description was produced for this frame." }));
  if (ev) {
    const d = state.eventDetail.get(ev.event_id);
    if (!d) sec.append(h("div", { class: "skel", style: "height:70px;margin-top:10px" }));
    else if (d.error) sec.append(h("p", { class: "small muted", text: `Evidence for this event could not be loaded: ${d.error}` }));
    else sec.append(h("p", { class: "small muted", style: "margin-top:10px", text: `${plural(d.observations.length, "observation")} behind ${ev.event_id}` }),
      h("div", { class: "evthumbs" }, ...d.observations.map((x) => h("button", { type: "button", class: `evthumb${x.observation_id === o.observation_id ? " sel" : ""}`,
        "data-obs": x.observation_id, "aria-label": `Show frame ${x.frame_id}`, onclick: () => selectObs(x.observation_id) },
      x.image ? thumb(x) : null, h("span", { class: "mono", text: `frame ${x.frame_id}` })))));
  }
  return sec;
}

async function ensureEventDetail(id) {
  if (state.eventDetail.has(id)) return;
  const token = state.token;
  try {
    const d = await runApi(`/events/${encodeURIComponent(id)}`);
    if (token !== state.token) return;
    state.eventDetail.set(id, d);
  } catch (err) {
    if (token !== state.token) return;
    state.eventDetail.set(id, { error: err.message });
  }
  if (state.sel.event === id) renderInvestigation();
}

/* ---------- selection: one place decides what is selected and what is highlighted ---------- */
function selectObs(id, opts = {}) {
  const o = state.idx.obsById.get(id);
  if (!o) return;
  const s = state.sel;
  s.obs = id; s.origin = "obs"; s.alert = null;
  let ev = s.event ? state.idx.eventById.get(s.event) : null;
  if (ev && !ev.observation_ids.includes(id)) { s.event = null; ev = null; }
  state.hlObs = new Set(ev ? ev.observation_ids : [id]);
  state.hlEvents = new Set(ev ? [ev.event_id] : o.event_refs.map(refId));
  renderSelection();
  if (opts.scroll) ensureVisible($("#viewer"));
}

function selectEvent(id, opts = {}) {
  const ev = state.idx.eventById.get(id);
  if (!ev) return;
  const s = state.sel;
  s.event = id; s.origin = "event"; s.alert = null;
  const first = ev.observation_ids.find((x) => state.idx.obsById.has(x));
  if (first) s.obs = first;
  state.hlEvents = new Set([id]);
  state.hlObs = new Set(ev.observation_ids);
  renderSelection();
  ensureEventDetail(id);
  if (opts.scroll !== false) ensureVisible($("#viewer"));
}

function renderSelection() {
  renderViewer(); renderInvestigation(); applyHighlights();
  const card = $(`#strip [data-obs="${state.sel.obs}"]`), strip = $("#strip");
  if (card && strip) strip.scrollTo({ left: card.offsetLeft - (strip.clientWidth - card.offsetWidth) / 2, behavior: "smooth" });
}

function applyHighlights() {
  const { sel } = state;
  const mark = (el, selected, related) => {
    el.classList.toggle("sel", selected);
    el.classList.toggle("hl", related && !selected);
    if (selected) el.setAttribute("aria-current", "true"); else el.removeAttribute("aria-current");
  };
  document.querySelectorAll("[data-event]").forEach((el) => mark(el, sel.event === el.dataset.event, state.hlEvents.has(el.dataset.event)));
  document.querySelectorAll("[data-obs]").forEach((el) => mark(el, sel.obs === el.dataset.obs, state.hlObs.has(el.dataset.obs)));
  document.querySelectorAll("[data-alert]").forEach((el) => el.classList.toggle("sel", sel.alert === el.dataset.alert));
  document.querySelectorAll("[data-entity]").forEach((el) => {
    const e = state.idx.entityByRef.get(el.dataset.entity);
    el.classList.toggle("hl", !!(e && sel.event && e.event_refs.includes(`event:${sel.event}`)));
  });
  document.querySelectorAll("svg.timeline .bar.sel").forEach((el) => el.parentNode.append(el));   // draw the selection on top
}

function openRef(ref) {
  const kind = refKind(ref), id = refId(ref);
  if (kind === "event") selectEvent(id, { scroll: true });
  else if (kind === "alert") { selectAlert(id); goPanel("panel-alerts"); }
  else if (kind === "entity") {
    const el = document.querySelector(`[data-entity="entity:${id}"]`);
    if (el) { goPanel("panel-overview"); el.classList.add("hl"); setTimeout(applyHighlights, 1800); }
  } else if (kind === "frame") {
    const o = state.idx.obs.find((x) => String(x.frame_id) === id);
    if (o) selectObs(o.observation_id, { scroll: true });
  } else if (kind === "obs") selectObs(id, { scroll: true });
}

/* ---------- citations inside text ---------- */
function linkCitations(text, citations) {
  const tokens = [...new Set(citations.map((c) => c.token))].sort((a, b) => b.length - a.length);
  if (!tokens.length) return [text];
  const esc = tokens.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.join("|")})`, "g");
  return text.split(re).map((part) => {
    const c = citations.find((x) => x.token === part);
    return c ? h("button", { class: "chip", type: "button", title: `Open ${c.kind}:${c.id}`, onclick: () => openRef(`${c.kind}:${c.id}`) }, part) : part;
  });
}

/* ---------- event timeline ---------- */
function filteredEvents() {
  return state.events.events.filter((e) =>
    (!state.zoneFilter || e.zone === state.zoneFilter) && (!state.typeFilter || e.type === state.typeFilter));
}

function renderTimeline() {
  const body = $("#timeline-body");
  if (state.errors.events) { panelError(body, "events"); return; }
  const all = state.events.events;
  if (!all.length) { empty(body, "No events are stored for this run."); return; }
  const zones = [...new Set(all.map((e) => e.zone))], types = [...new Set(all.map((e) => e.type))];
  const select = (id, label, options, value, set) => h("label", {}, `${label} `,
    h("select", { id, onchange: (e) => { set(e.target.value); renderTimeline(); applyHighlights(); } },
      h("option", { value: "", text: "all" }), ...options.map((o) => h("option", { value: o, text: pretty(o), selected: o === value }))));
  const filters = h("div", { class: "filters" },
    select("f-zone", "Zone", zones, state.zoneFilter, (v) => { state.zoneFilter = v; }),
    select("f-type", "Type", types, state.typeFilter, (v) => { state.typeFilter = v; }),
    h("span", { text: `${filteredEvents().length} of ${all.length} events` }));
  body.className = "";
  body.replaceChildren(filters, h("div", { class: "tl-scroll" }, timelineSvg()), timelineLegend(), eventTable());
  applyHighlights();
}

function timelineSvg() {
  const events = filteredEvents();
  const zoneLanes = [...new Set(state.events.events.map((e) => e.zone))];
  // alerts raised from drone telemetry have no camera event, so they get their own lane
  const teleAlerts = (state.zoneFilter || state.typeFilter || !state.alerts) ? [] : state.alerts.alerts.filter((a) => !a.event_refs.length);
  const lanes = teleAlerts.length ? [...zoneLanes, "drone telemetry"] : zoneLanes;
  const span = state.events.span;
  const stamps = teleAlerts.map((a) => localMs(a.triggered_at));
  const t0 = Math.min(localMs(span.first), ...stamps), t1 = Math.max(localMs(span.last), ...stamps, t0 + 5000);
  const W = 1000, left = 118, right = 18, rowH = 17, barH = 11, pad = 8, axis = 28;
  const x = (iso) => left + ((localMs(iso) - t0) / (t1 - t0)) * (W - left - right);
  const barW = (ev) => Math.max(9, x(ev.last_seen) - x(ev.first_seen));
  // pack overlapping events of a lane into rows, so none is drawn on top of another
  const rowsOf = new Map(), rowCount = new Map();
  for (const zone of zoneLanes) {
    const ends = [];
    const laneEvents = events.filter((e) => e.zone === zone).sort((a, b) => localMs(a.first_seen) - localMs(b.first_seen));
    for (const ev of laneEvents) {
      const x0 = x(ev.first_seen);
      let row = ends.findIndex((end) => end + 4 <= x0);
      if (row < 0) { row = ends.length; ends.push(0); }
      ends[row] = x0 + barW(ev);
      rowsOf.set(ev.event_id, row);
    }
    rowCount.set(zone, Math.max(1, ends.length));
  }
  rowCount.set("drone telemetry", 1);
  const laneTop = []; let cursor = 6;
  for (const zone of lanes) { laneTop.push(cursor); cursor += rowCount.get(zone) * rowH + pad * 2; }
  const H = cursor + axis;
  const alertSev = new Map(state.alerts ? state.alerts.alerts.map((a) => [a.ref, a.severity]) : []);
  const root = svg("svg", { class: "timeline", viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Event timeline by zone" });
  lanes.forEach((zone, i) => {
    const y = laneTop[i], height = rowCount.get(zone) * rowH + pad * 2;
    root.append(svg("line", { class: "lane", x1: 0, x2: W, y1: y + height, y2: y + height }),
      svg("text", { x: 8, y: y + height / 2 + 4 }, pretty(zone)));
  });
  for (let k = 0; k <= 6; k++) {
    const ms = t0 + ((t1 - t0) * k) / 6, px = left + ((W - left - right) * k) / 6;
    root.append(svg("line", { class: "grid", x1: px, x2: px, y1: 4, y2: cursor }),
      svg("text", { x: px, y: H - 12, "text-anchor": k === 6 ? "end" : "middle" }, clockLabel(ms, t1 - t0 < 15 * 60000)));
  }
  root.append(svg("text", { x: 8, y: H - 12 }, simulatedClock() ? "simulated clock" : "clock"));
  const activate = (fn) => (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(); } };
  const diamond = (cx, cy, sev, ref, label) => {
    const m = svg("path", { class: `marker sev-${sev}`, d: `M ${cx} ${cy - 6} l 6 6 l -6 6 l -6 -6 z`, tabindex: 0, role: "button",
      "aria-label": `${label}, ${sev} alert` }, svg("title", {}, `${ref} (${sev}): ${label}`));
    m.addEventListener("click", () => openRef(ref));
    m.addEventListener("keydown", activate(() => openRef(ref)));
    return m;
  };
  for (const ev of events) {
    const lane = lanes.indexOf(ev.zone), y = laneTop[lane] + pad + rowsOf.get(ev.event_id) * rowH, x0 = x(ev.first_seen);
    const bar = svg("rect", { class: `bar ${ev.type}`, x: x0, y, width: barW(ev), height: barH, rx: ev.type === "person" ? barH / 2 : 2,
      "data-event": ev.event_id, tabindex: 0, role: "button", "aria-label": `${ev.type} in ${pretty(ev.zone)}, ${ev.event_id}, ${fmtClock(ev.first_seen)}` },
    svg("title", {}, `${ev.event_id}: ${ev.type}, ${subject(ev.attributes)}, ${ev.observation_count === 1 ? "single observation" : fmtDur(ev.duration_seconds)}`));
    bar.addEventListener("click", () => selectEvent(ev.event_id));
    bar.addEventListener("keydown", activate(() => selectEvent(ev.event_id)));
    root.append(bar);
    ev.alert_refs.forEach((ref, n) => { const sev = alertSev.get(ref); if (sev) root.append(diamond(x0 + n * 9, y + barH / 2, sev, ref, ref)); });
  }
  teleAlerts.forEach((a) => root.append(diamond(x(a.triggered_at), laneTop[lanes.length - 1] + pad + barH / 2, a.severity, a.ref, a.title)));
  return root;
}

function timelineLegend() {
  return h("div", { class: "legend-line" },
    h("span", { class: "typelabel" }, h("i", { class: "shape person", "aria-hidden": "true" }), "person (rounded bar)"),
    h("span", { class: "typelabel" }, h("i", { class: "shape vehicle", "aria-hidden": "true" }), "vehicle (square bar)"),
    h("span", { text: "◆ alert, coloured and labelled by severity" }),
    h("span", { text: "Select a bar or a row: the viewer jumps to its frame." }));
}

function eventTable() {
  const rows = filteredEvents().map((ev) => {
    const entity = ev.entity_ref ? state.idx.entityByRef.get(ev.entity_ref) : null;
    return h("tr", { class: "clickable", "data-event": ev.event_id, onclick: () => selectEvent(ev.event_id) },
      h("td", { class: "mono", text: fmtClock(ev.first_seen) }),
      h("td", {}, h("button", { class: "rowbtn", type: "button", onclick: (e) => { e.stopPropagation(); selectEvent(ev.event_id); } },
        typeLabel(ev.type), " ", subject(ev.attributes) || "-"), h("div", { class: "mono muted", text: ev.event_id })),
      h("td", { text: pretty(ev.zone) }),
      h("td", { text: entity ? entity.label : "-" }),
      h("td", { text: ev.observation_count === 1 ? "1 observation" : `${fmtDur(ev.duration_seconds)} · ${ev.observation_count} obs` }),
      h("td", {}, ...ev.alert_refs.map((ref) => { const a = state.idx.alertByRef.get(ref); return h("div", {}, a ? [sevBadge(a.severity), " "] : null, chip(ref, refId(ref))); })));
  });
  return h("div", { class: "tbl-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ...["Time", "Event", "Zone", "Entity", "Duration", "Alerts"].map((t) => h("th", { text: t })))),
    h("tbody", {}, ...rows)));
}

/* ---------- entity memory ---------- */
function renderEntities() {
  const body = $("#overview-body");
  if (state.errors.entities) { panelError(body, "entities"); return; }
  const list = state.idx.entities;
  if (!list.length) { empty(body, "No vehicle entities in this run."); return; }
  body.className = "";
  body.replaceChildren(h("div", { class: "entity-list", role: "list" }, ...list.map((e) => {
    const type = state.idx.entityType.get(e.ref);
    const first = e.event_refs[0];
    return h("div", { class: "entity", role: "listitem", "data-entity": e.ref },
      h("div", { class: "top-row" },
        h("button", { class: "name", type: "button", title: first ? `Open ${first}` : null, onclick: () => first && openRef(first), text: e.label }),
        type ? typeLabel(type) : null),
      h("div", { class: "small muted" }, h("span", { class: "mono", text: e.entity_id }), ` · ${plural(e.appearance_count, "appearance")}`),
      h("div", { class: "small muted", text: `First seen ${fmtTime(e.first_seen)} · last seen ${fmtTime(e.last_seen)}` }),
      h("div", { class: "small" }, h("span", { class: "badge sim", text: e.identity_confirmed ? "identity supported" : "coarse match, identity NOT confirmed" })),
      h("div", {}, ...e.event_refs.map((r) => chip(r))));
  })));
}

/* ---------- alert center ---------- */
function renderAlerts() {
  const body = $("#alerts-body");
  if (state.errors.alerts) { panelError(body, "alerts"); return; }
  const list = state.idx.alerts;
  const rules = state.rules ? h("details", { class: "rules" }, h("summary", { text: `Security rules in force (${state.rules.rules.length})` }),
    h("p", { class: "small muted", text: state.rules.note }),
    ...state.rules.rules.map((r) => h("div", { class: "rule" },
      h("div", {}, h("code", { text: r.id }), " ", sevBadge(r.severity), " ", h("span", { class: "badge sim", text: r.enabled ? "enabled" : "disabled" }),
        ` ${plural(state.overview ? state.overview.rule_breakdown[r.id] || 0 : 0, "alert")} in this run`),
      h("div", { class: "small", text: r.description }),
      h("pre", { text: JSON.stringify(r.settings, null, 1) })))) : null;
  body.className = "";
  if (!list.length) {
    body.replaceChildren(h("div", { class: "zero" }, icon("check"),
      h("div", {}, h("div", { class: "n", text: "0 ALERTS" }),
        h("p", { text: `No security rules triggered in this ${storedOutputs() ? "replay" : "run"}.` }),
        state.rules ? h("p", { class: "small muted", text: `${plural(state.rules.rules.length, "security rule")} in force; none raised an alert. That is a result, not an error.` }) : null)),
    rules);
    return;
  }
  body.replaceChildren(h("div", { class: "alert-list" }, ...list.map(alertCard)), rules);
}

function alertCard(a) {
  const open = state.expanded === a.alert_id;
  const entity = a.entity_ref ? state.idx.entityByRef.get(a.entity_ref) : null;
  return h("article", { class: `alert-card sev-${a.severity}${state.sel.alert === a.alert_id ? " sel" : ""}`, "data-alert": a.alert_id },
    h("button", { class: "alert-head", type: "button", "aria-expanded": String(open), "aria-controls": `ad-${a.alert_id}`, onclick: () => toggleAlert(a.alert_id) },
      h("div", { class: "t", text: a.title }),
      h("div", { class: "row" }, sevBadge(a.severity), a.status ? statusBadge(a.status) : null, h("code", { text: a.rule_id })),
      h("div", { class: "row small muted" }, h("span", { text: pretty(a.zone || "-") }), h("span", { class: "mono", text: fmtTime(a.triggered_at) }),
        entity ? h("span", { text: entity.label }) : null)),
    h("div", { class: "alert-actions" },
      h("button", { class: "btn small primary", type: "button", onclick: () => selectAlert(a.alert_id, { scroll: true }) }, "View evidence"),
      a.entity_ref ? chip(a.entity_ref, refId(a.entity_ref)) : null),
    open ? h("div", { class: "alert-detail", id: `ad-${a.alert_id}` }, alertDetailView(state.alertDetail.get(a.alert_id))) : null);
}

function toggleAlert(id) {
  if (state.expanded === id) { state.expanded = null; renderAlerts(); applyHighlights(); return; }
  selectAlert(id);
}

function alertDetailView(d) {
  if (!d) return h("p", { class: "muted", text: "Loading evidence…" });
  if (d.error) return h("p", { class: "muted", text: `Evidence could not be loaded: ${d.error}` });
  const a = d.alert;
  return h("div", {},
    h("dl", { class: "kv" },
      h("dt", { text: "Rule" }), h("dd", {}, h("code", { text: a.rule_id })),
      h("dt", { text: "Severity" }), h("dd", { text: `${a.severity} (fixed by the rule, never chosen by a model)` }),
      h("dt", { text: "Triggered" }), h("dd", { class: "mono", text: fmtTime(a.triggered_at) }),
      // lifecycle fields exist only on a backend that has them; shown only when present
      a.status ? [h("dt", { text: "Status" }), h("dd", {}, statusBadge(a.status), a.resolved_at ? ` resolved ${fmtTime(a.resolved_at)}` : "")] : null,
      a.verdict ? [h("dt", { text: "Verdict" }), h("dd", { text: `${a.verdict}${a.confidence != null ? ` (rule-derived confidence ${Number(a.confidence).toFixed(2)})` : ""}` })] : null,
      h("dt", { text: "Events" }), h("dd", {}, a.event_refs.length ? a.event_refs.map((r) => chip(r)) : "none (raised from drone telemetry)"),
      h("dt", { text: "Frames" }), h("dd", {}, a.frame_ids.length ? a.frame_ids.map((f) => chip(`frame:${f}`, String(f))) : "none")),
    h("h4", { text: "Why it was raised" }), h("p", { text: a.reason }),
    a.recommended_action ? [h("h4", { text: "Recommended action" }), h("p", {}, a.recommended_action,
      h("span", { class: "small muted", text: " (a fixed table entry for this rule; not written by a model)" }))] : null,
    a.ladder && a.ladder.length > 1 ? [h("h4", { text: "Escalation ladder" }),
      h("p", {}, ...a.ladder.flatMap((s, i) => [i ? " → " : "", sevBadge(s.severity), ` ${fmtTime(s.reached_at)}`]))] : null,
    a.history && a.history.length ? [h("h4", { text: "Lifecycle history" }),
      h("ul", { class: "small" }, ...a.history.map((x) => h("li", { text: `${fmtTime(x.effective_at)} · ${x.kind} · ${x.to_severity}/${x.to_status} · ${x.actor}: ${x.reason}` })))] : null,
    d.telemetry && d.telemetry.length ? [h("h4", {}, "Drone telemetry behind the alert ", h("span", { class: "badge sim", text: "SIMULATED" })),
      h("div", { class: "tbl-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ...["Time", "Battery", "Signal", "Altitude", "Zone"].map((t) => h("th", { text: t })))),
        h("tbody", {}, ...d.telemetry.map((t) => h("tr", {},
          h("td", { class: "mono", text: fmtTime(t.time) }), h("td", { text: `${t.battery_percent}%` }),
          h("td", { text: t.signal_percent == null ? "not reported" : `${t.signal_percent}%` }),
          h("td", { text: `${t.altitude_m} m` }), h("td", { text: pretty(t.zone) }))))))] : null,
    h("h4", { text: "Numbers the rule compared" }),
    d.facts.length ? h("dl", { class: "kv" }, ...d.facts.flatMap((f) => [h("dt", { text: f.name }), h("dd", { class: "mono", text: String(f.value) })])) : h("p", { class: "muted", text: "none" }),
    h("h4", { text: "Evidence chain" }), h("ol", {}, ...d.steps.map((s) => h("li", { text: s }))),
    ...d.evidence.map((x) => h("div", {}, h("h4", { text: `${x.event_ref} · ${plural(x.observations.length, "observation")}` }),
      h("div", { class: "evthumbs" }, ...x.observations.map((o) => h("button", { type: "button", class: "evthumb", "data-obs": o.observation_id,
        "aria-label": `Show frame ${o.frame_id}`, onclick: () => selectObs(o.observation_id, { scroll: true }) },
      o.image ? thumb(o) : null, h("span", { class: "mono", text: `frame ${o.frame_id}` })))))),
    h("p", { style: "margin-top:12px" }, h("button", { class: "btn small primary", type: "button", onclick: () => askAbout(`Show me the evidence for ${a.alert_id}`) }, "Ask the analyst about this alert")));
}

async function selectAlert(id, opts = {}) {
  const token = state.token;
  Object.assign(state.sel, { alert: id, origin: "alert" });
  state.expanded = id;
  renderAlerts(); applyHighlights();
  if (!state.alertDetail.has(id)) {
    try { state.alertDetail.set(id, await runApi(`/alerts/${encodeURIComponent(id)}`)); }
    catch (err) { state.alertDetail.set(id, { error: err.message }); }
    if (token !== state.token) return;
  }
  const d = state.alertDetail.get(id);
  if (state.sel.alert !== id) return;
  if (!d.error) {
    const eventIds = d.events.map((e) => refId(e.ref));
    state.hlEvents = new Set(eventIds);
    state.hlObs = new Set(d.evidence.flatMap((x) => x.observations.map((o) => o.observation_id)));
    const event = eventIds.find((x) => state.idx.eventById.has(x)) || null;
    const obs = [...state.hlObs].find((x) => state.idx.obsById.has(x));
    state.sel.event = event;
    if (obs) state.sel.obs = obs;
    if (event) ensureEventDetail(event);
  }
  renderAlerts(); renderViewer(); renderInvestigation(); applyHighlights();
  if (opts.scroll) ensureVisible($("#viewer"));
}

/* ---------- analyst ---------- */
function renderAnalyst() {
  const body = $("#analyst-body");
  if (!state.overview) { panelError(body, "overview"); return; }
  const example = state.overview.suggested_questions[0] || "Summarize the run.";   // from the backend
  const log = h("div", { class: "chat-log", id: "chat-log", role: "log", "aria-live": "polite", "aria-label": "Conversation with the analyst" },
    h("div", { class: "msg bot" }, h("div", { class: "part" }, h("span", { class: "tag", text: "Analyst" }),
      h("div", { class: "body", text: "Ask about this run. Answers cite stored evidence as clickable references. I do not identify people, infer intent or call anyone a criminal." }))));
  const input = h("textarea", { id: "chat-q", rows: 1, placeholder: "Ask the security analyst...", "aria-label": `Your question, for example: ${example}` });
  const send = h("button", { class: "btn primary", id: "chat-send", type: "button", onclick: () => ask(input.value) }, "Ask");
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); } });
  body.className = "";
  body.replaceChildren(
    h("div", { class: "analyst-head" }, h("h2", { id: "h-analyst", text: "AI Security Analyst" }),
      h("span", { class: "chip-status ai", text: "Grounded in run evidence" }),
      h("p", { class: "sub", text: "Grounded answers from observed evidence" })),
    h("div", { class: "analyst" },
      h("div", {}, log, h("div", { class: "chat-input" }, input, send)),
      h("div", { class: "asides" },
        h("div", {}, h("h3", { id: "h-suggest", text: "Try asking" }), h("div", { class: "suggestions", id: "suggestions", role: "group", "aria-labelledby": "h-suggest" })),
        h("div", {}, h("h3", { text: "Run summary" }),
          h("p", { class: "small muted", text: "A deterministic summary of the stored run." }),
          h("button", { class: "btn small", type: "button", onclick: loadSummary }, "Show run summary"), h("div", { id: "summary-box" })))));
  setSuggestions(state.overview.suggested_questions, "Try asking");
}

function setSuggestions(list, lead) {
  $("#h-suggest").textContent = lead;
  $("#suggestions").replaceChildren(...list.map((q) => h("button", { type: "button", text: q, onclick: () => ask(q) })));
}

async function loadSummary() {
  const box = $("#summary-box"), token = state.token;
  box.replaceChildren(h("p", { class: "muted small", text: "Loading…" }));
  try {
    const s = await runApi("/summary");
    if (token !== state.token) return;
    box.replaceChildren(h("pre", { class: "summary" }, ...linkCitations(s.text, s.citations)), h("p", { class: "small muted", text: s.label }));
  } catch (err) { if (token === state.token) box.replaceChildren(h("p", { class: "small", text: `Summary unavailable: ${err.message}` })); }
}

function askAbout(q) { ensureVisible($("#panel-analyst")); goPanel("panel-analyst"); ask(q); }

async function ask(question) {
  question = (question || "").trim();
  if (!question) return;
  const log = $("#chat-log"), input = $("#chat-q"), send = $("#chat-send"), token = state.token;
  log.append(h("div", { class: "msg user", text: question }));
  const busy = h("div", { class: "msg bot busy", "aria-busy": "true", text: "Reading the stored evidence…" });
  log.append(busy);
  input.value = ""; send.disabled = true;
  log.scrollTop = log.scrollHeight;
  try {
    const a = await api(runUrl(state.run.name, "/ask"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    if (token !== state.token) return;   // another run was opened meanwhile: this answer belongs to the old one
    state.sessionId = a.session_id;
    busy.replaceWith(answerBubble(a));
    setSuggestions(a.refused ? a.suggested_questions : (a.followups.length ? a.followups : state.overview.suggested_questions),
      a.refused ? "Questions I can answer" : (a.followups.length ? "Follow up" : "Try asking"));
  } catch (err) {
    if (token !== state.token) return;
    busy.replaceWith(h("div", { class: "msg bot refused" }, h("div", { class: "part" }, h("span", { class: "tag", text: "Error" }), h("div", { class: "body", text: err.message }))));
  } finally {
    if (token === state.token) { send.disabled = false; log.scrollTop = log.scrollHeight; input.focus(); }
  }
}

function answerBubble(a) {
  if (a.refused) {
    return h("div", { class: "msg bot refused" }, h("div", { class: "part" },
      h("span", { class: "tag", text: `Not answered — ${a.guard.category} question` }),
      h("div", { class: "body" }, ...linkCitations(a.text, a.citations)),
      h("p", { class: "small muted", style: "margin-top:6px", text: `Matched wording: “${a.guard.matched}”. ${a.guard.policy}` })));
  }
  const unique = [...new Map(a.citations.map((c) => [c.token, c])).values()];
  const groups = new Map();
  for (const c of unique) groups.set(c.kind, [...(groups.get(c.kind) || []), c]);
  return h("div", { class: "msg bot" },
    h("div", { class: "part" }, h("span", { class: "tag", text: "Answer" }),
      h("div", { class: "body" }, ...linkCitations(a.text, a.citations)),
      !a.verified ? h("p", { class: "warn", text: `Check notes: ${a.notes.join("; ")}` }) : null),
    h("div", { class: "part" }, h("span", { class: "tag evi", text: "Evidence / citations" }),
      unique.length
        ? [...[...groups].map(([kind, items]) => h("div", { class: "cites" }, h("span", { class: "small muted", style: "margin-right:6px", text: `${kind}:` }),
          ...items.map((c) => h("button", { class: "chip", type: "button", title: `Open ${c.kind}:${c.id}`, onclick: () => openRef(`${c.kind}:${c.id}`) }, c.id)))),
          h("p", { class: "small muted", style: "margin-top:4px", text: `${plural(unique.length, "citation")} checked against the database` })]
        : h("p", { class: "small muted", text: "No evidence was cited by this answer." }),
      h("details", {}, h("summary", { text: `How this was answered · ${a.label}` }),
        h("p", {}, h("strong", { text: "Analyst: " }), a.origin.analyst), h("p", {}, h("strong", { text: "Data: " }), a.origin.observations))));
}

/* ---------- pipeline ---------- */
function renderPipeline() {
  const nav = $("#pipeline");
  if (!state.overview) { panelError(nav, "overview"); return; }
  const r = state.run;
  const stages = [{
    key: "video", title: "Video", panel: "panel-video",
    count: r.video ? r.video.frame_count : null,
    caption: r.video ? `${r.video.name}, ${r.video.fps} fps, ${r.video.duration_seconds}s` : "no video: scripted text frames",
  }, ...state.overview.pipeline.map((s) => (s.key === "observations" ? { ...s, title: "Observations" } : s))];
  nav.className = "pipeline";
  nav.replaceChildren(...stages.map((s) => h("a", { class: `pstage${s.key === "analyst" ? " ai" : ""}`, href: `#${s.panel}`, onclick: (e) => { e.preventDefault(); goPanel(s.panel); } },
    icon(s.key), h("div", { class: "name", text: s.title }),
    s.count !== null && s.count !== undefined ? h("div", { class: "count", text: String(s.count) }) : null,
    h("div", { class: "cap", text: s.caption }))));
}

init();
