"use strict";
/* Evaluator dashboard. Presentation only: it fetches what the backend computed and draws it.
   It only fetches, lays out and cross-links data; every security decision is made server-side. */

const $ = (sel, root = document) => root.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";

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

const state = {
  runs: [], run: null, overview: null, events: null, alerts: null, entities: null,
  observations: null, rules: null, sessionId: null,
  selEvent: null, selAlert: null, selObs: null, hlEvents: new Set(), hlObs: new Set(),
  zoneFilter: "", typeFilter: "",
};

/* ---------- formatting helpers ---------- */
const fmtTime = (iso) => (iso ? iso.slice(0, 19).replace("T", " ") : "");
const refId = (ref) => ref.slice(ref.indexOf(":") + 1);
const refKind = (ref) => ref.slice(0, ref.indexOf(":"));
function fmtDur(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}
function localMs(iso) {   // epoch shifted by the timestamp's own UTC offset, for axis labels
  const m = /([+-])(\d\d):(\d\d)$/.exec(iso);
  const off = m ? (m[1] === "-" ? -1 : 1) * (Number(m[2]) * 60 + Number(m[3])) : 0;
  return Date.parse(iso) + off * 60000;
}
const clockLabel = (ms, withSeconds) => new Date(ms).toISOString().slice(11, withSeconds ? 19 : 16);
const subject = (attrs) => Object.entries(attrs || {}).map(([k, v]) => `${k}: ${v}`).join(", ");
const pretty = (s) => s.replace(/_/g, " ");

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON error */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}
const runApi = (suffix) => api(`/api/runs/${encodeURIComponent(state.run.name)}${suffix}`);
const frameUrl = (frameId, w = 480) =>
  `/api/runs/${encodeURIComponent(state.run.name)}/frames/${frameId}.jpg?w=${w}`;

function showError(err) {
  const box = $("#error");
  box.hidden = false;
  box.textContent = `Error: ${err.message || err}`;
}
function clearError() { $("#error").hidden = true; }

/* ---------- small components ---------- */
function sevBadge(sev) { return h("span", { class: `badge sev sev-${sev}`, text: sev }); }
function typeBadge(t) { return h("span", { class: `badge type-${t}`, text: t }); }
function chip(ref, label) {
  return h("button", { class: "chip", type: "button", title: `Open ${ref}`, onclick: () => openRef(ref) },
    label || ref);
}
function sourceBadges(o) {
  const origin = state.run.description_origin;
  return [
    o.carried_forward
      ? h("span", { class: "badge carried",
          title: "A repeat of the last accepted description for a similar frame. It is NOT a new inference; no description was produced for this frame." },
        "CARRIED FORWARD")
      : h("span", { class: "badge fresh", title: `New description for this frame: ${origin}` }, "NEW OBSERVATION"),
    " ",
    h("span", { class: "badge src", title: "The source label stored with the observation" }, o.source),
  ];
}

/* ---------- boot ---------- */
async function init() {
  try {
    state.runs = (await api("/api/runs")).runs;
    const select = $("#run-select");
    select.replaceChildren(...state.runs.map((r) => h("option", { value: r.name, text: `${r.name}  -  ${r.label}` })));
    select.addEventListener("change", () => loadRun(select.value));
    const wanted = new URLSearchParams(location.hash.slice(1)).get("run");
    const start = state.runs.find((r) => r.name === wanted) || state.runs[0];
    select.value = start.name;
    await loadRun(start.name);
  } catch (err) { showError(err); }
}

async function loadRun(name) {
  clearError();
  Object.assign(state, { selEvent: null, selAlert: null, selObs: null, sessionId: null,
    hlEvents: new Set(), hlObs: new Set(), zoneFilter: "", typeFilter: "" });
  history.replaceState(null, "", `#run=${encodeURIComponent(name)}`);
  try {
    state.run = state.runs.find((r) => r.name === name);
    const get = (suffix) => api(`/api/runs/${encodeURIComponent(name)}${suffix}`);
    const [overview, events, alerts, entities, observations, rules] = await Promise.all([
      get("/overview"), get("/events"), get("/alerts"), get("/entities"),
      get("/observations?limit=200"), api("/api/rules"),
    ]);
    Object.assign(state, { overview, events, alerts, entities, observations, rules });
    state.run = overview.run;
    renderBanner(); renderPipeline(); renderOverview(); renderTimeline(); renderAlerts();
    renderAnalyst(); renderVideo(); applyHighlights();
  } catch (err) { showError(err); }
}

/* ---------- banner and pipeline ---------- */
function renderBanner() {
  const r = state.run;
  $("#kind-banner").className = `banner k-${r.kind}`;
  const parts = [h("span", { class: "kind", text: r.label }), r.detail];
  if (r.video) parts.push(h("div", { class: "small muted", text: `Video: ${r.video.name}, ${r.video.frame_count} frames at ${r.video.fps} fps (${r.video.duration_seconds}s), ${r.video.width}x${r.video.height}` }));
  $("#kind-banner").replaceChildren(...parts);
}

function renderPipeline() {
  $("#pipeline").replaceChildren(...state.overview.pipeline.map((s) =>
    h("a", { class: "stage", href: `#${s.panel}`, onclick: (e) => { e.preventDefault(); goPanel(s.panel); } },
      h("div", { class: "title", text: s.title }),
      h("div", { class: "count", text: s.count === null ? "✓" : s.count }),
      h("div", { class: "cap", text: s.caption }))));
}

function goPanel(id) {
  const el = document.getElementById(id);
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

/* ---------- overview ---------- */
function renderOverview() {
  const o = state.overview, c = o.counts;
  const total = Object.values(o.severity_breakdown).reduce((a, b) => a + b, 0);
  const bar = h("div", { class: "sevbar", role: "img", "aria-label": "Alerts by severity" },
    ...o.severity_order.filter((s) => o.severity_breakdown[s]).map((s) =>
      h("span", { class: `sev-${s}`, style: `width:${(100 * o.severity_breakdown[s]) / total}%`, title: `${s}: ${o.severity_breakdown[s]}` })));
  const legend = h("div", { class: "legend" },
    ...o.severity_order.map((s) => h("span", {}, h("i", { class: `dot sev-${s}` }), `${s} ${o.severity_breakdown[s] || 0}`)));
  const summaryBox = h("div", { id: "summary-box" });
  const body = [
    h("div", { class: "cards" },
      ...[["observations", "observations"], ["frames", "frames"], ["events", "events"], ["alerts", "alerts"], ["entities", "vehicle entities"]]
        .map(([k, l]) => h("div", { class: "card" }, h("div", { class: "n", text: c[k] }), h("div", { class: "l", text: l })))),
    h("div", { class: "grid2" },
      h("div", {}, h("h3", { text: "Alerts by severity" }),
        total ? [bar, legend] : h("p", { class: "muted", text: "No alerts in this run." }),
        h("h3", { text: "Alerts by rule" }),
        Object.keys(o.rule_breakdown).length
          ? h("ul", { class: "plain" }, ...Object.entries(o.rule_breakdown).map(([r, n]) => h("li", {}, h("code", { text: r }), `: ${n}`)))
          : h("p", { class: "muted", text: "None triggered." })),
      h("div", {}, h("h3", { text: "Observations by zone" }),
        h("ul", { class: "plain" }, ...o.zones.map((z) => h("li", {}, h("code", { text: z.zone }), `: ${z.observations}`))),
        h("h3", { text: "Detected objects" }),
        Object.keys(o.object_counts).length
          ? h("ul", { class: "plain" }, ...Object.entries(o.object_counts).map(([k, n]) => h("li", {}, `${k}: ${n}`)))
          : h("p", { class: "muted", text: "No objects detected." })),
      h("div", {}, h("h3", { text: "Where the observations came from" }),
        h("ul", { class: "plain" }, ...o.sources.map((s) => h("li", {},
          h("span", { class: `badge ${s.carried_forward ? "carried" : "fresh"}`, text: s.carried_forward ? "CARRIED FORWARD" : "NEW OBSERVATION" }),
          " ", h("code", { text: s.source }), `: ${s.count}`))),
        h("p", { class: "small muted", text: "CARRIED FORWARD = a repeat of the last accepted description (temporal propagation), not a new inference." })),
      h("div", {}, h("h3", { text: "Vehicle entity memory" }), entityList()),
    ),
    h("h3", { text: "Limitations stated by the backend" }),
    h("ul", { class: "plain" }, ...o.limitations.map((t) => h("li", { text: t }))),
    h("h3", { text: "Run summary (deterministic, offline)" }),
    h("button", { class: "btn", type: "button", onclick: loadSummary }, "Show run summary"),
    summaryBox,
  ];
  $("#overview-body").className = "";
  $("#overview-body").replaceChildren(...body);
}

function entityList() {
  const list = state.entities.entities;
  if (!list.length) return h("p", { class: "muted", text: "No vehicle entities." });
  return h("div", {}, ...list.map((e) => h("div", { id: `entity-${e.entity_id}`, class: "detail", style: "margin-bottom:8px" },
    h("div", {}, h("strong", { text: e.label }), " ", h("span", { class: "badge sim", text: e.identity_confirmed ? "identity supported" : "coarse match, identity NOT confirmed" })),
    h("div", { class: "small muted", text: `${e.appearance_count} separate appearance(s), ${fmtTime(e.first_seen)} to ${fmtTime(e.last_seen)}` }),
    h("div", {}, chip(e.ref), ...e.event_refs.map((r) => chip(r))))));
}

async function loadSummary() {
  const box = $("#summary-box");
  box.replaceChildren(h("p", { class: "muted", text: "Loading…" }));
  try {
    const s = await runApi("/summary");
    box.replaceChildren(h("pre", { class: "summary" }, ...linkCitations(s.text, s.citations)),
      h("p", { class: "small muted", text: s.label }));
  } catch (err) { showError(err); }
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

/* ---------- navigation between panels ---------- */
function openRef(ref) {
  const kind = refKind(ref), id = refId(ref);
  if (kind === "event") { selectEvent(id, true); goPanel("panel-timeline"); }
  else if (kind === "alert") { selectAlert(id); goPanel("panel-alerts"); }
  else if (kind === "entity") {
    const el = document.getElementById(`entity-${id}`);
    if (el) { goPanel("panel-overview"); el.classList.add("hl"); setTimeout(() => el.classList.remove("hl"), 1800); }
  } else if (kind === "frame") {
    const o = state.observations.observations.find((x) => String(x.frame_id) === id);
    if (o) { selectObs(o.observation_id); goPanel("panel-video"); }
  } else if (kind === "obs") { selectObs(id); goPanel("panel-video"); }
}

function applyHighlights() {
  document.querySelectorAll("[data-event]").forEach((el) => {
    el.classList.toggle("hl", state.hlEvents.has(el.dataset.event));
    el.classList.toggle("sel", state.selEvent === el.dataset.event);
  });
  document.querySelectorAll("[data-obs]").forEach((el) => {
    el.classList.toggle("hl", state.hlObs.has(el.dataset.obs));
    el.classList.toggle("sel", state.selObs === el.dataset.obs);
  });
}

/* ---------- event timeline ---------- */
function filteredEvents() {
  return state.events.events.filter((e) =>
    (!state.zoneFilter || e.zone === state.zoneFilter) && (!state.typeFilter || e.type === state.typeFilter));
}

function renderTimeline() {
  const all = state.events.events;
  const zones = [...new Set(all.map((e) => e.zone))];
  const types = [...new Set(all.map((e) => e.type))];
  const select = (id, label, options, value, set) => h("label", {}, `${label} `,
    h("select", { id, onchange: (e) => { set(e.target.value); renderTimeline(); applyHighlights(); } },
      h("option", { value: "", text: "all" }), ...options.map((o) => h("option", { value: o, text: o, selected: o === value }))));
  const filters = h("div", { class: "filters" },
    select("f-zone", "Zone", zones, state.zoneFilter, (v) => { state.zoneFilter = v; }),
    select("f-type", "Type", types, state.typeFilter, (v) => { state.typeFilter = v; }),
    h("span", { class: "muted small", text: `${filteredEvents().length} of ${all.length} events` }));
  const body = all.length
    ? [filters, timelineSvg(), timelineLegend(), eventTable(), h("div", { id: "event-detail" })]
    : [h("p", { class: "muted", text: "No events are stored for this run (build-events may not have been run)." })];
  $("#timeline-body").className = "";
  $("#timeline-body").replaceChildren(...body);
  if (state.selEvent) selectEvent(state.selEvent, false);
}

function timelineSvg() {
  const events = filteredEvents();
  const lanes = [...new Set(state.events.events.map((e) => e.zone))];
  const span = state.events.span;
  const t0 = localMs(span.first), t1 = Math.max(localMs(span.last), t0 + 60000);
  const W = 1000, left = 118, right = 16, laneH = 34, top = 8, axis = 26;
  const H = top + lanes.length * laneH + axis;
  const x = (iso) => left + ((localMs(iso) - t0) / (t1 - t0)) * (W - left - right);
  const alertSev = new Map(state.alerts.alerts.map((a) => [a.ref, a.severity]));
  const order = state.overview.severity_order;
  const root = svg("svg", { class: "timeline", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Event timeline by zone" });
  lanes.forEach((zone, i) => {
    const y = top + i * laneH;
    root.append(svg("line", { class: "lane", x1: left, x2: W - right, y1: y + laneH, y2: y + laneH }),
      svg("text", { x: 8, y: y + laneH / 2 + 4 }, pretty(zone)));
  });
  for (let k = 0; k <= 6; k++) {
    const ms = t0 + ((t1 - t0) * k) / 6, px = left + ((W - left - right) * k) / 6;
    root.append(svg("line", { class: "lane", x1: px, x2: px, y1: top, y2: top + lanes.length * laneH }),
      svg("text", { x: px, y: H - 8, "text-anchor": "middle" }, clockLabel(ms, t1 - t0 < 15 * 60000)));
  }
  for (const ev of events) {
    const lane = lanes.indexOf(ev.zone), y = top + lane * laneH + 6;
    const x0 = x(ev.first_seen), w = Math.max(6, x(ev.last_seen) - x0);
    const bar = svg("rect", { class: `bar type-${ev.type}`, x: x0, y, width: w, height: laneH - 12, rx: 3,
      "data-event": ev.event_id, tabindex: 0, role: "button", "aria-label": `${ev.type} in ${ev.zone}, ${ev.event_id}` },
      svg("title", {}, `${ev.event_id}: ${ev.type}, ${subject(ev.attributes)}, ${fmtDur(ev.duration_seconds)}`));
    bar.addEventListener("click", () => selectEvent(ev.event_id, false));
    bar.addEventListener("keydown", (e) => { if (e.key === "Enter") selectEvent(ev.event_id, false); });
    root.append(bar);
    ev.alert_refs.forEach((ar, n) => {
      const sev = alertSev.get(ar);
      if (!sev) return;
      const m = svg("path", { class: `marker sev-${sev}`, d: `M ${x0} ${y - 5} l 5 5 l -5 5 l -5 -5 z`,
        transform: `translate(0, ${n * 11})`, "data-sev": order.indexOf(sev) }, svg("title", {}, `${ar} (${sev})`));
      m.addEventListener("click", () => openRef(ar));
      root.append(m);
    });
  }
  return root;
}

function timelineLegend() {
  return h("div", { class: "legend-line" },
    h("span", {}, "Bars = events (colour = object type). Diamonds = alerts raised on that event, coloured by severity."),
    h("span", {}, typeBadge("person"), " ", typeBadge("vehicle"), " ", typeBadge("other")));
}

function eventTable() {
  const rows = filteredEvents().map((ev) => h("tr", { class: "clickable", "data-event": ev.event_id, onclick: () => selectEvent(ev.event_id, false) },
    h("td", { class: "mono", text: fmtTime(ev.first_seen) }),
    h("td", {}, typeBadge(ev.type)),
    h("td", { text: subject(ev.attributes) || "-" }),
    h("td", { text: pretty(ev.zone) }),
    h("td", { text: fmtDur(ev.duration_seconds) }),
    h("td", { class: "mono", text: ev.event_id }),
    h("td", {}, ...ev.alert_refs.map((r) => chip(r, refId(r)))),
    h("td", { class: "mono", text: `${ev.frames_total} (${ev.frame_ids[0]}–${ev.frame_ids[ev.frame_ids.length - 1]})` })));
  return h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ...["Time", "Type", "Subject", "Zone", "Duration", "Event ID", "Alerts", "Frames"].map((t) => h("th", { text: t })))),
    h("tbody", {}, ...rows)));
}

async function selectEvent(id, scroll) {
  state.selEvent = id;
  const ev = state.events.events.find((e) => e.event_id === id);
  state.hlEvents = new Set([id]);
  state.hlObs = new Set(ev ? ev.observation_ids : []);
  applyHighlights();
  const box = $("#event-detail");
  if (!box) return;
  box.replaceChildren(h("p", { class: "muted", text: "Loading evidence…" }));
  try {
    const d = await runApi(`/events/${encodeURIComponent(id)}`);
    const e = d.event;
    box.replaceChildren(h("div", { class: "detail", style: "margin-top:12px" },
      h("h3", {}, `${e.ref} `, typeBadge(e.type)),
      h("dl", { class: "kv" },
        h("dt", { text: "Zone" }), h("dd", { text: pretty(e.zone) }),
        h("dt", { text: "First / last seen" }), h("dd", { text: `${fmtTime(e.first_seen)} → ${fmtTime(e.last_seen)}` }),
        h("dt", { text: "Duration" }), h("dd", { text: fmtDur(e.duration_seconds) }),
        h("dt", { text: "Subject" }), h("dd", { text: subject(e.attributes) || "-" }),
        h("dt", { text: "Activity" }), h("dd", { text: e.activities.join(", ") || "-" }),
        h("dt", { text: "Closed because" }), h("dd", { text: e.close_reason }),
        h("dt", { text: "Vehicle entity" }), h("dd", {}, d.entity ? [chip(d.entity.ref, d.entity.label)] : "none (people are never linked)"),
        h("dt", { text: "Alerts" }), h("dd", {}, d.alerts.length ? d.alerts.map((a) => chip(a.ref)) : "none")),
      h("h3", { text: `Evidence: ${d.observations.length} observation(s) behind this event` }),
      ...d.observations.map((o) => obsCard(o))));
    if (scroll) box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) { showError(err); }
}

function obsCard(o) {
  return h("div", { class: "obs-card" },
    o.image ? h("img", { src: frameUrl(o.frame_id, 240), alt: `frame ${o.frame_id}`, loading: "lazy" }) : null,
    h("div", { class: "body" },
      h("div", {}, h("span", { class: "mono", text: fmtTime(o.time) }), " ", chip(o.frame_ref, `frame ${o.frame_id}`), " ", sourceBadges(o)),
      h("div", { text: o.description }),
      o.objects.length ? h("div", { class: "small muted", text: o.objects.map((x) => `${x.type}${x.activity ? " (" + x.activity + ")" : ""} ${subject(x.attributes)}`).join("; ") }) : null,
      o.telemetry_simulated ? h("div", { class: "small muted" }, h("span", { class: "badge sim", text: "SIMULATED telemetry" }), ` battery ${o.telemetry_simulated.battery_percent}%, altitude ${o.telemetry_simulated.altitude_m} m${o.telemetry_simulated.stale ? ", STALE sample" : ""}`) : null,
      h("div", {}, ...o.event_refs.map((r) => chip(r)))));
}

/* ---------- alerts ---------- */
function renderAlerts() {
  const list = state.alerts.alerts;
  const cards = list.map((a) => h("button", { class: `alert-card sev-${a.severity}`, type: "button", "data-alert": a.alert_id,
    onclick: () => selectAlert(a.alert_id) },
    h("div", { class: "t", text: a.title }),
    h("div", {}, sevBadge(a.severity), " ", h("code", { text: a.rule_id })),
    h("div", { class: "small muted", text: `${pretty(a.zone || "-")} · ${fmtTime(a.triggered_at)}` }),
    h("div", { class: "small mono", text: a.alert_id })));
  const rules = h("details", { class: "rules" }, h("summary", { text: `Security rules in force (${state.rules.rules.length})` }),
    h("p", { class: "small muted", text: state.rules.note }),
    ...state.rules.rules.map((r) => h("div", { class: "detail", style: "margin-bottom:8px" },
      h("div", {}, h("code", { text: r.id }), " ", sevBadge(r.severity), " ", h("span", { class: "badge sim", text: r.enabled ? "enabled" : "disabled" }),
        ` — ${state.overview.rule_breakdown[r.id] || 0} alert(s) in this run`),
      h("div", { class: "small", text: r.description }),
      h("pre", { class: "small mono", style: "margin:4px 0 0;white-space:pre-wrap", text: JSON.stringify(r.settings, null, 1) }))));
  $("#alerts-body").className = "";
  $("#alerts-body").replaceChildren(
    list.length
      ? h("div", { class: "split" }, h("div", { class: "alert-list" }, ...cards), h("div", { id: "alert-detail" }, h("p", { class: "muted", text: "Select an alert to see why it was raised and the evidence behind it." })))
      : h("p", { class: "muted", text: "No alerts were raised in this run. That is a result, not an error: no rule's condition was met." }),
    rules);
  if (state.selAlert) selectAlert(state.selAlert);
}

async function selectAlert(id) {
  state.selAlert = id;
  document.querySelectorAll("[data-alert]").forEach((el) => el.classList.toggle("sel", el.dataset.alert === id));
  const box = $("#alert-detail");
  if (!box) return;
  box.replaceChildren(h("p", { class: "muted", text: "Loading…" }));
  try {
    const d = await runApi(`/alerts/${encodeURIComponent(id)}`);
    const a = d.alert;
    state.hlEvents = new Set(d.events.map((e) => refId(e.ref)));
    state.hlObs = new Set(d.evidence.flatMap((x) => x.observations.map((o) => o.observation_id)));
    box.replaceChildren(h("div", { class: "detail" },
      h("h3", {}, a.title, " ", sevBadge(a.severity)),
      h("dl", { class: "kv" },
        h("dt", { text: "Rule" }), h("dd", {}, h("code", { text: a.rule_id })),
        h("dt", { text: "Severity" }), h("dd", { text: `${a.severity} (fixed by the rule, never chosen by a model)` }),
        h("dt", { text: "Zone" }), h("dd", { text: pretty(a.zone || "-") }),
        h("dt", { text: "Triggered" }), h("dd", { text: fmtTime(a.triggered_at) }),
        h("dt", { text: "Alert ID" }), h("dd", { class: "mono", text: a.alert_id }),
        h("dt", { text: "Event IDs" }), h("dd", {}, ...a.event_refs.map((r) => chip(r))),
        h("dt", { text: "Vehicle entity" }), h("dd", {}, a.entity_ref ? chip(a.entity_ref) : "none"),
        h("dt", { text: "Frame IDs" }), h("dd", {}, ...a.frame_ids.map((f) => chip(`frame:${f}`, String(f))))),
      h("h3", { text: "Why it was raised" }), h("p", { text: a.reason }),
      h("h3", { text: "Numbers the rule compared" }),
      d.facts.length ? h("table", {}, h("tbody", {}, ...d.facts.map((f) => h("tr", {}, h("td", { text: f.name }), h("td", { class: "mono", text: String(f.value) }))))) : h("p", { class: "muted", text: "none" }),
      h("h3", { text: "Evidence chain" }),
      h("ol", { class: "plain" }, ...d.steps.map((s) => h("li", { text: s }))),
      ...d.evidence.map((x) => h("div", {}, h("div", {}, chip(x.event_ref), h("span", { class: "muted small", text: ` ${x.observations.length} observation(s)` })), ...x.observations.map((o) => obsCard(o)))),
      h("p", {}, h("button", { class: "btn primary", type: "button", onclick: () => askAbout(`Show me the evidence for ${a.alert_id}`) }, "Ask the analyst about this alert"))));
    applyHighlights();
  } catch (err) { showError(err); }
}

/* ---------- analyst ---------- */
function renderAnalyst() {
  const suggestions = h("div", { class: "suggestions", id: "suggestions" });
  const log = h("div", { class: "chat-log", id: "chat-log", "aria-live": "polite" },
    h("div", { class: "msg bot" }, h("span", { class: "label", text: "Analyst (offline template, no language model)" }),
      "Ask about this run. Answers cite stored evidence as clickable references. I do not identify people, infer intent or call anyone a criminal."));
  const example = state.overview.suggested_questions[0] || "Summarize the run.";   // from the backend
  const input = h("textarea", { id: "chat-q", rows: 1, placeholder: `Ask a question, e.g. “${example}”, then ask a follow-up`, "aria-label": "Your question" });
  const send = h("button", { class: "btn primary", id: "chat-send", type: "button", onclick: () => ask(input.value) }, "Ask");
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); } });
  $("#analyst-body").className = "";
  $("#analyst-body").replaceChildren(h("div", { class: "chat" }, log, suggestions, h("div", { class: "chat-input" }, input, send)));
  setSuggestions(state.overview.suggested_questions, "Try:");
}

function setSuggestions(list, lead) {
  $("#suggestions").replaceChildren(h("span", { class: "lead", text: lead }),
    ...list.map((q) => h("button", { type: "button", text: q, onclick: () => ask(q) })));
}

function askAbout(q) { goPanel("panel-analyst"); ask(q); }

async function ask(question) {
  question = (question || "").trim();
  if (!question) return;
  const log = $("#chat-log"), input = $("#chat-q"), send = $("#chat-send");
  log.append(h("div", { class: "msg user", text: question }));
  input.value = ""; send.disabled = true;
  try {
    const a = await api(`/api/runs/${encodeURIComponent(state.run.name)}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    state.sessionId = a.session_id;
    log.append(answerBubble(a));
    setSuggestions(a.refused ? a.suggested_questions : (a.followups.length ? a.followups : state.overview.suggested_questions),
      a.refused ? "Questions I can answer:" : (a.followups.length ? "Follow up:" : "Try:"));
  } catch (err) {
    log.append(h("div", { class: "msg bot refused" }, h("span", { class: "label", text: "Error" }), err.message));
  } finally {
    send.disabled = false; log.scrollTop = log.scrollHeight; input.focus();
  }
}

function answerBubble(a) {
  const kids = [h("span", { class: "label", text: a.refused ? `NOT ANSWERED — ${a.guard.category} question` : a.label }), ...linkCitations(a.text, a.citations)];
  if (a.refused) {
    kids.push(h("span", { class: "origin", text: `Matched wording: “${a.guard.matched}”. ${a.guard.policy}` }));
  } else {
    if (!a.verified) kids.push(h("span", { class: "warn", text: `Check notes: ${a.notes.join("; ")}` }));
    kids.push(h("span", { class: "origin" }, h("strong", { text: "Analyst: " }), a.origin.analyst, h("br"), h("strong", { text: "Data: " }), a.origin.observations,
      a.citations.length ? [h("br"), h("strong", { text: "Citations checked against the database: " }), `${a.citations.length}`] : null));
  }
  return h("div", { class: `msg bot${a.refused ? " refused" : ""}` }, ...kids);
}

/* ---------- video / observations ---------- */
function renderVideo() {
  const list = state.observations.observations;
  if (!list.length) {
    $("#video-body").className = "";
    $("#video-body").replaceChildren(h("p", { class: "muted", text: "No observations stored." }));
    return;
  }
  const r = state.run;
  const note = h("p", { class: "hint", text: r.has_video
    ? `Each frame below is the real picture at the observation's frame index in ${r.video.name}. Its description is ${r.description_origin}.`
    : `This run has no video: each observation is ${r.description_origin}. No image exists for it.` });
  const slider = h("input", { type: "range", id: "obs-slider", min: 0, max: list.length - 1, value: 0, "aria-label": "Observation",
    oninput: (e) => selectObs(list[Number(e.target.value)].observation_id, true) });
  const viewer = h("div", {},
    h("div", { class: "frame-box", id: "frame-box" }),
    h("div", { class: "transport" },
      h("button", { class: "btn", type: "button", text: "◀", "aria-label": "Previous observation", onclick: () => stepObs(-1) }), slider,
      h("button", { class: "btn", type: "button", text: "▶", "aria-label": "Next observation", onclick: () => stepObs(1) })),
    h("div", { id: "obs-detail" }));
  const rows = list.map((o) => h("button", { class: `obs-row ${o.carried_forward ? "carried" : "fresh"}`, type: "button", "data-obs": o.observation_id,
    onclick: () => selectObs(o.observation_id) },
    o.image ? h("img", { src: frameUrl(o.frame_id, 160), alt: "", loading: "lazy" }) : null,
    h("div", {}, h("div", { class: "t", text: `${fmtTime(o.time)} · frame ${o.frame_id} · ${pretty(o.zone)}` }),
      h("div", {}, sourceBadges(o)), h("div", { class: "small", text: o.description }))));
  $("#video-body").className = "";
  $("#video-body").replaceChildren(note, h("div", { class: "viewer" }, viewer, h("div", {},
    h("h3", { text: `${state.observations.total} observation(s)${state.observations.truncated ? ` (showing ${state.observations.returned})` : ""}` }),
    h("div", { class: "obs-list" }, ...rows))));
  selectObs(state.selObs || list[0].observation_id, false, true);
}

function stepObs(delta) {
  const list = state.observations.observations;
  const i = list.findIndex((o) => o.observation_id === state.selObs);
  const next = list[Math.max(0, Math.min(list.length - 1, i + delta))];
  if (next) selectObs(next.observation_id);
}

function selectObs(id, fromSlider, quiet) {
  const list = state.observations.observations;
  const o = list.find((x) => x.observation_id === id);
  if (!o) return;
  state.selObs = id;
  if (!quiet) {
    state.hlEvents = new Set(o.event_refs.map(refId));
    state.hlObs = new Set([id]);
  }
  const box = $("#frame-box");
  if (!box) return;
  box.replaceChildren(o.image
    ? h("img", { src: frameUrl(o.frame_id, 640), alt: `Video frame ${o.frame_id}` })
    : h("div", { class: "none" }, "No image: this is a scripted text frame. It has a description but no pixels."));
  if (!fromSlider) $("#obs-slider").value = list.indexOf(o);
  $("#obs-detail").replaceChildren(h("div", { class: "detail", style: "margin-top:8px" },
    h("div", {}, h("strong", { text: `Frame ${o.frame_id}` }), ` · ${fmtTime(o.time)} · ${pretty(o.zone)} `, sourceBadges(o)),
    h("p", { text: o.description }),
    o.carried_forward ? h("p", { class: "small", style: "color:var(--carried)", text: "Carried forward: this repeats the last accepted description. No new description was produced for this frame." }) : null,
    o.objects.length ? h("div", { class: "small", text: `Objects: ${o.objects.map((x) => `${x.type}${x.activity ? " (" + x.activity + ")" : ""} ${subject(x.attributes)}`).join("; ")}` }) : h("div", { class: "small muted", text: "No objects detected in this frame." }),
    o.telemetry_simulated ? h("div", { class: "small muted" }, h("span", { class: "badge sim", text: "SIMULATED telemetry" }), ` battery ${o.telemetry_simulated.battery_percent}%, altitude ${o.telemetry_simulated.altitude_m} m, sample age ${o.telemetry_simulated.age_seconds}s${o.telemetry_simulated.stale ? " (STALE)" : ""}`) : null,
    h("div", {}, h("span", { class: "small muted", text: "Part of: " }), ...(o.event_refs.length ? o.event_refs.map((r) => chip(r)) : [h("span", { class: "small muted", text: "no event (nothing detected here)" })]))));
  applyHighlights();
}

init();
