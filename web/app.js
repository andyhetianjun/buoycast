// Map page. It runs in two modes:
//  - server mode: reads the FastAPI endpoints (buoycast-serve)
//  - static mode: reads precomputed JSON files in data/ (buoycast-site), no server needed
const STATIC = !!window.BUOYCAST_STATIC;
const SITE = window.BUOYCAST_SITE || {};
const LIVE = SITE.mode === "live";
const state = { station: null, stations: [], meta: null, cache: {} };
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const HOUR = 3600000;
const WINDOW_HOURS = 72;

const map = L.map("map").setView([45.5, -100], 3);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 12,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// <input type="datetime-local"> wants "YYYY-MM-DDTHH:MM" and we treat it as UTC.
const toInput = (iso) => iso.slice(0, 16);
const toApi = (v) => v + ":00Z";
const fmtTime = (iso) => iso.replace("T", " ").slice(0, 16);
const fmt = (x) => (x == null ? "n/a" : x.toFixed(2) + " m");
const pct = (x) => (x == null ? "n/a" : (x >= 0 ? "+" : "") + (x * 100).toFixed(1) + "%");

function el(name, attrs, text) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function html(tag, props, text) {
  const e = document.createElement(tag);
  Object.assign(e, props || {});
  if (text != null) e.textContent = text;
  return e;
}

// ---------- data access ----------

async function loadStations() {
  if (STATIC) {
    state.meta = await getJSON("data/index.json");
    return state.meta.stations.map((s) => ({
      ...s,
      first_obs: s.start,
      last_obs: new Date(Date.parse(s.start) + (s.hours - 1) * HOUR).toISOString(),
    }));
  }
  return getJSON("/stations");
}

async function stationData(id) {
  if (!state.cache[id]) state.cache[id] = await getJSON(`data/${id}.json`);
  return state.cache[id];
}

// What the chart and tables need for one station, window end and lead time.
async function loadView(s, endIso, horizon) {
  if (!STATIC) {
    const [fc, bt, qc] = await Promise.all([
      getJSON(`/forecast/${s.id}?as_of=${endIso}`).catch(() => null),
      getJSON(`/backtest/${s.id}?horizon=${horizon}&hours=${WINDOW_HOURS}&end=${endIso}`),
      getJSON(`/qc/${s.id}?hours=${WINDOW_HOURS}&end=${endIso}`).catch(() => null),
    ]);
    const strips = {};
    if (qc) for (const v of ["wvht", "wspd", "pres", "wtmp"]) strips[v] = qc.rows.map((r) => ({ time: r.time, flag: r[`${v}_overall`] }));
    return { fc, rows: bt.rows, strips, hasForecast: true };
  }
  const d = await stationData(s.id);
  const t0 = Date.parse(d.start);
  const e = Math.round((Date.parse(endIso) - t0) / HOUR);
  const f = d.forecast;
  const at = (arr, i) => (arr && i >= 0 && i < arr.length ? arr[i] : null);
  const fAt = (lead, m, i) => (f ? at(f[`h${lead}`]?.[m], i - f.offset) : null);
  const rows = [];
  const strips = { wvht: [], wspd: [], pres: [], wtmp: [] };
  for (let i = e - WINDOW_HOURS + 1; i <= e; i++) {
    if (i < 0 || i >= d.hours) continue;
    const time = new Date(t0 + i * HOUR).toISOString();
    rows.push({
      valid_time: time,
      observed: d.wvht[i],
      mlp: fAt(horizon, "mlp", i),
      linear: fAt(horizon, "linear", i),
      persistence: fAt(horizon, "persistence", i),
    });
    for (const v of Object.keys(strips)) strips[v].push({ time, flag: Number(d.qc[v][i]) });
  }
  if (LIVE && e === d.hours - 1) {
    // The newest forecast reaches past the last reading: show that stretch too.
    for (let i = e + 1; i <= e + horizon; i++) {
      rows.push({
        valid_time: new Date(t0 + i * HOUR).toISOString(),
        observed: null,
        mlp: fAt(horizon, "mlp", i),
        linear: fAt(horizon, "linear", i),
        persistence: fAt(horizon, "persistence", i),
      });
    }
  }
  const forecasts = [];
  for (const lead of [3, 6]) {
    const j = e + lead;
    const mlp = fAt(lead, "mlp", j);
    if (mlp == null) continue;
    forecasts.push({
      horizon_hours: lead,
      valid_time: new Date(t0 + j * HOUR).toISOString(),
      mlp,
      linear: fAt(lead, "linear", j),
      persistence: at(d.wvht, e),
    });
  }
  return { fc: forecasts.length ? { forecasts } : null, rows, strips, hasForecast: !!f };
}

// ---------- map ----------

function distanceKm(lat1, lon1, lat2, lon2) {
  const r = Math.PI / 180;
  const a = Math.sin(((lat2 - lat1) * r) / 2) ** 2 +
    Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(a));
}

function nearestStation(lat, lon) {
  let best = null;
  for (const s of state.stations) {
    const d = distanceKm(lat, lon, s.lat, s.lon);
    if (!best || d < best.d) best = { s, d };
  }
  return best;
}

async function init() {
  if (STATIC) {
    const b = $("banner");
    b.hidden = false;
    if (LIVE) {
      b.textContent = `Live data: the latest NOAA buoy observations, fetched ${SITE.built_utc} UTC (source: ${SITE.source}). ` +
        "This page is rebuilt every hour. There is no server: the forecasts and quality flags were calculated at that time by a model trained on 2022 to 2023 data.";
    } else {
      const why = SITE.mode === "sample-fallback"
        ? "The live NOAA files could not be reached when this page was built, so this is the committed sample instead. "
        : "";
      b.textContent = `${why}Demo: everything here is precomputed from the sample data committed to the repository (built ${SITE.built}). ` +
        "There is no server. The forecasts and quality flags were calculated once when the site was built, and the data runs from 2022 to 2024, not live.";
    }
  }
  state.stations = await loadStations();
  const pts = [];
  for (const s of state.stations) {
    const m = L.marker([s.lat, s.lon]).addTo(map);
    if (s.online === false) m.setOpacity(0.45);
    m.bindTooltip(s.id + (s.online === false ? " (no recent reading)" : ""));
    s.marker = m;
    m.on("click", (ev) => { L.DomEvent.stopPropagation(ev); select(s); });
    pts.push([s.lat, s.lon]);
  }
  map.fitBounds(pts, { padding: [30, 30] });
  if (STATIC) renderLiveReport();
  map.on("click", (ev) => {
    const n = nearestStation(ev.latlng.lat, ev.latlng.lng);
    L.popup()
      .setLatLng(ev.latlng)
      .setContent(`Nearest station: ${n.s.id}, ${Math.round(n.d)} km away`)
      .openOn(map);
  });
  if (STATIC) drawDrifters();
  $("horizon").addEventListener("change", refresh);
  $("end").addEventListener("change", refresh);
  $("vlead").addEventListener("change", renderVerification);
  loadReport();
  // Open on a station that has a current forecast (live), or one with test-period forecasts (sample).
  const first = state.stations.find((s) => s.current_forecast && s.online !== false);
  if (first) select(first);
}

function drawDrifters() {
  for (const d of state.meta.drifters) {
    const line = L.polyline(d.coords, { color: "#7a4fb0", weight: 2, opacity: 0.8 }).addTo(map);
    line.bindTooltip(`Drifting buoy ${d.id}: ${d.fixes} positions, ${d.flagged} flagged`);
  }
}

function select(s) {
  state.station = s;
  $("hint").hidden = true;
  $("detail").hidden = false;
  $("name").textContent = `${s.id}: ${s.name}`;
  if (LIVE) {
    const ago = s.age_hours < 1 ? "less than an hour ago" : `${Math.round(s.age_hours)} h before the page was built`;
    $("meta").textContent = `Latest wave reading ${fmtTime(s.last_obs)} UTC (${ago}). ` + (s.trained
      ? "The model was trained on this buoy."
      : "This buoy was not in the training data, so its forecast is less reliable.");
  } else {
    $("meta").textContent = `Data from ${s.first_obs.slice(0, 10)} to ${s.last_obs.slice(0, 10)} (UTC)`;
  }
  // The precomputed forecasts need their outcome to exist, so the last few hours are not offered
  // (except live, where the newest forecast is the point and has no outcome yet).
  const lastEnd = STATIC && !LIVE ? new Date(Date.parse(s.last_obs) - 6 * HOUR).toISOString() : s.last_obs;
  $("end").value = toInput(lastEnd);
  $("end").min = toInput(s.first_obs);
  $("end").max = toInput(lastEnd);
  if (STATIC) loadEvents(s);
  refresh();
  renderVerification();
}

async function refresh() {
  const s = state.station;
  if (!s || !$("end").value) return;
  const end = toApi($("end").value);
  const h = Number($("horizon").value);
  $("note").textContent = "";
  try {
    const view = await loadView(s, end, h);
    renderForecast(view.fc, s);
    renderChart(view);
    renderStrips(view.strips);
  } catch (e) {
    $("note").textContent = "Could not load data: " + e.message;
  }
}

// ---------- forecast table and chart ----------

function renderForecast(fc, s) {
  const body = document.querySelector("#fc tbody");
  body.replaceChildren();
  if (!fc) {
    const tr = body.insertRow();
    const td = tr.insertCell();
    td.colSpan = 4;
    td.textContent = LIVE
      ? "No forecast for this hour (the model needs 24 hours of readings before it)."
      : "No forecast for this hour (forecasts exist for July to December 2024 only).";
    return;
  }
  if (LIVE && s.online === false) {
    const tr = body.insertRow();
    const td = tr.insertCell();
    td.colSpan = 4;
    td.textContent = "This buoy has not reported recently, so these forecasts are old.";
  }
  for (const f of fc.forecasts) {
    const tr = body.insertRow();
    [fmtTime(f.valid_time), fmt(f.mlp), fmt(f.linear), fmt(f.persistence)]
      .forEach((t) => (tr.insertCell().textContent = t));
  }
}

function renderChart(view) {
  const rows = view.rows;
  const svg = $("chart");
  svg.replaceChildren();
  const W = 640, H = 300, m = { l: 44, r: 12, t: 12, b: 28 };
  if (!rows.length) {
    svg.appendChild(el("text", { x: 20, y: 30 }, "No readings in this window."));
    return;
  }
  const times = rows.map((r) => Date.parse(r.valid_time));
  const vals = rows.flatMap((r) => [r.observed, r.mlp, r.persistence]).filter((v) => v != null);
  const x0 = Math.min(...times), x1 = Math.max(...times);
  const y1 = Math.max(0.5, ...vals) * 1.1;
  const X = (t) => m.l + ((t - x0) / Math.max(1, x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v / y1) * (H - m.t - m.b);

  for (let i = 0; i <= 4; i++) {
    const v = (y1 * i) / 4;
    svg.appendChild(el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    svg.appendChild(el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
  }
  svg.appendChild(el("text", { x: 4, y: 12 }, "m"));
  for (let i = 0; i <= 3; i++) {
    const t = x0 + ((x1 - x0) * i) / 3;
    const label = new Date(t).toISOString().slice(5, 13).replace("T", " ") + "h";
    svg.appendChild(el("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, label));
  }
  const line = (key, cls) => {
    // Break the line where a value is missing instead of drawing across the gap.
    let d = "", pen = false;
    rows.forEach((r, i) => {
      if (r[key] == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(times[i]).toFixed(1)},${Y(r[key]).toFixed(1)}`;
      pen = true;
    });
    if (d) svg.appendChild(el("path", { class: "l " + cls, d }));
  };
  line("persistence", "per");
  line("mlp", "mlp");
  line("observed", "obs");
  // Observed readings the quality checks did not like.
  const flags = view.strips.wvht || [];
  flags.forEach((f, i) => {
    if ((f.flag === 3 || f.flag === 4) && rows[i] && rows[i].observed != null) {
      svg.appendChild(el("circle", { cx: X(times[i]), cy: Y(rows[i].observed), r: 5, class: "pt " + (f.flag === 4 ? "fail" : "suspect") }));
    }
  });
  $("note").textContent = view.hasForecast && rows.some((r) => r.mlp != null)
    ? `Each forecast is plotted at the time it is for, and was made ${$("horizon").value} hours earlier.` +
      (LIVE && rows.some((r) => r.observed == null && r.mlp != null) ? " The last hours (no observation yet) are the newest forecast." : "")
    : LIVE
      ? "No earlier forecasts to compare: a forecast needs 24 hours of readings first, and this buoy's file only holds the last 24 hours."
      : "No forecasts in this window: they exist for July to December 2024 only (the test period), and not at all for buoys with no data then.";
}

// ---------- quality flags ----------

const FLAG_CLASS = { 1: "pass", 2: "pass", 3: "suspect", 4: "fail", 9: "missing" };
const VAR_LABEL = { wvht: "wave height", wspd: "wind speed", pres: "pressure", wtmp: "water temp" };

function renderStrips(strips) {
  const svg = $("qcstrip");
  svg.replaceChildren();
  const vars = Object.keys(strips).filter((v) => strips[v].length);
  if (!vars.length) {
    svg.appendChild(el("text", { x: 20, y: 30 }, "No quality flags for this window."));
    return;
  }
  const left = 78, right = 12, rowH = 20;
  svg.setAttribute("viewBox", `0 0 640 ${vars.length * (rowH + 6) + 4}`);
  vars.forEach((v, r) => {
    const y = 4 + r * (rowH + 6);
    svg.appendChild(el("text", { x: left - 8, y: y + 14, "text-anchor": "end" }, VAR_LABEL[v]));
    const cells = strips[v];
    const w = (640 - left - right) / cells.length;
    cells.forEach((c, i) => {
      svg.appendChild(el("rect", { x: left + i * w, y, width: Math.max(w, 1), height: rowH, class: "cell " + FLAG_CLASS[c.flag] }));
    });
  });
}

async function loadEvents(s) {
  const d = await stationData(s.id);
  $("events-box").hidden = false;
  $("events-title").textContent = LIVE
    ? "Flagged stretches at this station (recent readings)"
    : "Flagged stretches at this station (2022 to 2024)";
  const ul = $("events");
  ul.replaceChildren();
  $("events-note").textContent = d.n_events
    ? `${d.n_events} ${d.n_events === 1 ? "stretch" : "stretches"} flagged by at least one test; showing the ${d.events.length === 1 ? "one" : "worst " + d.events.length}. Click one to move the window there.`
    : "No readings flagged at this station.";
  const t0 = Date.parse(d.start);
  for (const ev of d.events) {
    const from = new Date(t0 + ev.start * HOUR);
    const to = new Date(t0 + (ev.start + ev.hours - 1) * HOUR);
    const li = html("li");
    const btn = html("button", { type: "button", className: "link" },
      `${from.toISOString().slice(0, 16).replace("T", " ")}: ${VAR_LABEL[ev.var] || ev.var}, ${ev.test}, ${ev.flag === 4 ? "fail" : "suspect"} (${ev.hours} h)`);
    btn.addEventListener("click", () => {
      // Show the window that ends a day after the start of the stretch (or at its end).
      const endMs = Math.min(Math.max(to.getTime(), from.getTime() + 36 * HOUR), t0 + (d.hours - 1) * HOUR);
      $("end").value = toInput(new Date(endMs).toISOString());
      refresh();
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

// ---------- report: verification and quality-check summary ----------

async function loadReport() {
  try {
    if (STATIC) {
      state.verification = state.meta.verification;
      state.qc = state.meta.qc;
    } else {
      [state.verification, state.qc] = await Promise.all([getJSON("/verification"), getJSON("/qc/summary")]);
    }
  } catch (e) {
    $("verification").textContent = "Could not load the report: " + e.message;
    return;
  }
  renderVerification();
  renderQcSummary();
}

function renderLiveReport() {
  const h = state.meta.hindcast;
  if (!LIVE || !h || !Object.keys(h).length) return;
  $("live-report").hidden = false;
  const body = $("hindcast").createTBody();
  for (const lead of ["3", "6"]) {
    if (!h[lead]) continue;
    const rows = [["All buoys below", h[lead]], ...Object.entries(h[lead].stations).map(([id, g]) => [id, g])];
    for (const [name, g] of rows) {
      const r = body.insertRow();
      if (name.startsWith("All")) r.className = "group-row";
      [`${lead} h`, name, g.n.toLocaleString("en-US"), g.persistence.toFixed(3), g.linear.toFixed(3), g.mlp.toFixed(3)]
        .forEach((t) => (r.insertCell().textContent = t));
    }
  }
}

function skillBar(lo, hi, skill) {
  // Scale from -10% to +40% skill.
  const a = -0.1, b = 0.4, w = 140;
  const X = (v) => Math.max(0, Math.min(w, ((v - a) / (b - a)) * w));
  const svg = el("svg", { viewBox: `0 0 ${w} 14`, class: "skillbar", role: "img", "aria-label": "skill range" });
  svg.appendChild(el("line", { x1: X(0), x2: X(0), y1: 0, y2: 14, class: "zero" }));
  svg.appendChild(el("rect", { x: X(lo), y: 4, width: Math.max(2, X(hi) - X(lo)), height: 6, class: "range" }));
  if (skill != null) svg.appendChild(el("circle", { cx: X(skill), cy: 7, r: 3.5, class: "est" }));
  return svg;
}

function renderVerification() {
  const box = $("verification");
  if (!state.verification) return;
  const lead = state.verification.leads[$("vlead").value];
  box.replaceChildren();
  const groups = [
    ["All forecasts", [{ name: "All stations and heights", g: lead.overall }]],
    ["By wave height when the forecast was issued", lead.by_height.map((g) => ({ name: g.bin, g }))],
    ["By station", lead.by_station.map((g) => ({ name: g.station, g, id: g.station }))],
  ];
  const table = html("table", { className: "vtable" });
  const head = table.createTHead().insertRow();
  ["", "Forecasts", "Persistence error", "MLP error", "MLP bias", "Skill vs persistence", "Verdict"]
    .forEach((t) => head.appendChild(html("th", {}, t)));
  const body = table.createTBody();
  for (const [title, rows] of groups) {
    const tr = body.insertRow();
    const th = html("th", { colSpan: 7, className: "group" }, title);
    tr.appendChild(th);
    for (const { name, g, id } of rows) {
      const r = body.insertRow();
      if (id && state.station && state.station.id === id) r.className = "selected";
      const m = g.methods.mlp, p = g.methods.persistence;
      const [lo, hi] = g.mlp_skill_mae_95;
      r.insertCell().textContent = name;
      r.insertCell().textContent = g.n.toLocaleString("en-US");
      r.insertCell().textContent = p.mae.toFixed(3) + " m";
      r.insertCell().textContent = m.mae.toFixed(3) + " m";
      r.insertCell().textContent = (m.bias >= 0 ? "+" : "") + m.bias.toFixed(3) + " m";
      const c = r.insertCell();
      c.appendChild(skillBar(lo, hi, m.skill_mae));
      c.appendChild(html("span", { className: "muted" }, ` ${pct(m.skill_mae)} (${pct(lo)} to ${pct(hi)})`));
      r.insertCell().textContent = g.verdict;
    }
  }
  box.appendChild(table);
  box.appendChild(html("p", { className: "muted" },
    "Error is the average miss in metres (MAE). Bias is forecast minus observed, so negative means too low. " +
    "The dot is the skill, the bar its 95% range, the vertical line is zero (no better than persistence)."));
}

function renderQcSummary() {
  const rows = state.qc.stations.by_test.filter((r) => r.test && r.test !== "overall");
  const tests = ["gross_range", "spike", "rate_of_change", "flat_line", "gap"];
  const names = { gross_range: "Gross range", spike: "Spike", rate_of_change: "Rate of change", flat_line: "Flat line", gap: "Gap" };
  const vars = [...new Set(rows.map((r) => r.variable))];
  const d = state.qc.drifters;
  const driftFlagged = d.by_test.filter((r) => r.test !== "overall").reduce((a, r) => a + r.n_suspect + r.n_fail, 0);
  $("qcnote").textContent =
    `Readings flagged suspect or failed, out of ${state.qc.stations.rows.toLocaleString("en-US")} station-hours from 2022 to 2024. ` +
    `The ${d.rows.toLocaleString("en-US")} positions of the five drifting buoys (purple lines on the map, where the position changes every row) had ${driftFlagged} flagged: their data was already quality controlled by its publisher. ` +
    `When I planted known faults, the tests found ${Object.values(state.qc.injection_test.detected).reduce((a, b) => a + b, 0)} of ${Object.values(state.qc.injection_test.planted).reduce((a, b) => a + b, 0)} on the buoys and ` +
    `${Object.values(state.qc.injection_test.drifter_detected).reduce((a, b) => a + b, 0)} of ${Object.values(state.qc.injection_test.drifter_planted).reduce((a, b) => a + b, 0)} on the drifter tracks (obvious faults only).`;
  const table = html("table", { className: "vtable" });
  const head = table.createTHead().insertRow();
  ["Variable", ...tests.map((t) => names[t])].forEach((t) => head.appendChild(html("th", {}, t)));
  const body = table.createTBody();
  for (const v of vars) {
    const r = body.insertRow();
    r.insertCell().textContent = VAR_LABEL[v] || { gst: "gust", dpd: "wave period", atmp: "air temp" }[v] || v;
    for (const t of tests) {
      const x = rows.find((q) => q.variable === v && q.test === t);
      const n = x ? x.n_suspect + x.n_fail : 0;
      r.insertCell().textContent = n.toLocaleString("en-US");
    }
  }
  $("qctable").replaceChildren(table);
}

init().catch((e) => ($("hint").textContent = "Could not load stations: " + e.message));
