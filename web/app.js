const state = { station: null, stations: [] };
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

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

async function init() {
  state.stations = await getJSON("/stations");
  const pts = [];
  for (const s of state.stations) {
    const m = L.marker([s.lat, s.lon]).addTo(map);
    m.bindTooltip(s.id);
    m.on("click", () => select(s));
    pts.push([s.lat, s.lon]);
  }
  map.fitBounds(pts, { padding: [30, 30] });
  $("horizon").addEventListener("change", refresh);
  $("end").addEventListener("change", refresh);
}

function select(s) {
  state.station = s;
  $("hint").hidden = true;
  $("detail").hidden = false;
  $("name").textContent = `${s.id}: ${s.name}`;
  $("meta").textContent = `Data from ${s.first_obs.slice(0, 10)} to ${s.last_obs.slice(0, 10)} (UTC)`;
  $("end").value = toInput(s.last_obs);
  refresh();
}

async function refresh() {
  const s = state.station;
  if (!s) return;
  const end = toApi($("end").value);
  const h = $("horizon").value;
  $("note").textContent = "";
  try {
    const [fc, bt] = await Promise.all([
      getJSON(`/forecast/${s.id}?as_of=${end}`).catch(() => null),
      getJSON(`/backtest/${s.id}?horizon=${h}&hours=72&end=${end}`),
    ]);
    renderForecast(fc);
    renderChart(bt.rows);
  } catch (e) {
    $("note").textContent = "Could not load data: " + e.message;
  }
}

const fmt = (x) => (x == null ? "n/a" : x.toFixed(2) + " m");

function renderForecast(fc) {
  const body = document.querySelector("#fc tbody");
  body.replaceChildren();
  if (!fc) {
    const tr = body.insertRow();
    const td = tr.insertCell();
    td.colSpan = 4;
    td.textContent = "No complete observations at that hour.";
    return;
  }
  for (const f of fc.forecasts) {
    const tr = body.insertRow();
    [f.valid_time.replace("T", " ").slice(0, 16), fmt(f.mlp), fmt(f.linear), fmt(f.persistence)]
      .forEach((t) => (tr.insertCell().textContent = t));
  }
}

function el(name, attrs, text) {
  const e = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

function renderChart(rows) {
  const svg = $("chart");
  svg.replaceChildren();
  const W = 640, H = 300, m = { l: 44, r: 12, t: 12, b: 28 };
  if (!rows.length) {
    svg.appendChild(el("text", { x: 20, y: 30 }, "No forecast/observation pairs in this window."));
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
    svg.appendChild(el("path", { class: "l " + cls, d }));
  };
  line("persistence", "per");
  line("mlp", "mlp");
  line("observed", "obs");
  $("note").textContent =
    `Each forecast is plotted at the time it is for, and was made ${$("horizon").value} hours earlier.`;
}

init().catch((e) => ($("hint").textContent = "Could not load stations: " + e.message));
