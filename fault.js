/* =====================================================================
   FAULT DETECTION  (fault.js)
   Self-contained: include AFTER script.js in dashboard.html.
   Needs: PapaParse + Chart.js (already loaded in dashboard.html).
   Expected CSV columns (names are auto-detected, case-insensitive):
   DateTime, Output_V, Input_V, Current_mA, Temperature_C, [MOSFET], [Fault_Status]
   ===================================================================== */

// ---- Thresholds: change these to match your hardware -----------------
const FAULT_CFG = {
  outVMin: 11.0,      // V   below  -> output under-voltage
  outVMax: 12.5,      // V   above  -> output over-voltage
  inVMin: 1.0,        // V   below  -> input supply lost
  inVMax: 19.0,       // V   above  -> input surge / over-voltage
  curMax: 150,        // mA  above  -> over-current
  curShort: 1000,     // mA  above  -> short-circuit / spike (critical)
  tempMax: 40,        // °C  above  -> over-temperature
  tempCritical: 80,   // °C  above  -> critical over-temperature
  tempSensorMin: 0,   // °C  below  -> invalid sensor reading
  tempSensorMax: 100, // °C  above  -> invalid sensor reading
};

// ---- Fault definitions: where (param) + why (cause text) -------------
const FAULT_TYPES = {
  SHORT_CIRCUIT: { label: "Short-circuit / current spike", param: "Current", group: "current", severity: "critical",
    test: (r, c) => r.cur > c.curShort,
    why: r => `Current jumped to ${r.cur.toFixed(0)} mA, far beyond any normal load. Likely a short circuit, a MOSFET breakdown, or a current-sensor glitch.` },
  OVERCURRENT: { label: "Over-current", param: "Current", group: "current", severity: "warning",
    test: (r, c) => r.cur > c.curMax && r.cur <= c.curShort,
    why: r => {
      let s = `Load current (${r.cur.toFixed(0)} mA) exceeded the ${FAULT_CFG.curMax} mA limit.`;
      if (r.inV > FAULT_CFG.inVMax) s += " It coincides with an input-voltage surge.";
      else if (r.outV < FAULT_CFG.outVMin) s += " Output voltage is also sagging, so the load is too heavy for the converter.";
      else s += " Probable overload or a load-switching transient.";
      return s; } },
  REVERSE_CURRENT: { label: "Negative / reverse current", param: "Current", group: "current", severity: "warning",
    test: r => r.cur < 0,
    why: r => `Current reads ${r.cur.toFixed(1)} mA (negative). Indicates reverse current flow or a current-sensor offset/calibration error.` },
  OUT_UNDERVOLT: { label: "Output under-voltage", param: "Output_V", group: "voltage", severity: "warning",
    test: (r, c) => r.outV < c.outVMin,
    why: r => {
      let s = `Output dropped to ${r.outV.toFixed(2)} V (limit ${FAULT_CFG.outVMin} V).`;
      if (r.inV < FAULT_CFG.inVMin) s += " Input supply is missing at the same time.";
      else if (r.cur > 100) s += " High load current is pulling the output down (overload).";
      else s += " Input is healthy, so check duty-cycle/feedback control, or the load being too heavy.";
      return s; } },
  OUT_OVERVOLT: { label: "Output over-voltage", param: "Output_V", group: "voltage", severity: "critical",
    test: (r, c) => r.outV > c.outVMax,
    why: r => `Output rose to ${r.outV.toFixed(2)} V (limit ${FAULT_CFG.outVMax} V). Possible feedback-loop failure or sudden load removal; can damage the load.` },
  INPUT_LOSS: { label: "Input supply lost", param: "Input_V", group: "voltage", severity: "warning",
    test: (r, c) => r.inV < c.inVMin,
    why: r => `Input voltage is ${r.inV.toFixed(2)} V while output stays at ${r.outV.toFixed(2)} V. The source is disconnected or unstable (loose connector / source dropout / input-sense wire), and the output is running from stored energy or a backup.` },
  INPUT_SURGE: { label: "Input over-voltage / surge", param: "Input_V", group: "voltage", severity: "warning",
    test: (r, c) => r.inV > c.inVMax,
    why: r => `Input reached ${r.inV.toFixed(2)} V (limit ${FAULT_CFG.inVMax} V; nominal is about 18 V). Source surge or unstable supply, which stresses the MOSFET.` },
  TEMP_SENSOR: { label: "Invalid temperature reading", param: "Temperature", group: "temp", severity: "warning",
    test: (r, c) => r.temp < c.tempSensorMin || r.temp > c.tempSensorMax,
    why: r => `Temperature reads ${r.temp.toFixed(0)} °C, which is physically unrealistic for this board. Sensor disconnected, shorted, or noisy wiring. Check the sensor.` },
  OVERTEMP: { label: "Over-temperature", param: "Temperature", group: "temp", severity: "warning",
    test: (r, c) => r.temp > c.tempMax && r.temp <= c.tempSensorMax,
    why: r => `Temperature reached ${r.temp.toFixed(0)} °C (limit ${FAULT_CFG.tempMax} °C). Causes: sustained over-current, poor heat-sinking/airflow, or high switching losses.` },
  MOSFET_FAULT: { label: "MOSFET fault", param: "MOSFET", group: "current", severity: "critical",
    test: r => r.mosfet === "FAULT",
    why: r => `MOSFET flagged FAULT (current ${r.cur.toFixed(0)} mA, ${r.temp.toFixed(0)} °C, input ${r.inV.toFixed(1)} V). Typically caused by over-current, input surge, or overheating of the switch. Check gate drive and the switch.` },
};

// ---- Helpers -----------------------------------------------------------
function fEsc(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
function fPad(n) { return String(n).padStart(2, "0"); }
function fFmtTime(d) {
  return `${d.getFullYear()}-${fPad(d.getMonth() + 1)}-${fPad(d.getDate())} ${fPad(d.getHours())}:${fPad(d.getMinutes())}:${fPad(d.getSeconds())}`;
}
function fFmtDur(sec) {
  if (!isFinite(sec) || sec <= 0) return "instant";
  if (sec < 60) return Math.round(sec) + " s";
  if (sec < 3600) return Math.floor(sec / 60) + " min " + Math.round(sec % 60) + " s";
  return Math.floor(sec / 3600) + " h " + Math.round((sec % 3600) / 60) + " min";
}
function fParseTime(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/.exec(String(s || "").trim());
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
}

// ---- Core analysis (pure function, no DOM) ------------------------------
function analyzeFaults(csvText, cfg = FAULT_CFG) {
  const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true, transformHeader: h => h.trim() });
  if (!parsed.data.length) throw new Error("CSV is empty.");
  const headers = parsed.meta.fields || [];
  const find = re => headers.find(h => re.test(h));

  const col = {
    time: find(/time|date/i),
    outV: find(/^out|v_?out/i),
    inV: find(/^in(put)?[_ ]?v|v_?in/i),
    cur: find(/current|_ma$|^i_/i),
    temp: find(/temp/i),
    mosfet: find(/mosfet/i),
    status: find(/fault|status/i),
  };
  const missing = ["outV", "inV", "cur", "temp"].filter(k => !col[k]);
  if (missing.length)
    throw new Error("Could not find column(s) for: " + missing.join(", ") + ". Expected names like Output_V, Input_V, Current_mA, Temperature_C.");

  // Build rows
  const num = v => { const x = parseFloat(String(v).trim()); return isNaN(x) ? NaN : x; };
  const rows = [];
  parsed.data.forEach(d => {
    const r = {
      outV: num(d[col.outV]), inV: num(d[col.inV]), cur: num(d[col.cur]), temp: num(d[col.temp]),
      mosfet: col.mosfet ? String(d[col.mosfet] || "").trim().toUpperCase() : "",
      device: col.status ? String(d[col.status] || "").trim() : "",
      time: col.time ? fParseTime(d[col.time]) : null,
      est: false,
    };
    if ([r.outV, r.inV, r.cur, r.temp].every(isNaN)) return; // skip junk rows
    rows.push(r);
  });
  if (rows.length < 2) throw new Error("Not enough valid data rows.");

  // Fill missing timestamps (e.g. "NO_TIME") by estimating from the previous row
  const dts = [];
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].time && rows[i - 1].time) {
      const dt = (rows[i].time - rows[i - 1].time) / 1000;
      if (dt > 0 && dt < 600) dts.push(dt);
    }
  }
  dts.sort((a, b) => a - b);
  const medDt = dts.length ? dts[Math.floor(dts.length / 2)] : 1;
  let noTimeRows = 0;
  for (let i = 0; i < rows.length; i++) {
    if (!rows[i].time) {
      noTimeRows++;
      if (i > 0 && rows[i - 1].time) { rows[i].time = new Date(rows[i - 1].time.getTime() + medDt * 1000); rows[i].est = true; }
    }
  }

  // Detect faults per row
  rows.forEach(r => {
    r.faults = Object.keys(FAULT_TYPES).filter(k => FAULT_TYPES[k].test(r, cfg));
  });

  // Merge consecutive rows with the same fault into events
  const events = [];
  const open = {};
  rows.forEach((r, i) => {
    const active = new Set(r.faults);
    Object.keys(open).forEach(k => { if (!active.has(k)) { events.push(open[k]); delete open[k]; } });
    r.faults.forEach(k => {
      if (!open[k]) open[k] = { type: k, start: i, end: i, rows: [] };
      open[k].end = i;
      open[k].rows.push(r);
    });
  });
  Object.keys(open).forEach(k => events.push(open[k]));
  events.sort((a, b) => a.start - b.start || a.type.localeCompare(b.type));

  events.forEach(e => {
    const T = FAULT_TYPES[e.type];
    const first = rows[e.start], last = rows[e.end];
    e.startTime = first.time; e.endTime = last.time;
    e.est = first.est || last.est;
    e.durationSec = first.time && last.time ? (last.time - first.time) / 1000 + medDt : NaN;
    // worst row = furthest from normal for this fault
    const score = r => {
      switch (e.type) {
        case "SHORT_CIRCUIT": case "OVERCURRENT": return r.cur;
        case "REVERSE_CURRENT": return -r.cur;
        case "OUT_UNDERVOLT": return -r.outV;
        case "OUT_OVERVOLT": return r.outV;
        case "INPUT_LOSS": return -r.inV;
        case "INPUT_SURGE": return r.inV;
        case "OVERTEMP": case "TEMP_SENSOR": return Math.abs(r.temp - 32);
        default: return r.cur;
      }
    };
    e.worst = e.rows.reduce((a, b) => (score(b) > score(a) ? b : a), e.rows[0]);
    e.why = T.why(e.worst);
    const dev = {};
    e.rows.forEach(r => { if (r.device && r.device.toUpperCase() !== "NORMAL") dev[r.device] = (dev[r.device] || 0) + 1; });
    e.deviceFlag = Object.keys(dev).sort((a, b) => dev[b] - dev[a])[0] || "";
  });

  const faultyRows = rows.filter(r => r.faults.length).length;
  const byType = {};
  events.forEach(e => {
    byType[e.type] = byType[e.type] || { events: 0, rows: 0, first: e.startTime, last: e.endTime };
    byType[e.type].events++; byType[e.type].rows += e.rows.length;
    if (e.startTime && (!byType[e.type].first || e.startTime < byType[e.type].first)) byType[e.type].first = e.startTime;
    if (e.endTime && (!byType[e.type].last || e.endTime > byType[e.type].last)) byType[e.type].last = e.endTime;
  });

  return { rows, events, byType, faultyRows, noTimeRows, medDt, hasDevice: !!col.status };
}

// ---- UI ------------------------------------------------------------------
const faultCharts = {};
function faultTimeLabel(r, i) { return r.time ? fFmtTime(r.time).slice(5) : "#" + (i + 1); }

function drawFaultChart(canvasId, title, rows, lines, markers, yOpts) {
  if (faultCharts[canvasId]) faultCharts[canvasId].destroy();
  const labels = rows.map(faultTimeLabel);
  const datasets = lines.map(l => ({
    label: l.label, data: rows.map(l.get), borderColor: l.color, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0,
  }));
  markers.forEach(m => datasets.push({
    label: m.label, data: rows.map(r => (r.faults.some(f => m.types.includes(f)) ? m.get(r) : null)),
    borderColor: m.color, backgroundColor: m.color, showLine: false, pointRadius: 4, pointStyle: "rectRot",
  }));
  faultCharts[canvasId] = new Chart(document.getElementById(canvasId), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, animation: false,
      plugins: { title: { display: true, text: title } },
      scales: { x: { ticks: { maxTicksLimit: 10, maxRotation: 0 } }, y: yOpts || {} },
    },
  });
}

function handleFault(csvText, statusEl) {
  const res = analyzeFaults(csvText);
  const { rows, events, byType, faultyRows, noTimeRows } = res;

  // Summary cards
  document.getElementById("fault-total").textContent = rows.length;
  document.getElementById("fault-rows").textContent = faultyRows;
  document.getElementById("fault-events").textContent = events.length;
  document.getElementById("fault-health").textContent = ((1 - faultyRows / rows.length) * 100).toFixed(1) + "%";
  document.getElementById("fault-results").style.display = "block";

  // Breakdown by fault type (sorted by severity then count)
  const sevOrder = { critical: 0, warning: 1 };
  const typeKeys = Object.keys(byType).sort((a, b) =>
    sevOrder[FAULT_TYPES[a].severity] - sevOrder[FAULT_TYPES[b].severity] || byType[b].rows - byType[a].rows);
  const summaryEl = document.getElementById("fault-summary");
  if (!typeKeys.length) {
    summaryEl.innerHTML = `<div class="suggestion-item success"><strong>No faults found.</strong> All ${rows.length} samples are within the configured limits.</div>`;
  } else {
    summaryEl.innerHTML = `<h4>Faults by type</h4><div class="suggestion-list">` + typeKeys.map(k => {
      const T = FAULT_TYPES[k], b = byType[k];
      const cls = T.severity === "critical" ? "warning" : "info";
      return `<div class="suggestion-item ${cls}"><strong>${fEsc(T.label)}</strong> &mdash; ${b.events} event(s), ${b.rows} sample(s)
        <br>Where: <strong>${fEsc(T.param)}</strong> &nbsp;|&nbsp; First: ${b.first ? fFmtTime(b.first) : "n/a"} &nbsp;|&nbsp; Last: ${b.last ? fFmtTime(b.last) : "n/a"}</div>`;
    }).join("") + `</div>`;
  }

  // Charts
  const inFaultRows = g => ({ types: Object.keys(FAULT_TYPES).filter(k => FAULT_TYPES[k].group === g) });
  drawFaultChart("faultChartVoltage", "Input & Output Voltage (markers = faults)", rows,
    [{ label: "Input V", get: r => r.inV, color: "#3b82f6" }, { label: "Output V", get: r => r.outV, color: "#22c55e" }],
    [{ label: "Voltage fault (Output V)", types: inFaultRows("voltage").types.filter(t => t.startsWith("OUT")), get: r => r.outV, color: "#ef4444" },
     { label: "Voltage fault (Input V)", types: inFaultRows("voltage").types.filter(t => t.startsWith("INPUT")), get: r => r.inV, color: "#f59e0b" }]);

  const maxCur = Math.max(...rows.map(r => r.cur).filter(isFinite));
  const clipped = maxCur > 400;
  drawFaultChart("faultChartCurrent", "Current (mA)" + (clipped ? ` (axis clipped at 400; peak ${maxCur.toFixed(0)} mA)` : "") + " - markers = faults", rows,
    [{ label: "Current mA", get: r => r.cur, color: "#8b5cf6" }],
    [{ label: "Current fault", types: ["SHORT_CIRCUIT", "OVERCURRENT", "REVERSE_CURRENT"], get: r => r.cur, color: "#ef4444" },
     { label: "MOSFET fault", types: ["MOSFET_FAULT"], get: r => r.cur, color: "#f97316" }],
    clipped ? { max: 400 } : {});

  drawFaultChart("faultChartTemp", "Temperature (°C) - markers = faults", rows,
    [{ label: "Temperature °C", get: r => r.temp, color: "#f97316" }],
    [{ label: "Temperature fault", types: ["OVERTEMP", "TEMP_SENSOR"], get: r => r.temp, color: "#ef4444" }]);

  // Event table (first 300 events)
  const tbody = document.getElementById("fault-table-body");
  const shown = events.slice(0, 300);
  tbody.innerHTML = shown.map(e => {
    const T = FAULT_TYPES[e.type];
    const t1 = e.startTime ? fFmtTime(e.startTime) + (rows[e.start].est ? " (est.)" : "") : "unknown";
    const t2 = e.endTime ? fFmtTime(e.endTime).slice(11) : "";
    const when = e.start === e.end ? t1 : `${t1} &rarr; ${t2}`;
    return `<tr class="sev-${T.severity}">
      <td>${when}</td>
      <td>${fEsc(T.label)}<br><small>${T.severity.toUpperCase()}</small></td>
      <td>${fEsc(T.param)}</td>
      <td>${e.rows.length} (${fFmtDur(e.durationSec)})</td>
      <td>Out ${e.worst.outV.toFixed(2)} V, In ${e.worst.inV.toFixed(2)} V, ${e.worst.cur.toFixed(1)} mA, ${e.worst.temp.toFixed(0)} °C</td>
      <td>${fEsc(e.why)}${e.deviceFlag ? `<br><small>Device flag: ${fEsc(e.deviceFlag)}</small>` : ""}</td></tr>`;
  }).join("");
  document.getElementById("fault-table-note").textContent =
    events.length > shown.length ? `Showing first ${shown.length} of ${events.length} events.` : "";

  // Notes
  const notes = [];
  if (noTimeRows) notes.push(`${noTimeRows} row(s) had no valid timestamp; their time is estimated from the previous row and marked "(est.)".`);
  notes.push(`Thresholds used: Output ${FAULT_CFG.outVMin}-${FAULT_CFG.outVMax} V, Input ${FAULT_CFG.inVMin}-${FAULT_CFG.inVMax} V, Current &le; ${FAULT_CFG.curMax} mA, Temp ${FAULT_CFG.tempSensorMin}-${FAULT_CFG.tempMax} &deg;C (edit FAULT_CFG in fault.js).`);
  document.getElementById("fault-notes").innerHTML = notes.map(n => `<div class="suggestion-item info">${n}</div>`).join("");

  statusEl.textContent = `Fault analysis complete: ${events.length} fault event(s) in ${rows.length} samples.`;
  statusEl.className = "status-msg success";
}

// ---- Upload wiring (independent of script.js) -----------------------------
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("file-fault");
    const area = document.getElementById("upload-area-fault");
    const statusEl = document.getElementById("fault-status");
    if (!input || !area || !statusEl) return;

    function run(file) {
      if (!file.name.toLowerCase().endsWith(".csv")) {
        statusEl.textContent = "Please upload a CSV file."; statusEl.className = "status-msg error"; return;
      }
      statusEl.textContent = "Analyzing " + file.name + "..."; statusEl.className = "status-msg loading";
      const reader = new FileReader();
      reader.onload = e => {
        try { handleFault(e.target.result, statusEl); }
        catch (err) { statusEl.textContent = "Error: " + err.message; statusEl.className = "status-msg error"; }
      };
      reader.readAsText(file);
    }
    input.addEventListener("change", e => { if (e.target.files.length) run(e.target.files[0]); });
    area.addEventListener("dragover", e => { e.preventDefault(); area.classList.add("drag-over"); });
    area.addEventListener("dragleave", () => area.classList.remove("drag-over"));
    area.addEventListener("drop", e => { e.preventDefault(); area.classList.remove("drag-over"); if (e.dataTransfer.files.length) run(e.dataTransfer.files[0]); });
  });
}

if (typeof module !== "undefined") module.exports = { analyzeFaults, FAULT_TYPES, FAULT_CFG };
