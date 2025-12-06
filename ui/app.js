const API_BASE = "";

let chart = null;
function renderChart(payload) {
  const el = document.getElementById("chart");
  if (!el) return;
  const ctx = el.getContext("2d");
  if (chart) { chart.destroy(); chart = null; }

  if (payload.mode === "prophet" && payload.rows) {
    const labels = payload.rows.map(r => r.ds);
    const yhat = payload.rows.map(r => r.yhat);
    const lower = payload.rows.map(r => r.yhat_lower);
    const upper = payload.rows.map(r => r.yhat_upper);
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {label: "Forecast", data: yhat, tension: 0.2},
          {label: "Lower", data: lower, borderDash: [5,5], pointRadius: 0},
          {label: "Upper", data: upper, borderDash: [5,5], pointRadius: 0}
        ]
      },
      options: {responsive: true, scales: {y: {beginAtZero: false}}}
    });
  } else if (payload.dates && payload.yhat) {
    const labels = payload.dates;
    const yhat = payload.yhat;
    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets: [{label: "Forecast", data: yhat, tension: 0.2}]},
      options: {responsive: true, scales: {y: {beginAtZero: false}}}
    });
  }
}

async function postJSON(path, body) {
  const res = await fetch((API_BASE || "") + path, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return res.json();
}

async function getJSON(path) {
  const res = await fetch((API_BASE || "") + path);
  return res.json();
}

async function populateSeries() {
  const sel = document.getElementById("seriesSelect");
  sel.innerHTML = "<option>Loading...</option>";
  try {
    const data = await getJSON("/series?top=100");
    sel.innerHTML = "";
    data.series.forEach(s => {
      const opt = document.createElement("option");
      opt.value = `${s.Store},${s.Dept}`;
      opt.textContent = `Store ${s.Store} / Dept ${s.Dept}  (avg=${s.avg.toFixed(2)})`;
      sel.appendChild(opt);
    });
  } catch (e) {
    sel.innerHTML = "<option>Error loading series</option>";
  }
}

document.getElementById("applySeries").addEventListener("click", () => {
  const sel = document.getElementById("seriesSelect").value;
  if (!sel) return;
  const [st, dp] = sel.split(",").map(Number);
  document.getElementById("store").value = st;
  document.getElementById("dept").value = dp;
});

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const store = +document.getElementById("store").value;
  const dept = +document.getElementById("dept").value;
  const horizon = +document.getElementById("horizon").value;
  const mode = document.getElementById("mode").value;
  const out = document.getElementById("output");
  out.textContent = "Requesting forecast...";
  try {
    const data = await postJSON("/forecast", {store, dept, horizon, mode});
    out.textContent = JSON.stringify(data, null, 2);
    renderChart(data);
  } catch (err) {
    out.textContent = "Error: " + err.message;
  }
});

document.getElementById("train").addEventListener("click", async () => {
  const out = document.getElementById("output");
  out.textContent = "Training RF model...";
  const data = await postJSON("/train", {force: true});
  out.textContent = JSON.stringify(data, null, 2);
});

document.getElementById("loadSeries").addEventListener("click", async () => {
  const out = document.getElementById("output");
  out.textContent = "Loading top series...";
  await populateSeries();
  out.textContent = "Top series loaded. Pick one from the dropdown and click Apply.";
});

// Load series list on page start
window.addEventListener("DOMContentLoaded", populateSeries);
