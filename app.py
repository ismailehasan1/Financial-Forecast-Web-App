"""
Flask Application — LSTM Forecast API + Web UI
===============================================
Endpoints:
  GET  /                          → dashboard UI
  POST /api/train                 → train a new model
  POST /api/forecast              → run forecast from loaded pipeline
  GET  /api/models                → list available .pkl files
  GET  /api/download/<pkl_name>   → download a .pkl pipeline
  GET  /api/status/<job_id>       → poll training job status

Run:
  python app.py
  # or production:
  gunicorn app:app --workers 2 --timeout 300
"""

import io
import os
import json
import uuid
import pickle
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Flask, request, jsonify, send_file,
                   render_template_string, abort)

# ─── App config ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("static/plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store (use Redis/Celery in production)
JOBS: dict = {}
JOBS_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def list_models() -> list:
    return [
        {
            "filename": p.name,
            "size_mb": round(p.stat().st_size / 1_048_576, 2),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in sorted(MODELS_DIR.glob("*.pkl"))
    ]


def get_pipeline_meta(pkl_path: Path) -> dict:
    """Load only the metadata (no model bytes) for fast listing."""
    with open(pkl_path, "rb") as f:
        pl = pickle.load(f)
    return {k: v for k, v in pl.items() if k != "model_bytes"}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TRAINING WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _train_worker(job_id: str, ticker: str, horizon: int, lookback: int,
                  start: str, epochs: int, batch: int):
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = datetime.now().isoformat()

    try:
        from train_model import train as train_model
        pkl_path, fc_out, metrics = train_model(
            ticker=ticker,
            horizon=horizon,
            lookback=lookback,
            start=start,
            epochs=epochs,
            batch_size=batch,
            out_dir=str(MODELS_DIR),
            plots_dir=str(PLOTS_DIR),
        )
        # Serialise forecast for JSON (convert dates + np arrays)
        fc_serialised = {
            k: (v.tolist() if hasattr(v, "tolist") else [str(d) for d in v])
            for k, v in fc_out.items()
        }
        with JOBS_LOCK:
            JOBS[job_id].update({
                "status":      "done",
                "pkl_path":    str(pkl_path),
                "pkl_name":    Path(pkl_path).name,
                "metrics":     metrics,
                "forecast":    fc_serialised,
                "finished_at": datetime.now().isoformat(),
            })
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id].update({
                "status":      "error",
                "error":       str(e),
                "traceback":   traceback.format_exc(),
                "finished_at": datetime.now().isoformat(),
            })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Start a background training job.
    Body (JSON):
      ticker   : str   — e.g. "AAPL"
      horizon  : int   — 7 | 15 | 30 | 90
      lookback : int   — default 60
      start    : str   — "YYYY-MM-DD"
      epochs   : int   — default 100
      batch    : int   — default 32
    Returns: { job_id, status }
    """
    body    = request.get_json(force=True) or {}
    ticker  = body.get("ticker",  "AAPL").upper()
    horizon = int(body.get("horizon",  30))
    lookback= int(body.get("lookback", 60))
    start   = body.get("start", "2018-01-01")
    epochs  = int(body.get("epochs",  100))
    batch   = int(body.get("batch",    32))

    if horizon not in [7, 15, 30, 90]:
        return jsonify({"error": "horizon must be 7, 15, 30, or 90"}), 400

    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status":     "queued",
            "ticker":     ticker,
            "horizon":    horizon,
            "created_at": datetime.now().isoformat(),
        }

    t = threading.Thread(
        target=_train_worker,
        args=(job_id, ticker, horizon, lookback, start, epochs, batch),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "status": "queued",
                    "poll": f"/api/status/{job_id}"}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id: str):
    """Poll training job status."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """
    Run a forecast from a saved pipeline.
    Body (JSON):
      pkl_name : str  — filename in models/ dir
      date     : str  — "YYYY-MM-DD" (default: today)
      mc       : int  — MC-dropout samples (default: 200)
    """
    body     = request.get_json(force=True) or {}
    pkl_name = body.get("pkl_name")
    as_of    = body.get("date", datetime.today().strftime("%Y-%m-%d"))
    n_mc     = int(body.get("mc", 200))

    if not pkl_name:
        return jsonify({"error": "pkl_name is required"}), 400

    pkl_path = MODELS_DIR / pkl_name
    if not pkl_path.exists():
        return jsonify({"error": f"{pkl_name} not found in models/"}), 404

    try:
        from run_forecast import run_forecast
        results = run_forecast(
            pkl_path=str(pkl_path),
            as_of_date=as_of,
            n_mc=n_mc,
            plot_dir=str(PLOTS_DIR),
        )
        # Make JSON-serialisable
        fc = results["forecast"]
        response = {
            "ticker":     results["ticker"],
            "horizon":    results["horizon"],
            "as_of_date": results["as_of_date"],
            "trained_at": results["trained_at"],
            "metrics":    results["metrics"],
            "chart_url":  "/" + results["chart_path"].replace("\\", "/"),
            "forecast": {
                "dates": [str(d)[:10] for d in fc["dates"]],
                "mean":  [round(float(v), 4) for v in fc["mean"]],
                "p5":    [round(float(v), 4) for v in fc["p5"]],
                "p25":   [round(float(v), 4) for v in fc["p25"]],
                "p75":   [round(float(v), 4) for v in fc["p75"]],
                "p95":   [round(float(v), 4) for v in fc["p95"]],
            },
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/models", methods=["GET"])
def api_models():
    """List all saved pipeline .pkl files."""
    models = list_models()
    # Attach training metadata
    enriched = []
    for m in models:
        try:
            meta = get_pipeline_meta(MODELS_DIR / m["filename"])
            m.update({
                "ticker":     meta.get("ticker"),
                "horizon":    meta.get("horizon"),
                "metrics":    meta.get("metrics"),
                "trained_at": meta.get("trained_at"),
                "tf_version": meta.get("tf_version"),
            })
        except Exception:
            pass
        enriched.append(m)
    return jsonify(enriched)


@app.route("/api/download/<pkl_name>", methods=["GET"])
def api_download(pkl_name: str):
    """Download a pipeline pickle file."""
    pkl_path = MODELS_DIR / pkl_name
    if not pkl_path.exists():
        abort(404)
    return send_file(
        str(pkl_path),
        as_attachment=True,
        download_name=pkl_name,
        mimetype="application/octet-stream",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — WEB UI
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LSTM Forecast Dashboard</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--border:#21262d;--text:#c9d1d9;--muted:#8b949e;
        --blue:#4d9cf8;--green:#1dce8c;--amber:#f0a05a;--red:#f85149;--radius:8px}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'SF Mono',Consolas,monospace;font-size:14px;min-height:100vh}
  header{border-bottom:1px solid var(--border);padding:16px 32px;display:flex;align-items:center;gap:12px}
  header h1{font-size:18px;color:white;font-weight:600}
  header span{color:var(--muted);font-size:12px}
  .main{max-width:1200px;margin:0 auto;padding:24px 32px;display:grid;grid-template-columns:1fr 1fr;gap:24px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
  .card h2{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:16px}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px;margin-top:12px}
  input,select{width:100%;background:#0d1117;border:1px solid var(--border);border-radius:6px;
               color:var(--text);padding:8px 10px;font-size:13px;font-family:inherit}
  input:focus,select:focus{outline:none;border-color:var(--blue)}
  button{margin-top:16px;width:100%;padding:10px;border-radius:6px;border:none;cursor:pointer;
         font-size:13px;font-weight:600;font-family:inherit}
  .btn-primary{background:var(--blue);color:#0d1117}
  .btn-primary:hover{opacity:.88}
  .btn-secondary{background:var(--green);color:#0d1117}
  .btn-secondary:hover{opacity:.88}
  .btn-dl{background:transparent;border:1px solid var(--border);color:var(--muted);
           padding:5px 10px;font-size:11px;width:auto;margin:0}
  .status{margin-top:12px;padding:10px 12px;border-radius:6px;font-size:12px;display:none;
          background:#161b22;border:1px solid var(--border)}
  .status.visible{display:block}
  .tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
  .tag.done{background:rgba(29,206,140,.15);color:var(--green)}
  .tag.running{background:rgba(77,156,248,.12);color:var(--blue)}
  .tag.error{background:rgba(248,81,73,.12);color:var(--red)}
  .tag.queued{background:rgba(240,160,90,.12);color:var(--amber)}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{color:var(--muted);padding:6px 8px;text-align:left;border-bottom:1px solid var(--border);font-weight:500}
  td{padding:6px 8px;border-bottom:1px solid rgba(33,38,45,.5);color:var(--text)}
  .span2{grid-column:span 2}
  .chart-container{margin-top:16px;text-align:center}
  .chart-container img{max-width:100%;border-radius:var(--radius);border:1px solid var(--border)}
  .metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
  .metric{background:#0d1117;border-radius:6px;padding:10px;text-align:center}
  .metric .val{font-size:18px;font-weight:700;color:var(--green)}
  .metric .lbl{font-size:10px;color:var(--muted);margin-top:2px}
  pre{white-space:pre-wrap;word-break:break-word;color:var(--muted);font-size:11px;max-height:200px;overflow:auto}
</style>
</head>
<body>
<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#4d9cf8" stroke-width="1.8">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
  <h1>LSTM Forecast</h1>
  <span>Bidirectional LSTM · MC-Dropout · Yahoo Finance</span>
</header>

<div class="main">

  <!-- TRAIN -->
  <div class="card">
    <h2>Train new model</h2>
    <label>Ticker</label>
    <input id="tr-ticker" value="AAPL" placeholder="AAPL, MSFT, TSLA …">
    <label>Forecast horizon</label>
    <select id="tr-horizon">
      <option value="7">7 days</option>
      <option value="15">15 days</option>
      <option value="30" selected>30 days</option>
      <option value="90">3 months (90 days)</option>
    </select>
    <label>Training start date</label>
    <input id="tr-start" value="2018-01-01" type="date">
    <label>Max epochs</label>
    <input id="tr-epochs" value="100" type="number" min="10" max="500">
    <label>Lookback window (days)</label>
    <input id="tr-lookback" value="60" type="number" min="20" max="200">
    <button class="btn-primary" onclick="startTraining()">▶ Start Training</button>
    <div class="status" id="train-status"></div>
  </div>

  <!-- FORECAST -->
  <div class="card">
    <h2>Run forecast</h2>
    <label>Pipeline (.pkl)</label>
    <select id="fc-pkl" onchange="refreshModelSelect()"></select>
    <label>As-of date</label>
    <input id="fc-date" type="date">
    <label>MC-Dropout samples</label>
    <input id="fc-mc" value="200" type="number" min="50" max="1000">
    <button class="btn-secondary" onclick="runForecast()">🔮 Run Forecast</button>
    <div class="status" id="fc-status"></div>
  </div>

  <!-- MODELS TABLE -->
  <div class="card span2">
    <h2>Saved pipelines <button class="btn-dl" onclick="loadModels()" style="margin-left:8px">↻ Refresh</button></h2>
    <table id="models-table">
      <thead><tr><th>File</th><th>Ticker</th><th>Horizon</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Trained</th><th>Size</th><th></th></tr></thead>
      <tbody id="models-tbody"><tr><td colspan="9" style="color:var(--muted);text-align:center">Loading…</td></tr></tbody>
    </table>
  </div>

  <!-- FORECAST RESULTS -->
  <div class="card span2" id="results-card" style="display:none">
    <h2>Forecast results — <span id="res-title"></span></h2>
    <div class="metric-row" id="res-metrics"></div>
    <div class="chart-container" id="res-chart"></div>
    <div style="margin-top:16px;overflow:auto;max-height:300px">
      <table>
        <thead><tr><th>Date</th><th>Forecast</th><th>P5</th><th>P25</th><th>P75</th><th>P95</th></tr></thead>
        <tbody id="res-table"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);

// ── Training ──────────────────────────────────────────────────────────────
async function startTraining(){
  const body = {
    ticker:  $('tr-ticker').value.trim().toUpperCase(),
    horizon: parseInt($('tr-horizon').value),
    start:   $('tr-start').value,
    epochs:  parseInt($('tr-epochs').value),
    lookback:parseInt($('tr-lookback').value),
  };
  showStatus('train-status', `🚀 Queuing training for ${body.ticker} (${body.horizon}d) …`, 'queued');

  const r = await fetch('/api/train', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if(!r.ok){ showStatus('train-status', `❌ ${j.error}`, 'error'); return; }

  showStatus('train-status', `⏳ Job ${j.job_id} queued — polling …`, 'running');
  pollJob(j.job_id);
}

function pollJob(jobId){
  const iv = setInterval(async () => {
    const r = await fetch(`/api/status/${jobId}`);
    const j = await r.json();
    if(j.status === 'running'){
      showStatus('train-status', `⚙️ Training in progress … (job ${jobId})`, 'running');
    } else if(j.status === 'done'){
      clearInterval(iv);
      const mets = j.metrics || {};
      showStatus('train-status',
        `✅ Done! &nbsp;<strong>${j.pkl_name}</strong>&nbsp;` +
        `MAE=${mets.MAE} RMSE=${mets.RMSE} R²=${mets.R2}`, 'done');
      loadModels();
    } else if(j.status === 'error'){
      clearInterval(iv);
      showStatus('train-status', `❌ Error: ${j.error}`, 'error');
    }
  }, 3000);
}

// ── Forecast ─────────────────────────────────────────────────────────────
async function runForecast(){
  const pkl = $('fc-pkl').value;
  if(!pkl){ alert('Select a pipeline first'); return; }
  const body = { pkl_name: pkl, date: $('fc-date').value, mc: parseInt($('fc-mc').value) };
  showStatus('fc-status', `🔮 Running MC-Dropout forecast …`, 'running');

  const r = await fetch('/api/forecast', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if(!r.ok){ showStatus('fc-status', `❌ ${j.error}`, 'error'); return; }

  showStatus('fc-status', `✅ Forecast complete for ${j.ticker} (${j.horizon}d)`, 'done');
  renderResults(j);
}

function renderResults(j){
  $('results-card').style.display = 'block';
  $('res-title').textContent = `${j.ticker} — ${j.horizon}-day from ${j.as_of_date}`;

  const m = j.metrics || {};
  $('res-metrics').innerHTML = [
    ['MAE',  m.MAE], ['RMSE', m.RMSE], ['MAPE%', m.MAPE], ['R²', m.R2]
  ].map(([l,v]) => `<div class="metric"><div class="val">${v??'—'}</div><div class="lbl">${l}</div></div>`).join('');

  $('res-chart').innerHTML = j.chart_url
    ? `<img src="${j.chart_url}?t=${Date.now()}" alt="Forecast chart">`
    : '<p style="color:var(--muted)">No chart generated</p>';

  const fc = j.forecast;
  $('res-table').innerHTML = fc.dates.map((d,i) =>
    `<tr><td>${d}</td><td style="color:var(--green)">$${fc.mean[i]}</td>
    <td>$${fc.p5[i]}</td><td>$${fc.p25[i]}</td><td>$${fc.p75[i]}</td><td>$${fc.p95[i]}</td></tr>`
  ).join('');

  $('results-card').scrollIntoView({behavior:'smooth'});
}

// ── Models ────────────────────────────────────────────────────────────────
async function loadModels(){
  const r = await fetch('/api/models');
  const models = await r.json();

  // Update picker
  const pkl = $('fc-pkl');
  const cur = pkl.value;
  pkl.innerHTML = models.length
    ? models.map(m => `<option value="${m.filename}">${m.filename}</option>`).join('')
    : '<option value="">No models found — train one first</option>';
  if(cur) pkl.value = cur;

  // Update table
  $('models-tbody').innerHTML = models.length
    ? models.map(m => `<tr>
        <td style="color:var(--blue)">${m.filename}</td>
        <td>${m.ticker??'—'}</td>
        <td>${m.horizon??'—'}d</td>
        <td>${m.metrics?.MAE??'—'}</td>
        <td>${m.metrics?.RMSE??'—'}</td>
        <td>${m.metrics?.R2??'—'}</td>
        <td style="color:var(--muted);font-size:11px">${(m.trained_at||'').slice(0,16)}</td>
        <td style="color:var(--muted)">${m.size_mb} MB</td>
        <td><a href="/api/download/${m.filename}"><button class="btn-dl">⬇ Download</button></a></td>
      </tr>`).join('')
    : '<tr><td colspan="9" style="color:var(--muted);text-align:center">No .pkl files found</td></tr>';
}

function refreshModelSelect(){ /* already handled by loadModels */ }

function showStatus(id, html, type){
  const el = $(id);
  el.innerHTML = `<span class="tag ${type}">${type.toUpperCase()}</span> &nbsp;${html}`;
  el.className = 'status visible';
}

// ── Init ──────────────────────────────────────────────────────────────────
window.onload = () => {
  $('fc-date').value = new Date().toISOString().slice(0,10);
  loadModels();
};
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  LSTM Forecast Flask App                         ║")
    print("║  http://127.0.0.1:5001                           ║")
    print("╚══════════════════════════════════════════════════╝")
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
