# LSTM Stock Forecasting Pipeline

Bidirectional LSTM with Bahdanau Attention, MC-Dropout uncertainty,
Yahoo Finance data, Flask REST API + web dashboard.

---

## Project structure

```
lstm_forecaster/
├── train_model.py      ← data fetch, feature engineering, LSTM training, pickle export
├── run_forecast.py     ← load pipeline, run date-specific forecast, plot chart
├── app.py              ← Flask API + web dashboard
├── requirements.txt
├── models/             ← saved .pkl pipelines (auto-created)
└── static/plots/       ← generated forecast charts (auto-created)
```

---

## Quick start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Train a model
```bash
# 30-day AAPL forecast, data from 2018
python train_model.py --ticker AAPL --horizon 30

# 7-day MSFT forecast
python train_model.py --ticker MSFT --horizon 7

# 3-month TSLA forecast, custom start date
python train_model.py --ticker TSLA --horizon 90 --start 2019-01-01 --epochs 150
```

Available flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | AAPL | Yahoo Finance ticker |
| `--horizon` | 30 | Forecast days: 7 / 15 / 30 / 90 |
| `--lookback` | 60 | Input sequence length |
| `--start` | 2018-01-01 | Training data start |
| `--epochs` | 100 | Max training epochs (early stopping applies) |
| `--batch` | 32 | Batch size |
| `--out` | models/ | Output directory |

Output: `models/AAPL_lstm_30d.pkl`

---

### 3 — Run a date-specific forecast
```bash
# Forecast from today
python run_forecast.py --pkl models/AAPL_lstm_30d.pkl

# Forecast from a specific historical date
python run_forecast.py --pkl models/AAPL_lstm_30d.pkl --date 2024-01-15

# More MC-dropout samples = wider / more accurate confidence bands
python run_forecast.py --pkl models/AAPL_lstm_30d.pkl --date 2024-06-01 --mc 500
```

Output:
- Forecast table printed to terminal
- Chart saved to `static/plots/AAPL_2024-01-15_30d.png`

---

### 4 — Launch the Flask app
```bash
# Development
python app.py

# Production (gunicorn)
gunicorn app:app --workers 2 --timeout 300 --bind 0.0.0.0:5000
```

Open **http://localhost:5000** — you get:
- Train panel (start/stop background training jobs)
- Forecast panel (select pipeline, pick date, run)
- Models table with download links for every .pkl

---

## REST API reference

### `POST /api/train`
Start background training.
```json
{
  "ticker":   "AAPL",
  "horizon":  30,
  "start":    "2018-01-01",
  "epochs":   100,
  "lookback": 60,
  "batch":    32
}
```
Response: `{ "job_id": "a3f7c1b2", "status": "queued", "poll": "/api/status/a3f7c1b2" }`

### `GET /api/status/<job_id>`
Poll training progress.
```json
{
  "status":   "done",
  "pkl_name": "AAPL_lstm_30d.pkl",
  "metrics":  { "MAE": 2.14, "RMSE": 3.01, "MAPE": 1.82, "R2": 0.961 },
  "forecast": { "dates": [...], "mean": [...], "p5": [...], "p95": [...] }
}
```

### `POST /api/forecast`
Run inference from a saved pipeline.
```json
{
  "pkl_name": "AAPL_lstm_30d.pkl",
  "date":     "2024-06-01",
  "mc":       200
}
```
Response includes `forecast.dates`, `forecast.mean`, `forecast.p5/p25/p75/p95`, `chart_url`.

### `GET /api/models`
List all saved pipelines with metadata.

### `GET /api/download/<pkl_name>`
Download a .pkl file.

---

## Model architecture

```
Input (60, n_features)
   │
BiLSTM-1  (128 units × 2 directions = 256)  + Dropout 0.2
   │
BiLSTM-2  (64 units × 2 directions = 128)   + Dropout 0.2
   │
Bahdanau Attention  (64 units)
   │
Dense (64, ReLU) → Dropout 0.1 → Dense (32, ReLU)
   │
Dense (horizon)   ← forecast output
```

**Features** (17 total):
Open, High, Low, Volume, Returns, Log_Close,
MA_5/10/20/50, RSI_14, MACD, Signal, BB_width,
Volume_Change, ATR_14

**Uncertainty**: Monte Carlo dropout — run model N times
with dropout active, report mean ± percentile bands.

---

## Extending the pipeline

### Add a new feature
In `train_model.py → build_features()`:
```python
d["My_Feature"] = ...
```
The SequenceBuilder auto-picks up all non-Close columns.

### Change the model architecture
In `train_model.py → build_lstm_model()`.

### Use a different asset class
The pipeline is asset-agnostic — pass any Yahoo Finance ticker:
`EURUSD=X`, `GC=F` (gold), `BTC-USD`, `^GSPC` (S&P 500).

### Export to ONNX (faster inference)
```bash
pip install onnx tf2onnx
python -m tf2onnx.convert --saved-model models/_tmp_AAPL.keras --output models/AAPL.onnx
```

---

## Notes

- Early stopping (patience=10) prevents overfitting.
- The pickle bundles scalers + model weights — no separate files needed for deployment.
- MC-Dropout with 200 samples adds ~2–3 seconds per inference call.
- For production, replace the in-memory job store (`JOBS` dict) with Celery + Redis.

Select Python version and install library
py -3.11 -m pip install -r requirements.txt

Open a new terminal in VS Code and run these 3 commands one by one:
1 — Go to project folder:
powershellcd C:\forecast
2 — Activate the environment:
.\venv311\Scripts\Activate.ps1
3 — Start the app:
python app.py

Then open your browser and go to:
http://127.0.0.1:5001