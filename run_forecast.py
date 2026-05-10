"""
Forecast Runner
===============
Load a saved pipeline .pkl and generate a forecast from a given date.

Usage:
    python run_forecast.py --pkl models/AAPL_lstm_30d.pkl --date 2024-06-01
    python run_forecast.py --pkl models/AAPL_lstm_7d.pkl  --horizon 7 --mc 300
"""

import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─── Conditional imports ──────────────────────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline(pkl_path: str) -> dict:
    """Deserialise pipeline; reconstruct Keras model from embedded bytes."""
    print(f"📦  Loading pipeline from {pkl_path}")
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required to run inference: pip install tensorflow")

    import tempfile
    from train_model import BahdanauAttention  # custom layer must be importable

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(pipeline["model_bytes"])
        tmp_path = tmp.name

    pipeline["model"] = tf.keras.models.load_model(
        tmp_path,
        custom_objects={"BahdanauAttention": BahdanauAttention}
    )
    os.remove(tmp_path)

    print(f"✓  Model loaded  |  ticker={pipeline['ticker']}  horizon={pipeline['horizon']}d"
          f"  trained_at={pipeline['trained_at']}")
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH FRESH DATA UP TO THE REQUESTED DATE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data_up_to(ticker: str, end_date: str, lookback_days: int = 200) -> pd.DataFrame:
    """Download enough trading days to fill the lookback window ending at end_date."""
    from train_model import fetch_yahoo_data, build_features

    # Buffer: request extra calendar days to account for weekends/holidays
    end_dt  = pd.Timestamp(end_date)
    start_dt = end_dt - timedelta(days=lookback_days * 2)
    start   = start_dt.strftime("%Y-%m-%d")
    end     = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    df_raw  = fetch_yahoo_data(ticker, start=start, end=end)
    df_feat = build_features(df_raw)
    return df_raw, df_feat


# ══════════════════════════════════════════════════════════════════════════════
#  MC DROPOUT FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def mc_dropout_forecast(model, X_input: np.ndarray, n_samples: int = 200) -> dict:
    preds = np.stack(
        [model(X_input, training=True).numpy() for _ in range(n_samples)],
        axis=0
    )[:, 0, :]
    return {
        "mean": preds.mean(axis=0),
        "std":  preds.std(axis=0),
        "p5":   np.percentile(preds,  5, axis=0),
        "p25":  np.percentile(preds, 25, axis=0),
        "p75":  np.percentile(preds, 75, axis=0),
        "p95":  np.percentile(preds, 95, axis=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RECONSTRUCT SEQUENCE BUILDER FROM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def make_seq_builder_from_pipeline(pipeline: dict):
    """Re-hydrate a SequenceBuilder with saved scalers."""
    from train_model import SequenceBuilder
    sb = SequenceBuilder(lookback=pipeline["lookback"], horizon=pipeline["horizon"])
    sb.feature_scaler = pipeline["feature_scaler"]
    sb.target_scaler  = pipeline["target_scaler"]
    sb.feature_cols   = pipeline["feature_cols"]
    return sb


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_from_date(hist_df, future_dates, fc_dict, ticker, horizon,
                             as_of_date, out_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    hist_prices = hist_df["Close"].values
    hist_dates  = hist_df.index.to_list()

    ax.plot(hist_dates[-120:], hist_prices[-120:],
            color="#4d9cf8", lw=1.8, label="Historical", zorder=3)

    ax.fill_between(future_dates, fc_dict["p5"],  fc_dict["p95"],
                    alpha=0.18, color="#1dce8c", label="90% CI")
    ax.fill_between(future_dates, fc_dict["p25"], fc_dict["p75"],
                    alpha=0.32, color="#1dce8c", label="50% CI")
    ax.plot(future_dates, fc_dict["mean"],
            color="#1dce8c", lw=2.2, ls="--", label="Forecast", zorder=4)
    ax.plot([hist_dates[-1], future_dates[0]],
            [hist_prices[-1], fc_dict["mean"][0]],
            color="#1dce8c", lw=2.2, ls="--", zorder=4)

    # Mark as-of date
    ax.axvline(pd.Timestamp(as_of_date), color="#f0a05a", lw=1.2,
               ls=":", label=f"As of {as_of_date}", alpha=0.85)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3142")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(color="#161b22", linestyle="-", linewidth=0.8, alpha=0.8)
    legend = ax.legend(loc="upper left", framealpha=0.3,
                        labelcolor="white", fontsize=9)
    legend.get_frame().set_edgecolor("#2a3142")

    ax.set_title(f"{ticker} — {horizon}-Day Forecast from {as_of_date}",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Price (USD)", color="#8b949e")

    if out_path is None:
        out_path = f"{ticker}_forecast_{as_of_date}_{horizon}d.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊  Chart saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast(pkl_path: str, as_of_date: str = None, n_mc: int = 200,
                 plot_dir: str = "static/plots") -> dict:
    """
    Full end-to-end inference:
      1. Load pipeline from pickle
      2. Fetch / slice historical data up to as_of_date
      3. Build input sequence
      4. MC-dropout forecast
      5. Plot & return results dict
    """
    pipeline = load_pipeline(pkl_path)
    ticker   = pipeline["ticker"]
    horizon  = pipeline["horizon"]
    as_of_date = as_of_date or datetime.today().strftime("%Y-%m-%d")

    # Data
    df_raw, df_feat = fetch_data_up_to(ticker, as_of_date,
                                        lookback_days=pipeline["lookback"] + 100)

    # Check we have enough rows
    if len(df_feat) < pipeline["lookback"]:
        raise ValueError(
            f"Not enough data ({len(df_feat)} rows) for lookback={pipeline['lookback']}. "
            f"Try an earlier start date."
        )

    # Sequence builder
    sb = make_seq_builder_from_pipeline(pipeline)

    # Input
    X_last = sb.transform_last(df_feat)  # (1, lookback, n_features)

    # Forecast
    print(f"🔮  Running MC-Dropout inference ({n_mc} samples) …")
    fc_scaled = mc_dropout_forecast(pipeline["model"], X_last, n_samples=n_mc)

    # Inverse-transform
    fc = {k: sb.inverse_target(v) for k, v in fc_scaled.items() if k != "std"}
    fc["std"] = fc_scaled["std"] * (sb.target_scaler.data_max_[0] - sb.target_scaler.data_min_[0])

    last_date    = df_raw.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)
    fc["dates"]  = future_dates.tolist()

    # Print table
    table_rows = []
    for i, (d, m, lo, hi) in enumerate(zip(future_dates, fc["mean"], fc["p5"], fc["p95"])):
        table_rows.append([
            str(d.date()), f"${m:.2f}",
            f"${lo:.2f}", f"${hi:.2f}",
            f"±{fc['std'][i]:.2f}"
        ])
    print(f"\n── {ticker} {horizon}-Day Forecast from {as_of_date} ──────────────────")
    print(tabulate(table_rows, headers=["Date", "Forecast", "P5", "P95", "Std"],
                   tablefmt="rounded_outline"))

    # Metrics from training
    print(f"\n── Training metrics ───────────────────────────────")
    for k, v in pipeline["metrics"].items():
        print(f"  {k}: {v}")

    # Plot
    os.makedirs(plot_dir, exist_ok=True)
    chart_path = plot_forecast_from_date(
        hist_df=df_raw,
        future_dates=future_dates,
        fc_dict=fc,
        ticker=ticker,
        horizon=horizon,
        as_of_date=as_of_date,
        out_path=os.path.join(plot_dir, f"{ticker}_{as_of_date}_{horizon}d.png"),
    )

    return {
        "ticker":      ticker,
        "horizon":     horizon,
        "as_of_date":  as_of_date,
        "forecast":    fc,
        "metrics":     pipeline["metrics"],
        "chart_path":  chart_path,
        "trained_at":  pipeline["trained_at"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM forecast from saved pipeline")
    parser.add_argument("--pkl",    required=True,  help="Path to .pkl pipeline file")
    parser.add_argument("--date",   default=None,   help="As-of date YYYY-MM-DD (default: today)")
    parser.add_argument("--mc",     type=int, default=200, help="MC-Dropout samples (default: 200)")
    parser.add_argument("--plots",  default="static/plots", help="Directory to save charts")
    args = parser.parse_args()

    results = run_forecast(
        pkl_path=args.pkl,
        as_of_date=args.date,
        n_mc=args.mc,
        plot_dir=args.plots,
    )
    print(f"\n✅  Forecast complete.  Chart → {results['chart_path']}")
