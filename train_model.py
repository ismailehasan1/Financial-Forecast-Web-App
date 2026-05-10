"""
LSTM Forecasting Model — Training Script
=========================================
Uses Yahoo Finance data to train a Bidirectional LSTM with attention.
Exports a full pipeline as a .pkl file for use in the Flask app.

Usage:
    python train_model.py --ticker AAPL --horizon 30
    python train_model.py --ticker MSFT --horizon 7 --start 2020-01-01
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Try to import yfinance, else fallback ────────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠  yfinance not found. Install via: pip install yfinance")

# ─── Try to import TensorFlow / Keras ────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, LSTM, Bidirectional, Dense, Dropout,
        Layer, Multiply, Activation, Lambda, Flatten
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} detected")
except ImportError:
    TF_AVAILABLE = False
    print("⚠  TensorFlow not found. Install via: pip install tensorflow")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_yahoo_data(ticker: str, start: str = "2018-01-01", end: str = None) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is required: pip install yfinance")

    end = end or datetime.today().strftime("%Y-%m-%d")
    print(f"📥  Fetching {ticker} from {start} → {end}")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✓  {len(df)} trading days loaded  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as extra features."""
    d = df.copy()

    # Returns & log-price
    d["Returns"]   = d["Close"].pct_change()
    d["Log_Close"] = np.log(d["Close"])

    # Moving averages
    for w in [5, 10, 20, 50]:
        d[f"MA_{w}"] = d["Close"].rolling(w).mean()

    # RSI (14)
    delta  = d["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    d["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12      = d["Close"].ewm(span=12).mean()
    ema26      = d["Close"].ewm(span=26).mean()
    d["MACD"]  = ema12 - ema26
    d["Signal"]= d["MACD"].ewm(span=9).mean()

    # Bollinger Bands
    ma20         = d["Close"].rolling(20).mean()
    std20        = d["Close"].rolling(20).std()
    d["BB_upper"]= ma20 + 2 * std20
    d["BB_lower"]= ma20 - 2 * std20
    d["BB_width"]= (d["BB_upper"] - d["BB_lower"]) / ma20

    # Volume change
    d["Volume_Change"] = d["Volume"].pct_change()

    # ATR (14)
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    d["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    d = d.dropna()
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class SequenceBuilder:
    """Creates (X, y) sliding-window sequences and stores scalers."""

    def __init__(self, lookback: int = 60, horizon: int = 30):
        self.lookback = lookback
        self.horizon  = horizon
        self.feature_scaler = MinMaxScaler()
        self.target_scaler  = MinMaxScaler()
        self.feature_cols: list = []

    def fit_transform(self, df: pd.DataFrame):
        target = df[["Close"]].values
        features = df.drop(columns=["Close"]).values
        self.feature_cols = [c for c in df.columns if c != "Close"]

        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled   = self.target_scaler.fit_transform(target)

        X, y = [], []
        full = np.hstack([features_scaled, target_scaled])

        for i in range(self.lookback, len(full) - self.horizon + 1):
            X.append(full[i - self.lookback : i, :])
            y.append(target_scaled[i : i + self.horizon, 0])

        return np.array(X), np.array(y)

    def transform_last(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the last `lookback` rows for inference."""
        features = df[self.feature_cols].values[-self.lookback:]
        target   = df[["Close"]].values[-self.lookback:]
        features_scaled = self.feature_scaler.transform(features)
        target_scaled   = self.target_scaler.transform(target)
        full = np.hstack([features_scaled, target_scaled])
        return full[np.newaxis, ...]  # (1, lookback, n_features+1)

    def inverse_target(self, arr: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# ══════════════════════════════════════════════════════════════════════════════
#  KERAS ATTENTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

if TF_AVAILABLE:
    class BahdanauAttention(Layer):
        """Additive (Bahdanau) attention over the LSTM sequence output."""

        def __init__(self, units: int = 64, **kwargs):
            super().__init__(**kwargs)
            self.W = Dense(units)
            self.V = Dense(1)

        def call(self, encoder_outputs):
            score   = self.V(tf.nn.tanh(self.W(encoder_outputs)))  # (B, T, 1)
            weights = tf.nn.softmax(score, axis=1)                  # (B, T, 1)
            context = tf.reduce_sum(weights * encoder_outputs, axis=1)  # (B, d)
            return context, weights

        def get_config(self):
            cfg = super().get_config()
            return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(input_shape: tuple, horizon: int, units=(128, 64), dropout=0.2):
    """
    Bidirectional LSTM + Bahdanau Attention → Dense output.

    Args:
        input_shape : (lookback, n_features)
        horizon     : number of future steps to predict
        units       : tuple of LSTM units per layer
        dropout     : dropout rate

    Returns:
        Compiled Keras Model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required: pip install tensorflow")

    inp = Input(shape=input_shape, name="sequence_input")

    # BiLSTM layer 1 — return sequences for attention
    x = Bidirectional(LSTM(units[0], return_sequences=True, name="bilstm_1"),
                      name="bi_1")(inp)
    x = Dropout(dropout)(x)

    # BiLSTM layer 2 — return sequences for attention
    x = Bidirectional(LSTM(units[1], return_sequences=True, name="bilstm_2"),
                      name="bi_2")(x)
    x = Dropout(dropout)(x)

    # Attention
    context, _ = BahdanauAttention(units=64, name="attention")(x)

    # Dense head
    x = Dense(64, activation="relu")(context)
    x = Dropout(dropout / 2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(horizon, name="forecast_output")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  FORECAST & UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════

def mc_dropout_forecast(model, X_input: np.ndarray, n_samples: int = 100) -> dict:
    """
    Monte Carlo dropout for uncertainty estimation.
    Runs the model `n_samples` times with dropout active.
    """
    preds = np.stack(
        [model(X_input, training=True).numpy() for _ in range(n_samples)],
        axis=0
    )  # (n_samples, 1, horizon)
    preds = preds[:, 0, :]  # (n_samples, horizon)
    return {
        "mean":  preds.mean(axis=0),
        "std":   preds.std(axis=0),
        "p5":    np.percentile(preds,  5, axis=0),
        "p25":   np.percentile(preds, 25, axis=0),
        "p75":   np.percentile(preds, 75, axis=0),
        "p95":   np.percentile(preds, 95, axis=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, seq_builder: SequenceBuilder) -> dict:
    """Compute MAE, RMSE, MAPE, R² on the test set."""
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Flatten all horizon steps for aggregate metrics
    y_true_flat = seq_builder.inverse_target(y_test.flatten())
    y_pred_flat = seq_builder.inverse_target(y_pred_scaled.flatten())

    mae  = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-9))) * 100
    r2   = r2_score(y_true_flat, y_pred_flat)

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4), "R2": round(r2, 4)}


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast(hist_dates, hist_prices, future_dates, fc_dict: dict,
                  ticker: str, horizon: int, out_dir: str = "."):
    """Save a high-quality forecast chart."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Historical
    ax.plot(hist_dates[-120:], hist_prices[-120:],
            color="#4d9cf8", lw=1.8, label="Historical", zorder=3)

    mean  = fc_dict["mean"]
    p5    = fc_dict["p5"]
    p95   = fc_dict["p95"]
    p25   = fc_dict["p25"]
    p75   = fc_dict["p75"]

    # Confidence bands
    ax.fill_between(future_dates, p5,  p95,  alpha=0.18, color="#1dce8c", label="90% CI")
    ax.fill_between(future_dates, p25, p75,  alpha=0.32, color="#1dce8c", label="50% CI")
    ax.plot(future_dates, mean, color="#1dce8c", lw=2.2, ls="--", label="Forecast", zorder=4)

    # Connect history → forecast
    ax.plot([hist_dates[-1], future_dates[0]],
            [hist_prices[-1], mean[0]],
            color="#1dce8c", lw=2.2, ls="--", zorder=4)

    # Styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3142")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.yaxis.label.set_color("#8b949e")
    ax.grid(color="#161b22", linestyle="-", linewidth=0.8, alpha=0.8)

    legend = ax.legend(loc="upper left", framealpha=0.3,
                        labelcolor="white", fontsize=9)
    legend.get_frame().set_edgecolor("#2a3142")

    ax.set_title(f"{ticker} — {horizon}-Day LSTM Forecast  |  {datetime.today().strftime('%Y-%m-%d')}",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("Date", color="#8b949e")
    ax.set_ylabel("Price (USD)", color="#8b949e")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}_forecast_{horizon}d.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊  Chart saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_pipeline(model, seq_builder: SequenceBuilder, ticker: str,
                  horizon: int, metrics: dict, out_dir: str = "models"):
    """
    Save everything needed for inference into one .pkl file.
    The Keras model is saved separately as SavedModel then embedded as bytes.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save Keras model to a temp path then read as bytes
    tmp_path = os.path.join(out_dir, f"_tmp_{ticker}.keras")
    model.save(tmp_path)
    with open(tmp_path, "rb") as f:
        model_bytes = f.read()
    os.remove(tmp_path)

    pipeline = {
        "ticker":        ticker,
        "horizon":       horizon,
        "lookback":      seq_builder.lookback,
        "feature_cols":  seq_builder.feature_cols,
        "feature_scaler": seq_builder.feature_scaler,
        "target_scaler":  seq_builder.target_scaler,
        "model_bytes":   model_bytes,
        "metrics":       metrics,
        "trained_at":    datetime.now().isoformat(),
        "tf_version":    tf.__version__ if TF_AVAILABLE else "N/A",
    }

    pkl_path = os.path.join(out_dir, f"{ticker}_lstm_{horizon}d.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(pkl_path) / 1_048_576
    print(f"✓  Pipeline saved → {pkl_path}  ({size_mb:.1f} MB)")
    return pkl_path


def load_pipeline(pkl_path: str):
    """Load saved pipeline and reconstruct Keras model from bytes."""
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(pipeline["model_bytes"])
        tmp_path = tmp.name

    model = tf.keras.models.load_model(
        tmp_path,
        custom_objects={"BahdanauAttention": BahdanauAttention}
    )
    os.remove(tmp_path)
    pipeline["model"] = model
    return pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def train(ticker: str, horizon: int, lookback: int = 60, start: str = "2018-01-01",
          epochs: int = 100, batch_size: int = 32, test_ratio: float = 0.1,
          out_dir: str = "models", plots_dir: str = "static/plots"):

    # 1. Data
    df_raw   = fetch_yahoo_data(ticker, start=start)
    df_feat  = build_features(df_raw)

    # 2. Sequences
    seq = SequenceBuilder(lookback=lookback, horizon=horizon)
    X, y = seq.fit_transform(df_feat)
    print(f"✓  Sequences: X={X.shape}  y={y.shape}")

    # 3. Train / test split (chronological)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Build model
    model = build_lstm_model(
        input_shape=(lookback, X.shape[2]),
        horizon=horizon
    )
    model.summary()

    # 5. Callbacks
    ckpt_path = os.path.join(out_dir, f"{ticker}_best.keras")
    os.makedirs(out_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", verbose=0),
    ]

    # 6. Train
    print(f"\n🚀  Training on {len(X_train)} samples, validating on {len(X_test)}")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, seq)
    print(f"\n📈  Test metrics:  {metrics}")

    # 8. Forecast future dates
    X_last   = seq.transform_last(df_feat)
    fc_dict  = mc_dropout_forecast(model, X_last, n_samples=200)
    fc_mean  = seq.inverse_target(fc_dict["mean"])
    fc_p5    = seq.inverse_target(fc_dict["p5"])
    fc_p25   = seq.inverse_target(fc_dict["p25"])
    fc_p75   = seq.inverse_target(fc_dict["p75"])
    fc_p95   = seq.inverse_target(fc_dict["p95"])

    last_date    = df_raw.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    fc_out = {
        "mean": fc_mean, "p5": fc_p5, "p25": fc_p25,
        "p75": fc_p75, "p95": fc_p95,
        "dates": future_dates.tolist(),
    }

    # 9. Plot
    plot_forecast(
        hist_dates=df_raw.index.to_list(),
        hist_prices=df_raw["Close"].values,
        future_dates=future_dates,
        fc_dict={"mean": fc_mean, "p5": fc_p5, "p25": fc_p25, "p75": fc_p75, "p95": fc_p95},
        ticker=ticker,
        horizon=horizon,
        out_dir=plots_dir,
    )

    # 10. Save pipeline
    pkl_path = save_pipeline(model, seq, ticker, horizon, metrics, out_dir=out_dir)

    print(f"\n✅  Done!  Pickle → {pkl_path}")
    return pkl_path, fc_out, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM stock forecasting model")
    parser.add_argument("--ticker",    type=str,   default="AAPL",       help="Yahoo Finance ticker (default: AAPL)")
    parser.add_argument("--horizon",   type=int,   default=30,           help="Forecast horizon in days (7/15/30/90)")
    parser.add_argument("--lookback",  type=int,   default=60,           help="Lookback window in days")
    parser.add_argument("--start",     type=str,   default="2018-01-01", help="Training data start date")
    parser.add_argument("--epochs",    type=int,   default=100,          help="Max training epochs")
    parser.add_argument("--batch",     type=int,   default=32,           help="Batch size")
    parser.add_argument("--out",       type=str,   default="models",     help="Output directory for model pickle")
    args = parser.parse_args()

    pkl_path, fc, mets = train(
        ticker=args.ticker,
        horizon=args.horizon,
        lookback=args.lookback,
        start=args.start,
        epochs=args.epochs,
        batch_size=args.batch,
        out_dir=args.out,
    )

    print("\n── Forecast Summary ──────────────────────────────")
    for date, val in zip(fc["dates"][:10], fc["mean"][:10]):
        print(f"  {pd.Timestamp(date).date()}  →  ${val:.2f}")
    if args.horizon > 10:
        print(f"  ... ({args.horizon - 10} more days)")
