"""
features.py
───────────
Computes technical indicators:
  RSI, ATR, Rolling Volatility, Log-Returns, MACD, Bollinger Bands.
Also builds the windowed feature matrix for CNN/SVM input.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─── Raw Indicators ───────────────────────────────────────────────────────────

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=window - 1, min_periods=window).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["volatility"] = df["log_return"].rolling(window).std() * np.sqrt(252)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    return df


def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(window).mean()
    return df


# ─── Full Feature Pipeline ────────────────────────────────────────────────────

CNN_FEATURES = [
    "log_return", "volume", "sentiment",
    "rsi", "macd_hist", "bb_pct", "vol_ratio",
]

SVM_FEATURES = ["volatility", "atr", "rsi"]


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators and drop NaN rows."""
    df = raw_df.copy()
    df = add_log_returns(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_volatility(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_volume_ratio(df)
    df = df.dropna()
    return df


# ─── Labels ───────────────────────────────────────────────────────────────────

def make_cnn_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """+1 (Bullish) if price is higher in `horizon` days else -1 (Bearish)."""
    future = df["close"].shift(-horizon)
    label = np.where(future > df["close"], 1, -1)
    return pd.Series(label, index=df.index, name="cnn_label")


def make_svm_labels(df: pd.DataFrame, vol_threshold: float = 0.25) -> pd.Series:
    """1 = Safe, 0 = Risky  (based on rolling volatility threshold)."""
    label = np.where(df["volatility"] < vol_threshold, 1, 0)
    return pd.Series(label, index=df.index, name="svm_label")


# ─── Windowed Matrix for CNN ──────────────────────────────────────────────────

def make_windows(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over feature_matrix → (N, window, n_features)."""
    X, y = [], []
    for i in range(window, len(feature_matrix)):
        X.append(feature_matrix[i - window : i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y)


# ─── Convenience Builder ──────────────────────────────────────────────────────

def prepare_cnn_data(df: pd.DataFrame, window: int = 30):
    feat_df = df[CNN_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df.values)

    raw_labels = make_cnn_labels(df)
    aligned = raw_labels.reindex(feat_df.index).values

    X, y = make_windows(scaled, aligned, window=window)
    # convert -1/+1 to 0/1 for binary cross-entropy
    y_bin = ((y + 1) // 2).astype(int)
    return X, y_bin, scaler


def prepare_svm_data(df: pd.DataFrame):
    feat_df = df[SVM_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat_df.values)

    labels = make_svm_labels(df).values
    return scaled, labels, scaler


if __name__ == "__main__":
    from data_ingestion import load_dataset
    raw = load_dataset()
    df = build_features(raw)
    X_cnn, y_cnn, _ = prepare_cnn_data(df)
    X_svm, y_svm, _ = prepare_svm_data(df)
    print(f"CNN  X={X_cnn.shape}  y={y_cnn.shape}")
    print(f"SVM  X={X_svm.shape}  y={y_svm.shape}")