"""
strategy.py
───────────
Signal engine:
  BUY  → CNN Bullish  AND SVM Safe
  SELL → CNN Bearish  OR  SVM Risky (extreme)
  HOLD → everything else
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SIGNAL_BUY  =  1
SIGNAL_HOLD =  0
SIGNAL_SELL = -1


def generate_signals(
    cnn_predictions: np.ndarray,   # +1 Bullish / -1 Bearish
    svm_predictions: np.ndarray,   # 1 Safe / 0 Risky
    svm_safe_proba:  np.ndarray | None = None,  # P(Safe) ∈ [0,1]
    risk_threshold:  float = 0.35,              # below → extreme risk → SELL
) -> np.ndarray:
    """
    Combine model outputs into BUY / HOLD / SELL signals.

    Parameters
    ----------
    cnn_predictions : array of +1 / -1
    svm_predictions : array of 1 / 0
    svm_safe_proba  : optional continuous probability for fine-grained risk
    risk_threshold  : if P(Safe) < this → treat as extreme risk (force SELL)
    """
    n = len(cnn_predictions)
    signals = np.full(n, SIGNAL_HOLD, dtype=int)

    for i in range(n):
        bullish = (cnn_predictions[i] == 1)
        safe    = (svm_predictions[i] == 1)

        # optionally override with probability
        if svm_safe_proba is not None:
            extreme_risk = svm_safe_proba[i] < risk_threshold
        else:
            extreme_risk = not safe

        if bullish and safe:
            signals[i] = SIGNAL_BUY
        elif (not bullish) or extreme_risk:
            signals[i] = SIGNAL_SELL
        # else HOLD

    return signals


def signals_to_series(signals: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series(signals, index=index, name="signal")


def position_from_signals(signals: pd.Series) -> pd.Series:
    """
    Convert BUY/SELL/HOLD into a held position (1 = long, 0 = flat).
    We enter on BUY and exit/short on SELL.
    """
    position = pd.Series(0, index=signals.index, dtype=float)
    held = 0
    for dt, sig in signals.items():
        if sig == SIGNAL_BUY:
            held = 1
        elif sig == SIGNAL_SELL:
            held = 0
        position[dt] = held
    return position


# ─── Live inference helper ────────────────────────────────────────────────────

def live_signal(
    cnn_model,
    svm_model,
    X_cnn_latest: np.ndarray,   # shape (1, window, n_features)
    X_svm_latest: np.ndarray,   # shape (1, n_svm_features)
) -> dict:
    """
    Run a single forward pass and return a rich signal dict.
    """
    from models import predict_cnn, predict_cnn_proba, predict_svm, predict_svm_proba

    cnn_pred  = predict_cnn(cnn_model,  X_cnn_latest)[0]
    cnn_proba = float(predict_cnn_proba(cnn_model, X_cnn_latest)[0])
    svm_pred  = predict_svm(svm_model,  X_svm_latest)[0]
    svm_proba = float(predict_svm_proba(svm_model, X_svm_latest)[0])

    signal = generate_signals(
        np.array([cnn_pred]),
        np.array([svm_pred]),
        svm_safe_proba=np.array([svm_proba]),
    )[0]

    label_map = {SIGNAL_BUY: "BUY", SIGNAL_HOLD: "HOLD", SIGNAL_SELL: "SELL"}

    return {
        "signal":          label_map[signal],
        "signal_int":      int(signal),
        "cnn_trend":       "Bullish" if cnn_pred == 1 else "Bearish",
        "cnn_confidence":  round(cnn_proba * 100, 1),
        "svm_condition":   "Safe" if svm_pred == 1 else "Risky",
        "svm_safe_prob":   round(svm_proba * 100, 1),
    }