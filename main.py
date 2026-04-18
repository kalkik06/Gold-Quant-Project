"""
main.py
───────
Entry point: trains CNN + SVM, runs backtest, prints metrics.
Run once before launching the dashboard.

  python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_ingestion import load_dataset
from features      import build_features, prepare_cnn_data, prepare_svm_data, CNN_FEATURES, SVM_FEATURES
from models        import train_cnn, train_svm, evaluate_cnn, evaluate_svm, predict_cnn, predict_svm, predict_svm_proba
from strategy      import generate_signals, signals_to_series
from evaluator     import run_backtest, plot_equity_curve

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  Gold Quant Pro – Training & Backtest Pipeline")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Fetching & engineering features …")
    raw = load_dataset(period="2y")
    df  = build_features(raw)
    print(f"      Dataset shape after feature engineering: {df.shape}")

    # ── 2. Prepare datasets ───────────────────────────────────────────────────
    print("\n[2/5] Preparing CNN and SVM datasets …")
    WINDOW = 30
    X_cnn, y_cnn, cnn_scaler = prepare_cnn_data(df, window=WINDOW)
    X_svm, y_svm, svm_scaler = prepare_svm_data(df)

    # align svm to same rows as cnn (cnn starts at row WINDOW)
    X_svm_aligned = X_svm[WINDOW : WINDOW + len(X_cnn)]
    y_svm_aligned = y_svm[WINDOW : WINDOW + len(X_cnn)]

    split = int(len(X_cnn) * 0.8)
    X_cnn_tr, X_cnn_te = X_cnn[:split],        X_cnn[split:]
    y_cnn_tr, y_cnn_te = y_cnn[:split],        y_cnn[split:]
    X_svm_tr, X_svm_te = X_svm_aligned[:split], X_svm_aligned[split:]
    y_svm_tr, y_svm_te = y_svm_aligned[:split], y_svm_aligned[split:]

    print(f"      CNN  train={X_cnn_tr.shape}  test={X_cnn_te.shape}")
    print(f"      SVM  train={X_svm_tr.shape}  test={X_svm_te.shape}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[3/5] Training models …")
    cnn_model = train_cnn(X_cnn_tr, y_cnn_tr, epochs=40, batch_size=32)
    svm_model = train_svm(X_svm_tr, y_svm_tr)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set …")
    evaluate_cnn(cnn_model, X_cnn_te, y_cnn_te)
    evaluate_svm(svm_model, X_svm_te, y_svm_te)

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    print("\n[5/5] Running backtest …")
    # generate signals on full dataset (train+test for equity curve)
    cnn_preds  = predict_cnn(cnn_model,  X_cnn)
    svm_preds  = predict_svm(svm_model,  X_svm_aligned)
    svm_probas = predict_svm_proba(svm_model, X_svm_aligned)

    raw_signals = generate_signals(cnn_preds, svm_preds, svm_safe_proba=svm_probas)

    # align signal index with df (skip first WINDOW rows)
    signal_index = df.index[WINDOW : WINDOW + len(raw_signals)]
    sig_series   = signals_to_series(raw_signals, signal_index)

    price_slice = df.loc[signal_index, ["open", "high", "low", "close"]]
    metrics, bt_df = run_backtest(price_slice, sig_series)

    print("\n  ╔══════════════════════════════════╗")
    print("  ║     Backtest Performance         ║")
    print("  ╠══════════════════════════════════╣")
    print(f"  ║  Sharpe Ratio  : {metrics['sharpe']:>8.3f}         ║")
    print(f"  ║  Total Profit  : {metrics['total_profit']:>7.2f}%         ║")
    print(f"  ║  Max Drawdown  : {metrics['max_drawdown']:>7.2f}%         ║")
    print(f"  ║  Win Rate      : {metrics['win_rate']:>7.1f}%         ║")
    print(f"  ║  Trades        : {metrics['n_trades']:>8d}         ║")
    print(f"  ║  Final Capital : ${metrics['final_equity']:>9,.2f}   ║")
    print("  ╚══════════════════════════════════╝\n")

    curve_path = os.path.join(OUTPUTS_DIR, "equity_curve.png")
    plot_equity_curve(bt_df, save_path=curve_path)

    # save signal CSV for dashboard
    sig_out = bt_df[["close", "signal", "position", "equity"]].copy()
    sig_out.to_csv(os.path.join(OUTPUTS_DIR, "backtest_signals.csv"))
    print("[main] Outputs saved to ./outputs/")
    print("[main] ✅  Training complete. Launch the dashboard with:")
    print("       streamlit run dashboard.py\n")


if __name__ == "__main__":
    main()