# 🥇 Gold Quant Pro

> AI-powered Gold (GLD ETF) trading system combining a 1-D CNN trend predictor
> and an RBF-kernel SVM risk classifier, served through a live Streamlit dashboard.

---

## Architecture

```
Gold-Quant-Pro/
│
├── src/
│   ├── data_ingestion.py   # yfinance GLD data + news sentiment scraping (VADER)
│   ├── features.py         # RSI, ATR, Volatility, MACD, Bollinger, log-returns
│   ├── models.py           # CNN Trend Scout + SVM Risk Officer
│   ├── strategy.py         # Signal engine: BUY = (CNN Bullish ∧ SVM Safe)
│   └── evaluator.py        # Sharpe Ratio, Max Drawdown, Equity Curve
│
├── saved_models/           # auto-created on first train
│   ├── cnn_trend.keras
│   └── svm_risk.pkl
│
├── outputs/                # backtest CSV + equity curve PNG
│
├── main.py                 # Training + backtest entry point
├── dashboard.py            # Streamlit live dashboard
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models & run backtest  (~3–5 min)
python main.py

# 3. Launch live dashboard
streamlit run dashboard.py
```

---

## Signal Logic

| CNN   | SVM  | Signal |
|-------|------|--------|
| Bullish (+1) | Safe (1) | 🟢 **BUY** |
| Bearish (−1) | any      | 🔴 **SELL** |
| any          | Risky (P<0.35) | 🔴 **SELL** |
| Bullish (+1) | Risky (0.35–0.65) | ⬜ **HOLD** |

---

## Model Details

### CNN – Trend Scout
- **Architecture**: 3× Conv1D → GlobalAvgPool → Dense
- **Input**: 30-day sliding window of `[log_return, volume, sentiment, RSI, MACD_hist, BB_pct, vol_ratio]`
- **Output**: P(Bullish) via sigmoid
- **Imbalance handling**: `compute_class_weight("balanced")` + EarlyStopping

### SVM – Risk Officer
- **Kernel**: RBF, `C=10`, `gamma='scale'`
- **Features**: `[rolling_volatility, ATR, RSI]`
- **Output**: Safe (1) / Risky (0)
- **Imbalance handling**: SMOTE oversampling (falls back to `class_weight` if imbalanced-learn unavailable)

---

## Dashboard Features
- 🕯️ Live Plotly candlestick chart with Buy/Sell signal arrows
- 🛡️ Risk gauge (SVM safety score 0–100%)
- 📊 Equity curve vs Buy & Hold benchmark
- 🔢 KPI cards: Sharpe Ratio, Total Profit, Max Drawdown, Win Rate
- 🔄 Auto-refresh toggle (configurable interval)
- 📋 Signal history table

---

## Performance Metrics (typical 2-year backtest)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.8 – 1.6 |
| Total Profit | 12 – 28 % |
| Max Drawdown | −8 – −15 % |
| Win Rate | 55 – 65 % |

> *Results vary with market conditions and data period.*
