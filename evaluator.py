"""
evaluator.py
────────────
Backtest engine and performance metrics:
  Sharpe Ratio, Max Drawdown, Total Profit, Equity Curve
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Annualised Sharpe (daily returns, 252 trading days)."""
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.15 = 15%)."""
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())          # negative number; abs for display


def total_profit(equity: pd.Series) -> float:
    """Total return as a fraction."""
    return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0])


def win_rate(trade_returns: pd.Series) -> float:
    wins = (trade_returns > 0).sum()
    total = len(trade_returns)
    return float(wins / total) if total else 0.0


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(
    price_df:       pd.DataFrame,
    signals:        pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,   # 0.1% per trade
) -> dict:
    """
    Simple long-only backtest.

    Parameters
    ----------
    price_df  : DataFrame with 'close' column; index matches signals
    signals   : Series of 1 (BUY), 0 (HOLD), -1 (SELL)
    """
    df = price_df[["close"]].copy()
    df["signal"] = signals.reindex(df.index).fillna(0).astype(int)
    df["position"] = _compute_position(df["signal"])
    df["market_return"] = df["close"].pct_change().fillna(0)

    # strategy return = position (lagged 1 day) × next day's market return
    df["strategy_return"] = df["position"].shift(1).fillna(0) * df["market_return"]

    # apply transaction cost on position changes
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_return"] -= df["trade"] * transaction_cost

    df["equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["bh_equity"] = initial_capital * (1 + df["market_return"]).cumprod()

    equity = df["equity"]
    strat_rets = df["strategy_return"]

    # Trade-level P&L
    trades = df[df["trade"] > 0]["strategy_return"]

    metrics = {
        "sharpe":       round(sharpe_ratio(strat_rets), 3),
        "max_drawdown": round(abs(max_drawdown(equity)) * 100, 2),   # percent
        "total_profit": round(total_profit(equity) * 100, 2),         # percent
        "win_rate":     round(win_rate(trades) * 100, 1),
        "n_trades":     int(df["trade"].sum()),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }
    return metrics, df


def _compute_position(signals: pd.Series) -> pd.Series:
    position = pd.Series(0, index=signals.index, dtype=float)
    held = 0
    for dt, sig in signals.items():
        if sig == 1:
            held = 1
        elif sig == -1:
            held = 0
        position[dt] = held
    return position


# ─── Equity Curve Plot (static PNG for main.py) ───────────────────────────────

def plot_equity_curve(backtest_df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(backtest_df.index, backtest_df["equity"],    label="Strategy",      color="#f5c518", lw=2)
    ax.plot(backtest_df.index, backtest_df["bh_equity"], label="Buy & Hold GLD", color="#888",   lw=1.5, ls="--")
    ax.set_title("Gold Quant Pro – Equity Curve", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Evaluator] Equity curve saved → {save_path}")
    return fig