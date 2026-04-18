"""
dashboard.py
────────────
Streamlit live dashboard – Gold Quant Pro
Run with:  streamlit run dashboard.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Quant Pro",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Gold CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --gold:    #f5c518;
    --gold2:   #d4a017;
    --bg:      #0a0a0f;
    --surface: #12121a;
    --border:  #2a2a3a;
    --green:   #00e676;
    --red:     #ff1744;
    --text:    #e8e8f0;
    --muted:   #7a7a9a;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.metric-card {
    background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 10px;
}

.metric-label  { color: var(--muted); font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }
.metric-value  { color: var(--gold); font-size: 28px; font-weight: 700; font-family: 'Share Tech Mono'; }
.metric-delta  { font-size: 12px; color: var(--muted); }

.signal-buy  { background: rgba(0,230,118,0.12); border: 1px solid #00e676;
               color: #00e676; border-radius: 6px; padding: 6px 16px;
               font-size: 18px; font-weight: 700; letter-spacing: 2px; display:inline-block; }
.signal-sell { background: rgba(255,23,68,0.12);  border: 1px solid #ff1744;
               color: #ff1744; border-radius: 6px; padding: 6px 16px;
               font-size: 18px; font-weight: 700; letter-spacing: 2px; display:inline-block; }
.signal-hold { background: rgba(245,197,24,0.10); border: 1px solid #f5c518;
               color: #f5c518; border-radius: 6px; padding: 6px 16px;
               font-size: 18px; font-weight: 700; letter-spacing: 2px; display:inline-block; }

.header-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #f5c518, #d4a017, #f5c518);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}
.header-sub { color: var(--muted); font-size: 13px; letter-spacing: 2px; }

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px; padding: 12px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f5c518, #d4a017);
    color: #0a0a0f; border: none; font-weight: 700;
    font-family: 'Rajdhani'; letter-spacing: 1px;
    border-radius: 6px; width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    from models import load_cnn, load_svm
    cnn = load_cnn()
    svm = load_svm()
    return cnn, svm


@st.cache_data(ttl=300, show_spinner="Fetching market data …")
def get_data(period="1y"):
    from data_ingestion import load_dataset
    from features       import build_features
    raw = load_dataset(period=period)
    return build_features(raw)


def latest_signal(df, cnn_model, svm_model, window=30):
    from features  import CNN_FEATURES, SVM_FEATURES
    from models    import predict_cnn, predict_cnn_proba, predict_svm, predict_svm_proba
    from sklearn.preprocessing import StandardScaler

    cnn_feat = df[CNN_FEATURES].values
    svm_feat = df[SVM_FEATURES].values

    sc_cnn = StandardScaler().fit(cnn_feat)
    sc_svm = StandardScaler().fit(svm_feat)

    cnn_scaled = sc_cnn.transform(cnn_feat)
    svm_scaled = sc_svm.transform(svm_feat)

    # last window for CNN
    X_cnn = cnn_scaled[-window:].reshape(1, window, -1)
    X_svm = svm_scaled[[-1]]

    cnn_pred  = predict_cnn(cnn_model,  X_cnn)[0]
    cnn_proba = float(predict_cnn_proba(cnn_model, X_cnn)[0])
    svm_pred  = predict_svm(svm_model,  X_svm)[0]
    svm_proba = float(predict_svm_proba(svm_model, X_svm)[0])

    if cnn_pred == 1 and svm_pred == 1:
        sig = "BUY"
    elif cnn_pred == -1 or svm_proba < 0.35:
        sig = "SELL"
    else:
        sig = "HOLD"

    return {
        "signal":       sig,
        "cnn_trend":    "Bullish" if cnn_pred == 1 else "Bearish",
        "cnn_conf":     round(cnn_proba * 100, 1),
        "svm_cond":     "Safe" if svm_pred == 1 else "Risky",
        "svm_safe_pct": round(svm_proba * 100, 1),
    }


def backtest_signals(df, cnn_model, svm_model, window=30):
    from features  import CNN_FEATURES, SVM_FEATURES, prepare_cnn_data, prepare_svm_data
    from models    import predict_cnn, predict_svm, predict_svm_proba
    from strategy  import generate_signals, signals_to_series
    from evaluator import run_backtest

    X_cnn, y_cnn, _ = prepare_cnn_data(df, window=window)
    X_svm, y_svm, _ = prepare_svm_data(df)
    X_svm_al = X_svm[window: window + len(X_cnn)]

    cnn_p = predict_cnn(cnn_model,  X_cnn)
    svm_p = predict_svm(svm_model,  X_svm_al)
    svm_q = predict_svm_proba(svm_model, X_svm_al)

    sigs   = generate_signals(cnn_p, svm_p, svm_safe_proba=svm_q)
    idx    = df.index[window: window + len(sigs)]
    s_ser  = signals_to_series(sigs, idx)

    price  = df.loc[idx, ["open","high","low","close"]]
    metrics, bt_df = run_backtest(price, s_ser)
    return metrics, bt_df, s_ser


def make_candlestick(df, signals=None, n_candles=120):
    sub = df.tail(n_candles).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # candles
    fig.add_trace(
        go.Candlestick(
            x=sub.index,
            open=sub["open"], high=sub["high"],
            low=sub["low"],   close=sub["close"],
            name="GLD",
            increasing_line_color="#00e676", increasing_fillcolor="#00e676",
            decreasing_line_color="#ff1744", decreasing_fillcolor="#ff1744",
        ), row=1, col=1
    )

    # volume bars
    colors = [
    "rgba(0, 230, 118, 0.5)" if c >= o else "rgba(255, 23, 68, 0.5)"
    for c, o in zip(sub["close"], sub["open"])
]
    fig.add_trace(
        go.Bar(x=sub.index, y=sub["volume"], name="Volume",
               marker_color=colors, showlegend=False),
        row=2, col=1
    )

    # ── signal arrows ──────────────────────────────────────────────────────────
    if signals is not None:
        sig_sub = signals.reindex(sub.index).fillna(0)

        buy_idx  = sig_sub[sig_sub ==  1].index
        sell_idx = sig_sub[sig_sub == -1].index

        if len(buy_idx):
            fig.add_trace(go.Scatter(
                x=buy_idx,
                y=sub.loc[buy_idx, "low"] * 0.995,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14,
                            color="#00e676", line=dict(width=1, color="#fff")),
                name="BUY",
            ), row=1, col=1)

        if len(sell_idx):
            fig.add_trace(go.Scatter(
                x=sell_idx,
                y=sub.loc[sell_idx, "high"] * 1.005,
                mode="markers",
                marker=dict(symbol="triangle-down", size=14,
                            color="#ff1744", line=dict(width=1, color="#fff")),
                name="SELL",
            ), row=1, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(family="Share Tech Mono", color="#e8e8f0"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=10, t=30, b=0),
        legend=dict(bgcolor="#12121a", bordercolor="#2a2a3a", borderwidth=1),
        height=520,
    )
    fig.update_xaxes(gridcolor="#1e1e2e", zeroline=False)
    fig.update_yaxes(gridcolor="#1e1e2e", zeroline=False)
    return fig


def risk_gauge(svm_safe_pct: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=svm_safe_pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Market Safety Score", "font": {"color": "#e8e8f0", "size": 14}},
        number={"suffix": "%", "font": {"color": "#f5c518", "size": 28}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#7a7a9a"},
            "bar":       {"color": "#f5c518"},
            "bgcolor":   "#1a1a2e",
            "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [0,  35], "color": "#ff174420"},
                {"range": [35, 65], "color": "#f5c51820"},
                {"range": [65,100], "color": "#00e67620"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#12121a",
        font=dict(color="#e8e8f0"),
        height=240, margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def equity_chart(bt_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["equity"],
        name="Strategy", line=dict(color="#f5c518", width=2),
        fill="tozeroy", fillcolor="rgba(245,197,24,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["bh_equity"],
        name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
        font=dict(family="Share Tech Mono", color="#e8e8f0"),
        legend=dict(bgcolor="#12121a", bordercolor="#2a2a3a"),
        margin=dict(l=0, r=10, t=10, b=0), height=260,
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e", tickprefix="$")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-title">🥇 GOLD QUANT</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">AI-Powered Trading System</div>', unsafe_allow_html=True)
    st.divider()

    refresh_rate = st.slider("Refresh (seconds)", 30, 300, 60, step=30)
    n_candles    = st.slider("Candles to display", 60, 252, 120, step=20)
    period_opt   = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1)
    live_loop    = st.toggle("Live Auto-Refresh", value=False)
    st.divider()

    if st.button("🔄  Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <br>
    <div style='color:#7a7a9a;font-size:11px;letter-spacing:1px'>
    CNN: 1-D Convolutional Network<br>
    SVM: RBF Kernel Risk Classifier<br>
    Signal = CNN Bullish ∧ SVM Safe
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:10px 0 4px'>
  <span class='header-title'>GOLD QUANT PRO</span><br>
  <span class='header-sub'>REAL-TIME AI TRADING DASHBOARD · GLD ETF</span>
</div>
<hr style='border:0;border-top:1px solid #2a2a3a;margin:8px 0 20px'>
""", unsafe_allow_html=True)


# ── Model load gate ───────────────────────────────────────────────────────────
models_ready = os.path.exists(
    os.path.join(os.path.dirname(__file__), "saved_models", "cnn_trend.keras")
) and os.path.exists(
    os.path.join(os.path.dirname(__file__), "saved_models", "svm_risk.pkl")
)

if not models_ready:
    st.error(
        "⚠️  Trained models not found. Please run `python main.py` first to train "
        "the CNN & SVM models, then relaunch the dashboard."
    )
    st.info(
        "The dashboard will display **demo mode** with synthetic signals "
        "so you can preview the UI layout."
    )
    # ── DEMO MODE ─────────────────────────────────────────────────────────────
    df = get_data(period=period_opt)

    # synthetic signals
    rng = np.random.default_rng(7)
    n   = len(df)
    raw = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.5, 0.3])
    demo_signals = pd.Series(raw, index=df.index, name="signal")

    demo_info = {
        "signal": "BUY", "cnn_trend": "Bullish",
        "cnn_conf": 73.2, "svm_cond": "Safe", "svm_safe_pct": 68.5
    }
    demo_metrics = {
        "sharpe": 1.24, "total_profit": 18.7,
        "max_drawdown": 9.3, "win_rate": 61.0,
        "n_trades": 42, "final_equity": 11_870.0,
    }

    # top KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    sig = demo_info["signal"]
    sig_html = f'<span class="signal-{sig.lower()}">{sig}</span>'
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Signal</div><div style="margin-top:4px">{sig_html}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">CNN Trend</div><div class="metric-value" style="color:#00e676">{demo_info["cnn_trend"]}</div><div class="metric-delta">conf {demo_info["cnn_conf"]}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">SVM Condition</div><div class="metric-value" style="color:#00e676">{demo_info["svm_cond"]}</div><div class="metric-delta">safety {demo_info["svm_safe_pct"]}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{demo_metrics["sharpe"]}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Profit</div><div class="metric-value" style="color:#00e676">+{demo_metrics["total_profit"]}%</div></div>', unsafe_allow_html=True)

    # chart
    st.plotly_chart(make_candlestick(df, demo_signals, n_candles), use_container_width=True)

    gcol, ecol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(risk_gauge(demo_info["svm_safe_pct"]), use_container_width=True)

    st.stop()


# ── LIVE MODE ─────────────────────────────────────────────────────────────────
cnn_model, svm_model = load_models()
df = get_data(period=period_opt)

with st.spinner("Running signal engine …"):
    info    = latest_signal(df, cnn_model, svm_model)
    metrics, bt_df, sig_series = backtest_signals(df, cnn_model, svm_model)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
sig     = info["signal"]
sig_col = {"BUY": "#00e676", "SELL": "#ff1744", "HOLD": "#f5c518"}[sig]
sig_html = f'<span class="signal-{sig.lower()}">{sig}</span>'

def _card(label, value, delta="", val_color="var(--gold)"):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{val_color}">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>"""

with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Live Signal</div><div style="margin-top:6px">{sig_html}</div></div>', unsafe_allow_html=True)
with c2:
    trend_col = "#00e676" if info["cnn_trend"]=="Bullish" else "#ff1744"
    st.markdown(_card("CNN Trend", info["cnn_trend"], f"conf {info['cnn_conf']}%", trend_col), unsafe_allow_html=True)
with c3:
    cond_col = "#00e676" if info["svm_cond"]=="Safe" else "#ff1744"
    st.markdown(_card("Market Condition", info["svm_cond"], f"safety {info['svm_safe_pct']}%", cond_col), unsafe_allow_html=True)
with c4:
    sharp_col = "#00e676" if metrics["sharpe"] > 1 else ("#f5c518" if metrics["sharpe"] > 0 else "#ff1744")
    st.markdown(_card("Sharpe Ratio", metrics["sharpe"], "annualised", sharp_col), unsafe_allow_html=True)
with c5:
    prof_col = "#00e676" if metrics["total_profit"] > 0 else "#ff1744"
    prefix = "+" if metrics["total_profit"] > 0 else ""
    st.markdown(_card("Total Profit", f"{prefix}{metrics['total_profit']}%", f"{metrics['n_trades']} trades", prof_col), unsafe_allow_html=True)
with c6:
    st.markdown(_card("Max Drawdown", f"-{metrics['max_drawdown']}%", f"win rate {metrics['win_rate']}%", "#ff9800"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Candlestick Chart ─────────────────────────────────────────────────────────
st.markdown("#### 📈  GLD Candlestick  ·  Buy/Sell Signals")
fig_candle = make_candlestick(df, sig_series, n_candles)
st.plotly_chart(fig_candle, use_container_width=True)

# ── Bottom row ────────────────────────────────────────────────────────────────
g_col, e_col = st.columns([1, 2])

with g_col:
    st.markdown("#### 🛡️  Risk Gauge")
    st.plotly_chart(risk_gauge(info["svm_safe_pct"]), use_container_width=True)

with e_col:
    st.markdown("#### 💹  Equity Curve  vs  Buy & Hold")
    st.plotly_chart(equity_chart(bt_df), use_container_width=True)

# ── Signal log table ──────────────────────────────────────────────────────────
with st.expander("📋  Signal History (last 30 rows)"):
    recent = bt_df[["close","signal","position","equity"]].tail(30).copy()
    recent["signal"] = recent["signal"].map({1:"🟢 BUY", 0:"⬜ HOLD", -1:"🔴 SELL"})
    recent.index = recent.index.strftime("%Y-%m-%d")
    st.dataframe(recent.style.format({"close":"${:.2f}","equity":"${:,.0f}"}), use_container_width=True)

# ── Auto-refresh loop ─────────────────────────────────────────────────────────
if live_loop:
    st.caption(f"⟳  Auto-refreshing every {refresh_rate}s  ·  last update: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    time.sleep(refresh_rate)
    st.cache_data.clear()
    st.rerun()