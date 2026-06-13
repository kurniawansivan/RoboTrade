"""
Streamlit live dashboard. Reads from Redis + TimescaleDB.
Run: streamlit run monitoring/dashboard.py
Auto-refreshes every 5s.
"""

import json
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import redis
import streamlit as st
import yaml
from dotenv import load_dotenv
from sqlalchemy import text

from data.storage import build_engine, fetch_candles

load_dotenv(dotenv_path="config/.env")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoboTrade",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load config ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_config() -> dict:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.Redis(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        decode_responses=True,
    )

@st.cache_resource
def get_engine():
    return build_engine(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        db=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )

config = get_config()
r = get_redis()
engine = get_engine()
symbol = config["exchange"]["symbol"]
symbols = config["exchange"].get("symbols", [symbol])
sandbox = config["exchange"].get("sandbox", True)

# ── Auto-refresh every 5s ───────────────────────────────────────────────────
st.markdown("""
<style>
.big-metric { font-size: 2.2rem; font-weight: 700; }
.signal-long { color: #00d4aa; font-size: 1.8rem; font-weight: 700; }
.signal-short { color: #ff4b6e; font-size: 1.8rem; font-weight: 700; }
.signal-none { color: #888; font-size: 1.8rem; font-weight: 700; }
.status-ok { color: #00d4aa; }
.status-warn { color: #ffa500; }
</style>
""", unsafe_allow_html=True)

# ── Fetch data from Redis ────────────────────────────────────────────────────
def get_candles_from_redis() -> pd.DataFrame:
    key = f"candles:{symbol}:5m"
    raw = r.get(key)
    if not raw:
        return pd.DataFrame()
    data = json.loads(raw)
    df = pd.DataFrame(data)
    if not df.empty:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df

def get_balance() -> float:
    raw = r.get("balance")
    return float(raw) if raw else 0.0

def get_day_start_balance() -> float:
    raw = r.get("balance:day_start")
    return float(raw) if raw else 0.0

def get_positions() -> list:
    key = f"positions:{symbol.replace('/', '_')}"
    raw = r.get(key)
    return json.loads(raw) if raw else []

def get_last_signal() -> str:
    raw = r.get("last_signal")
    return raw or "none"

def get_recent_trades() -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT side, entry_price, exit_price, sl_price, tp_price,
                           quantity, pnl, entry_time, exit_time, strategy_type
                    FROM trades
                    ORDER BY entry_time DESC
                    LIMIT 20
                """),
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()

def get_indicator_state() -> dict:
    raw = r.get("indicator_state")
    if not raw:
        return {}
    return json.loads(raw)

def get_all_indicator_states() -> dict[str, dict]:
    """Return indicator state per symbol."""
    result = {}
    for sym in symbols:
        key = f"indicator_state:{sym.replace('/', '_')}"
        raw = r.get(key)
        if raw:
            result[sym] = json.loads(raw)
    return result

# ── Layout ───────────────────────────────────────────────────────────────────
mode_badge = "🟡 SANDBOX" if sandbox else "🔴 LIVE"
st.title(f"🤖 RoboTrade  {mode_badge}")

# Row 1: key metrics
balance = get_balance()
day_start = get_day_start_balance()
daily_pnl = balance - day_start if day_start > 0 else 0.0
daily_pnl_pct = (daily_pnl / day_start * 100) if day_start > 0 else 0.0
positions = get_positions()
indicator = get_indicator_state()

df_candles = get_candles_from_redis()
last_price = float(df_candles["close"].iloc[-1]) if not df_candles.empty else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("BTC/USDT", f"${last_price:,.2f}")
with col2:
    st.metric("Balance (USDT)", f"${balance:,.2f}")
with col3:
    pnl_color = "normal" if daily_pnl >= 0 else "inverse"
    st.metric("Daily P&L", f"${daily_pnl:+.2f}", f"{daily_pnl_pct:+.2f}%", delta_color=pnl_color)
with col4:
    st.metric("Open Positions", len(positions))
with col5:
    drawdown_pct = ((day_start - balance) / day_start * 100) if day_start > 0 else 0.0
    dd_color = "inverse" if drawdown_pct > 2 else "normal"
    st.metric("Daily DD", f"{drawdown_pct:.2f}%", delta_color=dd_color)

st.divider()

# Row 2: Signal + Indicators + Position
col_sig, col_ind, col_pos = st.columns([1, 2, 2])

with col_sig:
    st.subheader("📡 Signal")
    sig = indicator.get("signal", "none")
    if sig == "long":
        st.markdown('<div class="signal-long">▲ LONG</div>', unsafe_allow_html=True)
    elif sig == "short":
        st.markdown('<div class="signal-short">▼ SHORT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-none">— NONE</div>', unsafe_allow_html=True)

    htf = indicator.get("htf_trend", 0)
    htf_label = "🟢 Bullish" if htf == 1 else ("🔴 Bearish" if htf == -1 else "⚪ Neutral")
    st.write(f"**1h Trend:** {htf_label}")

    last_bar = indicator.get("bar_ts", "—")
    st.caption(f"Last bar: {last_bar}")

with col_ind:
    st.subheader("📊 Indicators")
    if indicator:
        rsi_val = indicator.get("rsi", 0)
        rsi_color = "🟢" if 40 < rsi_val < 60 else ("🔴" if rsi_val > 70 or rsi_val < 30 else "🟡")
        macd_hist = indicator.get("macd_hist", 0)
        macd_color = "🟢" if macd_hist > 0 else "🔴"
        adx_val = indicator.get("adx", 0)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("RSI (14)", f"{rsi_val:.1f}", help="<30 oversold, >70 overbought")
            st.metric("EMA Fast (20)", f"{indicator.get('ema_fast', 0):,.2f}")
        with col_b:
            st.metric("MACD Hist", f"{macd_hist:.2f}")
            st.metric("EMA Slow (200)", f"{indicator.get('ema_slow', 0):,.2f}")

        ema_f = indicator.get("ema_fast", 0)
        ema_s = indicator.get("ema_slow", 0)
        if ema_f and ema_s:
            cross_status = "EMA above 200 (bullish)" if ema_f > ema_s else "EMA below 200 (bearish)"
            st.caption(f"5m: {cross_status}")
    else:
        st.info("Waiting for first candle from bot…")

with col_pos:
    st.subheader("💼 Open Position")
    if positions:
        for pos in positions:
            side = pos.get("side", "")
            contracts = pos.get("contracts", 0)
            entry = pos.get("entryPrice", 0)
            upnl = pos.get("unrealizedPnl", 0)
            liq = pos.get("liquidationPrice", "—")
            color = "🟢" if side == "long" else "🔴"
            st.markdown(f"**{color} {side.upper()}** | {contracts} contracts @ ${float(entry):,.2f}")
            pnl_color = "🟢" if float(upnl or 0) >= 0 else "🔴"
            st.metric("Unrealized P&L", f"${float(upnl or 0):+.2f}")
            st.write(f"Liquidation: ${float(liq):,.2f}" if liq != "—" else "Liq: —")
    else:
        st.info("No open position")
        st.caption("Bot waits for EMA20/200 crossover + 1h HTF + MACD signal")

st.divider()

# Row 2.5: Multi-symbol scanner
st.subheader("🔭 Symbol Scanner")
all_states = get_all_indicator_states()
if all_states:
    rows = []
    for sym, state in all_states.items():
        htf = state.get("htf_trend", 0)
        htf_label = "🟢 Bull" if htf == 1 else ("🔴 Bear" if htf == -1 else "⚪ Neutral")
        sig = state.get("signal", "none")
        sig_display = f"🟢 {sig.upper()}" if sig == "long" else (f"🔴 {sig.upper()}" if sig == "short" else "—")
        rows.append({
            "Symbol": sym.split("/")[0],
            "Price": f"${state.get('price', 0):,.4f}",
            "Signal": sig_display,
            "RSI": f"{state.get('rsi', 0):.1f}",
            "MACD Hist": f"{state.get('macd_hist', 0):.4f}",
            "1h Trend": htf_label,
            "Last Bar": state.get("bar_ts", "—")[:19],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Waiting for symbol data… (bot must be running)")

st.divider()

# Row 3: Price chart
st.subheader("📈 BTC/USDT — Last 100 Candles (5m)")
if not df_candles.empty:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df_plot = df_candles.tail(100).copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot["open_time"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="BTC/USDT",
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff4b6e",
    ), row=1, col=1)

    # EMAs if available
    if "ema_fast" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot["open_time"], y=df_plot["ema_fast"],
            name="EMA 20", line=dict(color="#ffa500", width=1.5),
        ), row=1, col=1)
    if "ema_slow" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot["open_time"], y=df_plot["ema_slow"],
            name="EMA 200", line=dict(color="#8888ff", width=1.5),
        ), row=1, col=1)

    # Volume
    colors = ["#00d4aa" if c >= o else "#ff4b6e"
              for c, o in zip(df_plot["close"], df_plot["open"])]
    fig.add_trace(go.Bar(
        x=df_plot["open_time"], y=df_plot["volume"],
        name="Volume", marker_color=colors, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        height=450,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Waiting for candle data from bot…")

st.divider()

# Row 4: Recent trades
st.subheader("📋 Recent Trades")
df_trades = get_recent_trades()
if not df_trades.empty:
    df_trades["pnl"] = df_trades["pnl"].apply(
        lambda x: f"${x:+.4f}" if pd.notna(x) else "Open"
    )
    df_trades["side"] = df_trades["side"].apply(
        lambda s: "🟢 BUY" if s == "buy" else "🔴 SELL"
    )
    st.dataframe(
        df_trades[["side", "entry_price", "sl_price", "tp_price", "quantity", "pnl", "entry_time"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No trades yet — bot is live and scanning for signals.")

# Row 5: Strategy status
with st.expander("⚙️ Strategy Config"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.json({
            "strategy": config.get("strategy", {}),
        })
    with col_s2:
        st.json({
            "risk": config.get("risk", {}),
        })

# Auto-refresh
st.caption(f"Auto-refreshes every 5s  |  Symbol: {symbol}  |  Mode: {'SANDBOX' if sandbox else 'LIVE'}")
st.markdown("""
<script>
setTimeout(function() { window.location.reload(); }, 5000);
</script>
""", unsafe_allow_html=True)
