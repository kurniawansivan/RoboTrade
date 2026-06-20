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

import ccxt
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
def get_public_exchange():
    """Public ccxt exchange — no API keys needed for ticker data."""
    return ccxt.bitget()


def fetch_ticker_prices(syms: list[str]) -> dict[str, float]:
    """Fetch current price for each symbol directly from Bitget REST. Always fresh."""
    exchange = get_public_exchange()
    prices = {}
    for sym in syms:
        try:
            ticker = exchange.fetch_ticker(sym)
            prices[sym] = float(ticker["last"])
        except Exception:
            prices[sym] = 0.0
    return prices


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

def get_positions_for(sym: str) -> list:
    raw = r.get(f"positions:{sym.replace('/', '_')}")
    return json.loads(raw) if raw else []


def get_all_open_positions() -> list[dict]:
    """All open positions across every configured symbol."""
    out = []
    for sym in symbols:
        for p in get_positions_for(sym):
            p["_symbol"] = sym
            out.append(p)
    return out


def push_command(action: str, sym: str | None = None) -> None:
    """Send a command to the bot via Redis (consumed by command_consumer)."""
    payload = {"action": action}
    if sym:
        payload["symbol"] = sym
    r.lpush("bot:commands", json.dumps(payload))


def is_halted() -> bool:
    return r.get("bot:halted") == "1"


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

_lev = os.environ.get("LEVERAGE") or config.get("risk", {}).get("leverage", "?")
_strat = config.get("strategy", {}).get("type", "rule_based")
_mode = config.get("strategy", {}).get("mode", "")
_lev_warn = "  ⚠️ HIGH" if str(_lev).isdigit() and int(_lev) > 20 else ""
st.caption(f"Strategy: **{_strat}** {_mode}  |  Leverage: **{_lev}×**{_lev_warn}  |  Symbols: {len(symbols)}")


@st.fragment(run_every=2)
def control_center() -> None:
    """Live control center — reruns every 2s. Metrics, positions, manual controls."""
    prices = fetch_ticker_prices(symbols)
    bal = get_balance()
    day_s = get_day_start_balance()
    pnl = bal - day_s if day_s > 0 else 0.0
    pnl_pct = (pnl / day_s * 100) if day_s > 0 else 0.0
    open_pos = get_all_open_positions()
    btc_px = prices.get(symbol, 0.0)
    dd = ((day_s - bal) / day_s * 100) if day_s > 0 else 0.0
    halted = is_halted()

    # Total unrealized PnL across positions
    total_upnl = sum(float(p.get("unrealizedPnl") or 0) for p in open_pos)

    # ── Metric row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("BTC/USDT", f"${btc_px:,.2f}")
    c2.metric("Balance", f"${bal:,.2f}")
    c3.metric("Daily P&L", f"${pnl:+.2f}", f"{pnl_pct:+.2f}%",
              delta_color="normal" if pnl >= 0 else "inverse")
    c4.metric("Unrealized P&L", f"${total_upnl:+.2f}",
              delta_color="normal" if total_upnl >= 0 else "inverse")
    c5.metric("Open Positions", len(open_pos))
    c6.metric("Daily DD", f"{dd:.2f}%", delta_color="inverse" if dd > 2 else "normal")

    # ── Bot controls ─────────────────────────────────────────────────────────
    status = "🔴 HALTED" if halted else "🟢 TRADING"
    bc1, bc2, bc3 = st.columns([2, 1, 1])
    bc1.markdown(f"**Bot status:** {status}")
    if halted:
        if bc2.button("▶️ Resume", use_container_width=True):
            push_command("resume"); st.rerun()
    else:
        if bc2.button("⏸️ Halt entries", use_container_width=True):
            push_command("halt"); st.rerun()
    if bc3.button("🛑 Flatten ALL", type="primary", use_container_width=True):
        push_command("flatten_all"); st.toast("Flatten-all sent"); st.rerun()

    # ── Open positions table + per-position close ──────────────────────────
    st.subheader("💼 Open Positions")
    if open_pos:
        for i, p in enumerate(open_pos):
            sym = p["_symbol"]
            side = p.get("side", "")
            contracts = float(p.get("contracts") or 0)
            entry = float(p.get("entryPrice") or 0)
            upnl = float(p.get("unrealizedPnl") or 0)
            mark = prices.get(sym, entry)
            pct = ((mark - entry) / entry * 100 * (1 if side == "long" else -1)) if entry else 0.0
            liq = p.get("liquidationPrice")
            arrow = "🟢 LONG" if side == "long" else "🔴 SHORT"

            pc = st.columns([1.4, 1, 1.4, 1.4, 1.2, 1.2, 1])
            pc[0].markdown(f"**{sym.split('/')[0]}**  {arrow}")
            pc[1].markdown(f"{contracts:g}")
            pc[2].markdown(f"entry ${entry:,.4f}")
            pc[3].markdown(f"mark ${mark:,.4f}")
            pc[4].markdown(f"{'🟢' if upnl>=0 else '🔴'} ${upnl:+.2f}")
            pc[5].markdown(f"{pct:+.2f}%")
            if pc[6].button("Close", key=f"close_{sym}_{i}", use_container_width=True):
                push_command("close", sym); st.toast(f"Close {sym} sent"); st.rerun()
            if liq:
                st.caption(f"   ↳ {sym.split('/')[0]} liquidation ≈ ${float(liq):,.4f}")
    else:
        st.info("No open positions. Bot scanning for signals.")

    # ── Symbol scanner ───────────────────────────────────────────────────────
    st.subheader("🔭 Symbol Scanner")
    all_states = get_all_indicator_states()
    if all_states:
        rows = []
        for sym, state in all_states.items():
            htf = state.get("htf_trend", 0)
            htf_label = "🟢 Bull" if htf == 1 else ("🔴 Bear" if htf == -1 else "⚪ Neutral")
            sig = state.get("signal", "none")
            sig_display = f"🟢 {sig.upper()}" if sig == "long" else (
                f"🔴 {sig.upper()}" if sig == "short" else "—")
            live_px = prices.get(sym, state.get("price", 0))
            rows.append({
                "Symbol": sym.split("/")[0],
                "Price": f"${live_px:,.4f}",
                "Signal": sig_display,
                "RSI": f"{state.get('rsi', 0):.1f}",
                "MACD Hist": f"{state.get('macd_hist', 0):.4f}",
                "1h Trend": htf_label,
                "Last Bar": state.get("bar_ts", "—")[:19],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Waiting for symbol data… (bot must be running)")
    st.caption("⚡ Live — prices, P&L, positions & controls update every 2s")


control_center()

st.divider()

# Static data (indicators, chart, trades) — full page refresh handles these
indicator = get_indicator_state()
df_candles = get_candles_from_redis()

# Row 2: Signal + Indicators
col_sig, col_ind = st.columns([1, 2])

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

st.caption(f"⚡ Prices update every 2s  |  Chart/trades refresh on page reload  |  Mode: {'SANDBOX' if sandbox else 'LIVE'}")
