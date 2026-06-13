# Bitget Futures Trading Bot — Free Local Blueprint

> **100% free. Runs on your local machine.** No paid APIs, no cloud subscriptions.
> Only cost: electricity + your Bitget account (free to open).

---

## Cost Breakdown

| Component | Tool | Cost |
|---|---|---|
| Language | Python 3.11 | Free |
| Exchange API | Bitget (ccxt) | Free |
| Database | PostgreSQL + TimescaleDB | Free |
| Cache | Redis | Free |
| TA / Features | pandas-ta | Free |
| Backtesting | vectorbt | Free |
| ML / RL | XGBoost, stable-baselines3 | Free |
| Dashboard | Grafana (Docker) | Free |
| Alerts | Telegram Bot | Free |
| Container runtime | Docker Desktop | Free |
| IDE | VS Code | Free |
| 24/7 runtime | Your local machine | Electricity |

**Total: $0/month**

---

## Machine Requirements (minimum)

| Spec | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Disk | 20 GB free | 50 GB SSD |
| OS | Windows 10 / macOS 12 / Ubuntu 20 | Any |
| Python | 3.10+ | 3.11 |
| Docker | Desktop / Engine | Latest |

> **Keep-alive tip:** Disable sleep on your machine while the bot runs.
> On Mac: `caffeinate -i python bot.py`
> On Windows: adjust Power Plan → "High performance" + no sleep.

---

## Project Structure

```
bitget-bot/
│
├── config/
│   ├── config.yaml          # API keys, symbols, strategy params
│   └── .env                 # secrets (never commit this)
│
├── data/
│   ├── ingestion.py         # WebSocket + REST data fetcher (ccxt.pro)
│   ├── storage.py           # TimescaleDB write/read helpers
│   └── features.py          # TA indicator calculation (pandas-ta)
│
├── strategy/
│   ├── base.py              # Abstract Strategy class
│   ├── rule_based.py        # EMA cross + RSI + ATR strategy
│   ├── ml_strategy.py       # XGBoost signal generator
│   └── rl_strategy.py       # Stable-Baselines3 PPO agent
│
├── risk/
│   └── manager.py           # Position sizing, SL/TP, drawdown gate
│
├── execution/
│   ├── order_manager.py     # Place / cancel / track orders
│   └── portfolio.py         # Track positions, PnL, balance
│
├── backtest/
│   ├── run_backtest.py      # vectorbt backtest runner
│   └── walk_forward.py      # Walk-forward validation
│
├── training/
│   ├── train_ml.py          # Train XGBoost classifier
│   ├── train_rl.py          # Train PPO agent (gymnasium env)
│   └── envs/
│       └── futures_env.py   # Custom Gym environment
│
├── monitoring/
│   ├── dashboard.py         # Streamlit fallback dashboard
│   └── telegram_alerts.py   # Trade + error notifications
│
├── docker-compose.yml       # PostgreSQL + TimescaleDB + Redis + Grafana
├── requirements.txt
└── main.py                  # Bot entry point
```

---

## Docker Setup (databases + dashboard)

`docker-compose.yml` — paste this, then run `docker compose up -d`:

```yaml
version: "3.9"
services:

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: botpass
      POSTGRES_DB: trading
    volumes:
      - tsdb_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  tsdb_data:
  grafana_data:
```

---

## Python Dependencies

```
# requirements.txt
ccxt[pro]>=4.3          # exchange connectivity + WebSocket
pandas>=2.0
pandas-ta>=0.3
numpy>=1.26
psycopg2-binary         # PostgreSQL driver
redis>=5.0
SQLAlchemy>=2.0
vectorbt>=0.26          # backtesting
python-dotenv           # config
python-telegram-bot>=20 # Telegram alerts
streamlit>=1.30         # optional dashboard
pyyaml

# ML / RL (install separately or together)
xgboost>=2.0
scikit-learn>=1.4
stable-baselines3>=2.2
gymnasium>=0.29
```

Install: `pip install -r requirements.txt`

---

## Phased Build Plan

### Phase 0 — Setup (Day 1)
- [ ] Open Bitget account → enable Futures → get API key (read + trade permissions)
- [ ] Enable Bitget **Testnet** (demo futures) — free paper trading
- [ ] Install Docker Desktop, Python 3.11, VS Code
- [ ] Clone your project folder, add `.env` with API keys
- [ ] Run `docker compose up -d` → verify TimescaleDB + Redis + Grafana are up

### Phase 1 — Data Pipeline (Days 2–4)
- [ ] Connect to Bitget via `ccxt.pro` WebSocket — stream BTCUSDT perpetual candles
- [ ] Store raw OHLCV in TimescaleDB (hypertable for time-series performance)
- [ ] Cache last 200 candles in Redis for fast access
- [ ] Calculate features: EMA-9, EMA-21, RSI-14, ATR-14, MACD, Volume Z-score, Funding Rate
- [ ] Verify: run `python data/ingestion.py` for 10 minutes, query DB to confirm rows landing

### Phase 2 — Rule-Based Strategy + Backtest (Days 5–8)
- [ ] Code simple strategy: EMA9 > EMA21 + RSI < 65 → Long; EMA9 < EMA21 + RSI > 35 → Short
- [ ] Run vectorbt backtest on 1 year of historical data (download via ccxt REST)
- [ ] Check metrics: Sharpe ratio, max drawdown, win rate, profit factor
- [ ] Tune parameters (EMA periods, RSI thresholds) — optimize on 2022–2023, validate on 2024
- [ ] **Gate:** Only proceed if backtest Sharpe > 1.0 and max drawdown < 20%

### Phase 3 — Risk Manager + Order Executor (Days 9–12)
- [ ] Risk Manager:
  - Fixed fractional sizing: risk 1% of balance per trade
  - ATR-based stop-loss (1.5× ATR from entry)
  - Take-profit: 2× risk (2:1 reward/risk ratio)
  - Daily drawdown gate: halt if down >4% on the day
- [ ] Order Manager: place market order → confirm fill → set SL/TP bracket orders
- [ ] Portfolio tracker: poll positions every 5s, update Redis state
- [ ] Kill switch: `SIGTERM` handler → close all positions immediately
- [ ] **Test on Bitget Testnet for 2 weeks minimum**

### Phase 4 — Monitoring + Alerts (Days 13–14)
- [ ] Telegram bot: notify on trade open, trade close, daily PnL, errors
- [ ] Grafana dashboard: connect to TimescaleDB, build panels for PnL, open positions, drawdown
- [ ] Log all orders to DB with timestamps, fills, slippage
- [ ] Set up simple health check: if bot process dies, Telegram alert fires

### Phase 5 — ML Upgrade (Optional, Week 3–4)
- [ ] Label training data: for each candle, did price go up >0.5% in next 4 candles? → 1/0/-1
- [ ] Engineer 30–50 features: lagged returns, rolling stats, indicator values, funding rate delta
- [ ] Train XGBoost classifier with walk-forward CV (train 12 months, test 3 months, slide)
- [ ] Replace rule-based signal with ML signal if out-of-sample Sharpe improves
- [ ] Retrain weekly on new data (cron job or manual trigger)

### Phase 6 — RL Agent (Optional, Week 5–8)
- [ ] Build custom `gymnasium` environment simulating futures trading
  - State: (60, N_features) window — last 60 candles of features
  - Actions: 0=hold, 1=long, 2=short, 3=close
  - Reward: realized PnL − fees − 0.1×drawdown_penalty
- [ ] Train PPO agent with `stable-baselines3` on 2020–2023 data (~1M steps)
- [ ] Walk-forward validate on 2024 data
- [ ] Only deploy if RL agent beats ML strategy on out-of-sample Sharpe

---

## Key Config (config.yaml template)

```yaml
exchange:
  name: bitget
  sandbox: true          # flip to false for live trading
  symbol: BTC/USDT:USDT  # USDT-margined perpetual
  timeframe: 5m

strategy:
  type: rule_based       # options: rule_based | ml | rl
  ema_fast: 9
  ema_slow: 21
  rsi_period: 14
  rsi_overbought: 65
  rsi_oversold: 35

risk:
  risk_per_trade: 0.01   # 1% of balance
  max_daily_drawdown: 0.04
  leverage: 5
  atr_sl_multiplier: 1.5
  reward_risk_ratio: 2.0

alerts:
  telegram_enabled: true
```

---

## Trading Logic (Rule-Based, Fully Coded)

```python
def generate_signal(df):
    """
    df: pandas DataFrame with columns: close, ema9, ema21, rsi, atr
    Returns: 'long', 'short', or None
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Long condition: EMA cross up + RSI not overbought + momentum
    long = (
        last['ema9'] > last['ema21'] and
        prev['ema9'] <= prev['ema21'] and   # fresh crossover
        last['rsi'] < 65 and
        last['volume'] > df['volume'].rolling(20).mean().iloc[-1]
    )

    # Short condition: EMA cross down + RSI not oversold + momentum
    short = (
        last['ema9'] < last['ema21'] and
        prev['ema9'] >= prev['ema21'] and   # fresh crossover
        last['rsi'] > 35 and
        last['volume'] > df['volume'].rolling(20).mean().iloc[-1]
    )

    if long:
        return 'long'
    elif short:
        return 'short'
    return None
```

---

## Risk Rules (Never Break These)

1. **Never go live without 2+ weeks of testnet paper trading**
2. **Start with $50–100 max on live** — prove it works before scaling
3. **Leverage ≤ 5× until you have 3 months of positive live data**
4. **Never remove the drawdown circuit breaker**
5. **Always use bracket orders (SL + TP) — never naked positions**
6. **Backtest ≠ reality** — expect live performance to be ~40% worse than backtest

---

## Timeline Summary

| Week | Milestone |
|---|---|
| Week 1 | Setup + data pipeline + rule-based strategy backtest |
| Week 2 | Risk manager + order executor + testnet paper trading |
| Week 3 | Monitoring, alerts, dashboards. Go live with tiny size ($50) |
| Week 4 | Gather live data, tune parameters |
| Month 2 | ML classifier upgrade |
| Month 3+ | RL agent (optional) |

---

*Built 100% on free open-source tools. Your local machine is the server.*
*Upgrade to a cheap VPS ($5–6/mo on Hetzner) only when you want 24/7 uptime without leaving your PC on.*
