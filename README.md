# RoboTrade

Algorithmic futures trading bot for Bitget USDT-margined perpetuals. Runs entirely on your local machine — zero cloud cost, zero subscription fee.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Exchange](https://img.shields.io/badge/Exchange-Bitget-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Paper%20Trading-yellow)

---

## Overview

RoboTrade streams live 5-minute OHLCV candles from Bitget via WebSocket, computes technical indicators, generates trading signals, and executes bracket orders (entry + SL + TP) automatically. A Streamlit dashboard and Telegram alerts provide real-time visibility.

**Current strategy:** EMA 20/200 crossover + 1h HTF trend filter + MACD confirmation + RSI + volume filter. Signals across 5 concurrent symbols: BTC, ETH, SOL, BNB, XRP.

---

## Architecture

```
Bitget Exchange
      │
      │  WebSocket (ccxt.pro) — 5 parallel streams
      ▼
  Bot  (main.py)
      │  on every closed 5m candle:
      │   1. compute indicators (EMA, RSI, ATR, MACD, HTF trend)
      │   2. generate signal  → strategy/rule_based.py
      │   3. risk check       → risk/manager.py  (sizing, SL/TP, drawdown gate)
      │   4. place order      → execution/order_manager.py
      │   5. write state      → Redis
      │
      ├──────────────────────┐
      ▼                      ▼
   Redis                TimescaleDB
 (live state)         (OHLCV history + trades)
      │
      │  poll every 5s
      ▼
 Dashboard (Streamlit :8501)   +   Telegram Bot
```

---

## Features

- **Multi-symbol scanning** — 5 USDT-perp pairs running concurrently
- **Rule-based strategy** — EMA crossover with 1h higher-timeframe trend filter
- **ATR-based bracket orders** — dynamic SL (1.5× ATR) + TP (3× SL), always bracketed
- **Risk manager** — 1% per-trade risk, 5× leverage cap, 4% daily drawdown gate
- **Kill switch** — SIGTERM closes all positions and cancels all orders cleanly
- **Live dashboard** — candlestick chart, indicator table, open positions, trade log
- **Telegram alerts** — trade opened/closed, drawdown gate, daily summary
- **Walk-forward validation** — out-of-sample backtest framework before going live
- **TimescaleDB** — time-series optimised OHLCV storage with hypertable
- **Config hot-reload** — change strategy params without restarting the bot
- **22 unit tests** — risk manager + feature pipeline fully covered

---

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Exchange API | ccxt / ccxt.pro (WebSocket) |
| Database | TimescaleDB (PostgreSQL hypertable) |
| Cache / state bus | Redis |
| TA indicators | pandas-ta |
| Backtesting | vectorbt |
| Dashboard | Streamlit + Plotly |
| Alerts | python-telegram-bot |
| Infra | Docker Compose |
| ML (Phase 5) | XGBoost + scikit-learn |
| RL (Phase 6) | stable-baselines3 PPO + Gymnasium |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop | For TimescaleDB, Redis, Grafana |
| Python 3.12 | via pyenv recommended |
| Bitget account | Free — enable Futures |
| Telegram account | For alerts |

---

## Quick Start

### 1. Clone and create virtualenv

```bash
git clone <repo-url> && cd RoboTrade
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start infrastructure

```bash
docker compose up -d
```

Starts TimescaleDB on `:5434`, Redis on `:6379`, Grafana on `:3000`.

### 3. Configure secrets

```bash
cp config/.env.example config/.env
```

Edit `config/.env`:

```env
# Bitget Demo Trading API (bitget.com → Demo → API Management)
BITGET_API_KEY=your_demo_api_key
BITGET_API_SECRET=your_demo_api_secret
BITGET_API_PASSPHRASE=your_demo_passphrase

# Telegram (BotFather → /newbot)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Database (matches docker-compose.yml defaults)
DB_HOST=localhost
DB_PORT=5434
DB_NAME=trading
DB_USER=postgres
DB_PASSWORD=botpass

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 4. Run the bot

```bash
python main.py
```

### 5. Run the dashboard (separate terminal)

```bash
streamlit run monitoring/dashboard.py
```

Open `http://localhost:8501`.

---

## Configuration

All strategy and risk parameters live in `config/config.yaml`. The bot hot-reloads this file on every candle — no restart needed for parameter changes.

```yaml
exchange:
  name: bitget
  sandbox: true          # ← flip to false for live trading
  symbols:
    - BTC/USDT:USDT
    - ETH/USDT:USDT
    - SOL/USDT:USDT
    - BNB/USDT:USDT
    - XRP/USDT:USDT
  timeframe: 5m

strategy:
  type: rule_based
  ema_fast: 20           # 5m EMA crossover
  ema_slow: 200
  rsi_period: 14
  rsi_overbought: 60
  rsi_oversold: 40
  atr_period: 14
  volume_ma_period: 20
  # HTF filter: 1h EMA20/50 computed internally from 5m data

risk:
  risk_per_trade: 0.01   # 1% of balance per trade
  max_daily_drawdown: 0.04
  leverage: 5            # hard-capped at 5× in code
  atr_sl_multiplier: 1.5
  reward_risk_ratio: 3.0
  max_open_positions: 3
```

---

## Project Structure

```
RoboTrade/
├── config/
│   ├── config.yaml          # strategy + risk params (hot-reloaded)
│   └── .env                 # secrets (never commit)
├── data/
│   ├── ingestion.py         # WebSocket + REST fetcher (ccxt.pro)
│   ├── storage.py           # TimescaleDB helpers + hypertable init
│   └── features.py          # TA pipeline: EMA, RSI, ATR, MACD, HTF trend
├── strategy/
│   ├── base.py              # Abstract Strategy class
│   └── rule_based.py        # EMA cross + HTF + MACD + RSI + volume
├── risk/
│   └── manager.py           # sizing, SL/TP calc, drawdown gate, leverage cap
├── execution/
│   ├── order_manager.py     # bracket orders, emergency close, kill switch
│   └── portfolio.py         # balance + position sync → Redis
├── backtest/
│   ├── run_backtest.py      # vectorbt runner with gate check
│   ├── walk_forward.py      # 12m train / 3m test sliding window
│   └── tune_params.py       # parameter sweep with IS/OOS split
├── monitoring/
│   ├── dashboard.py         # Streamlit live dashboard
│   └── telegram_alerts.py   # trade + error + daily summary alerts
├── training/                # Phase 5–6 (ML/RL — coming)
│   ├── train_ml.py
│   ├── train_rl.py
│   └── envs/futures_env.py
├── tests/
│   ├── test_risk_manager.py # 10 tests
│   └── test_features.py     # 12 tests
├── docker-compose.yml
├── requirements.txt
└── main.py                  # entry point
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **22 passed**.

---

## Backtesting

```bash
# Single backtest (2023–2025)
python backtest/run_backtest.py 2023-01-01 2025-01-01

# Walk-forward validation (2022–2025, 4 folds)
python backtest/walk_forward.py 2022-01-01 2025-06-01

# Parameter sweep
python backtest/tune_params.py
```

### Current backtest results (rule-based, 2023–2025)

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.35 |
| Max Drawdown | 4.41% |
| Win Rate | 44.7% |
| Profit Factor | 1.18 |
| Total Return | +1.73% |
| Strategy | EMA20/200 + 1h HTF + MACD |

Walk-forward MaxDD across all folds: **≤ 1.6%** — capital protected in all market regimes.

---

## Build Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Docker infra, Bitget demo API, secrets | ✅ Done |
| 1 | WebSocket data pipeline → TimescaleDB → Redis → features | ✅ Done |
| 2 | Rule-based strategy + vectorbt backtest + walk-forward | ✅ Done |
| 3 | Risk manager + order executor + kill switch + tests | ✅ Done |
| 4 | Telegram alerts + Streamlit dashboard + multi-symbol | ✅ Done |
| 5 | XGBoost ML signal + walk-forward trainer | ✅ Done (train + validate before deploying) |
| 6 | PPO RL agent (Gymnasium env) | 🔜 Planned |

### Machine Learning (Phase 5)

Train an XGBoost direction classifier across all configured symbols:

```bash
python training/train_ml.py 2023-01-01 2025-01-01
```

This runs walk-forward CV (train 12m / test 3m), reports out-of-sample **directional accuracy**,
and saves the model to `training/models/xgb_signal.json`. Only deploy if directional accuracy > 0.50.

To switch the bot to the ML signal, set in `config/config.yaml`:

```yaml
strategy:
  type: ml
  ml:
    proba_threshold: 0.55   # min class probability to trade
```

Then restart the bot. The rule-based and ML strategies share the same risk manager, bracket orders, and dashboard.

### Leverage (env-driven)

Leverage is controlled by the `LEVERAGE` env var (overrides `config.yaml`), set on the exchange at startup:

```env
LEVERAGE=5      # in config/.env — no hard cap; config risk.max_leverage is a soft guard
```

---

## Risk Rules (Hard Limits)

These are enforced in code and must never be bypassed:

1. **Never go live without 2+ weeks of demo paper trading**
2. **Start live with small size** — prove it works before scaling
3. **Leverage ≤ 5×** until 3 months of positive live data
4. **Daily drawdown gate: halt if down >4%** on the day
5. **Always bracket orders** (SL + TP) — naked positions are blocked in code
6. **Expect live performance ~40% worse** than backtest

---

## Signal Logic

Two modes (config `strategy.mode`):

```
mode: trend_align  (default — active, ~1-10 signals/day/symbol)
  Long entry when ALL met:
    ✓ EMA20 > EMA200 on 5m (trend aligned up)
    ✓ 1h EMA20 ≥ EMA50 (higher-timeframe not bearish)
    ✓ MACD histogram > 0 (momentum confirming)
    ✓ RSI < rsi_overbought (not exhausted)
    ✓ 1h ADX ≥ adx_threshold (market is trending)
    ✓ cooldown elapsed (signal_cooldown_bars since last signal)
  Short: mirror with inverted filters

mode: fresh_cross  (rare — original, only on a fresh EMA20/200 crossover)

Exit: ATR-based bracket
  SL = entry ± 1.5 × ATR
  TP = entry ± 4.5 × ATR  (3:1 reward/risk)
```

---

## Dashboard

`http://localhost:8501` — auto-refreshes every 5s.

- **Top bar** — BTC price, balance, daily P&L, drawdown %, open positions
- **Signal panel** — current signal per symbol in large green/red text
- **Symbol scanner** — live table: all 5 symbols, RSI, MACD, 1h trend, signal
- **Chart** — last 100 candles with EMA20/200 overlay + volume
- **Trade log** — all trades with entry, SL, TP, P&L from DB

---

## Telegram Alerts

| Event | Message |
|---|---|
| Bot started | 🤖 symbols, mode |
| Trade opened | 🟢/🔴 entry, SL, TP, qty, risk$ |
| Drawdown gate | 🚨 halted for the day |
| Daily summary | 📈/📉 balance, P&L, trade count |
| Bot stopped | 🛑 reason |

---

## Grafana

`http://localhost:3000` (admin / admin) — connect to TimescaleDB datasource at `localhost:5434`, database `trading` for custom panels.

---

## License

MIT — free to use, modify, and distribute. No warranty. Trading involves risk of loss.

---

> **Disclaimer:** This software is for educational purposes. Past backtest performance does not guarantee future results. Never risk money you cannot afford to lose.
