# RoboTrade — Claude Agent Context

## Project
Bitget USDT-margined futures trading bot. Runs locally, zero subscription cost.
Exchange: Bitget (testnet → live). Primary pair: BTC/USDT perpetual, 5m candles.

## Stack
- Python 3.11
- ccxt[pro] — exchange API + WebSocket
- TimescaleDB (PostgreSQL + hypertable) — OHLCV + trade history
- Redis — candle cache (last 200 bars), live state
- pandas-ta — technical indicators
- vectorbt — backtesting
- XGBoost — ML signal (Phase 5)
- stable-baselines3 PPO — RL agent (Phase 6)
- Grafana (Docker) — dashboard
- Telegram bot — alerts
- Docker Compose — infra (TimescaleDB + Redis + Grafana)

## Project Structure
```
bitget-bot/
├── config/
│   ├── config.yaml          # strategy params, exchange settings
│   └── .env                 # API keys (never commit)
├── data/
│   ├── ingestion.py         # WebSocket + REST fetcher
│   ├── storage.py           # TimescaleDB helpers
│   └── features.py          # TA indicator pipeline
├── strategy/
│   ├── base.py              # Abstract Strategy
│   ├── rule_based.py        # EMA cross + RSI + ATR
│   ├── ml_strategy.py       # XGBoost signal
│   └── rl_strategy.py       # PPO agent
├── risk/
│   └── manager.py           # sizing, SL/TP, drawdown gate
├── execution/
│   ├── order_manager.py     # place/cancel/track orders
│   └── portfolio.py         # positions, PnL, balance
├── backtest/
│   ├── run_backtest.py      # vectorbt runner
│   └── walk_forward.py      # walk-forward validation
├── training/
│   ├── train_ml.py
│   ├── train_rl.py
│   └── envs/futures_env.py  # custom Gym env
├── monitoring/
│   ├── dashboard.py         # Streamlit fallback
│   └── telegram_alerts.py
├── docker-compose.yml
├── requirements.txt
└── main.py                  # entry point
```

## Build Phases
| Phase | Scope | Gate |
|---|---|---|
| 0 | Setup, Docker, Bitget testnet API | infra up |
| 1 | Data pipeline: WebSocket → TimescaleDB → Redis → features | data flowing |
| 2 | Rule-based strategy + vectorbt backtest | Sharpe >1.0, drawdown <20% |
| 3 | Risk manager + order executor + testnet paper trading | 2 weeks paper |
| 4 | Monitoring: Telegram + Grafana | alerts firing |
| 5 | XGBoost ML signal (optional) | beats rule-based OOS Sharpe |
| 6 | PPO RL agent (optional) | beats ML OOS Sharpe |

## Risk Rules (hard limits)
- Never go live without 2+ weeks testnet paper trading
- Start live with $50–100 max
- Leverage ≤ 5× until 3 months positive live data
- Daily drawdown gate: halt if >4% down on the day
- Always bracket orders (SL + TP), never naked positions
- Expect live perf ~40% worse than backtest

## Key Config Values
```yaml
exchange: bitget | sandbox: true | symbol: BTC/USDT:USDT | timeframe: 5m
risk_per_trade: 1% | max_daily_drawdown: 4% | leverage: 5x
atr_sl_multiplier: 1.5 | reward_risk_ratio: 2.0
```

## Infra
- TimescaleDB: localhost:5432, db=trading, pw=botpass
- Redis: localhost:6379
- Grafana: localhost:3000, admin/admin

## Conventions
- All strategy classes inherit from `strategy/base.py` `Strategy`
- Signal returns: `'long'` | `'short'` | `None`
- Risk manager must approve every signal before order submission
- All orders logged to DB with timestamps, fills, slippage
- `.env` holds secrets — never commit; use `python-dotenv`
- Config hot-reload via `config.yaml` (no restart needed for param tuning)

## Leverage Policy
Bitget BTC perp default = 20×, max = 125×.
Bot uses **5× hard cap** until 3 months positive live data.
Never raise leverage in code without explicit user decision.

## Coding Standards
- **Type hints** on every function signature (`def foo(df: pd.DataFrame) -> str | None:`)
- **No magic numbers** — all thresholds in `config.yaml`, loaded once at startup
- **Single responsibility** — each module does one thing; no cross-layer calls (e.g. strategy never touches DB)
- **Logging** — use `logging` stdlib, structured: `logger.info("signal", extra={"signal": s, "bar_ts": ts})`
- **Error handling** — catch only expected exceptions at system boundaries; let unexpected ones propagate + alert
- **No silent failures** — every caught exception must log.error + optionally Telegram alert
- **Config hot-reload** — read config.yaml values fresh each candle via `config.load()`; no module-level caching of params
- **Never hardcode secrets** — use `python-dotenv`; `.env` never committed
- **Tests** — unit test risk/manager.py and features.py at minimum; use `pytest`
- **Formatter**: `black` + `ruff`; type checker: `mypy --strict` on core modules
- **Dependency injection** — pass DB/Redis clients in; never import as module-level singletons
- **Idempotent DB writes** — upsert on `(symbol, timeframe, open_time)` primary key; no duplicates

## Review Workflow — Before Every Implementation
Claude must follow this sequence. No code written until checks done.

### 1 — Understand
- State what module/function is being built and why
- Identify which layer it lives in (data / strategy / risk / execution / monitoring)
- Confirm no existing module already does this

### 2 — Design Check
- Does it violate single-responsibility? Refactor scope if yes.
- Does it cross layer boundaries it shouldn't?
- Are all magic numbers going to config.yaml?
- Does it need DB/Redis access? If yes, inject clients — don't import singletons.

### 3 — Risk Check (for execution/risk changes)
- Could this change allow a naked position (no SL/TP)?
- Could this bypass the drawdown gate?
- Could this change leverage beyond 5×?
- If any yes → STOP. Get explicit user confirmation before proceeding.

### 4 — Implementation
- Write code following coding standards above
- Add type hints, logging, error handling
- No comments unless WHY is non-obvious

### 5 — Self-Review
- Read the diff: does it do exactly what was designed?
- Are there edge cases that could cause silent failures?
- Would a fresh reader understand this without context?
- Run `black` + `ruff` + `mypy` mentally on the diff

### 6 — Propose, Don't Just Do
For any change touching: risk/manager.py, order_manager.py, config defaults, leverage, SL/TP logic:
→ Show the plan + diff preview → wait for user approval → then implement.

## Current Status
Project initialized. No code written yet. Blueprint in `bitget-futures-bot-blueprint.md`.
Start with Phase 0 (infra) then Phase 1 (data pipeline).
