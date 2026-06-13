# RoboTrade — High-Level Design

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        BITGET EXCHANGE                       │
│         WebSocket (candles, orderbook, fills)                │
│         REST (place orders, account, historical OHLCV)       │
└───────────────────┬─────────────────────┬────────────────────┘
                    │ ccxt.pro             │ ccxt REST
                    ▼                     ▼
┌───────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐ │
│  │ ingestion.py│───▶│  TimescaleDB │    │  Redis Cache     │ │
│  │ WebSocket   │    │  OHLCV hyper │    │  last 200 bars   │ │
│  │ feed        │───▶│  table +     │    │  live state      │ │
│  └─────────────┘    │  trade log   │    └──────────────────┘ │
│  ┌─────────────┐    └──────────────┘            │            │
│  │ features.py │◀───────────────────────────────┘            │
│  │ pandas-ta   │  EMA9/21, RSI14, ATR14, MACD,               │
│  │ indicators  │  Volume Z-score, Funding Rate                │
│  └──────┬──────┘                                             │
└─────────│─────────────────────────────────────────────────────┘
          │ feature vector
          ▼
┌───────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ rule_based   │  │ ml_strategy  │  │  rl_strategy     │   │
│  │ EMA cross +  │  │ XGBoost      │  │  PPO agent       │   │
│  │ RSI + volume │  │ classifier   │  │  (gymnasium env) │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         └─────────────────┴────────────────────┘             │
│                           │                                   │
│              signal: 'long' | 'short' | None                 │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                     RISK LAYER                                │
│  risk/manager.py                                              │
│  • Fixed fractional sizing: 1% of balance per trade          │
│  • ATR-based SL: 1.5× ATR from entry                        │
│  • TP: 2× risk (2:1 R:R)                                    │
│  • Daily drawdown gate: halt if >4% down                     │
│  • Leverage cap: 5×                                          │
│  → Approves/blocks signal, computes size + SL/TP prices      │
└───────────────────────────┬───────────────────────────────────┘
                            │ approved order spec
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                   EXECUTION LAYER                             │
│  ┌────────────────────┐    ┌───────────────────────────────┐ │
│  │ order_manager.py   │    │ portfolio.py                  │ │
│  │ place market order │    │ poll positions every 5s       │ │
│  │ set bracket SL/TP  │    │ track PnL, balance            │ │
│  │ cancel / replace   │    │ update Redis state            │ │
│  │ SIGTERM kill switch│    └───────────────────────────────┘ │
│  └────────────────────┘                                      │
└───────────────────────────┬───────────────────────────────────┘
                            │ trade events
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                  MONITORING LAYER                             │
│  ┌─────────────────┐    ┌──────────────────────────────────┐ │
│  │ Telegram alerts │    │ Grafana dashboard                │ │
│  │ trade open/close│    │ PnL panel, positions, drawdown   │ │
│  │ daily PnL       │    │ source: TimescaleDB              │ │
│  │ errors, health  │    └──────────────────────────────────┘ │
│  └─────────────────┘                                         │
└───────────────────────────────────────────────────────────────┘
```

## Data Flow (per candle close)

```
Bitget WS candle close
  → ingestion.py writes OHLCV to TimescaleDB + Redis
  → features.py reads last 200 bars from Redis
  → computes indicator vector
  → active Strategy.generate_signal(feature_df) → signal
  → risk/manager.py validates + sizes → order_spec or HALT
  → order_manager.py submits market order → confirms fill
  → sets bracket SL/TP orders
  → logs trade to TimescaleDB
  → telegram_alerts.py fires trade notification
  → portfolio.py updates state in Redis
```

## Strategy Selection

```
config.yaml: strategy.type = rule_based | ml | rl
                              ↓
main.py loads correct Strategy subclass
Strategy.generate_signal() interface identical for all three
Risk manager and execution layer unchanged regardless of strategy
```

## Backtest Loop

```
vectorbt.run_backtest()
  ← historical OHLCV via ccxt REST (1 year)
  ← same feature pipeline (features.py)
  ← same signal logic (strategy/*.py)
  → Sharpe, max_drawdown, win_rate, profit_factor
Gate: Sharpe >1.0 AND drawdown <20% → proceed to Phase 3
```

## ML Training Pipeline (Phase 5)

```
historical OHLCV (2+ years)
  → features.py (30–50 features)
  → label: price up >0.5% in next 4 candles? → 1/0/-1
  → walk-forward CV: train 12m, test 3m, slide 3m
  → XGBoost.fit() → model artifact
  → ml_strategy.py loads model → replaces rule-based signal
  → weekly retrain cron
```

## RL Training Pipeline (Phase 6)

```
envs/futures_env.py (custom gymnasium Env)
  state:  (60, N_features) window
  actions: 0=hold 1=long 2=short 3=close
  reward: realized PnL − fees − 0.1×drawdown_penalty
  ↓
stable-baselines3 PPO.learn(total_timesteps=1_000_000)
  ← train on 2020–2023 data
  → validate on 2024 (OOS)
Gate: RL OOS Sharpe > ML OOS Sharpe → replace ml_strategy
```

## Deployment

```
Local machine (Phase 0–4):
  docker compose up -d      # TimescaleDB + Redis + Grafana
  caffeinate -i python main.py   # keep-alive on Mac

Optional VPS (24/7, Phase 4+):
  Hetzner CX22 (~$5/mo)
  same docker-compose.yml
  systemd service for main.py
```

## Risk Architecture — Hard Limits

| Limit | Value | Where enforced |
|---|---|---|
| Risk per trade | 1% balance | risk/manager.py |
| Daily drawdown halt | 4% | risk/manager.py |
| Leverage cap | 5× | risk/manager.py |
| Stop-loss | 1.5× ATR | risk/manager.py |
| Take-profit | 2× risk | risk/manager.py |
| Naked positions | NEVER | order_manager.py (SL/TP always set) |
| Live start capital | $50–100 | Operational rule |
| Testnet gate | 2 weeks minimum | Operational rule |

## Key Interfaces

```python
# All strategies implement:
class Strategy(ABC):
    def generate_signal(self, df: pd.DataFrame) -> Literal['long', 'short', None]: ...

# Risk manager:
def approve(signal, df, portfolio) -> OrderSpec | None: ...
# OrderSpec: {side, size_usdt, entry_price, sl_price, tp_price, leverage}

# Order manager:
def submit(order_spec: OrderSpec) -> FilledOrder: ...
def kill_all_positions() -> None: ...  # SIGTERM handler
```
