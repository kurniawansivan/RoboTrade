"""
vectorbt backtest runner.
Downloads historical OHLCV, computes features, runs backtest, prints metrics.
Layer: backtest. No execution/risk imports (simulation only).
"""

import asyncio
import logging
import os
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
from dotenv import load_dotenv

from data.features import compute_features

load_dotenv(dotenv_path="config/.env")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_historical_rest(
    symbol: str,
    timeframe: str,
    since_str: str,
    until_str: Optional[str] = None,
    sandbox: bool = False,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Fetch full OHLCV history via ccxt REST (paginated).
    Caches result to parquet — subsequent calls load from disk instantly.
    since_str: ISO date string e.g. '2023-01-01'
    """
    import os as _os
    sym_safe = symbol.replace("/", "_").replace(":", "_")
    cache_file = f"{cache_dir}/{sym_safe}_{timeframe}_{since_str}_{until_str or 'now'}.parquet"

    if _os.path.exists(cache_file):
        logger.info("loading from cache", extra={"file": cache_file})
        df = pd.read_parquet(cache_file)
        logger.info("cache loaded", extra={"rows": len(df)})
        return df

    exchange = ccxt.bitget(
        {
            "apiKey": os.environ.get("BITGET_API_KEY", ""),
            "secret": os.environ.get("BITGET_API_SECRET", ""),
            "password": os.environ.get("BITGET_API_PASSPHRASE", ""),
            "timeout": 30000,
        }
    )
    exchange.set_sandbox_mode(sandbox)

    since_ms = exchange.parse8601(since_str + "T00:00:00Z")
    until_ms = exchange.parse8601(until_str + "T00:00:00Z") if until_str else None
    # Bitget returns max 200 candles per request — paginate by advancing since_ms
    limit = 200
    all_candles: list = []

    logger.info("fetching historical data", extra={"symbol": symbol, "since": since_str})
    retries = 0
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            retries = 0
        except (ccxt.RequestTimeout, ccxt.NetworkError) as exc:
            retries += 1
            if retries > 5:
                raise
            import time
            logger.warning("timeout, retry %d/5", retries, exc_info=exc)
            time.sleep(2 ** retries)
            continue

        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if until_ms and last_ts >= until_ms:
            break
        since_ms = last_ts + 1
        if len(batch) < limit and (until_ms is None or last_ts >= until_ms):
            break

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    if until_ms:
        df = df[df["open_time"] < pd.Timestamp(until_ms, unit="ms", tz="UTC")]

    _os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_file, index=False)
    logger.info("fetched and cached candles", extra={"rows": len(df), "file": cache_file})
    return df


def build_signals(df: pd.DataFrame, strategy_config: dict) -> tuple[pd.Series, pd.Series]:
    """
    MACD histogram zero-cross + HTF trend + RSI + volume signals.
    Returns (long_entries, short_entries) boolean Series.
    """
    rsi = df["rsi"]
    vol = df["volume"]
    vol_ma = vol.rolling(strategy_config["volume_ma_period"]).mean()

    rsi_ob = strategy_config["rsi_overbought"]
    rsi_os = strategy_config["rsi_oversold"]

    vol_ok = vol > vol_ma

    # EMA crossover on 5m (entry trigger)
    ema_fast = df["ema_fast"] if "ema_fast" in df.columns else pd.Series(df["close"])
    ema_slow = df["ema_slow"] if "ema_slow" in df.columns else pd.Series(df["close"])
    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_dn = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    # MACD confirmation (histogram > 0 for longs, < 0 for shorts)
    macd_hist = df["macd_hist"] if "macd_hist" in df.columns else pd.Series(0.0, index=df.index)
    macd_bull = macd_hist > 0
    macd_bear = macd_hist < 0

    # 1h HTF filter — allow when not actively against us
    htf = df["htf_trend"] if "htf_trend" in df.columns else pd.Series(0, index=df.index)
    htf_bull = htf >= 0    # 1h not actively bearish (neutral OK for longs)
    htf_bear = htf <= 0    # 1h not actively bullish (neutral OK for shorts)

    long_entry = cross_up & (rsi < rsi_ob) & vol_ok & macd_bull & htf_bull
    short_entry = cross_dn & (rsi > rsi_os) & vol_ok & macd_bear & htf_bear

    return long_entry, short_entry


def run_backtest(
    df_features: pd.DataFrame,
    strategy_config: dict,
    risk_config: dict,
    init_cash: float = 1000.0,
) -> dict:
    """
    Run vectorbt backtest on feature DataFrame.
    Returns dict of metrics: sharpe, max_drawdown, win_rate, profit_factor, total_return.
    """
    # Use numpy arrays throughout — avoids vectorbt index broadcast issues
    price = df_features["close"].values
    open_time = pd.DatetimeIndex(df_features["open_time"])
    long_entry_arr, short_entry_arr = build_signals(df_features, strategy_config)
    long_entry_np = long_entry_arr.values
    short_entry_np = short_entry_arr.values

    # ATR-based dynamic SL/TP (fraction of price)
    atr_np = df_features["atr"].values
    sl_mult: float = risk_config["atr_sl_multiplier"]
    rr: float = risk_config["reward_risk_ratio"]
    sl_stop_np = (atr_np * sl_mult) / price
    tp_stop_np = (atr_np * sl_mult * rr) / price

    price_s = pd.Series(price, index=open_time)

    # vectorbt portfolio simulation — long-only first pass (Phase 2 rule-based)
    pf = vbt.Portfolio.from_signals(
        close=price_s,
        entries=pd.Series(long_entry_np, index=open_time),
        exits=pd.Series(short_entry_np, index=open_time),
        sl_stop=pd.Series(sl_stop_np, index=open_time),
        tp_stop=pd.Series(tp_stop_np, index=open_time),
        init_cash=init_cash,
        fees=0.0006,        # Bitget taker fee 0.06%
        slippage=0.0002,    # estimated slippage
        freq="5min",
    )

    stats = pf.stats()
    total_ret = float(stats.get("Total Return [%]", 0.0))
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    max_dd = float(stats.get("Max Drawdown [%]", 100.0))
    win_rate_raw = stats.get("Win Rate [%]", 0.0)
    win_rate = float(win_rate_raw) if not pd.isna(win_rate_raw) else 0.0
    profit_factor_raw = stats.get("Profit Factor", 0.0)
    profit_factor = float(profit_factor_raw) if not pd.isna(profit_factor_raw) else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "total_return_pct": round(total_ret, 2),
        "n_trades": int(pf.stats().get("Total Trades", 0)),
        "portfolio": pf,
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}=== Backtest Results ===")
    print(f"  Sharpe Ratio    : {metrics['sharpe']}")
    print(f"  Max Drawdown    : {metrics['max_drawdown_pct']}%")
    print(f"  Win Rate        : {metrics['win_rate_pct']}%")
    print(f"  Profit Factor   : {metrics['profit_factor']}")
    print(f"  Total Return    : {metrics['total_return_pct']}%")
    print(f"  # Trades        : {metrics['n_trades']}")

    gate_sharpe = metrics["sharpe"] >= 0.3      # realistic for rule-based trend-follow
    gate_pf = metrics["profit_factor"] >= 1.0   # must have positive expectancy
    gate_dd = metrics["max_drawdown_pct"] <= 20.0
    print(f"\n  Gate: Sharpe ≥ 0.3       → {'PASS ✓' if gate_sharpe else 'FAIL ✗'}  (realistic for rule-based trend-following)")
    print(f"  Gate: Profit Factor ≥ 1.0 → {'PASS ✓' if gate_pf else 'FAIL ✗'}")
    print(f"  Gate: MaxDD ≤ 20%         → {'PASS ✓' if gate_dd else 'FAIL ✗'}")
    print(f"  Overall: {'PASS — ready for Phase 3' if gate_sharpe and gate_pf and gate_dd else 'FAIL — tune strategy before proceeding'}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()

    since = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    until = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"

    print(f"Fetching {config['exchange']['symbol']} {config['exchange']['timeframe']} from {since} to {until}…")
    df_raw = fetch_historical_rest(
        symbol=config["exchange"]["symbol"],
        timeframe=config["exchange"]["timeframe"],
        since_str=since,
        until_str=until,
        sandbox=False,  # historical REST works on live endpoint
    )
    print(f"Downloaded {len(df_raw)} candles.")

    df_feat = compute_features(df_raw, config["strategy"])
    print(f"Features computed, {len(df_feat)} usable bars after warmup.")

    metrics = run_backtest(df_feat, config["strategy"], config["risk"])
    print_metrics(metrics)
