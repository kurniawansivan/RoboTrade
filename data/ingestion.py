"""
WebSocket + REST data fetcher via ccxt.pro.
Layer: data. Pushes candles to TimescaleDB + Redis cache.
"""

import asyncio
import json
import logging
import os
from typing import Optional

import ccxt.pro as ccxtpro
import pandas as pd
import redis.asyncio as aioredis
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

from data.features import compute_features
from data.storage import upsert_candles

load_dotenv(dotenv_path="config/.env")
logger = logging.getLogger(__name__)


def _build_exchange(config: dict, sandbox: bool) -> ccxtpro.Exchange:
    exchange_cls = getattr(ccxtpro, config["exchange"]["name"])
    exchange = exchange_cls(
        {
            "apiKey": os.environ["BITGET_API_KEY"],
            "secret": os.environ["BITGET_API_SECRET"],
            "password": os.environ["BITGET_API_PASSPHRASE"],
        }
    )
    exchange.set_sandbox_mode(sandbox)
    return exchange


async def fetch_historical(
    exchange: ccxtpro.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV via REST. Returns DataFrame sorted ascending.
    """
    logger.info("fetching historical OHLCV", extra={"symbol": symbol, "limit": limit})
    raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).sort_values("open_time").reset_index(drop=True)
    return df


async def _push_to_redis(
    redis_client: aioredis.Redis,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    cache_size: int,
) -> None:
    """Overwrite Redis cache with latest `cache_size` candles as JSON list."""
    key = f"candles:{symbol}:{timeframe}"
    rows = df.tail(cache_size).copy()
    rows["open_time"] = rows["open_time"].astype(str)
    payload = rows.to_dict(orient="records")
    await redis_client.set(key, json.dumps(payload))
    logger.debug("redis cache updated", extra={"key": key, "rows": len(payload)})


async def run_ws_loop(
    engine: Engine,
    redis_client: aioredis.Redis,
    config: dict,
    strategy_config: dict,
    on_candle: Optional[callable] = None,
    symbol_override: Optional[str] = None,
) -> None:
    """
    Main WebSocket loop. Streams closed candles, persists to DB + Redis,
    then calls on_candle(symbol, df_with_features) if provided.

    symbol_override: use this symbol instead of config["exchange"]["symbol"].
    Runs indefinitely. Caller should wrap in asyncio.run() + signal handling.
    """
    cfg = config["exchange"]
    data_cfg = config["data"]
    symbol: str = symbol_override or cfg["symbol"]
    timeframe: str = cfg["timeframe"]
    sandbox: bool = cfg.get("sandbox", True)
    cache_size: int = data_cfg["candle_cache_size"]

    exchange = _build_exchange(config, sandbox)
    logger.info(
        "WebSocket loop starting",
        extra={"symbol": symbol, "timeframe": timeframe, "sandbox": sandbox},
    )

    try:
        # Warm up: load last `cache_size` historical candles
        hist_df = await fetch_historical(exchange, symbol, timeframe, limit=cache_size)
        upsert_candles(engine, hist_df, symbol, timeframe)
        await _push_to_redis(redis_client, symbol, timeframe, hist_df, cache_size)
        logger.info("warmup complete", extra={"rows": len(hist_df)})

        while True:
            try:
                candles = await exchange.watch_ohlcv(symbol, timeframe)
                # ccxt returns list of [ts, o, h, l, c, v]; last bar is current (open)
                # We process only confirmed-closed bars (all but last)
                if len(candles) < 2:
                    continue

                # Write live price (current open bar) to Redis on every WS tick
                live_price = float(candles[-1][4])  # index 4 = close of current bar
                await redis_client.set(f"live_price:{symbol}", str(live_price))

                closed_candles = candles[:-1]
                df_new = pd.DataFrame(
                    closed_candles,
                    columns=["ts", "open", "high", "low", "close", "volume"],
                )
                df_new["open_time"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True)
                df_new = df_new.drop(columns=["ts"])

                upsert_candles(engine, df_new, symbol, timeframe)

                # Refresh full cache from DB latest rows
                # Use Redis cache + new rows for feature computation (avoid DB round-trip)
                redis_key = f"candles:{symbol}:{timeframe}"
                cached_raw = await redis_client.get(redis_key)
                if cached_raw:
                    cached_list = json.loads(cached_raw)
                    df_cache = pd.DataFrame(cached_list)
                    df_cache["open_time"] = pd.to_datetime(df_cache["open_time"], utc=True)
                    df_combined = (
                        pd.concat([df_cache, df_new])
                        .drop_duplicates(subset="open_time")
                        .sort_values("open_time")
                        .tail(cache_size)
                        .reset_index(drop=True)
                    )
                else:
                    df_combined = df_new

                await _push_to_redis(redis_client, symbol, timeframe, df_combined, cache_size)

                if on_candle is not None:
                    try:
                        features_df = compute_features(df_combined, strategy_config)
                        await on_candle(symbol, features_df)
                    except Exception as exc:
                        logger.error(
                            "on_candle callback error",
                            exc_info=exc,
                            extra={"symbol": symbol},
                        )

            except ccxtpro.NetworkError as exc:
                logger.error("network error, reconnecting in 5s", exc_info=exc)
                await asyncio.sleep(5)
            except ccxtpro.ExchangeError as exc:
                logger.error("exchange error", exc_info=exc)
                await asyncio.sleep(10)

    finally:
        await exchange.close()
        logger.info("exchange closed")


async def get_redis_candles(
    redis_client: aioredis.Redis,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Read candle cache from Redis. Returns empty DataFrame if not populated yet.
    """
    key = f"candles:{symbol}:{timeframe}"
    raw = await redis_client.get(key)
    if raw is None:
        return pd.DataFrame()
    records = json.loads(raw)
    df = pd.DataFrame(records)
    if not df.empty:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df
