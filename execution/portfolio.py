"""
Portfolio tracker. Syncs open positions + PnL from exchange → Redis.
Layer: execution. Depends on ccxt exchange + Redis client.
No strategy/risk imports.
"""

import json
import logging
import time
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_POSITIONS_KEY = "positions:{symbol}"
_BALANCE_KEY = "balance"
_DAY_START_KEY = "balance:day_start"
_DAILY_PNL_KEY = "pnl:daily"


async def sync_balance(
    exchange,
    redis_client: aioredis.Redis,
    symbol: str,
) -> float:
    """
    Fetch USDT balance from exchange, store in Redis.
    Returns current balance in USD.
    """
    try:
        balance_data = await exchange.fetch_balance({"type": "swap", "productType": "USDT-FUTURES"})
        info_list = balance_data.get("info", [])
        if info_list and isinstance(info_list, list):
            usdt = float(info_list[0].get("available", 0.0))
        else:
            usdt = float(balance_data.get("USDT", {}).get("free", 0.0) or 0.0)
        await redis_client.set(_BALANCE_KEY, str(usdt))
        logger.debug("balance synced", extra={"usdt": usdt})
        return usdt
    except Exception as exc:
        logger.error("balance sync failed", exc_info=exc)
        return 0.0


async def get_balance(redis_client: aioredis.Redis) -> float:
    """Read balance from Redis cache."""
    raw = await redis_client.get(_BALANCE_KEY)
    return float(raw) if raw else 0.0


async def init_day_start_balance(redis_client: aioredis.Redis) -> None:
    """
    Set day-start balance at UTC midnight (or first run of the day).
    Only sets if not already set today.
    """
    existing = await redis_client.get(_DAY_START_KEY)
    if existing is None:
        balance = await get_balance(redis_client)
        if balance > 0:
            await redis_client.set(_DAY_START_KEY, str(balance))
            # Expire at next UTC midnight
            import datetime
            now = datetime.datetime.utcnow()
            midnight = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            ttl_secs = int((midnight - now).total_seconds())
            await redis_client.expire(_DAY_START_KEY, ttl_secs)
            logger.info("day-start balance set", extra={"balance": balance, "ttl_secs": ttl_secs})


async def get_day_start_balance(redis_client: aioredis.Redis) -> float:
    raw = await redis_client.get(_DAY_START_KEY)
    return float(raw) if raw else 0.0


async def sync_positions(
    exchange,
    redis_client: aioredis.Redis,
    symbol: str,
) -> list[dict]:
    """
    Fetch open positions for symbol from exchange, store in Redis.
    Returns list of position dicts.
    """
    try:
        positions = await exchange.fetch_positions([symbol], params={"productType": "USDT-FUTURES"})
        open_pos = [p for p in positions if float(p.get("contracts", 0) or 0) != 0]

        key = _POSITIONS_KEY.format(symbol=symbol.replace("/", "_"))
        await redis_client.set(key, json.dumps(open_pos, default=str))

        if open_pos:
            logger.info(
                "positions synced",
                extra={"symbol": symbol, "count": len(open_pos)},
            )
        return open_pos
    except Exception as exc:
        logger.error("position sync failed", exc_info=exc, extra={"symbol": symbol})
        return []


async def get_positions(redis_client: aioredis.Redis, symbol: str) -> list[dict]:
    key = _POSITIONS_KEY.format(symbol=symbol.replace("/", "_"))
    raw = await redis_client.get(key)
    return json.loads(raw) if raw else []


async def has_open_position(redis_client: aioredis.Redis, symbol: str) -> bool:
    positions = await get_positions(redis_client, symbol)
    return len(positions) > 0


async def get_all_positions(redis_client: aioredis.Redis, symbols: list[str]) -> list[dict]:
    """Return all open positions across multiple symbols."""
    all_pos = []
    for sym in symbols:
        all_pos.extend(await get_positions(redis_client, sym))
    return all_pos


async def run_position_monitor(
    exchange,
    redis_client: aioredis.Redis,
    symbol: str,
    poll_interval_secs: float = 5.0,
) -> None:
    """
    Background loop: sync balance + positions every `poll_interval_secs`.
    Runs indefinitely — caller should cancel task on shutdown.
    """
    logger.info("position monitor started", extra={"symbol": symbol, "interval": poll_interval_secs})
    while True:
        try:
            await sync_balance(exchange, redis_client, symbol)
            await init_day_start_balance(redis_client)
            await sync_positions(exchange, redis_client, symbol)
        except Exception as exc:
            logger.error("position monitor error", exc_info=exc)
        import asyncio
        await asyncio.sleep(poll_interval_secs)
