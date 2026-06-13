"""
TimescaleDB helpers — OHLCV storage and retrieval.
Layer: data. No strategy/risk imports.
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def build_engine(
    host: str,
    port: int,
    db: str,
    user: str,
    password: str,
) -> Engine:
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def init_schema(engine: Engine) -> None:
    """Create hypertable + trades table if not present."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol      TEXT        NOT NULL,
                timeframe   TEXT        NOT NULL,
                open_time   TIMESTAMPTZ NOT NULL,
                open        DOUBLE PRECISION NOT NULL,
                high        DOUBLE PRECISION NOT NULL,
                low         DOUBLE PRECISION NOT NULL,
                close       DOUBLE PRECISION NOT NULL,
                volume      DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (symbol, timeframe, open_time)
            );
        """))
        # Promote to hypertable only if not already one
        conn.execute(text("""
            SELECT create_hypertable(
                'ohlcv', 'open_time',
                if_not_exists => TRUE,
                migrate_data  => TRUE
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trades (
                id              BIGSERIAL PRIMARY KEY,
                symbol          TEXT            NOT NULL,
                side            TEXT            NOT NULL,
                entry_price     DOUBLE PRECISION NOT NULL,
                exit_price      DOUBLE PRECISION,
                quantity        DOUBLE PRECISION NOT NULL,
                sl_price        DOUBLE PRECISION,
                tp_price        DOUBLE PRECISION,
                entry_time      TIMESTAMPTZ     NOT NULL,
                exit_time       TIMESTAMPTZ,
                pnl             DOUBLE PRECISION,
                fees            DOUBLE PRECISION,
                slippage        DOUBLE PRECISION,
                strategy_type   TEXT,
                signal_meta     JSONB
            );
        """))
    logger.info("DB schema initialized")


def upsert_candles(engine: Engine, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """
    Upsert OHLCV rows. df must have columns: open_time, open, high, low, close, volume.
    open_time must be timezone-aware UTC.
    Returns number of rows written.
    """
    if df.empty:
        return 0

    rows = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    rows["symbol"] = symbol
    rows["timeframe"] = timeframe

    with engine.begin() as conn:
        # Build temp table insert via pandas + ON CONFLICT upsert
        rows.to_sql("ohlcv_tmp", conn, if_exists="replace", index=False)
        result = conn.execute(text("""
            INSERT INTO ohlcv (symbol, timeframe, open_time, open, high, low, close, volume)
            SELECT symbol, timeframe, open_time, open, high, low, close, volume
            FROM ohlcv_tmp
            ON CONFLICT (symbol, timeframe, open_time) DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume;
        """))
        conn.execute(text("DROP TABLE IF EXISTS ohlcv_tmp;"))

    written = len(rows)
    logger.info("upserted candles", extra={"symbol": symbol, "timeframe": timeframe, "rows": written})
    return written


def fetch_candles(
    engine: Engine,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    since: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch most recent `limit` candles. Returns DataFrame sorted ascending by open_time.
    """
    query = """
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = :symbol AND timeframe = :timeframe
    """
    params: dict = {"symbol": symbol, "timeframe": timeframe}
    if since:
        query += " AND open_time >= :since"
        params["since"] = since
    query += " ORDER BY open_time DESC LIMIT :limit"
    params["limit"] = limit

    df = pd.read_sql(text(query), engine, params=params)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def log_trade(engine: Engine, trade: dict) -> None:
    """Insert a trade record. trade is a plain dict matching trades columns."""
    cols = ", ".join(trade.keys())
    placeholders = ", ".join(f":{k}" for k in trade.keys())
    sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
    with engine.begin() as conn:
        conn.execute(text(sql), trade)
    logger.info("trade logged", extra={"side": trade.get("side"), "symbol": trade.get("symbol")})
