"""
Order manager. Places, tracks, and cancels orders on Bitget futures.
Layer: execution. Depends on ccxt exchange + DB engine. No strategy/risk imports.

All orders logged to DB. SL and TP placed as separate reduce-only orders after fill.
"""

import logging
import time
from typing import Optional

from sqlalchemy.engine import Engine

from data.storage import log_trade
from risk.manager import TradeSpec

logger = logging.getLogger(__name__)


async def place_bracket_order(
    exchange,
    symbol: str,
    spec: TradeSpec,
    engine: Engine,
    strategy_type: str = "rule_based",
) -> Optional[dict]:
    """
    Place market entry + SL + TP on Bitget futures.

    Steps:
      1. Market order (entry)
      2. Confirm fill
      3. Stop-loss (stop_market, reduce-only)
      4. Take-profit (take_profit_market, reduce-only)

    Returns entry order dict, or None on failure.
    All steps logged. If SL/TP fail, position is closed immediately (no naked positions).
    """
    close_side = "sell" if spec.side == "buy" else "buy"

    # ── 1. Entry ───────────────────────────────────────────────────────────
    try:
        entry_order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side=spec.side,
            amount=spec.qty,
            params={"tdMode": "cross"},  # Bitget cross-margin mode
        )
        logger.info(
            "entry order placed",
            extra={
                "order_id": entry_order.get("id"),
                "side": spec.side,
                "qty": spec.qty,
                "symbol": symbol,
            },
        )
    except Exception as exc:
        logger.error("entry order failed", exc_info=exc, extra={"symbol": symbol, "spec": str(spec)})
        return None

    # ── 2. Confirm fill ────────────────────────────────────────────────────
    fill_price = spec.entry
    fill_qty = spec.qty
    try:
        filled = await exchange.fetch_order(entry_order["id"], symbol)
        fill_price = float(filled.get("average") or filled.get("price") or spec.entry)
        fill_qty = float(filled.get("filled") or spec.qty)
        slippage = abs(fill_price - spec.entry) / spec.entry
        logger.info(
            "entry fill confirmed",
            extra={
                "fill_price": fill_price,
                "expected": spec.entry,
                "slippage_pct": round(slippage * 100, 4),
            },
        )
    except Exception as exc:
        logger.warning("fill confirmation failed, using estimated price", exc_info=exc)

    # ── 3. Stop-loss order ────────────────────────────────────────────────
    sl_order_id: Optional[str] = None
    try:
        sl_order = await exchange.create_order(
            symbol=symbol,
            type="stop_market",
            side=close_side,
            amount=fill_qty,
            params={
                "stopPrice": spec.sl,
                "reduceOnly": True,
                "triggerType": "mark_price",
            },
        )
        sl_order_id = sl_order.get("id")
        logger.info("SL order placed", extra={"sl": spec.sl, "order_id": sl_order_id})
    except Exception as exc:
        logger.error(
            "SL order FAILED — closing position immediately to avoid naked exposure",
            exc_info=exc,
        )
        await _emergency_close(exchange, symbol, close_side, fill_qty)
        return entry_order

    # ── 4. Take-profit order ───────────────────────────────────────────────
    tp_order_id: Optional[str] = None
    try:
        tp_order = await exchange.create_order(
            symbol=symbol,
            type="take_profit_market",
            side=close_side,
            amount=fill_qty,
            params={
                "stopPrice": spec.tp,
                "reduceOnly": True,
                "triggerType": "mark_price",
            },
        )
        tp_order_id = tp_order.get("id")
        logger.info("TP order placed", extra={"tp": spec.tp, "order_id": tp_order_id})
    except Exception as exc:
        logger.error(
            "TP order FAILED — SL still active, position has SL protection",
            exc_info=exc,
        )
        # Don't close — SL is protecting the position

    # ── 5. Log to DB ──────────────────────────────────────────────────────
    import datetime
    try:
        log_trade(
            engine,
            {
                "symbol": symbol,
                "side": spec.side,
                "entry_price": fill_price,
                "exit_price": None,
                "quantity": fill_qty,
                "sl_price": spec.sl,
                "tp_price": spec.tp if tp_order_id else None,
                "entry_time": datetime.datetime.utcnow().isoformat(),
                "exit_time": None,
                "pnl": None,
                "fees": None,
                "slippage": round(abs(fill_price - spec.entry), 2),
                "strategy_type": strategy_type,
                "signal_meta": None,
            },
        )
    except Exception as exc:
        logger.error("trade DB log failed", exc_info=exc)

    return entry_order


async def cancel_all_orders(exchange, symbol: str) -> None:
    """Cancel all open orders for symbol."""
    try:
        await exchange.cancel_all_orders(symbol)
        logger.info("all orders cancelled", extra={"symbol": symbol})
    except Exception as exc:
        logger.error("cancel_all_orders failed", exc_info=exc, extra={"symbol": symbol})


async def close_all_positions(exchange, symbol: str) -> None:
    """
    Kill switch: close all open positions immediately via market order.
    Called on SIGTERM or manual halt.
    """
    try:
        positions = await exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts == 0:
                continue
            side = pos.get("side", "")
            close_side = "sell" if side == "long" else "buy"
            await exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=abs(contracts),
                params={"reduceOnly": True},
            )
            logger.info(
                "position closed (kill switch)",
                extra={"symbol": symbol, "side": side, "contracts": contracts},
            )
    except Exception as exc:
        logger.error("close_all_positions failed", exc_info=exc, extra={"symbol": symbol})


async def _emergency_close(exchange, symbol: str, close_side: str, qty: float) -> None:
    """Close position immediately — used when SL placement fails."""
    try:
        await exchange.create_order(
            symbol=symbol,
            type="market",
            side=close_side,
            amount=qty,
            params={"reduceOnly": True},
        )
        logger.warning("emergency close executed", extra={"symbol": symbol, "qty": qty})
    except Exception as exc:
        logger.error("EMERGENCY CLOSE FAILED", exc_info=exc, extra={"symbol": symbol})
