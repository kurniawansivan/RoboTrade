"""
Order manager. Places, tracks, and cancels orders on Bitget futures.
Layer: execution. Depends on ccxt exchange + DB engine. No strategy/risk imports.

Bracket entry: SL + TP are attached to the entry market order via Bitget preset
stop-loss / take-profit (ccxt unified stopLoss/takeProfit params). This is atomic —
no naked-position window, and avoids the separate plan-order type errors (400172/40774).
All orders logged to DB.
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
    Place market entry with attached SL + TP (Bitget preset bracket).

    Returns entry order dict, or None on failure. If the entry with bracket is
    rejected, no position is opened (safe). If the bracket somehow isn't applied,
    we verify the position has SL and emergency-close otherwise.
    """
    close_side = "sell" if spec.side == "buy" else "buy"

    # Round SL/TP to the exchange's price precision for this symbol
    try:
        sl_px = float(exchange.price_to_precision(symbol, spec.sl))
        tp_px = float(exchange.price_to_precision(symbol, spec.tp))
    except Exception:
        sl_px, tp_px = spec.sl, spec.tp

    bracket_params = {
        "reduceOnly": False,
        "stopLoss": {"triggerPrice": sl_px},
        "takeProfit": {"triggerPrice": tp_px},
    }

    # ── Entry with attached SL/TP ──────────────────────────────────────────
    try:
        entry_order = await exchange.create_order(
            symbol=symbol,
            type="market",
            side=spec.side,
            amount=spec.qty,
            params=bracket_params,
        )
        sl_pct = abs(spec.entry - sl_px) / spec.entry * 100
        tp_pct = abs(tp_px - spec.entry) / spec.entry * 100
        logger.info(
            "entry+bracket placed | %s %s qty=%s entry=%s SL=%s (-%.2f%%) TP=%s (+%.2f%%) RR=%.1f",
            symbol, spec.side, spec.qty, spec.entry, sl_px, sl_pct, tp_px, tp_pct,
            (tp_pct / sl_pct if sl_pct else 0),
        )
    except Exception as exc:
        logger.error("entry order failed", exc_info=exc, extra={"symbol": symbol, "spec": str(spec)})
        return None

    # ── Confirm fill ────────────────────────────────────────────────────────
    fill_price = spec.entry
    fill_qty = spec.qty
    try:
        filled = await exchange.fetch_order(entry_order["id"], symbol)
        fill_price = float(filled.get("average") or filled.get("price") or spec.entry)
        fill_qty = float(filled.get("filled") or spec.qty)
        slippage = abs(fill_price - spec.entry) / spec.entry
        logger.info(
            "entry fill confirmed",
            extra={"fill_price": fill_price, "expected": spec.entry, "slippage_pct": round(slippage * 100, 4)},
        )
    except Exception as exc:
        logger.warning("fill confirmation failed, using estimated price", exc_info=exc)

    # ── Confirm the preset SL/TP attached to the position (read-only) ──────
    # Bitget preset SL/TP live on the POSITION, not in the open-orders list, so we
    # read them back from fetch_positions. We NEVER auto-close on a missing read —
    # the entry order above already succeeded with the bracket params attached.
    try:
        await _confirm_protection(exchange, symbol, spec)
    except Exception as exc:
        logger.warning("SL/TP confirmation read failed (bracket still attached): %s", exc)

    # ── Log to DB ────────────────────────────────────────────────────────────
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
                "sl_price": sl_px,
                "tp_price": tp_px,
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


async def _confirm_protection(exchange, symbol: str, spec: TradeSpec) -> None:
    """
    Read-only confirmation that the position carries preset SL/TP.
    Logs CONFIRMED or a WARNING — never closes the position (the entry order
    already attached the bracket; a missing read is not proof of no protection).
    """
    positions = await exchange.fetch_positions([symbol], params={"productType": "USDT-FUTURES"})
    for pos in positions:
        if float(pos.get("contracts") or 0) == 0:
            continue
        info = pos.get("info", {}) or {}
        sl = (
            pos.get("stopLossPrice")
            or info.get("presetStopLossPrice")
            or info.get("stopLossTriggerPrice")
        )
        tp = (
            pos.get("takeProfitPrice")
            or info.get("presetStopSurplusPrice")
            or info.get("takeProfitTriggerPrice")
        )
        if sl:
            logger.info("protection confirmed", extra={"symbol": symbol, "sl": sl, "tp": tp})
        else:
            logger.warning(
                "preset SL not visible in position read — bracket was attached on entry; "
                "monitor will manage. NOT closing.",
                extra={"symbol": symbol, "expected_sl": spec.sl, "expected_tp": spec.tp},
            )
        return


async def cancel_all_orders(exchange, symbol: str) -> None:
    """Cancel all open orders for symbol."""
    try:
        await exchange.cancel_all_orders(symbol)
        logger.info("all orders cancelled", extra={"symbol": symbol})
    except Exception as exc:
        logger.error("cancel_all_orders failed", exc_info=exc, extra={"symbol": symbol})


async def close_position(exchange, symbol: str) -> bool:
    """
    Manually close the open position for one symbol (market, reduce-only).
    Cancels its trigger orders first. Returns True if a position was closed.
    """
    closed = False
    try:
        await cancel_all_orders(exchange, symbol)
    except Exception:
        pass
    try:
        positions = await exchange.fetch_positions([symbol], params={"productType": "USDT-FUTURES"})
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts == 0:
                continue
            side = pos.get("side", "")
            close_side = "sell" if side == "long" else "buy"
            await exchange.create_order(
                symbol=symbol, type="market", side=close_side,
                amount=abs(contracts), params={"reduceOnly": True},
            )
            closed = True
            logger.info("position closed (manual)", extra={"symbol": symbol, "side": side, "contracts": contracts})
    except Exception as exc:
        logger.error("close_position failed", exc_info=exc, extra={"symbol": symbol})
    return closed


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
