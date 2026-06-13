"""
Telegram alert sender.
Layer: monitoring. No DB/strategy imports — receives plain dicts.
All sends are fire-and-forget (async). Never blocks trading loop.
"""

import logging
import os
from typing import Optional

from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)

_bot: Optional[Bot] = None


def get_bot() -> Optional[Bot]:
    global _bot
    if _bot is None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not token:
            logger.warning("TELEGRAM_BOT_TOKEN not set — alerts disabled")
            return None
        _bot = Bot(token=token)
    return _bot


def _chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")


async def _send(text: str) -> None:
    """Send message. Silently swallows errors — never crash trading loop."""
    bot = get_bot()
    chat = _chat_id()
    if not bot or not chat:
        return
    try:
        await bot.send_message(
            chat_id=chat,
            text=text,
            parse_mode="HTML",
        )
    except TelegramError as exc:
        logger.error("telegram send failed", exc_info=exc)
    except Exception as exc:
        logger.error("telegram unexpected error", exc_info=exc)


async def alert_bot_started(symbol: str, sandbox: bool) -> None:
    mode = "🟡 SANDBOX" if sandbox else "🔴 LIVE"
    await _send(
        f"🤖 <b>RoboTrade started</b>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Mode: {mode}"
    )


async def alert_bot_stopped(reason: str = "SIGTERM") -> None:
    await _send(f"🛑 <b>RoboTrade stopped</b>\nReason: {reason}")


async def alert_trade_opened(
    signal: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    qty: float,
    risk_usd: float,
    symbol: str,
) -> None:
    emoji = "🟢" if side == "buy" else "🔴"
    direction = "LONG" if side == "buy" else "SHORT"
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    await _send(
        f"{emoji} <b>Trade Opened — {direction}</b>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Entry:  <code>${entry:,.2f}</code>\n"
        f"SL:     <code>${sl:,.2f}</code>  ({sl_pct:.2f}%)\n"
        f"TP:     <code>${tp:,.2f}</code>  ({tp_pct:.2f}%)\n"
        f"Qty:    <code>{qty} BTC</code>\n"
        f"Risk:   <code>${risk_usd:.4f}</code>"
    )


async def alert_trade_closed(
    side: str,
    entry: float,
    exit_price: float,
    pnl: float,
    reason: str,  # 'SL' | 'TP' | 'manual'
    symbol: str,
) -> None:
    pnl_emoji = "✅" if pnl >= 0 else "❌"
    direction = "LONG" if side == "buy" else "SHORT"
    pnl_pct = (exit_price - entry) / entry * 100 * (1 if side == "buy" else -1)
    await _send(
        f"{pnl_emoji} <b>Trade Closed — {direction} [{reason}]</b>\n"
        f"Symbol: <code>{symbol}</code>\n"
        f"Entry:  <code>${entry:,.2f}</code>\n"
        f"Exit:   <code>${exit_price:,.2f}</code>\n"
        f"P&L:    <code>${pnl:+.4f}  ({pnl_pct:+.2f}%)</code>"
    )


async def alert_drawdown_gate(
    balance: float,
    day_start: float,
    drawdown_pct: float,
) -> None:
    await _send(
        f"🚨 <b>DRAWDOWN GATE TRIGGERED</b>\n"
        f"Trading halted for today.\n"
        f"Balance:    <code>${balance:.2f}</code>\n"
        f"Day start:  <code>${day_start:.2f}</code>\n"
        f"Drawdown:   <code>{drawdown_pct:.2f}%</code>  (gate: 4%)"
    )


async def alert_daily_summary(
    balance: float,
    day_start: float,
    n_trades: int,
    sandbox: bool,
) -> None:
    pnl = balance - day_start
    pnl_pct = pnl / day_start * 100 if day_start > 0 else 0.0
    emoji = "📈" if pnl >= 0 else "📉"
    mode = "SANDBOX" if sandbox else "LIVE"
    await _send(
        f"{emoji} <b>Daily Summary [{mode}]</b>\n"
        f"Balance:  <code>${balance:.2f}</code>\n"
        f"Day P&L:  <code>${pnl:+.2f}  ({pnl_pct:+.2f}%)</code>\n"
        f"Trades:   <code>{n_trades}</code>"
    )


async def alert_error(context: str, error: str) -> None:
    await _send(
        f"⚠️ <b>Bot Error</b>\n"
        f"Context: <code>{context}</code>\n"
        f"Error:   <code>{error[:300]}</code>"
    )
