"""
Risk manager. Approves/rejects signals, sizes positions, enforces daily drawdown gate.
Layer: risk. No DB/order imports — pure calculation.

Hard rules (never bypass):
  - Risk per trade: 1% of balance
  - Leverage hard cap: 5×
  - Daily drawdown gate: halt if balance < day_start × (1 - max_daily_drawdown)
  - Never return a position without SL and TP
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Below this USD notional we cannot place a position (exchange minimum)
# Bitget BTCUSDT perp minimum: 0.001 BTC. Set conservatively; fetched from exchange at startup.
_DEFAULT_MIN_NOTIONAL_USD = 5.0


@dataclass
class TradeSpec:
    """Fully-specified trade approved by risk manager."""
    side: str           # 'buy' | 'sell'
    entry: float        # expected entry price (current market price)
    sl: float           # stop-loss price
    tp: float           # take-profit price
    qty: float          # contract quantity in base currency (BTC)
    notional: float     # face value USD = qty × entry
    risk_usd: float     # dollars at risk = qty × |entry - sl|
    risk_pct: float     # risk_usd / balance


class RiskManager:
    """
    Stateless calculator — all state (balance, daily start) passed in.
    Call approve_signal() once per candle signal.
    """

    def approve_signal(
        self,
        signal: str,
        price: float,
        atr: float,
        balance: float,
        day_start_balance: float,
        config: dict,
        min_qty: float = 0.001,
        min_notional: float = _DEFAULT_MIN_NOTIONAL_USD,
    ) -> Optional[TradeSpec]:
        """
        Returns TradeSpec if trade approved, None if rejected.

        Args:
            signal:           'long' | 'short'
            price:            current mid price (USD)
            atr:              ATR value from feature df
            balance:          current account balance (USD)
            day_start_balance: balance at UTC 00:00 today
            config:           full config dict (risk + exchange sections)
            min_qty:          minimum contract quantity from exchange
            min_notional:     minimum USD notional from exchange
        """
        risk_cfg = config["risk"]
        max_dd: float = risk_cfg["max_daily_drawdown"]
        risk_pct: float = risk_cfg["risk_per_trade"]
        leverage: int = int(risk_cfg["leverage"])
        sl_mult: float = risk_cfg["atr_sl_multiplier"]
        rr: float = risk_cfg["reward_risk_ratio"]

        # Hard cap leverage at 5× regardless of config
        leverage = min(leverage, 5)

        # ── Drawdown gate ──────────────────────────────────────────────────
        if day_start_balance > 0:
            daily_drawdown = (day_start_balance - balance) / day_start_balance
            if daily_drawdown >= max_dd:
                logger.warning(
                    "DRAWDOWN GATE: trading halted for today",
                    extra={
                        "daily_drawdown_pct": round(daily_drawdown * 100, 2),
                        "threshold_pct": round(max_dd * 100, 2),
                        "balance": balance,
                        "day_start": day_start_balance,
                    },
                )
                return None

        # ── Price / ATR sanity ─────────────────────────────────────────────
        if price <= 0 or atr <= 0:
            logger.error("invalid price or ATR", extra={"price": price, "atr": atr})
            return None

        # ── SL / TP ────────────────────────────────────────────────────────
        sl_distance = atr * sl_mult
        tp_distance = sl_distance * rr

        if signal == "long":
            side = "buy"
            sl = price - sl_distance
            tp = price + tp_distance
        elif signal == "short":
            side = "sell"
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            logger.error("unknown signal", extra={"signal": signal})
            return None

        # SL/TP must be on correct side of entry
        if side == "buy" and sl >= price:
            logger.error("SL above entry for long", extra={"sl": sl, "price": price})
            return None
        if side == "sell" and sl <= price:
            logger.error("SL below entry for short", extra={"sl": sl, "price": price})
            return None

        # ── Position sizing ────────────────────────────────────────────────
        risk_usd = balance * risk_pct                # dollars to risk
        qty = risk_usd / sl_distance                 # BTC to buy/sell
        qty = max(round(qty, 6), 0.0)

        # Enforce minimum quantity
        if qty < min_qty:
            logger.warning(
                "position too small — qty below minimum",
                extra={
                    "calculated_qty": qty,
                    "min_qty": min_qty,
                    "risk_usd": risk_usd,
                    "balance": balance,
                    "hint": "Add more capital or reduce min_qty threshold",
                },
            )
            return None

        notional = qty * price

        # Enforce minimum notional
        if notional < min_notional:
            logger.warning(
                "position too small — notional below minimum",
                extra={"notional": notional, "min_notional": min_notional},
            )
            return None

        # Margin check: notional / leverage must not exceed available balance
        required_margin = notional / leverage
        if required_margin > balance * 0.95:  # leave 5% buffer for fees
            logger.warning(
                "insufficient margin",
                extra={
                    "required_margin": required_margin,
                    "balance": balance,
                    "notional": notional,
                    "leverage": leverage,
                },
            )
            return None

        spec = TradeSpec(
            side=side,
            entry=price,
            sl=round(sl, 2),
            tp=round(tp, 2),
            qty=qty,
            notional=round(notional, 2),
            risk_usd=round(risk_usd, 4),
            risk_pct=risk_pct,
        )

        logger.info(
            "trade approved",
            extra={
                "signal": signal,
                "side": side,
                "qty": qty,
                "entry": price,
                "sl": spec.sl,
                "tp": spec.tp,
                "risk_usd": spec.risk_usd,
                "notional": spec.notional,
            },
        )
        return spec
