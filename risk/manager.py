"""
Risk manager. Approves/rejects signals, sizes positions, enforces daily drawdown gate.
Layer: risk. No DB/order imports — pure calculation.

Hard rules (never bypass):
  - Risk per trade: 1% of balance
  - Leverage: from config (env-injected via LEVERAGE). Soft guard: risk.max_leverage (0 = off)
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


def _round_sig(x: float, sig: int = 6) -> float:
    """Round to N significant figures — precision-safe across $0.5 alts and $60k BTC."""
    if x == 0:
        return 0.0
    import math
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


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

        # Soft sanity guard only (no hard 5× cap). max_leverage=0 disables guard.
        max_lev: int = int(risk_cfg.get("max_leverage", 0) or 0)
        if max_lev > 0 and leverage > max_lev:
            logger.warning(
                "leverage exceeds max_leverage guard — clamping",
                extra={"requested": leverage, "max_leverage": max_lev},
            )
            leverage = max_lev
        if leverage <= 0:
            leverage = 1

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
        # Stop = max(ATR-based, percentage floor). The floor guarantees the stop
        # is never tighter than market noise + round-trip fees+slippage, so the
        # take-profit is always a meaningful multiple of real trading cost.
        min_sl_pct: float = float(risk_cfg.get("min_sl_pct", 0.0) or 0.0)
        sl_distance = max(atr * sl_mult, price * min_sl_pct)
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

        # Keep full precision here — order_manager applies the exchange's price
        # precision per symbol. Rounding to 2 dp here would destroy SL/TP on
        # low-priced assets (e.g. XRP ~$1.14 → 1.13/1.15, breaking the R:R).
        spec = TradeSpec(
            side=side,
            entry=price,
            sl=_round_sig(sl),
            tp=_round_sig(tp),
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
