"""
Rule-based strategy: MACD histogram crossover + HTF trend filter + RSI filter.
Layer: strategy. No DB/Redis/order imports.

Entry logic (higher frequency than EMA cross):
  Long:  MACD histogram crosses 0 upward (momentum shift up)
         AND 1h trend bullish (HTF EMA20 > EMA50)
         AND RSI not overbought
         AND volume above average

  Short: MACD histogram crosses 0 downward (momentum shift down)
         AND 1h trend bearish (HTF EMA20 < EMA50)
         AND RSI not oversold
         AND volume above average

EMA20/200 on 5m still computed as trend context (used by features.py).
"""

import logging

import pandas as pd

from strategy.base import Strategy

logger = logging.getLogger(__name__)


class RuleBasedStrategy(Strategy):

    def generate_signal(self, df: pd.DataFrame, config: dict) -> str | None:
        if len(df) < 2:
            return None

        rsi_ob: float = config["rsi_overbought"]
        rsi_os: float = config["rsi_oversold"]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # 1h trend filter
        htf_trend: int = int(last.get("htf_trend", 0))

        # Volume confirmation
        vol_mean = df["volume"].rolling(config["volume_ma_period"]).mean().iloc[-1]
        vol_ok = last["volume"] > vol_mean

        # EMA crossover on 5m (entry trigger)
        ema_cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        ema_cross_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

        # MACD histogram confirmation
        macd_hist = float(last.get("macd_hist", 0))

        long_ok = (
            ema_cross_up
            and macd_hist > 0
            and last["rsi"] < rsi_ob
            and vol_ok
            and htf_trend >= 0             # 1h not actively bearish
        )
        short_ok = (
            ema_cross_dn
            and macd_hist < 0
            and last["rsi"] > rsi_os
            and vol_ok
            and htf_trend <= 0             # 1h not actively bullish
        )

        if long_ok:
            signal = "long"
        elif short_ok:
            signal = "short"
        else:
            signal = None

        if signal:
            logger.info(
                "signal generated",
                extra={
                    "signal": signal,
                    "bar_ts": str(last.get("open_time", "")),
                    "rsi": round(float(last["rsi"]), 2),
                    "ema_fast": round(float(last["ema_fast"]), 2),
                    "ema_slow": round(float(last["ema_slow"]), 2),
                },
            )

        return signal
