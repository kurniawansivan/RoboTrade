"""
Rule-based strategy with two modes.
Layer: strategy. No DB/Redis/order imports.

mode: fresh_cross   — original: only on a fresh EMA fast/slow crossover (rare, ~1/month)
mode: trend_align   — trades while trend aligned: EMA aligned + MACD momentum + RSI band
                      + 1h trend + ADX, with a per-symbol cooldown to prevent spam.
                      Fires far more often (~1-3 signals/day/symbol).

Cooldown: stored per-symbol in-instance; min `signal_cooldown_bars` between signals.
"""

import logging

import pandas as pd

from strategy.base import Strategy

logger = logging.getLogger(__name__)


class RuleBasedStrategy(Strategy):

    def __init__(self) -> None:
        # per-symbol last signal bar timestamp (for cooldown)
        self._last_signal_ts: dict[str, pd.Timestamp] = {}

    def _cooldown_ok(self, symbol: str | None, last_ts, cooldown_bars: int, tf_minutes: int = 5) -> bool:
        """True if enough bars have elapsed since last signal for this symbol."""
        if symbol is None or cooldown_bars <= 0:
            return True
        prev = self._last_signal_ts.get(symbol)
        if prev is None:
            return True
        try:
            elapsed_min = (pd.Timestamp(last_ts) - pd.Timestamp(prev)).total_seconds() / 60.0
            return elapsed_min >= cooldown_bars * tf_minutes
        except Exception:
            return True

    def _record_signal(self, symbol: str | None, ts) -> None:
        if symbol is not None:
            self._last_signal_ts[symbol] = pd.Timestamp(ts)

    def generate_signal(self, df: pd.DataFrame, config: dict, symbol: str | None = None) -> str | None:
        if len(df) < 2:
            return None

        mode: str = config.get("mode", "trend_align")
        rsi_ob: float = config["rsi_overbought"]
        rsi_os: float = config["rsi_oversold"]

        last = df.iloc[-1]
        prev = df.iloc[-2]
        bar_ts = last.get("open_time", "")

        htf_trend: int = int(last.get("htf_trend", 0))
        macd_hist = float(last.get("macd_hist", 0))
        rsi = float(last["rsi"])
        adx = float(last.get("adx", 0))

        vol_mean = df["volume"].rolling(config["volume_ma_period"]).mean().iloc[-1]
        vol_ok = last["volume"] > vol_mean

        if mode == "fresh_cross":
            signal = self._fresh_cross(last, prev, macd_hist, rsi, rsi_ob, rsi_os, vol_ok, htf_trend)
        else:
            signal = self._trend_align(config, last, macd_hist, rsi, rsi_ob, rsi_os, vol_ok, htf_trend, adx)

        if signal is None:
            return None

        # Cooldown gate
        cooldown_bars = int(config.get("signal_cooldown_bars", 0))
        if not self._cooldown_ok(symbol, bar_ts, cooldown_bars):
            return None
        self._record_signal(symbol, bar_ts)

        logger.info(
            "signal generated",
            extra={
                "symbol": symbol,
                "signal": signal,
                "mode": mode,
                "bar_ts": str(bar_ts),
                "rsi": round(rsi, 2),
                "macd_hist": round(macd_hist, 4),
                "adx": round(adx, 1),
                "htf_trend": htf_trend,
            },
        )
        return signal

    # ── Mode: fresh crossover (original, rare) ─────────────────────────────
    @staticmethod
    def _fresh_cross(last, prev, macd_hist, rsi, rsi_ob, rsi_os, vol_ok, htf_trend) -> str | None:
        cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        cross_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
        if cross_up and macd_hist > 0 and rsi < rsi_ob and vol_ok and htf_trend >= 0:
            return "long"
        if cross_dn and macd_hist < 0 and rsi > rsi_os and vol_ok and htf_trend <= 0:
            return "short"
        return None

    # ── Mode: trend alignment (active) ─────────────────────────────────────
    @staticmethod
    def _trend_align(config, last, macd_hist, rsi, rsi_ob, rsi_os, vol_ok, htf_trend, adx) -> str | None:
        adx_min = float(config.get("adx_threshold", 0))
        require_vol = bool(config.get("require_volume", False))

        ema_bull = last["ema_fast"] > last["ema_slow"]
        ema_bear = last["ema_fast"] < last["ema_slow"]
        trending = adx >= adx_min
        vol_pass = vol_ok if require_vol else True

        # Long: 5m trend up + 1h not bearish + positive momentum + not overbought
        # (no lower RSI bound — a pullback to low RSI in an uptrend is a good entry)
        long_ok = (
            ema_bull
            and htf_trend >= 0
            and macd_hist > 0
            and rsi < rsi_ob
            and trending
            and vol_pass
        )
        # Short: 5m trend down + 1h not bullish + negative momentum + not oversold
        short_ok = (
            ema_bear
            and htf_trend <= 0
            and macd_hist < 0
            and rsi > rsi_os
            and trending
            and vol_pass
        )
        if long_ok:
            return "long"
        if short_ok:
            return "short"
        return None
