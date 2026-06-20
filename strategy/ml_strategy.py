"""
ML strategy — XGBoost classifier predicting next-move direction.
Layer: strategy. No DB/Redis/order imports.

Model is trained offline by training/train_ml.py and saved to config ml.model_path.
Live: build ML features from the latest bars, predict class probabilities,
emit 'long'/'short' only when the winning class probability ≥ proba_threshold.

Classes: 0 = down (short), 1 = flat (no trade), 2 = up (long)
(XGBoost needs 0-indexed labels; train_ml maps -1/0/1 → 0/1/2.)
"""

import logging
import os

import pandas as pd

from data.features import ML_FEATURE_COLS, build_ml_features
from strategy.base import Strategy

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):

    def __init__(self) -> None:
        self._model = None
        self._model_path: str | None = None
        self._last_signal_ts: dict[str, pd.Timestamp] = {}

    def _load_model(self, model_path: str):
        """Lazy-load model once; reload if path changed (hot-swap retrained model)."""
        if self._model is not None and self._model_path == model_path:
            return self._model
        if not os.path.exists(model_path):
            logger.error("ML model not found — run training/train_ml.py first",
                         extra={"model_path": model_path})
            return None
        import xgboost as xgb
        booster = xgb.XGBClassifier()
        booster.load_model(model_path)
        self._model = booster
        self._model_path = model_path
        logger.info("ML model loaded", extra={"model_path": model_path})
        return booster

    def _cooldown_ok(self, symbol, last_ts, cooldown_bars, tf_minutes=5) -> bool:
        if symbol is None or cooldown_bars <= 0:
            return True
        prev = self._last_signal_ts.get(symbol)
        if prev is None:
            return True
        try:
            elapsed = (pd.Timestamp(last_ts) - pd.Timestamp(prev)).total_seconds() / 60.0
            return elapsed >= cooldown_bars * tf_minutes
        except Exception:
            return True

    def generate_signal(self, df: pd.DataFrame, config: dict, symbol: str | None = None) -> str | None:
        ml_cfg = config.get("ml", {})
        model_path = ml_cfg.get("model_path", "training/models/xgb_signal.json")
        threshold = float(ml_cfg.get("proba_threshold", 0.55))

        model = self._load_model(model_path)
        if model is None:
            return None

        feat = build_ml_features(df)
        if feat.empty:
            return None

        x = feat[ML_FEATURE_COLS].iloc[[-1]]
        bar_ts = feat.iloc[-1].get("open_time", "")

        try:
            proba = model.predict_proba(x)[0]  # [p_down, p_flat, p_up]
        except Exception as exc:
            logger.error("ML predict failed", exc_info=exc)
            return None

        cls = int(proba.argmax())
        conf = float(proba[cls])

        if conf < threshold or cls == 1:   # class 1 = flat → no trade
            return None

        signal = "long" if cls == 2 else "short"

        cooldown_bars = int(config.get("signal_cooldown_bars", 0))
        if not self._cooldown_ok(symbol, bar_ts, cooldown_bars):
            return None
        if symbol is not None:
            self._last_signal_ts[symbol] = pd.Timestamp(bar_ts)

        logger.info(
            "ML signal",
            extra={
                "symbol": symbol,
                "signal": signal,
                "confidence": round(conf, 3),
                "proba_down": round(float(proba[0]), 3),
                "proba_flat": round(float(proba[1]), 3),
                "proba_up": round(float(proba[2]), 3),
                "bar_ts": str(bar_ts),
            },
        )
        return signal
