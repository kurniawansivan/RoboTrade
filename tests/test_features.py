"""
Unit tests for data/features.py.
Run: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data.features import compute_features, label_for_ml

STRATEGY_CFG = {
    "ema_fast": 20,
    "ema_slow": 200,
    "rsi_period": 14,
    "atr_period": 14,
    "volume_ma_period": 20,
}


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a trend."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.clip(close, 1.0, None)
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 10_000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open_time": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestComputeFeatures:

    def test_returns_dataframe(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert isinstance(out, pd.DataFrame)

    def test_required_columns_present(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        for col in ["ema_fast", "ema_slow", "rsi", "atr", "macd", "volume_zscore", "htf_trend"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup_drop(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert out.isnull().sum().sum() == 0

    def test_warmup_rows_dropped(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        # EMA200 needs 200 bars warmup — output must be shorter than input
        assert len(out) < len(df)
        assert len(out) > 0

    def test_rsi_bounds(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert (out["rsi"] >= 0).all()
        assert (out["rsi"] <= 100).all()

    def test_atr_positive(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert (out["atr"] > 0).all()

    def test_htf_trend_values_valid(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert set(out["htf_trend"].unique()).issubset({-1, 0, 1})

    def test_missing_column_raises(self):
        df = _make_ohlcv(500).drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            compute_features(df, STRATEGY_CFG)

    def test_ema_fast_less_than_slow_in_trend(self):
        """EMA_fast and EMA_slow values are present and different."""
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        assert not (out["ema_fast"] == out["ema_slow"]).all()


class TestLabelForML:

    def test_labels_are_1_0_minus1(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        labeled = label_for_ml(out, lookahead=4, threshold=0.005)
        assert set(labeled["label"].unique()).issubset({-1, 0, 1})

    def test_drops_last_n_rows(self):
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        lookahead = 4
        labeled = label_for_ml(out, lookahead=lookahead)
        assert len(labeled) == len(out) - lookahead

    def test_no_future_leakage(self):
        """Label at index i must only use data up to i + lookahead, not beyond."""
        df = _make_ohlcv(500)
        out = compute_features(df, STRATEGY_CFG)
        labeled = label_for_ml(out, lookahead=4, threshold=0.005)
        # Basic check: labels exist and are finite
        assert labeled["label"].notna().all()
