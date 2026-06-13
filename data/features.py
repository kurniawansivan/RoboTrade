"""
TA indicator pipeline.
Layer: data. Input: raw OHLCV DataFrame. Output: feature DataFrame.
No DB/Redis/strategy imports.

Multi-timeframe: 5m OHLCV resampled to 1h internally.
1h EMA trend direction merged back onto 5m bars as htf_trend (+1 bull / -1 bear).
"""

import logging

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

def _resample_and_trend(
    df: pd.DataFrame,
    resample_rule: str,
    ema_fast: int,
    ema_slow: int,
    col_name: str,
) -> pd.Series:
    """
    Resample 5m OHLCV to `resample_rule`, compute EMA trend direction,
    merge back onto 5m index with no lookahead.
    Returns Series: +1 (bull), -1 (bear), 0 (undecided/NaN).
    """
    if "open_time" not in df.columns:
        return pd.Series(0, index=df.index)

    df_ts = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].copy()
    df_ts.index = pd.DatetimeIndex(df_ts.index)

    htf = df_ts.resample(resample_rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    htf["ema_f"] = pd.to_numeric(ta.ema(htf["close"], length=ema_fast), errors="coerce")
    htf["ema_s"] = pd.to_numeric(ta.ema(htf["close"], length=ema_slow), errors="coerce")
    htf[col_name] = 0
    valid = htf["ema_f"].notna() & htf["ema_s"].notna()
    htf.loc[valid & (htf["ema_f"] > htf["ema_s"]), col_name] = 1
    htf.loc[valid & (htf["ema_f"] < htf["ema_s"]), col_name] = -1

    trend_s = htf[[col_name]].reset_index()
    trend_s.columns = ["open_time", col_name]

    df_idx = df[["open_time"]].copy()
    df_idx["open_time"] = pd.to_datetime(df_idx["open_time"], utc=True)
    trend_s["open_time"] = pd.to_datetime(trend_s["open_time"], utc=True)

    merged = pd.merge_asof(
        df_idx.sort_values("open_time"),
        trend_s.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(df_idx.index)
    return merged[col_name].fillna(0).astype(int)


def _compute_htf_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute ADX on 1h bars, merge back onto 5m index (no lookahead).
    Returns ADX value per 5m bar.
    """
    if "open_time" not in df.columns:
        return pd.Series(30.0, index=df.index)

    df_ts = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].copy()
    df_ts.index = pd.DatetimeIndex(df_ts.index)

    htf = df_ts.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    adx_df = ta.adx(htf["high"], htf["low"], htf["close"], length=length)
    if adx_df is None or adx_df.empty:
        return pd.Series(30.0, index=df.index)

    htf["adx"] = adx_df.iloc[:, 0].fillna(0)

    adx_s = htf[["adx"]].reset_index()
    adx_s.columns = ["open_time", "adx"]

    df_idx = df[["open_time"]].copy()
    df_idx["open_time"] = pd.to_datetime(df_idx["open_time"], utc=True)
    adx_s["open_time"] = pd.to_datetime(adx_s["open_time"], utc=True)

    merged = pd.merge_asof(
        df_idx.sort_values("open_time"),
        adx_s.sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    merged = merged.set_index(df_idx.index)
    return merged["adx"].fillna(0.0)


def _compute_htf_trend(df: pd.DataFrame) -> pd.Series:
    """
    1h trend filter: EMA20 vs EMA50 on 1h bars.
    htf_trend = +1 (1h bullish), -1 (1h bearish), 0 (undecided/NaN).
    No lookahead — each 5m bar sees the last CLOSED 1h bar.
    """
    return _resample_and_trend(df, "1h", ema_fast=20, ema_slow=50, col_name="htf_trend")


def compute_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute all TA features from OHLCV.

    Args:
        df: DataFrame with columns open/high/low/close/volume (+ optional open_time), sorted ascending.
        config: strategy section from config.yaml.

    Returns:
        DataFrame with original columns + indicator columns.
        Rows with NaN indicators (warmup) are dropped.
    """
    if not _REQUIRED_COLS.issubset(df.columns):
        missing = _REQUIRED_COLS - set(df.columns)
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = df.copy()

    ema_fast: int = config["ema_fast"]
    ema_slow: int = config["ema_slow"]
    rsi_period: int = config["rsi_period"]
    atr_period: int = config["atr_period"]
    vol_ma_period: int = config["volume_ma_period"]

    # 5m EMAs
    out[f"ema_{ema_fast}"] = ta.ema(out["close"], length=ema_fast)
    out[f"ema_{ema_slow}"] = ta.ema(out["close"], length=ema_slow)

    # RSI
    out["rsi"] = ta.rsi(out["close"], length=rsi_period)

    # ATR
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=atr_period)

    # MACD
    macd_df = ta.macd(out["close"])
    if macd_df is not None:
        out["macd"] = macd_df.iloc[:, 0]
        out["macd_signal"] = macd_df.iloc[:, 1]
        out["macd_hist"] = macd_df.iloc[:, 2]

    # Volume Z-score
    vol_mean = out["volume"].rolling(vol_ma_period).mean()
    vol_std = out["volume"].rolling(vol_ma_period).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)

    # ADX on 1h bars — trend strength on macro level (not noisy 5m ADX)
    out["adx"] = _compute_htf_adx(out, atr_period)

    # Multi-timeframe trend (1h) — no lookahead, forward-fills
    out["htf_trend"] = _compute_htf_trend(out)

    # Convenience aliases
    out["ema_fast"] = out[f"ema_{ema_fast}"]
    out["ema_slow"] = out[f"ema_{ema_slow}"]

    # Drop NaN warmup rows
    out = out.dropna().reset_index(drop=True)

    logger.debug("features computed", extra={"rows": len(out), "cols": list(out.columns)})
    return out


def label_for_ml(df: pd.DataFrame, lookahead: int = 4, threshold: float = 0.005) -> pd.DataFrame:
    """
    Add binary label for ML training.
    label = 1  if close rises >threshold in next `lookahead` bars
    label = -1 if close falls >threshold
    label = 0  otherwise
    Drops last `lookahead` rows (no future data).
    """
    future_ret = df["close"].shift(-lookahead) / df["close"] - 1
    df = df.copy()
    df["label"] = 0
    df.loc[future_ret > threshold, "label"] = 1
    df.loc[future_ret < -threshold, "label"] = -1
    df = df.iloc[:-lookahead].copy()
    return df
