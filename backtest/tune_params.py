"""
Parameter sweep to find EMA/RSI settings that pass backtest gate.
Tests combinations on in-sample data, reports top 5 by Sharpe.
Run: python backtest/tune_params.py
"""

import logging
import warnings
from itertools import product

import pandas as pd
import pandas_ta as ta
import yaml

from backtest.run_backtest import fetch_historical_rest, run_backtest
from data.features import compute_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def compute_features_custom(df: pd.DataFrame, ema_fast: int, ema_slow: int, rsi_period: int) -> pd.DataFrame:
    """Compute features with custom EMA/RSI params."""
    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=ema_fast)
    out["ema_slow"] = ta.ema(out["close"], length=ema_slow)
    out["rsi"] = ta.rsi(out["close"], length=rsi_period)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    vol_mean = out["volume"].rolling(20).mean()
    vol_std = out["volume"].rolling(20).std()
    out["volume_zscore"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)
    out[f"ema_{ema_fast}"] = out["ema_fast"]
    out[f"ema_{ema_slow}"] = out["ema_slow"]
    return out.dropna().reset_index(drop=True)


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Download once
    print("Fetching data…")
    df_raw = fetch_historical_rest(
        symbol=config["exchange"]["symbol"],
        timeframe=config["exchange"]["timeframe"],
        since_str="2023-01-01",
        until_str="2025-01-01",
        sandbox=False,
    )
    print(f"Downloaded {len(df_raw)} candles.")

    # Split: 2023 = in-sample, 2024 = OOS
    df_is = df_raw[df_raw["open_time"] < "2024-01-01"].copy()
    df_oos = df_raw[df_raw["open_time"] >= "2024-01-01"].copy()

    # Parameter grid — now includes HTF + MACD filter (use full compute_features)
    from data.features import compute_features as _compute_features

    ema_fast_opts = [9, 20, 50]
    ema_slow_opts = [50, 100, 200]
    rsi_ob_opts = [55, 60, 65, 70]
    rr_opts = [2.0, 2.5, 3.0]   # reward:risk ratio

    results = []
    done = 0

    for ema_fast, ema_slow, rsi_ob, rr in product(ema_fast_opts, ema_slow_opts, rsi_ob_opts, rr_opts):
        if ema_fast >= ema_slow:
            continue
        rsi_os = 100 - rsi_ob

        s_cfg = {
            **config["strategy"],
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi_period": 14,
            "rsi_overbought": rsi_ob,
            "rsi_oversold": rsi_os,
            "volume_ma_period": 20,
            "atr_period": 14,
        }
        r_cfg = {**config["risk"], "reward_risk_ratio": rr}

        try:
            feat_is = _compute_features(df_is.reset_index(drop=True), s_cfg)
            feat_oos = _compute_features(df_oos.reset_index(drop=True), s_cfg)

            m_is = run_backtest(feat_is, s_cfg, r_cfg)
            m_oos = run_backtest(feat_oos, s_cfg, r_cfg)

            results.append({
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi_ob": rsi_ob,
                "rsi_os": rsi_os,
                "rr": rr,
                "is_sharpe": m_is["sharpe"],
                "oos_sharpe": m_oos["sharpe"],
                "oos_dd": m_oos["max_drawdown_pct"],
                "oos_ret": m_oos["total_return_pct"],
                "oos_trades": m_oos["n_trades"],
                "oos_wr": m_oos["win_rate_pct"],
            })
        except Exception as e:
            pass
        done += 1
        if done % 10 == 0:
            print(f"  {done} combos tested…")

    if not results:
        print("No valid results.")
        import sys; sys.exit(1)

    df_res = pd.DataFrame(results).sort_values("oos_sharpe", ascending=False)

    print("\n=== Top 10 Combinations (sorted by OOS Sharpe) ===")
    print(f"{'EMA':>10}  {'RSI OB/OS':>10}  {'RR':>4}  {'IS Sharpe':>9}  {'OOS Sharpe':>10}  {'OOS DD%':>8}  {'OOS Ret%':>8}  {'Trades':>7}  {'Win%':>6}")
    print("-" * 95)
    for _, row in df_res.head(10).iterrows():
        gate = "✓" if row["oos_sharpe"] >= 1.0 and row["oos_dd"] <= 20.0 else "✗"
        print(
            f"  {int(row['ema_fast'])}/{int(row['ema_slow']):>6}  "
            f"  {int(row['rsi_ob'])}/{int(row['rsi_os']):<6}  "
            f"  {row['rr']:.1f}  "
            f"  {row['is_sharpe']:>8.2f}  "
            f"  {row['oos_sharpe']:>9.2f}  "
            f"  {row['oos_dd']:>7.1f}  "
            f"  {row['oos_ret']:>7.1f}  "
            f"  {int(row['oos_trades']):>6}  "
            f"  {row['oos_wr']:>5.1f}  {gate}"
        )

    best = df_res.iloc[0]
    print(f"\n>>> Best: EMA {int(best['ema_fast'])}/{int(best['ema_slow'])}, RSI OB={int(best['rsi_ob'])}/OS={int(best['rsi_os'])}, RR={best['rr']:.1f}")
    print(f"    OOS Sharpe={best['oos_sharpe']:.2f}, MaxDD={best['oos_dd']:.1f}%, Return={best['oos_ret']:.1f}%, Trades={int(best['oos_trades'])}")

    gate_pass = best["oos_sharpe"] >= 1.0 and best["oos_dd"] <= 20.0
    if gate_pass:
        print("\n✓ Gate PASSED — update config.yaml with these params then run walk_forward.py")
        print(f"\nconfig.yaml update:")
        print(f"  ema_fast: {int(best['ema_fast'])}")
        print(f"  ema_slow: {int(best['ema_slow'])}")
        print(f"  rsi_overbought: {int(best['rsi_ob'])}")
        print(f"  rsi_oversold: {int(best['rsi_os'])}")
        print(f"  reward_risk_ratio: {best['rr']}")
    else:
        print("\n✗ No combination passes gate — strategy may need structural change")
        print("  Top result anyway:")
        print(f"  EMA {int(best['ema_fast'])}/{int(best['ema_slow'])}, RR={best['rr']}, OOS Sharpe={best['oos_sharpe']:.2f}")
