"""
Walk-forward validation.
Train window: 12 months. Test window: 3 months. Step: 3 months.
Reports out-of-sample metrics for each fold + aggregate.
Layer: backtest. No execution imports.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from backtest.run_backtest import fetch_historical_rest, run_backtest, print_metrics
from data.features import compute_features

logger = logging.getLogger(__name__)


@dataclass
class WFOConfig:
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    min_trades: int = 3   # skip fold if fewer trades; low-freq strategy OK with 3+


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: dict
    oos_metrics: dict


def run_walk_forward(
    df_raw: pd.DataFrame,
    strategy_config: dict,
    risk_config: dict,
    wfo: Optional[WFOConfig] = None,
    init_cash: float = 1000.0,
) -> list[FoldResult]:
    """
    Runs walk-forward validation on raw OHLCV DataFrame.
    Returns list of FoldResult objects.
    """
    if wfo is None:
        wfo = WFOConfig()

    results: list[FoldResult] = []
    df_raw = df_raw.copy()
    df_raw["open_time"] = pd.to_datetime(df_raw["open_time"], utc=True)
    start = df_raw["open_time"].min()
    end = df_raw["open_time"].max()

    fold = 0
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=wfo.train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=wfo.test_months)

        if test_end > end:
            break

        df_train = df_raw[(df_raw["open_time"] >= train_start) & (df_raw["open_time"] < train_end)].copy()
        df_test = df_raw[(df_raw["open_time"] >= test_start) & (df_raw["open_time"] < test_end)].copy()

        if len(df_train) < 200 or len(df_test) < 50:
            cursor += pd.DateOffset(months=wfo.step_months)
            continue

        try:
            feat_train = compute_features(df_train.reset_index(drop=True), strategy_config)
            feat_test = compute_features(df_test.reset_index(drop=True), strategy_config)

            m_train = run_backtest(feat_train, strategy_config, risk_config, init_cash)
            m_oos = run_backtest(feat_test, strategy_config, risk_config, init_cash)

            if m_train["n_trades"] < wfo.min_trades or m_oos["n_trades"] < wfo.min_trades:
                logger.warning(
                    "skipping fold — too few trades",
                    extra={"fold": fold, "train_trades": m_train["n_trades"], "oos_trades": m_oos["n_trades"]},
                )
                cursor += pd.DateOffset(months=wfo.step_months)
                continue

            result = FoldResult(
                fold=fold,
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_metrics={k: v for k, v in m_train.items() if k != "portfolio"},
                oos_metrics={k: v for k, v in m_oos.items() if k != "portfolio"},
            )
            results.append(result)
            fold += 1
            logger.info(
                "fold complete",
                extra={
                    "fold": fold,
                    "oos_sharpe": m_oos["sharpe"],
                    "oos_dd": m_oos["max_drawdown_pct"],
                },
            )
        except Exception as exc:
            logger.error("fold failed", exc_info=exc, extra={"fold": fold})

        cursor += pd.DateOffset(months=wfo.step_months)

    return results


def print_wfo_report(results: list[FoldResult]) -> None:
    if not results:
        print("No valid folds completed.")
        return

    print("\n=== Walk-Forward Validation Report ===")
    print(f"{'Fold':>4}  {'Train':>22}  {'Test':>22}  {'OOS Sharpe':>10}  {'OOS DD%':>8}  {'OOS Ret%':>8}  {'Trades':>7}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.fold:>4}  {r.train_start} → {r.train_end}  "
            f"{r.test_start} → {r.test_end}  "
            f"{r.oos_metrics['sharpe']:>10.2f}  "
            f"{r.oos_metrics['max_drawdown_pct']:>8.1f}  "
            f"{r.oos_metrics['total_return_pct']:>8.1f}  "
            f"{r.oos_metrics['n_trades']:>7}"
        )

    oos_sharpes = [r.oos_metrics["sharpe"] for r in results]
    oos_dds = [r.oos_metrics["max_drawdown_pct"] for r in results]
    oos_rets = [r.oos_metrics["total_return_pct"] for r in results]

    print("-" * 90)
    print(f"{'MEAN':>4}  {'':>47}  {np.mean(oos_sharpes):>10.2f}  {np.mean(oos_dds):>8.1f}  {np.mean(oos_rets):>8.1f}")
    print(f"{'WORST':>4}  {'':>47}  {np.min(oos_sharpes):>10.2f}  {np.max(oos_dds):>8.1f}  {np.min(oos_rets):>8.1f}")

    gate_sharpe = np.mean(oos_sharpes) >= 0.0    # mean OOS must not be significantly negative
    gate_dd = np.mean(oos_dds) <= 20.0           # hard DD limit
    gate_positive_folds = sum(1 for s in oos_sharpes if s >= 0) >= len(results) * 0.5  # ≥50% folds non-negative
    gate_pf_proxy = np.mean(oos_rets) >= -1.0    # mean OOS return not catastrophically negative

    print(f"\n  Gate: Mean OOS Sharpe ≥ 0.0          → {'PASS ✓' if gate_sharpe else 'FAIL ✗'}")
    print(f"  Gate: Mean OOS MaxDD  ≤ 20%           → {'PASS ✓' if gate_dd else 'FAIL ✗'}")
    print(f"  Gate: ≥50% folds non-negative Sharpe  → {'PASS ✓' if gate_positive_folds else 'FAIL ✗'}")
    overall = gate_sharpe and gate_dd and gate_positive_folds and gate_pf_proxy
    print(f"\n  Overall WFO: {'PASS — proceed to Phase 3 paper trading' if overall else 'FAIL — strategy needs tuning'}")
    print(f"  Note: Rule-based trend-following on 5m BTC; Phase 5 ML will improve regime detection")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    since = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01"
    until = sys.argv[2] if len(sys.argv) > 2 else "2025-06-01"

    print(f"Fetching {config['exchange']['symbol']} from {since} to {until}…")
    df_raw = fetch_historical_rest(
        symbol=config["exchange"]["symbol"],
        timeframe=config["exchange"]["timeframe"],
        since_str=since,
        until_str=until,
        sandbox=False,
    )
    print(f"Downloaded {len(df_raw)} candles.")

    results = run_walk_forward(df_raw, config["strategy"], config["risk"])
    print_wfo_report(results)
