"""
Train XGBoost direction classifier with walk-forward validation.

Run:
    python training/train_ml.py                # all config symbols, 2023→2025
    python training/train_ml.py 2022-01-01 2025-06-01

Pipeline:
  fetch OHLCV → compute_features → build_ml_features → label (next-move direction)
  → walk-forward CV (train 12m / test 3m) → report OOS accuracy + trade proxy
  → train final model on full set → save to config ml.model_path

Features are scale-free (ratios, returns, RSI/ADX) so one model generalises across symbols.
Labels: -1/0/1 mapped to 0/1/2 for XGBoost (down/flat/up).
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from backtest.run_backtest import fetch_historical_rest
from data.features import ML_FEATURE_COLS, build_ml_features, compute_features, label_for_ml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_dataset(df_raw: pd.DataFrame, strat_cfg: dict, lookahead: int, threshold: float) -> pd.DataFrame:
    """Raw OHLCV → labelled ML feature rows."""
    feat = compute_features(df_raw.reset_index(drop=True), strat_cfg)
    feat = build_ml_features(feat)
    feat = label_for_ml(feat, lookahead=lookahead, threshold=threshold)
    return feat


def map_labels(y: pd.Series) -> np.ndarray:
    """-1/0/1 → 0/1/2 for XGBoost."""
    return (y + 1).astype(int).to_numpy()


def walk_forward_eval(data: pd.DataFrame, params: dict, train_months=12, test_months=3, step_months=3) -> dict:
    """Walk-forward CV. Returns aggregate OOS metrics."""
    import xgboost as xgb

    data = data.sort_values("open_time").reset_index(drop=True)
    data["open_time"] = pd.to_datetime(data["open_time"], utc=True)
    start, end = data["open_time"].min(), data["open_time"].max()

    accs, dir_accs, trade_ns = [], [], []
    cursor = start
    fold = 0
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break

        tr = data[(data["open_time"] >= tr_start) & (data["open_time"] < tr_end)]
        te = data[(data["open_time"] >= tr_end) & (data["open_time"] < te_end)]
        if len(tr) < 500 or len(te) < 100:
            cursor += pd.DateOffset(months=step_months)
            continue

        Xtr, ytr = tr[ML_FEATURE_COLS], map_labels(tr["label"])
        Xte, yte = te[ML_FEATURE_COLS], map_labels(te["label"])

        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        acc = float((pred == yte).mean())
        # Directional accuracy: of bars where model said up/down (not flat), how often right
        nonflat = pred != 1
        dir_acc = float((pred[nonflat] == yte[nonflat]).mean()) if nonflat.sum() > 0 else 0.0
        accs.append(acc)
        dir_accs.append(dir_acc)
        trade_ns.append(int(nonflat.sum()))
        logger.info("fold %d  acc=%.3f  dir_acc=%.3f  trades=%d", fold, acc, dir_acc, int(nonflat.sum()))
        fold += 1
        cursor += pd.DateOffset(months=step_months)

    return {
        "folds": fold,
        "mean_acc": float(np.mean(accs)) if accs else 0.0,
        "mean_dir_acc": float(np.mean(dir_accs)) if dir_accs else 0.0,
        "mean_trades_per_fold": float(np.mean(trade_ns)) if trade_ns else 0.0,
    }


def main() -> None:
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    strat_cfg = config["strategy"]
    ml_cfg = strat_cfg.get("ml", {})
    lookahead = int(ml_cfg.get("label_lookahead", 6))
    threshold = float(ml_cfg.get("label_threshold", 0.004))
    model_path = ml_cfg.get("model_path", "training/models/xgb_signal.json")

    since = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
    until = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"

    symbols = config["exchange"].get("symbols", [config["exchange"]["symbol"]])
    timeframe = config["exchange"]["timeframe"]

    # ── Build combined dataset across all symbols ─────────────────────────
    frames = []
    for sym in symbols:
        logger.info("fetching %s %s→%s", sym, since, until)
        try:
            raw = fetch_historical_rest(sym, timeframe, since, until, sandbox=False)
            ds = build_dataset(raw, strat_cfg, lookahead, threshold)
            ds["symbol"] = sym
            frames.append(ds)
            logger.info("  %s: %d labelled rows", sym, len(ds))
        except Exception as exc:
            logger.error("  %s failed: %s", sym, exc)

    if not frames:
        logger.error("no data — aborting")
        sys.exit(1)

    data = pd.concat(frames, ignore_index=True)
    logger.info("total dataset: %d rows", len(data))

    # Class balance
    counts = data["label"].value_counts().to_dict()
    logger.info("label balance: down=%d flat=%d up=%d",
                counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0))

    # ── XGBoost params ─────────────────────────────────────────────────────
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    # ── Walk-forward validation ────────────────────────────────────────────
    logger.info("=== Walk-forward validation ===")
    wfo = walk_forward_eval(data, params)
    print("\n=== Walk-Forward OOS Results ===")
    print(f"  Folds:                 {wfo['folds']}")
    print(f"  Mean accuracy:         {wfo['mean_acc']:.3f}  (3-class baseline ≈ 0.33)")
    print(f"  Mean directional acc:  {wfo['mean_dir_acc']:.3f}  (>0.50 = edge on traded bars)")
    print(f"  Mean trades / fold:    {wfo['mean_trades_per_fold']:.0f}")

    edge = wfo["mean_dir_acc"] > 0.50
    print(f"\n  Gate: directional acc > 0.50 → {'PASS ✓ — model has edge' if edge else 'FAIL ✗ — no edge, do not deploy'}")

    # ── Train final model on full dataset ──────────────────────────────────
    import xgboost as xgb
    logger.info("training final model on full dataset…")
    final = xgb.XGBClassifier(**params)
    final.fit(data[ML_FEATURE_COLS], map_labels(data["label"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    final.save_model(model_path)
    logger.info("model saved → %s", model_path)

    # Feature importance
    importances = sorted(
        zip(ML_FEATURE_COLS, final.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    print("\n=== Feature Importance (top 10) ===")
    for name, imp in importances[:10]:
        print(f"  {name:<16} {imp:.4f}")

    print(f"\nModel saved to {model_path}")
    print("To use: set strategy.type: ml in config.yaml, then restart the bot.")


if __name__ == "__main__":
    main()
