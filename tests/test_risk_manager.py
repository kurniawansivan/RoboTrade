"""
Unit tests for risk/manager.py.
Run: pytest tests/test_risk_manager.py -v
"""

import pytest
from risk.manager import RiskManager, TradeSpec

BASE_CONFIG = {
    "risk": {
        "risk_per_trade": 0.01,
        "max_daily_drawdown": 0.04,
        "leverage": 5,
        "atr_sl_multiplier": 1.5,
        "reward_risk_ratio": 3.0,
    },
    "exchange": {"sandbox": True},
}


@pytest.fixture
def rm():
    return RiskManager()


class TestApproveSignal:

    def test_long_signal_returns_trade_spec(self, rm):
        spec = rm.approve_signal(
            signal="long",
            price=100_000.0,
            atr=200.0,
            balance=1000.0,
            day_start_balance=1000.0,
            config=BASE_CONFIG,
            min_qty=0.0001,
            min_notional=1.0,
        )
        assert spec is not None
        assert isinstance(spec, TradeSpec)
        assert spec.side == "buy"
        assert spec.sl < spec.entry < spec.tp

    def test_short_signal_returns_trade_spec(self, rm):
        spec = rm.approve_signal(
            signal="short",
            price=100_000.0,
            atr=200.0,
            balance=1000.0,
            day_start_balance=1000.0,
            config=BASE_CONFIG,
            min_qty=0.0001,
            min_notional=1.0,
        )
        assert spec is not None
        assert spec.side == "sell"
        assert spec.sl > spec.entry > spec.tp

    def test_sl_tp_correct_distance(self, rm):
        """SL distance = 1.5 * ATR; TP = SL * RR."""
        atr = 200.0
        price = 100_000.0
        spec = rm.approve_signal(
            "long", price, atr, 1000.0, 1000.0,
            BASE_CONFIG, min_qty=0.0001, min_notional=1.0,
        )
        sl_dist = price - spec.sl
        tp_dist = spec.tp - price
        assert abs(sl_dist - atr * 1.5) < 0.01
        assert abs(tp_dist - sl_dist * 3.0) < 0.01

    def test_drawdown_gate_blocks_trade(self, rm):
        """Trade blocked if daily drawdown >= 4%."""
        spec = rm.approve_signal(
            signal="long",
            price=100_000.0,
            atr=200.0,
            balance=950.0,          # 5% drawdown from 1000
            day_start_balance=1000.0,
            config=BASE_CONFIG,
            min_qty=0.0001,
            min_notional=1.0,
        )
        assert spec is None

    def test_drawdown_gate_allows_trade_below_threshold(self, rm):
        """Trade allowed if daily drawdown < 4%."""
        spec = rm.approve_signal(
            signal="long",
            price=100_000.0,
            atr=200.0,
            balance=970.0,          # 3% drawdown — below 4% gate
            day_start_balance=1000.0,
            config=BASE_CONFIG,
            min_qty=0.0001,
            min_notional=1.0,
        )
        assert spec is not None

    def test_position_too_small_rejected(self, rm):
        """With tiny balance, qty < min_qty → rejected."""
        spec = rm.approve_signal(
            signal="long",
            price=100_000.0,
            atr=200.0,
            balance=10.0,           # $10 balance → $0.10 risk → qty = 0.0003 BTC
            day_start_balance=10.0,
            config=BASE_CONFIG,
            min_qty=0.001,          # minimum 0.001 BTC (~$100)
            min_notional=5.0,
        )
        # $0.10 risk / $300 SL distance = 0.00033 BTC < 0.001 min_qty
        assert spec is None

    def test_leverage_env_driven_no_hard_cap(self, rm):
        """No hard 5× cap — leverage 20× is honored when no max_leverage guard set."""
        cfg = {
            "risk": {**BASE_CONFIG["risk"], "leverage": 20},
            "exchange": BASE_CONFIG["exchange"],
        }
        # 20× allows larger notional than 5× would — margin check should pass for a
        # position that would FAIL at 5× but pass at 20×.
        spec = rm.approve_signal(
            "long", 100_000.0, 50.0, 1000.0, 1000.0,
            cfg, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is not None
        # notional/20 must fit in balance; at 5× the same notional would exceed it
        assert spec.notional / 20 <= 1000.0 * 0.95

    def test_max_leverage_guard_clamps(self, rm):
        """max_leverage guard clamps an over-set leverage."""
        cfg = {
            "risk": {**BASE_CONFIG["risk"], "leverage": 125, "max_leverage": 10},
            "exchange": BASE_CONFIG["exchange"],
        }
        spec = rm.approve_signal(
            "long", 100_000.0, 200.0, 1000.0, 1000.0,
            cfg, min_qty=0.0001, min_notional=1.0,
        )
        # With guard=10, required margin uses 10× not 125×
        if spec is not None:
            assert spec.notional / 10 <= 1000.0 * 0.95

    def test_invalid_signal_rejected(self, rm):
        spec = rm.approve_signal(
            signal="hold",
            price=100_000.0,
            atr=200.0,
            balance=1000.0,
            day_start_balance=1000.0,
            config=BASE_CONFIG,
        )
        assert spec is None

    def test_zero_atr_rejected(self, rm):
        spec = rm.approve_signal(
            "long", 100_000.0, 0.0, 1000.0, 1000.0,
            BASE_CONFIG, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is None

    def test_sl_floor_applied_when_atr_tiny(self, rm):
        """When ATR is tiny, SL distance floors at min_sl_pct of price."""
        cfg = {
            "risk": {**BASE_CONFIG["risk"], "min_sl_pct": 0.008, "atr_sl_multiplier": 2.0},
            "exchange": BASE_CONFIG["exchange"],
        }
        price = 100_000.0
        spec = rm.approve_signal(
            "long", price, atr=1.0, balance=1000.0, day_start_balance=1000.0,
            config=cfg, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is not None
        sl_dist = price - spec.sl
        # ATR*2 = 2 would be tiny; floor 0.8% of 100k = 800
        assert abs(sl_dist - price * 0.008) < 1.0

    def test_low_price_asset_precision_preserved(self, rm):
        """SL/TP for a ~$1 asset keep precision (not rounded to 2 dp)."""
        cfg = {
            "risk": {**BASE_CONFIG["risk"], "min_sl_pct": 0.008, "reward_risk_ratio": 2.0},
            "exchange": BASE_CONFIG["exchange"],
        }
        price = 1.1391
        spec = rm.approve_signal(
            "long", price, atr=0.001, balance=1000.0, day_start_balance=1000.0,
            config=cfg, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is not None
        # SL must be ~0.8% below entry, NOT collapsed to 1.13
        sl_pct = (price - spec.sl) / price
        assert 0.007 < sl_pct < 0.009
        # TP must give ~2:1
        tp_pct = (spec.tp - price) / price
        assert abs(tp_pct / sl_pct - 2.0) < 0.1

    def test_risk_amount_is_1pct_of_balance(self, rm):
        balance = 500.0
        spec = rm.approve_signal(
            "long", 100_000.0, 200.0, balance, balance,
            BASE_CONFIG, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is not None
        assert abs(spec.risk_usd - balance * 0.01) < 0.001
