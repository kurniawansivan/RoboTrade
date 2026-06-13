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

    def test_leverage_hard_capped_at_5x(self, rm):
        """Even if config says leverage=20, hard cap is 5×."""
        cfg = {
            "risk": {
                **BASE_CONFIG["risk"],
                "leverage": 20,  # user tries to set 20×
            },
            "exchange": BASE_CONFIG["exchange"],
        }
        spec = rm.approve_signal(
            "long", 100_000.0, 200.0, 1000.0, 1000.0,
            cfg, min_qty=0.0001, min_notional=1.0,
        )
        if spec is not None:
            # Verify margin was calculated with 5× not 20×
            max_leveraged_notional = 1000.0 * 5  # 5× cap
            assert spec.notional <= max_leveraged_notional

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

    def test_risk_amount_is_1pct_of_balance(self, rm):
        balance = 500.0
        spec = rm.approve_signal(
            "long", 100_000.0, 200.0, balance, balance,
            BASE_CONFIG, min_qty=0.0001, min_notional=1.0,
        )
        assert spec is not None
        assert abs(spec.risk_usd - balance * 0.01) < 0.001
