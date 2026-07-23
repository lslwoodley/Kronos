"""Tests for broker layer: base, paper, and IBKR stub."""
from datetime import datetime

import pytest


class TestBrokerBase:
    def test_broker_base_has_required_interface(self):
        from kronosbot.broker.base import Broker

        required = {"market_order", "close_position", "position", "equity", "reset"}
        assert required.issubset(set(dir(Broker)))

    def test_broker_base_cannot_be_instantiated(self):
        from kronosbot.broker.base import Broker

        with pytest.raises(TypeError):
            Broker()


class TestPaperBroker:
    def _broker(self):
        from kronosbot.broker.paper import PaperBroker

        return PaperBroker(cash=10_000)

    def test_default_cash(self):
        broker = self._broker()
        assert broker.equity() == pytest.approx(10_000)
        assert broker.cash == pytest.approx(10_000)

    def test_default_trade_units(self):
        from kronosbot.broker.paper import PaperBroker

        broker = PaperBroker()
        assert broker.trade_units == 10_000

    def test_default_spread_for_eurusd(self):
        from kronosbot.broker.paper import PaperBroker

        broker = PaperBroker(symbol="EURUSD")
        # 0.6 pips in price terms: 0.6 * 0.0001 = 0.00006
        assert broker.spread == pytest.approx(0.00006)

    def test_default_spread_for_usdjpy(self):
        from kronosbot.broker.paper import PaperBroker

        broker = PaperBroker(symbol="USDJPY")
        # 0.6 pips in price terms: 0.6 * 0.01 = 0.006
        assert broker.spread == pytest.approx(0.006)

    def test_buy_market_order_opens_position(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        fill = broker.market_order("BUY", price=1.0850, timestamp=ts)

        assert fill["side"] == "BUY"
        assert fill["fill_price"] == pytest.approx(1.08503)  # price + half spread
        assert fill["commission"] == pytest.approx(5.0)
        assert broker.position()["side"] == "LONG"
        assert broker.position()["units"] == 10_000

    def test_sell_market_order_closes_position_with_profit(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        # Price rises before sell
        fill = broker.market_order("SELL", price=1.0900, timestamp=ts)

        assert fill["side"] == "SELL"
        assert fill["fill_price"] == pytest.approx(1.08997)  # price - half spread
        # PnL in pips: (1.08997 - 1.08003) / 0.0001 = 99.4 pips
        # USD PnL = 99.4 pips * 0.0001 * 10_000 = 99.4
        # Commission = 10 (buy + sell)
        assert broker.position()["side"] == "FLAT"
        assert broker.realized_pnl == pytest.approx(99.4 - 10.0, rel=1e-3)

    def test_sell_without_position_opens_short(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        fill = broker.market_order("SELL", price=1.0900, timestamp=ts)

        assert fill["side"] == "SELL"
        assert fill["fill_price"] == pytest.approx(1.08997)
        assert broker.position()["side"] == "SHORT"
        assert broker.position()["units"] == 10_000

    def test_sell_when_short_raises(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("SELL", price=1.0900, timestamp=ts)
        with pytest.raises(ValueError, match="Already short"):
            broker.market_order("SELL", price=1.0900, timestamp=ts)

    def test_buy_when_long_raises(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        with pytest.raises(ValueError, match="Already long"):
            broker.market_order("BUY", price=1.0800, timestamp=ts)

    def test_buy_when_short_closes_short(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("SELL", price=1.0900, timestamp=ts)
        close_fill = broker.market_order("BUY", price=1.0800, timestamp=ts)

        assert close_fill["side"] == "BUY"
        assert broker.position()["side"] == "FLAT"
        assert broker.realized_pnl > 0  # sold high, bought low

    def test_unknown_side_raises(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="side must be BUY or SELL"):
            broker.market_order("HOLD", price=1.0900, timestamp=ts)

    def test_position_unrealized_pnl(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        # Mark-to-market at a higher price
        pnl = broker.unrealized_pnl(1.0900)
        # Mark at 1.0900, sell fill = 1.08997. PnL = (1.08997 - 1.08003) * 10_000 = 99.4
        assert pnl == pytest.approx(99.4, rel=1e-3)

    def test_short_unrealized_pnl(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("SELL", price=1.0900, timestamp=ts)
        # Price drops, short is profitable
        pnl = broker.unrealized_pnl(1.0800)
        assert pnl > 0

    def test_reset_clears_position_and_pnl(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        broker.market_order("SELL", price=1.0900, timestamp=ts)
        assert broker.realized_pnl != 0

        broker.reset()
        assert broker.position()["side"] == "FLAT"
        assert broker.realized_pnl == 0
        assert broker.equity() == 10_000

    def test_commission_per_round_trip(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        broker.market_order("SELL", price=1.0800, timestamp=ts)
        # Spread loss: ~0.6 pips = $0.6 + 2 * $5 commission = $10.6
        assert broker.realized_pnl == pytest.approx(-10.6, rel=1e-3)
        assert broker.total_commission == pytest.approx(10.0)

    def test_close_position_flat_at_market(self):
        broker = self._broker()
        ts = datetime(2024, 1, 1)
        broker.market_order("BUY", price=1.0800, timestamp=ts)
        close_fill = broker.close_position(1.0900, ts)

        assert close_fill["side"] == "SELL"
        assert broker.position()["side"] == "FLAT"


class TestIBKRBroker:
    def test_ibkr_broker_raises_not_implemented(self):
        from kronosbot.broker.ibkr import IBKRBroker

        broker = IBKRBroker()
        with pytest.raises(NotImplementedError, match="IBKR"):
            broker.market_order("BUY", price=1.0, timestamp=datetime.now())
        with pytest.raises(NotImplementedError, match="IBKR"):
            broker.close_position(1.0, datetime.now())
        with pytest.raises(NotImplementedError, match="IBKR"):
            broker.position()
        with pytest.raises(NotImplementedError, match="IBKR"):
            broker.equity()
        with pytest.raises(NotImplementedError, match="IBKR"):
            broker.reset()
