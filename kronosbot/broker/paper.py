"""Paper broker implementation for forex market-order simulation (long/short)."""
from datetime import datetime
from typing import Any, Dict, Optional

from kronosbot.broker.base import Broker


class PaperBroker(Broker):
    """Simulate forex market orders with spread, commission, position tracking and PnL."""

    PIP_SIZE = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
    }

    def __init__(
        self,
        cash: float = 10_000.0,
        symbol: str = "EURUSD",
        spread_pips: float = 0.6,
        commission_per_lot: float = 5.0,
        trade_units: int = 10_000,
    ):
        self.symbol = symbol
        self.cash = float(cash)
        self.spread_pips = float(spread_pips)
        self.commission_per_lot = float(commission_per_lot)
        self.trade_units = int(trade_units)

        self.pip = self.PIP_SIZE.get(symbol, 0.0001)
        self.spread = self.spread_pips * self.pip
        self.half_spread = self.spread / 2.0

        self._position_side = "FLAT"
        self._units = 0
        self._entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_commission = 0.0

    def _adjust_price(self, price: float, side: str) -> float:
        if side == "BUY":
            return price + self.half_spread
        if side == "SELL":
            return price - self.half_spread
        raise ValueError("side must be BUY or SELL")

    def _commission_for_order(self, units: int) -> float:
        lots = units / 10_000.0
        return lots * self.commission_per_lot

    def market_order(self, side: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        fill_price = self._adjust_price(price, side)
        commission = self._commission_for_order(self.trade_units)

        if side == "BUY":
            if self._position_side == "LONG":
                raise ValueError("Already long; use close_position to exit")
            if self._position_side == "SHORT":
                # Closing a short
                return self.close_position(price, timestamp)
            # Open long
            self._position_side = "LONG"
            self._units = self.trade_units
            self._entry_price = fill_price
        else:  # SELL
            if self._position_side == "SHORT":
                raise ValueError("Already short; use close_position to exit")
            if self._position_side == "LONG":
                # Closing a long
                return self.close_position(price, timestamp)
            # Open short
            self._position_side = "SHORT"
            self._units = self.trade_units
            self._entry_price = fill_price

        self.total_commission += commission

        return {
            "symbol": self.symbol,
            "side": side,
            "units": self.trade_units,
            "price": price,
            "fill_price": fill_price,
            "commission": commission,
            "timestamp": timestamp,
        }

    def close_position(self, price: float, timestamp: datetime) -> Dict[str, Any]:
        if self._position_side == "FLAT":
            raise ValueError("No open position to close")
        close_side = "SELL" if self._position_side == "LONG" else "BUY"
        fill_price = self._adjust_price(price, close_side)
        commission = self._commission_for_order(self._units)

        pips = (self._entry_price - fill_price) / self.pip
        if self._position_side == "LONG":
            pips = (fill_price - self._entry_price) / self.pip
        pnl = pips * self.pip * self._units - (2 * commission)

        self.realized_pnl += pnl
        self.total_commission += commission

        result = {
            "symbol": self.symbol,
            "side": close_side,
            "units": self._units,
            "price": price,
            "fill_price": fill_price,
            "commission": commission,
            "pnl": pnl,
            "timestamp": timestamp,
        }

        self._position_side = "FLAT"
        self._units = 0
        self._entry_price = 0.0
        return result

    def position(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self._position_side,
            "units": self._units,
            "entry_price": self._entry_price,
        }

    def unrealized_pnl(self, price: float) -> float:
        if self._position_side == "FLAT":
            return 0.0
        if self._position_side == "LONG":
            mark = price - self.half_spread
            pips = (mark - self._entry_price) / self.pip
        else:
            mark = price + self.half_spread
            pips = (self._entry_price - mark) / self.pip
        return pips * self.pip * self._units

    def equity(self) -> float:
        return self.cash + self.realized_pnl

    def reset(self) -> None:
        self.cash = 10_000.0
        self._position_side = "FLAT"
        self._units = 0
        self._entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_commission = 0.0
