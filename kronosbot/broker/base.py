"""Abstract broker interface for Kronos Bot."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict


class Broker(ABC):
    """Abstract broker interface for live and paper trading."""

    @abstractmethod
    def market_order(self, side: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        """Execute a market order and return a fill dict."""

    @abstractmethod
    def close_position(self, price: float, timestamp: datetime) -> Dict[str, Any]:
        """Close the current position at market price."""

    @abstractmethod
    def position(self) -> Dict[str, Any]:
        """Return current position summary."""

    @abstractmethod
    def equity(self) -> float:
        """Return total account equity (cash + PnL)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the broker to initial state."""
