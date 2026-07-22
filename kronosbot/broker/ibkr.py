"""IBKR broker stub implementing the Broker interface as a future integration point."""
from datetime import datetime
from typing import Any, Dict

from kronosbot.broker.base import Broker


class IBKRBroker(Broker):
    """Stub broker for Interactive Brokers integration.

    All methods raise NotImplementedError. For setup, see the IBKR Web API docs:
    https://interactivebrokers.github.io/tws-api/introduction.html
    """

    def market_order(self, side: str, price: float, timestamp: datetime) -> Dict[str, Any]:
        raise NotImplementedError(
            "IBKR integration is a future workstream. "
            "See https://interactivebrokers.github.io/tws-api/introduction.html"
        )

    def close_position(self, price: float, timestamp: datetime) -> Dict[str, Any]:
        raise NotImplementedError(
            "IBKR integration is a future workstream. "
            "See https://interactivebrokers.github.io/tws-api/introduction.html"
        )

    def position(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "IBKR integration is a future workstream. "
            "See https://interactivebrokers.github.io/tws-api/introduction.html"
        )

    def equity(self) -> float:
        raise NotImplementedError(
            "IBKR integration is a future workstream. "
            "See https://interactivebrokers.github.io/tws-api/introduction.html"
        )

    def reset(self) -> None:
        raise NotImplementedError(
            "IBKR integration is a future workstream. "
            "See https://interactivebrokers.github.io/tws-api/introduction.html"
        )
