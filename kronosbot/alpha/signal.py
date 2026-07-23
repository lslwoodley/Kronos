"""Convert a Kronos point forecast into a trading signal and position size."""
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class AlphaSignal:
    symbol: str
    timestamp: pd.Timestamp
    direction: Literal[-1, 0, 1]
    expected_return: float
    current_price: float
    forecast_price: float
    stop_price: Optional[float]
    position_size: float
    atr_14: float
    rationale: str

    @classmethod
    def from_forecast(
        cls,
        symbol: str,
        historical_df: pd.DataFrame,
        forecast: pd.Series,
        account_equity: float = 10_000.0,
        risk_per_trade: float = 0.01,
        min_return_threshold: float = 0.001,
    ) -> "AlphaSignal":
        """Generate a signal from a Kronos forecast.

        Args:
            symbol: e.g., "EURUSD"
            historical_df: DataFrame with OHLCV bars. Must include at least 14 rows.
            forecast: pd.Series with one forecasted close price, indexed by target date.
            account_equity: current account equity for position sizing
            risk_per_trade: fraction of equity to risk per trade (default 1%)
            min_return_threshold: minimum absolute forecast return to generate a non-zero signal
        """
        close = historical_df["close"]
        atr = _atr_14(historical_df)
        current_price = float(close.iloc[-1])
        forecast_price = float(forecast.iloc[-1])
        expected_return = (forecast_price - current_price) / current_price
        timestamp = forecast.index[0]

        if abs(expected_return) < min_return_threshold:
            direction = 0
        else:
            direction = 1 if expected_return > 0 else -1

        # Risk-based position sizing: risk 1% of equity over 2*ATR stop distance.
        stop_distance = 2 * atr if not np.isnan(atr) else current_price * 0.005
        if direction == 0:
            stop_price = None
            position_size = 0.0
        else:
            stop_price = current_price - direction * stop_distance
            risk_amount = account_equity * risk_per_trade
            position_size = risk_amount / stop_distance  # units

        rationale = (
            f"Forecast {symbol} close {timestamp.date()} = {forecast_price:.5f} "
            f"({expected_return*100:+.3f}%). ATR(14)={atr:.5f}."
        )

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            expected_return=expected_return,
            current_price=current_price,
            forecast_price=forecast_price,
            stop_price=stop_price,
            position_size=position_size,
            atr_14=atr,
            rationale=rationale,
        )


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr_14(df: pd.DataFrame) -> float:
    return float(_true_range(df).rolling(window=14).mean().iloc[-1])
