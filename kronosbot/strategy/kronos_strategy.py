"""Kronos Strategy subclass for Backtesting.py."""
from typing import ClassVar, Optional

import pandas as pd
from backtesting import Strategy

from kronosbot.features.signals import SignalEngine


class KronosStrategy(Strategy):
    """Backtesting.py strategy driven by Kronos SignalEngine signals and ATR exits."""

    signal_engine: ClassVar[Optional[SignalEngine]] = None
    df_signals: ClassVar[Optional[pd.DataFrame]] = None
    size: ClassVar[int] = 10_000

    @classmethod
    def with_signal_engine(
        cls,
        signal_engine: SignalEngine,
        df_signals: Optional[pd.DataFrame] = None,
    ):
        """Return a subclass bound to the given signal engine and optional signals."""
        class BoundStrategy(cls):
            pass

        BoundStrategy.signal_engine = signal_engine
        BoundStrategy.df_signals = df_signals
        return BoundStrategy

    def _to_internal_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name is None:
            df.index.name = "timestamp"
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
        df = df.reset_index()
        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "timestamp": "timestamp",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["timestamp", "open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df.set_index("timestamp")
        return df

    def init(self):
        if self.signal_engine is None:
            raise ValueError(
                "KronosStrategy requires a signal_engine. "
                "Use KronosStrategy.with_signal_engine(engine) or set a class attribute."
            )

        data = self._to_internal_data(self.data.df)
        if self.df_signals is not None:
            signals = self.df_signals
        else:
            signals = self.signal_engine.generate(data)

        self._entry_signal = self.I(
            lambda: signals["entry_signal"].fillna(0).values, name="entry_signal"
        )
        self._atr = self.I(lambda: signals["atr"].fillna(0).values, name="atr")

    def next(self):
        if self._entry_signal[-1] and not self.position:
            self.buy(size=self.size)
        elif self.position.is_long and not self._entry_signal[-1]:
            self.position.close()
        elif self.position.is_long:
            stop_price = self._entry_price() - 2 * self._atr[-1]
            if self.data.Low[-1] <= stop_price:
                self.position.close()

    def _entry_price(self) -> float:
        if self.trades:
            return self.trades[-1].entry_price
        return self.data.Close[-1]
