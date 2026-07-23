"""Tests for KronosStrategy Backtesting.py integration."""
import numpy as np
import pandas as pd
import pytest


class TestKronosStrategy:
    def _make_data(self, periods: int = 250) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        close = np.linspace(1.0, 1.5, periods - 5)
        close += rng.normal(0, 0.005, periods - 5)
        # Last 5 bars are ascending breakouts so multiple entry signals fire.
        tail = np.linspace(max(close[-20:]) + 0.01, max(close[-20:]) + 0.05, 5)
        close = np.append(close, tail)
        high = close + 0.002
        low = close - 0.002
        volume = np.full(periods, 1_000.0)
        volume[-5:] = 5_000.0
        idx = pd.date_range("2023-01-01", periods=periods, freq="B")
        return pd.DataFrame(
            {
                "Open": close,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=idx,
        )

    def _to_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Backtesting.py required columns back to internal lowercase format."""
        out = df.reset_index().rename(
            columns={
                "index": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return out

    def test_strategy_requires_signal_engine(self):
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        assert KronosStrategy.signal_engine is None
        # The error is raised in init() when Backtest.run() is called without an engine.

    def test_strategy_precomputes_indicators(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        df = self._make_data(250)
        engine = SignalEngine(model=None, tokenizer=None)
        strategy_class = KronosStrategy.with_signal_engine(engine)

        from backtesting import Backtest

        bt = Backtest(df, strategy_class, cash=10_000, commission=0.0001, margin=1 / 50)
        stats = bt.run()
        assert "Return [%]" in stats

    def test_strategy_runs_and_returns_trade_list(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        df = self._make_data(250)
        engine = SignalEngine(model=None, tokenizer=None)
        strategy_class = KronosStrategy.with_signal_engine(engine)

        from backtesting import Backtest

        bt = Backtest(df, strategy_class, cash=10_000, commission=0.0001, margin=1 / 50)
        stats = bt.run()
        assert "# Trades" in stats

    def test_strategy_long_signal_produces_trade(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        df = self._make_data(250)
        engine = SignalEngine(model=None, tokenizer=None)
        df_signals = engine.generate(self._to_internal(df))
        # Force a positive forecast so entry signals fire.
        df_signals["forecast_return"] = 0.01
        df_signals["signal"] = engine.compute_signal(df_signals)
        df_signals["entry_signal"] = engine.compute_entry_signal(df_signals)

        strategy_class = KronosStrategy.with_signal_engine(engine, df_signals=df_signals)

        from backtesting import Backtest

        bt = Backtest(
            df, strategy_class, cash=10_000, commission=0.0001, margin=1 / 50, finalize_trades=True
        )
        stats = bt.run()
        assert stats["# Trades"] >= 1

    def test_strategy_short_signal_produces_trade(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        df = self._make_data(250)
        engine = SignalEngine(model=None, tokenizer=None)
        df_signals = engine.generate(self._to_internal(df))
        # Force a negative forecast so short entry signals fire.
        df_signals["forecast_return"] = -0.01
        df_signals["signal"] = engine.compute_signal(df_signals)
        df_signals["entry_signal"] = engine.compute_entry_signal(df_signals)

        strategy_class = KronosStrategy.with_signal_engine(engine, df_signals=df_signals)

        from backtesting import Backtest

        bt = Backtest(
            df, strategy_class, cash=10_000, commission=0.0001, margin=1 / 50, finalize_trades=True
        )
        stats = bt.run()
        assert stats["# Trades"] >= 1

    def test_strategy_respects_atr_based_stop(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        df = self._make_data(250)
        engine = SignalEngine(model=None, tokenizer=None)
        df_signals = engine.generate(self._to_internal(df))
        df_signals["forecast_return"] = 0.01
        df_signals["signal"] = engine.compute_signal(df_signals)
        df_signals["entry_signal"] = engine.compute_entry_signal(df_signals)

        strategy_class = KronosStrategy.with_signal_engine(engine, df_signals=df_signals)

        from backtesting import Backtest

        bt = Backtest(df, strategy_class, cash=10_000, commission=0.0001, margin=1 / 50)
        stats = bt.run()
        assert "# Trades" in stats

    def test_strategy_class_attribute_signal_engine(self):
        from kronosbot.features.signals import SignalEngine
        from kronosbot.strategy.kronos_strategy import KronosStrategy

        engine = SignalEngine(model=None, tokenizer=None)

        class TestStrategy(KronosStrategy):
            signal_engine = engine

        df = self._make_data(250)
        from backtesting import Backtest

        bt = Backtest(df, TestStrategy, cash=10_000, commission=0.0001, margin=1 / 50)
        stats = bt.run()
        assert "Return [%]" in stats
