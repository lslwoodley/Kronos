"""Tests for BacktestRunner."""
import pandas as pd
import pytest

from kronosbot.data.feed import DataFeed
from kronosbot.features.signals import SignalEngine
from kronosbot.strategy.kronos_strategy import KronosStrategy
from kronosbot.strategy.runner import BacktestRunner


class _MockFeed:
    def __init__(self, df):
        self._df = df

    def load(self, symbol, start=None, end=None):
        return self._df.copy()

    def _cache_path(self, *args, **kwargs):
        return None


def _make_bars(periods: int = 250) -> pd.DataFrame:
    import numpy as np

    close = np.linspace(1.0, 1.5, periods)
    high = close + 0.002
    low = close - 0.002
    volume = np.full(periods, 1_000.0)
    volume[-1] = 5_000.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=periods, freq="B"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestBacktestRunner:
    def test_runner_returns_metrics_and_trades(self, tmp_path):
        df = _make_bars(250)
        feed = _MockFeed(df)
        engine = SignalEngine(model=None, tokenizer=None)
        runner = BacktestRunner(feed, engine, KronosStrategy)

        result = runner.run(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-12-31",
            cash=10_000,
            output_dir=tmp_path,
        )
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result
        assert "Return [%]" in result["metrics"]
        assert result["metrics"]["# Trades"] is not None

    def test_runner_logs_to_journal(self, tmp_path):
        from kronosbot.journal.store import Journal

        df = _make_bars(250)
        feed = _MockFeed(df)
        engine = SignalEngine(model=None, tokenizer=None)
        runner = BacktestRunner(feed, engine, KronosStrategy)
        journal = Journal(tmp_path / "test.db")

        result = runner.run(
            symbol="EURUSD",
            start="2023-01-01",
            end="2023-12-31",
            cash=10_000,
            journal=journal,
        )
        runs = journal.read_backtest_runs()
        assert runs
        run_id = runs[0]["id"]
        trades = journal.read_backtest_trades(run_id)
        assert len(trades) == len(result["trades"])
