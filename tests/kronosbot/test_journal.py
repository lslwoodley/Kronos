"""Tests for SQLite journal store."""
from datetime import datetime

import pandas as pd
import pytest


class TestJournalStore:
    def _store(self, tmp_path):
        from kronosbot.journal.store import Journal

        return Journal(db_path=tmp_path / "test.db")

    def test_tables_created(self, tmp_path):
        store = self._store(tmp_path)
        tables = store.list_tables()
        for t in ["bars", "signals", "backtest_runs", "backtest_trades", "orders", "fills", "equity"]:
            assert t in tables

    def test_log_and_read_bars(self, tmp_path):
        store = self._store(tmp_path)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [100, 200],
            }
        )
        store.log_bars("EURUSD", df)
        rows = store.read_bars("EURUSD")
        assert len(rows) == 2

    def test_log_and_read_signals(self, tmp_path):
        store = self._store(tmp_path)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01"]),
                "baseline": [1],
                "primary_confirmation": [1],
                "secondary_confirmation": [1],
                "atr": [0.005],
                "forecast_return": [0.01],
                "signal": [1],
                "entry_signal": [1],
            }
        )
        store.log_signals("EURUSD", df)
        rows = store.read_signals("EURUSD")
        assert len(rows) == 1
        assert rows[0]["signal"] == 1
        assert rows[0]["entry_signal"] == 1

    def test_log_backtest_run_and_trades(self, tmp_path):
        store = self._store(tmp_path)
        run_id = store.log_backtest_run(
            symbol="EURUSD",
            start="2024-01-01",
            end="2024-12-31",
            return_pct=5.5,
            sharpe=0.85,
            max_drawdown=-3.2,
            params={"size": 10_000},
        )
        assert run_id is not None
        assert run_id > 0

        trades = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(["2024-02-01"]),
                "exit_time": pd.to_datetime(["2024-03-01"]),
                "size": [10_000],
                "entry_price": [1.08],
                "exit_price": [1.09],
                "pnl": [100.0],
                "return_pct": [1.0],
            }
        )
        store.log_backtest_trades(run_id, trades)
        rows = store.read_backtest_trades(run_id)
        assert len(rows) == 1
        assert rows[0]["pnl"] == pytest.approx(100.0)

    def test_log_order_and_fill(self, tmp_path):
        store = self._store(tmp_path)
        order_id = store.log_order(
            symbol="EURUSD",
            side="BUY",
            units=10_000,
            order_type="MARKET",
            timestamp=datetime(2024, 1, 1),
        )
        assert order_id > 0
        store.log_fill(
            order_id=order_id,
            fill_price=1.0850,
            commission=5.0,
            timestamp=datetime(2024, 1, 1),
        )
        fills = store.read_fills(order_id=order_id)
        assert len(fills) == 1
        assert fills[0]["fill_price"] == pytest.approx(1.0850)

    def test_log_equity(self, tmp_path):
        store = self._store(tmp_path)
        store.log_equity(timestamp=datetime(2024, 1, 1), cash=10_000, equity=10_000)
        rows = store.read_equity()
        assert len(rows) == 1
        assert rows[0]["equity"] == 10_000

    def test_read_backtest_run(self, tmp_path):
        store = self._store(tmp_path)
        run_id = store.log_backtest_run(
            symbol="EURUSD",
            start="2024-01-01",
            end="2024-12-31",
            return_pct=5.5,
        )
        row = store.read_backtest_run(run_id)
        assert row is not None
        assert row["symbol"] == "EURUSD"
        assert row["return_pct"] == pytest.approx(5.5)
