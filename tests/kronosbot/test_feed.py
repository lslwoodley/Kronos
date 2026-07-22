from pathlib import Path

import pandas as pd
import pytest


class TestDataFeed:
    def test_load_returns_dataframe_with_required_columns(self, tmp_path):
        from kronosbot.data.feed import DataFeed

        feed = DataFeed(cache_dir=tmp_path)
        df = feed.load("EURUSD", start="2024-01-01", end="2024-01-31")
        assert not df.empty
        assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_uses_csv_cache(self, tmp_path):
        from kronosbot.data.feed import DataFeed

        feed = DataFeed(cache_dir=tmp_path)
        df1 = feed.load("EURUSD", start="2024-01-01", end="2024-01-31")
        cache_files = list(tmp_path.glob("*.csv"))
        assert len(cache_files) == 1
        df2 = feed.load("EURUSD", start="2024-01-01", end="2024-01-31")
        pd.testing.assert_frame_equal(df1, df2)

    def test_normalize_flattens_multiindex_columns(self, tmp_path):
        from kronosbot.data.feed import DataFeed

        sample = pd.DataFrame(
            {
                ("Adj Close", "EURUSD=X"): [1.0],
                ("Close", "EURUSD=X"): [1.0],
                ("High", "EURUSD=X"): [1.0],
                ("Low", "EURUSD=X"): [1.0],
                ("Open", "EURUSD=X"): [1.0],
                ("Volume", "EURUSD=X"): [100],
            },
            index=pd.to_datetime(["2024-01-02"]),
        )
        sample.index.name = "Date"
        sample = sample.reset_index()

        df = DataFeed._normalize_columns(sample)
        assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(df.columns)
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_load_offline_uses_synthetic_fixture(self, tmp_path):
        from kronosbot.data.feed import DataFeed

        # Create a minimal cached CSV so no network call is needed.
        cache_path = tmp_path / "EURUSD_2024-01-01_2024-02-01.csv"
        dates = pd.date_range("2024-01-01", periods=22, freq="B")
        base = 1.10
        rows = []
        for i, d in enumerate(dates):
            close = base + i * 0.001
            rows.append(
                {
                    "timestamp": d,
                    "open": close - 0.0005,
                    "high": close + 0.0005,
                    "low": close - 0.0006,
                    "close": close,
                    "volume": 1000 + i,
                }
            )
        pd.DataFrame(rows).to_csv(cache_path, index=False)

        feed = DataFeed(cache_dir=tmp_path)
        df = feed.load("EURUSD", start="2024-01-01", end="2024-02-01")
        assert len(df) == 22
        assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_supported_symbols(self, tmp_path):
        from kronosbot.data.feed import DataFeed

        feed = DataFeed(cache_dir=tmp_path)
        assert feed._ticker("EURUSD") == "EURUSD=X"
        assert feed._ticker("GBPUSD") == "GBPUSD=X"
        assert feed._ticker("USDJPY") == "USDJPY=X"
        assert feed._ticker("BTCUSD") == "BTCUSD"
