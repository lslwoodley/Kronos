from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


class DataFeed:
    """Fetch and cache daily forex OHLCV bars from yfinance."""

    SYMBOL_TO_YFINANCE = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ticker(self, symbol: str) -> str:
        return self.SYMBOL_TO_YFINANCE.get(symbol, symbol)

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{symbol}_{start}_{end}.csv"

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten yfinance-style multi-index columns into required OHLCV names."""
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] for col in df.columns]

        df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

        rename = {
            "date": "timestamp",
            "unnamed:_0": "timestamp",
            "adj_close": "close",
        }
        df = df.rename(columns=rename)

        # Drop duplicate columns produced by Close + Adj Close both mapping to close.
        df = df.loc[:, ~df.columns.duplicated()].copy()

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        cache = self._cache_path(symbol, start, end)
        if cache.exists():
            df = pd.read_csv(cache)
            df = self._normalize_columns(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        ticker = self._ticker(symbol)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} ({ticker})")

        df = df.reset_index()
        df = self._normalize_columns(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.to_csv(cache, index=False)
        return df
