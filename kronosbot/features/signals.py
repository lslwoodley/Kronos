from typing import Optional

import numpy as np
import pandas as pd


class SignalEngine:
    """Compute NNFX-style trading signals from OHLCV bars plus a Kronos forecast."""

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        device: str = "cpu",
        max_context: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context = max_context
        self._predictor = None
        if model is not None and tokenizer is not None:
            self._predictor = self._create_predictor()

    def _create_predictor(self) -> Optional[object]:
        """Build a KronosPredictor once and cache it on the instance."""
        try:
            import sys
            from pathlib import Path

            repo_root = Path(__file__).parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from model.kronos import KronosPredictor

            return KronosPredictor(self.model, self.tokenizer, device=self.device)
        except Exception:
            return None

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ------------------------------------------------------------------
    # NNFX components
    # ------------------------------------------------------------------
    def compute_baseline(self, df: pd.DataFrame) -> pd.Series:
        """200 EMA baseline: +1 above, -1 below."""
        ema_200 = df["close"].ewm(span=200, adjust=False).mean()
        return np.where(df["close"] > ema_200, 1, np.where(df["close"] < ema_200, -1, 0))

    def compute_primary_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """ATR trend: +1 when ATR(14) > SMA20(ATR), -1 otherwise.

        Expanding volatility confirms that the market is moving.
        """
        atr = self.compute_atr(df)
        atr_sma20 = atr.rolling(window=20).mean()
        return np.where(atr > atr_sma20, 1, -1)

    def compute_secondary_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """RSI(14) momentum: +1 when RSI in bullish 50-70 band, -1 when 30-50.

        Avoids overbought (>70) and oversold (<30) conditions.
        """
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)
        return np.where((rsi > 50) & (rsi < 70), 1, np.where((rsi > 30) & (rsi < 50), -1, 0))

    # ------------------------------------------------------------------
    # Legacy / helper computations
    # ------------------------------------------------------------------
    def compute_trend(self, df: pd.DataFrame) -> pd.Series:
        """Compatibility alias for the 200 EMA baseline signal."""
        return self.compute_baseline(df)

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        return self._true_range(df).rolling(window=14).mean()

    def compute_forecast_return(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.Series:
        if self._predictor is None:
            return pd.Series(0.0, index=df.index)

        predictions = []
        for i in range(len(df)):
            start = max(0, i - self.max_context + 1)
            window = df.iloc[start : i + 1]
            if len(window) < 2:
                predictions.append(0.0)
                continue
            next_timestamp = window["timestamp"].iloc[-1] + pd.Timedelta(days=forecast_horizon)
            try:
                pred_df = self._predictor.predict(
                    df=window[["open", "high", "low", "close", "volume", "amount"]]
                    if "amount" in window.columns
                    else window[["open", "high", "low", "close", "volume"]],
                    x_timestamp=window["timestamp"],
                    y_timestamp=pd.Series([next_timestamp]),
                    pred_len=1,
                    verbose=False,
                )
                next_close = float(pred_df["close"].iloc[0])
                current_close = float(df["close"].iloc[i])
                predictions.append((next_close - current_close) / current_close)
            except Exception:
                predictions.append(0.0)
        return pd.Series(predictions, index=df.index)

    @staticmethod
    def compute_signal(df: pd.DataFrame) -> pd.Series:
        """NNFX entry signal: +1 long, -1 short, 0 flat.

        Long when baseline up + primary confirmation up + secondary
        confirmation up + forecast bullish.
        Short when all four align bearish.
        """
        long = (
            (df["baseline"] == 1)
            & (df["primary_confirmation"] == 1)
            & (df["secondary_confirmation"] == 1)
            & (df["forecast_return"] > 0)
        )
        short = (
            (df["baseline"] == -1)
            & (df["primary_confirmation"] == -1)
            & (df["secondary_confirmation"] == -1)
            & (df["forecast_return"] < 0)
        )
        return pd.Series(np.where(short, -1, np.where(long, 1, 0)), index=df.index)

    @staticmethod
    def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
        """Backward-compatible alias returning 1/0 for long-only consumers."""
        signal = SignalEngine.compute_signal(df)
        return (signal == 1).astype(int)

    def generate(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        result = df.copy()

        if isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index()
            if result.columns[0] != "timestamp":
                result = result.rename(columns={result.columns[0]: "timestamp"})
        elif "timestamp" not in result.columns:
            raise ValueError("Data must have a 'timestamp' column or a DatetimeIndex")

        result["timestamp"] = pd.to_datetime(result["timestamp"])

        if not self.REQUIRED_COLUMNS.issubset(result.columns):
            missing = self.REQUIRED_COLUMNS - set(result.columns)
            raise ValueError(f"Missing required columns: {missing}")

        if len(result) < 200:
            raise ValueError(f"Need at least 200 bars, got {len(result)}")

        # NNFX baseline + confirmations
        result["ema_200"] = result["close"].ewm(span=200, adjust=False).mean()
        result["baseline"] = self.compute_baseline(result)
        result["primary_confirmation"] = self.compute_primary_confirmation(result)
        result["secondary_confirmation"] = self.compute_secondary_confirmation(result)

        # Volatility + forecast
        result["atr"] = self.compute_atr(result)
        result["forecast_return"] = self.compute_forecast_return(result, forecast_horizon=forecast_horizon)

        # Final -1/0/+1 signal
        result["signal"] = self.compute_signal(result)
        # Backward-compatible long-only entry flag for legacy callers
        result["entry_signal"] = self.compute_entry_signal(result)

        return result
