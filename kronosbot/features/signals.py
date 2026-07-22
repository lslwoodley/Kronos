from typing import Optional

import numpy as np
import pandas as pd


class SignalEngine:
    """Compute Kronos + quantitative trading signals from OHLCV bars."""

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

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def compute_trend(self, df: pd.DataFrame) -> pd.Series:
        sma_200 = df["close"].rolling(window=200).mean()
        return (df["close"] > sma_200).astype(int)

    def compute_breakout(self, df: pd.DataFrame) -> pd.Series:
        # Current close exceeds max of previous 19 closes (excluding current bar).
        prev_high = df["close"].shift(1).rolling(window=19).max()
        return (df["close"] > prev_high).astype(int)

    def compute_volume_spike(self, df: pd.DataFrame) -> pd.Series:
        avg_vol = df["volume"].shift(1).rolling(window=20).mean()
        vol_spike = (df["volume"] > 1.5 * avg_vol).astype(int)
        # Fallback to ATR spike when volume is zero/missing.
        atr = self._true_range(df).rolling(window=14).mean()
        avg_atr = atr.shift(1).rolling(window=20).mean()
        atr_spike = (atr > 1.5 * avg_atr).astype(int)
        missing_vol = avg_vol.isna() | (df["volume"] == 0)
        return vol_spike.where(~missing_vol, other=atr_spike)

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        return self._true_range(df).rolling(window=14).mean()

    def compute_forecast_return(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.Series:
        if self.model is None or self.tokenizer is None:
            return pd.Series(0.0, index=df.index)

        # Kronos integration point: call the model on the last max_context bars.
        # For now we delegate to a lightweight predictor if available.
        try:
            import sys
            from pathlib import Path

            repo_root = Path(__file__).parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from model.kronos import KronosPredictor

            predictor = KronosPredictor(self.model, self.tokenizer, device=self.device)
            predictions = []
            for i in range(len(df)):
                start = max(0, i - self.max_context + 1)
                window = df.iloc[start : i + 1]
                if len(window) < 2:
                    predictions.append(0.0)
                    continue
                next_timestamp = window["timestamp"].iloc[-1] + pd.Timedelta(days=forecast_horizon)
                pred_df = predictor.predict(
                    df=window[["open", "high", "low", "close", "volume", "amount"]] if "amount" in window.columns else window[["open", "high", "low", "close", "volume"]],
                    x_timestamp=window["timestamp"],
                    y_timestamp=pd.Series([next_timestamp]),
                    pred_len=1,
                )
                next_close = float(pred_df["close"].iloc[0])
                current_close = float(df["close"].iloc[i])
                predictions.append((next_close - current_close) / current_close)
            return pd.Series(predictions, index=df.index)
        except Exception:
            return pd.Series(0.0, index=df.index)

    @staticmethod
    def compute_entry_signal(df: pd.DataFrame) -> pd.Series:
        return (
            (df["trend"] == 1)
            & (df["breakout"] == 1)
            & (df["volume_spike"] == 1)
            & (df["forecast_return"] > 0)
        ).astype(int)

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

        result["sma_200"] = result["close"].rolling(window=200).mean()
        result["trend"] = self.compute_trend(result)
        result["breakout"] = self.compute_breakout(result)
        result["volume_spike"] = self.compute_volume_spike(result)
        result["atr"] = self.compute_atr(result)
        result["forecast_return"] = self.compute_forecast_return(result, forecast_horizon=forecast_horizon)
        result["entry_signal"] = self.compute_entry_signal(result)
        return result
