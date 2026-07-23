"""Lightweight wrapper around KronosPredictor for one-day-ahead forecasting."""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


class ForecastEngine:
    """Load Kronos once and expose a single `forecast_next_day` method."""

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
            self._build_predictor()

    @classmethod
    def from_pretrained(cls, device: str = "cpu", max_context: int = 512) -> "ForecastEngine":
        from kronosbot.model_loader import load_kronos_model

        model, tokenizer = load_kronos_model(device=device)
        return cls(model=model, tokenizer=tokenizer, device=device, max_context=max_context)

    def _build_predictor(self):
        repo_root = Path(__file__).parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from model.kronos import KronosPredictor

        self._predictor = KronosPredictor(self.model, self.tokenizer, device=self.device)

    def forecast_next_day(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        """Return a one-step forecast for the next calendar day after `df`."""
        if self._predictor is None:
            return pd.Series(dtype=float)

        df = df.copy().reset_index(drop=True)
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last_date = df["timestamp"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Kronos expects OHLCV + optional amount
        if "amount" not in df.columns and "volume" in df.columns:
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

        pred_df = self._predictor.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=df["timestamp"],
            y_timestamp=pd.Series([next_date]),
            pred_len=1,
            sample_count=1,
            sample_logits=False,
            verbose=False,
        )
        return pred_df["close"]
