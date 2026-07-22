import numpy as np
import pandas as pd
import pytest


class TestSignalEngine:
    def _make_trending_data(self, periods: int = 210, start_price: float = 1.0) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        close = np.linspace(start_price, start_price + (periods - 1) * 0.002, periods)
        high = close + 0.001
        low = close - 0.001
        volume = np.full(periods, 1_000.0)
        volume[-1] = 5_000.0
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=periods, freq="B"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    def test_trend_above_200_sma(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["trend"] == 1
        assert "sma_200" in result.columns

    def test_breakout_when_close_exceeds_last_19_bars(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["breakout"] == 1

    def test_volume_spike_flag(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["volume_spike"] == 1

    def test_atr_14_present_and_positive(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["atr"] > 0

    def test_forecast_return_neutral_when_model_missing(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["forecast_return"] == pytest.approx(0.0)

    def test_entry_signal_on_confirmed_bullish_setup(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        last = result.iloc[-1]
        assert last["trend"] == 1
        assert last["breakout"] == 1
        assert last["volume_spike"] == 1
        # forecast_return is 0 when no model, so entry_signal should be 0 per the rule.
        assert last["entry_signal"] == 0

    def test_entry_signal_with_positive_forecast_return(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        result["forecast_return"] = 0.01
        result["entry_signal"] = engine.compute_entry_signal(result)
        assert result.iloc[-1]["entry_signal"] == 1

    def test_too_few_rows_raises(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(10)
        engine = SignalEngine(model=None, tokenizer=None)
        with pytest.raises(ValueError):
            engine.generate(df)

    def test_missing_required_columns_raises(self):
        from kronosbot.features.signals import SignalEngine

        df = pd.DataFrame({"close": [1.0] * 250})
        engine = SignalEngine(model=None, tokenizer=None)
        with pytest.raises(ValueError):
            engine.generate(df)
