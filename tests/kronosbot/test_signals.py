import numpy as np
import pandas as pd
import pytest


class TestSignalEngine:
    def _make_trending_data(
        self, periods: int = 210, start_price: float = 1.0, slope: float = 0.002
    ) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        close = np.linspace(start_price, start_price + (periods - 1) * slope, periods)
        high = close + 0.001 + rng.normal(0, 0.0003, periods)
        low = close - 0.001 + rng.normal(0, 0.0003, periods)
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

    def _make_bearish_data(self, periods: int = 210) -> pd.DataFrame:
        return self._make_trending_data(periods, start_price=1.5, slope=-0.002)

    def test_baseline_above_200_ema_in_uptrend(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["baseline"] == 1
        assert "ema_200" in result.columns

    def test_baseline_below_200_ema_in_downtrend(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_bearish_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["baseline"] == -1

    def test_primary_confirmation_present(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["primary_confirmation"] in (-1, 1)

    def test_secondary_confirmation_present(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["secondary_confirmation"] in (-1, 0, 1)

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

    def test_signal_zero_without_forecast(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        # forecast_return is 0 when no model, so signal should be 0 per NNFX rules.
        assert result.iloc[-1]["signal"] == 0
        assert result.iloc[-1]["entry_signal"] == 0

    def test_long_signal_with_positive_forecast_return(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        # Force all NNFX long conditions to align.
        result["baseline"] = 1
        result["primary_confirmation"] = 1
        result["secondary_confirmation"] = 1
        result["forecast_return"] = 0.01
        result["signal"] = engine.compute_signal(result)
        result["entry_signal"] = engine.compute_entry_signal(result)
        assert result.iloc[-1]["signal"] == 1
        assert result.iloc[-1]["entry_signal"] == 1

    def test_short_signal_with_negative_forecast_return(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_bearish_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        # Force all NNFX short conditions to align.
        result["baseline"] = -1
        result["primary_confirmation"] = -1
        result["secondary_confirmation"] = -1
        result["forecast_return"] = -0.01
        result["signal"] = engine.compute_signal(result)
        result["entry_signal"] = engine.compute_entry_signal(result)
        assert result.iloc[-1]["signal"] == -1
        assert result.iloc[-1]["entry_signal"] == 0

    def test_compute_signal_static(self):
        from kronosbot.features.signals import SignalEngine

        df = pd.DataFrame({
            "baseline": [1, 1, -1, -1],
            "primary_confirmation": [1, 1, -1, -1],
            "secondary_confirmation": [1, -1, -1, -1],
            "forecast_return": [0.01, 0.01, -0.01, -0.01],
        })
        signal = SignalEngine.compute_signal(df)
        # Row 0: long aligned. Row 1: secondary conflicts. Row 2: short aligned.
        # Row 3: all short aligned, so signal is -1.
        assert signal.tolist() == [1, 0, -1, -1]

    def test_secondary_confirmation_forces_long(self):
        from kronosbot.features.signals import SignalEngine

        df = pd.DataFrame({
            "baseline": [1],
            "primary_confirmation": [1],
            "secondary_confirmation": [1],
            "forecast_return": [0.01],
        })
        signal = SignalEngine.compute_signal(df)
        assert signal.iloc[-1] == 1

    def test_secondary_confirmation_forces_short(self):
        from kronosbot.features.signals import SignalEngine

        df = pd.DataFrame({
            "baseline": [-1],
            "primary_confirmation": [-1],
            "secondary_confirmation": [-1],
            "forecast_return": [-0.01],
        })
        signal = SignalEngine.compute_signal(df)
        assert signal.iloc[-1] == -1

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
