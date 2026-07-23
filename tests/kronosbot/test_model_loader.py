"""Tests for the Kronos model loader and SignalEngine model wiring."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestModelLoader:
    def test_load_kronos_model_caches_by_device(self, monkeypatch, tmp_path):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_cls_model = MagicMock()
        mock_cls_tok = MagicMock()
        mock_cls_model.from_pretrained.return_value = mock_model
        mock_cls_tok.from_pretrained.return_value = mock_tok

        import kronosbot.model_loader as ml

        # Ensure a clean cache and replace the real classes with mocks.
        monkeypatch.setattr(ml, "_MODEL_CACHE", {})
        monkeypatch.setattr(ml, "Kronos", mock_cls_model)
        monkeypatch.setattr(ml, "KronosTokenizer", mock_cls_tok)

        cache_dir = tmp_path / "hf_cache"
        model, tokenizer = ml.load_kronos_model(device="cpu", cache_dir=cache_dir)
        model2, tokenizer2 = ml.load_kronos_model(device="cpu", cache_dir=cache_dir)

        assert model is model2
        assert tokenizer is tokenizer2
        assert mock_cls_model.from_pretrained.call_count == 1
        assert mock_cls_tok.from_pretrained.call_count == 1

    def test_load_kronos_model_missing_dependency_raises(self, monkeypatch):
        import kronosbot.model_loader as ml

        monkeypatch.setattr(ml, "_MODEL_CACHE", {})
        monkeypatch.setattr(ml, "Kronos", None)
        monkeypatch.setattr(ml, "KronosTokenizer", None)
        monkeypatch.setattr(ml, "_IMPORT_ERROR", ImportError("no module named model"))

        with pytest.raises(RuntimeError):
            ml.load_kronos_model(device="cpu", cache_dir="/tmp/test_hf")


class TestSignalEngineWithModel:
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

    def test_forecast_return_zero_when_predictor_is_none(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        engine = SignalEngine(model=None, tokenizer=None)
        result = engine.generate(df)
        assert result.iloc[-1]["forecast_return"] == pytest.approx(0.0)

    def test_entry_signal_with_mock_predictor(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        mock_tok = MagicMock()
        mock_model = MagicMock()

        mock_pred = MagicMock()
        mock_pred.predict.return_value = pd.DataFrame({"close": [df["close"].iloc[-1] * 1.01]})

        with patch.object(SignalEngine, "_create_predictor", return_value=mock_pred):
            engine = SignalEngine(model=mock_model, tokenizer=mock_tok)
            result = engine.generate(df)
            # Force the remaining NNFX long conditions so the signal fires.
            result["baseline"] = 1
            result["primary_confirmation"] = 1
            result["secondary_confirmation"] = 1
            result["signal"] = engine.compute_signal(result)
            result["entry_signal"] = engine.compute_entry_signal(result)

        assert result.iloc[-1]["forecast_return"] > 0
        assert result.iloc[-1]["entry_signal"] == 1

    def test_predictor_failure_falls_back_to_zero(self):
        from kronosbot.features.signals import SignalEngine

        df = self._make_trending_data(210)
        mock_pred = MagicMock()
        mock_pred.predict.side_effect = RuntimeError("model failure")

        with patch.object(SignalEngine, "_create_predictor", return_value=mock_pred):
            engine = SignalEngine(model=MagicMock(), tokenizer=MagicMock())
            result = engine.generate(df)

        assert result.iloc[-1]["forecast_return"] == pytest.approx(0.0)
