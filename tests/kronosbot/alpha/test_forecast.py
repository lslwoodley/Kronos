from unittest.mock import MagicMock
import pandas as pd
import pytest


def test_forecast_next_day_returns_series():
    from kronosbot.alpha.forecast import ForecastEngine

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_pred = MagicMock()
    mock_pred.predict.return_value = pd.DataFrame(
        {"close": [1.11]},
        index=[pd.Timestamp("2024-01-04")],
    )

    engine = ForecastEngine(model=mock_model, tokenizer=mock_tok, max_context=30)
    engine._predictor = mock_pred

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [1.0, 1.05, 1.08],
            "high": [1.05, 1.08, 1.12],
            "low": [0.98, 1.02, 1.05],
            "close": [1.04, 1.07, 1.10],
            "volume": [1000, 1100, 1200],
        }
    )
    forecast = engine.forecast_next_day("EURUSD", df)
    assert len(forecast) == 1
    assert forecast.iloc[0] == pytest.approx(1.11)
