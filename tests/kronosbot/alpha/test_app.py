from unittest.mock import MagicMock
import pandas as pd
import pytest


def _make_mock_engine():
    engine = MagicMock()
    engine.forecast_next_day.return_value = pd.Series([1.11], index=[pd.Timestamp("2024-01-04")])
    return engine


def _make_mock_feed():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.0 + i * 0.001 for i in range(30)],
            "high": [1.005 + i * 0.001 for i in range(30)],
            "low": [0.995 + i * 0.001 for i in range(30)],
            "close": [1.002 + i * 0.001 for i in range(30)],
            "volume": [1000.0] * 30,
        }
    )

    class MockFeed:
        def __init__(self, cache_dir=None):
            pass

        def load(self, symbol, start, end):
            return df

    return MockFeed


def test_dashboard_smoke(monkeypatch):
    import kronosbot.alpha.app as app_mod

    monkeypatch.setattr(app_mod, "ForecastEngine", MagicMock(from_pretrained=MagicMock(return_value=_make_mock_engine())))
    monkeypatch.setattr(app_mod, "DataFeed", _make_mock_feed())

    from kronosbot.alpha.app import app

    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
