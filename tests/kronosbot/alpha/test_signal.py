import pandas as pd
import numpy as np
from kronosbot.alpha.signal import AlphaSignal


def test_expected_return_from_forecast():
    df = pd.DataFrame(
        {
            "close": [1.0, 1.01, 1.02],
            "high": [1.01, 1.02, 1.03],
            "low": [0.99, 1.00, 1.01],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    forecast = pd.Series([1.03], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast("EURUSD", df, forecast, min_return_threshold=0.005)
    assert signal.direction == 1
    assert signal.expected_return > 0


def test_no_signal_when_forecast_below_threshold():
    df = pd.DataFrame(
        {
            "close": [1.0, 1.0001, 1.0002],
            "high": [1.0001, 1.0002, 1.0003],
            "low": [0.9999, 1.0, 1.0001],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    forecast = pd.Series([1.00021], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast("EURUSD", df, forecast, min_return_threshold=0.001)
    assert signal.direction == 0


def test_short_signal_from_negative_forecast():
    df = pd.DataFrame(
        {
            "close": [1.0, 1.01, 1.02],
            "high": [1.01, 1.02, 1.03],
            "low": [0.99, 1.00, 1.01],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    forecast = pd.Series([0.98], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast("EURUSD", df, forecast, min_return_threshold=0.005)
    assert signal.direction == -1
    assert signal.stop_price > signal.current_price
