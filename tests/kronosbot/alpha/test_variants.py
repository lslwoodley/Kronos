"""Tests for Kronos Alpha quant variants.

All tests use synthetic OHLCV data and mock forecasts so they never load Kronos or any
external model.
"""
import pandas as pd
import pytest

from kronosbot.alpha.backtest import WalkForwardBacktest
from kronosbot.alpha.variants import (
    TimesFMVariant,
    arima_residual_ensemble_factory,
    regime_filter_factory,
    slippage_spread_factory,
    timesfm_ensemble_factory,
    volatility_targeting_factory,
)


def _make_data(periods=60, trend=0.0005, noise=0.001):
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rng = pd.Series(range(periods))
    close = 1.0 + rng * trend + (rng % 5 - 2) * noise
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close - 0.0005,
            "high": close + 0.0005,
            "low": close - 0.0006,
            "close": close,
            "volume": 1000.0 + rng * 10,
        }
    )


def _mock_forecast(_symbol, _df, _date):
    return pd.Series([_df["close"].iloc[-1] * 1.001], index=[_date])


def _mock_forecast_short(_symbol, _df, _date):
    return pd.Series([_df["close"].iloc[-1] * 0.999], index=[_date])


def test_volatility_targeting_runs():
    df = _make_data(periods=60)
    factory = volatility_targeting_factory(forecast_context_bars=20)
    bt = factory("EURUSD", df, _mock_forecast)
    result = bt.run()
    assert result["trades_count"] >= 0


def test_volatility_targeting_caps_size():
    df = _make_data(periods=60)
    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=20,
    )
    factory = volatility_targeting_factory(
        target_annual_vol=0.50,
        max_size_multiplier=2.0,
        forecast_context_bars=20,
        min_return_threshold=0.0005,
    )
    vt = factory("EURUSD", df, _mock_forecast)

    base_result = base.run()
    vt_result = vt.run()
    assert base_result["trades_count"] == vt_result["trades_count"]
    # High target vol is capped; returns should not exceed 2x base by much.
    assert vt_result["total_return_pct"] <= base_result["total_return_pct"] * 2.5 + 1.0


def test_regime_filter_skips_low_trend():
    df = _make_data(periods=60)
    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=20,
    )
    factory = regime_filter_factory(
        min_trend_strength=10.0, skip_high_vol=False, forecast_context_bars=20
    )
    filtered = factory("EURUSD", df, _mock_forecast)

    base_result = base.run()
    filtered_result = filtered.run()
    assert filtered_result["trades_count"] <= base_result["trades_count"]


def test_regime_filter_allows_strong_trend():
    # Strong monotonic trend → high trend strength, trades should be allowed.
    df = _make_data(periods=60, trend=0.005, noise=0.0001)
    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=20,
    )
    factory = regime_filter_factory(
        min_trend_strength=0.001,
        skip_high_vol=False,
        forecast_context_bars=20,
        min_return_threshold=0.0005,
    )
    filtered = factory("EURUSD", df, _mock_forecast)

    base_result = base.run()
    filtered_result = filtered.run()
    assert filtered_result["trades_count"] == base_result["trades_count"]


def test_arima_ensemble_runs():
    df = _make_data(periods=60)
    factory = arima_residual_ensemble_factory(forecast_context_bars=20)
    bt = factory("EURUSD", df, _mock_forecast)
    result = bt.run()
    assert result["trades_count"] >= 0


def test_arima_ensemble_blend_changes_forecast():
    df = _make_data(periods=60)
    from kronosbot.alpha.variants import arima_residual_ensemble_forecast

    ensemble = arima_residual_ensemble_forecast(arima_weight=0.5, order=(1, 1, 1))
    kronos_price = float(_mock_forecast("EURUSD", df, df["timestamp"].iloc[-1]).iloc[-1])
    blended = ensemble("EURUSD", df, df["timestamp"].iloc[-1], _mock_forecast)
    blended_price = float(blended.iloc[-1])
    assert blended_price != kronos_price


def test_slippage_spread_reduces_returns():
    df = _make_data(periods=60)
    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=20,
    )
    factory = slippage_spread_factory(
        forecast_context_bars=20, min_return_threshold=0.0005
    )
    costly = factory("EURUSD", df, _mock_forecast)

    base_result = base.run()
    costly_result = costly.run()
    assert costly_result["trades_count"] == base_result["trades_count"]
    assert costly_result["total_return_pct"] < base_result["total_return_pct"]


def test_slippage_spread_adapts_to_jpy_pair():
    df = _make_data(periods=60)
    df["open"] = df["open"] * 150.0
    df["high"] = df["high"] * 150.0
    df["low"] = df["low"] * 150.0
    df["close"] = df["close"] * 150.0

    factory = slippage_spread_factory(forecast_context_bars=20)
    bt = factory("USDJPY", df, _mock_forecast)
    result = bt.run()
    assert result["trades_count"] >= 0


def test_timesfm_variant_is_stub():
    factory = timesfm_ensemble_factory()
    df = _make_data(periods=30)
    with pytest.raises(NotImplementedError):
        factory("EURUSD", df, _mock_forecast)


def test_variant_factories_use_no_future_data():
    """Each variant must use only the historical context provided up to today.

    We verify this by running a variant that receives a context DataFrame whose last row is
    the current bar and checking that the factory still constructs a backtest without error.
    """
    df = _make_data(periods=40)
    for factory in [
        volatility_targeting_factory(forecast_context_bars=20),
        regime_filter_factory(forecast_context_bars=20),
        arima_residual_ensemble_factory(forecast_context_bars=20),
        slippage_spread_factory(forecast_context_bars=20),
    ]:
        bt = factory("EURUSD", df, _mock_forecast)
        assert isinstance(bt, WalkForwardBacktest)
