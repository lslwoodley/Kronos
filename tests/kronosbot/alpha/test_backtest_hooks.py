"""Tests for backtest hooks (custom sizing, execution cost, signal filters)."""
import pandas as pd
import pytest

from kronosbot.alpha.backtest import WalkForwardBacktest


def _make_data(periods=30, trend=0.001):
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = 1.0 + pd.Series(range(periods)) * trend
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close - 0.0005,
            "high": close + 0.0005,
            "low": close - 0.0006,
            "close": close,
            "volume": 1000.0,
        }
    )


def _mock_forecast(_symbol, _df, _date):
    return pd.Series([_df["close"].iloc[-1] * 1.001], index=[_date])


def test_custom_sizing_hook_changes_position_size():
    df = _make_data(periods=30)

    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
    )
    sized = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
        sizing_fn=lambda signal, _ctx, _equity: 100.0,  # fixed tiny size
    )

    base_result = base.run()
    sized_result = sized.run()

    assert base_result["trades_count"] == sized_result["trades_count"]
    # Very small fixed sizing should produce much smaller PnL than default sizing.
    assert abs(sized_result["total_return_pct"]) < abs(base_result["total_return_pct"]) * 0.5


def test_execution_cost_hook_reduces_returns():
    df = _make_data(periods=30)

    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
    )
    costly = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
        execution_cost_fn=lambda _entry, _exit, _direction, _ctx: 0.0001,
    )

    base_result = base.run()
    costly_result = costly.run()

    assert costly_result["trades_count"] == base_result["trades_count"]
    assert costly_result["total_return_pct"] < base_result["total_return_pct"]


def test_signal_filter_hook_skips_filtered_signals():
    df = _make_data(periods=30)

    base = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
    )
    filtered = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
        signal_filter_fn=lambda _signal, _ctx, _idx: False,
    )

    base_result = base.run()
    filtered_result = filtered.run()

    assert filtered_result["trades_count"] == 0
    assert base_result["trades_count"] > 0


def test_execution_cost_fn_signature():
    df = _make_data(periods=30)
    called = {}

    def capture_cost(entry, exit_row, direction, context):
        called["entry"] = float(entry["open"])
        called["exit"] = float(exit_row["open"])
        called["direction"] = direction
        called["context_len"] = len(context)
        return 0.0002

    bt = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
        execution_cost_fn=capture_cost,
    )
    bt.run()

    assert called["direction"] in (-1, 1)
    assert called["context_len"] >= 5
    assert called["entry"] > 0
    assert called["exit"] > 0


def test_sizing_fn_signature():
    df = _make_data(periods=30)
    called = {}

    def capture_sizing(signal, context, equity):
        called["signal"] = signal
        called["equity"] = equity
        called["context_len"] = len(context)
        return 1.0

    bt = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
        sizing_fn=capture_sizing,
    )
    bt.run()

    assert called["signal"].direction in (-1, 1)
    assert called["equity"] == pytest.approx(10_000.0, rel=0.01)
    assert called["context_len"] >= 5


