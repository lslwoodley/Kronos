"""Tests for the quant experiment harness."""
import pandas as pd
import pytest

from kronosbot.alpha.experiment import (
    Experiment,
    VariantResult,
    baseline_factory,
    run_variants,
)


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


def test_variant_result_to_row():
    metrics = {
        "total_return_pct": 4.0,
        "sharpe_ratio": 1.5,
        "max_drawdown_pct": -2.0,
        "trades_count": 50,
        "win_rate_pct": 55.0,
    }
    result = VariantResult("baseline", "EURUSD", metrics)
    row = result.to_row()
    assert row["variant"] == "baseline"
    assert row["calmar"] == pytest.approx(2.0)
    assert row["return_pct"] == 4.0


def test_experiment_runs_baseline_and_variants():
    df = _make_data(periods=30)

    def sizing_variant(symbol, data, forecast_fn):
        from kronosbot.alpha.backtest import WalkForwardBacktest

        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=forecast_fn,
            min_return_threshold=0.0005,
            risk_per_trade=0.02,
            forecast_context_bars=5,
        )

    baseline, results = Experiment(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        baseline_factory=baseline_factory,
        variant_factories=[("double_risk", sizing_variant)],
        account_equity=10_000.0,
    ).run()

    assert "total_return_pct" in baseline
    assert len(results) == 2
    assert results[0].variant_name == "baseline"
    assert results[1].variant_name == "double_risk"


def test_run_variants_returns_dataframe():
    df = _make_data(periods=30)
    table = run_variants(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        baseline_factory=lambda s, d, f: baseline_factory(s, d, f, forecast_context_bars=5),
        variant_factories=[],
        account_equity=10_000.0,
    )
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "variant",
        "symbol",
        "return_pct",
        "sharpe",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "calmar",
        "promoted",
    ]
    assert len(table) == 1
    assert table.iloc[0]["variant"] == "baseline"


def test_experiment_promotion_rule():
    df = _make_data(periods=30)

    def poor_variant(symbol, data, forecast_fn):
        from kronosbot.alpha.backtest import WalkForwardBacktest

        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=forecast_fn,
            min_return_threshold=0.1,
            risk_per_trade=0.01,
            forecast_context_bars=5,
        )

    _, results = Experiment(
        symbol="EURUSD",
        data=df,
        forecast_fn=_mock_forecast,
        baseline_factory=lambda s, d, f: baseline_factory(s, d, f, forecast_context_bars=5),
        variant_factories=[("no_trade", poor_variant)],
        account_equity=10_000.0,
    ).run()
    assert results[0].variant_name == "baseline"
    assert results[1].variant_name == "no_trade"
    # A variant that produces no trades should not be promoted.
    assert results[1].summary["promoted"] is False


def test_baseline_factory():
    df = _make_data(periods=30)
    bt = baseline_factory("EURUSD", df, _mock_forecast, forecast_context_bars=5)
    metrics = bt.run(account_equity=10_000.0)
    assert "total_return_pct" in metrics
    assert metrics["trades_count"] >= 0
