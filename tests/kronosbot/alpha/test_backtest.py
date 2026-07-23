import pandas as pd
import numpy as np


def test_walk_forward_backtest_returns_metrics():
    from kronosbot.alpha.backtest import WalkForwardBacktest

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.linspace(1.0, 1.1, 30),
            "high": np.linspace(1.01, 1.11, 30),
            "low": np.linspace(0.99, 1.09, 30),
            "close": np.linspace(1.0, 1.1, 30),
            "volume": np.full(30, 1000.0),
        }
    )

    def mock_forecast(_symbol, _df, _date):
        return pd.Series([_df["close"].iloc[-1] * 1.001], index=[_date])

    backtest = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=mock_forecast,
        min_return_threshold=0.0005,
        forecast_context_bars=5,
    )
    result = backtest.run(account_equity=10_000)
    assert "total_return_pct" in result
    assert "trades" in result
