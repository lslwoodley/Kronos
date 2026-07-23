"""Flask Alpha Terminal dashboard."""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request, jsonify

from kronosbot.data.feed import DataFeed
from kronosbot.alpha.forecast import ForecastEngine
from kronosbot.alpha.signal import AlphaSignal
from kronosbot.alpha.backtest import WalkForwardBacktest


app = Flask(__name__)
app.secret_key = os.environ.get("KRONOS_ALPHA_SECRET", "kronos-alpha-dev")

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]

_forecast_engine: Optional[ForecastEngine] = None


def get_forecast_engine() -> ForecastEngine:
    global _forecast_engine
    if _forecast_engine is None:
        _forecast_engine = ForecastEngine.from_pretrained(device="cpu", max_context=512)
    return _forecast_engine


@app.route("/")
def dashboard():
    symbol = request.args.get("symbol", "EURUSD")
    start = request.args.get("start", "2024-01-01")
    end = request.args.get("end", "2025-01-01")

    feed = DataFeed(cache_dir=DEFAULT_CACHE_DIR)
    df = feed.load(symbol, start=start, end=end)

    engine = get_forecast_engine()
    forecast = engine.forecast_next_day(symbol, df)
    signal = AlphaSignal.from_forecast(symbol, df, forecast)

    backtest = WalkForwardBacktest(
        symbol=symbol,
        data=df,
        forecast_fn=lambda s, d, date: engine.forecast_next_day(s, d),
        min_return_threshold=0.001,
    )
    backtest_result = backtest.run(account_equity=10_000)

    return render_template(
        "alpha_dashboard.html",
        symbol=symbol,
        supported_symbols=SUPPORTED_SYMBOLS,
        signal=signal,
        backtest=backtest_result,
    )


@app.route("/api/signal")
def api_signal():
    symbol = request.args.get("symbol", "EURUSD")
    feed = DataFeed(cache_dir=DEFAULT_CACHE_DIR)
    df = feed.load(symbol, start="2024-01-01", end="2025-01-01")
    engine = get_forecast_engine()
    forecast = engine.forecast_next_day(symbol, df)
    signal = AlphaSignal.from_forecast(symbol, df, forecast)
    return jsonify({
        "symbol": signal.symbol,
        "timestamp": signal.timestamp.isoformat(),
        "direction": signal.direction,
        "expected_return": signal.expected_return,
        "current_price": signal.current_price,
        "forecast_price": signal.forecast_price,
        "stop_price": signal.stop_price,
        "position_size": signal.position_size,
        "rationale": signal.rationale,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8052, debug=True)
