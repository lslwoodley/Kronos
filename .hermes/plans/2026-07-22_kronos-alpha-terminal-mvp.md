# Kronos Alpha Terminal — MVP Plan

> **For Hermes:** Implement this plan task-by-task using the `subagent-driven-development` skill. Treat the current Kronos Bot codebase as the starting point, but build a clean new `alpha/` module rather than patching the old NNFX filters.

---

## Goal

Build a **Kronos Alpha Terminal** MVP: a single web dashboard that uses the Kronos foundation model to forecast next-day FX prices, generates a daily trading signal, and runs a simple walk-forward backtest. The focus is on the model signal as the primary alpha source, with minimal risk management, not a stack of technical indicators.

## Architecture

- **Model layer**: Kronos (HuggingFace `NeoQuasar/Kronos-small`) produces a point forecast for the next day’s close.
- **Signal layer**: Forecasted return is converted to a directional signal and a volatility-scaled position size.
- **Backtest layer**: Walk-forward, one-day forecast → trade next day open → hold one day → close at next next day open. No intraday complexity.
- **UI layer**: Single-page Flask dashboard with symbol selector, forecast chart, signal card, and backtest metrics.
- **Execution layer**: Paper-only for MVP. Live broker adapters are out of scope.

## Tech Stack

- Python 3.12
- Kronos model via `model.Kronos` and `model.KronosPredictor`
- yfinance for daily FX data
- Flask + Tailwind CSS CDN + HTMX (same as existing webui)
- pytest for testing
- Backtesting.py is **not** used for this MVP; the backtest is event-based to keep the model signal front-and-center.

## Current Starting Context

- Repo: `/home/bruce/code/kronos` (fork of `lslwoodley/Kronos`)
- Kronos model loading already exists in `kronosbot/model_loader.py`
- Existing `kronosbot/data/feed.py` loads yfinance FX daily bars
- Existing `kronosbot/webui/app.py` is a multi-page management UI; we will create a separate `alpha_app.py` for the MVP
- Python venv: `.venv` with PyTorch CPU, Flask, yfinance, pandas, numpy
- Tests currently: 57 passing, 10 in the old NNFX branch pending (we will not fix them; we will add new tests for the alpha module)

---

## Task 1: Create the Alpha Signal Module

**Objective:** Implement a clean module that converts Kronos forecasts into a daily trading signal and position size.

**Files:**
- Create: `kronosbot/alpha/signal.py`
- Test: `tests/kronosbot/alpha/test_signal.py`

**Step 1: Write failing test**

```python
import pandas as pd
import numpy as np
from kronosbot.alpha.signal import AlphaSignal


def test_expected_return_from_forecast():
    df = pd.DataFrame({
        "close": [1.0, 1.01, 1.02],
    }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    forecast = pd.Series([1.03], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast(df, forecast, min_return_threshold=0.005)
    assert signal.direction == 1
    assert signal.expected_return > 0
```

Run: `pytest tests/kronosbot/alpha/test_signal.py::test_expected_return_from_forecast -v`
Expected: FAIL — `AlphaSignal` not defined

**Step 2: Implement `AlphaSignal`**

```python
"""Convert a Kronos point forecast into a trading signal and position size."""
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import numpy as np


@dataclass
class AlphaSignal:
    symbol: str
    timestamp: pd.Timestamp
    direction: Literal[-1, 0, 1]
    expected_return: float
    current_price: float
    forecast_price: float
    stop_price: Optional[float]
    position_size: float
    atr_14: float
    rationale: str

    @classmethod
    def from_forecast(
        cls,
        symbol: str,
        historical_df: pd.DataFrame,
        forecast: pd.Series,
        account_equity: float = 10_000.0,
        risk_per_trade: float = 0.01,
        min_return_threshold: float = 0.001,
    ) -> "AlphaSignal":
        """Generate a signal from a Kronos forecast.

        Args:
            symbol: e.g., "EURUSD"
            historical_df: DataFrame with OHLCV bars. Must include at least 14 rows.
            forecast: pd.Series with one forecasted close price, indexed by target date.
            account_equity: current account equity for position sizing
            risk_per_trade: fraction of equity to risk per trade (default 1%)
            min_return_threshold: minimum absolute forecast return to generate a non-zero signal
        """
        close = historical_df["close"]
        atr = _atr_14(historical_df)
        current_price = float(close.iloc[-1])
        forecast_price = float(forecast.iloc[-1])
        expected_return = (forecast_price - current_price) / current_price
        timestamp = forecast.index[0]

        if abs(expected_return) < min_return_threshold:
            direction = 0
        else:
            direction = 1 if expected_return > 0 else -1

        # Risk-based position sizing: risk 1% of equity over 2*ATR stop distance.
        stop_distance = 2 * atr if not np.isnan(atr) else current_price * 0.005
        if direction == 0:
            stop_price = None
            position_size = 0.0
        else:
            stop_price = current_price - direction * stop_distance
            risk_amount = account_equity * risk_per_trade
            position_size = risk_amount / stop_distance  # units

        rationale = (
            f"Forecast {symbol} close {timestamp.date()} = {forecast_price:.5f} "
            f"({expected_return*100:+.3f}%). ATR(14)={atr:.5f}."
        )

        return cls(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            expected_return=expected_return,
            current_price=current_price,
            forecast_price=forecast_price,
            stop_price=stop_price,
            position_size=position_size,
            atr_14=atr,
            rationale=rationale,
        )


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr_14(df: pd.DataFrame) -> float:
    return float(_true_range(df).rolling(window=14).mean().iloc[-1])
```

Run: `pytest tests/kronosbot/alpha/test_signal.py -v`
Expected: PASS

**Step 3: Add edge-case tests**

```python
def test_no_signal_when_forecast_below_threshold():
    df = pd.DataFrame({
        "close": [1.0, 1.0001, 1.0002],
        "high": [1.0001, 1.0002, 1.0003],
        "low": [0.9999, 1.0, 1.0001],
    }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    forecast = pd.Series([1.00021], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast("EURUSD", df, forecast, min_return_threshold=0.001)
    assert signal.direction == 0


def test_short_signal_from_negative_forecast():
    df = pd.DataFrame({
        "close": [1.0, 1.01, 1.02],
        "high": [1.01, 1.02, 1.03],
        "low": [0.99, 1.00, 1.01],
    }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
    forecast = pd.Series([0.98], index=[pd.Timestamp("2024-01-04")])
    signal = AlphaSignal.from_forecast("EURUSD", df, forecast, min_return_threshold=0.005)
    assert signal.direction == -1
    assert signal.stop_price > signal.current_price
```

Run: `pytest tests/kronosbot/alpha/test_signal.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add kronosbot/alpha/signal.py tests/kronosbot/alpha/test_signal.py
git commit -m "feat(alpha): add AlphaSignal from Kronos forecast"
```

---

## Task 2: Create a Forecast Engine Wrapper

**Objective:** Load Kronos model once and produce a clean `forecast_next_day(symbol, df)` function for the Alpha Terminal.

**Files:**
- Create: `kronosbot/alpha/forecast.py`
- Test: `tests/kronosbot/alpha/test_forecast.py` (with mock model to avoid heavy loading in tests)

**Step 1: Write failing test**

```python
from unittest.mock import MagicMock
import pandas as pd
import numpy as np


def test_forecast_next_day_returns_series():
    from kronosbot.alpha.forecast import ForecastEngine

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_pred = MagicMock()
    mock_pred.predict.return_value = pd.DataFrame({
        "close": [1.11],
    }, index=[pd.Timestamp("2024-01-04")])

    engine = ForecastEngine(model=mock_model, tokenizer=mock_tok, max_context=30)
    engine._predictor = mock_pred

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
        "open": [1.0, 1.05, 1.08],
        "high": [1.05, 1.08, 1.12],
        "low": [0.98, 1.02, 1.05],
        "close": [1.04, 1.07, 1.10],
        "volume": [1000, 1100, 1200],
    })
    forecast = engine.forecast_next_day("EURUSD", df)
    assert len(forecast) == 1
    assert forecast.iloc[0] == pytest.approx(1.11)
```

Run: `pytest tests/kronosbot/alpha/test_forecast.py::test_forecast_next_day_returns_series -v`
Expected: FAIL — `ForecastEngine` not defined

**Step 2: Implement `ForecastEngine`**

```python
"""Lightweight wrapper around KronosPredictor for one-day-ahead forecasting."""
from typing import Optional

import pandas as pd
import numpy as np

from kronosbot.model_loader import load_kronos_model


class ForecastEngine:
    """Load Kronos once and expose a single `forecast_next_day` method."""

    def __init__(
        self,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        device: str = "cpu",
        max_context: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context = max_context
        self._predictor = None
        if model is not None and tokenizer is not None:
            self._build_predictor()

    @classmethod
    def from_pretrained(cls, device: str = "cpu", max_context: int = 512) -> "ForecastEngine":
        model, tokenizer = load_kronos_model(device=device)
        return cls(model=model, tokenizer=tokenizer, device=device, max_context=max_context)

    def _build_predictor(self):
        import sys
        from pathlib import Path

        repo_root = Path(__file__).parent.parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from model.kronos import KronosPredictor

        self._predictor = KronosPredictor(self.model, self.tokenizer, device=self.device)

    def forecast_next_day(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        """Return a one-step forecast for the next calendar day after `df`."""
        if self._predictor is None:
            return pd.Series(dtype=float)

        df = df.copy().reset_index(drop=True)
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last_date = df["timestamp"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Kronos expects OHLCV + optional amount
        if "amount" not in df.columns and "volume" in df.columns:
            df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

        pred_df = self._predictor.predict(
            df=df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=df["timestamp"],
            y_timestamp=pd.Series([next_date]),
            pred_len=1,
            verbose=False,
        )
        return pred_df["close"]
```

Run: `pytest tests/kronosbot/alpha/test_forecast.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add kronosbot/alpha/forecast.py tests/kronosbot/alpha/test_forecast.py
git commit -m "feat(alpha): add ForecastEngine wrapper for Kronos one-day forecast"
```

---

## Task 3: Build the Walk-Forward Backtest Engine

**Objective:** Create a simple event-based backtest that forecasts each day, trades next day open, holds one day, and records PnL.

**Files:**
- Create: `kronosbot/alpha/backtest.py`
- Test: `tests/kronosbot/alpha/test_backtest.py`

**Step 1: Write failing test**

```python
import pandas as pd
import numpy as np
from datetime import datetime


def test_walk_forward_backtest_returns_metrics():
    from kronosbot.alpha.backtest import WalkForwardBacktest

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.linspace(1.0, 1.1, 30),
        "high": np.linspace(1.01, 1.11, 30),
        "low": np.linspace(0.99, 1.09, 30),
        "close": np.linspace(1.0, 1.1, 30),
        "volume": np.full(30, 1000.0),
    })

    def mock_forecast(_symbol, _df, _date):
        return pd.Series([_df["close"].iloc[-1] * 1.001], index=[_date])

    backtest = WalkForwardBacktest(
        symbol="EURUSD",
        data=df,
        forecast_fn=mock_forecast,
        min_return_threshold=0.0005,
    )
    result = backtest.run(account_equity=10_000)
    assert "total_return_pct" in result
    assert "trades" in result
```

Run: `pytest tests/kronosbot/alpha/test_backtest.py::test_walk_forward_backtest_returns_metrics -v`
Expected: FAIL — `WalkForwardBacktest` not defined

**Step 2: Implement `WalkForwardBacktest`**

```python
"""Simple walk-forward backtest: forecast today, trade tomorrow open, close next open."""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from kronosbot.alpha.signal import AlphaSignal


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float


class WalkForwardBacktest:
    """Event-based backtest that uses a model forecast each day to trade the next open."""

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        forecast_fn: Callable[[str, pd.DataFrame, pd.Timestamp], pd.Series],
        min_return_threshold: float = 0.001,
        risk_per_trade: float = 0.01,
        forecast_context_bars: int = 60,
    ):
        self.symbol = symbol
        self.data = data.copy().reset_index(drop=True)
        self.forecast_fn = forecast_fn
        self.min_return_threshold = min_return_threshold
        self.risk_per_trade = risk_per_trade
        self.forecast_context_bars = forecast_context_bars

    def run(self, account_equity: float = 10_000.0) -> Dict:
        df = self.data
        trades: List[Trade] = []
        equity_curve = []
        equity = float(account_equity)

        # Need at least context + 2 bars (forecast bar, trade entry, trade exit)
        for i in range(self.forecast_context_bars, len(df) - 1):
            today = df.iloc[i]
            tomorrow = df.iloc[i + 1]
            context = df.iloc[: i + 1]

            forecast = self.forecast_fn(self.symbol, context, tomorrow["timestamp"])
            signal = AlphaSignal.from_forecast(
                symbol=self.symbol,
                historical_df=context,
                forecast=forecast,
                account_equity=equity,
                risk_per_trade=self.risk_per_trade,
                min_return_threshold=self.min_return_threshold,
            )

            if signal.direction == 0:
                equity_curve.append({"timestamp": tomorrow["timestamp"], "equity": equity})
                continue

            # Trade: enter at tomorrow open, exit at next open
            entry_price = float(tomorrow["open"])
            if i + 2 < len(df):
                exit_price = float(df.iloc[i + 2]["open"])
                exit_time = df.iloc[i + 2]["timestamp"]
            else:
                exit_price = float(tomorrow["close"])
                exit_time = tomorrow["timestamp"]

            pnl_pct = signal.direction * (exit_price - entry_price) / entry_price
            pnl = pnl_pct * min(signal.position_size, equity)
            pnl = max(min(pnl, equity), -equity)  # cannot lose more than equity
            equity += pnl

            trade = Trade(
                entry_time=tomorrow["timestamp"],
                exit_time=exit_time,
                direction=signal.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                return_pct=pnl_pct * 100,
            )
            trades.append(trade)
            equity_curve.append({"timestamp": exit_time, "equity": equity})

        return self._metrics(equity, account_equity, trades, equity_curve)

    def _metrics(self, final_equity, start_equity, trades, equity_curve):
        total_return = (final_equity - start_equity) / start_equity
        win_trades = [t for t in trades if t.pnl > 0]
        returns = [t.return_pct / 100 for t in trades]
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
        max_dd = 0.0
        if not equity_df.empty:
            peak = equity_df["equity"].cummax()
            drawdown = (equity_df["equity"] - peak) / peak
            max_dd = float(drawdown.min())

        return {
            "symbol": self.symbol,
            "start_equity": start_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "trades_count": len(trades),
            "win_rate_pct": (len(win_trades) / len(trades) * 100) if trades else 0.0,
            "trades": [t.__dict__ for t in trades],
            "equity_curve": equity_curve,
        }
```

Run: `pytest tests/kronosbot/alpha/test_backtest.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add kronosbot/alpha/backtest.py tests/kronosbot/alpha/test_backtest.py
git commit -m "feat(alpha): add walk-forward backtest engine"
```

---

## Task 4: Create the Alpha Dashboard Flask App

**Objective:** Build a single-page dashboard that loads the model, shows today’s signal for a symbol, and runs the backtest.

**Files:**
- Create: `kronosbot/alpha/app.py`
- Create: `kronosbot/alpha/templates/alpha_dashboard.html`
- Create: `kronosbot/alpha/static/.gitkeep` (or reuse existing static)
- Test: `tests/kronosbot/alpha/test_app.py` (smoke test)

**Step 1: Write failing test**

```python
def test_dashboard_smoke():
    from kronosbot.alpha.app import app

    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
```

Run: `pytest tests/kronosbot/alpha/test_app.py::test_dashboard_smoke -v`
Expected: FAIL — `app.py` not found

**Step 2: Implement `app.py`**

```python
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
```

**Step 3: Create template `alpha_dashboard.html`**

Create `kronosbot/alpha/templates/alpha_dashboard.html`:

```html
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kronos Alpha Terminal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = { darkMode: 'class' }
    </script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
    <div class="max-w-5xl mx-auto p-6">
        <h1 class="text-3xl font-bold mb-6">Kronos Alpha Terminal</h1>

        <form method="get" class="mb-6 flex gap-4 items-end">
            <div>
                <label class="block text-sm text-slate-400">Symbol</label>
                <select name="symbol" class="bg-slate-800 border border-slate-700 rounded px-3 py-2">
                    {% for s in supported_symbols %}
                        <option value="{{ s }}" {% if s == symbol %}selected{% endif %}>{{ s }}</option>
                    {% endfor %}
                </select>
            </div>
            <div>
                <label class="block text-sm text-slate-400">Start</label>
                <input type="date" name="start" value="{{ request.args.get('start', '2024-01-01') }}" class="bg-slate-800 border border-slate-700 rounded px-3 py-2">
            </div>
            <div>
                <label class="block text-sm text-slate-400">End</label>
                <input type="date" name="end" value="{{ request.args.get('end', '2025-01-01') }}" class="bg-slate-800 border border-slate-700 rounded px-3 py-2">
            </div>
            <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded">Run</button>
        </form>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-slate-800 rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4">Today's Signal</h2>
                <div class="space-y-2">
                    <p><span class="text-slate-400">Symbol:</span> {{ signal.symbol }}</p>
                    <p><span class="text-slate-400">Date:</span> {{ signal.timestamp.date() }}</p>
                    <p><span class="text-slate-400">Direction:</span>
                        {% if signal.direction == 1 %}<span class="text-green-400">LONG</span>{% endif %}
                        {% if signal.direction == -1 %}<span class="text-red-400">SHORT</span>{% endif %}
                        {% if signal.direction == 0 %}<span class="text-slate-400">FLAT</span>{% endif %}
                    </p>
                    <p><span class="text-slate-400">Current:</span> {{ "%.5f"|format(signal.current_price) }}</p>
                    <p><span class="text-slate-400">Forecast:</span> {{ "%.5f"|format(signal.forecast_price) }}</p>
                    <p><span class="text-slate-400">Expected Return:</span> {{ "%.3f"|format(signal.expected_return * 100) }}%</p>
                    <p><span class="text-slate-400">Stop:</span> {{ "%.5f"|format(signal.stop_price) if signal.stop_price else "—" }}</p>
                    <p><span class="text-slate-400">Position Size:</span> {{ "%.0f"|format(signal.position_size) }} units</p>
                    <p class="text-sm text-slate-400 mt-4">{{ signal.rationale }}</p>
                </div>
            </div>

            <div class="bg-slate-800 rounded-lg p-6">
                <h2 class="text-xl font-semibold mb-4">Backtest (2024–2025)</h2>
                <div class="space-y-2">
                    <p><span class="text-slate-400">Return:</span> {{ "%.2f"|format(backtest.total_return_pct) }}%</p>
                    <p><span class="text-slate-400">Sharpe:</span> {{ "%.2f"|format(backtest.sharpe_ratio) }}</p>
                    <p><span class="text-slate-400">Max Drawdown:</span> {{ "%.2f"|format(backtest.max_drawdown_pct) }}%</p>
                    <p><span class="text-slate-400">Trades:</span> {{ backtest.trades_count }}</p>
                    <p><span class="text-slate-400">Win Rate:</span> {{ "%.1f"|format(backtest.win_rate_pct) }}%</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

**Step 4: Run smoke test**

Run: `pytest tests/kronosbot/alpha/test_app.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add kronosbot/alpha/app.py kronosbot/alpha/templates/alpha_dashboard.html tests/kronosbot/alpha/test_app.py
git commit -m "feat(alpha): add Kronos Alpha Terminal dashboard"
```

---

## Task 5: Add a CLI Command for the Alpha Terminal

**Objective:** Add `kronosbot alpha` CLI command to start the dashboard.

**Files:**
- Modify: `kronosbot/cli.py`
- Test: `tests/kronosbot/test_cli.py` (add one test for the new command)

**Step 1: Add failing test**

```python
def test_alpha_cli_command_exists():
    from kronosbot.cli import cli
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, ["alpha", "--help"])
    assert result.exit_code == 0
    assert "alpha" in result.output.lower()
```

Run: `pytest tests/kronosbot/test_cli.py::test_alpha_cli_command_exists -v`
Expected: FAIL — command not registered

**Step 2: Add command to `kronosbot/cli.py`**

Insert before `if __name__ == "__main__":`:

```python
@cli.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind to.")
@click.option("--port", default=8052, help="Port to run the Alpha Terminal on.", type=int)
@click.option("--debug/--no-debug", default=False, help="Run Flask in debug mode.")
def alpha(host, port, debug):
    """Start the Kronos Alpha Terminal."""
    from kronosbot.alpha.app import app

    click.echo(f"Starting Kronos Alpha Terminal at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
```

Run: `pytest tests/kronosbot/test_cli.py::test_alpha_cli_command_exists -v`
Expected: PASS

**Step 3: Commit**

```bash
git add kronosbot/cli.py tests/kronosbot/test_cli.py
git commit -m "feat(cli): add kronosbot alpha command to launch Alpha Terminal"
```

---

## Task 6: Run EURUSD 2024 Backtest and Validate Profitability

**Objective:** Run the walk-forward backtest with real Kronos forecasts on EURUSD 2024 and report metrics.

**Step 1: Run test command**

```bash
cd /home/bruce/code/kronos
source .venv/bin/activate
python -c "
from kronosbot.data.feed import DataFeed
from kronosbot.alpha.forecast import ForecastEngine
from kronosbot.alpha.backtest import WalkForwardBacktest
from pathlib import Path

feed = DataFeed(cache_dir=Path('data/cache'))
df = feed.load('EURUSD', start='2024-01-01', end='2025-01-01')
engine = ForecastEngine.from_pretrained(device='cpu', max_context=128)
bt = WalkForwardBacktest(
    symbol='EURUSD',
    data=df,
    forecast_fn=lambda s, d, date: engine.forecast_next_day(s, d),
    min_return_threshold=0.001,
    forecast_context_bars=60,
)
result = bt.run(account_equity=10_000)
print('Return:', result['total_return_pct'])
print('Sharpe:', result['sharpe_ratio'])
print('Max Drawdown:', result['max_drawdown_pct'])
print('Trades:', result['trades_count'])
print('Win Rate:', result['win_rate_pct'])
"
```

Expected: prints real metrics. If Return is positive and Sharpe > 0.5, we consider the MVP promising. If not, we stop and redesign the signal.

**Step 2: Add the result as a comment in the plan or commit a markdown note**

Create `docs/alpha_terminal_mvp_results.md` if results are worth keeping.

**Step 3: Commit**

```bash
git add docs/alpha_terminal_mvp_results.md

git commit -m "docs(alpha): record EURUSD 2024 walk-forward backtest metrics"
```

---

## Task 7: Run Full Test Suite for New Alpha Module

**Objective:** Ensure all new tests pass and the old test failures are isolated to the deprecated NNFX branch.

**Step 1: Run tests**

```bash
cd /home/bruce/code/kronos
source .venv/bin/activate
python -m pytest tests/kronosbot/alpha/ -q
```

Expected: all new tests pass.

**Step 2: Run the full suite (optional, to confirm old failures are unchanged)**

```bash
python -m pytest tests/kronosbot/ -q
```

Expected: 10 old NNFX tests fail as before; 63 total before + new alpha tests pass. If old failures break the new module, fix only what blocks the new alpha tests.

**Step 3: Commit any test fixes**

---

## Task 8: Update README and Architecture Doc

**Objective:** Document the Alpha Terminal in the project README and docs.

**Files:**
- Modify: `README.md`
- Modify: `docs/kronosbot.md`

**Step 1: Add Alpha Terminal section to `README.md`**

After the existing Kronos Bot section, add:

```markdown
## Kronos Alpha Terminal (MVP)

A model-first web dashboard for generating daily FX trading signals from Kronos forecasts.

```bash
kronosbot alpha
```

Open `http://localhost:8052` and select a symbol to see:
- Today's forecasted close and expected return
- Directional signal (LONG/SHORT/FLAT)
- Volatility-based position size and stop loss
- Walk-forward backtest metrics (Return, Sharpe, Max Drawdown, Trades, Win Rate)
```

**Step 2: Update `docs/kronosbot.md`**

Add an architecture section describing the `alpha/` module, the walk-forward backtest, and the dashboard.

**Step 3: Commit**

```bash
git add README.md docs/kronosbot.md
git commit -m "docs(alpha): add Alpha Terminal MVP documentation"
```

---

## Task 9: Push to GitHub (User Approval Required)

**Objective:** Push the clean MVP branch once the user is happy with the metrics.

**Step 1: Ensure the working tree is clean and committed**

```bash
git status
```

Expected: no uncommitted changes.

**Step 2: Push**

```bash
eval $(grep '^GITHUB_PERSONAL_ACCESS_TOKEN=' /home/bruce/.hermes/.env)
cd /home/bruce/code/kronos
git push https://${GITHUB_PERSONAL_ACCESS_TOKEN}@github.com/lslwoodley/Kronos.git master
```

Expected: push succeeds.

---

## Risks & Tradeoffs

| Risk | Mitigation |
|------|------------|
| Kronos forecast is too noisy for daily FX | Use min_return_threshold and ATR-based position sizing; if backtest Sharpe < 0.5, we pivot to multi-day horizon or ensemble |
| Model loading is slow and memory-heavy | Load once at app startup; use `max_context=128` for faster inference during testing |
| Walk-forward backtest may be too optimistic | Hold 1 day, trade next-day open; no intraday improvement; use conservative risk |
| yfinance data quality for FX | Validate against known broker data later; for MVP use free data |
| Single-symbol, single-timeframe | MVP scope; multi-asset/portfolio layer is Phase 2 |

## Open Questions

1. Should we add TimesFM as a secondary forecast to ensemble with Kronos? (Out of scope for MVP, but easy to add later.)
2. Should the signal include a confidence score from the Kronos forecast distribution? (Kronos currently returns a point forecast; quantile/spread is not exposed in the public API.)
3. What is the minimum acceptable Sharpe/Return for the user to consider the next phase? (To be decided after Task 6.)

## Success Criteria

- [ ] `kronosbot alpha` launches the dashboard
- [ ] Dashboard shows a signal for EURUSD, GBPUSD, USDJPY
- [ ] Walk-forward backtest runs on EURUSD 2024 and reports metrics
- [ ] All `tests/kronosbot/alpha/` tests pass
- [ ] README and docs are updated
- [ ] User reviews the metrics and decides whether to proceed to live paper trading
