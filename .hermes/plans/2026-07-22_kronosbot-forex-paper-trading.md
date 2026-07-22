# Kronos Bot v0 — Forex Paper Trading System

> **For Hermes/Pi:** Use TDD and subagent-driven-development to implement this plan task-by-task. All code goes in `/home/bruce/code/kronos/`.

**Goal:** Turn the Kronos forecasting model into a live-quality paper-trading bot for forex (EURUSD daily), with a clean broker adapter so Interactive Brokers Web API can be plugged in later.

**Architecture:**
- **Data layer:** yfinance/Alpha Vantage fallback for historical daily forex bars; IBKR adapter later.
- **Forecast layer:** `KronosPredictor` generates next-day OHLC forecast.
- **Signal layer:** Combine Kronos forecast with Quant Science rules (200 SMA, 20-day high breakout, volume/volatility confirmation, ATR stop).
- **Strategy layer:** Long/flat only for v0; entry on confirmed bullish setup, exit on ATR stop or forecast reversal.
- **Broker layer:** `PaperBroker` simulates spread, fills, commission, and rollover; `IBKRBroker` adapter stub for future integration.
- **Journal layer:** SQLite store for bars, signals, orders, fills, and positions.
- **CLI:** `python -m kronosbot.cli backtest EURUSD --start 2024-01-01 --end 2025-01-01` and `paper EURUSD`.

**Tech Stack:** Python 3.11, pandas, numpy, torch, yfinance, SQLite, pytest, click.

---

## Task 1: Scaffold Project Structure

**Objective:** Create the `kronosbot/` package directory and empty modules.

**Files:**
- Create: `kronosbot/__init__.py`
- Create: `kronosbot/data/__init__.py`
- Create: `kronosbot/data/feed.py`
- Create: `kronosbot/features/__init__.py`
- Create: `kronosbot/features/signals.py`
- Create: `kronosbot/strategy/__init__.py`
- Create: `kronosbot/strategy/strategy.py`
- Create: `kronosbot/broker/__init__.py`
- Create: `kronosbot/broker/base.py`
- Create: `kronosbot/broker/paper.py`
- Create: `kronosbot/broker/ibkr.py`
- Create: `kronosbot/journal/__init__.py`
- Create: `kronosbot/journal/store.py`
- Create: `kronosbot/cli.py`
- Create: `tests/kronosbot/__init__.py`
- Create: `tests/kronosbot/test_feed.py`
- Create: `tests/kronosbot/test_signals.py`
- Create: `tests/kronosbot/test_broker.py`
- Create: `tests/kronosbot/test_strategy.py`
- Create: `tests/kronosbot/test_journal.py`
- Modify: `requirements.txt` to add `yfinance`, `click`.
- Modify: `README.md` to add a "Kronos Bot" section.

**Step 1:** Create empty module files with `__init__.py` markers.

**Step 2:** Run `python -c "import kronosbot; print('ok')"` from repo root.
Expected: `ok`.

**Step 3:** Commit.

```bash
git add kronosbot/ tests/kronosbot/ requirements.txt README.md
git commit -m "feat(kronosbot): scaffold forex paper-trading package"
```

---

## Task 2: Data Feed — Historical Forex Bars

**Objective:** Implement `kronosbot/data/feed.py` to fetch daily EURUSD bars from yfinance or a CSV fallback.

**Interface:**
```python
from kronosbot.data.feed import DataFeed

df = DataFeed.load("EURUSD", start="2024-01-01", end="2025-01-01")
# Columns: timestamp, open, high, low, close, volume (volume may be 0 for forex)
```

**Files:**
- Create: `tests/kronosbot/test_feed.py`
- Modify: `kronosbot/data/feed.py`

**Step 1: Write failing test**

```python
def test_load_returns_dataframe_with_required_columns(tmp_path):
    feed = DataFeed(cache_dir=tmp_path)
    df = feed.load("EURUSD", start="2024-01-01", end="2024-01-31")
    assert not df.empty
    for col in ["open", "high", "low", "close"]:
        assert col in df.columns
```

**Step 2:** Run `pytest tests/kronosbot/test_feed.py -v`. Expected: FAIL — `DataFeed` not defined.

**Step 3: Implement minimal `DataFeed`**

```python
import os
from pathlib import Path
import pandas as pd
import yfinance as yf

class DataFeed:
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "../../data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ticker(self, symbol: str) -> str:
        mapping = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X"}
        return mapping.get(symbol, symbol)

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        cache = self.cache_dir / f"{symbol}_{start}_{end}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        ticker = self._ticker(symbol)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # yfinance multi-index columns: flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        rename = {
            "date": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj_close": "close",
            "volume": "volume",
        }
        df = df.rename(columns=rename)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.to_csv(cache, index=False)
        return df
```

**Step 4:** Run test. Expected: PASS (if network available; otherwise add a synthetic CSV fixture).

**Step 5:** Commit.

---

## Task 3: Feature Signals — Kronos Forecast + Quant Science Rules

**Objective:** Implement `kronosbot/features/signals.py` that computes:
- 200 SMA trend filter.
- 20-day high breakout.
- Volume/volatility confirmation (volume > 150% of 20-day average, or ATR spike).
- ATR(14) stop distance.
- Kronos forecast direction (next-day close vs current close).

**Interface:**
```python
from kronosbot.features.signals import SignalEngine

engine = SignalEngine(model, tokenizer, device="cpu")
signals = engine.generate(df, forecast_horizon=1)
# Returns DataFrame with columns: trend, breakout, volume_spike, atr, forecast_return, signal
```

**Files:**
- Modify: `kronosbot/features/signals.py`
- Create: `tests/kronosbot/test_signals.py`

**Step 1: Write failing test**

```python
def test_signal_values_on_trending_breakout():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=210),
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": [1.0 + i*0.001 for i in range(210)],
        "volume": 1000,
    })
    engine = SignalEngine(None, None, device="cpu")  # Kronos optional for unit tests
    result = engine.generate(df)
    assert result.iloc[-1]["trend"] == 1  # above 200 SMA
```

**Step 2:** Run test. Expected: FAIL.

**Step 3: Implement `SignalEngine`**

- `trend`: 1 if close > 200 SMA, else 0.
- `breakout`: 1 if close > max(close[-20:-1]), else 0.
- `volume_spike`: 1 if volume[-1] > 1.5 * mean(volume[-20:-1]).
- `atr`: true range rolling mean(14).
- `forecast_return`: if model is None, return 0.01 (placeholder bullish bias). Otherwise call `KronosPredictor.predict`.
- `signal`: 1 if trend + breakout + volume_spike > 0 and forecast_return > 0, else 0.

**Step 4:** Run test. Expected: PASS.

**Step 5:** Commit.

---

## Task 4: Paper Broker — Simulate Forex Orders and Fills

**Objective:** Implement `kronosbot/broker/paper.py` that simulates market orders, spread, commission, and tracks positions/PnL in pips.

**Interface:**
```python
from kronosbot.broker.paper import PaperBroker

broker = PaperBroker(spread_pips=0.6, commission_per_lot=5.0, pip_value=10.0)
broker.market_order("EURUSD", side="BUY", units=10000, price=1.0850, timestamp=...)
```

**Files:**
- Modify: `kronosbot/broker/base.py` (abstract base)
- Modify: `kronosbot/broker/paper.py`
- Create: `tests/kronosbot/test_broker.py`

**Step 1: Write failing test**

```python
def test_paper_broker_buy_and_sell():
    broker = PaperBroker(spread_pips=1.0, pip_value=10.0)
    broker.market_order("EURUSD", side="BUY", units=10000, price=1.1000, timestamp=pd.Timestamp.now())
    pos = broker.position("EURUSD")
    assert pos["units"] == 10000
    pnl = broker.close_position("EURUSD", price=1.1010, timestamp=pd.Timestamp.now())
    assert pnl > 0
```

**Step 2:** Run test. Expected: FAIL.

**Step 3: Implement `PaperBroker`**

- `market_order`: applies half-spread slippage to fill price, records fill, opens position.
- `close_position`: flips side at fill price, computes PnL in pips and USD, records realized PnL.
- `equity`: cash + unrealized PnL.
- `reset`: clear state for tests.

**Step 4:** Run test. Expected: PASS.

**Step 5:** Commit.

---

## Task 5: Trade Journal — SQLite Store

**Objective:** Implement `kronosbot/journal/store.py` to persist bars, signals, orders, fills, and daily equity snapshots.

**Interface:**
```python
from kronosbot.journal.store import Journal

journal = Journal("data/kronosbot.db")
journal.log_bar(...)
journal.log_signal(...)
journal.log_order(...)
journal.log_fill(...)
journal.log_equity(...)
```

**Files:**
- Modify: `kronosbot/journal/store.py`
- Create: `tests/kronosbot/test_journal.py`

**Step 1: Write failing test**

```python
def test_journal_logs_and_reads_orders(tmp_path):
    journal = Journal(tmp_path / "journal.db")
    journal.log_order(symbol="EURUSD", side="BUY", units=10000, price=1.1, timestamp=pd.Timestamp.now())
    orders = journal.read_orders()
    assert len(orders) == 1
```

**Step 2:** Run test. Expected: FAIL.

**Step 3: Implement `Journal`** with tables:
- `bars`: timestamp, symbol, open, high, low, close, volume.
- `signals`: timestamp, symbol, trend, breakout, volume_spike, atr, forecast_return, signal.
- `orders`: id, timestamp, symbol, side, units, price, status.
- `fills`: id, timestamp, order_id, symbol, side, units, price, commission.
- `equity`: timestamp, cash, unrealized, realized.

**Step 4:** Run test. Expected: PASS.

**Step 5:** Commit.

---

## Task 6: Strategy Engine — Backtest Loop

**Objective:** Implement `kronosbot/strategy/strategy.py` that iterates bars, generates signals, executes via broker, and logs to journal.

**Interface:**
```python
from kronosbot.strategy.strategy import Strategy

strategy = Strategy(feed, signal_engine, broker, journal)
strategy.backtest("EURUSD", start="2024-01-01", end="2025-01-01")
```

**Files:**
- Modify: `kronosbot/strategy/strategy.py`
- Create: `tests/kronosbot/test_strategy.py`

**Step 1: Write failing test**

```python
def test_backtest_runs_and_produces_equity():
    feed = DataFeed(cache_dir=tmp_path)
    broker = PaperBroker()
    journal = Journal(tmp_path / "journal.db")
    engine = SignalEngine(None, None)
    strategy = Strategy(feed, engine, broker, journal)
    strategy.backtest("EURUSD", start="2024-01-01", end="2024-03-01")
    equity = journal.read_equity()
    assert len(equity) > 0
```

**Step 2:** Run test. Expected: FAIL.

**Step 3: Implement `Strategy.backtest`**

1. Load historical bars.
2. For each day, append to growing window.
3. When window has 200 bars, compute signals.
4. If signal == 1 and no position: BUY 1 mini lot (10,000 units) at close.
5. If position open and (price hits ATR stop or signal flips to 0): CLOSE at close.
6. Log everything.

**Step 4:** Run test. Expected: PASS.

**Step 5:** Commit.

---

## Task 7: IBKR Broker Adapter Stub

**Objective:** Add `kronosbot/broker/ibkr.py` implementing the same base interface as a future integration point.

**Files:**
- Modify: `kronosbot/broker/ibkr.py`

**Step 1:** Define `IBKRBroker` class with methods: `market_order`, `close_position`, `position`, `equity`, `reset`. All methods raise `NotImplementedError` with a clear message pointing to IBKR Web API setup docs.

**Step 2:** Write test that instantiating works but calling methods raises.

**Step 3:** Commit.

---

## Task 8: CLI and Integration Smoke Test

**Objective:** Add `kronosbot/cli.py` with `backtest` and `paper` commands.

**Interface:**
```bash
python -m kronosbot.cli backtest EURUSD --start 2024-01-01 --end 2025-01-01
python -m kronosbot.cli paper EURUSD --start 2025-01-01
```

**Files:**
- Modify: `kronosbot/cli.py`
- Modify: `requirements.txt` to add `click`.

**Step 1:** Write failing test in `tests/kronosbot/test_cli.py` using `click.testing.CliRunner`.

**Step 2:** Implement CLI.

**Step 3:** Run smoke test. Expected: PASS.

**Step 4:** Commit.

---

## Task 9: Documentation and Final Push

**Objective:** Update README with Kronos Bot usage and push to GitHub.

**Files:**
- Modify: `README.md` with a new "Kronos Bot — Paper Trading" section.
- Create: `docs/kronosbot.md` with architecture and IBKR setup notes.

**Step 1:** Add section covering:
- How to install bot dependencies.
- How to run backtest.
- How to run paper mode.
- Where trades are stored.
- How to plug in IBKR Web API.

**Step 2:** Commit and push via git MCP.

---

## Risks and Tradeoffs

- **yfinance forex data is incomplete.** Add CSV fallback and Alpha Vantage adapter if needed.
- **Kronos model is heavy for daily bars.** v0 uses model only when available; tests bypass it.
- **Long/flat only for v0.** Shorting and leverage deferred.
- **No real-time streaming.** Daily bar re-evaluation only.

## Open Questions

1. What starting cash and lot size? (Default: $10,000, 10,000 units per trade.)
2. Should we run the bot as a Docker service or cron job? (Deferred to v1.)
3. Which forex pairs to support initially? (EURUSD, GBPUSD, USDJPY.)

---

## Verification Summary

- [ ] `pytest tests/kronosbot/ -v` passes.
- [ ] `python -m kronosbot.cli backtest EURUSD --start 2024-01-01 --end 2025-01-01` runs without error and logs to `data/kronosbot.db`.
- [ ] README updated.
- [ ] All commits pushed to `lslwoodley/Kronos`.
