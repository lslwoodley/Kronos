# Kronos Bot v0 — Forex Paper Trading System (Backtesting.py + WebUI)

> **For Hermes/Pi:** Use TDD and subagent-driven-development to implement this plan task-by-task. All code goes in `/home/bruce/code/kronos/`.

**Goal:** Turn the Kronos forecasting model into a live-quality, backtested paper-trading bot for forex (EURUSD daily), using the **Backtesting.py** engine for robust event-driven backtests and a polished **Flask management web UI** for strategies, symbols, results, and paper-trading control.

**Architecture:**
- **Data layer:** `kronosbot/data/feed.py` fetches daily forex bars from yfinance / CSV cache. Supports EURUSD, GBPUSD, USDJPY.
- **Signal layer:** `kronosbot/features/signals.py` computes 200 SMA trend, 20-day high breakout, volume/volatility confirmation, ATR(14), and Kronos forecast direction.
- **Backtest layer:** `kronosbot/strategy/kronos_strategy.py` subclasses `backtesting.Strategy`. It exposes Kronos predictions as a custom indicator and trades long/flat on confirmed signals.
- **Backtest runner:** `kronosbot/strategy/runner.py` runs `Backtest.optimize` or `Backtest.run`, captures metrics, equity curve, and trade list.
- **Paper trading layer:** `kronosbot/broker/paper.py` simulates market orders, spread, commission, and rollover. Used for walk-forward / paper runs on new bars.
- **Journal layer:** `kronosbot/journal/store.py` SQLite store for bars, signals, backtest results, orders, fills, and equity snapshots.
- **Web UI:** `kronosbot/webui/` Flask app with a polished, mobile-responsive dashboard (Tailwind CSS + dark theme, inspired by Astryx/shadcn design patterns). Pages: Dashboard, Symbols, Strategies, Backtest, Results, Paper Trading, Journal, Settings.
- **CLI:** `python -m kronosbot.cli` for headless backtest and paper mode.
- **IBKR adapter:** `kronosbot/broker/ibkr.py` stub for future live broker integration.

**Tech Stack:** Python 3.11, pandas, numpy, torch, yfinance, Backtesting.py, bokeh, Flask, plotly, SQLite, pytest, click, Tailwind CSS.

**Default Trading Parameters:**
- Starting cash: $10,000
- Trade size: 10,000 units per signal (1 mini lot)
- EURUSD spread: 0.6 pips
- Commission: $5 round-trip per mini lot
- Stop-loss: 2 × ATR(14)

---

## Task 1: Scaffold Project + Dependencies

**Objective:** Create `kronosbot/` package and install `backtesting`, `bokeh`, `flask`, `plotly`, `yfinance`.

**Files:**
- Create: `kronosbot/__init__.py`, `kronosbot/data/__init__.py`, `kronosbot/data/feed.py`, `kronosbot/features/__init__.py`, `kronosbot/features/signals.py`, `kronosbot/strategy/__init__.py`, `kronosbot/strategy/kronos_strategy.py`, `kronosbot/strategy/runner.py`, `kronosbot/broker/__init__.py`, `kronosbot/broker/base.py`, `kronosbot/broker/paper.py`, `kronosbot/broker/ibkr.py`, `kronosbot/journal/__init__.py`, `kronosbot/journal/store.py`, `kronosbot/cli.py`, `kronosbot/webui/__init__.py`, `kronosbot/webui/app.py`, `kronosbot/webui/templates/`, `kronosbot/webui/static/css/`, `kronosbot/webui/static/js/`.
- Modify: `requirements.txt` to add `backtesting`, `bokeh>=3.0`, `flask`, `flask-cors`, `plotly`, `yfinance`, `click`.
- Create: `tests/kronosbot/` package with test files.

**Steps:**
1. Create empty module tree.
2. Install dependencies in existing `.venv`.
3. Verify `python -c "import kronosbot; print('ok')"` works.
4. Commit.

---

## Task 2: Data Feed — Historical Forex Bars

**Objective:** Implement `DataFeed` that returns daily OHLCV bars for forex pairs from yfinance with CSV cache.

**Interface:**
```python
from kronosbot.data.feed import DataFeed
feed = DataFeed(cache_dir="data/cache")
df = feed.load("EURUSD", start="2024-01-01", end="2025-01-01")
# columns: timestamp, open, high, low, close, volume
```

**Files:**
- Modify: `kronosbot/data/feed.py`
- Create: `tests/kronosbot/test_feed.py`

**TDD steps:**
1. Write failing test for load shape and columns.
2. Implement `DataFeed` with yfinance `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` mapping, multi-index flattening, CSV cache.
3. Run test; add synthetic fixture if network fails.
4. Commit.

---

## Task 3: Signal Engine — Kronos + Quant Science Rules

**Objective:** Implement `SignalEngine` that computes trend, breakout, volume/volatility confirmation, ATR, and forecast return.

**Interface:**
```python
from kronosbot.features.signals import SignalEngine
engine = SignalEngine(model=None, tokenizer=None, device="cpu")
df_signals = engine.generate(df, forecast_horizon=1)
# columns: trend, breakout, volume_spike, atr, forecast_return, entry_signal
```

**Rules:**
- `trend`: 1 if close > 200 SMA.
- `breakout`: 1 if close > max(previous 19 closes, excluding current bar).
- `volume_spike`: 1 if volume > 1.5 × 20-day average volume (or ATR spike if volume is missing/zero).
- `atr`: ATR(14).
- `forecast_return`: if model/tokenizer provided, call `KronosPredictor.predict` on last `max_context` bars and compute `(next_close - current_close) / current_close`. If model is None, return 0.0 (neutral) for unit tests.
- `entry_signal`: 1 if trend == 1 and breakout == 1 and volume_spike == 1 and forecast_return > 0, else 0.

**Files:**
- Modify: `kronosbot/features/signals.py`
- Create: `tests/kronosbot/test_signals.py`

**TDD steps:**
1. Write test with a synthetic uptrending dataset that asserts `trend`, `breakout`, and `entry_signal`.
2. Run and watch fail.
3. Implement.
4. Run and pass.
5. Commit.

---

## Task 4: Backtesting.py Strategy

**Objective:** Implement `KronosStrategy` subclass of `backtesting.Strategy` that uses the signal engine for long entries and ATR-based exits.

**Interface:**
```python
from kronosbot.strategy.kronos_strategy import KronosStrategy
```

**Behavior:**
- `init()`: precompute indicators and signals on full data (Backtesting.py convention). Cache `entry_signal` and `atr` series.
- `next()`: if no position and `entry_signal` is true today: buy `size=10000` units (or cash-based sizing) at next open.
- Exit: if position is open and either (a) `entry_signal` becomes 0, or (b) current low hits `entry_price - 2*ATR`, close at stop price. For simplicity, use next close for exit fill in v0.
- `next()` has access to `self.data.Open`, `self.data.High`, etc.

**Files:**
- Modify: `kronosbot/strategy/kronos_strategy.py`
- Create: `tests/kronosbot/test_kronos_strategy.py`

**TDD steps:**
1. Write test that runs a backtest on synthetic data and asserts stats contains `Return [%]` and a non-empty trade list.
2. Run and fail.
3. Implement minimal strategy.
4. Run and pass.
5. Commit.

---

## Task 5: Backtest Runner + Results Capture

**Objective:** Implement `BacktestRunner` that wires feed, signal engine, and Backtesting.py, returning metrics, equity curve, and trades.

**Interface:**
```python
from kronosbot.strategy.runner import BacktestRunner
runner = BacktestRunner(feed, signal_engine, strategy_class=KronosStrategy)
result = runner.run("EURUSD", start="2024-01-01", end="2025-01-01")
# result: dict with metrics, equity_curve, trades, bokeh_plot_path
```

**Files:**
- Modify: `kronosbot/strategy/runner.py`
- Create: `tests/kronosbot/test_runner.py`

**TDD steps:**
1. Write test that runs `BacktestRunner` with synthetic data and checks for `equity_curve`, `trades`, and positive/negative return.
2. Implement runner (prepares data, instantiates `Backtest`, runs, extracts results, saves Bokeh plot to `results/`).
3. Run and pass.
4. Commit.

---

## Task 6: Paper Broker / Forward Engine

**Objective:** Implement `PaperBroker` that simulates daily forex orders with spread, commission, and rollover. Used for walk-forward paper trading.

**Interface:**
```python
from kronosbot.broker.paper import PaperBroker
broker = PaperBroker(cash=10_000, spread_pips=0.6, commission_per_lot=5.0, pip_value=1.0)
broker.market_order("EURUSD", side="BUY", units=10_000, price=1.0850, timestamp=...)
```

**Files:**
- Modify: `kronosbot/broker/base.py` (abstract broker)
- Modify: `kronosbot/broker/paper.py`
- Create: `tests/kronosbot/test_broker.py`

**TDD steps:**
1. Write test for buy, then sell with profit.
2. Implement fill price = price ± half-spread, commission, position tracking, realized PnL in pips/USD, equity.
3. Pass test.
4. Commit.

---

## Task 7: Journal Store — SQLite

**Objective:** SQLite journal for bars, signals, backtest results, orders, fills, and equity snapshots.

**Tables:**
- `bars`, `signals`, `backtest_runs`, `backtest_trades`, `orders`, `fills`, `equity`.

**Files:**
- Modify: `kronosbot/journal/store.py`
- Create: `tests/kronosbot/test_journal.py`

**TDD steps:**
1. Write test for `log_backtest_run` and `read_backtest_trades`.
2. Implement schema and logging methods.
3. Pass.
4. Commit.

---

## Task 8: CLI — Backtest and Paper Mode

**Objective:** Add `kronosbot.cli` commands using `click`.

**Commands:**
```bash
python -m kronosbot.cli backtest EURUSD --start 2024-01-01 --end 2025-01-01 --output results/
python -m kronosbot.cli paper EURUSD --start 2025-01-01 --db data/kronosbot.db
```

**Files:**
- Modify: `kronosbot/cli.py`
- Create: `tests/kronosbot/test_cli.py`

**TDD steps:**
1. Write `CliRunner` test for `backtest` command.
2. Implement CLI using `BacktestRunner` and `Journal`.
3. Pass.
4. Commit.

---

## Task 9: Polished Management Web UI

**Objective:** Build a Flask web UI in `kronosbot/webui/` with a clean, mobile-responsive dark theme using Tailwind CSS and design patterns from Astryx/shadcn.

**Pages:**
1. **Dashboard** (`/`): current cash, open positions, today’s signal, latest equity, quick actions.
2. **Symbols** (`/symbols`): list supported pairs, refresh data, view cached bars.
3. **Strategies** (`/strategies`): view Kronos strategy parameters; edit threshold toggles in v0 (read-only details + parameter form).
4. **Backtest** (`/backtest`): form with symbol, start/end, run button; results page with metrics, equity curve, trade list.
5. **Paper Trading** (`/paper`): start/stop toggle, status, latest orders, equity chart.
6. **Journal** (`/journal`): browsable logs of signals, orders, fills, backtest runs.
7. **Settings** (`/settings`): cache directory, DB path, broker mode, API key placeholders (not stored in Git).

**UI Stack:**
- Flask + Jinja2 templates.
- Tailwind CSS via CDN or vendored build.
- Plotly/Bokeh for charts (equity curve, trade distribution).
- HTMX or vanilla JS for interactive tables and tabs.

**Files:**
- Create: `kronosbot/webui/app.py`, `kronosbot/webui/templates/base.html`, `kronosbot/webui/templates/dashboard.html`, `kronosbot/webui/templates/symbols.html`, `kronosbot/webui/templates/strategies.html`, `kronosbot/webui/templates/backtest.html`, `kronosbot/webui/templates/paper.html`, `kronosbot/webui/templates/journal.html`, `kronosbot/webui/templates/settings.html`, `kronosbot/webui/static/css/style.css`, `kronosbot/webui/static/js/app.js`.
- Create: `tests/kronosbot/test_webui.py` using Flask test client.

**TDD steps:**
1. Write test that each route returns 200 and contains expected page title.
2. Implement base layout and route handlers.
3. Pass.
4. Add HTMX-powered backtest run form with loading state and result display.
5. Add Plotly equity chart on results page.
6. Commit.

---

## Task 10: IBKR Broker Stub

**Objective:** Add `IBKRBroker` adapter implementing the same base interface as a future integration point.

**Files:**
- Modify: `kronosbot/broker/ibkr.py`

**Steps:**
1. Implement `IBKRBroker` with `market_order`, `close_position`, `position`, `equity`, `reset`. All methods raise `NotImplementedError` with a link to IBKR Web API setup docs.
2. Write test confirming instantiation and raising behavior.
3. Commit.

---

## Task 11: Documentation and Final Push

**Objective:** Update README and create `docs/kronosbot.md` with clear usage instructions.

**Docs sections:**
- What Kronos Bot is and isn’t.
- Installation: `pip install -r requirements.txt` and `python -m kronosbot.webui.app`.
- Running a backtest from CLI and Web UI.
- Reading results: metrics, equity curve, trade list.
- Starting paper trading.
- Strategy parameter reference.
- IBKR integration roadmap.
- Database schema overview.

**Files:**
- Modify: `README.md` (add Kronos Bot section at top of Getting Started).
- Create: `docs/kronosbot.md`.

**Steps:**
1. Write docs.
2. Run full test suite: `pytest tests/kronosbot/ -v`.
3. Commit.
4. Push to `lslwoodley/Kronos` via git MCP.

---

## Risks and Tradeoffs

- **Backtesting.py dependency:** requires `bokeh>=3.0`; may conflict with pinned `matplotlib`. Use `pip install --no-deps` or unpin matplotlib if needed.
- **yfinance forex quality:** may be patchy. Add CSV fallback and Alpha Vantage adapter if data gaps appear.
- **Kronos model weight:** forecasting daily bars with Kronos-small is possible but heavy. Tests bypass model; real runs can optionally load model.
- **Long/flat only:** shorting and leverage deferred to v1.
- **No real-time streaming:** daily re-evaluation only. Intraday requires a scheduler + websocket broker adapter.

## Open Questions

1. Should the Web UI use Tailwind CDN or a vendored build? (Default: CDN for simplicity.)
2. Should the bot run as a Docker service or cron job? (Deferred to v1.)
3. Do we want strategy optimization via Backtesting.py’s built-in optimizer? (Deferred; keep simple run first.)

---

## Verification Checklist

- [ ] `pytest tests/kronosbot/ -v` passes.
- [ ] `python -m kronosbot.cli backtest EURUSD --start 2024-01-01 --end 2025-01-01` produces a results file and prints metrics.
- [ ] `python -m kronosbot.webui.app` starts and all pages return 200.
- [ ] Backtest result page shows an equity curve chart and a trade list.
- [ ] README and `docs/kronosbot.md` are clear and complete.
- [ ] All commits pushed to `lslwoodley/Kronos`.
