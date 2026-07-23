# Kronos Bot — Forex Paper Trading & Backtesting System

A minimal, modular trading bot built on the [Kronos foundation model](https://github.com/shiyu-coder/Kronos).

It now includes two modes:

1. **Classic Backtest Engine** — a Backtesting.py strategy combining trend/breakout/volume rules with an optional Kronos forecast.
2. **Kronos Alpha Terminal (MVP)** — a model-first daily signal system with a walk-forward backtest and a single-page web dashboard.

## Features

- **Backtesting.py integration** — event-driven, commission/spread-aware backtests with Bokeh HTML reports.
- **SignalEngine** — trend (`SMA200`), 20-bar breakout, volume spike, ATR, and optional Kronos forecast return.
- **AlphaSignal** — model-first daily signal, volatility-scaled position sizing, and ATR-based stop.
- **WalkForwardBacktest** — event-based backtest that forecasts each day and trades the next open.
- **PaperBroker** — forex mini-lot simulator with spread, commission, position tracking, and PnL.
- **SQLite Journal** — persistent store for bars, signals, backtest results, orders, fills, and equity.
- **Flask Web UI** — dark-themed management dashboard at `http://localhost:8051`.
- **Alpha Terminal** — single-page Kronos model-driven dashboard at `http://localhost:8052`.
- **CLI** — `kronosbot backtest`, `kronosbot paper`, and `kronosbot alpha`.
- **IBKR stub** — ready for future Interactive Brokers integration.

## Quick Start

```bash
cd /home/bruce/code/kronos
pip install -e .

# Run the test suite (new alpha tests + old engine tests)
pytest tests/kronosbot/ -v

# Run the Alpha Terminal backtest for EURUSD 2024
kronosbot alpha EURUSD --start 2024-01-01 --end 2025-01-01

# Launch the Alpha Terminal dashboard
kronosbot alpha --host 0.0.0.0 --port 8052

# Run the classic backtest engine
kronosbot backtest EURUSD --start 2024-01-01 --end 2025-01-01 --output results/

# Start the original web UI
python -m kronosbot.webui.app
# open http://localhost:8051
```

## Project Structure

```
kronosbot/
├── alpha/           # NEW: Alpha Terminal MVP
│   ├── signal.py
│   ├── forecast.py
│   ├── backtest.py
│   ├── app.py
│   └── templates/alpha_dashboard.html
├── broker/          # Abstract, Paper, and IBKR brokers
├── data/            # DataFeed with yfinance/CSV cache
├── features/        # SignalEngine rules
├── journal/         # SQLite journal store
├── strategy/        # Backtesting.py strategy + runner
├── webui/           # Flask management UI
└── cli.py           # Click CLI
```

## Configuration

Settings are controlled via environment variables or CLI flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `KRONOSBOT_CACHE_DIR` | `data/cache` | CSV cache directory |
| `KRONOSBOT_DB_PATH` | `data/kronosbot.db` | SQLite journal path |
| `KRONOSBOT_RESULTS_DIR` | `results` | Backtest plot output |
| `KRONOSBOT_SECRET` | `kronosbot-dev-secret` | Flask secret key |

## Strategy Rules (Classic Engine)

Long entry requires:
1. Price above 200-period SMA (`trend = 1`).
2. Price above the highest close of the previous 19 bars (`breakout = 1`).
3. Volume >= 1.5× trailing 20-period average (`volume_spike = 1`).
4. Forecast return is positive (`forecast_return > 0`).

Exit uses a 1.5× ATR trailing stop. Size defaults to 10,000 unit mini lots; backtests run with `margin=1/50` (50:1 leverage) to match typical forex leverage.

## Kronos Alpha Terminal (MVP)

The Alpha Terminal treats the Kronos foundation model as the primary alpha source:

1. Load historical daily OHLCV bars for the selected symbol.
2. Generate a one-day-ahead Kronos point forecast for the close price.
3. Convert the forecasted return into a directional signal (`LONG` / `SHORT` / `FLAT`).
4. Size the position so 1% of account equity is risked over a 2× ATR stop distance.
5. Run a walk-forward backtest: forecast at time `t`, enter at `t+1` open, exit at `t+2` open.
6. Report Return, Sharpe ratio, Max Drawdown, trade count, and win rate.

The web dashboard renders the latest signal and the full backtest metrics.

### Alpha CLI

```bash
kronosbot alpha EURUSD --start 2024-01-01 --end 2025-01-01 --cash 10000 --threshold 0.001
```

### Alpha Dashboard

```bash
kronosbot alpha --host 0.0.0.0 --port 8052
```

Open `http://localhost:8052` (or `http://<tailscale-ip>:8052` from mobile).

## Roadmap

- [x] DataFeed + SignalEngine
- [x] Backtesting.py strategy + runner
- [x] PaperBroker + SQLite journal
- [x] Flask Web UI + CLI
- [x] Kronos model wiring for forecasts
- [x] Kronos Alpha Terminal MVP
- [ ] Validate Alpha Terminal profitability on 2024 EURUSD
- [ ] IBKR live broker integration
- [ ] Multi-symbol portfolio allocation

## License

This project follows the license of the original Kronos project. See [LICENSE](./LICENSE).

## Disclaimer

This is research software. It is **not financial advice**. Trading involves risk; past performance does not guarantee future results.
