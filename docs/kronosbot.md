# Kronos Bot — Forex Paper Trading & Backtesting System

A minimal, modular trading bot built on the [Kronos foundation model](https://github.com/shiyu-coder/Kronos). It adds a classic quant layer (trend, breakout, volume, ATR) and runs both historical backtests and paper simulations.

## Features

- **Backtesting.py integration** — event-driven, commission/spread-aware backtests with Bokeh HTML reports.
- **SignalEngine** — trend (`SMA200`), 20-bar breakout, volume spike, ATR, and optional Kronos forecast return.
- **PaperBroker** — forex mini-lot simulator with spread, commission, position tracking, and PnL.
- **SQLite Journal** — persistent store for bars, signals, backtest results, orders, fills, and equity.
- **Flask Web UI** — dark-themed dashboard, symbol management, backtest forms, journal viewer.
- **CLI** — `kronosbot backtest` and `kronosbot paper` for headless operation.
- **IBKR stub** — ready for future Interactive Brokers integration.

## Quick Start

```bash
# Install in editable mode
cd /home/bruce/code/kronos
pip install -e .

# Run the test suite
pytest tests/kronosbot/ -v

# Run a backtest from the CLI
kronosbot backtest EURUSD --start 2024-01-01 --end 2025-01-01 --output results/

# Start the web UI
python -m kronosbot.webui.app
# open http://localhost:8051
```

## Project Structure

```
kronosbot/
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

## Strategy Rules

Long entry requires:
1. Price above 200-period SMA (`trend = 1`).
2. Price above the highest close of the previous 19 bars (`breakout = 1`).
3. Volume >= 1.5× trailing 20-period average (`volume_spike = 1`).
4. Forecast return is positive (`forecast_return > 0`).

Exit uses a 1.5× ATR trailing stop. Size defaults to 10,000 unit mini lots; backtests run with `margin=1/50` (50:1 leverage) to match typical forex leverage.

## Roadmap

- [x] DataFeed + SignalEngine
- [x] Backtesting.py strategy + runner
- [x] PaperBroker + SQLite journal
- [x] Flask Web UI + CLI
- [ ] IBKR live broker integration
- [ ] Kronos model forecast wiring (currently neutral fallback)
- [ ] Multi-symbol portfolio allocation

## License

This project follows the license of the original Kronos project. See [LICENSE](./LICENSE).

## Disclaimer

This is research software. It is **not financial advice**. Trading involves risk; past performance does not guarantee future results.
