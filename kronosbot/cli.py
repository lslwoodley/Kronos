import os
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kronosbot.data.feed import DataFeed
from kronosbot.features.signals import SignalEngine

DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DB_PATH = Path("data/kronosbot.db")
DEFAULT_CASH = 10_000.0


@click.group()
def cli():
    """Kronos Bot CLI for forex backtesting and paper trading."""
    pass


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


@cli.command()
@click.argument("symbol")
@click.option("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD).")
@click.option("--end", default="2025-01-01", help="Backtest end date (YYYY-MM-DD).")
@click.option("--cash", default=DEFAULT_CASH, help="Starting cash.")
@click.option("--commission", default=5.0, help="Round-trip commission per mini lot.")
@click.option("--output", default=str(DEFAULT_RESULTS_DIR), help="Directory to save results.")
@click.option("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite journal path.")
@click.option("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="CSV cache directory.")
def backtest(symbol, start, end, cash, commission, output, db_path, cache_dir):
    """Run a historical backtest for SYMBOL."""
    from kronosbot.strategy.runner import BacktestRunner
    from kronosbot.strategy.kronos_strategy import KronosStrategy
    from kronosbot.journal.store import Journal

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    engine = SignalEngine()
    runner = BacktestRunner(feed, engine, KronosStrategy)
    journal = Journal(_resolve_path(db_path))

    click.echo(f"Running backtest for {symbol} from {start} to {end}...")
    result = runner.run(
        symbol=symbol,
        start=start,
        end=end,
        cash=cash,
        commission=commission,
        output_dir=_resolve_path(output),
        journal=journal,
    )

    metrics = result.get("metrics", {})
    click.echo(f"Return: {metrics.get('Return [%]', 0):.2f}%")
    click.echo(f"Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
    click.echo(f"Max Drawdown: {metrics.get('Max. Drawdown [%]', 0):.2f}%")
    click.echo(f"Trades: {len(result.get('trades', []))}")
    click.echo(f"Plot saved to: {result.get('plot_path')}")


@cli.command()
@click.argument("symbol")
@click.option("--start", default="2025-01-01", help="Paper trading start date.")
@click.option("--end", default="2026-01-01", help="Paper trading end date.")
@click.option("--cash", default=DEFAULT_CASH, help="Starting cash.")
@click.option("--spread-pips", default=0.6, help="Spread in pips.")
@click.option("--commission", default=5.0, help="Round-trip commission per mini lot.")
@click.option("--trade-units", default=10_000, help="Units per trade.")
@click.option("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite journal path.")
@click.option("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="CSV cache directory.")
def paper(symbol, start, end, cash, spread_pips, commission, trade_units, db_path, cache_dir):
    """Run a paper trading simulation for SYMBOL from START to END."""
    from kronosbot.broker.paper import PaperBroker
    from kronosbot.journal.store import Journal

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    df = feed.load(symbol, start=start, end=end)
    engine = SignalEngine()
    signals = engine.generate(df)

    broker = PaperBroker(
        cash=cash,
        symbol=symbol,
        spread_pips=spread_pips,
        commission_per_lot=commission,
        trade_units=trade_units,
    )
    journal = Journal(_resolve_path(db_path))

    for _, row in signals.iterrows():
        ts = row["timestamp"]
        if row["entry_signal"] == 1 and broker.position()["side"] == "FLAT":
            fill = broker.market_order("BUY", price=row["close"], timestamp=ts)
            order_id = journal.log_order(symbol, "BUY", trade_units, "MARKET", ts)
            journal.log_fill(order_id, fill["fill_price"], ts, commission=fill["commission"])
        elif broker.position()["side"] == "LONG" and row["entry_signal"] == 0:
            fill = broker.close_position(price=row["close"], timestamp=ts)
            order_id = journal.log_order(symbol, "SELL", trade_units, "MARKET", ts)
            journal.log_fill(order_id, fill["fill_price"], ts, commission=fill["commission"])

        journal.log_equity(ts, cash=broker.equity(), equity=broker.equity() + broker.unrealized_pnl(row["close"]))

    click.echo(f"Paper run complete. Final equity: ${broker.equity():.2f}")


if __name__ == "__main__":
    cli()
