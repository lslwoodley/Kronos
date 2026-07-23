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
@click.option("--model-cache-dir", default=str(DEFAULT_CACHE_DIR.parent / "hf_cache"), help="HuggingFace model cache directory.")
@click.option("--no-model", is_flag=True, help="Skip loading the Kronos model (legacy behaviour, no forecast).")
def backtest(symbol, start, end, cash, commission, output, db_path, cache_dir, model_cache_dir, no_model):
    """Run a historical backtest for SYMBOL."""
    from kronosbot.model_loader import load_kronos_model
    from kronosbot.strategy.runner import BacktestRunner
    from kronosbot.strategy.kronos_strategy import KronosStrategy
    from kronosbot.journal.store import Journal

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    if no_model:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_kronos_model(device="cpu", cache_dir=_resolve_path(model_cache_dir))
    engine = SignalEngine(model=model, tokenizer=tokenizer)
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
@click.option("--model-cache-dir", default=str(DEFAULT_CACHE_DIR.parent / "hf_cache"), help="HuggingFace model cache directory.")
@click.option("--no-model", is_flag=True, help="Skip loading the Kronos model (legacy behaviour, no forecast).")
def paper(symbol, start, end, cash, spread_pips, commission, trade_units, db_path, cache_dir, model_cache_dir, no_model):
    """Run a paper trading simulation for SYMBOL from START to END."""
    from kronosbot.broker.paper import PaperBroker
    from kronosbot.journal.store import Journal
    from kronosbot.model_loader import load_kronos_model

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    df = feed.load(symbol, start=start, end=end)
    if no_model:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_kronos_model(device="cpu", cache_dir=_resolve_path(model_cache_dir))
    engine = SignalEngine(model=model, tokenizer=tokenizer)
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
        signal = row.get("signal", row.get("entry_signal", 0))
        position = broker.position()["side"]

        if signal == 1 and position != "LONG":
            if position != "FLAT":
                close_fill = broker.close_position(price=row["close"], timestamp=ts)
                close_id = journal.log_order(symbol, close_fill["side"], trade_units, "MARKET", ts)
                journal.log_fill(close_id, close_fill["fill_price"], ts, commission=close_fill["commission"])
            fill = broker.market_order("BUY", price=row["close"], timestamp=ts)
            order_id = journal.log_order(symbol, "BUY", trade_units, "MARKET", ts)
            journal.log_fill(order_id, fill["fill_price"], ts, commission=fill["commission"])
        elif signal == -1 and position != "SHORT":
            if position != "FLAT":
                close_fill = broker.close_position(price=row["close"], timestamp=ts)
                close_id = journal.log_order(symbol, close_fill["side"], trade_units, "MARKET", ts)
                journal.log_fill(close_id, close_fill["fill_price"], ts, commission=close_fill["commission"])
            fill = broker.market_order("SELL", price=row["close"], timestamp=ts)
            order_id = journal.log_order(symbol, "SELL", trade_units, "MARKET", ts)
            journal.log_fill(order_id, fill["fill_price"], ts, commission=fill["commission"])
        elif signal == 0 and position != "FLAT":
            fill = broker.close_position(price=row["close"], timestamp=ts)
            order_id = journal.log_order(symbol, fill["side"], trade_units, "MARKET", ts)
            journal.log_fill(order_id, fill["fill_price"], ts, commission=fill["commission"])

        journal.log_equity(ts, cash=broker.equity(), equity=broker.equity() + broker.unrealized_pnl(row["close"]))

    click.echo(f"Paper run complete. Final equity: ${broker.equity():.2f}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host interface to bind the web UI to.")
@click.option("--port", default=8051, help="Port to run the web UI on.", type=int)
@click.option("--debug/--no-debug", default=False, help="Run Flask in debug mode.")
def webui(host, port, debug):
    """Start the Kronos Bot web UI."""
    from kronosbot.webui.app import app

    click.echo(f"Starting Kronos Bot web UI at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


@cli.command()
@click.argument("symbol")
@click.option("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD).")
@click.option("--end", default="2025-01-01", help="Backtest end date (YYYY-MM-DD).")
@click.option("--cash", default=DEFAULT_CASH, help="Starting cash.")
@click.option("--threshold", default=0.001, help="Minimum expected return threshold to trade.")
@click.option("--model-cache-dir", default=str(DEFAULT_CACHE_DIR.parent / "hf_cache"), help="HuggingFace model cache directory.")
@click.option("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="CSV cache directory.")
@click.option("--output", default=str(DEFAULT_RESULTS_DIR / "alpha"), help="Directory to save results JSON.")
@click.option("--variants", default="all", help="Comma-separated variant names or 'all'.")
def alpha_experiment(symbol, start, end, cash, threshold, model_cache_dir, cache_dir, output, variants):
    """Run the Kronos Alpha Terminal quant-layer experiment for SYMBOL."""
    from pathlib import Path as _Path
    import json

    from kronosbot.alpha.forecast import ForecastEngine
    from kronosbot.alpha.experiment import run_variants, baseline_factory
    from kronosbot.alpha.variants import (
        volatility_targeting_factory,
        regime_filter_factory,
        arima_residual_ensemble_factory,
        slippage_spread_factory,
    )

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    df = feed.load(symbol, start=start, end=end)
    engine = ForecastEngine.from_pretrained(device="cpu")
    forecast_fn = lambda s, d, date: engine.forecast_next_day(s, d)

    all_variants = [
        ("vol_target", volatility_targeting_factory(min_return_threshold=threshold)),
        ("regime_filter", regime_filter_factory(min_return_threshold=threshold)),
        ("arima_ensemble", arima_residual_ensemble_factory(min_return_threshold=threshold, arima_weight=0.3)),
        ("slippage_spread", slippage_spread_factory(min_return_threshold=threshold)),
    ]
    if variants.lower() == "all":
        selected = all_variants
    else:
        names = set(variants.split(","))
        selected = [(n, f) for n, f in all_variants if n in names]

    comparison = run_variants(
        symbol=symbol,
        data=df,
        forecast_fn=forecast_fn,
        baseline_factory=baseline_factory,
        variant_factories=selected,
        account_equity=cash,
    )

    click.echo("Kronos Alpha Terminal Quant Experiment")
    click.echo(f"Symbol: {symbol}  Start: {start}  End: {end}")
    click.echo(comparison.to_string(index=False))

    out_dir = _Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_alpha_experiment_{start}_{end}.csv"
    comparison.to_csv(out_file, index=False)
    click.echo(f"Saved: {out_file}")


@cli.command()
@click.argument("symbol")
@click.option("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD).")
@click.option("--end", default="2025-01-01", help="Backtest end date (YYYY-MM-DD).")
@click.option("--cash", default=DEFAULT_CASH, help="Starting cash.")
@click.option("--threshold", default=0.001, help="Minimum expected return threshold to trade.")
@click.option("--model-cache-dir", default=str(DEFAULT_CACHE_DIR.parent / "hf_cache"), help="HuggingFace model cache directory.")
@click.option("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="CSV cache directory.")
@click.option("--output", default=str(DEFAULT_RESULTS_DIR / "alpha"), help="Directory to save results JSON.")
def alpha(symbol, start, end, cash, threshold, model_cache_dir, cache_dir, output):
    """Run the Kronos Alpha Terminal forecast + walk-forward backtest for SYMBOL."""
    from pathlib import Path as _Path
    import json

    from kronosbot.alpha.forecast import ForecastEngine
    from kronosbot.alpha.backtest import WalkForwardBacktest

    feed = DataFeed(cache_dir=_resolve_path(cache_dir))
    df = feed.load(symbol, start=start, end=end)
    engine = ForecastEngine.from_pretrained(device="cpu")

    backtest = WalkForwardBacktest(
        symbol=symbol,
        data=df,
        forecast_fn=lambda s, d, date: engine.forecast_next_day(s, d),
        min_return_threshold=threshold,
    )
    result = backtest.run(account_equity=cash)

    click.echo("Kronos Alpha Terminal Results")
    click.echo(f"Symbol: {result['symbol']}")
    click.echo(f"Return: {result['total_return_pct']:.2f}%")
    click.echo(f"Sharpe: {result['sharpe_ratio']:.2f}")
    click.echo(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    click.echo(f"Trades: {result['trades_count']}")
    click.echo(f"Win Rate: {result['win_rate_pct']:.1f}%")

    out_dir = _Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_alpha_{start}_{end}.json"
    with open(out_file, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    click.echo(f"Saved: {out_file}")


if __name__ == "__main__":
    cli()
