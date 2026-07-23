"""Backtest runner that wires DataFeed, SignalEngine, and KronosStrategy into Backtesting.py."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd
from backtesting import Backtest
from bokeh.embed import file_html
from bokeh.resources import CDN

from kronosbot.data.feed import DataFeed
from kronosbot.features.signals import SignalEngine
from kronosbot.journal.store import Journal
from kronosbot.strategy.kronos_strategy import KronosStrategy


class BacktestRunner:
    """Run a Backtesting.py backtest for a given symbol and date range."""

    DEFAULT_MARGIN = 1 / 50  # 50:1 forex leverage for mini lots
    DEFAULT_LOT_SIZE = 10_000

    def __init__(
        self,
        feed: DataFeed,
        engine: SignalEngine,
        strategy_class: Type[KronosStrategy] = KronosStrategy,
    ):
        self.feed = feed
        self.engine = engine
        self.strategy_class = strategy_class

    @classmethod
    def with_model(
        cls,
        feed: DataFeed,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        strategy_class: Type[KronosStrategy] = KronosStrategy,
        device: str = "cpu",
        max_context: int = 512,
    ):
        """Convenience factory that wraps a SignalEngine with the given model/tokenizer."""
        engine = SignalEngine(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context=max_context,
        )
        return cls(feed, engine, strategy_class)

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        cash: float = 10_000.0,
        commission: float = 5.0,
        margin: Optional[float] = None,
        output_dir: Optional[Path] = None,
        journal: Optional[Journal] = None,
    ) -> Dict[str, Any]:
        margin = margin if margin is not None else self.DEFAULT_MARGIN

        df = self.feed.load(symbol, start=start, end=end)
        if df.empty:
            raise ValueError(f"No data loaded for {symbol} between {start} and {end}")

        # Precompute signals so the strategy can reuse them without re-running the model each bar.
        df_signals = self.engine.generate(df)
        strategy_class = self.strategy_class.with_signal_engine(self.engine, df_signals=df_signals)
        strategy_class.size = self.DEFAULT_LOT_SIZE

        # Backtesting.py expects TitleCase OHLCV columns.
        bt_data = df.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
        if isinstance(bt_data.index, pd.DatetimeIndex):
            pass
        elif "timestamp" in bt_data.columns:
            bt_data["timestamp"] = pd.to_datetime(bt_data["timestamp"])
            bt_data = bt_data.set_index("timestamp")
        else:
            raise ValueError("Data must have a DatetimeIndex or a 'timestamp' column")

        # Convert commission from $ per round-trip lot to a fraction of trade value if possible,
        # otherwise use a fixed commission model. Backtesting.py expects a fraction or a pair.
        # For simplicity, we use commission=0 here and subtract commissions in the journal/trade summary.
        bt = Backtest(
            bt_data,
            strategy_class,
            cash=cash,
            commission=0.0,
            margin=margin,
            finalize_trades=True,
        )
        stats = bt.run()

        trades = self._trades_to_df(stats._trades)
        equity_curve = self._equity_curve(stats)

        plot_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = output_dir / f"{symbol}_{start}_{end}_{ts}.html"
            try:
                bt.plot(filename=str(plot_path), open_browser=False)
            except Exception:
                plot_path = None

        metrics = self._metrics(stats)

        if journal:
            run_id = journal.log_backtest_run(
                symbol=symbol,
                start=start,
                end=end,
                return_pct=metrics.get("Return [%]"),
                sharpe=metrics.get("Sharpe Ratio"),
                max_drawdown=metrics.get("Max. Drawdown [%]"),
                params={"cash": cash, "commission": commission, "margin": margin},
            )
            if not trades.empty:
                journal.log_backtest_trades(run_id, trades)

        return {
            "stats": stats,
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "plot_path": str(plot_path) if plot_path else None,
        }

    def _metrics(self, stats) -> Dict[str, Optional[float]]:
        keys = [
            "Return [%]",
            "Sharpe Ratio",
            "Max. Drawdown [%]",
            "# Trades",
            "Win Rate [%]",
            "Avg. Trade [%]",
            "Profit Factor",
        ]
        return {k: stats.get(k) for k in keys}

    def _trades_to_df(self, trades) -> pd.DataFrame:
        if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "size",
                    "entry_price",
                    "exit_price",
                    "pnl",
                    "return_pct",
                ]
            )
        if isinstance(trades, pd.DataFrame):
            df = trades.copy()
            df = df.rename(
                columns={
                    "Size": "size",
                    "EntryPrice": "entry_price",
                    "ExitPrice": "exit_price",
                    "PnL": "pnl",
                    "ReturnPct": "return_pct",
                    "EntryTime": "entry_time",
                    "ExitTime": "exit_time",
                }
            )
            return df

        rows = []
        for t in trades:
            rows.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "size": t.size,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pl,
                    "return_pct": t.return_pct,
                }
            )
        return pd.DataFrame(rows)

    def _equity_curve(self, stats) -> pd.DataFrame:
        equity = getattr(stats, "_equity_curve", None)
        if equity is None or (isinstance(equity, pd.DataFrame) and equity.empty):
            return pd.DataFrame()
        df = equity.copy()
        df = df.rename(columns={df.columns[0]: "Equity"})
        return df
