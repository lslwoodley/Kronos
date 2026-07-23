"""Simple walk-forward backtest: forecast today, trade tomorrow open, close next open."""
from dataclasses import dataclass
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

        max_dd = 0.0
        if equity_curve:
            equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
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
