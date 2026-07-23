"""Quant experiment harness for Kronos Alpha Terminal.

Provides a comparison harness to evaluate quant variants against a Kronos-only baseline.
Each variant is a factory that produces a configured ``WalkForwardBacktest``; the harness runs them
and returns a comparison table of risk-adjusted metrics.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from kronosbot.alpha.backtest import WalkForwardBacktest


class VariantFactory(Protocol):
    """Protocol for a variant builder.

    A variant factory takes a symbol, a DataFrame, and a forecast function and returns a
    ``WalkForwardBacktest`` that applies the variant's quant logic.
    """

    def __call__(
        self,
        symbol: str,
        data: pd.DataFrame,
        forecast_fn: Callable,
    ) -> WalkForwardBacktest: ...


@dataclass
class VariantResult:
    """Result container for one variant on one symbol."""

    variant_name: str
    symbol: str
    metrics: Dict
    summary: Dict = field(default_factory=dict)

    def to_row(self) -> Dict:
        """Flatten metrics into a single comparison row."""
        row = {
            "variant": self.variant_name,
            "symbol": self.symbol,
            "return_pct": round(self.metrics.get("total_return_pct", 0.0), 2),
            "sharpe": round(self.metrics.get("sharpe_ratio", 0.0), 2),
            "max_drawdown_pct": round(self.metrics.get("max_drawdown_pct", 0.0), 2),
            "trades": self.metrics.get("trades_count", 0),
            "win_rate_pct": round(self.metrics.get("win_rate_pct", 0.0), 1),
            "calmar": round(self._calmar(), 2),
        }
        row.update(self.summary)
        return row

    def _calmar(self) -> float:
        ret = self.metrics.get("total_return_pct", 0.0)
        dd = self.metrics.get("max_drawdown_pct", 0.0)
        if dd == 0.0:
            return 0.0
        return ret / abs(dd)


@dataclass
class Experiment:
    """Comparison harness that runs a baseline and a list of quant variants."""

    symbol: str
    data: pd.DataFrame
    forecast_fn: Callable
    baseline_factory: VariantFactory
    variant_factories: List[Tuple[str, VariantFactory]]
    account_equity: float = 10_000.0

    def run(
        self,
    ) -> Tuple[Dict, List[VariantResult]]:
        """Run the baseline and all variants for the symbol and return results.

        Returns:
            Tuple of (baseline metrics, list of VariantResult for every variant).
        """
        baseline = self.baseline_factory(self.symbol, self.data, self.forecast_fn)
        baseline_metrics = baseline.run(account_equity=self.account_equity)
        baseline_results = [
            VariantResult(
                variant_name="baseline",
                symbol=self.symbol,
                metrics=baseline_metrics,
                summary={"promoted": False},
            )
        ]
        variant_results: List[VariantResult] = []
        for name, factory in self.variant_factories:
            backtest = factory(self.symbol, self.data, self.forecast_fn)
            metrics = backtest.run(account_equity=self.account_equity)
            promoted = self._promote(baseline_metrics, metrics)
            variant_results.append(
                VariantResult(
                    variant_name=name,
                    symbol=self.symbol,
                    metrics=metrics,
                    summary={"promoted": promoted},
                )
            )
        return baseline_metrics, baseline_results + variant_results

    def _promote(self, baseline: Dict, candidate: Dict) -> bool:
        """Success rule: positive return, improved Sharpe or Calmar, and drawdown not doubled."""
        baseline_sharpe = baseline.get("sharpe_ratio", 0.0)
        candidate_sharpe = candidate.get("sharpe_ratio", 0.0)
        baseline_dd = abs(baseline.get("max_drawdown_pct", 0.0))
        candidate_dd = abs(candidate.get("max_drawdown_pct", 0.0))
        candidate_return = candidate.get("total_return_pct", 0.0)
        baseline_return = baseline.get("total_return_pct", 0.0)
        baseline_calmar = baseline_return / baseline_dd if baseline_dd else 0.0
        candidate_calmar = candidate_return / candidate_dd if candidate_dd else 0.0

        positive_return = candidate_return > 0.0
        improved = candidate_sharpe > baseline_sharpe or candidate_calmar > baseline_calmar
        dd_ok = candidate_dd <= baseline_dd * 2.0 if baseline_dd > 0 else candidate_dd <= 5.0
        return positive_return and improved and dd_ok


def run_variants(
    symbol: str,
    data: pd.DataFrame,
    forecast_fn: Callable,
    baseline_factory: VariantFactory,
    variant_factories: List[Tuple[str, VariantFactory]],
    account_equity: float = 10_000.0,
) -> pd.DataFrame:
    """High-level helper that runs an experiment and returns a comparison DataFrame.

    Args:
        symbol: market symbol, e.g. ``EURUSD``.
        data: OHLCV DataFrame with a ``timestamp`` column.
        forecast_fn: function ``(symbol, df, target_date) -> pd.Series``.
        baseline_factory: baseline backtest builder.
        variant_factories: list of ``(variant_name, builder)`` tuples.
        account_equity: starting equity.

    Returns:
        DataFrame of comparison rows (one per variant + baseline).
    """
    experiment = Experiment(
        symbol=symbol,
        data=data,
        forecast_fn=forecast_fn,
        baseline_factory=baseline_factory,
        variant_factories=variant_factories,
        account_equity=account_equity,
    )
    _, results = experiment.run()
    rows = [r.to_row() for r in results]
    return pd.DataFrame(rows)


def baseline_factory(
    symbol: str,
    data: pd.DataFrame,
    forecast_fn: Callable,
    min_return_threshold: float = 0.001,
    risk_per_trade: float = 0.01,
    forecast_context_bars: int = 60,
) -> WalkForwardBacktest:
    """Default Kronos-only baseline backtest factory."""
    return WalkForwardBacktest(
        symbol=symbol,
        data=data,
        forecast_fn=forecast_fn,
        min_return_threshold=min_return_threshold,
        risk_per_trade=risk_per_trade,
        forecast_context_bars=forecast_context_bars,
    )
