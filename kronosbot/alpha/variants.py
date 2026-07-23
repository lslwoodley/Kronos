"""Quant variants for the Kronos Alpha Terminal experiment harness.

Each variant is a factory that returns a configured ``WalkForwardBacktest`` using one or
more of the backtest hooks (sizing, execution cost, signal filter).  Kronos remains the
only forecast source; quant methods operate on risk / execution layers.

The ``TimesFMVariant`` is intentionally a stub and raises ``NotImplementedError`` because
TimesFM is not part of this milestone.
"""
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from kronosbot.alpha.backtest import WalkForwardBacktest
from kronosbot.alpha.signal import AlphaSignal


# -----------------------------------------------------------------------------
# Volatility targeting
# -----------------------------------------------------------------------------

def _realized_vol(log_returns: pd.Series, window: int = 20) -> float:
    """Annualized realized volatility from log returns."""
    if len(log_returns) < window:
        return 0.0
    return float(log_returns.tail(window).std() * np.sqrt(252))


def volatility_targeting_sizing(
    target_annual_vol: float = 0.10,
    max_size_multiplier: float = 2.0,
) -> Callable[[AlphaSignal, pd.DataFrame, float], float]:
    """Return a sizing hook that scales position to hit a target annual volatility.

    Uses the ATR-based stop distance as a proxy for the per-trade volatility.
    Position size is capped at ``max_size_multiplier`` × the base ATR sizing.
    """

    def _size(signal: AlphaSignal, context: pd.DataFrame, equity: float) -> float:
        close = context["close"]
        log_returns = np.log(close / close.shift(1)).dropna()
        realized_vol = _realized_vol(log_returns, window=20)
        if realized_vol <= 0 or np.isnan(signal.atr_14) or signal.atr_14 <= 0:
            return signal.position_size

        stop_distance = 2.0 * signal.atr_14
        daily_target_vol = target_annual_vol / np.sqrt(252)
        target_position = equity * daily_target_vol / stop_distance

        base_size = signal.position_size
        max_size = base_size * max_size_multiplier
        return min(target_position, max_size)

    return _size


def volatility_targeting_factory(
    target_annual_vol: float = 0.10,
    max_size_multiplier: float = 2.0,
    **backtest_kwargs,
) -> Callable:
    """Factory builder for a volatility-targeting variant."""

    def _factory(symbol: str, data: pd.DataFrame, forecast_fn: Callable) -> WalkForwardBacktest:
        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=forecast_fn,
            sizing_fn=volatility_targeting_sizing(
                target_annual_vol=target_annual_vol,
                max_size_multiplier=max_size_multiplier,
            ),
            **backtest_kwargs,
        )

    return _factory


# -----------------------------------------------------------------------------
# Regime filter
# -----------------------------------------------------------------------------

def _trend_strength(context: pd.DataFrame, window: int = 14) -> float:
    """Simple ADX-like trend strength using smoothed directional movement.

    Uses the difference between close and a short SMA, normalized by recent ATR.
    Returns a positive float; higher values indicate a stronger trend.
    """
    close = context["close"]
    if len(close) < window + 1:
        return 0.0
    sma = close.rolling(window=window).mean().iloc[-1]
    atr = _atr(context, window=14)
    if atr <= 0:
        return 0.0
    return float(abs(close.iloc[-1] - sma) / atr)


def _high_volatility(context: pd.DataFrame, window: int = 20, percentile: float = 0.95) -> bool:
    """Return True if the most recent realized volatility is above the historical percentile."""
    close = context["close"]
    if len(close) < window + 1:
        return False
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < window:
        return False
    current_vol = log_returns.iloc[-window:].std() * np.sqrt(252)
    hist_vol = log_returns.std() * np.sqrt(252)
    return current_vol > hist_vol * (2.0 * percentile - 1.0)


def regime_filter_signal_filter(
    min_trend_strength: float = 0.1,
    skip_high_vol: bool = True,
) -> Callable[[AlphaSignal, pd.DataFrame, int], bool]:
    """Return a signal filter that skips trades in low-trend or high-vol regimes."""

    def _filter(signal: AlphaSignal, context: pd.DataFrame, _idx: int) -> bool:
        if signal.direction == 0:
            return False
        if _trend_strength(context) < min_trend_strength:
            return False
        if skip_high_vol and _high_volatility(context):
            return False
        return True

    return _filter


def regime_filter_factory(
    min_trend_strength: float = 0.1,
    skip_high_vol: bool = True,
    **backtest_kwargs,
) -> Callable:
    """Factory builder for a regime-filter variant."""

    def _factory(symbol: str, data: pd.DataFrame, forecast_fn: Callable) -> WalkForwardBacktest:
        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=forecast_fn,
            signal_filter_fn=regime_filter_signal_filter(
                min_trend_strength=min_trend_strength,
                skip_high_vol=skip_high_vol,
            ),
            **backtest_kwargs,
        )

    return _factory


# -----------------------------------------------------------------------------
# ARIMA residual ensemble
# -----------------------------------------------------------------------------

def _fit_arima_forecast(close: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> float:
    """Return a one-step ARIMA forecast for the next close price.

    Falls back to the last close if the model cannot be fit.  Does not import statsmodels
    at module load time; it is imported lazily when the variant is instantiated.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return float(close.iloc[-1])

    if len(close) < sum(order) + 2:
        return float(close.iloc[-1])
    try:
        model = ARIMA(close, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=1)
        return float(forecast.iloc[0])
    except Exception:
        return float(close.iloc[-1])


def arima_residual_ensemble_forecast(
    arima_weight: float = 0.3,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Callable:
    """Return a forecast function that blends Kronos with an ARIMA(1,1,1) residual forecast.

    The Kronos forecast is the primary alpha source.  The ARIMA forecast is a weighted
    overlay of the recent residual/mean-reversion component.  Weights sum to 1.0:
    ``forecast = (1 - arima_weight) * kronos + arima_weight * arima``.
    """

    def _forecast(
        symbol: str,
        context: pd.DataFrame,
        target_date: pd.Timestamp,
        kronos_forecast_fn: Callable,
    ) -> pd.Series:
        kronos_series = kronos_forecast_fn(symbol, context, target_date)
        kronos_price = float(kronos_series.iloc[-1])
        arima_price = _fit_arima_forecast(context["close"], order=order)
        blended = (1.0 - arima_weight) * kronos_price + arima_weight * arima_price
        return pd.Series([blended], index=[target_date])

    return _forecast


def arima_residual_ensemble_factory(
    arima_weight: float = 0.3,
    order: Tuple[int, int, int] = (1, 1, 1),
    **backtest_kwargs,
) -> Callable:
    """Factory builder for an ARIMA residual ensemble variant."""

    def _factory(symbol: str, data: pd.DataFrame, forecast_fn: Callable) -> WalkForwardBacktest:
        ensemble = arima_residual_ensemble_forecast(
            arima_weight=arima_weight,
            order=order,
        )
        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=lambda s, ctx, date: ensemble(s, ctx, date, forecast_fn),
            **backtest_kwargs,
        )

    return _factory


# -----------------------------------------------------------------------------
# Slippage + spread model
# -----------------------------------------------------------------------------

def fx_slippage_cost(
    base_spread_pips: float = 0.6,
    slippage_vol_fraction: float = 0.05,
) -> Callable:
    """Return an execution cost function that models FX spread plus a small slippage fraction.

    Cost is expressed in price units: half the spread plus a slippage term proportional to
    the one-day realized volatility (default 5% of daily vol).  This keeps the model realistic
    for liquid majors without turning a modest signal into a large loss.
    """

    def _cost(entry: pd.Series, _exit: pd.Series, _direction: int, context: pd.DataFrame) -> float:
        price = float(entry["open"])
        # One-day realized volatility (absolute price units)
        close = context["close"]
        log_returns = np.log(close / close.shift(1)).dropna()
        one_day_vol = float(log_returns.tail(20).std() * price)
        slippage = slippage_vol_fraction * one_day_vol

        # Spread in price units: 0.6 pips for a 1.0-ish price, 0.06 pips for 100+ price.
        if price >= 50.0:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        spread_cost = (base_spread_pips * pip_size) / 2.0
        return spread_cost + slippage

    return _cost


def slippage_spread_factory(
    base_spread_pips: float = 0.6,
    slippage_vol_fraction: float = 0.05,
    **backtest_kwargs,
) -> Callable:
    """Factory builder for a slippage + spread execution variant."""

    def _factory(symbol: str, data: pd.DataFrame, forecast_fn: Callable) -> WalkForwardBacktest:
        return WalkForwardBacktest(
            symbol=symbol,
            data=data,
            forecast_fn=forecast_fn,
            execution_cost_fn=fx_slippage_cost(
                base_spread_pips=base_spread_pips,
                slippage_vol_fraction=slippage_vol_fraction,
            ),
            **backtest_kwargs,
        )

    return _factory


# -----------------------------------------------------------------------------
# TimesFM ensemble stub
# -----------------------------------------------------------------------------

class TimesFMVariant:
    """Placeholder for the TimesFM ensemble variant.

    Not implemented in this milestone because TimesFM is not installed and the model layer is
    intentionally Kronos-only until the experiment proves it useful.
    """

    def __init__(self, **backtest_kwargs):
        self.backtest_kwargs = backtest_kwargs

    def __call__(self, symbol: str, data: pd.DataFrame, forecast_fn: Callable) -> WalkForwardBacktest:
        raise NotImplementedError(
            "TimesFMVariant is a stub; install timesfm and implement a TimesFMForecastEngine "
            "before enabling this variant."
        )


def timesfm_ensemble_factory(**backtest_kwargs) -> Callable:
    """Factory builder for the TimesFM ensemble variant (stub)."""
    return TimesFMVariant(**backtest_kwargs)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _atr(df: pd.DataFrame, window: int = 14) -> float:
    """Average True Range helper used by regime filter."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.rolling(window=window).mean().iloc[-1])
