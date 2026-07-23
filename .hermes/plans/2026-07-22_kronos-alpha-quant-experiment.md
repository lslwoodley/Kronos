# Kronos Alpha Terminal — Quant Layer Experiment Plan

> **For Hermes:** Before adding any quant method to the roadmap, we test whether it improves risk-adjusted return over the Kronos-only baseline. Kronos stays the primary alpha source. Quant methods are applied in risk/execution layers, not as inputs to the forecast model.

---

## Goal

Determine which quant enhancements, if any, improve the Kronos Alpha Terminal before they are promoted to the official feature roadmap. The baseline is the Kronos-only walk-forward backtest (EURUSD 2024: Return 4.08%, Sharpe 2.12, Max DD -2.75%, 130 trades, 55.4% win).

## Core Principle

**Kronos is the signal. Quant methods are risk/execution tooling, not signal inputs.**

We do not feed Kelly size, volatility targets, regime labels, slippage models, or factor overlays into the Kronos forecast. We use them to decide how much to trade, when to stop, and how to simulate realistic execution.

## Candidates to Test

| Candidate | What it does | Why it might help | Why it might hurt |
|-----------|-----------|-------------------|-------------------|
| **Volatility targeting** | Scale position size so each trade targets a fixed expected volatility (e.g., 10% annualized). | Prevents over-leverage in calm periods and blow-ups in volatile ones. | Can under-size in strong trends where Kronos has edge; assumes vol is stable short-term. |
| **Regime filter (trend/mean-reversion/volatility)** | Disable trading or flip direction when recent market behavior is classified as non-trending or high-vol. | Avoids trades in environments where Kronos forecasts are unreliable. | Regime labels are lagging; can filter out valid signals. |
| **ARIMA residual ensemble** | Use ARIMA to forecast the residual / mean-reversion component and combine with Kronos. | Might capture short-term reversals Kronos misses. | ARIMA is mean-reversion biased; can cancel Kronos trend signals. |
| **TimesFM ensemble** | Combine Kronos point forecast with TimesFM point forecast. | Diversifies model error; TimesFM is trained on broader data. | TimesFM has <<0.01% financial data; likely weaker than Kronos and may dilute edge. |
| **Kelly fraction sizing** | Size each trade as `edge / variance` from recent backtest window. | Maximizes log-wealth growth if edge is real and stable. | Edge is not stable in FX; can cause over-sizing and ruin. |
| **Slippage + spread model** | Model spread widening and slippage around entry/exit. | Makes backtests more realistic. | Hard to parameterize without live data; can be over-fit. |
| **CVaR / max loss floor** | Cap daily or per-trade loss at a fixed fraction of equity. | Hard risk control. | Can stop out of winning trades during normal noise. |
| **Macro regime filter (NFP, FOMC, ECB days)** | Avoid trading around major scheduled macro events. | Avoids unpredictable gaps. | Reduces sample size; Kronos may already price in patterns. |
| **Multi-horizon forecast** | Use Kronos forecasts for 1-day, 3-day, 5-day horizons and combine into a single signal. | Captures short and medium-term alpha. | Increases inference cost; horizons may conflict. |
| **Post-trade calibration** | Track forecast error and adjust signal threshold or size based on recent bias. | Corrects model drift over time. | Assumes recent error predicts future error; can over-react. |

## Experiment Design

For each candidate, create a **single variant** of `WalkForwardBacktest` that applies the quant method to the signal, sizing, or execution layer while keeping the Kronos forecast unchanged.

### Baseline
- Kronos-only signal, ATR-based position sizing (1% risk), 2× ATR stop, trade next open, hold 1 day.
- Symbols: EURUSD, GBPUSD, USDJPY.
- Period: 2024-01-01 to 2025-01-01.
- Metrics: Return, Sharpe, Max Drawdown, Trades, Win Rate, Calmar.

### Variant Rules
- Only one quant change per variant.
- No method sees future data.
- Execution variant uses fixed spread/slippage assumption, not fitted to data.
- Ensemble variants combine forecasts with equal weight or simple median.

### Success Criteria
A variant is promoted to the roadmap only if it improves **Sharpe ratio** or **Calmar ratio** over the baseline on at least 2 of 3 symbols without increasing max drawdown by more than 2×.

## Implementation Plan

### Task 1: Add a backtest comparison harness

**File:** `kronosbot/alpha/experiment.py`
**Test:** `tests/kronosbot/alpha/test_experiment.py`

Create a `VariantBacktest` base class and a `run_variants` function that takes a list of variant factories and returns a comparison table.

### Task 2: Implement the baseline across EURUSD, GBPUSD, USDJPY

Run the Kronos-only backtest on all three symbols and record the baseline metrics.

### Task 3: Implement Volatility Targeting variant

Add `position_size = target_vol / (stop_distance * sqrt(252))` or similar, capped at max 2× baseline size.

### Task 4: Implement Regime Filter variant

Classify regime using a simple ADX-like trend strength or rolling Sharpe of returns. Skip trades when trend strength is below threshold or realized vol is above 95th percentile.

### Task 5: Implement ARIMA Residual Ensemble variant

Fit ARIMA on recent returns and combine with Kronos forecast. Keep model simple and fast (ARIMA(1,1,1)).

### Task 6: Implement TimesFM Ensemble variant

Add a `TimesFMForecastEngine` that wraps `timesfm` and ensemble with Kronos forecast via median or weighted average.

### Task 7: Implement Slippage + Spread Model variant

Add realistic spread and slippage to the backtest execution layer. For FX, assume 0.6 pips spread + slippage proportional to 1-day realized vol.

### Task 8: Run the full comparison

Run all variants on EURUSD, GBPUSD, USDJPY. Record metrics and produce a summary table.

### Task 9: Document the results

Write a markdown report `docs/alpha_quant_experiment_results.md` with the comparison table and which variants are promoted to the roadmap.

### Task 10: Update the roadmap

Create `docs/alpha_terminal_roadmap.md` with only the promoted features and the original 10 user-requested features ordered by impact and build order.

## Files Likely to Change

- `kronosbot/alpha/experiment.py` (new)
- `kronosbot/alpha/variants.py` (new)
- `kronosbot/alpha/backtest.py` (add hooks for spread/slippage, sizing, regime)
- `tests/kronosbot/alpha/test_experiment.py` (new)
- `tests/kronosbot/alpha/test_variants.py` (new)
- `docs/alpha_quant_experiment_results.md` (new)
- `docs/alpha_terminal_roadmap.md` (new)
- `README.md` (add experiment link)

## Risks

- **TimesFM installation** may require newer Python or JAX/torch dependencies. If it fails, skip that variant and note it.
- **Computation time** — running Kronos inference across 3 symbols × 6 variants could take hours. Run in background where possible.
- **Look-ahead bias** — must be avoided in every variant. Double-check that no variant uses future data.
- **Overfitting** — variants are only promoted if they improve on multiple symbols, not just EURUSD.

## Success Criteria

- [ ] Baseline metrics recorded for EURUSD, GBPUSD, USDJPY.
- [ ] At least 3 quant variants implemented and tested.
- [ ] Comparison report saved to `docs/alpha_quant_experiment_results.md`.
- [ ] Roadmap saved to `docs/alpha_terminal_roadmap.md`.
- [ ] Only statistically / economically superior variants promoted to the roadmap.
