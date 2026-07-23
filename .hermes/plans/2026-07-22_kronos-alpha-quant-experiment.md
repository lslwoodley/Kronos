# Kronos Alpha Terminal — Quant Layer Experiment Plan

> **Goal:** Run a controlled experiment to decide which quant methods should be promoted to the Kronos Alpha Terminal roadmap.
> **Conclusion:** None of the tested overlays produced positive risk-adjusted returns. The core Kronos signal must be improved before any quant layer can help. Volatility targeting is the least harmful overlay and execution-cost realism is mandatory.

## Principle

Kronos is the **primary alpha source**. Quant methods (sizing, execution cost, signal filters) operate in the **risk and execution layers**, not as inputs to the model. This document records the experiment design and results.

## Baseline

- Kronos-only deterministic forecast (1-day ahead, `sample_logits=False`, `sample_count=1`, `model.eval()` applied).
- Walk-forward backtest: trade at next open, exit at following open.
- Symbols: EURUSD, GBPUSD, USDJPY on 2024-01-01 → 2025-01-01 daily bars.

## Variants tested

| Variant | Layer | What it does |
|---|---|---|
| **baseline** | none | Kronos only, ATR-based sizing, no execution cost. |
| **vol_target** | sizing | Scale position to target 10% annualized volatility, capped at 2× base size. |
| **regime_filter** | signal filter | Skip trades when trend strength is low or realized volatility is in the 95th percentile. |
| **arima_ensemble** | forecast blend | Blend Kronos forecast with ARIMA(1,1,1) residual forecast (30% weight). |
| **slippage_spread** | execution cost | Deduct half-spread + 5% of daily volatility from each exit price. |

## Results

### EURUSD

| variant | return_pct | sharpe | max_drawdown_pct | trades | win_rate_pct | calmar | promoted |
|---|---|---|---|---|---|---|---|
| baseline | -0.94 | -0.02 | -2.93 | 116 | 50.9 | -0.32 | False |
| vol_target | -0.59 | -0.02 | -1.86 | 116 | 50.9 | -0.32 | False |
| regime_filter | -0.79 | -1.16 | -2.02 | 25 | 52.0 | -0.39 | False |
| arima_ensemble | -0.84 | -0.13 | -2.74 | 82 | 53.7 | -0.31 | False |
| slippage_spread | -2.70 | -0.97 | -4.01 | 116 | 46.6 | -0.67 | False |

### GBPUSD

| variant | return_pct | sharpe | max_drawdown_pct | trades | win_rate_pct | calmar | promoted |
|---|---|---|---|---|---|---|---|
| baseline | -0.56 | -0.29 | -2.16 | 113 | 47.8 | -0.26 | False |
| vol_target | -0.35 | -0.29 | -1.36 | 113 | 47.8 | -0.26 | False |
| regime_filter | -1.14 | -3.12 | -1.77 | 22 | 36.4 | -0.65 | False |
| arima_ensemble | -1.09 | -0.81 | -2.25 | 93 | 47.3 | -0.48 | False |
| slippage_spread | -1.98 | -1.14 | -3.06 | 113 | 41.6 | -0.65 | False |

### USDJPY

| variant | return_pct | sharpe | max_drawdown_pct | trades | win_rate_pct | calmar | promoted |
|---|---|---|---|---|---|---|---|
| baseline | -0.01 | -0.32 | -0.03 | 150 | 50.0 | -0.33 | False |
| vol_target | -0.01 | -0.32 | -0.02 | 150 | 50.0 | -0.33 | False |
| regime_filter | -0.01 | -0.59 | -0.02 | 35 | 51.4 | -0.39 | False |
| arima_ensemble | -0.03 | -1.41 | -0.04 | 129 | 48.8 | -0.66 | False |
| slippage_spread | -0.03 | -1.07 | -0.04 | 150 | 48.0 | -0.62 | False |

## Promotion rule

A variant is promoted **only if**:

1. Total return > 0%.
2. Sharpe ratio or Calmar ratio improves over baseline.
3. Max drawdown does not exceed 2× the baseline.

No variant was promoted on any symbol.

## Key findings

1. **Kronos-only deterministic inference is not profitable** on the three majors in 2024.
2. **Volatility targeting is the least harmful overlay** — it slightly reduces drawdown but cannot turn the system positive.
3. **Regime filters and ARIMA ensembles add complexity without improving outcomes.** They are rejected as roadmap candidates.
4. **Execution cost realism is mandatory.** A 5% daily-vol slippage model is a reasonable default; without it, backtests overstate performance.

## Implications for roadmap

Because the signal is not yet profitable, the roadmap must prioritize improving the **core forecast** over adding more risk/execution overlays. See `ROADMAP.md` for the ranked 10-feature plan.

## Implementation notes

- Determinism fix: `model.eval()` and `sample_logits=False` added to `model/kronos.py` and `kronosbot/alpha/forecast.py`.
- Experiment harness: `kronosbot/alpha/experiment.py` + `kronosbot/alpha/variants.py`.
- CLI command: `kronosbot alpha-experiment SYMBOL`.
- Promotion rule: updated to require positive return.
- Slippage model: updated to 5% of daily volatility instead of 100%.

## Reproduction

```bash
cd /home/bruce/code/kronos
source .venv/bin/activate
python -m kronosbot.cli alpha-experiment EURUSD --start 2024-01-01 --end 2025-01-01
python -m kronosbot.cli alpha-experiment GBPUSD --start 2024-01-01 --end 2025-01-01
python -m kronosbot.cli alpha-experiment USDJPY --start 2024-01-01 --end 2025-01-01
```

Results are saved to `results/alpha/*_alpha_experiment_2024-01-01_2025-01-01.csv`.
