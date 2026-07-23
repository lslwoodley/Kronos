# Kronos Alpha Terminal — Roadmap (10 Next Features)

> **Status:** Evidence-based roadmap derived from the 2024-01-01 → 2025-01-01 walk-forward quant experiment across EURUSD, GBPUSD, and USDJPY. Kronos-only deterministic inference is slightly negative on all three majors; no quant overlay tested produced positive risk-adjusted returns.

## What the experiment proved

- **Kronos is the only forecast source.** The quant experiment (volatility targeting, regime filter, ARIMA residual ensemble, slippage/spread model) showed that adding filters, ensembles, or sizing rules on top of a weak signal cannot create alpha.
- **Volatility targeting is the least harmful overlay** — it reduces drawdown modestly but still produces negative returns.
- **ARIMA and regime filters add complexity without improving outcomes**.
- **A realistic execution model is essential**, but it cannot save a non-profitable signal.

## Guiding principle

> **Fix the core signal first. Risk and execution layers can only preserve alpha, not create it.**

---

## 10 next features (ranked)

### 1. Multi-horizon Kronos forecast (1d, 3d, 5d)
**Why:** The current signal uses a single 1-day forecast. Markets often move over multiple days. A consensus across horizons should reduce noise and increase confidence when all horizons agree.
**Validation:** Walk-forward experiment comparing 1d-only vs 1d+3d+5d consensus. Promote if Sharpe improves on ≥2 of 3 majors with positive return.

### 2. Multi-symbol cross-sectional signal
**Why:** EURUSD, GBPUSD, and USDJPY share a USD component. Trading only when Kronos agrees across correlated pairs (or disagrees for a specific pair) may improve directional accuracy.
**Validation:** Run a portfolio-level walk-forward backtest with equal-risk allocation and a correlation-aware consensus rule.

### 3. Market regime context injection
**Why:** Instead of filtering trades *after* the forecast (regime filter failed), feed regime/volatility context into the forecast *input* or as a conditioning feature so Kronos can adapt. This is fundamentally different from a post-hoc filter.
**Validation:** Build a regime-aware context vector (trend strength, recent volatility, macro event window) and test whether Kronos forecasts improve when conditioned on it. If Kronos architecture cannot be conditioned, this becomes a feature-engineering experiment for a later model.

### 4. Kronos fine-tuning on FX-specific data
**Why:** The public HuggingFace model is trained on 45 markets. Fine-tuning on FX-specific daily bars (and/or higher-frequency FX data) should align the model’s token distribution with the target domain.
**Validation:** Fine-tune a small Kronos variant on 10+ years of daily FX data. Compare risk-adjusted returns vs the frozen base model on a held-out 2024 period.

### 5. Forecast attribution and confidence scoring
**Why:** Not every forecast should be traded. Build a confidence score from forecast horizon agreement, recent forecast error, and input distribution drift. Trade only high-confidence forecasts.
**Validation:** Calibrate confidence on a training window and test on 2024. Promote if it improves Sharpe/return on ≥2 majors.

### 6. Walk-forward cross-validation framework
**Why:** One 2024 backtest is not enough. We need monthly retraining/retesting splits to estimate whether results are robust or curve-fit.
**Validation:** Implement anchored/expanding window CV. Use it to evaluate all future signal changes. Do not promote a feature unless it improves CV Sharpe.

### 7. Realistic execution cost model as default
**Why:** The current slippage/spread experiment proved that ignoring costs overstates performance. The default backtest should include realistic spread + small slippage.
**Validation:** Make the 5% daily-vol slippage model the default in `WalkForwardBacktest`. Future signal improvements must beat this realistic baseline.

### 8. Drawdown circuit breaker
**Why:** Even a positive-edge system can blow up. A risk-layer rule that halves size or halts trading after a trailing drawdown threshold is essential before any live capital.
**Validation:** Backtest with circuit breakers at -3%, -5%, -10%. Confirm max drawdown is bounded without destroying too much upside.

### 9. Daily paper-trading loop with journal
**Why:** Move from historical backtest to real-time validation. A cron job that fetches the latest bar, generates the signal, logs it, and waits for the next day gives us live out-of-sample data without capital risk.
**Validation:** Run for 30+ trading days, compare live signal vs backtest-equivalent signal on the same days. Track forecast error and slippage.

### 10. Broker adapter behind a clean `Broker` interface
**Why:** Once the signal is profitable in paper trading, execution must be pluggable. Design the interface first so OANDA, IBKR, or Alpaca can be swapped without changing strategy code.
**Validation:** Implement the interface and a simulator that matches it. Write a paper-to-live migration test that verifies the same signal produces the same order sequence in sim and live stubs.

---

## What is *not* on this roadmap (and why)

| Item | Why it is excluded |
|---|---|
| **TimesFM ensemble** | General time-series models are inferior to Kronos for finance; adding them dilutes rather than diversifies. Revisit only after Kronos is fine-tuned. |
| **NNFX / indicator stacking** | Abandoned per user direction. Indicator filters cannot create alpha from a model that already subsumes price action. |
| **Kelly / optimal f sizing** | Premature for a signal that is not yet positive risk-adjusted. Risk sizing is addressed via volatility targeting and circuit breakers. |
| **Crypto / Hummingbot Condor** | Forex-first is the domain until we have a working FX signal. Condor remains a future crypto execution option. |
| **Intraday / tick execution** | Daily bars are the right granularity for fast model iteration. Intraday can follow once daily alpha is proven. |

---

## Definition of "promoted"

A feature is promoted to the active implementation queue only when it satisfies **all** of the following on walk-forward CV across at least two of EURUSD, GBPUSD, USDJPY:

1. Positive total return.
2. Higher Sharpe ratio than the Kronos-only baseline.
3. Max drawdown no worse than 2× the baseline.
4. Deterministic and reproducible results (seeded / model.eval() / no sampling).

## References

- Quant experiment plan: `.hermes/plans/2026-07-22_kronos-alpha-quant-experiment.md`
- Experiment results: `results/alpha/*_alpha_experiment_2024-01-01_2025-01-01.csv`
