# Liquidity Forecasting System

Intraday (hourly) and daily cash-position forecasting for central bank, agent/correspondent
bank and nostro accounts, with calibrated prediction intervals, regime detection, rolling
backtests and anomaly flags.

```
liquidity_forecast/
├── config.py       # all tunables: splits, horizons, quantiles, hyperparameters, thresholds
├── data.py         # ingest → business calendar → gap fill → outlier flag → causal scaling
├── features.py     # temporal / lag / rolling / cross-account / external / flow / regime features
├── models.py       # seasonal baseline + LightGBM×XGBoost quantile ensemble + conformal calibration
├── evaluation.py   # MAE, RMSE, (criticality-weighted) MAPE, directional accuracy, pinball, coverage
├── pipeline.py     # LiquidityForecaster: fit (time-series CV), forecast, backtest, anomalies, importance
└── synthetic.py    # realistic synthetic accounts/transactions/market data for the demo
run_demo.py         # end-to-end run on synthetic data
demo_output.txt     # captured output of run_demo.py
```

```bash
pip install pandas numpy scikit-learn lightgbm xgboost
python run_demo.py
```

## 1. How the pieces fit

**Target.** Every model predicts the *scaled forward change* in balance,
`y_h = (B[t+h] − B[t]) / scale[t]`, where `scale` is a causal rolling median of the account's
absolute balance. This is what makes a single global model work across a $2.5bn Fed account and a
$30m nostro: the trees learn *behaviour* (fractions of scale), not magnitudes. Forecast levels are
reconstructed as `B[t] + y_h · scale[t]`.

**Hybrid structure.** `forecast = seasonal baseline + ML residual`. The baseline is the mean scaled
flow by (account, hour, weekday) for intraday and (account, weekday, month-end flag) for daily,
estimated on the training window only. LightGBM and XGBoost are trained on the residual with
pinball loss for each quantile and averaged. This gives an interpretable "what the calendar says"
component plus a "what the data says beyond the calendar" component, reported separately in every
forecast row (`baseline_component`, `ml_component`).

**Account heterogeneity — one global model with account features, not one model per type.**
`account_type`, `currency` and `account_id` are categorical features, so the trees can split on
type where behaviour differs and share statistical strength where it doesn't. With ~7 accounts and
~1 year of history, per-type models each see a third of the data and per-account models overfit
badly. The global approach is a cheap form of hierarchical/multi-task learning. The diagnostic
`feature_importance_by_account_type()` refits small per-type models purely to explain what drives
each type — not for forecasting. Move to per-type models only when (a) you have dozens of accounts
per type, and (b) the global model's residuals show type-specific structure in the backtest.

**Horizons — direct multi-horizon.** One model per (granularity, horizon). Direct models avoid error
compounding of recursive forecasts and let each horizon pick its own features (the 1h model leans on
`flow_lag_1`, the 24h model on `seasonal_flow` and `days_to_month_end`). Intraday and daily are
separate model families because the feature semantics differ (an hourly lag is not a daily lag).

**Uncertainty — quantile regression + Conformalised Quantile Regression.** Raw quantile regressors are
typically under-covered out of sample (the demo's raw 90% interval covers ~80%). CQR holds out the
last 20% of the training window, measures how far the realised residual falls outside the predicted
quantile, and widens each quantile by the empirical (1−α)(1+1/n) quantile of that miss. Coverage
then holds on the test set. Quantiles are sorted to guarantee monotone bounds.

**Regime detection.** Short-window realised flow volatility vs. its long-window robust
distribution → `regime_vol_z` (continuous, fed to the model) and `stress_regime` (binary, with 6-period
persistence, used to slice backtests and to tag forecasts). The interface is deliberately simple; a
Markov-switching or HMM detector can replace `regime_features()` unchanged. In stress regimes the
right behaviour is *wider intervals*, not necessarily lower point error — check `by_regime` in the
backtest for `interval_width_scaled` rising with `coverage_90` roughly preserved.

**No leakage.** Calendar alignment uses open hours only, so lag *k* means *k open hours ago*.
Seasonal profiles are fit on the training mask. The rolling `scale` is causal. Cross-validation is
expanding-window with a purge gap equal to the horizon so no training target overlaps a validation
feature row. The final model is fit on train+valid; the test window is only touched by `backtest()`.

## 2. Model selection trade-offs

| Option | Accuracy | Interpretability | Latency / cost | When to use |
|---|---|---|---|---|
| Seasonal baseline only | Low (the demo's ML adds 8–25% MAE skill on top) | Highest — a lookup table | µs | Fallback, sanity check, accounts with < 3 months of history |
| Global GBM quantile ensemble (implemented) | High | Good — SHAP/gain importances, explicit baseline vs ML split | ~1 ms/row inference; minutes to retrain | Default for 5–500 accounts |
| Single GBM (LightGBM only) | ≈ ensemble −1–3% | Same | Half the cost | When retraining budget is tight |
| Per-account-type GBMs | Higher only with many accounts per type | Slightly better per type | 3× training | > 30 accounts per type |
| LSTM / Temporal Fusion Transformer | Higher for long intraday sequences with many accounts; worse at 1 year × 7 accounts | Low (attention maps only) | GPU training, 10–100× inference cost | > 1,000 account-series or when cross-account attention matters |
| Conformal on top of any of the above | No effect on point | Adds guaranteed coverage | Negligible | Always, for anything that feeds a limit or buffer decision |

Practical guidance: gradient boosting on well-engineered features beats deep sequence models on
tabular financial time series at this data volume, and the treasury desk can audit a feature
importance table. Revisit neural models when the account universe is in the thousands.

## 3. Retraining and monitoring

**Retraining cadence**
- Intraday models: retrain **weekly** (nightly if compute is free). Payment patterns drift with
  client behaviour and cut-off changes; a week of drift is visible in the 1h/4h skill.
- Daily models: retrain **monthly**, plus a forced retrain after each quarter-end so the new
  quarter-end observations are learned before the next one.
- **Recalibrate conformal offsets daily** — it's a quantile of residuals on a rolling window and
  costs nothing; this alone fixes most coverage drift without touching the trees.
- Event-driven retrain: policy rate decision, new correspondent onboarded, payment-system cut-off
  change, or when the monitors below trip.

**What to monitor (per account type, per horizon, rolling 20 business days)**
1. `mae_scaled` vs. its backtest distribution — alert at the 95th percentile.
2. `skill_vs_baseline` — if it approaches zero, the ML component is no longer adding value; check
   for a data feed change before retraining.
3. **Interval coverage** — nominal 90% should sit in [86%, 94%]. Under-coverage → recalibrate
   conformal; persistent under-coverage after recalibration → regime shift, retrain.
4. **Directional accuracy** — a drop below ~0.6 at h=1 usually means a timestamp/calendar problem
   (DST, holiday not in `DataConfig.holidays`) rather than model decay.
5. **Feature drift** — population-stability index on the top 15 features; PSI > 0.25 triggers
   investigation.
6. **Anomaly rate** — `flag_anomalies()` should fire on ~1–3% of observations; a spike is either
   a real liquidity event or a broken upstream feed. Route to the desk either way.
7. **Data quality** — `balance_filled_flag` rate and `outlier_flag` rate per account per day.

Keep the last N model versions and their backtests; roll back if a fresh model under-performs the
previous one on the most recent 10 days (a champion/challenger check inside the retrain job).

## 4. Which features matter, by account type

From the synthetic backtest (`feature_importance_by_account_type` in `demo_output.txt`). The
synthetic generator was built to reflect typical real-world behaviour, so the *ranking* is
indicative; validate on your data.

**Central bank accounts** — dominated by the intraday settlement cycle and calendar:
`hour`/`seasonal_flow`/`periods_to_cob` (RTGS outflows early, reserve top-ups before close),
`is_month_end_window`/`days_to_month_end` (reserve-maintenance and month-end squaring), and
`bal_dev_from_scale` (mean reversion to the target reserve level). Cross-account flow from same-
currency agent banks (`ccy_other_flow`) matters at the 4h horizon because settlement legs arrive
there first. External rates (`sofr`, `estr`) matter mostly at daily horizons via reserve-
remuneration incentives.

**Agent / correspondent bank accounts** — the most "flow-driven": `flow_lag_1`, `flow_mean_4/12`,
`tx_count_rel` and `inflow/outflow_scaled` lead, because client payment volume is autocorrelated
through the day. `corr_nostro` and `ccy_other_flow` are significant — nostro pay-aways land here.
Day-of-week effects (Monday inflow, Friday outflow) are stronger than for the other types.

**Nostro accounts** — the noisiest in scaled terms. Top features are `seasonal_flow` at the FX
settlement hours, `outlier_flag`/`max_abs_tx_scaled` (single wires are a large share of scale),
`fx_*_chg` and `vix_z` (settlement size tracks FX volumes), and `regime_vol_z`. Rolling
`bal_std_24/120` is what widens the intervals correctly. Lag features beyond 4 hours add little —
nostro balances are swept, so history decays fast.

Across all types, the **baseline seasonal component explains 60–75% of forecastable variance at
h=24 and h=1 day**; the ML residual earns its keep at short intraday horizons and around
month/quarter-end.

## 5. Using the API

```python
from liquidity_forecast import LiquidityForecaster, PipelineConfig
from liquidity_forecast.config import ModelConfig, SplitConfig, DataConfig

cfg = PipelineConfig(
    data=DataConfig(business_hours=(7, 19), holidays=["2025-12-25", "2026-01-01"]),
    model=ModelConfig(horizons=[1, 4, 24], quantiles=[0.05, 0.10, 0.50, 0.90, 0.95]),
    split=SplitConfig(train_end="2025-10-31", valid_end="2025-12-31"),
    account_criticality={"CB_USD_FED": 3.0},        # weights for MAPE
    per_account_anomaly_threshold={"NOS_USD_CITI": 1.0},
)
fc = LiquidityForecaster(cfg, daily_horizons=[1, 5])
fc.fit({"accounts": accounts_df, "balances": balances_df, "transactions": tx_df}, external=market_df)

f = fc.forecast(accounts=["CB_USD_FED"], granularity="intraday", horizons=[1, 4], confidence=[0.9])
bt = fc.backtest(granularity="daily", split="test")      # dict of metric tables + anomalies
```

Required input schemas: `accounts(account_id, account_type, currency)`,
`balances(timestamp, account_id, balance)`, `transactions(timestamp, account_id, amount)`; optional
`external(timestamp, fx_*, sofr, sonia, estr, vix)`. Multi-currency is handled by scaling — no FX
conversion is applied to balances; FX rates enter only as features.

Forecast output columns: `account_id, account_type, granularity, origin, target_time, horizon,
confidence, current_balance, point, lower, upper, baseline_component, ml_component,
regime_vol_z, stress_regime, confidence_score`.

## 6. Production notes

- Logging goes through the `liquidity_forecast` logger hierarchy; set `PipelineConfig.log_level`.
- Input validation raises `ValueError` with the missing table/column; forecasting for an untrained
  horizon or an unavailable confidence level raises before any model call.
- The panel/feature build is pure pandas and idempotent — it can run as a scheduled job that writes
  a feature store, with `forecast()` served from a lightweight process that loads pickled
  `HorizonModel` objects.
- For > 100 accounts, replace `_cross_account`'s per-account loop with a grouped matrix
  computation and consider `polars`; everything else scales linearly in rows.

## 7. Seasonality extension (`seasonality.py`)

Adds the seasonal drivers a plain (hour × weekday) profile misses. Plugs in without touching the
base pipeline:

```python
from liquidity_forecast.seasonality import (SeasonalityConfig, build_seasonal_features,
                                            CurrencyCalendar, account_open_mask)
sea = SeasonalityConfig(
    extra_holidays={"USD": ["2025-01-09"]},                       # ad-hoc closures
    custom_events={"rmp_end": ["2025-02-11", "2025-03-25"],        # ECB reserve maintenance periods
                   "dividend_run": ["2025-03-28"]},
    event_currencies={"us_tax": ["USD"], "uk_tax": ["GBP"], "ust_coupon": ["USD"], "rmp_end": ["EUR"]},
)
cal = CurrencyCalendar(["USD", "EUR", "GBP"], range(2024, 2027), sea)
fc = LiquidityForecaster(cfg,
                         extra_features=build_seasonal_features(sea),          # +32 features
                         open_mask=lambda p: account_open_mask(p, cal))        # drop own-ccy holidays
```

**Per-currency holiday calendars.** Built-in rules for USD (Fedwire: Sunday→Monday substitution,
Saturday not observed), EUR (TARGET2) and GBP (UK bank holidays incl. Easter, substitute days);
other currencies via `extra_holidays`. Features: `ccy_holiday`, `ccy_pre_holiday` (pre-funding
day), `ccy_post_holiday` (catch-up settlement), `days_to_ccy_holiday`, `counter_ccy_holiday`
(another currency in the book is closed → FX legs suppressed). With `open_mask`, an account's rows
on its own holidays are removed so frozen balances don't enter lags or rolling stats; the panel
becomes non-rectangular and cross-account features align by timestamp.

**Moving-date events.** IMM dates (3rd Wed Mar/Jun/Sep/Dec), US Treasury tax dates, UK PAYE/
corporation-tax dates, payroll (15th & last business day), UST coupon/settlement dates, plus any
`custom_events`. Each yields `is_E`, `days_to_E` (capped at `event_lookahead_days`) and
`days_since_E`; event dates roll forward to the currency's next open day. Restrict an event to
currencies with `event_currencies`. In practice the highest-value custom events are the bank's own:
reserve-maintenance-period ends, known client payroll/dividend runs, CLS/CCP margin cycles.

**Trend.** Causal rolling-OLS `trend_slope_{W}` and `trend_resid_{W}` on scaled balance (5-day and
20-day windows intraday; 10/30 days daily). `stl_decompose()` is provided as a *diagnostic* STL —
not causal, for reporting on whether a trend term is warranted.

**What to expect.** On the synthetic data these features cannot improve accuracy because the
generator has no holiday or event effects — the run in `demo_seasonality_output.txt` verifies
correctness of the flags (Good Friday/Easter Monday, spring bank holiday, US tax day, IMM), not
lift. On real treasury data, expect the largest gains at daily horizons and around month-end,
where tax and coupon dates are the dominant unexplained flows.

**Annual seasonality caveat remains.** Year-end still has one observation per year of history;
`is_year_end_window` and `month` get real support only after 2–3 years. Until then, treat year-end
forecasts as wide-interval and consider a manual uplift or a known-event calendar entry.
