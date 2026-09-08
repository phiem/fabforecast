"""Demo of the seasonality extension: per-currency holidays, moving-date events, trend.

    python run_demo_seasonality.py      # ~1 minute (short window, small models)
"""
import logging, sys, time
import pandas as pd
from liquidity_forecast import LiquidityForecaster, PipelineConfig, synthetic
from liquidity_forecast.config import ModelConfig, SplitConfig
from liquidity_forecast.seasonality import (SeasonalityConfig, build_seasonal_features,
                                            CurrencyCalendar, account_open_mask, stl_decompose)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

t0 = time.time()
tables = synthetic.generate(start="2025-01-01", end="2025-09-01")
external = tables.pop("external"); tables.pop("stress_window")

sea_cfg = SeasonalityConfig(
    extra_holidays={"USD": ["2025-01-09"]},                   # ad-hoc closure (national day of mourning)
    custom_events={"rmp_end": ["2025-02-11", "2025-03-25", "2025-05-06", "2025-06-17", "2025-07-29"],
                   "dividend_run": ["2025-03-28", "2025-06-27"]},
    event_currencies={"us_tax": ["USD"], "uk_tax": ["GBP"], "ust_coupon": ["USD"],
                      "rmp_end": ["EUR"]},                      # ECB reserve maintenance periods
)
cal = CurrencyCalendar(["USD", "EUR", "GBP"], range(2024, 2027), sea_cfg)

cfg = PipelineConfig(
    model=ModelConfig(horizons=[1, 4], quantiles=[0.05, 0.5, 0.95], cv_folds=2,
                      lgb_params=dict(n_estimators=120, learning_rate=0.05, num_leaves=15, verbose=-1),
                      use_xgboost=False),
    split=SplitConfig(train_end="2025-06-30", valid_end="2025-07-31"),
)
fc = LiquidityForecaster(cfg, daily_horizons=[1],
                         extra_features=build_seasonal_features(sea_cfg),
                         open_mask=lambda panel: account_open_mask(panel, cal))
fc.fit(tables, external, run_cv=False)

X = fc.features["intraday"]
sea_cols = [c for c in X.columns if c.startswith(("ccy_", "days_to_", "days_since_", "is_imm", "is_us_tax",
                                                   "is_uk_tax", "is_payroll", "is_ust", "is_rmp", "is_dividend", "trend_"))]
print(f"\n{len(sea_cols)} seasonality features added; feature matrix now {X.shape[1]} cols")
print("\nGBP agent bank around the 2025 spring bank holiday (26 May):")
print(X.xs("AGT_GBP_BARC").loc["2025-05-22 10:00":"2025-05-28 10:00":12,
      ["ccy_pre_holiday", "ccy_post_holiday", "days_to_ccy_holiday", "counter_ccy_holiday", "is_uk_tax", "days_to_uk_tax"]])
print("\nNote: 26 May rows are absent for GBP/USD accounts (open_mask) but present for EUR (TARGET2 open).")
print(X.xs("2025-05-26 10:00", level="timestamp")[["ccy_holiday", "counter_ccy_holiday"]])

print("\nUSD Fed account, US tax date 15 Apr and IMM 18 Jun:")
print(X.xs("CB_USD_FED").loc[["2025-04-14 10:00", "2025-04-15 10:00", "2025-06-17 10:00", "2025-06-18 10:00"],
      ["is_us_tax", "days_to_us_tax", "days_since_us_tax", "is_imm", "days_to_imm", "is_ust_coupon", "trend_slope_120"]])

print("\nTop 15 features, intraday h=4 (incl. seasonality block):")
print(fc.feature_importance("intraday", 4, top=15))

print("\nSTL diagnostic (daily, CB_EUR_ECB) — trend range vs seasonal amplitude:")
d = stl_decompose(fc.panels["daily"], "CB_EUR_ECB")
print(d.agg(["min", "max", "std"]).round(4))
print(f"\nBacktest (test):"); print(fc.backtest("intraday")["by_horizon"][["n", "mae_scaled", "coverage_90", "skill_vs_baseline"]])
print(f"\ntotal {time.time()-t0:.0f}s")
