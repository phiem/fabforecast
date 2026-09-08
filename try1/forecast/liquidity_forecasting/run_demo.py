"""End-to-end demo on synthetic data.

    python run_demo.py            # ~2-4 minutes on a laptop
"""
import logging, sys, time
import pandas as pd
from liquidity_forecast import LiquidityForecaster, PipelineConfig
from liquidity_forecast.config import ModelConfig, SplitConfig
from liquidity_forecast import synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    stream=sys.stdout)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4g}")

t0 = time.time()
tables = synthetic.generate()
external = tables.pop("external"); stress = tables.pop("stress_window")

cfg = PipelineConfig(
    model=ModelConfig(horizons=[1, 4, 24], quantiles=[0.05, 0.10, 0.50, 0.90, 0.95],
                      cv_folds=2, lgb_params=dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4),
                      xgb_params=dict(n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0, n_jobs=4)),
    split=SplitConfig(train_end="2025-10-31", valid_end="2025-12-31"),
    account_criticality={"CB_USD_FED": 3.0, "CB_EUR_ECB": 3.0, "AGT_USD_JPM": 2.0,
                         "AGT_GBP_BARC": 2.0, "NOS_USD_CITI": 1.0, "NOS_EUR_DB": 1.0, "NOS_GBP_HSBC": 1.0},
    anomaly_threshold=0.5,
)
fc = LiquidityForecaster(cfg, daily_horizons=[1, 5]).fit(tables, external)
print(f"\n=== trained in {time.time()-t0:.0f}s ===\n")

print("=== Time-series CV (intraday) ===")
print(fc.cv_results["intraday"].groupby("horizon")[["mae_scaled", "coverage_90", "directional_acc", "skill_vs_baseline"]].mean())
print("\n=== Time-series CV (daily) ===")
print(fc.cv_results["daily"].groupby("horizon")[["mae_scaled", "coverage_90", "directional_acc", "skill_vs_baseline"]].mean())

print("\n=== Live forecast, intraday, 80% & 90% intervals ===")
f = fc.forecast(accounts=["CB_USD_FED", "NOS_USD_CITI"], granularity="intraday", horizons=[1, 4, 24], confidence=[0.8, 0.9])
print(f[["account_id", "target_time", "horizon", "confidence", "current_balance", "point", "lower", "upper",
         "baseline_component", "ml_component", "confidence_score"]].to_string(index=False))

print("\n=== Live forecast, daily ===")
print(fc.forecast(granularity="daily", horizons=[1, 5], confidence=[0.9])[
    ["account_id", "target_time", "horizon", "point", "lower", "upper", "stress_regime"]].to_string(index=False))

for gran in ("intraday", "daily"):
    bt = fc.backtest(granularity=gran, split="test")
    cols = ["n", "mae_scaled", "mape_pct", "mape_weighted_pct", "directional_acc", "coverage_90", "interval_width_scaled", "skill_vs_baseline"]
    print(f"\n=== Backtest ({gran}, test period) by horizon ===");        print(bt["by_horizon"][cols])
    print(f"\n=== Backtest ({gran}) by account type ===");                 print(bt["by_account_type"][cols])
    print(f"\n=== Backtest ({gran}) by regime (1 = detected stress) ===");  print(bt["by_regime"][cols])
    print(f"\n{gran}: {len(bt['anomalies'])} anomalies flagged of {len(bt['predictions'])} forecasts")

print("\n=== Stress-window backtest (valid split contains the synthetic crisis) ===")
bt_v = fc.backtest(granularity="intraday", split="valid")
print(bt_v["by_regime"][["n", "mae_scaled", "coverage_90", "interval_width_scaled"]])

print("\n=== Top features, intraday h=1 (global model) ===")
print(fc.feature_importance("intraday", 1, top=12))
print("\n=== Top features by account type, intraday h=4 ===")
for t, s in fc.feature_importance_by_account_type("intraday", 4, top=8).items():
    print(f"\n-- {t}"); print(s)
print("\n=== Top features by account type, daily h=1 ===")
for t, s in fc.feature_importance_by_account_type("daily", 1, top=8).items():
    print(f"\n-- {t}"); print(s)
print(f"\ntotal {time.time()-t0:.0f}s")
