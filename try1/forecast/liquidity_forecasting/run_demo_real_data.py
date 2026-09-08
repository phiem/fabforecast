"""Real-data ingestion demo: source files -> loaders -> validation -> pipeline.

Writes sample MT940 / MT942 / camt.053 / payment-hub / FX-blotter / queue / benchmark
files from the synthetic generator, loads them back through the production loaders,
validates, then trains with known flows, treasury-flow exclusion and the residual
target, and compares against the desk benchmark.

    python run_demo_real_data.py     # ~2 minutes
"""
import logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from liquidity_forecast import synthetic, samples, schema, LiquidityForecaster, PipelineConfig
from liquidity_forecast.config import DataConfig, ModelConfig, SplitConfig
from liquidity_forecast.loaders import (LoaderConfig, load_mt940, load_mt942, load_camt, load_payment_hub,
                                        load_fx_blotter, load_queue_snapshots, load_benchmark,
                                        combine_balances, combine_transactions)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
t0 = time.time(); rng = np.random.default_rng(0)
src = Path("sample_source_files"); src.mkdir(exist_ok=True)

# ---- 1. produce source-system files from synthetic data ---------------------------------
t = synthetic.generate(start="2025-02-01", end="2025-08-01")
bal, tx, accounts, external = t["balances"], t["transactions"], t["accounts"], t["external"]
eod = bal[bal.timestamp.dt.hour == 18]                     # MT940 carries only EOD balances
intraday = bal[bal.account_id.isin(["CB_USD_FED", "NOS_USD_CITI", "AGT_GBP_BARC"])]  # MT942 feed for 3 accounts
camt_accts = ["CB_EUR_ECB"]                                # one account reports in camt.053
files = {
    "mt940": samples.write_mt940(eod[~eod.account_id.isin(camt_accts)], tx, accounts, src / "statements.mt940"),
    "mt942": samples.write_mt942(intraday, tx, accounts, src / "interim.mt942"),
    "camt053": samples.write_camt053(eod[eod.account_id.isin(camt_accts)], tx, accounts, src / "statements_camt053.xml"),
    "hub": samples.write_payment_hub_csv(tx, src / "payment_hub_cash.csv"),
    "fx": samples.write_fx_blotter(tx, accounts, src / "fx_blotter.csv", rng),
    "queue": samples.write_queue_snapshots(bal.dropna(), src / "queue_snapshots.csv", rng),
    "desk": samples.write_desk_benchmark(bal, src / "desk_forecasts.csv"),
}
print({k: f"{v.stat().st_size/1e6:.1f} MB" for k, v in files.items()})

# ---- 2. load through the production loaders --------------------------------------------
lc = LoaderConfig(eod_hour=18,
                  account_map={**{f"IBAN00{a}": a for a in accounts.account_id}, **{f"LGR-{a}": a for a in accounts.account_id}},
                  flow_class_map={"NTRF": "client", "NMSC": "client", "NFEX": "known", "NSWP": "treasury",
                                  "WIRE": "client", "ACH": "client", "FX_SETTLE": "known", "SWEEP": "treasury"})
m940 = load_mt940([files["mt940"]], lc)
m942 = load_mt942([files["mt942"]], lc)
c053 = load_camt([files["camt053"]], lc)
hub = load_payment_hub(files["hub"], lc, column_map={"timestamp": "BOOKING_TS", "account_id": "LEDGER_ACCT", "amount": "AMT",
                       "dc": "DR_CR", "value_date": "VALUE_DT", "tx_type": "PRODUCT", "rail": "RAIL",
                       "counterparty_bic": "CPTY_BIC", "reference": "REF"})
fx = load_fx_blotter(files["fx"], lc, column_map={"trade_time": "TRADE_TS", "value_date": "VALUE_DATE", "buy_ccy": "BUY_CCY",
                     "buy_amount": "BUY_AMT", "sell_ccy": "SELL_CCY", "sell_amount": "SELL_AMT", "buy_account": "BUY_ACCT",
                     "sell_account": "SELL_ACCT", "reference": "TRADE_ID", "cancelled_at": "CANCELLED_TS"},
                     settlement_hour={"USD": 10, "EUR": 9, "GBP": 9})
queue = load_queue_snapshots(files["queue"], lc, column_map={"timestamp": "SNAP_TS", "account_id": "ACCT", "queued_amount": "QUEUED",
                             "held_amount": "HELD", "time_critical_amount": "TIME_CRIT", "n_items": "N"})
desk = load_benchmark(files["desk"], lc, column_map={"origin": "FCST_TIME", "target_time": "FOR_TIME", "account_id": "ACCOUNT", "forecast": "DESK_FCST"})

# Balances: EOD from MT940/camt + intraday available from MT942 where we have it; the pipeline
# forward-fills the rest and flags it. Transactions: the hub is the intraday source of truth;
# statement lines only fill accounts the hub does not cover.
balances = combine_balances([m940["balances"], c053["balances"], m942["balances"]], prefer="available")
transactions = combine_transactions([hub, m942["transactions"]])      # MT940 lines dropped: hub covers all accounts

tables = schema.validate_all({"accounts": accounts.assign(criticality=[3, 3, 2, 2, 1, 1, 1]),
                              "balances": balances, "transactions": transactions,
                              "scheduled_flows": fx, "queue_snapshots": queue, "benchmark": desk})
tables["external"] = external
print("\n=== coverage ===\n", schema.describe({k: v for k, v in tables.items() if k != "external"}))
print("\nflow_class mix:", tables["transactions"].flow_class.value_counts().to_dict())
print("balance kinds:", tables["balances"].balance_kind.value_counts().to_dict())

# ---- 3. train with residual target, backtest, benchmark ------------------------------------
cfg = PipelineConfig(
    data=DataConfig(balance_kind="available", exclude_flow_classes=["treasury"]),
    model=ModelConfig(horizons=[1, 4], quantiles=[0.05, 0.5, 0.95], cv_folds=2, residual_target=True, use_xgboost=False,
                      lgb_params=dict(n_estimators=150, learning_rate=0.05, num_leaves=15, verbose=-1)),
    split=SplitConfig(train_end="2025-05-31", valid_end="2025-06-30"),
    account_criticality=dict(zip(accounts.account_id, [3, 3, 2, 2, 1, 1, 1])),
)
fc = LiquidityForecaster(cfg, daily_horizons=[1]).fit(tables, run_cv=False)

X = fc.features["intraday"]
print("\nknown-flow / queue features (NOS_USD_CITI, sample):")
print(X.xs("NOS_USD_CITI")[["known_h1", "known_h4", "known_count_h1", "queued_amount_scaled", "held_amount_scaled",
                            "excluded_h4", "y_h4_raw", "y_h4"]].query("known_h4 != 0").head(5))
print("\nshare of origins with a known flow in next 4h:", float((X["known_h4"] != 0).mean()).__round__(3),
      "| share with treasury flow excluded:", float((X["excluded_h4"] != 0).mean()).__round__(3))

f = fc.forecast(accounts=["NOS_USD_CITI", "CB_USD_FED"], horizons=[1, 4], confidence=[0.9])
print("\n=== forecast with component breakdown ===")
print(f[["account_id", "target_time", "horizon", "current_balance", "point", "lower", "upper",
         "baseline_component", "ml_component", "known_component"]].to_string(index=False))

bt = fc.backtest("intraday", split="test")
print("\n=== backtest by horizon (unmanaged position) ===")
print(bt["by_horizon"][["n", "mae_scaled", "mape_weighted_pct", "coverage_90", "directional_acc", "skill_vs_baseline"]])
if "benchmark_comparison" in bt:
    print("\n=== model vs desk forecast on identical rows (h=4) ===")
    print(bt["benchmark_comparison"])
print("\ntop features h=4:"); print(fc.feature_importance("intraday", 4, top=10))
print(f"\ntotal {time.time()-t0:.0f}s")
