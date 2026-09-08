# Integration guide: how the data is interpreted for training and testing

This explains what the code does with each table once it is loaded — which columns
become the target, which become features, which are held back for evaluation, and
where leakage is prevented. Read alongside `DATA_REQUIREMENTS.md` (what to source) and
`README.md` (the modelling design). The worked example is `run_demo_real_data.py`.

## 1. From source files to the pipeline in five calls

```python
lc = LoaderConfig(eod_hour=18, account_map={...}, flow_class_map={...})
balances     = combine_balances([load_mt940(mt940_files, lc)["balances"],
                                 load_camt(camt_files, lc)["balances"],
                                 load_mt942(mt942_files, lc)["balances"]], prefer="available")
transactions = combine_transactions([load_payment_hub(hub_file, lc, column_map, product_map=...),
                                     load_mt942(mt942_files, lc)["transactions"]])
tables = schema.validate_all({"accounts": accounts, "balances": balances, "transactions": transactions,
                              "scheduled_flows": fx_legs_and_maturities, "queue_snapshots": queue,
                              "benchmark": desk_forecasts})
tables["external"] = market_data
fc = LiquidityForecaster(cfg).fit(tables)
```

## 2. What each table becomes

### balances → the target series
`data.build_panel` keeps only `DataConfig.balance_kind` (default `available`), floors
timestamps to the hour, takes the last observation per (account, hour), re-indexes onto the
business calendar and forward-fills gaps with `balance_filled_flag = 1`.

Interpretation choices you must make:
* **Available vs booked.** The queue spends *available*; regulatory reserve tests use
  *booked*. Model the one the decision uses; if only booked is available intraday, model
  booked and note that intraday credit lines are not reflected.
* **EOD-only accounts.** An account with MT940 only has one true observation a day; the
  other 11 hours are forward-filled and flagged. The model still trains on them (the flag
  is a feature) but intraday accuracy for that account is an artefact — report it
  separately (`by_account` in the backtest) and push for an MT942/hub feed.
* **Opening balance.** `balance_kind='opening'` rows are kept in `balances` for the
  continuity check but are not modelled.

### transactions → flow features and the treasury exclusion
Aggregated per (account, hour) into `net_flow`, `inflow`, `outflow`, `tx_count`,
`max_abs_tx`. Two derived quantities matter:
* `net_flow_clean` — winsorised copy for rolling features (outliers flagged, target untouched).
* `excluded_flow` — the sum of `flow_class ∈ DataConfig.exclude_flow_classes` (default
  `treasury`). It is subtracted from the target so the model forecasts the **unmanaged**
  position. If you leave `flow_class` empty everything is `client` and nothing is excluded —
  the model will then learn to predict yesterday's sweeps.

Continuity check to run before training: per account,
`balance.diff() − net_flow` should be near zero. If it is not, a feed is missing or
booked/available are mixed.

### scheduled_flows → known-flow features and the residual target
`known_flows.known_flow_features` maps each scheduled flow onto every origin `t` at which
it was known and lay within the next `h` open periods, giving `known_h{h}` (scaled).
With `ModelConfig.residual_target=True` (default) the target becomes

```
y_h = raw_change_h − known_h − excluded_h
```

and forecasts add `known_h` back (`known_component` in the output). The model therefore
predicts *what the desk does not already know*. Cancelled flows stay in `known_h` until
`cancelled_at` — the shortfall they cause is exactly what the residual model should learn.

If `scheduled_flows` is absent, `known_h = 0` and the target reduces to the treasury-
excluded change. Point-in-time is enforced by `validate_all`; do **not** build
`scheduled_flows` from settled transactions with `known_at = value_time − 1 day` except
for a first dry run — that assumes every flow was foreseen, which overstates accuracy.

### queue_snapshots → state features
As-of joined (last snapshot ≤ t) into `queued_amount_scaled`, `held_amount_scaled`,
`time_critical_amount_scaled`, `queue_snapshot_age_h`. They tell the model how much outflow
is pending release; they are *not* subtracted from the target because release timing is
uncertain. If you also put queued items into `scheduled_flows` (with `known_at` = queue
entry time and `value_time` = expected release), they *are* subtracted — choose one
treatment per item type, not both.

### external → features
Forward-filled onto the calendar; each series and its short-window change are features,
plus a VIX z-score. Nothing here is a target.

### accounts → categoricals, criticality, hours
`account_type`, `currency`, `account_id` are categorical features of the global model.
`criticality` feeds `mape_weighted_pct` and is the natural queue-priority weight.

### benchmark → evaluation only
Never a feature. `backtest()` aligns desk forecasts with model predictions on identical
(account, origin, target_time) rows and reports both under `benchmark_comparison`.

## 3. Splits, cross-validation and leakage control

* `SplitConfig.train_end / valid_end / test_end` cut by **origin timestamp**. A target at
  `t+h` may fall past `train_end`; that is allowed for train rows but the CV folds purge
  `h` periods between fit and validation so no fit target overlaps a validation feature row.
* The final model is fit on train + valid; `valid` is for hyperparameter and CV reporting,
  `test` is touched only by `backtest(split="test")`.
* Conformal calibration uses the last `calibration_fraction` of the fit window — chronologically
  the most recent data — so intervals reflect current volatility.
* Seasonal profiles and the per-account `scale` are computed on training rows / causally.
* Recommended real-data split for 3 years: train = first 24 months, valid = next 6, test =
  last 6, then a **second** test cut around a known stress window (use `backtest(split="valid")`
  with `by_regime`).

## 4. Reading the backtest

`backtest()` returns `predictions` plus metric tables. Columns to understand:

| column | meaning |
|---|---|
| `current` | balance at origin |
| `point`, `q_*` | forecast levels of the unmanaged position (known flows added back) |
| `actual` | realised unmanaged position (treasury flows removed) — the like-for-like target |
| `actual_incl_treasury` | realised balance as booked — for reconciliation with statements |
| `baseline_point` | seasonal-only forecast; `skill_vs_baseline` compares against it |
| `known_component` | scheduled flows the forecast simply added |
| `target_time` | calendar-correct target timestamp |

Score against `actual`; if you score against `actual_incl_treasury` you are measuring the
desk's sweep behaviour, not the model.

Metrics to report by account type and by regime: `mae_scaled` (headline), `coverage_90`
(should sit in 0.86–0.94), `directional_acc`, `skill_vs_baseline`, and the
`benchmark_comparison` table. `mape_pct` is unreliable for accounts that cross zero.

## 5. Common mapping decisions

| Situation | What to do |
|---|---|
| Same transaction in hub and MT942 | `combine_transactions` de-dups on (account, amount, value_date, reference); make sure `reference` carries the same id in both feeds or tighten `dedupe_on` |
| Agent sends MT940 only, nostro has hub coverage | Use hub for transactions, MT940 for the EOD balance; drop MT940 lines from `combine_transactions` |
| Sub-accounts / ledger splits | Map all to one canonical `account_id` in `account_map` if the queue treats them as one pool |
| Currency holidays | `seasonality.account_open_mask` drops an account's rows on its own holidays; keep other currencies open |
| Account opened mid-history | Rows before opening are dropped by `build_panel`'s reindex only if no balance exists; set `SplitConfig` so the account has ≥ 3 months in train |
| Time-zone of sources differ | Convert in the loader (`LoaderConfig.tz`); the panel assumes one zone |
| Negative balances (overdrafts) | Fine — the target is a change; `scale` uses `|balance|` with a floor |

## 6. Retraining with real data

Store the raw canonical tables (parquet, partitioned by month); rebuild the panel and
features in the retrain job rather than appending to old feature files, so a change to
`flow_class_map` or business hours propagates cleanly. Keep the previous model's backtest
on the most recent 10 business days and reject the new model if `mae_scaled` or
`coverage_90` is worse (champion/challenger).
