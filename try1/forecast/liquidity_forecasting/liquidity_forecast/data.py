"""Data ingestion and cleaning.

Output of :func:`build_panel` is a single *long* panel indexed by
(account_id, timestamp) on a regular business-hour calendar with columns:

    balance, balance_filled_flag, net_flow, inflow, outflow, tx_count,
    outlier_flag, net_flow_clean, scale, account_type, currency, is_open

Design choices
--------------
* **Calendar alignment** – every account is re-indexed onto the same business
  calendar (weekdays x business hours minus holidays). Closed periods are
  removed rather than zero-filled, so lags mean "N *open* hours ago".
* **Gap filling** – missing balances are forward-filled and flagged; a
  flag column lets the model discount imputed observations.
* **Outliers** – detected on flows with a rolling median / MAD robust
  z-score. The *target* is never altered; outliers are winsorised only in the
  feature copy (``net_flow_clean``) so a single wire does not poison rolling
  statistics.
* **Normalisation** – each account gets a ``scale`` (rolling median absolute
  balance, computed causally). Targets and flow features are expressed as
  fractions of scale so one global model can be trained across accounts
  whose balances differ by 2-3 orders of magnitude.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .config import DataConfig

log = logging.getLogger(__name__)

REQUIRED = {
    "accounts": {"account_id", "account_type", "currency"},
    "balances": {"timestamp", "account_id", "balance"},
    "transactions": {"timestamp", "account_id", "amount"},
}


def validate(tables: Dict[str, pd.DataFrame]) -> None:
    for name, cols in REQUIRED.items():
        if name not in tables:
            raise ValueError(f"missing table '{name}'")
        missing = cols - set(tables[name].columns)
        if missing:
            raise ValueError(f"table '{name}' missing columns {sorted(missing)}")
    if tables["balances"].account_id.nunique() == 0:
        raise ValueError("no accounts in balances")


def business_calendar(start: pd.Timestamp, end: pd.Timestamp, cfg: DataConfig) -> pd.DatetimeIndex:
    idx = pd.date_range(start.floor("h"), end.ceil("h"), freq=cfg.freq)
    mask = idx.dayofweek < 5
    if cfg.freq.lower().startswith("h"):
        mask &= (idx.hour >= cfg.business_hours[0]) & (idx.hour < cfg.business_hours[1])
    if cfg.holidays:
        hol = pd.to_datetime(cfg.holidays).normalize()
        mask &= ~idx.normalize().isin(hol)
    return idx[mask]


def aggregate_transactions(tx: pd.DataFrame, calendar: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    """Bucket raw transactions into the calendar grid per account."""
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["bucket"] = tx["timestamp"].dt.floor(freq)
    # transactions landing in closed periods roll to the next open bucket
    pos = calendar.searchsorted(tx["bucket"].values, side="left")
    pos = np.clip(pos, 0, len(calendar) - 1)
    tx["bucket"] = calendar[pos]
    tx["_in"] = tx["amount"].clip(lower=0)
    tx["_out"] = (-tx["amount"]).clip(lower=0)
    tx["_abs"] = tx["amount"].abs()
    g = tx.groupby(["account_id", "bucket"])
    agg = pd.DataFrame({
        "net_flow": g["amount"].sum(), "inflow": g["_in"].sum(), "outflow": g["_out"].sum(),
        "tx_count": g["amount"].size(), "max_abs_tx": g["_abs"].max(),
    })
    agg.index.names = ["account_id", "timestamp"]
    return agg


def robust_z(x: pd.Series, window: int = 240) -> pd.Series:
    med = x.rolling(window, min_periods=24).median()
    mad = (x - med).abs().rolling(window, min_periods=24).median() * 1.4826
    return (x - med) / mad.replace(0, np.nan)


def build_panel(tables: Dict[str, pd.DataFrame], cfg: DataConfig,
                external: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    validate(tables)
    accounts = tables["accounts"].set_index("account_id")
    bal = tables["balances"].copy()
    bal["timestamp"] = pd.to_datetime(bal["timestamp"])
    if "balance_kind" in bal.columns and bal["balance_kind"].notna().any():
        kinds = bal["balance_kind"].fillna("booked")
        pick = kinds == cfg.balance_kind
        if not pick.any():
            log.warning("no balances of kind '%s'; falling back to booked", cfg.balance_kind)
            pick = kinds == "booked"
        bal = bal[pick]
    tx = tables["transactions"]

    cal = business_calendar(min(bal.timestamp.min(), pd.to_datetime(tx.timestamp).min()),
                            max(bal.timestamp.max(), pd.to_datetime(tx.timestamp).max()), cfg)
    log.info("calendar: %d open periods from %s to %s", len(cal), cal[0], cal[-1])

    bal["timestamp"] = bal["timestamp"].dt.floor(cfg.freq)
    bal = bal.groupby(["account_id", "timestamp"])["balance"].last()
    full_index = pd.MultiIndex.from_product([accounts.index, cal], names=["account_id", "timestamp"])
    panel = pd.DataFrame(index=full_index)
    panel["balance"] = bal.reindex(full_index)
    panel["balance_filled_flag"] = panel["balance"].isna().astype(int)
    if cfg.fill_gaps:
        panel["balance"] = panel.groupby(level="account_id")["balance"].ffill().bfill()
    n_filled = int(panel["balance_filled_flag"].sum())
    log.info("filled %d missing balance observations (%.2f%%)", n_filled, 100 * n_filled / len(panel))

    agg = aggregate_transactions(tx, cal, cfg.freq)
    panel = panel.join(agg)
    for c in ["net_flow", "inflow", "outflow", "tx_count", "max_abs_tx"]:
        panel[c] = panel[c].fillna(0.0)
    # treasury-class flows are decisions; kept separately so the target can exclude them
    if "flow_class" in tx.columns and cfg.exclude_flow_classes:
        excl = tx[tx["flow_class"].isin(cfg.exclude_flow_classes)]
        if len(excl):
            ex = aggregate_transactions(excl, cal, cfg.freq)["net_flow"].rename("excluded_flow")
            panel = panel.join(ex)
            log.info("excluded %d %s transactions from the modelling target", len(excl), cfg.exclude_flow_classes)
    panel["excluded_flow"] = panel.get("excluded_flow", 0.0)
    panel["excluded_flow"] = panel["excluded_flow"].fillna(0.0)

    # causal per-account scale: rolling median |balance| over ~3 months of open hours
    grp = panel.groupby(level="account_id")
    panel["scale"] = grp["balance"].transform(
        lambda s: s.abs().rolling(60 * 12, min_periods=24).median().bfill())
    panel["scale"] = panel["scale"].clip(lower=1.0)

    # outliers on flow (robust z) -> flag & winsorise the feature copy
    z = grp["net_flow"].transform(lambda s: robust_z(s))
    panel["outlier_flag"] = (z.abs() > cfg.outlier_mad_threshold).astype(int)
    panel["net_flow_clean"] = panel["net_flow"]
    if cfg.winsorize_outliers:
        cap = grp["net_flow"].transform(lambda s: s.abs().rolling(240, min_periods=24).quantile(0.99).bfill())
        panel["net_flow_clean"] = panel["net_flow"].clip(lower=-cap, upper=cap)
    log.info("flagged %d outlier flow buckets", int(panel["outlier_flag"].sum()))

    panel = panel.join(accounts[["account_type", "currency"]], on="account_id")
    panel["is_open"] = 1

    if external is not None:
        ext = external.copy()
        ext["timestamp"] = pd.to_datetime(ext["timestamp"]).dt.floor(cfg.freq)
        ext = ext.groupby("timestamp").last().reindex(cal).ffill().bfill()
        panel = panel.join(ext, on="timestamp")

    return panel.sort_index()


def make_splits(index: pd.DatetimeIndex, train_end: str, valid_end: str, test_end: Optional[str]):
    ts = pd.Series(index)
    train = ts <= pd.Timestamp(train_end)
    valid = (ts > pd.Timestamp(train_end)) & (ts <= pd.Timestamp(valid_end))
    test = ts > pd.Timestamp(valid_end)
    if test_end:
        test &= ts <= pd.Timestamp(test_end)
    return train.values, valid.values, test.values
