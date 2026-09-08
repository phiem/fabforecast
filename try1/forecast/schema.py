"""Canonical input tables for the forecaster and their validation.

Every loader in :mod:`loaders` produces one of these frames; :func:`validate_all`
checks types, keys, referential integrity and the point-in-time rule before
anything reaches the pipeline.

Tables
------
accounts          one row per account (reference data; current view is enough for
                  modelling, keep change history separately for audit)
balances          balance observations: (timestamp, account_id, balance, balance_kind)
transactions      settled cash movements with a flow_class tag
scheduled_flows   known-future flows AS KNOWN at ``known_at`` (point-in-time)
queue_snapshots   queued / held value per account at snapshot times
external          market data, wide format on timestamp
benchmark         the desk's own historical forecasts (for evaluation only)

Conventions
-----------
* All timestamps are tz-naive in a single reference zone (choose one — the
  treasury centre's local time — and convert on load; ``LoaderConfig.tz``).
* Amounts are signed from the account's point of view: + credit, - debit.
* ``flow_class`` in {client, known, treasury, fee, internal}. ``treasury`` flows
  (sweeps, funding transfers, FX swaps done to fund the account) are decisions
  and are excluded from the modelling target; ``known`` flows are settled
  items that were in ``scheduled_flows`` beforehand (FX legs, maturities).
"""
from __future__ import annotations
import logging
from typing import Dict, List
import pandas as pd

log = logging.getLogger(__name__)

ACCOUNT_TYPES = {"central_bank", "agent_bank", "nostro"}
FLOW_CLASSES = {"client", "known", "treasury", "fee", "internal"}
BALANCE_KINDS = {"available", "booked", "opening", "closing"}

SCHEMAS: Dict[str, Dict[str, str]] = {
    "accounts": {
        "account_id": "str", "account_type": "str", "currency": "str", "bic": "str?",
        "legal_entity": "str?", "agent_name": "str?", "target_balance": "float?",
        "min_balance": "float?", "intraday_line": "float?", "criticality": "float?",
        "business_hours_start": "int?", "business_hours_end": "int?",
    },
    "balances": {"timestamp": "datetime", "account_id": "str", "balance": "float", "balance_kind": "str?",
                 "source": "str?"},
    "transactions": {"timestamp": "datetime", "account_id": "str", "amount": "float", "value_date": "datetime?",
                     "tx_type": "str?", "rail": "str?", "flow_class": "str?", "counterparty_bic": "str?",
                     "source_system": "str?", "reference": "str?"},
    "scheduled_flows": {"account_id": "str", "known_at": "datetime", "value_time": "datetime", "amount": "float",
                        "flow_type": "str?", "reference": "str?", "cancelled_at": "datetime?"},
    "queue_snapshots": {"timestamp": "datetime", "account_id": "str", "queued_amount": "float",
                        "held_amount": "float?", "time_critical_amount": "float?", "n_items": "int?"},
    "external": {"timestamp": "datetime"},
    "benchmark": {"origin": "datetime", "target_time": "datetime", "account_id": "str", "forecast": "float",
                  "source": "str?"},
}


def _coerce(df: pd.DataFrame, spec: Dict[str, str], name: str) -> pd.DataFrame:
    df = df.copy()
    for col, typ in spec.items():
        optional = typ.endswith("?")
        base = typ.rstrip("?")
        if col not in df.columns:
            if optional:
                continue
            raise ValueError(f"{name}: required column '{col}' missing")
        if base == "datetime":
            df[col] = pd.to_datetime(df[col])
        elif base == "float":
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        elif base == "int":
            df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
        else:
            df[col] = df[col].astype(str).where(df[col].notna(), None)
    return df


def validate_all(tables: Dict[str, pd.DataFrame], strict: bool = True) -> Dict[str, pd.DataFrame]:
    """Coerce types, check keys and referential integrity. Returns coerced tables."""
    out = {}
    for name, df in tables.items():
        if name not in SCHEMAS:
            raise ValueError(f"unknown table '{name}'")
        out[name] = _coerce(df, SCHEMAS[name], name)

    acc = out["accounts"]
    if acc["account_id"].duplicated().any():
        raise ValueError("accounts: duplicate account_id")
    bad = set(acc["account_type"]) - ACCOUNT_TYPES
    if bad:
        raise ValueError(f"accounts: unknown account_type {bad}; expected {ACCOUNT_TYPES}")
    ids = set(acc["account_id"])

    for name in ("balances", "transactions", "scheduled_flows", "queue_snapshots", "benchmark"):
        if name in out:
            unknown = set(out[name]["account_id"]) - ids
            if unknown:
                msg = f"{name}: {len(unknown)} account_ids not in accounts (e.g. {sorted(unknown)[:3]})"
                if strict:
                    raise ValueError(msg)
                log.warning(msg + " — rows dropped")
                out[name] = out[name][out[name]["account_id"].isin(ids)]

    if "transactions" in out:
        tx = out["transactions"]
        if "flow_class" not in tx.columns:
            tx["flow_class"] = "client"
        tx["flow_class"] = tx["flow_class"].fillna("client")
        bad = set(tx["flow_class"]) - FLOW_CLASSES
        if bad:
            raise ValueError(f"transactions: unknown flow_class {bad}; expected {FLOW_CLASSES}")
        out["transactions"] = tx

    if "balances" in out:
        b = out["balances"]
        if "balance_kind" in b.columns and b["balance_kind"].notna().any():
            bad = set(b["balance_kind"].dropna()) - BALANCE_KINDS
            if bad:
                raise ValueError(f"balances: unknown balance_kind {bad}")
        dup = b.duplicated(["timestamp", "account_id"] + (["balance_kind"] if "balance_kind" in b else []))
        if dup.any():
            log.warning("balances: %d duplicate observations, keeping last", int(dup.sum()))
            out["balances"] = b[~dup.values | ~dup.duplicated(keep="last").values]

    if "scheduled_flows" in out:
        sf = out["scheduled_flows"]
        late = sf["known_at"] > sf["value_time"]
        if late.any():
            msg = f"scheduled_flows: {int(late.sum())} rows known_at > value_time (violates point-in-time)"
            if strict:
                raise ValueError(msg)
            log.warning(msg + " — rows dropped")
            out["scheduled_flows"] = sf[~late]
    return out


def describe(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Coverage summary per table: rows, accounts, date range — the first thing to
    check when a new extract arrives."""
    rows = []
    for name, df in tables.items():
        tcol = next((c for c in ("timestamp", "known_at", "origin") if c in df.columns), None)
        rows.append({"table": name, "rows": len(df),
                     "accounts": df["account_id"].nunique() if "account_id" in df else None,
                     "start": df[tcol].min() if tcol else None, "end": df[tcol].max() if tcol else None})
    return pd.DataFrame(rows).set_index("table")
