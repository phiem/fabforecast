"""Synthetic data generator.

Produces three tables mirroring what a treasury data warehouse would expose:

* ``accounts``      – account_id, account_type, currency, scale
* ``transactions``  – timestamp, account_id, amount (signed), tx_type
* ``balances``      – timestamp, account_id, balance (end-of-period)
* ``external``      – timestamp, fx_*, sofr, sonia, estr, vix

The generator embeds the structure real accounts show:
intraday payment cycles, day-of-week and month-end effects, cross-account
mirroring within a currency (nostro <-> agent bank), fat-tailed large wires,
and a configurable stress regime with elevated volatility.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ACCOUNT_SPECS = [
    # id,           type,           ccy,   scale (typical balance)
    ("CB_USD_FED",  "central_bank", "USD", 2.5e9),
    ("CB_EUR_ECB",  "central_bank", "EUR", 1.2e9),
    ("AGT_USD_JPM", "agent_bank",   "USD", 4.0e8),
    ("AGT_GBP_BARC", "agent_bank",  "GBP", 1.5e8),
    ("NOS_USD_CITI", "nostro",      "USD", 9.0e7),
    ("NOS_EUR_DB",  "nostro",       "EUR", 6.0e7),
    ("NOS_GBP_HSBC", "nostro",      "GBP", 3.0e7),
]


def _intraday_profile(hours: np.ndarray, acct_type: str) -> np.ndarray:
    """Expected net flow as a fraction of scale, by hour of day."""
    h = hours.astype(float)
    if acct_type == "central_bank":
        # settlement: large outflows early (RTGS), inflows near end-of-day
        return -0.010 * np.exp(-((h - 9) ** 2) / 2) + 0.011 * np.exp(-((h - 16) ** 2) / 3)
    if acct_type == "agent_bank":
        return -0.020 * np.exp(-((h - 10) ** 2) / 4) + 0.018 * np.exp(-((h - 15) ** 2) / 4)
    # nostro: FX settlement clusters (CLS-like) mid-morning, sweeps late afternoon
    return -0.030 * np.exp(-((h - 8) ** 2) / 1.5) + 0.028 * np.exp(-((h - 17) ** 2) / 2)


def generate(start: str = "2024-09-01", end: str = "2026-02-28",
             stress_start: str = "2025-11-10", stress_end: str = "2025-12-05",
             seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    is_open = (idx.dayofweek < 5) & (idx.hour >= 7) & (idx.hour < 19)
    open_idx = idx[is_open]
    n = len(open_idx)

    stress = (open_idx >= stress_start) & (open_idx < stress_end)
    month_end = open_idx.is_month_end | (open_idx.day >= 28)
    quarter_end = month_end & open_idx.month.isin([3, 6, 9, 12])

    # external market data (hourly, business hours; ffilled by pipeline over gaps)
    vix = 15 + np.cumsum(rng.normal(0, 0.15, n)); vix = np.clip(vix, 10, 60)
    vix = np.where(stress, vix + 18 + rng.normal(0, 2, n), vix)
    sofr = 4.3 + np.cumsum(rng.normal(0, 0.0015, n)) - 0.0004 * np.arange(n) / 12
    external = pd.DataFrame({
        "timestamp": open_idx,
        "fx_EURUSD": 1.08 + np.cumsum(rng.normal(0, 0.0006, n)),
        "fx_GBPUSD": 1.27 + np.cumsum(rng.normal(0, 0.0007, n)),
        "sofr": sofr, "sonia": sofr + 0.6 + np.cumsum(rng.normal(0, 0.001, n)),
        "estr": sofr - 1.0 + np.cumsum(rng.normal(0, 0.001, n)),
        "vix": vix,
    })

    accounts = pd.DataFrame(ACCOUNT_SPECS, columns=["account_id", "account_type", "currency", "scale"])
    tx_frames, bal_frames = [], []
    # shared currency shocks so same-ccy accounts co-move (nostro pays -> agent receives)
    ccy_shock = {c: rng.normal(0, 1, n) for c in accounts.currency.unique()}

    for _, a in accounts.iterrows():
        scale = a.scale
        prof = _intraday_profile(open_idx.hour.values, a.account_type)
        dow_eff = np.where(open_idx.dayofweek == 0, 0.004, 0) - np.where(open_idx.dayofweek == 4, 0.003, 0)
        me_eff = np.where(month_end, -0.006, 0) + np.where(quarter_end, -0.006, 0)
        vol = 0.004 * (1 + 2.5 * stress)
        noise = rng.normal(0, 1, n) * vol
        # cross-account structure: nostro outflow tends to land in same-ccy agent bank
        sign = {"nostro": -1.0, "agent_bank": 0.6, "central_bank": 0.3}[a.account_type]
        cross = sign * 0.004 * ccy_shock[a.currency]
        # fat-tailed large wires (flagged as outliers downstream)
        wires = np.zeros(n)
        wire_pos = rng.choice(n, size=max(3, n // 400), replace=False)
        wires[wire_pos] = rng.standard_t(2, len(wire_pos)) * 0.05
        # count of transactions per hour
        base_rate = {"central_bank": 60, "agent_bank": 120, "nostro": 40}[a.account_type]
        tx_count = rng.poisson(base_rate * (1 + 0.8 * np.abs(prof) / 0.02) * (1 + 0.5 * stress))

        net_frac = prof + dow_eff + me_eff + noise + cross + wires
        # mean reversion toward scale so balances stay bounded
        bal = np.empty(n); level = scale
        for t in range(n):
            level = level + net_frac[t] * scale + 0.02 * (scale - level)
            bal[t] = level
        bal_frames.append(pd.DataFrame({"timestamp": open_idx, "account_id": a.account_id, "balance": bal}))

        # explode hourly net flow into individual transactions
        for t in range(n):
            k = int(tx_count[t]); net = net_frac[t] * scale
            if k == 0:
                continue
            gross = np.abs(rng.exponential(scale * 0.0015, k)) * rng.choice([-1, 1], k)
            gross += (net - gross.sum()) / k
            if wires[t] != 0:  # ensure the wire shows as a single big ticket
                gross[0] += wires[t] * scale
            tx_frames.append(pd.DataFrame({
                "timestamp": open_idx[t] + pd.to_timedelta(rng.integers(0, 3600, k), unit="s"),
                "account_id": a.account_id, "amount": gross,
                "tx_type": rng.choice(["wire", "ach", "fx_settle", "sweep"], k, p=[0.3, 0.4, 0.2, 0.1]),
            }))

    balances = pd.concat(bal_frames, ignore_index=True)
    transactions = pd.concat(tx_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    # inject some missing balance observations (feed failures)
    drop = rng.choice(len(balances), size=len(balances) // 200, replace=False)
    balances.loc[drop, "balance"] = np.nan
    log.info("synthetic: %d balance rows, %d transactions, %d accounts",
             len(balances), len(transactions), len(accounts))
    return {"accounts": accounts, "transactions": transactions,
            "balances": balances, "external": external,
            "stress_window": (pd.Timestamp(stress_start), pd.Timestamp(stress_end))}
