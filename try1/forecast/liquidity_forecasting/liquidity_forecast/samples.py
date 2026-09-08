"""Write sample source-system files (MT940, MT942, camt.053, payment-hub CSV, FX blotter,
queue snapshots, desk benchmark) from the synthetic tables. Used by
run_demo_real_data.py to exercise the loaders end-to-end; also handy as format examples
when talking to the teams who own the real feeds."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CODES = {"wire": "NTRF", "ach": "NMSC", "fx_settle": "NFEX", "sweep": "NSWP"}


def _amt(x: float) -> str:
    return f"{abs(x):.2f}".replace(".", ",")


def _by_day(tx: pd.DataFrame, cap: int) -> dict:
    t = tx.assign(day=tx.timestamp.dt.normalize())
    return {k: g.head(cap) for k, g in t.groupby(["account_id", "day"])}


def _by_hour(tx: pd.DataFrame, cap: int) -> dict:
    t = tx.assign(hr=tx.timestamp.dt.floor("h"))
    return {k: g.head(cap) for k, g in t.groupby(["account_id", "hr"])}


def write_mt940(bal: pd.DataFrame, tx: pd.DataFrame, accounts: pd.DataFrame, out: Path) -> Path:
    lines = []
    ccy = accounts.set_index("account_id")["currency"]
    txd = _by_day(tx, 40)
    for (acct, day), b in bal.assign(day=bal.timestamp.dt.normalize()).groupby(["account_id", "day"]):
        closing = b.sort_values("timestamp")["balance"].iloc[-1]
        opening = b.sort_values("timestamp")["balance"].iloc[0]
        if np.isnan(closing) or np.isnan(opening):
            continue
        d = day.strftime("%y%m%d")
        t = txd.get((acct, day), tx.iloc[0:0])
        lines += [f":20:STMT{d}{acct[:6]}", f":25:IBAN00{acct}", f":28C:{day.dayofyear}/1",
                  f":60F:{'C' if opening >= 0 else 'D'}{d}{ccy[acct]}{_amt(opening)}"]
        for _, r in t.iterrows():          # capped lines per statement for the sample
            lines.append(f":61:{d}{d[2:]}{'C' if r.amount >= 0 else 'D'}{_amt(r.amount)}{CODES[r.tx_type]}NONREF//{r.name}")
            lines.append(f":86:{r.tx_type.upper()} sample narrative")
        lines += [f":62F:{'C' if closing >= 0 else 'D'}{d}{ccy[acct]}{_amt(closing)}",
                  f":64:{'C' if closing >= 0 else 'D'}{d}{ccy[acct]}{_amt(closing)}", "-}"]
    out.write_text("\n".join(lines))
    return out


def write_mt942(bal: pd.DataFrame, tx: pd.DataFrame, accounts: pd.DataFrame, out: Path) -> Path:
    lines = []
    txh = _by_hour(tx[tx.account_id.isin(bal.account_id.unique())], 20)
    for (acct, ts), b in bal.groupby(["account_id", "timestamp"]):
        v = b["balance"].iloc[0]
        d = ts.strftime("%y%m%d")
        lines += [f":20:INTR{ts.strftime('%y%m%d%H%M')}{acct[:4]}", f":25:IBAN00{acct}", f":28C:{ts.dayofyear}/{ts.hour}",
                  f":13D:{ts.strftime('%y%m%d%H%M')}+0000"]
        t = txh.get((acct, ts), tx.iloc[0:0])
        for _, r in t.iterrows():
            lines.append(f":61:{d}{d[2:]}{'C' if r.amount >= 0 else 'D'}{_amt(r.amount)}{CODES[r.tx_type]}NONREF//{r.name}")
        if not np.isnan(v):
            lines.append(f":86:/BAL/{'C' if v >= 0 else 'D'}{_amt(v)}")
        lines.append("-}")
    out.write_text("\n".join(lines))
    return out


def write_camt053(bal: pd.DataFrame, tx: pd.DataFrame, accounts: pd.DataFrame, out: Path) -> Path:
    ns = "urn:iso:std:iso:20022:tech:xsd:camt.053.001.08"
    ccy = accounts.set_index("account_id")["currency"]
    parts = [f'<?xml version="1.0" encoding="UTF-8"?><Document xmlns="{ns}"><BkToCstmrStmt>']
    txd = _by_day(tx[tx.account_id.isin(bal.account_id.unique())], 40)
    for (acct, day), b in bal.assign(day=bal.timestamp.dt.normalize()).groupby(["account_id", "day"]):
        closing = b.sort_values("timestamp")["balance"].iloc[-1]
        if np.isnan(closing):
            continue
        parts.append(f"<Stmt><Id>S{day:%Y%m%d}{acct}</Id><CreDtTm>{day:%Y-%m-%d}T18:05:00</CreDtTm>"
                     f"<Acct><Id><IBAN>IBAN00{acct}</IBAN></Id><Ccy>{ccy[acct]}</Ccy></Acct>"
                     f"<Bal><Tp><CdOrPrtry><Cd>CLBD</Cd></CdOrPrtry></Tp><Amt Ccy=\"{ccy[acct]}\">{abs(closing):.2f}</Amt>"
                     f"<CdtDbtInd>{'CRDT' if closing >= 0 else 'DBIT'}</CdtDbtInd><Dt><Dt>{day:%Y-%m-%d}</Dt></Dt></Bal>"
                     f"<Bal><Tp><CdOrPrtry><Cd>CLAV</Cd></CdOrPrtry></Tp><Amt Ccy=\"{ccy[acct]}\">{abs(closing):.2f}</Amt>"
                     f"<CdtDbtInd>{'CRDT' if closing >= 0 else 'DBIT'}</CdtDbtInd><Dt><Dt>{day:%Y-%m-%d}</Dt></Dt></Bal>")
        t = txd.get((acct, day), tx.iloc[0:0])
        for _, r in t.iterrows():
            fam = {"wire": "RCDT", "ach": "ICDT", "fx_settle": "FEX", "sweep": "TRSF"}[r.tx_type]  # illustrative codes
            parts.append(f"<Ntry><Amt Ccy=\"{ccy[acct]}\">{abs(r.amount):.2f}</Amt><CdtDbtInd>{'CRDT' if r.amount >= 0 else 'DBIT'}</CdtDbtInd>"
                         f"<Sts><Cd>BOOK</Cd></Sts><BookgDt><DtTm>{r.timestamp:%Y-%m-%dT%H:%M:%S}</DtTm></BookgDt>"
                         f"<ValDt><Dt>{day:%Y-%m-%d}</Dt></ValDt><BkTxCd><Domn><Cd>PMNT</Cd><Fmly><Cd>{fam}</Cd></Fmly></Domn>"
                         f"<Prtry><Cd>{CODES[r.tx_type]}</Cd></Prtry></BkTxCd><NtryDtls><TxDtls><Refs><EndToEndId>E2E{r.name}</EndToEndId></Refs></TxDtls></NtryDtls></Ntry>")
        parts.append("</Stmt>")
    parts.append("</BkToCstmrStmt></Document>")
    out.write_text("".join(parts))
    return out


def write_payment_hub_csv(tx: pd.DataFrame, out: Path) -> Path:
    hub = pd.DataFrame({
        "BOOKING_TS": tx.timestamp, "LEDGER_ACCT": "LGR-" + tx.account_id, "AMT": tx.amount.abs(),
        "DR_CR": np.where(tx.amount < 0, "D", "C"), "VALUE_DT": tx.timestamp.dt.normalize(),
        "PRODUCT": tx.tx_type.str.upper(), "RAIL": np.where(tx.tx_type == "wire", "FEDWIRE", "ACH"),
        "CPTY_BIC": "CPTYXXXX", "REF": "HUB" + tx.index.astype(str),
    })
    hub.to_csv(out, index=False)
    return out


def write_fx_blotter(tx: pd.DataFrame, accounts: pd.DataFrame, out: Path, rng: np.random.Generator) -> Path:
    """Turn a sample of fx_settle transactions into two-legged trades known one day ahead."""
    fx = tx[tx.tx_type == "fx_settle"].sample(frac=0.02, random_state=1)
    ccy = accounts.set_index("account_id")["currency"]
    other = {"USD": "NOS_EUR_DB", "EUR": "NOS_USD_CITI", "GBP": "NOS_USD_CITI"}
    rows = []
    for i, r in fx.iterrows():
        buy_acct, sell_acct = (r.account_id, other[ccy[r.account_id]]) if r.amount > 0 else (other[ccy[r.account_id]], r.account_id)
        rows.append({"TRADE_TS": r.timestamp - pd.Timedelta(days=1), "VALUE_DATE": r.timestamp.normalize(),
                     "BUY_CCY": ccy[buy_acct], "BUY_AMT": abs(r.amount), "SELL_CCY": ccy[sell_acct],
                     "SELL_AMT": abs(r.amount) * 0.92, "BUY_ACCT": buy_acct, "SELL_ACCT": sell_acct,
                     "TRADE_ID": f"FX{i}", "CANCELLED_TS": (r.timestamp - pd.Timedelta(hours=20)) if rng.random() < 0.03 else ""})
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_queue_snapshots(panel_like: pd.DataFrame, out: Path, rng: np.random.Generator) -> Path:
    """Hourly queue snapshots: queued value roughly proportional to the coming outflow."""
    q = panel_like[["timestamp", "account_id", "balance"]].copy()
    q["QUEUED"] = np.abs(rng.normal(0.01, 0.004, len(q))) * q["balance"].abs()
    q["HELD"] = q["QUEUED"] * rng.uniform(0, 0.3, len(q))
    q["TIME_CRIT"] = q["QUEUED"] * rng.uniform(0, 0.2, len(q))
    q["N"] = rng.poisson(12, len(q))
    q.rename(columns={"timestamp": "SNAP_TS", "account_id": "ACCT"}).drop(columns="balance").to_csv(out, index=False)
    return out


def write_desk_benchmark(bal: pd.DataFrame, out: Path) -> Path:
    """Desk forecast proxy: seasonal naive (same hour, previous business day) for h=4."""
    b = bal.dropna().sort_values(["account_id", "timestamp"]).copy()
    b["origin"] = b["timestamp"]
    b["target_time"] = b.groupby("account_id")["timestamp"].shift(-4)
    b["forecast"] = b.groupby("account_id")["balance"].shift(12 - 4)   # 12 open hours per day
    b = b.dropna(subset=["target_time", "forecast"])
    b[["origin", "target_time", "account_id", "forecast"]].rename(
        columns={"origin": "FCST_TIME", "target_time": "FOR_TIME", "account_id": "ACCOUNT", "forecast": "DESK_FCST"}).to_csv(out, index=False)
    return out
