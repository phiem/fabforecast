"""Loaders that map source-system extracts onto the canonical tables in :mod:`schema`.

Supported inputs
----------------
* SWIFT MT940 (end-of-day statement) and MT942 (interim/intraday report)   -> balances + transactions
* ISO 20022 camt.053 (statement) and camt.052 (account report)              -> balances + transactions
* Payment-hub / GL cash extract (CSV/parquet, column mapping supplied)      -> transactions
* FX blotter, MM deposits, securities pipeline, standing orders (CSV)       -> scheduled_flows (point-in-time)
* Queue snapshots from the payment-hub liquidity manager (CSV)              -> queue_snapshots
* Treasurer's historical forecasts (CSV)                                     -> benchmark

The SWIFT and camt parsers cover the fields the model needs (balances, statement
lines with value date, sign, amount, transaction code, reference). They are not
full message validators; feed them the raw archive and check :func:`schema.describe`.

Timestamps: MT940/camt.053 closing balances are stamped at ``LoaderConfig.eod_hour``
on the statement date; MT942/camt.052 use the message's own date/time (:13D:, CreDtTm).
Statement lines carry only a date, so intraday transaction timing must come from the
payment hub or from MT942 lines, not from the EOD statement — see INTEGRATION_GUIDE.md.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class LoaderConfig:
    tz: Optional[str] = None            # e.g. "Europe/London"; source tz converted to naive local
    eod_hour: int = 18                  # hour assigned to EOD statement balances
    # map source account identifiers (:25: content, IBAN, ledger id) -> canonical account_id
    account_map: Dict[str, str] = field(default_factory=dict)
    # map SWIFT transaction codes / hub product codes -> flow_class
    flow_class_map: Dict[str, str] = field(default_factory=lambda: {
        "NTRF": "client", "NMSC": "client", "NCHK": "client", "NCLR": "client",
        "NFEX": "known", "NCOL": "known", "NDIV": "known", "NINT": "known", "NSEC": "known",
        "NCHG": "fee", "NCOM": "fee",
        "NSWP": "treasury", "NTRS": "treasury", "NLDP": "treasury",
    })
    default_flow_class: str = "client"


# --------------------------------------------------------------------------- helpers

def _map_account(raw: str, cfg: LoaderConfig) -> str:
    key = raw.strip()
    if key in cfg.account_map:
        return cfg.account_map[key]
    tail = key.split("/")[-1].strip()
    if tail in cfg.account_map:
        return cfg.account_map[tail]
    return key


def _swift_amount(s: str) -> float:
    return float(s.replace(",", "."))


def _swift_date(yymmdd: str) -> pd.Timestamp:
    return pd.Timestamp(f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}")


def _split_messages(text: str) -> List[str]:
    # messages separated by "-}" trailer, "$" or blank line; :20: starts a new one
    parts = re.split(r"\n(?=:20:)", text.replace("\r", ""))
    return [p for p in parts if ":20:" in p]


def _tags(msg: str) -> List[tuple]:
    """Return [(tag, content)] preserving order; multi-line content joined."""
    out, cur, buf = [], None, []
    for line in msg.split("\n"):
        m = re.match(r"^:(\d{2}[A-Z]?):(.*)$", line)
        if m:
            if cur:
                out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1), [m.group(2)]
        elif cur and line and not line.startswith("-"):
            buf.append(line)
    if cur:
        out.append((cur, "\n".join(buf)))
    return out


_LINE61 = re.compile(
    r"^(?P<vd>\d{6})(?P<ed>\d{4})?(?P<dc>R?[CD])(?P<fc>[A-Z])?(?P<amt>[\d,]+)"
    r"(?P<code>[A-Z][A-Z0-9]{3})(?P<ref>[^/\n]*)(//(?P<bref>[^\n]*))?")


# --------------------------------------------------------------------------- MT940 / MT942

def load_mt940(paths: Iterable[PathLike], cfg: LoaderConfig) -> Dict[str, pd.DataFrame]:
    """Parse MT940 (and MT950) files: closing/opening balances + statement lines."""
    bal_rows, tx_rows = [], []
    for p in paths:
        text = Path(p).read_text(errors="ignore")
        for msg in _split_messages(text):
            tags = _tags(msg)
            acct, ccy, ref = None, None, None
            for tag, body in tags:
                if tag == "20":
                    ref = body.strip()
                elif tag == "25" or tag == "25P":
                    acct = _map_account(body, cfg)
                elif tag in ("60F", "60M"):            # opening balance
                    dc, d, ccy, amt = body[0], body[1:7], body[7:10], body[10:]
                    if tag == "60F":
                        bal_rows.append({"timestamp": _swift_date(d).replace(hour=cfg.eod_hour) - pd.Timedelta(days=0),
                                         "account_id": acct, "balance": (1 if dc == "C" else -1) * _swift_amount(amt),
                                         "balance_kind": "opening", "source": "MT940"})
                elif tag in ("62F", "62M"):            # closing booked balance
                    dc, d, ccy, amt = body[0], body[1:7], body[7:10], body[10:]
                    bal_rows.append({"timestamp": _swift_date(d).replace(hour=cfg.eod_hour), "account_id": acct,
                                     "balance": (1 if dc == "C" else -1) * _swift_amount(amt),
                                     "balance_kind": "booked" if tag == "62F" else "closing", "source": "MT940"})
                elif tag == "64":                      # closing available balance
                    dc, d, ccy, amt = body[0], body[1:7], body[7:10], body[10:]
                    bal_rows.append({"timestamp": _swift_date(d).replace(hour=cfg.eod_hour), "account_id": acct,
                                     "balance": (1 if dc == "C" else -1) * _swift_amount(amt),
                                     "balance_kind": "available", "source": "MT940"})
                elif tag == "61":
                    m = _LINE61.match(body.replace("\n", " "))
                    if not m:
                        log.warning("unparsed :61: line in %s: %s", ref, body[:40]); continue
                    sign = -1 if m.group("dc") in ("D", "RC") else 1
                    code = m.group("code")
                    tx_rows.append({"timestamp": _swift_date(m.group("vd")).replace(hour=cfg.eod_hour),
                                    "value_date": _swift_date(m.group("vd")), "account_id": acct,
                                    "amount": sign * _swift_amount(m.group("amt")), "tx_type": code,
                                    "rail": "SWIFT", "flow_class": cfg.flow_class_map.get(code, cfg.default_flow_class),
                                    "reference": (m.group("ref") or "").strip(), "source_system": "MT940"})
    log.info("MT940: %d balance rows, %d statement lines", len(bal_rows), len(tx_rows))
    return {"balances": pd.DataFrame(bal_rows), "transactions": pd.DataFrame(tx_rows)}


def load_mt942(paths: Iterable[PathLike], cfg: LoaderConfig) -> Dict[str, pd.DataFrame]:
    """Parse MT942 interim reports: :13D: date/time gives intraday timing; :90D:/:90C:
    give totals; :61: lines give individual movements since the previous report."""
    bal_rows, tx_rows = [], []
    for p in paths:
        text = Path(p).read_text(errors="ignore")
        for msg in _split_messages(text):
            tags = _tags(msg)
            acct, ts = None, None
            for tag, body in tags:
                if tag in ("25", "25P"):
                    acct = _map_account(body, cfg)
                elif tag == "13D":                     # YYMMDDHHMM+HHMM
                    ts = _swift_date(body[:6]) + pd.Timedelta(hours=int(body[6:8]), minutes=int(body[8:10]))
                elif tag == "61" and ts is not None:
                    m = _LINE61.match(body.replace("\n", " "))
                    if not m:
                        continue
                    sign = -1 if m.group("dc") in ("D", "RC") else 1
                    code = m.group("code")
                    tx_rows.append({"timestamp": ts, "value_date": _swift_date(m.group("vd")), "account_id": acct,
                                    "amount": sign * _swift_amount(m.group("amt")), "tx_type": code, "rail": "SWIFT",
                                    "flow_class": cfg.flow_class_map.get(code, cfg.default_flow_class),
                                    "reference": (m.group("ref") or "").strip(), "source_system": "MT942"})
                elif tag == "86" and body.startswith("/BAL/") and ts is not None:
                    # optional convention: /BAL/<C|D><amount> intraday available balance in :86:
                    mm = re.match(r"/BAL/([CD])([\d,]+)", body)
                    if mm:
                        bal_rows.append({"timestamp": ts, "account_id": acct,
                                         "balance": (1 if mm.group(1) == "C" else -1) * _swift_amount(mm.group(2)),
                                         "balance_kind": "available", "source": "MT942"})
    log.info("MT942: %d balance rows, %d movements", len(bal_rows), len(tx_rows))
    return {"balances": pd.DataFrame(bal_rows), "transactions": pd.DataFrame(tx_rows)}


# --------------------------------------------------------------------------- camt.053 / camt.052

def _ns(tag: str, root: ET.Element) -> str:
    m = re.match(r"\{(.*)\}", root.tag)
    return f"{{{m.group(1)}}}{tag}" if m else tag


def load_camt(paths: Iterable[PathLike], cfg: LoaderConfig) -> Dict[str, pd.DataFrame]:
    """Parse camt.053 / camt.052. Balances from <Bal> (CLBD booked, CLAV available, OPBD
    opening, ITBD/ITAV interim); entries from <Ntry> with BookgDt/ValDt, CdtDbtInd, Amt,
    BkTxCd domain/family codes and EndToEndId."""
    bal_rows, tx_rows = [], []
    for p in paths:
        root = ET.parse(p).getroot()
        n = lambda t: _ns(t, root)
        for stmt in root.iter():
            if stmt.tag not in (n("Stmt"), n("Rpt")):
                continue
            acct_el = stmt.find(f"{n('Acct')}/{n('Id')}")
            raw = "".join(acct_el.itertext()).strip() if acct_el is not None else ""
            acct = _map_account(raw, cfg)
            cre = stmt.findtext(n("CreDtTm"))
            cre_ts = pd.Timestamp(cre).tz_localize(None) if cre else None
            for bal in stmt.findall(n("Bal")):
                kind = bal.findtext(f"{n('Tp')}/{n('CdOrPrtry')}/{n('Cd')}") or ""
                amt = float(bal.findtext(n("Amt")) or 0)
                sign = 1 if bal.findtext(n("CdtDbtInd")) == "CRDT" else -1
                d = bal.findtext(f"{n('Dt')}/{n('Dt')}") or bal.findtext(f"{n('Dt')}/{n('DtTm')}")
                ts = pd.Timestamp(d).tz_localize(None)
                if ts.hour == 0 and ts.minute == 0:
                    ts = ts.replace(hour=cfg.eod_hour)
                if kind in ("ITBD", "ITAV") and cre_ts is not None:
                    ts = cre_ts
                bk = {"CLBD": "booked", "CLAV": "available", "OPBD": "opening", "ITBD": "booked",
                      "ITAV": "available", "PRCD": "opening"}.get(kind)
                if bk:
                    bal_rows.append({"timestamp": ts, "account_id": acct, "balance": sign * amt,
                                     "balance_kind": bk, "source": stmt.tag.split("}")[-1]})
            for e in stmt.findall(n("Ntry")):
                amt = float(e.findtext(n("Amt")) or 0)
                sign = 1 if e.findtext(n("CdtDbtInd")) == "CRDT" else -1
                bd = e.findtext(f"{n('BookgDt')}/{n('DtTm')}") or e.findtext(f"{n('BookgDt')}/{n('Dt')}")
                vd = e.findtext(f"{n('ValDt')}/{n('Dt')}") or bd
                ts = pd.Timestamp(bd).tz_localize(None)
                if ts.hour == 0 and ts.minute == 0:
                    ts = ts.replace(hour=cfg.eod_hour)
                fam = e.findtext(f"{n('BkTxCd')}/{n('Domn')}/{n('Fmly')}/{n('Cd')}") or ""
                dom = e.findtext(f"{n('BkTxCd')}/{n('Domn')}/{n('Cd')}") or ""
                prtry = e.findtext(f"{n('BkTxCd')}/{n('Prtry')}/{n('Cd')}") or ""
                code = prtry or f"{dom}/{fam}"
                ref = e.findtext(f".//{n('EndToEndId')}") or e.findtext(n("AcctSvcrRef")) or ""
                tx_rows.append({"timestamp": ts, "value_date": pd.Timestamp(vd).tz_localize(None), "account_id": acct,
                                "amount": sign * amt, "tx_type": code, "rail": dom or "ISO",
                                "flow_class": cfg.flow_class_map.get(prtry, cfg.flow_class_map.get(fam, cfg.default_flow_class)),
                                "reference": ref, "source_system": stmt.tag.split("}")[-1]})
    log.info("camt: %d balance rows, %d entries", len(bal_rows), len(tx_rows))
    return {"balances": pd.DataFrame(bal_rows), "transactions": pd.DataFrame(tx_rows)}


# --------------------------------------------------------------------------- tabular extracts

def _read_table(path: PathLike) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)


def load_payment_hub(path: PathLike, cfg: LoaderConfig, column_map: Dict[str, str],
                     flow_class_col: Optional[str] = None, product_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Payment-hub / GL cash extract -> transactions.

    column_map: {canonical: source}, must cover timestamp, account_id, amount; optional
                value_date, tx_type, rail, counterparty_bic, reference, source_system.
    product_map: {source product code: flow_class}; overrides cfg.flow_class_map.
    Debits must already be negative, or pass a 'direction' column mapped to 'dc' with
    values in {C, D} / {CRDT, DBIT}.
    """
    src = _read_table(path)
    df = pd.DataFrame({k: src[v] for k, v in column_map.items() if v in src.columns})
    if "dc" in df.columns:
        sign = np.where(df["dc"].astype(str).str.upper().str[0] == "D", -1, 1)
        df["amount"] = df["amount"].abs() * sign
        df = df.drop(columns="dc")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if cfg.tz:
        df["timestamp"] = df["timestamp"].dt.tz_localize(cfg.tz, ambiguous="NaT").dt.tz_convert(None)
    df["account_id"] = df["account_id"].astype(str).map(lambda a: _map_account(a, cfg))
    pm = {**cfg.flow_class_map, **(product_map or {})}
    if flow_class_col and flow_class_col in src.columns:
        df["flow_class"] = src[flow_class_col].map(pm).fillna(cfg.default_flow_class)
    elif "tx_type" in df.columns:
        df["flow_class"] = df["tx_type"].map(pm).fillna(cfg.default_flow_class)
    else:
        df["flow_class"] = cfg.default_flow_class
    df["source_system"] = df.get("source_system", "payment_hub")
    log.info("payment hub: %d rows, flow_class mix %s", len(df), df["flow_class"].value_counts().to_dict())
    return df


def load_fx_blotter(path: PathLike, cfg: LoaderConfig, column_map: Dict[str, str],
                    settlement_hour: Dict[str, int] = None) -> pd.DataFrame:
    """FX trades -> scheduled_flows (two legs per trade, point-in-time).

    column_map keys: trade_time, value_date, buy_ccy, buy_amount, sell_ccy, sell_amount,
                     buy_account, sell_account, optional cancelled_at, reference.
    settlement_hour: {ccy: hour} at which legs are expected to hit the account (CLS
                     pay-out hour, or scheme convention); default 10.
    """
    src = _read_table(path)
    sh = settlement_hour or {}
    rows = []
    for _, r in src.iterrows():
        vd = pd.Timestamp(r[column_map["value_date"]])
        known = pd.Timestamp(r[column_map["trade_time"]])
        ref = r.get(column_map.get("reference", ""), "")
        canc = r.get(column_map.get("cancelled_at", ""), None)
        for side, sign in (("buy", 1), ("sell", -1)):
            ccy = r[column_map[f"{side}_ccy"]]
            rows.append({"account_id": _map_account(str(r[column_map[f"{side}_account"]]), cfg),
                         "known_at": known, "value_time": vd.replace(hour=sh.get(ccy, 10)),
                         "amount": sign * abs(float(r[column_map[f"{side}_amount"]])),
                         "flow_type": "fx_leg", "reference": ref,
                         "cancelled_at": pd.Timestamp(canc) if pd.notna(canc) else pd.NaT})
    df = pd.DataFrame(rows)
    log.info("fx blotter: %d legs from %d trades", len(df), len(src))
    return df


def load_scheduled_flows(path: PathLike, cfg: LoaderConfig, column_map: Dict[str, str],
                         flow_type: str) -> pd.DataFrame:
    """Generic known-future flows (MM maturities, securities pipeline, standing orders,
    salary/tax files, CLS pay-in schedule). column_map keys: account_id, known_at,
    value_time, amount; optional reference, cancelled_at."""
    src = _read_table(path)
    df = pd.DataFrame({k: src[v] for k, v in column_map.items() if v in src.columns})
    df["account_id"] = df["account_id"].astype(str).map(lambda a: _map_account(a, cfg))
    df["known_at"] = pd.to_datetime(df["known_at"]); df["value_time"] = pd.to_datetime(df["value_time"])
    df["flow_type"] = flow_type
    if "cancelled_at" in df.columns:
        df["cancelled_at"] = pd.to_datetime(df["cancelled_at"])
    return df


def load_queue_snapshots(path: PathLike, cfg: LoaderConfig, column_map: Dict[str, str]) -> pd.DataFrame:
    """Queue snapshots -> queue_snapshots. column_map keys: timestamp, account_id,
    queued_amount; optional held_amount, time_critical_amount, n_items. Amounts are
    positive values of outgoing payments not yet released."""
    src = _read_table(path)
    df = pd.DataFrame({k: src[v] for k, v in column_map.items() if v in src.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_id"] = df["account_id"].astype(str).map(lambda a: _map_account(a, cfg))
    return df


def load_benchmark(path: PathLike, cfg: LoaderConfig, column_map: Dict[str, str], source: str = "desk") -> pd.DataFrame:
    src = _read_table(path)
    df = pd.DataFrame({k: src[v] for k, v in column_map.items() if v in src.columns})
    df["origin"] = pd.to_datetime(df["origin"]); df["target_time"] = pd.to_datetime(df["target_time"])
    df["account_id"] = df["account_id"].astype(str).map(lambda a: _map_account(a, cfg))
    df["source"] = source
    return df


# --------------------------------------------------------------------------- combine

def combine_balances(frames: List[pd.DataFrame], prefer: str = "available") -> pd.DataFrame:
    """Merge balance observations from several sources into one series per account:
    prefer the requested balance_kind at each timestamp, fall back to booked."""
    b = pd.concat([f for f in frames if len(f)], ignore_index=True)
    b["balance_kind"] = b.get("balance_kind", pd.Series(index=b.index)).fillna("booked")
    rank = {prefer: 0, "booked": 1, "closing": 2, "opening": 3}
    b["_r"] = b["balance_kind"].map(rank).fillna(9)
    b = b.sort_values(["account_id", "timestamp", "_r"]).drop_duplicates(["account_id", "timestamp"], keep="first")
    return b.drop(columns="_r").reset_index(drop=True)


def combine_transactions(frames: List[pd.DataFrame], dedupe_on: tuple = ("account_id", "amount", "value_date", "reference")) -> pd.DataFrame:
    """Concatenate transactions from several sources and drop duplicates that appear in
    both intraday (MT942/hub) and EOD (MT940) feeds — prefer the intraday timestamp."""
    tx = pd.concat([f for f in frames if len(f)], ignore_index=True)
    tx["_intraday"] = tx["source_system"].isin(["MT942", "payment_hub", "Rpt"]).astype(int)
    tx = tx.sort_values("_intraday", ascending=False)
    keys = [k for k in dedupe_on if k in tx.columns]
    before = len(tx)
    tx = tx.drop_duplicates(keys, keep="first").drop(columns="_intraday").sort_values("timestamp").reset_index(drop=True)
    log.info("combine_transactions: %d -> %d rows after de-duplication on %s", before, len(tx), keys)
    return tx
