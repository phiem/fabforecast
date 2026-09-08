"""Seasonality extension: per-currency holidays, moving-date treasury events, trend.

Plugs into the existing pipeline as an extra feature block:

    from liquidity_forecast.seasonality import SeasonalityConfig, build_seasonal_features
    fc = LiquidityForecaster(cfg, extra_features=build_seasonal_features(SeasonalityConfig()))

Three groups of features are produced, all causal (calendar lookups + backward-looking
regressions), all aligned to the (account_id, timestamp) panel index:

1. **Per-currency holiday calendar**  — USD (Fed/Fedwire), EUR (TARGET2), GBP (UK bank
   holidays) built in from rules, any currency extendable with explicit dates.
   ``ccy_holiday``            account's own currency is closed today (balance frozen)
   ``ccy_pre_holiday``        last open day before a currency holiday (pre-funding)
   ``ccy_post_holiday``       first open day after (catch-up settlement)
   ``days_to_ccy_holiday``    open days until the next closure
   ``counter_ccy_holiday``    some *other* currency in the book is closed (suppresses FX legs)

2. **Moving-date event calendar**  — the drivers that a plain (hour, weekday) profile misses:
   IMM dates, US/UK tax dates, payroll runs, Treasury coupon/settlement dates, reserve
   maintenance period ends, plus user-supplied events. For each event ``E``:
   ``is_E``, ``days_to_E`` (0..lookahead, capped), ``days_since_E`` (capped).

3. **Trend**  — the hybrid otherwise has seasonal + residual only.
   ``trend_slope_{W}``  causal rolling OLS slope of scaled balance over W periods
   ``trend_resid_{W}``  current scaled balance minus the rolling trend line
   ``stl_decompose()``  is a *diagnostic* STL (not causal) for reporting.

Also exposes ``account_open_mask()`` so the data layer can drop rows on an account's
own holidays instead of carrying frozen balances (recommended once holidays are
per-currency; see README section "Currency calendars").
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- date rules

def easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19; b, c = divmod(year, 100); d, e = divmod(b, 4)
    f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) weekday (Mon=0) of month; n=-1 -> last."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta((weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(1)
    return d - timedelta((d.weekday() - weekday) % 7)


def observed(d: date, rule: str = "us") -> date:
    """Weekend substitution. us: Sat->Fri, Sun->Mon (Fedwire: Saturday holidays are NOT
    observed Friday, so Sat stays Sat). uk: Sat/Sun -> next Monday(s)."""
    if rule == "us":
        return d + timedelta(1) if d.weekday() == 6 else d
    if rule == "uk":
        while d.weekday() >= 5:
            d += timedelta(1)
        return d
    return d


def us_fed_holidays(year: int) -> Dict[str, date]:
    return {
        "new_year": observed(date(year, 1, 1)), "mlk": nth_weekday(year, 1, 0, 3),
        "presidents": nth_weekday(year, 2, 0, 3), "memorial": nth_weekday(year, 5, 0, -1),
        "juneteenth": observed(date(year, 6, 19)), "independence": observed(date(year, 7, 4)),
        "labor": nth_weekday(year, 9, 0, 1), "columbus": nth_weekday(year, 10, 0, 2),
        "veterans": observed(date(year, 11, 11)), "thanksgiving": nth_weekday(year, 11, 3, 4),
        "christmas": observed(date(year, 12, 25)),
    }


def target2_holidays(year: int) -> Dict[str, date]:
    e = easter(year)
    return {"new_year": date(year, 1, 1), "good_friday": e - timedelta(2), "easter_monday": e + timedelta(1),
            "labour_day": date(year, 5, 1), "christmas": date(year, 12, 25), "boxing_day": date(year, 12, 26)}


def uk_bank_holidays(year: int) -> Dict[str, date]:
    e = easter(year)
    xmas = observed(date(year, 12, 25), "uk"); boxing = observed(date(year, 12, 26), "uk")
    if boxing == xmas:
        boxing += timedelta(1)
    return {"new_year": observed(date(year, 1, 1), "uk"), "good_friday": e - timedelta(2),
            "easter_monday": e + timedelta(1), "early_may": nth_weekday(year, 5, 0, 1),
            "spring": nth_weekday(year, 5, 0, -1), "summer": nth_weekday(year, 8, 0, -1),
            "christmas": xmas, "boxing_day": boxing}


BUILTIN_CALENDARS: Dict[str, Callable[[int], Dict[str, date]]] = {
    "USD": us_fed_holidays, "EUR": target2_holidays, "GBP": uk_bank_holidays,
}

# --------------------------------------------------------------------------- events

def imm_dates(year: int) -> List[date]:
    """3rd Wednesday of Mar/Jun/Sep/Dec — futures/swap rolls, large margin & settlement flows."""
    return [nth_weekday(year, m, 2, 3) for m in (3, 6, 9, 12)]


def us_tax_dates(year: int) -> List[date]:
    """Corporate estimated-tax & individual filing dates (Treasury General Account inflows)."""
    return [date(year, 1, 15), date(year, 3, 15), date(year, 4, 15), date(year, 6, 15),
            date(year, 9, 15), date(year, 10, 15), date(year, 12, 15)]


def uk_tax_dates(year: int) -> List[date]:
    """PAYE 22nd monthly; corporation tax quarter days; self-assessment 31 Jan / 31 Jul."""
    return [date(year, m, 22) for m in range(1, 13)] + [date(year, 1, 31), date(year, 7, 31)]


def payroll_dates(year: int) -> List[date]:
    out = []
    for m in range(1, 13):
        out.append(date(year, m, 15))
        out.append(nth_weekday(year, m, 4, -1) if nth_weekday(year, m, 4, -1).month == m
                   else date(year + (m == 12), m % 12 + 1, 1) - timedelta(1))
    return out


def ust_coupon_settle_dates(year: int) -> List[date]:
    """US Treasury coupon/redemption cash dates: 15th and month-end."""
    return [date(year, m, 15) for m in range(1, 13)] + \
           [date(year + (m == 12), m % 12 + 1, 1) - timedelta(1) for m in range(1, 13)]


BUILTIN_EVENTS: Dict[str, Callable[[int], List[date]]] = {
    "imm": imm_dates, "us_tax": us_tax_dates, "uk_tax": uk_tax_dates,
    "payroll": payroll_dates, "ust_coupon": ust_coupon_settle_dates,
}

# --------------------------------------------------------------------------- config

@dataclass
class SeasonalityConfig:
    # currencies -> extra explicit holiday dates (ISO), merged with built-in rules
    extra_holidays: Dict[str, List[str]] = field(default_factory=dict)
    # currencies with no built-in rules get *only* these dates (or none)
    events: List[str] = field(default_factory=lambda: list(BUILTIN_EVENTS))
    # user events: name -> list of ISO dates (e.g. reserve maintenance period ends, dividend dates)
    custom_events: Dict[str, List[str]] = field(default_factory=dict)
    # restrict an event to currencies (default: all). e.g. {"us_tax": ["USD"], "uk_tax": ["GBP"]}
    event_currencies: Dict[str, List[str]] = field(default_factory=lambda: {
        "us_tax": ["USD"], "uk_tax": ["GBP"], "ust_coupon": ["USD"]})
    event_lookahead_days: int = 10          # cap for days_to_*
    event_lookback_days: int = 5            # cap for days_since_*
    trend_windows: List[int] = field(default_factory=lambda: [24 * 5, 24 * 20])   # periods
    trend_windows_daily: List[int] = field(default_factory=lambda: [10, 30])
    roll_to_business_day: bool = True       # move weekend/holiday event dates to next open day


# --------------------------------------------------------------------------- calendars

class CurrencyCalendar:
    """Holiday sets per currency over a year range."""

    def __init__(self, currencies: Iterable[str], years: Iterable[int], cfg: SeasonalityConfig):
        self.holidays: Dict[str, pd.DatetimeIndex] = {}
        for ccy in currencies:
            dates = set()
            rule = BUILTIN_CALENDARS.get(ccy)
            if rule is None and ccy not in cfg.extra_holidays:
                log.warning("no holiday rules for currency %s; treating as always open", ccy)
            for y in years:
                if rule:
                    dates.update(rule(y).values())
            dates.update(pd.to_datetime(cfg.extra_holidays.get(ccy, [])).date)
            self.holidays[ccy] = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in dates))
            log.info("calendar %s: %d holidays", ccy, len(dates))

    def is_holiday(self, ccy: str, days: pd.DatetimeIndex) -> np.ndarray:
        return days.normalize().isin(self.holidays.get(ccy, pd.DatetimeIndex([])))

    def open_days(self, ccy: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        d = pd.bdate_range(start.normalize(), end.normalize())
        return d[~self.is_holiday(ccy, d)]

    def roll_forward(self, ccy: str, d: pd.Timestamp) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        hol = self.holidays.get(ccy, pd.DatetimeIndex([]))
        while d.weekday() >= 5 or d in hol:
            d += pd.Timedelta(days=1)
        return d


def account_open_mask(panel: pd.DataFrame, cal: CurrencyCalendar) -> np.ndarray:
    """True where the account's own currency is open. Feed to ``panel[mask]`` in the data
    layer to drop frozen-balance rows on currency holidays."""
    ts = panel.index.get_level_values("timestamp")
    ccy = panel["currency"].values
    mask = np.ones(len(panel), dtype=bool)
    for c in np.unique(ccy):
        sel = ccy == c
        mask[sel] = ~cal.is_holiday(c, ts[sel])
    return mask


# --------------------------------------------------------------------------- features

def _holiday_features(panel: pd.DataFrame, cal: CurrencyCalendar) -> pd.DataFrame:
    ts = panel.index.get_level_values("timestamp")
    days = ts.normalize()
    ccy = panel["currency"].values
    all_ccys = np.unique(ccy)
    out = pd.DataFrame(index=panel.index)
    out["ccy_holiday"] = 0; out["ccy_pre_holiday"] = 0; out["ccy_post_holiday"] = 0
    out["days_to_ccy_holiday"] = 30.0; out["counter_ccy_holiday"] = 0
    start, end = days.min() - pd.Timedelta(days=40), days.max() + pd.Timedelta(days=40)
    for c in all_ccys:
        sel = ccy == c
        d = days[sel]
        hol = cal.holidays.get(c, pd.DatetimeIndex([]))
        out.loc[sel, "ccy_holiday"] = d.isin(hol).astype(int)
        opn = cal.open_days(c, start, end)
        wk_hol = hol[(hol >= start) & (hol <= end) & (hol.weekday < 5)]   # weekend holidays are already closed
        pos_d = opn.searchsorted(d)                       # open-day index of d (or of next open day if d closed)
        if len(wk_hol):
            nxt = np.clip(wk_hol.searchsorted(d, side="left"), 0, len(wk_hol) - 1)  # first holiday >= d
            pos_h = opn.searchsorted(wk_hol[nxt])          # number of open days strictly before that holiday
            days_to = np.where(wk_hol[nxt] >= d, pos_h - pos_d, 30)
        else:
            days_to = np.full(len(d), 30)
        out.loc[sel, "days_to_ccy_holiday"] = np.clip(days_to, 0, 30).astype(float)
        out.loc[sel, "ccy_pre_holiday"] = (days_to == 1).astype(int)     # next open day is a holiday
        # post: previous calendar day(s) back to last open day include a weekday holiday
        prev_open = opn[np.clip(pos_d - 1, 0, len(opn) - 1)]
        gap = pd.DatetimeIndex(d) - pd.DatetimeIndex(prev_open)
        has_hol_in_gap = np.array([hol[(hol > p) & (hol < q) & (hol.weekday < 5)].size > 0
                                   for p, q in zip(prev_open, d)])
        out.loc[sel, "ccy_post_holiday"] = (has_hol_in_gap & (gap.days >= 1) & ~d.isin(hol)).astype(int)
        # some other currency closed today
        other = np.zeros(sel.sum(), dtype=int)
        for o in all_ccys:
            if o != c:
                other |= d.isin(cal.holidays.get(o, pd.DatetimeIndex([]))).astype(int)
        out.loc[sel, "counter_ccy_holiday"] = other
    return out


def _event_features(panel: pd.DataFrame, cal: CurrencyCalendar, cfg: SeasonalityConfig) -> pd.DataFrame:
    ts = panel.index.get_level_values("timestamp")
    days = ts.normalize()
    ccy = panel["currency"].values
    years = range(days.min().year - 1, days.max().year + 2)
    out = pd.DataFrame(index=panel.index)

    def dates_for(name: str) -> List[date]:
        if name in cfg.custom_events:
            return list(pd.to_datetime(cfg.custom_events[name]).date)
        return [d for y in years for d in BUILTIN_EVENTS[name](y)]

    for name in list(cfg.events) + list(cfg.custom_events):
        allowed = cfg.event_currencies.get(name)
        is_e = np.zeros(len(panel), dtype=int)
        d_to = np.full(len(panel), float(cfg.event_lookahead_days))
        d_since = np.full(len(panel), float(cfg.event_lookback_days))
        for c in np.unique(ccy):
            sel = ccy == c
            if allowed and c not in allowed:
                continue
            ev = pd.DatetimeIndex(sorted({pd.Timestamp(x) for x in dates_for(name)}))
            if cfg.roll_to_business_day:
                ev = pd.DatetimeIndex(sorted({cal.roll_forward(c, x) for x in ev}))
            d = days[sel]
            is_e[sel] = d.isin(ev).astype(int)
            nxt = np.searchsorted(ev.values, d.values, side="left")
            prv = nxt - 1
            nxt_d = ev.values[np.clip(nxt, 0, len(ev) - 1)]
            prv_d = ev.values[np.clip(prv, 0, len(ev) - 1)]
            d_to[sel] = np.clip((nxt_d - d.values) / np.timedelta64(1, "D"), 0, cfg.event_lookahead_days)
            d_since[sel] = np.clip((d.values - prv_d) / np.timedelta64(1, "D"), 0, cfg.event_lookback_days)
        out[f"is_{name}"] = is_e
        out[f"days_to_{name}"] = d_to
        out[f"days_since_{name}"] = d_since
    return out


def _rolling_slope(y: np.ndarray, w: int) -> np.ndarray:
    """Causal OLS slope of y on time over trailing window w (vectorised, O(n))."""
    n = len(y); out = np.full(n, np.nan)
    if n < w:
        return out
    x = np.arange(w, dtype=float); x -= x.mean()
    sxx = (x ** 2).sum()
    yw = np.lib.stride_tricks.sliding_window_view(y, w)
    slope = (yw * x).sum(axis=1) / sxx
    out[w - 1:] = slope
    return out


def _trend_features(panel: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    sb = (panel["balance"] / panel["scale"]).astype(float)
    for w in windows:
        slope = np.full(len(panel), np.nan); resid = np.full(len(panel), np.nan)
        for acct, s in sb.groupby(level="account_id", sort=False):
            idx = np.where(panel.index.get_level_values("account_id") == acct)[0]
            y = s.values
            sl = _rolling_slope(y, w)
            mean = pd.Series(y).rolling(w, min_periods=w).mean().values
            # trend line value at the window end = mean + slope*(w-1)/2
            resid_a = y - (mean + sl * (w - 1) / 2)
            slope[idx] = sl; resid[idx] = resid_a
        out[f"trend_slope_{w}"] = slope
        out[f"trend_resid_{w}"] = resid
    return out


def build_seasonal_features(cfg: Optional[SeasonalityConfig] = None) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    """Return an ``extra_features(panel, granularity) -> DataFrame`` callable for the pipeline."""
    cfg = cfg or SeasonalityConfig()

    def _fn(panel: pd.DataFrame, granularity: str) -> pd.DataFrame:
        ts = panel.index.get_level_values("timestamp")
        years = range(ts.min().year - 1, ts.max().year + 2)
        cal = CurrencyCalendar(panel["currency"].unique(), years, cfg)
        parts = [_holiday_features(panel, cal), _event_features(panel, cal, cfg)]
        windows = cfg.trend_windows if granularity == "intraday" else cfg.trend_windows_daily
        parts.append(_trend_features(panel, windows))
        X = pd.concat(parts, axis=1)
        log.info("seasonality features[%s]: %d cols", granularity, X.shape[1])
        return X

    return _fn


# --------------------------------------------------------------------------- diagnostics

def stl_decompose(panel: pd.DataFrame, account_id: str, period: int = 5, robust: bool = True) -> pd.DataFrame:
    """Diagnostic STL (trend + seasonal + resid) of one account's *daily* scaled balance.
    NOT causal — use for reporting and for deciding whether a trend term is needed,
    never as a model feature."""
    from statsmodels.tsa.seasonal import STL
    s = (panel.xs(account_id)["balance"] / panel.xs(account_id)["scale"]).astype(float)
    res = STL(s.values, period=period, robust=robust).fit()
    return pd.DataFrame({"observed": s.values, "trend": res.trend, "seasonal": res.seasonal,
                         "resid": res.resid}, index=s.index)
