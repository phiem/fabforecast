"""Feature engineering on the cleaned panel.

All features are **causal**: at row (account, t) they use only information
with timestamp <= t. Targets are the *scaled forward change* in balance:

    y_h(t) = (balance(t+h) - balance(t)) / scale(t)

Predicting the change (not the level) and dividing by the account scale is
what lets a single model generalise across account types and currencies.
The forecast level is reconstructed as balance(t) + y_h * scale(t).

The same code builds intraday (hourly) and daily feature sets; only the
lag/window lists and the calendar features differ (``granularity``).
"""
from __future__ import annotations
import logging
from typing import List
import numpy as np
import pandas as pd

from .config import FeatureConfig

log = logging.getLogger(__name__)

CATEGORICAL = ["account_type", "currency", "account_id"]


def extend_calendar(ts: pd.DatetimeIndex, granularity: str, n_extra: int) -> pd.DatetimeIndex:
    """Extend an observed business calendar ``n_extra`` open periods into the future,
    reusing the observed (weekday, hour) pattern so closed hours/weekends are skipped."""
    ts = pd.DatetimeIndex(sorted(ts.unique()))
    if granularity == "intraday":
        pattern = set(zip(ts.dayofweek, ts.hour))
        cand = pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=n_extra * 8 + 72, freq="h")
        fut = cand[[(d, h) in pattern for d, h in zip(cand.dayofweek, cand.hour)]][:n_extra]
    else:
        cand = pd.date_range(ts[-1] + pd.Timedelta(days=1), periods=n_extra * 2 + 7, freq="D")
        fut = cand[cand.dayofweek < 5][:n_extra]
    return ts.append(fut)


def to_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the intraday panel to end-of-day observations per account."""
    df = panel.reset_index()
    df["date"] = df["timestamp"].dt.normalize()
    g = df.groupby(["account_id", "date"])
    daily = pd.DataFrame({
        "balance": g["balance"].last(),
        "balance_filled_flag": g["balance_filled_flag"].max(),
        "net_flow": g["net_flow"].sum(), "inflow": g["inflow"].sum(),
        "outflow": g["outflow"].sum(), "tx_count": g["tx_count"].sum(),
        "max_abs_tx": g["max_abs_tx"].max(), "outlier_flag": g["outlier_flag"].max(),
        "net_flow_clean": g["net_flow_clean"].sum(), "scale": g["scale"].last(),
        "excluded_flow": g["excluded_flow"].sum() if "excluded_flow" in df else 0.0,
        "intraday_range": (g["balance"].max() - g["balance"].min()),
        "account_type": g["account_type"].first(), "currency": g["currency"].first(),
    })
    ext_cols = [c for c in panel.columns if c.startswith("fx_") or c in ("sofr", "sonia", "estr", "vix")]
    for c in ext_cols:
        daily[c] = g[c].last()
    daily.index.names = ["account_id", "timestamp"]
    return daily.sort_index()


def _temporal(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    ts = df.index.get_level_values("timestamp")
    out = pd.DataFrame(index=df.index)
    out["dow"] = ts.dayofweek
    out["dom"] = ts.day
    out["month"] = ts.month
    out["days_to_month_end"] = (ts.to_period("M").to_timestamp("M") - ts.normalize()).days
    out["is_month_end_window"] = (out["days_to_month_end"] <= 2).astype(int)
    out["is_quarter_end_window"] = (out["is_month_end_window"] & ts.month.isin([3, 6, 9, 12])).astype(int)
    out["is_year_end_window"] = (out["is_month_end_window"] & (ts.month == 12)).astype(int)
    out["is_month_start"] = (ts.day <= 2).astype(int)
    out["is_monday"] = (ts.dayofweek == 0).astype(int)
    out["is_friday"] = (ts.dayofweek == 4).astype(int)
    if granularity == "intraday":
        out["hour"] = ts.hour
        out["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        # periods until close of business today (open-hours grid)
        day = ts.normalize()
        out["periods_to_cob"] = pd.Series(np.arange(len(df)), index=df.index).groupby(
            [df.index.get_level_values("account_id"), day]).transform(lambda s: len(s) - 1 - np.arange(len(s))).values
        out["is_first_hour"] = (out["periods_to_cob"] == out.groupby(
            [df.index.get_level_values("account_id"), day])["periods_to_cob"].transform("max")).astype(int)
        out["is_last_hour"] = (out["periods_to_cob"] == 0).astype(int)
    return out


def _lag_and_rolling(df: pd.DataFrame, lags: List[int], windows: List[int]) -> pd.DataFrame:
    g = df.groupby(level="account_id", group_keys=False)
    out = pd.DataFrame(index=df.index)
    scaled_bal = df["balance"] / df["scale"]
    scaled_flow = df["net_flow_clean"] / df["scale"]
    out["bal_scaled"] = scaled_bal
    out["bal_dev_from_scale"] = scaled_bal - 1.0
    gb = scaled_bal.groupby(level="account_id", group_keys=False)
    gf = scaled_flow.groupby(level="account_id", group_keys=False)
    for L in lags:
        out[f"bal_lag_{L}"] = gb.shift(L)
        out[f"bal_chg_{L}"] = scaled_bal - out[f"bal_lag_{L}"]
        out[f"flow_lag_{L}"] = gf.shift(L)
    for W in windows:
        r = gb.rolling(W, min_periods=max(2, W // 4))
        out[f"bal_mean_{W}"] = r.mean().values
        out[f"bal_std_{W}"] = r.std().values
        out[f"bal_min_{W}"] = r.min().values
        out[f"bal_max_{W}"] = r.max().values
        out[f"bal_z_{W}"] = (scaled_bal - out[f"bal_mean_{W}"]) / out[f"bal_std_{W}"].replace(0, np.nan)
        rf = gf.rolling(W, min_periods=max(2, W // 4))
        out[f"flow_mean_{W}"] = rf.mean().values
        out[f"flow_std_{W}"] = rf.std().values
        out[f"flow_sum_{W}"] = rf.sum().values
    # flow characteristics
    out["inflow_scaled"] = df["inflow"] / df["scale"]
    out["outflow_scaled"] = df["outflow"] / df["scale"]
    out["tx_count"] = df["tx_count"]
    out["tx_count_rel"] = df["tx_count"] / g["tx_count"].transform(
        lambda s: s.rolling(windows[-1], min_periods=5).mean()).replace(0, np.nan)
    out["max_abs_tx_scaled"] = df["max_abs_tx"] / df["scale"]
    out["outlier_flag"] = df["outlier_flag"]
    out["balance_filled_flag"] = df["balance_filled_flag"]
    return out


def _seasonal_key(ts: pd.DatetimeIndex, granularity: str) -> pd.DataFrame:
    k = pd.DataFrame({"dow": ts.dayofweek}, index=ts)
    if granularity == "intraday":
        k["hour"] = ts.hour
    else:
        k["is_month_end_window"] = ((ts.to_period("M").to_timestamp("M") - ts.normalize()).days <= 2).astype(int)
    return k


def _seasonal_profile(df: pd.DataFrame, granularity: str, train_mask: np.ndarray, horizons: List[int]):
    """Seasonal baseline: mean scaled flow by (account, hour, dow) or (account, dow, month-end flag),
    estimated on the training portion only. Returns
      * ``seasonal_flow`` aligned to df.index
      * ``baseline_h{h}`` = cumulative seasonal flow over the next h *open* periods, computed on a
        calendar extended into the future so the last observations get a real baseline (no leakage:
        the profile is a fixed calendar lookup)."""
    ts = df.index.get_level_values("timestamp")
    accts = df.index.get_level_values("account_id")
    key_cols = list(_seasonal_key(ts[:1], granularity).columns)
    tmp = _seasonal_key(ts, granularity).reset_index(drop=True)
    tmp["account_id"] = accts.values
    tmp["flow"] = (df["net_flow_clean"] / df["scale"]).values
    prof = tmp[train_mask].groupby(["account_id"] + key_cols)["flow"].mean().rename("seasonal_flow")

    cal = extend_calendar(ts, granularity, max(horizons))
    fkey = _seasonal_key(cal, granularity)
    out = {}
    seasonal = pd.Series(index=df.index, dtype=float)
    per_h = {h: pd.Series(index=df.index, dtype=float) for h in horizons}
    for a in accts.unique():
        k = fkey.copy(); k["account_id"] = a
        prof_a = k.join(prof, on=["account_id"] + key_cols)["seasonal_flow"].fillna(0.0)  # indexed by cal
        obs_ts = ts[accts == a]
        seasonal.loc[a] = prof_a.reindex(obs_ts).values
        for h in horizons:
            # sum of seasonal flow over periods t+1..t+h
            fwd = prof_a[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
            per_h[h].loc[a] = fwd.reindex(obs_ts).fillna(0.0).values
    out["seasonal_flow"] = seasonal
    for h in horizons:
        out[f"baseline_h{h}"] = per_h[h]
    return out


def _cross_account(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Same-currency cross-account dynamics.

    * ``ccy_other_flow``  – aggregate scaled flow of *other* accounts in the same currency
                            (contemporaneous; at forecast time these are known).
    * ``corr_<type>``     – rolling correlation of this account's flow with the
                            same-currency account of each other type.
    """
    out = pd.DataFrame(index=df.index)
    flow = (df["net_flow_clean"] / df["scale"]).rename("f").to_frame()
    flow["currency"] = df["currency"].values
    flow["account_type"] = df["account_type"].values
    wide = flow["f"].unstack("account_id")  # timestamp x account
    meta = df.groupby(level="account_id")[["currency", "account_type"]].first()
    other_flow = pd.Series(np.nan, index=df.index)
    corr_cols = {t: pd.Series(np.nan, index=df.index) for t in meta.account_type.unique()}
    acct_level = df.index.get_level_values("account_id")
    ts_level = df.index.get_level_values("timestamp")
    for acct in wide.columns:
        ccy, typ = meta.loc[acct, "currency"], meta.loc[acct, "account_type"]
        peers = [a for a in wide.columns if a != acct and meta.loc[a, "currency"] == ccy]
        if not peers:
            continue
        own_ts = ts_level[acct_level == acct]           # this account's (possibly irregular) timestamps
        agg = wide[peers].sum(axis=1, min_count=1)       # NaN when every peer is closed
        other_flow.loc[acct] = agg.reindex(own_ts).values
        for p in peers:
            ptype = meta.loc[p, "account_type"]
            if ptype == typ:
                continue
            c = wide[acct].rolling(window, min_periods=window // 2).corr(wide[p])
            corr_cols[ptype].loc[acct] = c.reindex(own_ts).values
    out["ccy_other_flow"] = other_flow.fillna(0.0)
    out["ccy_other_flow_lag1"] = out.groupby(level="account_id")["ccy_other_flow"].shift(1)
    for t, s in corr_cols.items():
        out[f"corr_{t}"] = s
    return out


def _external(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cols = [c for c in df.columns if c.startswith("fx_") or c in ("sofr", "sonia", "estr", "vix")]
    g = df.groupby(level="account_id", group_keys=False)
    for c in cols:
        out[c] = df[c]
        out[f"{c}_chg_{windows[0]}"] = g[c].pct_change(windows[0])
    if "vix" in cols:
        out["vix_z"] = (df["vix"] - g["vix"].transform(lambda s: s.rolling(windows[-1], min_periods=5).mean())) / \
                       g["vix"].transform(lambda s: s.rolling(windows[-1], min_periods=5).std()).replace(0, np.nan)
    return out


def regime_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Volatility regime detection.

    Realised flow volatility over a short window is compared with its long-run
    distribution (robust z-score). Two outputs: a continuous ``regime_vol_z``
    the model can use, and a binary ``stress_regime`` used for reporting and
    for backtest slicing. This is deliberately simple and transparent; an
    HMM/Markov-switching model can be substituted with the same interface.
    """
    out = pd.DataFrame(index=df.index)
    flow = df["net_flow"] / df["scale"]
    g = flow.groupby(level="account_id", group_keys=False)
    short = g.rolling(cfg.regime_vol_window // 5, min_periods=4).std().values
    long_ = g.rolling(cfg.regime_vol_window, min_periods=cfg.regime_vol_window // 4)
    med = long_.median().values
    mad = pd.Series(np.abs(flow.values - med), index=df.index).groupby(level="account_id", group_keys=False) \
        .rolling(cfg.regime_vol_window, min_periods=cfg.regime_vol_window // 4).median().values * 1.4826
    z = (short - med) / np.where(mad == 0, np.nan, mad)
    out["regime_vol_z"] = z
    out["stress_regime"] = (pd.Series(z, index=df.index).fillna(0) > cfg.regime_z_threshold).astype(int)
    # smooth with a short persistence so the flag does not flicker
    out["stress_regime"] = out.groupby(level="account_id")["stress_regime"].transform(
        lambda s: s.rolling(6, min_periods=1).max())
    return out


def build_features(df: pd.DataFrame, cfg: FeatureConfig, granularity: str,
                   horizons: List[int], train_mask: np.ndarray) -> pd.DataFrame:
    """Return feature matrix + targets (``y_h{h}``) aligned to ``df.index``."""
    if granularity == "intraday":
        lags, windows = cfg.intraday_lags, cfg.intraday_windows
    else:
        lags, windows = cfg.daily_lags, cfg.daily_windows
    parts = [_temporal(df, granularity)]
    parts.append(_lag_and_rolling(df, lags, windows))
    parts.append(_cross_account(df, cfg.cross_account_corr_window))
    if cfg.use_external:
        parts.append(_external(df, windows))
    parts.append(regime_features(df, cfg))
    X = pd.concat(parts, axis=1)
    for name, series in _seasonal_profile(df, granularity, train_mask, horizons).items():
        X[name] = series
    for c in CATEGORICAL:
        X[c] = (df[c] if c in df.columns else df.index.get_level_values(c)).astype("category").values
    # targets
    scaled_bal = df["balance"] / df["scale"]
    gb = scaled_bal.groupby(level="account_id", group_keys=False)
    for h in horizons:
        X[f"y_h{h}"] = gb.shift(-h) - scaled_bal
    log.info("features[%s]: %d rows x %d cols", granularity, len(X), X.shape[1])
    return X


def feature_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if not (c.startswith("y_h") or c.startswith("baseline_h"))]
