"""Known flows, treasury-flow exclusion and queue features.

Why this module exists
----------------------
The raw target ``y_h = (B[t+h] - B[t]) / scale`` asks the model to predict three
very different things at once:

1. flows the desk already knows about at t (FX legs, maturities, salary files,
   CLS pay-ins)                                  -> should be ADDED, not predicted
2. flows the desk itself decides (sweeps, funding transfers)
                                                  -> decisions, not signals: EXCLUDE
3. everything else (client payments, counterparty legs)
                                                  -> the only part worth a model

So the pipeline, when given ``scheduled_flows`` and ``flow_class`` tags, trains on

    y_h_residual = y_h - known_h(t) - excluded_h(t)

where ``known_h(t)`` is the scaled sum of scheduled flows with value time in
(t, t+h] that were known at t (point-in-time), and ``excluded_h(t)`` is the scaled
sum of treasury-class settled flows in (t, t+h]. At forecast time the level is
rebuilt as ``B[t] + (baseline + ml + known_h) * scale`` — the *unmanaged* position
the desk needs before deciding its own sweeps.

Everything here is computed with calendar positions so that "next h periods"
means h *open* periods, consistent with the rest of the feature set.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _range_add(n: int, starts: np.ndarray, ends: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    """Sum of amounts over inclusive index ranges [start, end] via difference array."""
    diff = np.zeros(n + 1)
    ok = ends >= starts
    s, e, a = np.clip(starts[ok], 0, n), np.clip(ends[ok], -1, n - 1), amounts[ok]
    np.add.at(diff, s, a)
    np.add.at(diff, e + 1, -a)
    return np.cumsum(diff)[:n]


def known_flow_features(panel: pd.DataFrame, scheduled: pd.DataFrame, horizons: Sequence[int],
                        freq: str) -> pd.DataFrame:
    """Point-in-time sum of scheduled flows over the next h open periods, scaled.

    A flow with value_time in bucket v, known at k (and cancelled at c, if any)
    contributes to origins p with  max(v-h, k) <= p <= min(v-1, c-1).
    """
    out = pd.DataFrame(index=panel.index)
    for h in horizons:
        out[f"known_h{h}"] = 0.0
    out["known_next_1"] = 0.0
    out["known_count_h1"] = 0.0
    if scheduled is None or scheduled.empty:
        return out
    accts = panel.index.get_level_values("account_id")
    ts_all = panel.index.get_level_values("timestamp")
    scale = panel["scale"].values
    for a, sf in scheduled.groupby("account_id"):
        sel = np.where(accts == a)[0]
        if len(sel) == 0:
            continue
        cal = ts_all[sel]
        n = len(cal)
        v = cal.searchsorted(sf["value_time"].dt.floor(freq).values, side="left")
        k = cal.searchsorted(sf["known_at"].values, side="left")
        if "cancelled_at" in sf and sf["cancelled_at"].notna().any():
            c = np.where(sf["cancelled_at"].notna(), cal.searchsorted(sf["cancelled_at"].fillna(cal[-1] + pd.Timedelta(days=1)).values, side="left"), n + 1)
        else:
            c = np.full(len(sf), n + 1)
        amt = sf["amount"].values.astype(float)
        for h in horizons:
            tot = _range_add(n, np.maximum(v - h, k), np.minimum(v - 1, c - 1), amt)
            out.iloc[sel, out.columns.get_loc(f"known_h{h}")] = tot / scale[sel]
        cnt = _range_add(n, np.maximum(v - 1, k), np.minimum(v - 1, c - 1), np.ones(len(sf)))
        out.iloc[sel, out.columns.get_loc("known_count_h1")] = cnt
        out.iloc[sel, out.columns.get_loc("known_next_1")] = out.iloc[sel, out.columns.get_loc(f"known_h{horizons[0]}")]
    log.info("known flows: %d scheduled rows mapped onto %d origins", len(scheduled), len(panel))
    return out


def excluded_flow_adjustment(panel: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Scaled sum of ``panel['excluded_flow']`` (treasury-class flows) over (t, t+h]."""
    out = pd.DataFrame(index=panel.index)
    if "excluded_flow" not in panel.columns:
        for h in horizons:
            out[f"excluded_h{h}"] = 0.0
        return out
    g = (panel["excluded_flow"] / panel["scale"]).groupby(level="account_id", group_keys=False)
    for h in horizons:
        # sum of periods t+1..t+h  ==  reversed rolling sum, shifted
        out[f"excluded_h{h}"] = g.transform(lambda s: s[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)).fillna(0.0)
    return out


def queue_features(panel: pd.DataFrame, snapshots: Optional[pd.DataFrame]) -> pd.DataFrame:
    """As-of join of queue snapshots onto the panel (last snapshot at or before t)."""
    out = pd.DataFrame(index=panel.index)
    cols = ["queued_amount", "held_amount", "time_critical_amount"]
    for c in cols:
        out[f"{c}_scaled"] = 0.0
    out["queue_snapshot_age_h"] = np.nan
    if snapshots is None or snapshots.empty:
        return out
    snaps = snapshots.sort_values("timestamp")
    for c in cols:
        if c not in snaps:
            snaps[c] = 0.0
    left = pd.DataFrame({"timestamp": panel.index.get_level_values("timestamp"),
                         "account_id": panel.index.get_level_values("account_id")}).sort_values("timestamp")
    m = pd.merge_asof(left, snaps[["timestamp", "account_id"] + cols].rename(columns={"timestamp": "snap_ts"}),
                      left_on="timestamp", right_on="snap_ts", by="account_id", direction="backward")
    m = m.set_index(["account_id", "timestamp"]).reindex(panel.index)
    scale = panel["scale"].values
    for c in cols:
        out[f"{c}_scaled"] = m[c].fillna(0.0).values / scale
    age = pd.Series(m.index.get_level_values("timestamp"), index=m.index) - m["snap_ts"]
    out["queue_snapshot_age_h"] = (age.dt.total_seconds() / 3600).values
    return out


def benchmark_frame(benchmark: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Align the desk's historical forecasts with model predictions on
    (account_id, origin=timestamp, target_time) so both can be scored with
    :func:`evaluation.evaluate` on identical rows."""
    p = predictions.copy()
    if "target_time" not in p.columns:
        raise ValueError("predictions need a target_time column (backtest adds it)")
    b = benchmark.rename(columns={"origin": "timestamp", "forecast": "benchmark_point"})
    m = p.merge(b[["account_id", "timestamp", "target_time", "benchmark_point"]],
                on=["account_id", "timestamp", "target_time"], how="inner")
    log.info("benchmark: %d of %d prediction rows have a desk forecast", len(m), len(p))
    return m
