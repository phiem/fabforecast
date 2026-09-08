"""Evaluation metrics for point forecasts and prediction intervals."""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def directional_accuracy(actual_change: np.ndarray, pred_change: np.ndarray) -> float:
    m = actual_change != 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(actual_change[m]) == np.sign(pred_change[m])))


def weighted_mape(actual: np.ndarray, pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    denom = np.abs(actual)
    ok = denom > 0
    ape = np.abs(actual[ok] - pred[ok]) / denom[ok]
    w = np.ones(ok.sum()) if weights is None else weights[ok]
    return float(100 * np.sum(w * ape) / np.sum(w))


def evaluate(df: pd.DataFrame, quantiles: Sequence[float], criticality: Optional[pd.Series] = None,
             interval: tuple = (0.05, 0.95)) -> Dict[str, float]:
    """``df`` must contain: actual, point, current (balance at origin) and q_<q> columns
    in *level* units. Returns a flat metric dict."""
    d = df.dropna(subset=["actual", "point"])
    if d.empty:
        return {}
    a, p = d["actual"].values, d["point"].values
    w = None if criticality is None else criticality.reindex(d["account_id"]).fillna(1.0).values
    out = {
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(a - p))),
        "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
        "mape_pct": weighted_mape(a, p),
        "mape_weighted_pct": weighted_mape(a, p, w) if w is not None else np.nan,
        "mae_scaled": float(np.mean(np.abs(a - p) / d["scale"].values)),
        "directional_acc": directional_accuracy(a - d["current"].values, p - d["current"].values),
    }
    if "baseline_point" in d:
        out["mae_baseline"] = float(np.mean(np.abs(a - d["baseline_point"].values)))
        out["skill_vs_baseline"] = 1 - out["mae"] / out["mae_baseline"] if out["mae_baseline"] > 0 else np.nan
    for q in quantiles:
        col = f"q_{q}"
        if col in d:
            out[f"pinball_{q}"] = pinball_loss(a, d[col].values, q) / d["scale"].mean()
    lo, hi = f"q_{interval[0]}", f"q_{interval[1]}"
    if lo in d and hi in d:
        out[f"coverage_{int(round(100 * (interval[1] - interval[0])))}"] = float(
            np.mean((a >= d[lo].values) & (a <= d[hi].values)))
        out["interval_width_scaled"] = float(np.mean((d[hi].values - d[lo].values) / d["scale"].values))
    return out


def evaluate_by(df: pd.DataFrame, by: str, quantiles: Sequence[float], **kw) -> pd.DataFrame:
    rows = {k: evaluate(g, quantiles, **kw) for k, g in df.groupby(by)}
    return pd.DataFrame(rows).T
