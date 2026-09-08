"""Model layer.

`HorizonModel` is one forecaster for one (granularity, horizon) pair. It is a
hybrid:

    forecast = seasonal_baseline + ML(features)  ... where ML is trained on
    the *residual* of the baseline, so a degenerate ML model falls back to a
    sensible seasonal forecast.

The ML part is a quantile ensemble: for every requested quantile a LightGBM
and an XGBoost regressor are fitted with pinball loss and averaged. The
median model is the point forecast. Predicted quantiles are then made
monotone and (optionally) calibrated with Conformalised Quantile Regression
(Romano et al., 2019) on a held-out tail of the training window so that the
nominal coverage is honoured out of sample.

If LightGBM / XGBoost are not installed the class silently uses scikit-learn's
HistGradientBoostingRegressor with quantile loss.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .config import ModelConfig
from .features import CATEGORICAL

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False
from sklearn.ensemble import HistGradientBoostingRegressor


def _make_learner(kind: str, q: float, cfg: ModelConfig):
    if kind == "lgb":
        return lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=cfg.random_state, **cfg.lgb_params)
    if kind == "xgb":
        return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, enable_categorical=True,
                                tree_method="hist", random_state=cfg.random_state, **cfg.xgb_params)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                         categorical_features="from_dtype", random_state=cfg.random_state)


class HorizonModel:
    def __init__(self, horizon: int, cfg: ModelConfig, feature_cols: List[str]):
        self.horizon = horizon
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.quantiles = sorted(set(cfg.quantiles) | {0.5})
        self.learners: Dict[float, list] = {}
        self.conformal_offsets: Dict[float, float] = {}
        self.feature_importance_: Optional[pd.Series] = None
        kinds = []
        if cfg.use_lightgbm and HAS_LGB:
            kinds.append("lgb")
        if cfg.use_xgboost and HAS_XGB:
            kinds.append("xgb")
        self.kinds = kinds or ["sk"]

    # ------------------------------------------------------------------ utils
    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp = X[self.feature_cols].copy()
        for c in CATEGORICAL:
            if c in Xp.columns:
                Xp[c] = Xp[c].astype("category")
        return Xp

    @staticmethod
    def _valid(X: pd.DataFrame, y: pd.Series, baseline: pd.Series) -> np.ndarray:
        # drop rows without a target or where key lags are missing
        return (~y.isna()).values & (~baseline.isna()).values & (X.isna().mean(axis=1) < 0.5).values

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y: pd.Series, baseline: pd.Series,
            sample_weight: Optional[np.ndarray] = None) -> "HorizonModel":
        m = self._valid(X, y, baseline)
        Xp, resid = self._prep(X[m]), (y[m] - baseline[m]).values
        w = None if sample_weight is None else sample_weight[m]
        n = len(Xp)
        if self.cfg.conformal:
            n_cal = int(n * self.cfg.calibration_fraction)
            fit_idx, cal_idx = np.arange(n - n_cal), np.arange(n - n_cal, n)
        else:
            fit_idx, cal_idx = np.arange(n), np.array([], dtype=int)

        imps = []
        for q in self.quantiles:
            self.learners[q] = []
            for kind in self.kinds:
                mdl = _make_learner(kind, q, self.cfg)
                mdl.fit(Xp.iloc[fit_idx], resid[fit_idx], sample_weight=None if w is None else w[fit_idx])
                self.learners[q].append(mdl)
                if q == 0.5 and hasattr(mdl, "feature_importances_"):
                    imps.append(pd.Series(mdl.feature_importances_, index=self.feature_cols))
        if imps:
            fi = pd.concat(imps, axis=1).mean(axis=1)
            self.feature_importance_ = (fi / fi.sum()).sort_values(ascending=False)

        if self.cfg.conformal and len(cal_idx) > 20:
            raw = self._raw_quantiles(Xp.iloc[cal_idx])
            yc = resid[cal_idx]
            for q in self.quantiles:
                if q == 0.5:
                    continue
                # one-sided conformal correction per quantile: the lower (upper) bound is
                # shifted so that the empirical miss rate on the calibration set equals q (1-q)
                tail = min(q, 1 - q)
                lvl = min(1.0, (1 - tail) * (1 + 1 / len(yc)))
                scores = (raw[q] - yc) if q < 0.5 else (yc - raw[q])
                self.conformal_offsets[q] = float(np.quantile(scores, lvl))
            log.debug("h=%d conformal offsets %s", self.horizon,
                      {k: round(v, 5) for k, v in self.conformal_offsets.items()})
        log.info("h=%d fitted %s on %d rows (%d calibration)", self.horizon, self.kinds, len(fit_idx), len(cal_idx))
        return self

    # ---------------------------------------------------------------- predict
    def _raw_quantiles(self, Xp: pd.DataFrame) -> Dict[float, np.ndarray]:
        out = {}
        for q, mdls in self.learners.items():
            out[q] = np.mean([m.predict(Xp) for m in mdls], axis=0)
        return out

    def predict(self, X: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
        """Return DataFrame with columns ``q_<quantile>`` plus ``baseline`` and
        ``ml_adjustment`` (all in scaled-change units)."""
        Xp = self._prep(X)
        raw = self._raw_quantiles(Xp)
        res = pd.DataFrame(index=X.index)
        for q in self.quantiles:
            adj = raw[q].copy()
            off = self.conformal_offsets.get(q, 0.0)
            if q < 0.5:
                adj -= off
            elif q > 0.5:
                adj += off
            res[f"q_{q}"] = adj
        # enforce monotone quantiles
        qcols = [f"q_{q}" for q in self.quantiles]
        res[qcols] = np.sort(res[qcols].values, axis=1)
        res["ml_adjustment"] = res["q_0.5"]
        res["baseline"] = baseline.values
        for c in qcols:
            res[c] = res[c] + baseline.values
        return res
