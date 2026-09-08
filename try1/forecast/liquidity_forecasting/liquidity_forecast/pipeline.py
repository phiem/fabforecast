"""End-to-end orchestration.

    fc = LiquidityForecaster(cfg)
    fc.fit(tables, external)                         # builds panel, features, CV, final models
    fc.forecast(accounts=[...], granularity="intraday", horizons=[1, 4], confidence=[0.9])
    fc.backtest(granularity="daily")
"""
from __future__ import annotations
import logging
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .config import PipelineConfig
from . import data as D
from . import features as F
from .models import HorizonModel
from .evaluation import evaluate, evaluate_by
from . import known_flows as KF

log = logging.getLogger(__name__)

DAILY_HORIZONS_DEFAULT = [1, 5]


class LiquidityForecaster:
    def __init__(self, cfg: PipelineConfig, daily_horizons: Optional[List[int]] = None,
                 extra_features: Optional[Callable[[pd.DataFrame, str], pd.DataFrame]] = None,
                 open_mask: Optional[Callable[[pd.DataFrame], np.ndarray]] = None):
        """
        extra_features: callable(panel, granularity) -> DataFrame aligned to panel.index,
                        concatenated to the feature matrix (see seasonality.build_seasonal_features).
        open_mask:      callable(panel) -> bool array; rows where False are dropped before
                        feature building (e.g. seasonality.account_open_mask for currency holidays).
        """
        self.cfg = cfg
        self.extra_features = extra_features
        self.open_mask = open_mask
        self.daily_horizons = daily_horizons or DAILY_HORIZONS_DEFAULT
        self.panels: Dict[str, pd.DataFrame] = {}
        self.features: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Dict[int, HorizonModel]] = {"intraday": {}, "daily": {}}
        self.cv_results: Dict[str, pd.DataFrame] = {}
        self.masks: Dict[str, Dict[str, np.ndarray]] = {}
        logging.getLogger("liquidity_forecast").setLevel(cfg.log_level)

    # ------------------------------------------------------------------ setup
    def _horizons(self, granularity: str) -> List[int]:
        return self.cfg.model.horizons if granularity == "intraday" else self.daily_horizons

    def prepare(self, tables: Dict[str, pd.DataFrame], external: Optional[pd.DataFrame] = None) -> None:
        external = tables.get("external", external)
        self.scheduled = tables.get("scheduled_flows")
        self.queue_snapshots = tables.get("queue_snapshots")
        self.benchmark = tables.get("benchmark")
        panel = D.build_panel(tables, self.cfg.data, external)
        if self.open_mask is not None:
            m = self.open_mask(panel)
            log.info("open_mask drops %d rows on account-currency holidays", int((~m).sum()))
            panel = panel[m]
        self.panels["intraday"] = panel
        self.panels["daily"] = F.to_daily(panel)
        s = self.cfg.split
        for gran, p in self.panels.items():
            ts = p.index.get_level_values("timestamp")
            tr, va, te = D.make_splits(ts, s.train_end, s.valid_end, s.test_end)
            self.masks[gran] = {"train": tr, "valid": va, "test": te}
            self.features[gran] = F.build_features(p, self.cfg.features, gran, self._horizons(gran), tr)
            self._attach_known_flows(gran, p)
            if self.extra_features is not None:
                extra = self.extra_features(p, gran)
                dup = [c for c in extra.columns if c in self.features[gran].columns]
                if dup:
                    raise ValueError(f"extra_features collide with base features: {dup}")
                self.features[gran] = pd.concat([self.features[gran], extra], axis=1)
            log.info("%s split sizes train/valid/test = %d/%d/%d", gran, tr.sum(), va.sum(), te.sum())

    def _attach_known_flows(self, gran: str, panel: pd.DataFrame) -> None:
        """Add known_h / excluded_h / queue features and, if configured, convert the
        targets to residual form (see known_flows module docstring)."""
        X = self.features[gran]
        hs = self._horizons(gran)
        freq = self.cfg.data.freq if gran == "intraday" else "D"
        known = KF.known_flow_features(panel, self.scheduled, hs, freq)
        excl = KF.excluded_flow_adjustment(panel, hs)
        q = KF.queue_features(panel, self.queue_snapshots)
        raw = pd.DataFrame({f"y_h{h}_raw": X[f"y_h{h}"] for h in hs}, index=X.index)
        X = pd.concat([X, known, q, excl, raw], axis=1)
        if self.cfg.model.residual_target:
            for h in hs:
                X[f"y_h{h}"] = X[f"y_h{h}_raw"] - X[f"known_h{h}"] - X[f"excluded_h{h}"]
        self.features[gran] = X.copy()
        if self.cfg.model.residual_target:
            log.info("%s: residual target = raw change - known flows - excluded flows", gran)

    # -------------------------------------------------------------------- fit
    @staticmethod
    def _feature_cols(X: pd.DataFrame) -> List[str]:
        return [c for c in F.feature_columns(X) if not c.startswith("excluded_h") and not c.endswith("_raw")]

    def _cv_folds(self, X: pd.DataFrame, train_mask: np.ndarray, horizon: int):
        """Expanding-window folds over the training period with a purge gap of
        ``horizon`` periods between train and validation so that no target in
        the train fold overlaps a validation feature row."""
        ts = np.asarray(X.index.get_level_values("timestamp"))
        t_train = np.sort(np.unique(ts[train_mask]))
        k = self.cfg.model.cv_folds
        cut_points = np.array_split(t_train[len(t_train) // 3:], k + 1)
        for i in range(k):
            val_start, val_end = cut_points[i][0], cut_points[i][-1]
            purge_start = t_train[max(0, np.searchsorted(t_train, val_start) - horizon)]
            fit = train_mask & (ts < purge_start)
            val = train_mask & (ts >= val_start) & (ts <= val_end)
            yield fit, val

    def _to_levels(self, pred: pd.DataFrame, panel: pd.DataFrame, idx, known: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Scaled-change predictions -> levels. ``known`` (scaled) is added back when the
        model was trained on the residual target, so levels are the unmanaged position."""
        cur = panel.loc[idx, "balance"].values
        scale = panel.loc[idx, "scale"].values
        known = np.zeros(len(idx)) if known is None else known
        out = pd.DataFrame(index=idx)
        out["current"], out["scale"] = cur, scale
        for c in pred.columns:
            if c.startswith("q_"):
                out[c] = cur + (pred[c].values + known) * scale
        out["point"] = out["q_0.5"]
        out["baseline_point"] = cur + (pred["baseline"].values + known) * scale
        out["ml_adjustment"] = pred["ml_adjustment"].values * scale
        out["known_component"] = known * scale
        return out

    def fit(self, tables: Dict[str, pd.DataFrame], external: Optional[pd.DataFrame] = None,
            run_cv: bool = True) -> "LiquidityForecaster":
        self.prepare(tables, external)
        for gran in ("intraday", "daily"):
            X, panel = self.features[gran], self.panels[gran]
            cols = self._feature_cols(X)
            tr = self.masks[gran]["train"] | self.masks[gran]["valid"]   # final fit uses train+valid
            cv_rows = []
            for h in self._horizons(gran):
                y, base = X[f"y_h{h}"], X[f"baseline_h{h}"]
                if run_cv:
                    for i, (fit_m, val_m) in enumerate(self._cv_folds(X, self.masks[gran]["train"], h)):
                        m = HorizonModel(h, self.cfg.model, cols).fit(X[fit_m], y[fit_m], base[fit_m])
                        pred = m.predict(X[val_m], base[val_m])
                        lv = self._to_levels(pred, panel, X.index[val_m], self._known(X, val_m, h))
                        lv["actual"] = lv["current"] + (X.loc[val_m, f"y_h{h}"] + X.loc[val_m, f"known_h{h}"]).values * lv["scale"]
                        lv["account_id"] = lv.index.get_level_values("account_id")
                        r = evaluate(lv, self.cfg.model.quantiles); r.update(fold=i, horizon=h)
                        cv_rows.append(r)
                        log.info("CV %s h=%d fold=%d MAE=%.3g cov90=%.3f dir=%.3f skill=%.3f", gran, h, i,
                                 r["mae"], r.get("coverage_90", np.nan), r["directional_acc"], r.get("skill_vs_baseline", np.nan))
                self.models[gran][h] = HorizonModel(h, self.cfg.model, cols).fit(X[tr], y[tr], base[tr])
            if cv_rows:
                self.cv_results[gran] = pd.DataFrame(cv_rows)
        return self

    # --------------------------------------------------------------- forecast
    def forecast(self, accounts: Optional[Sequence[str]] = None, granularity: str = "intraday",
                 horizons: Optional[Sequence[int]] = None, confidence: Sequence[float] = (0.9,),
                 origin: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Forecast from a single origin timestamp (default: last observation).

        Returns one row per (account, horizon, confidence level) with point
        estimate, bounds, baseline/ML decomposition and a confidence score
        (relative interval width mapped to (0,1]; 1 = very tight)."""
        X, panel = self.features[granularity], self.panels[granularity]
        ts = X.index.get_level_values("timestamp")
        origin = pd.Timestamp(origin) if origin is not None else ts.max()
        rows = X.index[ts == origin]
        if accounts:
            rows = rows[rows.get_level_values("account_id").isin(accounts)]
        if len(rows) == 0:
            raise ValueError(f"no observations at origin {origin}")
        for cl in confidence:
            for q in ((1 - cl) / 2, 1 - (1 - cl) / 2):
                if round(q, 4) not in [round(x, 4) for x in self.cfg.model.quantiles]:
                    raise ValueError(f"confidence {cl} needs quantile {q}; add it to ModelConfig.quantiles")
        out = []
        for h in (horizons or self._horizons(granularity)):
            mdl = self.models[granularity].get(h)
            if mdl is None:
                raise ValueError(f"no {granularity} model for horizon {h}; trained: {list(self.models[granularity])}")
            pred = mdl.predict(X.loc[rows], X.loc[rows, f"baseline_h{h}"])
            lv = self._to_levels(pred, panel, rows, self._known(X, rows, h))
            cal = F.extend_calendar(ts, granularity, h)
            target_time = cal[cal.get_loc(origin) + h]
            for cl in confidence:
                lo = next(c for c in lv.columns if c.startswith("q_") and abs(float(c[2:]) - (1 - cl) / 2) < 1e-6)
                hi = next(c for c in lv.columns if c.startswith("q_") and abs(float(c[2:]) - (1 - (1 - cl) / 2)) < 1e-6)
                width = (lv[hi] - lv[lo]) / lv["scale"]
                out.append(pd.DataFrame({
                    "account_id": rows.get_level_values("account_id"),
                    "account_type": panel.loc[rows, "account_type"].values,
                    "granularity": granularity, "origin": origin,
                    "target_time": target_time, "horizon": h,
                    "confidence": cl, "current_balance": lv["current"].values,
                    "point": lv["point"].values, "lower": lv[lo].values, "upper": lv[hi].values,
                    "baseline_component": lv["baseline_point"].values - lv["current"].values - lv["known_component"].values,
                    "ml_component": lv["ml_adjustment"].values,
                    "known_component": lv["known_component"].values,
                    "regime_vol_z": X.loc[rows, "regime_vol_z"].values,
                    "stress_regime": X.loc[rows, "stress_regime"].values,
                    "confidence_score": (1 / (1 + 10 * width)).values,
                }))
        return pd.concat(out, ignore_index=True)

    # --------------------------------------------------------------- backtest
    def backtest(self, granularity: str = "intraday", split: str = "test",
                 horizons: Optional[Sequence[int]] = None) -> Dict[str, pd.DataFrame]:
        """Rolling-origin evaluation over every timestamp in ``split`` using the
        final fitted models. Returns predictions plus metric tables sliced by
        horizon, account type, regime and month."""
        X, panel = self.features[granularity], self.panels[granularity]
        m = self.masks[granularity][split]
        preds = []
        for h in (horizons or self._horizons(granularity)):
            mdl = self.models[granularity][h]
            pred = mdl.predict(X[m], X.loc[m, f"baseline_h{h}"])
            lv = self._to_levels(pred, panel, X.index[m], self._known(X, m, h))
            # actual = unmanaged position (excludes treasury-class flows), consistent with the target
            lv["actual"] = lv["current"] + (X.loc[m, f"y_h{h}"] + X.loc[m, f"known_h{h}"]).values * lv["scale"].values
            lv["actual_incl_treasury"] = lv["current"] + X.loc[m, f"y_h{h}_raw"].values * lv["scale"].values
            unit = "h" if granularity == "intraday" else "D"
            cal = F.extend_calendar(X.index.get_level_values("timestamp"), granularity, h)
            pos = cal.searchsorted(X.index[m].get_level_values("timestamp")) + h
            lv["target_time"] = cal[np.clip(pos, 0, len(cal) - 1)]
            lv["horizon"] = h
            lv["account_type"] = panel.loc[lv.index, "account_type"].values
            lv["stress_regime"] = X.loc[m, "stress_regime"].values
            lv["month"] = lv.index.get_level_values("timestamp").to_period("M").astype(str)
            preds.append(lv.reset_index())
        P = pd.concat(preds, ignore_index=True).dropna(subset=["actual"])
        crit = pd.Series(self.cfg.account_criticality) if self.cfg.account_criticality else None
        q = self.cfg.model.quantiles
        res = {
            "predictions": P,
            "overall": pd.DataFrame([evaluate(P, q, criticality=crit)]),
            "by_horizon": evaluate_by(P, "horizon", q, criticality=crit),
            "by_account_type": evaluate_by(P, "account_type", q, criticality=crit),
            "by_account": evaluate_by(P, "account_id", q, criticality=crit),
            "by_regime": evaluate_by(P, "stress_regime", q, criticality=crit),
            "by_month": evaluate_by(P, "month", q, criticality=crit),
        }
        if self.benchmark is not None and len(self.benchmark):
            B = KF.benchmark_frame(self.benchmark, P)
            if len(B):
                Bm = B.rename(columns={"point": "model_point"}).rename(columns={"benchmark_point": "point"})
                res["benchmark_comparison"] = pd.DataFrame({
                    "model": evaluate(B, q, criticality=crit),
                    "desk": evaluate(Bm, q, criticality=crit)}).T[["n", "mae", "rmse", "mape_pct", "directional_acc"]]
        P["anomaly"] = self.flag_anomalies(P)
        res["anomalies"] = P[P["anomaly"]][["account_id", "timestamp", "horizon", "actual", "point", "q_0.05", "q_0.95"]]
        return res

    @staticmethod
    def _known(X: pd.DataFrame, sel, h: int) -> np.ndarray:
        col = f"known_h{h}"
        return X.loc[sel, col].values if col in X.columns else None

    # --------------------------------------------------------------- anomaly
    def flag_anomalies(self, P: pd.DataFrame) -> pd.Series:
        """An observation is anomalous when the realised balance falls outside the
        90% interval by more than ``threshold`` interval-widths."""
        width = (P["q_0.95"] - P["q_0.05"]).replace(0, np.nan)
        excess = np.maximum(P["q_0.05"] - P["actual"], P["actual"] - P["q_0.95"]) / width
        thr = P["account_id"].map(self.cfg.per_account_anomaly_threshold).fillna(self.cfg.anomaly_threshold or 0.5)
        return (excess > thr).fillna(False)

    # ------------------------------------------------------------ importance
    def feature_importance(self, granularity: str = "intraday", horizon: Optional[int] = None,
                           top: int = 15) -> pd.Series:
        h = horizon or self._horizons(granularity)[0]
        return self.models[granularity][h].feature_importance_.head(top)

    def feature_importance_by_account_type(self, granularity: str = "intraday", horizon: Optional[int] = None,
                                           top: int = 10) -> Dict[str, pd.Series]:
        """Per-type importance via refitting a small median model per type (diagnostic only)."""
        h = horizon or self._horizons(granularity)[0]
        X, panel = self.features[granularity], self.panels[granularity]
        tr = self.masks[granularity]["train"]
        out = {}
        cols = [c for c in self._feature_cols(X) if c not in ("account_type", "account_id")]
        for t in panel["account_type"].unique():
            sel = tr & (panel["account_type"] == t).values
            m = HorizonModel(h, self.cfg.model, cols)
            m.quantiles = [0.5]; m.cfg = type(self.cfg.model)(**{**self.cfg.model.__dict__, "conformal": False})
            m.fit(X[sel], X.loc[sel, f"y_h{h}"], X.loc[sel, f"baseline_h{h}"])
            out[t] = m.feature_importance_.head(top)
        return out
