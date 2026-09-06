"""
Liquidity forecasting pipeline for Cash Management -- now with MLflow
experiment tracking and matplotlib/Plotly visual evaluation wired in.

This builds on the earlier reference pipeline (liquidity_forecast.py):
same labels, features, walk-forward validation, and quantile LightGBM
models. What's new in this version:

  1. Every backtest run is logged to MLflow: params, per-horizon metrics,
     and the full metrics table as a CSV artifact.
  2. Three visual evaluations are generated and logged as artifacts:
       - MAE: model vs. seasonal-naive baseline (bar chart, matplotlib)
       - Quantile coverage vs. the 10%/90% target (bar chart, matplotlib)
       - Forecast vs. actual over time with P10/P50/P90 bands
         (interactive HTML, Plotly) for a sample account
  3. A "champion vs challenger" comparison: run this script twice with
     different hyperparameters and compare runs in the MLflow UI.

Run it, then view results with:
    mlflow ui --backend-store-uri ./mlruns
and open http://localhost:5000

Dependencies:
    pip install pandas numpy lightgbm holidays scikit-learn mlflow matplotlib plotly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays as holidays_lib
import mlflow
import matplotlib
matplotlib.use("Agg")  # no display in this environment
import matplotlib.pyplot as plt
import plotly.graph_objects as go

QUANTILES = [0.10, 0.50, 0.90]
HORIZONS = [1, 5, 10]
US_HOLIDAYS = holidays_lib.UnitedStates()

# Recent MLflow versions deprecated the plain "./mlruns" file store in favor
# of a SQLite (or other database) backend -- this is now the recommended
# default for anything beyond a one-off local experiment.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "liquidity-forecasting"

# ---------------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------------

def build_labels(txns: pd.DataFrame) -> pd.DataFrame:
    daily = (txns.groupby(["account_id", pd.Grouper(key="settle_date", freq="B")])
                 ["amount"].sum().rename("net_flow").reset_index())
    out = []
    for acct, grp in daily.groupby("account_id"):
        idx = pd.bdate_range(grp["settle_date"].min(), grp["settle_date"].max())
        g = (grp.set_index("settle_date").reindex(idx)
                .rename_axis("settle_date").reset_index())
        g["account_id"] = acct
        g["net_flow"] = g["net_flow"].fillna(0.0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def flag_exceptional_flows(labels: pd.DataFrame, z: float = 8.0) -> pd.DataFrame:
    df = labels.copy()
    med = df.groupby("account_id")["net_flow"].transform("median")
    abs_dev = (df["net_flow"] - med).abs()
    mad = abs_dev.groupby(df["account_id"]).transform("median") + 1e-9
    df["is_exceptional"] = (abs_dev / (1.4826 * mad)) > z
    return df


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------

def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["settle_date"]
    df["dow"] = d.dt.dayofweek
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df["bday_of_month"] = d.apply(
        lambda x: np.busday_count(x.replace(day=1).date(), x.date()) + 1)
    df["is_month_end"] = d.dt.is_month_end | (
        d + pd.offsets.BDay(1)).dt.month.ne(d.dt.month)
    df["is_quarter_end"] = df["is_month_end"] & d.dt.month.isin([3, 6, 9, 12])
    df["is_holiday_adjacent"] = d.apply(
        lambda x: (x + pd.Timedelta(days=1)) in US_HOLIDAYS
                  or (x - pd.Timedelta(days=1)) in US_HOLIDAYS)
    return df


def autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("account_id")["net_flow"]
    for lag in [1, 2, 3, 5, 7, 10, 20]:
        df[f"lag_{lag}"] = g.shift(lag)
    df["same_dow_lag_5"] = g.shift(5)
    df["same_dow_lag_20"] = g.shift(20)
    for w in [5, 20, 60]:
        s = g.shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean().reset_index(0, drop=True)
        df[f"roll_std_{w}"] = s.rolling(w).std().reset_index(0, drop=True)
        df[f"roll_min_{w}"] = s.rolling(w).min().reset_index(0, drop=True)
    return df


def merge_known_future_flows(df: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(known, on=["account_id", "settle_date"], how="left")
    df[["scheduled_out", "scheduled_in"]] = (
        df[["scheduled_out", "scheduled_in"]].fillna(0.0))
    df["scheduled_net"] = df["scheduled_in"] - df["scheduled_out"]
    return df


def build_features(labels: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    df = labels.sort_values(["account_id", "settle_date"]).copy()
    df = calendar_features(df)
    df = autoregressive_features(df)
    df = merge_known_future_flows(df, known)
    df["account_id"] = df["account_id"].astype("category")
    return df


FEATURE_COLS = (
    ["dow", "dom", "month", "bday_of_month", "is_month_end",
     "is_quarter_end", "is_holiday_adjacent", "account_id",
     "scheduled_out", "scheduled_in", "scheduled_net"]
    + [f"lag_{l}" for l in [1, 2, 3, 5, 7, 10, 20]]
    + ["same_dow_lag_5", "same_dow_lag_20"]
    + [f"roll_{s}_{w}" for w in [5, 20, 60] for s in ["mean", "std", "min"]]
)


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("account_id")["net_flow"]
    for h in HORIZONS:
        df[f"y_h{h}"] = g.shift(-h)
    return df


# ---------------------------------------------------------------------------
# 3. Walk-forward splits
# ---------------------------------------------------------------------------

def walk_forward_splits(df: pd.DataFrame, n_folds: int = 6, test_days: int = 21):
    dates = np.sort(df["settle_date"].unique())
    for i in range(n_folds, 0, -1):
        test_end = len(dates) - (i - 1) * test_days
        test_start = test_end - test_days
        if test_start <= 60:
            continue
        train_idx = df["settle_date"] < dates[test_start]
        test_idx = (df["settle_date"] >= dates[test_start]) & \
                   (df["settle_date"] < dates[test_end - 1])
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# 4. Quantile GBM training
# ---------------------------------------------------------------------------

def train_quantile_models(train: pd.DataFrame, horizon: int, params: dict) -> dict:
    y_col = f"y_h{horizon}"
    data = train.dropna(subset=[y_col])
    data = data[~data.get("is_exceptional", False)]
    models = {}
    for q in QUANTILES:
        models[q] = lgb.train(
            {"objective": "quantile", "alpha": q, **params, "verbosity": -1},
            lgb.Dataset(data[FEATURE_COLS], label=data[y_col]),
            num_boost_round=params.get("num_boost_round", 500),
        )
    return models


def predict_quantiles(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    preds = pd.DataFrame({f"p{int(q*100)}": m.predict(X[FEATURE_COLS])
                          for q, m in models.items()}, index=X.index)
    preds["p10"] = preds[["p10", "p50"]].min(axis=1)
    preds["p90"] = preds[["p50", "p90"]].max(axis=1)
    return preds


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def seasonal_naive(test: pd.DataFrame, horizon: int) -> np.ndarray:
    cols = ["same_dow_lag_5", "same_dow_lag_20", "lag_10", "lag_20"]
    return test[cols].mean(axis=1).to_numpy()


def evaluate_fold(test: pd.DataFrame, preds: pd.DataFrame, horizon: int) -> dict:
    y_col = f"y_h{horizon}"
    mask = test[y_col].notna()
    y = test.loc[mask, y_col].to_numpy()
    p = preds.loc[mask]
    metrics = {
        "horizon": horizon,
        "n": int(mask.sum()),
        "mae_p50": float(np.mean(np.abs(y - p["p50"]))),
        "mae_naive": float(np.mean(np.abs(
            y - seasonal_naive(test.loc[mask], horizon)))),
        "pinball_p10": pinball_loss(y, p["p10"].to_numpy(), 0.10),
        "pinball_p90": pinball_loss(y, p["p90"].to_numpy(), 0.90),
        "below_p10_rate": float(np.mean(y < p["p10"])),
        "above_p90_rate": float(np.mean(y > p["p90"])),
    }
    me = test.loc[mask, "is_month_end"].to_numpy(dtype=bool)
    if me.any():
        metrics["mae_p50_month_end"] = float(
            np.mean(np.abs(y[me] - p["p50"].to_numpy()[me])))
    return metrics


def backtest(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, one saved prediction set for plotting)."""
    rows = []
    sample_preds = None  # keep last fold's predictions for the time-series plot
    for fold_i, (train_idx, test_idx) in enumerate(walk_forward_splits(df)):
        train, test = df[train_idx], df[test_idx]
        for h in HORIZONS:
            models = train_quantile_models(train, h, params)
            preds = predict_quantiles(models, test)
            rows.append(evaluate_fold(test, preds, h))
            if h == 1:  # keep the 1-day-ahead fold for the sample time-series chart
                sample_preds = test[["account_id", "settle_date", "y_h1"]].join(preds).copy()
    res = pd.DataFrame(rows)
    return res, sample_preds


# ---------------------------------------------------------------------------
# 6. Visual evaluation -- matplotlib + Plotly
# ---------------------------------------------------------------------------

def plot_mae_vs_baseline(summary: pd.DataFrame, path: str) -> None:
    """Bar chart: model MAE vs naive baseline MAE, per horizon (matplotlib)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width / 2, summary["mae_p50"], width, label="Model (P50)")
    ax.bar(x + width / 2, summary["mae_naive"], width, label="Seasonal-naive baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in summary["horizon"]])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE")
    ax.set_title("Model vs. baseline MAE by horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_coverage(summary: pd.DataFrame, path: str) -> None:
    """Bar chart: actual P10/P90 breach rates vs. the 10%/90% target (matplotlib).
    This is the calibration check -- the single most important chart for
    trusting the quantile forecasts operationally."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width / 2, summary["below_p10_rate"], width, label="Actual below-P10 rate")
    ax.bar(x + width / 2, summary["above_p90_rate"], width, label="Actual above-P90 rate")
    ax.axhline(0.10, color="black", linestyle="--", linewidth=1, label="10% target")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in summary["horizon"]])
    ax.set_ylabel("Breach rate")
    ax.set_title("Quantile coverage vs. target (calibration check)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast_vs_actual(sample_preds: pd.DataFrame, path: str,
                             account: str | None = None) -> None:
    """Interactive Plotly chart: actual net flow vs. P10/P50/P90 bands over
    time for one account. This is the chart an ops/treasury reviewer would
    actually want to scroll through."""
    if account is None:
        account = sample_preds["account_id"].iloc[0]
    d = sample_preds[sample_preds["account_id"] == account].sort_values("settle_date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["settle_date"], y=d["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["settle_date"], y=d["p10"], line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(100,149,237,0.25)",
                             name="P10-P90 band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["settle_date"], y=d["p50"], mode="lines",
                             name="P50 forecast", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=d["settle_date"], y=d["y_h1"], mode="markers+lines",
                             name="Actual net flow", line=dict(color="black", dash="dot"),
                             marker=dict(size=5)))
    fig.update_layout(title=f"1-day-ahead forecast vs. actual -- {account}",
                      xaxis_title="Settlement date", yaxis_title="Net flow",
                      template="plotly_white", height=420)
    fig.write_html(path)


# ---------------------------------------------------------------------------
# 7. MLflow-wrapped run
# ---------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, run_name: str, params: dict) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("horizons", HORIZONS)
        mlflow.log_param("quantiles", QUANTILES)

        per_fold, sample_preds = backtest(df, params)
        summary = per_fold.groupby("horizon").mean(numeric_only=True).reset_index().round(4)

        # 1. Numeric: log every metric per horizon
        for _, row in summary.iterrows():
            h = int(row["horizon"])
            for col in ["mae_p50", "mae_naive", "pinball_p10", "pinball_p90",
                       "below_p10_rate", "above_p90_rate"]:
                mlflow.log_metric(f"h{h}_{col}", row[col])
            if "mae_p50_month_end" in row and not pd.isna(row["mae_p50_month_end"]):
                mlflow.log_metric(f"h{h}_mae_p50_month_end", row["mae_p50_month_end"])

        # also log an overall "did we beat baseline" flag -- easy to sort/filter on in the UI
        beat_baseline = bool((summary["mae_p50"] < summary["mae_naive"]).all())
        mlflow.log_metric("beat_baseline_all_horizons", int(beat_baseline))

        # 2. Numeric artifact: full metrics table as CSV
        csv_path = "metrics_summary.csv"
        summary.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # 3. Visual artifacts
        plot_mae_vs_baseline(summary, "mae_vs_baseline.png")
        mlflow.log_artifact("mae_vs_baseline.png")

        plot_coverage(summary, "coverage_calibration.png")
        mlflow.log_artifact("coverage_calibration.png")

        plot_forecast_vs_actual(sample_preds, "forecast_vs_actual.html")
        mlflow.log_artifact("forecast_vs_actual.html")

        print(f"\n=== Run '{run_name}' ===")
        print(summary[["horizon", "mae_p50", "mae_naive", "below_p10_rate",
                       "above_p90_rate"]])
        print(f"Beat baseline on all horizons: {beat_baseline}")
        return summary


# ---------------------------------------------------------------------------
# 8. Synthetic data + demo: two runs to compare (champion vs. challenger)
# ---------------------------------------------------------------------------

def _synthetic_transactions(n_accounts: int = 5, years: int = 3) -> tuple:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=252 * years)
    tx_rows, known_rows = [], []
    for a in range(n_accounts):
        base = rng.uniform(0.5, 2.0)
        for d in dates:
            dow_effect = [1.2, 1.0, 0.9, 1.0, 1.6][d.dayofweek]
            flow = rng.normal(0, 3.0) * base * dow_effect
            if d.is_month_end or (d + pd.offsets.BDay(1)).month != d.month:
                flow -= rng.uniform(8, 15) * base
            sched_in = 0.0
            if d.day in (10, 25):
                sched_in = rng.uniform(4, 6) * base
                flow += sched_in
            tx_rows.append((f"ACCT{a}", d, round(flow, 2)))
            known_rows.append((f"ACCT{a}", d, 0.0, round(sched_in, 2)))
    txns = pd.DataFrame(tx_rows, columns=["account_id", "settle_date", "amount"])
    known = pd.DataFrame(known_rows, columns=[
        "account_id", "settle_date", "scheduled_out", "scheduled_in"])
    return txns, known


if __name__ == "__main__":
    txns, known = _synthetic_transactions()
    labels = flag_exceptional_flows(build_labels(txns))
    df = add_horizon_targets(build_features(labels, known))

    # "Champion" -- the original default hyperparameters
    champion_params = {"learning_rate": 0.05, "num_leaves": 63,
                       "min_data_in_leaf": 50, "feature_fraction": 0.8,
                       "bagging_fraction": 0.8, "bagging_freq": 1}
    run_experiment(df, "champion_default_params", champion_params)

    # "Challenger" -- more regularization, aimed at fixing the overconfident
    # quantiles (below_p10_rate was ~18-22% instead of ~10% last time)
    challenger_params = {"learning_rate": 0.03, "num_leaves": 31,
                        "min_data_in_leaf": 100, "feature_fraction": 0.7,
                        "bagging_fraction": 0.7, "bagging_freq": 1}
    run_experiment(df, "challenger_more_regularized", challenger_params)

    print("\nAll runs logged to MLflow at:", MLFLOW_TRACKING_URI)
    print("View them with:  mlflow ui --backend-store-uri", MLFLOW_TRACKING_URI)
