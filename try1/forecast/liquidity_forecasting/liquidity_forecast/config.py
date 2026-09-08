"""Configuration dataclasses. Everything tunable lives here."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    freq: str = "h"                          # base granularity of the balance series
    business_hours: tuple = (7, 19)          # inclusive start, exclusive end (local time)
    holidays: List[str] = field(default_factory=list)   # ISO dates treated as closed
    outlier_mad_threshold: float = 6.0       # robust z-score on flows above which a tx is flagged
    winsorize_outliers: bool = True          # cap outliers in *features* (target is never altered)
    fill_gaps: bool = True                   # forward-fill balances over closed periods
    balance_kind: str = "available"          # which balance_kind to model when several are supplied
    exclude_flow_classes: List[str] = field(default_factory=lambda: ["treasury"])  # decisions, not signals


@dataclass
class FeatureConfig:
    intraday_lags: List[int] = field(default_factory=lambda: [1, 4, 8, 24])       # hours
    daily_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 30])         # days
    intraday_windows: List[int] = field(default_factory=lambda: [4, 12, 24, 120]) # hours
    daily_windows: List[int] = field(default_factory=lambda: [5, 10, 30])         # days
    cross_account_corr_window: int = 48      # periods for rolling flow correlation
    regime_vol_window: int = 24 * 5          # periods for regime volatility estimate
    regime_z_threshold: float = 2.0          # z-score of vol marking a stress regime
    use_external: bool = True


@dataclass
class ModelConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 4, 24])   # in base periods
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.50, 0.90, 0.95])
    conformal: bool = True                   # CQR calibration of quantile models
    calibration_fraction: float = 0.2        # tail of train set held out for conformal
    use_lightgbm: bool = True
    use_xgboost: bool = True
    lgb_params: Dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1))
    xgb_params: Dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.03, max_depth=5, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0))
    cv_folds: int = 3                        # expanding-window folds
    residual_target: bool = True             # subtract known + excluded flows from the target
    random_state: int = 42


@dataclass
class SplitConfig:
    train_end: str = "2025-10-31"
    valid_end: str = "2025-12-31"
    test_end: Optional[str] = None           # None -> to end of data


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    # weight used by criticality-weighted MAPE; defaults to 1 for unknown accounts
    account_criticality: Dict[str, float] = field(default_factory=dict)
    # anomaly threshold on |actual - forecast| / interval width; None -> use per-account default
    anomaly_threshold: Optional[float] = None
    per_account_anomaly_threshold: Dict[str, float] = field(default_factory=dict)
    log_level: str = "INFO"
