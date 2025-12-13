"""
Training pipeline for DGXB
Includes model competition, cross-validation, and metrics tracking
"""

from .cv_splitter import create_rolling_origin_cv, create_sliding_backtest_cv
from .model_competition import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_model_with_hpo,
)
from .metrics_tracker import (
    compute_hotspot_metrics,
    measure_inference_latency,
    compute_model_quality_metrics,
    track_pipeline_metrics,
)
from .pipeline import run_training_competition

__all__ = [
    "create_rolling_origin_cv",
    "create_sliding_backtest_cv",
    "train_logistic_regression",
    "train_random_forest",
    "train_xgboost",
    "train_model_with_hpo",
    "compute_hotspot_metrics",
    "measure_inference_latency",
    "compute_model_quality_metrics",
    "track_pipeline_metrics",
    "run_training_competition",
]
