"""
GPU Model Competition (XGBoost GPU + scikit-learn fallbacks)

- XGBoost: GPU hist (primary workhorse for GBDT)
- scikit-learn: LinearRegression, RandomForest (CPU fallbacks when cuML unavailable)

This version works on ARM64 (Grace Blackwell) where cuML is not available.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Try to import cupy, fall back to numpy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np  # type: ignore
    HAS_CUPY = False

# scikit-learn fallbacks
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost as xgb

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GPU utilities
# -----------------------------------------------------------------------------

def _has_cuda_gpu() -> bool:
    """Check if CUDA GPU is available for XGBoost."""
    try:
        # Try to create a small XGBoost model with GPU
        import xgboost as xgb
        # Check if GPU device is available by trying to use it
        test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        import numpy as np
        X_test = np.random.rand(10, 2).astype(np.float32)
        y_test = np.random.rand(10).astype(np.float32)
        test_model.fit(X_test, y_test, verbose=False)
        return True
    except Exception:
        return False


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array to numpy (from cupy or pandas)."""
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    elif hasattr(arr, 'values'):  # pandas
        return arr.values
    return np.asarray(arr)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sample_param_dicts(
    param_space: Dict[str, List[Any]], n_trials: int, seed: int = 42
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(param_space.keys())
    trials: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        d = {k: rng.choice(param_space[k]) for k in keys}
        trials.append(d)
    return trials


# -----------------------------------------------------------------------------
# scikit-learn models (CPU fallbacks)
# -----------------------------------------------------------------------------

def train_linear_regression_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], float]:
    """
    LinearRegression with StandardScaler (CPU - scikit-learn fallback).
    """
    logger.info("Training LinearRegression (scikit-learn CPU fallback)...")
    t0 = time.time()

    X = _to_numpy(X_train)
    y = _to_numpy(y_train)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    grid = [
        {"fit_intercept": True},
        {"fit_intercept": False},
    ]

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_bundle: Optional[Dict[str, Any]] = None

    for params in grid:
        fold_rmses: List[float] = []

        for train_idx, test_idx in cv_splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = LinearRegression(fit_intercept=params["fit_intercept"])
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            fold_rmses.append(_rmse(y_te, y_pred))

        rmse_mean = float(np.mean(fold_rmses))
        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_params = dict(params)
            # Refit on full data
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(X)
            model_full = LinearRegression(fit_intercept=params["fit_intercept"])
            model_full.fit(X_full, y)
            best_bundle = {"scaler": scaler_full, "model": model_full}

    train_time = time.time() - t0
    if best_bundle is None:
        raise RuntimeError("LinearRegression failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_bundle, best_params, {"rmse": best_rmse}, train_time


def train_random_forest_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    RandomForestRegressor (CPU - scikit-learn fallback).
    """
    logger.info("Training RandomForest (scikit-learn CPU fallback)...")
    t0 = time.time()

    seed = 42
    X = _to_numpy(X_train)
    y = _to_numpy(y_train)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    param_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    trials = _sample_param_dicts(param_space, n_trials=n_trials, seed=seed)

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[RandomForestRegressor] = None
    best_rmse_std: float = 0.0

    for i, params in enumerate(trials, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"  Trial {i}/{len(trials)}: {params}")

        fold_rmses: List[float] = []
        for train_idx, test_idx in cv_splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            fold_rmses.append(_rmse(y_te, y_pred))

        rmse_mean = float(np.mean(fold_rmses))
        rmse_std = float(np.std(fold_rmses))

        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_rmse_std = rmse_std
            best_params = dict(params)
            # Refit on full data
            best_model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=params["max_depth"],
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                random_state=seed,
                n_jobs=-1,
            )
            best_model.fit(X, y)

    train_time = time.time() - t0
    if best_model is None:
        raise RuntimeError("RandomForest failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f} ± {best_rmse_std:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, {"rmse": best_rmse, "rmse_std": best_rmse_std}, train_time


# -----------------------------------------------------------------------------
# XGBoost GPU
# -----------------------------------------------------------------------------

def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    XGBoost with GPU acceleration (device='cuda').
    Falls back to CPU if GPU not available.
    """
    use_gpu = _has_cuda_gpu()
    device = "cuda" if use_gpu else "cpu"
    
    logger.info(f"Training XGBoost ({device.upper()})...")
    t0 = time.time()

    seed = 42
    X = _to_numpy(X_train)
    y = _to_numpy(y_train)

    param_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "objective": ["reg:squarederror", "count:poisson"],
    }
    trials = _sample_param_dicts(param_space, n_trials=n_trials, seed=seed)

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[xgb.XGBRegressor] = None
    best_rmse_std: float = 0.0

    for i, params in enumerate(trials, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"  Trial {i}/{len(trials)} ({device}): {params}")

        fold_rmses: List[float] = []
        failed = False

        for train_idx, test_idx in cv_splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = xgb.XGBRegressor(
                tree_method="hist",
                device=device,
                random_state=seed,
                n_jobs=1,
                **params,
            )

            try:
                # For poisson, ensure non-negative
                if params["objective"] == "count:poisson":
                    y_tr_fit = np.clip(y_tr, 0.0, None)
                else:
                    y_tr_fit = y_tr

                model.fit(X_tr, y_tr_fit, verbose=False)
                y_pred = model.predict(X_te)
                y_pred = np.clip(y_pred, 0, None)  # Clip predictions
                fold_rmses.append(_rmse(y_te, y_pred))
            except Exception as e:
                logger.warning(f"    XGBoost trial failed: {e}")
                failed = True
                break

        if failed:
            continue

        rmse_mean = float(np.mean(fold_rmses))
        rmse_std = float(np.std(fold_rmses))

        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_rmse_std = rmse_std
            best_params = dict(params)

            best_model = xgb.XGBRegressor(
                tree_method="hist",
                device=device,
                random_state=seed,
                n_jobs=1,
                **params,
            )
            # Fit on full data
            if params["objective"] == "count:poisson":
                y_fit = np.clip(y, 0.0, None)
            else:
                y_fit = y
            best_model.fit(X, y_fit, verbose=False)

    train_time = time.time() - t0
    if best_model is None:
        raise RuntimeError("XGBoost failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f} ± {best_rmse_std:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_model, best_params, {"rmse": best_rmse, "rmse_std": best_rmse_std}, train_time


# Aliases for backward compatibility
train_cuml_linear_regression = train_linear_regression_gpu
train_cuml_random_forest = train_random_forest_gpu
