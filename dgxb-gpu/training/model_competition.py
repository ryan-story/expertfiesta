"""
GPU Model Competition (cuML-first, with XGBoost GPU as the GBDT workhorse)

- cuML: LinearRegression, RandomForestRegressor, StandardScaler
- XGBoost: GPU hist (optional but strongly recommended)
- Manual HPO (grid/random) with rolling-origin CV splits

Notes:
- cuML does NOT currently offer a first-class PoissonRegressor equivalent like sklearn.
  For count data, use XGBoost objective='count:poisson' on GPU.
- This module assumes X_train_cp and y_train_cp are already CuPy arrays (float32).
- CV splits are list[(train_idx, test_idx)] with positional indices (as in your pipeline).

Dependencies:
  pip install cupy-cuda12x cudf-cu12 cuml-cu12 xgboost
  (exact packages vary by CUDA version; use your RAPIDS channel instructions)

"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import cupy as cp

from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
from cuml.preprocessing import StandardScaler as cuStandardScaler

import xgboost as xgb

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GPU utilities
# -----------------------------------------------------------------------------


def _ensure_cupy_float32(X: cp.ndarray) -> cp.ndarray:
    """Ensure CuPy array is float32 and contiguous."""
    if X.dtype != cp.float32:
        X = X.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(X)


def _ensure_cupy_y(y: cp.ndarray) -> cp.ndarray:
    """Ensure CuPy array is float32 and contiguous."""
    if y.dtype != cp.float32:
        y = y.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(y)


def _gpu_col_median_impute_inplace(X_cp: cp.ndarray) -> Dict[str, Any]:
    """
    In-place median imputation for NaNs (GPU).
    Returns stats for debugging.
    """
    nan_mask = cp.isnan(X_cp)
    if not bool(nan_mask.any()):
        return {"imputed": False, "n_nans": 0}

    # Column medians ignoring NaNs
    med = cp.nanmedian(X_cp, axis=0)
    # Replace NaN medians (all-NaN column) with 0
    med = cp.where(cp.isnan(med), cp.zeros_like(med), med)

    # Broadcast medians to NaN positions
    rows, cols = cp.where(nan_mask)
    X_cp[rows, cols] = med[cols]

    return {"imputed": True, "n_nans": int(nan_mask.sum().get())}


def _rmse_gpu(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    err = y_true - y_pred
    return float(cp.sqrt(cp.mean(err * err)).get())


def _make_cv_slices(
    X_cp: cp.ndarray,
    y_cp: cp.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    tr = cp.asarray(train_idx, dtype=cp.int32)
    te = cp.asarray(test_idx, dtype=cp.int32)

    X_tr = X_cp.take(tr, axis=0)
    X_te = X_cp.take(te, axis=0)
    y_tr = y_cp.take(tr, axis=0)
    y_te = y_cp.take(te, axis=0)
    return X_tr, X_te, y_tr, y_te


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
# cuML models (GPU)
# -----------------------------------------------------------------------------


def train_cuml_linear_regression(
    X_train_cp: cp.ndarray,
    y_train_cp: cp.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], float]:
    """
    cuML LinearRegression with simple grid over fit_intercept + normalize-like scaling.

    Args:
        X_train_cp: CuPy array (float32) of shape (n_samples, n_features)
        y_train_cp: CuPy array (float32) of shape (n_samples,)
        cv_splits: List of (train_idx, test_idx) tuples with numpy arrays

    Returns:
      best_model_bundle: dict with {"scaler": scaler_or_None, "model": model}
    """
    logger.info("Training cuML LinearRegression (GPU)...")
    t0 = time.time()

    X_cp = _ensure_cupy_float32(X_train_cp)
    y_cp = _ensure_cupy_y(y_train_cp)
    impute_stats = _gpu_col_median_impute_inplace(X_cp)
    if impute_stats["imputed"]:
        logger.info(f"  Median-imputed NaNs (GPU): {impute_stats['n_nans']:,}")

    # Small grid
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
            X_tr, X_te, y_tr, y_te = _make_cv_slices(X_cp, y_cp, train_idx, test_idx)

            # Always use scaler for cuML LinearRegression
            scaler = cuStandardScaler(with_mean=True, with_std=True)
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = cuLinearRegression(
                fit_intercept=params["fit_intercept"],
                output_type="cupy",
            )
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)
            fold_rmses.append(_rmse_gpu(y_te, y_pred))

        rmse_mean = float(np.mean(fold_rmses))
        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_params = dict(params)
            # Refit bundle on full data
            scaler_full = cuStandardScaler(with_mean=True, with_std=True)
            X_full = scaler_full.fit_transform(X_cp)

            model_full = cuLinearRegression(
                fit_intercept=params["fit_intercept"],
                output_type="cupy",
            )
            model_full.fit(X_full, y_cp)
            best_bundle = {"scaler": scaler_full, "model": model_full}

    train_time = time.time() - t0
    if best_bundle is None:
        raise RuntimeError("cuML LinearRegression failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return best_bundle, best_params, {"rmse": best_rmse}, train_time


def train_cuml_random_forest(
    X_train_cp: cp.ndarray,
    y_train_cp: cp.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    cuML RandomForestRegressor (GPU) with random search.

    cuML RF can be very fast on GPU, but memory-heavy; keep depth/trees reasonable.

    Args:
        X_train_cp: CuPy array (float32) of shape (n_samples, n_features)
        y_train_cp: CuPy array (float32) of shape (n_samples,)
        cv_splits: List of (train_idx, test_idx) tuples with numpy arrays
        n_trials: Number of random search trials
    """
    logger.info("Training cuML RandomForestRegressor (GPU)...")
    t0 = time.time()

    seed = 42  # Fixed seed for reproducibility
    X_cp = _ensure_cupy_float32(X_train_cp)
    y_cp = _ensure_cupy_y(y_train_cp)
    impute_stats = _gpu_col_median_impute_inplace(X_cp)
    if impute_stats["imputed"]:
        logger.info(f"  Median-imputed NaNs (GPU): {impute_stats['n_nans']:,}")

    param_space = {
        "n_estimators": [200, 400, 800],
        "max_depth": [8, 12, 16, 24],
        "max_features": [0.5, 0.7, 1.0],  # fraction of features per split
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    trials = _sample_param_dicts(param_space, n_trials=n_trials, seed=seed)

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[cuRandomForestRegressor] = None
    best_rmse_std: float = 0.0

    for i, params in enumerate(trials, 1):
        if i % 5 == 0 or i == 1:
            logger.info(f"  Trial {i}/{len(trials)}: {params}")

        fold_rmses: List[float] = []
        for train_idx, test_idx in cv_splits:
            X_tr, X_te, y_tr, y_te = _make_cv_slices(X_cp, y_cp, train_idx, test_idx)

            model = cuRandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                max_features=float(params["max_features"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                bootstrap=bool(params["bootstrap"]),
                random_state=seed,
                n_streams=1,
                output_type="cupy",
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            fold_rmses.append(_rmse_gpu(y_te, y_pred))

        rmse_mean = float(np.mean(fold_rmses))
        rmse_std = float(np.std(fold_rmses))

        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_rmse_std = rmse_std
            best_params = dict(params)
            # Refit on full data
            best_model = cuRandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                max_features=float(params["max_features"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                bootstrap=bool(params["bootstrap"]),
                random_state=seed,
                n_streams=1,
                output_type="cupy",
            )
            best_model.fit(X_cp, y_cp)

    train_time = time.time() - t0
    if best_model is None:
        raise RuntimeError("cuML RandomForestRegressor failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f} ± {best_rmse_std:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return (
        best_model,
        best_params,
        {"rmse": best_rmse, "rmse_std": best_rmse_std},
        train_time,
    )


# -----------------------------------------------------------------------------
# XGBoost GPU (recommended for best accuracy on tabular)
# -----------------------------------------------------------------------------


def train_xgboost_gpu(
    X_train_cp: cp.ndarray,
    y_train_cp: cp.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 30,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    XGBoost GPU with random search. Uses CuPy matrices for GPU-native training.

    If GPU training fails at runtime (no CUDA build), falls back to CPU hist.

    Args:
        X_train_cp: CuPy array (float32) of shape (n_samples, n_features)
        y_train_cp: CuPy array (float32) of shape (n_samples,)
        cv_splits: List of (train_idx, test_idx) tuples with numpy arrays
        n_trials: Number of random search trials
    """
    logger.info("Training XGBoost (GPU-first)...")
    t0 = time.time()

    seed = 42  # Fixed seed for reproducibility
    prefer_gpu = True
    X_cp = _ensure_cupy_float32(X_train_cp)
    y_cp = _ensure_cupy_y(y_train_cp)

    # XGBoost can handle NaNs natively; keep them (no imputation required).

    param_space = {
        "n_estimators": [300, 600, 1200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_alpha": [0.0, 1e-3, 1e-2, 1e-1],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        # For counts, include poisson. For general regression, include squarederror.
        "objective": ["count:poisson", "reg:squarederror"],
    }
    trials = _sample_param_dicts(param_space, n_trials=n_trials, seed=seed)

    # GPU-only: enforce gpu_hist (no CPU fallback in GPU module)
    if not (prefer_gpu and _has_cuda_gpu()):
        raise RuntimeError(
            "XGBoost GPU mode required in dgxb-gpu module. CUDA GPU not available."
        )
    tree_method = "gpu_hist"
    device = "cuda"

    best_rmse = float("inf")
    best_params: Dict[str, Any] = {}
    best_model: Optional[xgb.XGBRegressor] = None
    best_rmse_std: float = 0.0

    for i, params in enumerate(trials, 1):
        if i % 5 == 0 or i == 1:
            logger.info(f"  Trial {i}/{len(trials)} ({device}): {params}")

        fold_rmses: List[float] = []
        failed = False

        for train_idx, test_idx in cv_splits:
            X_tr, X_te, y_tr, y_te = _make_cv_slices(X_cp, y_cp, train_idx, test_idx)

            # XGBoost sklearn API can take CuPy arrays directly
            model = xgb.XGBRegressor(
                tree_method=tree_method,
                random_state=seed,
                n_jobs=1,
                **params,
            )

            try:
                # For poisson, ensure non-negative
                if params["objective"] == "count:poisson":
                    y_tr_fit = cp.clip(y_tr, 0.0, None)
                    y_te_eval = cp.clip(y_te, 0.0, None)
                else:
                    y_tr_fit = y_tr
                    y_te_eval = y_te

                model.fit(X_tr, y_tr_fit)
                y_pred = model.predict(X_te)
                y_pred_cp = (
                    cp.asarray(y_pred) if not isinstance(y_pred, cp.ndarray) else y_pred
                )
                fold_rmses.append(_rmse_gpu(y_te_eval, y_pred_cp))
            except Exception as e:
                logger.error(f"    XGBoost GPU trial failed: {e}")
                failed = True
                break

        if failed:
            # GPU-only: no CPU fallback
            logger.error(
                "XGBoost GPU training failed. Check CUDA availability and XGBoost GPU build."
            )
            continue

        rmse_mean = float(np.mean(fold_rmses))
        rmse_std = float(np.std(fold_rmses))

        if rmse_mean < best_rmse:
            best_rmse = rmse_mean
            best_rmse_std = rmse_std
            best_params = dict(params)

            best_model = xgb.XGBRegressor(
                tree_method=tree_method,
                random_state=seed,
                n_jobs=1,
                **params,
            )

            # Fit on full data
            if params["objective"] == "count:poisson":
                y_fit = cp.clip(y_cp, 0.0, None)
            else:
                y_fit = y_cp
            best_model.fit(X_cp, y_fit)

    train_time = time.time() - t0
    if best_model is None:
        raise RuntimeError("XGBoost trainer failed to produce a model.")

    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f} ± {best_rmse_std:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")

    return (
        best_model,
        best_params,
        {"rmse": best_rmse, "rmse_std": best_rmse_std},
        train_time,
    )


# -----------------------------------------------------------------------------
# Unified entrypoint
# -----------------------------------------------------------------------------


def train_model_with_hpo_gpu(
    model_name: str,
    X_train: cp.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    hpo_method: str = "random",
    n_trials: int = 30,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any], Dict[str, float], float]:
    """
    GPU-first model competition.

    model_name:
      - "cuML_LinearRegression"
      - "cuML_RandomForest"
      - "XGBoost_GPU"

    hpo_method is currently meaningful for:
      - cuML_RandomForest (random via n_trials)
      - XGBoost_GPU (random via n_trials)
      - cuML_LinearRegression uses a small grid internally

    Returns: (best_model_or_bundle, best_params, cv_scores, train_time_sec)
    """
    if model_name == "cuML_LinearRegression":
        return train_cuml_linear_regression(
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            use_scaler=True,
        )

    if model_name == "cuML_RandomForest":
        return train_cuml_random_forest(
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            n_trials=n_trials,
            seed=seed,
        )

    if model_name == "XGBoost_GPU":
        return train_xgboost_gpu(
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            n_trials=n_trials,
            seed=seed,
            prefer_gpu=True,
        )

    raise ValueError(f"Unknown model_name='{model_name}'.")


def _has_cuda_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False
