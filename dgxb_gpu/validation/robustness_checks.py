"""
Robustness validation checks (GPU / RAPIDS / XGBoost GPU)

Key fixes:
- time-safe horizon evaluation (rolling-origin CV, not random split)
- re-instantiate model per fold
"""

import logging
from typing import Dict, Any, List

import numpy as np

import cudf
import cupy as cp
import xgboost as xgb

from pathlib import Path
from dgxb_gpu.training.cv_splitter import create_rolling_origin_cv_gpu

logger = logging.getLogger(__name__)


# --------------------------
# GPU metric helpers (CuPy)
# --------------------------


def _rmse_cp(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    err = y_true - y_pred
    return float(cp.sqrt(cp.mean(err * err)).get())


def _mae_cp(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    return float(cp.mean(cp.abs(y_true - y_pred)).get())


def _r2_cp(y_true: cp.ndarray, y_pred: cp.ndarray) -> float:
    ss_res = cp.sum((y_true - y_pred) ** 2)
    ss_tot = cp.sum((y_true - cp.mean(y_true)) ** 2)
    # guard for constant targets
    r2 = 1.0 - (ss_res / (ss_tot + cp.float32(1e-12)))
    return float(r2.get())


def _xgb_gpu_regressor(seed: int = 42) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=200,
        max_depth=10,
        learning_rate=0.2,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        device="cuda",
        n_jobs=1,
    )


def _cudf_numeric(df: cudf.DataFrame) -> cudf.DataFrame:
    out = df.copy(deep=False)
    for c in out.columns:
        if out[c].dtype == "bool":
            out[c] = out[c].astype("int8")
    # numeric only
    cols = [c for c in out.columns if cudf.api.types.is_numeric_dtype(out[c].dtype)]
    return out[cols]


def _median_impute(df: cudf.DataFrame) -> cudf.DataFrame:
    return df.fillna(df.median())


def _to_cp_2d(df: cudf.DataFrame) -> cp.ndarray:
    X = df.to_cupy()
    if X.dtype != cp.float32:
        X = X.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(X)


def _to_cp_1d(s: cudf.Series) -> cp.ndarray:
    y = s.to_cupy()
    if y.dtype != cp.float32:
        y = y.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(y)


# --------------------------
# Test 1 (GPU): remove incident_count* features
# --------------------------


def test_without_incident_count_t(
    X_rich: cudf.DataFrame,
    y: cudf.Series,
    hour_ts: cudf.Series,  # GPU-friendly: accepts cudf Series
    h3_cells: np.ndarray,  # unused here; keep signature
    n_folds: int = 3,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("ROBUSTNESS TEST 1 (GPU): Remove incident_count* features")
    logger.info("=" * 80)

    # columns to remove
    columns_to_remove = [
        col
        for col in X_rich.columns
        if any(
            k in col.lower()
            for k in [
                "incident_count",
                "incident_count_t",
                "incident_count_lag",
                "incident_count_rolling",
            ]
        )
    ]

    if columns_to_remove:
        logger.info(
            f"  Removing {len(columns_to_remove)} incident_count-related columns"
        )
        for c in columns_to_remove[:10]:
            logger.info(f"    - {c}")
        if len(columns_to_remove) > 10:
            logger.info(f"    ... and {len(columns_to_remove) - 10} more")
        Xf = X_rich.drop(columns=columns_to_remove)
    else:
        logger.info("  ✓ No incident_count-related features found in X")
        Xf = X_rich

    Xf = _median_impute(_cudf_numeric(Xf))

    # CV splits (GPU-friendly: accepts cudf Series)
    cv_splits = create_rolling_origin_cv_gpu(
        hour_ts, n_folds=n_folds, val_window_hours=24, gap_hours=1, return_device="cpu"
    )

    X_cp = _to_cp_2d(Xf)
    y_cp = _to_cp_1d(y)

    y_true_all = []
    y_pred_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        model = _xgb_gpu_regressor(seed=42 + fold_idx)

        tr = cp.asarray(train_idx, dtype=cp.int32)
        te = cp.asarray(test_idx, dtype=cp.int32)

        X_train = X_cp.take(tr, axis=0)
        y_train = y_cp.take(tr, axis=0)
        X_test = X_cp.take(te, axis=0)
        y_test = y_cp.take(te, axis=0)

        model.fit(X_train, y_train)
        y_pred = cp.asarray(model.predict(X_test))
        y_pred = cp.maximum(y_pred, 0.0)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_all = cp.concatenate(y_true_all)
    y_pred_all = cp.concatenate(y_pred_all)

    results = {
        "test_name": "Without incident_count* features (GPU)",
        "rmse": _rmse_cp(y_true_all, y_pred_all),
        "mae": _mae_cp(y_true_all, y_pred_all),
        "r2": _r2_cp(y_true_all, y_pred_all),
        "n_features_removed": len(columns_to_remove),
        "n_features_remaining": int(Xf.shape[1]),
    }

    logger.info(f"  RMSE: {results['rmse']:.4f}")
    logger.info(f"  MAE:  {results['mae']:.4f}")
    logger.info(f"  R²:   {results['r2']:.4f}")
    return results


# --------------------------
# Test 2 (GPU): horizon sensitivity (FIXED: time-blocked CV)
# --------------------------


def test_horizon_sensitivity(
    X_rich: cudf.DataFrame,
    y_target_df: cudf.DataFrame,  # must include ['h3_cell','hour_ts','incident_count_t_plus_1']
    hour_ts: cudf.Series,  # GPU-friendly: accepts cudf Series
    h3_cells: np.ndarray,
    horizons: List[int] = [1, 3, 6],
    n_folds: int = 3,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("ROBUSTNESS TEST 2 (GPU): Horizon Sensitivity with rolling-origin CV")
    logger.info("=" * 80)

    X_num = _median_impute(_cudf_numeric(X_rich))

    # We will build y_{t+h} by shifting per cell in a time-sorted table on GPU.
    base = y_target_df[["h3_cell", "hour_ts", "incident_count_t_plus_1"]].copy()
    base["hour_ts"] = cudf.to_datetime(base["hour_ts"], utc=True)

    # Sort for stable groupby-shift
    base = base.sort_values(by=["h3_cell", "hour_ts"]).reset_index(drop=True)

    results_by_horizon = {}

    for h in horizons:
        logger.info(f"\n  Horizon t+{h}")

        if h == 1:
            base[f"y_t_plus_{h}"] = base["incident_count_t_plus_1"]
        else:
            # To get t+h, shift the t+1 label by (h-1) steps backward within each cell
            base[f"y_t_plus_{h}"] = base.groupby("h3_cell")[
                "incident_count_t_plus_1"
            ].shift(-(h - 1))

        y_h = base[f"y_t_plus_{h}"]

        # Align X and hour_ts to the same rows as base.
        # IMPORTANT: this assumes X_rich/y_target_df were built row-aligned already.
        # If not, you must join on (h3_cell, hour_ts) instead of relying on row order.
        valid = ~y_h.isna()
        if int(valid.sum()) == 0:
            results_by_horizon[h] = {
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "n_samples": 0,
            }
            logger.warning("    No valid samples for this horizon")
            continue

        Xh = X_num[valid].reset_index(drop=True)
        yh = y_h[valid].reset_index(drop=True)
        hour_h = hour_ts[valid].reset_index(drop=True)  # Keep as cudf Series

        cv_splits = create_rolling_origin_cv_gpu(
            hour_h,
            n_folds=n_folds,
            val_window_hours=24,
            gap_hours=1,
            return_device="cpu",
        )

        X_cp = _to_cp_2d(Xh)
        y_cp = _to_cp_1d(yh)

        y_true_all = []
        y_pred_all = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            model = _xgb_gpu_regressor(seed=1000 + h * 10 + fold_idx)

            tr = cp.asarray(train_idx, dtype=cp.int32)
            te = cp.asarray(test_idx, dtype=cp.int32)

            X_train = X_cp.take(tr, axis=0)
            y_train = y_cp.take(tr, axis=0)
            X_test = X_cp.take(te, axis=0)
            y_test = y_cp.take(te, axis=0)

            model.fit(X_train, y_train)
            y_pred = cp.asarray(model.predict(X_test))
            y_pred = cp.maximum(y_pred, 0.0)

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

        y_true_all = cp.concatenate(y_true_all)
        y_pred_all = cp.concatenate(y_pred_all)

        results_by_horizon[h] = {
            "rmse": _rmse_cp(y_true_all, y_pred_all),
            "mae": _mae_cp(y_true_all, y_pred_all),
            "r2": _r2_cp(y_true_all, y_pred_all),
            "n_samples": int(len(yh)),
        }

        logger.info(f"    RMSE: {results_by_horizon[h]['rmse']:.4f}")
        logger.info(f"    MAE:  {results_by_horizon[h]['mae']:.4f}")
        logger.info(f"    R²:   {results_by_horizon[h]['r2']:.4f}")

    return {
        "test_name": "Horizon Sensitivity (GPU, time-safe CV)",
        "results_by_horizon": results_by_horizon,
    }


# --------------------------
# Test 3 (GPU): sector cold-start (cells held out) + time-safe evaluation
# --------------------------


def test_sector_cold_start(
    X_rich: cudf.DataFrame,
    y: cudf.Series,
    hour_ts: cudf.Series,  # GPU-friendly: accepts cudf Series
    h3_cells: np.ndarray,
    test_cell_fraction: float = 0.2,
    n_folds: int = 3,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(
        "ROBUSTNESS TEST 3 (GPU): Sector cold-start (hold out cells) + time-safe CV"
    )
    logger.info("=" * 80)

    unique_cells = np.unique(h3_cells)
    n_test_cells = max(1, int(len(unique_cells) * test_cell_fraction))

    rng = np.random.default_rng(42)
    test_cells = rng.choice(unique_cells, size=n_test_cells, replace=False)

    cell_is_test = np.isin(h3_cells, test_cells)
    cell_is_train = ~cell_is_test

    # Train set: only train cells
    X_train_full = _median_impute(_cudf_numeric(X_rich[cell_is_train]))
    y_train_full = y[cell_is_train]
    hour_train = hour_ts[cell_is_train].reset_index(drop=True)  # Keep as cudf Series

    # Test set: only held-out cells
    X_test_full = _median_impute(_cudf_numeric(X_rich[cell_is_test]))
    y_test_full = y[cell_is_test]
    hour_test = hour_ts[cell_is_test].reset_index(drop=True)  # Keep as cudf Series

    # Time-safe: build splits on TRAIN hours only, evaluate on corresponding future windows in held-out cells
    # Simplest robust form: choose the same (last 24h * n_folds) windows on held-out cells and ensure train uses only earlier hours.
    cv_splits_train = create_rolling_origin_cv_gpu(
        hour_train,
        n_folds=n_folds,
        val_window_hours=24,
        gap_hours=1,
        return_device="cpu",
    )

    X_train_cp = _to_cp_2d(X_train_full)
    y_train_cp = _to_cp_1d(y_train_full)

    # For evaluation, we will use the held-out cells restricted to each fold's test hours
    # so the temporal question remains consistent.
    y_true_all = []
    y_pred_all = []

    # Precompute held-out arrays (GPU-friendly: keep as cudf Series)
    X_test_cp = _to_cp_2d(X_test_full)
    y_test_cp = _to_cp_1d(y_test_full)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits_train):
        # Determine the fold's test hour window from TRAIN data
        test_hours = hour_train.iloc[test_idx]
        tmin = test_hours.min()
        tmax = test_hours.max()

        # Select held-out samples in same window (use hour_test cudf Series)
        mask = (hour_test >= tmin) & (hour_test <= tmax)
        if mask.sum() == 0:
            continue

        model = _xgb_gpu_regressor(seed=5000 + fold_idx)

        tr = cp.asarray(train_idx, dtype=cp.int32)
        model.fit(X_train_cp.take(tr, axis=0), y_train_cp.take(tr, axis=0))

        te_mask_idx = np.where(mask.values)[0]
        te = cp.asarray(te_mask_idx, dtype=cp.int32)

        y_pred = cp.asarray(model.predict(X_test_cp.take(te, axis=0)))
        y_pred = cp.maximum(y_pred, 0.0)

        y_true_all.append(y_test_cp.take(te, axis=0))
        y_pred_all.append(y_pred)

    if not y_true_all:
        return {
            "test_name": "Sector Cold-Start (GPU, time-safe)",
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "n_train_cells": int(len(unique_cells) - len(test_cells)),
            "n_test_cells": int(len(test_cells)),
            "n_test_samples": int(cell_is_test.sum()),
            "note": "No held-out samples fell into the evaluated test windows.",
        }

    y_true_all = cp.concatenate(y_true_all)
    y_pred_all = cp.concatenate(y_pred_all)

    results = {
        "test_name": "Sector Cold-Start (GPU, time-safe)",
        "rmse": _rmse_cp(y_true_all, y_pred_all),
        "mae": _mae_cp(y_true_all, y_pred_all),
        "r2": _r2_cp(y_true_all, y_pred_all),
        "n_train_cells": int(len(unique_cells) - len(test_cells)),
        "n_test_cells": int(len(test_cells)),
        "n_train_samples": int(cell_is_train.sum()),
        "n_test_samples": int(cell_is_test.sum()),
    }

    logger.info(f"  RMSE: {results['rmse']:.4f}")
    logger.info(f"  MAE:  {results['mae']:.4f}")
    logger.info(f"  R²:   {results['r2']:.4f}")
    return results


def run_all_robustness_checks(
    rich_x_path: str = "rich-gold-gpu-traffic/X_features.parquet",
    y_path: str = "gold-gpu-traffic/y_target.parquet",
    output_path: str = "results/robustness_checks_gpu.json",
) -> Dict[str, Any]:
    import json

    logger.info("=" * 80)
    logger.info("RUNNING ALL ROBUSTNESS CHECKS (GPU)")
    logger.info("=" * 80)

    logger.info("\nLoading data to GPU (cuDF)...")
    X_rich = cudf.read_parquet(rich_x_path)
    y_target_df = cudf.read_parquet(y_path)

    y = y_target_df["incident_count_t_plus_1"]
    hour_ts = cudf.to_datetime(y_target_df["hour_ts"], utc=True)
    # CPU helper: cell sampling (small array conversion acceptable)
    h3_cells = y_target_df["h3_cell"].to_pandas().values  # for cell sampling

    valid = ~y.isna()
    X_rich = X_rich[valid].reset_index(drop=True)
    y_target_df = y_target_df[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    hour_ts = hour_ts[valid].reset_index(drop=True)  # Keep as cudf Series

    logger.info(f"  Loaded {len(X_rich):,} samples, {len(X_rich.columns)} cols")

    results: Dict[str, Any] = {}

    try:
        results["test_1_without_incident_count_t"] = test_without_incident_count_t(
            X_rich, y, hour_ts, h3_cells
        )
    except Exception as e:
        logger.exception("Test 1 failed")
        results["test_1_without_incident_count_t"] = {"error": str(e)}

    try:
        results["test_2_horizon_sensitivity"] = test_horizon_sensitivity(
            X_rich, y_target_df, hour_ts, h3_cells
        )
    except Exception as e:
        logger.exception("Test 2 failed")
        results["test_2_horizon_sensitivity"] = {"error": str(e)}

    try:
        results["test_3_sector_cold_start"] = test_sector_cold_start(
            X_rich, y, hour_ts, h3_cells
        )
    except Exception as e:
        logger.exception("Test 3 failed")
        results["test_3_sector_cold_start"] = {"error": str(e)}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Saved GPU robustness results to {output_path}")
    return results
