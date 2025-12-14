"""
Time-based cross-validation splitting for temporal data (GPU-friendly)

Reality check:
- CV split generation is control-flow + small arrays; GPU acceleration does not materially help.
- The "GPU version" here means: (a) accept cuDF / CuPy inputs, (b) do all masking/index ops
  using CuPy when possible, (c) avoid pandas-heavy operations.

This module:
- Works with pd.Series / np.ndarray / cudf.Series / cupy.ndarray inputs.
- Returns either NumPy indices (default) or CuPy indices (return_device="gpu").
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Literal, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_import_cudf_cupy():
    cudf = None
    cp = None
    try:
        import cudf as _cudf  # type: ignore

        cudf = _cudf
    except Exception:
        cudf = None
    try:
        import cupy as _cp  # type: ignore

        cp = _cp
    except Exception:
        cp = None
    return cudf, cp


def _to_datetime64ns_utc_numpy(x: Any) -> np.ndarray:
    """
    Convert timestamps-like to numpy datetime64[ns] in UTC (naive representation).
    """
    # cudf.Series
    cudf, _ = _try_import_cudf_cupy()
    if cudf is not None and isinstance(x, cudf.Series):
        s = x
        s = cudf.to_datetime(s, utc=True, errors="coerce")
        s = s.dropna()
        # to_numpy yields numpy datetime64[ns]
        return s.to_numpy()

    # cupy array of datetime64 is uncommon; fall back to pandas
    if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
        s = pd.to_datetime(pd.Series(x), utc=True, errors="coerce")
        s = s.dropna()
        return s.to_numpy()

    # pandas Series
    if isinstance(x, pd.Series):
        s = pd.to_datetime(x, utc=True, errors="coerce").dropna()
        return s.to_numpy()

    # anything else: attempt pandas conversion
    s = pd.to_datetime(pd.Series(x), utc=True, errors="coerce").dropna()
    return s.to_numpy()


def _floor_hour_np(dt64_ns: np.ndarray) -> np.ndarray:
    """
    Floor numpy datetime64[ns] to hour.
    """
    # Convert to hours then back to ns
    dt_h = dt64_ns.astype("datetime64[h]")
    return dt_h.astype("datetime64[ns]")


def create_rolling_origin_cv_gpu(
    hour_ts: Any,
    n_folds: int = 3,
    val_window_hours: int = 24,
    gap_hours: int = 0,
    return_device: Literal["cpu", "gpu"] = "cpu",
) -> List[Tuple[Any, Any]]:
    """
    Create time-blocked rolling-origin CV splits on hour_ts.

    Args:
        hour_ts: timestamps (already floored or not). Accepts pd.Series, np.ndarray,
                 cudf.Series, list-like.
        n_folds: number of folds
        val_window_hours: validation window length
        gap_hours: optional gap between train and validation
        return_device: "cpu" -> returns numpy indices; "gpu" -> returns cupy indices (if available)

    Returns:
        list of (train_idx, test_idx)
    """
    cudf, cp = _try_import_cudf_cupy()
    dt = _to_datetime64ns_utc_numpy(hour_ts)
    if dt.size == 0:
        logger.warning("create_rolling_origin_cv_gpu: empty hour_ts after parsing")
        return []

    # Ensure hour granularity
    dt = _floor_hour_np(dt)

    # Unique hours for windows
    unique_hours = np.unique(dt)
    unique_hours.sort()
    end_hour = unique_hours[-1]

    # timedelta in numpy units
    np.timedelta64(1, "h")
    val_window = np.timedelta64(int(val_window_hours), "h")
    gap = np.timedelta64(int(gap_hours), "h")

    splits: List[Tuple[Any, Any]] = []

    # Decide index backend
    use_gpu = (return_device == "gpu") and (cp is not None)

    # Work in CPU for datetime comparisons (numpy datetime64); masks -> optionally move to GPU for indices.
    n = dt.shape[0]
    row_index = np.arange(n)

    for fold in range(n_folds):
        test_end = end_hour - np.timedelta64(int(fold * val_window_hours), "h")
        test_start = test_end - val_window
        train_end = test_start - gap

        train_mask = dt < train_end
        test_mask = (dt >= test_start) & (dt < test_end)

        train_idx_cpu = row_index[train_mask]
        test_idx_cpu = row_index[test_mask]

        if train_idx_cpu.size == 0 or test_idx_cpu.size == 0:
            logger.warning(
                f"  Fold {fold+1}: Skipping (train={train_idx_cpu.size}, test={test_idx_cpu.size})"
            )
            continue

        logger.info(
            f"  Fold {fold+1}: train hours < {pd.Timestamp(train_end).isoformat()}Z, "
            f"test hours [{pd.Timestamp(test_start).isoformat()}Z, {pd.Timestamp(test_end).isoformat()}Z)"
        )
        logger.info(
            f"    Train: {train_idx_cpu.size:,} rows, Test: {test_idx_cpu.size:,} rows"
        )

        if use_gpu:
            train_idx = cp.asarray(train_idx_cpu)
            test_idx = cp.asarray(test_idx_cpu)
        else:
            train_idx = train_idx_cpu
            test_idx = test_idx_cpu

        splits.append((train_idx, test_idx))

    logger.info(f"Created {len(splits)} CV folds (time-blocked on hour_ts)")
    return splits


def create_nested_cv_gpu(
    hour_ts: Any,
    n_outer_folds: int = 3,
    n_inner_folds: int = 3,
    val_window_hours: int = 24,
    gap_hours: int = 0,
    return_device: Literal["cpu", "gpu"] = "cpu",
) -> List[Tuple[Any, Any, List[Tuple[Any, Any]]]]:
    """
    Nested rolling-origin CV for proper hyperparameter tuning.

    Returns:
        list of (outer_train_idx, outer_test_idx, inner_splits)
    """
    # Outer splits always computed from full index set
    outer_splits = create_rolling_origin_cv_gpu(
        hour_ts,
        n_folds=n_outer_folds,
        val_window_hours=val_window_hours,
        gap_hours=gap_hours,
        return_device="cpu",  # keep outer indices in CPU for slicing hour_ts reliably
    )

    # For inner splits, we create a reindexed hour_ts for the outer-train subset
    dt_full = _to_datetime64ns_utc_numpy(hour_ts)
    dt_full = _floor_hour_np(dt_full)

    cudf, cp = _try_import_cudf_cupy()
    use_gpu = (return_device == "gpu") and (cp is not None)

    nested: List[Tuple[Any, Any, List[Tuple[Any, Any]]]] = []

    for outer_fold_idx, (outer_train_idx_cpu, outer_test_idx_cpu) in enumerate(
        outer_splits
    ):
        logger.info(f"\n  Outer Fold {outer_fold_idx + 1}:")
        logger.info(f"    Outer train: {outer_train_idx_cpu.size:,} rows")
        logger.info(f"    Outer test:  {outer_test_idx_cpu.size:,} rows")

        # Build hour_ts for outer training only, reindexed 0..len-1
        dt_train = dt_full[outer_train_idx_cpu]
        # Inner splits are indices relative to dt_train
        inner_splits = create_rolling_origin_cv_gpu(
            dt_train,
            n_folds=n_inner_folds,
            val_window_hours=val_window_hours,
            gap_hours=gap_hours,
            return_device=return_device,
        )
        logger.info(f"    Inner CV: {len(inner_splits)} folds")

        if use_gpu:
            outer_train_idx = cp.asarray(outer_train_idx_cpu)
            outer_test_idx = cp.asarray(outer_test_idx_cpu)
        else:
            outer_train_idx = outer_train_idx_cpu
            outer_test_idx = outer_test_idx_cpu

        nested.append((outer_train_idx, outer_test_idx, inner_splits))

    logger.info(f"\nCreated {len(nested)} nested CV folds")
    return nested


def create_sliding_backtest_cv_gpu(
    timestamps: Any,
    n_folds: int = 3,
    val_window_hours: int = 3,
    step_hours: int = 1,
    return_device: Literal["cpu", "gpu"] = "cpu",
) -> List[Tuple[Any, Any]]:
    """
    Sliding backtest CV (demo metric)

    Notes:
    - Sorting timestamps dominates; GPU not meaningfully helpful.
    - This version supports cuDF/cupy inputs and can output GPU indices.

    Args:
        timestamps: timestamps-like
        n_folds: number of folds
        val_window_hours: validation window size
        step_hours: how far to slide back each fold (default 1 hour)
        return_device: "cpu" or "gpu" indices

    Returns:
        list of (train_idx, test_idx)
    """
    cudf, cp = _try_import_cudf_cupy()
    dt = _to_datetime64ns_utc_numpy(timestamps)
    if dt.size == 0:
        logger.warning("create_sliding_backtest_cv_gpu: empty timestamps after parsing")
        return []

    # Sort by time
    sorted_idx = np.argsort(dt)
    dt_sorted = dt[sorted_idx]

    end_time = dt_sorted[-1]
    val_window = np.timedelta64(int(val_window_hours), "h")
    np.timedelta64(int(step_hours), "h")

    use_gpu = (return_device == "gpu") and (cp is not None)
    splits: List[Tuple[Any, Any]] = []

    for fold in range(n_folds):
        test_end = end_time - (np.timedelta64(fold, "h") * step_hours)
        test_start = test_end - val_window

        train_mask = dt_sorted < test_start
        test_mask = (dt_sorted >= test_start) & (dt_sorted < test_end)

        train_idx_cpu = sorted_idx[train_mask]
        test_idx_cpu = sorted_idx[test_mask]

        if train_idx_cpu.size == 0:
            logger.warning(f"Backtest fold {fold+1}: No training data")
            continue
        if test_idx_cpu.size == 0:
            logger.warning(f"Backtest fold {fold+1}: No test data")
            continue

        logger.info(
            f"Backtest fold {fold+1}: Train={train_idx_cpu.size}, Test={test_idx_cpu.size}, "
            f"Test window=[{pd.Timestamp(test_start).isoformat()}Z, {pd.Timestamp(test_end).isoformat()}Z)"
        )

        if use_gpu:
            splits.append((cp.asarray(train_idx_cpu), cp.asarray(test_idx_cpu)))
        else:
            splits.append((train_idx_cpu, test_idx_cpu))

    return splits
