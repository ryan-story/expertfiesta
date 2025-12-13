"""
Time-based cross-validation splitting for temporal data
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def create_rolling_origin_cv(
    hour_ts: pd.Series,
    n_folds: int = 3,
    val_window_hours: int = 24,
    gap_hours: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-blocked rolling-origin CV splits on hour_ts

    Time-blocked splits: split on hours, not individual rows.
    Each fold's test set is a contiguous time window (e.g., last 24h).
    Train uses only hours strictly before the test window (with optional gap).

    Args:
        hour_ts: Series of hour timestamps (floored to hour) for all data points
        n_folds: Number of CV folds
        val_window_hours: Validation window size in hours
        gap_hours: Optional gap between train and test (prevents leakage via rolling features)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    hour_ts = pd.to_datetime(hour_ts, utc=True)

    # Get unique hours and their ranges
    unique_hours = hour_ts.unique()
    unique_hours = pd.Series(unique_hours).sort_values()

    end_hour = unique_hours.max()
    splits = []

    for fold in range(n_folds):
        # Test window: [T - (fold+1)*val_window, T - fold*val_window)
        test_end = end_hour - pd.Timedelta(hours=fold * val_window_hours)
        test_start = test_end - pd.Timedelta(hours=val_window_hours)

        # Train: all hours strictly before test_start (with gap)
        train_end = test_start - pd.Timedelta(hours=gap_hours)

        # Find all rows whose hour_ts is in train/test windows
        train_mask = hour_ts < train_end
        test_mask = (hour_ts >= test_start) & (hour_ts < test_end)

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) == 0 or len(test_indices) == 0:
            logger.warning(
                f"  Fold {fold+1}: Skipping (train={len(train_indices)}, test={len(test_indices)})"
            )
            continue

        logger.info(
            f"  Fold {fold+1}: train hours < {train_end}, test hours [{test_start}, {test_end})"
        )
        logger.info(
            f"    Train: {len(train_indices):,} rows, Test: {len(test_indices):,} rows"
        )

        splits.append((train_indices, test_indices))

    logger.info(f"Created {len(splits)} CV folds (time-blocked on hour_ts)")
    return splits


def create_nested_cv(
    hour_ts: pd.Series,
    n_outer_folds: int = 3,
    n_inner_folds: int = 3,
    val_window_hours: int = 24,
    gap_hours: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Create nested time-blocked CV splits for proper hyperparameter tuning.

    Outer CV: Used for final model evaluation (never seen during hyperparameter tuning)
    Inner CV: Used for hyperparameter tuning (only uses training data from outer fold)

    Args:
        hour_ts: Series of hour timestamps (floored to hour) for all data points
        n_outer_folds: Number of outer CV folds (for final evaluation)
        n_inner_folds: Number of inner CV folds (for hyperparameter tuning)
        val_window_hours: Validation window size in hours
        gap_hours: Optional gap between train and test

    Returns:
        List of (outer_train_idx, outer_test_idx, inner_cv_splits) tuples
        where inner_cv_splits are (inner_train_idx, inner_test_idx) tuples
        computed only from outer_train_idx data
    """
    hour_ts = pd.to_datetime(hour_ts, utc=True)

    # Create outer CV splits (for final evaluation)
    outer_splits = create_rolling_origin_cv(
        hour_ts,
        n_folds=n_outer_folds,
        val_window_hours=val_window_hours,
        gap_hours=gap_hours,
    )

    nested_splits = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        logger.info(f"\n  Outer Fold {outer_fold_idx + 1}:")
        logger.info(f"    Outer train: {len(outer_train_idx):,} rows")
        logger.info(f"    Outer test: {len(outer_test_idx):,} rows")

        # Extract hour_ts for outer training data only
        # Create a new Series with 0-based index for inner CV
        hour_ts_train_values = hour_ts.iloc[outer_train_idx].values
        hour_ts_train_series = pd.Series(
            hour_ts_train_values, index=range(len(hour_ts_train_values))
        )

        # Create inner CV splits from outer training data only
        # This returns indices relative to hour_ts_train_series (0-based: 0, 1, 2, ...)
        inner_splits = create_rolling_origin_cv(
            hour_ts_train_series,
            n_folds=n_inner_folds,
            val_window_hours=val_window_hours,
            gap_hours=gap_hours,
        )

        # Inner splits are already 0-based relative to outer_train_idx
        # They can be used directly with X_train_outer.iloc[inner_train_idx]
        # No mapping needed - they're already correct!
        inner_splits_mapped = inner_splits

        logger.info(f"    Inner CV: {len(inner_splits_mapped)} folds")

        nested_splits.append((outer_train_idx, outer_test_idx, inner_splits_mapped))

    logger.info(f"\nCreated {len(nested_splits)} nested CV folds")
    return nested_splits


def create_sliding_backtest_cv(
    timestamps: pd.Series,
    n_folds: int = 3,
    val_window_hours: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create 3-hour sliding backtest CV (demo metric only)

    Args:
        timestamps: Series of timestamps for all data points
        n_folds: Number of CV folds (default 3)
        val_window_hours: Validation window size in hours (default 3)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    timestamps = pd.to_datetime(timestamps)
    sorted_indices = timestamps.sort_values().index.values
    sorted_timestamps = timestamps.loc[sorted_indices]

    end_time = sorted_timestamps.max()
    splits = []

    for fold in range(n_folds):
        # Test window: [T - (fold+1)*val_window, T - fold*val_window)
        # Slide backward by 1 hour per fold
        test_end = end_time - pd.Timedelta(hours=fold * 1)
        test_start = test_end - pd.Timedelta(hours=val_window_hours)

        # Train: all data before test_start
        train_mask = sorted_timestamps < test_start
        test_mask = (sorted_timestamps >= test_start) & (sorted_timestamps < test_end)

        train_indices = sorted_indices[train_mask]
        test_indices = sorted_indices[test_mask]

        if len(train_indices) == 0:
            logger.warning(f"Backtest fold {fold+1}: No training data")
            continue
        if len(test_indices) == 0:
            logger.warning(f"Backtest fold {fold+1}: No test data")
            continue

        splits.append((train_indices, test_indices))
        logger.info(
            f"Backtest fold {fold+1}: Train={len(train_indices)}, Test={len(test_indices)}, "
            f"Test window=[{test_start}, {test_end})"
        )

    return splits
