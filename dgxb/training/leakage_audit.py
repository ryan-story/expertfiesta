"""
Data Leakage Audit Tools
Validates that features do not contain future information
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def audit_feature_timestamps(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    sample_size: int = 100,
) -> Dict[str, any]:
    """
    Audit feature timestamps vs label timestamps to detect leakage

    Args:
        X: Feature DataFrame
        y: Target Series
        timestamps: Timestamps for each sample
        sample_size: Number of samples to check

    Returns:
        Dictionary with audit results
    """
    timestamps = pd.to_datetime(timestamps)

    issues = []
    warnings = []

    # Check if any feature columns contain timestamp-like data
    timestamp_cols = []
    for col in X.columns:
        if "timestamp" in col.lower() or "time" in col.lower():
            timestamp_cols.append(col)

    if timestamp_cols:
        warnings.append(f"Found timestamp-like columns: {timestamp_cols}")

    # Check for lag features and verify direction
    lag_features = [col for col in X.columns if "_lag_" in col]
    if lag_features:
        logger.info(f"Found {len(lag_features)} lag features")
        # Verify lag features are backward-looking (negative shift)
        for lag_col in lag_features[:5]:  # Check first 5
            if "_lag_" in lag_col:
                parts = lag_col.split("_lag_")
                if len(parts) == 2:
                    try:
                        lag_hours = int(parts[1].replace("h", ""))
                        if lag_hours < 0:
                            issues.append(
                                f"Lag feature {lag_col} has negative lag hours (forward-looking!)"
                            )
                    except ValueError:
                        pass

    # Check for rolling features
    rolling_features = [
        col
        for col in X.columns
        if "_rolling_" in col or "_mean_" in col or "_std_" in col
    ]
    if rolling_features:
        logger.info(f"Found {len(rolling_features)} rolling/aggregate features")
        warnings.append(
            f"Found {len(rolling_features)} rolling features - verify they exclude future data"
        )

    # Check data distribution
    # Convert to Series if numpy array
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y)
    else:
        y_series = y

    if (
        y_series.dtype in [int, float]
        or hasattr(y_series, "dtype")
        and y_series.dtype in ["int64", "float64", "int32", "float32"]
    ):
        unique_values = y_series.nunique()
        value_counts = y_series.value_counts()
        most_common_pct = value_counts.iloc[0] / len(y_series) * 100

        if most_common_pct > 80:
            warnings.append(
                f"Target variable is highly imbalanced: {most_common_pct:.1f}% of samples are class '{value_counts.index[0]}'"
            )

        logger.info(
            f"Target variable: {unique_values} unique values, most common: {most_common_pct:.1f}%"
        )

    return {
        "issues": issues,
        "warnings": warnings,
        "lag_features_count": len(lag_features),
        "rolling_features_count": len(rolling_features),
        "timestamp_cols": timestamp_cols,
    }


def verify_lag_direction(
    df: pd.DataFrame,
    timestamp_col: str,
    lag_col: str,
    sample_indices: List[int],
) -> bool:
    """
    Verify that a lag feature is correctly backward-looking

    Args:
        df: DataFrame with timestamp and lag feature
        timestamp_col: Name of timestamp column
        lag_col: Name of lag feature column
        sample_indices: Indices to check

    Returns:
        True if lag is correctly backward-looking, False otherwise
    """
    # Extract lag hours from column name
    if "_lag_" not in lag_col:
        return True

    parts = lag_col.split("_lag_")
    if len(parts) != 2:
        return True

    try:
        lag_hours = int(parts[1].replace("h", ""))
        # Negative lag hours would indicate forward-looking (leakage)
        if lag_hours < 0:
            return False
    except ValueError:
        return True

    # Simplified check: verify lag hours is positive (backward-looking)
    # In practice, you'd verify the actual values match expected times
    return True


def audit_cv_splits(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    timestamps: pd.Series,
) -> Dict[str, any]:
    """
    Audit CV splits to ensure they're time-aware

    Args:
        cv_splits: List of (train_indices, test_indices) tuples
        timestamps: Timestamps for all samples

    Returns:
        Dictionary with audit results
    """
    timestamps = pd.to_datetime(timestamps)
    issues = []
    warnings = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        # Use .loc if indices are labels, .iloc if positional
        # After index reset in pipeline, indices are positional
        train_times = timestamps.iloc[train_idx]
        test_times = timestamps.iloc[test_idx]

        train_max = train_times.max()
        test_min = test_times.min()
        test_max = test_times.max()

        # Check: train max should be <= test min (no future data in train)
        if train_max > test_min:
            issues.append(
                f"Fold {fold_idx+1}: Train max time ({train_max}) > Test min time ({test_min}) - LEAKAGE!"
            )

        # Check: test window should be contiguous
        test_time_span = (test_max - test_min).total_seconds() / 3600
        expected_span = 24  # Assuming 24h windows
        if abs(test_time_span - expected_span) > 2:
            warnings.append(
                f"Fold {fold_idx+1}: Test window span is {test_time_span:.1f}h (expected ~{expected_span}h)"
            )

        # Check for overlap between folds (test windows should be sequential, not overlapping)
        # In rolling-origin CV, later folds should have earlier test windows
        # So test_max of fold N should be <= test_min of fold N-1
        if fold_idx > 0:
            # Use .iloc since indices are positional after pipeline reset
            prev_test_times = timestamps.iloc[cv_splits[fold_idx - 1][1]]
            prev_test_min = prev_test_times.min()

            # In rolling-origin, fold N test should be before fold N-1 test
            # So test_max should be <= prev_test_min
            if test_max > prev_test_min:
                issues.append(
                    f"Fold {fold_idx+1}: Test window overlaps with previous fold - LEAKAGE!"
                )

    return {
        "issues": issues,
        "warnings": warnings,
        "n_folds": len(cv_splits),
    }
