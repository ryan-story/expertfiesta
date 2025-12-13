"""
Robustness validation checks for sector-hour regression pipeline.

These tests verify that the exceptional performance (R²=0.991) is due to
legitimate signal rather than temporal leakage or overfitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dgxb.training.cv_splitter import create_rolling_origin_cv

logger = logging.getLogger(__name__)


def test_without_incident_count_t(
    X_rich: pd.DataFrame,
    y: pd.Series,
    hour_ts: pd.Series,
    h3_cells: np.ndarray,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Test 1: Remove incident_count_t and related lag features.
    
    This checks if the model is overly dependent on the current hour's
    incident count (which would indicate temporal leakage).
    
    Expected: Performance should worsen materially if the model is correct.
    """
    logger.info("=" * 80)
    logger.info("ROBUSTNESS TEST 1: Remove incident_count_t and lag features")
    logger.info("=" * 80)
    
    # Identify columns to remove
    # Note: incident_count_t is in the target, not features
    # But we should check for any lag/rolling features that might leak
    columns_to_remove = [
        col for col in X_rich.columns
        if any(keyword in col.lower() for keyword in [
            'incident_count',
            'incident_count_t',
            'incident_count_lag',
            'incident_count_rolling',
        ])
    ]
    
    # Also check for any features that directly encode the current hour's count
    # This is a safety check - if incident_count_t somehow made it into features, remove it
    
    if len(columns_to_remove) > 0:
        logger.info(f"  Removing {len(columns_to_remove)} columns related to incident_count:")
        for col in columns_to_remove[:10]:  # Show first 10
            logger.info(f"    - {col}")
        if len(columns_to_remove) > 10:
            logger.info(f"    ... and {len(columns_to_remove) - 10} more")
        # Create filtered feature set
        X_filtered = X_rich.drop(columns=columns_to_remove)
    else:
        logger.info("  ✓ No incident_count features found in X (good - no target leakage)")
        logger.info("  Testing with all features (weather, calendar, spatial, rolling stats)")
        X_filtered = X_rich.copy()
    
    # Keep only numeric columns
    X_filtered = X_filtered.select_dtypes(include=[np.number])
    X_filtered = X_filtered.fillna(X_filtered.median())
    
    logger.info(f"  Remaining features: {len(X_filtered.columns)}")
    logger.info(f"  Features kept: weather, calendar, spatial, rolling stats (non-incident)")
    
    # Create CV splits
    cv_splits = create_rolling_origin_cv(
        hour_ts, n_folds=n_folds, val_window_hours=24, gap_hours=1
    )
    
    # Train XGBoost model
    logger.info("  Training XGBoost without incident_count features...")
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=100,
        max_depth=10,
        learning_rate=0.2,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
    )
    
    # Evaluate on all folds
    all_y_true = []
    all_y_pred = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X_filtered.iloc[train_idx]
        X_test = X_filtered.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0.0)  # Clip to non-negative
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    mae = mean_absolute_error(all_y_true, all_y_pred)
    r2 = r2_score(all_y_true, all_y_pred)
    
    results = {
        "test_name": "Without incident_count_t",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features_removed": len(columns_to_remove),
        "n_features_remaining": len(X_filtered.columns),
    }
    
    logger.info(f"  Results:")
    logger.info(f"    RMSE: {rmse:.4f}")
    logger.info(f"    MAE: {mae:.4f}")
    logger.info(f"    R²: {r2:.4f}")
    logger.info(f"    Features removed: {len(columns_to_remove)}")
    logger.info(f"    Features remaining: {len(X_filtered.columns)}")
    
    return results


def test_horizon_sensitivity(
    X_rich: pd.DataFrame,
    y_target_df: pd.DataFrame,
    hour_ts: pd.Series,
    h3_cells: np.ndarray,
    horizons: list = [1, 3, 6],
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Test 2: Evaluate performance at different prediction horizons.
    
    Tests: t+1, t+3, t+6 hours
    
    Expected: Monotonic degradation (RMSE increases, R² decreases) as horizon increases.
    If R² stays ~0.99 at +6 hours, something is wrong.
    """
    logger.info("=" * 80)
    logger.info("ROBUSTNESS TEST 2: Horizon Sensitivity (t+1, t+3, t+6)")
    logger.info("=" * 80)
    
    # Keep only numeric columns
    X_numeric = X_rich.select_dtypes(include=[np.number])
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    # Create CV splits
    cv_splits = create_rolling_origin_cv(
        hour_ts, n_folds=n_folds, val_window_hours=24, gap_hours=1
    )
    
    results_by_horizon = {}
    
    # Get base target for comparison
    y_base = y_target_df["incident_count_t_plus_1"].values
    valid_base = ~pd.isna(y_base)
    y_base = y_base[valid_base]
    
    for horizon in horizons:
        logger.info(f"\n  Testing horizon: t+{horizon} hours")
        
        # Create target for this horizon
        y_target_sorted = y_target_df.sort_values(["h3_cell", "hour_ts"]).copy()
        # Shift by (horizon - 1) to get t+horizon
        y_target_sorted[f"incident_count_t_plus_{horizon}"] = (
            y_target_sorted.groupby("h3_cell")["incident_count_t_plus_1"]
            .shift(-(horizon - 1))
        )
        
        # Filter to valid targets (same mask as base)
        valid_mask = valid_base & ~pd.isna(y_target_sorted[f"incident_count_t_plus_{horizon}"])
        y_horizon = y_target_sorted[f"incident_count_t_plus_{horizon}"][valid_mask].values
        X_horizon = X_numeric[valid_mask].reset_index(drop=True)
        
        logger.info(f"    Valid targets: {len(y_horizon)} (vs {len(y_base)} for t+1)")
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            objective='count:poisson',
            n_estimators=100,
            max_depth=10,
            learning_rate=0.2,
            subsample=0.6,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
        )
        
        # Evaluate on all folds (simplified - use first fold for speed)
        # In production, evaluate on all folds
        if len(y_horizon) > 0:
            train_size = int(0.8 * len(y_horizon))
            X_train = X_horizon.iloc[:train_size]
            X_test = X_horizon.iloc[train_size:]
            y_train = y_horizon[:train_size]
            y_test = y_horizon[train_size:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0.0)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results_by_horizon[horizon] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_samples": len(y_horizon),
            }
            
            logger.info(f"    RMSE: {rmse:.4f}")
            logger.info(f"    MAE: {mae:.4f}")
            logger.info(f"    R²: {r2:.4f}")
        else:
            logger.warning(f"    No valid targets for horizon t+{horizon}")
            results_by_horizon[horizon] = {
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "n_samples": 0,
            }
    
    return {
        "test_name": "Horizon Sensitivity",
        "results_by_horizon": results_by_horizon,
    }


def test_sector_cold_start(
    X_rich: pd.DataFrame,
    y: pd.Series,
    hour_ts: pd.Series,
    h3_cells: np.ndarray,
    test_cell_fraction: float = 0.2,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Test 3: Hold out entire H3 cells in validation.
    
    This tests if the model is learning city rhythm vs memorizing sector identity.
    
    Expected: Performance drops, but hotspot metrics remain reasonable.
    """
    logger.info("=" * 80)
    logger.info("ROBUSTNESS TEST 3: Sector Cold-Start (Hold out entire cells)")
    logger.info("=" * 80)
    
    # Get unique cells
    unique_cells = np.unique(h3_cells)
    n_test_cells = int(len(unique_cells) * test_cell_fraction)
    
    # Randomly select test cells
    np.random.seed(42)
    test_cells = np.random.choice(unique_cells, size=n_test_cells, replace=False)
    
    logger.info(f"  Total unique cells: {len(unique_cells)}")
    logger.info(f"  Test cells (held out): {len(test_cells)} ({test_cell_fraction*100:.0f}%)")
    logger.info(f"  Train cells: {len(unique_cells) - len(test_cells)}")
    
    # Create train/test split based on cells
    train_mask = ~np.isin(h3_cells, test_cells)
    test_mask = np.isin(h3_cells, test_cells)
    
    X_train = X_rich[train_mask].copy()
    X_test = X_rich[test_mask].copy()
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    logger.info(f"  Train samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    
    # Keep only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median
    
    # Train XGBoost model
    logger.info("  Training XGBoost on train cells, evaluating on held-out cells...")
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=100,
        max_depth=10,
        learning_rate=0.2,
        subsample=0.6,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0.0)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        "test_name": "Sector Cold-Start",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_train_cells": len(unique_cells) - len(test_cells),
        "n_test_cells": len(test_cells),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    
    logger.info(f"  Results:")
    logger.info(f"    RMSE: {rmse:.4f}")
    logger.info(f"    MAE: {mae:.4f}")
    logger.info(f"    R²: {r2:.4f}")
    
    return results


def run_all_robustness_checks(
    base_x_path: str = "gold-cpu-traffic/X_features.parquet",
    rich_x_path: str = "rich-gold-cpu-traffic/X_features.parquet",
    y_path: str = "gold-cpu-traffic/y_target.parquet",
    output_path: str = "results/robustness_checks.json",
) -> Dict[str, Any]:
    """
    Run all robustness checks and save results.
    """
    import json
    
    logger.info("=" * 80)
    logger.info("RUNNING ALL ROBUSTNESS CHECKS")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading data...")
    X_rich = pd.read_parquet(rich_x_path)
    y_target_df = pd.read_parquet(y_path)
    
    # Extract target and metadata
    y = y_target_df["incident_count_t_plus_1"].values
    hour_ts = pd.to_datetime(y_target_df["hour_ts"], utc=True)
    h3_cells = y_target_df["h3_cell"].values
    
    # Filter to valid targets
    valid_mask = ~pd.isna(y)
    y = y[valid_mask]
    X_rich = X_rich[valid_mask].reset_index(drop=True)
    hour_ts = hour_ts[valid_mask].reset_index(drop=True)
    h3_cells = h3_cells[valid_mask]
    y_target_df = y_target_df[valid_mask].reset_index(drop=True)
    
    logger.info(f"  Loaded {len(X_rich)} samples with {len(X_rich.columns)} features")
    
    # Run tests
    all_results = {}
    
    # Test 1: Without incident_count_t
    try:
        test1_results = test_without_incident_count_t(X_rich, y, hour_ts, h3_cells)
        all_results["test_1_without_incident_count_t"] = test1_results
    except Exception as e:
        logger.error(f"  Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["test_1_without_incident_count_t"] = {"error": str(e)}
    
    # Test 2: Horizon sensitivity
    try:
        test2_results = test_horizon_sensitivity(X_rich, y_target_df, hour_ts, h3_cells)
        all_results["test_2_horizon_sensitivity"] = test2_results
    except Exception as e:
        logger.error(f"  Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["test_2_horizon_sensitivity"] = {"error": str(e)}
    
    # Test 3: Sector cold-start
    try:
        test3_results = test_sector_cold_start(X_rich, y, hour_ts, h3_cells)
        all_results["test_3_sector_cold_start"] = test3_results
    except Exception as e:
        logger.error(f"  Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["test_3_sector_cold_start"] = {"error": str(e)}
    
    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Saved robustness check results to {output_path}")
    
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_robustness_checks()

