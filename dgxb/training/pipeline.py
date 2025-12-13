"""
Main training pipeline orchestrator
Runs model competition across two feature channels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import Optional, List, Any
import h3
from .model_competition import (
    train_linear_regression,
    train_poisson_regression,
    train_random_forest,
    train_xgboost,
)
from .metrics_tracker import (
    compute_hotspot_metrics,
    measure_inference_latency,
    compute_regression_metrics,
    track_pipeline_metrics,
)

logger = logging.getLogger(__name__)


def extract_top_features(model: Any, model_name: str, feature_names: List[str]) -> str:
    """
    Extract top 5 most important features from a trained model

    Args:
        model: Trained model (LinearRegression, PoissonRegressor, RandomForestRegressor, or XGBRegressor)
        model_name: Name of the model type
        feature_names: List of feature names

    Returns:
        Comma-separated string of top 5 feature names
    """
    try:
        # Handle Pipeline objects (from LinearRegression/PoissonRegressor)
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            actual_model = model.named_steps['regressor']
        else:
            actual_model = model
            
        if model_name in ["LinearRegression", "PoissonRegressor"]:
            # For LinearRegression/PoissonRegressor, use absolute coefficients
            if hasattr(actual_model, 'coef_'):
                if actual_model.coef_.ndim > 1:
                    # Multi-output: average absolute coefficients across outputs
                    importances = np.abs(actual_model.coef_).mean(axis=0)
                else:
                    # Single output: use absolute coefficients
                    importances = np.abs(actual_model.coef_)
            else:
                return ""
        elif model_name in ["RandomForest", "XGBoost"]:
            # Use feature_importances_ directly
            importances = actual_model.feature_importances_
        else:
            # Fallback: return empty
            return ""

        # Get top 5 features
        top_indices = np.argsort(importances)[::-1][:5]
        top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]

        # Format as comma-separated string
        return ",".join(top_features)
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return ""


def get_h3_cell(lat: float, lon: float, resolution: int) -> Optional[str]:
    """Get H3 cell for coordinates, handling both h3-py v3 and v4+ APIs"""
    try:
        lat = float(lat)
        lon = float(lon)

        if pd.isna(lat) or pd.isna(lon) or np.isnan(lat) or np.isnan(lon):
            return None
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None

        # Try new API first (h3-py v4+)
        try:
            return str(h3.latlng_to_cell(lat, lon, resolution))
        except AttributeError:
            # Fallback to old API (h3-py v3)
            return str(h3.geo_to_h3(lat, lon, resolution))
    except Exception:
        return None


def run_training_competition(
    base_x_path: str = "gold-cpu-traffic/X_features.parquet",
    rich_x_path: str = "rich-gold-cpu-traffic/X_features.parquet",
    y_path: str = "gold-cpu-traffic/y_target.parquet",
    merged_intermediate_path: str = "gold-cpu-traffic/merged_intermediate.parquet",
    results_dir: str = "results",
    n_folds: int = 3,
    val_window_hours: int = 24,
    k_hotspots: int = 50,
    h3_resolution: int = 9,
    n_trials_rf: int = 30,
    n_trials_xgb: int = 30,
) -> pd.DataFrame:
    """
    Run model competition across base and rich feature channels

    Args:
        base_x_path: Path to base X features
        rich_x_path: Path to rich X features
        y_path: Path to y target
        merged_intermediate_path: Path to merged intermediate (for timestamps/H3)
        results_dir: Output directory for results
        n_folds: Number of CV folds
        val_window_hours: Validation window size in hours
        k_hotspots: Number of top hotspots for Precision@K/Recall@K
        h3_resolution: H3 resolution for spatial operations
        n_trials_rf: Number of trials for RandomForest HPO
        n_trials_xgb: Number of trials for XGBoost HPO

    Returns:
        DataFrame with all model results and champion selection
    """
    logger.info("=" * 70)
    logger.info("CPU TRAINING COMPETITION PIPELINE")
    logger.info("=" * 70)

    pipeline_start_time = time.time()

    # Step 1: Load data
    logger.info("\n[Step 1/6] Loading data...")
    ingest_start = time.time()

    base_X = pd.read_parquet(base_x_path)
    rich_X = pd.read_parquet(rich_x_path)
    y_target = pd.read_parquet(y_path)
    merged_df = pd.read_parquet(merged_intermediate_path)

    # Reset indices to ensure positional indexing works correctly
    base_X = base_X.reset_index(drop=True)
    rich_X = rich_X.reset_index(drop=True)
    y_target = y_target.reset_index(drop=True)
    merged_df = merged_df.reset_index(drop=True)

    ingest_cleaning_time = time.time() - ingest_start
    logger.info(
        f"  Loaded base X: {len(base_X):,} records, {len(base_X.columns)} features"
    )
    logger.info(
        f"  Loaded rich X: {len(rich_X):,} records, {len(rich_X.columns)} features"
    )
    logger.info(f"  Loaded y: {len(y_target):,} records")
    logger.info(f"  Ingest time: {ingest_cleaning_time:.2f}s")

    # Extract hour_ts and h3_cell from aggregated data FIRST (before filtering)
    logger.info("\n[Step 2/6] Extracting hour_ts and h3_cell from aggregated data...")
    
    if "hour_ts" in base_X.columns:
        hour_ts = pd.to_datetime(base_X["hour_ts"], utc=True)
    elif "hour_ts" in y_target.columns:
        hour_ts = pd.to_datetime(y_target["hour_ts"], utc=True)
    else:
        raise ValueError("hour_ts must be present in X or y_target for time-blocked CV")
    
    if "h3_cell" in base_X.columns:
        h3_cells = base_X["h3_cell"].values
    elif "h3_cell" in y_target.columns:
        h3_cells = y_target["h3_cell"].values
    else:
        raise ValueError("h3_cell must be present in X or y_target")
    
    logger.info(f"  hour_ts range: {hour_ts.min()} to {hour_ts.max()}")
    logger.info(f"  Unique h3_cells: {len(np.unique(h3_cells))}")
    
    # Extract target (regression: incident_count_t_plus_1)
    if "incident_count_t_plus_1" not in y_target.columns:
        raise ValueError("y_target must contain 'incident_count_t_plus_1' for regression")
    
    y = y_target["incident_count_t_plus_1"].values
    
    # Drop rows with NaN target (last hour per cell - excluded from training)
    valid_mask = ~pd.isna(y)
    y = y[valid_mask]
    base_X = base_X[valid_mask].reset_index(drop=True)
    rich_X = rich_X[valid_mask].reset_index(drop=True)
    y_target = y_target[valid_mask].reset_index(drop=True)
    hour_ts = hour_ts[valid_mask].reset_index(drop=True)
    h3_cells = h3_cells[valid_mask]
    
    logger.info(f"  After dropping NaN targets: {len(y):,} records")
    
    # For hotspot metrics, create actual_counts_df from y_target (after filtering)
    # CRITICAL: incident_count_t_plus_1 represents incidents at hour_ts + 1, so hour_actual must be hour_ts + 1
    actual_counts_df = y_target[["h3_cell", "hour_ts", "incident_count_t_plus_1"]].copy()
    actual_counts_df["hour_actual"] = pd.to_datetime(actual_counts_df["hour_ts"], utc=True) + pd.Timedelta(hours=1)
    actual_counts_df = actual_counts_df[["h3_cell", "hour_actual", "incident_count_t_plus_1"]].copy()
    actual_counts_df.columns = ["h3_cell", "hour_actual", "incident_count"]
    
    # Reconstruct timestamps for compatibility (use hour_ts)
    timestamps = hour_ts.copy()
    
    logger.info(f"  Reconstructed {len(timestamps):,} timestamps")
    logger.info(f"  Reconstructed {len(h3_cells):,} H3 cells")

    # Step 3: Create nested CV splits (time-blocked on hour_ts)
    logger.info("\n[Step 3/6] Creating nested time-blocked cross-validation splits...")
    from dgxb.training.cv_splitter import create_nested_cv
    
    nested_cv_splits = create_nested_cv(
        hour_ts, 
        n_outer_folds=n_folds, 
        n_inner_folds=3,  # Inner folds for hyperparameter tuning
        val_window_hours=val_window_hours, 
        gap_hours=1
    )
    logger.info(f"  Created {len(nested_cv_splits)} nested CV folds (outer for evaluation, inner for tuning)")

    # Step 4: Train models for each channel
    logger.info("\n[Step 4/6] Training models...")

    all_results = []

    channels = [
        ("base", base_X),
        ("rich", rich_X),
    ]

    models = [
        ("LinearRegression", train_linear_regression),
        ("PoissonRegressor", train_poisson_regression),
        ("RandomForest", train_random_forest),
        ("XGBoost", train_xgboost),
    ]
    
    # Baseline models (simple comparators)
    def train_persistence_baseline(X_train, y_train, cv_splits):
        """Persistence baseline: y_hat = incident_count_t"""
        from sklearn.base import BaseEstimator
        class PersistenceBaseline(BaseEstimator):
            def fit(self, X, y):
                # Use incident_count_t as prediction
                if "incident_count_t" in X.columns:
                    self.pred_value = X["incident_count_t"].median()
                else:
                    self.pred_value = 0.0
                return self
            def predict(self, X):
                if "incident_count_t" in X.columns:
                    return X["incident_count_t"].values
                else:
                    return np.full(len(X), self.pred_value)
        model = PersistenceBaseline()
        model.fit(X_train, y_train)
        return model, {}, {"rmse": 0.0}, 0.0
    
    def train_climatology_baseline(X_train, y_train, cv_splits):
        """Climatology baseline: y_hat = mean(incident_count | cell, hour_of_week)"""
        from sklearn.base import BaseEstimator
        class ClimatologyBaseline(BaseEstimator):
            def fit(self, X, y):
                # Compute mean per (h3_cell, hour_of_week)
                X_fit = X.copy()
                if "h3_cell" in X_fit.columns and "hour" in X_fit.columns and "day_of_week" in X_fit.columns:
                    X_fit["hour_of_week"] = X_fit["day_of_week"] * 24 + X_fit["hour"]
                    y_series = pd.Series(y, index=X_fit.index)
                    self.climatology = X_fit.groupby(["h3_cell", "hour_of_week"]).apply(
                        lambda group: y_series.loc[group.index].mean()
                    ).to_dict()
                else:
                    self.climatology = {}
                self.fallback = np.mean(y) if len(y) > 0 else 0.0
                return self
            def predict(self, X):
                X_pred = X.copy()
                if "h3_cell" in X_pred.columns and "hour" in X_pred.columns and "day_of_week" in X_pred.columns:
                    X_pred["hour_of_week"] = X_pred["day_of_week"] * 24 + X_pred["hour"]
                    preds = []
                    for idx, row in X_pred.iterrows():
                        key = (row["h3_cell"], row["hour_of_week"])
                        preds.append(self.climatology.get(key, self.fallback))
                    return np.array(preds)
                else:
                    return np.full(len(X), self.fallback)
        model = ClimatologyBaseline()
        model.fit(X_train, y_train)
        return model, {}, {"rmse": 0.0}, 0.0
    
    baseline_models = [
        ("Persistence", train_persistence_baseline),
        ("Climatology", train_climatology_baseline),
    ]

    for channel_name, X in channels:
        logger.info(f"\n  Channel: {channel_name} ({len(X.columns)} features)")

        # Audit features for leakage (sample check) - skip for regression
        # Leakage audit was designed for classification, skip for now
        if channel_name == "rich":
            logger.info("  Rich features loaded (leakage audit skipped for regression)")

        # Filter to numeric columns only (drop string/object columns)
        # Convert boolean columns to int
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if X_numeric[col].dtype == bool:
                X_numeric[col] = X_numeric[col].astype(int)

        # Select only numeric columns
        numeric_cols = X_numeric.select_dtypes(include=["number"]).columns
        X_numeric = X_numeric[numeric_cols]

        logger.info(f"  Filtered to {len(X_numeric.columns)} numeric features")

        # Drop any remaining non-numeric columns (safety check)
        non_numeric = X_numeric.select_dtypes(exclude=["number"]).columns
        if len(non_numeric) > 0:
            logger.warning(f"  Dropping non-numeric columns: {non_numeric.tolist()}")
            X_numeric = X_numeric.drop(columns=non_numeric)

        # Handle NaN values: fill with median for numeric columns
        nan_counts = X_numeric.isna().sum()
        if nan_counts.sum() > 0:
            logger.info(
                f"  Filling NaN values (columns with NaN: {nan_counts[nan_counts > 0].to_dict()})"
            )
            X_numeric = X_numeric.fillna(X_numeric.median())

        # Feature build time (0 for base, actual for rich)
        # Rich features already built, just measure load time
        feature_build_time = 0.0  # Already loaded

        # Train baseline models first (evaluate them like other models)
        for model_name, train_func in baseline_models:
            logger.info(f"    Baseline: {model_name}")
            try:
                # For baselines, we need to evaluate on test folds
                # Create a simple wrapper that evaluates on CV splits
                all_y_true_baseline = []
                all_y_pred_baseline = []
                all_h3_test_baseline = []
                all_hour_ts_test_baseline = []
                all_hour_ts_pred_baseline = []
                
                # Use nested CV: inner CV for hyperparameter tuning, outer test for evaluation
                for outer_fold_idx, (outer_train_idx, outer_test_idx, inner_cv_splits) in enumerate(nested_cv_splits):
                    X_train_fold = X_numeric.iloc[outer_train_idx].reset_index(drop=True)
                    X_test_fold = X_numeric.iloc[outer_test_idx]
                    y_train_fold = y[outer_train_idx]
                    y_test_fold = y[outer_test_idx]
                    
                    # Train baseline using inner CV for hyperparameter tuning
                    model, best_params, cv_scores, train_time = train_func(
                        X_train_fold, y_train_fold, inner_cv_splits
                    )
                    
                    # Predict on outer test fold (never seen during hyperparameter tuning)
                    y_pred_fold = model.predict(X_test_fold)
                    y_pred_fold = np.maximum(y_pred_fold, 0.0)  # Clip to non-negative
                    
                    all_y_true_baseline.extend(y_test_fold)
                    all_y_pred_baseline.extend(y_pred_fold)
                    all_h3_test_baseline.extend(h3_cells[outer_test_idx])
                    all_hour_ts_test_baseline.extend(hour_ts.iloc[outer_test_idx])
                    all_hour_ts_pred_baseline.extend(hour_ts.iloc[outer_test_idx])
                
                # Compute metrics for baseline
                all_y_true_baseline_arr = np.array(all_y_true_baseline)
                all_y_pred_baseline_arr = np.array(all_y_pred_baseline)
                
                regression_metrics_baseline = compute_regression_metrics(
                    all_y_true_baseline_arr, all_y_pred_baseline_arr
                )
                
                # Build actual counts for hotspot metrics
                hour_ts_pred_baseline_series = pd.Series(all_hour_ts_pred_baseline)
                if not isinstance(hour_ts_pred_baseline_series, pd.DatetimeIndex):
                    hour_ts_pred_baseline_series = pd.to_datetime(hour_ts_pred_baseline_series, utc=True)
                
                # Predictions are made at hour t, so actuals should be at hour t+1
                hour_ts_actual_baseline_series = hour_ts_pred_baseline_series + pd.Timedelta(hours=1)
                
                # Filter actual_counts_df to actual hours (t+1)
                actual_hours_baseline = hour_ts_actual_baseline_series.dt.floor("h").unique()
                actual_counts_baseline = actual_counts_df[
                    actual_counts_df["hour_actual"].isin(actual_hours_baseline)
                ].copy()
                
                hotspot_metrics_baseline = compute_hotspot_metrics(
                    all_y_pred_baseline_arr,
                    actual_counts_baseline,
                    np.array(all_h3_test_baseline),
                    hour_ts_pred_baseline_series,
                    hour_ts_actual_baseline_series,
                    k=k_hotspots,
                    h3_resolution=h3_resolution,
                )
                
                # Create result entry for baseline
                baseline_result = {
                    "model_name": model_name,
                    "channel": channel_name,
                    "rmse": regression_metrics_baseline["rmse"],
                    "mae": regression_metrics_baseline["mae"],
                    "r2": regression_metrics_baseline["r2"],
                    "smape": regression_metrics_baseline["smape"],
                    "mape_pos": regression_metrics_baseline["mape_pos"],
                    "hotspot_precision_at_k": hotspot_metrics_baseline["precision_at_k"],
                    "hotspot_recall_at_k": hotspot_metrics_baseline["recall_at_k"],
                    "hotspot_precision_at_k_conditional": hotspot_metrics_baseline.get(
                        "hotspot_precision_at_k_conditional", 0.0
                    ),
                    "hotspot_recall_at_k_conditional": hotspot_metrics_baseline.get(
                        "hotspot_recall_at_k_conditional", 0.0
                    ),
                    "staging_utility_coverage_pct": hotspot_metrics_baseline.get(
                        "staging_utility_coverage_pct", 0.0
                    ),
                    "staging_utility_with_neighbors_pct": hotspot_metrics_baseline.get(
                        "staging_utility_with_neighbors_pct", 0.0
                    ),
                    "train_time_sec": train_time,
                    "inference_latency_p50_ms": 0.0,
                    "inference_latency_p95_ms": 0.0,
                    "rows_processed_per_sec": 0.0,
                    "top_5_features": f"{model_name}_baseline",
                    "champion": False,
                }
                all_results.append(baseline_result)
                
                logger.info(f"      RMSE: {regression_metrics_baseline['rmse']:.4f}")
                logger.info(f"      MAE: {regression_metrics_baseline['mae']:.4f}")
                logger.info(f"      R²: {regression_metrics_baseline['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"  Failed to train/evaluate {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Train regression models
        for model_name, train_func in models:
            logger.info(f"    Model: {model_name}")

            try:
                # Train and evaluate using nested CV
                # Outer folds: final evaluation (never seen during hyperparameter tuning)
                # Inner folds: hyperparameter tuning (only uses outer training data)
                all_y_true = []
                all_y_pred = []
                all_h3_test = []
                all_hour_ts_test = []
                all_hour_ts_pred = []
                
                total_train_time = 0.0

                for outer_fold_idx, (outer_train_idx, outer_test_idx, inner_cv_splits) in enumerate(nested_cv_splits):
                    logger.info(f"      Outer fold {outer_fold_idx + 1}/{len(nested_cv_splits)}")
                    
                    # Train model using inner CV for hyperparameter tuning
                    # Only use outer training data
                    # Reset index so inner CV splits (0-based) work correctly
                    X_train_outer = X_numeric.iloc[outer_train_idx].reset_index(drop=True)
                    y_train_outer = y[outer_train_idx]
                    
                    if model_name in ["LinearRegression", "PoissonRegressor"]:
                        model, best_params, cv_scores, train_time = train_func(  # type: ignore
                            X_train_outer, y_train_outer, inner_cv_splits
                        )
                    elif model_name == "RandomForest":
                        model, best_params, cv_scores, train_time = train_func(  # type: ignore
                            X_train_outer, y_train_outer, inner_cv_splits, n_trials=n_trials_rf
                        )
                    elif model_name == "XGBoost":
                        model, best_params, cv_scores, train_time = train_func(  # type: ignore
                            X_train_outer, y_train_outer, inner_cv_splits, n_trials=n_trials_xgb
                        )
                    
                    total_train_time += train_time
                    
                    # Evaluate on outer test set (never seen during hyperparameter tuning)
                    X_test_outer = X_numeric.iloc[outer_test_idx]
                    y_test_outer = y[outer_test_idx]
                    hour_ts_test_outer = hour_ts.iloc[outer_test_idx]
                    
                    y_pred_outer = model.predict(X_test_outer)
                    y_pred_outer = np.maximum(y_pred_outer, 0.0)  # Clip to non-negative
                    
                    all_y_true.extend(y_test_outer)
                    all_y_pred.extend(y_pred_outer)
                    all_h3_test.extend(h3_cells[outer_test_idx])
                    all_hour_ts_test.extend(hour_ts_test_outer)
                    all_hour_ts_pred.extend(hour_ts_test_outer)
                
                # Use average training time across folds
                train_time = total_train_time / len(nested_cv_splits)

                all_y_true_arr = np.array(all_y_true)
                all_y_pred_arr = np.array(all_y_pred)

                # Compute regression metrics
                regression_metrics = compute_regression_metrics(
                    all_y_true_arr, all_y_pred_arr
                )

                # Build actual incident counts per (H3, hour) from test data
                # Use actual_counts_df (already created from y_target)
                hour_ts_test_series = pd.Series(all_hour_ts_test)
                if not isinstance(hour_ts_test_series, pd.DatetimeIndex):
                    hour_ts_test_series = pd.to_datetime(hour_ts_test_series, utc=True)
                
                # Filter actual_counts_df to test hours
                test_hours = hour_ts_test_series.dt.floor("h").unique()
                actual_counts = actual_counts_df[
                    actual_counts_df["hour_actual"].isin(test_hours)
                ].copy()

                # Fix timestamp alignment: predictions at t, actuals at t+1
                hour_ts_pred_series = pd.Series(all_hour_ts_pred)
                if not isinstance(hour_ts_pred_series, pd.DatetimeIndex):
                    hour_ts_pred_series = pd.to_datetime(hour_ts_pred_series, utc=True)
                hour_ts_actual_series = hour_ts_pred_series + pd.Timedelta(hours=1)

                # Compute hotspot metrics (regression: use y_pred directly)
                hotspot_metrics = compute_hotspot_metrics(
                    all_y_pred_arr,  # Predicted counts
                    actual_counts,
                    np.array(all_h3_test),
                    hour_ts_pred_series,  # When prediction was made (t)
                    hour_ts_actual_series,  # Actual incident time (t+1)
                    k=k_hotspots,
                    h3_resolution=h3_resolution,
                )

                # Measure inference latency
                # Use a sample of test data
                sample_size = min(10000, len(X_numeric))
                X_sample = X_numeric.iloc[:sample_size]
                inference_metrics = measure_inference_latency(model, X_sample)

                # Measure staging recommendation time
                staging_start = time.time()
                # Simulate: predict and get top-K
                _ = model.predict(X_sample)
                # Get top-K (simplified - in practice would group by H3/hour)
                staging_time = time.time() - staging_start

                # Track pipeline metrics
                pipeline_metrics = track_pipeline_metrics(
                    ingest_cleaning_time_sec=ingest_cleaning_time,
                    feature_build_time_sec=feature_build_time,
                    train_time_sec=train_time,
                    inference_latency_metrics=inference_metrics,
                    staging_recommendation_time_sec=staging_time,
                    rows_processed=len(X_numeric),
                    dataset_size=len(X_numeric),
                )

                # Extract feature importances
                feature_names = list(X_numeric.columns)
                top_5_features = extract_top_features(model, model_name, feature_names)

                # Combine all metrics with explicit naming
                result = {
                    "model_name": model_name,
                    "channel": channel_name,
                    "rmse": regression_metrics["rmse"],
                    "mae": regression_metrics["mae"],
                    "r2": regression_metrics["r2"],
                    "smape": regression_metrics["smape"],
                    "mape_pos": regression_metrics["mape_pos"],
                    "hotspot_precision_at_k": hotspot_metrics[
                        "precision_at_k"
                    ],  # Unconditional (all hours)
                    "hotspot_recall_at_k": hotspot_metrics[
                        "recall_at_k"
                    ],  # Unconditional (all hours)
                    "hotspot_precision_at_k_conditional": hotspot_metrics.get(
                        "hotspot_precision_at_k_conditional", 0.0
                    ),  # Conditional (hours with incidents)
                    "hotspot_recall_at_k_conditional": hotspot_metrics.get(
                        "hotspot_recall_at_k_conditional", 0.0
                    ),  # Conditional (hours with incidents)
                    **pipeline_metrics,
                    "best_params": str(best_params),
                    "top_5_features": top_5_features,
                    "champion": False,
                }

                all_results.append(result)

                logger.info(f"      RMSE: {regression_metrics['rmse']:.4f}")
                logger.info(f"      MAE: {regression_metrics['mae']:.4f}")
                logger.info(f"      R²: {regression_metrics['r2']:.4f}")
                logger.info(f"      SMAPE: {regression_metrics['smape']:.2f}%")
                logger.info(
                    f"      Hotspot Precision@K (unconditional): {hotspot_metrics['precision_at_k']:.4f}"
                )
                logger.info(
                    f"      Hotspot Precision@K (conditional): {hotspot_metrics.get('hotspot_precision_at_k_conditional', 0.0):.4f}"
                )
                logger.info(
                    f"      Hotspot Recall@K: {hotspot_metrics['recall_at_k']:.4f}"
                )
                logger.info(
                    f"      Staging Utility: {hotspot_metrics['staging_utility_coverage_pct']:.2f}%"
                )
                logger.info(f"      Top 5 features: {top_5_features}")

            except Exception as e:
                logger.error(f"      Error training {model_name}: {e}")
                continue

    # Step 5: Baseline models are already trained above
    # Historical baseline computation removed (replaced by persistence/climatology baselines)
    logger.info("\n[Step 5/7] Baseline models completed above")
    
    # Step 6: Champion selection (one per channel)
    logger.info("\n[Step 6/7] Selecting champion models...")

    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        logger.error("No models trained successfully!")
        return pd.DataFrame()

    # Select champion for each channel based on RMSE (minimize)
    results_df["champion"] = False

    for channel in ["base", "rich"]:
        channel_results = results_df[results_df["channel"] == channel]
        if len(channel_results) > 0:
            # Exclude baseline models from champion selection
            channel_results_models = channel_results[~channel_results["model_name"].isin(["Persistence", "Climatology"])]
            if len(channel_results_models) > 0:
                champion_idx = channel_results_models["rmse"].idxmin()
                results_df.loc[champion_idx, "champion"] = True

                champion = results_df.loc[champion_idx]
                logger.info(f"  Champion ({channel}): {champion['model_name']}")
                logger.info(f"    RMSE: {champion['rmse']:.4f}")
                logger.info(f"    MAE: {champion['mae']:.4f}")
                logger.info(f"    R²: {champion['r2']:.4f}")
                logger.info(
                    f"    Hotspot Precision@K (unconditional): {champion.get('hotspot_precision_at_k', 'N/A'):.4f}"
                )
                logger.info(
                    f"    Hotspot Precision@K (conditional): {champion.get('hotspot_precision_at_k_conditional', 'N/A'):.4f}"
                )
                logger.info(
                    f"    Hotspot Recall@K: {champion.get('hotspot_recall_at_k', 'N/A'):.4f}"
                )
                logger.info(
                    f"    Staging Utility: {champion.get('staging_utility_coverage_pct', 'N/A'):.2f}%"
                )
                logger.info(f"    Top 5 features: {champion.get('top_5_features', 'N/A')}")

    # Step 7: Save results
    logger.info("\n[Step 7/7] Saving results...")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    output_file = results_path / "cpu_training_results.csv"

    # Select columns for CSV (exclude best_params as it's complex)
    # Use explicit metric names for clarity
    csv_columns = [
        "model_name",
        "channel",
        "rmse",
        "mae",
        "r2",
        "smape",
        "mape_pos",
        "hotspot_precision_at_k",  # Unconditional (all hours)
        "hotspot_recall_at_k",  # Unconditional (all hours)
        "hotspot_precision_at_k_conditional",  # Conditional (hours with incidents)
        "hotspot_recall_at_k_conditional",  # Conditional (hours with incidents)
        "staging_utility_coverage_pct",
        "staging_utility_with_neighbors_pct",
        "train_time_sec",
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
        "rows_processed_per_sec",
        "top_5_features",
        "champion",
    ]

    # Ensure all columns exist (fill missing with empty string)
    for col in csv_columns:
        if col not in results_df.columns:
            results_df[col] = ""

    results_df[csv_columns].to_csv(output_file, index=False)
    logger.info(f"  Saved results to {output_file}")

    total_time = time.time() - pipeline_start_time
    logger.info(f"\n✓ Training competition complete! Total time: {total_time:.2f}s")

    # Add disclaimer about results interpretation
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS INTERPRETATION NOTES:")
    logger.info("=" * 70)
    logger.info("⚠️  High F1 scores (especially 1.0) may indicate:")
    logger.info("   - Data leakage (future info in features)")
    logger.info("   - Label definition too easy (class imbalance)")
    logger.info("   - Small test set with repetitive patterns")
    logger.info("")
    logger.info("⚠️  Precision@K/Recall@K = 0.0 indicates:")
    logger.info("   - Test horizon may be too small (< 24h recommended)")
    logger.info("   - K may be larger than available spatial units")
    logger.info("   - Hotspot ground truth may be empty in test hours")
    logger.info("")
    logger.info("Next steps: Review leakage audit results above and verify:")
    logger.info("   1. Lag features are backward-looking only")
    logger.info("   2. Rolling windows exclude future data")
    logger.info("   3. Test horizon has sufficient support (24h+ recommended)")
    logger.info("=" * 70)

    return results_df
