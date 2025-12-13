"""
Comprehensive metrics tracking for model competition
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import logging
import time
import h3
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_cell_hour_f1(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    h3_cells: np.ndarray,
    timestamps_pred: pd.Series,  # When prediction was made (t)
    timestamps_label: pd.Series,  # When actual incident occurred (t+1)
    hazard_class_id: int,
    hazard_class_ids: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute F1, precision, and recall at cell-hour aggregation level

    Aggregates predictions and labels to (H3 cell, hour) level:
    - Predictions: aggregated by prediction hour (t)
    - Labels: aggregated by actual incident hour (t+1)
    - Label: has_hazard_incident (binary: 1 if any incident in that cell-hour is CRASH or HAZARDOUS CONDITION, else 0)
    - Prediction: max_hazard_probability (max probability of hazard class across all incidents in that cell-hour)

    This provides F1 at the decision unit level, complementing event-level F1.

    Args:
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
        y_true: True labels (shape: [n_samples]) - encoded class IDs
        h3_cells: H3 cell identifiers for each sample
        timestamps_pred: Timestamps when predictions were made (time t)
        timestamps_label: Timestamps when actual incidents occurred (time t+1)
        hazard_class_id: Encoded class ID for hazard class (used for probability extraction)
        hazard_class_ids: Optional list of all hazard class IDs (for label checking, defaults to [hazard_class_id])
        threshold: Probability threshold for binary prediction (default 0.5)

    Returns:
        Dictionary with cell_hour_f1, cell_hour_precision, cell_hour_recall
    """
    timestamps_pred = pd.to_datetime(timestamps_pred)
    timestamps_label = pd.to_datetime(timestamps_label)
    
    # Use provided hazard_class_ids or default to single class
    if hazard_class_ids is None:
        hazard_class_ids = [hazard_class_id]

    # Get hazard class probability for each sample (use max across all hazard classes if multiple)
    if y_pred_proba.ndim > 1:
        if len(hazard_class_ids) > 1:
            # Use max probability across all hazard classes
            hazard_proba = np.max(y_pred_proba[:, hazard_class_ids], axis=1)
        else:
            hazard_proba = y_pred_proba[:, hazard_class_id]
    else:
        # Binary case - use probabilities directly
        hazard_proba = y_pred_proba

    # Aggregate predictions by prediction hour (t)
    df_pred = pd.DataFrame(
        {
            "h3_cell": h3_cells,
            "hour_pred": timestamps_pred.dt.floor("h"),
            "hazard_proba": hazard_proba,
        }
    )
    df_pred_agg = (
        df_pred.groupby(["h3_cell", "hour_pred"])
        .agg({"hazard_proba": "max"})  # Max probability across incidents in this cell-hour
        .reset_index()
    )
    # Shift prediction hour forward by 1 hour to align with label hour (t+1)
    df_pred_agg["hour"] = df_pred_agg["hour_pred"] + pd.Timedelta(hours=1)
    df_pred_agg = df_pred_agg[["h3_cell", "hour", "hazard_proba"]]

    # Aggregate labels by actual incident hour (t+1)
    df_label = pd.DataFrame(
        {
            "h3_cell": h3_cells,
            "hour_label": timestamps_label.dt.floor("h"),
            "is_hazard": np.isin(y_true, hazard_class_ids).astype(int),
        }
    )
    df_label_agg = (
        df_label.groupby(["h3_cell", "hour_label"])
        .agg({"is_hazard": "max"})  # Any hazard incident in this cell-hour
        .reset_index()
    )
    df_label_agg.columns = ["h3_cell", "hour", "is_hazard"]

    # Join predictions (at hour t, shifted to t+1) with labels (at hour t+1)
    # Predictions at hour t should be compared against labels at hour t+1
    df_merged = df_pred_agg.merge(
        df_label_agg, on=["h3_cell", "hour"], how="inner"
    )

    # Diagnostic logging
    logger = logging.getLogger(__name__)
    if len(df_merged) > 0:
        cell_hour_positive_rate = df_merged["is_hazard"].mean()
        unique_cell_hours = len(df_merged)
        # Use original prediction hours (before shift) for diagnostics
        pred_hour_min = df_pred["hour_pred"].min()
        pred_hour_max = df_pred["hour_pred"].max()
        label_hour_min = df_label_agg["hour"].min()
        label_hour_max = df_label_agg["hour"].max()

        logger.info(
            f"  Cell-hour F1 diagnostics:"
        )
        logger.info(
            f"    Unique cell-hours evaluated: {unique_cell_hours}"
        )
        logger.info(
            f"    Cell-hour positive rate: {cell_hour_positive_rate:.4f} ({cell_hour_positive_rate*100:.2f}%)"
        )
        logger.info(
            f"    Prediction hour range (t): {pred_hour_min} to {pred_hour_max}"
        )
        logger.info(
            f"    Label hour range (t+1): {label_hour_min} to {label_hour_max}"
        )

        # Sanity check: warn if positive rate is extremely high
        if cell_hour_positive_rate > 0.9:
            logger.warning(
                f"    ⚠️  Cell-hour positive rate > 0.9 - this may indicate label definition issue"
            )

        # Sanity check: verify temporal alignment (prediction hours should be before label hours)
        # After shifting predictions forward by 1 hour, they should align with labels
        pred_hour_max_shifted = pred_hour_max + pd.Timedelta(hours=1)
        if pred_hour_max_shifted > label_hour_max:
            logger.warning(
                "    ⚠️  Temporal alignment issue: shifted prediction hours exceed label hours"
            )
    else:
        logger.warning(
            "  ⚠️  No matching cell-hours found between predictions and labels"
        )

    if len(df_merged) == 0:
        # No matching cell-hours - return zero metrics
        return {
            "cell_hour_f1": 0.0,
            "cell_hour_precision": 0.0,
            "cell_hour_recall": 0.0,
        }

    # Convert probabilities to binary predictions
    y_pred_binary = (df_merged["hazard_proba"] >= threshold).astype(int)
    y_true_binary = df_merged["is_hazard"].values

    # Compute metrics
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        "cell_hour_f1": f1,
        "cell_hour_precision": precision,
        "cell_hour_recall": recall,
    }


def compute_hotspot_metrics(
    y_pred: np.ndarray,
    actual_counts_df: pd.DataFrame,
    h3_cells: np.ndarray,
    timestamps_pred: pd.Series,
    timestamps_actual: pd.Series,
    k: int = 50,
    h3_resolution: int = 9,
) -> Dict[str, float]:
    """
    Compute Precision@K and Recall@K for hotspot prediction (regression)

    Prediction unit: (H3 cell, hour) pairs
    Score: Model predicts incident_count(h3, t+1) at time t
    Ground truth: Top-K H3 cells by actual incident_count in hour t+1

    For each hour_ts in test:
    - Predicted hotspots: top-K cells by predicted y at t+1
    - Actual hotspots: top-K cells by actual y at t+1
    - Precision@K/Recall@K computed per hour, then averaged across hours
    - Dynamic K: K = min(K_fixed, num_cells_with_any_history_that_hour)
    - Conditional metrics: conditional on total_incidents(hour_t+1) > 0

    Args:
        y_pred: Predicted incident counts (shape: [n_samples])
        actual_counts_df: DataFrame with columns ['h3_cell', 'hour_actual', 'incident_count']
            containing actual incident counts per (H3 cell, hour)
        h3_cells: H3 cell identifiers for each sample
        timestamps_pred: Timestamps when prediction was made (hour t)
        timestamps_actual: Timestamps of actual incidents (hour t+1)
        k: Number of top hotspots to consider (will be dynamically adjusted per hour)
        h3_resolution: H3 resolution used

    Returns:
        Dictionary with precision_at_k, recall_at_k, conditional variants, and staging utility metrics
    """
    timestamps_pred = pd.to_datetime(timestamps_pred)
    timestamps_actual = pd.to_datetime(timestamps_actual)

    # Use predicted counts directly as risk scores
    risk_scores = y_pred

    # Create DataFrame for predictions
    df_pred = pd.DataFrame(
        {
            "h3_cell": h3_cells,
            "timestamp_pred": timestamps_pred,  # When prediction was made (t)
            "risk_score": risk_scores,
        }
    )
    df_pred["hour_pred"] = df_pred["timestamp_pred"].dt.floor("h")

    # Use actual_counts_df directly (already aggregated by h3_cell and hour_actual)
    # Ensure hour_actual is datetime
    actual_counts_df = actual_counts_df.copy()
    actual_counts_df["hour_actual"] = pd.to_datetime(actual_counts_df["hour_actual"])

    # Group predictions by (H3 cell, prediction hour)
    df_pred_grouped = (
        df_pred.groupby(["h3_cell", "hour_pred"])
        .agg({"risk_score": "max"})  # Max risk score for this cell-hour
        .reset_index()
    )

    # For each prediction hour t, compare with actuals at t+1
    precision_at_k_list = []
    recall_at_k_list = []
    precision_at_k_conditional_list = []  # Only hours with ≥1 incident
    recall_at_k_conditional_list = []
    coverage_list = []
    coverage_with_neighbors_list = []

    unique_pred_hours = df_pred_grouped["hour_pred"].unique()
    k_min = 5  # Minimum K for dynamic adjustment

    for hour_t in unique_pred_hours:
        # Get predictions made at hour t (for t+1)
        pred_data = df_pred_grouped[df_pred_grouped["hour_pred"] == hour_t].copy()

        if len(pred_data) == 0:
            continue

        # Dynamic K: adjust based on available candidates
        k_t = min(k, max(k_min, len(pred_data) // 10))
        k_t = max(1, k_t)  # Ensure at least 1

        if len(pred_data) < k_t:
            # Not enough cells for top-K
            continue

        # Get actuals at hour t+1
        hour_t_plus_1 = hour_t + pd.Timedelta(hours=1)
        actual_data = actual_counts_df[
            actual_counts_df["hour_actual"] == hour_t_plus_1
        ].copy()

        if len(actual_data) == 0:
            # No actuals at t+1, skip this hour
            continue

        # Sort predictions by risk score (descending)
        pred_data = pred_data.sort_values("risk_score", ascending=False)

        # Top-K predicted hotspots (at hour t, for t+1)
        top_k_predicted = pred_data.head(k_t)["h3_cell"].values

        # Merge actuals with predictions to get all cells that have both
        # For cells with predictions but no actuals, incident_count = 0
        merged_data = pred_data.merge(
            actual_data, left_on="h3_cell", right_on="h3_cell", how="left"
        )
        merged_data["incident_count"] = merged_data["incident_count"].fillna(0)

        # Top-K actual hotspots (by incident count at t+1)
        actual_data_sorted = actual_data.sort_values("incident_count", ascending=False)
        top_k_actual = actual_data_sorted.head(k_t)["h3_cell"].values

        # Precision@K: Of top-K predicted, how many are in top-K actual?
        predicted_set = set(top_k_predicted)
        actual_set = set(top_k_actual)
        precision_k = (
            len(predicted_set & actual_set) / len(predicted_set)
            if len(predicted_set) > 0
            else 0.0
        )

        # Recall@K: Of top-K actual, how many are in top-K predicted?
        recall_k = (
            len(predicted_set & actual_set) / len(actual_set)
            if len(actual_set) > 0
            else 0.0
        )

        precision_at_k_list.append(precision_k)
        recall_at_k_list.append(recall_k)

        # Conditional metrics: only for hours with ≥1 incident
        # This answers: "When incidents occur, how well do we rank them?"
        if len(actual_data) > 0 and actual_data["incident_count"].sum() > 0:
            # At least one incident in this hour
            precision_at_k_conditional_list.append(precision_k)
            recall_at_k_conditional_list.append(recall_k)
        else:
            # No incidents in this hour - skip for conditional metric
            pass

        # Staging utility: % of cell-hours with ≥1 incident that are inside predicted top-K
        # Binary coverage metric (not count-weighted) to avoid collapse when total_incidents is small
        # Use actual_data (all cell-hours with actual incidents at t+1), not merged_data
        cell_hours_with_incidents = actual_data[actual_data["incident_count"] >= 1]

        if len(cell_hours_with_incidents) > 0:
            # Count cell-hours with incidents that are in predicted top-K
            cell_hours_covered = cell_hours_with_incidents[
                cell_hours_with_incidents["h3_cell"].isin(predicted_set)
            ]
            coverage = len(cell_hours_covered) / len(cell_hours_with_incidents)
        else:
            coverage = 0.0
        coverage_list.append(coverage)

        # Coverage with 1-ring neighbors
        predicted_with_neighbors = set(top_k_predicted)
        for cell in top_k_predicted:
            try:
                neighbors = get_h3_neighbors(cell, k=1)
                predicted_with_neighbors.update(neighbors)
            except Exception:
                pass

        if len(cell_hours_with_incidents) > 0:
            # Count cell-hours with incidents that are in predicted top-K or neighbors
            cell_hours_covered_neighbors = cell_hours_with_incidents[
                cell_hours_with_incidents["h3_cell"].isin(predicted_with_neighbors)
            ]
            coverage_neighbors = len(cell_hours_covered_neighbors) / len(
                cell_hours_with_incidents
            )
        else:
            coverage_neighbors = 0.0
        coverage_with_neighbors_list.append(coverage_neighbors)

    # Average across all hours (unconditional)
    precision_at_k_val = float(
        np.mean(precision_at_k_list) if precision_at_k_list else 0.0
    )
    recall_at_k_val = float(np.mean(recall_at_k_list) if recall_at_k_list else 0.0)

    # Average across hours with ≥1 incident (conditional - operational relevance)
    precision_at_k_conditional_val = float(
        np.mean(precision_at_k_conditional_list)
        if precision_at_k_conditional_list
        else 0.0
    )
    recall_at_k_conditional_val = float(
        np.mean(recall_at_k_conditional_list) if recall_at_k_conditional_list else 0.0
    )

    staging_utility_coverage_val = float(
        np.mean(coverage_list) if coverage_list else 0.0
    )
    staging_utility_with_neighbors_val = float(
        np.mean(coverage_with_neighbors_list) if coverage_with_neighbors_list else 0.0
    )

    return {
        "precision_at_k": precision_at_k_val,  # Unconditional (all hours)
        "recall_at_k": recall_at_k_val,  # Unconditional (all hours)
        "hotspot_precision_at_k_conditional": precision_at_k_conditional_val,  # Conditional (hours with incidents)
        "hotspot_recall_at_k_conditional": recall_at_k_conditional_val,  # Conditional (hours with incidents)
        "staging_utility_coverage_pct": staging_utility_coverage_val * 100.0,
        "staging_utility_with_neighbors_pct": staging_utility_with_neighbors_val
        * 100.0,
    }


def get_h3_neighbors(cell: str, k: int = 1, h3_resolution: int = 9) -> set:
    """Get H3 neighbors, handling both h3-py v3 and v4+ APIs"""
    try:
        # Try new API first (h3-py v4+)
        try:
            neighbor_cells = h3.grid_ring(cell, k)
            return set(neighbor_cells)
        except AttributeError:
            # Fallback to old API (h3-py v3)
            neighbor_cells = h3.k_ring(cell, k)
            return set(neighbor_cells)
    except Exception as e:
        logger.debug(f"H3 neighbor lookup failed for {cell}: {e}")
        return {cell}


def measure_inference_latency(
    model: Any,
    X_test: pd.DataFrame,
    batch_size: int = 10000,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure inference latency (cold-start and warm-start)

    Args:
        model: Trained model
        X_test: Test features
        batch_size: Batch size for inference (default 10000 rows)
        n_warmup: Number of warmup runs
        n_runs: Number of runs for warm-start measurement

    Returns:
        Dictionary with latency metrics (p50, p95 in milliseconds)
    """
    # Cold-start: First inference
    start_time = time.time()
    _ = model.predict(X_test.iloc[:batch_size])
    cold_start_time = (time.time() - start_time) * 1000  # Convert to ms

    # Warm-start: Multiple runs
    warm_times = []
    for _ in range(n_warmup + n_runs):
        start_time = time.time()
        _ = model.predict(X_test.iloc[:batch_size])
        warm_times.append((time.time() - start_time) * 1000)

    # Skip warmup runs
    warm_times = warm_times[n_warmup:]

    # Compute percentiles
    p50 = np.percentile(warm_times, 50)
    p95 = np.percentile(warm_times, 95)

    return {
        "inference_latency_cold_start_ms": float(cold_start_time),
        "inference_latency_p50_ms": float(p50),
        "inference_latency_p95_ms": float(p95),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics for count prediction
    
    Args:
        y_true: True incident counts
        y_pred: Predicted incident counts
        
    Returns:
        Dictionary with rmse, mae, r2, smape
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # SMAPE: Symmetric Mean Absolute Percentage Error (handles zeros safely)
    # SMAPE = 100 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + epsilon))
    epsilon = 1e-8  # Small epsilon to avoid division by zero
    smape = 100 * np.mean(
        np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
    )
    
    # MAPE on positive values only (optional diagnostic)
    y_pos_mask = y_true > 0
    if y_pos_mask.sum() > 0:
        mape_pos = 100 * np.mean(
            np.abs(y_true[y_pos_mask] - y_pred[y_pos_mask]) / (y_true[y_pos_mask] + epsilon)
        )
    else:
        mape_pos = np.nan
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape,
        "mape_pos": mape_pos,
    }


def compute_model_quality_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard model quality metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)

    Returns:
        Dictionary with quality metrics
    """
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    # Weighted F1 (precision-weighted)
    weighted_f1 = 0.6 * precision + 0.4 * recall

    return {
        "f1_score": f1,
        "precision_score": precision,
        "recall_score": recall,
        "weighted_f1": weighted_f1,
    }


def track_pipeline_metrics(
    ingest_cleaning_time_sec: float,
    feature_build_time_sec: float,
    train_time_sec: float,
    inference_latency_metrics: Dict[str, float],
    staging_recommendation_time_sec: float,
    rows_processed: int,
    dataset_size: int,
) -> Dict[str, float]:
    """
    Track pipeline latency and throughput metrics

    Args:
        ingest_cleaning_time_sec: Time to load and clean data
        feature_build_time_sec: Time to build features
        train_time_sec: Model training time
        inference_latency_metrics: Dictionary from measure_inference_latency
        staging_recommendation_time_sec: Time to generate top-K hotspots
        rows_processed: Number of rows processed
        dataset_size: Total dataset size

    Returns:
        Dictionary with all pipeline metrics
    """
    # Throughput
    total_feature_time = feature_build_time_sec
    rows_per_sec = (
        rows_processed / total_feature_time if total_feature_time > 0 else 0.0
    )

    # Time to first dashboard update
    time_to_first_update = (
        ingest_cleaning_time_sec + feature_build_time_sec + train_time_sec
    )

    metrics = {
        "ingest_cleaning_time_sec": ingest_cleaning_time_sec,
        "feature_build_time_sec": feature_build_time_sec,
        "train_time_sec": train_time_sec,
        "staging_recommendation_time_sec": staging_recommendation_time_sec,
        "rows_processed_per_sec": rows_per_sec,
        "max_dataset_size_feasible": dataset_size,  # Could be estimated based on memory
        "time_to_first_dashboard_update_sec": time_to_first_update,
    }

    # Add inference latency metrics
    metrics.update(inference_latency_metrics)

    return metrics
