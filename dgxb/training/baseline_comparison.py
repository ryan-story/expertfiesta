"""
Baseline comparison for hotspot prediction
Compares model performance against naive historical baseline
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def compute_historical_baseline_patk(
    h3_cells: np.ndarray,
    timestamps: pd.Series,
    actual_counts_df: pd.DataFrame,
    k: int = 50,
) -> Dict[str, float]:
    """
    Compute Precision@K for naive baseline: always predict top-K historically
    highest incident cells for that hour-of-day

    Args:
        h3_cells: H3 cell identifiers for each sample
        timestamps: Timestamps for each sample
        actual_counts_df: DataFrame with actual incident counts per (H3, hour)
        k: Number of top hotspots to consider

    Returns:
        Dictionary with precision_at_k, recall_at_k for baseline
    """
    logger.info("=" * 70)
    logger.info("HISTORICAL BASELINE: Top-K cells by hour-of-day")
    logger.info("=" * 70)

    timestamps = pd.to_datetime(timestamps)
    timestamps_hour = timestamps.dt.floor("h")

    # Group actual counts by (H3, hour_of_day) and compute historical average
    actual_counts_df = actual_counts_df.copy()
    actual_counts_df["hour_actual"] = pd.to_datetime(actual_counts_df["hour_actual"])
    actual_counts_df["hour_of_day"] = actual_counts_df["hour_actual"].dt.hour

    # Historical average incidents per (H3, hour_of_day)
    historical_avg = (
        actual_counts_df.groupby(["h3_cell", "hour_of_day"])["incident_count"]
        .mean()
        .reset_index()
    )
    historical_avg.columns = ["h3_cell", "hour_of_day", "avg_incident_count"]

    # For each unique hour in test data, predict top-K by historical average
    precision_at_k_list = []
    recall_at_k_list = []

    unique_hours = timestamps_hour.unique()
    k_min = 5

    for hour_t in unique_hours:
        hour_of_day_t = hour_t.hour

        # Get historical averages for this hour-of-day
        historical_for_hour = historical_avg[
            historical_avg["hour_of_day"] == hour_of_day_t
        ].copy()

        if len(historical_for_hour) < k_min:
            continue

        # Top-K predicted (by historical average)
        historical_for_hour = historical_for_hour.sort_values(
            "avg_incident_count", ascending=False
        )
        k_t = min(k, max(k_min, len(historical_for_hour) // 10))
        top_k_predicted = historical_for_hour.head(k_t)["h3_cell"].values

        # Get actuals at t+1
        hour_t_plus_1 = hour_t + pd.Timedelta(hours=1)
        actual_data = actual_counts_df[
            actual_counts_df["hour_actual"] == hour_t_plus_1
        ].copy()

        if len(actual_data) == 0:
            continue

        # Top-K actual hotspots
        actual_data_sorted = actual_data.sort_values("incident_count", ascending=False)
        top_k_actual = actual_data_sorted.head(k_t)["h3_cell"].values

        # Compute Precision@K and Recall@K
        predicted_set = set(top_k_predicted)
        actual_set = set(top_k_actual)

        precision_k = (
            len(predicted_set & actual_set) / len(predicted_set)
            if len(predicted_set) > 0
            else 0.0
        )
        recall_k = (
            len(predicted_set & actual_set) / len(actual_set)
            if len(actual_set) > 0
            else 0.0
        )

        precision_at_k_list.append(precision_k)
        recall_at_k_list.append(recall_k)

    if len(precision_at_k_list) == 0:
        logger.warning("No valid hours for baseline computation")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    avg_precision = np.mean(precision_at_k_list)
    avg_recall = np.mean(recall_at_k_list)

    logger.info(f"\nBaseline Precision@K: {avg_precision:.4f}")
    logger.info(f"Baseline Recall@K:    {avg_recall:.4f}")
    logger.info(f"Evaluated over {len(precision_at_k_list)} hours")

    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "n_hours_evaluated": len(precision_at_k_list),
    }
