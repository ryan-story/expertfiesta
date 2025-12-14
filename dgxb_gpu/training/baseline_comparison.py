"""
Baseline comparison for hotspot prediction (GPU)
Compares model performance against naive historical baseline:
predict top-K historically hottest H3 cells for each hour-of-day.

GPU notes:
- Uses cuDF for grouping/sorting/joins.
- Avoids Python loop over hours by computing metrics via joins on GPU.
- Returns python floats for easy logging/serialization.

Inputs:
- h3_cells: array-like (np.ndarray / list) of H3 cell ids for each sample (not strictly needed)
- timestamps: pd.Series-like of timestamps for each sample (test set times t)
- actual_counts_df: pd.DataFrame or cudf.DataFrame with columns:
    - h3_cell
    - hour_actual (timestamp of the hour)
    - incident_count
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_gpu_libs():
    try:
        import cudf  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "cuDF not available. Install RAPIDS (cudf) for your CUDA version. "
            f"Original error: {e}"
        )


def compute_historical_baseline_patk_gpu(
    h3_cells: Any,  # kept for API parity; not required
    timestamps: pd.Series,
    actual_counts_df: Any,
    k: int = 50,
    k_min: int = 5,
    dynamic_k: bool = True,
    dynamic_k_frac: float = 0.10,
) -> Dict[str, float]:
    """
    GPU Precision@K / Recall@K for naive baseline: top-K historically hottest cells per hour-of-day.

    Differences vs CPU version:
    - No per-hour loop.
    - Computes overlap via GPU joins and groupbys.
    - Computes K per hour-of-day (optionally dynamic) and evaluates for each unique hour in test set.

    Args:
        h3_cells: unused (kept for signature compatibility)
        timestamps: timestamps for each sample (test hours t)
        actual_counts_df: pd.DataFrame or cudf.DataFrame with (h3_cell, hour_actual, incident_count)
        k: max K
        k_min: minimum K
        dynamic_k: if True, K is min(k, max(k_min, n_cells_for_hour//10)) per hour-of-day
        dynamic_k_frac: fraction for dynamic K (default 10% of available cells)

    Returns:
        dict: precision_at_k, recall_at_k, n_hours_evaluated
    """
    _require_gpu_libs()
    import cudf

    logger.info("=" * 70)
    logger.info("HISTORICAL BASELINE (GPU): Top-K cells by hour-of-day")
    logger.info("=" * 70)

    if timestamps is None or len(timestamps) == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    # ---- Move inputs to GPU-friendly structures ----
    # Test timestamps -> unique hours t (floored), and compute target hour t+1.
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    ts = ts.dropna()
    if len(ts) == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    hours_t = pd.Series(ts.dt.floor("h").unique())
    hours_t = pd.to_datetime(hours_t, utc=True, errors="coerce").dropna()
    if len(hours_t) == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    # Build a small GPU DF for evaluation hours: hour_t, hour_t_plus_1, hour_of_day
    eval_pdf = pd.DataFrame(
        {
            "hour_t": hours_t,
        }
    )
    eval_pdf["hour_t_plus_1"] = eval_pdf["hour_t"] + pd.Timedelta(hours=1)
    eval_pdf["hour_of_day"] = eval_pdf["hour_t"].dt.hour.astype(np.int32)
    eval_gdf = cudf.from_pandas(eval_pdf)

    # Actual counts DF to cuDF
    if isinstance(actual_counts_df, cudf.DataFrame):
        ac = actual_counts_df.copy(deep=False)
    else:
        ac = cudf.from_pandas(actual_counts_df.copy())

    # Ensure required columns
    required = {"h3_cell", "hour_actual", "incident_count"}
    missing = required - set(ac.columns)
    if missing:
        raise ValueError(f"actual_counts_df missing columns: {sorted(list(missing))}")

    # Normalize types
    ac["hour_actual"] = cudf.to_datetime(ac["hour_actual"], utc=True, errors="coerce")
    ac = ac.dropna(subset=["hour_actual", "h3_cell"])
    if len(ac) == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    ac["hour_of_day"] = ac["hour_actual"].dt.hour.astype("int32")
    ac["incident_count"] = cudf.to_numeric(
        ac["incident_count"], errors="coerce"
    ).fillna(0)

    # ---- Historical averages per (h3_cell, hour_of_day) ----
    hist = (
        ac.groupby(["h3_cell", "hour_of_day"])["incident_count"]
        .mean()
        .reset_index()
        .rename(columns={"incident_count": "avg_incident_count"})
    )

    # Determine how many candidate cells exist per hour_of_day (for dynamic K and k_min gating)
    n_cells = (
        hist.groupby("hour_of_day")["h3_cell"]
        .count()
        .reset_index()
        .rename(columns={"h3_cell": "n_cells"})
    )

    # Join n_cells onto eval hours and filter hours that have enough history
    eval2 = eval_gdf.merge(n_cells, on="hour_of_day", how="left")
    eval2["n_cells"] = eval2["n_cells"].fillna(0).astype("int32")
    eval2 = eval2[eval2["n_cells"] >= k_min]
    if len(eval2) == 0:
        logger.warning(
            "No valid hours for baseline computation (insufficient historical cells)"
        )
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    # Compute k_t per hour (dynamic or fixed)
    if dynamic_k:
        # k_t = min(k, max(k_min, floor(n_cells*dynamic_k_frac)))
        # Use integer ops on GPU
        kt = (
            (eval2["n_cells"].astype("float32") * float(dynamic_k_frac))
            .floor()
            .astype("int32")
        )
        kt = kt.clip(lower=k_min)
        eval2["k_t"] = kt.clip(upper=int(k)).astype("int32")
    else:
        eval2["k_t"] = int(k)

    # ---- Compute top-k predicted per hour_of_day from historical averages ----
    # Rank within each hour_of_day by avg_incident_count desc
    hist_sorted = hist.sort_values(
        ["hour_of_day", "avg_incident_count"], ascending=[True, False]
    )
    hist_sorted["pred_rank"] = (
        hist_sorted.groupby("hour_of_day").cumcount() + 1
    )  # 1-indexed

    # Attach k_t per hour_of_day (use max k_t needed per hour_of_day)
    # (k_t can vary per hour_t even within same hour_of_day, but that's rare; we handle it exactly below
    # by carrying k_t per hour_t and filtering by pred_rank <= k_t after join.)
    pred = eval2[["hour_t", "hour_t_plus_1", "hour_of_day", "k_t"]].merge(
        hist_sorted[["h3_cell", "hour_of_day", "pred_rank"]],
        on="hour_of_day",
        how="left",
    )
    pred = pred[pred["pred_rank"] <= pred["k_t"]]
    # pred rows: (hour_t, hour_t_plus_1, h3_cell) predicted

    # ---- Compute top-k actual per hour_t_plus_1 (based on incident_count at that hour) ----
    # Filter actuals to only the hours we evaluate (hour_t_plus_1 set)
    # Join eval2 hour_t_plus_1 -> actual counts
    actual_join = eval2[["hour_t", "hour_t_plus_1", "k_t"]].merge(
        ac[["h3_cell", "hour_actual", "incident_count"]],
        left_on="hour_t_plus_1",
        right_on="hour_actual",
        how="left",
    )
    actual_join = actual_join.drop(columns=["hour_actual"])
    actual_join["incident_count"] = actual_join["incident_count"].fillna(0)

    # Rank actual hotspots within each hour_t (for hour_t_plus_1 target)
    actual_sorted = actual_join.sort_values(
        ["hour_t", "incident_count"], ascending=[True, False]
    )
    actual_sorted["act_rank"] = actual_sorted.groupby("hour_t").cumcount() + 1
    actual_topk = actual_sorted[actual_sorted["act_rank"] <= actual_sorted["k_t"]]
    # actual_topk rows: (hour_t, h3_cell) actual hotspots at t+1

    # Some hours may have no actual data (all-null after join)
    # Filter to hours with at least 1 actual row
    n_actual = (
        actual_topk.groupby("hour_t")["h3_cell"]
        .count()
        .reset_index()
        .rename(columns={"h3_cell": "n_actual"})
    )
    eval3 = eval2.merge(n_actual, on="hour_t", how="left")
    eval3["n_actual"] = eval3["n_actual"].fillna(0).astype("int32")
    eval3 = eval3[eval3["n_actual"] > 0]
    if len(eval3) == 0:
        logger.warning("No valid hours for baseline computation (no actuals at t+1)")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_hours_evaluated": 0}

    # Restrict predictions to only hours that have actuals
    pred2 = pred.merge(eval3[["hour_t", "k_t"]], on="hour_t", how="inner")
    actual2 = actual_topk.merge(eval3[["hour_t", "k_t"]], on="hour_t", how="inner")

    # ---- Intersection size per hour_t via join on (hour_t, h3_cell) ----
    inter = pred2[["hour_t", "h3_cell"]].merge(
        actual2[["hour_t", "h3_cell"]],
        on=["hour_t", "h3_cell"],
        how="inner",
    )
    inter_counts = (
        inter.groupby("hour_t")["h3_cell"]
        .count()
        .reset_index()
        .rename(columns={"h3_cell": "n_intersection"})
    )

    # Predicted count per hour_t (should be k_t but guard)
    pred_counts = (
        pred2.groupby("hour_t")["h3_cell"]
        .count()
        .reset_index()
        .rename(columns={"h3_cell": "n_pred"})
    )
    act_counts = (
        actual2.groupby("hour_t")["h3_cell"]
        .count()
        .reset_index()
        .rename(columns={"h3_cell": "n_act"})
    )

    metrics = (
        eval3[["hour_t", "k_t"]]
        .merge(pred_counts, on="hour_t", how="left")
        .merge(act_counts, on="hour_t", how="left")
        .merge(inter_counts, on="hour_t", how="left")
    )

    metrics["n_pred"] = metrics["n_pred"].fillna(0).astype("float32")
    metrics["n_act"] = metrics["n_act"].fillna(0).astype("float32")
    metrics["n_intersection"] = metrics["n_intersection"].fillna(0).astype("float32")

    # Precision = |∩| / |pred| ; Recall = |∩| / |act|
    metrics["precision_k"] = (metrics["n_intersection"] / metrics["n_pred"]).fillna(0)
    metrics["recall_k"] = (metrics["n_intersection"] / metrics["n_act"]).fillna(0)

    # Mean over hours (GPU -> CPU scalar)
    avg_precision = float(metrics["precision_k"].mean())
    avg_recall = float(metrics["recall_k"].mean())
    n_hours = int(len(metrics))

    logger.info(f"\nBaseline Precision@K: {avg_precision:.4f}")
    logger.info(f"Baseline Recall@K:    {avg_recall:.4f}")
    logger.info(f"Evaluated over {n_hours} hours")

    return {
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "n_hours_evaluated": float(n_hours),
    }
