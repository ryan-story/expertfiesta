import cudf
import cupy as cp
import numpy as np
from pathlib import Path
import logging
import time
from typing import List, Any, Dict

from .model_competition import (
    train_cuml_linear_regression,
    train_cuml_random_forest,
    train_xgboost_gpu,
)
from .metrics_tracker import (
    compute_hotspot_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)


# ----------------------------
# GPU helper utilities
# ----------------------------


def _cudf_numeric_only(df: cudf.DataFrame) -> cudf.DataFrame:
    """Keep numeric + bool; cast bool -> int8; drop non-numerics."""
    out = df.copy(deep=False)
    for c in out.columns:
        if out[c].dtype == "bool":
            out[c] = out[c].astype("int8")
    # Select numeric
    numeric_cols = []
    for c in out.columns:
        if cudf.api.types.is_numeric_dtype(out[c].dtype):
            numeric_cols.append(c)
    return out[numeric_cols]


def _gpu_median_impute(df: cudf.DataFrame) -> cudf.DataFrame:
    """Fill NaNs with per-column median (GPU)."""
    # cudf median returns Series; fillna aligns by column name.
    med = df.median()
    return df.fillna(med)


def _to_cupy_matrix(df: cudf.DataFrame) -> cp.ndarray:
    """cudf -> cupy float32 (2D)."""
    X = df.to_cupy()
    if X.dtype != cp.float32:
        X = X.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(X)


def _to_cupy_y(y: cudf.Series) -> cp.ndarray:
    arr = y.to_cupy()
    if arr.dtype != cp.float32:
        arr = arr.astype(cp.float32, copy=False)
    return cp.ascontiguousarray(arr)


def _rmse_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def _measure_inference_latency_gpu(
    model: Any,
    X_cp: cp.ndarray,
    batch_size: int = 100_000,
    n_warmup: int = 5,
    n_runs: int = 30,
) -> Dict[str, float]:
    """GPU inference latency: returns p50/p95 (ms). Synchronizes CUDA for accurate timing."""
    import time

    n = min(batch_size, X_cp.shape[0])
    Xb = X_cp[:n]

    # Cold
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    _ = model.predict(Xb)
    cp.cuda.runtime.deviceSynchronize()
    cold_ms = (time.time() - t0) * 1000.0

    # Warm
    times = []
    for _ in range(n_warmup + n_runs):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        _ = model.predict(Xb)
        cp.cuda.runtime.deviceSynchronize()
        times.append((time.time() - t0) * 1000.0)

    times = times[n_warmup:]
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return {
        "inference_latency_cold_start_ms": cold_ms,
        "inference_latency_p50_ms": p50,
        "inference_latency_p95_ms": p95,
    }


# ----------------------------
# GPU baselines
# ----------------------------


def _baseline_persistence_predict(X_cudf: cudf.DataFrame) -> cp.ndarray:
    """
    Persistence baseline: y_hat = incident_count_t if present else 0.
    Returns CuPy array float32.
    """
    if "incident_count_t" in X_cudf.columns:
        yhat = X_cudf["incident_count_t"].fillna(0).astype("float32").to_cupy()
    else:
        yhat = cp.zeros((len(X_cudf),), dtype=cp.float32)
    return yhat


def _fit_climatology_mapping_gpu(
    X_train: cudf.DataFrame, y_train_cp: cp.ndarray
) -> cudf.DataFrame:
    """
    Build mapping table: mean(y) per (h3_cell, hour_of_week).
    Returns cudf DataFrame with columns [h3_cell, hour_of_week, y_mean].
    """
    required = {"h3_cell", "hour", "day_of_week"}
    if not required.issubset(set(X_train.columns)):
        # Empty mapping means fallback only
        return cudf.DataFrame({"h3_cell": [], "hour_of_week": [], "y_mean": []})

    tmp = X_train[["h3_cell", "hour", "day_of_week"]].copy(deep=False)
    tmp["hour_of_week"] = (tmp["day_of_week"] * 24 + tmp["hour"]).astype("int16")
    tmp["y"] = cudf.Series(y_train_cp)  # aligns by row order
    mapping = tmp.groupby(["h3_cell", "hour_of_week"], as_index=False)["y"].mean()
    mapping = mapping.rename(columns={"y": "y_mean"})
    return mapping


def _predict_climatology_gpu(
    X_test: cudf.DataFrame, mapping: cudf.DataFrame, fallback: float
) -> cp.ndarray:
    required = {"h3_cell", "hour", "day_of_week"}
    if not required.issubset(set(X_test.columns)) or len(mapping) == 0:
        return cp.full((len(X_test),), fallback, dtype=cp.float32)

    tmp = X_test[["h3_cell", "hour", "day_of_week"]].copy(deep=False)
    tmp["hour_of_week"] = (tmp["day_of_week"] * 24 + tmp["hour"]).astype("int16")

    joined = tmp.merge(mapping, on=["h3_cell", "hour_of_week"], how="left")
    yhat = joined["y_mean"].fillna(fallback).astype("float32").to_cupy()
    return yhat


# ----------------------------
# Main orchestrator (GPU)
# ----------------------------


def run_training_competition_gpu(
    base_x_path: str = "gold-gpu-traffic/X_features.parquet",
    rich_x_path: str = "rich-gold-gpu-traffic/X_features.parquet",
    y_path: str = "gold-gpu-traffic/y_target.parquet",
    results_dir: str = "results",
    n_folds: int = 3,
    val_window_hours: int = 24,
    k_hotspots: int = 50,
    h3_resolution: int = 9,
    n_trials_rf: int = 30,
    n_trials_xgb: int = 30,
) -> cudf.DataFrame:
    logger.info("=" * 70)
    logger.info("GPU TRAINING COMPETITION PIPELINE (RAPIDS)")
    logger.info("=" * 70)

    pipeline_start = time.time()

    # ---- Step 1: Load with cuDF (GPU) ----
    logger.info("\n[Step 1/7] Loading parquet to GPU (cuDF)...")
    ingest_t0 = time.time()

    base_X = cudf.read_parquet(base_x_path)
    rich_X = cudf.read_parquet(rich_x_path)
    y_target = cudf.read_parquet(y_path)

    ingest_time = time.time() - ingest_t0
    logger.info(f"  Loaded base X: {len(base_X):,} rows, {len(base_X.columns)} cols")
    logger.info(f"  Loaded rich X: {len(rich_X):,} rows, {len(rich_X.columns)} cols")
    logger.info(
        f"  Loaded y:      {len(y_target):,} rows, {len(y_target.columns)} cols"
    )
    logger.info(f"  Ingest time: {ingest_time:.2f}s")

    # ---- Step 2: Extract hour_ts / h3_cell / y ----
    logger.info("\n[Step 2/7] Extracting hour_ts, h3_cell, and target (GPU)...")

    if "hour_ts" in base_X.columns:
        hour_ts = cudf.to_datetime(base_X["hour_ts"], utc=True)
    elif "hour_ts" in y_target.columns:
        hour_ts = cudf.to_datetime(y_target["hour_ts"], utc=True)
    else:
        raise ValueError("hour_ts must be present in X or y_target")

    if "h3_cell" in base_X.columns:
        h3_cells = base_X["h3_cell"]
    elif "h3_cell" in y_target.columns:
        h3_cells = y_target["h3_cell"]
    else:
        raise ValueError("h3_cell must be present in X or y_target")

    if "incident_count_t_plus_1" not in y_target.columns:
        raise ValueError("y_target must contain 'incident_count_t_plus_1'")

    y = y_target["incident_count_t_plus_1"]

    # Drop NaN targets
    valid_mask = ~y.isna()
    base_X = base_X[valid_mask].reset_index(drop=True)
    rich_X = rich_X[valid_mask].reset_index(drop=True)
    y_target = y_target[valid_mask].reset_index(drop=True)
    hour_ts = hour_ts[valid_mask].reset_index(drop=True)
    h3_cells = h3_cells[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    logger.info(f"  After dropping NaN targets: {len(y):,} rows")

    # ---- Step 3: Create nested CV splits (GPU-friendly) ----
    logger.info("\n[Step 3/7] Creating nested time-blocked CV splits...")
    from dgxb_gpu.training.cv_splitter import create_nested_cv_gpu

    # CPU helper: CV splitter interface (accepts cudf Series, returns numpy indices)
    nested_cv_splits = create_nested_cv_gpu(
        hour_ts,
        n_outer_folds=n_folds,
        n_inner_folds=3,
        val_window_hours=val_window_hours,
        gap_hours=1,
        return_device="cpu",
    )
    logger.info(f"  Nested folds: {len(nested_cv_splits)}")

    # ---- Step 4: Build actual_counts_df for hotspot metrics ----
    logger.info("\n[Step 4/7] Building actual_counts_df for hotspot metrics...")
    # CPU helper: hotspot metrics interface (small conversion cost acceptable)
    actual_counts_df_gpu = y_target[
        ["h3_cell", "hour_ts", "incident_count_t_plus_1"]
    ].copy()
    actual_counts_df_gpu["hour_ts"] = cudf.to_datetime(
        actual_counts_df_gpu["hour_ts"], utc=True
    )
    actual_counts_df_gpu["hour_actual"] = actual_counts_df_gpu[
        "hour_ts"
    ] + cudf.Timedelta(hours=1)
    actual_counts_df_gpu = actual_counts_df_gpu.rename(
        columns={"incident_count_t_plus_1": "incident_count"}
    )
    actual_counts_df_gpu = actual_counts_df_gpu[
        ["h3_cell", "hour_actual", "incident_count"]
    ]
    actual_counts_df = (
        actual_counts_df_gpu.to_pandas()
    )  # Convert to pandas for hotspot metrics function

    # ---- Step 5: Train per channel ----
    logger.info("\n[Step 5/7] Training models (GPU)...")

    all_results: List[Dict[str, Any]] = []

    channels = [
        ("base", base_X),
        ("rich", rich_X),
    ]

    # Remove PoissonRegressor; replace with XGBoost poisson objective handled inside train_xgboost_gpu
    models = [
        ("cuML_LinearRegression", train_cuml_linear_regression),
        ("cuML_RandomForest", train_cuml_random_forest),
        ("XGBoost_GPU", train_xgboost_gpu),
    ]

    for channel_name, X_cudf in channels:
        logger.info(f"\n  Channel: {channel_name}")

        # Keep numerics; impute for cuML models; keep original for xgb if you want NaNs
        X_num = _cudf_numeric_only(X_cudf)

        # IMPORTANT: keep h3/hour/day_of_week columns if you need baselines or hotspot grouping.
        # If those are non-numeric, ensure they were encoded upstream. Otherwise they will be dropped here.
        logger.info(f"  Numeric features: {len(X_num.columns)}")

        # Median impute for cuML safety
        X_num_imp = _gpu_median_impute(X_num)

        # Convert to CuPy matrices once per channel
        X_cp = _to_cupy_matrix(X_num_imp)
        y_cp = _to_cupy_y(y)

        # Baselines (GPU)
        for baseline_name in ["Persistence", "Climatology"]:
            logger.info(f"    Baseline: {baseline_name}")

            all_y_true = []
            all_y_pred = []
            all_h3 = []
            all_hour_ts = []

            # For baselines we don't need inner CV, but keep the protocol: fit on outer train, eval on outer test.
            for outer_train_idx, outer_test_idx, _inner in nested_cv_splits:
                # y_true on CPU for metric functions
                y_test = y.iloc[outer_test_idx].to_pandas().values.astype(np.float32)

                X_test = X_cudf.iloc[
                    outer_test_idx
                ]  # keep original columns for baseline features
                if baseline_name == "Persistence":
                    yhat_cp = _baseline_persistence_predict(X_test)
                else:
                    X_train = X_cudf.iloc[outer_train_idx]
                    y_train_cp = y.iloc[outer_train_idx].astype("float32").to_cupy()
                    mapping = _fit_climatology_mapping_gpu(X_train, y_train_cp)
                    fallback = float(y.iloc[outer_train_idx].mean())
                    yhat_cp = _predict_climatology_gpu(X_test, mapping, fallback)

                yhat = cp.maximum(yhat_cp, 0.0).get()  # CPU numpy
                all_y_true.append(y_test)
                all_y_pred.append(yhat)

                all_h3.append(h3_cells.iloc[outer_test_idx].to_pandas().values)
                all_hour_ts.append(hour_ts.iloc[outer_test_idx].to_pandas())

            y_true_np = np.concatenate(all_y_true)
            y_pred_np = np.concatenate(all_y_pred)
            h3_np = np.concatenate(all_h3)
            # CPU helper: small Series concatenation for hotspot metrics
            hour_ts_pred = (
                cudf.concat([cudf.Series(ts) for ts in all_hour_ts])
                .reset_index(drop=True)
                .to_pandas()
            )

            reg = compute_regression_metrics(y_true_np, y_pred_np)

            hour_ts_actual = hour_ts_pred + cudf.Timedelta(hours=1).to_pytimedelta()
            actual_hours = hour_ts_actual.dt.floor("h").unique()
            actual_counts = actual_counts_df[
                actual_counts_df["hour_actual"].isin(actual_hours)
            ].copy()

            hotspot = compute_hotspot_metrics(
                y_pred_np,
                actual_counts,
                h3_np,
                hour_ts_pred,
                hour_ts_actual,
                k=k_hotspots,
                h3_resolution=h3_resolution,
            )

            all_results.append(
                {
                    "model_name": baseline_name,
                    "channel": channel_name,
                    "rmse": reg["rmse"],
                    "mae": reg["mae"],
                    "r2": reg["r2"],
                    "smape": reg["smape"],
                    "mape_pos": reg["mape_pos"],
                    "hotspot_precision_at_k": hotspot["precision_at_k"],
                    "hotspot_recall_at_k": hotspot["recall_at_k"],
                    "hotspot_precision_at_k_conditional": hotspot.get(
                        "hotspot_precision_at_k_conditional", 0.0
                    ),
                    "hotspot_recall_at_k_conditional": hotspot.get(
                        "hotspot_recall_at_k_conditional", 0.0
                    ),
                    "staging_utility_coverage_pct": hotspot.get(
                        "staging_utility_coverage_pct", 0.0
                    ),
                    "staging_utility_with_neighbors_pct": hotspot.get(
                        "staging_utility_with_neighbors_pct", 0.0
                    ),
                    "train_time_sec": 0.0,
                    "inference_latency_p50_ms": 0.0,
                    "inference_latency_p95_ms": 0.0,
                    "rows_processed_per_sec": 0.0,
                    "top_5_features": f"{baseline_name}_baseline",
                    "champion": False,
                }
            )

        # Models (GPU)
        for model_name, train_func in models:
            logger.info(f"    Model: {model_name}")

            all_y_true = []
            all_y_pred = []
            all_h3 = []
            all_hour_ts = []
            total_train_time = 0.0

            for outer_fold_idx, (
                outer_train_idx,
                outer_test_idx,
                inner_cv_splits,
            ) in enumerate(nested_cv_splits):
                logger.info(
                    f"      Outer fold {outer_fold_idx+1}/{len(nested_cv_splits)}"
                )

                # Slice GPU matrices
                tr = cp.asarray(outer_train_idx, dtype=cp.int32)
                te = cp.asarray(outer_test_idx, dtype=cp.int32)

                X_train_outer = X_cp.take(tr, axis=0)
                y_train_outer = y_cp.take(tr, axis=0)

                # Inner CV indices are relative to outer_train_idx because you reset index in nested CV creation.
                # Here we need them as numpy arrays (CPU) but they index into X_train_outer.
                # We pass inner_cv_splits directly and let the trainer index into X_train_outer by CuPy take.

                if model_name == "cuML_LinearRegression":
                    model_bundle, best_params, cv_scores, train_time = train_func(
                        X_train_outer, y_train_outer, inner_cv_splits
                    )
                    model = model_bundle  # bundle has scaler+model
                elif model_name == "cuML_RandomForest":
                    model, best_params, cv_scores, train_time = train_func(
                        X_train_outer,
                        y_train_outer,
                        inner_cv_splits,
                        n_trials=n_trials_rf,
                    )
                elif model_name == "XGBoost_GPU":
                    model, best_params, cv_scores, train_time = train_func(
                        X_train_outer,
                        y_train_outer,
                        inner_cv_splits,
                        n_trials=n_trials_xgb,
                    )
                else:
                    raise ValueError(model_name)

                total_train_time += float(train_time)

                # Predict on outer test (GPU)
                X_test_outer = X_cp.take(te, axis=0)
                y_test_outer = (
                    y.iloc[outer_test_idx].to_pandas().values.astype(np.float32)
                )

                if model_name == "cuML_LinearRegression":
                    scaler = model["scaler"]
                    reg_model = model["model"]
                    if scaler is not None:
                        X_test_outer = scaler.transform(X_test_outer)
                    yhat_cp = reg_model.predict(X_test_outer)
                else:
                    yhat_cp = model.predict(X_test_outer)

                yhat = cp.maximum(cp.asarray(yhat_cp), 0.0).get()

                all_y_true.append(y_test_outer)
                all_y_pred.append(yhat)
                all_h3.append(h3_cells.iloc[outer_test_idx].to_pandas().values)
                all_hour_ts.append(hour_ts.iloc[outer_test_idx].to_pandas())

            train_time_avg = total_train_time / max(1, len(nested_cv_splits))

            y_true_np = np.concatenate(all_y_true)
            y_pred_np = np.concatenate(all_y_pred)
            h3_np = np.concatenate(all_h3)
            # CPU helper: small Series concatenation for hotspot metrics
            hour_ts_pred = (
                cudf.concat([cudf.Series(ts) for ts in all_hour_ts])
                .reset_index(drop=True)
                .to_pandas()
            )

            reg = compute_regression_metrics(y_true_np, y_pred_np)

            hour_ts_actual = hour_ts_pred + cudf.Timedelta(hours=1).to_pytimedelta()
            actual_hours = hour_ts_actual.dt.floor("h").unique()
            actual_counts = actual_counts_df[
                actual_counts_df["hour_actual"].isin(actual_hours)
            ].copy()

            hotspot = compute_hotspot_metrics(
                y_pred_np,
                actual_counts,
                h3_np,
                hour_ts_pred,
                hour_ts_actual,
                k=k_hotspots,
                h3_resolution=h3_resolution,
            )

            # GPU latency measurement on a batch
            # Use already-built X_cp for channel
            try:
                latency = _measure_inference_latency_gpu(
                    model["model"] if model_name == "cuML_LinearRegression" else model,
                    X_cp,
                    batch_size=min(200_000, X_cp.shape[0]),
                )
            except Exception:
                latency = {
                    "inference_latency_cold_start_ms": 0.0,
                    "inference_latency_p50_ms": 0.0,
                    "inference_latency_p95_ms": 0.0,
                }

            all_results.append(
                {
                    "model_name": model_name,
                    "channel": channel_name,
                    "rmse": reg["rmse"],
                    "mae": reg["mae"],
                    "r2": reg["r2"],
                    "smape": reg["smape"],
                    "mape_pos": reg["mape_pos"],
                    "hotspot_precision_at_k": hotspot["precision_at_k"],
                    "hotspot_recall_at_k": hotspot["recall_at_k"],
                    "hotspot_precision_at_k_conditional": hotspot.get(
                        "hotspot_precision_at_k_conditional", 0.0
                    ),
                    "hotspot_recall_at_k_conditional": hotspot.get(
                        "hotspot_recall_at_k_conditional", 0.0
                    ),
                    "staging_utility_coverage_pct": hotspot.get(
                        "staging_utility_coverage_pct", 0.0
                    ),
                    "staging_utility_with_neighbors_pct": hotspot.get(
                        "staging_utility_with_neighbors_pct", 0.0
                    ),
                    "train_time_sec": train_time_avg,
                    "inference_latency_p50_ms": latency.get(
                        "inference_latency_p50_ms", 0.0
                    ),
                    "inference_latency_p95_ms": latency.get(
                        "inference_latency_p95_ms", 0.0
                    ),
                    "rows_processed_per_sec": 0.0,
                    "top_5_features": "",  # see note below
                    "best_params": str(best_params),
                    "champion": False,
                }
            )

    # ---- Step 6: Champion selection ----
    logger.info("\n[Step 6/7] Selecting champions (per channel, min RMSE)...")
    results_df = cudf.DataFrame(all_results)
    results_df["champion"] = False

    for channel in ["base", "rich"]:
        ch = results_df[results_df["channel"] == channel]
        ch_models = ch[~ch["model_name"].isin(["Persistence", "Climatology"])]
        if len(ch_models) > 0:
            idx = ch_models["rmse"].idxmin()
            results_df.loc[idx, "champion"] = True

    # ---- Step 7: Save ----
    logger.info("\n[Step 7/7] Saving results...")
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gpu_training_results.csv"
    # CPU helper: CSV output (small conversion acceptable for final results)
    results_df.to_pandas().to_csv(out_file, index=False)

    total_time = time.time() - pipeline_start
    logger.info(f"\n✓ GPU training competition complete. Total time: {total_time:.2f}s")
    logger.info(f"  Results: {out_file}")

    return results_df


def _has_cuda_gpu() -> bool:
    try:
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False
