"""
GPU Training Pipeline (XGBoost GPU + scikit-learn)

This version works on ARM64 (Grace Blackwell) without cuDF/cuML dependencies.
Uses pandas/numpy with GPU-accelerated XGBoost.

Model saving:
- Champion models are saved to {results_dir}/models/
- Feature columns saved to {results_dir}/models/feature_columns.json
- Use dgxb_gpu.inference to load and serve with real-time weather
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import List, Any, Dict, Tuple

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from .model_competition import (
    train_linear_regression_gpu,
    train_random_forest_gpu,
    train_xgboost_gpu,
)
from dgxb.training.metrics_tracker import (
    compute_hotspot_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Helper utilities
# ----------------------------

def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric + bool columns; cast bool -> int."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "bool":
            out[c] = out[c].astype("int8")
    return out.select_dtypes(include=["number"])


def _median_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs with per-column median."""
    return df.fillna(df.median())


def _rmse_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(np.sqrt(np.mean(err * err)))


def _measure_inference_latency(
    model: Any,
    X: np.ndarray,
    batch_size: int = 10000,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> Dict[str, float]:
    """Measure inference latency."""
    n = min(batch_size, X.shape[0])
    Xb = X[:n]

    # Cold start
    t0 = time.time()
    _ = model.predict(Xb)
    cold_ms = (time.time() - t0) * 1000.0

    # Warm runs
    times = []
    for _ in range(n_warmup + n_runs):
        t0 = time.time()
        _ = model.predict(Xb)
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
# Baselines
# ----------------------------

def _baseline_persistence_predict(X: pd.DataFrame) -> np.ndarray:
    """Persistence baseline: y_hat = incident_count_t if present else 0."""
    if "incident_count_t" in X.columns:
        return X["incident_count_t"].fillna(0).values.astype(np.float32)
    return np.zeros(len(X), dtype=np.float32)


def _fit_climatology_mapping(X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    """Build mapping: mean(y) per (h3_cell, hour_of_week)."""
    required = {"h3_cell", "hour", "day_of_week"}
    if not required.issubset(set(X_train.columns)):
        return pd.DataFrame({"h3_cell": [], "hour_of_week": [], "y_mean": []})

    tmp = X_train[["h3_cell", "hour", "day_of_week"]].copy()
    tmp["hour_of_week"] = tmp["day_of_week"] * 24 + tmp["hour"]
    tmp["y"] = y_train
    mapping = tmp.groupby(["h3_cell", "hour_of_week"])["y"].mean().reset_index()
    mapping = mapping.rename(columns={"y": "y_mean"})
    return mapping


def _predict_climatology(X_test: pd.DataFrame, mapping: pd.DataFrame, fallback: float) -> np.ndarray:
    required = {"h3_cell", "hour", "day_of_week"}
    if not required.issubset(set(X_test.columns)) or len(mapping) == 0:
        return np.full(len(X_test), fallback, dtype=np.float32)

    tmp = X_test[["h3_cell", "hour", "day_of_week"]].copy()
    tmp["hour_of_week"] = tmp["day_of_week"] * 24 + tmp["hour"]
    joined = tmp.merge(mapping, on=["h3_cell", "hour_of_week"], how="left")
    return joined["y_mean"].fillna(fallback).values.astype(np.float32)


# ----------------------------
# Main orchestrator
# ----------------------------

def run_training_competition_gpu(
    base_x_path: str = "gold-gpu-traffic/X_features.parquet",
    rich_x_path: str = "rich-gold-gpu-traffic/X_features.parquet",
    y_path: str = "gold-gpu-traffic/y_target.parquet",
    merged_intermediate_path: str = "gold-gpu-traffic/merged_intermediate.parquet",
    results_dir: str = "results",
    n_folds: int = 3,
    val_window_hours: int = 24,
    k_hotspots: int = 50,
    h3_resolution: int = 9,
    n_trials_rf: int = 20,
    n_trials_xgb: int = 20,
) -> pd.DataFrame:
    """
    Run GPU training competition pipeline.
    
    Uses XGBoost with GPU acceleration + scikit-learn for other models.
    """
    logger.info("=" * 70)
    logger.info("GPU TRAINING COMPETITION PIPELINE")
    logger.info("=" * 70)

    # Check GPU via XGBoost
    from .model_competition import _has_cuda_gpu
    if _has_cuda_gpu():
        logger.info("🚀 GPU detected - XGBoost will use CUDA")
    else:
        logger.info("⚠️ No GPU detected - using CPU")

    pipeline_start = time.time()

    # ---- Step 1: Load data ----
    logger.info("\n[Step 1/7] Loading parquet files...")
    ingest_t0 = time.time()

    base_X = pd.read_parquet(base_x_path)
    rich_X = pd.read_parquet(rich_x_path)
    y_target = pd.read_parquet(y_path)

    ingest_time = time.time() - ingest_t0
    logger.info(f"  Loaded base X: {len(base_X):,} rows, {len(base_X.columns)} cols")
    logger.info(f"  Loaded rich X: {len(rich_X):,} rows, {len(rich_X.columns)} cols")
    logger.info(f"  Loaded y:      {len(y_target):,} rows")
    logger.info(f"  Ingest time: {ingest_time:.2f}s")

    # ---- Step 2: Extract hour_ts / h3_cell / y ----
    logger.info("\n[Step 2/7] Extracting hour_ts, h3_cell, and target...")

    if "hour_ts" in base_X.columns:
        hour_ts = pd.to_datetime(base_X["hour_ts"], utc=True)
    elif "hour_ts" in y_target.columns:
        hour_ts = pd.to_datetime(y_target["hour_ts"], utc=True)
    else:
        raise ValueError("hour_ts must be present in X or y_target")

    if "h3_cell" in base_X.columns:
        h3_cells = base_X["h3_cell"].values
    elif "h3_cell" in y_target.columns:
        h3_cells = y_target["h3_cell"].values
    else:
        raise ValueError("h3_cell must be present in X or y_target")

    if "incident_count_t_plus_1" not in y_target.columns:
        raise ValueError("y_target must contain 'incident_count_t_plus_1'")

    y = y_target["incident_count_t_plus_1"].values

    # Drop NaN targets
    valid_mask = ~pd.isna(y)
    base_X = base_X[valid_mask].reset_index(drop=True)
    rich_X = rich_X[valid_mask].reset_index(drop=True)
    y_target = y_target[valid_mask].reset_index(drop=True)
    hour_ts = hour_ts[valid_mask].reset_index(drop=True)
    h3_cells = h3_cells[valid_mask]
    y = y[valid_mask]

    logger.info(f"  After dropping NaN targets: {len(y):,} rows")

    # ---- Step 3: Create nested CV splits ----
    logger.info("\n[Step 3/7] Creating nested time-blocked CV splits...")
    from dgxb.training.cv_splitter import create_nested_cv

    nested_cv_splits = create_nested_cv(
        hour_ts,
        n_outer_folds=n_folds,
        n_inner_folds=3,
        val_window_hours=val_window_hours,
        gap_hours=1,
    )
    logger.info(f"  Nested folds: {len(nested_cv_splits)}")

    # ---- Step 4: Build actual_counts_df for hotspot metrics ----
    logger.info("\n[Step 4/7] Building actual_counts_df for hotspot metrics...")
    actual_counts_df = y_target[["h3_cell", "hour_ts", "incident_count_t_plus_1"]].copy()
    actual_counts_df["hour_ts"] = pd.to_datetime(actual_counts_df["hour_ts"], utc=True)
    actual_counts_df["hour_actual"] = actual_counts_df["hour_ts"] + pd.Timedelta(hours=1)
    actual_counts_df = actual_counts_df.rename(columns={"incident_count_t_plus_1": "incident_count"})
    actual_counts_df = actual_counts_df[["h3_cell", "hour_actual", "incident_count"]]

    # ---- Step 5: Train per channel ----
    logger.info("\n[Step 5/7] Training models...")

    all_results: List[Dict[str, Any]] = []

    # Track trained models for saving
    trained_models: Dict[str, Dict[str, Any]] = {}

    channels = [
        ("base", base_X),
        ("rich", rich_X),
    ]

    models = [
        ("LinearRegression", train_linear_regression_gpu),
        ("RandomForest", train_random_forest_gpu),
        ("XGBoost_GPU", train_xgboost_gpu),
    ]

    for channel_name, X_df in channels:
        logger.info(f"\n  Channel: {channel_name}")
        trained_models[channel_name] = {}

        # Keep numerics and impute
        X_num = _numeric_only(X_df)
        X_num_imp = _median_impute(X_num)
        X_np = X_num_imp.values.astype(np.float32)

        logger.info(f"    Numeric features: {X_num.shape[1]}")

        # Baselines
        for baseline_name in ["Persistence", "Climatology"]:
            logger.info(f"    Baseline: {baseline_name}")

            all_y_true = []
            all_y_pred = []
            all_h3 = []
            all_hour_ts = []

            for outer_train_idx, outer_test_idx, _ in nested_cv_splits:
                y_test = y[outer_test_idx]
                X_test = X_df.iloc[outer_test_idx]

                if baseline_name == "Persistence":
                    yhat = _baseline_persistence_predict(X_test)
                else:
                    X_train = X_df.iloc[outer_train_idx]
                    y_train = y[outer_train_idx]
                    mapping = _fit_climatology_mapping(X_train, y_train)
                    fallback = float(np.mean(y_train))
                    yhat = _predict_climatology(X_test, mapping, fallback)

                yhat = np.maximum(yhat, 0.0)
                all_y_true.append(y_test)
                all_y_pred.append(yhat)
                all_h3.append(h3_cells[outer_test_idx])
                all_hour_ts.append(hour_ts.iloc[outer_test_idx])

            y_true_np = np.concatenate(all_y_true)
            y_pred_np = np.concatenate(all_y_pred)
            h3_np = np.concatenate(all_h3)
            hour_ts_pred = pd.concat(all_hour_ts).reset_index(drop=True)

            reg = compute_regression_metrics(y_true_np, y_pred_np)

            hour_ts_actual = hour_ts_pred + pd.Timedelta(hours=1)
            actual_hours = hour_ts_actual.dt.floor("h").unique()
            actual_counts = actual_counts_df[actual_counts_df["hour_actual"].isin(actual_hours)].copy()

            hotspot = compute_hotspot_metrics(
                y_pred_np, actual_counts, h3_np,
                hour_ts_pred, hour_ts_actual,
                k=k_hotspots,
            )

            all_results.append({
                "model_name": baseline_name,
                "channel": channel_name,
                "rmse": reg["rmse"],
                "mae": reg["mae"],
                "r2": reg["r2"],
                "smape": reg["smape"],
                "mape_pos": reg["mape_pos"],
                "hotspot_precision_at_k": hotspot["precision_at_k"],
                "hotspot_recall_at_k": hotspot["recall_at_k"],
                "train_time_sec": 0.0,
                "inference_latency_p50_ms": 0.0,
                "inference_latency_p95_ms": 0.0,
                "top_5_features": f"{baseline_name}_baseline",
                "champion": False,
            })

            logger.info(f"      RMSE: {reg['rmse']:.4f}, MAE: {reg['mae']:.4f}")

        # Models
        for model_name, train_func in models:
            logger.info(f"    Model: {model_name}")

            all_y_true = []
            all_y_pred = []
            all_h3 = []
            all_hour_ts = []
            total_train_time = 0.0
            best_params = {}

            for outer_fold_idx, (outer_train_idx, outer_test_idx, inner_cv_splits) in enumerate(nested_cv_splits):
                logger.info(f"      Outer fold {outer_fold_idx+1}/{len(nested_cv_splits)}")

                X_train_outer = X_np[outer_train_idx]
                y_train_outer = y[outer_train_idx]

                n_trials = n_trials_rf if "RandomForest" in model_name else n_trials_xgb

                if model_name == "LinearRegression":
                    model_bundle, best_params, cv_scores, train_time = train_func(
                        X_train_outer, y_train_outer, inner_cv_splits
                    )
                    model = model_bundle
                else:
                    model, best_params, cv_scores, train_time = train_func(
                        X_train_outer, y_train_outer, inner_cv_splits, n_trials=n_trials
                    )

                total_train_time += train_time

                # Predict on outer test
                X_test_outer = X_np[outer_test_idx]
                y_test_outer = y[outer_test_idx]

                if model_name == "LinearRegression":
                    scaler = model["scaler"]
                    reg_model = model["model"]
                    if scaler is not None:
                        X_test_outer = scaler.transform(X_test_outer)
                    yhat = reg_model.predict(X_test_outer)
                else:
                    yhat = model.predict(X_test_outer)

                yhat = np.maximum(yhat, 0.0)

                all_y_true.append(y_test_outer)
                all_y_pred.append(yhat)
                all_h3.append(h3_cells[outer_test_idx])
                all_hour_ts.append(hour_ts.iloc[outer_test_idx])

            train_time_avg = total_train_time / max(1, len(nested_cv_splits))

            y_true_np = np.concatenate(all_y_true)
            y_pred_np = np.concatenate(all_y_pred)
            h3_np = np.concatenate(all_h3)
            hour_ts_pred = pd.concat(all_hour_ts).reset_index(drop=True)

            reg = compute_regression_metrics(y_true_np, y_pred_np)

            hour_ts_actual = hour_ts_pred + pd.Timedelta(hours=1)
            actual_hours = hour_ts_actual.dt.floor("h").unique()
            actual_counts = actual_counts_df[actual_counts_df["hour_actual"].isin(actual_hours)].copy()

            hotspot = compute_hotspot_metrics(
                y_pred_np, actual_counts, h3_np,
                hour_ts_pred, hour_ts_actual,
                k=k_hotspots,
            )

            # Measure latency
            try:
                if model_name == "LinearRegression":
                    latency = _measure_inference_latency(model["model"], X_np)
                else:
                    latency = _measure_inference_latency(model, X_np)
            except Exception:
                latency = {"inference_latency_p50_ms": 0.0, "inference_latency_p95_ms": 0.0}

            all_results.append({
                "model_name": model_name,
                "channel": channel_name,
                "rmse": reg["rmse"],
                "mae": reg["mae"],
                "r2": reg["r2"],
                "smape": reg["smape"],
                "mape_pos": reg["mape_pos"],
                "hotspot_precision_at_k": hotspot["precision_at_k"],
                "hotspot_recall_at_k": hotspot["recall_at_k"],
                "train_time_sec": train_time_avg,
                "inference_latency_p50_ms": latency.get("inference_latency_p50_ms", 0.0),
                "inference_latency_p95_ms": latency.get("inference_latency_p95_ms", 0.0),
                "best_params": str(best_params),
                "top_5_features": "",
                "champion": False,
            })

            # Store trained model for potential saving
            trained_models[channel_name][model_name] = {
                "model": model,
                "feature_columns": list(X_num.columns),
                "best_params": best_params,
                "rmse": reg["rmse"],
            }

            logger.info(f"      RMSE: {reg['rmse']:.4f}, MAE: {reg['mae']:.4f}, Train: {train_time_avg:.1f}s")

    # ---- Step 6: Champion selection ----
    logger.info("\n[Step 6/7] Selecting champions (per channel, min RMSE)...")
    results_df = pd.DataFrame(all_results)
    results_df["champion"] = False

    for channel in ["base", "rich"]:
        ch = results_df[results_df["channel"] == channel]
        ch_models = ch[~ch["model_name"].isin(["Persistence", "Climatology"])]
        if len(ch_models) > 0:
            idx = ch_models["rmse"].idxmin()
            results_df.loc[idx, "champion"] = True
            champ = results_df.loc[idx]
            logger.info(f"  {channel} champion: {champ['model_name']} (RMSE: {champ['rmse']:.4f})")

    # ---- Step 7: Save results and models ----
    logger.info("\n[Step 7/7] Saving results and models...")
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gpu_training_results.csv"
    results_df.to_csv(out_file, index=False)

    # Save champion models for deployment
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_models = {}
    for channel in ["base", "rich"]:
        champions = results_df[
            (results_df["champion"]) & (results_df["channel"] == channel)
        ]
        if len(champions) > 0:
            champ_name = champions.iloc[0]["model_name"]
            if channel in trained_models and champ_name in trained_models[channel]:
                model_info = trained_models[channel][champ_name]

                # Save model
                if HAS_JOBLIB:
                    model_path = models_dir / f"champion_{channel}.joblib"
                    joblib.dump(model_info["model"], model_path)
                    logger.info(f"  Saved {channel} champion model: {model_path}")

                    saved_models[channel] = {
                        "model_name": champ_name,
                        "model_path": str(model_path),
                        "feature_columns": model_info["feature_columns"],
                        "best_params": model_info["best_params"],
                        "rmse": model_info["rmse"],
                    }
                else:
                    logger.warning("joblib not available, skipping model save")

    # Save model metadata (feature columns, etc.)
    if saved_models:
        metadata = {
            "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "models": saved_models,
            "weather_features": [
                col for col in saved_models.get("rich", saved_models.get("base", {}))
                .get("feature_columns", [])
                if "weather" in col.lower()
            ],
            "usage": {
                "load_model": "joblib.load('results/models/champion_rich.joblib')",
                "inference_with_weather": "from dgxb_gpu.inference import prepare_inference_features, batch_inference_with_weather",
            }
        }
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"  Saved model metadata: {metadata_path}")

    total_time = time.time() - pipeline_start
    logger.info(f"\n✓ GPU training competition complete. Total time: {total_time:.2f}s")
    logger.info(f"  Results: {out_file}")
    if saved_models:
        logger.info(f"  Models saved to: {models_dir}")
        logger.info("  Use dgxb_gpu.inference for deployment with real-time weather")

    return results_df
