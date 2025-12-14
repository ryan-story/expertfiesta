"""
Run leakage validation tests on trained models (GPU-optimized)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.training.cv_splitter import create_rolling_origin_cv_gpu
from dgxb_gpu.training.leakage_validation import (
    single_feature_smoke_test,
    permutation_test,
    print_base_feature_list,
)

import numpy as np

# RAPIDS
import cudf
import cupy as cp
from cuml.preprocessing import LabelEncoder as cuLabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("LEAKAGE VALIDATION SUITE (RAPIDS/GPU)")
    logger.info("=" * 70)

    # Ensure results dir exists
    Path("results").mkdir(parents=True, exist_ok=True)

    # Load data (GPU)
    logger.info("\n[Step 1/4] Loading data (cuDF)...")
    base_X = cudf.read_parquet("gold-gpu-traffic/X_features.parquet")
    rich_X = cudf.read_parquet("rich-gold-gpu-traffic/X_features.parquet")
    y_target = cudf.read_parquet("gold-gpu-traffic/y_target.parquet")
    merged_df = cudf.read_parquet("gold-gpu-traffic/merged_intermediate.parquet")

    # Extract target (categorical)
    y_raw_categories = y_target["incident_category"]

    # Reconstruct timestamps (GPU-friendly: use cudf Series)
    # CPU helper: CV splitter interface (small conversion acceptable)
    if "incident_id" in merged_df.columns and "incident_id" in y_target.columns:
        # Build mapping on GPU then convert to pandas for CV splitter
        timestamp_map = merged_df[["incident_id", "timestamp"]].dropna()
        y_with_ts = y_target[["incident_id"]].merge(
            timestamp_map, on="incident_id", how="left"
        )
        timestamps = y_with_ts["timestamp"]
    else:
        # Fallback: if row-aligned
        if len(y_target) == len(merged_df):
            timestamps = merged_df["timestamp"]
        else:
            raise ValueError("Cannot align timestamps")

    # Create CV splits (GPU-friendly: accepts cudf Series)
    logger.info("\n[Step 2/4] Creating CV splits...")
    cv_splits = create_rolling_origin_cv_gpu(
        timestamps, n_folds=3, val_window_hours=24, return_device="cpu"
    )

    # Encode labels (GPU) using only train indices across folds
    logger.info("\n[Step 3/4] Encoding labels (cuML)...")
    all_train_indices = np.concatenate([train_idx for train_idx, _ in cv_splits])
    y_train_cats = y_raw_categories.iloc[all_train_indices]

    le = cuLabelEncoder()
    le.fit(y_train_cats)

    # Transform ALL labels
    y_encoded_gpu = le.transform(y_raw_categories).astype("int32")
    # Keep y on GPU as cupy (for XGBoost GPU path in leakage_validation)
    y_encoded = cp.asarray(y_encoded_gpu.values)

    # Prepare features (numeric only + median fill) on GPU
    logger.info("\n[Step 4/4] Preparing features (cuDF)...")
    base_X_numeric = base_X.select_dtypes(include=[np.number])
    rich_X_numeric = rich_X.select_dtypes(include=[np.number])

    # Median fill: cuDF supports median() and fillna()
    base_medians = base_X_numeric.median()
    base_X_numeric = base_X_numeric.fillna(base_medians)

    rich_medians = rich_X_numeric.median()
    rich_X_numeric = rich_X_numeric.fillna(rich_medians)

    # Test 1: Base feature audit (CPU helper: small conversion for metadata)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: BASE FEATURE AUDIT")
    logger.info("=" * 70)
    # CPU helper: print_base_feature_list expects pandas for dtype inspection
    base_X_pd = base_X_numeric.to_pandas()
    print_base_feature_list(base_X_pd, output_file="results/gpu_base_features_list.csv")

    # Test 2: Single-feature smoke test (base)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: SINGLE-FEATURE SMOKE TEST (BASE, GPU XGBOOST)")
    logger.info("=" * 70)
    smoke_test_results_base = single_feature_smoke_test(
        base_X_numeric,  # cuDF
        y_encoded,  # cupy
        cv_splits,
        feature_names=list(base_X_numeric.columns),
        top_n=20,
    )
    # CPU helper: CSV output (small conversion acceptable)
    smoke_test_results_base.to_pandas().to_csv(
        "results/gpu_single_feature_smoke_test_base.csv", index=False
    )

    # Test 3: Permutation test (base)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: PERMUTATION TEST (BASE, GPU XGBOOST)")
    logger.info("=" * 70)
    permutation_results_base = permutation_test(
        base_X_numeric, y_encoded, cv_splits, n_trials=3
    )

    # Test 4: Single-feature smoke test (rich - top 50 by variance)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: SINGLE-FEATURE SMOKE TEST (RICH - TOP 50, GPU)")
    logger.info("=" * 70)

    # cuDF variance; returns cudf Series
    feature_variances = rich_X_numeric.var().sort_values(ascending=False)
    top_rich_features = feature_variances.head(50).index.to_pandas().tolist()

    smoke_test_results_rich = single_feature_smoke_test(
        rich_X_numeric[top_rich_features],
        y_encoded,
        cv_splits,
        feature_names=top_rich_features,
        top_n=20,
    )
    # CPU helper: CSV output (small conversion acceptable)
    smoke_test_results_rich.to_pandas().to_csv(
        "results/gpu_single_feature_smoke_test_rich.csv", index=False
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("LEAKAGE VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\nBase Features:")
    logger.info(f"  Total features: {len(base_X_numeric.columns)}")
    logger.info(
        f"  Suspicious features (F1 > 0.95 alone): {int(smoke_test_results_base['is_suspicious'].sum())}"
    )
    logger.info(
        f"  Permutation test F1 drop: {permutation_results_base['f1_drop']:.4f}"
    )

    logger.info("\nRich Features (sample):")
    logger.info(f"  Tested features: {len(top_rich_features)}")
    logger.info(
        f"  Suspicious features (F1 > 0.95 alone): {int(smoke_test_results_rich['is_suspicious'].sum())}"
    )

    logger.info("\n✓ Validation complete! Results saved to results/ directory")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
