"""
Run leakage validation tests on trained models
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb.training.pipeline import run_training_competition
from dgxb.training.cv_splitter import create_rolling_origin_cv
from dgxb.training.leakage_validation import (
    single_feature_smoke_test,
    permutation_test,
    print_base_feature_list,
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run all leakage validation tests"""
    logger.info("=" * 70)
    logger.info("LEAKAGE VALIDATION SUITE")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n[Step 1/4] Loading data...")
    base_X = pd.read_parquet("gold-cpu-traffic/X_features.parquet")
    rich_X = pd.read_parquet("rich-gold-cpu-traffic/X_features.parquet")
    y_target = pd.read_parquet("gold-cpu-traffic/y_target.parquet")
    merged_df = pd.read_parquet("gold-cpu-traffic/merged_intermediate.parquet")
    
    # Reset indices
    base_X = base_X.reset_index(drop=True)
    rich_X = rich_X.reset_index(drop=True)
    y_target = y_target.reset_index(drop=True)
    merged_df = merged_df.reset_index(drop=True)
    
    # Extract target
    y_raw_categories = y_target["incident_category"].values
    
    # Reconstruct timestamps
    if "incident_id" in merged_df.columns and "incident_id" in y_target.columns:
        timestamp_map = merged_df.set_index("incident_id")["timestamp"].to_dict()
        timestamps = y_target["incident_id"].map(timestamp_map)
    else:
        if len(y_target) == len(merged_df):
            timestamps = merged_df["timestamp"].values
        else:
            raise ValueError("Cannot align timestamps")
    
    timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
    
    # Create CV splits
    logger.info("\n[Step 2/4] Creating CV splits...")
    cv_splits = create_rolling_origin_cv(timestamps, n_folds=3, val_window_hours=24)
    
    # Encode labels
    all_train_indices = np.concatenate([train_idx for train_idx, _ in cv_splits])
    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw_categories[all_train_indices])
    y_encoded = label_encoder.transform(y_raw_categories)
    
    # Prepare features (numeric only, fill NaN)
    base_X_numeric = base_X.select_dtypes(include=["number"]).copy()
    base_X_numeric = base_X_numeric.fillna(base_X_numeric.median())
    
    rich_X_numeric = rich_X.select_dtypes(include=["number"]).copy()
    rich_X_numeric = rich_X_numeric.fillna(rich_X_numeric.median())
    
    # Test 1: Base feature audit
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: BASE FEATURE AUDIT")
    logger.info("=" * 70)
    feature_list = print_base_feature_list(
        base_X_numeric, output_file="results/base_features_list.csv"
    )
    
    # Test 2: Single-feature smoke test (base features)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: SINGLE-FEATURE SMOKE TEST (BASE)")
    logger.info("=" * 70)
    smoke_test_results_base = single_feature_smoke_test(
        base_X_numeric,
        y_encoded,
        cv_splits,
        feature_names=list(base_X_numeric.columns),
        top_n=20,
    )
    smoke_test_results_base.to_csv(
        "results/single_feature_smoke_test_base.csv", index=False
    )
    
    # Test 3: Permutation test (base features)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: PERMUTATION TEST (BASE)")
    logger.info("=" * 70)
    permutation_results_base = permutation_test(
        base_X_numeric, y_encoded, cv_splits, n_trials=3
    )
    
    # Test 4: Single-feature smoke test (rich features - top 50 only for speed)
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: SINGLE-FEATURE SMOKE TEST (RICH - TOP 50)")
    logger.info("=" * 70)
    logger.info("Testing top 50 rich features (by variance) for speed...")
    # Select top features by variance
    feature_variances = rich_X_numeric.var().sort_values(ascending=False)
    top_rich_features = feature_variances.head(50).index.tolist()
    
    smoke_test_results_rich = single_feature_smoke_test(
        rich_X_numeric[top_rich_features],
        y_encoded,
        cv_splits,
        feature_names=top_rich_features,
        top_n=20,
    )
    smoke_test_results_rich.to_csv(
        "results/single_feature_smoke_test_rich.csv", index=False
    )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("LEAKAGE VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\nBase Features:")
    logger.info(f"  Total features: {len(base_X_numeric.columns)}")
    logger.info(
        f"  Suspicious features (F1 > 0.95 alone): {smoke_test_results_base['is_suspicious'].sum()}"
    )
    logger.info(f"  Permutation test F1 drop: {permutation_results_base['f1_drop']:.4f}")
    
    logger.info("\nRich Features (sample):")
    logger.info(f"  Tested features: {len(top_rich_features)}")
    logger.info(
        f"  Suspicious features (F1 > 0.95 alone): {smoke_test_results_rich['is_suspicious'].sum()}"
    )
    
    logger.info("\n✓ Validation complete! Results saved to results/ directory")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

