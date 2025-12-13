"""
Leakage validation tests for training pipeline
Implements mechanistic audits to detect remaining data leakage
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def single_feature_smoke_test(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Train model with one feature at a time to detect label proxies

    If any single feature yields F1 > 0.95, that feature is likely a label proxy or leaked.

    Args:
        X: Feature matrix
        y: Target labels
        cv_splits: Cross-validation splits
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with columns: feature_name, f1_score, precision, recall, is_suspicious
    """
    logger.info("=" * 70)
    logger.info("SINGLE-FEATURE SMOKE TEST")
    logger.info("=" * 70)
    logger.info("Training XGBoost with one feature at a time...")
    logger.info(f"Testing {len(feature_names)} features")

    results = []

    for idx, feat_name in enumerate(feature_names):
        if idx % 10 == 0:
            logger.info(f"  Testing feature {idx+1}/{len(feature_names)}: {feat_name}")

        # Extract single feature
        X_single = X[[feat_name]].copy()

        # Fill NaN with median
        X_single = X_single.fillna(X_single.median())

        # Train XGBoost with single feature
        model = xgb.XGBClassifier(
            tree_method="hist",
            random_state=42,
            n_estimators=50,
            max_depth=3,
            n_jobs=1,
        )

        # Evaluate on all test folds
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in cv_splits:
            X_train_fold = X_single.iloc[train_idx]
            X_test_fold = X_single.iloc[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)

            all_y_true.extend(y_test_fold)
            all_y_pred.extend(y_pred)

        # Compute metrics
        f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)
        precision = precision_score(
            all_y_true, all_y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            all_y_true, all_y_pred, average="weighted", zero_division=0
        )

        is_suspicious = f1 > 0.95

        results.append(
            {
                "feature_name": feat_name,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "is_suspicious": is_suspicious,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_score", ascending=False)

    logger.info("\n" + "=" * 70)
    logger.info("SINGLE-FEATURE SMOKE TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nTop {top_n} features by F1 score:")
    top_features = results_df.head(top_n)
    for _, row in top_features.iterrows():
        flag = "⚠️  SUSPICIOUS" if row["is_suspicious"] else ""
        logger.info(f"  {row['feature_name']:40s} F1={row['f1_score']:.4f} {flag}")

    suspicious_count = results_df["is_suspicious"].sum()
    logger.info(
        f"\n⚠️  Suspicious features (F1 > 0.95): {suspicious_count}/{len(results_df)}"
    )

    if suspicious_count > 0:
        logger.warning("\n⚠️  WARNING: Found features that achieve >0.95 F1 alone!")
        logger.warning("   These are likely label proxies or contain leakage.")
        logger.warning("   Review these features carefully:")
        for _, row in results_df[results_df["is_suspicious"]].iterrows():
            logger.warning(f"     - {row['feature_name']}")

    return results_df


def permutation_test(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 3,
) -> Dict[str, float]:
    """
    Permutation test: randomly shuffle labels within each fold's training window

    Expected: F1 collapses to near-baseline (~0.5-0.7 for binary, or class frequency baseline)
    If F1 stays high, features encode identity/time in a way that reproduces labels (leakage/duplication).

    Args:
        X: Feature matrix
        y: Target labels
        cv_splits: Cross-validation splits
        n_trials: Number of permutation trials

    Returns:
        Dictionary with original_f1, permuted_f1_mean, permuted_f1_std, baseline_f1
    """
    logger.info("=" * 70)
    logger.info("PERMUTATION TEST")
    logger.info("=" * 70)
    logger.info("Shuffling labels within training folds and retraining...")

    # Compute baseline (class frequency)
    from collections import Counter

    class_counts = Counter(y)
    most_common_pct = max(class_counts.values()) / len(y)
    baseline_f1 = most_common_pct  # Baseline = always predict most common class

    # Train on original labels
    logger.info("Training on original labels...")
    model = xgb.XGBClassifier(
        tree_method="hist",
        random_state=42,
        n_estimators=100,
        max_depth=5,
        n_jobs=1,
    )

    all_y_true_orig = []
    all_y_pred_orig = []

    for train_idx, test_idx in cv_splits:
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        all_y_true_orig.extend(y_test_fold)
        all_y_pred_orig.extend(y_pred)

    original_f1 = f1_score(
        all_y_true_orig, all_y_pred_orig, average="weighted", zero_division=0
    )

    # Permutation trials
    logger.info(f"Running {n_trials} permutation trials...")
    permuted_f1s = []

    for trial in range(n_trials):
        # Create permuted labels (shuffle within each training fold)
        y_permuted = y.copy()
        np.random.seed(42 + trial)

        for train_idx, _ in cv_splits:
            # Shuffle labels within this training fold
            permuted_train_labels = y[train_idx].copy()
            np.random.shuffle(permuted_train_labels)
            y_permuted[train_idx] = permuted_train_labels

        # Train on permuted labels
        model_perm = xgb.XGBClassifier(
            tree_method="hist",
            random_state=42,
            n_estimators=100,
            max_depth=5,
            n_jobs=1,
        )

        all_y_true_perm = []
        all_y_pred_perm = []

        for train_idx, test_idx in cv_splits:
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold_perm = y_permuted[train_idx]
            y_test_fold = y[test_idx]

            model_perm.fit(X_train_fold, y_train_fold_perm)
            y_pred = model_perm.predict(X_test_fold)

            all_y_true_perm.extend(y_test_fold)
            all_y_pred_perm.extend(y_pred)

        permuted_f1 = f1_score(
            all_y_true_perm, all_y_pred_perm, average="weighted", zero_division=0
        )
        permuted_f1s.append(permuted_f1)
        logger.info(f"  Trial {trial+1}: F1 = {permuted_f1:.4f}")

    permuted_f1_mean = np.mean(permuted_f1s)
    permuted_f1_std = np.std(permuted_f1s)

    logger.info("\n" + "=" * 70)
    logger.info("PERMUTATION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Original F1:        {original_f1:.4f}")
    logger.info(f"Permuted F1 (mean): {permuted_f1_mean:.4f} ± {permuted_f1_std:.4f}")
    logger.info(f"Baseline F1:        {baseline_f1:.4f}")

    f1_drop = original_f1 - permuted_f1_mean
    logger.info(f"F1 drop:            {f1_drop:.4f}")

    if permuted_f1_mean > 0.85:
        logger.warning("\n⚠️  WARNING: Permuted F1 is still very high (>0.85)!")
        logger.warning("   This suggests features encode identity/time in a way that")
        logger.warning("   reproduces labels even when labels are shuffled.")
        logger.warning("   Possible causes: near-duplicate rows, temporal leakage, or")
        logger.warning("   features that directly encode the target.")
    elif permuted_f1_mean < baseline_f1 + 0.1:
        logger.info(
            "\n✓ Permuted F1 collapsed to near-baseline - this is expected and good."
        )
    else:
        logger.warning(
            f"\n⚠️  Permuted F1 ({permuted_f1_mean:.4f}) is higher than expected."
        )
        logger.warning(
            "   Expected near baseline ({baseline_f1:.4f}), got {permuted_f1_mean:.4f}"
        )

    return {
        "original_f1": original_f1,
        "permuted_f1_mean": permuted_f1_mean,
        "permuted_f1_std": permuted_f1_std,
        "baseline_f1": baseline_f1,
        "f1_drop": f1_drop,
    }


def print_base_feature_list(X_base: pd.DataFrame, output_file: str = None) -> None:
    """
    Print exact base feature column list for verification

    Args:
        X_base: Base feature DataFrame
        output_file: Optional file to save feature list to
    """
    logger.info("=" * 70)
    logger.info("BASE FEATURE AUDIT")
    logger.info("=" * 70)
    logger.info(f"\nTotal features: {len(X_base.columns)}")
    logger.info("\nBase feature list:")

    feature_list = []
    for i, col in enumerate(X_base.columns, 1):
        feature_type = (
            "numeric" if pd.api.types.is_numeric_dtype(X_base[col]) else "categorical"
        )
        logger.info(f"  {i:2d}. {col:40s} ({feature_type})")
        feature_list.append({"index": i, "name": col, "type": feature_type})

    # Check for suspicious patterns
    suspicious_patterns = [
        "incident_count",
        "incident_type",
        "severity",
        "vehicle_count",
        "cleared",
        "response",
    ]

    suspicious_features = []
    for col in X_base.columns:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                suspicious_features.append(col)
                break

    if suspicious_features:
        logger.warning("\n⚠️  WARNING: Found potentially suspicious features:")
        for feat in suspicious_features:
            logger.warning(f"     - {feat}")
        logger.warning(
            "\n   These features may be direct proxies for the target label."
        )
        logger.warning("   Review whether these should be in 'base' features.")
    else:
        logger.info("\n✓ No obviously suspicious features found in base set.")

    # Save to file if requested
    if output_file:
        df_features = pd.DataFrame(feature_list)
        df_features.to_csv(output_file, index=False)
        logger.info(f"\n✓ Feature list saved to {output_file}")

    return feature_list
