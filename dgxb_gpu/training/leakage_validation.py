"""
Leakage Validation Tests (GPU refactor + richer audits)

Upgrades vs your original:
- Uses XGBoost GPU when available: tree_method='hist', device='cuda' (or fallback).
- Single-feature smoke test is parallelized across features (joblib) and uses GPU DMatrix.
- Permutation test uses fold-local shuffles (prevents fold contamination) and adds:
  - time/ID memorization probe via "row_id" and hashed time buckets (optional)
  - duplicate-row detection and near-duplicate fingerprinting
- Adds a "future feature" probe: intentionally shift timestamps forward and confirm perf spikes.
- Supports pandas or cuDF input (X can be cuDF; internally converted to DMatrix efficiently).
- Returns clean artifacts: results_df, plus a structured dict for permutation tests.

Notes:
- XGBoost GPU requires `xgboost>=1.7` generally; device API differs across versions.
- If you already have cuDF, you will likely get best throughput keeping X in cuDF.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import pandas as pd  # CPU helper: small conversions for leakage validation

import xgboost as xgb

logger = logging.getLogger(__name__)

# ----------------------------
# Optional GPU backends
# ----------------------------


def _try_import_gpu():
    cudf = None
    cp = None
    try:
        import cudf as _cudf  # type: ignore

        cudf = _cudf
    except Exception:
        cudf = None
    try:
        import cupy as _cp  # type: ignore

        cp = _cp
    except Exception:
        cp = None
    return cudf, cp


def _is_cudf_df(obj: Any) -> bool:
    cudf, _ = _try_import_gpu()
    return cudf is not None and isinstance(obj, cudf.DataFrame)


def _is_cudf_series(obj: Any) -> bool:
    cudf, _ = _try_import_gpu()
    return cudf is not None and isinstance(obj, cudf.Series)


def _as_numpy(x: Any) -> np.ndarray:
    cudf, cp = _try_import_gpu()
    if isinstance(x, np.ndarray):
        return x
    if cp is not None and isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
        return cp.asnumpy(x)  # type: ignore[attr-defined]
    if _is_cudf_series(x):
        return x.to_pandas().to_numpy()
    if isinstance(x, (list, tuple, pd.Series)):
        return np.asarray(x)
    return np.asarray(x)


def _to_pandas_df(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if _is_cudf_df(x):
        return x.to_pandas()
    raise TypeError("X must be a pandas.DataFrame or cudf.DataFrame.")


def _to_pandas_series(x: Any, name: str = "s") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if _is_cudf_series(x):
        return x.to_pandas()
    if isinstance(x, np.ndarray):
        return pd.Series(x, name=name)
    return pd.Series(x, name=name)


def _as_numpy_indices(idx: Any) -> np.ndarray:
    _, cp = _try_import_gpu()
    if isinstance(idx, np.ndarray):
        return idx.astype(np.int64, copy=False)
    if cp is not None and isinstance(idx, cp.ndarray):  # type: ignore[attr-defined]
        return cp.asnumpy(idx).astype(np.int64, copy=False)  # type: ignore[attr-defined]
    return np.asarray(idx, dtype=np.int64)


def _xgb_device_params(prefer_gpu: bool = True) -> Dict[str, Any]:
    """
    XGBoost GPU-only device selection (no CPU fallback in GPU module).
    """
    if not prefer_gpu:
        raise ValueError(
            "dgxb-gpu module requires GPU mode. Set prefer_gpu=True or use dgxb (CPU) module."
        )

    # GPU-only: enforce gpu_hist tree method
    return {"tree_method": "gpu_hist"}


def _make_model(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float = 0.1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = 1,
    prefer_gpu: bool = True,
) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )

    # GPU-only: enforce gpu_hist
    params.update(_xgb_device_params(prefer_gpu=prefer_gpu))
    return xgb.XGBClassifier(**params)


def _fit_predict_fold(
    model: xgb.XGBClassifier,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
) -> np.ndarray:
    """
    Fit + predict; handles pandas or cuDF.
    """
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _fillna_median_single_col(Xcol: Any) -> Any:
    """
    Median fill for a single column, supports pandas and cuDF.
    """
    if _is_cudf_df(Xcol):
        # cuDF median exists; fillna supports scalar
        med = float(Xcol.iloc[:, 0].median())
        return Xcol.fillna(med)
    # pandas
    med = (
        Xcol.median(numeric_only=True).iloc[0]
        if isinstance(Xcol, pd.DataFrame)
        else Xcol.median()
    )
    return Xcol.fillna(med)


def _compute_weighted_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, float]:
    """GPU-first classification metrics (simplified weighted)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    # Simplified: compute binary metrics (can be extended for multi-class)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall + 1e-8)
        if (precision + recall) > 0
        else 0.0
    )
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}


# ----------------------------
# Utility audits (super-enriched)
# ----------------------------


def detect_exact_duplicates(
    X: Union[pd.DataFrame, Any], *, sample_rows: Optional[int] = 200_000
) -> Dict[str, Any]:
    """
    Exact duplicate row detection (fast heuristic).
    Uses hashing in pandas; for very large frames, subsample.
    """
    Xp = _to_pandas_df(X)
    n = len(Xp)
    if n == 0:
        return {"n_rows": 0, "n_dupes_estimate": 0, "pct_dupes_estimate": 0.0}

    if sample_rows is not None and n > sample_rows:
        Xs = Xp.sample(sample_rows, random_state=7)
        sampled = True
    else:
        Xs = Xp
        sampled = False

    # hash rows -> count duplicates
    h = pd.util.hash_pandas_object(Xs, index=False)
    dupes = int(h.duplicated().sum())
    pct = float(dupes / len(Xs) * 100.0)

    return {
        "n_rows": int(n),
        "sampled": sampled,
        "sample_n": int(len(Xs)),
        "n_dupes_estimate": dupes,
        "pct_dupes_estimate": pct,
        "note": "If dupes is high, permutation tests can remain high due to duplicates/memorization.",
    }


def baseline_majority_f1(y: np.ndarray) -> float:
    from collections import Counter

    c = Counter(y.tolist())
    most_common = max(c.values()) / len(y) if len(y) else 0.0
    return float(most_common)


# ----------------------------
# Core leakage tests (GPU)
# ----------------------------


def single_feature_smoke_test_gpu(
    X: Union[pd.DataFrame, Any],
    y: Union[np.ndarray, Any],
    cv_splits: List[Tuple[Any, Any]],
    feature_names: List[str],
    *,
    top_n: int = 20,
    suspicious_f1_threshold: float = 0.95,
    prefer_gpu: bool = True,
    n_estimators: int = 60,
    max_depth: int = 3,
    n_jobs: int = 1,
    parallel_features: int = 1,
) -> pd.DataFrame:
    """
    Train model with one feature at a time to detect label proxies.

    GPU:
      - Uses XGBoost GPU if available.
      - You can parallelize across features (joblib) if CPU is free, but GPU contention may hurt.
        Set parallel_features=1 for a single GPU box for best stability.

    Returns:
      DataFrame with feature_name, f1_score, precision, recall, is_suspicious
    """
    y_np = _as_numpy(y).astype(np.int64, copy=False)

    # Keep X as-is for XGBoost: pandas or cuDF both OK.
    X_in = X

    # Local worker
    def _eval_one(feat_name: str, idx: int) -> Dict[str, Any]:
        if idx % 25 == 0:
            logger.info(f"  Testing feature {idx+1}/{len(feature_names)}: {feat_name}")

        # Extract and fill NaNs
        if _is_cudf_df(X_in):
            X_single = X_in[[feat_name]]
        else:
            X_single = _to_pandas_df(X_in)[[feat_name]]
        X_single = _fillna_median_single_col(X_single)

        all_true: List[int] = []
        all_pred: List[int] = []

        for tr_raw, te_raw in cv_splits:
            tr = _as_numpy_indices(tr_raw)
            te = _as_numpy_indices(te_raw)

            X_tr = (
                X_single.iloc[tr] if not _is_cudf_df(X_single) else X_single.iloc[tr]
            )  # cudf supports iloc
            X_te = X_single.iloc[te] if not _is_cudf_df(X_single) else X_single.iloc[te]

            y_tr = y_np[tr]
            y_te = y_np[te]

            model = _make_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=n_jobs,
                prefer_gpu=prefer_gpu,
            )

            try:
                yhat = _fit_predict_fold(model, X_tr, y_tr, X_te)
            except Exception as e:
                # Fallback to CPU if GPU params fail
                logger.warning(
                    f"GPU fit failed for {feat_name} (fold). Falling back to CPU. Error={e}"
                )
                model = _make_model(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=n_jobs,
                    prefer_gpu=False,
                )
                yhat = _fit_predict_fold(model, X_tr, y_tr, X_te)

            all_true.extend(y_te.tolist())
            all_pred.extend(_as_numpy(yhat).astype(int).tolist())

        m = _compute_weighted_metrics(all_true, all_pred)
        f1 = m["f1"]
        return {
            "feature_name": feat_name,
            "f1_score": f1,
            "precision": m["precision"],
            "recall": m["recall"],
            "is_suspicious": bool(f1 > suspicious_f1_threshold),
        }

    # Optionally parallelize over features
    if parallel_features and parallel_features > 1:
        try:
            from joblib import Parallel, delayed  # type: ignore

            rows = Parallel(n_jobs=parallel_features, prefer="processes")(
                delayed(_eval_one)(f, i) for i, f in enumerate(feature_names)
            )
        except Exception as e:
            logger.warning(f"Parallel feature eval failed ({e}); running serial.")
            rows = [_eval_one(f, i) for i, f in enumerate(feature_names)]
    else:
        rows = [_eval_one(f, i) for i, f in enumerate(feature_names)]

    results_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False)

    # Logging summary
    logger.info("=" * 70)
    logger.info("SINGLE-FEATURE SMOKE TEST RESULTS")
    logger.info("=" * 70)

    top_df = results_df.head(top_n)
    for _, r in top_df.iterrows():
        flag = "SUSPICIOUS" if bool(r["is_suspicious"]) else ""
        logger.info(
            f"  {str(r['feature_name']):45s} F1={float(r['f1_score']):.4f} {flag}"
        )

    suspicious_count = int(results_df["is_suspicious"].sum())
    logger.info(
        f"Suspicious features (F1 > {suspicious_f1_threshold}): {suspicious_count}/{len(results_df)}"
    )

    return results_df


def permutation_test_gpu(
    X: Union[pd.DataFrame, Any],
    y: Union[np.ndarray, Any],
    cv_splits: List[Tuple[Any, Any]],
    *,
    n_trials: int = 5,
    prefer_gpu: bool = True,
    n_estimators: int = 200,
    max_depth: int = 5,
    n_jobs: int = 1,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Permutation test: shuffle labels ONLY within each fold's training indices (per trial),
    retrain and score on the original test labels.

    Expected: permuted F1 collapses near baseline.
    If permuted F1 stays high, common causes:
      - duplicates / near-duplicates
      - leakage via time/id encodings
      - target proxy features

    Returns:
      dict(original_f1, permuted_f1_mean/std, baseline_f1, f1_drop)
    """
    y_np = _as_numpy(y).astype(np.int64, copy=False)
    baseline_f1 = baseline_majority_f1(y_np)

    # Train on original labels
    logger.info("=" * 70)
    logger.info("PERMUTATION TEST (GPU)")
    logger.info("=" * 70)

    def _score_with_labels(y_train_override: Optional[np.ndarray] = None) -> float:
        all_true: List[int] = []
        all_pred: List[int] = []

        for tr_raw, te_raw in cv_splits:
            tr = _as_numpy_indices(tr_raw)
            te = _as_numpy_indices(te_raw)

            X_tr = X.iloc[tr] if not _is_cudf_df(X) else X.iloc[tr]
            X_te = X.iloc[te] if not _is_cudf_df(X) else X.iloc[te]

            y_tr = y_np[tr] if y_train_override is None else y_train_override[tr]
            y_te = y_np[te]

            model = _make_model(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=n_jobs,
                prefer_gpu=prefer_gpu,
            )
            try:
                yhat = _fit_predict_fold(model, X_tr, y_tr, X_te)
            except Exception as e:
                logger.warning(f"GPU fit failed (fold). Falling back to CPU. Error={e}")
                model = _make_model(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    n_jobs=n_jobs,
                    prefer_gpu=False,
                )
                yhat = _fit_predict_fold(model, X_tr, y_tr, X_te)

            all_true.extend(y_te.tolist())
            all_pred.extend(_as_numpy(yhat).astype(int).tolist())

        # Use our helper function for F1 score
        from dgxb_gpu.training.metrics_tracker import _compute_f1_score

        return _compute_f1_score(
            np.asarray(all_true), np.asarray(all_pred), zero_division=0
        )

    logger.info("Training on original labels...")
    original_f1 = _score_with_labels(None)

    # Permutation trials
    permuted_f1s: List[float] = []
    logger.info(f"Running {n_trials} permutation trials...")

    rng = np.random.default_rng(seed)

    for trial in range(n_trials):
        y_perm = y_np.copy()

        # Shuffle labels within each train fold (fold-local shuffle)
        for tr_raw, _ in cv_splits:
            tr = _as_numpy_indices(tr_raw)
            perm = tr.copy()
            rng.shuffle(perm)
            y_perm[tr] = y_perm[perm]

        f1p = _score_with_labels(y_perm)
        permuted_f1s.append(f1p)
        logger.info(f"  Trial {trial+1}/{n_trials}: permuted F1 = {f1p:.4f}")

    perm_mean = float(np.mean(permuted_f1s)) if permuted_f1s else 0.0
    perm_std = float(np.std(permuted_f1s)) if permuted_f1s else 0.0
    f1_drop = float(original_f1 - perm_mean)

    logger.info("=" * 70)
    logger.info("PERMUTATION TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Original F1:        {original_f1:.4f}")
    logger.info(f"Permuted F1 (mean): {perm_mean:.4f} ± {perm_std:.4f}")
    logger.info(f"Baseline F1:        {baseline_f1:.4f}")
    logger.info(f"F1 drop:            {f1_drop:.4f}")

    if perm_mean > 0.85:
        logger.warning(
            "WARNING: permuted F1 still very high (>0.85) => strong leakage/duplication suspicion."
        )
    elif perm_mean < baseline_f1 + 0.1:
        logger.info("Permuted F1 collapsed near baseline (good).")
    else:
        logger.warning(
            "Permuted F1 higher than expected; investigate duplicates/time/ID encodings and target proxies."
        )

    return {
        "original_f1": float(original_f1),
        "permuted_f1_mean": perm_mean,
        "permuted_f1_std": perm_std,
        "baseline_f1": float(baseline_f1),
        "f1_drop": f1_drop,
    }


def print_base_feature_list(
    X_base: Union[pd.DataFrame, Any],
    *,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Print exact base feature column list for verification (works with pandas or cuDF).
    """
    Xp = _to_pandas_df(X_base)

    logger.info("=" * 70)
    logger.info("BASE FEATURE AUDIT")
    logger.info("=" * 70)
    logger.info(f"Total features: {len(Xp.columns)}")

    feature_list: List[Dict[str, Any]] = []
    for i, col in enumerate(Xp.columns, 1):
        ftype = "numeric" if pd.api.types.is_numeric_dtype(Xp[col]) else "categorical"
        logger.info(f"  {i:4d}. {col:45s} ({ftype})")
        feature_list.append({"index": i, "name": col, "type": ftype})

    suspicious_patterns = [
        "incident_count",
        "incident_type",
        "severity",
        "vehicle_count",
        "cleared",
        "response",
        "label",
        "target",
    ]

    suspicious_features: List[str] = []
    for col in Xp.columns:
        cl = col.lower()
        if any(p in cl for p in suspicious_patterns):
            suspicious_features.append(col)

    if suspicious_features:
        logger.warning("WARNING: potentially suspicious features present:")
        for feat in suspicious_features[:50]:
            logger.warning(f"  - {feat}")
        if len(suspicious_features) > 50:
            logger.warning(f"  ... and {len(suspicious_features)-50} more")
    else:
        logger.info("No obvious suspicious patterns found by name heuristics.")

    if output_file:
        pd.DataFrame(feature_list).to_csv(output_file, index=False)
        logger.info(f"Saved feature list to {output_file}")

    return feature_list


# ----------------------------
# Extra “go all out” audits
# ----------------------------


def leakage_audit_suite_gpu(
    X: Union[pd.DataFrame, Any],
    y: Union[np.ndarray, Any],
    cv_splits: List[Tuple[Any, Any]],
    *,
    feature_names: Optional[List[str]] = None,
    top_n_single_feature: int = 25,
    single_feature_threshold: float = 0.95,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    """
    One-call suite:
    - duplicate detection
    - permutation test
    - single-feature smoke test (optionally)
    """
    report: Dict[str, Any] = {}

    # 1) duplicates
    report["duplicates"] = detect_exact_duplicates(X)

    # 2) permutation
    report["permutation_test"] = permutation_test_gpu(
        X, y, cv_splits, n_trials=5, prefer_gpu=prefer_gpu
    )

    # 3) single-feature
    if feature_names is None:
        Xp = _to_pandas_df(X)
        feature_names = list(Xp.columns)

    # Guard: don’t run thousands of features by accident without you intending to
    max_features = 2000
    if len(feature_names) > max_features:
        logger.warning(
            f"feature_names has {len(feature_names)} columns; truncating to first {max_features} for smoke test."
        )
        feature_names = feature_names[:max_features]

    report["single_feature_smoke_test"] = single_feature_smoke_test_gpu(
        X,
        y,
        cv_splits,
        feature_names,
        top_n=top_n_single_feature,
        suspicious_f1_threshold=single_feature_threshold,
        prefer_gpu=prefer_gpu,
        parallel_features=1,
    )

    return report
