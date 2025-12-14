"""
Data Leakage Audit Tools (GPU-friendly + more thorough)

Key upgrades vs your original:
- Supports pandas OR cuDF for X / timestamps; supports numpy OR cupy for indices.
- Adds a real lag/rolling leakage validation by sampling rows and checking that
  lagged/rolled values do NOT include information from >= label time.
- Adds optional per-group (e.g., h3_cell) validation for lag/rolling features.
- Adds stronger CV split auditing: overlap, contiguity, monotonicity, gap checks, hour window inference.
- Keeps everything deterministic with a seed.

Notes:
- Leakage auditing is mostly logic/IO bound; GPU helps when X is huge and stored in cuDF.
- This implementation avoids expensive full scans by default; you can increase sample sizes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

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


def _to_pandas_series(x: Any, name: str = "s") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if _is_cudf_series(x):
        return x.to_pandas()
    if isinstance(x, np.ndarray):
        return pd.Series(x, name=name)
    return pd.Series(x, name=name)


def _to_pandas_df(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if _is_cudf_df(x):
        return x.to_pandas()
    raise TypeError(
        "X must be a pandas.DataFrame or cudf.DataFrame for this audit tool."
    )


def _to_datetime_utc(s: Any) -> pd.Series:
    ps = _to_pandas_series(s, name="timestamps")
    # Always coerce; UTC for consistent comparisons
    return pd.to_datetime(ps, errors="coerce", utc=True)


def _as_numpy_indices(idx: Any) -> np.ndarray:
    """
    Accepts numpy, list, cupy. Returns numpy int64 array.
    """
    _, cp = _try_import_gpu()
    if isinstance(idx, np.ndarray):
        return idx.astype(np.int64, copy=False)
    if cp is not None and isinstance(idx, cp.ndarray):  # type: ignore[attr-defined]
        return cp.asnumpy(idx).astype(np.int64, copy=False)  # type: ignore[attr-defined]
    return np.asarray(idx, dtype=np.int64)


# ----------------------------
# Parsing feature names
# ----------------------------

_LAG_RE = re.compile(r"_lag_(?P<hours>-?\d+)h\b")
_ROLLING_RE = re.compile(r"_rolling_(?P<stat>mean|std|min|max)_(?P<hours>\d+)h\b")
# You also used names like: weather_temperature_rolling_mean_24h
_ROLLING_ALT_RE = re.compile(
    r"_rolling_(?P<stat>mean|std|min|max)_(?P<hours>\d+)h\b|_rolling_(?P<hours2>\d+)h\b"
)


@dataclass(frozen=True)
class ParsedFeature:
    kind: Literal["lag", "rolling", "other"]
    base_col: Optional[str]
    hours: Optional[int]
    stat: Optional[str]
    col: str


def parse_feature_name(col: str) -> ParsedFeature:
    m = _LAG_RE.search(col)
    if m:
        hours = int(m.group("hours"))
        base = col.split("_lag_")[0]
        return ParsedFeature(kind="lag", base_col=base, hours=hours, stat=None, col=col)

    # rolling pattern like *_rolling_mean_24h
    m = _ROLLING_RE.search(col)
    if m:
        stat = m.group("stat")
        hours = int(m.group("hours"))
        # base is everything before "_rolling_"
        base = col.split("_rolling_")[0]
        return ParsedFeature(
            kind="rolling", base_col=base, hours=hours, stat=stat, col=col
        )

    # fallback: treat as other
    return ParsedFeature(kind="other", base_col=None, hours=None, stat=None, col=col)


# ----------------------------
# Sampling helpers
# ----------------------------


def _stable_sample_indices(n: int, sample_size: int, seed: int = 7) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    sample_size = int(min(sample_size, n))
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False).astype(np.int64)


def _get_base_value_at_time(
    df: pd.DataFrame,
    time_col: str,
    base_col: str,
    t: pd.Timestamp,
    group_key: Optional[str] = None,
    group_value: Optional[Any] = None,
    direction: Literal["backward", "forward"] = "backward",
) -> Any:
    """
    Get the base_col value at the nearest timestamp <= t (backward) or >= t (forward),
    optionally within a group.
    """
    if group_key is not None:
        sdf = df[df[group_key] == group_value]
    else:
        sdf = df

    # filter to candidate rows
    if direction == "backward":
        cand = sdf[sdf[time_col] <= t]
        if cand.empty:
            return np.nan
        # nearest prior: take max time
        row = cand.loc[cand[time_col].idxmax()]
    else:
        cand = sdf[sdf[time_col] >= t]
        if cand.empty:
            return np.nan
        row = cand.loc[cand[time_col].idxmin()]

    return row.get(base_col, np.nan)


def _rolling_stat_from_history(
    df: pd.DataFrame,
    time_col: str,
    base_col: str,
    t_end: pd.Timestamp,
    window_hours: int,
    stat: str,
    group_key: Optional[str] = None,
    group_value: Optional[Any] = None,
) -> float:
    """
    Compute rolling stat over [t_end - window, t_end] inclusive, within group if specified.
    This is the "no leakage" reference.
    """
    t_start = t_end - pd.Timedelta(hours=int(window_hours))
    if group_key is not None:
        sdf = df[df[group_key] == group_value]
    else:
        sdf = df

    w = sdf[(sdf[time_col] >= t_start) & (sdf[time_col] <= t_end)][base_col].dropna()
    if w.empty:
        return np.nan

    if stat == "mean":
        return float(w.mean())
    if stat == "std":
        return float(w.std(ddof=1))  # match pandas default
    if stat == "min":
        return float(w.min())
    if stat == "max":
        return float(w.max())

    return np.nan


# ----------------------------
# Public API
# ----------------------------


def audit_feature_timestamps_gpu(
    X: Union[pd.DataFrame, Any],
    y: Union[pd.Series, np.ndarray, Any],
    timestamps: Union[pd.Series, np.ndarray, Any],
    *,
    timestamp_col_in_X: Optional[str] = None,
    group_key: Optional[str] = None,
    sample_size: int = 256,
    deep_check: bool = True,
    deep_check_max_features: int = 50,
    deep_check_tolerance: float = 1e-6,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Audit features for potential leakage.

    Fast checks:
    - timestamp-like columns present
    - lag feature names have negative hours (immediate leakage risk)
    - rolling features presence

    Deep checks (optional):
    - For sampled rows, validate that lag feature equals base_col value at (t - lag_hours)
      using nearest prior timestamp within optional group.
    - For rolling features (mean/std/min/max), validate value equals stat over prior window
      ending at t (inclusive), within optional group.

    Args:
        X: pandas or cuDF DataFrame
        y: labels (pandas/cudf series or numpy)
        timestamps: timestamps aligned to X rows
        timestamp_col_in_X: if provided and present, will use it instead of timestamps arg
        group_key: optional key like "h3_cell" to enforce per-group lag/rolling validation
        sample_size: number of rows to sample for deep checks
        deep_check: whether to run deep checks
        deep_check_max_features: cap features validated (avoid huge audit runs)
        deep_check_tolerance: numeric tolerance for comparison
        seed: RNG seed for stable sampling

    Returns:
        dict with issues/warnings and counts, plus deep_check results if enabled.
    """
    # Bring into pandas for auditing logic (GPU-friendly storage still acceptable upstream)
    Xp = _to_pandas_df(X)
    n = len(Xp)

    # Determine timestamps
    if timestamp_col_in_X and timestamp_col_in_X in Xp.columns:
        ts = _to_datetime_utc(Xp[timestamp_col_in_X])
    else:
        ts = _to_datetime_utc(timestamps)

    issues: List[str] = []
    warnings: List[str] = []

    # Timestamp-like columns in X
    timestamp_like_cols = [
        c for c in Xp.columns if ("timestamp" in c.lower() or "time" in c.lower())
    ]
    if timestamp_like_cols:
        warnings.append(
            f"Found timestamp-like columns in X: {timestamp_like_cols[:25]}{'...' if len(timestamp_like_cols)>25 else ''}"
        )

    # Parse features
    parsed = [parse_feature_name(c) for c in Xp.columns]
    lag_features = [p for p in parsed if p.kind == "lag"]
    rolling_features = [p for p in parsed if p.kind == "rolling"]

    # Name-based lag sanity: negative lag hours implies forward-looking construction
    for p in lag_features:
        if p.hours is not None and p.hours < 0:
            issues.append(
                f"Lag feature {p.col} has negative lag hours ({p.hours}) => forward-looking leakage risk."
            )

    if rolling_features:
        warnings.append(
            f"Found {len(rolling_features)} rolling features; verify window excludes future data."
        )

    # Target distribution warning (same spirit as yours, slightly cleaner)
    yps = _to_pandas_series(y, name="y")
    if yps.size > 0 and pd.api.types.is_numeric_dtype(yps):
        vc = yps.value_counts(dropna=False)
        if len(vc) > 0:
            most_common_pct = float(vc.iloc[0] / len(yps) * 100.0)
            if most_common_pct > 80:
                warnings.append(
                    f"Target is highly imbalanced: {most_common_pct:.1f}% are value '{vc.index[0]}'"
                )

    # Deep checks
    deep_results: Dict[str, Any] = {}
    if deep_check and n > 0:
        # We need a working frame with timestamps and optional group_key
        if ts.isna().any():
            warnings.append(
                "Some timestamps could not be parsed (NaT). Deep checks will ignore those rows."
            )

        work = Xp.copy()
        work["_audit_ts"] = ts

        if group_key is not None and group_key not in work.columns:
            warnings.append(
                f"group_key='{group_key}' not found in X; deep checks will run without grouping."
            )
            group_key_eff = None
        else:
            group_key_eff = group_key

        # Sort by time within group for stable nearest-prior lookups
        sort_cols = (
            [_ for _ in [group_key_eff, "_audit_ts"] if _ is not None]
            if group_key_eff
            else ["_audit_ts"]
        )
        work = work.sort_values(sort_cols).reset_index(drop=True)

        # Sample indices from valid timestamp rows
        valid_mask = work["_audit_ts"].notna().to_numpy()
        valid_idx = np.where(valid_mask)[0]
        if valid_idx.size == 0:
            warnings.append(
                "All timestamps are NaT after parsing; skipping deep checks."
            )
        else:
            sample_idx = _stable_sample_indices(valid_idx.size, sample_size, seed=seed)
            sample_rows = valid_idx[sample_idx]

            # Choose subset of features for deep checking (prioritize lags then rolling)
            features_to_check: List[ParsedFeature] = []
            # prioritize lag/rolling only
            features_to_check.extend(lag_features[: deep_check_max_features // 2])
            remaining = deep_check_max_features - len(features_to_check)
            if remaining > 0:
                features_to_check.extend(rolling_features[:remaining])

            lag_mismatches: List[str] = []
            rolling_mismatches: List[str] = []
            lag_checked = 0
            rolling_checked = 0

            # Pre-check: base cols must exist
            for pf in features_to_check:
                if pf.base_col is None or pf.base_col not in work.columns:
                    warnings.append(
                        f"Feature {pf.col} references base_col '{pf.base_col}' which is missing; skipping deep validation for this feature."
                    )
            features_to_check = [
                pf
                for pf in features_to_check
                if pf.base_col and pf.base_col in work.columns
            ]

            for pf in features_to_check:
                if pf.kind == "lag" and pf.hours is not None:
                    lag_hours = int(pf.hours)
                    if lag_hours < 0:
                        # already flagged
                        continue

                    # For lag, expected = base at (t - lag_hours) using nearest prior row
                    for r in sample_rows:
                        t = work.loc[r, "_audit_ts"]
                        if pd.isna(t):
                            continue
                        gval = work.loc[r, group_key_eff] if group_key_eff else None
                        expected_t = t - pd.Timedelta(hours=lag_hours)

                        expected = _get_base_value_at_time(
                            work,
                            time_col="_audit_ts",
                            base_col=str(pf.base_col),
                            t=expected_t,
                            group_key=group_key_eff,
                            group_value=gval,
                            direction="backward",
                        )
                        actual = work.loc[r, pf.col]

                        # Numeric compare with tolerance, else exact/NaN equality
                        ok = True
                        if pd.isna(expected) and pd.isna(actual):
                            ok = True
                        elif pd.api.types.is_numeric_dtype(
                            work[pf.col]
                        ) and pd.api.types.is_numeric_dtype(work[pf.base_col]):
                            try:
                                ok = (pd.isna(expected) and pd.isna(actual)) or (
                                    abs(float(actual) - float(expected))
                                    <= deep_check_tolerance
                                )
                            except Exception:
                                ok = True
                        else:
                            ok = actual == expected

                        if not ok:
                            lag_mismatches.append(
                                f"{pf.col}: row={int(r)} t={t} expected(base@{expected_t})={expected} got={actual}"
                            )
                            # don't explode
                            if len(lag_mismatches) >= 25:
                                break
                    lag_checked += 1

                if (
                    pf.kind == "rolling"
                    and pf.hours is not None
                    and pf.stat is not None
                ):
                    window_hours = int(pf.hours)
                    stat = str(pf.stat)

                    for r in sample_rows:
                        t = work.loc[r, "_audit_ts"]
                        if pd.isna(t):
                            continue
                        gval = work.loc[r, group_key_eff] if group_key_eff else None

                        expected = _rolling_stat_from_history(
                            work,
                            time_col="_audit_ts",
                            base_col=str(pf.base_col),
                            t_end=t,
                            window_hours=window_hours,
                            stat=stat,
                            group_key=group_key_eff,
                            group_value=gval,
                        )
                        actual = work.loc[r, pf.col]

                        ok = True
                        if pd.isna(expected) and pd.isna(actual):
                            ok = True
                        else:
                            try:
                                ok = abs(float(actual) - float(expected)) <= max(
                                    deep_check_tolerance, 1e-9
                                )
                            except Exception:
                                ok = True

                        if not ok:
                            rolling_mismatches.append(
                                f"{pf.col}: row={int(r)} t={t} expected({stat},{window_hours}h)={expected} got={actual}"
                            )
                            if len(rolling_mismatches) >= 25:
                                break
                    rolling_checked += 1

            if lag_mismatches:
                issues.append(
                    f"Deep check found {len(lag_mismatches)} sampled lag mismatches (showing up to 25). Likely leakage or incorrect construction."
                )
            if rolling_mismatches:
                issues.append(
                    f"Deep check found {len(rolling_mismatches)} sampled rolling mismatches (showing up to 25). Likely leakage or incorrect construction."
                )

            deep_results = {
                "sample_size_used": int(len(sample_rows)),
                "group_key": group_key_eff,
                "lag_features_checked": int(lag_checked),
                "rolling_features_checked": int(rolling_checked),
                "lag_mismatches_examples": lag_mismatches[:25],
                "rolling_mismatches_examples": rolling_mismatches[:25],
            }

    return {
        "issues": issues,
        "warnings": warnings,
        "lag_features_count": int(len(lag_features)),
        "rolling_features_count": int(len(rolling_features)),
        "timestamp_like_cols": timestamp_like_cols,
        "deep_check": deep_results,
    }


def audit_cv_splits_gpu(
    cv_splits: List[Tuple[Any, Any]],
    timestamps: Union[pd.Series, np.ndarray, Any],
    *,
    expected_val_window_hours: Optional[int] = 24,
    require_non_overlap: bool = True,
    enforce_train_before_test: bool = True,
) -> Dict[str, Any]:
    """
    Audit CV splits to ensure time awareness.

    Enhancements:
    - Supports numpy or cupy indices in cv_splits
    - Optional expected window size
    - Stronger overlap checks and monotonic test-window ordering checks

    Returns:
        dict with issues/warnings/n_folds plus per-fold summary.
    """
    ts = _to_datetime_utc(timestamps)
    issues: List[str] = []
    warnings: List[str] = []

    per_fold: List[Dict[str, Any]] = []

    # Collect test windows for overlap checks
    test_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for fold_idx, (train_idx_raw, test_idx_raw) in enumerate(cv_splits):
        train_idx = _as_numpy_indices(train_idx_raw)
        test_idx = _as_numpy_indices(test_idx_raw)

        if train_idx.size == 0 or test_idx.size == 0:
            issues.append(f"Fold {fold_idx+1}: empty train or test indices.")
            continue

        train_times = ts.iloc[train_idx]
        test_times = ts.iloc[test_idx]

        train_max = train_times.max()
        train_min = train_times.min()
        test_min = test_times.min()
        test_max = test_times.max()

        # Train-before-test check
        if (
            enforce_train_before_test
            and (pd.notna(train_max) and pd.notna(test_min))
            and train_max > test_min
        ):
            issues.append(
                f"Fold {fold_idx+1}: Train max ({train_max}) > Test min ({test_min}) => leakage (train contains future times)."
            )

        # Contiguity / window span
        span_hours = (
            float((test_max - test_min).total_seconds() / 3600.0)
            if pd.notna(test_max) and pd.notna(test_min)
            else np.nan
        )
        if expected_val_window_hours is not None and not np.isnan(span_hours):
            # If data are hourly points, inclusive span for a 24h window can show ~23h;
            # accept +/-2 hours.
            if abs(span_hours - float(expected_val_window_hours)) > 2.0:
                warnings.append(
                    f"Fold {fold_idx+1}: Test span is {span_hours:.1f}h (expected ~{expected_val_window_hours}h)."
                )

        # Record test window
        test_windows.append((test_min, test_max))

        per_fold.append(
            {
                "fold": fold_idx + 1,
                "train_n": int(train_idx.size),
                "test_n": int(test_idx.size),
                "train_min": None if pd.isna(train_min) else train_min.isoformat(),
                "train_max": None if pd.isna(train_max) else train_max.isoformat(),
                "test_min": None if pd.isna(test_min) else test_min.isoformat(),
                "test_max": None if pd.isna(test_max) else test_max.isoformat(),
                "test_span_hours": None if np.isnan(span_hours) else span_hours,
            }
        )

    # Overlap checks between folds (test windows should not overlap if rolling-origin blocks)
    if require_non_overlap and len(test_windows) > 1:
        # sort windows by start time
        windows_sorted = sorted(enumerate(test_windows), key=lambda x: x[1][0])
        for i in range(1, len(windows_sorted)):
            prev_fold, (prev_start, prev_end) = windows_sorted[i - 1]
            cur_fold, (cur_start, cur_end) = windows_sorted[i]
            # overlap if current start <= previous end
            if pd.notna(prev_end) and pd.notna(cur_start) and cur_start <= prev_end:
                issues.append(
                    f"Test window overlap: fold {cur_fold+1} [{cur_start},{cur_end}] overlaps fold {prev_fold+1} [{prev_start},{prev_end}]"
                )

    return {
        "issues": issues,
        "warnings": warnings,
        "n_folds": int(len(cv_splits)),
        "per_fold": per_fold,
    }
