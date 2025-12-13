"""
GPU Rich Feature Engineering Pipeline for DGXB (Sector-Hour)

GPU-first refactor + super-enrichment:
- cuDF groupby shifts (lags) per h3_cell
- cuDF rolling stats per h3_cell
- H3 neighbor expansion for spatial aggregates (fast)
- Extra temporal features (hour_of_week, holidays, etc.)
- Extra volatility/trend indicators

Assumptions:
- Input is sector-hour level with primary keys: (h3_cell, hour_ts)
- hour_ts is UTC or tz-aware; we normalize to UTC
- If incident_count is present in X (optional), spatial features leverage it;
  otherwise, spatial features are computed for weather vars only.

Dependencies:
  pip install cudf-cu12 cupy-cuda12x rmm h3 pandas pyarrow
Optional:
  pip install holidays

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import h3

logger = logging.getLogger(__name__)


def _require_gpu_libs():
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GPU libraries not available. Install cudf + cupy matching your CUDA runtime. "
            f"Original error: {e}"
        )


def _try_enable_rmm_pool(initial_pool_size: Optional[int] = None) -> None:
    try:
        import rmm  # type: ignore

        if initial_pool_size is None:
            rmm.reinitialize(pool_allocator=True)
        else:
            rmm.reinitialize(pool_allocator=True, initial_pool_size=initial_pool_size)
        logger.info("Enabled RMM pool allocator.")
    except Exception as e:
        logger.debug(f"RMM pool not enabled: {e}")


def _h3_str_to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(h3.str_to_int(s))  # type: ignore[attr-defined]
    except Exception:
        try:
            return int(h3.h3_to_int(s))  # type: ignore[attr-defined]
        except Exception:
            try:
                return int(s, 16)
            except Exception:
                return None


def _neighbors_for_cell(cell: str, k: int) -> List[str]:
    if cell is None:
        return []
    out = [cell]
    if k <= 0:
        return out
    try:
        try:
            ring = h3.grid_ring(cell, k)
        except AttributeError:
            ring = h3.k_ring(cell, k)
        out.extend(list(ring))
    except Exception:
        pass
    return out


@dataclass(frozen=True)
class RichGPUParams:
    # Temporal
    add_holidays: bool = True
    holiday_country: str = "US"

    # Lags/Rolling
    lag_windows: Tuple[int, ...] = (1, 3, 6, 12, 24)
    rolling_windows: Tuple[int, ...] = (3, 6, 12, 24)

    # Spatial (H3 neighbors)
    spatial_k_ring: int = 1  # ring size
    spatial_aggs: Tuple[str, ...] = ("mean", "max", "sum")  # supported: mean/max/sum
    spatial_feature_cols: Optional[Tuple[str, ...]] = None  # None => auto-detect

    # Super-enrichment
    add_trend_features: bool = True
    trend_windows: Tuple[int, ...] = (6, 12, 24)  # rolling slope windows
    add_volatility_features: bool = True
    volatility_windows: Tuple[int, ...] = (6, 12, 24)  # rolling CV windows

    # Output handling
    drop_timestamp_if_hour_ts_present: bool = True


def _safe_import_holidays():
    try:
        import holidays  # type: ignore

        return holidays
    except Exception:
        return None


def _add_time_binning_features_gdf(gdf, timestamp_col: str = "timestamp"):
    """
    Adds:
      - time_of_day_bucket (0..5)
      - rush_hour (0/1/2)
      - day_period (0 weekday / 1 weekend)
    """
    import cudf

    if timestamp_col not in gdf.columns:
        return gdf

    ts = gdf[timestamp_col]
    hour = ts.dt.hour
    dow = ts.dt.weekday  # 0=Mon

    # time_of_day_bucket: bins [-1,6,10,14,18,22,24] -> labels 0..5
    # We'll implement with boolean masks (GPU-friendly)
    bucket = cudf.Series(np.zeros(len(gdf), dtype=np.int32))
    bucket = bucket.where(~((hour >= 6) & (hour < 10)), 1)
    bucket = bucket.where(~((hour >= 10) & (hour < 14)), 2)
    bucket = bucket.where(~((hour >= 14) & (hour < 18)), 3)
    bucket = bucket.where(~((hour >= 18) & (hour < 22)), 4)
    bucket = bucket.where(~((hour >= 22) & (hour <= 23)), 5)
    gdf["time_of_day_bucket"] = bucket.astype("int8")

    # rush_hour: 1 for 7-9, 2 for 17-19
    rush = cudf.Series(np.zeros(len(gdf), dtype=np.int8))
    rush = rush.where(~((hour >= 7) & (hour < 9)), 1)
    rush = rush.where(~((hour >= 17) & (hour < 19)), 2)
    gdf["rush_hour"] = rush

    # day_period: weekend flag
    gdf["day_period"] = (dow >= 5).astype("int8")
    return gdf


def _add_extended_temporal_features_gdf(gdf, timestamp_col: str = "timestamp"):
    """
    Adds:
      - day_of_year
      - week_of_year (ISO)
      - quarter
      - is_month_start / is_month_end
      - days_since_year_start
      - hour_of_week (0..167)
      - cyclical encodings for hour_of_week
    """
    import cudf
    import cupy as cp

    if timestamp_col not in gdf.columns:
        return gdf

    ts = gdf[timestamp_col]

    gdf["day_of_year"] = ts.dt.dayofyear.astype("int16")

    # ISO week in cuDF: via dt.isocalendar().week if available; else fallback from pandas later
    try:
        gdf["week_of_year"] = ts.dt.isocalendar().week.astype("int16")
    except Exception:
        # fallback: compute in pandas at end if needed
        gdf["week_of_year"] = cudf.Series(np.nan, index=gdf.index)

    gdf["quarter"] = ts.dt.quarter.astype("int8")
    gdf["is_month_start"] = ts.dt.is_month_start.astype("int8")
    gdf["is_month_end"] = ts.dt.is_month_end.astype("int8")

    # days_since_year_start: ts - year_start
    # Build year_start per row (YYYY-01-01) and subtract
    year = ts.dt.year.astype("int16")
    # Construct year_start using string assembly (GPU-safe)
    year_start_str = year.astype("str") + "-01-01"
    year_start = cudf.to_datetime(year_start_str, utc=True)
    gdf["days_since_year_start"] = (ts - year_start).dt.days.astype("int16")

    # hour_of_week
    dow = ts.dt.weekday.astype("int16")
    hour = ts.dt.hour.astype("int16")
    how = (dow * 24 + hour).astype("int16")
    gdf["hour_of_week"] = how

    # cyclical encoding for hour_of_week
    how_f = how.astype("float32").to_cupy()
    gdf["hour_of_week_sin"] = cudf.Series(cp.sin(2 * cp.pi * how_f / 168.0))
    gdf["hour_of_week_cos"] = cudf.Series(cp.cos(2 * cp.pi * how_f / 168.0))

    return gdf


def _add_holiday_flags_gdf(gdf, timestamp_col: str = "timestamp", country: str = "US"):
    """
    Adds:
      - is_holiday (0/1)
      - is_day_before_holiday (0/1)
      - is_day_after_holiday (0/1)

    Implemented by creating a holiday calendar on CPU for date range, then GPU join.
    """
    import cudf

    holidays_lib = _safe_import_holidays()
    if holidays_lib is None:
        logger.info("holidays library not installed; skipping holiday flags.")
        return gdf

    ts = gdf[timestamp_col]
    # Pull min/max dates to CPU
    min_date = pd.Timestamp(ts.min().to_pandas()).date()
    max_date = pd.Timestamp(ts.max().to_pandas()).date()

    try:
        hol = holidays_lib.country_holidays(
            country, years=range(min_date.year, max_date.year + 1)
        )
    except Exception:
        logger.info("Failed to create holiday calendar; skipping holiday flags.")
        return gdf

    # Build a small CPU table of dates
    days = pd.date_range(start=min_date, end=max_date, freq="D")
    is_holiday = [1 if d.date() in hol else 0 for d in days]
    is_before = [1 if (d + pd.Timedelta(days=1)).date() in hol else 0 for d in days]
    is_after = [1 if (d - pd.Timedelta(days=1)).date() in hol else 0 for d in days]
    cal = pd.DataFrame(
        {
            "date": days.date,
            "is_holiday": is_holiday,
            "is_day_before_holiday": is_before,
            "is_day_after_holiday": is_after,
        }
    )
    gcal = cudf.from_pandas(cal)
    gcal["date"] = cudf.to_datetime(gcal["date"])

    # Extract date from ts
    gdf["date"] = ts.dt.floor("D")
    gdf = gdf.merge(gcal, on="date", how="left")
    for c in ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"]:
        gdf[c] = gdf[c].fillna(0).astype("int8")
    gdf = gdf.drop(columns=["date"])
    return gdf


def _auto_detect_weather_cols(columns: Sequence[str]) -> List[str]:
    keep = {
        "weather_temperature",
        "weather_humidity",
        "weather_wind_speed",
        "weather_precipitation_amount",
        "weather_precipitation_probability",
        "weather_dewpoint",
    }
    out = [c for c in columns if c in keep]
    return out


def _add_lags_gdf(
    gdf,
    group_key: str,
    cols: Sequence[str],
    lag_windows: Sequence[int],
):
    """
    Adds col_lag_{k}h = groupby(col).shift(k)
    Assumes rows are sorted by (group_key, timestamp).
    """
    for col in cols:
        if col not in gdf.columns:
            continue
        s = gdf[col]
        for k in lag_windows:
            gdf[f"{col}_lag_{k}h"] = s.groupby(gdf[group_key]).shift(k)
    return gdf


def _add_rollings_gdf(
    gdf,
    group_key: str,
    cols: Sequence[str],
    windows: Sequence[int],
):
    """
    Adds rolling mean/std/min/max over window size (in rows) within each group.
    Assumes hourly rows; window size = hours.
    """

    for col in cols:
        if col not in gdf.columns:
            continue

        grp = gdf.groupby(group_key)[col]
        for w in windows:
            # rolling per group (cudf supports groupby().rolling)
            r = grp.rolling(window=w, min_periods=1)

            gdf[f"{col}_rolling_mean_{w}h"] = r.mean().reset_index(drop=True)
            gdf[f"{col}_rolling_std_{w}h"] = r.std().reset_index(drop=True)
            gdf[f"{col}_rolling_min_{w}h"] = r.min().reset_index(drop=True)
            gdf[f"{col}_rolling_max_{w}h"] = r.max().reset_index(drop=True)

    return gdf


def _add_volatility_features_gdf(
    gdf,
    group_key: str,
    cols: Sequence[str],
    windows: Sequence[int],
):
    """
    Adds rolling coefficient of variation (CV = std / (mean + eps)) per group.
    """

    eps = 1e-6
    for col in cols:
        if col not in gdf.columns:
            continue
        grp = gdf.groupby(group_key)[col]
        for w in windows:
            r = grp.rolling(window=w, min_periods=1)
            mu = r.mean().reset_index(drop=True)
            sd = r.std().reset_index(drop=True)
            gdf[f"{col}_rolling_cv_{w}h"] = sd / (mu.abs() + eps)
    return gdf


def _rolling_slope_cupy(x_cp):
    """
    Compute slope of linear regression y~t for each window in a rolling manner is expensive.
    We use an efficient closed-form for slope for evenly spaced t = 0..(w-1).

    This helper expects x_cp shape: (n_windows, w) and returns slope per window.
    """
    import cupy as cp

    w = x_cp.shape[1]
    t = cp.arange(w, dtype=cp.float32)
    t_mean = (w - 1) / 2.0
    denom = cp.sum((t - t_mean) ** 2)
    x_mean = cp.mean(x_cp, axis=1)
    num = cp.sum((t[None, :] - t_mean) * (x_cp - x_mean[:, None]), axis=1)
    return num / (denom + 1e-6)


def _add_trend_features_gdf(
    gdf,
    group_key: str,
    cols: Sequence[str],
    windows: Sequence[int],
    timestamp_col: str = "timestamp",
):
    """
    Adds rolling trend slope per group for each col and window.
    Implementation strategy:
      - For each group, extract column values to CuPy, build rolling windows via stride tricks.
      - This is GPU-heavy but feasible for your scale (thousands of rows, hundreds of cells).
    """
    import cudf
    import cupy as cp

    # We do this per column, per group, to avoid exploding memory.
    # Store result as cudf Series aligned to gdf row order.
    # Assumes sorted by (group_key, timestamp).

    # Build group offsets once
    groups = gdf[group_key]
    # cudf groupby gives us group boundaries via value_counts? We'll compute by run-length encoding.
    # Approach: get group ids by factorize
    codes, uniques = groups.factorize()
    gdf["_gid"] = codes.astype("int32")

    # Get order indices for each gid
    # We'll operate on the already-sorted frame: contiguous blocks per gid.
    gid = gdf["_gid"].to_cupy()
    # Find boundaries where gid changes
    change = cp.concatenate([cp.array([True]), gid[1:] != gid[:-1]])
    starts = cp.where(change)[0]
    ends = cp.concatenate([starts[1:], cp.array([len(gdf)], dtype=cp.int64)])

    for col in cols:
        if col not in gdf.columns:
            continue
        x = gdf[col].astype("float32").fillna(np.nan).to_cupy()

        for w in windows:
            out = cp.full(len(gdf), cp.nan, dtype=cp.float32)
            for s, e in zip(starts.tolist(), ends.tolist()):
                seg = x[s:e]
                n = seg.shape[0]
                if n < 2:
                    continue
                if n < w:
                    # If shorter than window, compute slope on available segment prefix using w=n
                    ww = n
                else:
                    ww = w

                if ww < 2:
                    continue

                # rolling windows (n-ww+1, ww)
                # Use stride trick
                shape = (n - ww + 1, ww)
                strides = (seg.strides[0], seg.strides[0])
                windows_view = cp.lib.stride_tricks.as_strided(
                    seg, shape=shape, strides=strides
                )
                slopes = _rolling_slope_cupy(windows_view)

                # Align: place slope at the window end index
                # indices in segment: (ww-1 .. n-1)
                out[s + (ww - 1) : s + n] = slopes

            gdf[f"{col}_rolling_slope_{w}h"] = cudf.Series(out)

    gdf = gdf.drop(columns=["_gid"])
    return gdf


def _add_precip_indicators_gdf(gdf):
    """
    Adds precipitation indicator features if precipitation columns exist.
    """
    import cudf

    if "weather_precipitation_amount" in gdf.columns:
        p = gdf["weather_precipitation_amount"].fillna(0.0)
        gdf["precip_any"] = (p > 0).astype("int8")
        # heavy precipitation threshold via global 90th percentile (robust)
        try:
            p90 = float(p.quantile(0.90).to_pandas())
            gdf["precip_heavy_p90"] = (p >= p90).astype("int8")
        except Exception:
            gdf["precip_heavy_p90"] = cudf.Series(np.zeros(len(gdf), dtype=np.int8))

    if "weather_precipitation_probability" in gdf.columns:
        pp = gdf["weather_precipitation_probability"].fillna(0.0)
        gdf["precip_prob_gt_50"] = (pp >= 50).astype("int8")
        gdf["precip_prob_gt_80"] = (pp >= 80).astype("int8")

    return gdf


def _build_neighbor_edges_df(h3_cells: Iterable[str], k: int) -> pd.DataFrame:
    """
    CPU: Build edges table (h3_cell -> neighbor_cell) for unique cells.
    """
    rows = []
    for c in h3_cells:
        nbs = _neighbors_for_cell(c, k)
        for nb in nbs:
            rows.append((c, nb))
    return pd.DataFrame(rows, columns=["h3_cell", "neighbor_h3_cell"])


def _spatial_aggregates_h3_gdf(
    gdf,
    hour_col: str = "hour_ts",
    cell_col: str = "h3_cell",
    feature_cols: Sequence[str] = (),
    aggs: Sequence[str] = ("mean", "max", "sum"),
    k_ring: int = 1,
):
    """
    Spatial neighbor aggregates at the SAME hour_ts:
      For each (cell, hour), compute agg of feature over neighbors in k-ring.

    Implementation:
      - Build neighbor edges on CPU for unique cells
      - Move edges to GPU and join:
          left: gdf (cell, hour)
          edges: cell -> neighbor_cell
          right: gdf (neighbor_cell, hour) providing neighbor feature values
      - Groupby (cell, hour) aggregate across neighbors

    Adds columns:
      spatial_{agg}_{col}_k{k}
    """
    import cudf

    if not feature_cols:
        return gdf

    # Unique cells (CPU) for edge building
    uniq_cells = gdf[cell_col].unique().to_pandas().tolist()
    edges_pd = _build_neighbor_edges_df(uniq_cells, k_ring)
    edges_g = cudf.from_pandas(edges_pd)

    # Prepare left keys
    left = gdf[[cell_col, hour_col]].copy()
    left = left.reset_index(drop=True)
    left["_row"] = cudf.Series(np.arange(len(left), dtype=np.int64))

    # Join left -> edges to get neighbor cell per row
    le = left.merge(edges_g, on=cell_col, how="left")

    # Join neighbor features at same hour
    right_cols = [cell_col, hour_col] + list(feature_cols)
    right = gdf[right_cols].rename(columns={cell_col: "neighbor_h3_cell"})
    joined = le.merge(right, on=["neighbor_h3_cell", hour_col], how="left")

    # Groupby original row id
    grp = joined.groupby("_row")

    # Aggregate each feature col
    agg_frames = []
    for c in feature_cols:
        if c not in joined.columns:
            continue
        for a in aggs:
            if a == "mean":
                s = grp[c].mean()
            elif a == "max":
                s = grp[c].max()
            elif a == "sum":
                s = grp[c].sum()
            else:
                continue
            s = s.rename(f"spatial_{a}_{c}_k{k_ring}")
            agg_frames.append(s)

    if not agg_frames:
        return gdf

    out = cudf.concat(agg_frames, axis=1).reset_index()
    # Merge back by _row -> original rows
    gdf2 = gdf.reset_index(drop=True)
    gdf2["_row"] = cudf.Series(np.arange(len(gdf2), dtype=np.int64))
    gdf2 = gdf2.merge(out, on="_row", how="left")
    gdf2 = gdf2.drop(columns=["_row"])
    return gdf2


def enrich_X_features_gpu(
    X_input_path: str = "gold-gpu-traffic/X_features.parquet",
    rich_output_dir: str = "rich-gold-gpu-traffic",
    params: RichGPUParams = RichGPUParams(),
) -> pd.DataFrame:
    """
    GPU super-enriched rich feature pipeline for sector-hour X_features.

    Reads X_input_path (sector-hour), ensures timestamp exists,
    adds temporal + lag + rolling + spatial neighbor aggregates + super features,
    writes to rich_output_dir/X_features.parquet and metadata.
    """
    _require_gpu_libs()
    _try_enable_rmm_pool()

    import cudf

    logger.info("=" * 70)
    logger.info("RICH FEATURE ENGINEERING (GPU): Super-Enriching X Features")
    logger.info("=" * 70)

    # Load
    logger.info(f"Loading base X: {X_input_path}")
    X_base_pd = pd.read_parquet(X_input_path)

    if "hour_ts" in X_base_pd.columns:
        X_base_pd["hour_ts"] = pd.to_datetime(X_base_pd["hour_ts"], utc=True)
        X_base_pd["timestamp"] = X_base_pd["hour_ts"]
    elif "timestamp" in X_base_pd.columns:
        X_base_pd["timestamp"] = pd.to_datetime(X_base_pd["timestamp"], utc=True)
    else:
        raise ValueError(
            "X does not contain hour_ts or timestamp; cannot enrich for sector-hour."
        )

    if "h3_cell" not in X_base_pd.columns:
        raise ValueError("X must contain h3_cell for sector-hour enrichment.")

    # Convert to GPU
    gdf = cudf.from_pandas(X_base_pd)

    # Ensure correct dtypes
    gdf["timestamp"] = cudf.to_datetime(gdf["timestamp"], utc=True)
    if "hour_ts" in gdf.columns:
        gdf["hour_ts"] = cudf.to_datetime(gdf["hour_ts"], utc=True)

    group_key = "h3_cell"

    # Sort for groupwise ops
    gdf = gdf.sort_values([group_key, "timestamp"]).reset_index(drop=True)

    # ---- Temporal enrichment ----
    logger.info("Adding time binning features (GPU)...")
    gdf = _add_time_binning_features_gdf(gdf, timestamp_col="timestamp")

    logger.info("Adding extended temporal features (GPU)...")
    gdf = _add_extended_temporal_features_gdf(gdf, timestamp_col="timestamp")

    if params.add_holidays:
        logger.info("Adding holiday flags (CPU calendar + GPU join)...")
        gdf = _add_holiday_flags_gdf(
            gdf, timestamp_col="timestamp", country=params.holiday_country
        )

    # Add precipitation indicators
    gdf = _add_precip_indicators_gdf(gdf)

    # ---- Determine feature columns for lags/rollings ----
    if params.spatial_feature_cols is None:
        base_cols = list(gdf.columns)
        weather_cols = _auto_detect_weather_cols(base_cols)
        # Also include incident_count if present (helps model)
        if "incident_count" in gdf.columns:
            feature_cols = ["incident_count"] + weather_cols
        else:
            feature_cols = weather_cols
    else:
        feature_cols = list(params.spatial_feature_cols)

    # ---- Lags ----
    logger.info(
        f"Adding lag features (GPU) windows={params.lag_windows} cols={len(feature_cols)}..."
    )
    gdf = _add_lags_gdf(
        gdf, group_key=group_key, cols=feature_cols, lag_windows=params.lag_windows
    )

    # ---- Rolling stats ----
    logger.info(
        f"Adding rolling statistics (GPU) windows={params.rolling_windows} cols={len(feature_cols)}..."
    )
    gdf = _add_rollings_gdf(
        gdf, group_key=group_key, cols=feature_cols, windows=params.rolling_windows
    )

    # ---- Volatility ----
    if params.add_volatility_features:
        logger.info(
            f"Adding rolling volatility CV (GPU) windows={params.volatility_windows}..."
        )
        gdf = _add_volatility_features_gdf(
            gdf,
            group_key=group_key,
            cols=feature_cols,
            windows=params.volatility_windows,
        )

    # ---- Trend (rolling slope) ----
    if params.add_trend_features:
        logger.info(
            f"Adding rolling trend slopes (GPU) windows={params.trend_windows}..."
        )
        gdf = _add_trend_features_gdf(
            gdf, group_key=group_key, cols=feature_cols, windows=params.trend_windows
        )

    # ---- Spatial neighbor aggregates at SAME HOUR ----
    logger.info(
        f"Adding spatial H3 neighbor aggregates (GPU) k={params.spatial_k_ring} aggs={params.spatial_aggs}..."
    )
    # Choose spatial cols: by default include incident_count (if exists) + key weather cols
    spatial_cols = []
    if "incident_count" in gdf.columns:
        spatial_cols.append("incident_count")
    spatial_cols += _auto_detect_weather_cols(list(gdf.columns))
    # Dedup
    spatial_cols = [c for c in dict.fromkeys(spatial_cols) if c in gdf.columns]

    gdf = _spatial_aggregates_h3_gdf(
        gdf,
        hour_col="hour_ts" if "hour_ts" in gdf.columns else "timestamp",
        cell_col="h3_cell",
        feature_cols=spatial_cols,
        aggs=params.spatial_aggs,
        k_ring=params.spatial_k_ring,
    )

    # ---- Derived spatial deltas (own vs neighbor mean) ----
    # These help with localized anomaly signals (hotspots)
    for c in spatial_cols:
        mcol = f"spatial_mean_{c}_k{params.spatial_k_ring}"
        if mcol in gdf.columns and c in gdf.columns:
            gdf[f"{c}_minus_neighbor_mean_k{params.spatial_k_ring}"] = (
                gdf[c] - gdf[mcol]
            )

    # ---- Cleanup timestamp ----
    if (
        params.drop_timestamp_if_hour_ts_present
        and ("hour_ts" in gdf.columns)
        and ("timestamp" in gdf.columns)
    ):
        gdf = gdf.drop(columns=["timestamp"])

    # Some cudf versions may leave week_of_year as null if unsupported
    # Fix in pandas if needed
    out_pd = gdf.to_pandas()
    if "week_of_year" in out_pd.columns and out_pd["week_of_year"].isna().any():
        ts = pd.to_datetime(
            out_pd["hour_ts"] if "hour_ts" in out_pd.columns else out_pd["timestamp"],
            utc=True,
        )
        out_pd["week_of_year"] = ts.isocalendar().week.astype(int)

    # Save
    out_dir = Path(rich_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "X_features.parquet"
    out_pd.to_parquet(out_path, index=False)

    meta = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": len(out_pd),
        "base_feature_count": len(X_base_pd.columns),
        "enriched_feature_count": len(out_pd.columns),
        "new_features_added": int(len(out_pd.columns) - len(X_base_pd.columns)),
        "enrichment_parameters": {
            "lag_windows": list(params.lag_windows),
            "rolling_windows": list(params.rolling_windows),
            "spatial_k_ring": params.spatial_k_ring,
            "spatial_aggs": list(params.spatial_aggs),
            "trend_windows": list(params.trend_windows),
            "volatility_windows": list(params.volatility_windows),
            "add_holidays": params.add_holidays,
        },
        "pipeline": "rich_X_features_gpu_super",
        "base_source": X_input_path,
    }
    with open(out_dir / "X_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("GPU rich feature engineering complete.")
    logger.info(f"Saved: {out_path}")
    logger.info(
        f"Rows: {len(out_pd):,}  Cols: {len(out_pd.columns)}  Added: {meta['new_features_added']}"
    )
    logger.info("=" * 70)

    return out_pd


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    X = enrich_X_features_gpu(
        X_input_path="gold-gpu-traffic/X_features.parquet",
        rich_output_dir="rich-gold-gpu-traffic",
    )
    print("Done. Shape:", X.shape)
