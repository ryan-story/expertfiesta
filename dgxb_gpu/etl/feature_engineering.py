"""
GPU Feature Engineering Pipeline for DGXB

This module is a GPU-first refactor of the CPU feature engineering pipeline.
Major accelerations:
- Fuzzy merge via H3 neighbor expansion + GPU joins + GPU scoring (cuDF + CuPy)
- Sector-hour aggregations via cuDF groupby
- City-wide weather imputation via cuDF groupby + joins

Notes:
- H3 cell computation and k-ring neighbor generation remain CPU (h3-py is CPU).
- We convert H3 strings to int64 for fast GPU joins.
- Outputs match the CPU pipeline artifacts:
  - merged_events.parquet
  - sector_hour_base.parquet
  - X_features.parquet
  - X_metadata.json
  - y_target.parquet (via prepare_y_target_regression, CPU/pandas ok)

Dependencies:
  pip install cudf-cu12 cupy-cuda12x rmm  (version depends on your CUDA)
  pip install h3 pandas pyarrow

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import h3

logger = logging.getLogger(__name__)


# -----------------------------
# Optional: configure RMM pool
# -----------------------------
def _try_enable_rmm_pool(initial_pool_size: Optional[int] = None) -> None:
    """
    Optional memory pool to reduce GPU alloc overhead.
    initial_pool_size: bytes (e.g., 2 * 1024**3 for 2GB) or None
    """
    try:
        import rmm  # type: ignore

        if initial_pool_size is None:
            rmm.reinitialize(pool_allocator=True)
        else:
            rmm.reinitialize(pool_allocator=True, initial_pool_size=initial_pool_size)
        logger.info("Enabled RMM pool allocator.")
    except Exception as e:
        logger.debug(f"RMM pool not enabled: {e}")


def _require_gpu_libs():
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GPU libraries not available. Install cudf + cupy matching your CUDA runtime. "
            f"Original error: {e}"
        )


# -----------------------------
# CPU helpers (H3 is CPU)
# -----------------------------
def get_h3_cell(lat: float, lon: float, resolution: int) -> Optional[str]:
    try:
        lat = float(lat)
        lon = float(lon)
        if pd.isna(lat) or pd.isna(lon) or np.isnan(lat) or np.isnan(lon):
            return None
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None
        try:
            return str(h3.latlng_to_cell(lat, lon, resolution))
        except AttributeError:
            return str(h3.geo_to_h3(lat, lon, resolution))
    except Exception:
        return None


def get_h3_neighbors(cell: str, k: int = 1) -> List[str]:
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


def _h3_str_to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        # h3-py v4 exposes str_to_int
        return int(h3.str_to_int(s))  # type: ignore[attr-defined]
    except Exception:
        # fallback: v3 has h3_to_int maybe; otherwise parse hex
        try:
            return int(h3.h3_to_int(s))  # type: ignore[attr-defined]
        except Exception:
            try:
                return int(s, 16)  # last resort if string is hex-like
            except Exception:
                return None


def prepare_traffic_data(traffic_df: pd.DataFrame) -> pd.DataFrame:
    traffic_clean = traffic_df.dropna(subset=["timestamp"]).copy()

    if "lat" in traffic_clean.columns and "lon" in traffic_clean.columns:
        lat_valid = (traffic_clean["lat"].between(25, 35)) & traffic_clean[
            "lat"
        ].notna()
        if lat_valid.sum() < len(traffic_clean) * 0.5:
            if (
                "location_lat" in traffic_clean.columns
                and "location_lon" in traffic_clean.columns
            ):
                logger.info("Using location_lat/location_lon instead of lat/lon")
                traffic_clean["lat"] = traffic_clean["location_lat"]
                traffic_clean["lon"] = traffic_clean["location_lon"]

    traffic_clean = traffic_clean.dropna(subset=["lat", "lon"]).copy()
    traffic_clean = traffic_clean[
        traffic_clean["lat"].between(25, 35) & traffic_clean["lon"].between(-100, -95)
    ].copy()
    traffic_clean["timestamp"] = pd.to_datetime(traffic_clean["timestamp"], utc=True)
    return traffic_clean


# -----------------------------
# GPU math helpers (CuPy)
# -----------------------------
def _haversine_km_cupy(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine on GPU. Inputs are cupy arrays (float64/float32).
    Returns cupy array of km distances.
    """
    import cupy as cp

    R = 6371.0
    # radians
    lat1r = cp.deg2rad(lat1)
    lon1r = cp.deg2rad(lon1)
    lat2r = cp.deg2rad(lat2)
    lon2r = cp.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1r) * cp.cos(lat2r) * cp.sin(dlon / 2) ** 2
    return 2 * cp.arcsin(cp.sqrt(a)) * R


# -----------------------------
# GPU Fuzzy Merge
# -----------------------------
@dataclass(frozen=True)
class FuzzyMergeParams:
    h3_resolution: int = 9
    k_ring_size: int = 1
    max_spatial_km: float = 50.0
    max_time_hours: float = 1.0
    spatial_weight: float = 1.0
    temporal_weight: float = 0.5
    time_equiv_km_per_hour: float = 50.0  # your CPU scaling


def fuzzy_merge_h3_gpu(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    params: FuzzyMergeParams = FuzzyMergeParams(),
) -> pd.DataFrame:
    """
    GPU fuzzy merge:
    - Compute h3_cell on CPU
    - Build neighbor expansion table (traffic_unique_cell -> neighbor_cell)
    - Move to GPU and create candidate pairs via joins
    - Filter by time + distance (GPU)
    - Score and pick best candidate per traffic_row_id (GPU)
    - Return merged pandas df (traffic + weather_ cols)
    """
    _require_gpu_libs()
    import cudf
    import cupy as cp

    traffic_clean = prepare_traffic_data(traffic_df)
    weather_clean = weather_df.dropna(subset=["lat", "lon", "timestamp"]).copy()
    weather_clean["timestamp"] = pd.to_datetime(weather_clean["timestamp"], utc=True)

    if len(weather_clean) == 0 or len(traffic_clean) == 0:
        logger.warning(
            "Empty traffic/weather after cleaning; returning original traffic_df."
        )
        return traffic_df.copy()

    logger.info(
        f"GPU fuzzy merge: traffic={len(traffic_clean)} weather={len(weather_clean)}"
    )

    # Assign H3 cells (CPU)
    traffic_clean = traffic_clean.copy()
    weather_clean = weather_clean.copy()

    traffic_clean["h3_cell"] = [
        get_h3_cell(lat, lon, params.h3_resolution)
        for lat, lon in zip(
            traffic_clean["lat"].to_numpy(), traffic_clean["lon"].to_numpy()
        )
    ]
    weather_clean["h3_cell"] = [
        get_h3_cell(lat, lon, params.h3_resolution)
        for lat, lon in zip(
            weather_clean["lat"].to_numpy(), weather_clean["lon"].to_numpy()
        )
    ]

    traffic_valid = traffic_clean[traffic_clean["h3_cell"].notna()].copy()
    weather_valid = weather_clean[weather_clean["h3_cell"].notna()].copy()

    if len(traffic_valid) == 0 or len(weather_valid) == 0:
        logger.warning("No valid H3 cells; returning original traffic_df.")
        return traffic_df.copy()

    # Convert H3 to int64 (CPU) for fast GPU joins
    traffic_valid["h3_cell_int"] = (
        traffic_valid["h3_cell"].map(_h3_str_to_int).astype("Int64")
    )
    weather_valid["h3_cell_int"] = (
        weather_valid["h3_cell"].map(_h3_str_to_int).astype("Int64")
    )

    traffic_valid = traffic_valid.dropna(subset=["h3_cell_int"]).copy()
    weather_valid = weather_valid.dropna(subset=["h3_cell_int"]).copy()

    # Stable row id for grouping
    traffic_valid = traffic_valid.reset_index(drop=False).rename(
        columns={"index": "traffic_src_index"}
    )
    traffic_valid["traffic_row_id"] = np.arange(len(traffic_valid), dtype=np.int64)

    # Build neighbor expansion table (CPU) on unique traffic cells
    uniq_cells = traffic_valid["h3_cell"].unique().tolist()
    rows = []
    for c in uniq_cells:
        for nb in get_h3_neighbors(c, params.k_ring_size):
            rows.append((_h3_str_to_int(c), _h3_str_to_int(nb)))
    nb_df = pd.DataFrame(rows, columns=["traffic_h3_int", "neighbor_h3_int"]).dropna()

    # Move to GPU
    g_traffic = cudf.from_pandas(
        traffic_valid[
            [
                "traffic_row_id",
                "traffic_src_index",
                "timestamp",
                "lat",
                "lon",
                "h3_cell",
                "h3_cell_int",
            ]
        ]
    )
    g_weather = cudf.from_pandas(weather_valid)
    g_nb = cudf.from_pandas(nb_df)

    # Join traffic -> neighbors -> weather to create candidates
    g_traffic = g_traffic.rename(columns={"h3_cell_int": "traffic_h3_int"})
    g_weather = g_weather.rename(
        columns={"h3_cell_int": "neighbor_h3_int", "timestamp": "weather_timestamp"}
    )

    cand = g_traffic.merge(g_nb, on="traffic_h3_int", how="left")
    cand = cand.merge(
        g_weather,
        on="neighbor_h3_int",
        how="left",
        suffixes=("_tr", "_wx"),
    )

    # Drop candidates without weather match
    cand = cand.dropna(subset=["weather_timestamp", "lat_wx", "lon_wx"])

    if len(cand) == 0:
        logger.warning(
            "No candidate matches after neighbor join; returning traffic without weather cols."
        )
        return traffic_df.copy()

    # Time filter (GPU)
    # Compute time diff hours as float32
    # cudf datetime subtraction yields timedelta64[ns]
    td = (cand["weather_timestamp"] - cand["timestamp"]).abs()
    time_diff_hours = td.astype("timedelta64[s]").astype("float32") / 3600.0
    cand["weather_time_diff_hours"] = time_diff_hours
    cand = cand[cand["weather_time_diff_hours"] <= params.max_time_hours]

    if len(cand) == 0:
        logger.warning(
            "No candidates after time filter; returning traffic without weather cols."
        )
        return traffic_df.copy()

    # Distance + score on GPU via CuPy
    lat1 = cand["lat"].to_cupy()
    lon1 = cand["lon"].to_cupy()
    lat2 = cand["lat_wx"].to_cupy()
    lon2 = cand["lon_wx"].to_cupy()

    dist_km = _haversine_km_cupy(
        lat1.astype(cp.float32),
        lon1.astype(cp.float32),
        lat2.astype(cp.float32),
        lon2.astype(cp.float32),
    )
    cand["weather_distance_km"] = cudf.Series(dist_km)

    cand = cand[cand["weather_distance_km"] <= params.max_spatial_km]
    if len(cand) == 0:
        logger.warning(
            "No candidates after distance filter; returning traffic without weather cols."
        )
        return traffic_df.copy()

    # Combined score
    time_equiv_km = cand["weather_time_diff_hours"] * params.time_equiv_km_per_hour
    cand["weather_combined_score"] = (
        params.spatial_weight * cand["weather_distance_km"]
        + params.temporal_weight * time_equiv_km
    )

    # Select best weather per traffic_row_id (min score)
    cand = cand.sort_values(["traffic_row_id", "weather_combined_score"])
    best = cand.drop_duplicates(subset=["traffic_row_id"], keep="first")

    # Build output: attach weather columns (prefixed)
    # Determine which weather columns to bring over (exclude core)
    exclude = {
        "weather_timestamp",
        "lat_wx",
        "lon_wx",
        "h3_cell",
        "neighbor_h3_int",
        "traffic_h3_int",
    }
    wx_cols = [
        c for c in best.columns if c not in exclude and c not in g_traffic.columns
    ]

    # Prefix weather columns
    rename = {}
    for c in wx_cols:
        if c.startswith("weather_"):
            # already prefixed (e.g., weather_time_diff_hours)
            continue
        rename[c] = f"weather_{c}"
    best = best.rename(columns=rename)

    # Also attach weather_h3_cell (string) if present in weather_valid
    # g_weather includes "h3_cell" as weather cell string from original
    if "h3_cell_wx" in best.columns:
        best = best.rename(columns={"h3_cell_wx": "weather_h3_cell"})

    # Reduce to join keys + new columns
    keep_cols = (
        ["traffic_row_id"]
        + [c for c in best.columns if c.startswith("weather_")]
        + [
            "weather_distance_km",
            "weather_time_diff_hours",
            "weather_combined_score",
            "weather_h3_cell",
        ]
    )
    keep_cols = [c for c in keep_cols if c in best.columns]
    best_small = best[keep_cols]

    merged_gpu = g_traffic.merge(best_small, on="traffic_row_id", how="left")

    # Back to pandas; merge onto original traffic_df indices for full preservation
    merged_pd = merged_gpu.to_pandas()

    # Reconstruct final_df aligned to original traffic_df rows
    final_df = traffic_df.copy()
    # Add weather columns with default None
    for col in merged_pd.columns:
        if col not in final_df.columns and col not in (
            "traffic_src_index",
            "traffic_row_id",
            "traffic_h3_int",
        ):
            final_df[col] = None

    # Write back on the subset rows that survived traffic_valid filtering
    # traffic_src_index points to original index in traffic_df
    idx = merged_pd["traffic_src_index"].to_numpy()
    for col in merged_pd.columns:
        if col in final_df.columns and col not in (
            "traffic_src_index",
            "traffic_row_id",
            "traffic_h3_int",
        ):
            final_df.loc[idx, col] = merged_pd[col].to_numpy()

    logger.info("GPU merge complete.")
    return final_df


# -----------------------------
# GPU Aggregations
# -----------------------------
def aggregate_incidents_to_sector_hour_gpu(
    traffic_df: pd.DataFrame,
    h3_resolution: int = 9,
) -> pd.DataFrame:
    _require_gpu_libs()
    import cudf

    logger.info("GPU: Aggregating incidents to sector-hour...")

    # Ensure h3_cell
    if "h3_cell" not in traffic_df.columns:
        traffic_df = traffic_df.copy()
        traffic_df["h3_cell"] = [
            get_h3_cell(lat, lon, h3_resolution)
            for lat, lon in zip(
                traffic_df.get("lat", traffic_df.get("location_lat")).to_numpy(),
                traffic_df.get("lon", traffic_df.get("location_lon")).to_numpy(),
            )
        ]

    df = traffic_df[traffic_df["h3_cell"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["h3_cell", "hour_ts", "incident_count"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # hour_ts in pandas then to cudf (cudf floor works too, but keep stable)
    df["hour_ts"] = df["timestamp"].dt.floor("h")
    df["hour"] = df["hour_ts"].dt.hour
    df["day_of_week"] = df["hour_ts"].dt.dayofweek
    df["day_of_month"] = df["hour_ts"].dt.day
    df["month"] = df["hour_ts"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    gdf = cudf.from_pandas(df)

    agg_map = {
        "incident_id": "count",
        "hour": "first",
        "day_of_week": "first",
        "day_of_month": "first",
        "month": "first",
        "is_weekend": "first",
    }
    if "agency" in gdf.columns:
        agg_map["agency"] = "nunique"
    if "status" in gdf.columns:
        agg_map["status"] = "nunique"

    out = gdf.groupby(["h3_cell", "hour_ts"]).agg(agg_map).reset_index()
    out = out.rename(columns={"incident_id": "incident_count"})
    if "agency" in out.columns:
        out = out.rename(columns={"agency": "n_unique_agency"})
    if "status" in out.columns:
        out = out.rename(columns={"status": "n_unique_status"})

    pdf = out.to_pandas()
    logger.info(f"  GPU incidents aggregated rows: {len(pdf)}")
    return pdf


def aggregate_weather_to_sector_hour_gpu(
    weather_df: pd.DataFrame,
    h3_resolution: int = 9,
) -> pd.DataFrame:
    _require_gpu_libs()
    import cudf

    logger.info("GPU: Aggregating weather to sector-hour...")

    df = weather_df.dropna(subset=["lat", "lon", "timestamp"]).copy()
    if len(df) == 0:
        return pd.DataFrame()

    if "h3_cell" not in df.columns:
        df["h3_cell"] = [
            get_h3_cell(lat, lon, h3_resolution)
            for lat, lon in zip(df["lat"].to_numpy(), df["lon"].to_numpy())
        ]
    df = df[df["h3_cell"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour_ts"] = df["timestamp"].dt.floor("h")

    gdf = cudf.from_pandas(df)

    # Define desired aggregations (only if columns exist)
    spec = {
        "temperature": "median",
        "dewpoint": "median",
        "humidity": "median",
        "wind_speed": "median",
        "wind_direction": "median",
        "precipitation_amount": "max",
        "precipitation_probability": "max",
        "is_daytime": "first",
        "weather_code": "first",
        "data_source": "first",
    }

    agg_map = {k: v for k, v in spec.items() if k in gdf.columns}
    out = gdf.groupby(["h3_cell", "hour_ts"]).agg(agg_map).reset_index()

    # Prefix non-key columns with weather_
    rename = {c: f"weather_{c}" for c in out.columns if c not in ("h3_cell", "hour_ts")}
    out = out.rename(columns=rename)

    pdf = out.to_pandas()
    logger.info(f"  GPU weather aggregated rows: {len(pdf)}")
    return pdf


# -----------------------------
# Target creation (pandas is fine)
# -----------------------------
def make_regression_target(sector_hour_base: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating regression target...")
    sh = sector_hour_base.sort_values(["h3_cell", "hour_ts"]).copy()
    sh["incident_count_t"] = sh["incident_count"]
    sh["incident_count_t_plus_1"] = sh.groupby("h3_cell")["incident_count"].shift(-1)
    y = sh[["h3_cell", "hour_ts", "incident_count_t", "incident_count_t_plus_1"]].copy()
    logger.info(
        f"  Target rows: {len(y)}; NaNs in t+1: {y['incident_count_t_plus_1'].isna().sum()}"
    )
    return y


def prepare_y_target_regression(
    sector_hour_base_path: str = "gold-gpu-traffic/sector_hour_base.parquet",
    gold_output_dir: str = "gold-gpu-traffic",
) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("Y PIPELINE (GPU PATH): Regression Target Preparation")
    logger.info("=" * 70)

    base = pd.read_parquet(sector_hour_base_path)
    y = make_regression_target(base)

    out_dir = Path(gold_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_path = out_dir / "y_target.parquet"
    y.to_parquet(y_path, index=False)

    meta = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": len(y),
        "target_type": "regression",
        "target_variable": "incident_count_t_plus_1",
        "pipeline": "y_target_regression",
    }
    with open(out_dir / "y_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✓ Saved y_target to {y_path}")
    return y


# -----------------------------
# End-to-end X pipeline (GPU)
# -----------------------------
def merge_and_save_X_features_gpu(
    traffic_silver_path: str = "silver-gpu-traffic/data_silver.parquet",
    weather_silver_path: str = "silver-gpu-weather/data_silver.parquet",
    gold_output_dir: str = "gold-gpu-traffic",
    h3_resolution: int = 9,
    k_ring_size: int = 1,
    max_spatial_km: float = 50,
    max_time_hours: float = 1,
    spatial_weight: float = 1.0,
    temporal_weight: float = 0.5,
    enable_rmm_pool: bool = True,
) -> pd.DataFrame:
    """
    GPU version of merge_and_save_X_features().
    Produces:
      - merged_events.parquet (traceability)
      - sector_hour_base.parquet
      - X_features.parquet
      - X_metadata.json
    """
    _require_gpu_libs()
    if enable_rmm_pool:
        _try_enable_rmm_pool()

    logger.info("=" * 70)
    logger.info("X PIPELINE (GPU): Merge + Feature Engineering")
    logger.info("=" * 70)

    traffic_df = pd.read_parquet(traffic_silver_path)
    weather_df = pd.read_parquet(weather_silver_path)

    # GPU fuzzy merge (for traceability + optional debug)
    params = FuzzyMergeParams(
        h3_resolution=h3_resolution,
        k_ring_size=k_ring_size,
        max_spatial_km=max_spatial_km,
        max_time_hours=max_time_hours,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )
    merged_df = fuzzy_merge_h3_gpu(traffic_df, weather_df, params=params)

    out_dir = Path(gold_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_events_path = out_dir / "merged_events.parquet"
    merged_df.to_parquet(merged_events_path, index=False)
    logger.info(f"✓ Saved merged events: {merged_events_path}")

    # Sector-hour aggregates (GPU)
    inc_sector_hour = aggregate_incidents_to_sector_hour_gpu(
        traffic_df, h3_resolution=h3_resolution
    )
    if "incident_count" in inc_sector_hour.columns:
        inc_sector_hour["incident_count"] = inc_sector_hour["incident_count"].fillna(0)

    # Weather aggregates (GPU)
    # ensure weather has h3_cell if not (CPU compute)
    if "h3_cell" not in weather_df.columns:
        weather_df = weather_df.copy()
        weather_df["h3_cell"] = [
            get_h3_cell(lat, lon, h3_resolution)
            for lat, lon in zip(
                weather_df["lat"].to_numpy(), weather_df["lon"].to_numpy()
            )
        ]
    wx_sector_hour = aggregate_weather_to_sector_hour_gpu(
        weather_df, h3_resolution=h3_resolution
    )

    # Join (pandas join is fine at this size; if you want pure GPU, we can convert to cudf and join)
    sector_hour_base = inc_sector_hour.merge(
        wx_sector_hour, on=["h3_cell", "hour_ts"], how="outer"
    )
    sector_hour_base["hour_ts"] = pd.to_datetime(sector_hour_base["hour_ts"], utc=True)

    # Fill incident_count zeros for weather-only rows
    if "incident_count" in sector_hour_base.columns:
        sector_hour_base["incident_count"] = sector_hour_base["incident_count"].fillna(
            0
        )

    # Fill time features from hour_ts
    sector_hour_base["hour"] = sector_hour_base["hour_ts"].dt.hour
    sector_hour_base["day_of_week"] = sector_hour_base["hour_ts"].dt.dayofweek
    sector_hour_base["day_of_month"] = sector_hour_base["hour_ts"].dt.day
    sector_hour_base["month"] = sector_hour_base["hour_ts"].dt.month
    sector_hour_base["is_weekend"] = (sector_hour_base["day_of_week"] >= 5).astype(int)

    # Weather imputation (GPU city-wide fallback per hour_ts)
    # We'll do it in cuDF for speed + cleanliness.
    import cudf

    g_base = cudf.from_pandas(sector_hour_base)
    weather_cols = [c for c in g_base.columns if c.startswith("weather_")]

    if weather_cols:
        missing_mask = g_base[weather_cols].isna().any(axis=1)
        if int(missing_mask.sum()) > 0:
            logger.info(
                f"Imputing weather for {int(missing_mask.sum())} rows using city-wide per-hour fallback (GPU)."
            )

            # City-level aggregates per hour_ts
            # numeric -> median, non-numeric -> first of mode (approx: first)
            num_cols = [
                c for c in weather_cols if g_base[c].dtype.kind in ("i", "u", "f")
            ]
            cat_cols = [c for c in weather_cols if c not in num_cols]

            agg_map = {}
            for c in num_cols:
                agg_map[c] = "median"
            for c in cat_cols:
                agg_map[c] = "first"

            city = g_base.groupby("hour_ts").agg(agg_map).reset_index()
            city = city.rename(columns={c: f"{c}__city" for c in weather_cols})

            g_base = g_base.merge(city, on="hour_ts", how="left")

            # Fill nulls
            for c in weather_cols:
                city_c = f"{c}__city"
                if city_c in g_base.columns:
                    g_base[c] = g_base[c].fillna(g_base[city_c])
            drop_city = [
                f"{c}__city" for c in weather_cols if f"{c}__city" in g_base.columns
            ]
            g_base = g_base.drop(columns=drop_city)

    sector_hour_base = g_base.to_pandas()
    sector_hour_base_path = out_dir / "sector_hour_base.parquet"
    sector_hour_base.to_parquet(sector_hour_base_path, index=False)
    logger.info(
        f"✓ Saved sector-hour base: {sector_hour_base_path} (rows={len(sector_hour_base):,})"
    )

    # Build X_features (same logic, pandas is fine; can be moved to cuDF later if needed)
    X_features = sector_hour_base.copy()

    # Drop target cols if present
    for c in ["incident_count", "incident_count_t", "incident_count_t_plus_1"]:
        if c in X_features.columns:
            X_features = X_features.drop(columns=[c])

    # Cyclical encodings
    if "hour" in X_features.columns:
        X_features["hour_sin"] = np.sin(2 * np.pi * X_features["hour"] / 24)
        X_features["hour_cos"] = np.cos(2 * np.pi * X_features["hour"] / 24)
    if "day_of_week" in X_features.columns:
        X_features["day_of_week_sin"] = np.sin(
            2 * np.pi * X_features["day_of_week"] / 7
        )
        X_features["day_of_week_cos"] = np.cos(
            2 * np.pi * X_features["day_of_week"] / 7
        )
    if "month" in X_features.columns:
        X_features["month_sin"] = np.sin(2 * np.pi * X_features["month"] / 12)
        X_features["month_cos"] = np.cos(2 * np.pi * X_features["month"] / 12)

    # Wind direction transforms
    if "weather_wind_direction" in X_features.columns:
        wd = pd.to_numeric(X_features["weather_wind_direction"], errors="coerce")
        X_features["wind_direction_sin"] = np.sin(np.deg2rad(wd))
        X_features["wind_direction_cos"] = np.cos(np.deg2rad(wd))
        X_features = X_features.drop(columns=["weather_wind_direction"])

    # Encode categorical weather features (pandas get_dummies ok)
    for cat_col in ["weather_weather_code", "weather_data_source"]:
        if cat_col in X_features.columns:
            d = pd.get_dummies(X_features[cat_col], prefix=cat_col, drop_first=True)
            X_features = pd.concat([X_features.drop(columns=[cat_col]), d], axis=1)

    if "weather_is_daytime" in X_features.columns:
        X_features["weather_is_daytime"] = (
            X_features["weather_is_daytime"].fillna(0).astype(int)
        )

    # has_weather_data
    weather_cols2 = [
        c
        for c in X_features.columns
        if c.startswith("weather_")
        and c not in ["weather_data_source"]
        and not c.startswith("weather_data_source_")
    ]
    if weather_cols2:
        X_features["has_weather_data"] = (
            X_features[weather_cols2[0]].notna().astype(int)
        )

    X_path = out_dir / "X_features.parquet"
    X_features.to_parquet(X_path, index=False)
    logger.info(
        f"✓ Saved X_features: {X_path} (rows={len(X_features):,}, cols={len(X_features.columns)})"
    )

    meta = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": len(X_features),
        "feature_columns": list(X_features.columns),
        "aggregation_level": "sector_hour",
        "primary_key": ["h3_cell", "hour_ts"],
        "merge_parameters": {
            "h3_resolution": h3_resolution,
            "k_ring_size": k_ring_size,
            "max_spatial_km": max_spatial_km,
            "max_time_hours": max_time_hours,
            "spatial_weight": spatial_weight,
            "temporal_weight": temporal_weight,
        },
        "pipeline": "X_features_gpu",
        "date_range": {
            "start": (
                merged_df["timestamp"].min().isoformat()
                if "timestamp" in merged_df.columns
                else None
            ),
            "end": (
                merged_df["timestamp"].max().isoformat()
                if "timestamp" in merged_df.columns
                else None
            ),
        },
        "artifacts": {
            "merged_events": str(merged_events_path),
            "sector_hour_base": str(sector_hour_base_path),
            "X_features": str(X_path),
        },
    }
    with open(out_dir / "X_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("X pipeline (GPU) complete.")
    return X_features


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example run (adjust paths to your GPU-labeled folders)
    X = merge_and_save_X_features_gpu(
        traffic_silver_path="silver-gpu-traffic/data_silver.parquet",
        weather_silver_path="silver-gpu-weather/data_silver.parquet",
        gold_output_dir="gold-gpu-traffic",
        h3_resolution=9,
        k_ring_size=1,
    )
    y = prepare_y_target_regression(
        sector_hour_base_path="gold-gpu-traffic/sector_hour_base.parquet",
        gold_output_dir="gold-gpu-traffic",
    )
    print("X shape:", X.shape)
    print("y shape:", y.shape)
