"""
Silver Layer Processor for DGXB (GPU)
Consolidates bronze parquet files into structured silver layer (NO aggregation)
Preserves all individual records with full temporal granularity.

GPU-first using cuDF for:
- reading many parquet files
- concatenation
- cleaning/coercion
- sorting
- writing parquet

Location flattening supports:
1) Arrow/Parquet struct-like "location" (dict/struct) -> extract type/coordinates
2) JSON string in "location" -> json_extract if available, else CPU fallback
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_gpu_libs():
    try:
        import cudf  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GPU libraries not available. Install RAPIDS cuDF matching your CUDA runtime. "
            f"Original error: {e}"
        )


@dataclass(frozen=True)
class SilverGPUParams:
    single_file: bool = True
    max_rows_per_file: int = 1_000_000
    # If location JSON parsing on GPU is not available, we do CPU fallback once
    allow_cpu_location_fallback: bool = True


def load_bronze_parquets_gpu(bronze_dir: str, file_pattern: str = "*.parquet"):
    """
    Load all bronze parquet files matching pattern and combine into a single GPU DataFrame.
    Preserves all rows - no aggregation.
    """
    _require_gpu_libs()
    import cudf

    bronze_path = Path(bronze_dir)
    parquet_files = sorted(bronze_path.glob(file_pattern))

    if not parquet_files:
        logger.warning(
            f"No parquet files found in {bronze_dir} matching {file_pattern}"
        )
        return cudf.DataFrame()

    logger.info(
        f"Loading {len(parquet_files)} bronze parquet files from {bronze_dir}..."
    )

    # cuDF can read list of files directly (fast)
    try:
        gdf = cudf.read_parquet([str(p) for p in parquet_files])
        logger.info(
            f"Combined {len(gdf)} total records from {len(parquet_files)} files"
        )
        return gdf
    except Exception as e:
        logger.warning(f"Bulk cudf.read_parquet failed, falling back file-by-file: {e}")

    gdfs = []
    for p in parquet_files:
        try:
            part = cudf.read_parquet(str(p))
            gdfs.append(part)
        except Exception as ee:
            logger.error(f"Failed to load {p}: {ee}")

    if not gdfs:
        logger.warning("No dataframes loaded")
        return cudf.DataFrame()

    gdf = cudf.concat(gdfs, ignore_index=True)
    logger.info(f"Combined {len(gdf)} total records from {len(gdfs)} files")
    return gdf


def _coerce_numeric(gdf, col: str):
    """
    Coerce a column to numeric (float64) with nulls on failure.
    """
    import cudf

    if col not in gdf.columns:
        return gdf
    # to_numeric exists in cudf, errors='coerce' supported in most versions
    try:
        gdf[col] = cudf.to_numeric(gdf[col], errors="coerce")
    except Exception:
        # fallback: cast after replacing non-numeric
        gdf[col] = cudf.to_numeric(gdf[col].astype("str"), errors="coerce")
    return gdf


def flatten_location_column_gpu(gdf, params: SilverGPUParams):
    """
    Flatten location column into:
      - location_type
      - location_lon
      - location_lat

    Supports:
    - struct/dict-like location with fields 'type' and 'coordinates'
    - JSON string location

    Preserves all rows; expands nested structure only.
    """
    import cudf

    if "location" not in gdf.columns:
        return gdf

    gdf = gdf.copy()

    # Initialize outputs
    if "location_type" not in gdf.columns:
        gdf["location_type"] = None
    if "location_lon" not in gdf.columns:
        gdf["location_lon"] = np.nan
    if "location_lat" not in gdf.columns:
        gdf["location_lat"] = np.nan

    loc = gdf["location"]

    # Case A: location is a STRUCT type (Arrow struct written to parquet)
    # cuDF represents this as a struct column with .struct accessor in many versions.
    try:
        if hasattr(loc, "struct"):
            # Try extracting "type"
            try:
                gdf["location_type"] = loc.struct.field("type")
            except Exception:
                pass

            # Extract coordinates -> expected [lon, lat]
            # Some datasets store as list under "coordinates" (struct field)
            try:
                coords = loc.struct.field("coordinates")
                # coords may be a list column
                if hasattr(coords, "list"):
                    gdf["location_lon"] = coords.list.get(0)
                    gdf["location_lat"] = coords.list.get(1)
                else:
                    # if coords is string/other, fall through
                    pass
            except Exception:
                pass

            # Coerce numeric
            gdf = _coerce_numeric(gdf, "location_lon")
            gdf = _coerce_numeric(gdf, "location_lat")
            return gdf
    except Exception:
        pass

    # Case B: location is JSON string
    # Try GPU json_extract (available in newer cuDF)
    try:
        # json_extract returns strings typically
        # Extract $.type and $.coordinates[0/1]
        gdf["location_type"] = loc.astype("str").str.json_extract("$.type")
        gdf["location_lon"] = loc.astype("str").str.json_extract("$.coordinates[0]")
        gdf["location_lat"] = loc.astype("str").str.json_extract("$.coordinates[1]")

        gdf = _coerce_numeric(gdf, "location_lon")
        gdf = _coerce_numeric(gdf, "location_lat")
        return gdf
    except Exception as e:
        logger.info(f"GPU json_extract not available or failed: {e}")

    # Case C: CPU fallback (single pass) if allowed
    if not params.allow_cpu_location_fallback:
        logger.warning("Location column could not be flattened on GPU; skipping.")
        return gdf

    logger.info("Falling back to CPU parsing for location (single pass).")

    pdf = gdf[["location"]].to_pandas()

    def extract_location(loc_val):
        if pd.isna(loc_val):
            return (None, None, None)
        if isinstance(loc_val, dict):
            loc_type = loc_val.get("type", None)
            coords = loc_val.get("coordinates", []) or []
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None
            return (loc_type, lon, lat)
        if isinstance(loc_val, str):
            try:
                obj = json.loads(loc_val)
                loc_type = obj.get("type", None)
                coords = obj.get("coordinates", []) or []
                lon = coords[0] if len(coords) > 0 else None
                lat = coords[1] if len(coords) > 1 else None
                return (loc_type, lon, lat)
            except Exception:
                return (None, None, None)
        return (None, None, None)

    parsed = pdf["location"].apply(extract_location)
    pdf_out = pd.DataFrame(
        parsed.tolist(), columns=["location_type", "location_lon", "location_lat"]
    )

    # Bring back to GPU and set columns
    gdf["location_type"] = cudf.from_pandas(pdf_out["location_type"])
    gdf["location_lon"] = cudf.from_pandas(
        pd.to_numeric(pdf_out["location_lon"], errors="coerce")
    )
    gdf["location_lat"] = cudf.from_pandas(
        pd.to_numeric(pdf_out["location_lat"], errors="coerce")
    )

    return gdf


def clean_traffic_data_gpu(gdf, params: SilverGPUParams):
    """
    Clean and structure traffic data for silver layer (GPU).
    Preserves all individual records with full temporal granularity.
    """
    import cudf

    if gdf is None or len(gdf) == 0:
        return gdf

    gdf = gdf.copy()

    # Remove duplicate columns (keep first)
    # cuDF doesn't have duplicated() on columns; implement via ordered dict
    cols = list(gdf.columns)
    seen = set()
    keep = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            keep.append(c)
    if len(keep) != len(cols):
        gdf = gdf[keep]

    # Flatten location
    gdf = flatten_location_column_gpu(gdf, params)

    # Timestamp -> datetime UTC
    if "timestamp" in gdf.columns:
        try:
            gdf["timestamp"] = cudf.to_datetime(
                gdf["timestamp"], utc=True, errors="coerce"
            )
        except Exception:
            # older cudf may not accept errors; do best effort
            gdf["timestamp"] = cudf.to_datetime(gdf["timestamp"], utc=True)

    # lat/lon numeric
    if "lat" in gdf.columns:
        gdf = _coerce_numeric(gdf, "lat")
    if "lon" in gdf.columns:
        gdf = _coerce_numeric(gdf, "lon")

    # Fill lat/lon with location_lat/lon if present and missing
    if "lat" in gdf.columns and "location_lat" in gdf.columns:
        gdf["lat"] = gdf["lat"].fillna(gdf["location_lat"])
    if "lon" in gdf.columns and "location_lon" in gdf.columns:
        gdf["lon"] = gdf["lon"].fillna(gdf["location_lon"])

    # Drop rows with missing critical fields
    critical_fields = ["incident_id", "timestamp"]
    if all(c in gdf.columns for c in critical_fields):
        before = len(gdf)
        gdf = gdf.dropna(subset=critical_fields)
        after = len(gdf)
        if before > after:
            logger.info(f"Removed {before - after} rows with missing critical fields")

    # Sort by timestamp
    if "timestamp" in gdf.columns:
        gdf = gdf.sort_values("timestamp").reset_index(drop=True)

    return gdf


def clean_weather_data_gpu(gdf, params: SilverGPUParams):
    """
    Clean and structure weather data for silver layer (GPU).
    Preserves all individual records with full temporal granularity.
    """
    import cudf

    if gdf is None or len(gdf) == 0:
        return gdf

    gdf = gdf.copy()

    # Remove duplicate columns (keep first)
    cols = list(gdf.columns)
    seen = set()
    keep = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            keep.append(c)
    if len(keep) != len(cols):
        gdf = gdf[keep]

    # Timestamp -> datetime UTC
    if "timestamp" in gdf.columns:
        try:
            gdf["timestamp"] = cudf.to_datetime(
                gdf["timestamp"], utc=True, errors="coerce"
            )
        except Exception:
            gdf["timestamp"] = cudf.to_datetime(gdf["timestamp"], utc=True)

    # lat/lon numeric
    if "lat" in gdf.columns:
        gdf = _coerce_numeric(gdf, "lat")
    if "lon" in gdf.columns:
        gdf = _coerce_numeric(gdf, "lon")

    # Numeric weather fields
    numeric_fields = [
        "temperature",
        "dewpoint",
        "humidity",
        "wind_speed",
        "wind_direction",
        "precipitation_amount",
        "precipitation_probability",
        "weather_code",
    ]
    for f in numeric_fields:
        if f in gdf.columns:
            gdf = _coerce_numeric(gdf, f)

    # Drop rows with missing critical fields
    if "timestamp" in gdf.columns:
        before = len(gdf)
        gdf = gdf.dropna(subset=["timestamp"])
        after = len(gdf)
        if before > after:
            logger.info(f"Removed {before - after} rows with missing critical fields")

    # Sort by timestamp
    if "timestamp" in gdf.columns:
        gdf = gdf.sort_values("timestamp").reset_index(drop=True)

    return gdf


def save_silver_data_gpu(
    gdf,
    output_dir: str,
    params: SilverGPUParams,
):
    """
    Save cleaned data to silver layer (GPU write).
    """
    _require_gpu_libs()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if gdf is None or len(gdf) == 0:
        logger.warning("Empty DataFrame, nothing to save")
        return

    # Single file or partition
    if params.single_file or len(gdf) <= params.max_rows_per_file:
        file_path = output_path / "data_silver.parquet"
        gdf.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(gdf)} records to {file_path}")
    else:
        num_files = (len(gdf) // params.max_rows_per_file) + 1
        for i in range(num_files):
            start = i * params.max_rows_per_file
            end = min((i + 1) * params.max_rows_per_file, len(gdf))
            chunk = gdf.iloc[start:end]
            file_path = output_path / f"data_silver_part_{i+1:03d}.parquet"
            chunk.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(chunk)} records to {file_path}")

    # Metadata (computed with minimal CPU transfer)
    cols = list(gdf.columns)

    # Date range handling (convert just min/max to CPU)
    date_start = None
    date_end = None
    if "timestamp" in gdf.columns:
        try:
            tmin = gdf["timestamp"].min()
            tmax = gdf["timestamp"].max()
            if tmin is not None and tmax is not None:
                date_start = pd.Timestamp(tmin.to_pandas()).isoformat()
                date_end = pd.Timestamp(tmax.to_pandas()).isoformat()
        except Exception:
            pass

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": int(len(gdf)),
        "columns": cols,
        "date_range": {"start": date_start, "end": date_end},
        "engine": "cudf",
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def process_bronze_to_silver_traffic_gpu(
    bronze_dir: str = "bronze-traffic",
    silver_dir: str = "silver-gpu-traffic",
    params: SilverGPUParams = SilverGPUParams(),
):
    """
    Complete pipeline: Load bronze traffic, clean, save to silver (GPU).
    Preserves all individual records with full temporal granularity.
    """
    logger.info(
        f"Processing bronze->silver (traffic, GPU): {bronze_dir} -> {silver_dir}"
    )

    gdf = load_bronze_parquets_gpu(bronze_dir, file_pattern="traffic_*.parquet")
    if gdf is None or len(gdf) == 0:
        logger.warning("No data to process")
        return gdf

    gdf = clean_traffic_data_gpu(gdf, params)
    save_silver_data_gpu(gdf, silver_dir, params)

    logger.info(f"Silver layer processing complete (traffic): {len(gdf)} records")
    return gdf


def process_bronze_to_silver_weather_gpu(
    bronze_dir: str = "bronze-weather",
    silver_dir: str = "silver-gpu-weather",
    params: SilverGPUParams = SilverGPUParams(),
):
    """
    Complete pipeline: Load bronze weather, clean, save to silver (GPU).
    Preserves all individual records with full temporal granularity.
    """
    logger.info(
        f"Processing bronze->silver (weather, GPU): {bronze_dir} -> {silver_dir}"
    )

    gdf = load_bronze_parquets_gpu(bronze_dir, file_pattern="weather_*.parquet")
    if gdf is None or len(gdf) == 0:
        logger.warning("No data to process")
        return gdf

    gdf = clean_weather_data_gpu(gdf, params)
    save_silver_data_gpu(gdf, silver_dir, params)

    logger.info(f"Silver layer processing complete (weather): {len(gdf)} records")
    return gdf


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Processing traffic data (GPU)...")
    traffic_gdf = process_bronze_to_silver_traffic_gpu()
    print(f"Traffic: {0 if traffic_gdf is None else len(traffic_gdf)} records")

    print("\nProcessing weather data (GPU)...")
    weather_gdf = process_bronze_to_silver_weather_gpu()
    print(f"Weather: {0 if weather_gdf is None else len(weather_gdf)} records")
