"""
Silver Layer Processor for DGXB
Consolidates bronze parquet files into structured silver layer
Preserves all individual records with full temporal granularity (no aggregation)
"""

import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_bronze_parquets(
    bronze_dir: str, file_pattern: str = "*.parquet"
) -> pd.DataFrame:
    """
    Load all bronze parquet files matching pattern and combine into single DataFrame
    Preserves all rows - no aggregation
    """
    bronze_path = Path(bronze_dir)
    parquet_files = sorted(bronze_path.glob(file_pattern))

    if not parquet_files:
        logger.warning(
            f"No parquet files found in {bronze_dir} matching {file_pattern}"
        )
        return pd.DataFrame()

    logger.info(
        f"Loading {len(parquet_files)} bronze parquet files from {bronze_dir}..."
    )

    all_dfs = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
            logger.debug(f"Loaded {len(df)} records from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue

    if not all_dfs:
        logger.warning("No dataframes loaded")
        return pd.DataFrame()

    # Combine all dataframes (preserves all rows)
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Combined {len(combined)} total records from {len(parquet_files)} files"
    )

    return combined


def flatten_location_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten location JSON/dict column into structured columns
    Preserves all rows - just expands nested structure
    """
    if "location" not in df.columns:
        return df

    df = df.copy()

    # Extract coordinates from location.geometry if it's a dict
    def extract_location_data(loc):
        if pd.isna(loc):
            return pd.Series(
                {"location_type": None, "location_lon": None, "location_lat": None}
            )

        if isinstance(loc, dict):
            loc_type = loc.get("type", None)
            coords = loc.get("coordinates", [])
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None

            return pd.Series(
                {"location_type": loc_type, "location_lon": lon, "location_lat": lat}
            )

        return pd.Series(
            {"location_type": None, "location_lon": None, "location_lat": None}
        )

    # Extract location data (preserves all rows)
    location_df = df["location"].apply(extract_location_data)

    # Add extracted columns
    df["location_type"] = location_df["location_type"]
    df["location_lon"] = pd.to_numeric(location_df["location_lon"], errors="coerce")
    df["location_lat"] = pd.to_numeric(location_df["location_lat"], errors="coerce")

    # Use location coordinates if lat/lon are missing
    if "lat" in df.columns and df["lat"].isna().any():
        df["lat"] = df["lat"].fillna(df["location_lat"])
    if "lon" in df.columns and df["lon"].isna().any():
        df["lon"] = df["lon"].fillna(df["location_lon"])

    return df


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and structure traffic data for silver layer
    Preserves all individual records with full temporal granularity
    """
    if df.empty:
        return df

    df = df.copy()

    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Flatten location column (expands JSON, preserves rows)
    df = flatten_location_column(df)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Ensure lat/lon are numeric
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Sort by timestamp (preserves all records)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove rows with missing critical fields
    critical_fields = ["incident_id", "timestamp"]
    if all(field in df.columns for field in critical_fields):
        before = len(df)
        df = df.dropna(subset=critical_fields)
        after = len(df)
        if before > after:
            logger.info(f"Removed {before - after} rows with missing critical fields")

    return df


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and structure weather data for silver layer
    Preserves all individual records with full temporal granularity
    """
    if df.empty:
        return df

    df = df.copy()

    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Ensure lat/lon are numeric
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Ensure numeric weather fields
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
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    # Sort by timestamp (preserves all records)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove rows with missing critical fields
    critical_fields = ["timestamp"]
    if all(field in df.columns for field in critical_fields):
        before = len(df)
        df = df.dropna(subset=critical_fields)
        after = len(df)
        if before > after:
            logger.info(f"Removed {before - after} rows with missing critical fields")

    return df


def save_silver_data(
    df: pd.DataFrame,
    output_dir: str,
    single_file: bool = True,
    max_rows_per_file: int = 1000000,
):
    """
    Save cleaned data to silver layer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("Empty DataFrame, nothing to save")
        return

    if single_file or len(df) <= max_rows_per_file:
        # Single file
        file_path = output_path / "data_silver.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(df)} records to {file_path}")
    else:
        # Partition if too large
        num_files = (len(df) // max_rows_per_file) + 1
        for i in range(num_files):
            start_idx = i * max_rows_per_file
            end_idx = min((i + 1) * max_rows_per_file, len(df))
            chunk_df = df.iloc[start_idx:end_idx]

            file_path = output_path / f"data_silver_part_{i+1:03d}.parquet"
            chunk_df.to_parquet(file_path, index=False)
            logger.info(f"Saved {len(chunk_df)} records to {file_path}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "columns": list(df.columns),
        "date_range": {
            "start": (
                df["timestamp"].min().isoformat()
                if "timestamp" in df.columns
                and pd.api.types.is_datetime64_any_dtype(df["timestamp"])
                and not df["timestamp"].isna().all()
                else None
            ),
            "end": (
                df["timestamp"].max().isoformat()
                if "timestamp" in df.columns
                and pd.api.types.is_datetime64_any_dtype(df["timestamp"])
                and not df["timestamp"].isna().all()
                else None
            ),
        },
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved metadata to {metadata_path}")


def process_bronze_to_silver_traffic(
    bronze_dir: str = "bronze-traffic", silver_dir: str = "silver-cpu-traffic"
) -> pd.DataFrame:
    """
    Complete pipeline: Load bronze traffic, clean, save to silver
    Preserves all individual records with full temporal granularity
    """
    logger.info(f"Processing bronze to silver: {bronze_dir} -> {silver_dir}")

    # Load bronze data
    df = load_bronze_parquets(bronze_dir, file_pattern="traffic_*.parquet")

    if df.empty:
        logger.warning("No data to process")
        return df

    # Clean and structure (preserves all rows)
    df = clean_traffic_data(df)

    # Save to silver
    save_silver_data(df, silver_dir)

    logger.info(f"Silver layer processing complete: {len(df)} records")
    return df


def process_bronze_to_silver_weather(
    bronze_dir: str = "bronze-weather", silver_dir: str = "silver-cpu-weather"
) -> pd.DataFrame:
    """
    Complete pipeline: Load bronze weather, clean, save to silver
    Preserves all individual records with full temporal granularity
    """
    logger.info(f"Processing bronze to silver: {bronze_dir} -> {silver_dir}")

    # Load bronze data
    df = load_bronze_parquets(bronze_dir, file_pattern="weather_*.parquet")

    if df.empty:
        logger.warning("No data to process")
        return df

    # Clean and structure (preserves all rows)
    df = clean_weather_data(df)

    # Save to silver
    save_silver_data(df, silver_dir)

    logger.info(f"Silver layer processing complete: {len(df)} records")
    return df


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Process traffic
    print("Processing traffic data...")
    traffic_df = process_bronze_to_silver_traffic()
    print(f"Traffic: {len(traffic_df)} records")

    # Process weather
    print("\nProcessing weather data...")
    weather_df = process_bronze_to_silver_weather()
    print(f"Weather: {len(weather_df)} records")
