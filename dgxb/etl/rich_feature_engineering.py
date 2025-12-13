"""
Rich Feature Engineering Pipeline for DGXB
Enriches base X features with lags, rolling stats, spatial aggregates, and extended temporal features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import List, Optional
from math import radians, cos, sin, asin, sqrt

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in km"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * R


def add_time_binning_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Add time binning features (numeric representation)

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with time binning features added
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not found, skipping time binning"
        )
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    hour = df[timestamp_col].dt.hour

    # Time of day bucket (0-5)
    # 0: night (0-6), 1: morning (6-10), 2: midday (10-14),
    # 3: afternoon (14-18), 4: evening (18-22), 5: late (22-24)
    df["time_of_day_bucket"] = pd.cut(
        hour,
        bins=[-1, 6, 10, 14, 18, 22, 24],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(int)

    # Rush hour (0=no, 1=morning rush 7-9, 2=evening rush 17-19)
    df["rush_hour"] = 0
    df.loc[(hour >= 7) & (hour < 9), "rush_hour"] = 1
    df.loc[(hour >= 17) & (hour < 19), "rush_hour"] = 2

    # Day period (0=weekday, 1=weekend)
    day_of_week = df[timestamp_col].dt.dayofweek
    df["day_period"] = (day_of_week >= 5).astype(int)

    logger.info(
        "Added time binning features: time_of_day_bucket, rush_hour, day_period"
    )
    return df


def add_extended_temporal_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Add extended temporal features

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with extended temporal features added
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not found, skipping extended temporal"
        )
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Day of year (1-365)
    df["day_of_year"] = df[timestamp_col].dt.dayofyear

    # Week of year (1-52)
    df["week_of_year"] = df[timestamp_col].dt.isocalendar().week

    # Quarter (1-4)
    df["quarter"] = df[timestamp_col].dt.quarter

    # Month start/end indicators
    df["is_month_start"] = df[timestamp_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[timestamp_col].dt.is_month_end.astype(int)

    # Days since year start
    year_start = df[timestamp_col].dt.to_period("Y").dt.start_time
    df["days_since_year_start"] = (df[timestamp_col] - year_start).dt.days

    logger.info(
        "Added extended temporal features: day_of_year, week_of_year, quarter, is_month_start, is_month_end, days_since_year_start"
    )
    return df


def add_lag_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    feature_cols: Optional[List[str]] = None,
    lag_windows: List[int] = [1, 3, 6, 12, 24],
) -> pd.DataFrame:
    """
    Add lag features for specified columns

    Args:
        df: DataFrame sorted by timestamp
        timestamp_col: Name of timestamp column
        feature_cols: List of feature columns to create lags for (if None, auto-detect numeric weather features)
        lag_windows: List of lag windows in hours

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not found, skipping lag features"
        )
        return df

    # Auto-detect numeric weather features if not specified
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c.startswith("weather_")
            and c
            in [
                "weather_temperature",
                "weather_humidity",
                "weather_wind_speed",
                "weather_precipitation_amount",
                "weather_precipitation_probability",
            ]
        ]

    if not feature_cols:
        logger.warning("No feature columns found for lag features")
        return df

    # Ensure sorted by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Create lag features using merge_asof for efficiency
    df_indexed = df.set_index(timestamp_col).sort_index()

    for col in feature_cols:
        if col not in df_indexed.columns:
            continue

        for lag_hours in lag_windows:
            lag_col = f"{col}_lag_{lag_hours}h"

            # Create shifted dataframe
            df_shifted = df_indexed[[col]].copy()
            df_shifted.index = df_shifted.index - pd.Timedelta(hours=lag_hours)
            df_shifted.columns = [lag_col]

            # Merge back using merge_asof (backward direction = nearest earlier value)
            df_indexed = pd.merge_asof(
                df_indexed,
                df_shifted,
                left_index=True,
                right_index=True,
                direction="backward",
            )

    # Reset index
    df = df_indexed.reset_index()

    logger.info(
        f"Added lag features for {len(feature_cols)} features with windows {lag_windows}"
    )
    return df


def add_rolling_statistics(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    feature_cols: Optional[List[str]] = None,
    rolling_windows: List[int] = [3, 6, 12, 24],
) -> pd.DataFrame:
    """
    Add rolling statistics (mean, std, min, max) for specified columns

    Args:
        df: DataFrame sorted by timestamp
        timestamp_col: Name of timestamp column
        feature_cols: List of feature columns to create rolling stats for
        rolling_windows: List of rolling windows in hours

    Returns:
        DataFrame with rolling statistics added
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not found, skipping rolling stats"
        )
        return df

    # Auto-detect numeric weather features if not specified
    if feature_cols is None:
        feature_cols = [
            c
            for c in df.columns
            if c.startswith("weather_")
            and c
            in [
                "weather_temperature",
                "weather_humidity",
                "weather_wind_speed",
                "weather_precipitation_amount",
                "weather_precipitation_probability",
            ]
        ]

    if not feature_cols:
        logger.warning("No feature columns found for rolling statistics")
        return df

    # Ensure sorted by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Set timestamp as index for rolling operations
    df_indexed = df.set_index(timestamp_col)

    # Create rolling statistics
    for col in feature_cols:
        if col not in df.columns:
            continue

        for window_hours in rolling_windows:
            window_str = f"{window_hours}h"

            # Rolling mean
            df_indexed[f"{col}_rolling_mean_{window_str}"] = (
                df_indexed[col].rolling(window=window_str, min_periods=1).mean()
            )

            # Rolling std
            df_indexed[f"{col}_rolling_std_{window_str}"] = (
                df_indexed[col].rolling(window=window_str, min_periods=1).std()
            )

            # Rolling min
            df_indexed[f"{col}_rolling_min_{window_str}"] = (
                df_indexed[col].rolling(window=window_str, min_periods=1).min()
            )

            # Rolling max
            df_indexed[f"{col}_rolling_max_{window_str}"] = (
                df_indexed[col].rolling(window=window_str, min_periods=1).max()
            )

    # Reset index
    df = df_indexed.reset_index()

    logger.info(
        f"Added rolling statistics for {len(feature_cols)} features with windows {rolling_windows}"
    )
    return df


def add_spatial_neighbor_aggregates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    spatial_radius_km: float = 5.0,
    time_windows: List[int] = [1, 24],
) -> pd.DataFrame:
    """
    Add spatial neighbor aggregates (count, averages) within radius and time windows

    Args:
        df: DataFrame with lat/lon and timestamp
        timestamp_col: Name of timestamp column
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        spatial_radius_km: Radius in km for spatial neighbor search
        time_windows: List of time windows in hours

    Returns:
        DataFrame with spatial neighbor aggregates added
    """
    df = df.copy()

    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning("Lat/lon columns not found, skipping spatial aggregates")
        return df

    if timestamp_col not in df.columns:
        logger.warning(
            f"Timestamp column '{timestamp_col}' not found, skipping spatial aggregates"
        )
        return df

    # Ensure sorted by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Initialize new columns
    for window_hours in time_windows:
        df[f"nearby_incidents_count_{window_hours}h"] = 0
        df[f"nearby_avg_temperature_{window_hours}h"] = np.nan
        df[f"nearby_avg_humidity_{window_hours}h"] = np.nan

    # Compute spatial aggregates (vectorized where possible)
    logger.info(f"Computing spatial aggregates for {len(df)} records...")

    for i in range(len(df)):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(df)} records...")

        current_lat = df.loc[i, lat_col]
        current_lon = df.loc[i, lon_col]
        current_time = df.loc[i, timestamp_col]

        # Skip if invalid coordinates
        if pd.isna(current_lat) or pd.isna(current_lon):
            continue

        for window_hours in time_windows:
            time_threshold = current_time - pd.Timedelta(hours=window_hours)

            # Find nearby incidents within time window
            time_mask = df[timestamp_col] >= time_threshold
            time_mask &= df[timestamp_col] < current_time
            time_mask &= df.index != i  # Exclude self

            nearby_df = df[time_mask].copy()

            if len(nearby_df) == 0:
                continue

            # Calculate distances
            distances = nearby_df.apply(
                lambda row: haversine_distance(
                    current_lat, current_lon, row[lat_col], row[lon_col]
                ),
                axis=1,
            )

            # Filter by spatial radius
            within_radius = nearby_df[distances <= spatial_radius_km]

            if len(within_radius) > 0:
                # Count
                df.loc[i, f"nearby_incidents_count_{window_hours}h"] = len(
                    within_radius
                )

                # Average weather features
                if "weather_temperature" in within_radius.columns:
                    temps = within_radius["weather_temperature"].dropna()
                    if len(temps) > 0:
                        df.loc[i, f"nearby_avg_temperature_{window_hours}h"] = (
                            temps.mean()
                        )

                if "weather_humidity" in within_radius.columns:
                    humidity = within_radius["weather_humidity"].dropna()
                    if len(humidity) > 0:
                        df.loc[i, f"nearby_avg_humidity_{window_hours}h"] = (
                            humidity.mean()
                        )

    logger.info(f"Added spatial neighbor aggregates for windows {time_windows}")
    return df


def enrich_X_features(
    X_input_path: str = "gold-cpu-traffic/X_features.parquet",
    merged_intermediate_path: str = "gold-cpu-traffic/merged_intermediate.parquet",
    rich_output_dir: str = "rich-gold-cpu-traffic",
    lag_windows: List[int] = [1, 3, 6, 12, 24],
    rolling_windows: List[int] = [3, 6, 12, 24],
    spatial_radius_km: float = 5.0,
    time_windows: List[int] = [1, 24],
    lag_feature_cols: Optional[List[str]] = None,
    rolling_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Enrich base X features with lags, rolling stats, spatial aggregates, and extended temporal features

    Args:
        X_input_path: Path to base X_features.parquet
        merged_intermediate_path: Path to merged_intermediate.parquet (for timestamp)
        rich_output_dir: Output directory for rich features
        lag_windows: List of lag windows in hours
        rolling_windows: List of rolling windows in hours
        spatial_radius_km: Radius in km for spatial neighbor search
        time_windows: List of time windows in hours for spatial aggregates
        lag_feature_cols: Specific columns for lag features (None = auto-detect)
        rolling_feature_cols: Specific columns for rolling stats (None = auto-detect)

    Returns:
        Enriched X_features DataFrame
    """
    logger.info("=" * 70)
    logger.info("RICH FEATURE ENGINEERING: Enriching X Features")
    logger.info("=" * 70)

    # Step 1: Load base X features
    logger.info("\n[Step 1/6] Loading base X features...")
    logger.info(f"  Source: {X_input_path}")
    X_base = pd.read_parquet(X_input_path)
    logger.info(
        f"    Loaded {len(X_base):,} records with {len(X_base.columns)} features"
    )

    # Step 2: Ensure timestamp/hour_ts exists in X
    logger.info("\n[Step 2/6] Ensuring timestamp exists in X...")
    
    # Check if hour_ts exists (aggregated data) or if we need to load from merged_intermediate
    if "hour_ts" in X_base.columns:
        # Aggregated data - use hour_ts as timestamp
        logger.info("  Found hour_ts in X (aggregated data)")
        X_base = X_base.copy()
        X_base["timestamp"] = pd.to_datetime(X_base["hour_ts"], utc=True)
    elif "timestamp" in X_base.columns:
        # Already has timestamp
        logger.info("  Found timestamp in X")
        X_base["timestamp"] = pd.to_datetime(X_base["timestamp"], utc=True)
    else:
        # Need to load from merged_intermediate (legacy event-level data)
        logger.info(f"  Loading timestamp from {merged_intermediate_path}")
        merged_df = pd.read_parquet(merged_intermediate_path)
        
        # Merge timestamp back into X
        # Try to merge on incident_id first, then fall back to index alignment
        if "incident_id" in merged_df.columns:
            if "incident_id" in X_base.columns:
                # Merge on incident_id
                timestamp_map = merged_df.set_index("incident_id")["timestamp"].to_dict()
                X_base["timestamp"] = X_base["incident_id"].map(timestamp_map)
                logger.info("    Merged timestamp using incident_id")
            else:
                # X_base doesn't have incident_id, try to add it from merged
                # Assume same order if lengths match
                if len(X_base) == len(merged_df):
                    X_base["timestamp"] = merged_df["timestamp"].values
                    logger.info("    Merged timestamp using index alignment")
                else:
                    logger.warning("Cannot merge timestamp - data length mismatch")
                    X_base["timestamp"] = None
        else:
            # No incident_id, assume same order
            if len(X_base) == len(merged_df):
                X_base["timestamp"] = merged_df["timestamp"].values
                logger.info("    Merged timestamp using index alignment")
            else:
                logger.warning("Cannot merge timestamp - data length mismatch")
                X_base["timestamp"] = None
        
        logger.info("    Merged timestamp into X features")

    # Step 3: Ensure proper grouping key exists
    # For aggregated data, group by h3_cell; for event-level, use incident_id or index
    if "h3_cell" in X_base.columns:
        group_key = "h3_cell"
        logger.info("  Using h3_cell as grouping key (aggregated data)")
    elif "incident_id" in X_base.columns:
        group_key = "incident_id"
        logger.info("  Using incident_id as grouping key (event-level data)")
    else:
        group_key = None
        logger.warning("  No grouping key found - lag/rolling features may be incorrect")
    
    # Step 4: Add time binning features
    logger.info("\n[Step 4/7] Adding time binning features...")
    X_enriched = add_time_binning_features(X_base.copy(), timestamp_col="timestamp")

    # Step 5: Add extended temporal features
    logger.info("\n[Step 5/7] Adding extended temporal features...")
    X_enriched = add_extended_temporal_features(X_enriched, timestamp_col="timestamp")

    # Step 6: Add lag features (group by h3_cell for aggregated data)
    logger.info("\n[Step 6/7] Adding lag features...")
    if group_key:
        # For aggregated data, compute lags per h3_cell
        X_enriched = X_enriched.sort_values([group_key, "timestamp"]).reset_index(drop=True)
        # Group by h3_cell and compute lags within each group
        lag_cols = lag_feature_cols or [
            c for c in X_enriched.columns
            if c.startswith("weather_") and c in [
                "weather_temperature", "weather_humidity", "weather_wind_speed",
                "weather_precipitation_amount", "weather_precipitation_probability"
            ]
        ]
        for col in lag_cols:
            if col in X_enriched.columns:
                for lag_hours in lag_windows:
                    X_enriched[f"{col}_lag_{lag_hours}h"] = X_enriched.groupby(group_key)[col].shift(lag_hours)
    else:
        X_enriched = add_lag_features(
            X_enriched,
            timestamp_col="timestamp",
            feature_cols=lag_feature_cols,
            lag_windows=lag_windows,
        )

    # Step 7: Add rolling statistics (group by h3_cell for aggregated data)
    logger.info("\n[Step 7/7] Adding rolling statistics...")
    if group_key:
        # For aggregated data, compute rolling stats per h3_cell
        rolling_cols = rolling_feature_cols or lag_cols if 'lag_cols' in locals() else []
        if not rolling_cols:
            rolling_cols = [
                c for c in X_enriched.columns
                if c.startswith("weather_") and c in [
                    "weather_temperature", "weather_humidity", "weather_wind_speed",
                    "weather_precipitation_amount", "weather_precipitation_probability"
                ]
            ]
        for col in rolling_cols:
            if col in X_enriched.columns:
                for window_hours in rolling_windows:
                    # Rolling window in hours (assuming hourly data)
                    X_enriched[f"{col}_rolling_mean_{window_hours}h"] = (
                        X_enriched.groupby(group_key)[col].transform(
                            lambda x: x.rolling(window=window_hours, min_periods=1).mean()
                        )
                    )
                    X_enriched[f"{col}_rolling_max_{window_hours}h"] = (
                        X_enriched.groupby(group_key)[col].transform(
                            lambda x: x.rolling(window=window_hours, min_periods=1).max()
                        )
                    )
    else:
        X_enriched = add_rolling_statistics(
            X_enriched,
            timestamp_col="timestamp",
            feature_cols=rolling_feature_cols,
            rolling_windows=rolling_windows,
        )

    # Step 7: Add spatial neighbor aggregates
    logger.info("\n[Step 7/7] Adding spatial neighbor aggregates...")
    X_enriched = add_spatial_neighbor_aggregates(
        X_enriched,
        timestamp_col="timestamp",
        spatial_radius_km=spatial_radius_km,
        time_windows=time_windows,
    )

    # Keep hour_ts if it exists (needed for CV and lag features), but drop timestamp
    if "timestamp" in X_enriched.columns and "hour_ts" in X_enriched.columns:
        X_enriched = X_enriched.drop(columns=["timestamp"])
    elif "timestamp" in X_enriched.columns:
        # Keep timestamp if hour_ts doesn't exist (legacy event-level data)
        pass

    logger.info("\n" + "=" * 70)
    logger.info("Rich feature engineering complete!")
    logger.info(f"  Base features: {len(X_base.columns)}")
    logger.info(f"  Enriched features: {len(X_enriched.columns)}")
    logger.info(
        f"  New features added: {len(X_enriched.columns) - len(X_base.columns)}"
    )
    logger.info("=" * 70)

    # Save enriched features
    rich_path = Path(rich_output_dir)
    rich_path.mkdir(parents=True, exist_ok=True)

    X_file = rich_path / "X_features.parquet"
    X_enriched.to_parquet(X_file, index=False)
    logger.info(f"\n✓ Saved enriched feature matrix (X) to {X_file}")
    logger.info(f"  Total records: {len(X_enriched):,}")
    logger.info(f"  Total features: {len(X_enriched.columns)}")

    # Save metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "row_count": len(X_enriched),
        "base_feature_count": len(X_base.columns),
        "enriched_feature_count": len(X_enriched.columns),
        "new_features_added": len(X_enriched.columns) - len(X_base.columns),
        "feature_columns": list(X_enriched.columns),
        "enrichment_parameters": {
            "lag_windows": lag_windows,
            "rolling_windows": rolling_windows,
            "spatial_radius_km": spatial_radius_km,
            "time_windows": time_windows,
        },
        "pipeline": "rich_X_features",
        "base_source": X_input_path,
    }

    metadata_file = rich_path / "X_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"✓ Saved metadata to {metadata_file}")

    return X_enriched


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the enrichment pipeline
    X_enriched = enrich_X_features()

    print("\nRich feature engineering complete!")
    print(
        f"  Enriched X: {X_enriched.shape[0]:,} records, {X_enriched.shape[1]} features"
    )
    print("  Check ./rich-gold-cpu-traffic/ directory for results.")
