"""
Feature Engineering Pipeline for DGXB
Includes fuzzy merge, feature transformation, and zero-shot classification for target variable
"""

import pandas as pd
import numpy as np
import h3
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
import logging
import json
import pickle
from typing import Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Zero-shot classification categories
INCIDENT_CATEGORIES = [
    "Collision or Crash",
    "Vehicle Breakdown",
    "Traffic Hazard",
    "Road Closure",
    "Other"
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points in km"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * R


def get_h3_cell(lat: float, lon: float, resolution: int) -> str:
    """
    Get H3 cell for coordinates, handling both h3-py v3 and v4+ APIs
    """
    try:
        # Convert to float
        lat = float(lat)
        lon = float(lon)
        
        # Validate
        if pd.isna(lat) or pd.isna(lon) or np.isnan(lat) or np.isnan(lon):
            return None
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None
        
        # Try new API first (h3-py v4+)
        try:
            return h3.latlng_to_cell(lat, lon, resolution)
        except AttributeError:
            # Fallback to old API (h3-py v3)
            return h3.geo_to_h3(lat, lon, resolution)
    except Exception as e:
        logger.debug(f"H3 cell creation failed for ({lat}, {lon}): {e}")
        return None


def get_h3_neighbors(cell: str, k: int = 1) -> set:
    """
    Get H3 neighbors, handling both h3-py v3 and v4+ APIs
    """
    if cell is None:
        return set()
    
    neighbors = {cell}
    if k > 0:
        try:
            # Try new API first (h3-py v4+)
            try:
                neighbor_cells = h3.grid_ring(cell, k)
                neighbors.update(neighbor_cells)
            except AttributeError:
                # Fallback to old API (h3-py v3)
                neighbor_cells = h3.k_ring(cell, k)
                neighbors.update(neighbor_cells)
        except Exception as e:
            logger.debug(f"H3 neighbor lookup failed for {cell}: {e}")
    
    return neighbors


def prepare_traffic_data(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare traffic data, fixing lat/lon columns if needed
    """
    traffic_clean = traffic_df.dropna(subset=['timestamp']).copy()
    
    # Check if lat/lon are valid, otherwise use location_lat/location_lon
    if 'lat' in traffic_clean.columns and 'lon' in traffic_clean.columns:
        # Check if lat values are reasonable (Austin is ~30)
        lat_valid = (traffic_clean['lat'].between(25, 35)) & traffic_clean['lat'].notna()
        if lat_valid.sum() < len(traffic_clean) * 0.5:  # Less than 50% valid
            # Try location_lat/location_lon
            if 'location_lat' in traffic_clean.columns and 'location_lon' in traffic_clean.columns:
                logger.info("Using location_lat/location_lon instead of lat/lon")
                traffic_clean['lat'] = traffic_clean['location_lat']
                traffic_clean['lon'] = traffic_clean['location_lon']
    
    # Filter for valid lat/lon
    traffic_clean = traffic_clean.dropna(subset=['lat', 'lon']).copy()
    
    # Filter for reasonable coordinate ranges (Austin area)
    traffic_clean = traffic_clean[
        traffic_clean['lat'].between(25, 35) &
        traffic_clean['lon'].between(-100, -95)
    ].copy()
    
    # Ensure timestamp is datetime
    traffic_clean['timestamp'] = pd.to_datetime(traffic_clean['timestamp'])
    
    return traffic_clean


def fuzzy_merge_h3(
    traffic_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    h3_resolution: int = 9,
    k_ring_size: int = 1,
    max_spatial_km: float = 50,
    max_time_hours: float = 1,
    spatial_weight: float = 1.0,
    temporal_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Optimized H3-based fuzzy merge for thousands of records.
    
    Uses H3 spatial indexing to group records by geographic cells,
    then finds closest weather by time within spatial proximity.
    
    Args:
        traffic_df: DataFrame with [timestamp, lat, lon, ...]
        weather_df: DataFrame with [timestamp, lat, lon, ...]
        h3_resolution: H3 resolution (0-15). Higher = smaller cells
        k_ring_size: H3 k-ring size for neighbor search
        max_spatial_km: Maximum spatial distance in km
        max_time_hours: Maximum time difference in hours
        spatial_weight: Weight for spatial distance in combined metric
        temporal_weight: Weight for temporal distance in combined metric
    
    Returns:
        Merged DataFrame with weather columns prefixed with 'weather_'
    """
    # Prepare data
    traffic_clean = prepare_traffic_data(traffic_df)
    weather_clean = weather_df.dropna(subset=['lat', 'lon', 'timestamp']).copy()
    weather_clean['timestamp'] = pd.to_datetime(weather_clean['timestamp'])
    
    if len(weather_clean) == 0:
        logger.warning("No valid weather data")
        return traffic_df.copy()
    
    if len(traffic_clean) == 0:
        logger.warning("No valid traffic data")
        return traffic_df.copy()
    
    logger.info(f"Processing {len(traffic_clean)} traffic records against {len(weather_clean)} weather records...")
    
    # H3 resolution to approximate cell size (km)
    h3_cell_sizes = {7: 5.2, 8: 1.8, 9: 0.5, 10: 0.2, 11: 0.07}
    cell_size_km = h3_cell_sizes.get(h3_resolution, 0.5)
    logger.info(f"Using H3 resolution {h3_resolution} (cell size ~{cell_size_km:.2f}km)")
    
    # Create H3 cells
    traffic_clean['h3_cell'] = traffic_clean.apply(
        lambda row: get_h3_cell(row['lat'], row['lon'], h3_resolution),
        axis=1
    )
    weather_clean['h3_cell'] = weather_clean.apply(
        lambda row: get_h3_cell(row['lat'], row['lon'], h3_resolution),
        axis=1
    )
    
    # Filter valid H3 cells
    traffic_valid = traffic_clean[traffic_clean['h3_cell'].notna()].copy()
    weather_valid = weather_clean[weather_clean['h3_cell'].notna()].copy()
    
    logger.info(f"Valid H3 cells: {len(traffic_valid)} traffic, {len(weather_valid)} weather")
    
    if len(traffic_valid) == 0 or len(weather_valid) == 0:
        logger.warning("No valid H3 cells created. Returning original traffic data.")
        return traffic_df.copy()
    
    # Build H3 index: map H3 cell -> list of weather record indices
    h3_weather_index = {}
    for idx, row in weather_valid.iterrows():
        cell = row['h3_cell']
        if cell not in h3_weather_index:
            h3_weather_index[cell] = []
        h3_weather_index[cell].append(idx)
    
    # Cache for neighbors
    h3_neighbor_cache = {}
    
    def get_h3_neighbors_cached(cell: str) -> set:
        """Get H3 neighbors with caching"""
        if cell is None:
            return set()
        if cell not in h3_neighbor_cache:
            neighbors = get_h3_neighbors(cell, k_ring_size)
            h3_neighbor_cache[cell] = neighbors
        return h3_neighbor_cache[cell]
    
    # Process traffic records
    results = []
    no_match_count = 0
    match_count = 0
    
    batch_size = 1000
    for batch_start in range(0, len(traffic_valid), batch_size):
        batch_end = min(batch_start + batch_size, len(traffic_valid))
        traffic_batch = traffic_valid.iloc[batch_start:batch_end]
        
        for idx, traffic_row in traffic_batch.iterrows():
            neighbor_cells = get_h3_neighbors_cached(traffic_row['h3_cell'])
            
            # Collect candidate weather records
            candidate_indices = []
            for cell in neighbor_cells:
                if cell in h3_weather_index:
                    candidate_indices.extend(h3_weather_index[cell])
            
            if len(candidate_indices) == 0:
                results.append(traffic_row.to_dict())
                no_match_count += 1
                continue
            
            candidate_weather = weather_valid.loc[candidate_indices].copy()
            
            # Calculate spatial distances
            candidate_weather['spatial_distance_km'] = candidate_weather.apply(
                lambda w_row: haversine_distance(
                    traffic_row['lat'], traffic_row['lon'],
                    w_row['lat'], w_row['lon']
                ),
                axis=1
            )
            
            # Filter by max spatial distance
            candidate_weather = candidate_weather[
                candidate_weather['spatial_distance_km'] <= max_spatial_km
            ]
            
            if len(candidate_weather) == 0:
                results.append(traffic_row.to_dict())
                no_match_count += 1
                continue
            
            # Filter by temporal proximity
            time_diffs = abs(candidate_weather['timestamp'] - traffic_row['timestamp'])
            time_mask = time_diffs <= pd.Timedelta(hours=max_time_hours)
            
            if not time_mask.any():
                results.append(traffic_row.to_dict())
                no_match_count += 1
                continue
            
            # Get temporally valid candidates
            valid_candidates = candidate_weather[time_mask].copy()
            valid_time_diffs = time_diffs[time_mask]
            
            # Calculate combined distance metric (weighted)
            time_equivalent_km = (valid_time_diffs.dt.total_seconds() / 3600.0) * 50.0
            combined_distance = (
                spatial_weight * valid_candidates['spatial_distance_km'].values +
                temporal_weight * time_equivalent_km.values
            )
            
            # Select best match (minimum combined distance)
            best_idx = combined_distance.argmin()
            best_weather = valid_candidates.iloc[best_idx]
            
            # Merge weather data
            merged_row = traffic_row.to_dict()
            for col in weather_df.columns:
                if col not in ['timestamp', 'lat', 'lon', 'h3_cell']:
                    merged_row[f'weather_{col}'] = best_weather[col]
            
            # Add metadata
            merged_row['weather_distance_km'] = best_weather['spatial_distance_km']
            merged_row['weather_time_diff_hours'] = valid_time_diffs.iloc[best_idx].total_seconds() / 3600.0
            merged_row['weather_combined_score'] = combined_distance[best_idx]
            merged_row['weather_h3_cell'] = best_weather['h3_cell']
            
            results.append(merged_row)
            match_count += 1
        
        if (batch_end % 1000 == 0) or (batch_end == len(traffic_valid)):
            logger.info(f"Processed {batch_end}/{len(traffic_valid)} records...")
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Merge back with original traffic (preserve rows that were filtered out)
    final_df = traffic_df.copy()
    for col in result_df.columns:
        if col not in traffic_df.columns:
            final_df[col] = None
            if len(result_df) == len(traffic_valid):
                final_df.loc[traffic_valid.index, col] = result_df[col].values
    
    logger.info(f"Merge complete:")
    logger.info(f"  - Matched: {match_count} records")
    logger.info(f"  - No match: {no_match_count} records")
    if len(traffic_valid) > 0:
        logger.info(f"  - Match rate: {match_count/len(traffic_valid)*100:.1f}%")
    else:
        logger.warning("  - Match rate: N/A (no valid records)")
    
    return final_df


def setup_zero_shot_classifier(model_name: str = "microsoft/deberta-v3-base"):
    """
    Setup zero-shot classification pipeline
    
    Args:
        model_name: Hugging Face model name for zero-shot classification
    
    Returns:
        Zero-shot classifier pipeline
    """
    try:
        from transformers import pipeline
        
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except ImportError:
        logger.error("transformers library not installed. Install with: pip install transformers torch")
        raise
    except Exception as e:
        logger.error(f"Failed to load zero-shot classifier: {e}")
        raise


def classify_incident_zero_shot(
    descriptions: pd.Series,
    categories: list = None,
    classifier=None,
    model_name: str = "microsoft/deberta-v3-base",
    cache_path: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Classify incident descriptions using zero-shot classification
    
    Args:
        descriptions: Series of incident descriptions
        categories: List of category labels (defaults to INCIDENT_CATEGORIES)
        classifier: Pre-loaded classifier (optional)
        model_name: Model to use if classifier not provided
        cache_path: Path to cache results (optional)
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with original description, predicted category, and confidence
    """
    if categories is None:
        categories = INCIDENT_CATEGORIES
    
    # Check cache
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached classifications from {cache_path}")
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(descriptions):
            logger.info("Using cached zero-shot classifications")
            return cached
    
    # Setup classifier if needed
    if classifier is None:
        logger.info(f"Loading zero-shot classifier: {model_name}")
        classifier = setup_zero_shot_classifier(model_name)
    
    results = []
    
    logger.info(f"Classifying {len(descriptions)} descriptions into {len(categories)} categories...")
    
    # Process in batches for efficiency
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions.iloc[i:i+batch_size].tolist()
        
        # Classify batch
        try:
            batch_results = classifier(batch, categories)
            
            # Handle both single and batch results
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
            
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error classifying batch {i}: {e}")
            # Fallback: classify one at a time
            for desc in batch:
                try:
                    result = classifier([desc], categories)[0]
                    results.append(result)
                except Exception as e2:
                    logger.error(f"Error classifying '{desc}': {e2}")
                    # Default to "Other" if classification fails
                    results.append({
                        'labels': categories,
                        'scores': [0.0] * (len(categories) - 1) + [1.0]
                    })
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"  Processed {min(i + batch_size, len(descriptions))}/{len(descriptions)}")
    
    # Extract predictions
    df_results = pd.DataFrame({
        'description': descriptions.values,
        'predicted_category': [r['labels'][0] for r in results],
        'confidence': [r['scores'][0] for r in results],
        'all_scores': [dict(zip(r['labels'], r['scores'])) for r in results]
    })
    
    # Cache results
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(cache_path, index=False)
        logger.info(f"Cached classifications to {cache_path}")
    
    return df_results


def prepare_ml_features(
    df: pd.DataFrame,
    target_col: str = 'description',
    drop_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, object]:
    """
    Prepare features for ML by dropping artifacts and transforming features
    
    Args:
        df: Input DataFrame with merged traffic and weather data
        target_col: Name of target column
        drop_artifacts: Whether to drop artifact features
    
    Returns:
        X_features: Feature DataFrame (ready for ML)
        y_raw: Raw target (original descriptions)
        y_processed: Processed target (zero-shot classified categories)
        y_encoded: Encoded target (integers for ML)
        label_encoder: Fitted LabelEncoder
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    # Separate target variable FIRST
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    y_raw = df[target_col].copy()
    
    # Drop artifact features
    artifacts_to_drop = [
        'location',                    # JSON structure
        'location_lon', 'location_lat', # Duplicates
        'traffic_report_status_date_time', # Duplicate timestamp
        'weather_h3_cell',             # Merge artifact
        'weather_combined_score',      # Internal metric
        'weather_short_forecast',      # Text (drop unless using NLP)
        'weather_detailed_forecast',   # Text (drop unless using NLP)
        'h3_cell',                     # Processing artifact
        'incident_id',                 # Identifier (keep for tracking but drop from features)
        target_col,                    # Target variable (separate it)
    ]
    
    if drop_artifacts:
        df = df.drop(columns=[col for col in artifacts_to_drop if col in df.columns])
    
    # Extract temporal features from timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop original timestamp
        df = df.drop(columns=['timestamp'])
    
    # Transform wind direction (circular feature)
    if 'weather_wind_direction' in df.columns:
        # Convert to numeric and handle NaNs properly
        wind_dir = pd.to_numeric(df['weather_wind_direction'], errors='coerce')
        df['wind_direction_sin'] = np.sin(np.deg2rad(wind_dir))
        df['wind_direction_cos'] = np.cos(np.deg2rad(wind_dir))
        df = df.drop(columns=['weather_wind_direction'])
    
    # Encode categorical features
    categorical_cols = ['status', 'agency', 'location_type',
                       'weather_weather_code', 'weather_data_source']
    
    for col in categorical_cols:
        if col in df.columns:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    
    # Convert binary features
    if 'weather_is_daytime' in df.columns:
        # Fill NaN with 0 (default to nighttime if unknown)
        df['weather_is_daytime'] = df['weather_is_daytime'].fillna(0).infer_objects(copy=False).astype(int)
    
    # Handle missing weather data (create indicator)
    weather_cols = [c for c in df.columns if c.startswith('weather_') and 
                   c not in ['weather_data_source'] and 
                   not c.startswith('weather_data_source_')]
    if weather_cols:
        df['has_weather_data'] = df[weather_cols[0]].notna().astype(int)
    
    # X is all remaining columns (features)
    X = df
    
    # For now, return y_raw (will be processed by zero-shot classification separately)
    # y_processed and y_encoded will be set by zero-shot classification
    y_processed = None
    y_encoded = None
    label_encoder = None
    
    return X, y_raw, y_processed, y_encoded, label_encoder


def merge_and_save_X_features(
    traffic_silver_path: str = "silver-cpu-traffic/data_silver.parquet",
    weather_silver_path: str = "silver-cpu-weather/data_silver.parquet",
    gold_output_dir: str = "gold-cpu-traffic",
    h3_resolution: int = 9,
    k_ring_size: int = 1,
    max_spatial_km: float = 50,
    max_time_hours: float = 1,
    spatial_weight: float = 1.0,
    temporal_weight: float = 0.5,
) -> pd.DataFrame:
    """
    X Pipeline: Load silver data, perform fuzzy merge, apply feature engineering, and save X features
    
    This pipeline generates the feature matrix (X) only, without target processing.
    Fast to run and can be tested independently.
    
    Args:
        traffic_silver_path: Path to traffic silver data
        weather_silver_path: Path to weather silver data
        gold_output_dir: Output directory for gold layer
        h3_resolution: H3 resolution for spatial indexing
        k_ring_size: H3 k-ring size
        max_spatial_km: Maximum spatial distance for matching
        max_time_hours: Maximum time difference for matching
        spatial_weight: Weight for spatial distance
        temporal_weight: Weight for temporal distance
    
    Returns:
        X_features DataFrame (engineered features ready for ML)
    """
    logger.info("=" * 70)
    logger.info("X PIPELINE: Merge + Feature Engineering")
    logger.info("=" * 70)
    
    # Step 1: Load silver data
    logger.info(f"\n[Step 1/3] Loading silver data...")
    logger.info(f"  Traffic: {traffic_silver_path}")
    traffic_df = pd.read_parquet(traffic_silver_path)
    logger.info(f"    Loaded {len(traffic_df)} traffic records")
    
    logger.info(f"  Weather: {weather_silver_path}")
    weather_df = pd.read_parquet(weather_silver_path)
    logger.info(f"    Loaded {len(weather_df)} weather records")
    
    # Step 2: Perform fuzzy merge
    logger.info(f"\n[Step 2/3] Performing fuzzy merge...")
    merged_df = fuzzy_merge_h3(
        traffic_df=traffic_df,
        weather_df=weather_df,
        h3_resolution=h3_resolution,
        k_ring_size=k_ring_size,
        max_spatial_km=max_spatial_km,
        max_time_hours=max_time_hours,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )
    logger.info(f"  Merge complete: {len(merged_df)} records")
    
    # Save merged data as intermediate file (needed for Y pipeline)
    gold_path = Path(gold_output_dir)
    gold_path.mkdir(parents=True, exist_ok=True)
    merged_intermediate_path = gold_path / "merged_intermediate.parquet"
    merged_df.to_parquet(merged_intermediate_path, index=False)
    logger.info(f"  Saved merged data to {merged_intermediate_path} (for Y pipeline)")
    
    # Step 3: Feature engineering (X only - no target)
    logger.info(f"\n[Step 3/3] Feature engineering (X features)...")
    X_features, y_raw, _, _, _ = prepare_ml_features(
        df=merged_df,
        target_col='description',
        drop_artifacts=True,
    )
    
    logger.info(f"  Feature engineering complete")
    logger.info(f"    Original columns: {len(merged_df.columns)}")
    logger.info(f"    Feature columns: {len(X_features.columns)}")
    
    # Save X features
    X_file = gold_path / "X_features.parquet"
    X_features.to_parquet(X_file, index=False)
    logger.info(f"\n✓ Saved feature matrix (X) to {X_file}")
    logger.info(f"  Total records: {len(X_features):,}")
    logger.info(f"  Total features: {len(X_features.columns)}")
    
    # Save X metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "row_count": len(X_features),
        "feature_columns": list(X_features.columns),
        "merge_parameters": {
            "h3_resolution": h3_resolution,
            "k_ring_size": k_ring_size,
            "max_spatial_km": max_spatial_km,
            "max_time_hours": max_time_hours,
            "spatial_weight": spatial_weight,
            "temporal_weight": temporal_weight,
        },
        "pipeline": "X_features",
        "date_range": {
            "start": merged_df['timestamp'].min().isoformat() if 'timestamp' in merged_df.columns else None,
            "end": merged_df['timestamp'].max().isoformat() if 'timestamp' in merged_df.columns else None,
        },
    }
    
    metadata_file = gold_path / "X_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"✓ Saved X metadata to {metadata_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("X Pipeline complete!")
    logger.info("=" * 70)
    logger.info("Note: X_features.parquet is ready for both training and inference")
    logger.info("      Run Y pipeline separately to generate y_target.parquet")
    logger.info("=" * 70)
    
    return X_features


def prepare_and_save_y_target(
    merged_data_path: str = "gold-cpu-traffic/merged_intermediate.parquet",
    gold_output_dir: str = "gold-cpu-traffic",
    zero_shot_model: str = "microsoft/deberta-v3-base",
    incident_categories: list = None,
    use_cached_classifications: bool = True,
) -> pd.DataFrame:
    """
    Y Pipeline: Load merged data, perform zero-shot classification, and save y target
    
    This pipeline generates the target variable (y) with zero-shot classification.
    Can be run independently after X pipeline completes.
    
    Args:
        merged_data_path: Path to merged intermediate data (from X pipeline)
        gold_output_dir: Output directory for gold layer
        zero_shot_model: Model name for zero-shot classification
        incident_categories: List of categories for zero-shot (defaults to INCIDENT_CATEGORIES)
        use_cached_classifications: Whether to use cached zero-shot results if available
    
    Returns:
        y_target DataFrame (target variable with classifications)
    """
    if incident_categories is None:
        incident_categories = INCIDENT_CATEGORIES
    
    logger.info("=" * 70)
    logger.info("Y PIPELINE: Zero-Shot Classification + Target Preparation")
    logger.info("=" * 70)
    
    # Step 1: Load merged data
    logger.info(f"\n[Step 1/3] Loading merged data...")
    logger.info(f"  Source: {merged_data_path}")
    merged_df = pd.read_parquet(merged_data_path)
    logger.info(f"    Loaded {len(merged_df)} records")
    
    # Step 2: Zero-shot classification for target
    logger.info(f"\n[Step 2/3] Zero-shot classification for target variable...")
    gold_path = Path(gold_output_dir)
    cache_path = gold_path / "zero_shot_classifications.parquet" if use_cached_classifications else None
    
    try:
        classifications = classify_incident_zero_shot(
            descriptions=merged_df['description'],
            categories=incident_categories,
            model_name=zero_shot_model,
            cache_path=str(cache_path) if cache_path else None,
        )
        
        # Add classified categories to merged_df
        merged_df['incident_category'] = classifications['predicted_category']
        merged_df['incident_category_confidence'] = classifications['confidence']
        
        logger.info(f"  Classification complete")
        logger.info(f"  Category distribution:")
        for cat, count in merged_df['incident_category'].value_counts().items():
            logger.info(f"    {cat}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    except Exception as e:
        logger.warning(f"  Zero-shot classification failed: {e}")
        logger.warning(f"  Continuing without classification. Install transformers: pip install transformers torch")
        merged_df['incident_category'] = "Other"
        merged_df['incident_category_confidence'] = 0.0
    
    # Step 3: Prepare target variable (y)
    logger.info(f"\n[Step 3/3] Preparing target variable (y)...")
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(merged_df['incident_category'])
    
    # Create y DataFrame with all target-related information
    y_target = pd.DataFrame({
        'description': merged_df['description'],  # Original text
        'incident_category': merged_df['incident_category'],  # Zero-shot classified
        'incident_category_encoded': y_encoded,  # Encoded for ML
        'incident_category_confidence': merged_df['incident_category_confidence'],  # Confidence score
    })
    
    # Add incident_id for tracking if it exists
    if 'incident_id' in merged_df.columns:
        y_target['incident_id'] = merged_df['incident_id']
    
    logger.info(f"  Target preparation complete")
    logger.info(f"    Target records: {len(y_target):,}")
    logger.info(f"    Category distribution:")
    for cat, count in merged_df['incident_category'].value_counts().items():
        logger.info(f"      {cat}: {count} ({count/len(merged_df)*100:.1f}%)")
    
    # Save y target
    gold_path.mkdir(parents=True, exist_ok=True)
    y_file = gold_path / "y_target.parquet"
    y_target.to_parquet(y_file, index=False)
    logger.info(f"\n✓ Saved target variable (y) to {y_file}")
    logger.info(f"  Total records: {len(y_target):,}")
    logger.info(f"  Target columns: {list(y_target.columns)}")
    
    # Save label encoder (for inference to decode predictions)
    encoder_path = gold_path / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"✓ Saved label encoder to {encoder_path}")
    
    # Save Y metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "row_count": len(y_target),
        "target_columns": list(y_target.columns),
        "feature_engineering": {
            "zero_shot_model": zero_shot_model,
            "incident_categories": incident_categories,
            "target_column": "description",
            "target_encoded_column": "incident_category_encoded",
        },
        "target_distribution": merged_df['incident_category'].value_counts().to_dict(),
        "pipeline": "y_target",
    }
    
    metadata_file = gold_path / "y_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"✓ Saved Y metadata to {metadata_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Y Pipeline complete!")
    logger.info("=" * 70)
    logger.info("Note: y_target.parquet is for training only")
    logger.info("=" * 70)
    
    return y_target


def merge_and_save_to_gold(
    traffic_silver_path: str = "silver-cpu-traffic/data_silver.parquet",
    weather_silver_path: str = "silver-cpu-weather/data_silver.parquet",
    gold_output_dir: str = "gold-cpu-traffic",
    h3_resolution: int = 9,
    k_ring_size: int = 1,
    max_spatial_km: float = 50,
    max_time_hours: float = 1,
    spatial_weight: float = 1.0,
    temporal_weight: float = 0.5,
    zero_shot_model: str = "microsoft/deberta-v3-base",
    incident_categories: list = None,
    use_cached_classifications: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combined pipeline: Run both X and Y pipelines sequentially
    
    This is a convenience function that runs both pipelines.
    For faster iteration, use merge_and_save_X_features() and prepare_and_save_y_target() separately.
    
    Args:
        traffic_silver_path: Path to traffic silver data
        weather_silver_path: Path to weather silver data
        gold_output_dir: Output directory for gold layer
        h3_resolution: H3 resolution for spatial indexing
        k_ring_size: H3 k-ring size
        max_spatial_km: Maximum spatial distance for matching
        max_time_hours: Maximum time difference for matching
        spatial_weight: Weight for spatial distance
        temporal_weight: Weight for temporal distance
        zero_shot_model: Model name for zero-shot classification
        incident_categories: List of categories for zero-shot (defaults to INCIDENT_CATEGORIES)
        use_cached_classifications: Whether to use cached zero-shot results if available
    
    Returns:
        Tuple of (X_features, y_target) DataFrames
        - X_features: Engineered features (for both training and inference)
        - y_target: Target variable with classifications (for training only)
    """
    # Run X pipeline first
    X_features = merge_and_save_X_features(
        traffic_silver_path=traffic_silver_path,
        weather_silver_path=weather_silver_path,
        gold_output_dir=gold_output_dir,
        h3_resolution=h3_resolution,
        k_ring_size=k_ring_size,
        max_spatial_km=max_spatial_km,
        max_time_hours=max_time_hours,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )
    
    # Run Y pipeline
    merged_data_path = Path(gold_output_dir) / "merged_intermediate.parquet"
    y_target = prepare_and_save_y_target(
        merged_data_path=str(merged_data_path),
        gold_output_dir=gold_output_dir,
        zero_shot_model=zero_shot_model,
        incident_categories=incident_categories,
        use_cached_classifications=use_cached_classifications,
    )
    
    return X_features, y_target


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the pipeline
    X_features, y_target = merge_and_save_to_gold()
    
    print(f"\nFeature matrix (X) shape: {X_features.shape}")
    print(f"Target (y) shape: {y_target.shape}")
    print(f"Feature columns: {len(X_features.columns)}")
    print(f"Target columns: {list(y_target.columns)}")

