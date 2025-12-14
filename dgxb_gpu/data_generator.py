"""
GPU-Worthy Data Generator for Port-to-Rail Surge Forecasting

Generates realistic synthetic traffic/weather data at scale to demonstrate
GPU acceleration benefits:
- Configurable scale (100K to millions of rows)
- Realistic temporal patterns (hourly, daily, weekly seasonality)
- Spatial patterns across H3 hexagonal grid
- Weather correlations with traffic
- Incident hotspots with spatial clustering

Usage:
    python data_generator.py --rows 500000 --days 90 --h3_cells 200
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try GPU imports
try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    logger.info("GPU libraries not available, using CPU for data generation")


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation"""
    # Scale parameters
    n_days: int = 90                    # Number of days of data
    n_h3_cells: int = 200               # Number of H3 hexagonal cells (locations)
    h3_resolution: int = 8              # H3 resolution (8 = ~0.74 km² hexagons)
    
    # Port location (Long Beach, CA area)
    center_lat: float = 33.7701
    center_lon: float = -118.1937
    
    # Traffic patterns
    base_traffic_rate: float = 50.0     # Base hourly traffic count
    traffic_std: float = 15.0           # Standard deviation
    
    # Temporal patterns (amplitudes for seasonality)
    hourly_amplitude: float = 0.3       # Hour-of-day effect
    daily_amplitude: float = 0.2        # Day-of-week effect
    weekly_amplitude: float = 0.1       # Week-of-year trend
    
    # Weather parameters
    base_temp: float = 70.0             # Base temperature (°F)
    temp_daily_range: float = 15.0      # Daily temperature swing
    temp_seasonal_range: float = 20.0   # Seasonal temperature swing
    
    # Incident parameters
    incident_rate: float = 0.15         # Base incident probability
    incident_traffic_correlation: float = 0.4  # How much traffic affects incidents
    incident_weather_correlation: float = 0.2  # How much bad weather affects incidents
    hotspot_cells_pct: float = 0.1      # Percentage of cells that are hotspots
    hotspot_multiplier: float = 3.0     # Incident rate multiplier for hotspots
    
    # Output
    output_dir: str = "generated-data"
    seed: int = 42


def generate_h3_cells(config: DataGenConfig) -> List[str]:
    """Generate H3 cell IDs around the port area"""
    try:
        import h3
    except ImportError:
        raise ImportError("h3 library required: pip install h3")
    
    # Generate cells in a grid pattern around center
    cells = set()
    
    # Get center cell
    center_cell = h3.latlng_to_cell(config.center_lat, config.center_lon, config.h3_resolution)
    cells.add(center_cell)
    
    # Expand outward using k-ring
    k = 1
    while len(cells) < config.n_h3_cells:
        try:
            ring = h3.grid_disk(center_cell, k)
        except AttributeError:
            ring = h3.k_ring(center_cell, k)
        cells.update(ring)
        k += 1
    
    # Trim to exact count
    cells = list(cells)[:config.n_h3_cells]
    return cells


def generate_temporal_features(
    timestamps: pd.DatetimeIndex,
    config: DataGenConfig
) -> pd.DataFrame:
    """Generate temporal pattern multipliers"""
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Extract time components
    hour = timestamps.hour
    dow = timestamps.dayofweek  # 0=Monday
    doy = timestamps.dayofyear
    
    # Hour-of-day pattern (rush hours)
    # Peak at 8AM and 5PM
    hour_pattern = (
        config.hourly_amplitude * np.sin(2 * np.pi * (hour - 6) / 24) +
        config.hourly_amplitude * 0.5 * np.sin(4 * np.pi * (hour - 8) / 24)
    )
    
    # Day-of-week pattern (lower on weekends)
    dow_pattern = config.daily_amplitude * np.where(dow >= 5, -0.3, 0.1)
    
    # Seasonal pattern (higher in summer shipping season)
    seasonal_pattern = config.weekly_amplitude * np.sin(2 * np.pi * (doy - 80) / 365)
    
    df['hour'] = hour
    df['day_of_week'] = dow
    df['day_of_year'] = doy
    df['month'] = timestamps.month
    df['hour_pattern'] = hour_pattern
    df['dow_pattern'] = dow_pattern
    df['seasonal_pattern'] = seasonal_pattern
    df['combined_pattern'] = 1.0 + hour_pattern + dow_pattern + seasonal_pattern
    
    return df


def generate_weather_data(
    timestamps: pd.DatetimeIndex,
    config: DataGenConfig,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Generate realistic weather patterns"""
    n = len(timestamps)
    hour = timestamps.hour
    doy = timestamps.dayofyear
    
    # Temperature: daily cycle + seasonal cycle + noise
    daily_temp_cycle = config.temp_daily_range * np.sin(2 * np.pi * (hour - 6) / 24)
    seasonal_temp_cycle = config.temp_seasonal_range * np.sin(2 * np.pi * (doy - 80) / 365)
    temp_noise = rng.normal(0, 3, n)
    temperature = config.base_temp + daily_temp_cycle + seasonal_temp_cycle + temp_noise
    
    # Humidity: inverse correlation with temperature
    humidity = 60 - 0.5 * (temperature - config.base_temp) + rng.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    
    # Wind speed: log-normal distribution
    wind_speed = rng.lognormal(1.5, 0.5, n)
    wind_speed = np.clip(wind_speed, 0, 40)
    
    # Precipitation: seasonal pattern with random events
    precip_base_prob = 0.1 + 0.1 * np.sin(2 * np.pi * (doy - 350) / 365)  # Higher in winter
    precip_occurs = rng.random(n) < precip_base_prob
    precip_amount = np.where(precip_occurs, rng.exponential(0.2, n), 0)
    precip_prob = np.where(precip_occurs, rng.uniform(50, 100, n), rng.uniform(0, 30, n))
    
    # Dewpoint: derived from temp and humidity
    dewpoint = temperature - ((100 - humidity) / 5)
    
    return pd.DataFrame({
        'weather_temperature': temperature,
        'weather_humidity': humidity,
        'weather_wind_speed': wind_speed,
        'weather_precipitation_amount': precip_amount,
        'weather_precipitation_probability': precip_prob,
        'weather_dewpoint': dewpoint,
    })


def generate_traffic_incidents(
    df: pd.DataFrame,
    h3_cells: List[str],
    config: DataGenConfig,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Generate traffic counts and incident data with spatial patterns"""
    n = len(df)
    
    # Identify hotspot cells (higher incident rates)
    n_hotspots = int(len(h3_cells) * config.hotspot_cells_pct)
    hotspot_cells = set(rng.choice(h3_cells, n_hotspots, replace=False))
    
    # Cell-specific baseline (some cells are busier)
    cell_baselines = {cell: rng.uniform(0.7, 1.3) for cell in h3_cells}
    
    results = []
    for idx, row in df.iterrows():
        cell = row['h3_cell']
        
        # Traffic count: base * patterns * cell_factor * noise
        cell_factor = cell_baselines[cell]
        pattern_factor = row['combined_pattern']
        
        traffic_count = (
            config.base_traffic_rate * 
            cell_factor * 
            pattern_factor * 
            rng.lognormal(0, 0.2)
        )
        traffic_count = max(0, int(traffic_count))
        
        # Incident probability
        is_hotspot = cell in hotspot_cells
        hotspot_factor = config.hotspot_multiplier if is_hotspot else 1.0
        
        # Weather impact on incidents (bad weather = more incidents)
        weather_risk = 0
        if 'weather_precipitation_amount' in row and row['weather_precipitation_amount'] > 0:
            weather_risk += config.incident_weather_correlation * row['weather_precipitation_amount']
        if 'weather_wind_speed' in row and row['weather_wind_speed'] > 15:
            weather_risk += config.incident_weather_correlation * 0.5
        
        # Traffic impact on incidents (more traffic = more incidents)
        traffic_risk = config.incident_traffic_correlation * (traffic_count / config.base_traffic_rate - 1)
        
        incident_prob = config.incident_rate * hotspot_factor * (1 + weather_risk + traffic_risk)
        incident_prob = np.clip(incident_prob, 0, 0.8)
        
        # Generate incidents (Poisson-ish)
        incident_count = rng.poisson(incident_prob * 2)  # Scale for realistic counts
        
        results.append({
            'traffic_count': traffic_count,
            'incident_count': incident_count,
            'is_hotspot': int(is_hotspot),
        })
    
    return pd.DataFrame(results)


def generate_dataset_gpu(config: DataGenConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate large-scale dataset using GPU acceleration"""
    import cudf
    import cupy as cp
    
    logger.info("Generating dataset with GPU acceleration...")
    
    rng = cp.random.default_rng(config.seed)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=config.n_days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')[:-1]
    n_hours = len(timestamps)
    
    # Generate H3 cells
    h3_cells = generate_h3_cells(config)
    logger.info(f"Generated {len(h3_cells)} H3 cells")
    
    # Create full grid (cells × hours)
    n_total = n_hours * len(h3_cells)
    logger.info(f"Generating {n_total:,} rows ({n_hours} hours × {len(h3_cells)} cells)")
    
    # Build arrays on GPU
    hour_idx = cp.repeat(cp.arange(n_hours), len(h3_cells))
    cell_idx = cp.tile(cp.arange(len(h3_cells)), n_hours)
    
    # Temporal features (vectorized on GPU)
    hour_of_day = (hour_idx % 24).astype(cp.float32)
    day_of_year = ((hour_idx // 24) % 365 + 1).astype(cp.float32)
    day_of_week = ((hour_idx // 24) % 7).astype(cp.float32)
    
    # Hour pattern
    hour_pattern = (
        config.hourly_amplitude * cp.sin(2 * cp.pi * (hour_of_day - 6) / 24) +
        config.hourly_amplitude * 0.5 * cp.sin(4 * cp.pi * (hour_of_day - 8) / 24)
    )
    
    # Day pattern
    dow_pattern = cp.where(day_of_week >= 5, -0.3, 0.1) * config.daily_amplitude
    
    # Seasonal pattern
    seasonal_pattern = config.weekly_amplitude * cp.sin(2 * cp.pi * (day_of_year - 80) / 365)
    
    combined_pattern = 1.0 + hour_pattern + dow_pattern + seasonal_pattern
    
    # Weather (vectorized on GPU)
    daily_temp_cycle = config.temp_daily_range * cp.sin(2 * cp.pi * (hour_of_day - 6) / 24)
    seasonal_temp_cycle = config.temp_seasonal_range * cp.sin(2 * cp.pi * (day_of_year - 80) / 365)
    temperature = config.base_temp + daily_temp_cycle + seasonal_temp_cycle + rng.normal(0, 3, n_total)
    
    humidity = 60 - 0.5 * (temperature - config.base_temp) + rng.normal(0, 10, n_total)
    humidity = cp.clip(humidity, 20, 100)
    
    wind_speed = cp.clip(rng.lognormal(1.5, 0.5, n_total), 0, 40)
    
    precip_base_prob = 0.1 + 0.1 * cp.sin(2 * cp.pi * (day_of_year - 350) / 365)
    precip_occurs = rng.random(n_total) < precip_base_prob
    precip_amount = cp.where(precip_occurs, rng.exponential(0.2, n_total), 0)
    precip_prob = cp.where(precip_occurs, rng.uniform(50, 100, n_total), rng.uniform(0, 30, n_total))
    
    dewpoint = temperature - ((100 - humidity) / 5)
    
    # Traffic counts (vectorized)
    cell_baselines = cp.array(rng.uniform(0.7, 1.3, len(h3_cells)))
    cell_factor = cell_baselines[cell_idx.astype(cp.int32)]
    
    traffic_count = (
        config.base_traffic_rate * 
        cell_factor * 
        combined_pattern * 
        cp.exp(rng.normal(0, 0.2, n_total))
    )
    traffic_count = cp.maximum(0, traffic_count).astype(cp.int32)
    
    # Incidents (vectorized)
    n_hotspots = int(len(h3_cells) * config.hotspot_cells_pct)
    hotspot_mask = cp.zeros(len(h3_cells), dtype=cp.bool_)
    hotspot_indices = rng.choice(len(h3_cells), n_hotspots, replace=False)
    hotspot_mask[hotspot_indices] = True
    is_hotspot = hotspot_mask[cell_idx.astype(cp.int32)]
    
    hotspot_factor = cp.where(is_hotspot, config.hotspot_multiplier, 1.0)
    
    weather_risk = (
        config.incident_weather_correlation * cp.clip(precip_amount, 0, 2) +
        config.incident_weather_correlation * 0.5 * (wind_speed > 15).astype(cp.float32)
    )
    
    traffic_risk = config.incident_traffic_correlation * (traffic_count / config.base_traffic_rate - 1)
    
    incident_lambda = config.incident_rate * hotspot_factor * (1 + weather_risk + traffic_risk) * 2
    incident_lambda = cp.clip(incident_lambda, 0.01, 5)
    incident_count = rng.poisson(incident_lambda).astype(cp.int32)
    
    # Build cuDF DataFrame
    logger.info("Building cuDF DataFrame...")
    
    # Convert cell indices to strings (need to go through CPU for h3 strings)
    cell_strings = [h3_cells[i] for i in cell_idx.get().tolist()]
    timestamp_series = pd.Series(timestamps).iloc[hour_idx.get()].reset_index(drop=True)
    
    gdf = cudf.DataFrame({
        'hour_ts': cudf.Series(timestamp_series),
        'h3_cell': cudf.Series(cell_strings),
        'hour': cudf.Series(hour_of_day.astype(cp.int8)),
        'day_of_week': cudf.Series(day_of_week.astype(cp.int8)),
        'day_of_year': cudf.Series(day_of_year.astype(cp.int16)),
        'month': cudf.Series(((day_of_year - 1) // 30 + 1).astype(cp.int8)),
        'weather_temperature': cudf.Series(temperature.astype(cp.float32)),
        'weather_humidity': cudf.Series(humidity.astype(cp.float32)),
        'weather_wind_speed': cudf.Series(wind_speed.astype(cp.float32)),
        'weather_precipitation_amount': cudf.Series(precip_amount.astype(cp.float32)),
        'weather_precipitation_probability': cudf.Series(precip_prob.astype(cp.float32)),
        'weather_dewpoint': cudf.Series(dewpoint.astype(cp.float32)),
        'traffic_count': cudf.Series(traffic_count),
        'incident_count': cudf.Series(incident_count),
        'is_hotspot_cell': cudf.Series(is_hotspot.astype(cp.int8)),
    })
    
    # Add derived features
    gdf['hour_sin'] = cudf.Series(cp.sin(2 * cp.pi * hour_of_day / 24).astype(cp.float32))
    gdf['hour_cos'] = cudf.Series(cp.cos(2 * cp.pi * hour_of_day / 24).astype(cp.float32))
    gdf['dow_sin'] = cudf.Series(cp.sin(2 * cp.pi * day_of_week / 7).astype(cp.float32))
    gdf['dow_cos'] = cudf.Series(cp.cos(2 * cp.pi * day_of_week / 7).astype(cp.float32))
    
    # Convert to pandas for output
    X_df = gdf.to_pandas()
    
    # Create target (y)
    y_df = X_df[['hour_ts', 'h3_cell', 'incident_count']].copy()
    y_df = y_df.rename(columns={'incident_count': 'target'})
    
    # Remove target from X
    X_df = X_df.drop(columns=['incident_count'])
    
    logger.info(f"Generated X: {X_df.shape}, y: {y_df.shape}")
    
    return X_df, y_df


def generate_dataset_cpu(config: DataGenConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate dataset using CPU (fallback)"""
    logger.info("Generating dataset with CPU...")
    
    rng = np.random.default_rng(config.seed)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=config.n_days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H', tz='UTC')[:-1]
    n_hours = len(timestamps)
    
    # Generate H3 cells
    h3_cells = generate_h3_cells(config)
    logger.info(f"Generated {len(h3_cells)} H3 cells")
    
    # Create full grid
    n_total = n_hours * len(h3_cells)
    logger.info(f"Generating {n_total:,} rows ({n_hours} hours × {len(h3_cells)} cells)")
    
    # Build DataFrame row by row (slower but works on CPU)
    rows = []
    
    for hour_idx, ts in enumerate(timestamps):
        temporal = generate_temporal_features(pd.DatetimeIndex([ts]), config).iloc[0]
        weather = generate_weather_data(pd.DatetimeIndex([ts]), config, rng).iloc[0]
        
        for cell_idx, cell in enumerate(h3_cells):
            row = {
                'hour_ts': ts,
                'h3_cell': cell,
                **temporal.to_dict(),
                **weather.to_dict(),
            }
            rows.append(row)
        
        if (hour_idx + 1) % 100 == 0:
            logger.info(f"  Generated {(hour_idx + 1) * len(h3_cells):,} rows...")
    
    df = pd.DataFrame(rows)
    
    # Generate traffic and incidents
    traffic_incidents = generate_traffic_incidents(df, h3_cells, config, rng)
    df = pd.concat([df, traffic_incidents], axis=1)
    
    # Create X and y
    y_df = df[['hour_ts', 'h3_cell', 'incident_count']].copy()
    y_df = y_df.rename(columns={'incident_count': 'target'})
    
    X_df = df.drop(columns=['incident_count'])
    
    logger.info(f"Generated X: {X_df.shape}, y: {y_df.shape}")
    
    return X_df, y_df


def save_dataset(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    config: DataGenConfig,
) -> None:
    """Save generated dataset to parquet files"""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save X features
    X_path = out_dir / "X_features.parquet"
    X_df.to_parquet(X_path, index=False)
    logger.info(f"Saved X features: {X_path}")
    
    # Save y target
    y_path = out_dir / "y_target.parquet"
    y_df.to_parquet(y_path, index=False)
    logger.info(f"Saved y target: {y_path}")
    
    # Save metadata
    meta = {
        'generated_at': datetime.now().isoformat(),
        'config': {
            'n_days': config.n_days,
            'n_h3_cells': config.n_h3_cells,
            'h3_resolution': config.h3_resolution,
            'seed': config.seed,
        },
        'X_shape': list(X_df.shape),
        'y_shape': list(y_df.shape),
        'X_columns': list(X_df.columns),
        'n_unique_cells': X_df['h3_cell'].nunique(),
        'date_range': {
            'start': str(X_df['hour_ts'].min()),
            'end': str(X_df['hour_ts'].max()),
        },
        'statistics': {
            'traffic_count_mean': float(X_df['traffic_count'].mean()) if 'traffic_count' in X_df.columns else None,
            'target_mean': float(y_df['target'].mean()),
            'target_nonzero_pct': float((y_df['target'] > 0).mean() * 100),
        }
    }
    
    meta_path = out_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved metadata: {meta_path}")


def generate_and_save(config: DataGenConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point: generate and save dataset"""
    
    if HAS_GPU:
        X_df, y_df = generate_dataset_gpu(config)
    else:
        X_df, y_df = generate_dataset_cpu(config)
    
    save_dataset(X_df, y_df, config)
    
    return X_df, y_df


def main():
    parser = argparse.ArgumentParser(description="Generate GPU-worthy synthetic data")
    parser.add_argument('--days', type=int, default=90, help='Number of days of data')
    parser.add_argument('--cells', type=int, default=200, help='Number of H3 cells')
    parser.add_argument('--output', type=str, default='generated-data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = DataGenConfig(
        n_days=args.days,
        n_h3_cells=args.cells,
        output_dir=args.output,
        seed=args.seed,
    )
    
    # Calculate expected size
    expected_rows = args.days * 24 * args.cells
    logger.info(f"Generating dataset:")
    logger.info(f"  Days: {args.days}")
    logger.info(f"  H3 Cells: {args.cells}")
    logger.info(f"  Expected rows: {expected_rows:,}")
    logger.info(f"  Output: {args.output}")
    
    X_df, y_df = generate_and_save(config)
    
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"X features: {X_df.shape[0]:,} rows × {X_df.shape[1]} columns")
    print(f"y target: {y_df.shape[0]:,} rows")
    print(f"Output directory: {config.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
