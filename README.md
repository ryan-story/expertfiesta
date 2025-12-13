# DGXB Data Product & Benchmark Architecture

## Core Philosophy

**Data Product First**: Define a standardized data product structure (like an API contract) that any data source can conform to. Once data conforms to this structure, it automatically flows through the pipeline.

**Hotspot Agent Watcher**: Build a system that learns the "Rhythm of the City" to predict traffic incident hotspots based on time/day and visualizes where units should be stationed. The system divides Austin into sectors (H3 cells) and assigns a "Risk Score" to each sector for every hour of the day.

## Current Implementation Status

### ✅ Completed: Sector-Hour Regression Pipeline

**Architecture**: The system has been refactored from event-level classification to (sector, hour) regression:

- **Gold Layer**: Aggregated to (h3_cell, hour_ts) level (7,610 rows from ~3,000 events)
- **Target**: `incident_count_t_plus_1` (predicting next hour's incident count per sector)
- **Features**: Weather aggregates, temporal features, spatial context, lag/rolling features
- **Models**: LinearRegression, PoissonRegressor, RandomForestRegressor, XGBRegressor
- **Metrics**: RMSE, MAE, R², SMAPE, MAPE (positive only), Hotspot Precision@K/Recall@K
- **CV Splitting**: Time-blocked splits on `hour_ts` (not individual rows)
- **Baseline Models**: Persistence (y_hat = incident_count_t), Climatology (y_hat = mean per cell, hour_of_week)

**Performance**: 
- **Champion Model**: XGBoost (rich features) - RMSE: 0.0144, MAE: 0.0025, R²: 0.9910
- **Robustness Validated**: No target leakage, horizon sensitivity confirmed, sector generalization verified
- See `notes/BENCHMARK_REPORT.md` and `notes/ROBUSTNESS_REPORT.md` for detailed results

### ✅ Implemented Components

#### Data Layers (Bronze → Silver → Gold → Rich Gold)

1. **Bronze Layer** (`bronze-traffic/`, `bronze-weather/`)
   - Raw ingested data (parquet files per day)
   - Traffic incidents and weather observations

2. **Silver Layer** (`silver-cpu-traffic/`, `silver-cpu-weather/`)
   - Cleaned and standardized data
   - Basic transformations applied

3. **Gold Layer** (`gold-cpu-traffic/`)
   - **Aggregated to sector-hour level**: `sector_hour_base.parquet`
   - **Event-level traceability**: `merged_events.parquet`
   - **Feature matrix**: `X_features.parquet` (with `hour_ts` preserved)
   - **Target variable**: `y_target.parquet` (incident_count_t_plus_1)
   - Primary key: `(h3_cell, hour_ts)` where `hour_ts` is timezone-aware UTC timestamp

4. **Rich Gold Layer** (`rich-gold-cpu-traffic/`)
   - Enriched features with lags and rolling statistics
   - Computed per `h3_cell` group on aggregated data
   - Preserves `hour_ts` for CV and lag feature generation

#### ETL Pipeline (`dgxb/etl/`)

- **`feature_engineering.py`**:
  - `build_sector_hour_index()`: Creates complete sector-hour grid with explicit zeros
  - `aggregate_incidents_to_sector_hour()`: Aggregates incidents to (h3_cell, hour_ts)
  - `aggregate_weather_to_sector_hour()`: Aggregates weather separately
  - `join_and_impute_sector_hour()`: Joins with neighbor/city-wide fallback
  - `make_regression_target()`: Creates `incident_count_t_plus_1` target
  - `merge_and_save_X_features()`: Orchestrates aggregation and feature engineering
  - `prepare_y_target_regression()`: Y pipeline for regression target

- **`rich_feature_engineering.py`**:
  - `enrich_X_features()`: Adds lags, rolling stats, spatial aggregates
  - Works with aggregated data using `hour_ts` and `h3_cell` grouping

- **`traffic_fetcher.py`**, **`weather_fetcher.py`**, **`silver_processor.py`**: Data ingestion and cleaning

#### Training Pipeline (`dgxb/training/`)

- **`pipeline.py`**:
  - `run_training_competition()`: Main orchestrator
  - Loads aggregated data, performs time-blocked CV, trains models, computes metrics
  - Champion selection based on RMSE (minimize)

- **`model_competition.py`**:
  - `train_linear_regression()`: LinearRegression with StandardScaler
  - `train_poisson_regression()`: PoissonRegressor (count-native)
  - `train_random_forest()`: RandomForestRegressor
  - `train_xgboost()`: XGBRegressor (reg:squarederror or count:poisson)
  - Baseline models: Persistence (y_hat = incident_count_t), Climatology (y_hat = mean per cell, hour_of_week)

- **`metrics_tracker.py`**:
  - `compute_regression_metrics()`: RMSE, MAE, R², SMAPE, MAPE (positive only)
  - `compute_hotspot_metrics()`: Precision@K, Recall@K (per hour, then averaged)
  - `measure_inference_latency()`: P50, P95 latency metrics

- **`cv_splitter.py`**:
  - `create_rolling_origin_cv()`: Time-blocked splits on `hour_ts`
  - Test set: contiguous time window (e.g., last 24h)
  - Train set: hours strictly before test window (with optional gap)

- **`leakage_audit.py`**: Feature timestamp validation (for classification, skipped for regression)

- **`baseline_comparison.py`**: Historical baseline metrics (legacy, replaced by baseline models)

#### Entry Points

- **`run_X_pipeline.py`**: Runs X feature engineering pipeline
- **`run_Y_pipeline.py`**: Runs Y target preparation (regression)
- **`run_rich_features.py`**: Runs rich feature engineering
- **`run_training.py`**: Runs model training competition

### ✅ Recent Improvements

1. **Performance Optimization**: Removed unnecessary full grid creation (reduced from 1.5M to 7,610 rows)
2. **Robustness Validation**: Comprehensive validation suite confirms legitimate signal (no leakage)
3. **Baseline Models**: Persistence and Climatology baselines fully integrated
4. **Documentation**: Benchmark and robustness reports moved to `notes/` directory

### 📋 Remaining Work

1. **Production Deployment**:
   - Deploy XGBoost (rich) model for real-time predictions
   - Hotspot visualization dashboard
   - Real-time prediction API
   - Monitoring and alerting

2. **Model Improvements**:
   - Sector embedding for better cold-start performance
   - Ensemble methods (XGBoost + RandomForest)
   - Multi-horizon training (separate models for t+1, t+3, t+6)

3. **GPU Pipeline** (Future):
   - Implement GPU equivalents using cuDF, cuML
   - Benchmark CPU vs GPU performance

4. **Advanced Features**:
   - More sophisticated spatial-temporal features
   - External data sources (events, holidays, traffic patterns)
   - Deep learning models (LSTM, Transformer)

## Architecture Overview

### Data Flow

```
Bronze (Raw) → Silver (Cleaned) → Gold (Aggregated) → Rich Gold (Enriched) → Training → Results
```

### Key Design Decisions

1. **Sector-Hour Aggregation**:
   - Primary key: `(h3_cell, hour_ts)` with timezone-aware timestamps
   - Explicit zero rows: Complete grid ensures "no row = no incidents" is explicit
   - Weather aggregated separately, then joined (prevents leakage)

2. **Target Construction**:
   - `incident_count_t_plus_1` = `incident_count.groupby(h3_cell).shift(-1)`
   - Rows with null target (last hour per cell) excluded from training

3. **Time-Blocked CV**:
   - Splits on `hour_ts`, not individual rows
   - All cells in test hours evaluated simultaneously
   - Optional gap between train and test to prevent rolling feature leakage

4. **Feature Engineering**:
   - Weather: median (continuous), max (precipitation), mode (categorical)
   - Avoid "mode" for agency/status (use counts/shares instead)
   - Lag/rolling features computed per `h3_cell` group

5. **Metrics**:
   - Primary: RMSE, MAE (overall)
   - Secondary: R², SMAPE (handles zeros safely)
   - Hotspot: Precision@K, Recall@K (per hour, then averaged)

## File Structure

```
dgxb/
├── etl/
│   ├── feature_engineering.py      # Aggregation & base features
│   ├── rich_feature_engineering.py # Lag/rolling/spatial features
│   ├── traffic_fetcher.py          # Traffic data ingestion
│   ├── weather_fetcher.py          # Weather data ingestion
│   └── silver_processor.py         # Silver layer processing
├── training/
│   ├── pipeline.py                 # Main training orchestrator
│   ├── model_competition.py        # Model training functions
│   ├── metrics_tracker.py         # Regression & hotspot metrics
│   ├── cv_splitter.py             # Time-blocked CV splits
│   ├── leakage_audit.py           # Feature validation (legacy)
│   └── baseline_comparison.py     # Historical baseline (legacy)
├── run_X_pipeline.py              # X feature engineering entry point
├── run_Y_pipeline.py              # Y target preparation entry point
├── run_rich_features.py           # Rich features entry point
└── run_training.py                # Training entry point
```

## Usage

### 1. Run X Pipeline (Feature Engineering)

```bash
python -m dgxb.run_X_pipeline
# or
python dgxb/run_X_pipeline.py
```

Creates:
- `gold-cpu-traffic/merged_events.parquet` (event-level, for traceability)
- `gold-cpu-traffic/sector_hour_base.parquet` (aggregated, canonical)
- `gold-cpu-traffic/X_features.parquet` (features ready for ML)

### 2. Run Y Pipeline (Target Preparation)

```bash
python -m dgxb.run_Y_pipeline
# or
python dgxb/run_Y_pipeline.py
```

Creates:
- `gold-cpu-traffic/y_target.parquet` (incident_count_t_plus_1)

### 3. Run Rich Features

```bash
python -m dgxb.run_rich_features
# or
python dgxb/run_rich_features.py
```

Creates:
- `rich-gold-cpu-traffic/X_features.parquet` (enriched features)

### 4. Run Training

```bash
python -m dgxb.run_training
# or
python dgxb/run_training.py
```

Creates:
- `results/cpu_training_results.csv` (all model results)
- Champion selection based on RMSE

## Results

Results are saved to `results/` directory:

- `cpu_training_results.csv`: All model results with metrics
- Columns include: `rmse`, `mae`, `r2`, `smape`, `hotspot_precision_at_k`, `hotspot_recall_at_k`, etc.

**Documentation** (in `notes/` directory):

- `BENCHMARK_REPORT.md`: Comprehensive benchmark results and model comparison
- `ROBUSTNESS_REPORT.md`: Validation results confirming legitimate signal (no leakage)
- `LEAKAGE_VALIDATION_REPORT.md`: Feature leakage audit results
- `TRAINING_RESULTS_ANALYSIS.md`: Detailed training analysis

## Key Metrics

### Regression Metrics
- **RMSE**: Root Mean Squared Error (primary)
- **MAE**: Mean Absolute Error (primary)
- **R²**: Coefficient of determination
- **SMAPE**: Symmetric Mean Absolute Percentage Error (handles zeros)
- **MAPE (positive only)**: Mean Absolute Percentage Error on non-zero targets

### Hotspot Metrics
- **Precision@K**: Of top-K predicted hotspots, how many are in top-K actual?
- **Recall@K**: Of top-K actual hotspots, how many are in top-K predicted?
- **Conditional Precision@K**: Same, but only for hours with ≥1 incident
- **Staging Utility**: % of cell-hours with incidents that are in predicted top-K

## Future Enhancements

1. **GPU Pipeline**: Implement cuDF/cuML equivalents for performance comparison
2. **Serving Layer**: Real-time prediction API and visualization
3. **Advanced Features**: More sophisticated spatial-temporal features
4. **Model Improvements**: Ensemble methods, deep learning models
5. **Production Deployment**: Containerization, API endpoints, monitoring

## Notes

- The system uses H3 resolution 9 (cell size ~0.50km) for spatial indexing
- All timestamps are timezone-aware (UTC preferred)
- The pipeline is designed for in-memory processing (pandas-based)
- For larger datasets, consider chunking or distributed processing
