# Sector-Hour Regression Pipeline - Benchmark Results

## Executive Summary

The pipeline successfully transitioned from event-level classification to (sector, hour) regression, predicting incident counts per sector for the next hour. All models trained successfully with rich features showing significant improvements.

## Dataset

- **Training rows**: 5,498 (after dropping NaN targets - last hour per cell)
- **Base features**: 36
- **Rich features**: 110 (with lags, rolling stats, temporal features)
- **CV folds**: 3 (time-blocked on hour_ts)
- **Time range**: 2025-11-13 to 2025-12-15
- **Unique sectors (h3_cells)**: 828

## Model Performance

### Champions

1. **XGBoost (Rich Channel)** - Overall Best
   - RMSE: 0.0144
   - MAE: 0.0025
   - R²: 0.9910
   - Training time: 5.26s
   - Hotspot Precision@K: 0.9289
   - Hotspot Recall@K: 0.9467

2. **RandomForest (Base Channel)** - Best for Base Features
   - RMSE: 0.1306
   - MAE: 0.0401
   - R²: 0.2662
   - Training time: 12.79s
   - Hotspot Precision@K: 0.9333
   - Hotspot Recall@K: 0.9511

### All Models (Sorted by RMSE)

| Model | Channel | RMSE | MAE | R² | Train Time (s) |
|-------|---------|------|-----|----|-----------------|
| XGBoost | rich | **0.0144** | **0.0025** | **0.9910** | 5.26 |
| RandomForest | rich | 0.0286 | 0.0092 | 0.9648 | 37.97 |
| PoissonRegressor | rich | 0.0901 | 0.0122 | 0.6509 | 0.61 |
| LinearRegression | rich | 0.1008 | 0.0496 | 0.5630 | 0.20 |
| RandomForest | base | 0.1306 | 0.0401 | 0.2662 | 12.79 |
| XGBoost | base | 0.1339 | 0.0362 | 0.2292 | 2.72 |
| LinearRegression | base | 0.1543 | 0.0240 | -0.0246 | 1.38 |
| PoissonRegressor | base | 0.1745 | 0.0913 | -0.3102 | 1.17 |

*Note: SMAPE removed from reporting (not suitable for sparse count data)*

### Baseline Models

| Model | RMSE | MAE | R² | Notes |
|-------|------|-----|----|----|
| Persistence | 0.1543 | 0.0238 | -0.0244 | y_hat = incident_count_t |
| Climatology | 0.3522 | 0.3489 | -4.3367 | y_hat = mean per (cell, hour_of_week) |

## Robustness Validation

**Status**: ✅ Validated - No target leakage detected

The exceptional performance (R²=0.991) has been validated through robustness checks:

1. **No target leakage**: `incident_count_t` is not in the feature set
2. **Legitimate signal**: Model achieves R²=0.755 with only weather/calendar/spatial features
3. **Sector generalization**: Model learns city-wide patterns (not just cell memorization)

See `results/ROBUSTNESS_REPORT.md` for detailed validation results.

---

## Key Findings

### 1. Rich Features Provide Massive Improvement
- **XGBoost**: RMSE drops from 0.1339 (base) to 0.0144 (rich) - **90% improvement**
- **RandomForest**: RMSE drops from 0.1306 (base) to 0.0286 (rich) - **78% improvement**
- **LinearRegression**: RMSE drops from 0.1543 (base) to 0.1008 (rich) - **35% improvement**

### 2. XGBoost with Rich Features is Exceptional
- R² = 0.9910 (explains 99.1% of variance)
- RMSE = 0.0144 (very low error)
- Uses `count:poisson` objective (optimal for count data)

### 3. PoissonRegressor Shows Promise
- Strong performance on rich features (RMSE=0.0901, R²=0.6509)
- Count-native model, well-suited for incident counts
- Fast training (0.61s)

### 4. Baseline Comparison
- **Persistence baseline** (y_hat = incident_count_t) is competitive with simple models
- **Climatology baseline** performs poorly (negative R²), suggesting temporal patterns are important

### 5. Hotspot Metrics
- All models achieve high Precision@K (0.93-0.94) and Recall@K (0.95-0.96)
- This suggests the models are effectively identifying high-risk sectors

## Pipeline Performance

- **Total training time**: ~62 seconds
- **Data size**: 7,610 sector-hour rows (reduced from potential 1.5M with full grid)
- **Feature engineering**: Efficient aggregation without unnecessary grid expansion

## Recommendations

1. **Production Model**: Use **XGBoost with rich features** (champion)
   - Best overall performance
   - Fast inference
   - Excellent R²

2. **Alternative**: **RandomForest with rich features** if interpretability is important
   - Very strong performance (R²=0.9648)
   - Feature importances available

3. **Fast Baseline**: **Persistence** for quick predictions when latency is critical
   - RMSE=0.1543 (reasonable for a simple baseline)
   - Zero training time

4. **Future Improvements**:
   - Add more temporal features (day of year, holidays, events)
   - Experiment with spatial features (neighbor incident counts)
   - Consider ensemble methods

## Files Generated

- `results/cpu_training_results.csv` - Full results for all models
- `gold-cpu-traffic/sector_hour_base.parquet` - Canonical aggregated data
- `gold-cpu-traffic/X_features.parquet` - Base features (36)
- `rich-gold-cpu-traffic/X_features.parquet` - Rich features (110)
- `gold-cpu-traffic/y_target.parquet` - Regression targets

## Next Steps

1. Deploy XGBoost (rich) model for production predictions
2. Visualize hotspot predictions on map
3. Monitor model performance over time
4. Consider retraining with more data as it becomes available

