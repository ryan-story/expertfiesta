# Training Results Analysis & Next Steps

## Current Results Summary

### Champions Selected

**Base Channel Champion: RandomForest**
- Weighted F1: 0.9725
- F1 Score: 0.9716
- Precision: 0.9727
- Train Time: 8.66s
- Inference Latency (p50): 37.44ms

**Rich Channel Champion: XGBoost**
- Weighted F1: 1.0000
- F1 Score: 1.0000
- Precision: 1.0000
- Train Time: 5.36s
- Inference Latency (p50): 23.01ms

## Critical Issues Identified

### 1. Perfect Scores (F1=1.0) - RED FLAG ⚠️

**What this likely indicates:**
- **Data leakage**: Future information accidentally included in features
- **Label definition too easy**: Most samples belong to one class
- **Non-time-aware splitting**: Same-hour patterns appearing in both train and test
- **Small test set**: Limited unique hours/H3 cells causing repetitive patterns

**Specific concerns:**
- Rich channel XGBoost achieving perfect scores is highly suspicious
- RandomForest base channel at ~0.97 is also suspiciously high for a sparse incident prediction task

### 2. Precision@K and Recall@K = 0.0 - Metric Framing Issue

**Root causes:**
- K (50) may be larger than available spatial units in test hours
- Hotspot ground truth may be empty in test hours
- Test horizon may be too small (< 24h recommended)
- Metrics computed at wrong granularity (events vs H3 cells)

**Current test window:** 24 hours per fold (3 folds)
**Issue:** If incident density is low, many hours may have no hotspots

### 3. Data Leakage Audit Needed

**Potential leakage vectors:**
1. **Lag features**: Must verify they're backward-looking only (t-lag, not t+lag)
2. **Rolling statistics**: Must verify windows exclude future data (use [t-window, t], not [t, t+window])
3. **Spatial aggregates**: Must verify time windows exclude future incidents
4. **Label definition**: If thresholds computed on full dataset (including test), that's leakage

## Implemented Fixes

### ✅ 1. Leakage Audit Module (`dgxb/training/leakage_audit.py`)

**Functions:**
- `audit_feature_timestamps()`: Checks for timestamp-like columns, lag features, rolling features
- `audit_cv_splits()`: Validates time-aware splitting (train max ≤ test min)
- `verify_lag_direction()`: Verifies lag features are backward-looking

**Integration:**
- Automatically runs during training pipeline
- Logs warnings and errors for detected issues

### ✅ 2. CV Split Validation

**Checks:**
- Train max time ≤ Test min time (no future data in train)
- Test windows are contiguous and non-overlapping
- Test window span matches expected duration (24h)

### ✅ 3. Results Interpretation Warnings

**Added to pipeline output:**
- Disclaimer about high F1 scores
- Explanation of Precision@K/Recall@K = 0.0
- Next steps checklist

## Verification Checklist

### Immediate Actions Required

- [ ] **Verify lag features are backward-looking**
  - Check `add_lag_features()` in `rich_feature_engineering.py`
  - Line 183: `df_shifted.index = df_shifted.index - pd.Timedelta(hours=lag_hours)` ✓ (correct)
  - Line 192: `direction="backward"` ✓ (correct)
  - **Status**: ✅ Appears correct - lags look backward in time

- [ ] **Verify rolling statistics exclude future data**
  - Check `add_rolling_statistics()` in `rich_feature_engineering.py`
  - Line 267: `df_indexed[col].rolling(window=window_str, min_periods=1).mean()`
  - **Issue**: Pandas `.rolling()` with time-based index includes current row
  - **For predicting t+1**: Window [t-window, t] is correct (includes current time t)
  - **Status**: ⚠️ Need to verify: For row at time t, does it include data from t+1?

- [ ] **Verify spatial aggregates exclude future data**
  - Check `add_spatial_neighbor_aggregates()` in `rich_feature_engineering.py`
  - Line 354: `time_threshold = current_time - pd.Timedelta(hours=window_hours)` ✓
  - Line 357: `time_mask = df[timestamp_col] >= time_threshold` ✓
  - Line 358: `time_mask &= df[timestamp_col] < current_time` ✓
  - **Status**: ✅ Appears correct - only uses data before current_time

- [ ] **Verify label definition**
  - Check how `incident_category_encoded` is computed
  - If thresholds/quantiles computed on full dataset → LEAKAGE
  - Must compute thresholds on train fold only
  - **Status**: ⚠️ Need to verify zero-shot classification doesn't use test data

- [ ] **Increase test horizon support**
  - Current: 24 hours per fold (3 folds)
  - **Recommendation**: Ensure test windows have sufficient incident density
  - Consider: Increase to 48-72 hours per fold if data allows

- [ ] **Fix hotspot metrics implementation**
  - Current: Computes per-hour P@K and R@K
  - **Issue**: If test hours have no incidents, metrics collapse to 0
  - **Fix**: 
    - Only compute metrics for hours with incidents
    - Adjust K dynamically based on available cells per hour
    - Use `K = min(50, 5% of active cells per hour)`

## Next Steps

### Phase 1: Validation (Immediate)

1. **Run leakage audit** on current rich features
   ```bash
   python dgxb/run_training.py
   ```
   Review audit output for warnings/errors

2. **Spot-check feature timestamps**
   - Sample 10-20 rows from test set
   - Verify all features are computable using data ≤ timestamp
   - Check lag features point to past values

3. **Verify label distribution**
   - Check class balance in train vs test
   - If >80% one class → label too easy
   - Consider: Use incident_count threshold instead of category

### Phase 2: Fixes (Before Next Run)

1. **Fix rolling statistics** (if needed)
   - Ensure rolling windows are strictly backward-looking
   - For row at time t, window should be [t-window, t) (exclusive of t if needed)

2. **Fix hotspot metrics**
   - Implement dynamic K adjustment
   - Only compute for hours with incidents
   - Add minimum support threshold (e.g., require ≥10 incidents per hour)

3. **Improve label definition** (if needed)
   - If using thresholds, compute on train fold only
   - Consider binary hotspot label: `incident_count >= threshold` (threshold from train)

### Phase 3: Re-run & Validate

1. **Re-run training** with fixes
2. **Expected outcomes:**
   - F1 scores drop to believable range (0.3-0.7 for sparse incidents)
   - Precision@K and Recall@K become non-zero
   - Staging utility shows meaningful coverage

3. **If scores still suspicious:**
   - Investigate label definition further
   - Consider simpler binary classification (hotspot vs not)
   - Verify test set has sufficient diversity

## Safe Presentation of Current Results

**Do NOT present F1=1.0 as a win.**

**Instead, present as:**
- "Pipeline is operational; champions selected based on current evaluation"
- "Next: Hardening evaluation using walk-forward CV and hotspot-centric KPIs"
- "Initial results show high F1 scores which may indicate data leakage or label definition issues - under investigation"

**Key metrics to highlight:**
- ✅ Pipeline latency (train time, inference latency) - these are reliable
- ✅ Throughput metrics - these are reliable
- ⚠️ Model quality metrics (F1, Precision@K) - need validation

## Files Modified

1. `dgxb/training/leakage_audit.py` - New module for leakage detection
2. `dgxb/training/pipeline.py` - Added audit calls and interpretation warnings
3. `TRAINING_RESULTS_ANALYSIS.md` - This document

## References

- CV splitting: `dgxb/training/cv_splitter.py` - ✅ Time-aware (rolling-origin)
- Lag features: `dgxb/etl/rich_feature_engineering.py:121-201` - ✅ Backward-looking
- Rolling stats: `dgxb/etl/rich_feature_engineering.py:204-291` - ⚠️ Need verification
- Spatial aggregates: `dgxb/etl/rich_feature_engineering.py:294-399` - ✅ Backward-looking
- Hotspot metrics: `dgxb/training/metrics_tracker.py:25-170` - ⚠️ Needs improvement

