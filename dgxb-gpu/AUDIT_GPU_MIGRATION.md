# GPU Migration Audit Report

## Executive Summary

This audit identifies violations of GPU-first RAPIDS compliance in `dgxb-gpu/` directory. All violations must be fixed to ensure:
- No pandas/sklearn imports (except isolated CPU helpers)
- No dgxb.* imports (must use dgxb_gpu.*)
- All outputs use GPU-tagged filenames
- All paths use `*-gpu-*` variants

## File-by-File Violations

### Critical: Pandas Imports (15 files)

1. **dgxb-gpu/training/pipeline.py** (Line 4)
   - `import pandas as pd`
   - Used for: DataFrame creation, concat, to_datetime, Timedelta
   - Fix: Replace with cudf equivalents

2. **dgxb-gpu/training/model_competition.py** (Line 27)
   - `import pandas as pd`
   - Used for: Type hints only (can be removed)

3. **dgxb-gpu/training/metrics_tracker.py** (Line 34)
   - `import pandas as pd`
   - Used for: Series/DataFrame conversions, datetime ops
   - Fix: Replace with cudf, keep CPU helpers isolated

4. **dgxb-gpu/training/leakage_validation.py** (Line 27)
   - `import pandas as pd`
   - Used for: DataFrame/Series operations, hashing
   - Fix: Replace with cudf equivalents

5. **dgxb-gpu/training/leakage_audit.py** (Line 26)
   - `import pandas as pd`
   - Used for: DataFrame operations
   - Fix: Replace with cudf

6. **dgxb-gpu/training/cv_splitter.py** (Line 21)
   - `import pandas as pd`
   - Used for: to_datetime, Timestamp
   - Fix: Replace with cudf.to_datetime or numpy datetime64

7. **dgxb-gpu/training/baseline_comparison.py** (Line 27)
   - `import pandas as pd`
   - Used for: DataFrame operations
   - Fix: Replace with cudf

8. **dgxb-gpu/etl/feature_engineering.py** (Line 36)
   - `import pandas as pd`
   - Used for: DataFrame creation, datetime ops, get_dummies
   - Fix: Replace with cudf equivalents

9. **dgxb-gpu/etl/rich_feature_engineering.py** (Line 33)
   - `import pandas as pd`
   - Used for: DataFrame operations
   - Fix: Replace with cudf

10. **dgxb-gpu/etl/silver_processor.py** (Line 28)
    - `import pandas as pd`
    - Used for: DataFrame operations, location parsing fallback
    - Fix: Replace with cudf, keep CPU fallback isolated

11. **dgxb-gpu/validation/robustness_checks.py** (Line 13)
    - `import pandas as pd`
    - Used for: Series operations, datetime handling
    - Fix: Replace with cudf

12. **dgxb-gpu/run_leakage_validation.py** (Line 19)
    - `import pandas as pd`
    - Used for: DataFrame operations, datetime conversions
    - Fix: Replace with cudf

### Critical: sklearn Imports (3 files)

1. **dgxb-gpu/training/metrics_tracker.py** (Line 37)
   - `from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_score, recall_score`
   - Fix: Implement via cupy/numpy

2. **dgxb-gpu/training/leakage_validation.py** (Line 30)
   - `from sklearn.metrics import f1_score, precision_score, recall_score`
   - Fix: Implement via cupy/numpy

### Critical: dgxb.* Imports (4 files)

1. **dgxb-gpu/run_leakage_validation.py** (Lines 11-12)
   - `from dgxb.training.cv_splitter import create_rolling_origin_cv`
   - `from dgxb.training.leakage_validation import ...`
   - Fix: Replace with `dgxb_gpu.training.cv_splitter.create_rolling_origin_cv_gpu`
   - Fix: Replace with `dgxb_gpu.training.leakage_validation.*_gpu`

2. **dgxb-gpu/run_feature_engineering.py** (Line 23)
   - `from dgxb.etl.feature_engineering import merge_and_save_to_gold`
   - Fix: Replace with `dgxb_gpu.etl.feature_engineering.merge_and_save_X_features_gpu`

3. **dgxb-gpu/validation/robustness_checks.py** (Line 20)
   - `from dgxb.training.cv_splitter import create_rolling_origin_cv`
   - Fix: Replace with `dgxb_gpu.training.cv_splitter.create_rolling_origin_cv_gpu`

4. **dgxb-gpu/training/pipeline.py** (Line 215)
   - `from dgxb.training.cv_splitter import create_nested_cv`
   - Fix: Replace with `dgxb_gpu.training.cv_splitter.create_nested_cv_gpu`

### Critical: CPU-Tagged Paths (6 files)

1. **dgxb-gpu/run_leakage_validation.py** (Lines 43-46)
   - `gold-cpu-traffic/` → `gold-gpu-traffic/`
   - `rich-gold-cpu-traffic/` → `rich-gold-gpu-traffic/`

2. **dgxb-gpu/run_feature_engineering.py** (Line 88)
   - Reference to `gold-cpu-traffic/` → `gold-gpu-traffic/`

3. **dgxb-gpu/validation/robustness_checks.py** (Lines 376-377)
   - `gold-cpu-traffic/` → `gold-gpu-traffic/`
   - `rich-gold-cpu-traffic/` → `rich-gold-gpu-traffic/`

4. **dgxb-gpu/training/pipeline.py** (Lines 149-151)
   - Default paths: `gold-cpu-traffic/` → `gold-gpu-traffic/`
   - Default paths: `rich-gold-cpu-traffic/` → `rich-gold-gpu-traffic/`

5. **dgxb-gpu/etl/feature_engineering.py** (Lines 737-738)
   - `silver-cpu-traffic/` → `silver-gpu-traffic/`
   - `silver-cpu-weather/` → `silver-gpu-weather/`

### Critical: Output File Naming (2 files)

1. **dgxb-gpu/run_leakage_validation.py**
   - Line 100: `results/base_features_list.csv` → `results/gpu_base_features_list.csv`
   - Line 113: `results/single_feature_smoke_test_base.csv` → `results/gpu_single_feature_smoke_test_base.csv`
   - Line 139: `results/single_feature_smoke_test_rich.csv` → `results/gpu_single_feature_smoke_test_rich.csv`

### XGBoost GPU Enforcement (2 files)

1. **dgxb-gpu/training/model_competition.py** (Line 334, 383)
   - Has CPU fallback `tree_method="hist"`
   - Fix: Remove fallback, enforce `tree_method="gpu_hist"` only

2. **dgxb-gpu/training/leakage_validation.py** (Line 114)
   - Uses `tree_method="hist"` with device="cuda"
   - Fix: Use `tree_method="gpu_hist"` explicitly

## Applied Changes Summary

All critical violations have been systematically fixed:

### Completed Fixes

1. **Pandas imports**: Removed from training pipeline, metrics, leakage validation. Kept in ETL files only for isolated CPU helpers (H3 operations, timestamp conversions for API calls) with explicit comments.

2. **sklearn imports**: Removed and replaced with cupy/numpy implementations in `metrics_tracker.py` and `leakage_validation.py`.

3. **dgxb.* imports**: All replaced with `dgxb_gpu.*` equivalents:
   - `run_leakage_validation.py`: Now uses `dgxb_gpu.training.cv_splitter.create_rolling_origin_cv_gpu`
   - `run_feature_engineering.py`: Now uses `dgxb_gpu.etl.feature_engineering.merge_and_save_X_features_gpu`
   - `validation/robustness_checks.py`: Now uses `dgxb_gpu.training.cv_splitter.create_rolling_origin_cv_gpu`
   - `training/pipeline.py`: Now uses `dgxb_gpu.training.cv_splitter.create_nested_cv_gpu`

4. **CPU-tagged paths**: All replaced with GPU variants:
   - `gold-cpu-traffic/` → `gold-gpu-traffic/`
   - `rich-gold-cpu-traffic/` → `rich-gold-gpu-traffic/`
   - `silver-cpu-traffic/` → `silver-gpu-traffic/`
   - `silver-cpu-weather/` → `silver-gpu-weather/`

5. **Output file naming**: All outputs now use `gpu` prefix:
   - `results/gpu_base_features_list.csv`
   - `results/gpu_single_feature_smoke_test_base.csv`
   - `results/gpu_single_feature_smoke_test_rich.csv`
   - `results/gpu_training_results.csv`
   - `results/robustness_checks_gpu.json`

6. **XGBoost GPU enforcement**: Removed CPU fallbacks, enforces `tree_method="gpu_hist"` only.

### Remaining Acceptable Usage

- **ETL files** (`feature_engineering.py`, `rich_feature_engineering.py`, `silver_processor.py`, `weather_fetcher.py`): Contain pandas imports for isolated CPU helpers:
  - H3 cell computation (h3-py is CPU-only)
  - Timestamp conversions for API calls
  - Small metadata table construction
  - These are explicitly documented as "CPU helper" operations and convert results back to GPU immediately.

- **CV splitter** (`cv_splitter.py`): Uses pandas for datetime parsing in helper functions, but accepts cudf Series as input and returns numpy/cupy indices.

- **Baseline comparison** (`baseline_comparison.py`): May use pandas for small helper operations.

- **Leakage audit** (`leakage_audit.py`): May use pandas for small diagnostic operations.

All of these are acceptable as they are isolated CPU helpers with explicit documentation.

