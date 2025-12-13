# GPU Pipeline (dgxb-gpu)

This directory contains a GPU-first implementation of the DGXB pipeline using RAPIDS (cuDF, cuML, CuPy) for accelerated data processing and model training.

## Requirements

- **NVIDIA GPU** with CUDA 12.x support
- **RAPIDS stack** (cudf, cuml, cupy, rmm)
- **XGBoost** with CUDA-enabled build

See `requirements-gpu.txt` for detailed dependencies.

## Installation

```bash
# Install RAPIDS from conda (recommended)
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf-cu12 cuml-cu12 cupy-cuda12x rmm-cu12

# Install other dependencies
pip install -r requirements-gpu.txt
```

## Pipeline Stages

### 1. X Pipeline (Feature Engineering)

```bash
python dgxb_gpu/run_X_pipeline.py
```

**Inputs:**
- `silver-gpu-traffic/data_silver.parquet`
- `silver-gpu-weather/data_silver.parquet`

**Outputs:**
- `gold-gpu-traffic/merged_events.parquet` (event-level, for traceability)
- `gold-gpu-traffic/sector_hour_base.parquet` (aggregated sector-hour base table)
- `gold-gpu-traffic/X_features.parquet` (feature matrix)
- `gold-gpu-traffic/X_metadata.json`

### 2. Y Pipeline (Target Preparation)

```bash
python dgxb_gpu/run_Y_pipeline.py
```

**Inputs:**
- `gold-gpu-traffic/sector_hour_base.parquet`

**Outputs:**
- `gold-gpu-traffic/y_target.parquet` (regression target: `incident_count_t_plus_1`)

### 3. Rich Features

```bash
python dgxb_gpu/run_rich_features.py
```

**Inputs:**
- `gold-gpu-traffic/sector_hour_base.parquet`

**Outputs:**
- `rich-gold-gpu-traffic/X_features.parquet` (enriched with lags, rolling stats, spatial aggregates)

### 4. Training Competition

```bash
python dgxb_gpu/run_training.py
```

**Inputs:**
- `gold-gpu-traffic/X_features.parquet` (base features)
- `rich-gold-gpu-traffic/X_features.parquet` (rich features)
- `gold-gpu-traffic/y_target.parquet`

**Outputs:**
- `results/gpu_training_results.csv` (model performance metrics)

### 5. Robustness Checks

```bash
python dgxb_gpu/run_robustness_checks.py
```

**Inputs:**
- `rich-gold-gpu-traffic/X_features.parquet`
- `gold-gpu-traffic/y_target.parquet`

**Outputs:**
- `results/robustness_checks_gpu.json`

### 6. Leakage Validation

```bash
python dgxb_gpu/run_leakage_validation.py
```

**Inputs:**
- `gold-gpu-traffic/X_features.parquet`
- `rich-gold-gpu-traffic/X_features.parquet`
- `gold-gpu-traffic/y_target.parquet`

**Outputs:**
- `results/gpu_base_features_list.csv`
- `results/gpu_single_feature_smoke_test_base.csv`
- `results/gpu_single_feature_smoke_test_rich.csv`

## GPU-Tagged Outputs

All outputs from the GPU pipeline are tagged with `gpu` prefix/suffix:
- Directory paths: `gold-gpu-traffic/`, `rich-gold-gpu-traffic/`, `silver-gpu-traffic/`
- Result files: `gpu_training_results.csv`, `robustness_checks_gpu.json`, etc.

This ensures no confusion with CPU pipeline outputs.

## CPU-Only Helpers

Some operations remain CPU-only due to library limitations:
- **H3 spatial indexing**: The `h3-py` library is CPU-only. H3 cell computation and neighbor expansion are performed on CPU, then converted to GPU for joins.
- **Holiday calendars**: Small metadata tables built on CPU, then joined to GPU dataframes.
- **CV splitter interface**: Accepts cudf Series but may convert to pandas internally for datetime operations (small conversion cost).
- **Final CSV/JSON outputs**: Small final results converted to pandas for CSV/JSON writing.

These are explicitly documented in code comments as "CPU helper" operations.

## Module Boundaries

The `dgxb-gpu/` module is self-contained:
- **No imports from `dgxb.*`**: All imports use `dgxb_gpu.*` namespace
- **GPU-first libraries**: Uses `cudf`, `cupy`, `cuml` instead of `pandas`, `numpy`, `sklearn`
- **XGBoost GPU-only**: Enforces `tree_method="gpu_hist"` with no CPU fallback

## Performance Notes

- **Data loading**: Parquet files are read directly to GPU memory via `cudf.read_parquet()`
- **Feature engineering**: All aggregations, joins, and transformations use cuDF groupby operations
- **Model training**: cuML models (LinearRegression, RandomForest) and XGBoost GPU operate directly on CuPy arrays
- **Inference**: Models predict on GPU batches for low-latency inference

## Troubleshooting

**CUDA out of memory:**
- Reduce batch sizes in training
- Process data in chunks for large datasets
- Use RMM memory pool allocator (enabled by default)

**XGBoost GPU errors:**
- Ensure XGBoost was built with CUDA support
- Check CUDA version compatibility (requires CUDA 12.x)
- Verify GPU is accessible: `nvidia-smi`

**Import errors:**
- Verify RAPIDS installation: `python -c "import cudf; print(cudf.__version__)"`
- Check CUDA version matches RAPIDS build: `nvcc --version`

