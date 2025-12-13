# DGXB Data Product & Benchmark Architecture

## Core Philosophy

**Data Product First**: Define a standardized data product structure (like an API contract) that any data source can conform to. Once data conforms to this structure, it automatically flows through the pipeline.

**Dual-Pipeline Benchmarking**: Build parallel CPU and GPU pipelines with identical logic, enabling fair comparison of performance (time, throughput, quality).

**Model Competition**: Both pipelines run multiple model candidates and select a champion based on objective metrics.

## Architecture Overview

### 1. Data Product Design (ETL Contract)

**Standardized Schema Contract**:

- Input data must conform to a defined schema
- Output artifacts follow a standardized structure
- Enables plug-and-play data sources

**Data Product Structure**:

```
data_product/
├── raw/                    # Raw ingested data
│   ├── incidents/
│   ├── weather/
│   └── geography/
├── base/                   # Base feature artifacts
│   ├── features_base.parquet
│   └── metadata.json
├── enriched/               # Enriched feature artifacts
│   ├── features_enriched.parquet
│   └── metadata.json
└── registry/               # Dataset registry
    ├── schema_hash.json
    ├── time_range.json
    └── row_counts.json
```

### 2. Dual-Pipeline Architecture

**CPU Pipeline** (Standard Stack):

- pandas/polars for ETL
- h3-py for spatial indexing
- scikit-learn for models
- XGBoost CPU (tree_method="hist")
- OR-Tools for optimization (optional)

**GPU Pipeline** (NVIDIA Stack):

- cuDF for ETL
- cuSpatial for spatial indexing
- cuML for models
- XGBoost GPU (tree_method="gpu_hist")
- cuOpt for optimization (optional)

**Both pipelines**:

- Same transformations
- Same feature definitions
- Same train/test splits
- Same model families/hyperparams
- Produce identical artifact structure

### 3. Benchmark Matrix (2×2)

| Stack | Base Features | Enriched Features |

|-------|--------------|-------------------|

| CPU   | CPU + Base   | CPU + Enriched    |

| GPU   | GPU + Base   | GPU + Enriched    |

**Four Comparable Runs**:

1. CPU + Base (~50-100 features)
2. GPU + Base (~50-100 features)
3. CPU + Enriched (~1000-5000 features)
4. GPU + Enriched (~1000-5000 features)

### 4. Feature Engineering Strategy

**Base Features**:

- H3 indexing
- Time binning
- Basic temporal (hour, day_of_week, month)
- Optional: simple weather join

**Enriched Features**:

- All base features
- Lags (multiple windows)
- Rolling stats (mean, std, min, max)
- Spatial neighbor aggregates
- Extended temporal features
- Weather features (comprehensive)

### 5. Model Competition Framework

**CPU Candidates**:

- sklearn LogisticRegression
- sklearn RandomForest
- XGBoost CPU (tree_method="hist")

**GPU Candidates**:

- cuML LogisticRegression
- cuML RandomForest
- XGBoost GPU (tree_method="gpu_hist")

**Champion Selection**:

- Primary: Maximize Precision@K
- Tie-break: Minimize inference latency
- Deterministic rule for reproducibility

## Implementation Structure

```
dgxb/
├── data_product/           # Data product design & contracts
│   ├── __init__.py
│   ├── schema.py          # Standardized schema definitions
│   ├── registry.py         # Dataset registry & metadata
│   ├── validator.py        # Schema validation
│   └── converter.py       # Convert raw data to data product format
├── etl/                    # ETL pipelines
│   ├── __init__.py
│   ├── cpu_ingest.py       # CPU ingestion (pandas/polars)
│   ├── gpu_ingest.py       # GPU ingestion (cuDF)
│   ├── cpu_feature_builder.py  # CPU feature engineering
│   ├── gpu_feature_builder.py  # GPU feature engineering
│   └── splitter.py         # Time-aware CV splitter
├── models/                 # Model competition
│   ├── __init__.py
│   ├── cpu_competition.py  # CPU model competition orchestrator
│   ├── gpu_competition.py  # GPU model competition orchestrator
│   ├── evaluator.py       # Metrics & KPI store
│   └── champion_selector.py
├── serving/                # Serving layer
│   ├── __init__.py
│   ├── cpu_serving.py      # CPU serving (sklearn, XGBoost CPU)
│   ├── gpu_serving.py      # GPU serving (cuML, FIL for XGBoost)
│   └── fil_engine.py      # FIL engine for XGBoost GPU models
├── optimization/           # Optimization layer
│   ├── __init__.py
│   ├── cpu_staging.py     # CPU staging (heuristic/OR-Tools)
│   └── gpu_staging.py     # GPU staging (heuristic/cuOpt)
├── benchmarks/             # Benchmarking harness
│   ├── __init__.py
│   ├── benchmark_runner.py  # Orchestrates 4 runs
│   ├── metrics_collector.py # Collects all metrics
│   └── report_generator.py  # Generates comparison reports
├── visualization/          # Dashboard
│   ├── __init__.py
│   └── dashboard.py        # Streamlit dashboard
└── pipeline.py             # Main orchestration
```

## Key Components

### Data Product Schema (`data_product/schema.py`)

- Defines input/output contracts
- Schema validation
- Metadata tracking

### Feature Builders (`etl/cpu_feature_builder.py`, `etl/gpu_feature_builder.py`)

- Base feature generation
- Enriched feature generation
- Identical logic, different backends

### Model Competition (`models/cpu_competition.py`, `models/gpu_competition.py`)

- Train multiple candidates
- Cross-validation per model
- Metric collection
- Champion selection

### Serving Layer (`serving/`)

- CPU: sklearn models, XGBoost CPU
- GPU: cuML models, FIL for XGBoost
- Hotspot generation
- Staging recommendations

### Benchmark Harness (`benchmarks/`)

- Orchestrates 4 runs
- Collects: latency, throughput, quality
- Generates comparison reports
- Cold-start vs warm-start timing

## Benchmark Metrics

### A) Pipeline Latency (End-to-End)

- Ingest + cleaning time
- Feature build time
- Train time
- Inference latency
- Staging recommendation generation

### B) Throughput and Scale

- Rows processed/sec for feature engineering
- Max dataset size feasible
- Time-to-first-dashboard-update

### C) Model Quality

- Precision@K hotspots
- Recall@K
- Staging utility proxy (incidents "covered")

## Artifact Contract

**Standardized Outputs** (both pipelines):

- `features_base.parquet`
- `features_enriched.parquet`
- `cv_folds.json` (time splits)
- `model_results.parquet` (all candidates × folds)
- `champion_model.*` (pickle/joblib or XGBoost JSON)
- `serving_bundle/` (includes FIL engine if XGBoost wins)
- `inference_snapshot.parquet` (next-hour predictions + staging recs)

## Critical Fairness Rules

1. **Same Transformations**: Identical feature definitions, window sizes
2. **Same Splits**: Identical train/test split strategy
3. **Same Models**: Equivalent algorithms (CPU vs GPU versions)
4. **Same Hyperparams**: Identical hyperparameter grids
5. **Report Both**: Cold-start and warm-start timing

## Implementation Phases

### Phase 1: Data Product Design

- Define schema contracts
- Build data product validator
- Create converter utilities

### Phase 2: ETL Pipelines

- CPU ingestion & feature builder
- GPU ingestion & feature builder
- Time-aware CV splitter
- Ensure identical transformations

### Phase 3: Model Competition

- CPU model competition orchestrator
- GPU model competition orchestrator
- Evaluator & KPI store
- Champion selector

### Phase 4: Serving Layer

- CPU serving module
- GPU serving module (cuML + FIL)
- Hotspot & staging engine

### Phase 5: Benchmarking

- Benchmark runner (4 runs)
- Metrics collector
- Report generator

### Phase 6: Visualization

- Streamlit dashboard
- Comparison visualizations
- Performance reports

## Success Criteria

1. **Data Product**: Any data source conforming to schema runs through pipeline automatically
2. **Fair Comparison**: CPU vs GPU pipelines produce identical artifacts (except performance)
3. **Measurable Wins**: Clear deltas in time, throughput, quality
4. **Reproducibility**: Deterministic results, same splits, same models
5. **Scalability**: Demonstrate GPU advantages at scale