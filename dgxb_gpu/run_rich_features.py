#!/usr/bin/env python3
"""
GPU Rich Feature Engineering Runner (RAPIDS / cuDF)

Runs the GPU-first rich feature engineering pipeline:
- Reads sector-hour base features from Parquet (GPU)
- Adds temporal features, lags, rolling stats, spatial neighbor aggregates
- Writes enriched features to Parquet (GPU)

Usage:
  python -m dgxb_gpu.run_rich_features
  OR
  python dgxb_gpu/run_rich_features.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.etl.rich_feature_engineering_gpu import (  # noqa: E402
    enrich_X_features_gpu,
    RichGPUParams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    params = RichGPUParams(
        # Temporal
        add_holidays=True,
        holiday_country="US",
        # Lags/Rolling (hours)
        lag_windows=(1, 3, 6, 12, 24),
        rolling_windows=(3, 6, 12, 24),
        # Spatial (H3 neighbors)
        spatial_k_ring=1,
        spatial_aggs=("mean", "max", "sum"),
        spatial_feature_cols=None,  # auto-detect
        # Super-enrichment (keep trend off by default; can be expensive)
        add_trend_features=False,
        trend_windows=(6, 12, 24),
        add_volatility_features=True,
        volatility_windows=(6, 12, 24),
        # Output
        drop_timestamp_if_hour_ts_present=True,
    )

    X_enriched = enrich_X_features_gpu(
        X_input_path="gold-gpu-traffic/X_features.parquet",
        rich_output_dir="rich-gold-gpu-traffic",
        params=params,
    )

    print("\n✓ GPU Rich feature engineering complete!")
    print(
        f"  Enriched X: {X_enriched.shape[0]:,} records, {X_enriched.shape[1]} features"
    )
    print("  Check ./rich-gold-gpu-traffic/ directory for results.")


if __name__ == "__main__":
    main()
