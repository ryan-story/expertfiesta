#!/usr/bin/env python3
"""
GPU X Pipeline: Merge + Feature Engineering (RAPIDS / cuDF)

Generates X_features.parquet on GPU (fast, no zero-shot classification)

Usage:
  python -m dgxb_gpu.run_X_pipeline
Or:
  python dgxb_gpu/run_X_pipeline.py

Notes:
- This runner calls the GPU implementation in dgxb_gpu/etl/.
- Output is expected under ./gold-gpu-traffic/X_features.parquet
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb_gpu
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb_gpu.etl.feature_engineering_gpu import (
    merge_and_save_X_features_gpu,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    X_features = merge_and_save_X_features_gpu(
        h3_resolution=9,  # ~0.5km cells
        k_ring_size=1,  # Include immediate neighbors
        max_spatial_km=50,  # Max 50km spatial distance
        max_time_hours=1,  # Max 1 hour time difference
        spatial_weight=1.0,  # Weight for spatial distance
        temporal_weight=0.5,  # Weight for temporal distance
    )

    print("\n✓ GPU X Pipeline complete!")
    print(
        f"  X (features): {X_features.shape[0]:,} records, {X_features.shape[1]} features"
    )
    print("  Check ./gold-gpu-traffic/X_features.parquet")
    print("  Run Y pipeline separately: python dgxb_gpu/run_Y_pipeline.py")
