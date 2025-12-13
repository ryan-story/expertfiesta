#!/usr/bin/env python3
"""
Combined Pipeline: Run both X and Y pipelines sequentially
For faster iteration, use run_X_pipeline.py and run_Y_pipeline.py separately
Usage: python -m dgxb.run_feature_engineering
Or: python dgxb/run_feature_engineering.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb.etl.feature_engineering import merge_and_save_to_gold

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    print("=" * 70)
    print("Running combined X + Y pipeline")
    print("For faster iteration, run separately:")
    print("  python dgxb/run_X_pipeline.py  (fast, no zero-shot)")
    print("  python dgxb/run_Y_pipeline.py (slow, zero-shot classification)")
    print("=" * 70)
    print()

    X_features, y_target = merge_and_save_to_gold(
        h3_resolution=9,  # ~0.5km cells
        k_ring_size=1,  # Include immediate neighbors
        max_spatial_km=50,  # Max 50km spatial distance
        max_time_hours=1,  # Max 1 hour time difference
        spatial_weight=1.0,  # Weight for spatial distance
        temporal_weight=0.5,  # Weight for temporal distance
    )

    print("\n✓ Combined pipeline complete!")
    print(
        f"  X (features): {X_features.shape[0]:,} records, {X_features.shape[1]} features"
    )
    print(f"  y (target): {y_target.shape[0]:,} records")
    print("  Check ./gold-cpu-traffic/ directory for results.")
