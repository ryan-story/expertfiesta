#!/usr/bin/env python3
"""
Y Pipeline: Zero-Shot Classification + Target Preparation
Generates y_target.parquet (requires X pipeline to run first)
Usage: python -m dgxb.run_Y_pipeline
Or: python dgxb/run_Y_pipeline.py
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path so we can import dgxb
sys.path.insert(0, str(Path(__file__).parent.parent))

from dgxb.etl.feature_engineering import prepare_y_target_regression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    y_target = prepare_y_target_regression(
        sector_hour_base_path="gold-cpu-traffic/sector_hour_base.parquet",
    )

    print("\n✓ Y Pipeline complete!")
    print(f"  y (target): {y_target.shape[0]:,} records")
    print("  Check ./gold-cpu-traffic/y_target.parquet")
